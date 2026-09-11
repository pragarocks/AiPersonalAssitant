import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

// ── Math & Smoothing Helpers from Maya ──
export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function smoothFactor(smoothing, dt) {
  return 1 - Math.pow(smoothing, dt);
}

export function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// ── LipSync Engine from Maya ──
export class MayaLipSync {
  constructor() {
    this.analyser = null;
    this.data = new Uint8Array(0);
    this.value = 0;
  }

  setAnalyser(analyser) {
    this.analyser = analyser;
    this.data = new Uint8Array(analyser.fftSize || 256);
  }

  clear() {
    this.analyser = null;
    this.value = 0;
  }

  sample(dt) {
    if (!this.analyser) {
      this.value = lerp(this.value, 0, smoothFactor(1e-4, dt));
      return this.value;
    }
    this.analyser.getByteTimeDomainData(this.data);
    let sum = 0;
    const n = this.data.length;
    for (let i = 0; i < n; i++) {
      const x = (this.data[i] - 128) / 128;
      sum += x * x;
    }
    const rms = Math.sqrt(sum / n);

    let target = clamp01(rms * 2.6);
    if (target < 0.08) target = 0; // noise gate: fully closes mouth between words

    const s = target > this.value ? smoothFactor(1e-11, dt) : smoothFactor(1e-4, dt);
    this.value = lerp(this.value, target, s);
    return this.value;
  }
}

// ── Aliveness Engine from Maya ──
export class MayaAliveness {
  constructor() {
    this.blinkT = 0;
    this.nextBlink = 1.5 + Math.random() * 3.5;
    this.blinkValue = 0;

    this.yaw = 0;
    this.pitch = 0;
    this.targetYaw = 0;
    this.targetPitch = 0;
    this.retarget = 0;
    this.enabled = true;
  }

  update(dt, vrm) {
    if (!this.enabled || !vrm) return;
    this.updateBlink(dt, vrm);
    this.updateHead(dt, vrm);
  }

  updateBlink(dt, vrm) {
    this.blinkT += dt;
    if (this.blinkT >= this.nextBlink) {
      const into = this.blinkT - this.nextBlink;
      const dur = 0.16;
      if (into >= dur) {
        this.blinkValue = 0;
        this.blinkT = 0;
        this.nextBlink = 1.5 + Math.random() * 4.5;
      } else {
        const half = dur / 2;
        this.blinkValue = into < half ? into / half : 1 - (into - half) / half;
      }
    }
    try {
      vrm.expressionManager?.setValue('blink', this.blinkValue);
    } catch {}
  }

  updateHead(dt, vrm) {
    this.retarget -= dt;
    if (this.retarget <= 0) {
      this.targetYaw = (Math.random() - 0.5) * 0.15;
      this.targetPitch = (Math.random() - 0.5) * 0.08;
      this.retarget = 1.5 + Math.random() * 2.5;
    }
    const s = smoothFactor(0.2, dt);
    this.yaw = lerp(this.yaw, this.targetYaw, s);
    this.pitch = lerp(this.pitch, this.targetPitch, s);

    const neck = vrm.humanoid?.getNormalizedBoneNode('neck');
    const spine = vrm.humanoid?.getNormalizedBoneNode('spine');
    if (neck) {
      neck.rotation.y = this.yaw;
      neck.rotation.x = this.pitch;
    }
    if (spine) {
      spine.rotation.y = this.yaw * 0.3;
    }
  }
}

// ── Expressions Engine from Maya ──
const PRESETS = {
  neutral: {},
  happy: { happy: 0.55 },
  sad: { sad: 0.8 },
  angry: { angry: 0.8 },
  surprised: { surprised: 0.85 },
  relaxed: { relaxed: 0.55 },
  thinking: { relaxed: 0.25, lookUp: 0.2 },
};

const ALL_KEYS = ['happy', 'sad', 'angry', 'surprised', 'relaxed', 'lookUp'];

export class MayaExpressions {
  constructor() {
    this.current = {};
    this.target = {};
  }

  setEmotion(emotion) {
    this.target = { ...(PRESETS[emotion] || {}) };
  }

  update(dt, vrm) {
    const em = vrm?.expressionManager;
    if (!em) return;
    const s = smoothFactor(0.02, dt);
    for (const key of ALL_KEYS) {
      const want = this.target[key] ?? 0;
      const now = lerp(this.current[key] ?? 0, want, s);
      this.current[key] = now;
      try {
        em.setValue(key, now);
      } catch {}
    }
  }
}

// ── Gestures Engine from Maya ──
const GESTURE_DURATIONS = {
  nod: 0.9,
  jump: 0.7,
  lookAround: 1.8,
  wave: 2.0,
  goodbye: 2.0,
  clap: 1.6,
  thinking: 1.5,
};

const TRACKED_BONES = [
  'head',
  'hips',
  'rightUpperArm',
  'rightLowerArm',
  'leftUpperArm',
  'leftLowerArm',
];

export class MayaGestures {
  constructor() {
    this.active = null;
    this.elapsed = 0;
    this.bases = new Map();
  }

  play(gesture, vrm) {
    if (!vrm) return;
    this.active = gesture;
    this.elapsed = 0;
    this.bases.clear();
    for (const name of TRACKED_BONES) {
      const node = vrm.humanoid?.getNormalizedBoneNode(name);
      if (node) {
        this.bases.set(name, { rot: node.rotation.clone(), pos: node.position.clone() });
      }
    }
  }

  update(dt, vrm) {
    if (!this.active || !vrm) return;
    this.elapsed += dt;
    const dur = GESTURE_DURATIONS[this.active] || 1.5;
    const p = Math.min(1, this.elapsed / dur);
    this.apply(this.active, p, vrm);
    if (p >= 1) {
      this.restore(vrm);
      this.active = null;
    }
  }

  node(vrm, name) {
    return vrm.humanoid?.getNormalizedBoneNode(name) ?? null;
  }

  base(name) {
    return this.bases.get(name) ?? { rot: new THREE.Euler(), pos: new THREE.Vector3() };
  }

  restore(vrm) {
    for (const name of TRACKED_BONES) {
      const node = this.node(vrm, name);
      const b = this.bases.get(name);
      if (node && b) {
        node.rotation.copy(b.rot);
        node.position.copy(b.pos);
      }
    }
  }

  apply(g, p, vrm) {
    const env = Math.sin(p * Math.PI);
    switch (g) {
      case 'nod': {
        const head = this.node(vrm, 'head');
        if (head) head.rotation.x = this.base('head').rot.x + Math.sin(p * Math.PI * 2) * 0.35;
        break;
      }
      case 'lookAround': {
        const head = this.node(vrm, 'head');
        if (head) head.rotation.y = this.base('head').rot.y + Math.sin(p * Math.PI * 2) * 0.5;
        break;
      }
      case 'jump': {
        const hips = this.node(vrm, 'hips');
        if (hips) hips.position.y = this.base('hips').pos.y + env * 0.12;
        break;
      }
      case 'wave':
      case 'goodbye': {
        const up = this.node(vrm, 'rightUpperArm');
        const lo = this.node(vrm, 'rightLowerArm');
        const bu = this.base('rightUpperArm').rot;
        const bl = this.base('rightLowerArm').rot;
        if (up) {
          up.rotation.z = bu.z + env * 2.2;
          up.rotation.x = bu.x - env * 0.1;
        }
        if (lo) lo.rotation.z = bl.z + env * 1.2 + env * Math.sin(p * Math.PI * 6) * 0.35;
        break;
      }
      case 'clap': {
        const lu = this.node(vrm, 'leftUpperArm');
        const ru = this.node(vrm, 'rightUpperArm');
        const ll = this.node(vrm, 'leftLowerArm');
        const rl = this.node(vrm, 'rightLowerArm');
        const beat = Math.sin(p * Math.PI * 8) * 0.2;
        if (lu && ru) {
          lu.rotation.x = this.base('leftUpperArm').rot.x + env * 0.8;
          ru.rotation.x = this.base('rightUpperArm').rot.x + env * 0.8;
          lu.rotation.z = this.base('leftUpperArm').rot.z - env * 0.5 + beat;
          ru.rotation.z = this.base('rightUpperArm').rot.z + env * 0.5 - beat;
        }
        if (ll && rl) {
          ll.rotation.y = this.base('leftLowerArm').rot.y + env * 0.9;
          rl.rotation.y = this.base('rightLowerArm').rot.y - env * 0.9;
        }
        break;
      }
      case 'thinking': {
        const head = this.node(vrm, 'head');
        const ru = this.node(vrm, 'rightUpperArm');
        const rl = this.node(vrm, 'rightLowerArm');
        if (head) {
          head.rotation.z = this.base('head').rot.z + env * 0.15;
          head.rotation.x = this.base('head').rot.x - env * 0.1;
        }
        if (ru) ru.rotation.z = this.base('rightUpperArm').rot.z + env * 0.8;
        if (rl) rl.rotation.z = this.base('rightLowerArm').rot.z + env * 0.6;
        break;
      }
    }
  }
}

// ── Full Maya Avatar Stage & Controller ──
export class MayaAvatarController {
  constructor(container, options = {}) {
    this.container = container;
    this.options = options;

    this.scene = new THREE.Scene();
    this.clock = new THREE.Clock();

    const w = container.clientWidth || 400;
    const h = container.clientHeight || 400;

    // Camera setup framed head-and-shoulders
    this.camera = new THREE.PerspectiveCamera(28, w / h, 0.1, 50);
    this.camera.position.set(0, 1.32, 1.1);
    this.camera.lookAt(0, 1.25, 0);

    // Renderer (transparent alpha)
    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      premultipliedAlpha: false,
      antialias: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
    container.appendChild(this.renderer.domElement);

    // Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1.25, 0);
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 2.5;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
    this.controls.enablePan = false;

    // Lighting (from Maya)
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.3));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1, 1.4, 1.2);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xbcd0ff, 0.6);
    rim.position.set(-1, 0.8, -0.6);
    this.scene.add(rim);

    // Engines
    this.vrm = null;
    this.lipsync = new MayaLipSync();
    this.aliveness = new MayaAliveness();
    this.expressions = new MayaExpressions();
    this.gestures = new MayaGestures();

    // Loader
    this.loader = new GLTFLoader();
    this.loader.crossOrigin = 'anonymous';
    this.loader.register((parser) => new VRMLoaderPlugin(parser));
    this.loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

    // Resize observer
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);

    this.animate = this.animate.bind(this);
    this.looping = true;
    requestAnimationFrame(this.animate);

    if (options.vrmUrl) {
      this.loadVRM(options.vrmUrl);
    }
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w > 0 && h > 0) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    }
  }

  async loadVRM(url) {
    try {
      console.log('[Maya] Loading VRM:', url);
      const gltf = await this.loader.loadAsync(url);
      const vrm = gltf.userData.vrm;
      if (!vrm) throw new Error('No VRM payload in file');

      VRMUtils.removeUnnecessaryVertices(gltf.scene);
      try {
        VRMUtils.combineSkeletons(gltf.scene);
      } catch {}
      vrm.scene.traverse((o) => {
        if (o.isMesh) o.frustumCulled = false;
      });
      VRMUtils.rotateVRM0(vrm);

      if (this.vrm) {
        this.scene.remove(this.vrm.scene);
        VRMUtils.deepDispose(this.vrm.scene);
      }

      this.vrm = vrm;
      this.scene.add(vrm.scene);
      if (vrm.lookAt) vrm.lookAt.target = this.camera;

      // Lower arms to natural relaxed A-pose (Maya rest pose)
      this.applyRestPose(vrm);
      this.frameToVRM(vrm);

      console.log('[Maya] VRM loaded successfully');
      return true;
    } catch (err) {
      console.error('[Maya] VRM load error:', err);
      return false;
    }
  }

  applyRestPose(vrm) {
    const h = vrm.humanoid;
    if (!h) return;
    h.getNormalizedBoneNode('leftUpperArm')?.rotation.set(0, 0, 1.2);
    h.getNormalizedBoneNode('rightUpperArm')?.rotation.set(0, 0, -1.2);
    h.getNormalizedBoneNode('leftLowerArm')?.rotation.set(0, -0.15, 0.1);
    h.getNormalizedBoneNode('rightLowerArm')?.rotation.set(0, 0.15, -0.1);
  }

  frameToVRM(vrm) {
    const head = vrm.humanoid?.getNormalizedBoneNode('head');
    const headPos = new THREE.Vector3(0, 1.35, 0);
    head?.getWorldPosition(headPos);
    const y = headPos.y;
    this.camera.position.set(0, y - 0.05, 1.1);
    this.controls.target.set(0, y - 0.15, 0);
    this.controls.update();
  }

  connectAnalyser(analyser) {
    this.lipsync.setAnalyser(analyser);
  }

  disconnectAnalyser() {
    this.lipsync.clear();
  }

  setEmotion(emotion) {
    this.expressions.setEmotion(emotion);
  }

  playGesture(gesture) {
    this.gestures.play(gesture, this.vrm);
  }

  onStateChange(state) {
    if (state === 'speaking') {
      this.setEmotion('happy');
    } else if (state === 'processing') {
      this.setEmotion('thinking');
      this.playGesture('thinking');
    } else if (state === 'listening') {
      this.setEmotion('neutral');
    }
  }

  animate() {
    if (!this.looping) return;
    requestAnimationFrame(this.animate);
    const dt = Math.min(this.clock.getDelta(), 0.1);

    if (this.controls) this.controls.update();

    if (this.vrm) {
      const mouth = this.lipsync.sample(dt);
      try {
        this.vrm.expressionManager?.setValue('aa', mouth);
      } catch {}
      this.aliveness.update(dt, this.vrm);
      this.expressions.update(dt, this.vrm);
      this.gestures.update(dt, this.vrm);
      this.vrm.update(dt);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  destroy() {
    this.looping = false;
    if (this.ro) this.ro.disconnect();
    if (this.renderer) this.renderer.dispose();
    if (this.vrm) VRMUtils.deepDispose(this.vrm.scene);
  }
}
