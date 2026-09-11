import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

/**
 * VRMViewer - Amica-inspired 3D Avatar Controller with Real-time LipSync,
 * Auto-Blink, and tk256ailab VRMA animation playback.
 */
export class VRMViewer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.options = options;

    // Core Three.js components
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.clock = new THREE.Clock();

    // Model & Animation state
    this.currentVrm = null;
    this.mixer = null;
    this.currentAction = null;
    this.animationClips = new Map(); // name -> AnimationClip
    this.loader = null;

    // Lip Sync & Audio
    this.audioCtx = null;
    this.analyser = null;
    this.audioData = new Float32Array(512);
    this.isSpeaking = false;
    this.currentMouthOpen = 0;

    // Auto Blink
    this.blinkTimer = 0;
    this.nextBlinkInterval = 3.0;
    this.blinkDuration = 0.2; // total blink time
    this.blinking = false;
    this.blinkProgress = 0;

    // Gaze & Procedural Motion
    this.lookAtTarget = new THREE.Object3D();
    this.mouse = new THREE.Vector2(0, 0);
    this.currentState = 'loading';

    this.initScene();
    this.initLoader();
    this.setupEventListeners();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initScene() {
    this.scene = new THREE.Scene();

    // Camera setup - framed for portrait / upper body
    const rect = this.canvas.getBoundingClientRect();
    const aspect = (rect.width || 400) / (rect.height || 400);
    this.camera = new THREE.PerspectiveCamera(30, aspect, 0.1, 20.0);
    this.camera.position.set(0, 1.35, 1.25);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(rect.width || 400, rect.height || 400);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1.25, 0);
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 2.5;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
    this.controls.enablePan = false;

    // Lighting (Warm Key + Lavender Rim + Neutral Ambient)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xfff5ea, 1.3);
    dirLight.position.set(1.0, 2.0, 1.5).normalize();
    this.scene.add(dirLight);

    const rimLight = new THREE.DirectionalLight(0xdbeafe, 0.7);
    rimLight.position.set(-1.0, 1.5, -1.0).normalize();
    this.scene.add(rimLight);

    // Look-at dummy target
    this.scene.add(this.lookAtTarget);
    this.lookAtTarget.position.set(0, 1.3, 2.0);
  }

  initLoader() {
    this.loader = new GLTFLoader();
    this.loader.crossOrigin = 'anonymous';

    // Register VRM and VRMA plugins
    this.loader.register((parser) => new VRMLoaderPlugin(parser));
    this.loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
  }

  setupEventListeners() {
    // Window / container resize
    const resize = () => {
      if (!this.canvas.parentElement) return;
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      if (width > 0 && height > 0) {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
      }
    };
    window.addEventListener('resize', resize);
    this.resizeObserver = new ResizeObserver(resize);
    if (this.canvas.parentElement) {
      this.resizeObserver.observe(this.canvas.parentElement);
    }

    // Mouse tracking for subtle look-at
    window.addEventListener('mousemove', (e) => {
      const nx = (e.clientX / window.innerWidth) * 2 - 1;
      const ny = -(e.clientY / window.innerHeight) * 2 + 1;
      this.mouse.set(nx, ny);
    });

    // Drag and drop custom VRM file onto canvas
    this.canvas.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    this.canvas.addEventListener('drop', async (e) => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.name.endsWith('.vrm')) {
          const blobUrl = URL.createObjectURL(file);
          console.log('Loading dropped VRM:', file.name);
          await this.loadVRM(blobUrl);
        }
      }
    });
  }

  setAudioSource(audioCtx, analyser) {
    this.audioCtx = audioCtx;
    this.analyser = analyser;
  }

  async loadVRM(url) {
    try {
      console.log('Loading VRM from:', url);
      const gltf = await this.loader.loadAsync(url);
      const vrm = gltf.userData.vrm;

      if (!vrm) {
        console.error('GLTF does not contain VRM metadata');
        return;
      }

      // Remove existing model if any
      if (this.currentVrm) {
        this.scene.remove(this.currentVrm.scene);
        VRMUtils.deepDispose(this.currentVrm.scene);
      }

      // VRM0 models need rotation to face +Z
      VRMUtils.rotateVRM0(vrm);
      this.currentVrm = vrm;
      this.scene.add(vrm.scene);

      // Setup Animation Mixer
      this.mixer = new THREE.AnimationMixer(vrm.scene);

      // Adjust camera to frame avatar head & shoulders
      if (vrm.humanoid) {
        const headNode = vrm.humanoid.getNormalizedBoneNode('head');
        if (headNode) {
          const headPos = new THREE.Vector3();
          headNode.getWorldPosition(headPos);
          this.camera.position.set(0, headPos.y + 0.05, 1.15);
          this.controls.target.set(0, headPos.y - 0.08, 0);
          this.controls.update();
        }
      }

      console.log('VRM loaded successfully:', vrm.meta?.name || 'Avatar');

      // Preload animation set
      await this.preloadAnimations();

      // Start idle animation
      this.playAnimation('relax');
    } catch (err) {
      console.error('Failed to load VRM:', err);
    }
  }

  async preloadAnimations() {
    const animationFiles = {
      relax: '/static/animations/Relax.vrma',
      thinking: '/static/animations/Thinking.vrma',
      lookAround: '/static/animations/LookAround.vrma',
      clapping: '/static/animations/Clapping.vrma',
      surprised: '/static/animations/Surprised.vrma',
      goodbye: '/static/animations/Goodbye.vrma',
    };

    for (const [name, path] of Object.entries(animationFiles)) {
      try {
        const gltf = await this.loader.loadAsync(path);
        const vrmAnimation = gltf.userData.vrmAnimations?.[0];
        if (vrmAnimation && this.currentVrm) {
          const clip = createVRMAnimationClip(vrmAnimation, this.currentVrm);
          clip.name = name;
          this.animationClips.set(name, clip);
        }
      } catch (err) {
        console.warn(`Failed to preload animation ${name}:`, err);
      }
    }
  }

  playAnimation(name, duration = 0.5) {
    if (!this.mixer || !this.animationClips.has(name)) return;
    const nextClip = this.animationClips.get(name);
    const nextAction = this.mixer.clipAction(nextClip);

    if (this.currentAction === nextAction) return;

    nextAction.reset();
    nextAction.setLoop(name === 'clapping' || name === 'goodbye' ? THREE.LoopOnce : THREE.LoopRepeat);
    nextAction.clampWhenFinished = true;
    nextAction.fadeIn(duration);
    nextAction.play();

    if (this.currentAction) {
      this.currentAction.fadeOut(duration);
    }
    this.currentAction = nextAction;
  }

  onStateChange(state) {
    this.currentState = state;
    if (!this.currentVrm) return;

    if (state === 'speaking') {
      this.isSpeaking = true;
      this.playAnimation('relax', 0.4);
      this.setExpression('happy', 0.25);
    } else if (state === 'processing') {
      this.isSpeaking = false;
      this.currentMouthOpen = 0;
      this.setExpression('happy', 0.0);
      if (this.animationClips.has('thinking')) {
        this.playAnimation('thinking', 0.5);
      }
    } else if (state === 'listening') {
      this.isSpeaking = false;
      this.currentMouthOpen = 0;
      this.setExpression('happy', 0.0);
      this.playAnimation('relax', 0.5);
    }
  }

  setExpression(name, value) {
    if (!this.currentVrm?.expressionManager) return;
    try {
      this.currentVrm.expressionManager.setValue(name, value);
    } catch {}
  }

  updateLipSync(delta) {
    if (!this.currentVrm?.expressionManager) return;

    let targetMouth = 0;
    if (this.isSpeaking && this.analyser) {
      this.analyser.getFloatTimeDomainData(this.audioData);
      let sum = 0;
      for (let i = 0; i < this.audioData.length; i++) {
        sum += Math.abs(this.audioData[i]);
      }
      const avg = sum / this.audioData.length;
      if (avg > 0.015) {
        // Boost and clamp mouth openness
        targetMouth = Math.min(1.0, (avg - 0.015) * 6.0);
      }
    }

    // Smooth lerp mouth motion
    this.currentMouthOpen += (targetMouth - this.currentMouthOpen) * Math.min(1.0, delta * 25);

    // Drive VRM expression blendshapes
    try {
      this.currentVrm.expressionManager.setValue('aa', this.currentMouthOpen * 0.9);
      this.currentVrm.expressionManager.setValue('ih', this.currentMouthOpen * 0.3);
      this.currentVrm.expressionManager.setValue('oh', this.currentMouthOpen * 0.2);
    } catch {}
  }

  updateAutoBlink(delta) {
    if (!this.currentVrm?.expressionManager) return;

    this.blinkTimer += delta;
    if (!this.blinking) {
      if (this.blinkTimer >= this.nextBlinkInterval) {
        this.blinking = true;
        this.blinkProgress = 0;
        this.blinkTimer = 0;
        this.nextBlinkInterval = 2.5 + Math.random() * 3.5; // randomize interval
      }
    } else {
      this.blinkProgress += delta / this.blinkDuration;
      let blinkWeight = 0;
      if (this.blinkProgress < 0.5) {
        blinkWeight = this.blinkProgress * 2.0; // closing eyes
      } else if (this.blinkProgress < 1.0) {
        blinkWeight = (1.0 - this.blinkProgress) * 2.0; // opening eyes
      } else {
        this.blinking = false;
        blinkWeight = 0;
      }
      try {
        this.currentVrm.expressionManager.setValue('blink', blinkWeight);
      } catch {}
    }
  }

  updateGaze(delta) {
    if (!this.currentVrm) return;

    // Subtle gaze target follows mouse slightly
    const targetX = this.mouse.x * 0.4;
    const targetY = 1.35 + this.mouse.y * 0.2;
    this.lookAtTarget.position.x += (targetX - this.lookAtTarget.position.x) * delta * 4;
    this.lookAtTarget.position.y += (targetY - this.lookAtTarget.position.y) * delta * 4;

    if (this.currentVrm.lookAt) {
      this.currentVrm.lookAt.target = this.lookAtTarget;
    }
  }

  animate() {
    requestAnimationFrame(this.animate);
    const delta = Math.min(this.clock.getDelta(), 0.1);

    if (this.controls) {
      this.controls.update();
    }

    if (this.mixer) {
      this.mixer.update(delta);
    }

    if (this.currentVrm) {
      this.updateLipSync(delta);
      this.updateAutoBlink(delta);
      this.updateGaze(delta);
      this.currentVrm.update(delta);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  destroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    if (this.renderer) {
      this.renderer.dispose();
    }
    if (this.currentVrm) {
      VRMUtils.deepDispose(this.currentVrm.scene);
    }
  }
}
