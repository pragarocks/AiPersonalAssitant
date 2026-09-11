"""Platform-aware TTS: mlx-audio on Apple Silicon, omnivoice.cpp elsewhere."""

import atexit
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# Comprehensive language detection for OmniVoice
_SCRIPT_MAP = [
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),  # Tamil
    (re.compile(r"[\u0900-\u097F]"), "hi"),  # Hindi / Devanagari
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),  # Telugu
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),  # Kannada
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),  # Malayalam
    (re.compile(r"[\u0980-\u09FF]"), "bn"),  # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),  # Gujarati
    (re.compile(r"[\u0A00-\u0A7F]"), "pa"),  # Punjabi
    (re.compile(r"[\u3040-\u30FF]"), "ja"),  # Japanese
    (re.compile(r"[\u4E00-\u9FFF]"), "zh"),  # Chinese
    (re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"), "ko"),  # Korean
    (re.compile(r"[\u0600-\u06FF]"), "ar"),  # Arabic / Urdu
    (re.compile(r"[\u0400-\u04FF]"), "ru"),  # Cyrillic / Russian
    (re.compile(r"[\u0370-\u03FF]"), "el"),  # Greek
    (re.compile(r"[\u0E00-\u0E7F]"), "th"),  # Thai
]

_LATIN_STOPWORDS = {
    "de": {
        "der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich",
        "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als",
        "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
        "wird", "bei", "einer", "um", "am", "sind", "hallo", "danke", "guten",
        "tag", "wie", "geht", "ihnen", "heute", "bitte", "ja", "nein", "wir"
    },
    "fr": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une", "que",
        "est", "pour", "qui", "dans", "par", "plus", "pas", "au", "sur", "ne",
        "ce", "avec", "se", "sont", "ou", "son", "cette", "comme", "aux", "mais",
        "nous", "vous", "ils", "tout", "sa", "ces", "ses", "bonjour", "merci",
        "comment", "allez", "aujourd", "hui"
    },
    "es": {
        "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las",
        "un", "por", "con", "no", "una", "para", "es", "al", "lo", "como",
        "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
        "son", "entre", "está", "cuando", "muy", "sin", "sobre", "también", "me",
        "hasta", "hay", "donde", "hola", "cómo", "estás", "gracias", "buenos", "días"
    },
    "pt": {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é",
        "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as",
        "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "olá",
        "obrigado", "você", "está", "bom", "dia"
    },
    "it": {
        "di", "e", "il", "che", "la", "a", "in", "per", "un", "del", "da",
        "non", "si", "una", "dei", "sono", "le", "con", "ed", "della", "anche",
        "ciao", "grazie", "come", "stai", "buongiorno"
    },
}


def detect_language(text: str) -> str:
    """Detect language ISO tag from text (e.g. 'ta', 'en', 'hi', 'zh', 'es', etc.)."""
    if not text:
        return "en"
    for pattern, lang in _SCRIPT_MAP:
        if pattern.search(text):
            return lang
    words = set(re.findall(r"\b[a-zA-Záéíóúüñàèìòùâêîôûçãõäöß]+\b", text.lower()))
    best_lang = "en"
    max_matches = 0
    for lang, stopwords in _LATIN_STOPWORDS.items():
        matches = len(words & stopwords)
        if matches > max_matches:
            max_matches = matches
            best_lang = lang
    return best_lang


LANGUAGE_NAMES = {
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "ko": "Korean (한국어)",
    "ar": "Arabic (العربية)",
    "ru": "Russian (Русский)",
    "el": "Greek (Ελληνικά)",
    "th": "Thai (ไทย)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "en": "English",
}


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(
        self,
        text: str,
        voice: str = "female, young adult, moderate pitch",
        language: str | None = None,
        speed: float = 1.1,
    ) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        language: str | None = None,
        speed: float = 1.1,
    ) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class OmniVoiceCPPBackend(TTSBackend):
    """omnivoice.cpp backend via local tts-server HTTP server."""

    DEFAULT_VOICE = "female, young adult, moderate pitch"

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.sample_rate = 24000
        self._process: subprocess.Popen | None = None

        if not self._check_health():
            self._start_server()

    def _check_health(self) -> bool:
        url = f"http://{self.host}:{self.port}/health"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _start_server(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        omni_dir = project_root / "Omni_Dependency_models"
        exe_name = "tts-server.exe" if sys.platform == "win32" else "tts-server"
        exe_path = omni_dir / exe_name

        model_path = omni_dir / "models" / "omnivoice-base-Q8_0.gguf"
        codec_path = omni_dir / "models" / "omnivoice-tokenizer-Q8_0.gguf"

        if not exe_path.exists():
            raise FileNotFoundError(f"omnivoice.cpp executable not found at {exe_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"omnivoice model file not found at {model_path}")
        if not codec_path.exists():
            raise FileNotFoundError(f"omnivoice codec file not found at {codec_path}")

        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--codec", str(codec_path),
            "--host", self.host,
            "--port", str(self.port),
        ]

        print(f"Starting omnivoice tts-server on {self.host}:{self.port}...")
        self._process = subprocess.Popen(
            cmd,
            cwd=str(omni_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self._cleanup)

        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < 30:
            if self._check_health():
                print(f"omnivoice tts-server ready on {self.host}:{self.port}")
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"omnivoice tts-server process exited unexpectedly with code {self._process.returncode}"
                )
            time.sleep(0.5)

        raise TimeoutError(f"omnivoice tts-server failed to start on {self.host}:{self.port} within 30 seconds")

    def _cleanup(self):
        if self._process and self._process.poll() is None:
            print("Stopping omnivoice tts-server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def generate(
        self,
        text: str,
        voice: str = "female, young adult, moderate pitch",
        language: str | None = None,
        speed: float = 1.1,
    ) -> np.ndarray:
        if not voice or voice in ("af_heart", "default"):
            voice = self.DEFAULT_VOICE

        if language is None:
            language = detect_language(text)

        url = f"http://{self.host}:{self.port}/v1/audio/speech"
        # tts-server.exe expects 'instructions' for voice design persona
        # and 'language' for ISO language tag (e.g. 'ta', 'en', 'hi', etc.)
        payload = json.dumps({
            "input": text,
            "instructions": voice,
            "voice": voice,
            "language": language,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"OmniVoice speech API returned HTTP status {resp.status}")
                pcm_bytes = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OmniVoice speech API error: {e.code} - {e.read().decode()}") from e

        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (pcm_int16.astype(np.float32) / 32768.0)


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if _is_apple_silicon() and not os.environ.get("OMNIVOICE"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to omnivoice.cpp")

    backend = OmniVoiceCPPBackend()
    print(f"TTS: omnivoice.cpp (sample_rate={backend.sample_rate})")
    return backend
