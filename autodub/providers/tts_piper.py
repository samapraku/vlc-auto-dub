"""Real local TTS via piper (ONNX, CPU). Voice models download on demand.

A voice is a piper model name (``en_US-lessac-medium``), optionally followed by
``#<speaker>`` for one speaker of a multi-speaker model
(``en_US-arctic-medium#slt``). Loaded models are kept, so a run that switches
between voices line by line loads each model once.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

from ..errors import DependencyError, ProviderError

DEFAULT_VOICE = "en_US-lessac-medium"


def default_cache_dir() -> Path:
    base = os.environ.get("AUTODUB_PIPER_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vlc-auto-dub", "piper"
    )
    return Path(base)


def split_voice(voice: str) -> tuple[str, str | None]:
    """``"model#speaker"`` -> ``("model", "speaker")``; no ``#`` -> ``(voice, None)``."""
    model, sep, speaker = voice.partition("#")
    return model, (speaker if sep and speaker else None)


class PiperTTS:
    def __init__(self, voice: str | None = None, cache_dir: str | None = None):
        self.voice = voice or DEFAULT_VOICE
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._models: dict[str, object] = {}

    def _ensure_model(self, name: str) -> Path:
        model = self.cache_dir / f"{name}.onnx"
        if model.exists():
            return model
        try:
            from piper.download_voices import download_voice
        except ImportError as exc:  # pragma: no cover
            raise DependencyError(
                "provider 'piper' needs the 'piper-tts' package (pip install piper-tts)"
            ) from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_voice(name, self.cache_dir)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"could not download piper voice {name!r}: {exc}"
            ) from exc
        if not model.exists():
            raise ProviderError(f"piper voice {name!r} not found after download")
        return model

    def _load(self, name: str):
        if name not in self._models:
            try:
                from piper import PiperVoice
            except ImportError as exc:  # pragma: no cover
                raise DependencyError(
                    "provider 'piper' needs the 'piper-tts' package (pip install piper-tts)"
                ) from exc
            self._models[name] = PiperVoice.load(str(self._ensure_model(name)))
        return self._models[name]

    def _speaker_id(self, model, name: str, speaker: str | None) -> int | None:
        if speaker is None:
            return None
        ids = getattr(model.config, "speaker_id_map", None) or {}
        count = getattr(model.config, "num_speakers", 1) or 1
        if count > 1:
            if speaker in ids:
                return ids[speaker]
            if speaker.isdigit() and int(speaker) < count:
                return int(speaker)
        raise ProviderError(f"piper voice {name!r} has no speaker {speaker!r}")

    def synthesize(self, text: str, out_path: str, voice: str | None) -> None:
        if voice:
            self.voice = voice
        name, speaker = split_voice(self.voice)
        model = self._load(name)
        speaker_id = self._speaker_id(model, name, speaker)
        try:
            syn_config = None
            if speaker_id is not None:
                from piper import SynthesisConfig

                syn_config = SynthesisConfig(speaker_id=speaker_id)
            with wave.open(out_path, "wb") as wav_file:
                model.synthesize_wav(text, wav_file, syn_config=syn_config)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"piper synthesis failed: {exc}") from exc
