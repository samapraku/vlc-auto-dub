"""TTS voice selection that follows the target language.

Policy
------
- ``espeak``: the target language code *is* the espeak-ng voice (espeak-ng
  ships ``de``, ``fr``, ``es``, ``tr``, ``ru``, ``it`` ...). If espeak-ng does
  not ship it, the provider raises a clear error rather than guessing.
- ``piper``: a small map of known language -> voice. A language not in the map
  falls back to the default English voice, and the run records a warning so the
  fallback is never silent.
- other providers (e.g. the test fake): no voice selection.
"""
from __future__ import annotations

DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_ESPEAK_VOICE = "en"

# Verified against the rhasspy/piper-voices index (medium quality, one per lang).
PIPER_VOICE_BY_LANG = {
    "en": "en_US-lessac-medium",
    "de": "de_DE-thorsten-medium",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
    "it": "it_IT-paola-medium",
    "tr": "tr_TR-dfki-medium",
    "ru": "ru_RU-dmitri-medium",
    "pt": "pt_BR-faber-medium",
}


def effective_voice(provider: str, lang: str | None) -> tuple[str | None, str | None]:
    """Return ``(voice, warning)`` for ``provider`` targeting ``lang``.

    ``warning`` is a human-readable note when a documented fallback was used.
    """
    if provider == "espeak":
        return (lang or DEFAULT_ESPEAK_VOICE), None
    if provider == "piper":
        if lang and lang in PIPER_VOICE_BY_LANG:
            return PIPER_VOICE_BY_LANG[lang], None
        if not lang:
            return DEFAULT_PIPER_VOICE, None
        return (
            DEFAULT_PIPER_VOICE,
            f"no piper voice mapped for language {lang!r}; using {DEFAULT_PIPER_VOICE}",
        )
    return None, None


# Voices for --speaker-voices, best first. "model#speaker" picks one speaker of
# a multi-speaker model (one download covers them all).
# English: CMU ARCTIC speakers; each one's median pitch was measured from piper
# output (male 96-140 Hz, female 177-216 Hz), matching the dataset's notes.
# Other languages: genders from the voices' datasets, not measured.
_ARCTIC = "en_US-arctic-medium#"
PIPER_SPEAKER_POOL = {
    "en": {"male": [_ARCTIC + s for s in ("rms", "bdl", "aew", "jmk", "fem", "awb")],
           "female": [_ARCTIC + s for s in ("slt", "clb", "lnh", "ljm", "eey", "axb")]},
    "de": {"male": ["de_DE-thorsten-medium", "de_DE-karlsson-low"],
           "female": ["de_DE-kerstin-medium", "de_DE-ramona-low"]},
    "fr": {"male": ["fr_FR-upmc-medium#pierre", "fr_FR-tom-medium"],
           "female": ["fr_FR-upmc-medium#jessica", "fr_FR-siwis-medium"]},
    "es": {"male": ["es_ES-sharvard-medium#M", "es_ES-davefx-medium"],
           "female": ["es_ES-sharvard-medium#F"]},
    "it": {"male": ["it_IT-riccardo-x_low"], "female": ["it_IT-paola-medium"]},
    "ru": {"male": ["ru_RU-dmitri-medium", "ru_RU-denis-medium", "ru_RU-ruslan-medium"],
           "female": ["ru_RU-irina-medium"]},
    "tr": {"male": ["tr_TR-dfki-medium"], "female": []},
    "pt": {"male": ["pt_BR-faber-medium", "pt_BR-cadu-medium", "pt_BR-jeff-medium"],
           "female": []},
}


def speaker_pool(provider: str, lang: str | None) -> tuple[dict[str, list[str]], str | None]:
    """Return ``(pool, warning)``: the voices to share out between speakers."""
    if provider != "piper":
        return {}, f"different voices per speaker need the piper TTS, not {provider!r}"
    pool = PIPER_SPEAKER_POOL.get(lang or "en")
    if pool is None:
        return {}, f"no speaker voices for language {lang!r}; every speaker uses one voice"
    missing = [g for g in ("male", "female") if not pool[g]]
    if missing:
        return pool, (f"piper has no {missing[0]} voice for {lang!r}; "
                      f"{missing[0]} speakers get one of the other voices")
    return pool, None
