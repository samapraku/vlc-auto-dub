"""Tell the speakers in a film apart so each can get a voice of their own.

1. Every dialogue line of a second or more gets a speaker embedding (a
   "voiceprint") from the original audio, using Microsoft's WavLM
   speaker-verification model.
2. Lines whose voiceprints are close are grouped into speakers (average-linkage
   clustering on cosine distance). Groups of only one or two lines are mostly
   fragments of a real speaker (shouting, a noisy moment), so they join the
   nearest larger group. Short lines carry too little voice to group on their
   own, so they join the nearest speaker afterwards.
3. Each speaker is called male or female only when the AudioSet classifier
   already used for sound tags ("Male speech" / "Female speech") is clear about
   it over all of their lines together. Otherwise the gender stays unknown.
   Pitch was tried and dropped: on real recordings a shouting man and a woman
   both measured around 320 Hz, and a male narrator at 170 Hz.
4. Speakers take voices from the language's pool (see ``voices.py``), the one
   with the most dialogue first. Every speaker gets a different voice while the
   pool lasts; that, more than gender, is what tells them apart.

The speaker model is downloaded once from the Hugging Face hub (about 400 MB,
CC-BY-SA 3.0, https://huggingface.co/microsoft/wavlm-base-plus-sv).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import DependencyError, ProviderError

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE = 16000
MIN_EMBED_SECONDS = 1.0     # shorter lines join the nearest speaker afterwards
MIN_TRY_SECONDS = 0.3       # shorter still: same speaker as the line before
MAX_EMBED_SECONDS = 10.0    # enough voice for an embedding; keeps it fast
# Cosine distance below which lines are one speaker. On a TV episode 0.2 kept
# the narrator and each main participant apart; 0.3 merged the narrator with a
# participant.
SAME_SPEAKER = 0.2
MIN_LINES = 3               # smaller groups join the nearest larger one
MAX_SPEAKERS = 8
# Mean (male - female) logit over a speaker's lines needed to call a gender.
GENDER_MARGIN = 0.75
GENDER_LINES = 12           # score at most this many (the longest) lines each


@dataclass(frozen=True)
class Speaker:
    id: int                 # 0 = the speaker with the most dialogue
    gender: str             # "male" | "female" | "unknown"
    gender_score: float | None   # mean male-minus-female logit; >0 leans male
    lines: int
    seconds: float


# --------------------------------------------------------------------------
# Grouping lines into speakers
# --------------------------------------------------------------------------

def _unit(e: np.ndarray) -> np.ndarray:
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)


def _centroids(e: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _unit(np.stack([e[labels == k].mean(axis=0) for k in range(labels.max() + 1)]))


def _merge_into_nearest(e: np.ndarray, labels: np.ndarray, group: int,
                        candidates: list[int]) -> np.ndarray:
    cents = _centroids(e, labels)
    best = max(candidates, key=lambda k: float(cents[k] @ cents[group]))
    labels = labels.copy()
    labels[labels == group] = best
    return np.unique(labels, return_inverse=True)[1]


def cluster(embeddings: np.ndarray, threshold: float = SAME_SPEAKER,
            min_lines: int = MIN_LINES, max_speakers: int = MAX_SPEAKERS) -> np.ndarray:
    """Speaker label (0..k-1) for each row of ``embeddings``."""
    count = len(embeddings)
    if count < 2:
        return np.zeros(count, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage

    e = _unit(np.asarray(embeddings, dtype=np.float64))
    labels = fcluster(linkage(e, method="average", metric="cosine"),
                      t=threshold, criterion="distance") - 1
    labels = np.unique(labels, return_inverse=True)[1]
    # Fold fragments into real speakers, smallest first; then enforce the cap.
    while True:
        sizes = np.bincount(labels)
        small = [k for k in range(len(sizes)) if sizes[k] < min_lines]
        big = [k for k in range(len(sizes)) if sizes[k] >= min_lines]
        if small and big:
            labels = _merge_into_nearest(e, labels, min(small, key=lambda k: sizes[k]), big)
        elif len(sizes) > max_speakers:
            k = int(np.argmin(sizes))
            labels = _merge_into_nearest(e, labels, k, [j for j in range(len(sizes)) if j != k])
        else:
            return labels


def nearest(vec: np.ndarray, centroids: np.ndarray) -> int:
    return int(np.argmax(centroids @ _unit(vec)))


def gender_of(score: float | None, margin: float = GENDER_MARGIN) -> str:
    if score is None or abs(score) < margin:
        return "unknown"
    return "male" if score > 0 else "female"


# --------------------------------------------------------------------------
# Voice assignment
# --------------------------------------------------------------------------

def assign_voices(speakers: list[Speaker], pool: dict[str, list[str]],
                  default: str | None) -> dict[int, str | None]:
    """Voice for each speaker id. ``pool`` maps "male"/"female" to voices.

    Speakers come in order of how much they talk. Each takes the next unused
    voice of their gender; a speaker of unknown gender takes one from whichever
    gender has the most voices left, so speakers alternate and sound different.
    When a gender runs out, the other's voices are used, and only when every
    voice is taken is one reused (the least-used one of the right gender).
    """
    males, females = list(pool.get("male", [])), list(pool.get("female", []))
    every = males + females
    if not every:
        return {s.id: default for s in speakers}
    used: list[str] = []
    out: dict[int, str | None] = {}
    for s in speakers:
        free_m = [v for v in males if v not in used]
        free_f = [v for v in females if v not in used]
        if s.gender == "male":
            order = free_m + free_f
        elif s.gender == "female":
            order = free_f + free_m
        else:
            order = free_m + free_f if len(free_m) >= len(free_f) else free_f + free_m
        if order:
            choice = order[0]
        else:
            mine = {"male": males, "female": females}.get(s.gender) or every
            choice = min(mine, key=lambda v: (used.count(v), mine.index(v)))
        used.append(choice)
        out[s.id] = choice
    return out


# --------------------------------------------------------------------------
# The models
# --------------------------------------------------------------------------

class SpeakerEmbedder:
    """WavLM x-vectors for clips of 16 kHz mono audio."""

    def __init__(self):
        self._model = None
        self._extractor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except ImportError as exc:  # pragma: no cover
            raise DependencyError(
                "different voices per speaker need 'torch' and 'transformers' "
                "(pip install -r requirements.txt)") from exc
        import warnings

        from .hf import from_pretrained

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._extractor = from_pretrained(AutoFeatureExtractor, MODEL_ID)
                self._model = from_pretrained(WavLMForXVector, MODEL_ID).eval()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"could not load the speaker model {MODEL_ID!r}: {exc}") from exc

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.empty((0, 512))
        import warnings

        import torch

        self._load()
        out = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for clip in clips:
                inputs = self._extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                out.append(self._model(**inputs).embeddings[0].numpy())
        return np.stack(out)


class GenderScorer:
    """Male-minus-female speech logit from the AudioSet classifier (AST)."""

    def __init__(self, tagger=None):
        self._tagger = tagger

    def score(self, clips: list[np.ndarray]) -> list[float]:
        import torch

        if self._tagger is None:
            from .sound_events import SoundTagger

            self._tagger = SoundTagger()
        model, extractor = self._tagger._load()
        device = self._tagger._device
        ids = model.config.label2id
        male, female = ids["Male speech, man speaking"], ids["Female speech, woman speaking"]
        out = []
        with torch.no_grad():
            for clip in clips:
                inputs = extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits[0]
                out.append(float(logits[male] - logits[female]))
        return out


def label_speakers(samples: np.ndarray, segments: list[tuple[float, float]],
                   embedder=None, scorer=None) -> tuple[list[int], list[Speaker]]:
    """Speaker id for each ``(start, end)`` line, and the speakers found.

    Speaker ids are ordered by how much each one talks (0 = the most).
    """
    if not segments:
        return [], []
    embedder = embedder or SpeakerEmbedder()
    scorer = scorer or GenderScorer()

    def clip(i: int) -> np.ndarray:
        start, end = segments[i]
        a = max(0, int(start * SAMPLE_RATE))
        b = min(len(samples), int(min(end, start + MAX_EMBED_SECONDS) * SAMPLE_RATE))
        return np.asarray(samples[a:b], dtype=np.float32)

    length = [e - s for s, e in segments]
    long_idx = [i for i in range(len(segments)) if length[i] >= MIN_EMBED_SECONDS]
    raw = np.zeros(len(segments), dtype=int)
    if long_idx:
        emb = embedder.embed([clip(i) for i in long_idx])
        labels = cluster(emb)
        raw[long_idx] = labels
        cents = _centroids(_unit(emb), labels)
        medium = [i for i in range(len(segments))
                  if MIN_TRY_SECONDS <= length[i] < MIN_EMBED_SECONDS]
        for i, vec in zip(medium, embedder.embed([clip(i) for i in medium])):
            raw[i] = nearest(vec, cents)
        for i in range(len(segments)):
            if length[i] < MIN_TRY_SECONDS:   # too short to tell: as the line before
                raw[i] = raw[i - 1] if i > 0 else 0

    # Order speakers by talk time.
    count = int(raw.max()) + 1
    seconds = np.bincount(raw, weights=length, minlength=count)
    order = np.argsort(-seconds, kind="stable")
    rank = np.empty(count, dtype=int)
    rank[order] = np.arange(count)

    speakers = []
    for new_id, k in enumerate(order):
        mine = [i for i in range(len(segments)) if raw[i] == k]
        scored = sorted((i for i in mine if length[i] >= MIN_EMBED_SECONDS),
                        key=lambda i: -length[i])[:GENDER_LINES]
        scores = scorer.score([clip(i) for i in scored]) if scored else []
        score = round(float(np.mean(scores)), 2) if scores else None
        speakers.append(Speaker(id=new_id, gender=gender_of(score), gender_score=score,
                                lines=len(mine), seconds=round(float(seconds[k]), 2)))
    return [int(rank[k]) for k in raw], speakers
