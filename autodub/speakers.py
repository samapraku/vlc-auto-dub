"""Tell the speakers in a film apart so each can get a voice of their own.

1. Lines are joined into speaking turns: a line that doesn't end a sentence and
   is followed closely by the next one is the same person carrying on
   ("We got dressed to the nines / since it's a once-in-a-lifetime event."),
   so the whole turn gets one voice. Longer turns also give steadier
   voiceprints.
2. Each turn is called female or male by a model trained for exactly that
   (audeering's wav2vec2 age/gender model). Turns it is sure about are split
   by gender before grouping, so a group can't mix a man and a woman.
3. Within each gender, turns are grouped by voiceprint (3D-Speaker's CAM++
   speaker model): average-linkage clustering on cosine distance. Groups of
   fewer than ``MIN_TURNS`` turns join the nearest larger group of the same
   gender. Turns the gender model isn't sure about, and short ones, join the
   nearest group afterwards.
4. Speakers take voices from the language's pool (see ``voices.py``), the one
   with the most dialogue first.

Tuned on a Korean TV drama episode (848 turns) and a 4-minute English phone
recording. On the drama, the two halves of the same turn got the same voice
87% of the time (the first version of this module, WavLM voiceprints at
distance 0.2 with AudioSet's male/female classes: ~70%), and 14 turns landed
with the wrong gender (before: ~37). The same person's turns sit 0.3-0.7 apart
across scenes, different people mostly further, hence ``SAME_SPEAKER = 0.8``;
0.85 merged the drama's two main men into one voice.

Models, downloaded once:
- speaker: 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced (28 MB,
  Apache-2.0), from the sherpa-onnx model releases, run with sherpa-onnx.
- gender: audeering/wav2vec2-large-robust-6-ft-age-gender (363 MB,
  CC-BY-NC-SA 4.0: non-commercial use), from the Hugging Face hub.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import DependencyError, ProviderError

SAMPLE_RATE = 16000
SPEAKER_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced"
SPEAKER_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     f"speaker-recongition-models/{SPEAKER_MODEL}.onnx")
GENDER_MODEL = "audeering/wav2vec2-large-robust-6-ft-age-gender"

MIN_EMBED_SECONDS = 1.0     # shorter turns join the nearest speaker afterwards
MIN_TRY_SECONDS = 0.5       # shorter still: same speaker as the turn before
MAX_CLIP_SECONDS = 10.0     # enough voice for both models; keeps them fast
TURN_GAP = 1.0              # max pause inside one speaking turn, seconds
SAME_SPEAKER = 0.8          # cosine distance (CAM++) below which turns merge
MIN_TURNS = 5               # smaller groups join the nearest larger one
MAX_PER_GENDER = 6          # English has 6 voices of each
SURE = 0.7                  # P(female) >= SURE: female; <= 1 - SURE: male

_SENTENCE_END = re.compile(r"""[.!?…♪"”')\]]\s*$""")


@dataclass(frozen=True)
class Speaker:
    id: int                 # 0 = the speaker with the most dialogue
    gender: str             # "male" | "female" | "unknown"
    female: float | None    # mean P(female) over the speaker's turns
    lines: int
    seconds: float


# --------------------------------------------------------------------------
# Speaking turns
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    return text.replace("‎", "").replace("‏", "").strip()


def turns(lines: list[tuple[float, float, str]], max_gap: float = TURN_GAP) -> list[list[int]]:
    """Indices of ``lines`` grouped into speaking turns, in order."""
    if not lines:
        return []
    out = [[0]]
    for i in range(1, len(lines)):
        prev, cur = _clean(lines[i - 1][2]), _clean(lines[i][2])
        carries_on = (lines[i][0] - lines[i - 1][1] < max_gap
                      and prev and not _SENTENCE_END.search(prev)
                      and not prev.startswith("-") and not cur.startswith("-"))
        if carries_on:
            out[-1].append(i)
        else:
            out.append([i])
    return out


# --------------------------------------------------------------------------
# Grouping turns into speakers
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
            min_size: int = MIN_TURNS, max_speakers: int = MAX_PER_GENDER) -> np.ndarray:
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
        small = [k for k in range(len(sizes)) if sizes[k] < min_size]
        big = [k for k in range(len(sizes)) if sizes[k] >= min_size]
        if small and big:
            labels = _merge_into_nearest(e, labels, min(small, key=lambda k: sizes[k]), big)
        elif len(sizes) > max_speakers:
            k = int(np.argmin(sizes))
            labels = _merge_into_nearest(e, labels, k, [j for j in range(len(sizes)) if j != k])
        else:
            return labels


def gender_of(female: float | None, sure: float = SURE) -> str:
    if female is None:
        return "unknown"
    if female >= sure:
        return "female"
    if female <= 1 - sure:
        return "male"
    return "unknown"


# --------------------------------------------------------------------------
# Voice assignment
# --------------------------------------------------------------------------

def assign_voices(speakers: list[Speaker], pool: dict[str, list[str]],
                  default: str | None) -> dict[int, str | None]:
    """Voice for each speaker id. ``pool`` maps "male"/"female" to voices.

    Speakers come in order of how much they talk. Each takes the next unused
    voice of their gender; a speaker of unknown gender takes one from whichever
    gender has the most voices left. When a gender runs out, the other's voices
    are used, and only when every voice is taken is one reused (the least-used
    one of the right gender).
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

def model_dir() -> Path:
    return Path(os.environ.get("AUTODUB_SPEAKER_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vlc-auto-dub", "speaker"))


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        part.replace(dest)
    except Exception as exc:  # noqa: BLE001
        part.unlink(missing_ok=True)
        raise ProviderError(f"could not download the speaker model from {url}: {exc}") from exc


class SpeakerEmbedder:
    """CAM++ voiceprints (sherpa-onnx) for clips of 16 kHz mono audio."""

    def __init__(self):
        self._extractor = None

    def _load(self):
        if self._extractor is not None:
            return
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise DependencyError("a voice for each speaker needs 'sherpa-onnx' "
                                  "(pip install -r requirements.txt)") from exc
        path = model_dir() / f"{SPEAKER_MODEL}.onnx"
        if not path.exists():
            from .progress import progress

            progress("downloading the speaker model (28 MB, once)")
            _download(SPEAKER_MODEL_URL, path)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path), num_threads=max(1, min(4, os.cpu_count() or 1)))
        if not config.validate():
            raise ProviderError(f"speaker model {path} is not usable; delete it to re-download")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        self._load()
        out = []
        for clip in clips:
            stream = self._extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.asarray(clip, dtype=np.float32))
            stream.input_finished()
            out.append(np.asarray(self._extractor.compute(stream)))
        return np.stack(out) if out else np.empty((0, self._extractor.dim))


class GenderScorer:
    """P(female) for clips, from audeering's wav2vec2 age/gender model."""

    def __init__(self):
        self._model = None
        self._extractor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from torch import nn
            from transformers import Wav2Vec2FeatureExtractor
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2Model,
                Wav2Vec2PreTrainedModel,
            )
        except ImportError as exc:  # pragma: no cover
            raise DependencyError("a voice for each speaker needs 'torch' and 'transformers' "
                                  "(pip install -r requirements.txt)") from exc
        import warnings

        from .hf import from_pretrained

        class Head(nn.Module):  # the model card's ModelHead
            def __init__(self, config, labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = Head(config, 1)
                self.gender = Head(config, 3)   # female, male, child (config id2label)
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return torch.softmax(self.gender(hidden), dim=1)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._extractor = from_pretrained(Wav2Vec2FeatureExtractor, GENDER_MODEL)
                self._model = from_pretrained(AgeGenderModel, GENDER_MODEL).eval()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"could not load the gender model {GENDER_MODEL!r}: {exc}") from exc

    def female(self, clips: list[np.ndarray]) -> list[float]:
        import torch

        self._load()
        out = []
        with torch.inference_mode():
            for clip in clips:
                values = self._extractor(clip, sampling_rate=SAMPLE_RATE,
                                         return_tensors="pt").input_values
                female, _male, child = self._model(values)[0].tolist()
                out.append(female + child)   # a child voice suits a female voice best
        return out


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

def label_speakers(samples: np.ndarray, lines: list[tuple[float, float, str]],
                   embedder=None, scorer=None) -> tuple[list[int], list[Speaker]]:
    """Speaker id for each ``(start, end, text)`` line, and the speakers found.

    Speaker ids are ordered by how much each one talks (0 = the most).
    """
    if not lines:
        return [], []
    embedder = embedder or SpeakerEmbedder()
    scorer = scorer or GenderScorer()
    groups = turns(lines)
    span = [(lines[t[0]][0], lines[t[-1]][1]) for t in groups]
    length = np.array([e - s for s, e in span])

    def clip(k: int) -> np.ndarray:
        start, end = span[k]
        a = max(0, int(start * SAMPLE_RATE))
        b = min(len(samples), int(min(end, start + MAX_CLIP_SECONDS) * SAMPLE_RATE))
        return np.asarray(samples[a:b], dtype=np.float32)

    usable = [k for k in range(len(groups)) if length[k] >= MIN_TRY_SECONDS]
    emb = {k: v for k, v in zip(usable, embedder.embed([clip(k) for k in usable]))}
    female = {k: p for k, p in zip(usable, scorer.female([clip(k) for k in usable]))}

    raw = np.full(len(groups), -1)
    gender_of_group: list[str] = []
    long_turns = [k for k in usable if length[k] >= MIN_EMBED_SECONDS]
    by_gender = [("female", [k for k in long_turns if female[k] >= SURE]),
                 ("male", [k for k in long_turns if female[k] <= 1 - SURE])]
    if not any(members for _, members in by_gender):
        by_gender = [("unknown", long_turns)]
    for gender, members in by_gender:
        if not members:
            continue
        labels = cluster(np.stack([emb[k] for k in members]))
        raw[members] = labels + len(gender_of_group)
        gender_of_group += [gender] * (int(labels.max()) + 1)

    placed = [k for k in range(len(groups)) if raw[k] >= 0]
    if placed:
        cents = _centroids(_unit(np.stack([emb[k] for k in placed])), raw[placed])
        for k in usable:                      # unsure gender, or 0.5-1 s long
            if raw[k] < 0:
                raw[k] = int(np.argmax(cents @ _unit(emb[k])))
    for k in range(len(groups)):              # too short to tell: as the turn before
        if raw[k] < 0:
            raw[k] = raw[k - 1] if k > 0 and raw[k - 1] >= 0 else 0
    if not gender_of_group:
        gender_of_group = ["unknown"]

    # Order speakers by talk time.
    count = int(raw.max()) + 1
    seconds = np.bincount(raw, weights=length, minlength=count)
    order = np.argsort(-seconds, kind="stable")
    rank = np.empty(count, dtype=int)
    rank[order] = np.arange(count)

    line_ids = [0] * len(lines)
    for k, members in enumerate(groups):
        for i in members:
            line_ids[i] = int(rank[raw[k]])
    speakers = []
    for new_id, g in enumerate(order):
        mine = [k for k in range(len(groups)) if raw[k] == g]
        probs = [female[k] for k in mine if k in female]
        mean = round(float(np.mean(probs)), 2) if probs else None
        gender = gender_of_group[g] if gender_of_group[g] != "unknown" else gender_of(mean)
        speakers.append(Speaker(id=new_id, gender=gender, female=mean,
                                lines=sum(len(groups[k]) for k in mine),
                                seconds=round(float(seconds[g]), 2)))
    return line_ids, speakers
