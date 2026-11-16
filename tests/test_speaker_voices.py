"""A different voice for each speaker (--speaker-voices).

The speaker and gender models are replaced by small fakes here, so these tests
cover the grouping, ordering, voice sharing and wiring; they don't judge how
well WavLM tells real voices apart.
"""
from __future__ import annotations

import numpy as np
import pytest

import main as cli
from autodub import pipeline, speakers
from autodub.config import build_config
from autodub.errors import ProviderError
from autodub.providers.tts_fake import FakeTTS
from autodub.providers.tts_piper import PiperTTS, split_voice
from autodub.speakers import Speaker, assign_voices, cluster, gender_of, label_speakers
from autodub.voices import PIPER_SPEAKER_POOL, speaker_pool

FAKES = ["--stt", "fake", "--tts", "fake", "--translate", "fake"]
RNG = np.random.default_rng(7)


def voices_around(center: np.ndarray, n: int, noise: float = 0.05) -> np.ndarray:
    return center + noise * RNG.standard_normal((n, len(center)))


def axis(i: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim)
    v[i] = 1.0
    return v


# --- cluster ---------------------------------------------------------------

def test_cluster_finds_each_speaker():
    e = np.vstack([voices_around(axis(0), 6), voices_around(axis(1), 5),
                   voices_around(axis(2), 5)])
    labels = cluster(e)
    assert len(set(labels)) == 3
    assert len(set(labels[:6])) == len(set(labels[6:11])) == len(set(labels[11:])) == 1


def test_fragments_join_the_nearest_real_speaker():
    a, b = axis(0), axis(1)
    near_b = axis(2) + 0.2 * b               # its own group at 0.8, but nearer b than a
    e = np.vstack([voices_around(a, 5), voices_around(b, 5), voices_around(near_b, 2, 0.01)])
    labels = cluster(e)
    assert len(set(labels)) == 2
    assert set(labels[10:]) == {labels[5]}


def test_cluster_caps_the_speaker_count():
    e = np.vstack([voices_around(axis(i), 5) for i in range(7)])
    assert len(set(cluster(e, max_speakers=4))) == 4
    assert len(set(cluster(e))) == speakers.MAX_PER_GENDER


def test_cluster_edge_cases():
    assert list(cluster(np.empty((0, 4)))) == []
    assert list(cluster(axis(0)[None])) == [0]
    # Too few lines for any group to count as a real speaker: kept as found.
    assert len(set(cluster(np.vstack([axis(0), axis(1)])))) == 2


# --- speaking turns --------------------------------------------------------

def test_a_sentence_carried_across_lines_is_one_turn():
    lines = [(0, 1, "We got dressed to the nines"), (1.1, 2, "since it was a big day."),
             (2.2, 3, "\u200eYou could've put this cap on,"), (3.1, 4, "\u200ethen let her try it."),
             (4.1, 5, "Really?"), (5.1, 6, "Yes"), (8, 9, "and that's it."),
             (9.1, 10, "-Go ahead."), (10.1, 11, "-Yes")]
    assert speakers.turns(lines) == [[0, 1], [2, 3], [4], [5], [6], [7], [8]]
    assert speakers.turns([]) == []


# --- gender and voice sharing ----------------------------------------------

def test_gender_needs_a_clear_margin():
    assert gender_of(0.9) == "female" and gender_of(0.1) == "male"
    assert gender_of(0.5) == gender_of(0.69) == gender_of(None) == "unknown"


def spk(i: int, gender: str) -> Speaker:
    return Speaker(id=i, gender=gender, female=None, lines=1, seconds=1.0)


POOL = {"male": ["m1", "m2"], "female": ["f1", "f2"]}


def test_each_speaker_gets_a_voice_of_their_gender():
    got = assign_voices([spk(0, "female"), spk(1, "male"), spk(2, "female")], POOL, "d")
    assert got == {0: "f1", 1: "m1", 2: "f2"}


def test_unknown_speakers_alternate_so_they_sound_different():
    got = assign_voices([spk(0, "male"), spk(1, "unknown"), spk(2, "unknown")], POOL, "d")
    assert got == {0: "m1", 1: "f1", 2: "m2"}


def test_a_missing_gender_borrows_and_a_full_pool_reuses():
    only_male = {"male": ["m1"], "female": []}
    assert assign_voices([spk(0, "female"), spk(1, "female")], only_male, "d") == \
        {0: "m1", 1: "m1"}
    got = assign_voices([spk(i, "male") for i in range(5)] + [spk(5, "female")], POOL, "d")
    assert [got[i] for i in range(4)] == ["m1", "m2", "f1", "f2"]  # distinct while they last
    assert got[4] == "m1" and got[5] == "f1"                       # then least-used, own gender
    assert assign_voices([spk(0, "male")], {}, "default") == {0: "default"}


# --- label_speakers --------------------------------------------------------

class FakeEmbedder:
    """Voiceprint from the clip's level: each fake speaker talks at its own level."""
    LEVELS = {0.1: axis(0), 0.2: axis(1), 0.3: axis(2), 0.4: axis(3)}

    def embed(self, clips):
        out = []
        for c in clips:
            level = min(self.LEVELS, key=lambda lv: abs(lv - float(np.mean(c))))
            out.append(self.LEVELS[level] + 0.01 * RNG.standard_normal(16))
        return np.stack(out) if out else np.empty((0, 16))


class FakeScorer:
    """0.1 and 0.4 are men, 0.2 a woman, 0.3 unclear."""
    def female(self, clips):
        return [{0.1: 0.0, 0.4: 0.05, 0.2: 1.0}.get(round(float(np.mean(c)), 1), 0.5)
                for c in clips]


def timeline(lines):
    """Samples where each (start, end, level[, text]) line is a constant ``level``."""
    rate = speakers.SAMPLE_RATE
    total = max(line[1] for line in lines) + 1
    x = np.zeros(int(total * rate), dtype=np.float32)
    for line in lines:
        x[int(line[0] * rate):int(line[1] * rate)] = line[2]
    return x, [(line[0], line[1], line[3] if len(line) > 3 else "Line.") for line in lines]


def label(lines):
    x, segs = timeline(lines)
    return label_speakers(x, segs, FakeEmbedder(), FakeScorer())


def test_label_speakers_orders_by_talk_time_and_names_genders():
    ids, found = label([(0, 2, 0.2), (3, 5, 0.1), (6, 9, 0.1), (10, 12, 0.2), (13, 15, 0.1),
                        (16, 18, 0.2), (19, 20.5, 0.1)])
    # 0.1 talks 8.5 s, 0.2 6 s.
    assert [(s.gender, s.lines, s.female) for s in found] == [("male", 4, 0.0),
                                                             ("female", 3, 1.0)]
    assert ids == [1, 0, 0, 1, 0, 1, 0]


def test_men_and_women_are_never_grouped_together():
    # 0.1 and 0.4 are both men; a woman with a voiceprint like 0.1's stays apart.
    class LookAlike(FakeEmbedder):
        LEVELS = {0.1: axis(0), 0.2: axis(0), 0.4: axis(1)}
    lines = [(i * 3, i * 3 + 2, lv) for i, lv in enumerate([0.1, 0.2, 0.4] * 5)]
    x, segs = timeline(lines)
    ids, found = label_speakers(x, segs, LookAlike(), FakeScorer())
    assert sorted(s.gender for s in found) == ["female", "male", "male"]
    assert len({ids[i] for i in range(1, 15, 3)}) == 1                     # the woman
    assert {ids[i] for i in range(1, 15, 3)}.isdisjoint(ids[i] for i in range(0, 15, 3))


def test_unsure_turns_join_an_existing_speaker():
    ids, found = label([(i * 3, i * 3 + 2, 0.1) for i in range(5)] + [(15, 17, 0.3)])
    assert len(found) == 1 and ids == [0] * 6 and found[0].gender == "male"


def test_a_turn_keeps_one_voice_even_when_the_voiceprints_differ():
    ids, _found = label([(0, 2, 0.1), (3, 5, 0.1), (6, 8, 0.2), (9, 11, 0.2),
                         (12, 14, 0.1, "I was going to"), (14.3, 16, 0.2, "say that.")])
    assert ids[4] == ids[5]


def test_short_turns_join_a_speaker():
    ids, found = label([(0, 2, 0.1), (3, 5, 0.1), (6, 8, 0.1), (9, 11, 0.2), (12, 14, 0.2),
                        (15, 17, 0.2), (18, 18.6, 0.2), (19, 19.2, 0.1)])
    assert len(found) == 2
    assert ids[6] == ids[3]              # 0.6 s: matched by its own voiceprint
    assert ids[7] == ids[6]              # 0.2 s: too short, same as the turn before


def test_label_speakers_with_only_short_lines_or_none():
    ids, found = label([(0, 0.3, 0.1), (1, 1.3, 0.2)])
    assert ids == [0, 0] and len(found) == 1 and found[0].gender == "unknown"
    assert label_speakers(np.zeros(10), [], FakeEmbedder(), FakeScorer()) == ([], [])


# --- voice pools -----------------------------------------------------------

def test_speaker_pools():
    pool, warning = speaker_pool("piper", "en")
    assert warning is None and len(pool["male"]) >= 4 and len(pool["female"]) >= 4
    assert speaker_pool("piper", None)[0] is PIPER_SPEAKER_POOL["en"]
    pool, warning = speaker_pool("piper", "tr")
    assert pool["male"] and "no female voice" in warning
    assert speaker_pool("piper", "xx") == (
        {}, "no speaker voices for language 'xx'; every speaker uses one voice")
    assert speaker_pool("espeak", "en")[0] == {}
    for lang, p in PIPER_SPEAKER_POOL.items():
        names = p["male"] + p["female"]
        assert len(names) == len(set(names)), lang
        assert all(split_voice(v)[0].startswith(lang) for v in names), lang


# --- piper speakers --------------------------------------------------------

def test_split_voice():
    assert split_voice("en_US-arctic-medium#slt") == ("en_US-arctic-medium", "slt")
    assert split_voice("en_US-lessac-medium") == ("en_US-lessac-medium", None)
    assert split_voice("x#") == ("x", None)


class _Config:
    def __init__(self, ids, n):
        self.speaker_id_map, self.num_speakers = ids, n


class _Model:
    def __init__(self, ids, n):
        self.config = _Config(ids, n)


def test_piper_speaker_ids():
    tts = PiperTTS()
    m = _Model({"slt": 2, "rms": 1}, 3)
    assert tts._speaker_id(m, "a", None) is None
    assert tts._speaker_id(m, "a", "slt") == 2
    assert tts._speaker_id(m, "a", "1") == 1
    with pytest.raises(ProviderError):
        tts._speaker_id(m, "a", "nobody")
    with pytest.raises(ProviderError):
        tts._speaker_id(_Model({}, 1), "a", "0")       # single-speaker model


def test_piper_keeps_every_loaded_model(monkeypatch):
    loads = []
    tts = PiperTTS()
    monkeypatch.setattr(tts, "_ensure_model", lambda name: loads.append(name) or name)
    import piper

    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda path: _Model({}, 1)))
    for name in ["a", "b", "a", "b"]:
        tts._load(name)
    assert loads == ["a", "b"]


# --- config and pipeline ---------------------------------------------------

def test_off_by_default_and_configurable(monkeypatch, tmp_path):
    assert build_config().speaker_voices is False
    assert build_config(speaker_voices=True).speaker_voices is True
    monkeypatch.setenv("AUTODUB_SPEAKER_VOICES", "1")
    assert build_config().speaker_voices is True
    cfg = tmp_path / "c.conf"
    cfg.write_text("speaker_voices = false\n")
    assert build_config(config_path=str(cfg)).speaker_voices is False
    assert cli.build_parser().parse_args(["x.mp4", "--in-place"]).speaker_voices is False


@pytest.fixture
def recorded_voices(monkeypatch):
    voices: list = []
    real = FakeTTS.synthesize

    def record(self, text, out_path, voice):
        voices.append(voice)
        real(self, text, out_path, voice)

    monkeypatch.setattr(FakeTTS, "synthesize", record)
    return voices


def test_each_line_is_spoken_by_its_speakers_voice(tmp_path, tone_video, monkeypatch,
                                                   recorded_voices, capsys):
    monkeypatch.setattr(pipeline, "speaker_pool", lambda provider, lang: (POOL, None))

    def fake_label(samples, segs):
        assert len(samples) > 0
        return [i % 2 for i in range(len(segs))], [Speaker(0, "female", -2.0, 2, 4.0),
                                                   Speaker(1, "male", 2.0, 1, 2.0)]

    monkeypatch.setattr(speakers, "label_speakers", fake_label)
    assert cli.main([tone_video, "--output-base", str(tmp_path / "o"), *FAKES,
                     "--speaker-voices", "--dub-only"]) == 0
    assert recorded_voices == ["f1", "m1", "f1"][:len(recorded_voices)]
    assert len(recorded_voices) >= 2
    out = capsys.readouterr().out
    assert "[autodub] speakers: 2" in out
    assert "#1 female, 2 lines, 4s -> f1" in out and "#2 male, 1 lines, 2s -> m1" in out


def test_without_a_pool_it_warns_and_uses_one_voice(tmp_path, tone_video, recorded_voices):
    config = build_config(stt="fake", tts="fake", translate="fake", speaker_voices=True,
                          dub_only=True)
    summary = pipeline.run(tone_video, str(tmp_path / "o"), config)
    assert "speakers" not in summary
    assert any("need the piper TTS" in w for w in summary["warnings"])
    assert len(set(recorded_voices)) == 1


def test_an_explicit_voice_wins(tmp_path, tone_video, monkeypatch, recorded_voices):
    monkeypatch.setattr(pipeline, "speaker_pool", lambda provider, lang: (POOL, None))
    monkeypatch.setattr(speakers, "label_speakers",
                        lambda *a: pytest.fail("speakers should not be analysed"))
    config = build_config(stt="fake", tts="fake", translate="fake", speaker_voices=True,
                          voice="mine", dub_only=True)
    summary = pipeline.run(tone_video, str(tmp_path / "o"), config)
    assert set(recorded_voices) == {"mine"}
    assert any("every speaker uses it" in w for w in summary["warnings"])


def test_subtitles_only_never_analyses_speakers(tmp_path, tone_video, monkeypatch):
    monkeypatch.setattr(pipeline, "speaker_pool", lambda provider, lang: (POOL, None))
    monkeypatch.setattr(speakers, "label_speakers",
                        lambda *a: pytest.fail("speakers should not be analysed"))
    config = build_config(stt="fake", tts="fake", translate="fake", speaker_voices=True,
                          subtitle_only=True)
    assert pipeline.run(tone_video, str(tmp_path / "o"), config)["mode"] == "subtitle-only"
