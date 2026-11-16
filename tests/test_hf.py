"""Hugging Face models load from the cache first, and the hub stays quiet."""
from __future__ import annotations

import logging

from autodub import hf


class Model:
    calls: list = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append(kwargs)
        if kwargs.get("local_files_only") and model_id == "not-cached":
            raise OSError("not in the cache")
        return model_id


def test_cached_models_load_without_the_network():
    Model.calls = []
    assert hf.from_pretrained(Model, "cached") == "cached"
    assert Model.calls == [{"local_files_only": True}]


def test_a_missing_model_is_downloaded_once():
    Model.calls = []
    assert hf.from_pretrained(Model, "not-cached") == "not-cached"
    assert Model.calls == [{"local_files_only": True}, {}]


def test_quiet_hides_hub_notices():
    from huggingface_hub.utils import logging as hub_logging

    hf.quiet()
    # The hub's "You are sending unauthenticated requests" is a WARNING.
    assert hub_logging.get_logger("huggingface_hub.utils._http").getEffectiveLevel() > logging.WARNING
    assert logging.getLogger(
        "torch.distributed.elastic.multiprocessing.redirects").getEffectiveLevel() > logging.WARNING
