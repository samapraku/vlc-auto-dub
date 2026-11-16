"""Loading Hugging Face models quietly, from the local cache when possible."""
from __future__ import annotations

import logging
import os


def quiet() -> None:
    """Keep the job log readable: no download bars or hub notices.

    The hub answers every request without a token with "You are sending
    unauthenticated requests to the HF Hub"; it is advice, not an error.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # torch logs "NOTE: Redirects are currently not supported in MacOs." when
    # transformers imports its distributed module; nothing here uses it.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
        logging.ERROR)
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except ImportError:  # pragma: no cover
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:  # pragma: no cover
        pass


def from_pretrained(cls, model_id: str):
    """``cls.from_pretrained(model_id)``, from the cache without touching the
    network when the model is already downloaded; online only the first time."""
    quiet()
    try:
        return cls.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError):
        return cls.from_pretrained(model_id)
