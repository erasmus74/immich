"""Florence-2 cannot run on this build.

Its remote code only works with transformers 4.x, which the core huggingface-hub>=1.0 pin rules out,
and transformers 5's native support does not accept the published Florence-2 checkpoints either. The
class the engine selection maps 'florence' repos to therefore refuses with a clear message instead.
"""

from typing import Any

from immich_ml.models.captioning.base import Captioner

MESSAGE = (
    "Florence-2 requires transformers 4.x, which is incompatible with this build; "
    "use a Qwen/Qwen3-VL or joy-caption model"
)


class Florence2Captioner(Captioner):
    """Refuses to run: a Florence-2 repo needs a machine-learning instance on transformers 4.x."""

    def load(self) -> None:
        raise RuntimeError(MESSAGE)

    def _load(self) -> Any:
        raise RuntimeError(MESSAGE)
