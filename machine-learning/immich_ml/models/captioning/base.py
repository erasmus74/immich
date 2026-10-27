from typing import Any, Protocol, cast

from huggingface_hub import snapshot_download
from PIL import Image

from immich_ml.config import log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import decode_pil
from immich_ml.schemas import CaptionOptions, ModelSource, ModelTask, ModelType


class CaptionEngine(Protocol):
    """What a loaded captioning engine provides, whatever backend it runs on."""

    def generate(self, image: Image.Image, options: CaptionOptions) -> str: ...


class Captioner(InferenceModel[CaptionOptions]):
    """Base for image captioning models.

    The engine is chosen from the repo name (pipeline.captioning): 'joy' runs JoyCaptionCaptioner,
    'florence' is refused, and anything else, Qwen repos included, runs QwenVLCaptioner. Caption models
    name plain HuggingFace repos, so the full id is kept in `repo_id` before cleaning strips the org
    from `model_name`.
    """

    depends = []
    identity = (ModelType.CAPTION, ModelTask.CAPTION)
    sources = (ModelSource.HUGGINGFACE,)

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        self.repo_id = model_name if "/" in model_name else f"{settings.model_organization}/{model_name}"
        super().__init__(model_name, **model_kwargs)

    def build(self) -> None:
        self.load()  # the engines have nothing to warm

    def download(self) -> None:
        if self.cached:
            return
        log.info(f"Downloading caption model '{self.repo_id}' to {self.model_dir}. This may take a while.")
        snapshot_download(
            self.repo_id,
            revision=settings.model_revision,
            cache_dir=self.cache_dir,
            local_dir=self.model_dir,
        )
        if not self.cached:
            raise FileNotFoundError(f"Model file not found: {self.model_dir}")

    @property
    def cached(self) -> bool:
        return any(self.model_dir.glob("*"))

    def _predict(self, inputs: Image.Image | bytes, options: CaptionOptions) -> str:
        engine = cast(CaptionEngine, self.session)
        return engine.generate(decode_pil(inputs), options)
