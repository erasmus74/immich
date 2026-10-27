"""Qwen captioning.

The default caption engine: repos like 'Qwen/Qwen3-VL-2B-Instruct' load natively through
transformers' image-text-to-text classes, so no remote code runs.
"""

from typing import Any

from PIL import Image

from immich_ml.config import log
from immich_ml.models.captioning.base import Captioner
from immich_ml.schemas import CaptionOptions

USER_PROMPT = (
    "Describe this image factually and in detail for a photo library's search index: the subject or "
    "subjects and what they are doing, their clothing and appearance, the setting and background, the "
    "season or time of day, the lighting, and the photographic style. Use concrete, searchable wording; "
    "do not guess identities, moralize, or add commentary."
)


class _Engine:
    """A loaded Qwen image-text-to-text model with its processor."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def generate(self, image: Image.Image, options: CaptionOptions) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": options.prompt or USER_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        tensors = {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value for name, value in inputs.items()
        }
        with torch.no_grad():
            generated = self.model.generate(**tensors, max_new_tokens=options.max_new_tokens, do_sample=False)
        new_tokens = generated[:, tensors["input_ids"].shape[1] :]
        caption: str = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return caption.strip()


class QwenVLCaptioner(Captioner):
    """Captioner for Qwen vision-language repos, the default engine for any repo not named for another.

    The configured repo loads as-is through the processor and image-text-to-text classes, fp16 on CUDA
    and fp32 elsewhere; once downloaded, its local snapshot is what loads.
    """

    def _load(self) -> Any:  # the engine is not an ORT session
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        source = self.model_dir if any(self.model_dir.glob("*")) else self.repo_id
        log.info(f"Loading caption model '{self.repo_id}' on {device}")
        # transformers exports these lazily, which mypy sees as untyped calls, so they are used through Any
        auto_processor: Any = AutoProcessor
        auto_model: Any = AutoModelForImageTextToText
        processor = auto_processor.from_pretrained(source)
        model = auto_model.from_pretrained(source, dtype=dtype).to(device).eval()
        return _Engine(model, processor, device)
