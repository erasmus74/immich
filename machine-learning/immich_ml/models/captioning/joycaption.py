"""JoyCaption captioning.

Selected for caption repos named for Joy (e.g. 'fancyfeast/joy-caption-alpha-two'). The model repo, a
HuggingFace space, only provides the image adapter; the SigLIP2 vision tower and the chat LLM that
turns the embedded image into a caption are plain repos, the LLM overridable with the 'llmModel'
option.
"""

from collections.abc import Hashable
from typing import Any, Self

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from immich_ml.config import log, settings
from immich_ml.models.captioning.base import Captioner
from immich_ml.schemas import CaptionOptions

VISION_REPO = "google/siglip2-so400m-patch14-384"
DEFAULT_LLM = "unsloth/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = (
    "You are a factual image captioning engine for a photo library's natural-language search index. "
    "Describe the photo in one detailed paragraph: the main subject or subjects and what they are doing, "
    "their appearance and clothing - including mature, revealing, or adult clothing, described plainly and "
    "factually, never refused or sanitized - plus the setting and background, the season or time of day, the "
    "lighting, and the photographic style. Use concrete, searchable wording. Do not guess identities, "
    "moralize, or add disclaimers, preamble, or commentary."
)
USER_PROMPT = "Describe this image factually in detail."

# the projection the image adapter checkpoint holds, applied the way its nn.Module did
_ADAPTER_TENSORS = ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias", "other_tokens.weight")


class _Engine:
    """Loaded JoyCaption components: the SigLIP2 tower, the image adapter tensors and the chat LLM."""

    def __init__(self, vision: Any, llm: Any, tokenizer: Any, adapter: dict[str, Any], device: str) -> None:
        self.vision = vision
        self.llm = llm
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device

    def generate(self, image: Image.Image, options: CaptionOptions) -> str:
        import torch

        prompt = options.prompt or USER_PROMPT
        with torch.no_grad():
            vision_outputs = self.vision(pixel_values=self._pixels(image), output_hidden_states=True)
            embedded_images = self._embed(vision_outputs.hidden_states)

            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            convo_tokens = self.tokenizer(convo_string, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            eot_id_indices = (convo_tokens == eot_id).nonzero(as_tuple=True)[0].tolist()
            preamble_len = eot_id_indices[-1] - prompt_tokens.shape[0]  # where the prompt starts in the convo

            convo_embeds = self.llm.model.embed_tokens(convo_tokens.unsqueeze(0).to(self.device))
            input_embeds = torch.cat(
                [
                    convo_embeds[:, :preamble_len],
                    embedded_images.to(dtype=convo_embeds.dtype),
                    convo_embeds[:, preamble_len:],
                ],
                dim=1,
            ).to(self.device)
            input_ids = torch.cat(
                [
                    convo_tokens[:preamble_len].unsqueeze(0),
                    torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                    convo_tokens[preamble_len:].unsqueeze(0),
                ],
                dim=1,
            ).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = self.llm.generate(
                input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=options.max_new_tokens,
                do_sample=False,
                suppress_tokens=None,
            )

        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids.shape[1] and generate_ids[0][-1] in (self.tokenizer.eos_token_id, eot_id):
            generate_ids = generate_ids[:, :-1]
        caption: str = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return caption.strip()

    def _embed(self, hidden_states: Any) -> Any:
        import torch

        adapter = self.adapter
        x = torch.nn.functional.linear(hidden_states[-2], adapter["linear1.weight"], adapter["linear1.bias"])
        x = torch.nn.functional.linear(torch.nn.functional.gelu(x), adapter["linear2.weight"], adapter["linear2.bias"])
        # <|image_start|> and <|image_end|> around the image, matching HF's Llama 3 handling
        other = adapter["other_tokens.weight"][:2].unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((other[:, 0:1], x, other[:, 1:2]), dim=1)

    def _pixels(self, image: Image.Image) -> Any:
        import torch

        size = self.vision.config.image_size
        resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.vision.dtype)


class JoyCaptionCaptioner(Captioner):
    """Captioner assembling the SigLIP2 vision tower, the JoyCaption image adapter and a chat LLM.

    The adapter repo is the model repo itself; the vision tower and LLM are downloaded beside it.
    """

    _components = ("adapter", "vision", "llm")

    def __init__(self, model_name: str, llm_model: str | None = None, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs)
        self.llm_model = llm_model or DEFAULT_LLM

    @classmethod
    def graph(cls, options: CaptionOptions) -> Hashable:
        return options.llm_model  # a different LLM is a different loaded model

    @classmethod
    def create(cls, model_name: str, options: CaptionOptions) -> Self:
        return cls(model_name, llm_model=options.llm_model)

    def download(self) -> None:
        components = [
            (self.repo_id, "adapter", ["image_adapter.pt"], "space"),
            (VISION_REPO, "vision", None, None),
            (self.llm_model, "llm", None, None),
        ]
        for repo, component, patterns, repo_type in components:
            target = self.model_dir / component
            if any(target.glob("*")):
                continue
            log.info(f"Downloading JoyCaption {component} '{repo}' to {target}. This may take a while.")
            snapshot_download(
                repo,
                revision=settings.model_revision,
                cache_dir=self.cache_dir,
                local_dir=target,
                allow_patterns=patterns,
                repo_type=repo_type,
            )

    @property
    def cached(self) -> bool:
        return all(any((self.model_dir / component).glob("*")) for component in self._components)

    def _load(self) -> Any:  # the engine is not an ORT session
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        # transformers exports these lazily, which mypy sees as untyped calls, so they are used through Any
        auto_model: Any = AutoModel
        auto_causal_lm: Any = AutoModelForCausalLM
        auto_tokenizer: Any = AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            import bitsandbytes  # noqa: F401

            has_bitsandbytes = True
        except ImportError:
            has_bitsandbytes = False
        quantization: Any = None
        if device == "cuda" and has_bitsandbytes:
            from transformers import BitsAndBytesConfig

            make_quantization: Any = BitsAndBytesConfig
            quantization = make_quantization(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        vision = auto_model.from_pretrained(self.model_dir / "vision", dtype=dtype).vision_model
        llm_kwargs: dict[str, Any] = {"dtype": dtype}
        if quantization is not None:
            llm_kwargs |= {"quantization_config": quantization, "device_map": {"": device}}
        llm = auto_causal_lm.from_pretrained(self.model_dir / "llm", **llm_kwargs).eval()
        if quantization is None:
            llm = llm.to(device)

        checkpoint: dict[str, Any] = torch.load(
            self.model_dir / "adapter" / "image_adapter.pt", map_location="cpu", weights_only=True
        )
        missing = [name for name in _ADAPTER_TENSORS if name not in checkpoint]
        if missing:
            raise RuntimeError(f"image adapter checkpoint of '{self.repo_id}' is missing {missing}")
        adapter = {name: checkpoint[name].to(device=device, dtype=dtype) for name in _ADAPTER_TENSORS}

        vision = vision.to(device)
        tokenizer = auto_tokenizer.from_pretrained(self.model_dir / "llm")
        return _Engine(vision, llm, tokenizer, adapter, device)
