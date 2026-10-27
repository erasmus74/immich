import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi import HTTPException
from PIL import Image
from pytest_mock import MockerFixture

from immich_ml.config import settings
from immich_ml.main import get_entries
from immich_ml.main import run_inference as run_request
from immich_ml.models.base import InferenceEntry, InferenceModel
from immich_ml.models.captioning.florence2 import MESSAGE as FLORENCE_MESSAGE
from immich_ml.models.captioning.florence2 import Florence2Captioner
from immich_ml.models.captioning.joycaption import DEFAULT_LLM, VISION_REPO, JoyCaptionCaptioner
from immich_ml.models.captioning.qwen import USER_PROMPT, QwenVLCaptioner, _Engine
from immich_ml.models.constants import get_model_source
from immich_ml.pipeline import captioning
from immich_ml.schemas import CaptionOptions, ModelSource, ModelTask, ModelType

QWEN = "Qwen/Qwen3-VL-2B-Instruct"


class TestEngineSelection:
    def test_runs_qwen_for_qwen_repos(self) -> None:
        assert captioning(QWEN) is QwenVLCaptioner
        assert captioning("QWEN/Qwen2.5-VL-3B-Instruct") is QwenVLCaptioner  # the name is matched case-insensitively

    def test_runs_qwen_for_any_repo_it_does_not_know(self) -> None:
        assert captioning("some/other-caption-model") is QwenVLCaptioner

    def test_runs_joycaption_for_joy_repos(self) -> None:
        assert captioning("fancyfeast/joy-caption-alpha-two") is JoyCaptionCaptioner

    def test_maps_florence_repos_to_the_refused_engine(self) -> None:
        assert captioning("microsoft/Florence-2-base") is Florence2Captioner


def test_florence2_explains_why_it_cannot_run(tmp_path: Path) -> None:
    captioner = Florence2Captioner("microsoft/Florence-2-base", cache_dir=tmp_path)

    with pytest.raises(RuntimeError) as refused:
        captioner.load()
    assert str(refused.value) == FLORENCE_MESSAGE

    with pytest.raises(RuntimeError) as refused:
        captioner._load()
    assert str(refused.value) == FLORENCE_MESSAGE


def test_caption_models_answer_the_caption_task() -> None:
    for engine in (QwenVLCaptioner, JoyCaptionCaptioner, Florence2Captioner):
        assert engine.identity == (ModelType.CAPTION, ModelTask.CAPTION)


def test_plain_repos_are_a_huggingface_model_source() -> None:
    assert get_model_source(QWEN) is ModelSource.HUGGINGFACE
    assert get_model_source("microsoft/Florence-2-base") is ModelSource.HUGGINGFACE
    assert get_model_source("fancyfeast/joy-caption-alpha-two") is ModelSource.HUGGINGFACE
    assert get_model_source("Qwen3-VL-2B-Instruct") is None  # a caption model names its org, unlike the mirrored ones


def test_keeps_the_full_repo_id_of_a_caption_model(tmp_path: Path) -> None:
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)

    assert captioner.repo_id == QWEN
    assert captioner.model_name == "Qwen3-VL-2B-Instruct"
    assert captioner.model_dir == tmp_path / "caption"


def test_defaults_the_joycaption_llm(tmp_path: Path) -> None:
    captioner = JoyCaptionCaptioner("fancyfeast/joy-caption-alpha-two", cache_dir=tmp_path)

    assert captioner.llm_model == DEFAULT_LLM


def test_downloads_the_joycaption_components_beside_each_other(mocker: MockerFixture, tmp_path: Path) -> None:
    snapshot = mocker.patch("immich_ml.models.captioning.joycaption.snapshot_download")
    captioner = JoyCaptionCaptioner("fancyfeast/joy-caption-alpha-two", cache_dir=tmp_path)

    captioner.download()

    calls = snapshot.call_args_list
    assert [(call.args[0], call.kwargs.get("repo_type")) for call in calls] == [
        ("fancyfeast/joy-caption-alpha-two", "space"),  # only a space carries the image adapter
        (VISION_REPO, None),
        (DEFAULT_LLM, None),
    ]
    assert [call.kwargs["local_dir"] for call in calls] == [
        tmp_path / "caption" / component for component in ("adapter", "vision", "llm")
    ]
    assert captioner.cached is False  # nothing landed on disk


def test_joycaption_is_cached_only_once_every_component_is(tmp_path: Path) -> None:
    captioner = JoyCaptionCaptioner("fancyfeast/joy-caption-alpha-two", cache_dir=tmp_path)

    for component in ("adapter", "vision"):
        (captioner.model_dir / component).mkdir(parents=True)
        (captioner.model_dir / component / "weights.bin").touch()
    assert captioner.cached is False

    (captioner.model_dir / "llm").mkdir(parents=True)
    (captioner.model_dir / "llm" / "config.json").touch()
    assert captioner.cached is True


def _mocked_backends(mocker: MockerFixture, cuda: bool) -> mock.MagicMock:
    torch = mock.MagicMock()
    torch.cuda.is_available.return_value = cuda
    torch.float16, torch.float32 = "float16", "float32"
    transformers = mock.MagicMock()
    mocker.patch.dict(sys.modules, {"torch": torch, "transformers": transformers})
    return transformers


def test_qwen_loads_the_model_on_cuda_in_half_precision(mocker: MockerFixture, tmp_path: Path) -> None:
    transformers = _mocked_backends(mocker, cuda=True)
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)

    engine = captioner._load()

    transformers.AutoProcessor.from_pretrained.assert_called_once_with(QWEN)
    transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(QWEN, dtype="float16")
    assert engine.device == "cuda"


def test_qwen_loads_the_model_on_cpu_in_full_precision(mocker: MockerFixture, tmp_path: Path) -> None:
    transformers = _mocked_backends(mocker, cuda=False)
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)

    engine = captioner._load()

    transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(QWEN, dtype="float32")
    assert engine.device == "cpu"


def test_qwen_loads_the_local_snapshot_once_downloaded(mocker: MockerFixture, tmp_path: Path) -> None:
    transformers = _mocked_backends(mocker, cuda=False)
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)
    captioner.model_dir.mkdir(parents=True)
    (captioner.model_dir / "config.json").touch()

    captioner._load()

    transformers.AutoProcessor.from_pretrained.assert_called_once_with(captioner.model_dir)


def test_reads_caption_options_as_the_engine_takes_them() -> None:
    request = {
        "caption": {
            "modelName": QWEN,
            "options": {"prompt": "list the clothing", "maxNewTokens": 64, "llmModel": DEFAULT_LLM},
        }
    }

    entries = get_entries(json.dumps(request))

    assert [entry.options for entry in entries] == [
        CaptionOptions(prompt="list the clothing", max_new_tokens=64, llm_model=DEFAULT_LLM)
    ]


def test_caption_options_default_to_the_engine_choices() -> None:
    entries = get_entries(json.dumps({"caption": {"modelName": QWEN}}))

    assert entries[0].options == CaptionOptions()
    assert entries[0].options.prompt is None
    assert entries[0].options.max_new_tokens == 120


def test_joycaption_keys_the_loaded_model_by_its_llm() -> None:
    assert QwenVLCaptioner.graph(CaptionOptions(llm_model=DEFAULT_LLM)) is None
    assert JoyCaptionCaptioner.graph(CaptionOptions()) is None
    assert JoyCaptionCaptioner.graph(CaptionOptions(llm_model=DEFAULT_LLM)) == DEFAULT_LLM


@pytest.mark.parametrize(
    "options",
    [
        {"maxNewTokens": "many"},
        {"cache_dir": "/cache"},  # what no caption engine takes it does not quietly ignore
    ],
)
def test_refuses_caption_options_an_engine_cannot_take(options: dict[str, Any]) -> None:
    with pytest.raises(HTTPException) as refused:
        get_entries(json.dumps({"caption": {"modelName": QWEN, "options": options}}))

    assert refused.value.status_code == 422


def test_refuses_a_disabled_model(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "disabled_models", QWEN)

    with pytest.raises(HTTPException) as refused:
        get_entries(json.dumps({"caption": {"modelName": QWEN}}))

    assert refused.value.status_code == 422
    assert refused.value.detail == "Model disabled on this instance."


def test_refuses_a_model_a_disabled_prefix_matches(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "disabled_models", "buffalo_l, Qwen/Qwen3-VL*")

    with pytest.raises(HTTPException) as refused:
        get_entries(json.dumps({"caption": {"modelName": QWEN}}))

    assert refused.value.status_code == 422


def test_allows_a_model_the_disabled_list_does_not_match(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "disabled_models", "Qwen/Qwen3-VL*")

    entries = get_entries(json.dumps({"caption": {"modelName": "fancyfeast/joy-caption-alpha-two"}}))

    assert [entry.name for entry in entries] == ["fancyfeast/joy-caption-alpha-two"]


def test_allows_every_model_while_no_disabled_list_is_set(mocker: MockerFixture) -> None:
    mocker.patch.object(settings, "disabled_models", None)

    entries = get_entries(json.dumps({"caption": {"modelName": QWEN}}))

    assert [entry.name for entry in entries] == [QWEN]


@pytest.mark.asyncio
async def test_answers_with_the_caption_and_the_image_size(mocker: MockerFixture, pil_image: Image.Image) -> None:
    model = mock.Mock(spec=InferenceModel, depends=[], loaded=True)
    model.predict.return_value = "a photo of a dog in the snow"
    mocker.patch("immich_ml.main.model_cache.get", side_effect=lambda entry, ttl: model)
    entries: list[InferenceEntry[Any]] = [InferenceEntry(QwenVLCaptioner, QWEN, CaptionOptions())]

    response = await run_request(pil_image, entries)

    assert response == {
        "caption": "a photo of a dog in the snow",
        "imageHeight": 800,
        "imageWidth": 600,
    }
    model.predict.assert_called_once_with(pil_image, options=CaptionOptions())


def test_decodes_the_image_and_hands_it_to_the_engine(tmp_path: Path, pil_image: Image.Image) -> None:
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)
    engine = mock.Mock()
    engine.generate.return_value = "a caption"
    captioner.session, captioner.loaded = engine, True
    options = CaptionOptions(prompt="list the clothing", max_new_tokens=32)

    assert captioner.predict(pil_image, options=options) == "a caption"
    image, passed = engine.generate.call_args.args
    assert isinstance(image, Image.Image)
    assert passed is options


def test_qwen_generates_with_the_requested_prompt_and_token_budget(
    mocker: MockerFixture, tmp_path: Path, pil_image: Image.Image
) -> None:
    class FakeTensor:
        shape = (1, 7)

        def to(self, device: str) -> "FakeTensor":
            return self

    torch = mock.MagicMock()
    torch.Tensor = FakeTensor
    mocker.patch.dict(sys.modules, {"torch": torch})
    processor = mock.MagicMock()
    processor.apply_chat_template.return_value = {"input_ids": FakeTensor(), "pixel_values": FakeTensor()}
    processor.batch_decode.return_value = [" a caption\n"]
    model = mock.MagicMock()
    captioner = QwenVLCaptioner(QWEN, cache_dir=tmp_path)
    captioner.session, captioner.loaded = _Engine(model, processor, "cpu"), True

    caption = captioner.predict(pil_image, options=CaptionOptions(prompt="list the clothing", max_new_tokens=32))
    captioner.predict(pil_image, options=CaptionOptions())

    assert caption == "a caption"
    first, second = processor.apply_chat_template.call_args_list
    assert first.args[0][0]["content"][0]["image"] is pil_image
    assert first.args[0][0]["content"][1] == {"type": "text", "text": "list the clothing"}
    assert second.args[0][0]["content"][1] == {"type": "text", "text": USER_PROMPT}
    assert model.generate.call_args_list[0].kwargs["max_new_tokens"] == 32
    assert model.generate.call_args.kwargs["max_new_tokens"] == 120
    assert model.generate.call_args.kwargs["do_sample"] is False


def test_serves_requests_without_importing_the_caption_engines() -> None:
    code = (
        "import sys; import immich_ml.main; "
        "heavy = {'torch', 'transformers', 'torchvision', 'onnxruntime_genai'} & set(sys.modules); "
        "assert not heavy, sorted(heavy)"
    )

    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
