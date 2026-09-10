"""Unit tests for the `imagegen` toolkit.

The gateway itself (`services/elitea-llm-gateway` in elitea-platform) is
mocked at the `requests` boundary: these tests assert that `generate_image`
calls `POST {base_url}/llm/v1/images/generations` with the toolkit's
configured model, decodes the base64 image(s) it gets back, and writes them
through `elitea.artifact(bucket).create(...)` — and that `edit_image` reads
the source image via `get_raw_content_by_filepath` and posts a multipart
request to `.../images/edits`.
"""

from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.tools.imagegen import ImageGenToolkit, get_tools
from elitea_sdk.tools.imagegen.api_wrapper import ImageGenAPIWrapper


def _fake_elitea():
    elitea = MagicMock()
    elitea.base_url = "https://gateway.internal"
    elitea.llm_path = "/llm/v1"
    elitea.headers = {"Authorization": "Bearer token"}
    elitea.model_timeout = 30
    return elitea


def _b64_response(count: int = 1):
    import base64

    return {
        "data": [{"b64_json": base64.b64encode(b"\x89PNG-fake-bytes").decode()} for _ in range(count)]
    }


class TestGenerateImage:
    def test_requires_a_configured_model(self):
        wrapper = ImageGenAPIWrapper(elitea=_fake_elitea(), image_generation_model=None, bucket="images")
        with pytest.raises(ValueError, match="Image generation model is not configured"):
            wrapper.generate_image(prompt="a red fox")

    @patch("elitea_sdk.tools.imagegen.api_wrapper.requests.post")
    def test_generates_and_saves_one_image(self, mock_post):
        elitea = _fake_elitea()
        artifact = MagicMock()
        artifact.create.return_value = {"filepath": "/images/generated-generate-0.png", "sanitized_name": "generated-generate-0.png"}
        elitea.artifact.return_value = artifact
        mock_post.return_value = MagicMock(status_code=200, json=lambda: _b64_response(1))
        mock_post.return_value.raise_for_status = lambda: None

        wrapper = ImageGenAPIWrapper(elitea=elitea, image_generation_model="gpt-image-1", bucket="images")
        result = wrapper.generate_image(prompt="a red fox in the snow")

        called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
        assert called_url == "https://gateway.internal/llm/v1/images/generations"
        assert called_kwargs["json"]["model"] == "gpt-image-1"
        assert called_kwargs["json"]["prompt"] == "a red fox in the snow"
        elitea.artifact.assert_called_once_with("images")
        artifact.create.assert_called_once()
        assert result["artifacts"] == [{"filepath": "/images/generated-generate-0.png", "filename": "generated-generate-0.png"}]

    @patch("elitea_sdk.tools.imagegen.api_wrapper.requests.post")
    def test_multiple_images_are_all_saved(self, mock_post):
        elitea = _fake_elitea()
        artifact = MagicMock()
        artifact.create.side_effect = [
            {"filepath": f"/images/generated-generate-{i}.png", "sanitized_name": f"generated-generate-{i}.png"}
            for i in range(3)
        ]
        elitea.artifact.return_value = artifact
        mock_post.return_value = MagicMock(status_code=200, json=lambda: _b64_response(3))
        mock_post.return_value.raise_for_status = lambda: None

        wrapper = ImageGenAPIWrapper(elitea=elitea, image_generation_model="gpt-image-1", bucket="images")
        result = wrapper.generate_image(prompt="three cats", n=3)

        assert artifact.create.call_count == 3
        assert len(result["artifacts"]) == 3

    def test_requires_a_configured_bucket(self):
        wrapper = ImageGenAPIWrapper(elitea=_fake_elitea(), image_generation_model="gpt-image-1", bucket=None)
        with patch("elitea_sdk.tools.imagegen.api_wrapper.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: _b64_response(1))
            mock_post.return_value.raise_for_status = lambda: None
            with pytest.raises(ValueError, match="artifact bucket is not configured"):
                wrapper.generate_image(prompt="a red fox")


class TestEditImage:
    @patch("elitea_sdk.tools.imagegen.api_wrapper.requests.post")
    def test_edits_an_existing_artifact_image(self, mock_post):
        elitea = _fake_elitea()
        artifact = MagicMock()
        artifact.get_raw_content_by_filepath.return_value = (b"source-bytes", "source.png")
        artifact.create.return_value = {"filepath": "/images/generated-edit-0.png", "sanitized_name": "generated-edit-0.png"}
        elitea.artifact.return_value = artifact
        mock_post.return_value = MagicMock(status_code=200, json=lambda: _b64_response(1))
        mock_post.return_value.raise_for_status = lambda: None

        wrapper = ImageGenAPIWrapper(elitea=elitea, image_generation_model="gpt-image-1", bucket="images")
        result = wrapper.edit_image(image="/images/source.png", prompt="add a hat")

        called_url = mock_post.call_args[0][0]
        called_kwargs = mock_post.call_args[1]
        assert called_url == "https://gateway.internal/llm/v1/images/edits"
        assert called_kwargs["data"]["model"] == "gpt-image-1"
        assert "image" in called_kwargs["files"]
        assert "mask" not in called_kwargs["files"]
        assert result["artifacts"] == [{"filepath": "/images/generated-edit-0.png", "filename": "generated-edit-0.png"}]

    @patch("elitea_sdk.tools.imagegen.api_wrapper.requests.post")
    def test_edit_with_a_mask_sends_both_files(self, mock_post):
        elitea = _fake_elitea()
        artifact = MagicMock()
        artifact.get_raw_content_by_filepath.side_effect = [
            (b"source-bytes", "source.png"),
            (b"mask-bytes", "mask.png"),
        ]
        artifact.create.return_value = {"filepath": "/images/generated-edit-0.png", "sanitized_name": "generated-edit-0.png"}
        elitea.artifact.return_value = artifact
        mock_post.return_value = MagicMock(status_code=200, json=lambda: _b64_response(1))
        mock_post.return_value.raise_for_status = lambda: None

        wrapper = ImageGenAPIWrapper(elitea=elitea, image_generation_model="gpt-image-1", bucket="images")
        wrapper.edit_image(image="/images/source.png", prompt="add a hat", mask="/images/mask.png")

        called_kwargs = mock_post.call_args[1]
        assert set(called_kwargs["files"].keys()) == {"image", "mask"}


class TestToolkitRegistration:
    def test_toolkit_config_schema_declares_the_three_fields(self):
        schema = ImageGenToolkit.toolkit_config_schema().schema()
        assert set(["image_generation_model", "bucket", "name_prefix", "selected_tools"]).issubset(
            schema["properties"].keys()
        )
        assert schema["properties"]["image_generation_model"]["configuration_model"] == "image_generation"

    def test_get_toolkit_builds_generate_and_edit_tools(self):
        toolkit = ImageGenToolkit.get_toolkit(
            elitea=_fake_elitea(), image_generation_model="gpt-image-1", bucket="images", toolkit_name="My ImageGen"
        )
        names = sorted(t.name for t in toolkit.get_tools())
        assert names == ["edit_image", "generate_image"]

    def test_selected_tools_filters_the_toolkit(self):
        toolkit = ImageGenToolkit.get_toolkit(
            elitea=_fake_elitea(),
            image_generation_model="gpt-image-1",
            bucket="images",
            selected_tools=["generate_image"],
        )
        names = [t.name for t in toolkit.get_tools()]
        assert names == ["generate_image"]

    def test_get_tools_entry_point_reads_settings_and_injected_elitea(self):
        elitea = _fake_elitea()
        tool = {
            "settings": {
                "elitea": elitea,
                "image_generation_model": "gpt-image-1",
                "bucket": "images",
                "selected_tools": [],
            },
            "toolkit_name": "My ImageGen",
        }
        tools = get_tools(tool)
        assert sorted(t.name for t in tools) == ["edit_image", "generate_image"]
