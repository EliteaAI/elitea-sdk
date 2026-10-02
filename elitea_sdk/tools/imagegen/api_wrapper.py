"""Calls the gateway's images API and lands results in a project artifact bucket.

Ported by decision from the legacy `imagegen` plugin's invoke path
(`legacy/plugins/imagegen/methods/invoke.py:96-350` in elitea-platform): resolve
the project's image-generation model, call the OpenAI-compatible images API,
write every returned image into the configured bucket, and return the object
references. The plugin's OWN generic descriptor/health/invoke provider-hub
surface is intentionally NOT ported here — that stays deferred to ADR-0012/P3
(elitea-platform #864). This wrapper only carries the two tools an agent
actually attaches: `generate_image` and `edit_image`.

The gateway routes this wrapper calls (`/llm/v1/images/generations` and
`/llm/v1/images/edits`) are served by `services/elitea-llm-gateway` in
elitea-platform (`internal/llmproxy/handler.go`, `multipart.go`) — this module
does not duplicate that logic, it is a thin, direct client of it, mirroring
`elitea_sdk.runtime.clients.client.EliteAClient.generate_image` but taking the
model from the TOOLKIT's own configured `image_generation_model` (one image
toolkit per project may point at a different model) rather than a client-level
default that nothing else in this SDK ever sets.
"""

import base64
import logging
from typing import Any, List, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, create_model

from ...runtime.utils.tool_groups import tool_group, with_tool_groups

logger = logging.getLogger(__name__)

DEFAULT_NAME_PREFIX = "generated-"
DEFAULT_RESPONSE_FORMAT = "b64_json"

GenerateImageSchema = create_model(
    "ImageGenGenerateImageSchema",
    prompt=(str, Field(description="Text prompt describing the image to generate.")),
    size=(
        Optional[str],
        Field(default=None, description="Image size, e.g. '1024x1024'. Leave empty for the model's default."),
    ),
    n=(Optional[int], Field(default=1, description="Number of images to generate.")),
)

EditImageSchema = create_model(
    "ImageGenEditImageSchema",
    image=(
        str,
        Field(description="Artifact filepath (/{bucket}/{filename}) of the source image to edit."),
    ),
    prompt=(str, Field(description="Text prompt describing the edit to apply.")),
    mask=(
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional artifact filepath (/{bucket}/{filename}) of a mask image; "
                "transparent areas of the mask mark where the edit applies."
            ),
        ),
    ),
)


class ImageGenAPIWrapper(BaseModel):
    """Generates and edits images through the gateway, saving results to a bucket."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    elitea: Any = None
    image_generation_model: Optional[str] = None
    bucket: Optional[str] = None
    name_prefix: str = DEFAULT_NAME_PREFIX

    def _images_url(self, operation: str) -> str:
        return f"{self.elitea.base_url}{self.elitea.llm_path}/images/{operation}"

    def _headers(self, content_type: Optional[str] = None) -> dict:
        headers = dict(self.elitea.headers)
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _require_model(self) -> str:
        if not self.image_generation_model:
            raise ValueError("Image generation model is not configured for this toolkit")
        return self.image_generation_model

    def _save_images(self, payload: dict, prefix: str) -> List[dict]:
        """Writes every image the gateway returned into the toolkit's bucket.

        Mirrors the legacy plugin's `_process_and_save_images`: accepts either
        `b64_json` or `url` entries (the two `response_format` shapes the
        images API can answer with) and always ends with bytes written through
        the SAME artifact upload path the Artifact toolkit and the chat
        attachments feature use (`EliteAClient.artifact(bucket).create(...)`),
        so a generated image shows up in Artifacts exactly like any other
        upload.
        """
        if not self.bucket:
            raise ValueError("An artifact bucket is not configured for this toolkit")
        artifact = self.elitea.artifact(self.bucket)
        saved: List[dict] = []
        for index, item in enumerate(payload.get("data") or []):
            if item.get("b64_json"):
                image_bytes = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                response = requests.get(item["url"], timeout=self.elitea.model_timeout)
                response.raise_for_status()
                image_bytes = response.content
            else:
                continue
            filename = f"{self.name_prefix}{prefix}-{index}.png"
            result = artifact.create(filename, image_bytes)
            if "error" in result:
                raise RuntimeError(f"Failed to save generated image: {result['error']}")
            saved.append({"filepath": result["filepath"], "filename": result.get("sanitized_name", filename)})
        if not saved:
            raise RuntimeError("The image generation model returned no image data")
        return saved

    @tool_group("write")
    def generate_image(self, prompt: str, size: Optional[str] = None, n: Optional[int] = 1) -> dict:
        """Generate one or more images from a text prompt and save them to the project's artifact bucket."""
        model = self._require_model()
        request_body = {
            "prompt": prompt,
            "model": model,
            "n": n or 1,
            "response_format": DEFAULT_RESPONSE_FORMAT,
        }
        if size and size.lower() != "auto":
            request_body["size"] = size
        response = requests.post(
            self._images_url("generations"),
            headers=self._headers("application/json"),
            json=request_body,
            timeout=self.elitea.model_timeout,
        )
        response.raise_for_status()
        artifacts = self._save_images(response.json(), "generate")
        return {"artifacts": artifacts}

    @tool_group("write")
    def edit_image(self, image: str, prompt: str, mask: Optional[str] = None) -> dict:
        """Edit an existing image (an artifact filepath) from a text prompt and save the result to the project's artifact bucket."""
        model = self._require_model()
        artifact = self.elitea.artifact(self.bucket)
        image_bytes, image_name = artifact.get_raw_content_by_filepath(image)
        files = {"image": (image_name, image_bytes)}
        if mask:
            mask_bytes, mask_name = artifact.get_raw_content_by_filepath(mask)
            files["mask"] = (mask_name, mask_bytes)
        data = {"prompt": prompt, "model": model, "response_format": DEFAULT_RESPONSE_FORMAT}
        response = requests.post(
            self._images_url("edits"),
            headers=self._headers(),
            data=data,
            files=files,
            timeout=self.elitea.model_timeout,
        )
        response.raise_for_status()
        artifacts = self._save_images(response.json(), "edit")
        return {"artifacts": artifacts}

    @with_tool_groups
    def get_available_tools(self):
        return [
            {
                "name": "generate_image",
                "description": self.generate_image.__doc__,
                "args_schema": GenerateImageSchema,
                "ref": self.generate_image,
            },
            {
                "name": "edit_image",
                "description": self.edit_image.__doc__,
                "args_schema": EditImageSchema,
                "ref": self.edit_image,
            },
        ]

    def run(self, mode: str, *args: Any, **kwargs: Any):
        for tool in self.get_available_tools():
            if tool["name"] == mode:
                return tool["ref"](*args, **kwargs)
        raise ValueError(f"Unknown mode: {mode}")
