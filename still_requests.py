from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_name: Literal["flux2_text", "flux2_ref"] = "flux2_text"
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    seed: int = 42
    steps: int = Field(default=28, ge=1, le=200)
    cfg: float = Field(default=3.5, gt=0.0, le=30.0)
    sampler_name: str = "euler"
    scheduler: str = "simple"
    filename_prefix: str = "filmforge_still"
    ref_images: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("prompt must not be blank")
        return text

    @field_validator("width", "height")
    @classmethod
    def _validate_dimension(cls, value: int) -> int:
        if value < 256 or value > 2048:
            raise ValueError("dimension must be between 256 and 2048")
        return int(value)

    @field_validator("filename_prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("filename_prefix must not be blank")
        return text

    @field_validator("ref_images")
    @classmethod
    def _validate_ref_images(cls, value: list[str]) -> list[str]:
        cleaned = []
        for item in value or []:
            text = str(item or "").strip()
            if not text:
                continue
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _validate_workflow_requirements(self) -> "StillRequest":
        if self.workflow_name == "flux2_ref":
            if not self.ref_images:
                raise ValueError("ref_images is required when workflow_name='flux2_ref'")
            if len(self.ref_images) > 1:
                raise ValueError("flux2_ref supports exactly one ref image")
        return self
