"""Strict JSON experiment configuration with explicit inheritance and path resolution."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(Settings):
    name_or_path: str
    revision: str
    dtype: Literal["bfloat16", "float16", "float32"]
    attention_implementation: Literal["sdpa", "eager", "flash_attention_2"]
    load_in_4bit: bool
    quantization_type: Literal["nf4", "fp4"]
    double_quantization: bool
    local_files_only: bool


class DataSettings(Settings):
    sources: list[str] = Field(min_length=1)
    output_dir: str
    max_samples: int | None = Field(ge=3)
    seed: int
    split_ratios: list[float] = Field(min_length=3, max_length=3)
    system_prompt_file: str

    @model_validator(mode="after")
    def ratios(self):
        if any(value <= 0 for value in self.split_ratios) or abs(sum(self.split_ratios) - 1) > 1e-8:
            raise ValueError("Three positive split ratios must sum to one")
        return self


class LoRASettings(Settings):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)
    bias: Literal["none"]


class TrainSettings(Settings):
    output_dir: str
    seed: int
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_seq_length: int = Field(ge=128)
    long_sequence_policy: Literal["chunk_assistant", "error"]
    overlap_tokens: int = Field(ge=1)
    batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    gradient_checkpointing: bool
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    optimizer: str
    lr_scheduler: str
    dataloader_workers: int = Field(ge=0)
    pad_to_multiple_of: int = Field(gt=0)
    full_determinism: bool
    resume_from_checkpoint: str | None


class InferenceSettings(Settings):
    max_new_tokens: int = Field(gt=0)
    do_sample: bool
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    repetition_penalty: float = Field(gt=0)
    seed: int


class SmokeSettings(Settings):
    enabled: bool
    num_samples: int = Field(gt=0)
    minimum_loss_reduction: float = Field(gt=0, lt=1)
    minimum_schema_pass_rate: float = Field(ge=0, le=1)


class ExperimentConfig(Settings):
    project_root: str
    model: ModelSettings
    data: DataSettings
    lora: LoRASettings
    training: TrainSettings
    inference: InferenceSettings
    smoke: SmokeSettings

    @model_validator(mode="after")
    def context(self):
        if self.training.overlap_tokens >= self.training.max_seq_length:
            raise ValueError("overlap_tokens must be less than max_seq_length")
        return self

    def path(self, value: str) -> Path:
        return (Path(self.project_root) / value).resolve()


def _merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = (
            _merge(result[key], value) if isinstance(value, dict) and key in result else value
        )
    return result


def _read(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    seen.add(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    return _merge(_read(path.parent / parent, seen), data) if parent else data


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    config = ExperimentConfig.model_validate(_read(path, set()))
    config.project_root = str((path.parent / config.project_root).resolve())
    return config
