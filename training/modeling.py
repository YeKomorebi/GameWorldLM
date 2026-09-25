"""Lazy optional ML dependencies and the same base-model loading for train and inference."""

import gc

from .config import ExperimentConfig


def load_tokenizer(config: ExperimentConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        local_files_only=config.model.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base(config: ExperimentConfig):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Qwen 7B training pipeline")
    dtype = getattr(torch, config.model.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU cannot use bfloat16; set model.dtype to float16")
    options = {}
    if config.model.load_in_4bit:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.quantization_type,
            bnb_4bit_use_double_quant=config.model.double_quantization,
            bnb_4bit_compute_dtype=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        torch_dtype=dtype,
        attn_implementation=config.model.attention_implementation,
        device_map={"": 0},
        trust_remote_code=False,
        use_safetensors=True,
        local_files_only=config.model.local_files_only,
        **options,
    )


def attach_lora(model, config: ExperimentConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    return model


def release_memory():
    import torch

    gc.collect()
    torch.cuda.empty_cache()
