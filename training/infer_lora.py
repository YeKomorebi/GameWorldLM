"""Deterministic direct model inference without generation repairs or JSON rewriting."""

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

from dataset.storage import write_json

from .config import load_config
from .data import read_jsonl
from .modeling import load_base, load_tokenizer


def predict(
    model, tokenizer, samples, config, destination: Path, *, base: bool = False
) -> list[dict]:
    import torch
    from transformers import set_seed

    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.config.use_cache = True
    settings = config.inference
    metadata = {
        "model": config.model.model_dump(),
        "inference": settings.model_dump(),
        "variant": "base" if base else "lora",
        "sample_ids": [row["id"] for row in samples],
    }
    write_json(destination.with_suffix(".metadata.json"), metadata)
    predictions = []
    context = (
        model.disable_adapter() if base and hasattr(model, "disable_adapter") else nullcontext()
    )
    with context, torch.inference_mode(), destination.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(samples, 1):
            started = time.perf_counter()
            messages = row["messages"][:2]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            max_context = getattr(model.config, "max_position_embeddings", None)
            if max_context is not None and len(encoded) + settings.max_new_tokens > max_context:
                raise ValueError(
                    "Inference prompt plus max_new_tokens exceeds model context; "
                    "no truncation applied"
                )
            inputs = torch.tensor([encoded], device=model.device)
            seed = settings.seed + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            set_seed(seed % (2**32))
            options = {
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": settings.do_sample,
                "repetition_penalty": settings.repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if settings.do_sample:
                options.update(temperature=settings.temperature, top_p=settings.top_p)
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), **options
            )
            generated = output[0, len(encoded) :].tolist()
            prediction = {
                "id": row["id"],
                "variant": metadata["variant"],
                "raw_response": tokenizer.decode(generated, skip_special_tokens=True),
                "prompt_tokens": len(encoded),
                "completion_tokens": len(generated),
                "finish_reason": "eos"
                if generated and generated[-1] == tokenizer.eos_token_id
                else "length",
                "latency_seconds": time.perf_counter() - started,
                "error": None,
            }
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{metadata['variant']} inference {index}/{len(samples)}: {row['id'][:12]}",
                flush=True,
            )
    return predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen base or a trained LoRA adapter")
    parser.add_argument("--config", default="training/configs/qwen7b_lora.json")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    model = load_base(config)
    if not args.base:
        from peft import PeftModel

        adapter = args.adapter or config.path(config.training.output_dir) / "adapter"
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    samples = read_jsonl(args.samples or config.path(config.data.output_dir) / "test.jsonl")
    predict(model, tokenizer, samples, config, args.output, base=args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
