"""PEFT training, epoch audit records, and automatic paired base/adapter evaluation."""

import argparse
import importlib.metadata
import json
import os
import platform

from dataset.storage import utc_now, write_json

from .config import ExperimentConfig, load_config
from .data import AssistantCollator, read_jsonl, tokenize_rows
from .evaluate_lora import comparison_report, evaluate_predictions
from .infer_lora import predict
from .modeling import attach_lora, load_base, load_tokenizer, release_memory
from .prepare_dataset import prepare, verify_prepared


def check_run_directory(config: ExperimentConfig) -> None:
    output = config.path(config.training.output_dir)
    checkpoint = config.training.resume_from_checkpoint
    if checkpoint is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "Training output is not empty; configure a checkpoint to resume or a new directory"
            )
        return
    checkpoint = config.path(checkpoint)
    if checkpoint.parent != output or not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("Resume checkpoint must be a saved Trainer checkpoint in output_dir")
    previous = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    current = config.model_dump()
    for item in (previous, current):
        item["training"]["resume_from_checkpoint"] = None
    if previous != current:
        raise ValueError("Resume configuration differs from the recorded experiment")
    prepared = verify_prepared(config.path(config.data.output_dir))
    recorded = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    if prepared["dataset_sha256"] != recorded["dataset_sha256"]:
        raise ValueError("Resume dataset differs from the recorded experiment")


def run(config: ExperimentConfig) -> dict:
    check_run_directory(config)
    output = config.path(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run(config)
    except Exception as exc:
        import torch

        out_of_memory = isinstance(exc, torch.cuda.OutOfMemoryError)
        write_json(
            output / "status.json",
            {
                "status": "out_of_memory" if out_of_memory else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": utc_now(),
                "recommendation": (
                    "Use qwen7b_qlora.json or --qlora with a fresh output directory"
                    if out_of_memory
                    else "Inspect the error and keep this run's artifacts for diagnosis"
                ),
            },
        )
        raise


def _run(config: ExperimentConfig) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    settings = config.training
    output = config.path(settings.output_dir)
    prepare(config)
    data_root = config.path(config.data.output_dir)
    manifest = verify_prepared(data_root)
    train_rows, val_rows, test_rows = [
        read_jsonl(data_root / f"{name}.jsonl") for name in ("train", "val", "test")
    ]
    if config.smoke.enabled:
        if len(train_rows) < config.smoke.num_samples:
            raise ValueError("Not enough training rows for the overfit smoke test")
        train_rows = train_rows[: config.smoke.num_samples]
        test_rows = train_rows
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config.model_dump())
    write_json(output / "dataset_manifest.json", manifest)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "safetensors",
                )
            },
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    write_json(output / "status.json", {"status": "tokenizing", "started_at": utc_now()})
    set_seed(settings.seed)
    tokenizer = load_tokenizer(config)
    train_data, train_stats = tokenize_rows(tokenizer, train_rows, settings)
    val_data, val_stats = tokenize_rows(tokenizer, val_rows, settings)
    write_json(output / "tokenization.json", {"train": train_stats, "val": val_stats})
    from dataset_generation.splits import _write_jsonl

    _write_jsonl(output / "evaluation_samples.jsonl", test_rows)
    write_json(output / "status.json", {"status": "loading_model", "updated_at": utc_now()})
    model = attach_lora(load_base(config), config)
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA weights may be trainable")
    write_json(
        output / "trainable_parameters.json",
        {
            "trainable": sum(trainable.values()),
            "total": sum(p.numel() for p in model.parameters()),
            "names": sorted(trainable),
        },
    )

    class EpochAudit(TrainerCallback):
        def __init__(self):
            self.losses = []
            self.history = []
            if settings.resume_from_checkpoint and (output / "epoch_metrics.json").is_file():
                self.history = json.loads(
                    (output / "epoch_metrics.json").read_text(encoding="utf-8")
                )
                state_path = config.path(settings.resume_from_checkpoint) / "trainer_state.json"
                resumed_step = json.loads(state_path.read_text(encoding="utf-8"))["global_step"]
                self.history = [row for row in self.history if row["global_step"] <= resumed_step]

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append(logs["loss"])

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if state.epoch is not None and state.epoch > 0 and metrics and "eval_loss" in metrics:
                epoch = int(round(state.epoch))
                self.history.append(
                    {
                        "epoch": epoch,
                        "global_step": state.global_step,
                        "training_loss": sum(self.losses) / len(self.losses)
                        if self.losses
                        else None,
                        "validation_loss": metrics["eval_loss"],
                        "training_loss_definition": (
                            "mean of per-optimizer-step logged assistant-only losses"
                        ),
                        "checkpoint": None,
                    }
                )
                self.losses = []
                write_json(output / "epoch_metrics.json", self.history)

        def on_save(self, args, state, control, **kwargs):
            if self.history and self.history[-1]["global_step"] == state.global_step:
                self.history[-1]["checkpoint"] = f"checkpoint-{state.global_step}"
                write_json(output / "epoch_metrics.json", self.history)

    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.max_grad_norm,
        optim=settings.optimizer,
        lr_scheduler_type=settings.lr_scheduler,
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=None,
        save_safetensors=True,
        report_to=[],
        seed=settings.seed,
        data_seed=settings.seed,
        full_determinism=settings.full_determinism,
        dataloader_num_workers=settings.dataloader_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        logging_nan_inf_filter=False,
    )
    audit = EpochAudit()
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=AssistantCollator(tokenizer.pad_token_id, settings.pad_to_multiple_of),
        processing_class=tokenizer,
        callbacks=[audit],
    )
    initial_loss = trainer.evaluate(train_data, metric_key_prefix="initial_train")[
        "initial_train_loss"
    ]
    write_json(
        output / "status.json", {"status": "training", "initial_training_loss": initial_loss}
    )
    try:
        result = trainer.train(
            resume_from_checkpoint=(
                str(config.path(settings.resume_from_checkpoint))
                if settings.resume_from_checkpoint
                else None
            )
        )
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        final_loss = trainer.evaluate(train_data, metric_key_prefix="final_train")[
            "final_train_loss"
        ]
        write_json(
            output / "training_summary.json",
            result.metrics
            | {
                "initial_training_loss": initial_loss,
                "final_training_loss": final_loss,
                "relative_loss_reduction": 1 - final_loss / initial_loss,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        trainer.optimizer = None
        trainer.lr_scheduler = None
        del trainer
        model.gradient_checkpointing_disable()
        release_memory()
        write_json(output / "status.json", {"status": "evaluating", "updated_at": utc_now()})
        base_predictions = predict(
            model, tokenizer, test_rows, config, output / "base_predictions.jsonl", base=True
        )
        lora_predictions = predict(
            model, tokenizer, test_rows, config, output / "lora_predictions.jsonl"
        )
        base_metrics = evaluate_predictions(test_rows, base_predictions)
        lora_metrics = evaluate_predictions(test_rows, lora_predictions)
        write_json(output / "base_metrics.json", base_metrics)
        write_json(output / "lora_metrics.json", lora_metrics)
        comparison_report(
            output,
            base_metrics,
            lora_metrics,
            {
                "model": config.model.name_or_path,
                "revision": config.model.revision,
                "mode": "overfit training-set memorization"
                if config.smoke.enabled
                else "held-out test",
                "dataset_sha256": manifest["dataset_sha256"],
                "quantized": config.model.load_in_4bit,
                "dtype": config.model.dtype,
                "inference": config.inference.model_dump(),
            },
        )
        summary = {
            "status": "completed",
            "finished_at": utc_now(),
            "base": base_metrics["metrics"],
            "lora": lora_metrics["metrics"],
        }
        if config.smoke.enabled:
            reduction = 1 - final_loss / initial_loss
            schema_rate = lora_metrics["metrics"]["schema_valid_rate"]
            summary["smoke_passed"] = (
                reduction >= config.smoke.minimum_loss_reduction
                and schema_rate >= config.smoke.minimum_schema_pass_rate
            )
            write_json(
                output / "overfit_smoke_report.json",
                summary
                | {
                    "samples": len(train_rows),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "relative_loss_reduction": reduction,
                    "criteria": config.smoke.model_dump(),
                    "interpretation": "Training-set memorization only; not held-out performance",
                },
            )
        write_json(output / "status.json", summary)
        return summary
    except torch.cuda.OutOfMemoryError:
        write_json(
            output / "status.json",
            {
                "status": "out_of_memory",
                "recommendation": "Use qwen7b_qlora.json or --qlora with a fresh output directory",
            },
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train PEFT LoRA and evaluate base versus adapter")
    parser.add_argument("--config", default="training/configs/qwen7b_lora.json")
    parser.add_argument("--qlora", action="store_true", help="Explicitly enable 4-bit loading")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.qlora:
        config.model.load_in_4bit = True
    summary = run(config)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("smoke_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
