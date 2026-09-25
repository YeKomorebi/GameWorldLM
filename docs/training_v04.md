# GameWorldLM v0.4 LoRA Pipeline

This release validates training, inference, and evaluation with a small recorded dataset. It does not change the World State schema, API generator, spatial validator, or Pygame renderer. The base model is `Qwen/Qwen2.5-7B-Instruct`, pinned to revision `a09a35458c702b33eeacc393d103063234e8bc28`.

## Installation

Use Python 3.10-3.12 and a CUDA GPU. Training dependencies are optional, so API generation and rendering do not require PyTorch.

```bash
python -m pip install -e ".[training,dev]"
```

The training extra pins PyTorch, Transformers, PEFT, Accelerate, and Linux bitsandbytes versions. Downloading the base requires approximately 15 GB of storage; retain additional space for epoch checkpoints and optimizer states. Model weights and all private training artifacts belong under ignored `outputs/` or an external model cache.

On the configured training server, the pinned model is already cached. Use the existing virtual environment and cache from the project root:

```bash
cd /root/autodl-tmp/GameWorldLM
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
```

For another machine, set `HF_HOME` to its model cache or use the Hugging Face default. Enable offline mode only after downloading the pinned revision. QLoRA uses the Linux bitsandbytes dependency; the standard BF16 LoRA path does not require bitsandbytes. Training and inference consume no Qwen API quota.

## Prepare Recorded Data

```bash
python -m training.prepare_dataset --config training/configs/qwen7b_lora.json
```

The default sources are `dataset/` and `outputs/real/`. Only real LLM successes with matching original request messages, world checksums, current schema, spatial constraints, object expectations, and PNG artifacts qualify. Older requests are checked against their saved system prompt. Normalized duplicate user prompts are removed before partitioning. The training system message describes the current schema and is separate from the historical API generation system message.

Each JSONL row contains `id`, `messages` (system/user/assistant), `expectations`, object counts, and source provenance. The assistant message contains the complete serialized valid world. A seed-42 hash ordering and largest-remainder allocation provide reproducible 80/10/10 partitions. The default selects 143 valid rows: **115 train, 14 val, 14 test**. Changing dataset size, sources, seed, or the system message requires a new `data.output_dir` so an existing snapshot cannot silently change. The manifest records file, schema, and dataset hashes and object distributions.

## Train and Compare

```bash
python -m training.train_lora --config training/configs/qwen7b_lora.json
# Use a separate output directory and 4-bit NF4 base weights.
python -m training.train_lora --config training/configs/qwen7b_qlora.json
```

Every hyperparameter is in the JSON config. Child configs use `extends`; paths are resolved from `project_root`, relative to the selected config file. Default LoRA parameters are rank 16, alpha 32, dropout 0.05, 3 epochs, learning rate 0.0002, and maximum sequence length 2048. Attention and MLP projections receive adapters. Only LoRA parameters are trainable. The base weights stay frozen. QLoRA additionally uses NF4 double quantization and PEFT's k-bit preparation.

The tokenizer's actual chat template determines the assistant boundary. System, user, assistant header, and padding labels are `-100`. The causal LM shifts the labels internally. Longer samples are split into overlapping 2048-token windows; repeated context is masked and each assistant target is supervised exactly once. The complete target JSON is preserved. Continuation windows contain the preceding 256 tokens, rather than the entire original prompt; this limits long-distance conditioning and is a deliberate small-pipeline tradeoff. Set `long_sequence_policy` to `error` to reject long sequences instead. Tokenization statistics expose long samples, windows, and supervised tokens.

Each epoch evaluates the validation set and saves a checkpoint. `epoch_metrics.json` records the mean logged optimizer-step training loss, validation loss, and checkpoint path. Losses use assistant labels; they are not whole-prompt language-modeling losses. Initial and final training-set evaluation losses use the same windows, enabling the overfit check. Seeds and deterministic settings improve repeatability on the same software and hardware; cross-device bitwise equality is not guaranteed.

To resume an interrupted run, set `training.resume_from_checkpoint` to one of its saved checkpoint directories and keep all other settings unchanged. The loader checks the previous configuration and dataset digest before writing. A fresh experiment must use an empty output directory; failures preserve diagnostics and write `status.json`.

After training, the same frozen base is evaluated with adapters disabled, then with the trained adapter enabled. Both use identical held-out prompts, precision, tokenizer, greedy decoding, and token limits. No API calls, JSON repair, or generation retries are used during evaluation.

Main outputs under `outputs/training/qwen7b_lora/`:

| Artifact | Purpose |
| --- | --- |
| `resolved_config.json`, `environment.json` | Exact settings and runtime versions |
| `dataset_manifest.json`, `tokenization.json` | Frozen dataset and token accounting |
| `checkpoint-*` | Adapter, optimizer, scheduler, RNG, and Trainer state per epoch |
| `epoch_metrics.json`, `training_summary.json` | Epoch losses and overall training results |
| `adapter/` | Final PEFT adapter and tokenizer |
| `evaluation_samples.jsonl` | Exact evaluation sample order |
| `base_predictions.jsonl`, `lora_predictions.jsonl` | Raw completions, token counts, latency, finish reason |
| `base_metrics.json`, `lora_metrics.json` | Aggregate metrics and per-sample failures |
| `base_vs_lora_report.md` | Paired comparison |
| `status.json` | Current stage or completion status |

## Inference and Offline Evaluation

```bash
python -m training.infer_lora --config training/configs/qwen7b_lora.json --output outputs/inference/lora.jsonl
python -m training.infer_lora --config training/configs/qwen7b_lora.json --base --output outputs/inference/base.jsonl
python -m training.evaluate_lora --samples outputs/training/prepared/test.jsonl --base outputs/inference/base.jsonl --lora outputs/inference/lora.jsonl --output-dir outputs/inference/report
```

`--adapter` selects another saved adapter, and `--samples` selects another prepared chat JSONL. All rates use every evaluation sample as the denominator, including malformed output:

| Metric | Definition |
| --- | --- |
| JSON parse rate | Raw response parses as JSON; Markdown fences are a failure |
| Schema valid rate | Parsed response passes the unchanged `WorldState` schema |
| Spatial validator pass rate | Schema-valid response passes the existing spatial validator |
| Object count accuracy | Every explicitly expected type has the correct count; invalid schemas fail |
| All expectations pass rate | Existing expectation evaluator passes, including map and relation requirements |

When explicit object counts are absent, counts from the reference world are used. Metrics measure structural compliance; they do not establish gameplay quality or human preference.

## Independent Overfit Smoke Test

```bash
python -m training.train_lora --config training/configs/overfit_smoke.json
```

The smoke run reloads the original base and trains a fresh adapter on the first 20 rows of the frozen training split. It uses 20 epochs with constant learning rate, writes its own checkpoints, and evaluates those same 20 training examples to measure memorization. It never trains on the held-out test set. Validation losses still use the original validation split.

`overfit_smoke_report.json` passes when training-set loss falls by at least 50% and at least 80% of generated training examples satisfy the schema. The command exits nonzero if either criterion fails. These thresholds and sample count are configurable. Passing this test demonstrates that the data and gradient path can learn; it is not a claim of generalization.

## Module Boundaries

`prepare_dataset.py` handles provenance and splits without loading a model. `data.py` owns masking and windowing. `modeling.py` shares base/quantization loading between training and inference. `train_lora.py` handles optimization and epoch artifacts. `infer_lora.py` saves unmodified responses. `evaluate_lora.py` calls the existing validators and can rerun without a GPU. This separation keeps training experiments independent of the production API generation and rendering flow.
