# GameWorldLM v0.4: Base vs LoRA

The 143-example training pipeline completed on 2026-09-24. The adapter learned the required JSON format and object counts on this small held-out set. Spatial validity remains limited: only 3 of 14 LoRA worlds pass the unchanged validator. This is a pipeline validation, not a production-quality world model.

## Held-Out Test Results

| Metric | Frozen base | LoRA | Change |
| --- | ---: | ---: | ---: |
| JSON parse rate | 2/14 (14.29%) | 14/14 (100.00%) | +85.71 pp |
| Schema valid rate | 2/14 (14.29%) | 14/14 (100.00%) | +85.71 pp |
| Spatial validator pass rate | 0/14 (0.00%) | 3/14 (21.43%) | +21.43 pp |
| Object count accuracy | 2/14 (14.29%) | 14/14 (100.00%) | +85.71 pp |
| All generation expectations | 0/14 (0.00%) | 3/14 (21.43%) | +21.43 pp |

All denominators include every test example. Parsing uses the raw completion without JSON extraction, repair, or retries. Twelve base responses have invalid JSON; nine contain extra quotes between object records. Object count accuracy requires a valid schema and every explicitly requested type count to match. It does not require a spatially valid arrangement, which is measured separately.

LoRA spatial failures contain 11 `blocked_entity`, 7 `overlap`, 3 `not_connected`, and 1 `not_inside` issues. These are issue occurrences, not disjoint sample counts. An output can contain multiple issues.

## Training Evidence

| Epoch | Mean logged training loss | Validation loss | Checkpoint |
| --- | ---: | ---: | --- |
| 1 | 0.202838 | 0.148230 | `checkpoint-29` |
| 2 | 0.124690 | 0.136970 | `checkpoint-58` |
| 3 | 0.093586 | 0.136171 | `checkpoint-87` |

Initial and final evaluation loss on the same training examples: **0.294725 -> 0.086612**, a **70.61% reduction**. These evaluation losses differ from the per-epoch logged training averages because training uses dropout and updates weights between batches. Training completed all 87 optimizer updates in 350.80 seconds. Base inference took 377.04 seconds; unmerged LoRA inference took 954.74 seconds for the same 14 prompts.

Only the 40,370,176 LoRA parameters were trainable. All base-model weights were frozen. Each epoch checkpoint, the final adapter, resolved configuration, runtime versions, and raw predictions are preserved under `outputs/training/qwen7b_lora/` locally and on the server. Runtime artifacts and model weights are excluded from Git.

## Reproduction

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Config: `training/configs/qwen7b_lora.json`; BF16, unquantized base.
- LoRA: rank 16, alpha 32, dropout 0.05; attention and MLP projections.
- Optimization: 3 epochs, learning rate 0.0002, batch size 1, accumulation 4, context 2048.
- Partition: 115 train / 14 validation / 14 test; seed 42; normalized prompt deduplication.
- Dataset SHA-256: `398ac3ad751a675342d3e424b4193d8beebac554a2e8f70e0854d58d5336158f`.
- Schema SHA-256: `84f4ae89879096eebb4088d0cf0996bd37ebef90efa3546fb8e1ef38f9b1e2d1`.
- Runtime: Python 3.10.8, PyTorch 2.6.0 / CUDA 12.4, Transformers 4.57.6, PEFT 0.15.2, Accelerate 1.6.0, RTX 4080 SUPER.
- Decoding: greedy, maximum 4096 new tokens, repetition penalty 1.0, seed 42. Both variants use identical prompts and precision. The base is evaluated with adapters disabled.

All 143 complete chat examples fit within 2048 tokens; the longest contains 1586 tokens. No target tokens were truncated, and no window splitting was needed in this run. Loss labels cover assistant outputs only.

See [training instructions](training_v04.md) for the commands and independent overfit smoke test. The small test set and automatically generated training data do not establish generalization or gameplay quality. Improving spatial placement remains a separate next-stage task.
