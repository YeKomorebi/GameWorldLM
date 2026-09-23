# GameWorldLM v0.3.1 训练数据流水线

本版本验证从 prompt 到标准训练数据的完整流程：Qwen 生成、World State 校验、PNG 渲染、失败过滤、train/val/test 划分和统计。输出可用于后续 SFT/LoRA 的数据加载；本版本没有执行模型权重训练。

## 规模与已有 Prompt

```bash
python -m pip install -e ".[dev]"

# 默认行为仍为生成 10,000 条 prompt 和对应世界。
gameworldlm-build-dataset

# 新的独立实验，计划处理 1,000 条 prompt。
gameworldlm-build-dataset --num-samples 1000 --output-dir outputs/v031-1000

# 使用已有清单中的前 1,000 条，保留 ID、文本、主题和校验期望。
gameworldlm-build-dataset --prompts-file dataset/prompts.jsonl --num-samples 1000 --output-dir outputs/v031-1000

# 只准备 prompt，完全离线；随后去掉 --prepare-only 即可开始生成。
gameworldlm-build-dataset --prompts-file dataset/prompts.jsonl --num-samples 1000 --prepare-only --output-dir outputs/v031-1000
```

以上新实验命令是可选方式；同一输出目录的配置和 prompt 必须一致。源清单本身不会被修改。CLI 的 Python 等价入口为 `python -m dataset_generation.dataset_builder`。

| 参数 | 含义 |
| --- | --- |
| `--num-samples N` | 整批计划处理的 prompt 数，包含后续可能失败的样本；不保证产出 N 条有效世界。 |
| `--count N` | `--num-samples` 的兼容别名，原命令继续可用。 |
| `--limit N` | 本次调用最多处理 N 条新样本；恢复时跳过已处理样本，整批规模不变。 |
| `--prompts-file PATH` | 已有 PromptSpec JSONL，或以 `.json` 结尾的 PromptSpec 对象数组。 |
| `--prepare-only` | 只创建所选 prompt 清单，不调用 Qwen。 |
| `--output-dir PATH` | 规模或生成配置变化时应使用独立目录。 |

没有 `--prompts-file` 且没有指定数量时仍为 10,000；指定文件但未指定数量时使用文件中的全部 prompt。数量超过文件长度会明确报错，不循环复制或静默补齐。

输入结构与 v0.3 自动生成的 `prompts.jsonl` 相同，字段为 `id`、`theme`、`prompt`、`expectations`、`descriptors`。读取时检查空文本、长度、重复 ID 和归一化后的重复 prompt。选择顺序与源文件一致，因此原万条清单按七主题轮转的平衡性可以保留。

## 自动划分

每次生成调用结束、达到 `--limit` 或正常暂停后，都会从已提交检查点重建有效数据并自动划分。通过校验和渲染的有效数据按 80/10/10 分到三个互斥集合。

```text
<output-dir>/
  prompts.jsonl
  checkpoint.sqlite3
  valid.jsonl              # 所有有效样本，生成时持续追加
  train.jsonl              # 80%
  val.jsonl                # 10%
  test.jsonl               # 10%
  failed.jsonl             # 失败及导出复检拒绝的记录
  split_membership.jsonl   # 样本 ID、prompt 分组与分区
  split_manifest.json      # 比例、seed、实际数量、文件 SHA-256
  dataset_info.json        # gameworldlm_train / val / test 映射
  report.json              # 当前生成进度、对象统计、ETA
  statistics.json          # 最新训练数据快照统计
  artifacts/               # 原始调用记录、world.json 和 map.png
```

先按归一化 prompt 的带 seed SHA-256 排序，再分配样本；相同数据、相同 seed 的成员归属一致。比例使用最大余数法取整，同余数时依次优先 train、val、test。例如：

| 有效数量 | Train | Val | Test |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 |
| 7 | 5 | 1 | 1 |
| 10 | 8 | 1 | 1 |
| 1,000 | 800 | 100 | 100 |
| 10,000 | 8,000 | 1,000 | 1,000 |

数量指有效数据；失败记录从不进入这些分区。数据不足时会有空分区，不通过重复样本填充。导出前复检原始请求、World State schema/几何/期望、world 文件摘要、PNG 标识和审计来源。重复归一化 prompt 或复检失败的记录进入 `failed.jsonl`。

`train.jsonl`、`val.jsonl`、`test.jsonl` 每行均含标准 `messages` 数组和可追溯元信息，训练目标是最终有效 World JSON。`dataset_info.json` 给出 LLaMA-Factory ShareGPT/messages 映射。以 `gameworldlm_train` 训练、`gameworldlm_val` 调参，保留 `gameworldlm_test` 用于最终评估。

新增有效样本后重新划分，成员归属可能变化。因此开始模型训练时应使用独立快照，避免一边生成一边读取尚在更新的分区。归一化去重可阻止同一指令跨集；相似模板场景还应在后续质量评估中单独检查。

## 复用已有生成结果

```bash
# 所有已完成记录，快照目标必须不存在。
gameworldlm-split-dataset --source-dir dataset --output-dir outputs/training-snapshot

# 只选前 1,000 条已完成记录（包含失败项）；不足 1,000 条则报错。
gameworldlm-split-dataset --source-dir dataset --num-samples 1000 --output-dir outputs/training-1000
```

等价入口为 `python -m dataset_generation.splits`。该命令不需要 API Key，通过只读 SQLite 查询获取一致的已提交结果，不会调用模型、修改源数据或中断正在生成的任务。运行中的未提交样本不计入快照。

快照的 `messages` 可独立加载用于训练。PNG 和原始审计文件不重复复制；每条记录的 `artifact_root` 指向源目录，`artifacts` 中的路径相对于它。需要迁移审计文件时请同时迁移源数据目录。

v0.3 的原始成功全集位于 `train.jsonl`。使用 v0.3.1 恢复同一批次时会从 SQLite 重建到 `valid.jsonl`，再输出三个分区；完整成功记录和审计文件仍被保留，已处理项不重新调用 Qwen。World State schema、生成清单版本、manifest 生成契约保持兼容。

已经启动的 v0.3 进程继续使用已加载的代码，不自动中断或重启；可以用新快照与统计命令立即读取它的数据。原进程退出后，再次启动生成命令即可使用 v0.3.1。

## 统计与完成时间

```bash
gameworldlm-dataset-stats --source-dir dataset
# 等价命令
python -m dataset_generation.statistics --source-dir dataset
```

该命令只读取数据，兼容 v0.3 检查点，并从原始审计文件恢复对象数量及耗时。v0.3.1 的 `report.json` 也会在每个样本完成后更新这些指标：

| 字段 | 定义 |
| --- | --- |
| `total_count` | 整批计划数量；独立导出快照则为选中的已完成记录数。 |
| `processed` / `pending` | 已处理与尚未处理数量。 |
| `valid_count` / `failed_count` | 有效和失败数量；快照统计包含导出复检结果。 |
| `object_distribution` | 有效世界中各对象类型的 token 总数，未出现的合法类型为零。 |
| `average_object_count` | 每个有效世界的平均对象数量。 |
| `average_generation_seconds` | 已获得有效计时的样本平均生成耗时，包含修正和渲染。 |
| `timed_samples` | 用于时间估计的样本数量。 |
| `estimated_remaining_seconds` | 平均生成耗时 × pending。 |
| `estimated_completion_at` | 以 UTC 表示的预计完成时间；仅运行中且有计时数据时提供。 |

暂停、配额中断或没有可用耗时时，绝对完成时间为 null；有历史计时时仍提供恢复运行后预计需要的剩余秒数。全部完成时剩余时间为零。估算假设串行连续运行、API 额度充足，不包含未来停机和人工暂停，初期样本较少时可能波动。

进度报告同时保留原有 `succeeded`/`failed` 生成历史字段；若成功记录在训练导出复检时被拒绝，`export_rejected_count` 会单独记录，状态为 `export_failed`，实际训练数据数量以 `valid_count` 和 `statistics.json` 为准。

## 验证范围

原有 105 项测试未修改，新增测试覆盖数量别名、已有清单子集、无效输入、800/100/100 划分、去重与互斥性、失败过滤、续跑复现、只读快照、产物复检和 ETA。134 项测试通过，World State schema 未改动。

已从原有 10,000 条清单离线选出 1,000 条，七类数量为 143/143/143/143/143/143/142；另使用真实 Qwen 验证新入口的一条完整流程，生成有效 JSON、PNG、训练记录及统计报告。完整 1,000 条世界生成及 LoRA 权重训练尚不属于这次已完成的验证结果。

2026-09-24 对服务器的 141 条已完成真实记录导出训练快照：134 条有效数据划分为 train 107、val 14、test 13；6 条校验失败及 1 条 provider 错误单独保存。快照位于 `outputs/v031-training-validation/`，已同步到本地，校验全部输出文件 SHA-256，并在 Windows 再次通过产物复检。独立快照包含 1,808 个对象，平均每个有效世界约 13.49 个对象。
