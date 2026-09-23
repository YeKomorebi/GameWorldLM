# GameWorldLM v0.3 数据流水线

## 模块职责

| 模块 | 职责与设计原因 |
| --- | --- |
| `prompt_generator.py` | 离线组合场景、材质、氛围、布局、数量和关系；固定 seed 保证复现，无须为生成 prompt 消耗 API。按主题轮转，短批次也能覆盖多个主题。 |
| `qwen_generator.py` | 复用 `ProviderConfig`、`GenerationPipeline` 和原有校验修正循环；对每次 SDK 调用限速，包括修正请求。 |
| `dataset_builder.py` | 管理批次、恢复、质量准入、训练格式、统计与停止条件；数据生成策略不进入 World State schema。 |
| `storage.py` | SQLite 事务作为已处理状态的依据，JSONL 为可重建输出；操作系统文件锁防止两个进程同时写同一批次。进程退出会自动释放锁。 |

既有 `world/schema.py`、验证器、渲染器和原有测试无需修改。当前使用串行生成，避免 pygame 并发渲染和供应商限流带来的复杂性。

## 场景覆盖

| 主题 | 原有 schema biome | 默认 10,000 条中的数量 |
| --- | --- | ---: |
| forest | forest | 1429 |
| desert | desert | 1429 |
| ice | snow | 1429 |
| city | city | 1429 |
| dungeon | arcane | 1428 |
| fantasy | arcane | 1428 |
| cyberpunk | city | 1428 |

主题保存在数据集元信息与 prompt 中；建筑风格使用已有 `attributes.style/material`。未添加 biome、对象类型或 schema 字段。

每类主题有八种场景、四种建筑风格、五种氛围、五种布局、四种地图尺寸和随机对象数量。关系要求包含无必需关系、`near`、`inside`、`connected_to`。每个 prompt 同时生成可执行的数量、地图和最小关系数量期望；未列出的对象类型期望数量为零。文本去重后才分配样本 ID。默认 seed 为 42。

自动校验覆盖 JSON schema、空间几何、对象数量、地图以及最小关系数量。它不保证氛围、材质表达、指定对象对之间的自然语言关系或地图可玩性完全符合描述；这些仍需要后续语义评估和人工筛选。

## 运行

```bash
python -m pip install -e ".[dev]"

# 仅生成 prompt；无网络调用，不会创建虚假的训练样本。
gameworldlm-build-dataset --prepare-only

# QWEN_API_KEY / QWEN_MODEL / QWEN_BASE_URL 从 .env 或环境读取。
# 先真实生成七条，然后恢复剩余任务。
gameworldlm-build-dataset --limit 7
gameworldlm-build-dataset

# 独立的小实验：总数不同请使用新的输出目录。
gameworldlm-build-dataset --count 70 --seed 43 --output-dir outputs/pilot

# 可选的调用限速、预算阈值。
gameworldlm-build-dataset --requests-per-minute 10 --max-total-tokens 1000000
```

等价入口为 `python -m dataset_generation.dataset_builder`。独立 prompt 入口：

```bash
gameworldlm-prompts --count 10000 --seed 42 --output dataset/prompts.jsonl
```

恢复必须使用相同 count、seed、模型、endpoint、schema、system prompt、attempts、tile size、token 上限和 timeout。限速、单次样本数量和累计 token 阈值可以调整。修改生成配置时请使用新的输出目录。

`--max-total-tokens` 检查已报告的累计 token，且仅在样本之间检查，因此可能超出一个样本的用量；缺失的 usage 不计入总和。这不是供应商账单的硬额度。OpenAI SDK 自带一次传输重试，可能产生额外 HTTP 请求；RPM 控制的是逻辑 SDK 调用的启动间隔。

完整批次可能运行很久；仅启动 10,000 次请求在默认间隔下就至少需要约 8.3 小时，实际还包含模型响应、修正和渲染耗时。不会自动重跑失败项以追求表面上的 100% 成功率。

## 文件与格式

```text
dataset/
  __init__.py, storage.py, export.py  # 原有源代码
  prompts.jsonl                     # 完整清单、主题、期望与描述元信息
  prompt_report.json                # prepare-only 的清单统计
  manifest.json                     # 配置、schema/system prompt/清单摘要
  checkpoint.sqlite3                # 持久化运行状态和已提交结果
  train.jsonl                       # 通过生成、校验、渲染的真实 LLM 样本
  failed.jsonl                      # 失败或中断样本，不含训练 messages
  dataset_info.json                 # LLaMA-Factory ShareGPT/messages 映射
  report.json                       # 每个样本完成后更新的统计
  artifacts/<sample_id>/runs/<run_id>/
    prompt.txt
    record.json
    validation.json
    world.json                      # 生成并通过校验后才存在
    map.png                         # 渲染成功后才存在
    attempts/001/
      request.json
      response.txt
      provider.json
      validation.json
```

`dataset/` 同时是原有 Python 包，只忽略明确列出的生成文件和 `artifacts/`；源码继续版本管理。密钥、完整批量输出和原始响应不上传公共 GitHub。

`train.jsonl` 每行保留：`id`、`theme`、`prompt`、`model`、`actual_models`、`expectations`、`validation`、`artifacts`、对象数、调用用量、schema 摘要和质量标记，以及标准 `messages` 数组：

```json
{
  "messages": [
    {"role": "system", "content": "原始 World State 生成约束及 schema"},
    {"role": "user", "content": "原始场景 prompt"},
    {"role": "assistant", "content": "最终通过校验的 World State JSON 字符串"}
  ]
}
```

该片段仅展示训练字段结构，不是实际训练样本。修正后的最终答案与原始请求配对；修正对话保存在审计目录，不混入训练输入。训练目标是 World State JSON，不是 PNG。`artifacts` 路径相对于数据集根目录。

`failed.jsonl` 使用相同的元信息，附有 `status`、`error`、`failure_reasons` 和最后一次 `validation`；没有有效世界或 PNG 时不伪造对应路径。渲染失败可同时有 `validation.passed=true` 与 `status=render_error`，仍然属于失败样本。

当前 `train.jsonl` 是自动收集的候选训练池。LoRA 开始前应进行人工抽检、语义评估、相似场景分组和独立验证集划分；模板生成的相近 prompt 不应随意拆到训练与评估两侧。

## 恢复与停止

1. 调用前提交 `running` 检查点，调用后保存审计产物并事务提交结果。
2. 已提交的成功和失败样本恢复时均跳过，避免重复计数和重复调用。
3. 若进程在完整产物落盘后、提交结果前退出，恢复时校验产物并收录，无需再调用模型。
4. 若请求结果不明或运行中断，记录为 `interrupted` 失败，不自动重新调用可能已经计费的请求。保留全部现有审计文件，后续可人工决定是否重新实验。
5. JSONL 尾部写入中断时，下次运行从 SQLite 重新生成输出；无需人工删除尾行。

创建 `<output-dir>/STOP` 文件可在当前样本结束后停止。移除文件后重启命令。HTTP 401、403、429 立即暂停后续样本；连续三次其他 provider 错误或渲染、存储产物异常也会停止。暂停期间未处理 prompt 仍计为 pending，绝不计为失败或成功。

退出码：全部完成且无失败为 0；有限批次暂停、其他暂停或有失败为 1；键盘中断为 130。使用 `report.json` 区分暂停与样本失败。硬性结束进程时，报告可能暂时显示 running，下一次恢复会重建。

## 统计口径

- `success_rate`：成功数 / 已处理数，不含 pending；没有处理样本时为 null。
- `average_object_count`：仅成功样本的平均对象数，没有成功样本时为 null。
- `failure_reason_distribution`：每个原因影响的失败样本数；同一样本同一原因只计一次，可同时有多个原因，总数可能大于失败样本数。
- `status_distribution`：每个最终状态的样本数，互斥。
- `themes`：七类主题分别记录成功数与失败数。
- `sdk_attempts`、`reported_total_tokens`、`attempts_without_usage`：实际调用与可获得的用量，保留统计缺口。

## 验证

新增测试文件 `tests/test_dataset_generation.py`，原有测试不变。覆盖 10,000 条 prompt 唯一性/平衡性/复现、真实 pipeline 的测试替身、修正对话过滤、PNG、失败统计、断点恢复、JSONL 尾部恢复、锁、STOP、预算阈值、provider 熔断以及配置变化检测。

```bash
pytest -q
ruff check .
```

## 首批真实 Qwen 验证

2026-09-23 使用 `qwen-plus`，默认 seed=42、最多三次尝试、tile size=16，运行完整清单的前七条。七类各一条均成功，成功率 7/7，平均对象数 13，共 10 次 SDK 调用，供应商报告 29,680 tokens。该结果仅代表首批七条，不代表 10,000 条已经完成或预期全部成功。

| 主题 | 对象数 | SDK 调用次数 | 报告 token 数 |
| --- | ---: | ---: | ---: |
| forest | 9 | 1 | 2100 |
| desert | 15 | 1 | 2950 |
| ice | 11 | 1 | 2593 |
| city | 11 | 2 | 6489 |
| dungeon | 16 | 1 | 2458 |
| fantasy | 16 | 2 | 6052 |
| cyberpunk | 13 | 2 | 7038 |

全部产物保存在本地 `dataset/`，完整 prompt 清单 SHA-256 为 `b5f2e43aadc7890ebebef3f6a14aa1e8721fba5e210f559edf60469f56853d7a`。本地 105 项测试及 Ruff 检查通过；与 v0.2 基线相比，原有 schema 和原有测试文件内容未改变。
