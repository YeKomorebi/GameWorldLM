# GameWorldLM v0.4.1 Spatial Analysis

## Scope and Evidence

本版完成离线错误分析、固定 benchmark、关系指标和 curriculum 数据准备，**没有重新训练或修改 v0.4 checkpoints**。以下模型分数来自原 14 条 held-out test 的原始预测重评，不是新增 100 条 benchmark 的结果。新的 benchmark 已冻结并验证可解，但尚未运行模型推理。

分析对象为 Qwen2.5-7B-Instruct 基座与 v0.4 LoRA，原数据划分仍为 115 / 14 / 14，seed 42。WorldState schema、validator 语义、geometry 和 renderer 均沿用原实现。Schema SHA-256：`84f4ae89879096eebb4088d0cf0996bd37ebef90efa3546fb8e1ef38f9b1e2d1`。

可审计产物：

- [逐项错误报告](spatial_failure_report.md)，含原始错误码、次数和样本 ID。
- [数据统计 JSON](v0_4_1_dataset_statistics.json)，含全部对象类型、关系、尺寸及地图分布。
- [固定 benchmark](../evaluation/prompts.jsonl) 与 [manifest](../evaluation/prompts.manifest.json)。
- 本地与服务器 `outputs/evaluation/v0_4_1/{base,lora}_spatial_samples.jsonl`：每条样本的 prompt、完整 raw response、解析后的 JSON、validator 结果、精确原因、footprint 和关系级检查。
- `outputs/curriculum/v0_4_1/`：成功/失败记录、chat 数据、world.json、validation.json、PNG 及文件校验和。

## Existing Model Results

| Metric | Base | LoRA |
| --- | ---: | ---: |
| JSON parse rate | 2/14 (14.29%) | 14/14 (100%) |
| Schema valid rate | 2/14 (14.29%) | 14/14 (100%) |
| Spatial validator pass rate | 0/14 (0%) | 3/14 (21.43%) |
| Object count accuracy | 2/14 (14.29%) | 14/14 (100%) |
| Near accuracy | 0/4 (0%) | 3/4 (75.00%) |
| Inside accuracy | 0/3 (0%) | 2/3 (66.67%) |
| Connected-to accuracy | 0/7 (0%) | 4/7 (57.14%) |
| Boundary accuracy | 2/14 (14.29%) | 14/14 (100%) |
| Overlap-free rate | 0/14 (0%) | 4/14 (28.57%) |

这些关系分母很小，不能据此判断真实泛化排名。基座有 12 条解析失败，主要是 JSON 语法错误；统计严格保留失败，不移除代码围栏、不截取 JSON、不修复生成结果。

### LoRA Failure Distribution

| Requested category | Issue occurrences | Affected samples |
| --- | ---: | ---: |
| overlap | 18 | 10 |
| out_of_bounds | 0 | 0 |
| near violation | 0 | 0 |
| inside violation | 1 | 1 |
| connected_to violation | 3 | 2 |
| invalid reference | 0 | 0 |
| other | 0 | 0 |

`overlap` 汇总包含 7 次同层 `overlap` 和 11 次 `blocked_entity`，原始子类型与消息全部保留。同一样本可出现多种错误，受影响样本数不可相加。当前最明显的瓶颈是实体占地与结构占地冲突，其次是把接近、重叠或角点接触当成有效连接。

三条连接失败的实际 footprint（右边、下边为 exclusive）为：道路 `[4,14,7,16]` 与桥 `[12,12,20,14]` 分离；道路 `[20,14,23,16]` 与同一座桥仅在 `(20,14)` 角点接触；道路 `[0,10,32,12]` 穿过城堡 `[8,6,24,18]`，没有共享边。最后一种跨层重叠本身符合地面承载规则，但不满足 connected_to，不能通过放宽碰撞规则解决。

另有 1 条样本缺少必需的 near 关系。原 validator 只检查已有关系，因此不会产生 `not_near`；新指标把缺失槽位计为失败，并在 `requirement_failures` 单独说明。上表是原 validator 的错误分类，不把提示词遗漏伪装成 validator 报错。

### Metric Definitions

JSON、schema、整体空间、对象数量、boundary、overlap-free 均以所有样本为分母；无效 JSON、schema 失败、推理失败不会从分母移除。

关系准确率使用 micro aggregation。对每个样本与关系类型，分母为 `max(预测关系数 + 缺失的指定端点关系数, 请求的最少关系数)`，分子为引用明确、非自引用、非重复且满足几何规则的预测关系数。重复关系计入分母，反向 near/connected_to 视为同一关系；inside 有方向。无任何请求或预测关系时为 N/A。新 benchmark 指定端点，额外生成的无关关系不能完全替代漏掉的必需关系。

关系分数只衡量该条关系的几何正确性，与全局碰撞、边界指标分开。near 距离为 0 仍满足原 near 语义，但没有合法 inside 的碰撞仍会使整体空间检查失败。

Boundary accuracy 表示所有 footprint 均处于生成世界声明的地图内；地图尺寸是否遵从 prompt 由 expectations 检查。Overlap-free 沿用原 layer、bridge 和显式 inside 例外。两者均要求 schema 有效且 ID 唯一。

新增 `task_constraint_pass_rate` 检查 benchmark 明确指定的 ID、类型、尺寸、关系、贴边要求及全部既有 expectations；v0.4 样本没有这组标注，显示 N/A。它可防止通过缩小物体尺寸等方式规避 benchmark 难度。

## Frozen Evaluation Benchmark

Seed 为 **20260925**，共 100 条，主题覆盖 forest、desert、ice、city、dungeon、fantasy、cyberpunk。固定文件只包含 system/user 消息和评估约束，不输出 assistant 答案。

| Difficulty | Prompts | Construction |
| --- | ---: | --- |
| easy | 30 | 单个关系、3 个对象，含干扰对象 |
| medium | 40 | 共享目标或连接链、2 个关系、5 个对象 |
| hard | 30 | 多关系组合、贴边约束、共享目标或连接链、9 个对象 |

每条 prompt 都有经过原 validator、对象数量、尺寸、端点和贴边检查的构造见证，其哈希保存在行记录中。难度是按约束复杂度定义的工程分级，尚未用模型通过率做校准。benchmark 使用 expedition stars/chains 模板，curriculum 使用 settlement disjoint-pairs 模板，二者复用的仅是几何放置与验证机制。

隔离措施包括：固定文件重建必须内容相同；同一路径拒绝替换不同 seed 的清单；与原 143 条数据核对规范化 prompt；curriculum 拒绝 benchmark、原 val/test、重复 prompt 和重复见证世界；训练准备入口拒绝 `evaluation_only` 或命中 benchmark 规范化哈希的记录。规范化消除大小写、Unicode 兼容形式和空白差异，不声称能够识别所有语义改写。

## Dataset Statistics

| Data | Total | Valid | Rejected | Use |
| --- | ---: | ---: | ---: | --- |
| v0.4 snapshot | 143 | 143 | 0 | 保持 115/14/14 |
| New curriculum attempts | 270 | 269 | 1 | 训练候选，仅剔除 1 条重复 prompt |
| Candidate training pool | 384 | 384 | 0 | 115 原训练样本 + 269 合成样本 |
| Original validation/test | 28 | 28 | 0 | 继续隔离，不计入 384 |
| New benchmark | 100 | 100 feasible tasks | 0 | 仅评估，不加入训练池 |

因此现有 143 条加新增 269 条，共有 **412 条有效带标签数据**；其中可用于本轮后续训练设计的候选池是 **384 条**。未把所有 143 条旧样本重新混入训练。

| Statistic | Original 143 | New 269 |
| --- | ---: | ---: |
| Average objects | 13.35 | 6.35 |
| near relation tokens | 54 | 240 |
| inside relation tokens | 43 | 240 |
| connected_to relation tokens | 37 | 239 |

新增难度分布为 easy 89 / medium 90 / hard 90；focus_relation 为 near 90 / inside 90 / connected_to 89。hard 样本包含另外两类关系，因此关系 token 总数不同于按 focus 分类的样本数。

| Object type | Original tokens | New tokens |
| --- | ---: | ---: |
| bridge | 1 | 0 |
| castle | 50 | 104 |
| courtyard | 3 | 0 |
| crystal | 108 | 94 |
| dune | 69 | 0 |
| house | 145 | 208 |
| library | 47 | 102 |
| monster | 329 | 247 |
| npc | 335 | 233 |
| portal | 45 | 0 |
| river | 1 | 0 |
| road | 140 | 239 |
| rock | 107 | 83 |
| ruin | 157 | 207 |
| tower | 99 | 98 |
| tree | 122 | 93 |
| vehicle | 91 | 0 |
| wall | 60 | 0 |

所有新增成功样本均完成 generation -> schema/validator/expectation/task validation -> PNG render，并保存 `system/user/assistant` messages。`source=synthetic_constructive`、`model=null` 明确区分于原真实 Qwen 数据；本轮没有调用 Qwen API。269 条的自动检查通过率不代表模型成功率，也不等于人工质量评审。

数据的主要局限是构造模板有限，平均对象数小于旧数据，而且仅覆盖 11 种对象类型。它适合作为空间关系课程，不能取代复杂真实场景数据。增加关系密度与正例可提高后续学习信号，但是否改善泛化必须通过冻结 benchmark 测量。

## Modules and Commands

| Module | Design reason |
| --- | --- |
| `evaluation/spatial.py` | 直接重评原始预测，复用原指标与几何函数，保存原错误信息，避免修复掩盖模型问题。 |
| `evaluation/benchmark.py` | 将固定评估清单与可解性证明独立管理，供现有 infer_lora 直接读取。 |
| `evaluation/holdout.py` | 在训练数据准入阶段集中排除冻结提示词。 |
| `evaluation/dataset_statistics.py` | 分开统计模型成绩与数据质量，避免把标签通过率视为模型能力。 |
| `dataset_generation/spatial_tasks.py` | 参数化关系、难度、尺寸与地图，通过有界搜索生成合法布局；不改变世界 schema。 |
| `dataset_generation/spatial_curriculum.py` | 负责去重、隔离、质量过滤、渲染和数据落盘，输出真实的合成来源及文件哈希。 |

```bash
# 重评已有模型输出，无 GPU、无 API、无训练。
python -m evaluation.spatial --samples outputs/training/qwen7b_lora/evaluation_samples.jsonl --base outputs/training/qwen7b_lora/base_predictions.jsonl --lora outputs/training/qwen7b_lora/lora_predictions.jsonl --output-dir outputs/evaluation/v0_4_1

# 验证并重建相同的固定 benchmark。
python -m evaluation.benchmark --exclude-file outputs/training/prepared/train.jsonl --exclude-file outputs/training/prepared/val.jsonl --exclude-file outputs/training/prepared/test.jsonl

# 完整小规模课程；270 是候选数，去重或验证失败项保留在 failed.jsonl。
# 必须选择新目录，避免覆盖任何已生成数据。
python -m dataset_generation.spatial_curriculum --num-samples 270 --seed 42 --prepared-dir outputs/training/prepared --output-dir outputs/curriculum/new-run

python -m dataset_generation.spatial_curriculum --relation near --difficulty easy --num-samples 30 --output-dir outputs/curriculum/near-easy
python -m dataset_generation.spatial_curriculum --relation inside --difficulty medium --num-samples 30 --output-dir outputs/curriculum/inside-medium
python -m dataset_generation.spatial_curriculum --relation connected_to --difficulty hard --num-samples 30 --output-dir outputs/curriculum/connected-hard

python -m evaluation.dataset_statistics --curriculum-dir outputs/curriculum/v0_4_1
```

curriculum 默认 seed 42、270 个候选，每次最多 500 个；这只是新增构造器的批次上限，原 Qwen 10,000 条能力不变。有被剔除的候选时 CLI 返回非零并保留成功、失败和统计文件。`train.jsonl` 仅含新合成数据；指定 `--prepared-dir` 后另存 `candidate_train.jsonl`，合并旧 train，保留各自 provenance 与相对 artifact_root。

这批数据是候选训练导出，不自动接入 v0.4 的默认训练配置。原 `training.prepare_dataset` 仍要求经过审计的真实 LLM 记录，不会把合成数据冒充 Qwen 输出；后续训练版本需要显式接受合成来源，并将候选训练池与原 held-out 数据分别配置。

以下命令供下一阶段对冻结 benchmark 做推理使用，本版没有执行这 200 次模型推理：

```bash
python -m training.infer_lora --base --samples evaluation/prompts.jsonl --output outputs/evaluation/benchmark/base_predictions.jsonl
python -m training.infer_lora --samples evaluation/prompts.jsonl --output outputs/evaluation/benchmark/lora_predictions.jsonl
python -m evaluation.spatial --samples evaluation/prompts.jsonl --base outputs/evaluation/benchmark/base_predictions.jsonl --lora outputs/evaluation/benchmark/lora_predictions.jsonl --output-dir outputs/evaluation/benchmark
```

## Next Experiment

先在冻结 benchmark 上测现有 base 与 v0.4 LoRA，保存逐难度、逐关系基线。之后再显式设计新训练配置，比较相同预算下的原数据、加入 curriculum、分阶段 curriculum；保持解码参数、随机种子与测试清单一致。优先观察 blocked_entity/overlap 的下降以及 connected_to 的正确性，同时检查 JSON 和对象数量指标是否回退。

训练阶段可先使用 easy，再引入 medium/hard，并保留旧训练数据以维持对象类型和复杂场景覆盖。当前没有训练新 adapter，也没有证据支持宣称空间泛化已经提升。

## Verification

服务器完整环境通过 **188 项测试**，包括全部 154 项原测试和 34 项新增测试，Ruff 检查通过。新增测试覆盖严格解析、缺失与重复关系、非法引用、inside 方向、边界、角点连接、合法包含例外、benchmark 可复现性与隔离、curriculum 渲染与失败过滤。既有测试文件没有修改。

269 张 PNG 全部通过非空像素检查，逐条 world、validation、PNG 与 JSONL 文件校验和一致。候选训练池与 100 条 benchmark 加原 28 条 held-out 提示词的交集为 0。原数据快照、schema 和两个 v0.4 最终 adapter 的哈希保持不变，详见 [产物审计记录](v0_4_1_verification.json)。训练目录中的 epoch checkpoints 未被写入。
