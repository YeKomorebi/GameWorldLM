# v0.2 Generation Records and LoRA Data

v0.2 保持 `world/schema.py` 和 World State `schema_version=1.0` 不变。
新增的记录、校验报告和训练数据格式属于生成流程，不向世界数据添加模型或日志字段。

## 运行真实生成

先安装或更新项目，再在本机 `.env` 配置 Qwen。密钥不写入代码、记录或训练文件。

```bash
python -m pip install -e ".[dev]"
python -m examples.real_generation --provider qwen --limit 1
python -m examples.real_generation --provider qwen
```

完整套件位于 `examples/real_prompts.json`，包含十个中文 prompt：森林村庄、冰雪城堡、
沙漠遗迹、末日城市、魔法学院、小型森林营地、森林哨所、冰原前哨、奥术圣所、沙漠商队。
每项配置明确的对象数量、地图大小、biome，部分场景还要求关系数量下限。

```bash
gameworldlm-real --provider qwen --prompt "一座森林里的木屋，旁边有一个守卫" --output-dir outputs/custom
gameworldlm-real --provider openai --model gpt-4o-mini --limit 1
gameworldlm-real --provider qwen --prompts-file my_prompts.json --attempts 5 --no-export
gameworldlm-real --provider qwen --case-id forest_watchtowers --attempts 5
```

`--provider` 只接受真实 API provider，不接受 fixture。默认使用 Qwen，默认每个场景最多
三次生成或修正。HTTP 401、403、429 会停止剩余场景，避免重复调用无效凭据或耗尽配额。
SDK 内部最多重试一次临时网络错误；这里的 attempt 代表一次 SDK 调用，不代表底层每个
HTTP 请求。计费数据以返回的 `usage` 为准，网络失败时可能没有用量信息。

Qwen 默认通过 `extra_body={"enable_thinking": false}` 使用非思考输出，保持非流式 JSON
流程明确；不支持这个参数的旧模型可设置 `QWEN_ENABLE_THINKING=omit`。
当前保留 `max_tokens` 参数以兼容 `qwen-plus`。更换模型时应核对其输出模式、token 上限和
接口地域：[Qwen Chat Completions 文档](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)。

## 每次运行的产物

```text
outputs/real/
  runs/<UTC时间_UUID>/
    prompt.txt
    record.json
    validation.json
    world.json                   最终通过全部校验时保存
    map.png                      渲染成功时保存
    attempts/001/
      request.json               实际发出的 system/user/修正上下文
      response.txt               模型原始文本，包括无效或截断的响应
      provider.json              实际模型、request ID、usage、耗时、finish_reason
      validation.json            本次响应的校验结果
    attempts/002/...
  batches/<batch_id>.json         套件通过数、失败项和 run_id 列表
  datasets/<batch_id>/            本次结束时对现有记录导出的数据快照
```

每次运行使用新的目录，重复运行不会覆盖历史数据。`record.json` 和校验报告通过临时
文件原子替换；每个 attempt 结束即写盘。进程中断时，已有响应仍保留。强制终止或机器
断电可能留下 `running` 状态，不能被视为成功样本。

`record.json` 保存：prompt、case_id、请求的模型、provider、应用版本、schema hash、
生成参数、时间、attempt 列表、最终状态、产物相对路径和 world 文件 SHA-256。
每次响应的 `provider.json` 保存接口实际返回的模型名称，便于追踪模型别名。

最终状态包含 `success`、`validation_failed`、`provider_error`、`render_error` 等。
若服务未返回合法世界，不会伪造 `world.json` 或 PNG；这些产物在 record 中为 `null`，
原始文本和错误仍然保留。若世界有效但渲染失败，会保留 JSON 并明确标记失败。

## 校验分层

```json
{
  "status": "valid",
  "schema_valid": true,
  "spatial_valid": true,
  "expectations_valid": true,
  "errors": [],
  "context": {},
  "repair_hints": [],
  "passed": true
}
```

- Schema：继续使用现有 Pydantic World State。
- Spatial：继续使用现有 validator，检查引用、边界、重叠、包含、邻近和接边。
- Expectations：单独检查测试 prompt 配置的对象数量、地图规格和关系数量下限。
- 未执行的检查记录为 `null`。普通自定义 prompt 没有显式 expectations，其对应字段为 `null`。

失败反馈包含错误和计算后的矩形边界，交给模型修正。流程不偷偷移动对象或放宽原有
空间规则。校验不等于完整自然语言理解，例如“每个守卫靠近不同的塔”目前只部分由
关系数量约束覆盖，仍需看图和检查 JSON。

## 导出训练数据

```bash
gameworldlm-dataset --runs outputs/real --output outputs/dataset-v02
# 或
python -m dataset.export --runs outputs/real --output outputs/dataset-v02 --validation-fraction 0.2
```

导出目标必须是新目录。默认真实生成入口会自动导出到新的 batch 目录，可用
`--no-export` 关闭。不同批次的导出都以 runs 中的全部现有记录为输入。

| 文件 | 用途 |
| --- | --- |
| `records.jsonl` | 所有可解析记录，包括失败项，用于质量分析；不能直接作为 SFT 输入 |
| `train.jsonl` / `validation.jsonl` | `messages` 对话格式的训练集和验证集 |
| `alpaca_train.jsonl` / `alpaca_validation.jsonl` | `instruction/input/output/system` 格式 |
| `membership.jsonl` | 训练样本到 run_id、prompt 分组和 split 的追溯映射 |
| `dataset_info.json` | LLaMA-Factory 数据集字段映射 |
| `manifest.json` | schema hash、接受数、排除原因、去重数、划分策略和质量标记 |

messages 格式，每行一个 JSON 对象：

```json
{"messages":[{"role":"system","content":"实际使用的系统提示词和schema..."},{"role":"user","content":"原始场景描述..."},{"role":"assistant","content":"{\"schema_version\":\"1.0\", ...}"}]}
```

Alpaca 格式：

```json
{"instruction":"原始场景描述...","input":"","output":"完整World JSON字符串...","system":"实际使用的系统提示词和schema..."}
```

上述缩写只用于说明。真正导出的是完整 World JSON 和完整系统提示词，不含省略号。
输出字段是文本，训练时模型学习预测序列化的 Spatial Token，而不是执行游戏代码。

导出规则：

1. 仅接受 `source=llm` 且整个流程成功的记录，排除 fixture、测试替身和失败项。
2. 重新解析 world，核对 schema hash、文件 hash、PNG 标识和原始请求，并重新执行空间
   与 expectations 校验；不只相信历史 `passed` 标记。
3. 相同规范化 prompt 加相同世界去重；重复 prompt 的不同世界保留，但放在相同 split。
4. 按 prompt 组做可复现划分，默认约 80/20；仅一个 prompt 组时全部放训练集。
   规范化包括 Unicode NFKC、空白和大小写，不会识别不同措辞的语义重复。
5. 只导出原始 system/user 加最终有效 assistant，不把错误回答、校验错误或修正对话
   混入基础场景生成 SFT 数据；修正历史保留在 runs 中，可用于后续单独训练修复任务。

数据质量标记为 `automatically_validated_not_human_reviewed`。十个 prompt 用于验证链路和
数据格式，尚未形成规模化训练语料。本阶段完成数据格式准备，未执行 LoRA 训练。
运行记录和数据集保存在本地输出目录，不会随代码提交到公共仓库。

## 模块边界

`world/generator.py` 提供可选 attempt 回调，原有调用方式仍有效；`world/evaluation.py`
处理测试要求；`gameworldlm/pipeline.py` 负责完整流程；`dataset/storage.py` 管理记录；
`dataset/export.py` 负责质量过滤和训练格式；`examples/real_generation.py` 只组织批量调用。
数据来源、日志与训练字段始终留在世界 schema 外。
