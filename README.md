---
title:my agent
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
tags:
- smolagents
- agent
- smolagent
- tool
- agent-course
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## 项目结构

```
app.py                 入口：组装记忆 + agent + UI
agent/factory.py       创建 CodeAgent（模型 + 工具 + prompt）
agent/model.py         模型网关（ollama / deepseek 双 profile）
config/settings.py     全局配置与 .env 读取
config/prompts.yaml    CodeAgent 的 prompt 模板
tools/                 工具集（唯一入口 tools/__init__.py）
memory/                长期记忆：存储、向量、偏好、检索、上下文拼装
Gradio_UI.py           对话界面 + run 记录
documents/             只读文档目录（read_text_file 的数据源）
data/                  SQLite 数据库
```

### 记忆调用链

```
用户输入
  → memory/agent.py::MemoryAgent.run
  → memory/context.py::build_augmented_task（检索偏好 + 拼装，唯一注入点）
  → agent/factory.py::create_agent 产出的 CodeAgent 执行
  → Gradio_UI 记录 run（agent_runs）
```

## 工具清单

所有工具都通过 `tools/__init__.py` 注册，`build_tools()` 返回给 agent 的列表，
`describe_tools()` 打印「用途 + 调用方式 + 示例」。每个工具在 CodeAgent 眼里
就是一个可以直接写的 Python 函数：

| 工具 | 调用方式 | 用途 |
|---|---|---|
| `final_answer` | `final_answer(answer)` | 提交最终答案并结束任务，必须是最后一步 |
| `get_current_time_in_timezone` | `get_current_time_in_timezone("Asia/Shanghai")` | 查询指定时区当前时间 |
| `list_files` | `list_files()` | 列出 `documents/` 下可读的 .txt |
| `read_text_file` | `read_text_file("1.txt", start_line=1, end_line=20)` | 读取 `documents/` 下 .txt，支持行截取 |
| `web_search` | `web_search("关键词")` | DuckDuckGo 搜索，返回标题/链接/摘要 |
| `visit_webpage` | `visit_webpage(url)` | 抓取网页正文转 Markdown（截断约 1 万字符） |

新增工具：在 `tools/` 下用 `@tool` 装饰函数或继承 `Tool`，然后在
`tools/__init__.py` 的 `build_tools()` 里追加，并在 `TOOL_GUIDE` 补一条说明。

## 启动

```bash
cp .env.example .env   # 填模型配置；本地可先起 ollama
pip install -r requirements.txt
python app.py          # 默认只监听本机，不生成公网链接
```

常用环境变量（都能写在 `.env` 里）：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `ACTIVE_MODEL` | `ollama` | 使用哪套模型配置（`ollama` / `deepseek`） |
| `AGENT_MAX_STEPS` | `6` | 单次任务最大步数 |
| `AGENT_MAX_TOKENS` | `2048` | 单次补全 token 上限 |
| `AGENT_TEMPERATURE` | `0.5` | 采样温度 |
| `AGENT_VERBOSITY_LEVEL` | `2` | 日志详细程度 |
| `GRADIO_SHARE` | `0` | 设为 `1` 才生成公网可访问链接 |
| `GRADIO_DEBUG` | `0` | 设为 `1` 开调试模式 |

> `share` 之前是硬编码 `True`，等于每次启动都把界面开到公网；现在默认关闭，
> 需要临时共享时才设 `GRADIO_SHARE=1`。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

用例全部离线可跑——不需要模型、不需要 ollama、不联网，也不碰
`data/agent_memory.db`（每个用例用独立的临时数据库）。覆盖范围：

- `test_store.py` 建表、run 生命周期、偏好版本化（同名 key 只留一条有效）、
  过期过滤、向量 upsert 与 scope 合并；
- `test_retriever.py` 余弦相似度的边界、向量排序、模型不匹配/服务不可用两条降级路径；
- `test_context.py` 偏好注入口径与「未命中就原样放行」；
- `test_preference_manager.py` 向量失败仍落库、`rebuild_embeddings` 统计；
- `test_memory_agent.py` 偏好注入位置、`stream/reset` 等参数透传、`create_agent` 组装；
- `test_file_tools.py` 只读 `documents/`、只允许 `.txt`、`../` 穿越与子目录拦截；
- `test_tools_registry.py` 工具清单与 `TOOL_GUIDE` 一致性、`final_answer` 必须在末尾；
- `test_embedding.py` / `test_settings.py` / `test_gradio_flags.py`
  默认配置、`health_check` 不抛异常、执行预算与 UI 开关的默认值。

## 配置与密钥

- 真实密钥只写在 `.env`，该文件已被 `.gitignore` 忽略；仓库里只有占位模板 `.env.example`。
- `.envrc`（direnv）不含任何明文，只负责把 `.env` 载入环境变量，本身也被忽略。
- `data/` 是运行期目录（SQLite 记忆库），不入库；首次运行 `MemoryStore` 会自动创建。
- 若密钥曾以任何形式提交或外泄，唯一有效的补救是到服务商控制台**吊销并重新签发**，
  再把新值写入本地 `.env`（改代码或删文件都无法让已泄露的旧密钥失效）。

## Embedding（偏好向量检索）

程序**不会**替你启动 embedding 服务，也不会自动 pull 模型，它只连接下列外部服务：

| 环境变量 | 默认值 |
|---|---|
| `EMBEDDING_PROVIDER` | `ollama` |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` |
| `EMBEDDING_MODEL` | `nomic-embed-text` |

本地起 ollama 的做法：

```bash
ollama serve                      # 另开一个终端常驻
ollama pull nomic-embed-text      # 只需一次
python app.py                     # 启动时会打印 [memory] Embedding 服务可用：ollama / nomic-embed-text
```

启动日志就是唯一可靠的判断依据：`EmbeddingClient` 的构造函数不发网络请求，
所以「app 能起来」不代表服务可用。服务不可用时日志会明确打印
`Embedding 服务不可用（...）` 以及上述修复命令，此时偏好检索会**回退到结构化读取**
（丢掉相关性排序，但不会整轮失忆）。

换了 embedding 模型后，旧向量会因 `embedding_model` 不匹配被过滤，需要重建：

```bash
python -c "
from memory.store import MemoryStore
from memory.embedding import EmbeddingClient
from memory.preference_manager import PreferenceManager
print(PreferenceManager(MemoryStore(), EmbeddingClient()).rebuild_embeddings('default_user'))"
```

