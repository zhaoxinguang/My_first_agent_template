import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# override=True：.env 是唯一权威来源。
# 原因：本目录的 .envrc（direnv）会把 .env 导出成 shell 环境变量，
# 而 load_dotenv 默认 override=False（已有环境变量优先），
# 结果就是「改了 .env 里的 ACTIVE_MODEL / AGENT_* 完全不生效」。
load_dotenv(PROJECT_ROOT / ".env", override=True)

PROMPT_PATH = PROJECT_ROOT / "config" / "prompts.yaml"

# 选择使用哪个模型配置：ollama / deepseek
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "llama")

MODEL_PROFILES = {
    "ollama": {
        "model_id": os.getenv("OLLAMA_MODEL_ID", "qwen3.5:9b"),
        "api_base": os.getenv(
            "OLLAMA_API_BASE", "http://localhost:11434/v1"
        ),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
    },
    # llama 档走同一个 Ollama 服务，只是模型名不同。
    # 存在的意义是「换模型不必改 ollama 档」：实测本机（2GB 显存、纯 CPU）
    # 上 llama3.2:3.2B 生成约 9.7 tok/s，qwen3.5:9b 只有 4.8 tok/s，
    # 需要快速响应的任务可以切到这一档；代价是小模型写代码的可靠性更低。
    # supports_thinking=False：llama3.2 不支持思考，Ollama 会给带
    # reasoning_effort 的请求直接返回 400（不是「无害忽略」）。
    "llama": {
        "model_id": os.getenv("LLAMA_MODEL_ID", "llama3.2:latest"),
        "api_base": os.getenv(
            "LLAMA_API_BASE",
            os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        ),
        "api_key": os.getenv("LLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", "ollama")),
        "supports_thinking": False,
    },
    "deepseek": {
        "model_id": os.getenv("DEEPSEEK_MODEL_ID", "deepseek-flash"),
        "api_base": os.getenv(
            "DEEPSEEK_API_BASE", "https://api.deepseek.com"
        ),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
}

# Embedding 配置不在这里定义：memory/embedding.py 直接读取
# EMBEDDING_PROVIDER / EMBEDDING_BASE_URL / EMBEDDING_MODEL /
# EMBEDDING_API_KEY 这组环境变量，避免出现两份互相矛盾的默认值
# （曾出现 settings 写 text-embedding-3-small、实际生效却是
# nomic-embed-text 的情况）。


def _env_int(name: str, default: int) -> int:
    """从环境变量读整数；空值走默认值，非法值直接报错而不是静默降级。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值：{raw!r}") from None


# 执行预算全部可由 .env 覆盖，避免改行为必须改代码
MAX_STEPS = _env_int("AGENT_MAX_STEPS", 6)
# 默认 8192：本地思考模型（qwen3.x）的思维链会和正文共用同一份预算，
# 2048 时经常「思维链写完就没配额写代码块了」，表现为
# smolagents 报 code parsing 失败（抠不到 ```py 代码块）。
MAX_TOKENS = _env_int("AGENT_MAX_TOKENS", 8192)
TEMPERATURE = _env_float("AGENT_TEMPERATURE", 0.5)
VERBOSITY_LEVEL = _env_int("AGENT_VERBOSITY_LEVEL", 2)

# 思考（思维链）开关，默认 "high"：开。
# 依据实测：在本项目的 agent 场景（system prompt 里已有完整 Thought/Code
# 示例、每步都有 Observation）下，qwen3.5:9b 通常根本不产生思考内容
# ——think=True 与 False 的响应长度和耗时基本一致，所以开着几乎免费。
# 真正需要权衡的是「模型真的开始思考」的那些任务：本机生成只有约
# 4.8 tok/s，300 token 的思维链就是 60 秒，且思维链与正文共用
# num_predict / num_ctx（撞窗口会退化成「只有 Thought 没有代码块」）。
# 想清空这份不确定性就设成 none；只想让它「少想一点」可用 low / minimal。
# 取值对齐 OpenAI / Ollama 的 reasoning_effort：none|minimal|low|medium|high|xhigh|max；
# 设为空字符串表示「什么都不传」，沿用服务端默认（ollama 上即等于开思考）。
VALID_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
REASONING_EFFORT = (os.getenv("AGENT_REASONING_EFFORT", "high") or "").strip().lower()
if REASONING_EFFORT and REASONING_EFFORT not in VALID_REASONING_EFFORTS:
    raise ValueError(
        f"环境变量 AGENT_REASONING_EFFORT 取值非法：{REASONING_EFFORT!r}，"
        f"可选值：{list(VALID_REASONING_EFFORTS)}，或留空表示不传"
    )

# 单次模型请求的超时（秒）与重试次数。
# 原来硬编码 60 秒 / 0 次重试：本地 ollama 首次加载 9B 模型、或 CPU 推理，
# 单步很容易超过 60 秒，直接 ReadTimeout 打挂 Step 1。
# 云端（deepseek）网络抖动也可以靠 AGENT_MAX_RETRIES=1~2 兜一下。
REQUEST_TIMEOUT = _env_float("AGENT_REQUEST_TIMEOUT", 300.0)
MAX_RETRIES = _env_int("AGENT_MAX_RETRIES", 0)



# ---- 联网工具的上下文预算 ----
# 本机实测：llama3.2 单步 30~170 秒，qwen3.5:9b 约 20~40 秒，
# 而每一步都要把**全部历史 Observation** 重新 prefill 一遍（4.6~15 tok/s）。
# 所以「一次工具调用塞多少字」直接决定总耗时，不只是 token 账单。
# 一次 count=10、summary 600~800 字的搜索就能吃掉 8000 字符，
# 折算成本机约 1~2 分钟的纯 prefill。默认收紧到 3 条 / 240 字 / 1800 字总量。
WEB_SEARCH_MAX_RESULTS = _env_int("WEB_SEARCH_MAX_RESULTS", 3)
WEB_SEARCH_ITEM_CHARS = _env_int("WEB_SEARCH_ITEM_CHARS", 240)
WEB_SEARCH_MAX_CHARS = _env_int("WEB_SEARCH_MAX_CHARS", 1800)
# extract：只回传与问题相关的行；visit_webpage：整页正文上限
WEB_EXTRACT_MAX_CHARS = _env_int("WEB_EXTRACT_MAX_CHARS", 1200)
WEB_PAGE_MAX_CHARS = _env_int("WEB_PAGE_MAX_CHARS", 6000)

# ---- 历史步骤的 Observation 折叠 ----
# 只保留最近 N 步的完整 Observation，更早的截成开头一小段。
# 模型要的是「结论和证据」，不是每一步的原始输出；而每多留一步原文，
# 后面每一步都要重新读一遍。设 0 表示不折叠（恢复旧行为）。
AGENT_KEEP_RECENT_OBSERVATIONS = _env_int("AGENT_KEEP_RECENT_OBSERVATIONS", 2)
AGENT_OLD_OBSERVATION_CHARS = _env_int("AGENT_OLD_OBSERVATION_CHARS", 400)

# ---- 每个任务里联网工具最多调几次 ----
# 真实现场：llama3.2 跑「广州今日天气」连搜 7 次（每次换一点措辞绕过去重），
# 单步 40~100 秒，最后仍然编了个 22℃。次数上限是硬刹车：
# 到顶后工具只回一句「用已有结果作答」，模型只能往下走。
# 0 表示不限制。
WEB_SEARCH_MAX_CALLS_PER_TASK = _env_int("WEB_SEARCH_MAX_CALLS_PER_TASK", 4)
WEB_EXTRACT_MAX_CALLS_PER_TASK = _env_int("WEB_EXTRACT_MAX_CALLS_PER_TASK", 3)
WEB_PAGE_MAX_CALLS_PER_TASK = _env_int("WEB_PAGE_MAX_CALLS_PER_TASK", 2)
