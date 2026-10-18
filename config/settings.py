import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROMPT_PATH = PROJECT_ROOT / "config" / "prompts.yaml"

# 选择使用哪个模型配置：ollama / deepseek
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "ollama")

MODEL_PROFILES = {
    "ollama": {
        "model_id": os.getenv("OLLAMA_MODEL_ID", "qwen3.5:9b"),
        "api_base": os.getenv(
            "OLLAMA_API_BASE", "http://localhost:11434/v1"
        ),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
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
MAX_TOKENS = _env_int("AGENT_MAX_TOKENS", 2048)
TEMPERATURE = _env_float("AGENT_TEMPERATURE", 0.5)
VERBOSITY_LEVEL = _env_int("AGENT_VERBOSITY_LEVEL", 2)


