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

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)


MAX_STEPS = 6
MAX_TOKENS = 2048
TEMPERATURE = 0.5
VERBOSITY_LEVEL = 2


