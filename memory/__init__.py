"""长期记忆模块。

- store：SQLite 持久化（任务历史、执行轨迹、偏好、向量、每轮提炼记忆）
- embedding：文本向量化客户端
- preference_manager：偏好的写入/更新/遗忘/向量重建（写侧，带字符预算）
- history：历史问答的向量索引（写入 / 历史数据补建）
- retriever：历史问答的向量召回 + 偏好的「目录 / 按需完整加载」
- context：记忆 → prompt 的唯一拼装点
- agent：MemoryAgent，执行前注入记忆

偏好没有自动提炼环节：写不写、写什么，由模型在任务收尾时自行决断并调用
``save_user_preference``（见 config/prompts.yaml 的规则 16）。
"""

from memory.agent import MemoryAgent
from memory.context import (
    DEFAULT_USER_ID,
    augment_task,
    build_augmented_task,
    build_context,
    build_history_context,
)
from memory.embedding import EmbeddingClient
from memory.history import HistoryIndex
from memory.preference_manager import (
    PreferenceBudgetError,
    PreferenceManager,
)
from memory.retriever import MemoryRetriever
from memory.store import MemoryStore

__all__ = [
    "MemoryAgent",
    "MemoryStore",
    "MemoryRetriever",
    "PreferenceManager",
    "PreferenceBudgetError",
    "HistoryIndex",
    "EmbeddingClient",
    "DEFAULT_USER_ID",
    "build_context",
    "build_history_context",
    "augment_task",
    "build_augmented_task",
]
