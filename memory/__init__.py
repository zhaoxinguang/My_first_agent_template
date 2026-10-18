"""长期记忆模块。

- store：SQLite 持久化（任务历史、执行轨迹、偏好、向量）
- embedding：文本向量化客户端
- preference_manager：偏好的写入/更新/遗忘/向量重建
- retriever：偏好与历史任务的召回
- context：记忆 → prompt 的唯一拼装点
- agent：MemoryAgent，执行前注入记忆
"""

from memory.agent import MemoryAgent
from memory.context import (
    DEFAULT_USER_ID,
    augment_task,
    build_augmented_task,
    build_context,
)
from memory.embedding import EmbeddingClient
from memory.preference_manager import PreferenceManager
from memory.retriever import MemoryRetriever
from memory.store import MemoryStore

__all__ = [
    "MemoryAgent",
    "MemoryStore",
    "MemoryRetriever",
    "PreferenceManager",
    "EmbeddingClient",
    "DEFAULT_USER_ID",
    "build_context",
    "augment_task",
    "build_augmented_task",
]
