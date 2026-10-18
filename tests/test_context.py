"""memory.context：唯一注入点的行为契约。"""

from memory.context import (
    DEFAULT_USER_ID,
    PREFERENCE_INSTRUCTION,
    augment_task,
    build_augmented_task,
    build_context,
)
from memory.retriever import MemoryRetriever


class _BrokenRetriever:
    def retrieve_preferences(self, **kwargs):
        raise ConnectionError("embedding 服务挂了")


class _RecordingRetriever:
    """记录调用参数，用于断言注入点的入参透传。"""

    def __init__(self, memories=None):
        self.memories = memories or []
        self.kwargs = None

    def retrieve_preferences(self, **kwargs):
        self.kwargs = kwargs
        return self.memories

    @staticmethod
    def format_preferences(memories):
        return "偏好块"


def test_default_user_id_is_single_user_placeholder():
    assert DEFAULT_USER_ID == "default_user"


def test_empty_task_returns_empty_context():
    retriever = _RecordingRetriever()
    assert build_context("", retriever) == ""
    assert build_context("   ", retriever) == ""


def test_no_memories_returns_empty_context():
    """没有命中时返回空串，而不是塞一句「没检索到」占 token。"""
    assert build_context("任务", _RecordingRetriever()) == ""


def test_build_context_passes_query_and_scope():
    retriever = _RecordingRetriever(memories=[{"memory_key": "k"}])
    build_context("任务", retriever, user_id="u1", scope="proj", limit=3)

    assert retriever.kwargs == {
        "query": "任务",
        "user_id": "u1",
        "scope": "proj",
        "limit": 3,
    }


def test_retriever_exception_is_swallowed():
    """检索失败不能阻断对话。"""
    assert build_context("任务", _BrokenRetriever()) == ""


def test_augment_task_passthrough_when_no_context():
    assert augment_task("原始任务", "") == "原始任务"


def test_augment_task_keeps_original_task_and_instruction():
    task = augment_task("原始任务", "偏好块")

    assert "偏好块" in task
    assert PREFERENCE_INSTRUCTION in task
    assert task.rstrip().endswith("原始任务")


def test_instruction_forbids_overriding_explicit_request():
    assert "以当前指令为准" in PREFERENCE_INSTRUCTION


def test_build_augmented_task_end_to_end(store):
    store.save_preference("u", "format", "language", "中文")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")

    assert "language" in result
    assert "中文" in result
    assert PREFERENCE_INSTRUCTION in result
    assert result.rstrip().endswith("写个快排")


def test_build_augmented_task_without_preferences_keeps_task_intact(store):
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert build_augmented_task("写个快排", retriever, user_id="u") == "写个快排"
