"""memory.context：唯一注入点的行为契约。"""

from memory.context import (
    DEFAULT_USER_ID,
    HISTORY_INSTRUCTION,
    PREFERENCE_HIDDEN_NOTE,
    PREFERENCE_INSTRUCTION,
    augment_task,
    build_augmented_task,
    build_context,
    build_history_context,
)
from memory.retriever import HISTORY_MAX_ITEMS, MemoryRetriever


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


class _HistoryRetriever:
    """只实现历史召回 + 可选的偏好召回，用来验证两个块的拼装顺序。"""

    def __init__(self, history=None, preferences=None):
        self.history = history or []
        self.preferences = preferences or []
        self.history_kwargs = None

    def retrieve_history(self, **kwargs):
        self.history_kwargs = kwargs
        return self.history

    def retrieve_preferences(self, **kwargs):
        return self.preferences

    @staticmethod
    def format_history(memories):
        return "历史块"

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


def test_catalog_reports_how_many_preferences_are_hidden(store):
    """目录被条数上限截断时，必须告诉模型「还有 N 条」。

    否则模型以为目录就是全部，用户也看不出记忆只生效了一部分。
    说明只占一行，且仍然是列表形态（不再有整段提示语）。
    """
    for i in range(5):
        store.save_preference("u", "cat", f"k{i}", f"值{i}")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u", limit=2)

    assert PREFERENCE_HIDDEN_NOTE.format(hidden=3) in result
    assert 'load_user_preferences("")' in result


def test_catalog_without_truncation_has_no_count_note(store):
    """没被截断时不该多出提示，避免每轮都塞一句无用的话。"""
    store.save_preference("u", "cat", "only", "唯一一条")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")

    assert "另有" not in result


def test_catalog_note_is_skipped_when_retriever_cannot_count():
    """替身 retriever 没有计数能力时宁可不提示，也不猜一个数字。"""
    retriever = _RecordingRetriever(memories=[{"memory_key": "k"}])

    result = build_augmented_task("任务", retriever)

    assert "另有" not in result


# ---------- 历史对话块 ----------

def test_build_history_context_includes_instruction():
    retriever = _HistoryRetriever(history=[{"task": "t", "final_answer": "a"}])

    text = build_history_context("任务", retriever)

    assert HISTORY_INSTRUCTION in text
    assert "历史块" in text


def test_build_history_context_empty_when_no_hit():
    assert build_history_context("任务", _HistoryRetriever()) == ""


def test_build_history_context_tolerates_retriever_without_history():
    """只实现偏好检索的替身对象不该被历史召回拖挂。"""
    retriever = _RecordingRetriever(memories=[{"memory_key": "k"}])

    assert build_history_context("任务", retriever) == ""


def test_history_limit_is_clamped_to_hard_cap():
    """调用方要 10 条也只给 2 条：条数上限不允许被绕过。"""
    retriever = _HistoryRetriever(history=[{"task": "t"}])

    build_history_context("任务", retriever, limit=10)

    assert retriever.history_kwargs["limit"] == HISTORY_MAX_ITEMS


def test_history_can_be_disabled_with_zero():
    retriever = _HistoryRetriever(history=[{"task": "t"}])

    assert build_context("任务", retriever, history_limit=0) == ""
    assert retriever.history_kwargs is None  # 0 条时压根不检索


def test_build_context_puts_history_before_preferences():
    retriever = _HistoryRetriever(
        history=[{"task": "t"}], preferences=[{"memory_key": "k"}]
    )

    context = build_context("任务", retriever)

    assert context.index("历史块") < context.index("偏好块")


def test_build_context_with_only_history_skips_preference_instruction():
    retriever = _HistoryRetriever(history=[{"task": "t"}])

    result = build_augmented_task("写个快排", retriever)

    assert HISTORY_INSTRUCTION in result
    assert PREFERENCE_INSTRUCTION not in result
    assert result.rstrip().endswith("写个快排")


# ---------- 偏好：目录注入 + 按需加载 ----------


def test_preference_block_is_catalog_not_full_text(store):
    """长偏好只进目录要点，全文留给 load_user_preferences 按需取。"""
    long_value = "用中文回答，先给结论再给步骤，代码必须可直接运行。" * 10
    store.save_preference("u", "style", "answer_style", long_value)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")

    assert "answer_style" in result
    assert long_value not in result          # 全文没有进 prompt
    assert "…" in result                     # 要点被截断


def test_preference_block_is_plain_list_without_prose(store):
    """偏好块就是一列 `- key：value`：不再塞任何说明文字。

    每轮都进 prompt 的固定开销要小到只剩内容本身；「怎么按需取全文」
    「什么时候该写偏好」这类说明放在 prompts.yaml 的固定规则里。
    """
    store.save_preference("u", "response_style", "answer_format", "分条")
    store.save_preference("u", "response_style", "answer_length", "简短")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    block = build_augmented_task("写个快排", retriever, user_id="u")
    block = block.split(PREFERENCE_INSTRUCTION)[0].strip()

    lines = block.splitlines()
    # 内容对，且顺序不依赖写入先后（目录按 updated_at 排序，同秒写入会并列）
    assert set(lines) == {"- answer_format：分条", "- answer_length：简短"}
    # 每一行都是 `- key：value`：没有标题、没有说明句
    assert all(line.startswith("- ") for line in lines)
    # 不再出现「目录 / 要点 / load_user_preferences」这类说明文字
    assert "目录" not in block
    assert "要点" not in block
    assert "load_user_preferences" not in block


def test_short_preference_value_stays_visible(store):
    """短偏好直接可读，不必为了它多跑一次工具。"""
    store.save_preference("u", "style", "language", "中文")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")

    assert "language" in result
    assert "中文" in result


def test_no_preferences_means_no_catalog_and_no_hint(store):
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")

    assert result == "写个快排"
    assert "load_user_preferences" not in result


def test_preference_catalog_stays_within_budget(store):
    """偏好很多时目录也不能无限膨胀——每轮固定开销要有上限。"""
    for i in range(30):
        store.save_preference("u", "style", f"k{i}", "很长的偏好内容" * 5)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    result = build_augmented_task("写个快排", retriever, user_id="u")
    catalog = result.split(PREFERENCE_INSTRUCTION)[0]

    assert len(catalog) <= 620 + 80  # 目录块预算（含换行与截断说明）


def test_catalog_reads_live_database_each_turn(store):
    """动态更新：写完立刻生效，不需要重启或重建索引。"""
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert "新加的偏好" not in build_augmented_task("任务", retriever, user_id="u")

    store.save_preference("u", "style", "fresh", "新加的偏好")

    assert "新加的偏好" in build_augmented_task("任务", retriever, user_id="u")
