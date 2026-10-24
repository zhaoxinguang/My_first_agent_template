"""TurnDistiller：每轮结束的自动提炼（偏好 + 本轮摘要）。"""

import json

import pytest

from memory.distiller import (
    DEFAULT_MAX_PREFERENCES,
    MAX_VALUE_CHARS,
    TurnDistiller,
)
from memory.preference_manager import PreferenceManager


class _Message:
    """模拟 smolagents 的 ChatMessage：提炼器只读 .content。"""

    def __init__(self, content):
        self.content = content


class FakeModel:
    """可编程的模型替身：记录调用、可返回固定 JSON、可模拟故障。"""

    def __init__(self, payload=None, fail=False):
        self.payload = payload
        self.fail = fail
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("model down")
        if isinstance(self.payload, str):
            return _Message(self.payload)
        return _Message(json.dumps(self.payload, ensure_ascii=False))


def _run(store, task="写个快排", answer="这是快排代码"):
    run_id = store.start_run(task)
    store.finish_run(run_id, final_answer=answer, status="success")
    return run_id


def _distiller(store, payload=None, **kwargs):
    model = kwargs.pop("model", None) or FakeModel(payload)
    return TurnDistiller(
        store=store,
        embedding_client=None,
        model=model,
        enabled=kwargs.pop("enabled", True),
        **kwargs,
    )


PAYLOAD = {
    "preferences": [
        {
            "key": "answer_style",
            "value": "回答简洁，先给可运行代码",
            "category": "response_style",
            "confidence": 0.9,
        }
    ],
    "summary": "用户要快排实现，给出了 Python 递归版并说明复杂度。",
    "facts": ["快排平均复杂度 O(n log n)"],
}


def test_distill_saves_preference_and_summary(store):
    run_id = _run(store)
    distiller = _distiller(store, PAYLOAD)

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["preferences_added"] == 1
    assert stats["summary_saved"] is True
    prefs = distiller.manager.list_preferences("default_user")
    assert [p["memory_key"] for p in prefs] == ["answer_style"]
    assert prefs[0]["memory_value"] == "回答简洁，先给可运行代码"

    memory = store.get_turn_memory(run_id)
    assert memory["summary"].startswith("用户要快排实现")
    assert memory["facts"] == ["快排平均复杂度 O(n log n)"]


def test_distill_records_source_run_id(store):
    """能追到「这条偏好是哪一轮学来的」——工具路径做不到，提炼器可以。"""
    run_id = _run(store)
    distiller = _distiller(store, PAYLOAD)

    distiller.distill(run_id, "写个快排", "这是快排代码")

    assert distiller.manager.list_preferences("default_user")[0]["source_run_id"] == run_id


def test_distill_is_idempotent_for_same_run(store):
    """同一轮重复触发（UI 重试、进程重启补跑）不能写出两条偏好。"""
    run_id = _run(store)
    model = FakeModel(PAYLOAD)
    distiller = _distiller(store, model=model)

    distiller.distill(run_id, "写个快排", "这是快排代码")
    second = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert second["skipped"] == "already_distilled"
    assert len(model.calls) == 1                 # 第二次连模型都没调
    assert len(distiller.manager.list_preferences("default_user")) == 1


def test_distill_survives_bad_json(store):
    run_id = _run(store)
    distiller = _distiller(store, "我觉得你应该先装依赖……")

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["skipped"] == "bad_model_output"
    assert stats["preferences_added"] == 0
    assert store.get_turn_memory(run_id) is None


def test_distill_survives_model_failure(store):
    run_id = _run(store)
    distiller = _distiller(store, model=FakeModel(fail=True))

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["skipped"] == "model_failed"
    assert stats["error"]


def test_distill_skips_empty_turn(store):
    distiller = _distiller(store, PAYLOAD)

    assert distiller.distill(1, "", "答案")["skipped"] == "empty_turn"
    assert distiller.distill(1, "问题", "   ")["skipped"] == "empty_turn"


def test_disabled_distiller_makes_no_model_call(store):
    run_id = _run(store)
    model = FakeModel(PAYLOAD)
    distiller = _distiller(store, model=model, enabled=False)

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["skipped"] == "disabled"
    assert model.calls == []


def test_budget_rejection_keeps_summary(store):
    """偏好写不进去（预算满）也不该妨碍记录本轮摘要。"""
    run_id = _run(store)
    distiller = _distiller(store, PAYLOAD)
    distiller.manager = PreferenceManager(
        store=store, embedding_client=None, char_limit=5
    )

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["preferences_skipped"] == 1
    assert stats["preferences_added"] == 0
    assert stats["summary_saved"] is True
    assert store.get_turn_memory(run_id) is not None


def test_long_preference_value_is_clipped(store):
    """偏好要短到「目录里的要点就是全文」，否则小模型还得再调一次工具。"""
    run_id = _run(store)
    distiller = _distiller(
        store,
        {
            "preferences": [{"key": "style", "value": "很长的偏好" * 50}],
            "summary": "摘要",
        },
    )

    distiller.distill(run_id, "写个快排", "这是快排代码")

    value = distiller.manager.list_preferences("default_user")[0]["memory_value"]
    assert len(value) <= MAX_VALUE_CHARS


def test_max_preferences_per_turn_is_capped(store):
    run_id = _run(store)
    distiller = _distiller(
        store,
        {
            "preferences": [
                {"key": f"k{i}", "value": f"v{i}"} for i in range(8)
            ],
            "summary": "摘要",
        },
    )

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["preferences_added"] == DEFAULT_MAX_PREFERENCES
    assert len(distiller.manager.list_preferences("default_user")) == DEFAULT_MAX_PREFERENCES


def test_preferences_are_ignored_when_invalid(store):
    """模型常会给空 key / 超长 key / 非法 confidence，这些必须被挡掉。"""
    run_id = _run(store)
    distiller = _distiller(
        store,
        {
            "preferences": [
                {"key": "", "value": "没有 key"},
                {"key": "k" * 200, "value": "超长 key"},
                {"key": "ok", "value": ""},
                {"key": "good", "value": "正常偏好"},
            ],
            "summary": "摘要",
        },
    )

    stats = distiller.distill(run_id, "写个快排", "这是快排代码")

    assert stats["preferences_added"] == 1
    assert [p["memory_key"] for p in distiller.manager.list_preferences("default_user")] == ["good"]


def test_same_key_across_turns_updates_instead_of_duplicating(store):
    """两轮学到同一件事时应该是覆盖，不是堆两条。"""
    distiller = _distiller(
        store,
        {
            "preferences": [{"key": "answer_style", "value": "第一轮版本"}],
            "summary": "第一轮",
        },
    )
    first = _run(store, task="问 A", answer="答 A")
    distiller.distill(first, "问 A", "答 A")

    distiller._model = FakeModel(
        {
            "preferences": [{"key": "answer_style", "value": "第二轮版本"}],
            "summary": "第二轮",
        }
    )
    second = _run(store, task="问 B", answer="答 B")
    distiller.distill(second, "问 B", "答 B")

    prefs = distiller.manager.list_preferences("default_user")
    assert len(prefs) == 1
    assert prefs[0]["memory_value"] == "第二轮版本"


def test_summary_falls_back_to_answer_when_model_returns_none(store):
    """模型只给偏好不给摘要时，本轮记忆也要落库（否则幂等闸门失效）。"""
    run_id = _run(store)
    distiller = _distiller(
        store, {"preferences": [{"key": "k", "value": "v"}]}
    )

    distiller.distill(run_id, "写个快排", "这是快排代码")

    memory = store.get_turn_memory(run_id)
    assert memory is not None
    assert memory["summary"].strip()


def test_distill_async_runs_in_background_and_swallows_errors(store):
    run_id = _run(store)
    distiller = _distiller(store, model=FakeModel(fail=True))

    thread = distiller.distill_async(run_id, "写个快排", "这是快排代码")
    thread.join(timeout=5)

    assert not thread.is_alive()


def test_distill_async_success_path(store):
    run_id = _run(store)
    distiller = _distiller(store, PAYLOAD)

    thread = distiller.distill_async(run_id, "写个快排", "这是快排代码")
    thread.join(timeout=5)

    assert store.get_turn_memory(run_id) is not None


def test_enabled_defaults_from_env(store, monkeypatch):
    monkeypatch.setenv("PREF_AUTO_EXTRACT", "0")

    distiller = TurnDistiller(store=store, embedding_client=None, model=FakeModel())

    assert distiller.enabled is False


def test_learned_preference_reaches_next_turn_prompt(store):
    """端到端：第 N 轮提炼出的偏好，第 N+1 轮的 prompt 里必须能看到。

    这是「每轮自动提炼」的意义所在——不依赖模型主动调任何工具。
    """
    from memory.agent import MemoryAgent
    from memory.context import DEFAULT_USER_ID
    from memory.retriever import MemoryRetriever

    # 第 1 轮：用户表达偏好
    run_id = _run(store, task="以后回答都用中文", answer="好的，之后都用中文。")
    distiller = _distiller(
        store,
        {
            "preferences": [
                {"key": "answer_language", "value": "始终用中文回答"}
            ],
            "summary": "用户要求之后都用中文回答。",
        },
    )
    distiller.distill(run_id, "以后回答都用中文", "好的，之后都用中文。")

    # 第 2 轮：不带任何工具调用，只靠注入
    class _Recorder:
        def __init__(self):
            self.tasks = []

        def run(self, task, *args, **kwargs):
            self.tasks.append(task)
            return "ok"

    inner = _Recorder()
    memory_agent = MemoryAgent(
        agent=inner,
        retriever=MemoryRetriever(store=store, embedding_client=None),
        user_id=DEFAULT_USER_ID,
    )
    memory_agent.run("写个快排")

    assert "始终用中文回答" in inner.tasks[0]


def test_value_fits_inside_catalog_gist(store):
    """提炼出的偏好值必须短到目录要点能完整显示（不必再调工具）。"""
    from memory.retriever import PREF_INDEX_VALUE_CHARS

    assert MAX_VALUE_CHARS <= PREF_INDEX_VALUE_CHARS


def test_newly_distilled_preference_reaches_prompt_in_a_crowded_catalog(store):
    """回归：目录条数已满时，本轮提炼出的偏好下一轮仍必须进 prompt。

    这是「动态存储」的最后一公里：目录是自动记忆唯一的注入入口，
    一旦新学到的偏好被老条目挤出目录，提炼得再准也等于没学到。
    """
    from memory.agent import MemoryAgent
    from memory.context import DEFAULT_USER_ID
    from memory.retriever import MemoryRetriever

    retriever = MemoryRetriever(store=store, embedding_client=None)
    for i in range(10):
        store.save_preference("u", "cat", f"old_{i}", f"老偏好{i}")
        retriever.load_preferences(query=f"old_{i}", user_id="u")

    run_id = _run(store, task="以后回答都用中文", answer="好的，之后都用中文。")
    _distiller(
        store,
        {
            "preferences": [
                {"key": "answer_language", "value": "始终用中文回答"}
            ],
            "summary": "用户要求之后都用中文回答。",
        },
    ).distill(run_id, "以后回答都用中文", "好的，之后都用中文。")

    class _Recorder:
        def __init__(self):
            self.tasks = []

        def run(self, task, *args, **kwargs):
            self.tasks.append(task)
            return "ok"

    inner = _Recorder()
    MemoryAgent(
        agent=inner,
        retriever=MemoryRetriever(store=store, embedding_client=None),
        user_id=DEFAULT_USER_ID,
    ).run("写个快排")

    assert "始终用中文回答" in inner.tasks[0]


# ---------- 归一：同义 key 收敛 + 英文翻译噪声过滤 ----------

def test_normalize_collapses_free_form_detail_keys(store):
    """模型每轮自己造 key，同一维度必须收敛成一条。"""
    distiller = _distiller(store)

    prefs = distiller._normalize_preferences([
        {"key": "detail_level", "value": "简短", "confidence": 0.7},
        {"key": "detail_degree", "value": "short", "confidence": 0.8},
        {"key": "verbosity", "value": "简短", "confidence": 0.9},
    ])

    assert [p["key"] for p in prefs] == ["answer_length"]
    # 同 key 多条时保留置信度最高的那条，而不是「谁先来算谁」
    assert prefs[0]["confidence"] == 0.9


def test_normalize_moves_style_value_to_response_style(store):
    """``answer_format：可爱风格`` 说的是语气，落到 response_style 才不会
    被后来的 ``answer_format：text`` 顶掉（真实数据里就这么丢过）。"""
    distiller = _distiller(store)

    prefs = distiller._normalize_preferences([
        {"key": "answer_format", "value": "可爱风格", "confidence": 0.9},
    ])

    assert prefs[0]["key"] == "response_style"
    assert prefs[0]["value"] == "可爱风格"


def test_normalize_drops_english_translation_noise(store):
    """白名单之外的纯英文单词值 = 模型把中文偏好翻译了，整条丢掉。"""
    distiller = _distiller(store)

    prefs = distiller._normalize_preferences([
        {"key": "answer_format", "value": "informal", "confidence": 0.8},
        {"key": "detail_level", "value": "low", "confidence": 0.7},
        {"key": "answer_length", "value": "简短", "confidence": 0.8},
    ])

    assert [p["key"] for p in prefs] == ["answer_length"]


def test_normalize_canonicalizes_category(store):
    distiller = _distiller(store)

    prefs = distiller._normalize_preferences([
        {"key": "detail_degree", "value": "short", "category": "detail_degree"},
        {"key": "python_cmd", "value": "用 uv 跑", "category": "tooling"},
    ])

    assert [p["category"] for p in prefs] == ["response_style", "tooling"]


def test_extract_prompt_pins_canonical_keys(store):
    """提示词必须点名固定 key，且不能再出现「英文小写下划线」这种诱导。

    真实事故：示例里写着 ``"key": "英文小写下划线"``，小模型把它读成
    「语言偏好应该是英文」，于是凭空写出一条 language：english。
    """
    from memory.distiller import EXTRACT_INSTRUCTION

    for key in ("language", "answer_length", "answer_format", "response_style"):
        assert key in EXTRACT_INSTRUCTION
    assert "英文小写下划线" not in EXTRACT_INSTRUCTION
    assert "明确要求" in EXTRACT_INSTRUCTION


def test_extract_prompt_lists_known_preferences(store):
    """已知偏好要喂进提示词，模型才会复用 key 而不是重造一条同义记录。"""
    store.save_preference("u", "response_style", "answer_length", "简短")
    model = FakeModel({"preferences": [], "summary": "无"})
    distiller = _distiller(store, model=model)
    run_id = _run(store)

    distiller.distill(run_id, "写个快排", "这是快排代码")

    prompt = model.calls[0][0]["content"]
    assert "已知偏好" in prompt
    assert "answer_length" in prompt
