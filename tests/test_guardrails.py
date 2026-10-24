"""运行时护栏：旧 Observation 折叠 + 每轮重置。

折叠只作用于「发给模型的消息副本」，不改 memory.steps——原始记录要留着
给日志和事后审计看。这里同时断言这两件事。
"""

import pytest

pytest.importorskip("smolagents")

from agent.guardrails import (  # noqa: E402
    FOLDED_MARKER,
    ObservationTrimmer,
    attach_guards,
    reset_tool_state,
)
from smolagents.memory import ActionStep, TaskStep  # noqa: E402
from tools.web_tools import TaskScopedCache  # noqa: E402

LONG = "结果：" + "字" * 500


def _messages(*observations):
    """构造一个最小的消息列表：每步一个 tool-response。"""
    messages = [{"role": "system", "content": [{"type": "text", "text": "sys prompt"}]}]
    for text in observations:
        messages.append(
            {
                "role": "tool-response",
                "content": [{"type": "text", "text": f"Observation:\n{text}"}],
            }
        )
    return messages


def _text(message):
    return message["content"][0]["text"]


# ---------- 折叠 ----------

def test_keeps_recent_observations_intact():
    trimmer = ObservationTrimmer(keep_recent=1, old_chars=100)
    messages = _messages("第一步" + LONG, "最后一步" + LONG)

    out = trimmer(messages)

    # 最近 1 步原样，较早的折叠
    assert _text(out[-1]).endswith(LONG)
    assert FOLDED_MARKER.format(omitted=trimmer.saved_chars) in _text(out[1])
    assert trimmer.folded_steps == 1


def test_no_folding_when_steps_within_budget():
    trimmer = ObservationTrimmer(keep_recent=3, old_chars=100)
    messages = _messages("a" * 500, "b" * 500)

    out = trimmer(messages)

    assert out == messages
    assert trimmer.folded_steps == 0


def test_keep_recent_zero_disables_folding():
    trimmer = ObservationTrimmer(keep_recent=0, old_chars=100)
    messages = _messages("a" * 500, "b" * 500)

    assert trimmer(messages) == messages


def test_short_observation_is_not_touched():
    trimmer = ObservationTrimmer(keep_recent=1, old_chars=100)
    messages = _messages("短结果", "a" * 500)

    out = trimmer(messages)

    assert _text(out[1]) == "Observation:\n短结果"


def test_folded_marker_reports_omitted_chars():
    trimmer = ObservationTrimmer(keep_recent=1, old_chars=100)
    messages = _messages("x" * 300, "latest")

    out = trimmer(messages)

    text = _text(out[1])
    assert "已折叠" in text
    assert text.startswith("Observation:\nx" * 1)
    assert trimmer.saved_chars > 0


def test_trimmer_does_not_mutate_memory_steps():
    """折叠只影响「发给模型的副本」，原始 step 记录必须完好。"""

    class _Agent:
        def __init__(self):
            self.tools = {}
            self.final_answer_checks = []
            self.memory = type("M", (), {"steps": []})()

    agent = _Agent()
    agent.memory.steps = [
        TaskStep(task="查询广州今日的天气"),
        ActionStep(step_number=1, observations=LONG),
        ActionStep(step_number=2, observations=LONG),
    ]
    # 3 步输出：默认 keep_recent=2，所以只有第 1 步会被折叠
    original_write = lambda *a, **k: _messages(LONG, LONG, LONG)   # noqa: E731
    agent.write_memory_to_messages = original_write
    agent.run = lambda *a, **k: "ok"                          # noqa: E731

    guards = attach_guards(agent)
    out = agent.write_memory_to_messages()

    assert guards["trimmer"].folded_steps == 1
    assert agent.memory.steps[1].observations == LONG          # 原文没被改
    assert FOLDED_MARKER.split("{")[0] in _text(out[1])   # out[0] 是 system prompt


# ---------- 每轮重置 ----------

def test_reset_tool_state_clears_each_tool_cache():
    class _ToolWithCache:
        name = "web_search"

        def __init__(self):
            self.cache = TaskScopedCache()

    class _ToolWithBeginTask:
        name = "extract"

        def __init__(self):
            self.calls = 0

        def begin_task(self):
            self.calls += 1

    first, second = _ToolWithCache(), _ToolWithBeginTask()
    first.cache.put("k", "v")
    agent = type("A", (), {"tools": {"web_search": first, "extract": second}})()

    reset_names = reset_tool_state(agent)

    assert sorted(reset_names) == ["extract", "web_search"]
    assert first.cache.get("k") is None
    assert second.calls == 1


def test_run_resets_trimmer_counters_and_tool_cache():
    class _Tool:
        name = "web_search"

        def __init__(self):
            self.cache = TaskScopedCache()

        def forward(self, query):
            return "结果"

        def __call__(self, query):
            return self.forward(query)

    tool = _Tool()
    tool.cache.put("q", "cached")

    class _Agent:
        def __init__(self):
            self.tools = {"web_search": tool}
            self.final_answer_checks = []
            self.memory = type("M", (), {"steps": []})()
            self.write_memory_to_messages = lambda *a, **k: _messages(LONG, LONG, LONG)

        def run(self, task):
            return "answer"

    agent = _Agent()
    guards = attach_guards(agent)
    guards["trimmer"].folded_steps = 7           # 模拟上一轮留下的计数

    assert agent.run("查询广州今日的天气") == "answer"
    assert guards["trimmer"].folded_steps == 0   # 每轮清零
    assert tool.cache.get("q") is None           # 工具缓存也清空
