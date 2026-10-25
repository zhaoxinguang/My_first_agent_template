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


# ---------------------------------------------------------------------------
# 工具名拼错：把「没有这个函数」改写成「你大概想用 X」
#
# 真实现场（查高铁班次）：llama3.2 把 load_user_preferences 写成单数，
# 撞上 Forbidden function evaluation 后彻底放弃这个工具，改成
# result.split("\n") 直接把原始搜索结果当答案交出去。
# ---------------------------------------------------------------------------

_NAME_ERROR = (
    "Code execution failed at line 'x = load_user_preference(query=\"a\")' due to: "
    "InterpreterError: Forbidden function evaluation: 'load_user_preference' is not "
    "among the explicitly allowed tools or defined/imported in the preceding code"
)


def _name_error_messages(text=_NAME_ERROR):
    return [{"role": "tool-response", "content": [{"type": "text", "text": text}]}]


def test_repair_tool_name_points_at_the_closest_tool():
    from agent.guardrails import repair_tool_name_errors

    fixed = repair_tool_name_errors(_name_error_messages(), ["load_user_preferences", "web_search"])

    text = fixed[0]["content"][0]["text"]
    assert "load_user_preferences" in text
    assert "没有名为 load_user_preference 的工具" in text
    # 必须给下一步动作，否则小模型不知道该重跑还是该换做法
    assert "重新执行这一步" in text
    assert "不要改用字符串切分" in text


def test_repair_tool_name_keeps_original_code_for_audit():
    from agent.guardrails import repair_tool_name_errors

    fixed = repair_tool_name_errors(_name_error_messages(), ["load_user_preferences"])

    # 模型自己的那行代码要留着，它得知道自己写错了什么
    assert "load_user_preference(query=" in fixed[0]["content"][0]["text"]
    # 原来的英文模板不该再出现在模型眼前
    assert "is not among the explicitly allowed tools" not in fixed[0]["content"][0]["text"]


def test_repair_tool_name_lists_available_tools_when_nothing_is_close():
    from agent.guardrails import repair_tool_name_errors

    text = (
        "InterpreterError: Forbidden function evaluation: 'zzzqqq' is not among "
        "the explicitly allowed tools"
    )
    fixed = repair_tool_name_errors(_name_error_messages(text), ["web_search", "extract"])

    out = fixed[0]["content"][0]["text"]
    assert "可用工具：extract, web_search" in out


def test_repair_tool_name_leaves_other_observations_alone():
    from agent.guardrails import repair_tool_name_errors

    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": _NAME_ERROR}]},
        {"role": "tool-response", "content": [{"type": "text", "text": "正常的工具结果"}]},
    ]
    fixed = repair_tool_name_errors(messages, ["load_user_preferences"])

    # assistant 自己写的话不动（那是模型的输出，不是回执）
    assert fixed[0]["content"][0]["text"] == _NAME_ERROR
    assert fixed[1]["content"][0]["text"] == "正常的工具结果"


def test_repair_tool_name_does_not_mutate_memory_steps():
    from agent.guardrails import repair_tool_name_errors

    original = _name_error_messages()
    snapshot = original[0]["content"][0]["text"]
    repair_tool_name_errors(original, ["load_user_preferences"])

    assert original[0]["content"][0]["text"] == snapshot
