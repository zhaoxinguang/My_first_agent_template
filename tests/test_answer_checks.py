"""final_answer 硬校验（agent/answer_checks.py）的测试。

对应真实事故：模型把上一轮「成都 阴天转小雨 14-19°C」的历史答案
当成「广州今日天气」直接提交，且本轮从未调用 web_search。
这层校验要保证：**只有本轮真的调过联网工具，才允许报时效性数据。**
"""

import pytest
from smolagents import Tool
from smolagents.memory import ActionStep, TaskStep
from smolagents.models import ChatMessage, MessageRole, Model

from agent.answer_checks import (
    MAX_REJECTIONS,
    FreshnessCheck,
    ToolCallLedger,
    asserts_concrete_value,
    attach_answer_checks,
    matched_topic,
    record_web_tool_calls,
)


class _FakeMemory:
    """只提供 steps 的最小替身（真 AgentMemory 也只有一个 steps 列表）。"""

    def __init__(self, task, observations="Execution logs:\n结果：56088"):
        # observations 默认非空：默认模拟「成功跑通了一步」，
        # 否则会先被「一步都没成功」那条守卫拦住（另有用例专门测它）
        self.steps = [
            TaskStep(task=task),
            ActionStep(step_number=1, observations=observations),
        ]


# ---------- 关键词判定 ----------

@pytest.mark.parametrize(
    "task",
    [
        "查询广州今日的天气，以可爱风格回答",
        "贵州茅台股价多少",
        "昨天英超比分",
        "帮我搜一下今天的新闻",
        "what is the weather in Guangzhou",
    ],
)
def test_matched_topic_hits_fact_questions(task):
    assert matched_topic(task)


@pytest.mark.parametrize(
    "task",
    [
        "帮我写个 python 装饰器",
        "解释一下快速排序",
        "计算123*456",
        "今天先帮我把这段代码重构一下",   # 时间副词不算时效主题
        "",
    ],
)
def test_matched_topic_ignores_stable_tasks(task):
    assert matched_topic(task) is None


def test_asserts_concrete_value_detects_numbers_and_weather_words():
    assert asserts_concrete_value("广州今日天气：阴天转小雨，气温14-19°C")
    assert asserts_concrete_value("广州今天是晴天")
    assert not asserts_concrete_value("我查不到实时数据，建议你打开天气 App")
    assert not asserts_concrete_value("")


# ---------- 核心校验 ----------

def test_allows_unrelated_task_without_evidence():
    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)

    assert check("123乘456等于56088", _FakeMemory("计算123*456")) is True


def test_allows_honest_fallback_answer():
    """没编数据的兜底回答不拦：拦它只会白花一步。"""
    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)

    answer = "抱歉，我无法获取实时天气数据。"
    assert check(answer, _FakeMemory("查询广州今日的天气")) is True


def test_rejects_fabricated_weather_without_tool_call():
    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)

    with pytest.raises(AssertionError) as excinfo:
        check("广州今日天气：阴天转小雨，气温14-19°C", _FakeMemory("查询广州今日的天气"))

    message = str(excinfo.value)
    assert "web_search" in message      # 告诉模型该怎么补救
    assert "历史对话" in message        # 点明污染来源
    assert check.rejected == 1


def test_passes_when_web_search_was_called():
    ledger = ToolCallLedger()
    ledger.record(
        "web_search", ("广州 今天 天气",), {}, output="广州 晴 气温 34 度 湿度 60%"
    )
    check = FreshnessCheck(ledger)

    assert check("广州今天晴，34°C", _FakeMemory("查询广州今日的天气")) is True
    assert check.rejected == 0


def test_visit_webpage_also_counts_as_evidence():
    ledger = ToolCallLedger()
    ledger.record(
        "visit_webpage",
        ("https://example.com/weather",),
        {},
        output="广州 今天 多云 气温 34 度",
    )
    check = FreshnessCheck(ledger)

    assert check("广州今天晴，34°C", _FakeMemory("查询广州今日的天气")) is True


def test_non_web_tool_is_not_evidence():
    """load_user_preferences 之类的工具不能为时效数据背书。"""
    ledger = ToolCallLedger()
    ledger.record("load_user_preferences", ("",), {})
    check = FreshnessCheck(ledger)

    with pytest.raises(AssertionError):
        check("广州今天晴，34°C", _FakeMemory("查询广州今日的天气"))


def test_stops_rejecting_after_limit():
    """反复给同一个答案时放行，避免把 max_steps 耗光。"""
    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)
    memory = _FakeMemory("查询广州今日的天气")
    answer = "广州今日天气：阴天转小雨，气温14-19°C"

    for _ in range(MAX_REJECTIONS):
        with pytest.raises(AssertionError):
            check(answer, memory)

    assert check(answer, memory) is True


def test_reset_clears_rejection_counter():
    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)
    memory = _FakeMemory("查询广州今日的天气")

    with pytest.raises(AssertionError):
        check("广州今天晴，34°C", memory)

    check.reset()
    assert check.rejected == 0
    with pytest.raises(AssertionError):
        check("广州今天晴，34°C", memory)


def test_task_comes_from_taskstep_not_augmented_action_step():
    """ActionStep 里存的是注入过记忆的任务，不能被注入内容带偏。"""

    class _Memory:
        def __init__(self):
            self.steps = [
                TaskStep(task="帮我写个 python 装饰器"),
                ActionStep(step_number=1, observations="历史：广州今天晴，34°C"),
            ]

    ledger = ToolCallLedger()
    check = FreshnessCheck(ledger)

    # 注入块里出现了「天气」也不能让稳定任务被要求联网
    assert check("装饰器写法如下……", _Memory()) is True


def test_check_name_is_exposed_for_agent_error():
    """smolagents 用 check_function.__name__ 拼 AgentError 消息。"""
    check = FreshnessCheck(ToolCallLedger())
    assert check.__name__ == "check_freshness"


# ---------- 工具记账 ----------

class _FakeWebTool(Tool):
    name = "web_search"
    description = "fake"
    inputs = {"query": {"type": "string", "description": "q"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return f"结果：{query} → 广州晴 34°C"


class _FakeOtherTool(Tool):
    name = "load_user_preferences"
    description = "fake"
    inputs = {"query": {"type": "string", "description": "q"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return "偏好"


def test_record_web_tool_calls_records_and_preserves_output():
    ledger = ToolCallLedger()
    web_tool = _FakeWebTool()

    assert record_web_tool_calls([web_tool], ledger) == ["web_search"]
    # 返回值不变（记账是透明的）
    assert web_tool("广州 天气") == "结果：广州 天气 → 广州晴 34°C"
    assert ledger.names() == ["web_search"]
    assert ledger.calls[0]["args"] == ("广州 天气",)


def test_record_web_tool_calls_skips_non_web_tools():
    ledger = ToolCallLedger()
    other = _FakeOtherTool()

    assert record_web_tool_calls([other], ledger) == []
    other("x")
    assert ledger.calls == []


def test_record_web_tool_calls_is_idempotent():
    """重复挂载不能让一次调用记两笔账。"""
    ledger = ToolCallLedger()
    web_tool = _FakeWebTool()

    record_web_tool_calls([web_tool], ledger)
    record_web_tool_calls([web_tool], ledger)
    web_tool("广州 天气")

    assert ledger.names() == ["web_search"]


def test_ledger_has_any():
    ledger = ToolCallLedger()
    ledger.record("web_search", (), {})
    assert ledger.has_any({"web_search"}) is True
    assert ledger.has_any({"visit_webpage"}) is False


# ---------- 挂载到 agent 上 ----------

class _FakeAgent:
    def __init__(self):
        self.tools = {"web_search": _FakeWebTool()}
        self.runs = 0
        self.final_answer_checks = None

    def run(self, task):
        self.runs += 1
        return task


def test_attach_answer_checks_wires_check_and_run_hook():
    agent = _FakeAgent()
    check = attach_answer_checks(agent)

    assert agent.final_answer_checks == [check]
    assert agent.runs == 0

    agent.run("查询广州今日的天气")
    assert agent.runs == 1                       # 原 run 仍然被调用

    # 上一轮的记账不能为下一轮背书
    agent.tools["web_search"]("广州 天气")
    assert check.ledger.has_any({"web_search"}) is True
    agent.run("查询广州今日的天气")
    assert check.ledger.has_any({"web_search"}) is False


def test_attach_answer_checks_is_idempotent():
    agent = _FakeAgent()
    first = attach_answer_checks(agent)
    second = attach_answer_checks(agent)

    assert first is second
    agent.tools["web_search"]("广州 天气")
    assert first.ledger.names() == ["web_search"]   # 没被记两次


def test_create_agent_attaches_answer_checks(tmp_path, monkeypatch):
    """工厂装配的真实 CodeAgent 也带上了校验（不联网也能构造）。"""
    from agent.factory import create_agent

    agent = create_agent(include_web=False)

    assert agent.final_answer_checks
    check = agent.final_answer_checks[0]
    assert isinstance(check, FreshnessCheck)
    # 离线装配时没有联网工具：记账为空，时效性数据一律要求查证
    assert check.ledger.calls == []


# ---------- 端到端：AgentError 回喂后模型能自纠，任务不中断 ----------

class _ScriptedModel(Model):
    """按剧本逐次作答的模型替身（不联网、不耗 token）。"""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, messages, stop_sequences=None, grammar=None, tools_to_call_from=None, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply)


def test_fabricated_answer_is_rejected_then_self_corrected():
    """复刻真实事故：先编答案 → 被拦 → 补查 → 用真实数据作答。"""
    from smolagents import CodeAgent

    from tools.final_answer import FinalAnswerTool

    model = _ScriptedModel(
        [
            # 第 1 步：不查，直接抄历史
            "```python\nfinal_answer('广州今日天气：阴天转小雨，气温14-19°C')\n```",
            # 第 2 步：被 AgentError 提示后乖乖查
            "```python\nprint(web_search('广州 今天 天气'))\n```",
            # 第 3 步：用查到的数据回答
            "```python\nfinal_answer('广州今天晴，34°C')\n```",
        ]
    )
    agent = CodeAgent(
        model=model,
        tools=[_FakeWebTool(), FinalAnswerTool()],
        max_steps=5,
        verbosity_level=0,
    )
    attach_answer_checks(agent)

    answer = agent.run("查询广州今日的天气，以可爱风格回答")

    assert answer == "广州今天晴，34°C"     # 事故里的 14-19°C 没有漏出去
    assert model.calls == 3                  # 任务没中断，多了自纠一步
    rejected = [step for step in agent.memory.steps if getattr(step, "error", None)]
    assert len(rejected) == 1                # 只被拦一次，模型一次就改对了
    first_step = rejected[0]
    assert "web_search" in str(first_step.error)


# ---------- 兜底路径：max_steps 用尽时不冒充「查证过」 ----------

def test_fallback_answer_gets_unverified_warning():
    """max_steps 用尽走 provide_final_answer，不经过 final_answer_checks。

    真实 llama3.2 实测就是这样漏出去的：7 步全在解析偏好 JSON 时报错，
    兜底答案里写着「阴天，温度在22℃左右」——一次 web_search 都没调。
    """
    from agent.answer_checks import UNVERIFIED_PREFIX

    memory = _FakeMemory("查询广州今日的天气，以可爱风格回答")
    check = FreshnessCheck(ToolCallLedger())

    guarded = check.guard_fallback("广州今天阴天，温度在22℃左右", memory)

    assert guarded.startswith(UNVERIFIED_PREFIX)
    assert "天气" in guarded


def test_fallback_answer_untouched_when_verified():
    memory = _FakeMemory("查询广州今日的天气，以可爱风格回答")
    ledger = _ledger_with_weather_output()
    check = FreshnessCheck(ledger)

    answer = "广州今天晴，34°C，湿度 60%"
    assert check.guard_fallback(answer, memory) == answer


def test_fallback_answer_marked_when_number_has_no_evidence():
    """实测现场：搜过 7 次，兜底答案里的「22℃」哪次结果里都没有。"""
    memory = _FakeMemory("查询广州今日的天气，以可爱风格回答")
    check = FreshnessCheck(_ledger_with_weather_output())

    guarded = check.guard_fallback("广州今日的天气是晴天，温度在22℃左右", memory)

    assert guarded.startswith("⚠️ 未核实：")
    assert "22℃左右" in guarded


def test_fallback_answer_untouched_for_non_fact_task():
    memory = _FakeMemory("帮我写个 python 装饰器")
    check = FreshnessCheck(ToolCallLedger())

    answer = "示例：@cache 装饰器，第 2 行是 wrap"
    assert check.guard_fallback(answer, memory) == answer


def test_max_steps_fallback_is_marked_unverified():
    """端到端复刻 llama3.2 的真实漏法：一步都没查，最后走 max_steps 兜底。"""
    from smolagents import CodeAgent

    from agent.answer_checks import UNVERIFIED_PREFIX
    from tools.final_answer import FinalAnswerTool

    model = _ScriptedModel(
        [
            # 一直在写坏代码，从不调 final_answer
            "```python\nprint(1 / 0)\n```",
            "```python\nprint(1 / 0)\n```",
            # provide_final_answer 用它当兜底答案
            "广州今日天气：阴天，温度在22℃左右。",
        ]
    )
    agent = CodeAgent(
        model=model,
        tools=[_FakeWebTool(), FinalAnswerTool()],
        max_steps=2,
        verbosity_level=0,
    )
    attach_answer_checks(agent)

    answer = agent.run("查询广州今日的天气，以可爱风格回答")

    assert answer.startswith(UNVERIFIED_PREFIX)
    assert "阴天" in answer                       # 原答案保留，但明确标成未核实
    assert agent.final_answer_checks[0].ledger.names() == []   # 确实一次没查


# ---------- 依据校验：答案里的数字要能在工具结果里找到 ----------

def _ledger_with_weather_output():
    ledger = ToolCallLedger()
    ledger.record(
        "web_search",
        ("广州 今天 天气",),
        {},
        output="[1] 广州天气  https://example.com/1\n  广州今天晴，气温34度，湿度60%",
    )
    return ledger


def test_grounded_answer_passes():
    check = FreshnessCheck(_ledger_with_weather_output())

    assert check("广州今天晴，34°C，湿度 60%", _FakeMemory("查询广州今日的天气")) is True


def test_ungrounded_number_is_rejected():
    """照抄历史数据的典型形态：搜到了 34 度，却报 14-19 度。"""
    check = FreshnessCheck(_ledger_with_weather_output())

    with pytest.raises(AssertionError) as exc:
        check("广州今日天气：阴天转小雨，气温14-19°C", _FakeMemory("查询广州今日的天气"))

    message = str(exc.value)
    assert "找不到" in message
    assert "14" in message


def test_grounding_ignores_stable_tasks():
    """算出来的数字本来就不在工具结果里，不能因此拦截。"""
    ledger = ToolCallLedger()
    ledger.record("web_search", ("x",), {}, output="无数字的搜索结果")
    check = FreshnessCheck(ledger)

    assert check("123×456=56088", _FakeMemory("计算123*456")) is True


def test_grounding_skipped_without_evidence():
    """本轮没查过：交给「没查证」那条规则，不走数字比对。"""
    check = FreshnessCheck(ToolCallLedger())

    with pytest.raises(AssertionError) as exc:
        check("广州今天晴，34°C", _FakeMemory("查询广州今日的天气"))

    assert "没有调用过" in str(exc.value)


def test_extract_counts_as_evidence():
    ledger = ToolCallLedger()
    ledger.record(
        "extract", ("https://e.com/1", "气温"), {}, output="广州今天晴，气温34度"
    )
    check = FreshnessCheck(ledger)

    assert check("广州今天晴，34°C", _FakeMemory("查询广州今日的天气")) is True


# ---------- 失败步骤守卫：一步都没跑通时不许给结论 ----------

class _BrokenMemory:
    """所有步骤都报错（真实现场：连续 KeyError / 代码解析失败）。"""

    def __init__(self, task):
        self.steps = [
            TaskStep(task=task),
            ActionStep(step_number=1, error=KeyError("response_style")),
            ActionStep(step_number=2, error=NameError("user_preferences")),
        ]


def test_all_failed_steps_blocks_concrete_answer():
    check = FreshnessCheck(ToolCallLedger())

    # 用非时效任务：时效任务会先被「没查证」那条规则拦下（另有用例覆盖）
    with pytest.raises(AssertionError) as exc:
        check("账单总额是 1234 元", _BrokenMemory("读一下 documents 里的账单，算总额"))

    assert "没有任何一步成功" in str(exc.value)


def test_all_failed_steps_allows_honest_failure_report():
    """「我失败/查不到」这种回答不该被拦，否则白白多花一步。"""
    check = FreshnessCheck(ToolCallLedger())

    assert check("抱歉，我没能取到数据。", _BrokenMemory("查询广州今日的天气")) is True


def test_one_successful_step_is_enough():
    memory = _FakeMemory("查询广州今日的天气")
    check = FreshnessCheck(_ledger_with_weather_output())

    assert check("广州今天晴，34°C", memory) is True


def test_usable_steps_counts_only_successful_ones():
    from agent.answer_checks import usable_steps

    assert usable_steps(_BrokenMemory("任意任务")) == 0
    assert usable_steps(_FakeMemory("任意任务")) == 1


# ---------- 数字抽取 ----------

def test_key_numbers_ignores_single_digits():
    from agent.answer_checks import key_numbers

    assert key_numbers("第1步，气温 34 度，湿度 60%") == ["34", "60"]


def test_ungrounded_numbers_fullwidth_and_spacing():
    from agent.answer_checks import ungrounded_numbers

    # 全角数字与多空格都要能对上
    assert ungrounded_numbers("气温３４度", ["气温 34 度"]) == []
