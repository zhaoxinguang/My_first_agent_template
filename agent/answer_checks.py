"""最终答案硬校验：把「不许凭空报时效性事实」从提示词约束升级成代码约束。

起因（真实事故）：
    用户问「查询广州今日的天气」，本地 llama3.2 第二步直接
    ``final_answer("广州今日天气：阴天转小雨，气温14-19°C")``——
    这串数字来自上一轮**成都**的答案，是从历史注入块里抄来的。
    事后排查：``web_search("广州 今天 天气")`` 本身完全正常，
    返回了 4172 字符真实数据（广州 34°C 晴）；模型压根没查。

    只在提示词里写「实时信息必须重新查询」治不住它：本地小模型
    会照抄注入块里离得最近的那段答案。所以加一层**代码级**校验。

机制（smolagents 1.13）：
    ``MultiStepAgent.__init__(final_answer_checks=[...])``；
    提交 final_answer 时调用 ``check(final_answer, self.memory)``；
    check 抛 AssertionError → 包装成 ``AgentError`` →
    ``_run`` 里 ``except AgentError: action_step.error = e``。
    也就是说**任务不中断**：错误文本作为 Observation 回喂模型，
    模型下一步可以自己纠正（去调 web_search 再回答）。

两个设计要点：

1. 取证不能只看 ``memory.steps``。
   ``_validate_final_answer`` 在 ``_finalize_step`` **之前**执行，
   当前这一步还没进 ``memory.steps``；如果模型在同一段代码里
   ``r = web_search(...)`` 后立刻 ``final_answer(r)``，
   扫 memory 会误判成「没查过」。所以用 ``ToolCallLedger``：
   在工具对象上挂一层记录，调用发生时就记账。

2. 必须有放行上限（``max_rejections``）。
   模型可能反复给同一个答案；无限拦截会把 ``max_steps`` 耗光，
   最终走 ``provide_final_answer`` 兜底，反而更糟。
   拦过 N 次仍不改就放行，并把问题留给上层（日志里能看到）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

# ---- 取证：哪些工具的调用才算「本轮真的查过」 ----
# 只认「从外部取回数据」的工具。load_user_preferences 之类不算，
# 它只是把偏好取出来，不产生任何时效性事实。
WEB_TOOL_NAMES = frozenset({"web_search", "visit_webpage", "extract"})

# 账本里每个工具调用最多留多少输出字符：用于「答案里的数字有没有依据」的核对
OUTPUT_KEEP_CHARS = 4000

# 答案里被认为「值得核对依据」的数字：至少两位，或带小数
_KEY_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_MIN_KEY_NUMBER_DIGITS = 2

# ---- 判定：任务是否在问「此刻的事实」 ----
# 注意这里比 memory/retriever.py 的 TIME_SENSITIVE_HINTS **更窄**：
# 那边是「命中就不注入历史」，宁可放宽（误判只是少一条历史）；
# 这里是「命中就必须查证」，误判会让正常任务被拦一次（多花一步）。
# 所以这里只收**主题词**，不收「今天 / 现在 / 最新」这类时间副词——
# 「今天帮我写个装饰器」不该被要求联网。
CURRENT_FACT_TOPICS = (
    "天气", "气温", "温度", "下雨", "降雨", "台风", "空气质量",
    "股价", "股票", "汇率", "金价", "油价", "房价", "票价", "价格", "多少钱",
    "比分", "赛果", "赛程", "热搜", "新闻", "头条", "航班", "路况",
    "weather", "price", "stock", "score", "news",
)

# 答案里出现这些，说明它在下「天气结论」而不是在说别的
WEATHER_STATE_WORDS = (
    "晴", "阴", "多云", "雨", "雪", "雾", "霾", "台风", "降温", "升温",
)

_DIGIT_RE = re.compile(r"\d")

# 拦几次之后放行：拦太多会把 max_steps 耗光，得不偿失
MAX_REJECTIONS = 2

# max_steps 用尽时的兜底答案退回线上时加的前缀：
# 那条路径不经过 final_answer_checks，只能把「没查证」写在答案里
UNVERIFIED_PREFIX = "⚠️ 未核实："


def _task_text(memory: Any) -> str:
    """从 AgentMemory 里取回本轮任务原文。

    memory.steps 形状：[SystemPromptStep, TaskStep(task=...), ActionStep, ...]
    TaskStep 是本轮任务最可靠的来源——ActionStep 里存的是**被注入过记忆的**
    任务（带历史块和偏好目录），拿它判关键词会被注入内容干扰。
    """
    for step in getattr(memory, "steps", []) or []:
        task = getattr(step, "task", None)
        if isinstance(task, str) and task.strip():
            return task
    return ""


def matched_topic(task: str) -> str | None:
    """任务命中的时效主题词；返回 None 表示这不是「问此刻事实」的任务。"""
    text = str(task or "").lower()
    for topic in CURRENT_FACT_TOPICS:
        if topic in text:
            return topic
    return None


def asserts_concrete_value(answer: Any) -> bool:
    """答案是否给出了具体数值/天气结论。

    用来避免误拦「我查不到实时天气」这类诚实的兜底回答——
    它没有编造数据，拦它只会白白多花一步。
    """
    text = str(answer or "")
    if not text.strip():
        return False
    if _DIGIT_RE.search(text):
        return True
    return any(word in text for word in WEATHER_STATE_WORDS)


class ToolCallLedger:
    """一次任务内「真正被调用过哪些工具」的账本。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(
        self, name: str, args: tuple, kwargs: dict, output: Any = None
    ) -> None:
        # output 是工具真正回传给模型的文本：答案里的数字要在这里找依据。
        # 只留前 OUTPUT_KEEP_CHARS 字符——够覆盖搜索结果，又不会把内存吃满。
        self.calls.append(
            {
                "name": name,
                "args": tuple(args),
                "kwargs": dict(kwargs),
                "output": str(output or "")[:OUTPUT_KEEP_CHARS],
            }
        )

    def outputs(self) -> list[str]:
        return [call["output"] for call in self.calls if call["output"]]

    def reset(self) -> None:
        self.calls.clear()

    def names(self) -> list[str]:
        return [call["name"] for call in self.calls]

    def has_any(self, names: Iterable[str]) -> bool:
        wanted = set(names)
        return any(call["name"] in wanted for call in self.calls)


def record_web_tool_calls(tools: Iterable[Any], ledger: ToolCallLedger) -> list[str]:
    """给联网工具挂一层记录，返回实际挂上的工具名。

    只替换实例上的 ``forward``（``Tool.__call__`` 就是转发到它），
    不动工具类的定义，也不改变返回值和异常。
    """
    attached: list[str] = []
    for tool in tools or []:
        name = getattr(tool, "name", None)
        if name not in WEB_TOOL_NAMES:
            continue
        if getattr(tool, "_answer_checks_recorded", False):
            continue
        original_forward = getattr(tool, "forward", None)
        if not callable(original_forward):    # 非 Tool（测试替身 / 已包装过的函数）
            continue

        def forward(*args, __name=name, __original=original_forward, **kwargs):
            output = __original(*args, **kwargs)
            ledger.record(__name, args, kwargs, output=output)
            return output

        forward.__name__ = getattr(original_forward, "__name__", "forward")
        tool.forward = forward
        tool._answer_checks_recorded = True
        attached.append(name)
    return attached


def _normalize_text(text: Any) -> str:
    """全半角统一、去空白，只为了「数字能不能对上」这种字面比较。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(value.split())


def key_numbers(text: Any) -> list[str]:
    """从文本里抽出「值得核对」的数字（≥2 位或带小数）。

    单位、序号、单个数字都排除在外：答案里的「1」几乎总是「第一步」，
    拿它去核对依据只会制造假阳性。
    """
    numbers = []
    for token in _KEY_NUMBER_RE.findall(_normalize_text(text)):
        digits = token.replace(".", "")
        if len(digits) < _MIN_KEY_NUMBER_DIGITS:
            continue
        numbers.append(token)
    return numbers


def ungrounded_numbers(answer: Any, tool_outputs: Iterable[str]) -> list[str]:
    """答案里出现、但在本轮工具输出里找不到的数字。"""
    haystack = " ".join(_normalize_text(output) for output in tool_outputs)
    grounded = set(key_numbers(haystack))
    return [number for number in key_numbers(answer) if number not in grounded]


def usable_steps(memory: Any) -> int:
    """本轮「成功跑通」的步数：没有 error 且有 Observation。

    小模型常见失败模式：连续几步都是 NameError / KeyError / 代码解析失败，
    然后基于一个不存在的变量直接给出结论。这类结论没有任何依据。
    """
    count = 0
    for step in getattr(memory, "steps", []) or []:
        if getattr(step, "step_number", None) is None:   # 跳过 system prompt / task
            continue
        if getattr(step, "error", None):
            continue
        if getattr(step, "observations", None):
            count += 1
    return count


class FreshnessCheck:
    """final_answer_checks 用的校验器。

    ``check_function.__name__`` 会被写进 AgentError 的消息里，
    所以实例上也放一个 ``__name__``（用实例而非闭包，方便测试重置计数）。
    """

    __name__ = "check_freshness"

    def __init__(
        self,
        ledger: ToolCallLedger,
        max_rejections: int = MAX_REJECTIONS,
        evidence_tools: Iterable[str] = WEB_TOOL_NAMES,
    ) -> None:
        self.ledger = ledger
        self.max_rejections = max_rejections
        self.evidence_tools = frozenset(evidence_tools)
        self.rejected = 0

    def reset(self) -> None:
        self.rejected = 0

    def violation_topic(self, final_answer: Any, memory: Any) -> str | None:
        """返回「这次答案违反了哪条」的时效主题词；None 表示合规。

        三个条件同时满足才算违规：① 任务在问此刻的事实；
        ② 答案给出了具体数值/天气结论；③ 本轮没调过取证类工具。
        """
        topic = matched_topic(_task_text(memory))
        if topic is None:
            return None
        if not asserts_concrete_value(final_answer):
            return None
        if self.ledger.has_any(self.evidence_tools):
            return None
        return topic

    def guard_fallback(self, answer: Any, memory: Any) -> Any:
        """兜底答案（max_steps 用尽时）的守卫。

        ``provide_final_answer`` 不经过 ``final_answer_checks``，
        所以兜底路径上编造的数据完全不受校验——小模型恰恰最容易走到这里。
        兜底时已经没有下一步可以自纠了，只能把风险写在答案里。
        """
        if not isinstance(answer, str):
            return answer

        topic = self.violation_topic(answer, memory)
        if topic is not None:
            return f"{UNVERIFIED_PREFIX}（「{topic}」相关数据本轮没有查证）\n{answer}"

        # 查过了也要核对：实测 llama3.2 搜了 7 次，兜底答案仍写「22℃」，
        # 而这个数字在任何一个 Observation 里都没有出现过。
        problem = self.grounding_problem(answer, memory) or self.failed_steps_problem(
            answer, memory
        )
        if problem is not None:
            return f"{UNVERIFIED_PREFIX}（{problem.splitlines()[0][:60]}）\n{answer}"
        return answer

    def grounding_problem(self, final_answer: Any, memory: Any) -> str | None:
        """答案里的关键数字有没有依据：必须出现在本轮工具输出里。

        只管「本轮确实查过」的情况——没查过的情况由 ``violation_topic``
        先一步拦掉；而没查也允许的数字（算出来的结果）不该被这条误伤。
        所以这条只对「问此刻事实」的任务生效。
        """
        if matched_topic(_task_text(memory)) is None:
            return None
        if not self.ledger.has_any(self.evidence_tools):
            return None

        missing = ungrounded_numbers(final_answer, self.ledger.outputs())
        if not missing:
            return None
        return (
            "答案里的这些数字本轮工具结果里找不到："
            + "、".join(missing[:5])
            + "。请不要自己推算或沿用历史数据，直接用工具返回的原始数值回答；"
            "查不到就说明查不到。"
        )

    def failed_steps_problem(self, final_answer: Any, memory: Any) -> str | None:
        """本轮一步都没跑通，却给出了具体结论 → 拦。

        典型现场（llama3.2 实测）：连续几步都是 KeyError / NameError /
        代码解析失败，模型最后一句话给出「阴天，温度在22℃左右」。
        """
        if not asserts_concrete_value(final_answer):
            return None
        if usable_steps(memory) > 0:
            return None
        return (
            "本轮还没有任何一步成功执行（每一步都报错或没有输出），"
            "你手上的变量可能根本不存在。"
            "请先写出能跑通的一步并把结果 print 出来，再基于真实结果作答；"
            "实在做不到就说明失败原因，不要给出具体数值。"
        )

    def __call__(self, final_answer: Any, memory: Any) -> bool:
        topic = self.violation_topic(final_answer, memory)
        problem = None
        if topic is not None:
            problem = topic
        else:
            problem = self.grounding_problem(final_answer, memory)
            if problem is None:
                problem = self.failed_steps_problem(final_answer, memory)
        if problem is None:
            return True

        self.rejected += 1
        if self.rejected > self.max_rejections:
            # 已经纠正过 max_rejections 次仍不改：放行。
            # 继续拦只会把 max_steps 耗光，最终得到的兜底答案更差。
            return True

        if topic is None:
            # grounding / 失败步 这两条守卫自己带了完整的补救说明
            assert False, problem

        tools = " / ".join(sorted(self.evidence_tools))
        assert False, (
            f"答案里给出了「{topic}」的具体数据，但本轮没有调用过 {tools}，"
            f"无法证明这些数据是这次查到的（很可能是从历史对话或记忆里抄来的过期数据）。"
            f"请先调用工具查证，例如 web_search(\"{_example_query(topic)}\")，"
            f"再用查询结果里的数字回答；如果确实查不到，就直接说明查不到，"
            f"不要给出具体数值。"
        )


def _example_query(topic: str) -> str:
    if topic in {"天气", "气温", "温度", "weather"}:
        return "广州 今天 天气"
    if topic in {"股价", "股票", "stock"}:
        return "贵州茅台 股价"
    if topic in {"比分", "score"}:
        return "昨天 英超 比分"
    return f"{topic} 最新"


def attach_answer_checks(agent: Any, ledger: ToolCallLedger | None = None) -> FreshnessCheck:
    """给 CodeAgent 挂上 final_answer 校验 + 工具记账 + 每轮重置。

    Args:
        agent: 已构造好的 smolagents agent（工具已经装配在 ``agent.tools``）。
        ledger: 复用已有账本（测试用）；默认新建。

    Returns:
        挂上的 ``FreshnessCheck``，测试里可以直接调用它的 ``reset()``。
    """
    existing = getattr(agent, "_answer_checks_attached", False)
    if existing:
        # 重复挂载会让「每轮重置」指向旧账本，直接返回已挂的那个
        return agent.final_answer_checks[0]

    ledger = ledger if ledger is not None else ToolCallLedger()
    check = FreshnessCheck(ledger)

    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        tools = list(tools.values())
    record_web_tool_calls(tools or [], ledger)

    agent.final_answer_checks = [check]

    # smolagents 里一个 agent 实例会被反复 run（Gradio 后台常驻），
    # 所以每轮开始前必须清账，否则上一轮的 web_search 会为这一轮背书。
    original_run = agent.run

    def run(*args, **kwargs):
        ledger.reset()
        check.reset()
        return original_run(*args, **kwargs)

    agent.run = run

    # 兜底路径（max_steps 用尽 → provide_final_answer）不走 final_answer_checks，
    # 单独包一层，把「没查证」标在答案前面，而不是让它冒充查证过的结果。
    original_provide_final_answer = getattr(agent, "provide_final_answer", None)
    if callable(original_provide_final_answer):

        def provide_final_answer(*args, **kwargs):
            answer = original_provide_final_answer(*args, **kwargs)
            return check.guard_fallback(answer, agent.memory)

        agent.provide_final_answer = provide_final_answer
    agent._answer_checks_attached = True

    return check


__all__ = [
    "FreshnessCheck",
    "MAX_REJECTIONS",
    "ToolCallLedger",
    "UNVERIFIED_PREFIX",
    "WEB_TOOL_NAMES",
    "attach_answer_checks",
    "asserts_concrete_value",
    "matched_topic",
    "record_web_tool_calls",
]
