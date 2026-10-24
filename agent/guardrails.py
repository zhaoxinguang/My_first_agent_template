"""运行时护栏：把「省时间」的三件事装配到一个 agent 上。

三件事互相依赖，所以放在一个入口里挂载：

1. **答案校验**（`agent/answer_checks.py`）：时效性数据必须本轮查证过，
   答案里的关键数字必须能在本轮工具结果里找到依据；
2. **上下文预算**（本文件 `ObservationTrimmer`）：只保留最近 N 步的完整
   Observation，更早的折叠成开头一小段。本机每步都要把全部历史重新 prefill
   一遍，少留一步原文就是实打实省几秒；
3. **每轮重置**：工具缓存（`TaskScopedCache`）和调用账本都属于「一次任务」，
   agent 实例会被反复 run，不清就会串轮。

为什么不并进 `answer_checks.py`：那个模块只回答「这个答案能不能交」，
而这里的第 2、3 件事跟答案对错无关，是纯执行层预算。分开以后两者都能单独测试。
"""

from __future__ import annotations

from typing import Any, Iterable

from agent.answer_checks import attach_answer_checks
from config.settings import (
    AGENT_KEEP_RECENT_OBSERVATIONS,
    AGENT_OLD_OBSERVATION_CHARS,
)

# Observation 被折叠时追加的说明。写清楚「这是被省略的历史输出」，
# 模型才不会把半句话当成完整结论，也不会以为自己看错了。
FOLDED_MARKER = " …（该步输出已折叠，共省略 {omitted} 字）"

_TOOL_RESPONSE_ROLES = ("tool-response", "tool_response", "tool")


def _role_of(message: Any) -> str:
    role = message.get("role") if isinstance(message, dict) else None
    return str(getattr(role, "value", role) or "").lower()


def _text_parts(message: Any) -> list[dict]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict) and "text" in part]
    return []


class ObservationTrimmer:
    """把较早步骤的 Observation 截短，只留最近 N 步完整。

    只动「发给模型的副本」，不改 `agent.memory.steps`——
    原始记录仍然完整，日志和事后审计不受影响。
    """

    def __init__(
        self,
        keep_recent: int = AGENT_KEEP_RECENT_OBSERVATIONS,
        old_chars: int = AGENT_OLD_OBSERVATION_CHARS,
    ) -> None:
        self.keep_recent = max(0, int(keep_recent))
        self.old_chars = max(80, int(old_chars))
        # 便于测试与排障：折叠发生了几次、一共省了多少字符
        self.folded_steps = 0
        self.saved_chars = 0

    def reset(self) -> None:
        self.folded_steps = 0
        self.saved_chars = 0

    def __call__(self, messages: Iterable[Any]) -> list[Any]:
        messages = list(messages)
        if self.keep_recent == 0:
            return messages

        indexes = [i for i, m in enumerate(messages) if _role_of(m) in _TOOL_RESPONSE_ROLES]
        if len(indexes) <= self.keep_recent:
            return messages

        for index in indexes[: -self.keep_recent]:
            shrunk = self._shrink(messages[index])
            if shrunk is not None:
                messages[index] = shrunk
                self.folded_steps += 1
        return messages

    def _shrink(self, message: dict) -> dict | None:
        parts = _text_parts(message)
        if not parts:
            return None

        new_parts = []
        changed = False
        for part in parts:
            text = str(part.get("text") or "")
            if len(text) <= self.old_chars:
                new_parts.append(part)
                continue
            omitted = len(text) - self.old_chars
            self.saved_chars += omitted
            new_parts.append(
                {**part, "text": text[: self.old_chars] + FOLDED_MARKER.format(omitted=omitted)}
            )
            changed = True

        if not changed:
            return None
        return {**message, "content": new_parts}


def reset_tool_state(agent: Any) -> list[str]:
    """清空各工具的「本轮缓存」；返回被重置的工具名。

    约定：工具要么自己有 ``begin_task()``，要么把缓存放在 ``tool.cache`` 上
    （`TaskScopedCache.begin_task()`）。两条都支持，避免为了重置去改工具签名。
    """
    reset_names: list[str] = []
    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        tools = list(tools.values())
    for tool in tools or []:
        name = getattr(tool, "name", None)
        reset = getattr(tool, "begin_task", None)
        if not callable(reset):
            cache = getattr(tool, "cache", None)
            reset = getattr(cache, "begin_task", None)
        if callable(reset):
            reset()
            if name:
                reset_names.append(name)
    return reset_names


def attach_guards(agent: Any, ledger: Any = None) -> dict:
    """给 agent 挂上全部护栏。

    Args:
        agent: 已装配好工具的 smolagents agent。
        ledger: 复用调用账本（测试用）。

    Returns:
        ``{"check": FreshnessCheck, "trimmer": ObservationTrimmer}``，
        测试里可以直接断言它们的内部计数。
    """
    check = attach_answer_checks(agent, ledger=ledger)
    trimmer = ObservationTrimmer()

    # 1) 上下文预算：包在 write_memory_to_messages 外面。
    #    这是 smolagents 里唯一把 memory.steps 变成「发给模型的消息」的地方，
    #    包这里既不丢原始记录，也能覆盖 step() 和 provide_final_answer() 两条路径。
    original_write = agent.write_memory_to_messages

    def write_memory_to_messages(*args, **kwargs):
        return trimmer(original_write(*args, **kwargs))

    agent.write_memory_to_messages = write_memory_to_messages

    # 2) 每轮重置工具缓存。
    #    answer_checks 已经包过一层 run（重置账本），这里再包一层，
    #    两层各管各的状态，互不依赖。
    original_run = agent.run

    def run(*args, **kwargs):
        trimmer.reset()
        reset_tool_state(agent)
        return original_run(*args, **kwargs)

    agent.run = run
    return {"check": check, "trimmer": trimmer}


__all__ = [
    "FOLDED_MARKER",
    "ObservationTrimmer",
    "attach_guards",
    "reset_tool_state",
]
