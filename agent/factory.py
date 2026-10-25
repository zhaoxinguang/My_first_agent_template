"""Agent 工厂：模型 + 工具 + prompt 模板 → CodeAgent。"""

import yaml
from typing import Any, cast

from smolagents import CodeAgent

from agent.final_answer_tool import (
    FinalAnswerRewritingExecutor,
    LenientFinalAnswerTool,
)
from agent.guardrails import attach_guards
from agent.model import create_model
from config.settings import MAX_STEPS, PROMPT_PATH, VERBOSITY_LEVEL
from tools import build_tools

# smolagents 不会把传入的 prompt_templates 与内置默认模板合并：
# 传了就用传入的整份字典，缺哪段就在用到时报 KeyError。
# 这里在启动阶段就校验，避免跑到 max_steps 才崩（提供兜底答案时才读 final_answer）。
REQUIRED_TEMPLATE_KEYS = ("system_prompt", "planning", "managed_agent", "final_answer")


def _load_prompt_templates() -> dict:
    with PROMPT_PATH.open("r", encoding="utf-8") as stream:
        prompt_templates = yaml.safe_load(stream)

    missing = [key for key in REQUIRED_TEMPLATE_KEYS if key not in prompt_templates]
    if missing:
        raise ValueError(
            f"{PROMPT_PATH} 缺少 prompt 模板段落：{missing}。"
            f"smolagents 不会与内置默认模板合并，必须完整提供 "
            f"{list(REQUIRED_TEMPLATE_KEYS)}。"
        )

    return prompt_templates


def create_agent(
    include_web: bool = True,
    store=None,
    embedding_client=None,
) -> CodeAgent:
    """创建底层 CodeAgent。

    Args:
        include_web: 是否挂载联网工具，离线调试时可传 False。
        store: 记忆库。偏好工具的读写都走它；与 MemoryAgent 用同一个
            store，才能让「模型刚存的偏好」下一轮立刻出现在目录里。
        embedding_client: 偏好向量化客户端；None 时偏好读侧退化为结构化读取。
    """
    prompt_templates = _load_prompt_templates()

    agent = CodeAgent(
        model=create_model(),
        tools=build_tools(
            include_web=include_web,
            store=store,
            embedding_client=embedding_client,
        ),
        max_steps=MAX_STEPS,
        verbosity_level=VERBOSITY_LEVEL,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=cast(Any, prompt_templates),
    )
    # final_answer 换成宽容版：模型把偏好当成它的参数时（
    # final_answer(x, answer_format="分条")）不再抛 TypeError 白跑一步。
    # 必须在 run 之前换——send_tools 只在 run() 开头同步一次。
    agent.tools["final_answer"] = LenientFinalAnswerTool()
    # 光换工具不够：evaluate_python_code 执行前会把 final_answer 换成
    # 只收一个位置参数的闭包，所以还要在代码进 executor 之前删掉多余参数。
    agent.python_executor = FinalAnswerRewritingExecutor(agent.python_executor)

    # 护栏必须在工具装配之后：它要给联网工具挂「调用记账」，
    # 用来判断模型报的时效性数据到底是不是本轮查来的；
    # 同时挂上旧 Observation 折叠（上下文预算）和每轮工具缓存重置。
    attach_guards(agent)
    return agent


__all__ = ["create_agent"]
