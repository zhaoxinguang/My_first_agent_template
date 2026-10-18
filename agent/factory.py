"""Agent 工厂：模型 + 工具 + prompt 模板 → CodeAgent。"""

import yaml

from smolagents import CodeAgent

from agent.model import create_model
from config.settings import MAX_STEPS, PROMPT_PATH, VERBOSITY_LEVEL
from tools import build_tools


def create_agent(include_web: bool = True) -> CodeAgent:
    """创建底层 CodeAgent。

    Args:
        include_web: 是否挂载联网工具，离线调试时可传 False。
    """
    with PROMPT_PATH.open("r", encoding="utf-8") as stream:
        prompt_templates = yaml.safe_load(stream)

    return CodeAgent(
        model=create_model(),
        tools=build_tools(include_web=include_web),
        max_steps=MAX_STEPS,
        verbosity_level=VERBOSITY_LEVEL,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates,
    )


__all__ = ["create_agent"]
