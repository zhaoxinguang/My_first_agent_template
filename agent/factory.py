import yaml

from smolagents import CodeAgent, DuckDuckGoSearchTool

from agent.model import create_model
from config.settings import (
    PROMPT_PATH,
    MAX_STEPS,
    VERBOSITY_LEVEL,
)
from tools.custom_tools import get_current_time_in_timezone
from tools.file_tools import read_text_file, list_files
from tools.final_answer import FinalAnswerTool


def create_agent():
    with PROMPT_PATH.open("r", encoding="utf-8") as stream:
        prompt_templates = yaml.safe_load(stream)

    model = create_model()

    tools = [
        get_current_time_in_timezone,
        FinalAnswerTool(),
        DuckDuckGoSearchTool(),
        read_text_file,
        list_files,
    ]

    return CodeAgent(
        model=model,
        tools=tools,
        max_steps=MAX_STEPS,
        verbosity_level=VERBOSITY_LEVEL,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates,
    )

