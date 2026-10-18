"""MemoryAgent：注入偏好、透明转发；以及 agent.factory 的组装。"""

import pytest

from memory.agent import MemoryAgent
from memory.retriever import MemoryRetriever


class _FakeAgent:
    """替代 CodeAgent，只记录被喂进来的任务。"""

    def __init__(self):
        self.tasks = []
        self.kwargs = None
        self.model = "fake-model"

    def run(self, task, *args, **kwargs):
        self.tasks.append(task)
        self.kwargs = kwargs
        return "fake-answer"


@pytest.fixture
def wired(store):
    store.save_preference("u", "format", "language", "中文")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    agent = _FakeAgent()
    return MemoryAgent(agent=agent, retriever=retriever, user_id="u"), agent


def test_run_injects_preferences_into_task(wired):
    memory_agent, inner = wired

    result = memory_agent.run("写个快排")

    assert result == "fake-answer"
    injected = inner.tasks[0]
    assert "中文" in injected
    assert injected.rstrip().endswith("写个快排")


def test_run_still_works_without_preferences(store):
    retriever = MemoryRetriever(store=store, embedding_client=None)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(agent=inner, retriever=retriever, user_id="u")

    memory_agent.run("随便问问")

    assert inner.tasks[0] == "随便问问"


def test_run_forwards_stream_and_kwargs(wired):
    memory_agent, inner = wired

    memory_agent.run("任务", stream=True, reset=True)

    assert inner.kwargs == {"stream": True, "reset": True}


def test_prepare_task_does_not_execute(wired):
    memory_agent, inner = wired

    prepared = memory_agent.prepare_task("写个快排")

    assert "中文" in prepared
    assert inner.tasks == []  # 只拼装，不执行


def test_model_property_is_forwarded(wired):
    memory_agent, inner = wired

    assert memory_agent.model == "fake-model"


def test_unknown_attribute_is_forwarded(wired):
    memory_agent, inner = wired
    inner.tools = {"final_answer": object()}

    assert "final_answer" in memory_agent.tools


def test_scope_and_limit_are_used(store):
    store.save_preference("u", "cat", "k1", "v1", scope="proj")
    store.save_preference("u", "cat", "k2", "v2", scope="global")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(
        agent=inner, retriever=retriever, user_id="u", scope="proj", limit=1
    )

    memory_agent.run("任务")

    # scope 精确匹配 + limit 1，只应注入 proj 那条
    assert "v1" in inner.tasks[0]
    assert "v2" not in inner.tasks[0]


# ---------- agent.factory ----------

def test_create_agent_assembles_code_agent():
    """真实组装一次 CodeAgent：验证模型 + 工具 + prompt 模板都能接上。

    只构造不执行，不发网络请求，所以不需要真的能连上模型。
    """
    smolagents = pytest.importorskip("smolagents")
    from agent.factory import create_agent
    from config.settings import MAX_STEPS

    agent = create_agent(include_web=False)

    assert isinstance(agent, smolagents.CodeAgent)
    tool_names = set(agent.tools)
    assert "final_answer" in tool_names
    assert "read_text_file" in tool_names
    assert agent.max_steps == MAX_STEPS


def test_create_agent_loads_custom_prompt():
    pytest.importorskip("smolagents")
    from agent.factory import create_agent

    agent = create_agent(include_web=False)

    # prompts.yaml 里的 system_prompt 应替换掉 smolagents 默认模板
    assert "expert assistant" in str(
        agent.prompt_templates.get("system_prompt", "")
    )
