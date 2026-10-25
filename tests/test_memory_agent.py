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


# ---------- 历史对话注入 ----------

def _index_run(store, client, task, answer):
    from memory.history import HistoryIndex

    run_id = store.start_run(task)
    store.finish_run(run_id, final_answer=answer, status="success")
    HistoryIndex(store=store, embedding_client=client).index_run(
        run_id, task, answer
    )


def test_run_injects_retrieved_history(store):
    from tests.helpers import FakeSemanticEmbeddingClient

    client = FakeSemanticEmbeddingClient()
    _index_run(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(agent=inner, retriever=retriever, user_id="u")

    memory_agent.run("帮我计算 123 乘 456")

    injected = inner.tasks[0]
    assert "123乘456等于56088" in injected
    assert injected.rstrip().endswith("帮我计算 123 乘 456")


def test_history_limit_zero_disables_history(store):
    from tests.helpers import FakeSemanticEmbeddingClient

    client = FakeSemanticEmbeddingClient()
    _index_run(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(
        agent=inner,
        retriever=retriever,
        user_id="u",
        history_limit=0,
    )

    memory_agent.run("帮我计算 123 乘 456")

    assert inner.tasks[0] == "帮我计算 123 乘 456"


def test_scope_includes_global_and_skips_other_scopes(store):
    """作用域口径与向量检索一致：当前作用域 + global，其它作用域不注入。"""
    store.save_preference("u", "cat", "k1", "v1", scope="proj")
    store.save_preference("u", "cat", "k2", "v2", scope="global")
    store.save_preference("u", "cat", "k3", "v3", scope="other")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(
        agent=inner, retriever=retriever, user_id="u", scope="proj", limit=5
    )

    memory_agent.run("任务")

    injected = inner.tasks[0]
    assert "v1" in injected
    assert "v2" in injected
    assert "v3" not in injected


def test_preference_limit_is_respected(store):
    """limit=1 时目录只列一条，不能因为条数少就放开。"""
    store.save_preference("u", "cat", "k1", "v1")
    store.save_preference("u", "cat", "k2", "v2")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    inner = _FakeAgent()
    memory_agent = MemoryAgent(
        agent=inner, retriever=retriever, user_id="u", limit=1
    )

    memory_agent.run("任务")

    catalog = [
        line for line in inner.tasks[0].splitlines()
        if line.startswith("- [")
    ]
    assert len(catalog) == 1


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


def test_create_agent_installs_lenient_final_answer():
    """装配出来的 final_answer 必须容忍模型多传的偏好参数。

    真实现场：final_answer(result, answer_format="分条") 抛 TypeError，
    整步白跑；前面已经拿到的搜索结果一步都没交出去。
    """
    pytest.importorskip("smolagents")
    from agent.factory import create_agent
    from agent.final_answer_tool import FinalAnswerRewritingExecutor, LenientFinalAnswerTool

    agent = create_agent(include_web=False)

    assert isinstance(agent.tools["final_answer"], LenientFinalAnswerTool)
    assert isinstance(agent.python_executor, FinalAnswerRewritingExecutor)


def test_create_agent_loads_custom_prompt():
    pytest.importorskip("smolagents")
    from agent.factory import create_agent

    agent = create_agent(include_web=False)

    # prompts.yaml 里的 system_prompt 应替换掉 smolagents 默认模板
    assert "expert assistant" in str(
        agent.prompt_templates.get("system_prompt", "")
    )


def test_prompt_templates_are_complete():
    """prompts.yaml 必须自带全部四段模板。

    smolagents 不会把传入的 prompt_templates 与内置默认模板合并，
    缺 final_answer 段时，一旦跑满 max_steps 需要兜底作答就 KeyError。
    """
    pytest.importorskip("smolagents")
    from agent.factory import REQUIRED_TEMPLATE_KEYS, _load_prompt_templates

    templates = _load_prompt_templates()

    assert set(REQUIRED_TEMPLATE_KEYS) <= set(templates)
    assert templates["final_answer"]["pre_messages"].strip()
    assert "{{task}}" in templates["final_answer"]["post_messages"]

