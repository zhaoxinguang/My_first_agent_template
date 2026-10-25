"""偏好工具：动态更新（save/forget）与按需加载（load）。

这些工具是模型唯一能写长期偏好的入口，所以重点验证三件事：
1. 读写真的落到同一个 store（写完之后立刻能读出来）；
2. 失败一律返回 ``Error:`` 字符串，不抛异常（否则小模型会原地重试）；
3. 错误信息可执行（预算满时告诉它删哪条）。
"""

import pytest

from tools import TOOL_GUIDE, build_tools, get_tool
from tools import memory_tools


@pytest.fixture
def toolset(store):
    """三个偏好工具，共用同一个临时 store。"""
    return {
        name: get_tool(name, include_web=False, store=store)
        for name in (
            "load_user_preferences",
            "save_user_preference",
            "forget_user_preference",
        )
    }


def test_tools_are_registered_with_guides():
    names = [getattr(tool, "name", None) for tool in build_tools(include_web=False)]

    for name in (
        "load_user_preferences",
        "save_user_preference",
        "forget_user_preference",
    ):
        assert name in names
        assert name in TOOL_GUIDE


def test_building_tools_does_not_touch_the_database(monkeypatch):
    """构造工具不该建库：测试/离线场景不应该有副作用。"""
    monkeypatch.setattr(memory_tools, "_DEFAULT_STORE", None)

    def boom(*args, **kwargs):
        raise AssertionError("build_tools 不应构造 MemoryStore")

    monkeypatch.setattr("memory.store.MemoryStore.__init__", boom)

    assert build_tools(include_web=False)


def test_save_then_load_roundtrip(toolset):
    saved = toolset["save_user_preference"]("style", "回答简洁，先给可运行代码")

    assert "已保存偏好 style" in saved

    loaded = toolset["load_user_preferences"]("")
    assert "回答简洁，先给可运行代码" in loaded
    assert "style" in loaded


def test_save_same_key_updates_in_place(toolset, store):
    toolset["save_user_preference"]("style", "回答简洁")
    toolset["save_user_preference"]("style", "回答详细")

    assert store.count_active_preferences("default_user") == 1
    assert "回答详细" in toolset["load_user_preferences"]("style")
    assert "回答简洁" not in toolset["load_user_preferences"]("style")


def test_load_by_key_name_does_not_need_semantic_search(toolset):
    """点名一个同义 key 也能命中（写侧会把同义 key 收敛到规范维度）。"""
    toolset["save_user_preference"]("preferred_detail_level", "简洁")

    loaded = toolset["load_user_preferences"]("detail")

    assert "answer_length" in loaded  # 规范 key，而不是写进去的那个别名
    assert "简短" in loaded  # 值也归一了（简洁 → 简短）


def test_load_unknown_query_degrades_to_all_without_embedding(toolset):
    """没有向量服务时「按语义找」退化为「列全部」——好过整轮失忆。"""
    toolset["save_user_preference"]("style", "回答简洁")

    loaded = toolset["load_user_preferences"]("完全无关的查询词")

    assert "style" in loaded


def test_load_missing_key_on_empty_store(toolset):
    loaded = toolset["load_user_preferences"]("style")

    assert "没有与" in loaded
    assert "load_user_preferences" in loaded  # 告诉它怎么列全部


def test_load_without_any_preference(toolset):
    assert "没有" in toolset["load_user_preferences"]("")


def test_save_rejects_blank_inputs(toolset):
    assert toolset["save_user_preference"]("", "v").startswith("Error:")
    assert toolset["save_user_preference"]("k", "  ").startswith("Error:")


def test_save_reports_skipped_write_as_error(toolset, store, monkeypatch):
    """``upsert`` 没写进库时必须如实报错，而不是「已保存」。

    早期版本这种情况返回的是**旧条目的 memory_id**，工具层看到非空 id
    就回「已保存」，模型以为写成功了，下一轮在目录里找不到又写一遍。
    """
    save = toolset["save_user_preference"]

    def _skip(*args, **kwargs):
        raise memory_tools.PreferenceSkipped(
            "现有条目置信度更高（0.90 > 0.10），保留原偏好。"
        )

    # _manager() 每次都新建一个实例，要打在类上才拦得住
    monkeypatch.setattr(
        memory_tools.PreferenceManager, "upsert", lambda self, **kw: _skip()
    )

    out = save("language", "english")

    assert out.startswith("Error:")
    assert "未写入" in out
    assert "置信度" in out
    assert store.count_active_preferences("default_user") == 0


def test_save_reports_budget_error_as_string(store):
    tool = get_tool("save_user_preference", include_web=False, store=store)
    tool._manager = lambda: memory_tools.PreferenceManager(
        store=store, embedding_client=None, char_limit=14
    )

    assert "已保存" in tool("k1", "0123456789")  # 2 + 10 = 12 字符
    failed = tool("k2", "0123456789")

    assert failed.startswith("Error:")
    assert "forget_user_preference" in failed   # 告诉模型怎么腾空间
    assert "k1" in failed                       # 现有哪些条目


def test_forget_removes_and_reports_budget(toolset, store):
    toolset["save_user_preference"]("style", "回答简洁")

    removed = toolset["forget_user_preference"]("style")

    assert "已删除偏好 style" in removed
    assert store.count_active_preferences("default_user") == 0


def test_forget_unknown_key_lists_existing(toolset):
    toolset["save_user_preference"]("style", "回答简洁")

    out = toolset["forget_user_preference"]("no_such_key")

    assert out.startswith("Error:")
    assert "style" in out  # 列出已有的 key，模型据此纠正


def test_forget_unknown_key_on_empty_store(toolset):
    out = toolset["forget_user_preference"]("no_such_key")

    assert out.startswith("Error:")
    assert "没有任何偏好" in out


def test_forget_rejects_blank_key(toolset):
    assert toolset["forget_user_preference"]("").startswith("Error:")


def test_load_failure_returns_error_string(store, monkeypatch):
    tool = get_tool("load_user_preferences", include_web=False, store=store)

    def boom(self, **kwargs):
        raise RuntimeError("database is locked")

    # 工具每次调用都会新建 retriever，所以必须补在类上而不是实例上
    monkeypatch.setattr(
        "memory.retriever.MemoryRetriever.load_preferences", boom
    )

    out = tool("style")

    assert out.startswith("Error:")
    assert "读取偏好失败" in out


def test_tools_share_the_injected_store(store):
    """写进去的偏好必须能被 MemoryAgent 注入层读到（同一个 store）。"""
    from memory.context import build_augmented_task
    from memory.retriever import MemoryRetriever

    get_tool("save_user_preference", include_web=False, store=store)(
        "style", "回答简洁"
    )

    task = build_augmented_task(
        "写个快排",
        MemoryRetriever(store=store, embedding_client=None),
        user_id="default_user",
    )

    assert "style" in task
    assert task.rstrip().endswith("写个快排")
