"""工具注册表：清单、顺序、TOOL_GUIDE 与实现保持一致。"""

import pytest

from tools import TOOL_GUIDE, build_tools, describe_tools, get_tool


def _names(tools):
    return [getattr(tool, "name", getattr(tool, "__name__", None)) for tool in tools]


def test_offline_toolset_exact():
    """离线模式不依赖任何网络库，工具集合应固定。"""
    assert _names(build_tools(include_web=False)) == [
        "get_current_time_in_timezone",
        "read_text_file",
        "list_files",
        "load_user_preferences",
        "save_user_preference",
        "forget_user_preference",
        "final_answer",
    ]


def test_final_answer_is_last():
    """final_answer 必须是最后一步可用的工具，顺序错了模型容易提前结束。"""
    assert _names(build_tools(include_web=True))[-1] == "final_answer"


def test_guide_covers_every_registered_tool():
    """注册了却没写说明，模型就不知道该怎么用。"""
    for name in _names(build_tools(include_web=True)):
        assert name in TOOL_GUIDE, f"缺少 {name} 的 TOOL_GUIDE 条目"


def test_guide_entries_are_complete():
    for name, info in TOOL_GUIDE.items():
        assert info["purpose"].strip(), f"{name} 缺 purpose"
        assert info["usage"].strip(), f"{name} 缺 usage"
        assert info["example"].strip(), f"{name} 缺 example"


def test_web_tools_registered_when_available(monkeypatch):
    # 联网工具现在只依赖 requests / markdownify（搜索走 DeepSeek Search API），
    # 不再依赖 duckduckgo_search；用错守卫会让这条用例被无声跳过。
    pytest.importorskip("requests")
    pytest.importorskip("markdownify")
    # 缺 DEEPSEEK_API_KEY 时 web_search 会有意缺席，这里补一个假密钥保证可构造
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    names = _names(build_tools(include_web=True))

    assert "web_search" in names
    assert "visit_webpage" in names
    assert "extract" in names


def test_web_tools_absent_when_disabled():
    names = _names(build_tools(include_web=False))

    assert "web_search" not in names
    assert "visit_webpage" not in names


def test_build_tools_tolerates_missing_web_dependencies(monkeypatch, capsys):
    """没装 requests / markdownify 时联网工具整组缺席，本地工具照常返回。

    回归点：这些依赖以前是在 ``import tools`` 时就硬引入的——缺一个包，
    整个包都 import 不进来，``include_web=False`` 等于空话。
    """
    import tools as tools_pkg

    for attr in ("WebSearchTool", "ExtractTool", "VisitWebpageTool", "TaskScopedCache"):
        monkeypatch.setattr(tools_pkg, attr, None)
    monkeypatch.setattr(
        tools_pkg, "_WEB_IMPORT_ERROR", ImportError("No module named 'requests'")
    )

    names = _names(tools_pkg.build_tools(include_web=True))

    assert names == [
        "get_current_time_in_timezone",
        "read_text_file",
        "list_files",
        "load_user_preferences",
        "save_user_preference",
        "forget_user_preference",
        "final_answer",
    ]
    assert "未加载联网工具" in capsys.readouterr().out


def test_get_tool_forwards_user_id():
    """get_tool 必须和 build_tools 一样能指定用户，否则多用户只能绕道取下标。"""
    tool = get_tool("save_user_preference", include_web=False, user_id="someone")

    assert tool.user_id == "someone"


def test_get_tool_by_name():
    tool = get_tool("final_answer", include_web=False)

    assert tool is not None
    assert getattr(tool, "name", None) == "final_answer"


def test_get_tool_unknown_returns_none():
    assert get_tool("no_such_tool", include_web=False) is None


def test_describe_tools_mentions_every_entry():
    text = describe_tools()

    for name in TOOL_GUIDE:
        assert name in text


def test_get_current_time_in_timezone_handles_invalid_zone():
    tool = get_tool("get_current_time_in_timezone", include_web=False)

    assert "Unknown timezone" in tool("Mars/Olympus")


def test_get_current_time_in_timezone_returns_time():
    tool = get_tool("get_current_time_in_timezone", include_web=False)

    assert "Asia/Shanghai" in tool("Asia/Shanghai")


def test_visit_webpage_reports_bad_url():
    """坏 URL 应返回错误字符串，而不是抛异常打断 agent。"""
    pytest.importorskip("requests")
    pytest.importorskip("markdownify")
    tool = get_tool("visit_webpage", include_web=True)

    assert "Error" in tool("http://127.0.0.1:65535/nope")


def test_web_search_failure_returns_error_string(monkeypatch):
    """搜索后端限流/异常时必须返回字符串，不能抛异常。

    抛异常会变成一步工具执行失败，模型容易原地重试到耗尽 max_steps，
    最后既没答案也没 final_answer。

    现在 web_search 走 DeepSeek Search API，直接在 requests.post 这一层注入异常。
    """
    pytest.importorskip("smolagents")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    tool = get_tool("web_search", include_web=True)

    def boom(*args, **kwargs):
        raise RuntimeError("backend down")

    monkeypatch.setattr("tools.web_tools.requests.post", boom)

    out = tool("anything")

    assert out.startswith("Error: web search failed")
    assert "visit_webpage" in out

