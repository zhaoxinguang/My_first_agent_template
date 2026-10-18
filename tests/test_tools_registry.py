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


def test_web_tools_registered_when_available():
    pytest.importorskip("duckduckgo_search")
    names = _names(build_tools(include_web=True))

    assert "web_search" in names
    assert "visit_webpage" in names


def test_web_tools_absent_when_disabled():
    names = _names(build_tools(include_web=False))

    assert "web_search" not in names
    assert "visit_webpage" not in names


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
    pytest.importorskip("duckduckgo_search")
    tool = get_tool("visit_webpage", include_web=True)

    assert "Error" in tool("http://127.0.0.1:65535/nope")
