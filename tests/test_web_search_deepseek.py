"""web_search（DeepSeek Search API）的行为约定。

接口文档：https://api-docs.deepseek.com/zh-cn/api/search
这里全部用假的 ``requests.post``，不打真实网络、不消耗额度。
"""

import pytest

pytest.importorskip("smolagents")

from tools.web_tools import DeepSeekSearchTool, WebSearchTool  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _make_tool(monkeypatch, responses, max_results=5):
    """构造工具，并把 requests.post 换成按顺序吐 responses 的假实现。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_SEARCH_BASE_URL", "https://api.deepseek.com")
    tool = DeepSeekSearchTool(max_results=max_results)

    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return responses.pop(0)

    monkeypatch.setattr("tools.web_tools.requests.post", fake_post)
    return tool, calls


def test_web_search_tool_keeps_public_interface():
    """接口名不能改：工具名 web_search，入参只有 query。"""
    assert getattr(WebSearchTool, "__name__") == "WebSearchTool"
    assert DeepSeekSearchTool.name == "web_search"
    assert list(DeepSeekSearchTool.inputs) == ["query"]
    assert DeepSeekSearchTool.output_type == "string"


def test_request_payload_matches_search_api(monkeypatch):
    """端点、鉴权头、请求体字段都要符合 Search API 规范。"""
    payload = {"results": [{"title": "t", "url": "https://e.com", "summary": "s"}]}
    tool, calls = _make_tool(monkeypatch, [_FakeResponse(200, payload)], max_results=7)

    tool("smolagents 文档")

    call = calls[0]
    assert call["url"] == "https://api.deepseek.com/v1/tools/search"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["headers"]["Content-Type"] == "application/json"
    # query / count / need_content 是 Search API 的字段名，写错会被服务端忽略
    assert call["json"] == {
        "query": "smolagents 文档",
        "count": 7,
        "need_content": False,
    }


def test_query_is_truncated_and_count_is_clamped(monkeypatch):
    """query 上限 128 字符、count 上限 1~50，越界要在本地兜住而不是吃 400。"""
    tool, calls = _make_tool(
        monkeypatch, [_FakeResponse(200, {"results": []})], max_results=999
    )

    tool("长" * 300)

    assert len(calls[0]["json"]["query"]) == 128
    assert calls[0]["json"]["count"] == 50


def test_results_are_formatted_for_the_model(monkeypatch):
    payload = {
        "results": [
            {
                "rerank_score": 2.4,
                "title": "DeepSeek V4",
                "url": "https://example.com/a",
                "summary": "长摘要内容",
                "snippet": "短摘要",
                "site_name": "Example News",
                "date": "2026-06-01T00:00:00+00:00",
            }
        ]
    }
    tool, _ = _make_tool(monkeypatch, [_FakeResponse(200, payload)])

    out = tool("DeepSeek V4")

    assert "DeepSeek V4" in out
    assert "https://example.com/a" in out
    assert "长摘要内容" in out  # 优先用 summary，缺了才用 snippet
    assert "Example News" in out
    assert "score 2.4" in out


@pytest.mark.parametrize(
    "status,payload,expect",
    [
        (401, {"error": {"message": "bad key"}}, "invalid DEEPSEEK_API_KEY"),
        (402, {"error": {"message": "no balance"}}, "余额不足"),
        (429, {"error": {"message": "slow down"}}, "rate limited"),
        (500, {"error": {"message": "boom"}}, "server busy"),
    ],
)
def test_http_errors_become_error_strings(monkeypatch, status, payload, expect):
    """任何 HTTP 错误都要变成字符串（agent 要能读懂并换招），不能抛异常。"""
    tool, _ = _make_tool(monkeypatch, [_FakeResponse(status, payload, text="raw")])

    out = tool("q")

    assert out.startswith("Error: web search failed")
    assert expect in out


def test_empty_results_and_bad_json(monkeypatch):
    tool, _ = _make_tool(
        monkeypatch,
        [_FakeResponse(200, {"results": []}), _FakeResponse(200, None, text="<html>")],
    )

    # 两个不同 query：同一 query 本轮第二次会走缓存（见 test_repeat_query_*）
    assert tool("q1") == "No results found."
    assert "not valid JSON" in tool("q2")


def test_blank_query_short_circuits(monkeypatch):
    """空 query 不打网络，直接返回提示。"""
    tool, calls = _make_tool(monkeypatch, [])

    assert "empty query" in tool("   ")
    assert calls == []


def test_network_exception_becomes_error_string(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    tool = DeepSeekSearchTool()

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("tools.web_tools.requests.post", boom)

    out = tool("q")

    assert out.startswith("Error: web search failed")
    assert "visit_webpage" in out
