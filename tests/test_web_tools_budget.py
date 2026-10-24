"""联网工具的上下文预算：条数、长度、去重、extract。

这三件事都是为了省时间而不是省账单：本机每步都要把全部历史 Observation
重新 prefill 一遍（llama3.2 约 15 tok/s、qwen3.5:9b 约 5 tok/s），
一次搜索多塞 6000 字符就是多几十秒，而模型可能压根用不到。
"""

import pytest

pytest.importorskip("markdownify")

from tools.web_tools import (  # noqa: E402
    ExtractTool,
    TaskScopedCache,
    VisitWebpageTool,
    WebSearchTool,
    _clip,
    extract_relevant_text,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", encoding="utf-8"):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = text.encode(encoding, "ignore")
        self.encoding = encoding
        self.apparent_encoding = encoding

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _results(count, summary_len=800):
    return {
        "results": [
            {
                "title": f"标题{i}",
                "url": f"https://example.com/{i}",
                "summary": "气" * summary_len,
                "site_name": "站点",
            }
            for i in range(count)
        ]
    }


def _make_search(monkeypatch, responses, **kwargs):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_SEARCH_BASE_URL", "https://api.deepseek.com")
    tool = WebSearchTool(**kwargs)
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr("tools.web_tools.requests.post", fake_post)
    return tool, calls


def _make_get(monkeypatch, pages):
    """pages: {url: html}；返回 (调用记录, 恢复函数)。"""
    calls = []

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        if url not in pages:
            return _FakeResponse(404, None, text="not found")
        return _FakeResponse(200, None, text=pages[url])

    monkeypatch.setattr("tools.web_tools.requests.get", fake_get)
    return calls


# ---------- 搜索：条数与长度预算 ----------

def test_search_result_count_is_capped(monkeypatch):
    """API 返回 10 条也只回 3 条（默认，与 WEB_SEARCH_MAX_RESULTS 默认值一致）。"""
    tool, _ = _make_search(monkeypatch, [_FakeResponse(200, _results(10))])

    out = tool("广州 天气")

    assert out.count("https://example.com/") == 3
    assert "返回 10 条" in out


def test_search_total_length_is_capped(monkeypatch):
    """整次输出不超过 max_chars（默认 1800），超了明确写出省略了几条。"""
    tool, _ = _make_search(
        monkeypatch, [_FakeResponse(200, _results(10))], max_chars=900
    )

    out = tool("广州 天气")

    assert len(out) <= 1100          # 900 + 表头/省略说明的余量
    assert "已省略" in out


def test_search_item_summary_is_truncated(monkeypatch):
    """单条摘要截到 item_chars，并标出截断了多少字（不静默截断）。"""
    tool, _ = _make_search(
        monkeypatch, [_FakeResponse(200, _results(1))], item_chars=100
    )

    out = tool("广州 天气")

    assert len(out) < 400
    assert "已截断" in out


def test_repeat_query_hits_cache_without_network(monkeypatch):
    """同一 query 本轮搜第二次不发请求，直接回缓存 + 明确指令。"""
    tool, calls = _make_search(monkeypatch, [_FakeResponse(200, _results(2))])

    first = tool("广州 天气")
    second = tool(" 广州   天气 ")     # 空白与大小写差异不算新 query

    assert len(calls) == 1
    assert "本轮已经搜过" in second
    assert "不要再用同样的 query" in second
    assert "https://example.com/0" in second      # 缓存里仍带结果，模型不至于两眼一抹黑
    assert first and "本轮已经搜过" not in first


def test_begin_task_clears_cache(monkeypatch):
    """新任务（begin_task）之后同一 query 要重新请求。"""
    tool, calls = _make_search(
        monkeypatch,
        [_FakeResponse(200, _results(1)), _FakeResponse(200, _results(1))],
    )

    tool("广州 天气")
    tool.begin_task()
    tool("广州 天气")

    assert len(calls) == 2


# ---------- visit_webpage：去重 + 长度预算 ----------

_PAGE = "<html><body><p>" + "广州今天晴，气温34度。" * 200 + "</p></body></html>"


def test_visit_webpage_is_length_capped(monkeypatch):
    _make_get(monkeypatch, {"https://example.com/a": _PAGE})
    tool = VisitWebpageTool(max_chars=500)

    out = tool("https://example.com/a")

    assert len(out) <= 700
    assert "34度" in out


def test_visit_webpage_same_url_is_not_refetched(monkeypatch):
    calls = _make_get(monkeypatch, {"https://example.com/a": _PAGE})
    tool = VisitWebpageTool()

    tool("https://example.com/a")
    second = tool("https://example.com/a")

    assert len(calls) == 1
    assert "本轮已经抓过" in second
    assert "extract" in second          # 引导它改用便宜的工具


# ---------- extract：只回相关句子，且复用已抓过的页面 ----------

def test_extract_reuses_cached_page(monkeypatch):
    calls = _make_get(monkeypatch, {"https://example.com/a": _PAGE})
    visit = VisitWebpageTool()
    extract = ExtractTool()

    visit("https://example.com/a")
    # 让两个工具共享同一份缓存（真实场景里由 begin_task 一起清空）
    extract.cache = visit.cache
    out = extract("https://example.com/a", "广州今天气温多少度")

    assert len(calls) == 1              # 没有第二次网络请求
    assert "34度" in out
    assert "已抓取过" in out
    assert len(out) < len(_PAGE)


def test_extract_fetches_when_cache_is_empty(monkeypatch):
    calls = _make_get(monkeypatch, {"https://example.com/a": _PAGE})
    tool = ExtractTool(max_chars=200)

    out = tool("https://example.com/a", "广州今天气温多少度")

    assert len(calls) == 1
    assert "34度" in out
    assert len(out) <= 400


def test_extract_reports_missing_question(monkeypatch):
    tool = ExtractTool()

    assert "question" in tool("https://example.com/a", "  ")


def test_extract_rejects_bad_url(monkeypatch):
    _make_get(monkeypatch, {})
    tool = ExtractTool()

    out = tool("example.com/a", "气温")

    assert out.startswith("Error")


# ---------- 抽取算法本身 ----------

def test_extract_relevant_text_picks_matching_sentence():
    page = (
        "首页 关于我们 联系我们 版权声明 京ICP备12345号。"
        "广州今天白天晴，最高气温34度，夜间最低26度。"
        "相关新闻：某地举办马拉松。"
    )

    out = extract_relevant_text(page, "广州今天气温多少", 120)

    assert "34度" in out
    assert "马拉松" not in out


def test_extract_relevant_text_falls_back_when_nothing_matches():
    page = "完全没有相关内容的页面文本，只有一些无关的段落。"

    out = extract_relevant_text(page, "广州气温", 50)

    assert out            # 不能返回空：宁可给原文，也不要让模型以为抓取失败


def test_clip_marks_truncation():
    assert _clip("短文本", 100) == "短文本"
    clipped = _clip("字" * 50, 10)
    assert clipped.startswith("字" * 10)
    assert "已截断 40 字" in clipped


def test_task_scoped_cache_basic():
    cache = TaskScopedCache()
    cache.put("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1
    cache.begin_task()
    assert cache.get("k") is None


# ---------- 每任务调用次数上限（硬刹车） ----------

def test_search_call_cap_stops_the_loop(monkeypatch):
    """真实现场：llama3.2 换着措辞连搜 7 次。到上限后不再打网络。"""
    tool, calls = _make_search(
        monkeypatch,
        [
            _FakeResponse(200, _results(2)),
            _FakeResponse(200, _results(2)),
        ],
        max_calls=2,
    )

    tool("广州 天气")
    tool("广州 今日 天气")          # 不同 query，仍然在额度内
    out = tool("广州 今天 天气情况")  # 第三次：被刹车

    assert "达到上限" in out
    assert len(calls) == 2           # 没有发第三次请求


def test_call_cap_zero_means_unlimited(monkeypatch):
    tool, calls = _make_search(
        monkeypatch,
        [_FakeResponse(200, _results(1)) for _ in range(3)],
        max_calls=0,
    )

    for query in ("a", "b", "c"):
        tool(query)

    assert len(calls) == 3


def test_caps_reset_at_new_task(monkeypatch):
    tool, calls = _make_search(
        monkeypatch,
        [_FakeResponse(200, _results(1)), _FakeResponse(200, _results(1))],
        max_calls=1,
    )

    tool("广州 天气")
    assert "达到上限" in tool("广州 天气2")

    tool.begin_task()
    tool("广州 天气")               # 新任务，额度恢复

    assert len(calls) == 2


def test_extract_and_visit_share_page_cache_and_have_caps(monkeypatch):
    page = (
        "<html><body><p>广州今天晴，气温34度，湿度60%，东北风3级。</p>"
        "<p>明日多云，气温 22 到 28 度，空气质量优，适合户外活动。</p>"
        "<p>后日小雨，气温 20 到 25 度，注意携带雨具，出行注意路面积水。</p>"
        "<p>周末转晴，气温 23 到 30 度，紫外线较强，注意防晒补水。</p></body></html>"
    )
    calls = _make_get(monkeypatch, {"https://e.com/a": page})
    # build_tools 会给三个联网工具注入同一个 TaskScopedCache 实例
    shared = TaskScopedCache()
    visit = VisitWebpageTool(max_calls=2, cache=shared)
    extract = ExtractTool(max_calls=2, cache=shared)

    visit("https://e.com/a")          # 抓一次
    first = extract("https://e.com/a", "广州气温")   # 命中 page 缓存，不再抓
    assert "34度" in first
    assert len(calls) == 1

    # extract 自身额度：第 2 次仍可用（缓存命中不消耗），第 3 次被刹车
    extract("https://e.com/a", "广州湿度")
    third = extract("https://e.com/a", "广州风力")
    assert "达到上限" in third or "相关片段" in third
