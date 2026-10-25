"""联网工具的上下文预算：条数、长度、去重、extract。

这三件事都是为了省时间而不是省账单：本机每步都要把全部历史 Observation
重新 prefill 一遍（llama3.2 约 15 tok/s、qwen3.5:9b 约 5 tok/s），
一次搜索多塞 6000 字符就是多几十秒，而模型可能压根用不到。
"""

import pytest

pytest.importorskip("markdownify")

from tools.web_tools import (  # noqa: E402
    ExtractTool,
    _fetch_page,
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
    assert "搜索结果 10 条：" in out    # 总量要写出来，模型才知道只看到 3 条中的几条


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
    assert "缓存：本轮已搜过同一个 query" in second
    assert "未重新请求网络" in second
    assert "https://example.com/0" in second      # 缓存里仍带结果，模型不至于两眼一抹黑
    assert first and "缓存：本轮已搜过" not in first


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
    assert "缓存：本轮已抓过这个 URL" in second
    # 工具不再教它改用 extract——那是 prompts.yaml 规则 12 的事
    assert "改用 extract" not in second


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
    assert "缓存：本轮已抓过该页面" in out
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


def test_blocked_page_returns_error_prefixed_string(monkeypatch):
    """回归：抓取失败必须回 ``Error:`` 开头的串。

    真实现场（12306）：页面是 JS 渲染 + 反爬，``_fetch_page`` 正确判定
    「没有可读正文」，但错误串**没带前缀**，llama3.2 把它当网页正文
    去 ``splitlines()`` 里过滤「高铁」，拿到空列表后继续换 URL 瞎试，
    最后交出一个链接当答案。
    """
    _make_get(monkeypatch, {"https://example.com/spa": "<html><body><div id=app></div></body></html>"})

    payload, error = _fetch_page("https://example.com/spa")

    assert payload is None
    assert error.startswith("Error:")
    # 只留 Error: 前缀（模型靠它判断失败），不再附带行为指令
    assert "不要拿它当正文解析" not in error


def test_visit_webpage_surfaces_blocked_page_as_error(monkeypatch):
    _make_get(monkeypatch, {"https://example.com/spa": "<html><body><div id=app></div></body></html>"})

    out = VisitWebpageTool()("https://example.com/spa")

    assert out.startswith("Error:")


def test_extract_accepts_search_output_as_text_without_network(monkeypatch):
    """真实现场：llama3.2 写 ``extract(web_search结果, query)``。

    它想的是「从搜索结果里挑相关行」，旧实现回一句
    「url 必须以 http:// 开头」并把 1800 字入参原样回显，
    模型随后连 4 步在 print URL 里打转，最后交出一个链接当答案。
    """
    def _boom(*args, **kwargs):
        raise AssertionError("文本模式不该发起网络请求")

    monkeypatch.setattr("tools.web_tools._fetch_page", _boom)
    tool = ExtractTool(max_chars=400)

    search_out = (
        "搜索结果（3 条原始数据。这不是答案）：\n"
        "- MU5102\n  https://jipiao.cncn.com/mobile/flight_MU5102.html\n"
        "  MU5102 08:00 北京首都国际机场 上海 10:15 上海虹桥国际机场\n"
        "- 无关结果\n  https://example.com/x\n  这里讲的是完全不相干的事情。\n"
    )

    out = tool(search_out, "北京飞上海机票")

    assert "08:00 北京首都国际机场" in out
    assert "未联网" in out


def test_extract_text_mode_rejects_input_too_short_to_be_text():
    tool = ExtractTool()

    out = tool("随便写两个字", "气温")

    assert out.startswith("Error")
    assert "http://" in out        # 同时给出「你是不是想抓页面」的出路


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


def test_extract_prefers_current_weather_card_over_navigation():
    """真实现场：问题词（天气/预报）全在标题和导航上，数据块一个都不含。

    早期版本按关键词打分，抽出的是「今日天气提示 / 7天预报 / 24小时预报」，
    模型据此判定「没查到天气」又搜一遍，最后打满 max_steps 走兜底。
    """
    page = "\n".join([
        "随时随地 想查就查",
        "深圳市，广东省，中国",
        "45 优",
        "28",
        "晴",
        "今天17:04更新",
        "湿度 53%",
        "东北风4级",
        "今日天气提示",
        "略微偏热，注意衣物变化。",
        "Windows 下载",
        "7天预报",
        "10天预报",
        "15天预报",
        "23° / 31°",
        "24小时预报",
        "温度",
        "风力",
        "7天预报省份表",
        "10天预报城市列表",
    ])

    out = extract_relevant_text(page, "天气预报", 1200)

    # 真正的答案必须在：当前气温、天气状况、湿度、风力
    assert "28" in out
    assert "晴" in out
    assert "湿度 53%" in out
    assert "东北风4级" in out
    # 纯导航标签不能被当成数据喂回去
    assert "10天预报" not in out
    assert "省份表" not in out
    assert "Windows 下载" not in out


def test_extract_keeps_keyword_sentences_when_top_match_is_content():
    """命中块本身就是正文时，不要改成「连续数据窗口」——否则会丢掉分散的句子。"""
    page = "\n".join([
        "首页 关于我们 联系我们",
        "广州今天白天晴，最高气温34度，夜间最低26度，体感闷热。",
        "相关新闻：某地举办马拉松，报名人数破万。",
        "门票开售时间为每天上午十点，售完即止。",
    ])

    out = extract_relevant_text(page, "广州今天气温", 1200)

    assert "34度" in out
    assert "马拉松" not in out


def test_extract_keeps_short_answer_on_long_page():
    """短答案 + 长页面：答案必须留下，不能被「整页兜底」冲掉。

    真实事故：问「fastapi 依赖注入怎么写」，答案只有一句
    「用 Depends 声明参数即可」，短短十几个字、没有数字，
    早期版本把它判成「页面壳」，换成整页开头 —— 答案彻底丢失。
    """
    page = "\n".join(
        ["Python 进阶"]
        + [
            f"第 {i} 章讲的是与问题无关的基础语法，"
            f"这一节的内容用来把答案推到页面很靠后的位置。"
            for i in range(20)
        ]
        + [
            "装饰器",
            "用 functools.wraps 装饰内层函数，可以保留原函数的函数名和文档字符串。",
        ]
    )

    out = extract_relevant_text(page, "装饰器怎么保留原函数名", 1200)

    assert "functools.wraps" in out          # 答案必须在
    assert "第 0 章" not in out              # 无关正文不能靠「整页兜底」混进来


def test_extract_matches_synonym_time_words():
    """问「今日天气」、页面写「今天」：同义归一后必须命中，而不是整页兜底。"""
    page = "\n".join([
        "首页",
        "北京天气预报",
        "今天",
        "晴",
        "25℃",
        "湿度 40%",
        "东北风3级",
        "未来一周冷空气频繁，注意保暖。",
    ])

    out = extract_relevant_text(page, "今日天气", 1200)

    assert "25℃" in out
    assert "湿度 40%" in out
    assert "未来一周冷空气频繁" not in out      # 与问题无关的长句不该被带上


def test_extract_does_not_treat_news_sentence_as_data():
    """新闻句里带「13℃」也不能被当成天气数据块并进答案。

    真实事故：问「今日天气」，结果是一句
    「北京今晚起有雨，明天最高温 13℃，或创下半年新低」——
    卡片上的 28℃/晴/湿度全被挤掉了。
    """
    page = "\n".join([
        "首页",
        "北京天气预报",
        "今天",
        "晴",
        "28℃",
        "湿度 53%",
        "东北风4级",
        "北京今晚起有雨，明天最高温 only 13℃，或创下半年新低。",
    ])

    out = extract_relevant_text(page, "今日天气", 1200)

    assert "28℃" in out
    assert "晴" in out
    assert "湿度 53%" in out
    assert "今晚起有雨" not in out
    assert "only" not in out                 # 模板漏出来的英文填充词要清掉


def test_nav_label_does_not_eat_body_sentences():
    """「下载」类词只删纯菜单项，不能把正文里的句子一起删掉。"""
    from tools.web_tools import _is_nav_label

    assert _is_nav_label("客户端下载") is True
    assert _is_nav_label("7天预报省份表") is True
    assert _is_nav_label("温度") is True
    # 带动作词的是句子，不是菜单项
    assert _is_nav_label("点击这里下载") is False
    assert _is_nav_label("点击这里下载安装包") is False
    assert _is_nav_label("下载安装说明") is False


def test_nav_numbers_are_not_data_values():
    """``7天预报`` 含数字但是导航标签，不能算「数据值」。"""
    from tools.web_tools import _is_data_value

    assert _is_data_value("28") is True
    assert _is_data_value("湿度 53%") is True
    assert _is_data_value("¥ 68") is True
    assert _is_data_value("涨") is True
    assert _is_data_value("7天预报") is False
    assert _is_data_value("10天预报城市列表") is False


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

    assert "已达上限" in out
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
    assert "已达上限" in tool("广州 天气2")

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
    assert "已达上限" in third


def test_repeat_notice_does_not_nest_clip_markers():
    """回归：重贴缓存结果时不能把已有的「已截断 N 字」标记切开套娃。

    日志现场（查高铁班次，重复搜同一个 query）：

        '…（已…（已截断 397 字）'

    ``_repeat_notice`` 原来用 ``_clip(cached, 800)`` 按字符切，而 cached 是
    上一步**已经渲染好**的结果，摘要行尾本来就有 ``…（已截断 268 字）``，
    800 的切点正好落在标记中间；追加的新标记还相对"已被截断的副本"算字数，
    两个数字互相矛盾。
    """
    from tools.web_tools import _clip, WebSearchTool

    tool = WebSearchTool(max_results=3)
    # 构造：切点落在既有标记内部
    cached = "a" * 793 + "…（已截断 268 字）" + "b" * 200

    out = tool._repeat_notice(cached)

    assert "已…（已" not in out
    assert out.count("…（已截断") <= 1


def test_repeat_notice_keeps_whole_lines():
    """重贴时按整行切：行尾标记保持完整，末尾说明还省了多少行。"""
    from tools.web_tools import WebSearchTool

    tool = WebSearchTool(max_results=3)
    cached = "\n".join(f"- 结果{i} " + "字" * 60 for i in range(30))

    out = tool._repeat_notice(cached)

    assert "未显示" in out
    # 被保留的最后一行必须是完整行，不能是半截
    body = out.split("\n", 1)[1]
    last_data_line = [ln for ln in body.splitlines() if ln.startswith("- 结果")][-1]
    assert last_data_line.endswith("字" * 60)


# ---------- JS 渲染骨架页：别让模型在一个站上耗光步数 ----------

_SHELL_TEXT = (
    "Today\nHourly\n10-Day\nMonthly\nAdvertisement\n"
    "Xiuying District Weather\n"
    "Xiuying District, 570312, Haikou, Hainan, People's Republic of China\n"
    "Today's outlook\nWind\nHumidity\nAir Quality\nDew Point\nPressure\n"
    "UV Index\nVisibility\nMoon Phase\nSunrise\nSunset\nRadar\nAdvertisement"
)


def test_js_rendered_shell_page_is_detected():
    """真实现场：weather.com 正文由 JS 渲染，requests 只拿到 361 字导航标签。"""
    from tools.web_tools import _looks_like_page_skeleton

    assert _looks_like_page_skeleton(_SHELL_TEXT) is True


def test_normal_content_page_is_not_treated_as_skeleton():
    from tools.web_tools import _looks_like_page_skeleton

    body = (
        "海口天气预报\n全国 > 海南 > 海口 11:30更新\n"
        "25日（今天） 多云 31/23℃ <3级\n26日（明天） 雷阵雨 29/25℃ 3-4级"
    )
    assert _looks_like_page_skeleton(body) is False

    # 纯文字页面（没有数值也没有状态词）也不该被误判：导航占比低
    prose = (
        "装饰器是一个接收函数并返回函数的可调用对象。\n"
        "用 functools.wraps 可以保留原函数的名字和文档字符串。\n"
        "如果不加 wraps，被装饰函数的 __name__ 会变成 wrapper。"
    )
    assert _looks_like_page_skeleton(prose) is False


def test_extract_on_shell_page_says_give_up_on_this_site(monkeypatch):
    """抽不出东西时必须明说「别再抓这个站」，模型才不会再换天气站瞎试。

    真实现场（「Haikou current weather」）：extract 从中挑出 82 字
    （Advertisement + 一行地名）当成功返回，模型于是接着抓第二个、第三个
    天气站，403 / 超时，直到打满 max_steps 走兜底答案。
    """
    url = "https://weather.example.com/haikou/today"
    monkeypatch.setattr(
        "tools.web_tools._fetch_page",
        lambda u: ({"markdown": _SHELL_TEXT, "text": _SHELL_TEXT}, None),
    )

    out = ExtractTool()(url, "Current temperature in Haikou")

    assert out.startswith("Error:")
    assert "JavaScript" in out
    assert out.rstrip().endswith("静态请求拿不到。")   # 只陈述事实，不给建议


def test_extract_on_normal_page_still_returns_snippets(monkeypatch):
    url = "https://weather.example.cn/haikou"
    page = "海口天气预报 11:30更新 25日（今天） 多云 31/23℃ <3级 湿度 68%"
    monkeypatch.setattr(
        "tools.web_tools._fetch_page",
        lambda u: ({"markdown": page, "text": page}, None),
    )

    out = ExtractTool()(url, "海口今天气温")

    assert not out.startswith("Error:")
    assert "31/23" in out
