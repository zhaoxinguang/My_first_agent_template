"""联网工具：网页搜索、网页正文抓取、按问题抽取。

- ``web_search``：DeepSeek 官方 Search API（``/v1/tools/search``），Bearer 鉴权；
- ``extract``：抓页面，但**只回传与问题相关的行**（比整页便宜一个数量级）；
- ``visit_webpage``：抓整页正文并转 Markdown。

三个工具都返回「字符串」，可以直接被 CodeAgent 当作 Python 函数调用，
调用结果会出现在下一步的 Observation 里。

三者共享同一份 ``TaskScopedCache``（由 ``tools/__init__.py::build_tools`` 注入）：
- 同一 query / 同一 url 本轮第二次调用**不再打网络**，回缓存 + 一行「缓存：未重新请求网络」；
- 每个工具还有「每任务调用次数上限」：换个措辞就能绕过去重，用硬刹车兜住
  （``TaskScopedCache.allow()``，``max_calls<=0`` 表示不限制）。
每轮任务开始时由 ``agent/guardrails.py::reset_tool_state`` 调 ``begin_task()`` 清空。

关于 extract 的抽取质量（踩过的坑）：
早期版本直接把 markdownify 出来的 markdown 切句打分，结果 ``[墨迹天气](https://…)``
这类导航链接、面包屑（``成都市，四川省，中国``）因为含「天气」「成都」而被选中，
真正的内容行（``14 阴 今天7:53更新 湿度 100% 北风2级``）反而被挤掉。
现在的做法：
1. 从**原始 HTML** 取正文（``_readable_text``）：丢掉 script/nav/header/footer，
   图片丢掉，``<a>`` 只留文字，块级标签换成换行 —— 链接语法根本不进候选集；
2. 以「块（行）」为单位当候选，超长行才切成句子 —— 同一块里的数据和它的描述不会被拆散；
3. 打分：问题关键词按 idf 加权重（``天气`` 这种到处都是的词权重低），
   带数字/单位的块加分，符号噪声（``[]()|#*`` 等）占比高的块减分；
4. 只留分数 ≥ 最佳分 60% 的块，最多 6 块 —— 宁可少给，也不给一堆导航碎片；
5. 关键词全落在导航上时（问天气，页面上只有 ``今日天气提示`` ``7天预报``），
   改用「以命中块为中心的一段**连续**数据」：数据块连着数据块，
   ``28 晴 今天17:04更新 湿度 53% 北风3级`` 这种卡片才是模型要的答案；
   多个候选窗口都够干净时取最靠前的一个（首屏是当前值，往下是 24 小时表、
   7 天表，数据更密却答不了「今天多少度」）；
6. 结果里再剔除纯导航标签块（``7天预报`` / ``24小时预报`` / ``省份表`` / ``温度``），
   它们含问题里的词又带数字，很容易被当成「相关信息」拼进去。

判据是**通用**的，不是给天气写的（踩过的坑，按严重程度排序）：

- 关键词要**同义归一**再比对：页面写「今天 晴 25℃」、用户问「今日天气」，
  字面一个词都不重合，整张卡片得 0 分，再好的后处理也救不回来；
- 「数据」= ``数字+单位``（``53%`` / ``4级`` / ``¥ 68`` / ``120ms`` / ``3 年``）
  或纯状态词（晴/小雨/涨/跌/胜/已结束，可用 ``WEB_EXTRACT_STATE_WORDS`` 追加）；
  **必须短**——新闻标题里也常有 ``13℃``，那是正文，不是数据块；
- 「短但是正确的答案」不能被当成导航壳：``用 Depends 声明参数即可`` 只有十几个字、
  没有数字，早期版本判成 chrome 后退回整页，答案就丢了；
- 导航标签要认准：``点击这里下载`` 是正文句子，早期用
  ``[^，。！？]{0,10}(?:下载)$`` 一刀切，把它整行删了。
"""

import math
import os
import re
from collections import Counter
from html import unescape
from urllib.parse import urlparse

import requests
from markdownify import markdownify
from smolagents.tools import Tool

# 网页正文最多回传给模型多少字符（太长会吃掉上下文预算）
MAX_CONTENT_LENGTH = 6000
REQUEST_TIMEOUT = 20.0

# 搜索结果的上下文预算默认值（正常路径由 config/settings.py 的 WEB_SEARCH_* 注入，
# 这里只是直接构造工具时的兜底，取值与 settings 默认值保持一致）
DEFAULT_SEARCH_MAX_RESULTS = 3
DEFAULT_SEARCH_ITEM_CHARS = 240
DEFAULT_SEARCH_MAX_CHARS = 1800
# extract 默认上限：只回相关行，1200 字符足够覆盖一两个事实
DEFAULT_EXTRACT_MAX_CHARS = 1200

# 页面正文短于这个长度就当成「导航壳 / JS 渲染页」，不要喂给模型
_MIN_CONTENT_LEN = 80


# =============================================================================
# 本轮缓存 + 调用次数账本
# =============================================================================

class TaskScopedCache:
    """一次任务内的「去重缓存 + 调用计数」，任务结束（换任务）即清空。

    放在一个对象里而不是每工具一份，是为了让 ``extract`` 能直接复用
    ``visit_webpage`` 已经抓过的页面：换个问题问同一页只重做抽取，不重抓。

    两类 key 前缀互不干扰：``search:<query>`` / ``page:<url>``。
    """

    def __init__(self):
        self._store: dict = {}
        self._calls: dict = {}

    # ---- 缓存 ----

    def get(self, key, default=None):
        return self._store.get(key, default)

    def put(self, key, value):
        self._store[key] = value
        return value

    def __len__(self):
        return len(self._store)

    def __contains__(self, key):
        return key in self._store

    # ---- 调用计数 ----

    def calls(self, name: str) -> int:
        return self._calls.get(name, 0)

    def allow(self, name: str, limit: int) -> bool:
        """记账并判断这次调用是否还在预算内；``limit<=0`` 表示不限制。"""
        used = self._calls.get(name, 0) + 1
        self._calls[name] = used
        if not limit or limit <= 0:
            return True
        return used <= limit

    # ---- 换任务 ----

    def begin_task(self):
        """新任务开始：缓存与计数一起归零。"""
        self._store.clear()
        self._calls.clear()


# =============================================================================
# 通用小工具
# =============================================================================

def _clip(text: str, limit: int) -> str:
    """按字符数截断，并**显式**标出截掉了多少（不静默截断）。"""
    text = str(text or "").strip()
    if not limit or limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…（已截断 {len(text) - limit} 字）"


_OLD_CLIP_MARKER_RE = re.compile(r"…（已截断 \d+ 字）")


def _clip_lines(text: str, limit: int) -> str:
    """按整行截断，绝不切开行尾已有的「已截断 N 字」标记。

    只用于**已经渲染过**的文本（比如 `_repeat_notice` 重贴一份缓存结果）。
    `_clip` 是按字符切的，切点落在既有标记中间时会出现嵌套：

        …（已截断 2…（已截断 205 字）

    两个数字还互相矛盾（后一个相对已被截断的副本算），读的人无从判断。
    按行切就没这个问题：标记总在行尾，不会被打断。
    """
    text = str(text or "")
    if len(text) <= limit:
        return text

    lines = text.splitlines()
    if lines and len(lines[0]) > limit:
        # 极端情况：整段结果挤成一行（没有换行可切）。只能退回按字符截，
        # 但先把它自己已有的「已截断 N 字」换成裸省略号——
        # 否则新标记会套在旧标记里，两个数字互相矛盾。
        return _clip(_OLD_CLIP_MARKER_RE.sub("…", lines[0]), limit)

    kept: list[str] = []
    used = 0
    for line in lines:
        # 第一条无论如何要留下，否则模型看到一片空白以为搜索失败
        if kept and used + len(line) + 1 > limit:
            break
        kept.append(line)
        used += len(line) + 1

    omitted = len(lines) - len(kept)
    if not omitted:
        return text
    return "\n".join(kept) + f"\n（重复的结果还有 {omitted} 行未显示）"


def _validate_url(url: str) -> str:
    """返回错误字符串；URL 合法时返回空字符串。"""
    if not url:
        return "Error: url 为空。"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return (
            f"Error: url 必须以 http:// 或 https:// 开头（当前值：{_clip(url, 60)}），"
            "例如 https://www.weather.com.cn/。"
        )
    return ""


# 整个字符串就是一个网页地址：单行、无空白、http(s) 开头
_URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
# 走「文本模式」的最小长度：再短的多半是模型把半个域名当文本传进来了
_MIN_TEXT_MODE_CHARS = 40


def _looks_like_url(text: str) -> bool:
    """整个字符串是不是一个网页地址（不允许多行、不允许多余空白）。

    ``extract`` 靠它区分两种模式：传 URL 就抓页面，传一段文本就在文本里抽。
    真实事故：llama3.2 写了 ``extract(web_search的结果, query)``——
    模型的直觉没错（从搜索结果里抽相关行），是工具太窄。
    """
    return bool(_URL_ONLY_RE.match(text or ""))


def _search_key(query: str) -> str:
    """query 归一化：大小写与多余空白不算新 query。"""
    return "search:" + " ".join(query.split()).lower()


def _page_key(url: str) -> str:
    return "page:" + url.strip()


def _fix_encoding(response: requests.Response) -> None:
    """修正 requests 对中文页面的编码误判。

    requests 在响应头没有 charset 时会把 text 按 ISO-8859-1 解码，
    导致 GBK 页面出现 ``æµ·æ·`` 这类 mojibake。
    """
    # 1) 从 HTML meta 里找 charset（最可靠）
    meta_charset = None
    m = re.search(
        rb'<meta[^>]+charset=["\']?([a-zA-Z0-9_\-]+)',
        response.content[:2048],
        re.IGNORECASE,
    )
    if m:
        meta_charset = m.group(1).decode("ascii", "ignore").lower()

    # 2) 响应头声明的编码
    header_enc = (response.encoding or "").lower()
    if header_enc in ("iso-8859-1", "latin-1", ""):
        header_enc = None

    # 3) 优先 meta，其次 header，最后让 requests/chardet 猜
    if meta_charset:
        response.encoding = meta_charset
    elif header_enc:
        response.encoding = header_enc
    else:
        guessed = response.apparent_encoding
        response.encoding = guessed or "utf-8"


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _page_error(message: str) -> str:
    """网页抓取错误的统一格式：**必须以 ``Error:`` 开头**。

    模型靠前缀判断「这一步失败了」。早期版本这条不带 ``Error:``，真实现场：
    llama3.2 拿到 "The page returned little to no readable content…"，
    把它当成网页正文去 ``splitlines()`` 里过滤「高铁」，得到空列表后
    继续换 URL 瞎试，最后交出一个链接当答案。

    这里只留 ``Error:`` 前缀 + 事实描述。以前还跟过一句「（这不是网页内容，
    不要拿它当正文解析）」，属于在工具回执里写行为指令——已挪到
    ``config/prompts.yaml`` 的 web 规则里，回执只说自己发生了什么。
    """
    return f"Error: {message}"


# 「抓回来的只是一副页面骨架」的判定阈值：短到这个长度，且一个数值/状态词
# 都没有，再叠加导航占比，基本可以断定正文是 JavaScript 渲染出来的。
_SKELETON_MAX_CHARS = 600
_SKELETON_CHROME_RATIO = 0.6


def _looks_like_page_skeleton(text: str) -> bool:
    """抓回来的正文是不是只有导航骨架（真内容由 JS 注入）。

    真实现场（「Haikou current weather」）：模型按搜索回执的提示去
    ``extract(url=weather.com/...)``，该站正文全是 JS 渲染，
    ``requests`` 只拿到 361 字——``Today / Hourly / 10-Day / Advertisement /
    Humidity / Air Quality …``，一个温度都没有。抽取函数老老实实从中挑了
    82 字交回去（``Advertisement`` + 一行地名），看着像成功，模型于是
    继续换下一个天气站抓，403、超时，直到打满 max_steps。

    三个条件同时成立才算骨架：够短、没有任何「数字+单位」或状态词、
    且候选块里导航占比过半。纯文字页面（散文、文档）不会被误判——
    它们导航占比低。
    """
    text = str(text or "")
    if not text or len(text) >= _SKELETON_MAX_CHARS:
        return False
    # 用严格的「数字+单位」而不是 _UNIT_HINT_RE——后者只要有数字旁证就命中，
    # 而骨架页里的 "10-Day" / 邮编 570312 全是数字，会把它放过去。
    if _UNIT_VALUE_RE.search(text) or _STATE_WORD_RE.search(text):
        return False
    fragments = _fragments(text)
    if not fragments:
        return True
    # 这里不能拿 _looks_like_chrome(fragment) 数「导航块」——它的单参语义是
    # 「整段结果像不像导航壳」，对所有短片段都返回 True，会把一篇正常短文
    # 也算成骨架。用 _is_content_sentence 判「这一块是不是像样的句子」。
    non_sentence = sum(1 for fragment in fragments if not _is_content_sentence(fragment))
    return non_sentence / len(fragments) >= _SKELETON_CHROME_RATIO


def _fetch_page(url: str) -> tuple:
    """抓一个 URL，返回 ``(payload, error)``，只填一个。

    ``payload = {"markdown": 整页 Markdown, "text": 可读纯文本}``：
    visit_webpage 用 markdown，extract 用 text，缓存里只存一份。
    """
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return None, _page_error(
            "The request timed out. Please try again later or check the URL."
        )
    except requests.exceptions.RequestException as exc:
        return None, _page_error(f"Error fetching the webpage: {exc}")

    _fix_encoding(response)
    html = response.text or ""

    # extract 用：HTML 直接转可读纯文本（不带 markdown 链接语法）
    readable = _readable_text(html)

    try:
        markdown_content = markdownify(html).strip()
    except Exception as exc:  # HTML 结构异常时不要让整个 agent 崩掉
        return None, _page_error(
            f"An unexpected error occurred while parsing the page: {exc}"
        )

    # 压缩连续空行，减少无意义的 token
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

    # 内容有效性校验，避免把导航壳页当结果喂给模型
    if len(markdown_content) < _MIN_CONTENT_LEN and len(readable) < _MIN_CONTENT_LEN:
        return None, _page_error(
            "The page returned little to no readable content "
            "(possibly JS-rendered or blocked). "
            "Try another URL or a direct data API."
        )

    return {"markdown": markdown_content, "text": readable}, None


def _page_markdown(payload) -> str:
    """从缓存里取整页 Markdown（兼容只存了字符串的旧缓存）。"""
    if isinstance(payload, dict):
        return str(payload.get("markdown") or "")
    return str(payload or "")


def _page_text(payload) -> str:
    """从缓存里取可读纯文本（旧缓存只有 Markdown 时现场反推）。"""
    if isinstance(payload, dict):
        text = str(payload.get("text") or "")
        return text or _demarkdown(payload.get("markdown") or "")
    return _demarkdown(str(payload or ""))


# =============================================================================
# HTML → 可读纯文本（extract 的候选集来源）
# =============================================================================

# 整块都是噪声的标签：连内容一起丢掉
_DROP_BLOCK_RE = re.compile(
    r"(?is)<(script|style|noscript|template|svg|iframe|canvas|form|button|"
    r"select|option|nav|header|footer|aside|menu)\b[^>]*>.*?</\1\s*>"
)
# 图片/输入框等：留着只会变成图标名和占位文本
_VOID_NOISE_RE = re.compile(
    r"(?is)<(?:img|input|link|meta|source|track|path|use|area|base)\b[^>]*>"
)
# 块级标签边界换成换行：同一块里的「14 阴 湿度 100%」不会被拆成三行
_BLOCK_BREAK_RE = re.compile(
    r"(?is)</?(?:p|div|br|hr|li|ul|ol|dl|dt|dd|tr|td|th|table|thead|tbody|tfoot|"
    r"h[1-6]|section|article|blockquote|pre|figure|figcaption|main|center)\b[^>]*>"
)
_LINK_OPEN_RE = re.compile(r"(?is)<a\b[^>]*>")
_LINK_CLOSE_RE = re.compile(r"(?is)</a\s*>")
# <title> 整块丢掉：它对正文没有信息量，却是「问题关键词命中率最高的一行」——
# 真实事故：问「天气预报」，标题「【深圳市天气】_深圳市天气预报_天气查询 - 墨迹天气」
# 拿下最高分独占结果，页面里真正的「28 晴 湿度 53%」一块都没进去。
_TITLE_TAG_RE = re.compile(r"(?is)<title\b[^>]*>.*?</title\s*>")
# 短链接（导航项、标签、面包屑）单独成行，别和正文粘在一行里。
# 只匹配「独占一个块的链接」：前后紧挨着标签（说明块里只有这个链接），
# 正文句子中间的链接（「…降温，<a>气象台</a>发布了…」）不会被切断。
_BLOCK_LINK_RE = re.compile(r"(?is)(?<=>)\s*<a\b[^>]*>(.{0,40}?)</a\s*>(?=\s*<)")
_ANY_TAG_RE = re.compile(r"(?is)<[^>]+>")
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\ufeff\u00a0]")

# 链接文字短于这个长度就当成导航项：单独成行，避免「天气 空气质量 成都…」粘一起
_SHORT_LINK_LEN = 12


def _split_short_link(match: "re.Match") -> str:
    """独占一个块的短链接只留文字并单独成行；长链接（正文里的引用）保持原样。"""
    inner = _ANY_TAG_RE.sub(" ", match.group(1))
    inner = " ".join(unescape(inner).split())
    if not inner:
        return " "
    if len(inner) <= _SHORT_LINK_LEN:
        return f"\n{inner}\n"
    return f" {inner} "


def _readable_text(html: str) -> str:
    """HTML → 纯文本：去脚本/导航/页脚，链接只留文字，块级标签保留换行。"""
    if not html:
        return ""

    text = _DROP_BLOCK_RE.sub("\n", html)
    text = _TITLE_TAG_RE.sub("\n", text)
    text = _VOID_NOISE_RE.sub(" ", text)
    text = _BLOCK_LINK_RE.sub(_split_short_link, text)   # 导航链接单独成行
    text = _LINK_OPEN_RE.sub(" ", text)                  # 其余 <a> 只保留文字
    text = _LINK_CLOSE_RE.sub(" ", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = unescape(text)
    text = _INVISIBLE_RE.sub(" ", text)

    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


# =============================================================================
# 按问题抽取相关行
# =============================================================================

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_MD_MARK_RE = re.compile(r"[*_`#>~]+")
# 行首的 Markdown 列表符：``- only 13℃`` 里的 ``-`` 会在清洗后粘到数字上
_MD_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*+•][ \t]+")
# 夹在中文里的英文「填充词」：页面模板漏出来的标注词（真实事故：
# ``最高温 only 13℃``）。只认这一小张表，不要按「长度 ≤4 的英文词」一刀切：
# 那样会把 TCP、JSON、__name__ 这类正文内容一起删掉。
_DANGLING_DASH_RE = re.compile(r"(?<=[\u4e00-\u9fff])[-–—]{1,2}\s+(?=[\d\u4e00-\u9fff])")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_FILLER_EN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"only|total|more|read|view|free|hot|top|new|best|latest|updated|"
    r"click|share|loading|default|"
    r"advertisement|advertising|sponsored|subscribe|newsletter|"
    r"sign in|log in|sign up"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"[。！？；!?;]+")
_UNIT_HINT_RE = re.compile(r"\d|℃|度|%|级|hPa|百帕|mm|km")
_NOISE_CHARS = set("[](){}<>|\\/*#_~`•…·=+")

_TOKEN_NOISE = set("的了和与及或是在有也都就而请帮我查一下吗呢吧啊这个那个如何什么多少怎么")

# 「带单位的数据值」：任何「数字 + 计量单位」都算数据，**不限于天气**。
# 天气页面写 25℃ / 湿度 40% / 东北风3级，行情页面写 7.1234 / +0.12% / 12.5 元，
# 文档页面写 120ms / 8 GB，歌单页面写 3 分 42 秒 —— 单位不同，判据是同一个。
_UNIT_VALUE_RE = re.compile(
    r"(?:\d\s*(?:℃|°|%|‰|级|hPa|百帕|mm|cm|km|kg|"
    r"GB|MB|KB|TB|MHz|GHz|ms|µs|ns|req/s|qps|tps|fps|m/s|bit|dpi|px|"
    r"万元|亿元|亿美元|元|美元|欧元|港元|日元|"
    r"分钟|小时|点|分|秒|天|周|月|年|岁|人|次|票|股|手|"
    r"度|瓦|伏|安|毫升|升|克|斤|公斤)"
    r"|[¥￥$€£]\s*\d)"
)
_PURE_NUMBER_RE = re.compile(r"^[-+]?\d[\d.,]*\s*(?:%|℃|度|级|km|mm|hPa)?$")
_DIGITS_RE = re.compile(r"[\d]")
# 单独成块也有信息量的「状态词」：天气（晴/小雨）、行情（涨/跌/涨停）、
# 赛事（胜/负/已结束）。只对 ≤4 字的整块生效，避免把正文句子当数据。
# 想追加词表：环境变量 WEB_EXTRACT_STATE_WORDS，用 | 分隔。
_DEFAULT_STATE_WORDS = (
    "晴|多云|阴|小雨|中雨|大雨|暴雨|雷阵雨|阵雨|雪|小雪|中雪|大雪|雨夹雪|"
    "雾|霾|扬尘|浮尘|沙尘|优|良|轻度污染|中度污染|重度污染|严重污染|"
    "涨|跌|平|涨停|跌停|停牌|休市|开盘|收盘|"
    "胜|负|平局|进行中|已结束|延期|取消"
)


def _state_word_pattern() -> str:
    """状态词表：内置中文状态词 + 环境变量追加的词。"""
    extra = [
        word.strip()
        for word in re.split(r"[|,，\s]+", os.environ.get("WEB_EXTRACT_STATE_WORDS", ""))
        if word.strip()
    ]
    return "|".join([_DEFAULT_STATE_WORDS] + [re.escape(word) for word in extra])


_STATE_WORD_RE = re.compile(rf"^(?:{_state_word_pattern()})$")
# 同一批词的非锚定版本，只给 _looks_like_chrome 判「这段里有没有状态信息」用
_STATE_WORD_ANY_RE = re.compile(rf"(?:{_state_word_pattern()})")

# 纯导航标签：整块就是一个菜单项 / 按钮名，单独拼给模型没有任何信息量
_NAV_FRAGMENT_RE = re.compile(
    r"(?:"
    r"\d+\s*(?:天|日|小时|周|月|年)\s*预报(?:省份|城市)?(?:列表|表)?"
    r"|(?:省份|城市|地区|区县)(?:列表|表|大全)"
    r"|温度|风力|湿度|空气质量|生活指数|历史天气|天气日历|天气预警"
    r"|(?:客户端|应用|软件|电脑版|手机版|网页版|APP|Windows|Mac|iOS|"
    r"安卓|Android|安装包|PC版)?\s*下载"
    r"|首页|上一页|下一页|末页|返回顶部|更多|展开|收起"
    r")$"
)
# 导航标签里出现「动作词」说明这块是句子（「点击这里下载安装包」），不是菜单项。
# 早期版本用 ``[^，。！？]{0,10}(?:下载)$`` 一刀切，把正文里的「点击这里下载」整行删了。
_NAV_ACTION_RE = re.compile(
    r"点击|点此|这里|此处|立即|马上|前往|安装|如何|怎么|怎样|"
    r"获取|地址|链接|教程|说明|流程|步骤"
)
# 超过这个长度的块不可能是纯菜单项
_MAX_NAV_LABEL_LEN = 16
# 「数据行」的长度上限：新闻标题里常带「13℃」，那是正文不是数据块
_MAX_DATA_BLOCK_LEN = 24
# 数据格（``湿度 53%`` / ``4 核 16G``）不含句读标点；含标点的是句子
_SENTENCE_PUNCT_RE = re.compile(r"[，。；、！？,;!?]")

# 同义归一：只作用于**匹配**，绝不改写返回给模型的原文。
# 真实事故：用户问「今日天气」，页面上写的是「今天 晴 25℃」——
# 字面一个词都不重合，整张卡片得分为 0，最后走了「退回原文开头」，
# 模型拿到的是一句新闻标题。这里把时间词/常用近义词映射到同一写法，两侧都归一。
_SYNONYM_MAP = {
    "今天": "今日", "本日": "今日", "当天": "今日", "今日": "今日",
    "明天": "明日", "次日": "明日",
    "昨天": "昨日", "前天": "前日", "后天": "后日",
    "气温": "温度", "降雨": "降水", "下雪": "降雪",
    "几度": "温度", "多少度": "温度",
}
_SYNONYM_RE = re.compile(
    "|".join(sorted((re.escape(k) for k in _SYNONYM_MAP), key=len, reverse=True))
)
# 页面上单独成块的日期标签（「今天」「10月8日」「周三」），紧挨着数据块时要一起带上
_TIME_LABEL_RE = re.compile(
    r"^(?:今日|明日|昨日|后日|前日|"
    r"\d{1,2}\s*[月日号]|\d{4}-\d{1,2}-\d{1,2}|"
    r"周[一二三四五六日天]|星期[一二三四五六日天])$"
)


def _normalize_for_match(text: str) -> str:
    """把同义词统一成一种写法，只用于比对，不用于输出。"""
    return _SYNONYM_RE.sub(lambda m: _SYNONYM_MAP[m.group(0)], str(text or ""))


def _is_content_sentence(text: str) -> bool:
    """这块像不像「一句话」（而不是标题/标签）。

    短答案要能挡住「退整页」的兜底，但页面标题也是短块：``FastAPI 指南``
    只有 10 个字、没有标点，判成内容就会把整页答案错过。所以要求
    够长、或者带句读标点。
    """
    unit = (text or "").strip()
    if not unit:
        return False
    return len(unit) >= 16 or any(mark in unit for mark in "，。、；：,;:")


def _is_time_label(text: str) -> bool:
    """这个块是不是一个纯日期标签（「今日」「10月8日」「周三」）。

    归一后再判：「今天」「明天」和「今日」「明日」都算。
    """
    return bool(_TIME_LABEL_RE.match(_normalize_for_match(text).strip()))


def _is_data_value(text: str) -> bool:
    """这个块是不是「一个数据值」（不是含数字的标题）。

    比 :func:`_is_data_block` 严：``7天预报`` 含数字但只是导航标签，
    换行拼起来会把导航当数据喂给模型。
    """
    unit = (text or "").strip()
    if not unit or len(unit) > 16:
        return False
    if _is_nav_label(unit):
        return False
    if _PURE_NUMBER_RE.match(unit) or _STATE_WORD_RE.match(unit):
        return True
    return bool(_UNIT_VALUE_RE.search(unit)) and bool(_DIGITS_RE.search(unit))


def _is_nav_label(text: str) -> bool:
    """整块是不是一个纯导航标签（``7天预报`` / ``Windows 下载`` / ``省份表``）。

    只认短的、纯名词式的块：``点击这里下载安装包`` 带动作词，是正文句子。
    """
    unit = (text or "").strip()
    if not unit or len(unit) > _MAX_NAV_LABEL_LEN:
        return False
    if _NAV_ACTION_RE.search(unit):
        return False
    return bool(_NAV_FRAGMENT_RE.fullmatch(unit))

# 候选比这个短的多半是导航项 / 标签，直接不算候选
_MIN_FRAGMENT_LEN = 6
# 抽出来的正文短于这个、又没有任何数字/数据词，就当成「只抽到标题/导航」
_MIN_RESULT_CHARS = 120
# 「数据窗口」最多几块：天气/行情这类页面，数值常被拆成一串连续短块
_DATA_WINDOW_SIZE = 8
# 只保留分数 >= 最佳分 × 该比例的候选项：宁可少给，也不给一堆碎片
_MIN_SCORE_RATIO = 0.6
# 无论如何最多回几块（数据簇会额外补几块）
_MAX_FRAGMENTS = 6
# 最多拿几个「关键词命中块」当数据中心，挑数据最多的那一窗
# （命中块常常全是导航标签，只看前几名会集体落空，这里放宽到十几个）
_ANCHOR_CANDIDATES = 12
# 连续数据窗口要比散点结果「干净」这么多（数据块占比）才顶掉散点结果
_WINDOW_GAIN_MARGIN = 0.1
_MAX_CLUSTER = 8
# 带数字 / 单位的块多半是真正的数据行，加分
_CONTENT_BONUS = 1.5
# 符号噪声占比的惩罚系数
_NOISE_PENALTY = 2.0


def _demarkdown(text: str) -> str:
    """去掉 Markdown 语法：图片整个丢，链接只留文字，裸 URL 丢，强调符号丢。

    另外清掉两样在真实页面上很常见、却会直接污染答案的东西：

    - 行首的列表符号（``- only 13℃`` 里的 ``-``），它会让数字变成一个怪字符串；
    - 页面模板漏出来的英文填充词（``最高温 only 13℃`` 的 ``only``）。
    """
    text = _MD_IMAGE_RE.sub(" ", str(text or ""))
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\\", "")
    text = _MD_MARK_RE.sub(" ", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _FILLER_EN_RE.sub(" ", text)
    # 填充词被摘掉后常留下「最高温- 13℃」这种悬空连接符
    text = _DANGLING_DASH_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text) if "\n" not in text else text


def _question_tokens(question: str) -> set:
    """把问题拆成用于打分的「关键词集合」：中文二元组 + 英文/数字词。

    先做同义归一（今天→今日、气温→温度），否则问「今日天气」时
    页面上写的「今天 晴 25℃」一个词都不命中，整张卡片得 0 分。
    """
    question = _normalize_for_match(question)
    compact = "".join(ch for ch in question if ch.isalnum())
    tokens = {compact[i:i + 2] for i in range(len(compact) - 1)}
    tokens.update(w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}", question))
    return {t for t in tokens if t and t not in _TOKEN_NOISE}


def _is_data_block(text: str) -> bool:
    """这个块像不像「数据行」（25℃、湿度 100%、北风2级、阴、112 分）。

    **必须短**：新闻标题里也常出现「13℃」，但那是正文句子，
    早期版本只要撞上单位就当归数据块，于是「北京今晚起有雨，明天最高温
    13℃」被当成天气数据拼进了答案。
    """
    text = (text or "").strip()
    if not text:
        return False
    if _is_data_value(text):
        return True
    if _SENTENCE_PUNCT_RE.search(text):
        # 带逗号/句号的是句子：``北京今晚起有雨，明天最高温 13℃`` 是正文，
        # 不是卡片里的数据格
        return False
    return len(text) <= _MAX_DATA_BLOCK_LEN and bool(_DIGITS_RE.search(text))


def _keep_fragment(unit: str) -> bool:
    """太短的块通常是导航项 / 标签，除非它本身就是数据（「14」「阴」「今天」）。"""
    return (
        len(unit) >= _MIN_FRAGMENT_LEN
        or _is_data_value(unit)
        or _is_time_label(unit)
    )


def _fragments(text: str) -> list:
    """切成候选块：一行按句切（块级换行由 HTML 清洗负责），去重、去太短的。"""
    fragments, seen = [], set()
    for line in str(text or "").split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        for unit in _SENTENCE_SPLIT.split(line):
            unit = " ".join(unit.split()).strip(" ，,、·|-—…")
            if not _keep_fragment(unit) or unit in seen:
                continue
            seen.add(unit)
            fragments.append(unit)
    return fragments


def _noise_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ch in _NOISE_CHARS) / len(text)


def _looks_like_chrome(
    text: str,
    fragments: list = None,
    indexes=None,
) -> bool:
    """抽到的东西是不是「标题/导航壳」而不是正文。

    判据：既没有数字、也没有状态词（晴/涨/胜…），还短。
    真正的内容行总带点数据或够长（``广州今天晴，34度`` 有数字和「晴」，
    不会被误判）。

    ``fragments``/``indexes`` 给的是「这些字是从哪几块拼出来的」：只要其中
    有一块是正常句子（不是纯导航标签），就认为抽到的是正文——
    否则「用 Depends 声明参数即可」这种**短但是正确**的答案会被当成导航，
    退回到整页开头，答案就丢了。
    """
    text = text or ""
    if len(text) >= _MIN_RESULT_CHARS:
        return False
    if _DIGITS_RE.search(text) or _STATE_WORD_ANY_RE.search(text):
        return False
    if fragments is not None and indexes:
        if any(
            _is_content_sentence(fragments[i]) and not _is_nav_label(fragments[i])
            for i in indexes
        ):
            return False
    return True


def _count_data_values(fragments: list, indexes) -> int:
    """这段下标里有多少块是「数据值」。"""
    return sum(1 for index in indexes if _is_data_value(fragments[index]))


def _drop_nav_fragments(fragments: list, indexes) -> list:
    """扔掉纯导航标签块（``7天预报`` / ``24小时预报`` / ``省份表`` / ``温度``）。

    它们含问题里的词、还带数字，很容易被当成「相关内容」拼进结果里；
    但模型读到 ``7天预报`` 只会以为你给了它一个按钮名。全删空时不删
    （宁可给导航，也别给空字符串）。
    """
    kept = [index for index in indexes if not _is_nav_label(fragments[index])]
    return kept or list(indexes)


def _strip_nav_lines(text: str) -> str:
    """整页兜底时把开头那串菜单项去掉（``首页`` ``今日天气`` ``7天预报``…）。

    兜底本来给的就是「原文开头」，而原文开头往往是一排导航按钮；
    模型读到会以为抽到了内容，其实是页面壳。
    """
    lines = [line.strip() for line in str(text or "").split("\n")]
    kept = [line for line in lines if line and not _is_nav_label(line)]
    return "\n".join(kept) or str(text or "")


def _anchor_window(
    fragments: list,
    anchor: int,
    limit: int,
    size: int = _MAX_CLUSTER,
) -> list:
    """以 anchor 为中心，向两侧取**连续**的一段，尽量把数据块全带上。

    为什么必须连续：天气/行情页把一张卡片拆成一串相邻的短 <div>
    （``28`` ``晴`` ``今天17:04更新`` ``湿度 53%`` ``东北风4级``），
    散点拼接会把远处的 ``7天预报`` ``温度`` ``风力`` 这类导航标签也粘进来，
    模型读到的就是一锅碎片。

    两侧的扩展规则：
    - 相邻块是数据值 → 直接带上（数据永远不嫌多）；
    - 相邻块不是数据，但它更外侧还是数据 → 当桥接带上（``今天17:04更新``
      就是这种夹在数字中间的描述行）；
    - 否则停手 —— 再往外就是导航区了，继续扩只会污染结果。
    """
    if not fragments or not 0 <= anchor < len(fragments):
        return []

    start = end = anchor
    used = len(fragments[anchor]) + 1

    def _fits(index: int) -> bool:
        return bool(fragments[index]) and used + len(fragments[index]) + 1 <= limit

    while end - start + 1 < size:
        left, right = start - 1, end + 1
        left_data = left >= 0 and _is_data_value(fragments[left])
        right_data = right < len(fragments) and _is_data_value(fragments[right])
        left_bridge = (
            left > 0
            and not left_data
            and (
                _is_data_value(fragments[left - 1])
                or _is_time_label(fragments[left])
            )
        )
        right_bridge = (
            right < len(fragments) - 1
            and not right_data
            and (
                _is_data_value(fragments[right + 1])
                or _is_time_label(fragments[right])
            )
        )

        # 数据多的一侧优先扩；两侧都不带数据就停
        grow = None
        if right_data and _fits(right):
            grow = right
        elif left_data and _fits(left):
            grow = left
        elif right_bridge and _fits(right):
            grow = right
        elif left_bridge and _fits(left):
            grow = left
        if grow is None:
            break
        if grow < start:
            start = grow
        else:
            end = grow
        used += len(fragments[grow]) + 1

    return list(range(start, end + 1))


def _best_data_window(fragments: list, anchors, limit: int):
    """在若干「关键词命中块」周围各取一段连续窗口，挑数据最多的那段。

    为什么要试多个锚点：最高分那块可能落在导航上（``7天预报省份表``），
    紧挨着它根本没有数据；次高分的 ``今日天气提示`` 才是天气卡片的入口。
    返回 ``(下标列表, 数据块个数, 数据占比)``；占比用来和「散点结果」比谁更干净。

    多个窗口都够干净时取**最靠前**的那个：时效性页面（天气/行情/比分）
    首屏那张卡片就是当前值，再往下才是 24 小时表、7 天表、历史表 ——
    那些表数据更密，却答不了「今天多少度」。
    """
    best, best_count, best_ratio = [], 0, 0.0
    for anchor in anchors:
        window = _anchor_window(fragments, anchor, limit)
        count = _count_data_values(fragments, window)
        if count < 2 or not window:      # 只有一块数据多半是巧合，不顶掉原结果
            continue
        ratio = count / len(window)
        if not best:
            best, best_count, best_ratio = window, count, ratio
            continue
        # 更靠前优先；起点相同时才比数据多、比占比
        if window[0] > best[0]:
            continue
        if window[0] == best[0] and (count, ratio) <= (best_count, best_ratio):
            continue
        best, best_count, best_ratio = window, count, ratio
    return best, best_count, best_ratio


def _densest_data_window(fragments: list, size: int = _DATA_WINDOW_SIZE) -> list:
    """找出「数据块最密集」的连续窗口，返回原文下标列表；没有则返回空。

    为什么需要这一步：问题里的词（``天气``/``预报``）只出现在标题和导航里，
    而真正的答案（``28`` ``晴`` ``湿度 53%``）不含这些词，靠关键词打分永远捞不到。
    所以抽到「标题壳」时，改为把数据最密的那一段带上——这才是模型要的数字。
    """
    if not fragments:
        return []

    best_start, best_count = -1, 0
    for start in range(len(fragments)):
        window = fragments[start:start + size]
        count = sum(1 for unit in window if _is_data_value(unit))
        if count > best_count:
            best_start, best_count = start, count

    # 只有一块「数据」多半是巧合（页脚年份之类），不值得顶掉原结果
    if best_count < 2:
        return []

    window = fragments[best_start:best_start + size]
    data_positions = [
        offset for offset, unit in enumerate(window) if _is_data_value(unit)
    ]
    return list(
        range(
            best_start + data_positions[0],
            best_start + data_positions[-1] + 1,
        )
    )


def _score_fragments(fragments: list, tokens: set) -> list:
    """给每个候选块打分：罕见关键词命中多、带数据、噪声少 → 分高。

    关键词按 idf 加权：``天气`` 在导航里到处都是，权重低；
    ``今日``/``湿度`` 这类只出现在正文里的词权重高 —— 这决定了
    内容行能压过导航碎片。

    比对走同义归一（``_normalize_for_match``）：页面写「今天」、问题问「今日」
    也要算命中，但返回给模型的仍是原文写法。
    """
    if not fragments or not tokens:
        return [0.0] * len(fragments)

    normalized = [_normalize_for_match(fragment) for fragment in fragments]
    total = len(fragments)
    df = Counter()
    for fragment in normalized:
        for token in tokens:
            if token in fragment:
                df[token] += 1

    weights = {
        token: math.log(1.0 + total / (1.0 + df[token]))
        for token in tokens
        if df[token]
    }
    if not weights:
        return [0.0] * len(fragments)

    scores = []
    for fragment, matchable in zip(fragments, normalized):
        hits = [token for token in weights if token in matchable]
        if not hits:
            scores.append(0.0)
            continue
        score = sum(weights[token] for token in hits)
        if _UNIT_HINT_RE.search(fragment):
            score += _CONTENT_BONUS
        score -= _NOISE_PENALTY * _noise_ratio(fragment)
        scores.append(max(score, 0.0))
    return scores


def extract_relevant_text(
    text: str,
    question: str,
    max_chars: int = DEFAULT_EXTRACT_MAX_CHARS,
) -> str:
    """从整页文本里挑出与 question 最相关的几块，拼成不超过 max_chars 的片段。

    一句都匹配不上时**回退成原文开头**：宁可给原文，
    也不要让模型以为抓取失败。
    """
    text = _demarkdown(text).strip()
    if not text:
        return ""

    limit = max(80, int(max_chars or DEFAULT_EXTRACT_MAX_CHARS))
    fragments = _fragments(text)
    if not fragments:
        return _clip(text, limit)

    scores = _score_fragments(fragments, _question_tokens(question))
    best = max(scores)
    if best <= 0:
        # 一个词都没命中：先试「数据最密的一段」（页面上真正的数值区），
        # 再退回原文开头，且把开头的菜单项剔掉。
        window = _densest_data_window(fragments)
        if window:
            return _clip(
                _join_fragments(_drop_nav_fragments(fragments, window), fragments),
                limit,
            )
        return _clip(_strip_nav_lines(text), limit)

    threshold = best * _MIN_SCORE_RATIO
    ranked = sorted(
        ((score, index) for index, score in enumerate(scores) if score >= threshold),
        key=lambda item: (-item[0], item[1]),
    )

    chosen, used = [], 0
    for _, index in ranked[:_MAX_FRAGMENTS]:
        size = len(fragments[index]) + 1
        if chosen and used + size > limit:
            break
        chosen.append(index)
        used += size

    # 命中的那句话旁边往往才是真正的数字（「今日天气提示…」旁边就是
    # 「14 阴 湿度 100%」）。这些块不含问题里的词，单靠打分捞不到，
    # 所以在选中块的两侧把连续的「数据块」一起带上。
    picked = set(chosen)
    added = 0
    for index in sorted(picked):
        for step in (-1, 1):
            cursor = index + step
            while 0 <= cursor < len(fragments):
                if cursor in picked:
                    cursor += step
                    continue
                if added >= _MAX_CLUSTER or not _is_data_block(fragments[cursor]):
                    break
                size = len(fragments[cursor]) + 1
                if used + size > limit:
                    break
                picked.add(cursor)
                used += size
                added += 1
                cursor += step

    picked = sorted(picked)             # 按原文顺序拼，读起来才连贯
    source = _drop_nav_fragments(fragments, picked)
    result = _join_fragments(source, fragments)

    # 最高分那块往往落在导航上（``今日天气提示`` / ``7天预报省份表``）——
    # 问题里的词全在标题和菜单里，真正的答案（``28`` ``晴`` ``湿度 53%``）
    # 一个都不含。这时改用以命中块为中心的一段**连续**数据窗口。
    # 真实事故：问「天气预报」，返回的全是 ``7天预报`` ``24小时预报``，
    # 模型据此判定「没查到天气」又搜一遍，最后打满 max_steps 走兜底答案。
    scatter_count = _count_data_values(fragments, picked)
    scatter_ratio = scatter_count / len(picked) if picked else 0.0
    anchors = [index for _, index in ranked[:_ANCHOR_CANDIDATES]]
    window, window_count, window_ratio = _best_data_window(fragments, anchors, limit)
    # 两条判据任一成立就换：数据更多，或者更「干净」（数据块占比更高）。
    # 散点结果常常是「两块数据 + 一串导航」，占比低但个数不少，所以两个都要看。
    if window and (
        window_count > scatter_count
        or window_ratio >= scatter_ratio + _WINDOW_GAIN_MARGIN
    ):
        source = _drop_nav_fragments(fragments, window)
        result = _join_fragments(source, fragments)

    # 兜底：结果仍然只有标题/导航，就找数据最密的一段，再不行退回正文开头。
    # 「短但确实是正文」（例如「用 Depends 声明参数即可」）不算导航壳，
    # 不会走到这里——否则短答案会被换成整页噪声。
    if _looks_like_chrome(result, fragments, source):
        window = _densest_data_window(fragments)
        if window:
            result = _join_fragments(_drop_nav_fragments(fragments, window), fragments)
        elif len(text) > len(result):
            # 页面上连数据块都找不到：退回正文开头（含首屏内容），
            # 总比把「标题」当结论交回去强
            result = _strip_nav_lines(text)

    return _clip(result, limit)


def _join_fragments(chosen: list, fragments: list) -> str:
    """按原文顺序拼块；相邻的短数据块（「14」「阴」「湿度 100%」）合成一行。

    页面把天气卡片拆成一个个 <div> 时，分开回传模型要自己拼，
    合成一行更接近它理解的「一句话」。
    """
    lines, buffer = [], []
    for index in chosen:
        fragment = fragments[index]
        if len(fragment) <= 12:
            buffer.append(fragment)
            continue
        if buffer:
            lines.append(" ".join(buffer))
            buffer = []
        lines.append(fragment)
    if buffer:
        lines.append(" ".join(buffer))
    return "\n".join(lines)


DEEPSEEK_SEARCH_DEFAULT_BASE = "https://api.deepseek.com"
DEEPSEEK_SEARCH_PATH = "/v1/tools/search"
DEEPSEEK_SEARCH_TIMEOUT = 20.0

# Search API 的硬性约束，写死在代码里避免每次调用被 400 打回
_MAX_QUERY_LEN = 128       # query / ori_question：1~128 个 Unicode 字符
_MIN_COUNT, _MAX_COUNT = 1, 50   # count：1~50


def _read_env(name: str, default: str = "") -> str:
    """读环境变量；当前进程里没有就从项目根目录的 .env 兜一次。

    工具可能被单独 import（脚本、测试），这时 config.settings 还没跑过
    load_dotenv，直接 os.getenv 只会拿到空值。
    """
    value = (os.getenv(name) or "").strip()
    if value:
        return value

    try:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:  # 没装 python-dotenv 或者没有 .env 时静默降级
        return default

    return (os.getenv(name) or default).strip()


def _deepseek_search_endpoint() -> str:
    """拼出 Search 端点，默认 https://api.deepseek.com/v1/tools/search。

    优先 DEEPSEEK_SEARCH_BASE_URL（网关 / 私有部署），
    否则复用模型配置里的 DEEPSEEK_API_BASE（.env 里就是 https://api.deepseek.com）。
    base 后面多写的 /v1、/v1/tools 会被吃掉，避免拼出 /v1/v1/tools/search。
    """
    base = (
        _read_env("DEEPSEEK_SEARCH_BASE_URL")
        or _read_env("DEEPSEEK_API_BASE")
        or DEEPSEEK_SEARCH_DEFAULT_BASE
    ).rstrip("/")

    for suffix in ("/v1/tools", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    return base + DEEPSEEK_SEARCH_PATH


class DeepSeekSearchTool(Tool):
    """DeepSeek 官方搜索工具（Search API）。

    与模型共用同一个 Bearer Key（DEEPSEEK_API_KEY），不用再申请第三方 Key。
    - 端点：POST https://api.deepseek.com/v1/tools/search
    - 计费：每 1000 次「成功且有结果」的请求 5 元；失败 / 无结果不计费
    - 限速：每个账号 5 QPS

    构造函数参数与 ``tools/__init__.py::build_tools`` 传进来的完全对应：
    ``max_results`` 条数、``item_chars`` 单条摘要、``max_chars`` 整次输出上限、
    ``max_calls`` 每任务调用上限（0=不限）、``cache`` 本轮共享缓存。
    """

    name = "web_search"
    description = (
        "Performs a web search and returns the top results. "
        "Input is a single search query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
        item_chars: int = DEFAULT_SEARCH_ITEM_CHARS,
        max_chars: int = DEFAULT_SEARCH_MAX_CHARS,
        max_calls: int = 0,
        cache=None,
        need_content: bool = False,
    ):
        super().__init__()
        # 夹到 1~50：配置里写了非法值也不至于每次调用都 400
        self.max_results = max(_MIN_COUNT, min(int(max_results), _MAX_COUNT))
        # 单条摘要上限 / 整次输出上限（<=0 表示不限制）
        self.item_chars = max(40, int(item_chars or 0))
        self.max_chars = int(max_chars or 0)
        # 每个任务最多调用几次（<=0 表示不限制，由 build_tools 从 settings 注入）
        self.max_calls = int(max_calls or 0)
        # need_content=True 时 API 会回传网页正文（更长的上下文，也更贵在 token 上）
        self.need_content = bool(need_content)
        # 本轮缓存：build_tools 时三个联网工具共享同一个实例
        self.cache = cache if cache is not None else TaskScopedCache()

        # 启动时校验，避免跑起来才暴露问题
        self._api_key = _read_env("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. DeepSeek Search API 复用模型同一个 Key，"
                "请先在 .env 里配置 DEEPSEEK_API_KEY。"
            )

        self._endpoint = _deepseek_search_endpoint()

    # ---- 任务边界 ----

    def begin_task(self):
        """新一轮任务：清空去重缓存与调用计数。"""
        self.cache.begin_task()

    # ---- 调用 ----

    def forward(self, query: str) -> str:
        """按 query 搜索，返回可直接塞进 Observation 的纯文本。

        任何异常都转成以 ``Error: web search failed`` 开头的字符串返回：
        抛异常会变成一步工具执行失败，模型容易原地重试到耗尽 max_steps。
        """
        query = " ".join(str(query or "").split())
        if not query:
            return "Error: web search failed: empty query."

        try:
            return self._search(query)
        except Exception as exc:  # 兜底：解析异常 / 未知异常都不要打断 agent
            return (
                f"Error: web search failed: unexpected {type(exc).__name__}: {exc}. "
                "Try another query, or use visit_webpage on a known URL."
            )

    def _search(self, query: str) -> str:
        key = _search_key(query)

        # 1) 去重：同一 query 本轮第二次不请求网络，直接回缓存
        cached = self.cache.get(key)
        if cached is not None:
            return self._repeat_notice(cached)

        # 2) 硬刹车：换措辞就能绕过去重，所以再加每任务次数上限
        if not self.cache.allow(self.name, self.max_calls):
            return (
                f"Error: 本轮 web_search 调用已达上限（{self.max_calls} 次）。"
            )

        payload = {
            "query": query[:_MAX_QUERY_LEN],
            "count": self.max_results,
            "need_content": self.need_content,
        }

        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=DEEPSEEK_SEARCH_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return (
                "Error: web search failed: search request timed out."
            )
        except requests.exceptions.RequestException as exc:
            return (
                f"Error: web search failed: {exc}. "
                "Check network, or use visit_webpage on a known URL."
            )

        # 状态码按官方文档逐条翻译成人话，模型才知道下一步该干什么
        status = resp.status_code
        if status == 200:
            text = self._format_results(resp)
            if text.startswith("Error:") or text.startswith("No results"):
                return text          # 失败 / 空结果不进缓存，允许重试
            self.cache.put(key, text)
            return text
        if status == 400:
            return f"Error: web search failed: bad request (400): {self._error_message(resp)}"
        if status == 401:
            return "Error: web search failed: invalid DEEPSEEK_API_KEY (401)."
        if status == 402:
            return "Error: web search failed: DeepSeek 账户余额不足 (402)。"
        if status == 408:
            return "Error: web search failed: request timeout on server (408)."
        if status == 429:
            return "Error: web search failed: rate limited (429, 上限 5 QPS)."
        if status >= 500:
            return f"Error: web search failed: server busy ({status})."
        return f"Error: web search failed: HTTP {status}: {self._error_message(resp)}"

    def _repeat_notice(self, cached: str) -> str:
        """重复 query 的回应：明确禁止重复，同时把已有结果再贴一次。

        ``cached`` 是**上一步已经渲染好**的完整结果，每条摘要行尾可能已经带
        ``…（已截断 N 字）``。所以这里按行截，不能用 ``_clip``——
        按字符切会把既有标记切成两半，输出 ``…（已…（已截断 397 字）``。

        截断预算跟着 ``max_chars`` 走，不再额外砍一道：``cached`` 本来就是
        ``_format_results`` 的产物，已经受过条数和长度约束。原来在这里写死
        800 等于把刚拿到的摘要又削掉一截（实测第 3 条摘要整段变成
        「（重复的结果还有 2 行未显示）」），模型反而得再搜一次。
        """
        return "（缓存：本轮已搜过同一个 query，未重新请求网络）\n" + _clip_lines(
            cached, self.max_chars
        )

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        """尽量把服务端给的错误描述捞出来，捞不到就退回原始文本片段。"""
        try:
            data = resp.json()
        except ValueError:
            return (resp.text or "").strip()[:200] or "(no body)"

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)[:200]
            if error:
                return str(error)[:200]
            if data.get("message"):
                return str(data["message"])[:200]
        return str(data)[:200]

    def _format_results(self, resp: requests.Response) -> str:
        """把 results 数组渲染成紧凑文本：标题 / 链接 / 站点·时间·评分 / 摘要。

        两层预算一起生效：条数（``max_results``）先砍，字符数（``max_chars``）
        再砍，被砍掉的条数都会写在末尾，避免模型以为「搜索结果就这么点」。
        """
        try:
            data = resp.json()
        except ValueError:
            return "Error: web search failed: response is not valid JSON."

        results = data.get("results") or []
        if not results:
            return "No results found."

        total = len(results)
        shown = results[: self.max_results]
        # 只报事实。行为要求（先读摘要、别抄元信息、别重复搜）统一写在
        # config/prompts.yaml 的 web 规则里：夹在这里既每次调用重复一遍浪费
        # 上下文，又会被小模型当成正文抄进答案（「本轮已经搜过这个 query」
        # 就是这么漏出去的）。
        header = f"搜索结果 {total} 条：\n"

        blocks, used, omitted = [], len(header), 0
        for index, item in enumerate(shown, start=1):
            block = self._render_item(item, index)
            # 第一条永远要给（否则模型以为搜索失败），之后受长度预算约束
            if self.max_chars > 0 and blocks and used + len(block) + 1 > self.max_chars:
                omitted = len(shown) - index + 1
                break
            blocks.append(block)
            used += len(block) + 1

        out = header + "\n".join(blocks)
        if omitted:
            out += f"\n已省略 {omitted} 条（超出长度预算 {self.max_chars} 字）。"
        return out

    def _render_item(self, item: dict, index: int = 0) -> str:
        """渲染单条结果；摘要 / 正文都按 item_chars 截断并标出截断量。

        每条用 ``1.`` / ``2.`` 编号开头、字段带中文标签，模型才能分清
        「哪句是正文」「哪句是链接和元信息」。``rerank_score`` 刻意不渲染：
        它对本机模型没有可用的语义（三条结果常常同分），只会漏进答案。
        """
        title = " ".join(str(item.get("title") or "").split()) or "(no title)"
        url = str(item.get("url") or "").strip()

        # summary（600~800 字符，官方推荐给模型读）优先，退而用 snippet（100~200）
        summary = " ".join(
            str(item.get("summary") or item.get("snippet") or "").split()
        )

        meta = []
        if item.get("site_name"):
            meta.append(str(item["site_name"]))
        if item.get("date"):
            meta.append(str(item["date"])[:19].replace("T", " "))

        head = f"{index}. {title}" if index else f"- {title}"
        block = [head]
        if url:
            block.append(f"   链接：{url}")
        if meta:
            block.append("   " + " · ".join(meta))
        if summary:
            block.append(f"   摘要：{_clip(summary, self.item_chars)}")

        content = " ".join(str(item.get("content") or "").split())
        if content:  # need_content=False 时恒为空
            block.append(f"   正文：{_clip(content, self.item_chars * 5)}")

        return "\n".join(block)


def WebSearchTool(
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    item_chars: int = DEFAULT_SEARCH_ITEM_CHARS,
    max_chars: int = DEFAULT_SEARCH_MAX_CHARS,
    max_calls: int = 0,
    cache=None,
    need_content: bool = False,
) -> DeepSeekSearchTool:
    """构造 web_search 工具（DeepSeek 官方 Search API）。

    参数与 ``DeepSeekSearchTool.__init__`` 完全一致，``tools/__init__.py``
    直接按名字传参即可，不存在「定义了一个、传了另一个」的错位。
    """
    return DeepSeekSearchTool(
        max_results=max_results,
        item_chars=item_chars,
        max_chars=max_chars,
        max_calls=max_calls,
        cache=cache,
        need_content=need_content,
    )


class VisitWebpageTool(Tool):
    """打开一个 URL，把 HTML 正文转成 Markdown 返回。

    只适合抓取静态页面；需要登录或重前端渲染的页面可能拿不到内容。
    同一 URL 本轮第二次调用直接回缓存（只标「未重新请求网络」，不给行为建议）。
    """

    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a "
        "markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        max_chars: int = MAX_CONTENT_LENGTH,
        max_calls: int = 0,
        cache=None,
    ):
        super().__init__()
        self.max_chars = max(200, int(max_chars or MAX_CONTENT_LENGTH))
        self.max_calls = int(max_calls or 0)
        self.cache = cache if cache is not None else TaskScopedCache()

    def begin_task(self):
        self.cache.begin_task()

    def forward(self, url: str) -> str:
        url = str(url or "").strip()

        error = _validate_url(url)
        if error:
            return error

        # 1) 去重：同一 URL 本轮已经抓过就不重复请求网络
        cached = self.cache.get(_page_key(url))
        if cached is not None:
            return "（缓存：本轮已抓过这个 URL，未重新请求网络）\n\n" + _clip(
                _page_markdown(cached), min(self.max_chars, 1200)
            )

        # 2) 硬刹车：每任务最多抓几次
        if not self.cache.allow(self.name, self.max_calls):
            return f"Error: 本轮 visit_webpage 调用已达上限（{self.max_calls} 次）。"

        payload, error = _fetch_page(url)
        if error:
            return error

        self.cache.put(_page_key(url), payload)
        return _clip(payload["markdown"], self.max_chars)


class ExtractTool(Tool):
    """抓一个 URL，但只回传与问题相关的行。

    搜索结果摘要不够时优先用它：比 ``visit_webpage`` 省一个数量级的上下文。
    与 ``visit_webpage`` 共享页面缓存，同一页换个问题只重做抽取、不重抓。

    第一个参数**也可以是一段文本**（例如 ``web_search`` 的整段输出）：
    这时不联网，直接在这段文本里抽相关行。真实事故：llama3.2 写了
    ``extract(result, query)``（result 是搜索结果），旧实现报 url 格式错误、
    还把 1800 字入参原样回显，模型随后连 4 步在 print URL 里打转，
    最后交出一个链接当答案。模型的直觉没错，是工具太窄。
    """

    name = "extract"
    description = (
        "Return ONLY the lines relevant to the question. The first argument is "
        "either a webpage url to fetch, or a block of text you already have "
        "(such as the output of web_search) to pick lines out of — in that case "
        "no network request is made. Cheaper than visit_webpage when you need one "
        "or two facts."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": (
                "A webpage url (http:// or https://), or the text to extract "
                "from when you do not want to fetch anything."
            ),
        },
        "question": {
            "type": "string",
            "description": "The specific question you want this page to answer.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        max_chars: int = DEFAULT_EXTRACT_MAX_CHARS,
        max_calls: int = 0,
        cache=None,
    ):
        super().__init__()
        self.max_chars = max(200, int(max_chars or DEFAULT_EXTRACT_MAX_CHARS))
        self.max_calls = int(max_calls or 0)
        self.cache = cache if cache is not None else TaskScopedCache()

    def begin_task(self):
        self.cache.begin_task()

    def forward(self, url: str, question: str) -> str:
        first_arg = str(url or "").strip()
        question = str(question or "").strip()

        # question 是抽取的依据，缺了就退化成 visit_webpage，不如直接报错
        if not question:
            return (
                "Error: extract 缺少 question 参数（要从这一页里找什么问题），"
                '例如 extract("https://www.weather.com.cn/...", "广州今天气温多少度")。'
            )

        # 文本模式：第一个参数不是网页地址 → 当成「已经拿到的文本」直接抽取。
        # 不联网，也就不占用每任务的调用次数（那是给网络请求踩刹车的）。
        if not _looks_like_url(first_arg):
            return self._extract_from_text(first_arg, question)

        error = _validate_url(first_arg)
        if error:
            return error

        if not self.cache.allow(self.name, self.max_calls):
            return f"Error: 本轮 extract 调用已达上限（{self.max_calls} 次）。"

        payload = self.cache.get(_page_key(first_arg))
        reused = payload is not None
        if not reused:
            payload, error = _fetch_page(first_arg)
            if error:
                return error
            self.cache.put(_page_key(first_arg), payload)   # 与 visit_webpage 共享

        raw_text = _page_text(payload)
        if _looks_like_page_skeleton(raw_text):
            # 只陈述事实（「抽不出东西」），不再附带「别在这个站上试」这类
            # 行为建议——那是 prompts.yaml 规则 12 的职责。
            return _page_error(
                "页面只有导航骨架，正文由 JavaScript 渲染，静态请求拿不到。"
            )

        snippet = extract_relevant_text(raw_text, question, self.max_chars)
        if not snippet:
            return "Error: 该页面没有可读正文。"

        source = (
            "（缓存：本轮已抓过该页面，未重新请求网络）\n从页面抽取的相关片段：\n"
            if reused
            else "从页面抽取的相关片段：\n"
        )
        return source + snippet

    def _extract_from_text(self, text: str, question: str) -> str:
        """在「已经拿到的文本」里抽相关行（不联网）。

        典型输入是 ``web_search`` 的整段输出：模型手里已经有 3 条摘要，
        只是想挑出与问题相关的那几行，这时候根本不需要再抓页面。
        """
        if not text:
            return (
                "Error: extract 的第一个参数为空（应为 http(s) 地址，或一段待抽取的文本）。"
            )
        if len(text) < _MIN_TEXT_MODE_CHARS:
            return (
                f"Error: 第一参数既不是网页地址（需以 http:// 或 https:// 开头），"
                f"也不是可抽取的文本（只有 {len(text)} 字）。"
            )

        snippet = extract_relevant_text(text, question, self.max_chars)
        if not snippet:
            return "Error: 这段文本里没有可抽取的内容（可能是空的或全是符号）。"
        return "从你传入的文本里抽取的相关片段（未联网）：\n" + snippet


__all__ = [
    "TaskScopedCache",
    "WebSearchTool",
    "VisitWebpageTool",
    "ExtractTool",
    "DeepSeekSearchTool",
    "extract_relevant_text",
    "MAX_CONTENT_LENGTH",
]
