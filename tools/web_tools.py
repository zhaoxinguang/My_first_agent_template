"""联网工具：网页搜索、网页正文抓取、按问题抽取。

- ``web_search``：DeepSeek 官方 Search API（``/v1/tools/search``），Bearer 鉴权；
- ``extract``：抓页面，但**只回传与问题相关的行**（比整页便宜一个数量级）；
- ``visit_webpage``：抓整页正文并转 Markdown。

三个工具都返回「字符串」，可以直接被 CodeAgent 当作 Python 函数调用，
调用结果会出现在下一步的 Observation 里。

三者共享同一份 ``TaskScopedCache``（由 ``tools/__init__.py::build_tools`` 注入）：
- 同一 query / 同一 url 本轮第二次调用**不再打网络**，直接回缓存 + 「别重复」提示；
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
4. 只留分数 ≥ 最佳分 60% 的块，最多 6 块 —— 宁可少给，也不给一堆导航碎片。
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


def _validate_url(url: str) -> str:
    """返回错误字符串；URL 合法时返回空字符串。"""
    if not url:
        return "Error: url 为空，请传入要抓取的完整 http(s) 地址。"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return (
            f"Error: url 必须以 http:// 或 https:// 开头（当前值：{url!r}），"
            "例如 https://www.weather.com.cn/。"
        )
    return ""


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
        return None, "The request timed out. Please try again later or check the URL."
    except requests.exceptions.RequestException as exc:
        return None, f"Error fetching the webpage: {exc}"

    _fix_encoding(response)
    html = response.text or ""

    # extract 用：HTML 直接转可读纯文本（不带 markdown 链接语法）
    readable = _readable_text(html)

    try:
        markdown_content = markdownify(html).strip()
    except Exception as exc:  # HTML 结构异常时不要让整个 agent 崩掉
        return None, f"An unexpected error occurred while parsing the page: {exc}"

    # 压缩连续空行，减少无意义的 token
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

    # 内容有效性校验，避免把导航壳页当结果喂给模型
    if len(markdown_content) < _MIN_CONTENT_LEN and len(readable) < _MIN_CONTENT_LEN:
        return None, (
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

_SENTENCE_SPLIT = re.compile(r"[。！？；!?;]+")
_UNIT_HINT_RE = re.compile(r"\d|℃|度|%|级|hPa|百帕|mm|km")
_NOISE_CHARS = set("[](){}<>|\\/*#_~`•…·=+")

_TOKEN_NOISE = set("的了和与及或是在有也都就而请帮我查一下吗呢吧啊这个那个如何什么多少怎么")

# 「数据块」特征：气温/湿度/风力这类真正的天气数据，往往是一个短 <div>，
# 本身不含问题里的词（问题问「今日天气」，页面上写的是「14 阴 湿度 100%」）。
_DATA_HINT_RE = re.compile(
    r"℃|度|%|级|hPa|百帕|mm|km|湿度|气温|温度|风力|风向|气压|能见度|紫外线|"
    r"空气质量|PM2\.5|日出|日落|AQI|阴|晴|多云|雷|雨|雪|雾|霾"
)
_PURE_NUMBER_RE = re.compile(r"^[-+]?\d[\d.,]*\s*(?:%|℃|度|级|km|mm|hPa)?$")
_DIGITS_RE = re.compile(r"[\d]")

# 候选比这个短的多半是导航项 / 标签，直接不算候选
_MIN_FRAGMENT_LEN = 6
# 只保留分数 >= 最佳分 × 该比例的候选项：宁可少给，也不给一堆碎片
_MIN_SCORE_RATIO = 0.6
# 无论如何最多回几块（数据簇会额外补几块）
_MAX_FRAGMENTS = 6
_MAX_CLUSTER = 8
# 带数字 / 单位的块多半是真正的数据行，加分
_CONTENT_BONUS = 1.5
# 符号噪声占比的惩罚系数
_NOISE_PENALTY = 2.0


def _demarkdown(text: str) -> str:
    """去掉 Markdown 语法：图片整个丢，链接只留文字，裸 URL 丢，强调符号丢。"""
    text = _MD_IMAGE_RE.sub(" ", str(text or ""))
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\\", "")
    return _MD_MARK_RE.sub(" ", text)


def _question_tokens(question: str) -> set:
    """把问题拆成用于打分的「关键词集合」：中文二元组 + 英文/数字词。"""
    compact = "".join(ch for ch in (question or "") if ch.isalnum())
    tokens = {compact[i:i + 2] for i in range(len(compact) - 1)}
    tokens.update(w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}", question or ""))
    return {t for t in tokens if t and t not in _TOKEN_NOISE}


def _is_data_block(text: str) -> bool:
    """这个块像不像「数据行」（气温 14、湿度 100%、北风2级、阴）。"""
    text = text or ""
    if _DATA_HINT_RE.search(text) or _PURE_NUMBER_RE.match(text):
        return True
    # 「今天7:53更新」这种短块也算：数据常被拆成一串短 <div>
    return len(text) <= 8 and bool(_DIGITS_RE.search(text))


def _keep_fragment(unit: str) -> bool:
    """太短的块通常是导航项 / 标签，除非它本身就是数据（「14」「阴」）。"""
    return (
        len(unit) >= _MIN_FRAGMENT_LEN
        or _DATA_HINT_RE.search(unit) is not None
        or _PURE_NUMBER_RE.match(unit) is not None
        or (len(unit) <= 8 and bool(_DIGITS_RE.search(unit)))
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


def _score_fragments(fragments: list, tokens: set) -> list:
    """给每个候选块打分：罕见关键词命中多、带数据、噪声少 → 分高。

    关键词按 idf 加权：``天气`` 在导航里到处都是，权重低；
    ``今日``/``湿度`` 这类只出现在正文里的词权重高 —— 这决定了
    内容行能压过导航碎片。
    """
    if not fragments or not tokens:
        return [0.0] * len(fragments)

    total = len(fragments)
    df = Counter()
    for fragment in fragments:
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
    for fragment in fragments:
        hits = [token for token in weights if token in fragment]
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
    if best <= 0:                       # 没有任何相关块：退回原文开头
        return _clip(text, limit)

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
    return _clip(_join_fragments(picked, fragments), limit)


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


# =============================================================================
# 旧实现（Tavily）：已停用，按要求整段注释保留，方便对照 / 回滚。
# 停用原因：改用 DeepSeek 官方 Search API，与模型共用同一个 API Key，
#           不用再单独维护第三方密钥和免费额度。
#
# class TavilySearchTool(Tool):
#     """Tavily 搜索工具（Bearer Token 认证，替代 DDGS）。
#
#     环境变量：TAVILY_API_KEY（格式 tvly-xxxx）
#     免费额度：1000 credits/月，无需信用卡
#     """
#
#     name = "web_search"
#     description = (
#         "Performs a web search and returns the top results. "
#         "Input is a single search query."
#     )
#     inputs = {
#         "query": {
#             "type": "string",
#             "description": "The search query to perform.",
#         }
#     }
#     output_type = "string"
#
#     _BASE_URL = "https://api.tavily.com/search"
#
#     def __init__(self, max_results: int = 10):
#         super().__init__()
#         self.max_results = max_results
#         # 启动时校验，避免运行时才暴露问题
#         self._api_key = os.getenv("TAVILY_API_KEY")
#         if not self._api_key:
#             raise RuntimeError(
#                 "TAVILY_API_KEY not set. "
#                 "Get a free key at https://app.tavily.com (1000 credits/month, no credit card)."
#             )
#
#     def forward(self, query: str) -> str:
#         try:
#             resp = requests.post(
#                 self._BASE_URL,
#                 json={
#                     "query": query,
#                     "max_results": self.max_results,
#                     "search_depth": "basic",  # basic 消耗 1 credit；advanced 消耗 2
#                 },
#                 headers={
#                     "Authorization": f"Bearer {self._api_key}",
#                     "Content-Type": "application/json",
#                 },
#                 timeout=20,
#             )
#         except requests.exceptions.Timeout:
#             return "Search request timed out. Please retry later."
#         except requests.exceptions.RequestException as exc:
#             return f"Search request failed: {exc}"
#
#         if resp.status_code == 401:
#             return "Search authentication failed: invalid TAVILY_API_KEY."
#         if resp.status_code == 429:
#             return "Search rate-limited (429). Retry later or upgrade quota."
#
#         try:
#             resp.raise_for_status()
#         except requests.exceptions.HTTPError as exc:
#             return f"Search HTTP error: {exc}"
#
#         data = resp.json()
#         results = data.get("results", [])
#         if not results:
#             return "No results found."
#
#         lines = []
#         for r in results:
#             title = r.get("title", "")
#             url = r.get("url", "")
#             content = (r.get("content") or "")[:200]
#             lines.append(f"- {title}\n  {url}\n  {content}")
#         return "\n".join(lines)
#
#
# def WebSearchTool(max_results: int = 10):
#     """构造 web_search 工具（Tavily Bearer 认证）。"""
#     return TavilySearchTool(max_results=max_results)
# =============================================================================


# =============================================================================
# 新增实现：DeepSeek 官方搜索接口（Search API，自研搜索引擎）
#   文档：https://api-docs.deepseek.com/zh-cn/api/search
#   接口名保持不变：函数 WebSearchTool()，工具名 web_search，入参只有 query
# =============================================================================

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
                f"Error: web search failed: 本轮 web_search 调用已达到上限"
                f"（{self.max_calls} 次），请用已拿到的结果作答，不要再搜索。"
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
                "Error: web search failed: search request timed out. "
                "Retry later or use visit_webpage on a known URL."
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
            return "Error: web search failed: DeepSeek 账户余额不足 (402)，充值后重试。"
        if status == 408:
            return "Error: web search failed: request timeout on server (408). Retry later."
        if status == 429:
            return "Error: web search failed: rate limited (429, 上限 5 QPS). Retry later."
        if status >= 500:
            return f"Error: web search failed: server busy ({status}). Retry later."
        return f"Error: web search failed: HTTP {status}: {self._error_message(resp)}"

    def _repeat_notice(self, cached: str) -> str:
        """重复 query 的回应：明确禁止重复，同时把已有结果再贴一次。"""
        return (
            "本轮已经搜过这个 query，重复调用不会再请求网络。"
            "不要再用同样的 query 搜索，直接使用下面的结果作答；\n"
            "如果结果确实不够，请换更具体的关键词，或改用 extract 抓具体页面。\n\n"
            + _clip(cached, 800)
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
        header = (
            f"DeepSeek 搜索返回 {total} 条结果"
            f"（本次最多展示 {min(self.max_results, total)} 条）：\n"
        )

        blocks, used, omitted = [], len(header), 0
        for index, item in enumerate(shown):
            block = self._render_item(item)
            # 第一条永远要给（否则模型以为搜索失败），之后受长度预算约束
            if self.max_chars > 0 and blocks and used + len(block) + 1 > self.max_chars:
                omitted = len(shown) - index
                break
            blocks.append(block)
            used += len(block) + 1

        out = header + "\n".join(blocks)
        if omitted:
            out += (
                f"\n已省略 {omitted} 条（超出本次长度预算 {self.max_chars} 字），"
                "需要更多请换更精确的 query。"
            )
        return out

    def _render_item(self, item: dict) -> str:
        """渲染单条结果；摘要 / 正文都按 item_chars 截断并标出截断量。"""
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
        score = item.get("rerank_score")
        if isinstance(score, (int, float)):
            meta.append(f"score {round(float(score), 2)}")

        block = [f"- {title}", f"  {url}"]
        if meta:
            block.append("  " + " · ".join(meta))
        if summary:
            block.append(f"  {_clip(summary, self.item_chars)}")

        content = " ".join(str(item.get("content") or "").split())
        if content:  # need_content=False 时恒为空
            block.append(f"  {_clip(content, self.item_chars * 5)}")

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
    同一 URL 本轮第二次调用直接回缓存，并引导模型改用 extract。
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
            return (
                "本轮已经抓过这个 URL，正文见下，不要重复抓取。"
                "只需要其中一两个事实时改用 extract(url, question)，"
                "它只回相关行，比整页省一个数量级的上下文。\n\n"
                + _clip(_page_markdown(cached), min(self.max_chars, 1200))
            )

        # 2) 硬刹车：每任务最多抓几次
        if not self.cache.allow(self.name, self.max_calls):
            return (
                f"本轮 visit_webpage 调用已达到上限（{self.max_calls} 次），"
                "请用已经拿到的内容作答，不要再抓新页面。"
            )

        payload, error = _fetch_page(url)
        if error:
            return error

        self.cache.put(_page_key(url), payload)
        return _clip(payload["markdown"], self.max_chars)


class ExtractTool(Tool):
    """抓一个 URL，但只回传与问题相关的行。

    搜索结果摘要不够时优先用它：比 ``visit_webpage`` 省一个数量级的上下文。
    与 ``visit_webpage`` 共享页面缓存，同一页换个问题只重做抽取、不重抓。
    """

    name = "extract"
    description = (
        "Fetch a webpage and return ONLY the lines relevant to the question. "
        "Cheaper than visit_webpage when you need one or two facts."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to read.",
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
        url = str(url or "").strip()
        question = str(question or "").strip()

        # question 是抽取的依据，缺了就退化成 visit_webpage，不如直接报错
        if not question:
            return (
                "Error: extract 需要 question 参数：请传入你希望这一页回答的具体问题，"
                "例如 extract(\"https://www.weather.com.cn/...\", \"广州今天气温多少度\")。"
            )

        error = _validate_url(url)
        if error:
            return error

        if not self.cache.allow(self.name, self.max_calls):
            return (
                f"本轮 extract 调用已达到上限（{self.max_calls} 次），"
                "请用已经拿到的内容作答，不要再抓新页面。"
            )

        payload = self.cache.get(_page_key(url))
        reused = payload is not None
        if not reused:
            payload, error = _fetch_page(url)
            if error:
                return error
            self.cache.put(_page_key(url), payload)   # 与 visit_webpage 共享

        snippet = extract_relevant_text(_page_text(payload), question, self.max_chars)
        if not snippet:
            return "Error: 该页面没有可读正文，换一个 URL 或用 visit_webpage 试试。"

        source = (
            "该页面本轮已抓取过，下面是从缓存里抽取的相关片段（没有重复请求网络）：\n"
            if reused
            else "从页面抽取的相关片段（只回与问题相关的行）：\n"
        )
        return source + snippet


__all__ = [
    "TaskScopedCache",
    "WebSearchTool",
    "VisitWebpageTool",
    "ExtractTool",
    "DeepSeekSearchTool",
    "extract_relevant_text",
    "MAX_CONTENT_LENGTH",
]
