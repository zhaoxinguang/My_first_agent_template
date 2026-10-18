"""联网工具：网页搜索与网页正文抓取。

两个工具都返回「字符串」，可以直接被 CodeAgent 当作 Python 函数调用，
调用结果会出现在下一步的 Observation 里。
"""

import re

import requests
from markdownify import markdownify
from smolagents import DuckDuckGoSearchTool
from smolagents.tools import Tool
from smolagents.utils import truncate_content

# 网页正文最多回传给模型多少字符（太长会吃掉上下文预算）
MAX_CONTENT_LENGTH = 10000
REQUEST_TIMEOUT = 20.0


def create_web_search_tool(max_results: int = 10):
    """构造 DuckDuckGo 搜索工具。

    直接复用 smolagents 内置实现，避免维护一份重复代码。
    返回的 tool 名字是 ``web_search``，入参只有 ``query``。
    """
    return DuckDuckGoSearchTool(max_results=max_results)


class VisitWebpageTool(Tool):
    """打开一个 URL，把 HTML 正文转成 Markdown 返回。

    只适合抓取静态页面；需要登录或重前端渲染的页面可能拿不到内容。
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

    def forward(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (compatible; smolagent)"},
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except requests.exceptions.RequestException as exc:
            return f"Error fetching the webpage: {exc}"

        try:
            markdown_content = markdownify(response.text).strip()
        except Exception as exc:  # HTML 结构异常时不要让整个 agent 崩掉
            return f"An unexpected error occurred while parsing the page: {exc}"

        # 压缩连续空行，减少无意义的 token
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return truncate_content(markdown_content, MAX_CONTENT_LENGTH)


__all__ = [
    "VisitWebpageTool",
    "create_web_search_tool",
    "MAX_CONTENT_LENGTH",
]
