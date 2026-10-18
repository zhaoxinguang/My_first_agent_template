"""工具集的唯一入口。

外部只需要 ``from tools import build_tools`` 就能拿到 agent 的全部工具，
不需要知道每个工具具体放在哪个文件里。

工具分三类：
- 基础/文件：本地能力，无网络依赖（basic_tools、file_tools）
- 联网：搜索与抓取（web_tools）
- 协议：final_answer，agent 用它声明任务结束

每个工具都是 smolagents 的 Tool 或 @tool 函数，
在 CodeAgent 眼里就是「一个可以直接调用的 Python 函数」。
"""

from tools.basic_tools import get_current_time_in_timezone
from tools.file_tools import list_files, read_text_file
from tools.final_answer import FinalAnswerTool
from tools.web_tools import VisitWebpageTool, create_web_search_tool

__all__ = [
    "build_tools",
    "get_tool",
    "describe_tools",
    "TOOL_GUIDE",
    "get_current_time_in_timezone",
    "read_text_file",
    "list_files",
    "VisitWebpageTool",
    "FinalAnswerTool",
    "create_web_search_tool",
]


# 用途 + 调用方式说明。写给「读代码的人」，也是给 agent 的行为预期。
TOOL_GUIDE = {
    "final_answer": {
        "purpose": "提交最终答案并结束当前任务，必须是最后一步。",
        "usage": "final_answer(answer)",
        "example": 'final_answer("北京现在时间是 14:03")',
    },
    "get_current_time_in_timezone": {
        "purpose": "查询指定时区的当前时间，避免让模型自己算日期。",
        "usage": "get_current_time_in_timezone(timezone)",
        "example": 'get_current_time_in_timezone("Asia/Shanghai")',
    },
    "list_files": {
        "purpose": "列出 documents/ 目录下可读的 .txt 文件，读文件前先调用它。",
        "usage": "list_files()",
        "example": "list_files()",
    },
    "read_text_file": {
        "purpose": "读取 documents/ 下某个 .txt 文件，支持按行截取。",
        "usage": "read_text_file(filename, start_line=1, end_line=-1)",
        "example": 'read_text_file("1.txt", start_line=1, end_line=20)',
    },
    "web_search": {
        "purpose": "DuckDuckGo 搜索，返回标题、链接和摘要，用于找入口 URL。",
        "usage": "web_search(query)",
        "example": 'web_search("smolagents CodeAgent 文档")',
    },
    "visit_webpage": {
        "purpose": "抓取单个网页正文并转成 Markdown（已截断到约 1 万字符）。",
        "usage": "visit_webpage(url)",
        "example": 'visit_webpage("https://huggingface.co/docs/smolagents")',
    },
}


def build_tools(include_web: bool = True):
    """构造 agent 使用的工具列表。

    Args:
        include_web: 是否包含联网工具。离线调试或没装
            duckduckgo_search 时传 False。
    """
    tools = [
        get_current_time_in_timezone,
        read_text_file,
        list_files,
    ]

    if include_web:
        try:
            tools.append(create_web_search_tool())
        except ImportError as exc:
            print(f"[tools] 跳过 web_search：{exc}")
        try:
            tools.append(VisitWebpageTool())
        except Exception as exc:  # 工具初始化失败不应该拖垮整个 agent
            print(f"[tools] 跳过 visit_webpage：{exc}")

    tools.append(FinalAnswerTool())
    return tools


def get_tool(name: str, include_web: bool = True):
    """按名字取单个工具，找不到返回 None。"""
    for item in build_tools(include_web=include_web):
        if getattr(item, "name", None) == name:
            return item
    return None


def describe_tools() -> str:
    """生成「用途 + 调用方式」的文本清单，可直接 print。"""
    lines = ["可用工具清单："]
    for name, info in TOOL_GUIDE.items():
        lines.append(
            f"\n- {name}\n"
            f"  用途：{info['purpose']}\n"
            f"  调用：{info['usage']}\n"
            f"  示例：{info['example']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_tools())
