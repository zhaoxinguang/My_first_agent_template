"""工具集的唯一入口。

外部只需要 ``from tools import build_tools`` 就能拿到 agent 的全部工具，
不需要知道每个工具具体放在哪个文件里。

工具分四类：
- 基础/文件：本地能力，无网络依赖（basic_tools、file_tools）
- 长期偏好：按需加载 + 动态更新 + 遗忘（memory_tools）
- 联网：搜索与抓取（web_tools）
- 协议：final_answer，agent 用它声明任务结束

每个工具都是 smolagents 的 Tool 或 @tool 函数，
在 CodeAgent 眼里就是「一个可以直接调用的 Python 函数」。
"""

from typing import Any

from config.settings import (
    WEB_EXTRACT_MAX_CALLS_PER_TASK,
    WEB_EXTRACT_MAX_CHARS,
    WEB_PAGE_MAX_CALLS_PER_TASK,
    WEB_PAGE_MAX_CHARS,
    WEB_SEARCH_ITEM_CHARS,
    WEB_SEARCH_MAX_CALLS_PER_TASK,
    WEB_SEARCH_MAX_CHARS,
    WEB_SEARCH_MAX_RESULTS,
)
from tools.basic_tools import get_current_time_in_timezone
from tools.file_tools import list_files, read_text_file
from tools.final_answer import FinalAnswerTool
from tools.memory_tools import (
    ForgetPreferenceTool,
    LoadPreferencesTool,
    SavePreferenceTool,
)

# 联网工具是可选的：它依赖 requests / markdownify。
# 这里必须容错——否则「没装这两个包」会让 ``import tools`` 整个失败，
# 连带把本地工具和偏好工具一起带走（include_web=False 也就成了空话）。
try:
    from tools.web_tools import (
        ExtractTool,
        TaskScopedCache,
        VisitWebpageTool,
        WebSearchTool,
    )
except ImportError as exc:  # pragma: no cover - 取决于运行环境是否装依赖
    _WEB_IMPORT_ERROR = exc
    ExtractTool = None
    TaskScopedCache = None
    VisitWebpageTool = None
    WebSearchTool = None
else:
    _WEB_IMPORT_ERROR = None

_WEB_NAMES = ("VisitWebpageTool", "WebSearchTool", "ExtractTool", "TaskScopedCache")

__all__ = [
    "build_tools",
    "get_tool",
    "describe_tools",
    "TOOL_GUIDE",
    "get_current_time_in_timezone",
    "read_text_file",
    "list_files",
    "LoadPreferencesTool",
    "SavePreferenceTool",
    "ForgetPreferenceTool",
    "FinalAnswerTool",
]

# 联网工具没装上依赖时不导出：把 None 挂进 __all__ 只会让
# ``from tools import WebSearchTool`` 拿到一个 None，比 ImportError 更难查。
if _WEB_IMPORT_ERROR is None:
    __all__ += list(_WEB_NAMES)


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
        "purpose": (
            "DeepSeek 官方搜索接口（/v1/tools/search），返回标题、链接、摘要和评分，"
            "用于找入口 URL。只回前几条（默认 3 条、总计 ≤1800 字符）；"
            "同一个 query 本轮搜第二次会直接回缓存，不要重复搜。"
            "失败时返回 Error 字符串（不抛异常），此时改 query 或换 extract。"
        ),
        "usage": "web_search(query)",
        "example": 'web_search("smolagents CodeAgent 文档")',
    },
    "extract": {
        "purpose": (
            "抓一个 URL，但只回传与问题相关的句子（≤1200 字符）。"
            "搜索结果摘要不够时优先用它，比 visit_webpage 省一个数量级的上下文。"
        ),
        "usage": "extract(url, question)",
        "example": 'extract("https://www.weather.com.cn/...", "广州今天气温多少度")',
    },
    "visit_webpage": {
        "purpose": (
            "抓取单个网页全文并转成 Markdown（已截断到约 6000 字符）。"
            "只在确实需要通读整页时用；取一两个事实请用 extract。"
        ),
        "usage": "visit_webpage(url)",
        "example": 'visit_webpage("https://huggingface.co/docs/smolagents")',
    },
    "load_user_preferences": {
        "purpose": (
            "按需加载用户偏好的完整内容。prompt 里只给偏好目录（要点），"
            "要点不足以判断具体要求时用它取全文；传空字符串列出全部。"
        ),
        "usage": 'load_user_preferences(query="response_style")',
        "example": 'load_user_preferences("回答风格")',
    },
    "save_user_preference": {
        "purpose": (
            "新增或覆盖一条用户长期偏好（同 key 再次写入即覆盖）。"
            "没有自动提炼环节：是否写入由模型在任务收尾时自行判断，"
            "只记下一轮仍然成立的偏好；预算已满时返回 Error，需要先删再写。"
        ),
        "usage": 'save_user_preference(key, value, category="general")',
        "example": 'save_user_preference("response_style", "回答简洁，先给可运行代码")',
    },
    "forget_user_preference": {
        "purpose": "删除一条过时或错误的用户偏好；预算满时用它腾空间。",
        "usage": 'forget_user_preference(key)',
        "example": 'forget_user_preference("response_style")',
    },
}


def build_tools(
    include_web: bool = True,
    store=None,
    embedding_client=None,
    user_id=None,
):
    """构造 agent 使用的工具列表。

    Args:
        include_web: 是否包含联网工具。离线调试、或运行环境缺
            requests / markdownify 时传 False（缺依赖时这一组会自动缺席，
            本地工具照常返回）。
        store: 记忆库。传 None 时偏好工具会在**第一次被调用**时
            才建默认库（构造工具本身不碰 SQLite，测试/离线无副作用）。
        embedding_client: 偏好向量化客户端；None 时读侧退化为结构化读取，
            写侧只落库、不建向量（与历史记忆的降级策略一致）。
        user_id: 偏好归属的用户；None 用项目默认的单一用户。
    """
    memory_kwargs: dict[str, Any] = {
        "store": store,
        "embedding_client": embedding_client,
    }
    if user_id is not None:
        memory_kwargs["user_id"] = user_id

    tools = [
        get_current_time_in_timezone,
        read_text_file,
        list_files,
        # 偏好工具紧跟本地文件工具：都是"本地记忆/数据"，与联网工具分开
        LoadPreferencesTool(**memory_kwargs),
        SavePreferenceTool(**memory_kwargs),
        ForgetPreferenceTool(**memory_kwargs),
    ]

    if include_web and WebSearchTool is None:
        # 依赖没装全：联网工具整组缺席，但仍然给本地工具，别让启动直接失败。
        print(
            f"[tools] 未加载联网工具（{_WEB_IMPORT_ERROR}）："
            "本次只有本地工具与偏好工具；需要联网请安装 requests 与 markdownify"
        )
    elif include_web:
        # 三个联网工具共享一份「本轮缓存」：重复的 query/url 不再打网络，
        # 抓过的页面也能被 extract 直接复用。每轮由 agent/guardrails.py 清空。
        web_cache = TaskScopedCache()
        try:
            tools.append(
                WebSearchTool(
                    max_results=WEB_SEARCH_MAX_RESULTS,
                    item_chars=WEB_SEARCH_ITEM_CHARS,
                    max_chars=WEB_SEARCH_MAX_CHARS,
                    max_calls=WEB_SEARCH_MAX_CALLS_PER_TASK,
                    cache=web_cache,
                )
            )
        except ImportError as exc:
            print(f"[tools] 跳过 web_search：{exc}")
        except Exception as exc:
            # 目前 web_search 用 DeepSeek Search API，缺 DEEPSEEK_API_KEY 时
            # 构造会抛 RuntimeError；这时只降级掉搜索，不要让整个 agent 起不来。
            print(f"[tools] 跳过 web_search（{type(exc).__name__}）：{exc}")
        try:
            # extract 排在 visit_webpage 前面：prompt 里的工具顺序会影响
            # 小模型的选择，便宜且省上下文的那一个应该先出现。
            tools.append(
                ExtractTool(
                    max_chars=WEB_EXTRACT_MAX_CHARS,
                    max_calls=WEB_EXTRACT_MAX_CALLS_PER_TASK,
                    cache=web_cache,
                )
            )
        except Exception as exc:
            print(f"[tools] 跳过 extract：{exc}")
        try:
            tools.append(
                VisitWebpageTool(
                    max_chars=WEB_PAGE_MAX_CHARS,
                    max_calls=WEB_PAGE_MAX_CALLS_PER_TASK,
                    cache=web_cache,
                )
            )
        except Exception as exc:  # 工具初始化失败不应该拖垮整个 agent
            print(f"[tools] 跳过 visit_webpage：{exc}")

    tools.append(FinalAnswerTool())
    return tools


def get_tool(
    name: str,
    include_web: bool = True,
    store=None,
    embedding_client=None,
    user_id=None,
):
    """按名字取单个工具，找不到返回 None。

    ``user_id`` 与 :func:`build_tools` 同名同义：不给就用项目默认用户。
    少了它就没法为「非默认用户」单独取一个偏好工具，测试与多用户调用
    只能绕道 ``build_tools(...)[i]``。
    """
    for item in build_tools(
        include_web=include_web,
        store=store,
        embedding_client=embedding_client,
        user_id=user_id,
    ):
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
