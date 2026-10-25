"""用户偏好工具：让 agent 自己「按需加载」并「动态更新」长期偏好。

三个工具对应三种动作，命名刻意保持动词开头，方便小模型区分：

- ``load_user_preferences(query)``  读：目录里只有要点，完整取值从这取；
- ``save_user_preference(...)``     写：新增或覆盖同 key 的偏好（带预算校验）；
- ``forget_user_preference(key)``   删：过时的偏好自己清掉。

设计约束（都是从「模型会怎么用」倒推出来的）：

1. **一律返回字符串，不抛异常**。抛异常会变成一步工具执行失败，
   小模型容易原地重试到耗尽 max_steps；返回 ``Error: ...`` 它才知道换路。
2. **错误信息必须可执行**。预算超了要告诉它当前占用、上限、
   以及「用 forget_user_preference 删哪条」，否则它只会再试一次同样的写入。
3. **懒建库**。构造工具时不碰 SQLite（测试和纯离线场景不应该产生副作用），
   第一次真正调用时才拿默认 store。
"""

from smolagents.tools import Tool

from memory.context import DEFAULT_USER_ID
from memory.preference_manager import (
    PreferenceBudgetError,
    PreferenceManager,
    PreferenceSkipped,
)
from memory.retriever import PREF_LOAD_MAX_ITEMS, MemoryRetriever

# 懒加载的默认 store：只有真的调用工具时才建库/连库
_DEFAULT_STORE = None


def _resolve_store(store):
    """返回注入的 store；没有注入时用进程内单例，避免每步都开新连接。"""
    if store is not None:
        return store
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        from memory.store import MemoryStore

        _DEFAULT_STORE = MemoryStore()
    return _DEFAULT_STORE


def _clip(text, limit):
    text = " ".join(str(text or "").split())
    if limit and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class _PreferenceTool(Tool):
    """三个偏好工具共用的底座：store / embedding / user_id 的解析。"""

    def __init__(
        self,
        store=None,
        embedding_client=None,
        user_id: str = DEFAULT_USER_ID,
    ):
        self._store = store
        self._embedding_client = embedding_client
        self.user_id = user_id
        super().__init__()

    # 每次调用现取，保证「上一个工具刚写的偏好」下一次调用就能读到，
    # 不在工具实例上缓存任何偏好状态。
    def _store_now(self):
        return _resolve_store(self._store)

    def _retriever(self):
        return MemoryRetriever(
            store=self._store_now(),
            embedding_client=self._embedding_client,
        )

    def _manager(self):
        return PreferenceManager(
            store=self._store_now(),
            embedding_client=self._embedding_client,
        )


class LoadPreferencesTool(_PreferenceTool):
    name = "load_user_preferences"
    description = (
        "按需加载用户偏好的完整内容。参考上下文里只给了偏好目录（要点），"
        "当某条偏好可能影响当前任务、但要点不足以判断具体要求时调用它。"
        "query 可以是一条偏好的 key（如 response_style），"
        "也可以是一句话（按语义找），传空字符串表示列出全部偏好。"
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "偏好的 key、key 的一部分，或一句描述需求的话；"
                "传空字符串表示列出全部偏好"
            ),
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        query = str(query or "").strip()

        try:
            rows = self._retriever().load_preferences(
                query=query,
                user_id=self.user_id,
                limit=PREF_LOAD_MAX_ITEMS,
            )
        except Exception as exc:  # 读失败不该让整轮任务失败
            return f"Error: 读取偏好失败（{type(exc).__name__}）：{exc}"

        if not rows:
            if query:
                return (
                    f"没有与「{query}」匹配的偏好。"
                    "可以调用 load_user_preferences(\"\") 查看全部，"
                    "或按当前任务直接执行。"
                )
            return "当前没有任何长期偏好，按当前任务要求执行即可。"

        lines = [f"共 {len(rows)} 条偏好（完整内容）："]
        for row in rows:
            # 单条再长也留个上限：偏好是「一句话说清」的东西，
            # 这里截断只是防呆，正常条目远不到这个长度。
            value = _clip(row.get("memory_value"), 500)
            lines.append(
                f"- [{row.get('category') or 'general'}] "
                f"{row.get('memory_key')} = {value}"
                f"（作用域 {row.get('scope') or 'global'}，"
                f"置信度 {row.get('confidence', 1.0)}）"
            )
        lines.append(
            "这些是用户的长期偏好，仅在不违背当前任务的前提下参考。"
        )
        return "\n".join(lines)


class SavePreferenceTool(_PreferenceTool):
    name = "save_user_preference"
    description = (
        "新增或覆盖一条用户长期偏好。系统不会自动提炼偏好，"
        "是否写入由你自己在任务收尾时判断：本轮暴露了下一轮仍然成立的信息"
        "（语言、回答长度、输出格式、语气、工具环境、禁忌）才调用，"
        "没有就完全不要调用。同一个 key 再次写入视为覆盖旧值。"
        "不要把任务进度、临时数据或一次性要求写进来。"
    )
    inputs = {
        "key": {
            "type": "string",
            "description": "偏好的名字，用英文小写下划线，例如 response_style",
        },
        "value": {
            "type": "string",
            "description": "偏好的具体内容，一句话说清，例如「回答简洁，优先给可运行代码」",
        },
        "category": {
            "type": "string",
            "description": "分类，例如 response_style / tooling / environment，不确定就用 general",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, key: str, value: str, category: str = "general") -> str:
        key = str(key or "").strip()
        value = str(value or "").strip()

        if not key:
            return "Error: key 不能为空，请给出英文小写下划线的偏好名。"
        if not value:
            return "Error: value 不能为空，请用一句话写清偏好内容。"

        manager = self._manager()
        try:
            memory_id = manager.upsert(
                user_id=self.user_id,
                memory_key=key,
                memory_value=value,
                category=str(category or "general").strip() or "general",
            )
        except PreferenceSkipped as exc:
            # 「没写进去」但不算错误：如实回执，别让模型以为写成功了
            return f"Error: 未写入 {key}={value}。{exc}"
        except PreferenceBudgetError as exc:
            # 预算不足是可预期的「让模型自己腾地方」，不是异常路径
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: 保存偏好失败（{type(exc).__name__}）：{exc}"

        status = manager.budget_status(self.user_id)
        return (
            f"已保存偏好 {key} = {value}。"
            f"当前共 {status['items']} 条，占用 {status['used_chars']}"
            f"/{status['char_limit']} 字符。"
        )


class ForgetPreferenceTool(_PreferenceTool):
    name = "forget_user_preference"
    description = (
        "删除一条已经过时或错误的用户偏好。参数是本条偏好的 key"
        "（可从偏好目录或 load_user_preferences 的回执里看到）。"
        "偏好写不进去时（预算已满），也用它先腾出空间。"
    )
    inputs = {
        "key": {
            "type": "string",
            "description": "要删除的偏好 key，例如 response_style",
        }
    }
    output_type = "string"

    def forward(self, key: str) -> str:
        key = str(key or "").strip()
        if not key:
            return "Error: key 不能为空。"

        manager = self._manager()
        try:
            removed = manager.forget(self.user_id, key)
        except Exception as exc:
            return f"Error: 删除偏好失败（{type(exc).__name__}）：{exc}"

        if removed:
            status = manager.budget_status(self.user_id)
            return (
                f"已删除偏好 {key}。"
                f"当前共 {status['items']} 条，占用 {status['used_chars']}"
                f"/{status['char_limit']} 字符。"
            )

        keys = [
            row["memory_key"]
            for row in manager.list_preferences(self.user_id)
        ]
        hint = "、".join(keys[:8]) if keys else "（当前没有任何偏好）"
        return (
            f"Error: 没有找到 key 为「{key}」的偏好。"
            f"现有偏好：{hint}。请确认 key 后重试，"
            f"或调用 load_user_preferences(\"\") 查看全部。"
        )


__all__ = [
    "LoadPreferencesTool",
    "SavePreferenceTool",
    "ForgetPreferenceTool",
]
