"""带长期记忆的 Agent 包装器。

它在每次执行前做一件事：把与本轮任务相关的记忆拼进任务描述，
然后交给底层 CodeAgent 执行。注入两类记忆：

- 历史对话：按语义相关性召回的「当时的问题 + 当时的回答」（有严格条数/字数上限）；
- 用户偏好：长期偏好，语气是「参考」而不是「硬指令」。

分层意图：
- MemoryAgent 只负责「任务增强」，不关心 UI，也不负责写记忆；
- 检索与格式化都在 memory/ 层（retriever / context），避免多处注入；
- 底层 agent 仍然是普通的 smolagents CodeAgent，可单独测试。
"""

from memory.context import DEFAULT_USER_ID, build_augmented_task
from memory.retriever import (
    DEFAULT_HISTORY_LIMIT,
    PREF_INDEX_MAX_ITEMS,
    MemoryRetriever,
)


class MemoryAgent:
    """给任意 smolagents agent 套一层记忆注入。"""

    def __init__(
        self,
        agent,
        retriever: MemoryRetriever | None = None,
        user_id: str = DEFAULT_USER_ID,
        scope: str | None = None,
        # 目录条数默认放开到上限（8 条）：偏好预算可以到 40 条，
        # 而目录是它们唯一的自动注入入口，默认只给 5 条等于让大半偏好
        # 连「被看见」的机会都没有。字符预算由 PREF_INDEX_MAX_CHARS 单独把住。
        limit: int = PREF_INDEX_MAX_ITEMS,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ):
        self.agent = agent
        self.retriever = retriever or MemoryRetriever()
        self.user_id = user_id
        self.scope = scope
        self.limit = limit
        # 0 表示关闭历史注入；上限由 memory/retriever.py 的常量把住
        self.history_limit = history_limit

    # ---------- 对外能力 ----------

    def prepare_task(self, task: str) -> str:
        """只做「检索 + 拼装」，不执行。便于单独调试注入效果。"""
        return build_augmented_task(
            task=task,
            retriever=self.retriever,
            user_id=self.user_id,
            scope=self.scope,
            limit=self.limit,
            history_limit=self.history_limit,
        )

    def run(self, task: str, *args, **kwargs):
        """执行任务；stream=True 时原样返回底层生成器。"""
        return self.agent.run(
            self.prepare_task(task),
            *args,
            **kwargs,
        )

    # ---------- 透明转发 ----------

    @property
    def model(self):
        """供 GradioUI 统计 token 用量。"""
        return getattr(self.agent, "model", None)

    def __getattr__(self, name):
        # 只有常规属性查找失败时才会走到这里，
        # 因此 tools / prompt_templates 等仍可直接访问。
        return getattr(self.agent, name)


__all__ = ["MemoryAgent"]
