"""带长期记忆的 Agent 包装器。

它在每次执行前做一件事：把与本轮任务相关的记忆（当前是用户偏好）
拼进任务描述，然后交给底层 CodeAgent 执行。

分层意图：
- MemoryAgent 只负责「任务增强」，不关心 UI，也不负责写记忆；
- 检索与格式化都在 memory/context.py 里，避免多处注入；
- 底层 agent 仍然是普通的 smolagents CodeAgent，可单独测试。
"""

from memory.context import DEFAULT_USER_ID, build_augmented_task
from memory.retriever import MemoryRetriever


class MemoryAgent:
    """给任意 smolagents agent 套一层偏好注入。"""

    def __init__(
        self,
        agent,
        retriever: MemoryRetriever | None = None,
        user_id: str = DEFAULT_USER_ID,
        scope: str | None = None,
        limit: int = 5,
    ):
        self.agent = agent
        self.retriever = retriever or MemoryRetriever()
        self.user_id = user_id
        self.scope = scope
        self.limit = limit

    # ---------- 对外能力 ----------

    def prepare_task(self, task: str) -> str:
        """只做「检索 + 拼装」，不执行。便于单独调试注入效果。"""
        return build_augmented_task(
            task=task,
            retriever=self.retriever,
            user_id=self.user_id,
            scope=self.scope,
            limit=self.limit,
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
