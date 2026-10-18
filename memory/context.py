"""记忆 → prompt 上下文的唯一拼装点。

整个项目只允许在这里决定「哪条记忆、以什么格式、带什么语气」进入 prompt，
避免同一个偏好被多处注入（重复注入会浪费 token，也会让模型误判偏好的权重）。
"""

from memory.retriever import MemoryRetriever

# 单用户项目先用固定 id；接入多用户时由调用方传入真实 id
DEFAULT_USER_ID = "default_user"

# 注入时给模型的约束说明。偏好是「参考」，不是「硬指令」。
PREFERENCE_INSTRUCTION = (
    "请在不违背当前任务要求的前提下参考用户偏好。"
    "偏好仅用于个性化，不得覆盖用户当前的明确指令；"
    "若偏好与当前指令冲突，以当前指令为准，并简要说明你做了调整。"
)


def build_context(
    task: str,
    retriever: MemoryRetriever,
    user_id: str = DEFAULT_USER_ID,
    scope: str | None = None,
    limit: int = 5,
) -> str:
    """检索与任务相关的用户偏好，返回可直接拼进 prompt 的文本。

    没有命中、或检索过程出错时返回空字符串——调用方据此决定原样放行任务，
    而不是塞一句「没有检索到相关用户偏好」占用上下文。
    """
    if not task or not task.strip():
        return ""

    try:
        memories = retriever.retrieve_preferences(
            query=task,
            user_id=user_id,
            scope=scope,
            limit=limit,
        )
    except Exception as exc:  # Embedding 服务挂掉不应阻断对话
        print(f"[memory] 偏好检索失败，本次不注入偏好：{exc}")
        return ""

    if not memories:
        return ""

    return retriever.format_preferences(memories)


def augment_task(task: str, context: str) -> str:
    """把上下文拼到任务前面；context 为空时原样返回任务。"""
    if not context:
        return task

    return (
        f"{context}\n\n"
        f"{PREFERENCE_INSTRUCTION}\n\n"
        f"当前用户任务：\n{task}"
    )


def build_augmented_task(
    task: str,
    retriever: MemoryRetriever,
    user_id: str = DEFAULT_USER_ID,
    scope: str | None = None,
    limit: int = 5,
) -> str:
    """一步完成「检索 + 拼装」，等价于 build_context() 后接 augment_task()。"""
    context = build_context(
        task=task,
        retriever=retriever,
        user_id=user_id,
        scope=scope,
        limit=limit,
    )
    return augment_task(task, context)


__all__ = [
    "DEFAULT_USER_ID",
    "PREFERENCE_INSTRUCTION",
    "build_context",
    "augment_task",
    "build_augmented_task",
]
