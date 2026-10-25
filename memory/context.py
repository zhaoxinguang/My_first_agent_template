"""记忆 → prompt 上下文的唯一拼装点。

整个项目只允许在这里决定「哪条记忆、以什么格式、带什么语气」进入 prompt，
避免同一个偏好/同一段历史被多处注入（重复注入会浪费 token，
也会让模型误判记忆的权重）。

两类记忆各占一个块：
- 历史对话块：按语义相关性召回的「当时的问题 + 当时的回答」，
  条数与字符预算由 memory/retriever.py 的常量硬约束，块首带约束语；
- 偏好块：用户的长期偏好，只有一列 `- key：value`，不带说明文字
  （怎么按需取全文写在 prompts.yaml 的固定规则里）。

"""

from memory.retriever import (
    DEFAULT_HISTORY_LIMIT,
    HISTORY_MAX_ITEMS,
    HISTORY_MIN_SIMILARITY,
    PREF_INDEX_MAX_ITEMS,
    MemoryRetriever,
    is_time_sensitive,
)

# 单用户项目先用固定 id；接入多用户时由调用方传入真实 id
DEFAULT_USER_ID = "default_user"

# 注入时给模型的约束说明。偏好是「参考」，不是「硬指令」。
# 刻意压成一行：偏好块本身只是一列 `- key：value`，每轮固定进 prompt 的文字
# 越少越好（本地 ollama 的 num_ctx 默认只有 4096）。怎么按需取全文、什么时候
# 该写偏好，都放在 config/prompts.yaml 的固定规则里，不在这里重复一遍。
PREFERENCE_INSTRUCTION = (
    "以上偏好仅供参考；与当前指令冲突时，以当前指令为准。"
)

# 目录被条数上限截断时补的一行。仍然用列表形态，和上面几行读起来一致；
# 不补这一行，模型会以为目录就是全部。
PREFERENCE_HIDDEN_NOTE = (
    '- （另有 {hidden} 条偏好，需要时用 load_user_preferences("") 查看）'
)

# 历史对话的约束说明。
# 措辞要点（针对本地小模型）：
# 1. **先说不许用什么**。「会随时间变化的事实一律重新查」必须放在第一句——
#    旧措辞把「实时信息除外」放在整段最后，本地模型读不到，于是照抄了
#    上一轮别的城市的天气数据（真实事故）。
# 2. 再说可以用什么，并且要求核对「记录时间」。
# 3. 正例在前、例外在后是给「这段内容能不能用」定调；但涉及会不会出错时，
#    禁则必须在前，否则模型只会记住"可以直接用"。
HISTORY_INSTRUCTION = (
    "以下是历史问答，仅供参考。"
    "凡是会随时间变化的事实（天气、价格、比分、新闻、日程、库存等），"
    "一律不要沿用历史，必须重新调用工具查询。"
    "只有稳定不变的信息（定义、算法、代码写法、用户自己的偏好、"
    "已经确认过的计算结论）才可以参考，"
    "并请核对「记录于」的时间是否仍然成立。"
    "沿用了历史事实时，请在回答里简要说明依据来自历史。"
)


def _build_preference_block(
    task: str,
    retriever: MemoryRetriever,
    user_id: str,
    scope: str | None,
    limit: int,
) -> str:
    """注入「偏好目录」；完整内容由工具按需加载。

    返回的就是一列 ``- key：value``，不带任何说明文字：目录每轮都进 prompt，
    本地 ollama 的 num_ctx 只有 4096，说明文字直接等于秒数。按需取全文的
    用法写在 config/prompts.yaml 的固定规则里（规则 14/15），不在这里重复。

    两处用 getattr 做兼容：只实现 ``retrieve_preferences`` /
    ``format_preferences`` 的替身对象（测试、外部注入）仍能工作，
    不会被这次改动拖挂。
    """
    retrieve_index = getattr(
        retriever, "retrieve_preference_index", None
    )
    index_limit = min(limit, PREF_INDEX_MAX_ITEMS)

    try:
        if retrieve_index is not None:
            memories = retrieve_index(
                query=task,
                user_id=user_id,
                scope=scope,
                limit=index_limit,
            )
        else:
            memories = retriever.retrieve_preferences(
                query=task,
                user_id=user_id,
                scope=scope,
                limit=index_limit,
            )
    except Exception as exc:  # Embedding 服务挂掉不应阻断对话
        print(f"[memory] 偏好检索失败，本次不注入偏好：{exc}")
        return ""

    if not memories:
        return ""

    formatter = getattr(retriever, "format_preference_index", None)
    if formatter is not None:
        block = formatter(memories)
    else:
        block = retriever.format_preferences(memories)

    if not block or not block.strip():
        return ""

    total = _count_preferences(retriever, user_id, scope)
    hidden = total - len(memories)
    if hidden > 0:
        # 目录有上限（PREF_INDEX_MAX_ITEMS / PREF_INDEX_MAX_CHARS），
        # 偏好却可以攒到 PREF_MAX_ITEMS 条。不说明「还有几条没列出来」，
        # 模型会以为目录就是全部，用户也看不出记忆只生效了一小部分。
        return f"{block}\n{PREFERENCE_HIDDEN_NOTE.format(hidden=hidden)}"

    return block


def _count_preferences(retriever, user_id, scope) -> int:
    """有效偏好总数，只用于「还有几条没列出来」这句话。

    拿不到就返回 0（不提示），绝不猜一个数字：替身 retriever（测试、
    外部注入）可能根本没有这个能力，而错的条数比没有条数更容易误导模型。
    """
    counter = getattr(retriever, "count_preferences", None)
    if counter is None:
        return 0
    try:
        return int(counter(user_id=user_id, scope=scope))
    except Exception as exc:
        print(f"[memory] 偏好条数统计失败，本次不提示未列出的条目：{exc}")
        return 0


def build_history_context(
    task: str,
    retriever: MemoryRetriever,
    limit: int = DEFAULT_HISTORY_LIMIT,
    min_similarity: float = HISTORY_MIN_SIMILARITY,
) -> str:
    """检索语义相关的历史问答，返回可直接拼进 prompt 的文本。

    没有命中、检索出错、或 retriever 不支持历史召回时返回空字符串——
    调用方据此原样放行任务，而不是塞一句「没有相关历史」占 token。
    """
    if not task or not task.strip() or limit < 1:
        return ""

    # 时效性问题不注入历史。retriever 内部也拦了一道，这里再拦一次：
    # 注入决策在 context 层，不能指望所有替身 retriever 都记得这条规则。
    if is_time_sensitive(task):
        return ""

    # 用 getattr 而不是直接调用：允许只实现偏好检索的替身对象（测试、外部注入）。
    retrieve = getattr(retriever, "retrieve_history", None)
    if retrieve is None:
        return ""

    try:
        memories = retrieve(
            query=task,
            limit=min(limit, HISTORY_MAX_ITEMS),
            min_similarity=min_similarity,
        )
        if not memories:
            return ""
        block = retriever.format_history(memories)
    except Exception as exc:
        print(f"[memory] 历史检索失败，本次不注入历史：{exc}")
        return ""

    if not block or not block.strip():
        return ""

    return f"{HISTORY_INSTRUCTION}\n{block}"


def build_context(
    task: str,
    retriever: MemoryRetriever,
    user_id: str = DEFAULT_USER_ID,
    scope: str | None = None,
    limit: int = 5,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
) -> str:
    """拼出本轮注入的全部记忆块（历史在前、偏好在后）。

    任何一块为空都不占位：返回的是可直接放进 prompt 的纯文本，
    两块都为空时返回空字符串。
    """
    if not task or not task.strip():
        return ""

    blocks = []

    history_block = build_history_context(
        task, retriever, limit=history_limit
    )
    if history_block:
        blocks.append(history_block)

    preference_block = _build_preference_block(
        task, retriever, user_id, scope, limit
    )
    if preference_block:
        blocks.append(preference_block)

    return "\n\n".join(blocks)


def augment_task(
    task: str,
    context: str,
    instruction: str = PREFERENCE_INSTRUCTION,
) -> str:
    """把上下文拼到任务前面；context 为空时原样返回任务。"""
    if not context:
        return task

    if not instruction:
        return f"{context}\n\n当前用户任务：\n{task}"

    return (
        f"{context}\n\n"
        f"{instruction}\n\n"
        f"当前用户任务：\n{task}"
    )


def build_augmented_task(
    task: str,
    retriever: MemoryRetriever,
    user_id: str = DEFAULT_USER_ID,
    scope: str | None = None,
    limit: int = 5,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
) -> str:
    """一步完成「检索 + 拼装」，等价于 build_context() 后接 augment_task()。

    历史块自带 HISTORY_INSTRUCTION；只有偏好块需要额外补
    PREFERENCE_INSTRUCTION，所以指令跟着「偏好是否命中」走，
    避免没有偏好时还让模型「参考用户偏好」。
    """
    if not task or not task.strip():
        return task

    history_block = build_history_context(
        task, retriever, limit=history_limit
    )
    preference_block = _build_preference_block(
        task, retriever, user_id, scope, limit
    )

    blocks = [block for block in (history_block, preference_block) if block]
    if not blocks:
        return task

    instruction = PREFERENCE_INSTRUCTION if preference_block else ""

    return augment_task(task, "\n\n".join(blocks), instruction)


__all__ = [
    "DEFAULT_USER_ID",
    "PREFERENCE_INSTRUCTION",
    "PREFERENCE_HIDDEN_NOTE",
    "HISTORY_INSTRUCTION",
    "build_context",
    "build_history_context",
    "augment_task",
    "build_augmented_task",
]
