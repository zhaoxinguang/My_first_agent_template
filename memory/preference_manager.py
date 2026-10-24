"""用户偏好的写入、更新与遗忘。

分工：
- 本模块是**写侧**（策略 + 落地）：字符预算、覆盖语义、按 key 遗忘；
- 读侧在 ``memory/retriever.py``：目录（零 embedding）与按需完整加载。

为什么要有字符预算
------------------
偏好是被拼进每一轮 prompt 的固定开销。没有上限时，模型会不断 append
新条目，最后要么把上下文吃光，要么每条都被截断成不可用的碎片。
这里给「单用户所有有效偏好」一个总字数上限，超了**报错**而不是静默截断，
逼调用方（人或模型）先合并/删除再写——策展而不是堆积。
"""

import json
import os

from memory.store import MemoryStore

# 总预算：key + value 的字符数之和（单用户、所有作用域）。
# 3500 字符约 1000~1500 token，和 hermes 对 MEMORY.md / USER.md 的总量
# 上限（3575 字符）同一量级：足够记十几条稳定偏好，又不至于挤占任务上下文。
PREF_TOTAL_CHAR_LIMIT = 3500
# 条数上限：防止「每条形如 20 字」的碎片把预算切得太碎。
PREF_MAX_ITEMS = 40
# 新值的置信度比旧值低这么多时视为「猜测」，不覆盖旧值。
# 真实事故：自动提炼把 confidence=0.1 的 language=english 写上去，
# 顶掉了用户明确说过的 confidence=0.8 的 language=中文，
# 之后每轮 prompt 里都写着 english，模型却按中文答，看起来「偏好完全没生效」。
PREF_CONFIDENCE_DROP_LIMIT = 0.2


# ---------------------------------------------------------------------------
# (key, value) 规范化：同一件事只允许一个 key
# ---------------------------------------------------------------------------
# 自动提炼让模型自由起 key，结果同一件事攒出
# ``response_style`` / ``detail_level`` / ``detail_degree`` 三条，
# 目录里看着 5 条、模型能用的其实只有一两条。
# 这里给「有明确枚举范围」的维度定死规范 key，其余 key 原样保留
# （工具路径由用户/模型显式写入，指纹即语义，不该被悄悄改名）。
CANONICAL_KEY_ALIASES = {
    # 详细程度 / 长度
    "detail": "answer_length",
    "detail_degree": "answer_length",
    "detail_level": "answer_length",
    "preferred_detail_level": "answer_length",
    "verbosity": "answer_length",
    "length": "answer_length",
    "response_length": "answer_length",
    "answer_length": "answer_length",
    "详细程度": "answer_length",
    "简洁程度": "answer_length",
    # 语言
    "lang": "language",
    "reply_language": "language",
    "answer_language": "language",
    "language": "language",
    "回答语言": "language",
    # 格式
    "format": "answer_format",
    "output_format": "answer_format",
    "response_format": "answer_format",
    "answer_format": "answer_format",
    "回答格式": "answer_format",
}

# 「值本身就能说明属于哪一维」的单词式取值。
# 只认单词（``简短`` / ``short``），复合值（``回答简洁，先给可运行代码``）
# 保持原 key：把一个复合偏好硬塞进某个维度只会丢信息。
_VALUE_DIMENSIONS = {
    "answer_length": {
        "简短", "简洁", "简略", "精炼", "短", "详细", "详尽", "长",
        "short", "brief", "concise", "detailed", "long", "verbose",
    },
    "answer_format": {
        "text", "纯文本", "markdown", "md", "json", "table", "表格",
        "bullet", "列表", "分条", "plain",
    },
    "language": {
        "中文", "汉语", "英文", "英语", "日文", "日语",
        "chinese", "english", "japanese", "中英混排",
    },
    "response_style": {
        "可爱", "可爱风格", "幽默", "风趣", "严肃", "正式", "亲切", "温柔",
        "cute", "funny", "formal", "friendly", "professional",
    },
}

# 值归一：同一个意思只留一种写法，目录里才不会一会儿「简短」一会儿「short」
_VALUE_CANONICAL = {
    "short": "简短", "brief": "简短", "concise": "简短", "简洁": "简短",
    "简略": "简短", "精炼": "简短", "短": "简短",
    "detailed": "详细", "long": "详细", "verbose": "详细", "详尽": "详细",
    "长": "详细",
    "md": "markdown", "plain": "纯文本", "列表": "分条", "bullet": "分条",
    # 模型把「分条陈述」写成 list / bullets 很常见；这条映射能保住用户
    # 真实表达的偏好，而不是让 _looks_translated 把它当噪声丢掉
    "list": "分条", "bullets": "分条", "bullet points": "分条",
    "汉语": "中文", "英文": "英文", "英语": "英文",
    "chinese": "中文", "english": "英文", "japanese": "日文",
}


def known_values_for(memory_key):
    """某一维度的所有已知取值（小写）；未知维度返回空集合。

    给提炼器当「防幻觉」白名单用：``answer_format：informal`` 这种既不在
    枚举里、又是纯英文单词的值，基本可以断定是模型把中文偏好翻译成了英文
    （真实数据里出现过 informal / low / english 三条）。
    """
    key = str(memory_key or "").strip().lower()
    values = _VALUE_DIMENSIONS.get(key)
    if not values:
        return set()
    return {value.lower() for value in values}


def _value_dimension(memory_value):
    """值若是单词式的维度取值，返回它属于哪一维；否则返回 None。"""
    if not isinstance(memory_value, str):
        return None
    needle = memory_value.strip().lower()
    if not needle:
        return None
    for dimension, values in _VALUE_DIMENSIONS.items():
        if needle in {value.lower() for value in values}:
            return dimension
    return None


def canonicalize_preference(memory_key, memory_value):
    """把 ``(key, value)`` 收敛到固定维度，返回新的 ``(key, value)``。

    两条规则：

    1. **key 同义写法 → 规范 key**：``detail_level`` / ``detail_degree`` /
       ``verbosity`` 都是 ``answer_length``；
    2. **值是单词式的维度取值时以值判维度**：真实数据里
       ``response_style：简短``（其实说的是长度）、
       ``answer_format：可爱风格``（其实说的是语气）这类错位都能纠回来。

    非单词值、非字符串值一律原样返回，绝不猜测。
    """
    key = str(memory_key or "").strip()
    if not key:
        return key, memory_value

    key_dimension = CANONICAL_KEY_ALIASES.get(key.lower())
    value_dimension = _value_dimension(memory_value)

    if value_dimension and value_dimension != key_dimension:
        canonical_key = value_dimension
    elif key_dimension:
        canonical_key = key_dimension
    else:
        canonical_key = key

    if isinstance(memory_value, str):
        stripped = memory_value.strip()
        memory_value = _VALUE_CANONICAL.get(stripped.lower(), stripped)

    return canonical_key, memory_value


# ---------------------------------------------------------------------------
# category 也跟着 key 收敛
# ---------------------------------------------------------------------------
# 真实数据里出现过 ``category=detail_degree``（照抄 key）、``category=preference``
# 这种既不可读、也不能按类筛选的写法。category 只保留四个值，默认归类由 key 决定。
CATEGORY_BY_KEY = {
    "language": "response_style",
    "answer_length": "response_style",
    "answer_format": "response_style",
    "response_style": "response_style",
    "tooling": "tooling",
    "environment": "environment",
}
VALID_CATEGORIES = ("response_style", "tooling", "environment", "other")


def default_category_for(memory_key):
    """某个 key 的默认归类；读侧用它判断 category 是否值得印出来。"""
    key = str(memory_key or "").strip().lower()
    return CATEGORY_BY_KEY.get(key, "other")


def canonical_category(category, memory_key):
    """把模型给的 category 收敛到 ``VALID_CATEGORIES`` 之一。

    已知合法类别但和 key 的维度冲突时以 key 为准（key 是从值里判出来的，
    比自由填写的 category 可靠）；模型把 key 当 category 也按 key 归一。
    """
    key = str(memory_key or "").strip().lower()
    text = str(category or "").strip().lower()
    if text in VALID_CATEGORIES:
        return CATEGORY_BY_KEY.get(key, text)
    if text in CANONICAL_KEY_ALIASES:
        return CATEGORY_BY_KEY.get(key, "response_style")
    return CATEGORY_BY_KEY.get(key, "other")


def _env_int(name, default):
    """读整数环境变量；空值或非法值回落到默认值（不抛异常）。

    和 config/settings.py 里「非法值直接报错」的取舍不同：这里读的是
    可选调优参数，配错了不该让整个应用起不来。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _as_confidence(raw, default=1.0):
    """把库里的 confidence 转成 float；缺列/脏值按默认值处理（不抛异常）。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.0), 1.0)


class PreferenceBudgetError(ValueError):
    """偏好超出预算。

    消息里必须带上「现在占了多少 / 上限多少 / 该怎么腾空间」，
    因为这条消息会被工具层原样回给模型，是它决定下一步的唯一依据。
    """


class PreferenceManager:
    def __init__(
        self,
        store=None,
        embedding_client=None,
        char_limit=None,
        max_items=None,
    ):
        self.store = store or MemoryStore()
        self.embedding_client = embedding_client
        self.char_limit = char_limit or _env_int(
            "PREF_CHAR_LIMIT", PREF_TOTAL_CHAR_LIMIT
        )
        self.max_items = max_items or _env_int(
            "PREF_MAX_ITEMS", PREF_MAX_ITEMS
        )

    @staticmethod
    def _embedding_text(category, memory_key, memory_value, scope):
        if not isinstance(memory_value, str):
            memory_value = json.dumps(
                memory_value,
                ensure_ascii=False,
                sort_keys=True,
            )

        return (
            f"记忆类型：{category}\n"
            f"适用范围：{scope}\n"
            f"属性：{memory_key}\n"
            f"内容：{memory_value}"
        )

    # ---------- 预算 ----------

    @staticmethod
    def _plain_value(memory_value):
        """把库里的值转成「人类/模型看到的字符串」，用于算字符数。"""
        if isinstance(memory_value, str):
            return memory_value
        return json.dumps(
            memory_value, ensure_ascii=False, sort_keys=True
        )

    def _active_rows(self, user_id):
        return self.store.get_preferences(user_id=user_id)

    def _find_active(self, user_id, memory_key, scope):
        """取同一 (key, scope) 的有效行；没有则返回 None。"""
        for row in self._active_rows(user_id):
            if row["memory_key"] == memory_key and row["scope"] == scope:
                return row
        return None

    def budget_status(self, user_id="default_user"):
        """当前预算占用，供启动日志、工具回执与排障使用。"""
        rows = self._active_rows(user_id)
        used = sum(
            len(row["memory_key"]) + len(self._plain_value(row["memory_value"]))
            for row in rows
        )

        return {
            "user_id": user_id,
            "items": len(rows),
            "used_chars": used,
            "char_limit": self.char_limit,
            "max_items": self.max_items,
            "remaining_chars": max(0, self.char_limit - used),
            "remaining_items": max(0, self.max_items - len(rows)),
            "ok": used <= self.char_limit and len(rows) <= self.max_items,
        }

    def _check_budget(self, user_id, memory_key, memory_value, scope):
        """写入前的预算校验：把「覆盖同一条」的情况算作不新增。"""
        rows = self._active_rows(user_id)
        new_chars = len(memory_key) + len(self._plain_value(memory_value))

        same_key_chars = 0
        is_update = False
        for row in rows:
            if row["memory_key"] == memory_key and row["scope"] == scope:
                same_key_chars = len(row["memory_key"]) + len(
                    self._plain_value(row["memory_value"])
                )
                is_update = True
                break

        used = sum(
            len(row["memory_key"]) + len(self._plain_value(row["memory_value"]))
            for row in rows
        )
        projected = used - same_key_chars + new_chars
        projected_items = len(rows) + (0 if is_update else 1)

        if projected > self.char_limit:
            oldest = "、".join(
                row["memory_key"] for row in rows[::-1][:5]
            )
            raise PreferenceBudgetError(
                f"偏好预算不足：当前 {len(rows)} 条 / {used} 字符，"
                f"上限 {self.char_limit} 字符；"
                f"写入 {memory_key} 需要 {new_chars} 字符，"
                f"合计 {projected} 字符。"
                f"请先用 forget_user_preference(<key>) 删掉过时条目，"
                f"或把相近条目合并成一条再写。"
                + (f"最久未更新的是：{oldest}。" if oldest else "")
            )

        if projected_items > self.max_items:
            raise PreferenceBudgetError(
                f"偏好条数超出上限：当前 {len(rows)} 条，上限 {self.max_items} 条。"
                f"请先 forget_user_preference(<key>) 删掉过时条目，"
                f"或把多条合并成一条再写。"
            )

    # ---------- 写：动态更新 ----------

    def upsert(
        self,
        user_id,
        memory_key,
        memory_value,
        category="general",
        scope="global",
        confidence=1.0,
        source_run_id=None,
        expires_at=None,
    ):
        """新增或覆盖一条偏好，并强制预算。

        与 :meth:`save_preference` 的区别：这就是「动态更新」的生产入口，
        带预算校验和更友好的错误信息（会被工具层原样回给模型）。
        同一 (user, scope, key) 已存在时语义是**覆盖**：
        旧值自动失活保留为历史版本，不需要先删后写。
        """
        if not str(user_id or "").strip():
            raise ValueError("user_id 不能为空")
        if not str(memory_key or "").strip():
            raise ValueError("memory_key 不能为空")
        if not str(memory_value or "").strip():
            raise ValueError("memory_value 不能为空")

        memory_key = str(memory_key).strip()

        # 低置信度的「猜测」不许顶掉高置信度的既有偏好：
        # 自动提炼偶尔给出 confidence=0.1 的条目，直接覆盖会让用户
        # 明确说过的偏好在下一轮悄悄消失（见 PREF_CONFIDENCE_DROP_LIMIT 注释）。
        existing = self._find_active(user_id, memory_key, scope)
        if existing is not None:
            old_confidence = _as_confidence(existing.get("confidence"))
            if confidence + PREF_CONFIDENCE_DROP_LIMIT < old_confidence:
                print(
                    "[memory] 跳过低置信度覆盖："
                    f"{memory_key} 现有 {old_confidence:.2f} > 新值 {confidence:.2f}，"
                    "保留原偏好。"
                )
                return existing.get("memory_id")

        self._check_budget(user_id, memory_key, memory_value, scope)

        return self.save_preference(
            user_id=user_id,
            category=category or "general",
            memory_key=memory_key,
            memory_value=memory_value,
            scope=scope or "global",
            confidence=confidence,
            source_run_id=source_run_id,
            expires_at=expires_at,
        )

    def forget(self, user_id, memory_key, scope="global"):
        """按 key 遗忘一条偏好；返回是否命中。

        模型手里只有 key（没有自增 id），所以工具层必须按 key 删除。
        退化路径：先在 global 找，找不到再看该用户所有作用域，
        避免「记在 proj 作用域、按 global 删」这种静默失败。
        """
        memory_key = str(memory_key or "").strip()
        if not memory_key:
            return False

        if self.store.deactivate_preference_by_key(
            user_id=user_id, memory_key=memory_key, scope=scope
        ):
            return True

        for row in self.store.get_preferences(user_id=user_id):
            if row["memory_key"] == memory_key:
                return self.store.deactivate_preference(
                    memory_id=row["memory_id"], user_id=user_id
                )
        return False

    def save_preference(
        self,
        user_id,
        category,
        memory_key,
        memory_value,
        scope="global",
        confidence=1.0,
        source_run_id=None,
        expires_at=None,
    ):
        """底层写入：先落权威数据，再尽力补向量。

        Embedding 故障不影响数据落库（向量缺失只是「下次召回弱一点」），
        预算校验也不在这里做——需要策展约束请走 :meth:`upsert`。
        """
        memory_id = self.store.save_preference(
            user_id=user_id,
            category=category,
            memory_key=memory_key,
            memory_value=memory_value,
            scope=scope,
            confidence=confidence,
            source_run_id=source_run_id,
            expires_at=expires_at,
        )

        if self.embedding_client is None:
            return memory_id

        text = self._embedding_text(
            category, memory_key, memory_value, scope
        )

        try:
            vector = self.embedding_client.embed_query(text)
            self.store.save_embedding(
                memory_id=memory_id,
                embedding=vector,
                model=self.embedding_client.model,
                version="v1",
            )
        except Exception:
            # 数据已持久化，向量失败可在后续重建。
            # 实际部署中建议在这里记录日志并监控失败。
            pass

        return memory_id

    # ---------- 读：给工具与排障用 ----------

    def list_preferences(
        self, user_id, scope=None, category=None
    ):
        return self.store.get_preferences(
            user_id=user_id,
            scope=scope,
            category=category,
        )

    def forget_preference(self, memory_id, user_id):
        return self.store.deactivate_preference(
            memory_id=memory_id,
            user_id=user_id,
        )

    def rebuild_embeddings(self, user_id="default_user", limit=10000):
        """使用当前 embedding 模型重建用户已有偏好的向量。"""
        if self.embedding_client is None:
            raise RuntimeError("EmbeddingClient 未初始化")

        if limit < 1:
            raise ValueError("limit 必须大于 0")

        # 你的 get_preferences() 不支持 limit，因此在这里截取。
        memories = self.store.get_preferences(user_id=user_id)[:limit]

        success = 0
        failed = []

        for item in memories:
            memory_id = item["memory_id"]

            try:
                text = self._embedding_text(
                    item["category"],
                    item["memory_key"],
                    item["memory_value"],
                    item["scope"],
                )

                vector = self.embedding_client.embed_query(text)

                self.store.save_embedding(
                    memory_id=memory_id,
                    embedding=vector,
                    model=self.embedding_client.model,
                    version="v1",
                )

                success += 1

            except Exception as exc:
                failed.append({
                    "memory_id": memory_id,
                    "error": str(exc),
                })

        return {
            "total": len(memories),
            "success": success,
            "failed": len(failed),
            "errors": failed,
            "model": self.embedding_client.model,
        }


__all__ = [
    "PreferenceManager",
    "PreferenceBudgetError",
    "PREF_TOTAL_CHAR_LIMIT",
    "PREF_MAX_ITEMS",
    "PREF_CONFIDENCE_DROP_LIMIT",
    "CANONICAL_KEY_ALIASES",
    "canonicalize_preference",
    "CATEGORY_BY_KEY",
    "VALID_CATEGORIES",
    "canonical_category",
    "default_category_for",
    "known_values_for",
]
