
import json
import math

from memory.embedding import EmbeddingClient  # noqa: F401  (对外类型标注/兼容导出)
from memory.preference_manager import (
    CANONICAL_KEY_ALIASES,
    VALID_CATEGORIES,
    canonical_category,
    canonicalize_preference,
    default_category_for,
)
from memory.store import MemoryStore

# ---- 历史对话召回：条数与字符预算的硬上限 ----
# 这些值是「策略」，单独放常量：以后要放宽/收紧只改这里，
# 不会散落在 prompt 拼装和格式化逻辑里。
HISTORY_MAX_ITEMS = 2      # 单次最多注入几条历史，默认 2 条
# 整个历史块字符上限（含每条的行首缩进）。
# 从 700 收到 400：这 700 字符大约 500+ token，和 system prompt 一起
# 很容易把本地 ollama 的 num_ctx（默认 4096）占满，导致模型连代码块
# 都写不完。历史召回本来就是「宁缺毋滥」，宁可少注入一条。
HISTORY_MAX_CHARS = 400
HISTORY_TASK_CHARS = 80    # 单条：当时的问题最多保留多少字符
HISTORY_ANSWER_CHARS = 200  # 单条：当时的回答最多保留多少字符
HISTORY_MIN_SIMILARITY = 0.6
# 为什么是 0.6 而不是 0.35：本库实测（nomic-embed-text，中文短句）余弦相似度
# 的「地板」很高——「介绍一下 pandas 的 groupby 用法」对「查询成都今日天气」
# 也能拿到 0.57，0.35 的阈值基本等于不设阈值，任何任务都会注入两条天气历史。
# 0.6 是「同一类任务」才过得去的线：广州天气仍然命中广州/广西/成都天气（0.68~0.87），
# 而 pandas 这种无关任务被挡在外面。调低会重新引入无关历史噪声。
HISTORY_SCAN_LIMIT = 200   # 参与打分的「最近 N 条」run，防库变大后变慢
DEFAULT_HISTORY_LIMIT = HISTORY_MAX_ITEMS

# 时效性关键词：命中即**完全不注入历史**。
# 起因是一次真实事故：用户问「广州今日天气」，注入块把上一轮
# 「成都 14-19°C 阴天转小雨」的历史答案排在最前面，而 HISTORY_INSTRUCTION
# 当时写着「已包含答案就直接基于它作答，不要重复搜索」——模型照抄了成都的数据。
# 天气/价格/比分这类问题的答案**过期即错**，而历史注入没有任何时间感
# （旧条目甚至没有时间戳），所以对它们宁可完全不注入：不给历史顶多是少了个辅助，
# 给了历史则可能直接产出错误答案，还可能写回库里继续污染下一轮。
TIME_SENSITIVE_HINTS = (
    # 天气
    "天气", "气温", "温度", "下雨", "降雨", "暴雨", "台风", "阴晴", "空气质",
    "空气质量", "雾霾", "紫外线",
    # 时效
    "今天", "今日", "现在", "目前", "当前", "此刻", "最新", "实时", "刚刚",
    "本周", "这周", "今晚", "明天", "昨日", "昨天",
    # 行情 / 价格
    "股价", "股票", "汇率", "金价", "油价", "房价", "票价", "价格", "多少钱",
    "涨跌", "行情",
    # 赛事 / 动态
    "比分", "赛果", "赛程", "热搜", "新闻", "头条", "航班", "路况", "拥堵",
    # 英文
    "weather", "today", "latest", "current", "now", "news", "price",
    "stock", "score", "live", "real-time", "realtime",
)


def is_time_sensitive(task: str) -> bool:
    """任务是否在问「此刻的事实」。

    只做子串匹配，不追求精确：误判为时效性问题只是少注入一条历史
    （最多少一点辅助），漏判则可能把过期答案喂给模型并写回库里。
    所以这里宁可放宽。
    """
    text = str(task or "").lower()
    if not text.strip():
        return False
    return any(hint in text for hint in TIME_SENSITIVE_HINTS)

# ---- 用户偏好：「目录注入 + 按需加载」的预算 ----
# 目录是每轮都会进 prompt 的固定开销，所以它必须小而稳：
# 只放 key 和要点，完整取值由工具按需取（load_user_preferences）。
PREF_INDEX_MAX_ITEMS = 8      # 目录最多列几条
PREF_INDEX_VALUE_CHARS = 48   # 目录里每条要点最多多少字符
PREF_INDEX_MAX_CHARS = 620    # 整个目录块的字符上限
PREF_LOAD_MAX_ITEMS = 10      # 一次按需加载最多返回几条完整偏好
# 低于这个置信度的偏好不进目录：真实数据里出现过 language=english(0.1)
# 这种模型「猜的」条目，它既没被用户确认过，还会挤掉真正的 language=中文。
PREF_INDEX_MIN_CONFIDENCE = 0.3

# 这些 category 只是「没分类」的同义词，印在目录里纯属噪声。
_UNINFORMATIVE_CATEGORIES = {"", "general", "preference", "other", "misc"}


def _confidence_of(item):
    """取一条偏好的置信度；缺字段/脏值按 1.0 算（手写的偏好默认可信）。"""
    raw = item.get("confidence", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return min(max(value, 0.0), 1.0)


def _catalog_category(item, key):
    """目录里这一条该用哪个 category。

    只修「明显错」的两种写法（照抄 key、旧别名），**自定义标签原样保留**：
    ``category="project_notes"`` 是人/模型显式写进去的分类，读者靠它找东西，
    不能因为不在固定枚举里就被抹成 ``other``。
    """
    raw_key = str(item.get("memory_key") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    if not category or category == raw_key:
        # category=detail_degree（等于 key）说明根本没分类，按 key 的默认归类算
        return default_category_for(key)
    if category in CANONICAL_KEY_ALIASES or category in VALID_CATEGORIES:
        return canonical_category(category, key)
    return category


def _catalog_rows(items, min_confidence=PREF_INDEX_MIN_CONFIDENCE):
    """目录口径：按规范 key 归并同维度条目 + 丢掉低置信度噪声。

    为什么要在读取侧归并（而不是只靠写入侧）：

    - **存量数据纠不回来**。库里已经躺着 ``detail_degree：short``、
      ``detail_level：简短``、``response_style：简短`` 三条同义记录，
      写入侧改得再对，它们照样一起进 prompt，模型看到的偏好越多越糊。
    - **目录预算被吃掉**。同一个意思占三行，上限一满，真正新的偏好
      就被挤出去了。
    - **矛盾会直接打架**：``language：中文`` 和低置信度的
      ``language：english`` 并排出现时，模型只能瞎猜该听谁的。

    归并规则：同一规范 key 只保留 ``(置信度, updated_at)`` 最大的一条；
    顺序保持入参顺序（调用方已按「最近写入优先」排好），输出稳定。

    这里只做**归并**，不做语义纠错：早期版本还会拦「英文单词 + 该维度不认」
    的值（``answer_format：informal``），但那是替小模型猜语义、且有误伤
    （用户真说过 english），已删除。库里的怪值会原样出现在目录里，
    需要清理就用 ``forget`` 或直接改库。
    """
    best = {}
    order = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        confidence = _confidence_of(item)
        if confidence < min_confidence:
            continue
        key, value = canonicalize_preference(
            item.get("memory_key"), item.get("memory_value")
        )
        merged = dict(item)
        merged["memory_key"] = key
        merged["memory_value"] = value
        merged["confidence"] = confidence
        # category 一起归一：胜出的那条可能是 ``detail_degree`` 这种照抄 key
        # 的旧写法，原样印出来就是 ``- [detail_degree] answer_length：简短``。
        merged["category"] = _catalog_category(item, key)

        current = best.get(key)
        if current is None:
            best[key] = merged
            order.append(key)
            continue
        # 同 key 多条：置信度高的赢，平手时最近写入的赢
        if (confidence, str(merged.get("updated_at") or "")) > (
            current["confidence"],
            str(current.get("updated_at") or ""),
        ):
            best[key] = merged
    return [best[key] for key in order]


def _clip(text, limit):
    """压平空白并截断到 limit 字符，超长补省略号。"""
    text = " ".join(str(text or "").split())
    if not limit or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1] + "…"


def _format_recorded_at(value) -> str:
    """把 ISO 时间压成 ``2026-10-21 06:13``；解析不出就返回空串。

    历史块每轮都进 prompt，时间戳必须短且稳定——失败时宁可不显示，
    也不要把一长串 ISO 字符串塞进去。
    """
    raw = str(value or "").strip()
    if len(raw) < 16 or "T" not in raw:
        return ""
    return f"{raw[:10]} {raw[11:16]}"


class MemoryRetriever:
    def __init__(
        self,
        store=None,
        embedding_client=None,
    ):
        self.store = store or MemoryStore()
        self.embedding_client = embedding_client

    # ---------- 原有历史任务检索 ----------

    
    def retrieve(self, task: str, limit: int = 5) -> list[dict]:
        """关键词召回；成功记录优先，同状态按时间倒序。

        .. deprecated::
           这是「关键词 LIKE 整句」的旧实现：只有用户把老问题原样重打一遍
           才能命中，改写/指代（“刚才那个数是多少”）必然召回为空。
           生产路径请用 :meth:`retrieve_history`（向量相关性）。
           保留它只为兼容既有调用与测试。
        """
        if not task or not task.strip() or limit < 1:
            return []

        results = self.store.search_runs(
            task.strip(),
            limit=max(limit * 3, limit),
        )

        # Python 排序稳定：先按时间倒序，再按成功状态稳定排序。
        results.sort(
            key=lambda item: item.get("started_at") or "",
            reverse=True,
        )
        results.sort(
            key=lambda item: item.get("status") != "success"
        )

        return results[:limit]

    # ---------- 历史对话召回（向量相关性） ----------

    def retrieve_history(
        self,
        query: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
        min_similarity: float = HISTORY_MIN_SIMILARITY,
        scan_limit: int = HISTORY_SCAN_LIMIT,
    ) -> list[dict]:
        """按语义相关性召回历史问答。

        和旧的 ``retrieve()``（关键词 LIKE 整句）不同，这里把「当前任务」和
        「历史任务+历史回答」放进同一个向量空间比余弦相似度，
        所以“刚才那个数是多少”能命中“计算 123*456”。

        行为约束：
        - 只召回成功且有答案的 run（失败/中断的记录没有可复用事实）；
        - 只比较同一 embedding 模型的向量（换模型后旧向量不参与）；
        - 低于 ``min_similarity`` 一律丢弃，宁缺毋滥；
        - 同一句话被问过多次时只保留分最高的一条；
        - 返回条数硬上限 ``limit``，参与打分的候选硬上限 ``scan_limit``。

        没有 embedding 客户端（或服务不可用）时返回 ``[]``：
        偏好在检索失败时还能退化成「按时间取最新」，历史不能——
        不相关的历史被注入是纯噪声，比不注入更糟。
        """
        if not query or not query.strip() or limit < 1:
            return []

        # 时效性问题不注入历史：过期答案 + 「不要重复搜索」的旧措辞
        # 会让模型直接照抄上一轮的数据（真实踩过：成都的气温被当成广州的）。
        if is_time_sensitive(query):
            return []

        if self.embedding_client is None:
            return []

        try:
            query_vector = self.embedding_client.embed_query(
                query.strip()
            )
        except Exception as exc:  # ollama 没起 / 超时
            print(
                "[memory] 历史召回不可用，本次不注入历史："
                f"{type(exc).__name__}: {exc}"
            )
            return []

        candidates = self.store.get_run_embeddings(
            model=self.embedding_client.model,
            limit=scan_limit,
        )
        if not candidates:
            return []

        scored = []
        normalized_query = " ".join(str(query).split()).lower()
        for item in candidates:
            # 没有可复用内容的条目不进候选：只剩一句「回答了某某问题」的
            # 空摘要既占名额又误导（实测它会以最高相似度挤掉真正有用的历史）。
            if not (item.get("summary") or "").strip() and not (
                item.get("final_answer") or ""
            ).strip():
                continue

            # 与当前问题一字不差的记录不召回：要么是同一句话问了两遍，
            # 要么是上一轮刚写完、词向量还没变化——把它当「历史知识」
            # 注入只会得到「我上轮说过什么」这种循环自证。
            same_task = " ".join(str(item.get("task") or "").split()).lower()
            if same_task and same_task == normalized_query:
                continue

            similarity = self._cosine_similarity(
                query_vector, item["embedding"]
            )
            if similarity < min_similarity:
                continue
            item["similarity"] = similarity
            scored.append(item)

        scored.sort(
            key=lambda item: (
                item["similarity"],
                item.get("started_at") or "",
            ),
            reverse=True,
        )

        # 同一句话问过多次会命中多条重复记录，
        # 去重后才不会让 2 条预算被同一个问题占满。
        deduped = []
        seen = set()
        for item in scored:
            key = " ".join(str(item.get("task") or "").split()).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break

        return deduped

    @staticmethod
    def format_history(
        memories: list[dict],
        max_chars: int = HISTORY_MAX_CHARS,
        task_chars: int = HISTORY_TASK_CHARS,
        answer_chars: int = HISTORY_ANSWER_CHARS,
    ) -> str:
        """把召回的历史问答渲染成有硬字符预算的文本。

        两层预算：条数由调用方（``retrieve_history``）限制，
        字符数在这里截断，保证再长的历史回答也吃不掉上下文。
        预算耗尽立即停止，不会输出半条之后又接一条。
        """
        if not memories:
            return ""

        lines = []
        used = 0
        for item in memories:
            remaining = max_chars - used
            if remaining <= 0:
                break

            task = _clip(item.get("task"), task_chars)
            # 优先用提炼摘要：它是「这轮解决了什么」的浓缩版，
            # 同样 300 字符预算能表达更多信息；没有摘要才退回原始答案。
            answer = _clip(
                item.get("summary") or item.get("final_answer"),
                answer_chars,
            )
            # 带上记录时间：没有时间信息时，模型无法区分「三分钟前」
            # 和「上周」的答案，会把过期数据当现状复用。
            recorded = _format_recorded_at(item.get("started_at"))
            suffix = f"（记录于 {recorded}）" if recorded else ""
            line = f"- 当时的问题：{task}\n  当时的回答：{answer}{suffix}"

            if len(line) > remaining:
                line = line[: max(remaining - 1, 0)] + "…"

            lines.append(line)
            used += len(line) + 1  # +1 是行间换行

        return "\n".join(lines)

    # ---------- 用户偏好检索 ----------

    @staticmethod
    def _cosine_similarity(a, b):
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _structured_preferences(
        self,
        user_id: str,
        scope: str | None,
        limit: int,
    ) -> list[dict]:
        """不用向量的兜底：按更新时间取有效偏好。

        相关性排序会丢失，但至少让记忆继续生效，
        而不是因为向量服务抖动就整轮失忆。
        """
        return self.store.get_preferences(
            user_id=user_id,
            scope=scope,
        )[:limit]

    # ---------- 用户偏好：目录（按需加载的前半段） ----------

    def _scope_rows(self, user_id, scope, limit=None):
        """取有效偏好；scope 给定时把 global 一并算进来（与向量检索口径一致）。"""
        rows = self.store.get_preferences(user_id=user_id)
        if scope is not None:
            rows = [
                row for row in rows
                if row["scope"] in (scope, "global")
            ]
        return rows if limit is None else rows[:limit]

    def retrieve_preference_index(
        self,
        query: str = "",
        user_id: str = "default_user",
        scope: str | None = None,
        limit: int = PREF_INDEX_MAX_ITEMS,
    ) -> list[dict]:
        """便宜版偏好召回：只取「目录」，不调 embedding、不做相关性排序。

        为什么不复用 ``retrieve_preferences``：

        - **按需**：没有偏好时连一次 embedding 请求都不该付。先数条数，
          为 0 直接返回；这也让"没配 embedding 服务"的部署照样有目录可用。
        - **稳定**：目录每轮进 prompt，排序若随 query 漂移，对 prompt cache
          不友好，模型也难形成"我记得有这么一条"的印象。这里按
          「最近写入 → 被取用次数」排，query 只作为接口对称参数保留。
        - **可见**：最近学到/刚改过的偏好必须排在最前。目录是自动记忆
          唯一的注入入口，一旦它排在列表外，「这轮学到的东西」就永远
          进不了 prompt——学习闭环会断在最后一步（详见下面的排序注释）。
        - **省字**：值只保留要点（``PREF_INDEX_VALUE_CHARS``），
          完整内容交给 ``load_preferences`` 按需取。
        """
        if limit is None or limit < 1:
            return []

        limit = min(int(limit), PREF_INDEX_MAX_ITEMS)

        # 第一道闸：没有偏好就直接返回，不查详情、不碰向量服务。
        if self.store.count_active_preferences(user_id, scope=scope) < 1:
            return []

        rows = self._scope_rows(user_id, scope)
        # 排序：最近写入优先，被取用次数只做并列时的次序。
        #
        # 为什么不能以 use_count 为首键（曾经就是这么写的，结果是学习闭环断掉）：
        # ``MemoryStore.save_preference`` 覆盖同 key 时是「旧版本转 inactive +
        # 插入新行」，新行的 use_count 归 0，于是
        #   - 刚写入的偏好（use_count=0）永远排在最后，
        #     条数一超过目录上限就再也进不了 prompt；
        #   - 刚被改过值的偏好同样掉到最后，比没改的还靠后。
        # 也就是说：越新的记忆越看不见，和「动态更新」正好相反。
        #
        # 以 updated_at 为首键不会有这个问题，而且顺序依然是稳定的：
        # ``touch_preferences`` 故意不动 updated_at，光「被读过」不会改变
        # 目录顺序，只有这条偏好真的被写入（学到的新的、或值变了）才变化。
        rows.sort(
            key=lambda row: (
                row.get("updated_at") or "",
                row.get("use_count") or 0,
            ),
            reverse=True,
        )
        # 先归并/过滤，再截断：同义条目不该占掉目录名额
        return _catalog_rows(rows)[:limit]

    def count_preferences(
        self,
        user_id: str = "default_user",
        scope: str | None = None,
    ) -> int:
        """有效偏好总条数，口径与目录完全一致（scope 给定时含 global）。

        注意是「归并后」的条数：库里同一维度可能存着三条同义记录
        （``detail_degree：short`` / ``detail_level：简短`` …），目录只显示
        一条，总数也必须按一条算——否则「还有 N 条没列出来」里的 N 是虚的，
        模型会以为漏了很多东西。
        """
        return len(_catalog_rows(self._scope_rows(user_id, scope)))

    def load_preferences(
        self,
        query: str = "",
        keys=None,
        user_id: str = "default_user",
        scope: str | None = None,
        limit: int = PREF_LOAD_MAX_ITEMS,
    ) -> list[dict]:
        """按需加载偏好的完整内容（工具调用路径）。

        三种入口，优先级从高到低：

        1. ``keys``：点名要哪几条。不做相关性排序，也不能被阈值裁掉——
           模型说"我要看 language 这条"，就必须给它。
        2. ``query``：按语义找（走向量；服务不可用时退化为结构化）。
        3. 都没有：返回全部有效偏好（上限 ``limit``），相当于"列全部"。

        命中后记一次使用（use_count），供目录排序使用；计数失败不影响读取。
        """
        if limit is None or limit < 1:
            return []

        wanted = [
            str(key).strip()
            for key in (keys or [])
            if str(key).strip()
        ]

        if wanted:
            # 目录里显示的是规范 key（answer_length），库里可能还存着同义的
            # 旧 key（detail_degree）——点了目录上的名字必须也能命中。
            # 这里只做 key 同义收敛（value 传 None），不按值改判维度：
            # 点名要「哪一维」是明确指令，不该被值的内容改写成别的维度。
            wanted_norm = set()
            for key in wanted:
                wanted_norm.add(key.lower())
                alias, _ = canonicalize_preference(key, None)
                wanted_norm.add(str(alias).lower())

            matched = []
            for row in self._scope_rows(user_id, scope):
                row_key = str(row["memory_key"]).strip().lower()
                # 三种可能的叫法都算命中：原样 key、key 的规范写法（按别名）、
                # 以及「值本身就说明了维度」时目录里显示的那个 key
                # （存的是 response_style：简短，目录里显示 answer_length：简短）。
                alias, _ = canonicalize_preference(row["memory_key"], None)
                by_value, _ = canonicalize_preference(
                    row["memory_key"], row.get("memory_value")
                )
                candidates = {
                    row_key,
                    str(alias).strip().lower(),
                    str(by_value).strip().lower(),
                }
                if candidates & wanted_norm:
                    matched.append(row)
            # 点名要的条目不受置信度阈值拦截（min_confidence=0）：
            # 「我要看这条」是明确指令，哪怕它是模型猜的低置信偏好也要给。
            rows = _catalog_rows(matched, min_confidence=0.0)[:limit]
        elif query and query.strip():
            # 先按 key / category 做子串匹配：模型常常直接点名
            # 「看看 response_style 那条」，这类请求用关键词就够，
            # 不该为它付一次 embedding 请求，也不该被相似度阈值裁掉。
            needle = query.strip().lower()
            matched = [
                row for row in self._scope_rows(user_id, scope)
                if needle in str(row["memory_key"]).lower()
                or needle in str(row.get("category") or "").lower()
            ]
            if matched:
                rows = matched[:limit]
            else:
                rows = self.retrieve_preferences(
                    query=query,
                    user_id=user_id,
                    scope=scope,
                    limit=limit,
                )
        else:
            # 「列全部」按目录口径给：同维度只出一条、低置信度噪声不出现，
            # 否则模型点了目录里的 answer_length，加载后看到三条同义条目，
            # 又是一轮「到底听谁的」。
            rows = _catalog_rows(self._scope_rows(user_id, scope))[:limit]

        self._touch_preferences(rows)
        return rows

    def _touch_preferences(self, rows) -> None:
        """记录使用次数；纯统计信号，失败不打断读取。"""
        ids = [
            row.get("memory_id") for row in rows
            if row.get("memory_id") is not None
        ]
        if not ids:
            return
        try:
            self.store.touch_preferences(ids)
        except Exception as exc:  # 老库没迁移 / 并发写锁
            print(
                "[memory] 偏好使用计数更新失败（不影响本次读取）："
                f"{type(exc).__name__}: {exc}"
            )

    @staticmethod
    def format_preference_index(
        items: list[dict],
        max_chars: int = PREF_INDEX_MAX_CHARS,
        value_chars: int = PREF_INDEX_VALUE_CHARS,
    ) -> str:
        """把目录渲染成短行；值只留要点，超预算立即停。"""
        if not items:
            return ""

        lines = []
        used = 0
        # 再归并一次：format 也可能被直接调用（测试、外部拼装），
        # 不能假设上游已经去过重。对已归并的输入这一步是幂等的。
        for item in _catalog_rows(items):
            remaining = max_chars - used
            if remaining <= 0:
                break

            key = item.get("memory_key") or ""
            raw_value = item.get("memory_value")
            if not isinstance(raw_value, str):
                # 结构化值（dict/list）用 JSON 展示，避免 Python repr 的引号混进 prompt；
                # sort_keys 保证同样的值每次渲染成同一串，目录内容才稳定
                raw_value = json.dumps(
                    raw_value, ensure_ascii=False, sort_keys=True
                )
            value = _clip(raw_value, value_chars)
            category = str(item.get("category") or "").strip().lower()
            # category 只在它真的带来额外信息时才印：
            # 等于「没分类」、照抄 key、或正好是该 key 的默认归类时都省掉——
            # 否则每行都以 `- [response_style]` 开头，白占预算还干扰阅读。
            if (
                category in _UNINFORMATIVE_CATEGORIES
                or category == str(key).strip().lower()
                or category == default_category_for(key)
            ):
                line = f"- {key}：{value}"
            else:
                line = f"- [{category}] {key}：{value}"

            if len(line) > remaining:
                line = line[: max(remaining - 1, 0)] + "…"

            lines.append(line)
            used += len(line) + 1

        return "\n".join(lines)

    def retrieve_preferences(
        self,
        query: str,
        user_id: str = "default_user",
        scope: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if not query or not query.strip() or limit < 1:
            return []

        # 没有 Embedding 客户端时退化为读取结构化偏好。
        if self.embedding_client is None:
            return self._structured_preferences(
                user_id, scope, limit
            )

        # 客户端创建成功不代表服务可用（构造过程不联网），
        # 因此这里必须捕获运行期失败并回退，否则一次服务抖动
        # 就会让偏好完全不被注入。
        try:
            query_vector = self.embedding_client.embed_query(
                query.strip()
            )
            memories = self.store.get_embeddings(
                user_id=user_id,
                scope=scope,
            )
        except Exception as exc:
            print(
                "[memory] 向量检索不可用，本次回退到结构化偏好："
                f"{type(exc).__name__}: {exc}"
            )
            return self._structured_preferences(
                user_id, scope, limit
            )

        scored = []
        for item in memories:
            if item["embedding_model"] != self.embedding_client.model:
                continue

            similarity = self._cosine_similarity(
                query_vector, item["embedding"]
            )

            item["similarity"] = similarity
            scored.append(item)

        if not scored:
            # 有偏好但没有当前模型的向量：换过 embedding 模型，
            # 或写入时服务不可用导致向量缺失。降级而不是静默返回空。
            fallback = self._structured_preferences(
                user_id, scope, limit
            )
            if fallback:
                print(
                    "[memory] 未找到 "
                    f"{self.embedding_client.model} 的向量，"
                    "本次回退到结构化偏好；"
                    "如需向量检索请执行 rebuild_embeddings()"
                )
            return fallback

        scored.sort(
            key=lambda item: (
                item["similarity"],
                item["confidence"],
                item["updated_at"],
            ),
            reverse=True,
        )

        return scored[:limit]

    @staticmethod
    def format_preferences(memories: list[dict]) -> str:
        if not memories:
            return "没有检索到相关的用户偏好。"

        lines = [
            "以下是与当前任务相关的用户偏好，"
            "仅在适用时使用："
        ]

        for item in memories:
            lines.append(
                f"- 类别：{item['category']}\n"
                f"  作用域：{item['scope']}\n"
                f"  偏好：{item['memory_key']} = "
                f"{item['memory_value']}"
            )

        return "\n".join(lines)

    @staticmethod
    def format_context(memories: list[dict]) -> str:
        """格式化历史任务记录，保留原有接口。"""
        if not memories:
            return "没有检索到相关历史任务。"

        lines = ["以下是检索到的历史任务记录，仅供参考："]

        for item in memories:
            lines.append(
                f"- 历史任务：{item['task']}\n"
                f"  状态：{item['status']}\n"
                f"  历史结果：{item.get('final_answer') or '无'}"
            )

        return "\n".join(lines)
