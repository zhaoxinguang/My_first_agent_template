"""每轮对话结束后的自动记忆提炼。

为什么不做成工具
----------------
工具要靠模型自己想起来调。本地小模型跑完一轮往往连 ``save_user_preference``
都不会碰——存偏好对「完成当前任务」没有即时收益，它没有理由调。
所以「提炼偏好 + 记录本轮」必须是**运行时保证**：每轮结束固定跑一次，
不依赖模型的自由裁量。工具只保留给「用户明确说『记住这条』」的场景。

产出两类记忆
------------
1. 用户偏好 → ``user_memories``：走 ``PreferenceManager.upsert``，
   带 ``source_run_id``（能追溯是哪一轮学到的）、带 ``confidence``；
2. 本轮记忆 → ``turn_memories``：summary（这轮解决了什么）+ facts（可复用事实），
   历史召回时优先用 summary 代替原始长答案。

可靠性约束（都是踩过的坑）
--------------------------
- **永不抛异常**：它在回答已经交付之后的后台线程里跑，抛了也没人接；
- **幂等**：同一个 run_id 只提炼一次（``turn_memories`` 主键兜底）；
- **预算满只丢一条偏好**：不因为一条写不进去就放弃整轮提炼；
- **输出必须能解析**：模型吐出非 JSON / 空内容时只跳过提炼，不影响已答内容；
- **偏好要短**：每条 ≤60 字，短到「注入目录里的要点就是全文」，
  这样小模型不需要为了一条偏好再调一次 ``load_user_preferences``。
"""

import json
import os
import re
import threading

from memory.preference_manager import (
    CANONICAL_KEY_ALIASES,
    PREF_MAX_ITEMS,
    PreferenceBudgetError,
    PreferenceManager,
    canonicalize_preference,
    canonical_category as _canonical_category,
    known_values_for,
)
from memory.store import MemoryStore

# 单轮最多提炼几条偏好：宁可少而准，多了会挤占预算且互相冲突
DEFAULT_MAX_PREFERENCES = 3
MAX_KEY_CHARS = 60
# 关键：必须小于 retriever.PREF_INDEX_VALUE_CHARS（48），
# 这样「目录里的要点」就是全文，小模型不必为了一条偏好再去调工具。
MAX_VALUE_CHARS = 40
MAX_CATEGORY_CHARS = 40
MAX_SUMMARY_CHARS = 400
MAX_FACTS = 5
MAX_FACT_CHARS = 120
# 喂给提炼模型的输入上限：长答案整段塞进去既慢又会把要点淹掉
MAX_INPUT_CHARS = 1500

EXTRACT_INSTRUCTION = """你是长期记忆提炼器。读下面这一轮问答，只输出一个 JSON 对象，
不要写解释、不要写 Markdown 代码块以外的任何文字。字段：

- "preferences": 数组，0~3 条。只收「下一轮对话仍然成立」的用户偏好：
  语言、回答格式、详细程度、称呼、禁忌、常用环境或工具。
  每条形如 {"key": "answer_length", "value": "简短",
  "category": "response_style", "confidence": 0.8}。
  key 只有一个维度、必须从这五个里选一个（不要自己造 key）：
  language（语言）/ answer_length（详细程度）/ answer_format（输出格式）/
  response_style（语气风格）/ tooling（工具环境）。
  同一维度只能有一条：已经有 answer_length 就不要再写 detail_level、
  detail_degree、verbosity，也不要重复记录「已知偏好」里已有的 key。
  value 必须简短到一眼能看懂——它会以这个长度直接进每轮对话的提示词。
  **value 用用户说话的语言写**：用户说中文就写中文（"中文"、"简短"、
  "可爱风格"），不要翻译成英文（不要写 english / short / informal / low）。
  **只有用户明确要求「用某种语言回答」时才收 language**：
  用户拿中文提问本身不算偏好，答案里夹了几个英文单词也不算——不要收。
  绝对不要收：任务进度、临时数据、一次性要求、助手自己的结论、用户随口问的内容。
  用户没有明确表达过的偏好一律不收——**不要从示例、答案里的英文单词、
  或"已知偏好"的写法去猜**。
  拿不准用户是否真的长期需要时，把 confidence 压到 0.3 以下（会被忽略）。
- "summary": 字符串，不超过 120 字，概括这一轮解决了什么、得到什么结论。
- "facts": 字符串数组，0~5 条以后可能复用的事实（数字、日期、结论、文件名）。

没有可提炼的内容时返回 {"preferences": [], "summary": "...", "facts": []}。
"""


def _known_preferences_block(rows):
    """把「已经记住的偏好」渲染成提示词里的一段，让提炼器复用而不是重造。

    不给这份清单，模型每轮都会重新起 key（detail_level / detail_degree /
    response_style 各来一条），目录越来越长、真正生效的却只有一两条。
    """
    if not rows:
        return "已知偏好：暂无。\n"
    lines = ["已知偏好（同维度请复用这些 key，不要重复添加）："]
    for row in rows:
        value = row.get("memory_value")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"- {row.get('memory_key')}：{_clip(value, MAX_VALUE_CHARS)}")
    return "\n".join(lines) + "\n"


def _env_flag(name, default=True):
    """读布尔环境变量；非法值回落到默认值。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# category 归一在 preference_manager 里实现（读侧也要用同一张表判断
# 「这个 category 值不值得印进目录」），这里只做别名导入。

# 「取值本来就是有限枚举」的维度。这些维度里出现白名单之外的纯英文单词，
# 几乎一定是模型把中文偏好翻译成了英文（真实数据里出现过
# answer_format：informal、detail_level：low 这种既不好读、也没法复用的值）。
_ENUM_DIMENSIONS = ("language", "answer_length", "answer_format", "response_style")
_ASCII_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z _-]{0,19}$")


def _looks_translated(key, value):
    """值是不是「英文单词 + 该维度不认」的翻译产物。"""
    if key not in _ENUM_DIMENSIONS or not isinstance(value, str):
        return False
    if not _ASCII_WORD_RE.match(value.strip()):
        return False
    known = known_values_for(key)
    return bool(known) and value.strip().lower() not in known


def _clip(text, limit):
    text = " ".join(str(text or "").split())
    if limit and len(text) > limit:
        return text[: max(limit - 1, 0)] + "…"
    return text


def _parse_json_object(text):
    """从模型输出里抠出第一个完整 JSON 对象。

    模型常见的三种脏输出都要能处理：整段 JSON、```json 围栏、
    前后带「好的，这是提炼结果：」这类话。
    """
    if not text:
        return None

    raw = str(text).strip()

    if "```" in raw:
        parts = raw.split("```")
        # 取最长的一段代码块内容，跳过语言标记行
        candidates = []
        for part in parts:
            block = part.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                candidates.append(block)
        if candidates:
            raw = max(candidates, key=len)

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(raw[start: end + 1])
    except (TypeError, ValueError):
        return None

    return data if isinstance(data, dict) else None


class TurnDistiller:
    """把一轮问答提炼成「偏好 + 本轮记忆」。"""

    def __init__(
        self,
        store=None,
        embedding_client=None,
        model=None,
        model_factory=None,
        enabled=None,
        max_preferences=DEFAULT_MAX_PREFERENCES,
    ):
        self.store = store or MemoryStore()
        self.embedding_client = embedding_client
        # model 用于测试注入；生产用 model_factory（延迟创建，避免启动即连模型）
        self._model = model
        self._model_factory = model_factory
        self.enabled = (
            _env_flag("PREF_AUTO_EXTRACT", True)
            if enabled is None
            else bool(enabled)
        )
        self.max_preferences = max_preferences
        self.manager = PreferenceManager(
            store=self.store,
            embedding_client=embedding_client,
        )

    # ---------- 对外入口 ----------

    def distill(self, run_id, task, final_answer):
        """提炼一轮；返回统计字典，**永不抛异常**。"""
        stats = {
            "run_id": run_id,
            "preferences_added": 0,
            "preferences_skipped": 0,
            "summary_saved": False,
            "skipped": None,
            "error": None,
        }

        if not self.enabled:
            stats["skipped"] = "disabled"
            return stats

        if not str(task or "").strip() or not str(final_answer or "").strip():
            stats["skipped"] = "empty_turn"
            return stats

        try:
            if self.store.has_turn_memory(run_id):
                stats["skipped"] = "already_distilled"
                return stats
        except Exception as exc:  # 老库缺表 / 锁
            stats["skipped"] = f"store_unavailable: {type(exc).__name__}"
            return stats

        try:
            data = self._extract(task, final_answer)
        except Exception as exc:  # 模型不可用、超时
            stats["skipped"] = "model_failed"
            stats["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[memory] 偏好提炼失败（run_id={run_id}）：{stats['error']}")
            return stats

        if data is None:
            stats["skipped"] = "bad_model_output"
            return stats

        # 1) 偏好：一条写不进去不影响其它条
        for pref in self._normalize_preferences(data.get("preferences")):
            try:
                self.manager.upsert(
                    user_id=self.user_id(),
                    memory_key=pref["key"],
                    memory_value=pref["value"],
                    category=pref["category"],
                    confidence=pref["confidence"],
                    source_run_id=run_id,
                )
                stats["preferences_added"] += 1
            except PreferenceBudgetError as exc:
                # 预算满：这是预期路径（偏好该被策展），不算错误
                stats["preferences_skipped"] += 1
                print(
                    f"[memory] 偏好预算已满，本轮跳过 {pref['key']}：{exc}"
                )
            except Exception as exc:
                stats["preferences_skipped"] += 1
                print(
                    "[memory] 偏好写入失败"
                    f"（run_id={run_id}, key={pref['key']}）："
                    f"{type(exc).__name__}: {exc}"
                )

        # 2) 本轮记忆：摘要一定落库（哪怕模型给的 summary 是空的），
        #    它同时是「已提炼」的标记，保证幂等。
        summary = _clip(
            data.get("summary") or final_answer, MAX_SUMMARY_CHARS
        )
        facts = self._normalize_facts(data.get("facts"))
        try:
            self.store.save_turn_memory(
                run_id=run_id,
                summary=summary,
                facts=facts,
                model=self._model_name(),
            )
            stats["summary_saved"] = True
        except Exception as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[memory] 本轮记忆写入失败（run_id={run_id}）：{stats['error']}"
            )

        return stats

    def distill_async(self, run_id, task, final_answer):
        """后台提炼：答案已经交付给用户了，不该让提炼拖慢这一轮。"""
        thread = threading.Thread(
            target=self._distill_safely,
            args=(run_id, task, final_answer),
            name=f"distill-{run_id}",
            daemon=True,
        )
        thread.start()
        return thread

    # ---------- 内部 ----------

    @staticmethod
    def user_id():
        # 单用户项目；接多用户时由调用方传入
        from memory.context import DEFAULT_USER_ID

        return DEFAULT_USER_ID

    def _distill_safely(self, run_id, task, final_answer):
        """线程里再兜一层：任何异常都不该冒到 threading 的默认 hook。"""
        try:
            self.distill(run_id, task, final_answer)
        except Exception as exc:  # pragma: no cover - 兜底
            print(
                "[memory] 自动提炼线程异常"
                f"（run_id={run_id}）：{type(exc).__name__}: {exc}"
            )

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory()
        else:
            from agent.model import create_model

            # 固定不思考：提炼只要一段 JSON，多出的思维链既慢又会被
            # num_predict / num_ctx 吃掉，反而更容易截断掉 JSON 本体。
            # 这里显式传 none，不跟随 AGENT_REASONING_EFFORT——
            # 用户把主任务调成 high 时，后台提炼不该跟着变慢。
            self._model = create_model(reasoning_effort="none")
        return self._model

    def _model_name(self):
        model = self._model
        return getattr(model, "model_id", None) or type(model).__name__

    def _extract(self, task, final_answer):
        """调模型拿结构化提炼结果；拿不到合法 JSON 时返回 None。"""
        known = _known_preferences_block(self._known_preferences())
        prompt = (
            f"{EXTRACT_INSTRUCTION}\n"
            f"---\n{known}"
            f"\n用户问题：\n{_clip(task, MAX_INPUT_CHARS)}\n\n"
            f"助手回答：\n{_clip(final_answer, MAX_INPUT_CHARS)}\n---"
        )

        response = self._get_model()(
            [{"role": "user", "content": prompt}]
        )
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)

        return _parse_json_object(content)

    def _known_preferences(self):
        """取当前有效偏好（读失败就当没有，绝不阻断提炼）。"""
        try:
            return self.store.get_preferences(user_id=self.user_id()) or []
        except Exception as exc:
            print(f"[memory] 读取已知偏好失败（提炼照常进行）：{type(exc).__name__}: {exc}")
            return []

    def _normalize_preferences(self, raw):
        """校验并裁剪模型给的偏好；不合规的整条丢掉，不做猜测。"""
        if not isinstance(raw, list):
            return []

        results = []
        # key → results 里的下标；同一维度出现多次时按置信度取优
        seen_keys = {}
        # 注意：条数上限按「通过的条数」算，不能先切片再校验——
        # 否则前几条不合规时，后面合规的会被无辜丢掉。
        for item in raw:
            if len(results) >= self.max_preferences and not seen_keys:
                break
            if not isinstance(item, dict):
                continue

            raw_key = str(item.get("key") or "").strip()
            # 超长 key 直接丢掉而不是截断：截断出来的 key
            # 既不可读也不可复用，下次没法按它覆盖，只会污染目录。
            if not raw_key or len(raw_key) > MAX_KEY_CHARS:
                continue

            key = _clip(raw_key, MAX_KEY_CHARS)
            value = _clip(item.get("value"), MAX_VALUE_CHARS)
            if not value:
                continue

            # 同义 key 收敛到固定维度（detail_degree / detail_level / verbosity
            # → answer_length…），否则同一件事会在目录里堆成好几条。
            key, value = canonicalize_preference(key, value)
            key = _clip(key, MAX_KEY_CHARS)
            value = _clip(value, MAX_VALUE_CHARS)
            if not key or not value:
                continue

            # 防幻觉：值不在该维度的白名单里、又是纯英文单词，就丢掉整条。
            # 这类值（informal / low）读起来像偏好、实际上是把中文偏好
            # 逐字翻译的产物，留着只会占目录名额。
            if _looks_translated(key, value):
                continue

            category = _clip(
                item.get("category") or "general", MAX_CATEGORY_CHARS
            )
            category = _canonical_category(category, key)

            try:
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = min(max(confidence, 0.0), 1.0)

            # 同一轮里同一个维度出现两次：留置信度高的那条，而不是留先出现的。
            # 小模型常常先吐一个泛化的猜测（language：english 0.9），
            # 再吐真正的偏好；先到先得会把前者留下。
            if key.lower() in seen_keys:
                previous = seen_keys[key.lower()]
                if confidence <= results[previous]["confidence"]:
                    continue
                results[previous] = {
                    "key": key,
                    "value": value,
                    "category": category,
                    "confidence": confidence,
                }
                continue
            if len(results) >= self.max_preferences:
                continue
            seen_keys[key.lower()] = len(results)

            results.append(
                {
                    "key": key,
                    "value": value,
                    "category": category,
                    "confidence": confidence,
                }
            )

        return results

    @staticmethod
    def _normalize_facts(raw):
        if not isinstance(raw, list):
            return []

        facts = []
        for item in raw[:MAX_FACTS]:
            text = _clip(item, MAX_FACT_CHARS)
            if text:
                facts.append(text)
        return facts


__all__ = [
    "TurnDistiller",
    "EXTRACT_INSTRUCTION",
    "DEFAULT_MAX_PREFERENCES",
    "PREF_MAX_ITEMS",
]
