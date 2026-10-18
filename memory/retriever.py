
import math

from memory.store import MemoryStore
from memory.embedding import EmbeddingClient


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
        """关键词召回；成功记录优先，同状态按时间倒序。"""
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
