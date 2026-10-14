
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

    def retrieve_preferences(
        self,
        query: str,
        user_id: str = "default_user",
        scope: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if not query or not query.strip() or limit < 1:
            return []

        # 无 Embedding 服务时退化为读取结构化偏好。
        if self.embedding_client is None:
            return self.store.get_preferences(
                user_id=user_id,
                scope=scope,
            )[:limit]

        query_vector = self.embedding_client.embed_query(
            query.strip()
        )
        memories = self.store.get_embeddings(
            user_id=user_id,
            scope=scope,
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

    def format_context(self, memories: list[dict]) -> str:
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
