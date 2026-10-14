import json
from memory.store import MemoryStore
from memory.embedding import EmbeddingClient


class PreferenceManager:
    def __init__(
        self,
        store=None,
        embedding_client=None,
    ):
        self.store = store or MemoryStore()
        self.embedding_client = embedding_client

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
        # 先保存权威数据，确保 Embedding API 故障
        # 不会导致已经识别的偏好丢失。
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

    