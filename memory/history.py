"""历史对话索引：把已完成的 run 建成向量，供后续对话做语义召回。

与 preference_manager 的分工：
- preference_manager 管「用户偏好」的向量写入；
- history 管「历史问答」的向量写入。

两者共用同一个 EmbeddingClient（.env 里的 ollama / nomic-embed-text），
所以召回时能用同一套向量空间比较「当前任务」和「历史任务」的语义距离。
"""

from memory.store import MemoryStore

# 建向量时喂给 embedding 的文本上限：
# 答案动辄几千字，整段塞进去会把向量稀释成「什么都像」，
# 也会让每次 embedding 请求变慢，得不偿失。
INDEX_TASK_CHARS = 200
INDEX_ANSWER_CHARS = 500
EMBEDDING_VERSION = "v1"


class HistoryIndex:
    """给 agent_runs 建向量，并支持历史数据补建。"""

    def __init__(
        self,
        store=None,
        embedding_client=None,
    ):
        self.store = store or MemoryStore()
        self.embedding_client = embedding_client

    @staticmethod
    def _index_text(task, final_answer):
        """拼出用于向量化的文本：问题权重高于答案，答案只取开头。"""
        task_text = " ".join(str(task or "").split())[:INDEX_TASK_CHARS]
        answer_text = " ".join(
            str(final_answer or "").split()
        )[:INDEX_ANSWER_CHARS]
        return f"用户问题：{task_text}\n助手回答：{answer_text}"

    def index_run(self, run_id, task, final_answer):
        """给一次已完成的任务建向量。

        失败只返回 False，不抛异常：答案已经落库，
        向量缺失是「下次召回弱一点」，不该让这一轮回答变成错误。
        """
        if self.embedding_client is None:
            return False
        # 没有答案的 run 建向量没有意义：召回时只会往 prompt 里塞
        # 一句「当时的回答：」，纯噪声，不如不索引。
        if not str(task or "").strip():
            return False
        if not str(final_answer or "").strip():
            return False

        text = self._index_text(task, final_answer)

        try:
            vector = self.embedding_client.embed_query(text)
            self.store.save_run_embedding(
                run_id=run_id,
                embedding=vector,
                model=self.embedding_client.model,
                version=EMBEDDING_VERSION,
            )
        except Exception as exc:  # ollama 没起 / 超时 / 维度异常
            print(
                "[memory] 历史向量写入失败"
                f"（run_id={run_id}）：{type(exc).__name__}: {exc}"
            )
            return False

        return True

    def backfill(self, limit=200):
        """给还没有当前模型向量的历史 run 补建向量。

        只在启动时跑一次；limit 限制单次补建条数，
        避免历史很长的库把启动拖成几分钟。
        """
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "model": None,
        }
        if self.embedding_client is None:
            return stats

        stats["model"] = self.embedding_client.model
        pending = self.store.get_runs_missing_embedding(
            model=self.embedding_client.model,
            limit=limit,
        )
        stats["total"] = len(pending)

        for run in pending:
            ok = self.index_run(
                run_id=run["run_id"],
                task=run["task"],
                final_answer=run["final_answer"],
            )
            stats["success" if ok else "failed"] += 1

        return stats


__all__ = [
    "HistoryIndex",
    "INDEX_TASK_CHARS",
    "INDEX_ANSWER_CHARS",
    "EMBEDDING_VERSION",
]
