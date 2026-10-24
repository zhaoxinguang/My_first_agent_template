"""HistoryIndex：run 完成后建向量、失败不抛、backfill 只补缺失。"""

from memory.history import HistoryIndex
from tests.helpers import FakeSemanticEmbeddingClient


def _finished_run(store, task, answer="答"):
    run_id = store.start_run(task)
    store.finish_run(run_id, final_answer=answer, status="success")
    return run_id


def test_index_run_writes_vector(store):
    client = FakeSemanticEmbeddingClient()
    run_id = _finished_run(store, "查询天气", "晴")

    ok = HistoryIndex(store=store, embedding_client=client).index_run(
        run_id, "查询天气", "晴"
    )

    assert ok is True
    rows = store.get_run_embeddings(model=client.model)
    assert [row["run_id"] for row in rows] == [run_id]
    assert rows[0]["task"] == "查询天气"


def test_index_run_without_client_returns_false(store):
    run_id = _finished_run(store, "任务")

    assert HistoryIndex(store=store).index_run(run_id, "任务", "答") is False
    assert store.get_run_embeddings() == []


def test_index_run_requires_non_empty_task(store):
    client = FakeSemanticEmbeddingClient()
    run_id = _finished_run(store, "任务")

    assert (
        HistoryIndex(store=store, embedding_client=client).index_run(
            run_id, "   ", "答"
        )
        is False
    )


def test_index_run_swallows_embedding_failure(store):
    """ollama 没起时只丢向量，不抛异常、不影响已回答的结果。"""
    run_id = _finished_run(store, "任务")
    index = HistoryIndex(
        store=store, embedding_client=FakeSemanticEmbeddingClient(fail=True)
    )

    assert index.index_run(run_id, "任务", "答") is False
    assert store.get_run_embeddings() == []


def test_index_text_is_capped(store):
    """超长答案只取开头建向量，避免向量被稀释、请求变慢。"""
    client = FakeSemanticEmbeddingClient()
    run_id = _finished_run(store, "任务", "答" * 5000)

    HistoryIndex(store=store, embedding_client=client).index_run(
        run_id, "任务", "答" * 5000
    )

    assert len(client.calls[0]) < 800


def test_backfill_indexes_only_missing(store):
    client = FakeSemanticEmbeddingClient()
    index = HistoryIndex(store=store, embedding_client=client)
    first = _finished_run(store, "任务一")
    second = _finished_run(store, "任务二")
    index.index_run(first, "任务一", "答")

    stats = index.backfill()

    assert stats == {"total": 1, "success": 1, "failed": 0, "model": client.model}
    indexed = {row["run_id"] for row in store.get_run_embeddings(model=client.model)}
    assert indexed == {first, second}


def test_backfill_is_noop_when_nothing_pending(store):
    client = FakeSemanticEmbeddingClient()
    index = HistoryIndex(store=store, embedding_client=client)
    run_id = _finished_run(store, "任务")
    index.index_run(run_id, "任务", "答")

    assert index.backfill()["total"] == 0


def test_backfill_without_client(store):
    _finished_run(store, "任务")

    assert HistoryIndex(store=store).backfill() == {
        "total": 0,
        "success": 0,
        "failed": 0,
        "model": None,
    }
