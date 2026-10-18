"""MemoryRetriever：向量召回、模型过滤、两条降级路径、历史任务召回。"""

import pytest

from memory.retriever import MemoryRetriever
from tests.helpers import FakeEmbeddingClient


def _seed(store):
    store.save_preference("u", "response_style", "detail", "回答简洁分步骤")
    store.save_preference("u", "format", "language", "中文")


# ---------- 余弦相似度 ----------

def test_cosine_similarity_identical_vectors():
    # 浮点累计误差，用 approx 而不是精确相等
    assert MemoryRetriever._cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert MemoryRetriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_guards_dimension_mismatch():
    assert MemoryRetriever._cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_guards_zero_norm():
    assert MemoryRetriever._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_guards_empty():
    assert MemoryRetriever._cosine_similarity([], []) == 0.0


# ---------- 向量召回 ----------

def test_retrieve_preferences_uses_vectors_when_available(store):
    _seed(store)
    client = FakeEmbeddingClient()
    # 先按同一模型写入向量
    from memory.preference_manager import PreferenceManager

    PreferenceManager(store=store, embedding_client=client).save_preference(
        "u", "format", "language", "中文"
    )

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_preferences("中文", user_id="u")

    assert len(results) == 1
    assert "similarity" in results[0]
    assert client.calls  # 确实调用了 embedding 服务


def test_retrieve_preferences_respects_limit(store):
    for index in range(5):
        store.save_preference("u", "cat", f"key{index}", f"value{index}")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    assert len(retriever.retrieve_preferences("任意", user_id="u", limit=2)) == 2


# ---------- 降级路径 ----------

def test_falls_back_to_structured_without_client(store):
    _seed(store)

    retriever = MemoryRetriever(store=store, embedding_client=None)
    results = retriever.retrieve_preferences("任意", user_id="u")

    assert {r["memory_key"] for r in results} == {"detail", "language"}
    assert all("similarity" not in r for r in results)


def test_falls_back_when_embedding_service_fails(store):
    """ollama 没起时必须回退，而不是整轮失忆。"""
    _seed(store)
    retriever = MemoryRetriever(
        store=store, embedding_client=FakeEmbeddingClient(fail=True)
    )

    results = retriever.retrieve_preferences("任意", user_id="u")

    assert {r["memory_key"] for r in results} == {"detail", "language"}


def test_falls_back_when_model_mismatch(store):
    """换过 embedding 模型后旧向量被过滤，此时应回退而不是返回空。"""
    from memory.preference_manager import PreferenceManager

    PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(model="old-model")
    ).save_preference("u", "format", "language", "中文")

    retriever = MemoryRetriever(
        store=store, embedding_client=FakeEmbeddingClient(model="new-model")
    )
    results = retriever.retrieve_preferences("中文", user_id="u")

    assert [r["memory_key"] for r in results] == ["language"]


def test_vectors_from_other_model_are_not_mixed(store):
    """存在当前模型向量时，不能把别的模型的向量算进去。"""
    from memory.preference_manager import PreferenceManager

    PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(model="old-model")
    ).save_preference("u", "cat", "old", "旧模型向量")
    client = FakeEmbeddingClient(model="new-model")
    PreferenceManager(store=store, embedding_client=client).save_preference(
        "u", "cat", "new", "新模型向量"
    )

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_preferences("任意", user_id="u")

    assert [r["memory_key"] for r in results] == ["new"]


def test_empty_query_returns_empty(store):
    _seed(store)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert retriever.retrieve_preferences("   ", user_id="u") == []
    assert retriever.retrieve_preferences("任意", user_id="u", limit=0) == []


# ---------- 格式化 ----------

def test_format_preferences_empty():
    assert "没有检索到" in MemoryRetriever.format_preferences([])


def test_format_preferences_lists_each_item(store):
    _seed(store)
    retriever = MemoryRetriever(store=store, embedding_client=None)
    text = retriever.format_preferences(
        retriever.retrieve_preferences("任意", user_id="u")
    )

    assert "language" in text
    assert "中文" in text
    assert text.count("- 类别：") == 2


# ---------- 历史任务召回 ----------

def test_retrieve_history_prefers_success(store):
    failed = store.start_run("查天气")
    store.finish_run(failed, final_answer="失败了", status="failed")
    ok = store.start_run("查天气")
    store.finish_run(ok, final_answer="晴", status="success")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    results = retriever.retrieve("查天气")

    assert results[0]["status"] == "success"
    assert results[0]["run_id"] == ok


def test_retrieve_history_empty_keyword(store):
    retriever = MemoryRetriever(store=store, embedding_client=None)
    assert retriever.retrieve("  ") == []
    assert retriever.retrieve("任意", limit=0) == []


def test_format_context_of_history(store):
    run_id = store.start_run("查天气")
    store.finish_run(run_id, final_answer="晴", status="success")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    text = retriever.format_context(retriever.retrieve("查天气"))

    assert "查天气" in text
    assert "晴" in text


def test_format_context_empty():
    assert "没有检索到" in MemoryRetriever.format_context([])
