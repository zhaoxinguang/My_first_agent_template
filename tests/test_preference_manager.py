"""PreferenceManager：写入优先、向量失败可恢复、重建向量。"""

import pytest

from memory.preference_manager import PreferenceManager
from memory.store import MemoryStore
from tests.helpers import FakeEmbeddingClient


def test_save_preference_persists_even_if_embedding_fails(store):
    """向量服务故障不能导致已识别的偏好丢失。"""
    manager = PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(fail=True)
    )

    memory_id = manager.save_preference("u", "cat", "key", "值")

    assert memory_id
    assert [p["memory_value"] for p in manager.list_preferences("u")] == ["值"]
    assert store.get_embeddings("u") == []  # 向量确实没写成


def test_save_preference_writes_vector_when_available(store):
    client = FakeEmbeddingClient(model="m1")
    manager = PreferenceManager(store=store, embedding_client=client)

    manager.save_preference("u", "cat", "key", "值")

    rows = store.get_embeddings("u")
    assert len(rows) == 1
    assert rows[0]["embedding_model"] == "m1"
    assert rows[0]["embedding_version"] == "v1"


def test_save_preference_skips_vector_without_client(store):
    manager = PreferenceManager(store=store, embedding_client=None)

    manager.save_preference("u", "cat", "key", "值")

    assert store.get_embeddings("u") == []


def test_embedding_text_is_stable_for_dict_values():
    text = PreferenceManager._embedding_text(
        "cat", "key", {"b": 2, "a": 1}, "global"
    )

    assert "cat" in text and "key" in text and "global" in text
    # sort_keys 保证同样的 dict 生成同样的文本，避免向量漂移
    assert '{"a": 1, "b": 2}' in text


def test_rebuild_embeddings_reports_success(store):
    manager = PreferenceManager(store=store, embedding_client=None)
    manager.save_preference("u", "cat", "k1", "v1")
    manager.save_preference("u", "cat", "k2", "v2")

    client = FakeEmbeddingClient(model="m2")
    report = PreferenceManager(
        store=store, embedding_client=client
    ).rebuild_embeddings("u")

    assert report["total"] == 2
    assert report["success"] == 2
    assert report["failed"] == 0
    assert report["model"] == "m2"
    assert {row["embedding_model"] for row in store.get_embeddings("u")} == {"m2"}


def test_rebuild_embeddings_collects_failures(store):
    manager = PreferenceManager(store=store, embedding_client=None)
    manager.save_preference("u", "cat", "k1", "v1")

    report = PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(fail=True)
    ).rebuild_embeddings("u")

    assert report["success"] == 0
    assert report["failed"] == 1
    assert report["errors"][0]["memory_id"]


def test_rebuild_embeddings_requires_client(store):
    with pytest.raises(RuntimeError, match="EmbeddingClient"):
        PreferenceManager(store=store, embedding_client=None).rebuild_embeddings("u")


def test_rebuild_embeddings_validates_limit(store):
    with pytest.raises(ValueError, match="limit"):
        PreferenceManager(
            store=store, embedding_client=FakeEmbeddingClient()
        ).rebuild_embeddings("u", limit=0)


def test_list_and_forget(store):
    manager = PreferenceManager(store=store, embedding_client=None)
    memory_id = manager.save_preference("u", "cat", "key", "值")

    assert len(manager.list_preferences("u")) == 1
    assert manager.forget_preference(memory_id, "u") is True
    assert manager.list_preferences("u") == []


def test_manager_uses_single_store_instance(tmp_path):
    """retriever/manager 共享同一个 store，避免多连接写不同库。"""
    store = MemoryStore(db_path=tmp_path / "m.db")
    manager = PreferenceManager(store=store, embedding_client=None)

    assert manager.store is store
