"""PreferenceManager：写入优先、向量失败可恢复、重建向量。"""

import pytest

from memory.preference_manager import (
    PreferenceBudgetError,
    PreferenceManager,
    PreferenceSkipped,
    canonical_category,
    canonicalize_preference,
    normalize_preference,
)
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


# ---------- 动态更新：upsert / 预算 / 按 key 遗忘 ----------

def _manager(store, client=None, **kwargs):
    return PreferenceManager(
        store=store, embedding_client=client, **kwargs
    )


def test_upsert_same_key_overwrites_instead_of_appending(store):
    manager = _manager(store)

    manager.upsert("u", "style", "回答简洁")
    manager.upsert("u", "style", "回答详细")

    active = manager.list_preferences("u")
    assert len(active) == 1
    assert active[0]["memory_value"] == "回答详细"
    # 旧值保留为历史版本（is_active=0），不是被物理删除
    assert store.count_active_preferences("u") == 1


def test_upsert_rejects_blank_key_or_value(store):
    manager = _manager(store)

    for key, value in (("", "v"), ("k", ""), ("   ", "v"), ("k", "   ")):
        with pytest.raises(ValueError):
            manager.upsert("u", key, value)


def test_upsert_enforces_char_budget_with_actionable_message(store):
    """超预算必须报错（而不是静默截断），且错误信息要能指导下一步。"""
    manager = _manager(store, char_limit=20)
    manager.upsert("u", "k1", "0123456789")  # key+value = 12 字符

    with pytest.raises(PreferenceBudgetError) as err:
        manager.upsert("u", "k2", "0123456789")

    message = str(err.value)
    assert "20" in message          # 上限
    assert "forget_user_preference" in message  # 怎么腾空间
    assert "k1" in message          # 现有哪些条目可以删


def test_upsert_budget_counts_overwrite_as_not_new(store):
    """覆盖同一条不新增占用，否则写满之后连改错别字都会被拒。"""
    manager = _manager(store, char_limit=30)
    manager.upsert("u", "k1", "0123456789")
    manager.upsert("u", "k1", "0123456789")
    manager.upsert("u", "k1", "0123456789")

    assert manager.budget_status("u")["items"] == 1


def test_upsert_enforces_item_count_limit(store):
    manager = _manager(store, max_items=2)
    manager.upsert("u", "k1", "v1")
    manager.upsert("u", "k2", "v2")

    with pytest.raises(PreferenceBudgetError, match="条数"):
        manager.upsert("u", "k3", "v3")


def test_budget_status_reports_usage(store):
    manager = _manager(store, char_limit=100, max_items=5)
    manager.upsert("u", "ab", "cde")  # 5 字符

    status = manager.budget_status("u")

    assert status["items"] == 1
    assert status["used_chars"] == 5
    assert status["remaining_chars"] == 95
    assert status["ok"] is True


def test_forget_removes_by_key(store):
    manager = _manager(store)
    manager.upsert("u", "style", "回答简洁")

    assert manager.forget("u", "style") is True
    assert manager.list_preferences("u") == []
    assert manager.forget("u", "style") is False


def test_forget_falls_back_to_other_scopes(store):
    """偏好记在 proj 作用域、按 global 删也要能删掉（避免静默失败）。"""
    manager = _manager(store)
    manager.upsert("u", "style", "回答简洁", scope="proj")

    assert manager.forget("u", "style") is True
    assert store.count_active_preferences("u") == 0


def test_forget_ignores_blank_key(store):
    assert _manager(store).forget("u", "  ") is False


def test_write_path_still_works_without_embedding_client(store):
    """向量服务没起时，动态写入只丢向量、不丢数据。"""
    manager = _manager(store, client=None)

    manager.upsert("u", "style", "回答简洁")

    assert [p["memory_value"] for p in manager.list_preferences("u")] == ["回答简洁"]
    assert store.get_embeddings("u") == []


def test_upsert_writes_vector_when_client_available(store):
    manager = _manager(store, client=FakeEmbeddingClient(model="m-up"))

    manager.upsert("u", "style", "回答简洁")

    rows = store.get_embeddings("u")
    assert [row["embedding_model"] for row in rows] == ["m-up"]


# ---------- 规范化：同义 key 收敛、低置信度不许顶掉高置信度 ----------

def test_canonicalize_preference_collapses_detail_dimension():
    """`detail_level` / `detail_degree` / `verbosity` 是同一件事。"""
    for key in ("detail_level", "detail_degree", "verbosity", "response_length"):
        assert canonicalize_preference(key, "回答简短一点")[0] == "answer_length"


def test_canonicalize_preference_does_not_guess_key_from_value():
    """只按 key 归一，**不再看值猜维度**。

    早期版本会把 ``response_style：简短`` 改写成 ``answer_length：简短``——
    等于替模型猜语义，猜错方向就悄悄改写了用户真的想设的东西。
    现在错位的 key 原样保留，要纠就改库或用 forget 重写。
    """
    assert canonicalize_preference("response_style", "简短")[0] == "response_style"
    assert canonicalize_preference("answer_format", "可爱风格")[0] == "answer_format"
    # 值归一仍然生效，只是不再反过来改 key
    assert canonicalize_preference("verbosity", "short") == ("answer_length", "简短")


def test_canonicalize_preference_leaves_compound_values_alone():
    """复合值不硬塞维度：那会丢信息。"""
    key, value = canonicalize_preference("style", "回答简洁，先给可运行代码")
    assert key == "style"
    assert value == "回答简洁，先给可运行代码"


def test_canonical_category_keeps_explicit_tooling():
    assert canonical_category("tooling", "python_cmd") == "tooling"
    assert canonical_category("detail_degree", "answer_length") == "response_style"
    assert canonical_category("preference", "language") == "response_style"
    assert canonical_category("", "python_cmd") == "other"


def test_normalize_preference_bundles_key_value_category():
    """归一化是纯函数：不碰数据库，工具层/排障脚本都能单独预演。"""
    assert normalize_preference("detail_level", "short", "detail_level") == (
        "answer_length",
        "简短",
        "response_style",
    )


def test_upsert_refuses_low_confidence_downgrade(store):
    """回归：confidence=0.1 的猜测不许顶掉用户明确说过的偏好。

    真实事故：自动提炼写了一条 language=english(0.1)，把
    language=中文(0.8) 顶成失效，之后每轮 prompt 都写着 english。
    """
    manager = PreferenceManager(store=store, embedding_client=None)
    kept_id = manager.upsert("u", "language", "中文", confidence=0.8)

    with pytest.raises(PreferenceSkipped):
        manager.upsert("u", "language", "english", confidence=0.1)

    assert [p["memory_value"] for p in manager.list_preferences("u")] == ["中文"]


def test_upsert_allows_newer_high_confidence_value(store):
    """置信度不降时照常覆盖：闸门只挡「猜的」，不挡「用户改主意」。"""
    manager = PreferenceManager(store=store, embedding_client=None)
    manager.upsert("u", "language", "中文", confidence=0.8)

    manager.upsert("u", "language", "英文", confidence=0.9)

    assert [p["memory_value"] for p in manager.list_preferences("u")] == ["英文"]


def test_upsert_canonicalizes_key_before_dedup(store):
    """写侧归一化下沉到 upsert：同义 key 只占一行，覆盖而不是并列。

    自动提炼移除后，写侧只剩 upsert 一个入口；不在这里归一，
    ``detail_level`` 和 ``answer_length`` 会各存一行、互不覆盖，
    目录里看着两条、模型能用的只有一个维度。
    """
    manager = PreferenceManager(store=store, embedding_client=None)
    manager.upsert("u", "answer_length", "简短")

    manager.upsert("u", "detail_level", "详细")

    rows = manager.list_preferences("u")
    assert len(rows) == 1
    assert rows[0]["memory_key"] == "answer_length"
    assert rows[0]["memory_value"] == "详细"
