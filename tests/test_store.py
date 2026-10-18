"""MemoryStore：建表、run 生命周期、偏好版本管理、向量 upsert。"""

import sqlite3

import pytest

from memory.store import MemoryStore

PAST = "2000-01-01T00:00:00+00:00"
FUTURE = "2999-01-01T00:00:00+00:00"


# ---------- 初始化 ----------

def test_creates_db_file_and_tables(tmp_path):
    db_path = tmp_path / "nested" / "memory.db"
    MemoryStore(db_path=db_path)

    assert db_path.is_file()  # 父目录会自动创建

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"agent_runs", "agent_steps", "user_memories", "memory_embeddings"} <= tables


def test_init_is_idempotent(tmp_path):
    db_path = tmp_path / "memory.db"
    MemoryStore(db_path=db_path)
    MemoryStore(db_path=db_path)  # 再次初始化不应报错或丢数据

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM agent_runs").fetchone()[0] == 0


# ---------- run 生命周期 ----------

def test_start_and_finish_run(store):
    run_id = store.start_run("查天气")
    store.finish_run(run_id, final_answer="晴", status="success")

    run = store.get_recent_runs(limit=1)[0]
    assert run["run_id"] == run_id
    assert run["task"] == "查天气"
    assert run["status"] == "success"
    assert run["final_answer"] == "晴"
    assert run["finished_at"] is not None


def test_run_defaults_to_running(store):
    store.start_run("查天气")

    assert store.get_recent_runs(limit=1)[0]["status"] == "running"


def test_finish_run_rejects_unknown_status(store):
    run_id = store.start_run("任务")

    with pytest.raises(ValueError, match="status"):
        store.finish_run(run_id, final_answer="", status="done")


def test_finish_run_rejects_unknown_run(store):
    with pytest.raises(ValueError, match="run_id"):
        store.finish_run(9999, final_answer="", status="success")


def test_finish_run_records_error(store):
    run_id = store.start_run("任务")
    store.finish_run(run_id, final_answer="", status="failed", error="boom")

    run = store.get_recent_runs(limit=1)[0]
    assert run["status"] == "failed"
    assert run["error"] == "boom"


def test_get_recent_runs_order_and_limit(store):
    first = store.start_run("第一条")
    second = store.start_run("第二条")

    runs = store.get_recent_runs(limit=2)

    assert [r["run_id"] for r in runs] == [second, first]


def test_get_recent_runs_validates_limit(store):
    with pytest.raises(ValueError, match="limit"):
        store.get_recent_runs(limit=0)


def test_search_runs_matches_task_and_answer(store):
    a = store.start_run("北京天气")
    store.finish_run(a, final_answer="晴", status="success")
    b = store.start_run("上海天气")
    store.finish_run(b, final_answer="北京今天也晴", status="success")

    found = {r["run_id"] for r in store.search_runs("北京")}

    assert found == {a, b}


def test_search_runs_empty_keyword(store):
    store.start_run("任意任务")

    assert store.search_runs("  ") == []
    assert store.search_runs("任意", limit=0) == []


# ---------- 步骤（当前未接线，但 API 契约要固定） ----------

def test_add_step_and_get_steps(store):
    run_id = store.start_run("任务")
    store.add_step(run_id, 0, "ActionStep", '{"step": 0}')
    store.add_step(run_id, 1, "ActionStep", '{"step": 1}')

    steps = store.get_steps(run_id)

    assert [s["step_index"] for s in steps] == [0, 1]
    assert steps[0]["step_json"] == '{"step": 0}'


def test_steps_are_deleted_with_run(store):
    run_id = store.start_run("任务")
    store.add_step(run_id, 0, "ActionStep", "{}")

    with store._connect() as conn:
        conn.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))

    assert store.get_steps(run_id) == []


# ---------- 偏好：校验 ----------

def test_save_preference_validates_user_id(store):
    with pytest.raises(ValueError, match="user_id"):
        store.save_preference("  ", "cat", "key", "值")


def test_save_preference_validates_key(store):
    with pytest.raises(ValueError, match="memory_key"):
        store.save_preference("u", "cat", "  ", "值")


def test_save_preference_validates_confidence(store):
    with pytest.raises(ValueError, match="confidence"):
        store.save_preference("u", "cat", "key", "值", confidence=1.5)


# ---------- 偏好：读写与版本 ----------

def test_save_and_get_preference(store):
    memory_id = store.save_preference(
        "u", "response_style", "detail", "回答简洁", confidence=0.9
    )

    rows = store.get_preferences("u")

    assert len(rows) == 1
    assert rows[0]["memory_id"] == memory_id
    assert rows[0]["memory_value"] == "回答简洁"  # JSON 反序列化回来了
    assert rows[0]["is_active"] == 1


def test_preference_value_roundtrip_for_non_string(store):
    store.save_preference("u", "cat", "key", {"语言": "中文", "长度": 3})

    assert store.get_preferences("u")[0]["memory_value"] == {"语言": "中文", "长度": 3}


def test_update_preference_keeps_old_version_inactive(store):
    old_id = store.save_preference("u", "cat", "key", "旧值")
    new_id = store.save_preference("u", "cat", "key", "新值")

    active = store.get_preferences("u")
    assert [r["memory_value"] for r in active] == ["新值"]
    assert active[0]["memory_id"] == new_id

    # 历史版本仍然留在库里，只是 is_active=0
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT memory_id, memory_value, is_active FROM user_memories "
            "WHERE user_id = 'u' ORDER BY memory_id"
        ).fetchall()
    assert [(r["memory_id"], r["is_active"]) for r in rows] == [
        (old_id, 0),
        (new_id, 1),
    ]


def test_same_key_in_different_scope_both_active(store):
    store.save_preference("u", "cat", "key", "全局值", scope="global")
    store.save_preference("u", "cat", "key", "项目值", scope="proj")

    assert len(store.get_preferences("u")) == 2
    assert store.get_preferences("u", scope="proj")[0]["memory_value"] == "项目值"


def test_same_key_for_different_users_is_independent(store):
    store.save_preference("u1", "cat", "key", "甲")
    store.save_preference("u2", "cat", "key", "乙")

    assert store.get_preferences("u1")[0]["memory_value"] == "甲"
    assert store.get_preferences("u2")[0]["memory_value"] == "乙"


def test_expired_preference_is_excluded(store):
    store.save_preference("u", "cat", "old", "过期", expires_at=PAST)
    store.save_preference("u", "cat", "new", "有效", expires_at=FUTURE)

    keys = {r["memory_key"] for r in store.get_preferences("u")}

    assert keys == {"new"}


def test_get_preferences_filter_by_category(store):
    store.save_preference("u", "format", "k1", "v1")
    store.save_preference("u", "style", "k2", "v2")

    rows = store.get_preferences("u", category="format")

    assert [r["memory_key"] for r in rows] == ["k1"]


def test_deactivate_preference(store):
    memory_id = store.save_preference("u", "cat", "key", "值")

    assert store.deactivate_preference(memory_id, "u") is True
    assert store.get_preferences("u") == []
    # 已失效的再删一次返回 False
    assert store.deactivate_preference(memory_id, "u") is False


def test_deactivate_preference_wrong_user(store):
    memory_id = store.save_preference("u", "cat", "key", "值")

    assert store.deactivate_preference(memory_id, "other") is False


# ---------- 向量 ----------

def test_save_embedding_upsert(store):
    memory_id = store.save_preference("u", "cat", "key", "值")

    store.save_embedding(memory_id, [1.0, 2.0], "m1", "v1")
    store.save_embedding(memory_id, [3.0, 4.0], "m2", "v1")

    rows = store.get_embeddings("u")
    assert len(rows) == 1  # 同一 memory_id 只保留一行
    assert rows[0]["embedding"] == [3.0, 4.0]
    assert rows[0]["embedding_model"] == "m2"


def test_get_embeddings_skips_inactive_and_expired(store):
    keep = store.save_preference("u", "cat", "keep", "v")
    gone = store.save_preference("u", "cat", "gone", "v")
    expired = store.save_preference("u", "cat", "expired", "v", expires_at=PAST)

    for memory_id in (keep, gone, expired):
        store.save_embedding(memory_id, [1.0], "m1", "v1")
    store.deactivate_preference(gone, "u")

    assert [r["memory_key"] for r in store.get_embeddings("u")] == ["keep"]


def test_get_embeddings_scope_includes_global(store):
    global_id = store.save_preference("u", "cat", "g", "全局", scope="global")
    proj_id = store.save_preference("u", "cat", "p", "项目", scope="proj")
    other_id = store.save_preference("u", "cat", "o", "别的项目", scope="other")

    for memory_id in (global_id, proj_id, other_id):
        store.save_embedding(memory_id, [1.0], "m1", "v1")

    keys = {r["memory_key"] for r in store.get_embeddings("u", scope="proj")}

    # 向量检索按「当前 scope + global」合并
    assert keys == {"g", "p"}


def test_embedding_removed_with_preference_row(store):
    memory_id = store.save_preference("u", "cat", "key", "值")
    store.save_embedding(memory_id, [1.0], "m1", "v1")

    with store._connect() as conn:
        conn.execute("DELETE FROM user_memories WHERE memory_id = ?", (memory_id,))

    assert store.get_embeddings("u") == []
