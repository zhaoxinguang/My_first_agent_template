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
    assert {"agent_runs", "agent_steps", "user_memories", "memory_embeddings", "run_embeddings"} <= tables


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


# ---------- 历史对话向量（run_embeddings） ----------

def _ok_run(store, task="查天气", answer="晴"):
    run_id = store.start_run(task)
    store.finish_run(run_id, final_answer=answer, status="success")
    return run_id


def test_save_run_embedding_upsert(store):
    run_id = _ok_run(store)

    store.save_run_embedding(run_id, [1.0, 2.0], "m1", "v1")
    store.save_run_embedding(run_id, [3.0, 4.0], "m2", "v1")

    rows = store.get_run_embeddings()
    assert len(rows) == 1  # 同一 run 只保留一行
    assert rows[0]["embedding"] == [3.0, 4.0]
    assert rows[0]["embedding_model"] == "m2"


def test_get_run_embeddings_joins_run_content(store):
    run_id = _ok_run(store, task="计算123*456", answer="56088")
    store.save_run_embedding(run_id, [1.0], "m1", "v1")

    row = store.get_run_embeddings()[0]

    assert row["run_id"] == run_id
    assert row["task"] == "计算123*456"
    assert row["final_answer"] == "56088"
    assert row["status"] == "success"


def test_run_embeddings_skip_failed_and_empty_answers(store):
    ok = _ok_run(store)
    failed = store.start_run("失败任务")
    store.finish_run(failed, final_answer="", status="failed")
    empty = _ok_run(store, task="空答案", answer="   ")

    for run_id in (ok, failed, empty):
        store.save_run_embedding(run_id, [1.0], "m1", "v1")

    # 失败/空答案的 run 没有可复用事实，不参与召回
    assert [r["run_id"] for r in store.get_run_embeddings()] == [ok]


def test_get_run_embeddings_filters_by_model(store):
    run_id = _ok_run(store)
    store.save_run_embedding(run_id, [1.0], "old-model", "v1")

    assert store.get_run_embeddings(model="old-model")
    assert store.get_run_embeddings(model="new-model") == []


def test_get_run_embeddings_limit_and_order(store):
    first = _ok_run(store, task="任务一")
    second = _ok_run(store, task="任务二")
    for run_id in (first, second):
        store.save_run_embedding(run_id, [1.0], "m1", "v1")

    rows = store.get_run_embeddings(limit=1)

    assert len(rows) == 1
    assert rows[0]["run_id"] == second  # 时间倒序，先拿最近的
    assert store.get_run_embeddings(limit=0) == []


def test_get_runs_missing_embedding(store):
    indexed = _ok_run(store, task="已建向量")
    pending = _ok_run(store, task="待建向量")
    store.save_run_embedding(indexed, [1.0], "current", "v1")

    missing = store.get_runs_missing_embedding(model="current")

    assert [r["run_id"] for r in missing] == [pending]


def test_get_runs_missing_embedding_treats_other_model_as_missing(store):
    run_id = _ok_run(store)
    store.save_run_embedding(run_id, [1.0], "old-model", "v1")

    # 换模型后旧向量视为缺失 → 需要重建
    assert [r["run_id"] for r in store.get_runs_missing_embedding(model="new-model")] == [run_id]
    # model=None 表示「任意模型都算已建」
    assert store.get_runs_missing_embedding(model=None) == []


def test_run_embedding_removed_with_run(store):
    run_id = _ok_run(store)
    store.save_run_embedding(run_id, [1.0], "m1", "v1")

    with store._connect() as conn:
        conn.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))

    assert store.get_run_embeddings() == []


# ---------- 偏好：按需加载需要的查询 ----------

def _pref(store, key="k", value="v", scope="global", user="u"):
    return store.save_preference(user, "cat", key, value, scope=scope)


def test_old_schema_gets_usage_columns(tmp_path):
    """老库（没有 last_used_at / use_count）升级后必须能正常读写。"""
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE user_memories (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                confidence REAL NOT NULL DEFAULT 1.0,
                source_run_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)

    upgraded = MemoryStore(db_path=db_path)  # 触发迁移
    upgraded.save_preference("u", "cat", "k", "v")

    rows = upgraded.get_preferences("u")
    assert rows[0]["use_count"] == 0
    assert rows[0]["last_used_at"] is None


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "m.db"
    MemoryStore(db_path=db_path)
    MemoryStore(db_path=db_path)  # 二次初始化不应抛 duplicate column


def test_count_active_preferences(store):
    _pref(store, key="k1")
    _pref(store, key="k2", scope="proj")

    assert store.count_active_preferences("u") == 2
    assert store.count_active_preferences("u", scope="proj") == 2  # 含 global
    # 换一个作用域时 global 仍然算数（与向量检索口径一致），proj 那条不算
    assert store.count_active_preferences("u", scope="other") == 1
    assert store.count_active_preferences("nobody") == 0


def test_count_active_preferences_skips_expired(store):
    _pref(store, key="old")
    with store._connect() as conn:
        conn.execute(
            "UPDATE user_memories SET expires_at = ? WHERE memory_key = 'old'",
            (PAST,),
        )

    assert store.count_active_preferences("u") == 0


def test_touch_preferences_increments_usage(store):
    memory_id = _pref(store)

    store.touch_preferences([memory_id])
    store.touch_preferences([memory_id])

    row = store.get_preferences("u")[0]
    assert row["use_count"] == 2
    assert row["last_used_at"] is not None
    # 用过的偏好不该被当成「刚被改写」，否则目录排序会把老记忆顶到最前
    assert row["updated_at"] < row["last_used_at"]


def test_touch_preferences_ignores_empty_input(store):
    _pref(store)

    assert store.touch_preferences([]) == 0
    assert store.touch_preferences(None) == 0


def test_overwrite_inherits_usage_count(store):
    """覆盖同 key 时 use_count 记的是「这条偏好」，不随版本清零。

    覆盖的实现是「旧行转 inactive + 插新行」，新行的 use_count 默认 0；
    不继承的话，改一次值就等于把使用历史抹掉，目录排序随之把
    刚更新的条目推到最末（甚至推出去）。
    """
    memory_id = _pref(store, key="style", value="简洁")
    store.touch_preferences([memory_id])
    store.touch_preferences([memory_id])

    new_id = _pref(store, key="style", value="详细")

    row = store.get_preferences("u")[0]
    assert row["memory_id"] == new_id
    assert row["use_count"] == 2
    assert row["last_used_at"] is not None


def test_first_write_has_zero_usage(store):
    """首次写入（没有旧版本）不该凭空继承出计数。"""
    _pref(store, key="fresh")

    row = store.get_preferences("u")[0]
    assert row["use_count"] == 0
    assert row["last_used_at"] is None


def test_deactivate_preference_by_key(store):
    _pref(store, key="k", scope="proj")

    assert store.deactivate_preference_by_key("u", "k", scope="proj") is True
    assert store.deactivate_preference_by_key("u", "k", scope="proj") is False
    assert store.count_active_preferences("u") == 0


# ---------- 每轮对话的提炼记忆 ----------

def test_save_and_get_turn_memory(store):
    run_id = _ok_run(store)

    store.save_turn_memory(
        run_id, summary="本轮解决了快排实现", facts=["复杂度 O(n log n)"], model="m1"
    )

    memory = store.get_turn_memory(run_id)
    assert memory["summary"] == "本轮解决了快排实现"
    assert memory["facts"] == ["复杂度 O(n log n)"]
    assert memory["model"] == "m1"
    assert store.has_turn_memory(run_id) is True


def test_turn_memory_upsert_keeps_single_row(store):
    run_id = _ok_run(store)
    store.save_turn_memory(run_id, summary="第一版")
    store.save_turn_memory(run_id, summary="第二版")

    assert store.get_turn_memory(run_id)["summary"] == "第二版"
    with store._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM turn_memories WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert count == 1


def test_turn_memory_absent_and_facts_default(store):
    run_id = _ok_run(store)
    assert store.get_turn_memory(run_id) is None
    assert store.has_turn_memory(run_id) is False

    store.save_turn_memory(run_id, summary="无事实")

    assert store.get_turn_memory(run_id)["facts"] == []


def test_turn_memory_removed_with_run(store):
    run_id = _ok_run(store)
    store.save_turn_memory(run_id, summary="摘要")

    with store._connect() as conn:
        conn.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))

    assert store.get_turn_memory(run_id) is None


def test_run_embeddings_expose_summary(store):
    """历史召回要能拿到摘要（用来替代原始长答案）。"""
    run_id = _ok_run(store)
    store.save_run_embedding(run_id, [1.0], "m1", "v1")
    store.save_turn_memory(run_id, summary="本轮摘要")

    rows = store.get_run_embeddings(model="m1")

    assert rows[0]["summary"] == "本轮摘要"
