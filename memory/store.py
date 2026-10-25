import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class MemoryStore:
    """负责任务历史、执行轨迹、用户偏好及向量的持久化。"""

    def __init__(self, db_path=None):
        project_root = Path(__file__).resolve().parent.parent
        if db_path is None:
            db_path = project_root / "data" / "agent_memory.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _ensure_columns(conn, table, columns):
        """幂等补列：只补缺失的列，已有列一律不动。

        SQLite 没有 ``ADD COLUMN IF NOT EXISTS``，直接 ALTER 会在
        第二次启动时抛 duplicate column。所以先查 table_info。
        """
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"
                )

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    final_answer TEXT,
                    error TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_steps (
                    run_id INTEGER NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_type TEXT NOT NULL,
                    step_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step_index),
                    FOREIGN KEY (run_id)
                        REFERENCES agent_runs(run_id)
                        ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_started_at
                ON agent_runs(started_at)
            """)

            # 结构化长期记忆
            # last_used_at / use_count 是「按需加载」的排序信号：
            # 被真正取用过的偏好排在目录前面，长期没人用的自然沉底。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
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
                    is_active INTEGER NOT NULL DEFAULT 1,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (source_run_id)
                        REFERENCES agent_runs(run_id)
                        ON DELETE SET NULL
                )
            """)

            # 老库补列：CREATE TABLE IF NOT EXISTS 不会改已存在的表，
            # 已有用户升级时必须走 ALTER，否则读写会直接报 no such column。
            self._ensure_columns(conn, "user_memories", {
                "last_used_at": "TEXT",
                "use_count": "INTEGER NOT NULL DEFAULT 0",
            })

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_lookup
                ON user_memories(
                    user_id, scope, is_active, category
                )
            """)

            # 同一用户、同一作用域下，同一个 key 只允许
            # 存在一条有效记忆；历史版本保留为 inactive。
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_key
                ON user_memories(user_id, scope, memory_key)
                WHERE is_active = 1
            """)

            # Embedding 与原始记忆分离存储
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedded_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id)
                        REFERENCES user_memories(memory_id)
                        ON DELETE CASCADE
                )
            """)

            # 历史对话的向量，与偏好向量分开建表：
            # memory_embeddings.memory_id 是 user_memories 的外键，
            # 把 run 的向量塞进去会撞外键，也会让两种记忆的语义混在一起。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_embeddings (
                    run_id INTEGER PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedded_at TEXT NOT NULL,
                    FOREIGN KEY (run_id)
                        REFERENCES agent_runs(run_id)
                        ON DELETE CASCADE
                )
            """)

            # 每轮对话的「提炼记忆」：摘要 + 可复用事实，供历史召回优先使用。
            # 与 agent_runs 的分工：
            # - agent_runs 存原始问答（原文，可能几千字）；
            # - turn_memories 存提炼后的摘要与事实（短，可直接进 prompt）。
            # run_id 主键保证「同一轮只能提炼一次」，天然幂等。
            # 现状：写入方只剩手工/排障脚本（每轮自动提炼已取消，偏好改由
            # 模型在任务收尾时自行调用 save_user_preference）；存量行仍会被
            # MemoryRetriever.format_history 优先使用。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turn_memories (
                    run_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL,
                    facts TEXT,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id)
                        REFERENCES agent_runs(run_id)
                        ON DELETE CASCADE
                )
            """)

    # ---------- 任务历史：保留原有接口 ----------

    def start_run(self, task):
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO agent_runs (task, started_at)
                VALUES (?, ?)
            """, (task, self._now()))
            return cursor.lastrowid

    def finish_run(
        self, run_id, final_answer, status="success", error=None
    ):
        if status not in {"success", "failed"}:
            raise ValueError("status 必须是 success 或 failed")

        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE agent_runs
                SET status = ?, finished_at = ?,
                    final_answer = ?, error = ?
                WHERE run_id = ?
            """, (
                status, self._now(), final_answer, error, run_id
            ))
            if cursor.rowcount == 0:
                raise ValueError(f"不存在 run_id={run_id} 的任务")

    def add_step(self, run_id, step_index, step_type, step_json):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO agent_steps
                    (run_id, step_index, step_type, step_json, saved_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                run_id, step_index, step_type,
                step_json, self._now()
            ))

    def get_recent_runs(self, limit=10):
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM agent_runs
                ORDER BY started_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]

    def search_runs(self, keyword, limit=10):
        if not keyword or not keyword.strip():
            return []
        if limit < 1:
            return []

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM agent_runs
                WHERE task LIKE ? OR final_answer LIKE ?
                ORDER BY started_at DESC LIMIT ?
            """, (
                f"%{keyword.strip()}%",
                f"%{keyword.strip()}%",
                limit
            )).fetchall()
            return [dict(row) for row in rows]

    def get_steps(self, run_id):
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM agent_steps
                WHERE run_id = ?
                ORDER BY step_index
            """, (run_id,)).fetchall()
            return [dict(row) for row in rows]

    # ---------- 用户偏好记忆 ----------

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
        """新增或更新有效偏好；更新时保留旧版本。

        覆盖语义是「旧行转 inactive + 插新行」，所以新行必须显式继承
        旧行的 ``use_count`` / ``last_used_at``：这两个字段描述的是
        「这条偏好」，而不是「这一行版本」。否则每改一次值，
        使用历史就清零一次，目录排序里刚更新的条目会掉到最后。
        """
        if not user_id.strip():
            raise ValueError("user_id 不能为空")
        if not memory_key.strip():
            raise ValueError("memory_key 不能为空")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")

        value_json = json.dumps(
            memory_value, ensure_ascii=False
        )
        now = self._now()

        with self._connect() as conn:
            # 覆盖同 key 时先把这个版本的使用计数接过来：use_count 记的是
            # 「这条偏好被取用过几次」，不是「这一行版本被取用过几次」。
            # 不继承的话，每改一次值就把计数清零，刚更新的偏好会在目录里
            # 掉到最后（目录上限一卡就再也看不见），而它恰恰是最该被看到的。
            previous = conn.execute("""
                SELECT use_count, last_used_at FROM user_memories
                WHERE user_id = ? AND scope = ?
                  AND memory_key = ? AND is_active = 1
            """, (user_id, scope, memory_key)).fetchone()
            carried_use_count = int(previous["use_count"] or 0) if previous else 0
            carried_last_used = previous["last_used_at"] if previous else None

            # 先让旧版本失效，避免唯一索引冲突。
            conn.execute("""
                UPDATE user_memories
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND scope = ?
                  AND memory_key = ? AND is_active = 1
            """, (now, user_id, scope, memory_key))

            cursor = conn.execute("""
                INSERT INTO user_memories (
                    user_id, category, memory_key, memory_value,
                    scope, confidence, source_run_id,
                    created_at, updated_at, expires_at, is_active,
                    use_count, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                user_id, category, memory_key, value_json,
                scope, confidence, source_run_id,
                now, now, expires_at,
                carried_use_count, carried_last_used,
            ))
            return cursor.lastrowid

    def get_preferences(
        self, user_id, scope=None, category=None
    ):
        """获取有效偏好；全局记忆不在这里自动合并项目记忆。"""
        now = self._now()
        sql = """
            SELECT * FROM user_memories
            WHERE user_id = ? AND is_active = 1
              AND (expires_at IS NULL OR expires_at > ?)
        """
        params = [user_id, now]

        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope)
        if category is not None:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["memory_value"] = json.loads(
                item["memory_value"]
            )
            results.append(item)
        return results

    def deactivate_preference(self, memory_id, user_id):
        """停用指定用户的一条偏好。"""
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE user_memories
                SET is_active = 0, updated_at = ?
                WHERE memory_id = ? AND user_id = ?
                  AND is_active = 1
            """, (self._now(), memory_id, user_id))
            return cursor.rowcount > 0

    def deactivate_preference_by_key(
        self, user_id, memory_key, scope="global"
    ):
        """按 (用户, 作用域, key) 停用偏好，返回是否命中。

        工具层是「模型给一个 key 让我忘掉」，模型手里只有 key，
        没有自增 id，所以必须提供按 key 停用的入口。
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE user_memories
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND scope = ?
                  AND memory_key = ? AND is_active = 1
            """, (self._now(), user_id, scope, memory_key))
            return cursor.rowcount > 0

    def count_active_preferences(self, user_id, scope=None):
        """数有效偏好条数。

        这是「按需加载」最便宜的那道闸：为 0 时上层直接不注入、
        也不该为此付一次 embedding 请求。
        """
        now = self._now()
        sql = """
            SELECT COUNT(*) AS n FROM user_memories
            WHERE user_id = ? AND is_active = 1
              AND (expires_at IS NULL OR expires_at > ?)
        """
        params = [user_id, now]

        if scope is not None:
            sql += " AND scope IN (?, 'global')"
            params.append(scope)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

    def touch_preferences(self, memory_ids):
        """记录「这条偏好被真正取用过」：use_count + 1，刷新 last_used_at。

        故意不动 ``confidence`` 和 ``updated_at``：用过的偏好不等于
        「刚被改写」，否则目录排序会把所有老记忆都顶到前面，
        真正的更新反而看不出来。
        """
        ids = []
        for item in memory_ids or []:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(f"""
                UPDATE user_memories
                SET use_count = use_count + 1, last_used_at = ?
                WHERE memory_id IN ({placeholders})
            """, [self._now(), *ids])
            return cursor.rowcount

    # ---------- Embedding 持久化 ----------

    def save_embedding(
        self, memory_id, embedding, model, version
    ):
        """新增或更新某条记忆的向量。"""
        embedding_json = json.dumps(embedding)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO memory_embeddings (
                    memory_id, embedding, embedding_model,
                    embedding_version, embedded_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    embedded_at = excluded.embedded_at
            """, (
                memory_id, embedding_json, model,
                version, self._now()
            ))

    def get_embeddings(self, user_id, scope=None):
        """只返回有效、未过期记忆的向量。"""
        now = self._now()
        sql = """
            SELECT m.*, e.embedding, e.embedding_model,
                   e.embedding_version
            FROM user_memories m
            JOIN memory_embeddings e
              ON m.memory_id = e.memory_id
            WHERE m.user_id = ? AND m.is_active = 1
              AND (m.expires_at IS NULL OR m.expires_at > ?)
        """
        params = [user_id, now]

        if scope is not None:
            sql += " AND m.scope IN (?, 'global')"
            params.append(scope)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["memory_value"] = json.loads(
                item["memory_value"]
            )
            item["embedding"] = json.loads(item["embedding"])
            results.append(item)
        return results

    # ---------- 历史对话向量 ----------

    def save_run_embedding(
        self, run_id, embedding, model, version
    ):
        """新增或更新某次任务的向量。"""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO run_embeddings (
                    run_id, embedding, embedding_model,
                    embedding_version, embedded_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    embedded_at = excluded.embedded_at
            """, (
                run_id, json.dumps(embedding), model,
                version, self._now()
            ))

    def get_run_embeddings(self, model=None, limit=200):
        """取出可以召回的历史任务向量。

        只返回「成功且有答案」的任务：失败/中断的 run 没有可复用的事实，
        召回它们只会把空字符串喂给模型。按时间倒序取最近 limit 条，
        让打分阶段的计算量有硬上限，而不是随库增长无限变慢。
        """
        if limit is not None and limit < 1:
            return []

        sql = """
            SELECT r.run_id, r.task, r.status, r.started_at,
                   r.final_answer,
                   e.embedding, e.embedding_model,
                   e.embedding_version,
                   t.summary
            FROM agent_runs r
            JOIN run_embeddings e ON r.run_id = e.run_id
            LEFT JOIN turn_memories t ON r.run_id = t.run_id
            WHERE r.status = 'success'
              AND r.final_answer IS NOT NULL
              AND TRIM(r.final_answer) <> ''
        """
        params = []

        if model is not None:
            sql += " AND e.embedding_model = ?"
            params.append(model)

        sql += " ORDER BY r.started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["embedding"] = json.loads(item["embedding"])
            results.append(item)
        return results

    def get_runs_missing_embedding(self, model=None, limit=200):
        """找出还没有当前模型向量的历史任务，用于启动时补建。

        换过 embedding 模型后旧向量会失效，这些 run 也会被重新算一遍。
        """
        if limit is not None and limit < 1:
            return []

        # model 为 None 表示「任意模型的向量都算已建」；
        # 指定 model 时，其他模型的旧向量视为缺失（换模型后需要重建）。
        if model is None:
            join_condition = "ON r.run_id = e.run_id"
            params = []
        else:
            join_condition = (
                "ON r.run_id = e.run_id "
                "AND e.embedding_model = ?"
            )
            params = [model]

        sql = f"""
            SELECT r.run_id, r.task, r.final_answer
            FROM agent_runs r
            LEFT JOIN run_embeddings e {join_condition}
            WHERE r.status = 'success'
              AND r.final_answer IS NOT NULL
              AND TRIM(r.final_answer) <> ''
              AND e.run_id IS NULL
            ORDER BY r.started_at DESC
        """

        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ---------- 每轮对话的提炼记忆 ----------

    def save_turn_memory(
        self, run_id, summary, facts=None, model=None
    ):
        """写入一轮的提炼记忆；run_id 主键使重复调用变成覆盖。

        facts 是「可复用事实」列表（数字、日期、结论），
        没有就用 NULL，避免把空数组和「没提炼」混在一起。
        """
        facts_json = None
        if facts:
            facts_json = json.dumps(
                facts, ensure_ascii=False
            )

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO turn_memories (
                    run_id, summary, facts, model, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    summary = excluded.summary,
                    facts = excluded.facts,
                    model = excluded.model,
                    created_at = excluded.created_at
            """, (
                run_id, str(summary), facts_json,
                model, self._now()
            ))

    def get_turn_memory(self, run_id):
        """取一轮的提炼记忆；没有返回 None。"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM turn_memories WHERE run_id = ?
            """, (run_id,)).fetchone()

        if row is None:
            return None

        item = dict(row)
        if item.get("facts"):
            item["facts"] = json.loads(item["facts"])
        else:
            item["facts"] = []
        return item

    def has_turn_memory(self, run_id):
        """这一轮是否已经提炼过（提炼器的幂等闸门）。"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT 1 FROM turn_memories WHERE run_id = ?
            """, (run_id,)).fetchone()
        return row is not None
