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
                    FOREIGN KEY (source_run_id)
                        REFERENCES agent_runs(run_id)
                        ON DELETE SET NULL
                )
            """)

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
        """新增或更新有效偏好；更新时保留旧版本。"""
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
                    created_at, updated_at, expires_at, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                user_id, category, memory_key, value_json,
                scope, confidence, source_run_id,
                now, now, expires_at
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
