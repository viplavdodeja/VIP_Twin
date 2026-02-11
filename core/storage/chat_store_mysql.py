import time
import json
import uuid
from datetime import datetime

import pymysql
from pymysql.cursors import DictCursor


class ChatStoreMySQL:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _connect(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            cursorclass=DictCursor,
            autocommit=True
        )

    def init(self):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()

    def ensure_chat_session_exists(self, chat_id, user_id):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT IGNORE INTO chat_sessions "
                    "(chat_id, user_id, title, last_message_at) "
                    "VALUES (%s, %s, NULL, NULL)"
                )
                cur.execute(sql, (chat_id, user_id))
        except pymysql.err.IntegrityError as exc:
            msg = str(exc).lower()
            if "foreign key" in msg:
                raise RuntimeError(
                    "User not found in MySQL; create users row first "
                    "(user_id=" + str(user_id) + ", chat_id=" + str(chat_id) + ")"
                )
            raise
        finally:
            conn.close()

    def add_memory_item(self, user_id, chat_id, memory_type, role, content, created_at_dt, extra_json, memory_id=None):
        mem_id = memory_id
        if mem_id is None:
            mem_id = str(uuid.uuid4())
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO memory_items "
                    "(memory_id, user_id, chat_id, type, role, content, created_at, extra) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                )
                cur.execute(
                    sql,
                    (mem_id, user_id, chat_id, memory_type, role, content, created_at_dt, extra_json)
                )
                return mem_id
        finally:
            conn.close()

    def update_chat_last_message_at(self, chat_id, created_at_dt):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = "UPDATE chat_sessions SET last_message_at=%s WHERE chat_id=%s"
                cur.execute(sql, (created_at_dt, chat_id))
        finally:
            conn.close()

    def _dt_to_epoch(self, value):
        if value is None:
            return 0
        if isinstance(value, datetime):
            return int(time.mktime(value.timetuple()))
        return 0

    def get_recent_messages(self, user_id, chat_id, limit):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT role, content, created_at FROM memory_items "
                    "WHERE user_id=%s AND chat_id=%s AND type='chat_message' "
                    "ORDER BY created_at DESC LIMIT %s"
                )
                cur.execute(sql, (user_id, chat_id, limit))
                rows = list(cur.fetchall())
        finally:
            conn.close()

        rows = list(reversed(rows))
        items = []
        i = 0
        while i < len(rows):
            row = rows[i]
            created_at_epoch = self._dt_to_epoch(row.get("created_at"))
            items.append(
                {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "memory_type": "chat_message",
                    "role": row.get("role", ""),
                    "text": row.get("content", ""),
                    "created_at": created_at_epoch
                }
            )
            i += 1
        return items

    def get_last_n_messages(self, user_id, chat_id, n):
        return self.get_recent_messages(user_id, chat_id, n)

    def count_messages(self, user_id, chat_id):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT COUNT(*) AS cnt FROM memory_items "
                    "WHERE user_id=%s AND chat_id=%s AND type='chat_message'"
                )
                cur.execute(sql, (user_id, chat_id))
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return 0
        return int(row.get("cnt", 0))

    def _safe_json_loads(self, value):
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if len(text) == 0:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return None

    def get_session_memory(self, user_id, session_id):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT rolling_summary, active_threads, open_loops, short_term_prefs "
                    "FROM session_memory WHERE user_id=%s AND session_id=%s LIMIT 1"
                )
                cur.execute(sql, (user_id, session_id))
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return {
                "rolling_summary": "",
                "active_threads": None,
                "open_loops": None,
                "short_term_prefs": None
            }

        return {
            "rolling_summary": row.get("rolling_summary") or "",
            "active_threads": self._safe_json_loads(row.get("active_threads")),
            "open_loops": self._safe_json_loads(row.get("open_loops")),
            "short_term_prefs": self._safe_json_loads(row.get("short_term_prefs"))
        }

    def upsert_session_memory(
        self,
        user_id,
        session_id,
        rolling_summary,
        active_threads=None,
        open_loops=None,
        short_term_prefs=None
    ):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO session_memory "
                    "(session_id, user_id, rolling_summary, active_threads, open_loops, short_term_prefs, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, NOW()) "
                    "ON DUPLICATE KEY UPDATE "
                    "rolling_summary=VALUES(rolling_summary), "
                    "active_threads=VALUES(active_threads), "
                    "open_loops=VALUES(open_loops), "
                    "short_term_prefs=VALUES(short_term_prefs), "
                    "updated_at=NOW()"
                )
                cur.execute(
                    sql,
                    (
                        session_id,
                        user_id,
                        rolling_summary,
                        json.dumps(active_threads) if active_threads is not None else None,
                        json.dumps(open_loops) if open_loops is not None else None,
                        json.dumps(short_term_prefs) if short_term_prefs is not None else None
                    )
                )
        finally:
            conn.close()

    def insert_claim(self, user_id, claim_type, claim_text, confidence, status="active"):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO memory_claims "
                    "(user_id, claim_type, claim_text, confidence, last_seen_at, status, superseded_by_id) "
                    "VALUES (%s, %s, %s, %s, NOW(), %s, NULL)"
                )
                cur.execute(sql, (user_id, claim_type, claim_text, confidence, status))
                return cur.lastrowid
        finally:
            conn.close()

    def update_claim(self, claim_id, claim_text, confidence):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "UPDATE memory_claims "
                    "SET claim_text=%s, confidence=%s, last_seen_at=NOW() "
                    "WHERE id=%s"
                )
                cur.execute(sql, (claim_text, confidence, claim_id))
        finally:
            conn.close()

    def touch_claim(self, claim_id, confidence=None):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if confidence is None:
                    sql = "UPDATE memory_claims SET last_seen_at=NOW() WHERE id=%s"
                    cur.execute(sql, (claim_id,))
                else:
                    sql = (
                        "UPDATE memory_claims SET last_seen_at=NOW(), confidence=%s WHERE id=%s"
                    )
                    cur.execute(sql, (confidence, claim_id))
        finally:
            conn.close()

    def mark_claim_superseded(self, claim_id, superseded_by_id):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "UPDATE memory_claims "
                    "SET status='superseded', superseded_by_id=%s, last_seen_at=NOW() "
                    "WHERE id=%s"
                )
                cur.execute(sql, (superseded_by_id, claim_id))
        finally:
            conn.close()

    def get_claims_by_ids(self, user_id, claim_ids):
        if claim_ids is None or len(claim_ids) == 0:
            return []
        placeholders = ",".join(["%s"] * len(claim_ids))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT id, user_id, claim_type, claim_text, confidence, last_seen_at, status, superseded_by_id "
                    "FROM memory_claims WHERE user_id=%s AND id IN (" + placeholders + ")"
                )
                cur.execute(sql, (user_id, *claim_ids))
                rows = list(cur.fetchall())
        finally:
            conn.close()

        items = []
        i = 0
        while i < len(rows):
            row = rows[i]
            items.append(
                {
                    "id": row.get("id"),
                    "user_id": row.get("user_id"),
                    "claim_type": row.get("claim_type"),
                    "claim_text": row.get("claim_text"),
                    "confidence": float(row.get("confidence") or 0.0),
                    "last_seen_at": row.get("last_seen_at"),
                    "status": row.get("status"),
                    "superseded_by_id": row.get("superseded_by_id")
                }
            )
            i += 1
        return items

    def get_memory_items_by_ids(self, user_id, ids):
        if ids is None or len(ids) == 0:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT memory_id, user_id, chat_id, type, role, content, created_at, extra "
                    "FROM memory_items WHERE user_id=%s AND memory_id IN (" + placeholders + ")"
                )
                cur.execute(sql, (user_id, *ids))
                rows = list(cur.fetchall())
        finally:
            conn.close()

        items = []
        i = 0
        while i < len(ids):
            item_id = ids[i]
            j = 0
            found = None
            while j < len(rows):
                if rows[j].get("memory_id") == item_id:
                    found = rows[j]
                    break
                j += 1
            if found is not None:
                extra_val = found.get("extra")
                items.append(
                    {
                        "id": found.get("memory_id"),
                        "user_id": found.get("user_id"),
                        "chat_id": found.get("chat_id"),
                        "type": found.get("type"),
                        "role": found.get("role"),
                        "content": found.get("content"),
                        "created_at": found.get("created_at"),
                        "extra": extra_val
                    }
                )
            i += 1
        return items

    def get_profile_items(self, user_id):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT memory_id, content, extra, created_at FROM memory_items "
                    "WHERE user_id=%s AND type='profile_fact' ORDER BY created_at DESC"
                )
                cur.execute(sql, (user_id,))
                rows = list(cur.fetchall())
        finally:
            conn.close()

        items = []
        i = 0
        while i < len(rows):
            items.append(rows[i])
            i += 1
        return items

    def find_profile_memory_by_key(self, user_id, memory_key):
        if memory_key is None or len(str(memory_key).strip()) == 0:
            return None
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                like_pattern = "%" + str(memory_key) + "%"
                sql = (
                    "SELECT memory_id FROM memory_items "
                    "WHERE user_id=%s AND type='profile_fact' AND extra LIKE %s "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                cur.execute(sql, (user_id, like_pattern))
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return row.get("memory_id")

    def get_chat_summaries(self, user_id, chat_id, limit):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT content, created_at FROM memory_items "
                    "WHERE user_id=%s AND chat_id=%s AND type='chat_summary' "
                    "ORDER BY created_at DESC LIMIT %s"
                )
                cur.execute(sql, (user_id, chat_id, limit))
                rows = list(cur.fetchall())
        finally:
            conn.close()
        rows = list(reversed(rows))
        items = []
        i = 0
        while i < len(rows):
            items.append(rows[i])
            i += 1
        return items

    def get_doc_chunks(self, user_id, limit):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT memory_id, content, extra, created_at FROM memory_items "
                    "WHERE user_id=%s AND type='doc_chunk' "
                    "ORDER BY created_at DESC LIMIT %s"
                )
                cur.execute(sql, (user_id, limit))
                rows = list(cur.fetchall())
        finally:
            conn.close()
        items = []
        i = 0
        while i < len(rows):
            items.append(rows[i])
            i += 1
        return items

    def insert_document(
        self,
        doc_id,
        user_id,
        title,
        doc_type,
        source,
        filename,
        mime_type,
        file_size_bytes,
        sha256
    ):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO documents "
                    "(doc_id, user_id, filename, mime_type, file_size_bytes, sha256, source, title, doc_type, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"
                )
                cur.execute(
                    sql,
                    (
                        doc_id,
                        user_id,
                        filename,
                        mime_type,
                        file_size_bytes,
                        sha256,
                        source,
                        title,
                        doc_type
                    )
                )
                return doc_id
        finally:
            conn.close()

    def insert_document_chunk(self, doc_id, user_id, chunk_index, chunk_text):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO document_chunks "
                    "(chunk_id, doc_id, user_id, chunk_index, content_text, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, NOW())"
                )
                chunk_id = str(uuid.uuid4())
                cur.execute(sql, (chunk_id, doc_id, user_id, chunk_index, chunk_text))
                return chunk_id
        finally:
            conn.close()

    def get_document_chunks_by_ids(self, chunk_ids):
        if chunk_ids is None or len(chunk_ids) == 0:
            return []
        placeholders = ",".join(["%s"] * len(chunk_ids))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT chunk_id, doc_id, chunk_index, content_text, created_at "
                    "FROM document_chunks WHERE chunk_id IN (" + placeholders + ")"
                )
                cur.execute(sql, (*chunk_ids,))
                rows = list(cur.fetchall())
        finally:
            conn.close()
        return rows

    def get_documents_by_ids(self, document_ids):
        if document_ids is None or len(document_ids) == 0:
            return []
        placeholders = ",".join(["%s"] * len(document_ids))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "SELECT doc_id, user_id, title, doc_type, source, created_at "
                    "FROM documents WHERE doc_id IN (" + placeholders + ")"
                )
                cur.execute(sql, (*document_ids,))
                rows = list(cur.fetchall())
        finally:
            conn.close()
        return rows

    def log_retrieval_run(self, user_id, layer, query_text, top_k):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO retrieval_runs "
                    "(user_id, layer, query_text, top_k, created_at) "
                    "VALUES (%s, %s, %s, %s, NOW())"
                )
                cur.execute(sql, (user_id, layer, query_text, top_k))
        finally:
            conn.close()
