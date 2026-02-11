import os
import sys
import time
import uuid
import hashlib
from datetime import datetime

import pymysql
from pymysql.cursors import DictCursor
from qdrant_client import QdrantClient, models


ALLOWED_ROLES = {"user", "assistant", "system", "tool"}


def parse_bool(value):
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def normalize_role(role):
    if role is None:
        return None
    r = str(role).strip().lower()
    if r == "ai":
        r = "assistant"
    if r in ALLOWED_ROLES:
        return r
    return None


def make_message_id(user_id, chat_id, created_at, role, text):
    sha = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    base = f"{user_id}|{chat_id}|{created_at}|{role}|{sha}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def main():
    mysql_host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    mysql_port = int(os.environ.get("MYSQL_PORT", "3306"))
    mysql_user = os.environ.get("MYSQL_USER", "root")
    mysql_password = os.environ.get("MYSQL_PASSWORD", "")
    mysql_database = os.environ.get("MYSQL_DATABASE", "")

    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    chat_collection = os.environ.get("CHAT_COLLECTION", "chat_memory")
    migrate_user_id = os.environ.get("MIGRATE_USER_ID", "").strip()
    dry_run = parse_bool(os.environ.get("DRY_RUN", "false"))
    limit = os.environ.get("LIMIT", "").strip()
    limit = int(limit) if limit else None

    if not migrate_user_id:
        print("MIGRATE_USER_ID is required.")
        return 1

    print("MIGRATE: start user_id=", migrate_user_id, "dry_run=", dry_run)

    try:
        qdrant = QdrantClient(url=qdrant_url)
        qdrant.get_collections()
    except Exception as exc:
        print("MIGRATE: qdrant connection failed:", str(exc))
        return 1

    try:
        conn = pymysql.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
            cursorclass=DictCursor,
            autocommit=True
        )
    except Exception as exc:
        print("MIGRATE: mysql connection failed:", str(exc))
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions LIMIT 1")
            cur.execute("SELECT 1 FROM chat_messages LIMIT 1")
            cur.execute("SELECT 1 FROM users WHERE user_id=%s LIMIT 1", (migrate_user_id,))
            if cur.fetchone() is None:
                print("MIGRATE: user_id not found in MySQL users table:", migrate_user_id)
                return 1
    except Exception as exc:
        print("MIGRATE: mysql schema check failed:", str(exc))
        return 1

    scanned = 0
    valid = 0
    skipped = 0
    invalid_role = 0
    inserted = 0
    rows_affected = 0
    chat_last_ts = {}
    seen_chats = set()

    t0 = time.perf_counter()
    offset = None
    batch = []
    batch_size = 200

    q_filter = models.Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=migrate_user_id)),
            models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_message")),
        ]
    )

    def flush_batch(cur):
        nonlocal inserted, rows_affected
        if len(batch) == 0:
            return
        if dry_run:
            inserted += len(batch)
            batch.clear()
            return
        sql = (
            "INSERT INTO chat_messages "
            "(message_id, chat_id, user_id, role, content, created_at) "
            "VALUES (%s, %s, %s, %s, %s, FROM_UNIXTIME(%s)) "
            "ON DUPLICATE KEY UPDATE content=content"
        )
        rows = [(
            b["message_id"],
            b["chat_id"],
            b["user_id"],
            b["role"],
            b["content"],
            b["created_at"]
        ) for b in batch]
        affected = cur.executemany(sql, rows)
        rows_affected += affected
        inserted += len(batch)
        batch.clear()

    try:
        with conn.cursor() as cur:
            while True:
                points, next_page = qdrant.scroll(
                    collection_name=chat_collection,
                    scroll_filter=q_filter,
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )

                if points is None or len(points) == 0:
                    break

                i = 0
                while i < len(points):
                    scanned += 1
                    payload = points[i].payload or {}
                    chat_id = payload.get("chat_id")
                    role = payload.get("role")
                    text = payload.get("text")
                    created_at = payload.get("created_at")

                    if not chat_id or not role or not text or not created_at:
                        skipped += 1
                        i += 1
                        continue

                    norm_role = normalize_role(role)
                    if norm_role is None:
                        invalid_role += 1
                        i += 1
                        continue

                    try:
                        created_at_int = int(created_at)
                    except Exception:
                        skipped += 1
                        i += 1
                        continue

                    msg_id = make_message_id(
                        migrate_user_id, chat_id, created_at_int, norm_role, str(text)
                    )

                    if chat_id not in chat_last_ts or created_at_int > chat_last_ts[chat_id]:
                        chat_last_ts[chat_id] = created_at_int
                    if chat_id not in seen_chats:
                        seen_chats.add(chat_id)
                        if not dry_run:
                            cur.execute(
                                "INSERT IGNORE INTO chat_sessions "
                                "(chat_id, user_id, title, last_message_at) "
                                "VALUES (%s, %s, NULL, FROM_UNIXTIME(%s))",
                                (chat_id, migrate_user_id, created_at_int)
                            )

                    batch.append(
                        {
                            "message_id": msg_id,
                            "chat_id": chat_id,
                            "user_id": migrate_user_id,
                            "role": norm_role,
                            "content": str(text),
                            "created_at": created_at_int
                        }
                    )
                    valid += 1

                    if len(batch) >= batch_size:
                        flush_batch(cur)

                    if limit is not None and scanned >= limit:
                        break
                    i += 1

                if limit is not None and scanned >= limit:
                    break

                offset = next_page
                if offset is None:
                    break

            flush_batch(cur)

            if not dry_run:
                for chat_id, ts in chat_last_ts.items():
                    cur.execute(
                        "INSERT IGNORE INTO chat_sessions "
                        "(chat_id, user_id, title, last_message_at) "
                        "VALUES (%s, %s, NULL, FROM_UNIXTIME(%s))",
                        (chat_id, migrate_user_id, ts)
                    )
                    cur.execute(
                        "UPDATE chat_sessions SET last_message_at=FROM_UNIXTIME(%s) "
                        "WHERE chat_id=%s",
                        (ts, chat_id)
                    )
    finally:
        conn.close()

    t1 = time.perf_counter()
    print("MIGRATE: done in", round(t1 - t0, 2), "s")
    print("MIGRATE: scanned=", scanned, "valid=", valid, "inserted=", inserted)
    print("MIGRATE: skipped=", skipped, "invalid_role=", invalid_role, "rows_affected=", rows_affected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
