import os
import uuid
from datetime import datetime

import pymysql
from pymysql.cursors import DictCursor
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer


def mysql_connect():
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DATABASE", ""),
        cursorclass=DictCursor,
        autocommit=True
    )


def main():
    user_id = os.environ.get("TEST_USER_ID", "user_1")
    chat_id = os.environ.get("TEST_CHAT_ID", "test_chat")
    content = "sanity check message"
    created_at = datetime.now()

    conn = mysql_connect()
    memory_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        sql = (
            "INSERT INTO memory_items "
            "(memory_id, user_id, chat_id, type, role, content, created_at, extra) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        cur.execute(sql, (memory_id, user_id, chat_id, "chat_message", "user", content, created_at, None))
        mysql_id = memory_id

    embed = SentenceTransformer(os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2"))
    vector = embed.encode(content)
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))

    payload = {
        "user_id": user_id,
        "chat_id": chat_id,
        "memory_type": "chat_message",
        "mysql_id": mysql_id,
        "created_at": created_at.isoformat(timespec="seconds")
    }
    point = models.PointStruct(id=str(mysql_id), vector=vector, payload=payload)
    client.upsert(collection_name="chat_memory", points=[point])

    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
            models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_message"))
        ]
    )
    results = client.search(
        collection_name="chat_memory",
        query_vector=vector,
        limit=1,
        query_filter=qdrant_filter,
        with_payload=True
    )

    if results and len(results) > 0:
        payload = results[0].payload or {}
        if "text" in payload or "content" in payload:
            print("FAIL: Qdrant payload contains raw text")
        else:
            print("OK: Qdrant payload is pointer-only")
        q_mysql_id = payload.get("mysql_id")
        if q_mysql_id is None:
            print("FAIL: Qdrant payload missing mysql_id")
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM memory_items WHERE id=%s AND user_id=%s",
                    (q_mysql_id, user_id)
                )
                row = cur.fetchone()
                if row and row.get("content") == content:
                    print("OK: MySQL content retrieved via mysql_id")
                else:
                    print("FAIL: MySQL content mismatch or missing")

    conn.close()


if __name__ == "__main__":
    main()
