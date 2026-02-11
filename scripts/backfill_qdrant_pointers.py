import os
import json
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


def embedder():
    model_name = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def qdrant_client():
    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    return QdrantClient(url=url)


def upsert_point(client, collection, point_id, vector, payload):
    if isinstance(point_id, str) and point_id.strip().isdigit():
        try:
            point_id = int(point_id.strip())
        except Exception:
            pass
    point = models.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
    )
    client.upsert(collection_name=collection, points=[point])


def backfill_memory_items(client, embed, conn):
    print("Backfill memory_items -> Qdrant pointers")
    last_id = ""
    while True:
        with conn.cursor() as cur:
            sql = (
                "SELECT memory_id, user_id, chat_id, type, content, created_at "
                "FROM memory_items WHERE memory_id > %s ORDER BY memory_id ASC LIMIT 500"
            )
            cur.execute(sql, (last_id,))
            rows = list(cur.fetchall())

        if len(rows) == 0:
            break

        i = 0
        while i < len(rows):
            row = rows[i]
            mem_type = row.get("type")
            collection = None
            if mem_type == "profile_fact":
                collection = "user_profile"
            elif mem_type == "chat_message" or mem_type == "chat_summary":
                collection = "chat_memory"
            elif mem_type == "doc_chunk":
                collection = "uploaded_context"
            if collection is None:
                i += 1
                last_id = row.get("memory_id")
                continue

            content = row.get("content") or ""
            vector = embed.encode(content)
            payload = {
                "user_id": row.get("user_id"),
                "chat_id": row.get("chat_id"),
                "memory_type": mem_type,
                "mysql_id": row.get("memory_id"),
                "created_at": row.get("created_at").isoformat(timespec="seconds")
            }
            upsert_point(client, collection, str(row.get("memory_id")), vector, payload)
            last_id = row.get("memory_id")
            i += 1
        print("  processed up to id", last_id)


def backfill_claims(client, embed, conn):
    print("Backfill memory_claims -> Qdrant pointers")
    with conn.cursor() as cur:
        cur.execute("SELECT id, user_id, claim_text, last_seen_at FROM memory_claims")
        rows = list(cur.fetchall())

    i = 0
    while i < len(rows):
        row = rows[i]
        claim_text = row.get("claim_text") or ""
        vector = embed.encode(claim_text)
        payload = {
            "user_id": row.get("user_id"),
            "chat_id": None,
            "memory_type": "claim",
            "mysql_id": row.get("id"),
            "created_at": row.get("last_seen_at").isoformat(timespec="seconds") if row.get("last_seen_at") else datetime.now().isoformat(timespec="seconds")
        }
        upsert_point(client, "user_claims", str(row.get("id")), vector, payload)
        i += 1


def backfill_documents(client, embed, conn):
    print("Backfill document_chunks -> Qdrant pointers")
    with conn.cursor() as cur:
        sql = (
            "SELECT dc.chunk_id, dc.doc_id, dc.chunk_index, dc.content_text, "
            "d.user_id, dc.created_at "
            "FROM document_chunks dc JOIN documents d ON dc.doc_id = d.doc_id"
        )
        cur.execute(sql)
        rows = list(cur.fetchall())

    i = 0
    while i < len(rows):
        row = rows[i]
        chunk_text = row.get("content_text") or ""
        doc_id = row.get("doc_id")
        chunk_index = row.get("chunk_index")
        extra = {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "metadata": {}
        }
        extra_json = json.dumps(extra)
        with conn.cursor() as cur:
            like_doc = "%\"document_id\": " + str(doc_id) + "%"
            like_chunk = "%\"chunk_index\": " + str(chunk_index) + "%"
            sql_check = (
                "SELECT memory_id FROM memory_items "
                "WHERE user_id=%s AND type='doc_chunk' AND extra LIKE %s AND extra LIKE %s "
                "ORDER BY created_at DESC LIMIT 1"
            )
            cur.execute(sql_check, (row.get("user_id"), like_doc, like_chunk))
            existing = cur.fetchone()
            if existing is not None:
                mem_id = existing.get("memory_id")
            else:
                sql_insert = (
                    "INSERT INTO memory_items "
                    "(memory_id, user_id, chat_id, type, role, content, created_at, extra) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                )
                mem_id = str(uuid.uuid4())
                cur.execute(
                    sql_insert,
                    (mem_id, row.get("user_id"), None, "doc_chunk", None, chunk_text, row.get("created_at"), extra_json)
                )
        vector = embed.encode(chunk_text)
        payload = {
            "user_id": row.get("user_id"),
            "chat_id": None,
            "memory_type": "doc_chunk",
            "mysql_id": mem_id,
            "created_at": row.get("created_at").isoformat(timespec="seconds") if row.get("created_at") else datetime.now().isoformat(timespec="seconds")
        }
        upsert_point(client, "user_docs", str(mem_id), vector, payload)
        i += 1


def rewrite_qdrant_points(client, embed, conn, collection, forced_type):
    print("Rewrite Qdrant points for", collection)
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=200,
            offset=next_page,
            with_payload=True,
            with_vectors=False
        )
        if points is None or len(points) == 0:
            break

        i = 0
        while i < len(points):
            payload = points[i].payload or {}
            if payload.get("mysql_id") is not None:
                i += 1
                continue

            text = payload.get("text") or payload.get("content")
            if text is None or len(str(text).strip()) == 0:
                i += 1
                continue

            user_id = payload.get("user_id")
            if user_id is None:
                i += 1
                continue

            chat_id = payload.get("chat_id")
            mem_type = payload.get("memory_type") or forced_type
            created_at = datetime.now()
            extra = {}
            for key in payload:
                if key not in ("text", "content"):
                    extra[key] = payload.get(key)

            extra_json = json.dumps(extra) if len(extra) > 0 else None

            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO memory_items "
                    "(memory_id, user_id, chat_id, type, role, content, created_at, extra) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                )
                mysql_id = str(uuid.uuid4())
                cur.execute(sql, (mysql_id, user_id, chat_id, mem_type, payload.get("role"), text, created_at, extra_json))

            clean_payload = {
                "user_id": user_id,
                "chat_id": chat_id,
                "memory_type": mem_type,
                "mysql_id": mysql_id,
                "created_at": created_at.isoformat(timespec="seconds")
            }
            vector = embed.encode(text)
            upsert_point(client, collection, points[i].id, vector, clean_payload)
            i += 1

        if next_page is None:
            break


def rewrite_qdrant_claim_points(client, embed, conn):
    print("Rewrite Qdrant points for user_claims")
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name="user_claims",
            scroll_filter=None,
            limit=200,
            offset=next_page,
            with_payload=True,
            with_vectors=False
        )
        if points is None or len(points) == 0:
            break

        i = 0
        while i < len(points):
            payload = points[i].payload or {}
            if payload.get("mysql_id") is not None:
                i += 1
                continue
            text = payload.get("text") or payload.get("content")
            if text is None or len(str(text).strip()) == 0:
                i += 1
                continue
            user_id = payload.get("user_id")
            if user_id is None:
                i += 1
                continue

            claim_type = payload.get("claim_type") or "fact"
            confidence = float(payload.get("confidence") or 0.5)
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO memory_claims "
                    "(user_id, claim_type, claim_text, confidence, last_seen_at, status, superseded_by_id) "
                    "VALUES (%s, %s, %s, %s, NOW(), 'active', NULL)"
                )
                cur.execute(sql, (user_id, claim_type, text, confidence))
                claim_id = cur.lastrowid

            clean_payload = {
                "user_id": user_id,
                "chat_id": None,
                "memory_type": "claim",
                "mysql_id": claim_id,
                "created_at": datetime.now().isoformat(timespec="seconds")
            }
            vector = embed.encode(text)
            upsert_point(client, "user_claims", points[i].id, vector, clean_payload)
            i += 1

        if next_page is None:
            break


def main():
    conn = mysql_connect()
    embed = embedder()
    client = qdrant_client()

    backfill_memory_items(client, embed, conn)
    backfill_claims(client, embed, conn)
    backfill_documents(client, embed, conn)

    rewrite_qdrant_points(client, embed, conn, "user_profile", "profile_fact")
    rewrite_qdrant_points(client, embed, conn, "chat_memory", "chat_message")
    rewrite_qdrant_points(client, embed, conn, "uploaded_context", "doc_chunk")
    rewrite_qdrant_claim_points(client, embed, conn)
    rewrite_qdrant_points(client, embed, conn, "user_docs", "doc_chunk")

    conn.close()
    print("Backfill complete.")


if __name__ == "__main__":
    main()
