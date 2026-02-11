Twin MVP Notes

Memory storage model (source of truth)
- All memory text/content is stored in MySQL.
- Qdrant stores only embeddings + pointers: user_id, chat_id, memory_type, mysql_id, created_at.
- Retrieval flow: Qdrant -> mysql_id list -> MySQL content fetch.

Key tables
- memory_items: unified store for profile_fact, chat_message, chat_summary, doc_chunk.
- session_memory: rolling session summary.
- memory_claims: long-term claims with confidence/recency.
- documents / document_chunks: document metadata + chunks (L3 evidence).

Required env vars
- MYSQL_ENABLED=1
- MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
- MYSQL_REQUIRED=true (default) enforces MySQL as source of truth

Backfill script
- Run `python scripts/backfill_qdrant_pointers.py` to rebuild Qdrant pointers and migrate any text-only Qdrant payloads into MySQL.

Two-pass decision flow
- The LLM returns a JSON header (first line) plus plain answer text.
- Backend parses the header and only escalates when explicitly requested.
- Decision header format:
  {"action":"DIRECT"|"ASK"|"ESCALATE","next_layer":null|"L2"|"L3"|"L4","reason":"..."}

env run commands:
$env:MYSQL_ENABLED="true"
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="4205Mowry!@"
$env:MYSQL_DATABASE="vip_twin2"

$env:LLM_MODEL="vip-twin2"
$env:VECTOR_DB="qdrant"
$env:QDRANT_URL="http://localhost:6333"
