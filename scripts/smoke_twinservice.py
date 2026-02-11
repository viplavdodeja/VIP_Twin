import os
import sys
import time
from requests.exceptions import RequestException

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.twin_service import TwinService


def main():
    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    ollama_model_name = os.environ.get("OLLAMA_MODEL", "vip-twin2")
    profile_collection = os.environ.get("PROFILE_COLLECTION", "user_profile")
    chat_collection = os.environ.get("CHAT_COLLECTION", "chat_memory")
    embed_model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

    service = TwinService(
        qdrant_url=qdrant_url,
        ollama_model_name=ollama_model_name,
        profile_collection=profile_collection,
        chat_collection=chat_collection,
        embed_model_name=embed_model_name,
    )

    print("SMOKE: init start")
    try:
        service.init()
    except Exception as exc:
        print("SMOKE: init failed (qdrant/embeddings?):", str(exc))
        return 0

    user_id = "11111111-1111-1111-1111-111111111111"
    chat_id = service.start_chat(user_id)

    print("SMOKE: mysql enabled =", service.mysql_enabled)
    if service.mysql_enabled:
        try:
            service.add_chat_message(user_id, chat_id, role="user", text="smoke hello")
            service.add_chat_message(user_id, chat_id, role="assistant", text="smoke reply")
            rows = service.get_chat_messages(user_id, chat_id, limit=10)
            last_text = ""
            if len(rows) > 0:
                last_text = rows[-1].get("text", "")
            ordered = True
            i = 1
            while i < len(rows):
                if rows[i].get("created_at", 0) < rows[i - 1].get("created_at", 0):
                    ordered = False
                    break
                i += 1
            print("SMOKE: mysql rows =", len(rows), "last_text =", last_text)
            print("SMOKE: mysql ordering ok =", ordered)
        except Exception as exc:
            print("SMOKE: mysql check failed:", str(exc))

    print("SMOKE: profile dedupe check")
    try:
        service.add_profile_memory(user_id, "smoke_category", "first value")
        service.add_profile_memory(user_id, "smoke_category", "second value")
        memories = service.list_profile_memories(user_id, limit=200)
        count = 0
        i = 0
        while i < len(memories):
            row = memories[i]
            if row.get("category") == "smoke_category":
                count += 1
            i += 1
        print("SMOKE: profile entries for smoke_category =", count)
    except Exception as exc:
        print("SMOKE: profile dedupe check failed:", str(exc))

    print("SMOKE: ask start")
    t0 = time.perf_counter()
    try:
        result = service.ask(user_id, chat_id, "hello")
    except RequestException as exc:
        print("SMOKE: ask failed (ollama/qdrant request error):", str(exc))
        return 0
    except Exception as exc:
        print("SMOKE: ask failed:", str(exc))
        return 0
    t1 = time.perf_counter()

    context_used = result.get("context_used", {})
    print("SMOKE: ask ok in", int((t1 - t0) * 1000), "ms")
    print("SMOKE: context_used:", context_used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
