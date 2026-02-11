#python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload --log-level debug
#python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

from core.twin_service import TwinService

def main():
    # Update these if your setup differs
    qdrant_url = "http://localhost:6333"
    ollama_model_name = "vip-twin2"  # or your custom Ollama model name
    profile_collection = "user_profile"
    chat_collection = "chat_memory"
    embed_model_name = "all-MiniLM-L6-v2"

    service = TwinService(
        qdrant_url=qdrant_url,
        ollama_model_name=ollama_model_name,
        profile_collection=profile_collection,
        chat_collection=chat_collection,
        embed_model_name=embed_model_name
    )

    service.init()

    user_id = "user_1"
    chat_id = service.start_chat(user_id)

    print("Digital Twin MVP (CLI)")
    print("User:", user_id)
    print("Chat:", chat_id)
    print("Type /exit to quit.")
    print("Optional: /profile <category> <text>  (adds profile memory)")
    print()

    while True:
        raw = input("You: ").strip()
        if raw == "/exit":
            break

        if raw.startswith("/profile "):
            # Format: /profile goals I want to build a digital twin
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print("Usage: /profile <category> <text>")
                print()
                continue

            category = parts[1].strip()
            text = parts[2].strip()

            point_id = service.add_profile_memory(user_id, category, text)
            print("Saved profile memory:", point_id)
            print()
            continue

        result = service.ask(user_id, chat_id, raw)
        assistant_message = result.get("assistant_message", "")
        context_used = result.get("context_used", {})

        print()
        print("Twin:", assistant_message)
        print()
        print("Context used:", context_used)
        print()


if __name__ == "__main__":
    main()