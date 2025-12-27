import time
import uuid
import requests

from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer


class TwinService:
    def __init__(
        self,
        qdrant_url,
        ollama_model_name,
        profile_collection,
        chat_collection,
        embed_model_name
    ):
        self.qdrant_url = qdrant_url
        self.ollama_model_name = ollama_model_name
        self.profile_collection = profile_collection
        self.chat_collection = chat_collection
        self.embed_model_name = embed_model_name

        self.client = None
        self.embedder = None
        self.vector_size = None

        # Uploaded PDF chunks / image captions go here
        self.upload_collection = "uploaded_context"

    # ----------------------------
    # Setup
    # ----------------------------

    def init(self):
        self.client = QdrantClient(url=self.qdrant_url)
        self.embedder = SentenceTransformer(self.embed_model_name)

        test_vector = self.embed_text("test")
        self.vector_size = len(test_vector)

        self.ensure_collection(self.profile_collection)
        self.ensure_collection(self.chat_collection)
        self.ensure_collection(self.upload_collection)

    def ensure_collection(self, collection_name):
        existing = self.client.get_collections()
        found = False

        i = 0
        while i < len(existing.collections):
            if existing.collections[i].name == collection_name:
                found = True
                break
            i += 1

        if found is False:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    # ----------------------------
    # Embeddings
    # ----------------------------

    def embed_text(self, text):
        # Returns a numpy array; Qdrant client accepts it
        return self.embedder.encode(text)

    # ----------------------------
    # Ollama
    # ----------------------------

    def ollama_generate(self, prompt, max_tokens=200, temperature=0.2, timeout_seconds=180):
        url = "http://127.0.0.1:11434/api/generate"

        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        response = requests.post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    # ----------------------------
    # Chat/session
    # ----------------------------

    def start_chat(self, user_id):
        # MVP: stateless chat id
        return str(uuid.uuid4())

    # ----------------------------
    # Qdrant operations
    # ----------------------------

    def upsert(self, collection_name, point_id, vector, payload):
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )

        points = []
        points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def scroll(self, collection_name, qdrant_filter, limit):
        points, next_page = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return points

    def search(self, collection_name, query_vector, limit, qdrant_filter):
        # qdrant-client API changed over time; support both

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True
            )
            return results

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True
            )

            # Depending on version, response may contain points/result
            if hasattr(response, "points"):
                return response.points
            if hasattr(response, "result"):
                return response.result

            # fallback
            return response

        raise RuntimeError("qdrant-client does not support search/query_points. Please upgrade qdrant-client.")

    # ----------------------------
    # Profile memory
    # ----------------------------

    def add_profile_memory(self, user_id, category, text):
        point_id = str(uuid.uuid4())
        vector = self.embed_text(text)

        payload = {
            "user_id": user_id,
            "memory_type": "profile",
            "category": category,
            "text": text,
            "updated_at": int(time.time()),
            "memory_key": "profile:" + user_id + ":" + category
        }

        self.upsert(self.profile_collection, point_id, vector, payload)
        return point_id

    def get_profile_texts(self, user_id):
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="profile"))
            ]
        )

        points = self.scroll(self.profile_collection, qdrant_filter, limit=200)

        texts = []
        i = 0
        while i < len(points):
            payload = points[i].payload
            if payload is not None:
                t = payload.get("text", "")
                if t is not None and len(t) > 0:
                    texts.append(t)
            i += 1

        return texts

    # ----------------------------
    # Chat summaries (optional; currently read-only)
    # ----------------------------

    def get_recent_summaries(self, user_id, limit):
        # Kept for compatibility with your UI/debug, but summaries are not auto-generated in MVP speed mode.
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_summary"))
            ]
        )

        points = self.scroll(self.chat_collection, qdrant_filter, limit=500)

        items = []
        i = 0
        while i < len(points):
            payload = points[i].payload
            if payload is not None:
                created_at = payload.get("created_at", 0)
                text = payload.get("text", "")
                if text is not None and len(text) > 0:
                    items.append((created_at, text))
            i += 1

        # simple insertion sort by created_at
        sorted_items = []
        j = 0
        while j < len(items):
            pair = items[j]
            inserted = False

            k = 0
            while k < len(sorted_items):
                if pair[0] < sorted_items[k][0]:
                    sorted_items.insert(k, pair)
                    inserted = True
                    break
                k += 1

            if inserted is False:
                sorted_items.append(pair)

            j += 1

        recent = []
        start_index = 0
        if len(sorted_items) > limit:
            start_index = len(sorted_items) - limit

        m = start_index
        while m < len(sorted_items):
            recent.append(sorted_items[m][1])
            m += 1

        return recent

    def retrieve_relevant_summaries(self, user_id, query_text, top_k):
        query_vector = self.embed_text(query_text)

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_summary"))
            ]
        )

        results = self.search(self.chat_collection, query_vector, top_k, qdrant_filter)

        texts = []
        i = 0
        while i < len(results):
            payload = results[i].payload
            if payload is not None:
                t = payload.get("text", "")
                if t is not None and len(t) > 0:
                    texts.append(t)
            i += 1

        return texts

    # ----------------------------
    # Uploaded context (PDF chunks + image captions)
    # ----------------------------

    def add_uploaded_context_chunk(self, user_id, file_id, file_name, context_type, chunk_index, text):
        point_id = str(uuid.uuid4())
        vector = self.embed_text(text)

        payload = {
            "user_id": user_id,
            "memory_type": "uploaded_context",
            "context_type": context_type,
            "file_id": file_id,
            "file_name": file_name,
            "chunk_index": chunk_index,
            "text": text,
            "created_at": int(time.time())
        }

        self.upsert(self.upload_collection, point_id, vector, payload)
        return point_id

    def retrieve_relevant_uploaded_context(self, user_id, query_text, top_k):
        query_vector = self.embed_text(query_text)

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="uploaded_context")),
            ]
        )

        results = self.search(self.upload_collection, query_vector, top_k, qdrant_filter)

        texts = []
        i = 0
        while i < len(results):
            payload = results[i].payload
            if payload is not None:
                t = payload.get("text", "")
                if t is not None and len(t) > 0:
                    texts.append(t)
            i += 1

        return texts

    # ----------------------------
    # Main chat logic
    # ----------------------------

    def ask(self, user_id, chat_id, user_message):
        msg = user_message.lower()

        # Document summarization intent
        doc_mode = False
        if ("summarize" in msg) or ("summary" in msg):
            if ("pdf" in msg) or ("document" in msg) or ("upload" in msg) or ("file" in msg):
                doc_mode = True

        # Retrieve memory with MVP limits
        profile_texts = self.get_profile_texts(user_id)
        if len(profile_texts) > 3:
            profile_texts = profile_texts[:3]

        recent_summaries = self.get_recent_summaries(user_id, limit=2)
        retrieved_summaries = self.retrieve_relevant_summaries(user_id, user_message, top_k=1)

        # Upload retrieval: get more chunks for doc summaries
        if doc_mode:
            retrieved_uploads = self.retrieve_relevant_uploaded_context(user_id, user_message, top_k=8)
        else:
            retrieved_uploads = self.retrieve_relevant_uploaded_context(user_id, user_message, top_k=1)

        prompt = self.build_chat_prompt(
            profile_texts,
            recent_summaries,
            retrieved_summaries,
            retrieved_uploads,
            user_message,
            doc_mode=doc_mode
        )

        # Debug visibility
        print("PROMPT_CHARS:", len(prompt))
        print("CTX_COUNTS:",
              "profile=", len(profile_texts),
              "recent=", len(recent_summaries),
              "retrieved_sum=", len(retrieved_summaries),
              "retrieved_upload=", len(retrieved_uploads),
              "doc_mode=", doc_mode)

        if len(retrieved_uploads) > 0:
            preview = retrieved_uploads[0]
            if len(preview) > 200:
                preview = preview[:200] + "…"
            print("UPLOAD_PREVIEW:", preview)
        else:
            print("UPLOAD_PREVIEW: (none)")

        # Generation settings
        # Doc summaries: smaller output cap, very direct
        max_tokens = 180
        temp = 0.2

        if doc_mode:
            max_tokens = 220
            temp = 0.1

        try:
            assistant_message = self.ollama_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                timeout_seconds=180
            ).strip()
        except Exception as e:
            print("OLLAMA_ERROR:", str(e))
            assistant_message = (
                "I couldn’t generate a response in time (local model latency). "
                "Try again, or switch to a faster local model for the demo."
            )

        # MVP speed: do NOT generate/save summaries every message
        result = {
            "assistant_message": assistant_message,
            "summary_saved": "",
            "context_used": {
                "profile_count": len(profile_texts),
                "recent_summary_count": len(recent_summaries),
                "retrieved_summary_count": len(retrieved_summaries),
                "retrieved_upload_count": len(retrieved_uploads),
                "doc_mode": doc_mode
            }
        }
        return result

    # ----------------------------
    # Prompt builder
    # ----------------------------

    def build_chat_prompt(
        self,
        profile_texts,
        recent_summaries,
        retrieved_summaries,
        uploaded_context_texts,
        user_message,
        doc_mode=False
    ):
        prompt = ""

        if doc_mode:
            # STRICT doc summarization mode: prevents the “digital twin” persona from hijacking the answer
            prompt += "You are summarizing an uploaded document for the user.\n"
            prompt += "RULES (STRICT):\n"
            prompt += "- You MUST base the summary ONLY on the UPLOADED CONTEXT section.\n"
            prompt += "- If UPLOADED CONTEXT is empty, say exactly: I cannot access the uploaded PDF text in memory.\n"
            prompt += "- Do NOT ask questions. Do NOT mention goals/career unless it appears in the uploaded text.\n"
            prompt += "- Output exactly:\n"
            prompt += "  1) 5 bullet summary\n"
            prompt += "  2) 3 key quotes (short phrases) copied from the uploaded context\n"
            prompt += "\n"

            prompt += "## UPLOADED CONTEXT (DOCUMENT EXCERPTS)\n"
            if len(uploaded_context_texts) == 0:
                prompt += "(none)\n"
            else:
                i = 0
                while i < len(uploaded_context_texts):
                    t = uploaded_context_texts[i]
                    if len(t) > 320:
                        t = t[:320] + "…"
                    prompt += "[" + str(i + 1) + "] " + t + "\n"
                    i += 1
            prompt += "\n"

            prompt += "## USER REQUEST\n"
            prompt += user_message.strip() + "\n\n"
            prompt += "## ANSWER\n"
            return prompt

        # Normal digital twin chat mode
        prompt += "You are the user's private digital twin running locally.\n"
        prompt += "Rules:\n"
        prompt += "- Use memory sections as the source of truth.\n"
        prompt += "- If information is missing, ask a clarifying question.\n"
        prompt += "- Ground insights in memory and state assumptions clearly.\n"
        prompt += "- Only use UPLOADED CONTEXT when the user asks about the document/upload.\n"
        prompt += "\n"

        # Profile
        prompt += "## USER PROFILE\n"
        if len(profile_texts) == 0:
            prompt += "(none)\n"
        else:
            i = 0
            while i < len(profile_texts):
                t = profile_texts[i]
                if len(t) > 180:
                    t = t[:180] + "…"
                prompt += "- " + t + "\n"
                i += 1
        prompt += "\n"

        # Recent summaries
        prompt += "## RECENT CHAT SUMMARIES\n"
        if len(recent_summaries) == 0:
            prompt += "(none)\n"
        else:
            i = 0
            while i < len(recent_summaries):
                t = recent_summaries[i]
                if len(t) > 220:
                    t = t[:220] + "…"
                prompt += "- " + t + "\n"
                i += 1
        prompt += "\n"

        # Retrieved summaries
        prompt += "## RELEVANT PAST MEMORY\n"
        if len(retrieved_summaries) == 0:
            prompt += "(none)\n"
        else:
            i = 0
            while i < len(retrieved_summaries):
                t = retrieved_summaries[i]
                if len(t) > 220:
                    t = t[:220] + "…"
                prompt += "- " + t + "\n"
                i += 1
        prompt += "\n"

        # Uploaded context (only 1 chunk usually in normal mode)
        prompt += "## UPLOADED CONTEXT\n"
        if len(uploaded_context_texts) == 0:
            prompt += "(none)\n"
        else:
            i = 0
            while i < len(uploaded_context_texts):
                t = uploaded_context_texts[i]
                if len(t) > 300:
                    t = t[:300] + "…"
                prompt += "- " + t + "\n"
                i += 1
        prompt += "\n"

        prompt += "## USER QUESTION\n"
        prompt += user_message.strip() + "\n\n"
        prompt += "## ANSWER\n"

        return prompt

    # ----------------------------
    # Memory listing for viewer page
    # ----------------------------

    def list_profile_memories(self, user_id, limit=50):
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="profile")),
            ]
        )
        points = self.scroll(self.profile_collection, qdrant_filter, limit=limit)

        rows = []
        i = 0
        while i < len(points):
            payload = points[i].payload
            if payload is None:
                payload = {}
            rows.append(payload)
            i += 1
        return rows

    def list_chat_summaries(self, user_id, limit=50):
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_summary")),
            ]
        )
        points = self.scroll(self.chat_collection, qdrant_filter, limit=limit)

        rows = []
        i = 0
        while i < len(points):
            payload = points[i].payload
            if payload is None:
                payload = {}
            rows.append(payload)
            i += 1

        # bubble sort by created_at ascending
        j = 0
        while j < len(rows):
            k = j + 1
            while k < len(rows):
                a = rows[j].get("created_at", 0)
                b = rows[k].get("created_at", 0)
                if b < a:
                    tmp = rows[j]
                    rows[j] = rows[k]
                    rows[k] = tmp
                k += 1
            j += 1

        if len(rows) > limit:
            rows = rows[len(rows) - limit:]
        return rows

    def list_uploaded_context(self, user_id, limit=100):
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="uploaded_context")),
            ]
        )
        points = self.scroll(self.upload_collection, qdrant_filter, limit=limit)

        rows = []
        i = 0
        while i < len(points):
            payload = points[i].payload
            if payload is None:
                payload = {}
            rows.append(payload)
            i += 1
        return rows
