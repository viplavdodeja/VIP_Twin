import time
import json
import uuid
import os
import csv
from datetime import datetime
import requests
import threading
import queue

from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from core.context.router import classify_route
from core.context.prompt_builder import build_pass_a_prompt, build_pass_b_prompt, build_session_summary_prompt, build_claim_extraction_prompt
from core.storage.chat_store_mysql import ChatStoreMySQL


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
        self.http = None
        self.chat_store = None
        self.mysql_enabled = False
        self.mysql_connected = False
        self.qdrant_connected = False
        self.mode = "dev_no_memory"
        self.claims_collection = "user_claims"
        self.docs_collection = "user_docs"
        self.claim_job_queue = queue.Queue()
        self.claim_worker_started = False

        # Ask metrics logging
        base_dir = os.path.dirname(__file__)
        self.ask_metrics_csv_path = os.path.normpath(
            os.path.join(base_dir, "..", "logs", "ask_metrics.csv")
        )

        # Uploaded PDF chunks / image captions go here
        self.upload_collection = "uploaded_context"

        # Chat persistence settings
        # Generate and store a rolling summary every N chat messages (user+assistant)
        self.summary_every_n_messages = 10

    # ----------------------------
    # Setup
    # ----------------------------

    def init(self):
        mysql_required = os.environ.get("MYSQL_REQUIRED", "true").strip().lower() not in ("0", "false", "no")
        mysql_enabled = os.environ.get("MYSQL_ENABLED", "").strip().lower()

        if mysql_enabled in ("1", "true", "yes", "y"):
            host = os.environ.get("MYSQL_HOST", "127.0.0.1")
            port = int(os.environ.get("MYSQL_PORT", "3306"))
            user = os.environ.get("MYSQL_USER", "root")
            password = os.environ.get("MYSQL_PASSWORD", "")
            database = os.environ.get("MYSQL_DATABASE", "")
            try:
                self.chat_store = ChatStoreMySQL(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database
                )
                self.chat_store.init()
                self.mysql_enabled = True
                self.mysql_connected = True
                print("CHAT_STORE_MYSQL: enabled")
            except Exception as e:
                print("CHAT_STORE_MYSQL: disabled (init failed):", str(e))
                self.chat_store = None
                self.mysql_enabled = False
                self.mysql_connected = False
        else:
            self.mysql_enabled = False
            self.mysql_connected = False

        if self.mysql_enabled is False and mysql_required:
            raise RuntimeError("MySQL required as source of truth")

        self.mode = "mysql_truth" if self.mysql_enabled else "dev_no_memory"

        self.embedder = SentenceTransformer(self.embed_model_name)
        self.http = requests.Session()

        test_vector = self.embed_text("test")
        self.vector_size = len(test_vector)

        try:
            self.client = QdrantClient(url=self.qdrant_url)
            self.qdrant_connected = True
        except Exception as e:
            print("QDRANT: disabled (init failed):", str(e))
            self.client = None
            self.qdrant_connected = False

        if self.qdrant_connected:
            self.ensure_collection(self.profile_collection)
            self.ensure_collection(self.chat_collection)
            self.ensure_collection(self.upload_collection)
            self.ensure_collection(self.claims_collection)
            self.ensure_collection(self.docs_collection)

        if self.mysql_enabled:
            self._start_claim_worker()

    def ensure_collection(self, collection_name):
        if self.client is None:
            return
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

    def _start_claim_worker(self):
        if self.claim_worker_started:
            return
        self.claim_worker_started = True
        worker = threading.Thread(target=self._claim_worker_loop, daemon=True)
        worker.start()

    # ----------------------------
    # Embeddings
    # ----------------------------

    def embed_text(self, text):
        # Returns a numpy array; Qdrant client accepts it
        return self.embedder.encode(text)

    # ----------------------------
    # Ollama
    # ----------------------------

    def ollama_generate(
        self,
        prompt,
        max_tokens=500,
        temperature=0.2,
        timeout_seconds=240,
        num_ctx=None,
        return_stats=False
    ):
        url = "http://127.0.0.1:11434/api/generate"

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "USER:", "## USER QUESTION", "## ANSWER"]
        }
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 900,
            "options": options
        }

        response = self.http.post(url, json=payload, timeout=(5, timeout_seconds))
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        if return_stats:
            stats = {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count")
            }
            return text, stats
        return text

    def ollama_generate_stream(self, prompt, max_tokens=500, temperature=0.2, timeout_seconds=240, num_ctx=None):
        url = "http://127.0.0.1:11434/api/generate"

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "USER:", "## USER QUESTION", "## ANSWER"]
        }
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": True,
            "keep_alive": 900,
            "options": options
        }

        with self.http.post(url, json=payload, stream=True, timeout=(5, timeout_seconds)) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                stats = None
                if data.get("done"):
                    stats = {
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "eval_count": data.get("eval_count")
                    }
                if chunk:
                    yield chunk, stats
                if data.get("done"):
                    if chunk is None or len(chunk) == 0:
                        yield "", stats
                    break

    # ----------------------------
    # Chat/session
    # ----------------------------

    def start_chat(self, user_id):
        # MVP: stateless chat id
        chat_id = str(uuid.uuid4())
        if self.mysql_enabled:
            try:
                self.chat_store.ensure_chat_session_exists(chat_id, user_id)
            except Exception as e:
                print("CHAT_STORE_MYSQL: ensure_chat_session failed:", str(e))
        return chat_id

    # ----------------------------
    # Qdrant operations
    # ----------------------------

    def _coerce_point_id(self, point_id):
        if isinstance(point_id, str):
            s = point_id.strip()
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return point_id
        return point_id

    def upsert(self, collection_name, point_id, vector, payload):
        if self.client is None:
            return
        point_id = self._coerce_point_id(point_id)
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
        if self.client is None:
            return []
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
        if self.client is None:
            return []

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

    def add_profile_memory(self, user_id, category, text, scope="identity"):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for profile memory")

        memory_key = "profile:" + user_id + ":" + category
        created_at_dt = datetime.now()
        extra = {
            "scope": scope,
            "category": category,
            "memory_key": memory_key
        }
        extra_json = json.dumps(extra)
        mysql_id = self.chat_store.add_memory_item(
            user_id=user_id,
            chat_id=None,
            memory_type="profile_fact",
            role=None,
            content=text,
            created_at_dt=created_at_dt,
            extra_json=extra_json
        )

        if mysql_id is None:
            return None

        vector = self.embed_text(text)
        payload = {
            "user_id": user_id,
            "chat_id": None,
            "memory_type": "profile_fact",
            "mysql_id": mysql_id,
            "created_at": created_at_dt.isoformat(timespec="seconds")
        }
        self.upsert(self.profile_collection, str(mysql_id), vector, payload)
        return mysql_id

    def get_identity_profile_texts(self, user_id):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for profile memory")
        rows = self.chat_store.get_profile_items(user_id)
        texts = []
        i = 0
        while i < len(rows):
            extra_raw = rows[i].get("extra")
            scope = ""
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                    scope = extra.get("scope", "")
                except json.JSONDecodeError:
                    scope = ""
            if scope == "identity":
                text = rows[i].get("content", "")
                if text is not None and len(text) > 0:
                    texts.append(text)
            i += 1
        return texts

    def get_focus_profile_texts(self, user_id):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for profile memory")
        rows = self.chat_store.get_profile_items(user_id)
        texts = []
        i = 0
        while i < len(rows):
            extra_raw = rows[i].get("extra")
            scope = ""
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                    scope = extra.get("scope", "")
                except json.JSONDecodeError:
                    scope = ""
            if scope == "focus":
                text = rows[i].get("content", "")
                if text is not None and len(text) > 0:
                    texts.append(text)
            i += 1
        return texts

    def get_profile_texts(self, user_id):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for profile memory")
        rows = self.chat_store.get_profile_items(user_id)
        texts = []
        i = 0
        while i < len(rows):
            text = rows[i].get("content", "")
            if text is not None and len(text) > 0:
                texts.append(text)
            i += 1
        return texts

    # ----------------------------
    # Chat memory (messages + summaries)
    # ----------------------------

    def add_chat_message(self, user_id, chat_id, role, text, skip_vector=False):
        clean_text = ""
        if text is not None:
            clean_text = text.strip()

        if len(clean_text) == 0:
            return None

        created_at_epoch = int(time.time())
        created_at_dt = datetime.fromtimestamp(created_at_epoch)
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat messages")
        try:
            self.chat_store.ensure_chat_session_exists(chat_id, user_id)
            mysql_id = self.chat_store.add_memory_item(
                user_id=user_id,
                chat_id=chat_id,
                memory_type="chat_message",
                role=role,
                content=clean_text,
                created_at_dt=created_at_dt,
                extra_json=None
            )
            self.chat_store.update_chat_last_message_at(chat_id, created_at_dt)
        except Exception as e:
            print("CHAT_STORE_MYSQL_WRITE_ERROR:", str(e))
            raise

        if skip_vector:
            return None

        point_id = str(mysql_id)
        vector = self.embed_text(clean_text)

        payload = {
            "user_id": user_id,
            "chat_id": chat_id,
            "memory_type": "chat_message",
            "mysql_id": mysql_id,
            "created_at": created_at_dt.isoformat(timespec="seconds")
        }

        self.upsert(self.chat_collection, point_id, vector, payload)
        return mysql_id

    def add_chat_summary(self, user_id, chat_id, summary_text):
        clean_text = ""
        if summary_text is not None:
            clean_text = summary_text.strip()

        if len(clean_text) == 0:
            return None

        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat summaries")

        created_at_dt = datetime.fromtimestamp(int(time.time()))
        mysql_id = self.chat_store.add_memory_item(
            user_id=user_id,
            chat_id=chat_id,
            memory_type="chat_summary",
            role=None,
            content=clean_text,
            created_at_dt=created_at_dt,
            extra_json=None
        )

        vector = self.embed_text(clean_text)

        payload = {
            "user_id": user_id,
            "chat_id": chat_id,
            "memory_type": "chat_summary",
            "mysql_id": mysql_id,
            "created_at": created_at_dt.isoformat(timespec="seconds")
        }

        self.upsert(self.chat_collection, str(mysql_id), vector, payload)
        return mysql_id

    def get_chat_messages(self, user_id, chat_id, limit=20):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat messages")
        try:
            rows = self.chat_store.get_recent_messages(user_id, chat_id, limit)
            return rows if rows is not None else []
        except Exception as e:
            print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
            raise

    def get_chat_messages_fast(self, user_id, chat_id, limit=20, fetch_cap=200):
        return self.get_chat_messages(user_id, chat_id, limit=limit)

    def maybe_generate_chat_summary(self, user_id, chat_id):
        # Only summarize every N messages to avoid latency.
        # Returns "" if nothing was saved.
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat summaries")
        try:
            msgs = self.chat_store.get_last_n_messages(user_id, chat_id, 200)
        except Exception as e:
            print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
            raise
        msg_count = len(msgs)

        if msg_count == 0:
            return ""

        if self.summary_every_n_messages <= 0:
            return ""

        if (msg_count % self.summary_every_n_messages) != 0:
            return ""

        # Build a short transcript window
        window = 10
        if msg_count < window:
            window = msg_count

        start_index = msg_count - window
        if start_index < 0:
            start_index = 0

        transcript = ""
        i = start_index
        while i < msg_count:
            role = msgs[i].get("role", "")
            text = msgs[i].get("text", "")
            if role is None:
                role = ""
            if text is None:
                text = ""
            transcript += role.upper() + ": " + text + "\n"
            i += 1

        prompt = ""
        prompt += "Summarize the conversation so far in 4-6 bullet points.\n"
        prompt += "Focus on stable facts, preferences, goals, and decisions.\n"
        prompt += "Do not include private identifiers.\n\n"
        prompt += "CONVERSATION:\n"
        prompt += transcript
        prompt += "\nSUMMARY:\n"

        try:
            summary_text = self.ollama_generate(
                prompt,
                max_tokens=180,
                temperature=0.1,
                timeout_seconds=120
            ).strip()
        except Exception as e:
            print("SUMMARY_ERROR:", str(e))
            return ""

        saved = self.add_chat_summary(user_id, chat_id, summary_text)
        if saved is None:
            return ""
        return summary_text

    # ----------------------------
    # Chat summaries (optional; currently read-only)
    # ----------------------------

    def get_recent_summaries(self, user_id, chat_id, limit):
        # Kept for compatibility with your UI/debug.
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat summaries")
        rows = self.chat_store.get_chat_summaries(user_id, chat_id, limit)
        items = []
        i = 0
        while i < len(rows):
            text = rows[i].get("content", "")
            if text is not None and len(text) > 0:
                items.append(text)
            i += 1
        return items

    def retrieve_relevant_summaries(self, user_id, chat_id, query_text, top_k):
        query_vector = self.embed_text(query_text)

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="chat_summary"))
            ]
        )

        results = self.search(self.chat_collection, query_vector, top_k, qdrant_filter)
        ids = []
        i = 0
        while i < len(results):
            payload = results[i].payload or {}
            mysql_id = payload.get("mysql_id")
            if mysql_id is not None:
                ids.append(str(mysql_id))
            i += 1

        if self.mysql_enabled is False or self.chat_store is None:
            return []

        items = self.chat_store.get_memory_items_by_ids(user_id, ids)
        texts = []
        j = 0
        while j < len(items):
            content = items[j].get("content", "")
            if content is not None and len(content) > 0:
                texts.append(content)
            j += 1
        return texts

    # ----------------------------
    # Uploaded context (PDF chunks + image captions)
    # ----------------------------

    def retrieve_relevant_uploaded_context(self, user_id, query_text, top_k, file_id=None):
        query_vector = self.embed_text(query_text)

        must_conditions = []
        must_conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))
        must_conditions.append(models.FieldCondition(key="memory_type", match=models.MatchValue(value="doc_chunk")))

        qdrant_filter = models.Filter(must=must_conditions)

        results = self.search(self.upload_collection, query_vector, top_k, qdrant_filter)

        ids = []
        i = 0
        while i < len(results):
            payload = results[i].payload or {}
            mysql_id = payload.get("mysql_id")
            if mysql_id is not None:
                ids.append(str(mysql_id))
            i += 1

        if self.mysql_enabled is False or self.chat_store is None:
            return []

        items = self.chat_store.get_memory_items_by_ids(user_id, ids)
        texts = []
        j = 0
        while j < len(items):
            content = items[j].get("content", "")
            if content is not None and len(content) > 0:
                texts.append(content)
            j += 1
        return texts

    def enforce_context_budget(self, texts, max_chars):
        if max_chars <= 0:
            return []

        trimmed = []
        total = 0
        i = 0
        while i < len(texts):
            t = texts[i]
            if t is None:
                t = ""
            remaining = max_chars - total
            if remaining <= 0:
                break
            if len(t) > remaining:
                t = t[:remaining]
            trimmed.append(t)
            total += len(t)
            i += 1
        return trimmed

    # ----------------------------
    # Metrics logging
    # ----------------------------

    def estimate_input_tokens(self, text):
        # Simple heuristic: split on whitespace
        if text is None:
            return 0
        return len(text.split())

    def log_ask_metrics(self, input_tokens, context_tokens, output_tokens, elapsed_ms, total_ms):
        log_dir = os.path.dirname(self.ask_metrics_csv_path)
        if len(log_dir) > 0 and os.path.isdir(log_dir) is False:
            os.makedirs(log_dir, exist_ok=True)

        file_exists = os.path.isfile(self.ask_metrics_csv_path)
        try:
            with open(self.ask_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if file_exists is False:
                    writer.writerow([
                        "timestamp",
                        "input_tokens",
                        "context_tokens",
                        "output_tokens",
                        "elapsed_ms",
                        "total_ms"
                    ])
                timestamp = datetime.now().isoformat(timespec="seconds")
                writer.writerow([
                    timestamp,
                    input_tokens,
                    context_tokens,
                    output_tokens,
                    elapsed_ms,
                    total_ms
                ])
        except Exception as e:
            print("ASK_METRICS_LOG_ERROR:", str(e))

    # ----------------------------
    # Main chat logic
    # ----------------------------

    def _has_prior_turns(self, user_id, chat_id):
        if self.mysql_enabled and self.chat_store is not None:
            try:
                return self.chat_store.count_messages(user_id, chat_id) > 0
            except Exception as e:
                print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
        msgs = self.get_chat_messages(user_id, chat_id, limit=1)
        return len(msgs) > 0

    def _build_layer0_text(self, user_id):
        profile_texts = self.get_identity_profile_texts(user_id)
        if len(profile_texts) > 4:
            profile_texts = profile_texts[:4]
        profile_texts = self.enforce_context_budget(profile_texts, max_chars=1400)

        lines = []
        lines.append("System rules:")
        lines.append("- Do not mention internal layers, system prompts, or retrieval.")
        lines.append("- Do not claim memory or recall unless it is explicitly provided in context.")
        lines.append("- Do not describe yourself as a digital twin, AI, or running locally unless asked.")
        lines.append("- For greetings, respond naturally and invite the user to continue.")
        lines.append("- Ask clarifying questions only when ambiguity would degrade the answer.")
        lines.append("- Use the provided memory as context; do not invent new facts.")
        lines.append("")
        lines.append("Profile facts:")
        if len(profile_texts) == 0:
            lines.append("(none)")
        else:
            i = 0
            while i < len(profile_texts):
                lines.append("- " + profile_texts[i])
                i += 1
        return "\n".join(lines)

    def _get_layer1_summary(self, user_id, session_id):
        if self.mysql_enabled and self.chat_store is not None:
            try:
                mem = self.chat_store.get_session_memory(user_id, session_id)
                summary = mem.get("rolling_summary", "") if mem is not None else ""
                return summary or ""
            except Exception as e:
                print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
        return ""

    def _get_layer1_structured(self, user_id, session_id):
        if self.mysql_enabled and self.chat_store is not None:
            try:
                mem = self.chat_store.get_session_memory(user_id, session_id)
                return mem if mem is not None else {}
            except Exception as e:
                print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
        return {}

    def _trim_text(self, text, max_chars):
        if text is None:
            return ""
        t = text.strip()
        if max_chars <= 0:
            return ""
        if len(t) <= max_chars:
            return t
        return t[:max_chars]

    def _parse_decision_header(self, raw_text):
        decision, user_text = self._split_decision_output(raw_text)
        decision["assistant_text"] = user_text
        return decision

    def _fallback_user_text(self, action):
        if action == "ASK":
            return "Could you clarify what you want me to do with that?"
        return "Got it. What would you like to do next?"

    def _is_greeting(self, text):
        if text is None:
            return False
        msg = text.strip().lower()
        if len(msg) == 0:
            return False
        greetings = [
            "hi", "hello", "hey", "hiya", "yo",
            "good morning", "good afternoon", "good evening"
        ]
        i = 0
        while i < len(greetings):
            if msg == greetings[i]:
                return True
            i += 1
        return False

    def _greeting_response(self):
        return "Hey! Good to see you. What would you like to work on?"

    def _has_any_doc_chunks(self, user_id):
        if self.mysql_enabled is False or self.chat_store is None:
            return False
        try:
            items = self.chat_store.get_doc_chunks(user_id, limit=1)
            return True if items and len(items) > 0 else False
        except Exception as e:
            print("DOC_CHUNK_CHECK_ERROR:", str(e))
            return False

    def _should_force_doc_retrieval(self, user_id, user_message):
        if user_message is None:
            return False
        msg = user_message.strip().lower()
        if len(msg) == 0:
            return False
        keywords = [
            "pdf", "document", "doc", "transcript", "assignment", "homework",
            "notes", "lecture", "syllabus", "slides", "reading", "paper", "report"
        ]
        hit = False
        i = 0
        while i < len(keywords):
            if keywords[i] in msg:
                hit = True
                break
            i += 1
        if hit is False:
            return False
        return self._has_any_doc_chunks(user_id)

    def _split_decision_output(self, raw_text):
        # Minimal regression protection for decision header parsing.
        fallback_decision = {
            "action": "DIRECT",
            "next_layer": None,
            "reason": ""
        }
        if raw_text is None:
            return fallback_decision, ""

        lines = raw_text.splitlines()
        header_index = None
        i = 0
        while i < len(lines):
            if len(lines[i].strip()) > 0:
                header_index = i
                break
            i += 1

        if header_index is None:
            return fallback_decision, ""

        header_line = lines[header_index].strip()
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError:
            return fallback_decision, raw_text.strip()

        if not isinstance(header, dict):
            return fallback_decision, raw_text.strip()

        action = str(header.get("action", "DIRECT")).upper()
        next_layer = header.get("next_layer")
        reason = header.get("reason", "")
        if action not in ("DIRECT", "ASK", "ESCALATE"):
            return fallback_decision, raw_text.strip()
        if action == "ESCALATE" and next_layer not in ("L2", "L3", "L4"):
            return fallback_decision, raw_text.strip()

        assistant_text = "\n".join(lines[header_index + 1:]).strip()
        return {
            "action": action,
            "next_layer": next_layer,
            "reason": reason
        }, assistant_text

    def _self_check_decision_parser(self):
        samples = [
            "{\"action\":\"DIRECT\",\"next_layer\":null,\"reason\":\"ok\"}\nHello there.",
            "Just text, no header.",
            "   \n  {\"action\":\"ASK\",\"next_layer\":null,\"reason\":\"need more\"}\nWhat do you mean?",
            "{\"action\":\"DIRECT\",\"next_layer\":null,\"reason\":bad}\nHello"
        ]
        results = []
        i = 0
        while i < len(samples):
            decision, text = self._split_decision_output(samples[i])
            results.append({"decision": decision, "text": text})
            i += 1
        return results

    def _build_recent_turns_text(self, messages):
        lines = []
        i = 0
        while i < len(messages):
            role = messages[i].get("role", "")
            text = messages[i].get("text", "")
            if role is None:
                role = ""
            if text is None:
                text = ""
            lines.append(role.upper() + ": " + text)
            i += 1
        return "\n".join(lines)

    def _format_layer1_text(self, mem):
        if mem is None:
            return ""
        summary = mem.get("rolling_summary", "") or ""
        threads = mem.get("active_threads")
        loops = mem.get("open_loops")
        prefs = mem.get("short_term_prefs")

        lines = []
        lines.append("Rolling summary:")
        lines.append(summary if len(summary.strip()) > 0 else "(none)")
        lines.append("")
        lines.append("Active threads:")
        lines.append(json.dumps(threads) if threads is not None else "[]")
        lines.append("")
        lines.append("Open loops:")
        lines.append(json.dumps(loops) if loops is not None else "[]")
        lines.append("")
        lines.append("Short-term prefs:")
        lines.append(json.dumps(prefs) if prefs is not None else "{}")
        return "\n".join(lines)

    def _sanitize_summary(self, summary_text):
        if summary_text is None:
            return ""
        text = summary_text.strip()
        if len(text) == 0:
            return ""
        raw_lines = text.splitlines()
        lines = []
        i = 0
        while i < len(raw_lines):
            stripped = raw_lines[i].strip()
            if len(stripped) > 0:
                lines.append(stripped)
            i += 1
        bullets = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("-"):
                bullets.append(line if line.startswith("- ") else "- " + line.lstrip("- ").strip())
            else:
                bullets.append("- " + line)
            i += 1
        if len(bullets) > 6:
            bullets = bullets[:6]
        return "\n".join(bullets)

    def _truncate_bullets(self, text, max_chars):
        if text is None:
            return ""
        raw_lines = text.splitlines()
        lines = []
        i = 0
        while i < len(raw_lines):
            stripped = raw_lines[i].strip()
            if len(stripped) > 0:
                lines.append(stripped)
            i += 1
        bullets = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("-"):
                bullets.append(line if line.startswith("- ") else "- " + line.lstrip("- ").strip())
            else:
                bullets.append("- " + line)
            i += 1
        if len(bullets) == 0:
            return ""
        trimmed = []
        total = 0
        j = len(bullets) - 1
        while j >= 0:
            t = bullets[j]
            if total + len(t) + 1 > max_chars:
                j -= 1
                continue
            trimmed.append(t)
            total += len(t) + 1
            j -= 1
        if len(trimmed) == 0:
            return bullets[-1][:max_chars]
        trimmed.reverse()
        return "\n".join(trimmed)

    def _claims_to_text(self, claims, max_chars):
        if claims is None or len(claims) == 0:
            return ""
        lines = []
        i = 0
        while i < len(claims):
            claim = claims[i]
            ctype = claim.get("claim_type", "fact")
            ctext = claim.get("claim_text", "")
            if ctext:
                lines.append("- [" + str(ctype) + "] " + ctext)
            i += 1
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _rerank_claims(self, claims, top_k):
        if claims is None or len(claims) == 0:
            return []
        now = time.time()
        scored = []
        i = 0
        while i < len(claims):
            claim = claims[i]
            confidence = float(claim.get("confidence") or 0.0)
            last_seen = claim.get("last_seen_at")
            if isinstance(last_seen, datetime):
                age_days = max((now - time.mktime(last_seen.timetuple())) / 86400.0, 0.0)
            else:
                age_days = 365.0
            recency_score = 1.0 / (1.0 + age_days)
            score = (0.7 * confidence) + (0.3 * recency_score)
            scored.append((score, claim))
            i += 1
        scored.sort(key=lambda pair: pair[0], reverse=True)
        trimmed = []
        j = 0
        while j < len(scored) and j < top_k:
            trimmed.append(scored[j][1])
            j += 1
        return trimmed

    def _retrieve_layer2_claims(self, user_id, user_message, rolling_summary, top_k=8):
        if self.mysql_enabled is False or self.chat_store is None:
            return []
        query_text = user_message.strip()
        if rolling_summary:
            query_text += "\n\nSession summary:\n" + rolling_summary
        query_vector = self.embed_text(query_text)

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="claim"))
            ]
        )

        results = self.search(self.claims_collection, query_vector, top_k * 2, qdrant_filter)
        claim_ids = []
        i = 0
        while i < len(results):
            payload = results[i].payload or {}
            mysql_id = payload.get("mysql_id")
            if mysql_id is not None:
                claim_ids.append(int(mysql_id))
            i += 1

        claims = self.chat_store.get_claims_by_ids(user_id, claim_ids)
        active_claims = []
        j = 0
        while j < len(claims):
            if claims[j].get("status") == "active":
                active_claims.append(claims[j])
            j += 1
        return self._rerank_claims(active_claims, top_k)

    def _extract_keywords(self, text, limit=6):
        if text is None:
            return []
        words = []
        parts = text.lower().split()
        p = 0
        while p < len(parts):
            part = parts[p]
            chars = []
            c = 0
            while c < len(part):
                if part[c].isalnum():
                    chars.append(part[c])
                c += 1
            w = "".join(chars)
            if len(w) >= 4:
                words.append(w)
            p += 1
        seen = set()
        uniq = []
        i = 0
        while i < len(words):
            if words[i] not in seen:
                seen.add(words[i])
                uniq.append(words[i])
            i += 1
        return uniq[:limit]

    def _keyword_overlap_score(self, query_text, snippet_text):
        q_terms = self._extract_keywords(query_text, limit=10)
        if len(q_terms) == 0 or snippet_text is None:
            return 0.0
        s_terms = set(self._extract_keywords(snippet_text, limit=30))
        match = 0
        i = 0
        while i < len(q_terms):
            if q_terms[i] in s_terms:
                match += 1
            i += 1
        return float(match) / float(len(q_terms))

    def _why_snippet_matters(self, user_message, snippet_text):
        terms = self._extract_keywords(user_message, limit=4)
        if len(terms) == 0:
            return "Relevant to the user's question."
        return "Matches query terms: " + ", ".join(terms) + "."

    def _evidence_to_text(self, evidence_items, max_chars):
        if evidence_items is None or len(evidence_items) == 0:
            return ""
        lines = []
        total = 0
        i = 0
        while i < len(evidence_items):
            item = evidence_items[i]
            snippet = item.get("snippet", "")
            source = item.get("source", "")
            why = item.get("why", "")
            block = []
            block.append("- Snippet: " + snippet)
            block.append("  Source: " + source)
            block.append("  Why: " + why)
            text = "\n".join(block)
            if total + len(text) + 1 > max_chars:
                break
            lines.append(text)
            total += len(text) + 1
            i += 1
        return "\n".join(lines)

    def _retrieve_layer3_evidence(self, user_id, user_message, rolling_summary, top_k=6):
        if self.mysql_enabled is False or self.chat_store is None:
            return []
        query_text = user_message.strip()
        if rolling_summary:
            query_text += "\n\nSession summary:\n" + rolling_summary
        query_vector = self.embed_text(query_text)

        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="memory_type", match=models.MatchValue(value="doc_chunk"))
            ]
        )

        results = self.search(self.docs_collection, query_vector, top_k * 3, qdrant_filter)
        scored = []
        i = 0
        while i < len(results):
            payload = results[i].payload or {}
            mysql_id = payload.get("mysql_id")
            if mysql_id is None:
                i += 1
                continue
            score = getattr(results[i], "score", 0.0) or 0.0
            scored.append((str(mysql_id), score))
            i += 1

        chunk_ids = []
        j = 0
        while j < len(scored):
            chunk_ids.append(scored[j][0])
            j += 1

        items = self.chat_store.get_memory_items_by_ids(user_id, chunk_ids)
        chunk_map = {}
        k = 0
        while k < len(items):
            chunk_map[str(items[k]["id"])] = items[k]
            k += 1

        reranked = []
        k = 0
        while k < len(scored):
            cid, sem_score = scored[k]
            chunk = chunk_map.get(cid)
            if chunk is None:
                k += 1
                continue
            keyword_score = self._keyword_overlap_score(user_message, chunk.get("content", ""))
            score = (0.8 * sem_score) + (0.2 * keyword_score)
            reranked.append((score, chunk))
            k += 1

        reranked.sort(key=lambda pair: pair[0], reverse=True)
        top_chunks = []
        m2 = 0
        while m2 < len(reranked) and m2 < top_k:
            top_chunks.append(reranked[m2][1])
            m2 += 1

        doc_ids = []
        m = 0
        while m < len(top_chunks):
            extra_raw = top_chunks[m].get("extra") or ""
            doc_id_val = None
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                    doc_id_val = extra.get("document_id")
                except json.JSONDecodeError:
                    doc_id_val = None
            if doc_id_val is not None:
                doc_id_str = str(doc_id_val)
                if doc_id_str not in doc_ids:
                    doc_ids.append(doc_id_str)
            m += 1

        docs = self.chat_store.get_documents_by_ids(doc_ids)
        doc_map = {}
        n = 0
        while n < len(docs):
            doc_map[str(docs[n].get("doc_id"))] = docs[n]
            n += 1

        evidence = []
        m = 0
        while m < len(top_chunks):
            c = top_chunks[m]
            snippet = c.get("content", "")
            if len(snippet) > 360:
                snippet = snippet[:360] + "..."
            meta = {}
            doc_id = None
            chunk_index = None
            extra_raw = c.get("extra") or ""
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                except json.JSONDecodeError:
                    extra = {}
                meta = extra.get("metadata") or {}
                doc_id = extra.get("document_id")
                chunk_index = extra.get("chunk_index")
            doc = doc_map.get(str(doc_id), {}) if doc_id is not None else {}
            source_parts = [
                "doc_id=" + str(doc_id),
                "chunk_id=" + str(c.get("id")),
                "chunk_index=" + str(chunk_index)
            ]
            if "page" in meta:
                source_parts.append("page=" + str(meta.get("page")))
            if "timecode" in meta:
                source_parts.append("time=" + str(meta.get("timecode")))
            if doc.get("title"):
                source_parts.append("title=" + str(doc.get("title")))
            source = ", ".join(source_parts)
            evidence.append({
                "snippet": snippet,
                "source": source,
                "why": self._why_snippet_matters(user_message, snippet)
            })
            m += 1

        self._log_retrieval_run(user_id, "L3", query_text, top_k)
        return evidence

    def _log_retrieval_run(self, user_id, layer, query_text, top_k):
        if self.mysql_enabled is False or self.chat_store is None:
            return
        try:
            self.chat_store.log_retrieval_run(user_id, layer, query_text, top_k)
        except Exception as e:
            print("RETRIEVAL_LOG_ERROR:", str(e))

    def index_document(
        self,
        doc_id,
        user_id,
        title,
        doc_type,
        source,
        filename,
        mime_type,
        file_size_bytes,
        sha256,
        chunks
    ):
        if self.mysql_enabled is False or self.chat_store is None:
            return None
        doc_id = self.chat_store.insert_document(
            doc_id=doc_id,
            user_id=user_id,
            title=title,
            doc_type=doc_type,
            source=source,
            filename=filename,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            sha256=sha256
        )
        if doc_id is None:
            return None
        i = 0
        while i < len(chunks):
            chunk_text = chunks[i].get("text", "")
            chunk_index = int(chunks[i].get("chunk_index", i))
            metadata = chunks[i].get("metadata") or {}
            chunk_id = self.chat_store.insert_document_chunk(
                doc_id, user_id, chunk_index, chunk_text
            )
            extra = {
                "document_id": doc_id,
                "chunk_index": chunk_index,
                "metadata": metadata
            }
            extra_json = json.dumps(extra) if extra else None
            mem_id = self.chat_store.add_memory_item(
                user_id=user_id,
                chat_id=None,
                memory_type="doc_chunk",
                role=None,
                content=chunk_text,
                created_at_dt=datetime.now(),
                extra_json=extra_json
            )
            if mem_id is not None:
                vector = self.embed_text(chunk_text)
                payload = {
                    "user_id": user_id,
                    "chat_id": None,
                    "memory_type": "doc_chunk",
                    "mysql_id": mem_id,
                    "created_at": datetime.now().isoformat(timespec="seconds")
                }
                self.upsert(self.docs_collection, str(mem_id), vector, payload)
            i += 1
        return doc_id

    def _enqueue_claim_extraction(self, user_id, session_id):
        if self.mysql_enabled is False:
            return
        self.claim_job_queue.put({
            "user_id": user_id,
            "session_id": session_id
        })

    def _claim_worker_loop(self):
        while True:
            job = self.claim_job_queue.get()
            try:
                self._process_claim_job(job)
            except Exception as e:
                print("CLAIM_WORKER_ERROR:", str(e))
            finally:
                self.claim_job_queue.task_done()

    def _process_claim_job(self, job):
        user_id = job.get("user_id")
        session_id = job.get("session_id")
        if not user_id or not session_id or self.chat_store is None:
            return

        rolling_summary = self._get_layer1_summary(user_id, session_id)
        rolling_summary = self._truncate_bullets(rolling_summary, max_chars=1200)
        recent_msgs = self.get_chat_messages(user_id, session_id, limit=6)
        if len(recent_msgs) == 0:
            return
        recent_turns_text = self._build_recent_turns_text(recent_msgs)

        prompt = build_claim_extraction_prompt(rolling_summary, recent_turns_text)
        try:
            raw = self.ollama_generate(
                prompt,
                max_tokens=220,
                temperature=0.1,
                timeout_seconds=120
            )
        except Exception as e:
            print("CLAIM_EXTRACT_ERROR:", str(e))
            return

        claims = self._parse_claims_json(raw)
        i = 0
        while i < len(claims):
            self._upsert_claim(user_id, claims[i])
            i += 1

    def _parse_claims_json(self, raw_text):
        if raw_text is None:
            return []
        text = raw_text.strip()
        if len(text) == 0:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        cleaned = []
        i = 0
        while i < len(data):
            item = data[i] or {}
            claim_type = str(item.get("claim_type", "fact"))
            claim_text = str(item.get("claim_text", "")).strip()
            if len(claim_text) == 0:
                i += 1
                continue
            confidence = float(item.get("confidence", 0.5))
            supersedes = item.get("supersedes")
            cleaned.append({
                "claim_type": claim_type,
                "claim_text": claim_text,
                "confidence": max(0.0, min(1.0, confidence)),
                "supersedes": supersedes
            })
            i += 1
        return cleaned

    def _upsert_claim(self, user_id, claim):
        claim_text = claim.get("claim_text", "")
        claim_type = claim.get("claim_type", "fact")
        confidence = float(claim.get("confidence") or 0.5)
        supersedes = claim.get("supersedes")

        claim_id = None
        if supersedes is not None:
            try:
                superseded_id = int(supersedes)
                claim_id = self.chat_store.insert_claim(
                    user_id, claim_type, claim_text, confidence, status="active"
                )
                self.chat_store.mark_claim_superseded(superseded_id, claim_id)
            except Exception:
                claim_id = None

        if claim_id is None:
            vector = self.embed_text(claim_text)
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
                ]
            )
            results = self.search(self.claims_collection, vector, 1, qdrant_filter)
            if results and len(results) > 0:
                top = results[0]
                score = getattr(top, "score", 0.0) or 0.0
                payload = top.payload or {}
                existing_id = payload.get("claim_id")
                if existing_id is not None and score >= 0.86:
                    try:
                        existing_id = int(existing_id)
                        self.chat_store.update_claim(existing_id, claim_text, confidence)
                        payload_update = {
                            "user_id": user_id,
                            "chat_id": None,
                            "memory_type": "claim",
                            "mysql_id": existing_id,
                            "created_at": datetime.now().isoformat(timespec="seconds")
                        }
                        self.upsert(self.claims_collection, str(existing_id), vector, payload_update)
                        return
                    except Exception:
                        pass

            claim_id = self.chat_store.insert_claim(
                user_id, claim_type, claim_text, confidence, status="active"
            )

        if claim_id is None:
            return

        vector = self.embed_text(claim_text)
        payload = {
            "user_id": user_id,
            "chat_id": None,
            "memory_type": "claim",
            "mysql_id": claim_id,
            "created_at": datetime.now().isoformat(timespec="seconds")
        }
        self.upsert(self.claims_collection, str(claim_id), vector, payload)

    def _update_session_memory(self, user_id, session_id):
        if self.mysql_enabled is False or self.chat_store is None:
            return ""

        try:
            existing = self.chat_store.get_session_memory(user_id, session_id)
            previous_summary = existing.get("rolling_summary", "") if existing else ""
            previous_threads = json.dumps(existing.get("active_threads") if existing else None)
            previous_loops = json.dumps(existing.get("open_loops") if existing else None)
            previous_prefs = json.dumps(existing.get("short_term_prefs") if existing else None)
        except Exception as e:
            print("CHAT_STORE_MYSQL_READ_ERROR:", str(e))
            previous_summary = ""
            previous_threads = "[]"
            previous_loops = "[]"
            previous_prefs = "{}"

        recent_msgs = self.get_chat_messages(user_id, session_id, limit=12)
        if len(recent_msgs) == 0:
            return previous_summary

        recent_turns_text = self._build_recent_turns_text(recent_msgs)
        prompt = build_session_summary_prompt(
            previous_summary,
            recent_turns_text,
            previous_threads,
            previous_loops,
            previous_prefs
        )
        try:
            summary_text = self.ollama_generate(
                prompt,
                max_tokens=200,
                temperature=0.1,
                timeout_seconds=120
            )
        except Exception as e:
            print("SESSION_SUMMARY_ERROR:", str(e))
            return previous_summary

        try:
            data = json.loads(summary_text)
        except json.JSONDecodeError:
            data = {}

        rolling_summary = data.get("rolling_summary") if isinstance(data, dict) else None
        active_threads = data.get("active_threads") if isinstance(data, dict) else None
        open_loops = data.get("open_loops") if isinstance(data, dict) else None
        short_term_prefs = data.get("short_term_prefs") if isinstance(data, dict) else None

        cleaned = self._sanitize_summary(rolling_summary if isinstance(rolling_summary, str) else "")
        if len(cleaned) == 0:
            cleaned = self._sanitize_summary(previous_summary)

        try:
            self.chat_store.upsert_session_memory(
                user_id=user_id,
                session_id=session_id,
                rolling_summary=cleaned,
                active_threads=active_threads,
                open_loops=open_loops,
                short_term_prefs=short_term_prefs
            )
        except Exception as e:
            print("SESSION_MEMORY_WRITE_ERROR:", str(e))
        return cleaned

    def ask(self, user_id, chat_id, user_message):
        t0 = time.perf_counter()
        fast_mode = len(user_message.strip()) <= 20
        has_prior_turns = self._has_prior_turns(user_id, chat_id)
        route = classify_route(user_message, has_prior_turns=has_prior_turns)
        # Persist the user chat message so chat history survives restarts.
        self.add_chat_message(user_id, chat_id, role="user", text=user_message, skip_vector=fast_mode)
        t1 = time.perf_counter()

        if self._is_greeting(user_message):
            assistant_message = self._greeting_response()
            self.add_chat_message(user_id, chat_id, role="assistant", text=assistant_message, skip_vector=True)
            timings = {
                "add_user_msg_ms": int((t1 - t0) * 1000),
                "build_prompt_ms": 0,
                "ollama_generate_ms": 0,
                "add_assistant_msg_ms": 0,
                "summary_ms": 0,
                "total_ms": int((time.perf_counter() - t0) * 1000),
            }
            print("ASK_TIMINGS:", timings)
            return {
                "assistant_message": assistant_message,
                "summary_saved": "",
                "context_used": {
                    "profile_count": 0,
                    "recent_chat_count": 0,
                    "retrieved_summary_count": 0,
                    "retrieved_upload_count": 0,
                    "doc_mode": False
                }
            }

        layer0_text = self._build_layer0_text(user_id)
        layer1_text = ""
        if route.include_layer1:
            mem = self._get_layer1_structured(user_id, chat_id)
            layer1_text = self._format_layer1_text(mem)
            layer1_text = self._trim_text(layer1_text, max_chars=1400)

        prompt = build_pass_a_prompt(layer0_text, layer1_text, user_message)
        t2 = time.perf_counter()

        prompt_chars = len(prompt)
        print("PROMPT_CHARS:", prompt_chars)
        print("PROMPT_KB:", round(prompt_chars / 1024, 2))
        print("LAYER1_INCLUDED:", route.include_layer1)


        # ---- Generation settings ----
        max_tokens = 300
        temp = 0.2
        num_ctx = None
        if fast_mode:
            max_tokens = 40
            temp = 0.2
            num_ctx = 512


        stats = {}
        raw_output = ""
        try:
            raw_output, stats = self.ollama_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                timeout_seconds=180,
                num_ctx=num_ctx,
                return_stats=True
            )
            raw_output = raw_output.strip()
        except Exception as e:
            print("OLLAMA_ERROR:", str(e))
            raw_output = (
                "I couldn’t generate a response in time (local model latency). "
                "Try again, or switch to a faster local model for the demo."
            )

        decision = self._parse_decision_header(raw_output)
        if decision.get("action") == "ESCALATE" and decision.get("next_layer") == "L2":
            rolling_summary = layer1_text
            claims = self._retrieve_layer2_claims(
                user_id, user_message, rolling_summary, top_k=8
            )
            layer2_text = self._claims_to_text(claims, max_chars=1400)
            pass_b_prompt = build_pass_b_prompt(layer0_text, layer1_text, layer2_text, "", user_message)
            try:
                assistant_message = self.ollama_generate(
                    pass_b_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    timeout_seconds=180,
                    num_ctx=num_ctx
                ).strip()
            except Exception as e:
                print("OLLAMA_ERROR:", str(e))
                assistant_message = (
                    "I couldn't generate a response in time. Try again in a moment."
                )
        elif decision.get("action") == "ESCALATE" and decision.get("next_layer") == "L3":
            rolling_summary = layer1_text
            evidence = self._retrieve_layer3_evidence(
                user_id, user_message, rolling_summary, top_k=6
            )
            layer3_text = self._evidence_to_text(evidence, max_chars=1800)
            pass_b_prompt = build_pass_b_prompt(layer0_text, layer1_text, "", layer3_text, user_message)
            try:
                assistant_message = self.ollama_generate(
                    pass_b_prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    timeout_seconds=180,
                    num_ctx=num_ctx
                ).strip()
            except Exception as e:
                print("OLLAMA_ERROR:", str(e))
                assistant_message = (
                    "I couldn't generate a response in time. Try again in a moment."
                )
        elif decision.get("action") == "ESCALATE":
            assistant_message = (
                "I can answer that better if you tell me which prior doc or session you mean."
            )
        else:
            assistant_message = decision.get("assistant_text", "").strip()
            if len(assistant_message) == 0:
                assistant_message = self._fallback_user_text(decision.get("action"))

        t3 = time.perf_counter()

        # Persist assistant reply
        self.add_chat_message(user_id, chat_id, role="assistant", text=assistant_message, skip_vector=fast_mode)
        t4 = time.perf_counter()

        # Update rolling session memory
        summary_saved = ""
        if fast_mode is False and len(user_message.strip()) >= 40:
            summary_saved = self._update_session_memory(user_id, chat_id)
            self._enqueue_claim_extraction(user_id, chat_id)
        t5 = time.perf_counter()

        timings = {
            "add_user_msg_ms": int((t1 - t0) * 1000),
            "build_prompt_ms": int((t2 - t1) * 1000),
            "ollama_generate_ms": int((t3 - t2) * 1000),
            "add_assistant_msg_ms": int((t4 - t3) * 1000),
            "summary_ms": int((t5 - t4) * 1000),
            "total_ms": int((t5 - t0) * 1000),
        }
        print("ASK_TIMINGS:", timings)

        prompt_eval_count = stats.get("prompt_eval_count")
        eval_count = stats.get("eval_count")
        input_tokens = self.estimate_input_tokens(user_message)
        context_tokens = 0
        if isinstance(prompt_eval_count, int):
            context_tokens = max(prompt_eval_count - input_tokens, 0)
        output_tokens = eval_count if isinstance(eval_count, int) else 0
        self.log_ask_metrics(
            input_tokens=input_tokens,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            elapsed_ms=timings.get("ollama_generate_ms", 0),
            total_ms=timings.get("total_ms", 0)
        )

        result = {
            "assistant_message": assistant_message,
            "summary_saved": summary_saved,
            "context_used": {
                "profile_count": 0,
                "recent_chat_count": 0,
                "retrieved_summary_count": 0,
                "retrieved_upload_count": 0,
                "doc_mode": False
            }
        }
        return result

    def ask_stream(self, user_id, chat_id, user_message):
        t0 = time.perf_counter()
        fast_mode = len(user_message.strip()) <= 20
        has_prior_turns = self._has_prior_turns(user_id, chat_id)
        route = classify_route(user_message, has_prior_turns=has_prior_turns)
        force_doc_retrieval = self._should_force_doc_retrieval(user_id, user_message)
        # Persist the user chat message so chat history survives restarts.
        self.add_chat_message(user_id, chat_id, role="user", text=user_message, skip_vector=fast_mode)
        t1 = time.perf_counter()

        if self._is_greeting(user_message):
            assistant_message = self._greeting_response()
            def gen():
                yield json.dumps({"type": "delta", "text": assistant_message}) + "\n"
                self.add_chat_message(user_id, chat_id, role="assistant", text=assistant_message, skip_vector=True)
                result = {
                    "assistant_message": assistant_message,
                    "summary_saved": "",
                    "context_used": {
                        "profile_count": 0,
                        "recent_chat_count": 0,
                        "retrieved_summary_count": 0,
                        "retrieved_upload_count": 0,
                        "doc_mode": False
                    }
                }
                yield json.dumps({"type": "done", **result}) + "\n"
            return gen()

        layer0_text = self._build_layer0_text(user_id)
        layer1_text = ""
        if route.include_layer1:
            mem = self._get_layer1_structured(user_id, chat_id)
            layer1_text = self._format_layer1_text(mem)
            layer1_text = self._trim_text(layer1_text, max_chars=1400)

        prompt = build_pass_a_prompt(layer0_text, layer1_text, user_message)
        t2 = time.perf_counter()

        prompt_chars = len(prompt)
        print("PROMPT_CHARS:", prompt_chars)
        print("PROMPT_KB:", round(prompt_chars / 1024, 2))
        print("LAYER1_INCLUDED:", route.include_layer1)

        # ---- Generation settings ----
        max_tokens = 300
        temp = 0.2
        num_ctx = None
        if fast_mode:
            max_tokens = 40
            temp = 0.2
            num_ctx = 512

        def gen():
            stats = {}
            raw_output = ""
            if force_doc_retrieval:
                print("LAYER_REACHED: L3 (FORCED)")
                rolling_summary = layer1_text
                evidence = self._retrieve_layer3_evidence(
                    user_id, user_message, rolling_summary, top_k=6
                )
                layer3_text = self._evidence_to_text(evidence, max_chars=1800)
                pass_b_prompt = build_pass_b_prompt(layer0_text, layer1_text, "", layer3_text, user_message)
                try:
                    assistant_message = self.ollama_generate(
                        pass_b_prompt,
                        max_tokens=max_tokens,
                        temperature=temp,
                        timeout_seconds=180,
                        num_ctx=num_ctx
                    ).strip()
                except Exception as e:
                    print("OLLAMA_ERROR:", str(e))
                    assistant_message = (
                        "I couldn't generate a response in time. Try again in a moment."
                    )
            else:
                try:
                    raw_output, stats = self.ollama_generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temp,
                        timeout_seconds=180,
                        num_ctx=num_ctx,
                        return_stats=True
                    )
                    raw_output = raw_output.strip()
                except Exception as e:
                    print("OLLAMA_ERROR:", str(e))
                    raw_output = (
                        "I couldn't generate a response in time (local model latency). "
                        "Try again, or switch to a faster local model for the demo."
                    )

                decision = self._parse_decision_header(raw_output)
                print("DECISION_ACTION:", decision.get("action"), "NEXT_LAYER:", decision.get("next_layer"))
                if decision.get("action") == "ESCALATE" and decision.get("next_layer") == "L2":
                    print("LAYER_REACHED: L2")
                    rolling_summary = layer1_text
                    claims = self._retrieve_layer2_claims(
                        user_id, user_message, rolling_summary, top_k=8
                    )
                    layer2_text = self._claims_to_text(claims, max_chars=1400)
                    pass_b_prompt = build_pass_b_prompt(layer0_text, layer1_text, layer2_text, "", user_message)
                    try:
                        assistant_message = self.ollama_generate(
                            pass_b_prompt,
                            max_tokens=max_tokens,
                            temperature=temp,
                            timeout_seconds=180,
                            num_ctx=num_ctx
                        ).strip()
                    except Exception as e:
                        print("OLLAMA_ERROR:", str(e))
                        assistant_message = (
                            "I couldn't generate a response in time. Try again in a moment."
                        )
                elif decision.get("action") == "ESCALATE" and decision.get("next_layer") == "L3":
                    print("LAYER_REACHED: L3")
                    rolling_summary = layer1_text
                    evidence = self._retrieve_layer3_evidence(
                        user_id, user_message, rolling_summary, top_k=6
                    )
                    layer3_text = self._evidence_to_text(evidence, max_chars=1800)
                    pass_b_prompt = build_pass_b_prompt(layer0_text, layer1_text, "", layer3_text, user_message)
                    try:
                        assistant_message = self.ollama_generate(
                            pass_b_prompt,
                            max_tokens=max_tokens,
                            temperature=temp,
                            timeout_seconds=180,
                            num_ctx=num_ctx
                        ).strip()
                    except Exception as e:
                        print("OLLAMA_ERROR:", str(e))
                        assistant_message = (
                            "I couldn't generate a response in time. Try again in a moment."
                        )
                elif decision.get("action") == "ESCALATE":
                    print("LAYER_REACHED: ESCALATE_OTHER")
                    assistant_message = (
                        "I can answer that better if you tell me which prior doc or session you mean."
                    )
                else:
                    print("LAYER_REACHED: L0/L1")
                    assistant_message = decision.get("assistant_text", "").strip()
                    if len(assistant_message) == 0:
                        assistant_message = self._fallback_user_text(decision.get("action"))

            t3 = time.perf_counter()
            yield json.dumps({"type": "delta", "text": assistant_message}) + "\n"

            # Persist assistant reply
            self.add_chat_message(user_id, chat_id, role="assistant", text=assistant_message, skip_vector=fast_mode)
            t4 = time.perf_counter()

            # Update rolling session memory
            summary_saved = ""
            if fast_mode is False and len(user_message.strip()) >= 40:
                summary_saved = self._update_session_memory(user_id, chat_id)
                self._enqueue_claim_extraction(user_id, chat_id)
            t5 = time.perf_counter()

            result = {
                "assistant_message": assistant_message,
                "summary_saved": summary_saved,
                "context_used": {
                    "profile_count": 0,
                    "recent_chat_count": 0,
                    "retrieved_summary_count": 0,
                    "retrieved_upload_count": 0,
                    "doc_mode": False
                }
            }

            yield json.dumps({"type": "done", **result}) + "\n"

            timings = {
                "add_user_msg_ms": int((t1 - t0) * 1000),
                "build_prompt_ms": int((t2 - t1) * 1000),
                "ollama_generate_ms": int((t3 - t2) * 1000),
                "add_assistant_msg_ms": int((t4 - t3) * 1000),
                "summary_ms": int((t5 - t4) * 1000),
                "total_ms": int((t5 - t0) * 1000),
            }
            print("ASK_STREAM_TIMINGS:", timings)

            prompt_eval_count = stats.get("prompt_eval_count")
            eval_count = stats.get("eval_count")
            input_tokens = self.estimate_input_tokens(user_message)
            context_tokens = 0
            if isinstance(prompt_eval_count, int):
                context_tokens = max(prompt_eval_count - input_tokens, 0)
            output_tokens = eval_count if isinstance(eval_count, int) else 0
            self.log_ask_metrics(
                input_tokens=input_tokens,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                elapsed_ms=timings.get("ollama_generate_ms", 0),
                total_ms=timings.get("total_ms", 0)
            )

        return gen()

    # Memory listing for viewer page
    # ----------------------------

    def list_profile_memories(self, user_id, limit=50):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for profile memory")
        rows = self.chat_store.get_profile_items(user_id)
        items = []
        i = 0
        while i < len(rows) and len(items) < limit:
            extra_raw = rows[i].get("extra")
            category = ""
            scope = ""
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                    category = extra.get("category", "")
                    scope = extra.get("scope", "")
                except json.JSONDecodeError:
                    category = ""
                    scope = ""
            items.append(
                {
                    "user_id": user_id,
                    "memory_type": "profile_fact",
                    "text": rows[i].get("content", ""),
                    "created_at": self.chat_store._dt_to_epoch(rows[i].get("created_at")),
                    "category": category,
                    "scope": scope,
                    "updated_at": rows[i].get("created_at").isoformat(timespec="seconds") if rows[i].get("created_at") else None,
                    "extra": rows[i].get("extra")
                }
            )
            i += 1
        return items

    def list_chat_summaries(self, user_id, chat_id=None, limit=50):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for chat summaries")
        cid = chat_id
        if cid is None or len(str(cid).strip()) == 0:
            return []
        rows = self.chat_store.get_chat_summaries(user_id, cid, limit)
        items = []
        i = 0
        while i < len(rows):
            items.append(
                {
                    "user_id": user_id,
                    "chat_id": cid,
                    "memory_type": "chat_summary",
                    "text": rows[i].get("content", ""),
                    "created_at": self.chat_store._dt_to_epoch(rows[i].get("created_at"))
                }
            )
            i += 1
        return items

    def list_uploaded_context(self, user_id, limit=100):
        if self.mysql_enabled is False or self.chat_store is None:
            raise RuntimeError("MySQL required for doc chunks")
        rows = self.chat_store.get_doc_chunks(user_id, limit)
        items = []
        i = 0
        while i < len(rows):
            items.append(
                {
                    "user_id": user_id,
                    "memory_type": "doc_chunk",
                    "text": rows[i].get("content", ""),
                    "created_at": self.chat_store._dt_to_epoch(rows[i].get("created_at")),
                    "extra": rows[i].get("extra")
                }
            )
            i += 1
        return items
