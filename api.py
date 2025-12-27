# python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload --log-level debug
import os
import uuid
import shutil

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File, Form

from core.twin_service import TwinService


def build_service():
    qdrant_url = "http://127.0.0.1:6333"
    ollama_model_name = "twinModel"
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
    return service


app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Single-user MVP state (good enough for today)
SERVICE = build_service()
USER_ID = "user_1"
ACTIVE_CHAT_ID = None

@app.post("/api/upload")
async def upload_context(file: UploadFile = File(...), caption: str = Form("")):
    # Save file to disk
    file_id = str(uuid.uuid4())
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}__{safe_name}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_name = safe_name.lower()

    # PDF path: extract + chunk + embed/store
    if file_name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except Exception:
            return JSONResponse(status_code=500, content={"error": "PyMuPDF not installed. Run: pip install pymupdf"})

        doc = fitz.open(save_path)
        full_text = ""
        page_index = 0
        while page_index < len(doc):
            full_text += doc[page_index].get_text("text") + "\n"
            page_index += 1

        doc.close()

        full_text = full_text.strip()
        if len(full_text) == 0:
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "file_id": file_id,
                    "file_name": safe_name,
                    "indexed_chunks": 0,
                    "note": "No extractable text found (likely a scanned PDF)."
                }
            )

        # simple chunking (MVP)
        chunk_size = 800
        overlap = 120

        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end].strip()
            if len(chunk) > 0:
                chunks.append(chunk)
            start = start + (chunk_size - overlap)

        i = 0
        while i < len(chunks):
            SERVICE.add_uploaded_context_chunk(
                user_id=USER_ID,
                file_id=file_id,
                file_name=safe_name,
                context_type="pdf_chunk",
                chunk_index=i,
                text=chunks[i]
            )
            i += 1

        return JSONResponse(
            content={
                "ok": True,
                "file_id": file_id,
                "file_name": safe_name,
                "indexed_chunks": len(chunks)
            }
        )

    # Image (or other file) path: store caption as retrievable context
    # MVP: no OCR today; caption is what gets embedded & retrieved
    cap = (caption or "").strip()
    if len(cap) == 0:
        cap = f"User uploaded a file named '{safe_name}'. (No caption provided.)"

    SERVICE.add_uploaded_context_chunk(
        user_id=USER_ID,
        file_id=file_id,
        file_name=safe_name,
        context_type="image_caption",
        chunk_index=0,
        text=cap
    )

    return JSONResponse(
        content={
            "ok": True,
            "file_id": file_id,
            "file_name": safe_name,
            "indexed_chunks": 1,
            "note": "Stored caption as retrievable context (no OCR)."
        }
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    global ACTIVE_CHAT_ID

    if ACTIVE_CHAT_ID is None:
        ACTIVE_CHAT_ID = SERVICE.start_chat(USER_ID)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_id": USER_ID,
            "chat_id": ACTIVE_CHAT_ID
        }
    )


@app.post("/api/new_chat")
def new_chat():
    global ACTIVE_CHAT_ID

    ACTIVE_CHAT_ID = SERVICE.start_chat(USER_ID)

    data = {
        "user_id": USER_ID,
        "chat_id": ACTIVE_CHAT_ID
    }
    return JSONResponse(content=data)


@app.post("/api/profile")
async def add_profile(request: Request):
    body = await request.json()

    category = body.get("category", "")
    text = body.get("text", "")

    if category is None or len(category.strip()) == 0:
        return JSONResponse(status_code=400, content={"error": "category is required"})

    if text is None or len(text.strip()) == 0:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    point_id = SERVICE.add_profile_memory(USER_ID, category.strip(), text.strip())

    data = {
        "ok": True,
        "point_id": point_id
    }
    return JSONResponse(content=data)


@app.post("/api/ask")
async def ask(request: Request):
    global ACTIVE_CHAT_ID

    if ACTIVE_CHAT_ID is None:
        ACTIVE_CHAT_ID = SERVICE.start_chat(USER_ID)

    body = await request.json()
    message = body.get("message", "")

    if message is None or len(message.strip()) == 0:
        return JSONResponse(status_code=400, content={"error": "message is required"})

    result = SERVICE.ask(USER_ID, ACTIVE_CHAT_ID, message.strip())

    data = {
        "chat_id": ACTIVE_CHAT_ID,
        "assistant_message": result.get("assistant_message", ""),
        "context_used": result.get("context_used", {}),
        "summary_saved": result.get("summary_saved", "")
    }
    return JSONResponse(content=data)

@app.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request):
    return templates.TemplateResponse(
        "memory.html",
        {"request": request, "user_id": USER_ID}
    )

@app.get("/api/memory/profile")
def api_memory_profile():
    return {"ok": True, "items": SERVICE.list_profile_memories(USER_ID, limit=50)}

@app.get("/api/memory/summaries")
def api_memory_summaries():
    return {"ok": True, "items": SERVICE.list_chat_summaries(USER_ID, limit=50)}

@app.get("/api/memory/uploads")
def api_memory_uploads():
    return {"ok": True, "items": SERVICE.list_uploaded_context(USER_ID, limit=100)}
