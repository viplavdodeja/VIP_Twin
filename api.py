# python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload --log-level debug
#$env:MYSQL_ENABLED="1"
#$env:MYSQL_HOST="127.0.0.1"
#$env:MYSQL_PORT="3306"
#$env:MYSQL_USER="root"
#$env:MYSQL_PASSWORD="4205Mowry!@"
#$env:MYSQL_DATABASE="vip_twin2"

import os
import uuid
import shutil
import hashlib

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File, Form
from fastapi.staticfiles import StaticFiles

from core.twin_service import TwinService

def build_service():
    qdrant_url = "http://127.0.0.1:6333"
    ollama_model_name = "vip-twin2"
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Single-user MVP state (good enough for today)
SERVICE = build_service()
import core.twin_service as ts
print("RUNNING TwinService from:", ts.__file__)
#USER_ID = "user_1"
USER_ID = "11111111-1111-1111-1111-111111111111"
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
    file_size_bytes = os.path.getsize(save_path)
    sha256 = ""
    with open(save_path, "rb") as rf:
        hasher = hashlib.sha256()
        while True:
            chunk = rf.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        sha256 = hasher.hexdigest()
    mime_type = file.content_type or ""

    # PDF path: extract + chunk + embed/store
    if file_name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except Exception:
            return JSONResponse(status_code=500, content={"error": "PyMuPDF not installed. Run: pip install pymupdf"})

        doc = fitz.open(save_path)
        page_texts = []
        page_index = 0
        while page_index < len(doc):
            page_texts.append(doc[page_index].get_text("text"))
            page_index += 1
        doc.close()

        full_text = "\n".join(page_texts).strip()
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

        # simple chunking per page (MVP)
        chunk_size = 800
        overlap = 120

        chunks = []
        chunk_index = 0
        p = 0
        while p < len(page_texts):
            page_text = (page_texts[p] or "").strip()
            if len(page_text) > 0:
                start = 0
                while start < len(page_text):
                    end = start + chunk_size
                    chunk = page_text[start:end].strip()
                    if len(chunk) > 0:
                        chunks.append({
                            "text": chunk,
                            "chunk_index": chunk_index,
                            "metadata": {"page": p + 1, "file_name": safe_name}
                        })
                        chunk_index += 1
                    start = start + (chunk_size - overlap)
            p += 1

        SERVICE.index_document(
            doc_id=file_id,
            user_id=USER_ID,
            title=safe_name,
            doc_type="pdf",
            source=save_path,
            filename=safe_name,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            chunks=chunks
        )

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

    SERVICE.index_document(
        doc_id=file_id,
        user_id=USER_ID,
        title=safe_name,
        doc_type="file",
        source=save_path,
        filename=safe_name,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        sha256=sha256,
        chunks=[{"text": cap, "chunk_index": 0, "metadata": {"file_name": safe_name}}]
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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon.ico"))


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

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_stub():
    return Response(status_code=204)


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

@app.post("/api/ask_stream")
async def ask_stream(request: Request):
    global ACTIVE_CHAT_ID

    if ACTIVE_CHAT_ID is None:
        ACTIVE_CHAT_ID = SERVICE.start_chat(USER_ID)

    body = await request.json()
    message = body.get("message", "")

    if message is None or len(message.strip()) == 0:
        return JSONResponse(status_code=400, content={"error": "message is required"})

    def stream():
        for line in SERVICE.ask_stream(USER_ID, ACTIVE_CHAT_ID, message.strip()):
            yield line

    return StreamingResponse(stream(), media_type="application/x-ndjson")

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
def api_memory_summaries(chat_id: str = ""):
    cid = chat_id.strip()
    if len(cid) == 0:
        cid = None
    return {"ok": True, "items": SERVICE.list_chat_summaries(USER_ID, chat_id=cid, limit=50)}

@app.get("/api/memory/uploads")
def api_memory_uploads():
    return {"ok": True, "items": SERVICE.list_uploaded_context(USER_ID, limit=100)}


@app.get("/api/health")
def api_health():
    return {
        "mysql_connected": bool(SERVICE.mysql_connected),
        "qdrant_connected": bool(SERVICE.qdrant_connected),
        "mode": SERVICE.mode
    }
    file_size_bytes = os.path.getsize(save_path)
    sha256 = ""
    with open(save_path, "rb") as rf:
        hasher = hashlib.sha256()
        while True:
            chunk = rf.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        sha256 = hasher.hexdigest()

    mime_type = file.content_type or ""
