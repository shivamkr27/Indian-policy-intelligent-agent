"""
Document endpoints — upload, batch ingest, list, delete.

Mirrors the old Chainlit app's _handle_file_elements / on_ingest_all_action.

All uploaded files live on the single machine running this backend — there is
no per-user physical disk. Isolation between users is enforced here via:
  - per-user subfolder under DOCS_DIR (also prevents two users' same-named
    files from overwriting each other on disk)
  - a per-user total storage quota (MAX_USER_STORAGE_MB)
  - a per-user upload rate limit (protects the shared server from spam uploads)
"""

import asyncio
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from api import singletons
from api.deps import get_user_id
from core.config import DOCS_DIR, MAX_UPLOAD_SIZE_MB, MAX_USER_STORAGE_MB, SUPPORTED_UPLOAD_EXT
from core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["documents"])

MAX_FILES_PER_UPLOAD = 5


def _sanitize_filename(name: str) -> str:
    safe = Path(name).name
    safe = safe.replace("..", "").strip()
    return safe or "upload.bin"


def _is_valid_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def _user_docs_dir(user_id: str) -> Path:
    d = Path(DOCS_DIR) / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_storage_bytes(user_id: str) -> int:
    d = Path(DOCS_DIR) / user_id
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_user_id),
):
    if not singletons.upload_limiter.is_allowed(user_id):
        raise HTTPException(429, "Too many uploads. Please slow down.")

    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(400, f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.")

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    quota_bytes = MAX_USER_STORAGE_MB * 1024 * 1024
    user_dir = _user_docs_dir(user_id)
    used_bytes = _user_storage_bytes(user_id)

    results = []
    for upload in files:
        filename = _sanitize_filename(upload.filename or "upload.bin")

        if not filename.lower().endswith(SUPPORTED_UPLOAD_EXT):
            results.append({"file": filename, "status": "error", "detail": "Unsupported file type. Use PDF, DOCX, or TXT."})
            continue

        content = await upload.read()
        if len(content) > max_bytes:
            results.append({"file": filename, "status": "error", "detail": f"Exceeds {MAX_UPLOAD_SIZE_MB}MB limit."})
            continue

        if used_bytes + len(content) > quota_bytes:
            results.append({
                "file": filename,
                "status": "error",
                "detail": f"Storage quota exceeded ({MAX_USER_STORAGE_MB}MB total per user). Delete an existing document first.",
            })
            continue

        if filename.lower().endswith(".pdf") and not _is_valid_pdf(content):
            results.append({"file": filename, "status": "error", "detail": "Does not appear to be a valid PDF."})
            continue

        dest = user_dir / filename
        dest.write_bytes(content)
        used_bytes += len(content)

        stats = await asyncio.to_thread(singletons.ingestion.ingest_single, str(dest), user_id)
        await asyncio.to_thread(singletons.tool_factory.rebuild_bm25)

        if "error" in stats:
            results.append({"file": filename, "status": "error", "detail": stats["error"]})
        else:
            results.append({
                "file": filename,
                "status": "ingested",
                "parent_chunks": stats.get("parent_chunks", 0),
                "child_chunks": stats.get("child_chunks", 0),
            })

    return {"results": results}


@router.post("/documents/ingest-all")
async def ingest_all_documents(user_id: str = Depends(get_user_id)):
    user_dir = _user_docs_dir(user_id)
    stats = await asyncio.to_thread(singletons.ingestion.ingest_all, str(user_dir), user_id)
    await asyncio.to_thread(singletons.tool_factory.rebuild_bm25)
    return stats


@router.get("/documents")
async def list_documents(user_id: str = Depends(get_user_id)):
    files = singletons.ingestion.list_ingested_files(user_id=user_id)
    return {"files": files}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, user_id: str = Depends(get_user_id)):
    owned = set(singletons.ingestion.list_ingested_files(user_id=user_id))
    if filename not in owned:
        raise HTTPException(404, "Document not found.")

    await asyncio.to_thread(singletons.ingestion.delete_document, filename, user_id)
    await asyncio.to_thread(singletons.tool_factory.rebuild_bm25)

    dest = _user_docs_dir(user_id) / _sanitize_filename(filename)
    if dest.exists():
        dest.unlink()

    return {"deleted": True}
