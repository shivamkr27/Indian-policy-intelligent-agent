"""
Document endpoints — upload, batch ingest, list ingested files.

Mirrors the old Chainlit app's _handle_file_elements / on_ingest_all_action.
"""

import asyncio
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from api import singletons
from api.deps import get_user_id
from core.config import DOCS_DIR, MAX_UPLOAD_SIZE_MB, SUPPORTED_UPLOAD_EXT
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


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_user_id),
):
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(400, f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.")

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

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

        if filename.lower().endswith(".pdf") and not _is_valid_pdf(content):
            results.append({"file": filename, "status": "error", "detail": "Does not appear to be a valid PDF."})
            continue

        dest = Path(DOCS_DIR) / filename
        dest.write_bytes(content)

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
    stats = await asyncio.to_thread(singletons.ingestion.ingest_all, DOCS_DIR, user_id)
    await asyncio.to_thread(singletons.tool_factory.rebuild_bm25)
    return stats


@router.get("/documents")
async def list_documents(user_id: str = Depends(get_user_id)):
    files = singletons.ingestion.list_ingested_files(user_id=user_id)
    return {"files": files}
