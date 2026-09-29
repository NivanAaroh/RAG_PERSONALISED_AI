from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from backend.ingestion import ingest_pdf
from backend.chunking import chunk_document
from backend.embeddings import embed_chunks
from backend.vector_store import upsert_chunks
from backend.llm import generate_response


app = FastAPI(title="RAG Personalised AI")


UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty or whitespace-only.")
        return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = str(uuid4())
    document_name = Path(file.filename).name
    destination = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        contents = await file.read()
        destination.write_bytes(contents)

        document = ingest_pdf(
            destination,
            document_id=document_id,
            document_name=document_name,
        )

        pages_with_text = sum(
            1
            for page in document["pages"]
            if page["text"]
        )

        chunks = chunk_document(
            document["document_id"],
            document["document_name"],
            document["pages"],
        )

        embeddings = embed_chunks(chunks)

        stored_count = upsert_chunks(
            chunks,
            embeddings,
        )

        return {
            "document_id": document["document_id"],
            "document_name": document["document_name"],
            "page_count": document["page_count"],
            "pages_with_text": pages_with_text,
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "stored_count": stored_count,
        }

    except Exception as exc:
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=400,
            detail=f"PDF ingestion failed: {exc}",
        ) from exc

    finally:
        await file.close()


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response_stream = generate_response(request.message)

        return StreamingResponse(
            response_stream,
            media_type="text/plain; charset=utf-8",
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM service request failed.",
        ) from exc
