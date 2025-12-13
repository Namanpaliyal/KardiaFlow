from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import shutil
import traceback
import os
import time

try:
    from .rag import get_answer, build_vector_store_from_upload, build_vector_store
except ImportError:
    from app.rag import get_answer, build_vector_store_from_upload, build_vector_store

# Persistence dir for Chroma DB (can be overridden by env)
PERSIST_DIR = os.getenv("PERSIST_DIR", str(Path(__file__).parent / ".." / "chroma_db"))

app = FastAPI()

class Query(BaseModel):
    question: str
    top_k: Optional[int] = 3

# Mount static files FIRST
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.post("/api/ask")
async def ask_question(query: Query):
    """Handle question requests"""
    try:
        print(f"[LOG] Received question: {query.question}")
        start = time.time()

        response = get_answer(query.question, persist_dir=PERSIST_DIR, top_k=(query.top_k or 3))
        duration = time.time() - start
        print(f"[LOG] Got response successfully in {duration:.2f}s")

        answer = response.get("answer", "No answer generated")
        sources = [
            {"source": doc.metadata.get("source", "unknown")} 
            for doc in response.get("source_documents", [])
        ]

        # no telemetry

        return {
            "answer": answer,
            "sources": sources
        }
    except FileNotFoundError as e:
        print(f"[ERROR] FileNotFoundError: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "No documents uploaded yet",
                "answer": "Please upload a PDF document first before asking questions."
            }
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Exception: {e}")
        print(error_trace)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "answer": f"Error processing question: {str(e)}"
            }
        )

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF uploads"""
    try:
        print(f"[LOG] Uploading file: {file.filename}")
        
        upload_dir = Path(__file__).parent / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[LOG] File saved to: {file_path}")
        print(f"[LOG] Processing PDF...")
        
        build_vector_store_from_upload(str(file_path), persist_dir=PERSIST_DIR)
        
        print(f"[LOG] PDF processed successfully")
        # no telemetry

        return {
            "success": True,
            "message": f"PDF '{file.filename}' uploaded and processed successfully!"
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Upload error: {e}")
        print(error_trace)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/")
def root():
    """Serve the main HTML file"""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(str(static_path))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)