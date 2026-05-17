"""
Docling OCR microservice — runs on EC2 GPU alongside Ollama.

Start with:
    source ~/docling-env/bin/activate
    nohup python ocr_service.py &

Listens on 0.0.0.0:8001.
"""

import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Docling OCR Service")

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix.lower() or ".pdf"
    if suffix not in (".pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data = await file.read()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
        os.close(fd)

        converter = _get_converter()
        result = converter.convert(tmp_path)
        markdown = result.document.export_to_markdown()

        return JSONResponse({
            "markdown": markdown,
            "chars": len(markdown),
            "filename": file.filename,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
