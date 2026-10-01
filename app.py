"""FastAPI web interface for the AI Resume Screener."""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.config import SUPPORTED_EXTENSIONS, ensure_dirs
from src.pipeline import ingest_folder

app = FastAPI(title="AI Resume Screener")
templates = Jinja2Templates(directory="templates")

ensure_dirs()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/screen")
async def screen(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            dest = tmp_path / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)

        result = ingest_folder(tmp_path)

    return result.model_dump()
