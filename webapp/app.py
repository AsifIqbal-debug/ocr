"""
NID OCR Web Application - FastAPI Backend
==========================================
Supports 3 OCR models:
1. NID OCR (nid_ocr.py) - Structured NID-specific extraction
2. Surya OCR (ocr_surya.py) - Modern ML-based OCR
3. EasyOCR (ocr.py) - Fast CPU-based OCR
"""

import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="Bangladesh NID OCR API",
    description="OCR API for Bangladesh National ID Cards with multiple model support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run_nid_ocr(image_path: str) -> dict:
    """Run NID OCR pipeline."""
    from nid_ocr import NIDOCRPipeline
    
    pipeline = NIDOCRPipeline()
    result = pipeline.process_full_image_ocr(image_path)
    return {"model": "NID OCR", "type": "structured", "data": result}


def run_surya_ocr(image_path: str) -> dict:
    """Run Surya OCR."""
    from ocr_surya import run_surya_ocr as surya_ocr, apply_corrections
    
    items = surya_ocr(image_path, ["bn", "en"])
    items = apply_corrections(items)
    return {"model": "Surya OCR", "type": "lines", "data": {"items": items}}


def run_easyocr(image_path: str) -> dict:
    """Run EasyOCR."""
    import cv2
    import easyocr
    import re
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Preprocess
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    # Run OCR
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    results = reader.readtext(processed, detail=1)
    
    # Format results
    items = []
    for bbox, text, conf in results:
        # Classify text type
        bangla_count = len(re.findall(r'[\u0980-\u09FF]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        digit_count = len(re.findall(r'[0-9]', text))
        total = len(text.replace(" ", ""))
        
        if total == 0:
            text_type = "unknown"
        elif digit_count > total * 0.5:
            text_type = "number"
        elif bangla_count > english_count:
            text_type = "bangla"
        elif english_count > bangla_count:
            text_type = "english"
        else:
            text_type = "mixed"
        
        items.append({
            "text": text,
            "type": text_type,
            "confidence": round(conf, 3)
        })
    
    return {"model": "EasyOCR", "type": "lines", "data": {"items": items}}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return """
    <html>
        <head><title>NID OCR</title></head>
        <body>
            <h1>NID OCR API</h1>
            <p>Please create static/index.html</p>
            <p>API docs: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


@app.post("/api/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    model: str = Form(default="nid_ocr")
):
    """
    Process an image with the selected OCR model.
    
    - **file**: Image file (JPEG, PNG)
    - **model**: OCR model to use (nid_ocr, surya, easyocr)
    
    Returns OCR results in JSON format.
    """
    # Validate model
    valid_models = ["nid_ocr", "surya", "easyocr"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {valid_models}"
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: JPEG, PNG"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    file_ext = Path(file.filename).suffix or ".jpg"
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Run OCR based on selected model
        start_time = datetime.now()
        
        if model == "nid_ocr":
            result = run_nid_ocr(str(file_path))
        elif model == "surya":
            result = run_surya_ocr(str(file_path))
        elif model == "easyocr":
            result = run_easyocr(str(file_path))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Add metadata
        result["metadata"] = {
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@app.get("/api/models")
async def list_models():
    """List available OCR models."""
    return {
        "models": [
            {
                "id": "nid_ocr",
                "name": "NID OCR Pipeline",
                "description": "Structured extraction for Bangladesh NID cards. Returns fields: name, father, mother, DOB, NID number.",
                "output_type": "structured"
            },
            {
                "id": "surya",
                "name": "Surya OCR",
                "description": "State-of-the-art multilingual OCR with high accuracy for Bangla text.",
                "output_type": "lines"
            },
            {
                "id": "easyocr",
                "name": "EasyOCR",
                "description": "Fast CPU-based OCR supporting Bangla and English.",
                "output_type": "lines"
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404 errors."""
    return Response(content=b"", media_type="image/x-icon")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
