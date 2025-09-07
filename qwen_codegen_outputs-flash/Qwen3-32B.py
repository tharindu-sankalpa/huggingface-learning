from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header, status, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from typing import List
import uvicorn
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import io
from fastapi.responses import JSONResponse

# --- Configuration ---
MODEL_NAME = "Falconsai/nsfw_image_detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEYS = ["your-secret-api-key"]  # Replace with your secret key(s) or load from env

# --- Model Initialization ---
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# --- FastAPI App Setup ---
app = FastAPI(title="NSFW Image Detection API")

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# --- Authentication Middleware ---
def get_api_key(api_key: str = Header(..., description="Your API key for authentication")):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key

# --- Image Processing Function ---
def load_image(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# --- Prediction Endpoint ---
@app.post("/predict/", responses={429: {"description": "Too Many Requests"}})
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def predict(
    request: Request,
    files: List[UploadFile] = File(..., description="List of image files to analyze"),
    api_key: str = Depends(get_api_key)
):
    images = []
    filenames = []

    for file in files:
        try:
            img = load_image(file)
            images.append(img)
            filenames.append(file.filename)
        except HTTPException as e:
            raise e

    # Batch preprocessing
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predictions = []

    for i, filename in enumerate(filenames):
        scores = probs[i].tolist()
        label_idx = int(outputs.logits[i].argmax())
        label = model.config.id2label[label_idx]

        predictions.append({
            "filename": filename,
            "label": label,
            "scores": {model.config.id2label[j]: round(scores[j], 4) for j in range(len(scores))}
        })

    return {"predictions": predictions}

# --- Error Handler for Rate Limits ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "message": "Too many requests per minute"}
    )

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)