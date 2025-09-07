from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io

# Initialize FastAPI app
app = FastAPI(title="NSFW Image Detection API", description="Detect NSFW (Not Safe For Work) content in images.")

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Falconsai/nsfw_image_detection"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

class Prediction(BaseModel):
    label: str
    score: float

def load_image(image_file: UploadFile) -> Image.Image:
    try:
        contents = image_file.file.read()
        image = Image.open(io.BytesIO(contents))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {e}")

@app.post("/predict/", response_model=Prediction)
async def predict(image: UploadFile = File(...)):
    # Load and preprocess the image
    image = load_image(image)
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the results
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_id]
    score = torch.softmax(logits, dim=-1)[0, predicted_class_id].item()
    
    return Prediction(label=label, score=score)

# Root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "NSFW Image Detection API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)