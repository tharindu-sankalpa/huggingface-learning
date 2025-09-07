import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from transformers import AutoModelForImageClassification, AutoImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model_name = "Falconsai/nsfw_image_detection"
try:
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise

# Move model to device
model.to(device)
model.eval()  # Set to evaluation mode

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class_idx = probabilities.argmax().item()
        
        # Get the label
        label_map = model.config.id2label
        predicted_label = label_map.get(predicted_class_idx, "Unknown")
        
        return {
            "predicted_class": predicted_class_idx,
            "predicted_label": predicted_label,
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)