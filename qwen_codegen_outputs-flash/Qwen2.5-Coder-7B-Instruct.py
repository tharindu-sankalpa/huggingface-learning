from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import requests
import torch
from io import BytesIO
from transformers import ViTForImageClassification, ViTFeatureExtractor

app = FastAPI()

# Load the pre-trained model and feature extractor
model_name = "Falconsai/nsfw_image_detection"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

@app.post("/predict/")
async def predict(image_url: str):
    try:
        # Download and preprocess the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # Get the predicted class and confidence
        _, predicted_class_id = torch.max(probabilities, 1)
        confidence = probabilities[0][predicted_class_id].item()

        # Map class ID to label (assuming binary classification for NSFW vs Safe)
        if predicted_class_id == 0:
            label = "Safe"
        else:
            label = "NSFW"

        return JSONResponse(content={"label": label, "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)