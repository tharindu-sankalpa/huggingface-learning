=== PROMPT ===
I need to create a FastAPI server to serve the HuggingFace Vision Transformer model, Falconsai/nsfw_image_detection. The API should follow best practices and be optimized for high inference performance. Please provide the complete implementation.

=== FULL OUTPUT ===
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant that provides complete, working code implementations.
user
I need to create a FastAPI server to serve the HuggingFace Vision Transformer model, Falconsai/nsfw_image_detection. The API should follow best practices and be optimized for high inference performance. Please provide the complete implementation.
assistant
To create a FastAPI server for serving the HuggingFace Vision Transformer model, Falconsai/nsfw_image_detection, you can follow these steps:

1. Install the necessary dependencies:
```bash
pip install fastapi uvicorn transformers torch torchvision
```

2. Create a Python script for your FastAPI server:
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import requests
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
```

3. Run the FastAPI server:
```bash
python your_script_name.py
```

This code sets up a FastAPI server that accepts POST requests with an image URL. It downloads the image, preprocesses it using the Vision Transformer feature extractor, performs inference with the model, and returns the predicted class (NSFW or Safe) along with the confidence level.

The model used is "Falconsai/nsfw_image_detection", which is assumed to be a pre-trained Vision Transformer model for detecting NSFW content. You may need to adjust the model name and preprocessing steps based on the actual model's requirements.

Remember to replace `your_script_name.py` with the actual name of your Python script.

=== STATS ===
Model: Qwen/Qwen2.5-Coder-7B-Instruct
Flash Attention: True
Generation time: 10.35s
Input tokens: 81
Output tokens: 550
Tokens per second: 53.14
Model memory: 15.23 GB
GPU total: 99.87 GB
GPU allocated: 15.23 GB
GPU reserved: 15.25 GB
GPU free: 84.63 GB
