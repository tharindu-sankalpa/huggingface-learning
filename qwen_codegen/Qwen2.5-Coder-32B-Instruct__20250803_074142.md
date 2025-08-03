=== PROMPT ===
I need to create a FastAPI server to serve the HuggingFace Vision Transformer model, Falconsai/nsfw_image_detection. The API should follow best practices and be optimized for high inference performance. Please provide the complete implementation.

=== FULL OUTPUT ===
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant that provides complete, working code implementations.
user
I need to create a FastAPI server to serve the HuggingFace Vision Transformer model, Falconsai/nsfw_image_detection. The API should follow best practices and be optimized for high inference performance. Please provide the complete implementation.
assistant
Certainly! Below is a complete implementation of a FastAPI server that serves the Hugging Face Vision Transformer model `Falconsai/nsfw_image_detection`. This implementation includes best practices such as using dependency injection for model loading, handling image uploads, and optimizing inference performance.

First, ensure you have the necessary packages installed:

```bash
pip install fastapi uvicorn transformers torch pillow
```

Now, here's the complete implementation:

```python
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
```

### Explanation:

1. **Model Loading**: The model and image processor are loaded once when the application starts. This avoids reloading the model for each request, which significantly improves performance.

2. **Dependency Injection**: The `load_image` function handles the image loading and preprocessing, making it easier to test and maintain.

3. **Error Handling**: The `load_image` function includes error handling to catch issues related to image loading and conversion.

4. **Asynchronous Operations**: The `/predict/` endpoint is defined as an asynchronous function, which is beneficial for handling I/O-bound operations like file uploads and model inference.

5. **Response Model**: A Pydantic model (`Prediction`) is used to define the structure of the response, ensuring consistency and validation.

6. **Health Check Endpoint**: A root endpoint (`/`) is provided to check if the API is running.

### Running the Server:

To run the server, save the code to a file (e.g., `main.py`) and execute:

```bash
uvicorn main:app --reload
```

The `--reload` flag is useful during development as it restarts the server on code changes. For production, remove this flag.

This setup provides a robust, efficient, and scalable way to serve the NSFW image detection model using FastAPI.

=== STATS ===
Model: Qwen/Qwen2.5-Coder-32B-Instruct
Flash Attention: True
Generation time: 31.39s
Input tokens: 81
Output tokens: 800
Tokens per second: 25.49
Model memory: 65.53 GB
GPU total: 99.87 GB
GPU allocated: 65.56 GB
GPU reserved: 79.23 GB
GPU free: 20.64 GB
