# Save as scripts/convert_model.py
import torch
import os
from transformers import AutoModelForImageClassification

def convert_model():
    """Convert Hugging Face model to ONNX format."""
    print("🔄 Converting model to ONNX...")
    
    model_name = "Falconsai/nsfw_image_detection"
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)
    
    # Export to ONNX
    onnx_path = "../models/nsfw_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Check file size
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ Model converted successfully!")
    print(f"📁 File: {onnx_path}")
    print(f"📊 Size: {file_size:.1f} MB")

if __name__ == "__main__":
    convert_model()