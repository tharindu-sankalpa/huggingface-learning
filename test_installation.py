# test_installation.py
import torch
import transformers
import datasets
from transformers import pipeline

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")

# Test a simple pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love learning about AI!")
print(f"Test result: {result}")