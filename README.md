# 🚀 Hugging Face Learning Repository

A comprehensive collection of machine learning projects and learning resources built with Hugging Face Transformers, featuring both educational examples and production-ready APIs.

## 📦 Repository Contents

### 🖼️ Production API: NSFW Image Detection
**Location**: `nsfw-api/`

A production-grade FastAPI service for NSFW (Not Safe For Work) image classification using a fine-tuned Vision Transformer (ViT) model.

#### Features
- **Modular Architecture**: Clean package structure with separated concerns
- **Authentication**: Bearer token authentication for secure access
- **Monitoring**: Comprehensive Prometheus metrics for observability
- **Performance**: ONNX-optimized model for fast inference
- **Rate Limiting**: Built-in request throttling
- **Production Ready**: Docker support, error handling, logging

#### API Endpoints
- `POST /predict` - Upload image files for NSFW detection
- `POST /predict/url` - Analyze images from URLs
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint
- `GET /docs` - Interactive API documentation

#### Quick Start
```bash
# Set up environment
export API_KEY="your-secure-api-key"
cd nsfw-api/

# Run the API
python main.py

# Test the API
curl -X POST "http://localhost:8000/predict" \
    -H "Authorization: Bearer your-secure-api-key" \
    -F "file=@image.jpg"
```

#### Architecture
```
nsfw-api/
├── api/                    # Main package
│   ├── core/               # Core functionality
│   │   ├── auth.py         # Authentication & rate limiting
│   │   ├── config.py       # Configuration settings
│   │   ├── metrics.py      # Prometheus metrics
│   │   └── utils.py        # Utility functions
│   ├── models/             # Data models
│   │   └── models.py       # Pydantic response models
│   └── services/           # Business logic
│       └── inference.py    # ML model inference
├── main.py                 # FastAPI application
├── models/                 # ONNX model files
└── scripts/                # Utility scripts
```

### 📚 Learning Resources

#### 📓 Interactive Notebook
**File**: `notebook.ipynb`

Comprehensive Jupyter notebook covering:
- **Pipeline Examples**: Sentiment analysis, text generation, Q&A, NER
- **NSFW Classification**: Image classification with ViT models
- **Model Optimization**: ONNX conversion and performance comparison
- **Mac M1/M2 Optimization**: MPS (Metal Performance Shaders) usage

#### 🧪 Test Scripts
- **`test_installation.py`** - Verify Hugging Face ecosystem installation
- **`test_mps.py`** - Test Apple Silicon GPU acceleration

### 🛠️ Technical Stack

#### Core Dependencies
- **FastAPI** - Modern, fast web framework for APIs
- **Transformers** - Hugging Face's transformer models library
- **ONNX Runtime** - Optimized inference engine
- **Prometheus Client** - Metrics and monitoring
- **PyTorch** - Deep learning framework with MPS support
- **Pillow** - Image processing
- **Uvicorn** - ASGI server

#### Development Tools
- **UV** - Fast Python package manager
- **Jupyter** - Interactive development environment
- **Docker** - Containerization support

## ⚡ Setup Instructions

### Prerequisites
- Python 3.11+
- UV package manager (recommended) or pip

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd huggingface-learning

# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Environment Configuration
```bash
# For NSFW API
export API_KEY="your-secure-api-key"
export MODEL_PATH="nsfw-api/models/nsfw_model.onnx"
```

## 📈 Performance Metrics

The NSFW API includes comprehensive Prometheus metrics:

### HTTP Metrics
- `http_requests_total` - Request counts by endpoint/method/status
- `http_request_duration_seconds` - Request latency histograms
- `http_requests_active` - Active request gauge

### ML Inference Metrics
- `nsfw_predictions_total` - Prediction counts by classification
- `nsfw_inference_duration_seconds` - Inference time distribution
- `nsfw_prediction_confidence` - Confidence score histograms

### System Metrics
- `nsfw_model_load_time_seconds` - Model initialization time
- `nsfw_model_loaded` - Model status indicator
- `rate_limit_hits_total` - Rate limiting statistics

## 🚀 Deployment

### Docker Deployment
```bash
cd nsfw-api/
docker build -t nsfw-api .
docker run -p 8000:8000 -e API_KEY=your-key nsfw-api
```

### Production Considerations
- Use environment variables for configuration
- Set up reverse proxy (nginx) for SSL termination
- Configure monitoring with Prometheus/Grafana
- Implement log aggregation
- Set appropriate rate limits for your use case

## 💡 Use Cases

### Educational
- Learn Hugging Face Transformers library
- Understand modern ML API development
- Practice with different model types (NLP, Computer Vision)
- Explore model optimization techniques

### Production
- Content moderation systems
- Automated image filtering
- Safety compliance tools
- Custom ML inference APIs

## 🏆 Model Performance

The NSFW detection model provides:
- **Accuracy**: High precision for content classification
- **Speed**: ~300-400ms inference time on CPU
- **Efficiency**: ONNX optimization reduces model size by ~50%
- **Scalability**: Async processing with configurable workers

## 🤝 Contributing

This repository serves as both a learning resource and a production example. Feel free to:
- Explore the code structure
- Run the examples
- Modify for your own use cases
- Suggest improvements

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with relevant content policies when using NSFW detection models in production.

## 🆘 Support

For questions about implementation or usage:
1. Check the interactive documentation at `/docs` when running the API
2. Review the Jupyter notebook for detailed examples
3. Examine the test scripts for setup verification

---

*Built with ❤️ using Hugging Face Transformers, FastAPI, and modern Python best practices.*