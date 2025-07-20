import os


class Config:
    """Production configuration settings."""
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/nsfw_model.onnx")
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    SUPPORTED_FORMATS = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
    
    # API settings
    API_TITLE = "NSFW Image Detection API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Production-grade NSFW content detection API"
    
    # Performance settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Security settings
    API_KEY = os.getenv("API_KEY", "your-secure-api-key-here")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # Rate limiting (requests per minute)
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", "100"))