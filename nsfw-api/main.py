import logging
import time
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
import os
import requests

from api.core.config import Config
from api.models.models import PredictionResponse, ErrorResponse, HealthResponse
from api.core.auth import APIKeyAuth, RateLimiter
from api.services.inference import ModelManager
from api.core.utils import generate_request_id, validate_image, get_client_ip
from api.core import metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
model_manager = ModelManager()
api_key_auth = APIKeyAuth(Config.API_KEY)
rate_limiter = RateLimiter(Config.RATE_LIMIT)

# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(api_key_auth)]
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=Config.ALLOWED_HOSTS
)

# Store startup time for uptime calculation
startup_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting NSFW Detection API...")
    
    try:
        # Ensure model directory exists
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
        
        # Load model
        model_manager.load_model()
        
        # Set model metrics
        metrics.set_model_metrics(model_manager.load_time, model_manager.model_loaded)
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        metrics.set_model_metrics(None, False)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down NSFW Detection API...")
    model_manager.executor.shutdown(wait=True)


@app.get("/", response_model=dict)
@metrics.track_request_metrics("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NSFW Image Detection API",
        "version": Config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
@metrics.track_request_metrics("/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "unhealthy",
        model_loaded=model_manager.model_loaded,
        version=Config.API_VERSION,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
@metrics.track_request_metrics("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Predict NSFW content in uploaded image.
    
    - **file**: Image file (JPEG, PNG, WebP, BMP)
    - **Authorization**: Bearer token with API key (REQUIRED)
    
    Returns prediction with confidence scores.
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Rate limiting
        client_ip = get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            metrics.track_rate_limit_hit(client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate input
        validate_image(file)
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Log request
        logger.info(f"Processing request {request_id} from {client_ip}")
        
        # Make prediction
        prediction = await model_manager.predict_async(image)
        
        # Track inference metrics
        metrics.track_inference_metrics(prediction)
        
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            success=True,
            request_id=request_id,
            prediction=prediction,
            processing_time_ms=total_time
        )
        
        # Log successful prediction
        logger.info(
            f"Request {request_id} completed: "
            f"{prediction['label']} ({prediction['confidence']:.3f}) "
            f"in {total_time:.1f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"Request {request_id} failed: {error_msg}")
        
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=error_msg,
                error_code="INTERNAL_ERROR",
                request_id=request_id
            ).dict()
        )


@app.post("/predict/url")
@metrics.track_request_metrics("/predict/url")
async def predict_image_url(
    request: Request,
    image_url: str
):
    """
    Predict NSFW content from image URL.
    
    - **image_url**: Direct URL to image file
    - **Authorization**: Bearer token with API key (REQUIRED)
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Rate limiting
        client_ip = get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            metrics.track_rate_limit_hit(client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Download and process image
        response = requests.get(image_url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not any(fmt in content_type for fmt in Config.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=415,
                detail="Unsupported image format from URL"
            )
        
        image = Image.open(io.BytesIO(response.content))
        
        # Make prediction
        prediction = await model_manager.predict_async(image)
        
        # Track inference metrics
        metrics.track_inference_metrics(prediction)
        
        total_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            request_id=request_id,
            prediction=prediction,
            processing_time_ms=total_time
        )
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"URL prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    prometheus_metrics = metrics.get_prometheus_metrics()
    return Response(content=prometheus_metrics, media_type=metrics.CONTENT_TYPE_LATEST)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="info",
        access_log=True
    )