from pydantic import BaseModel
from typing import Dict, Any


class PredictionResponse(BaseModel):
    """Standard API response model."""
    success: bool
    request_id: str
    prediction: Dict[str, Any]
    processing_time_ms: float
    model_version: str = "1.0"


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    error_code: str
    request_id: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float