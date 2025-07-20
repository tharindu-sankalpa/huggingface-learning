import uuid
from fastapi import HTTPException, UploadFile, Request
from api.core.config import Config


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    # Check file size
    if file.size and file.size > Config.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size too large. Maximum allowed: {Config.MAX_IMAGE_SIZE} bytes"
        )
    
    # Check content type
    if file.content_type not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format. Supported: {', '.join(Config.SUPPORTED_FORMATS)}"
        )


def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host