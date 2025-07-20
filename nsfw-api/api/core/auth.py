import time
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Create a global HTTPBearer instance
security = HTTPBearer(auto_error=True)


class APIKeyAuth:
    """Simple API key authentication."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def __call__(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        if not credentials or credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "success": False,
                    "error": "Invalid or missing API key",
                    "error_code": "UNAUTHORIZED"
                }
            )
        return credentials.credentials


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        
        # Clean old entries
        cutoff = now - self.window_seconds
        self.requests = {
            ip: timestamps for ip, timestamps in self.requests.items()
            if any(t > cutoff for t in timestamps)
        }
        
        # Filter timestamps for this IP
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if t > cutoff
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True