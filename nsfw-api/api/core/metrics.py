from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from functools import wraps

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Inference metrics
INFERENCE_COUNT = Counter(
    'nsfw_predictions_total',
    'Total NSFW predictions made',
    ['label', 'is_safe']
)

INFERENCE_DURATION = Histogram(
    'nsfw_inference_duration_seconds',
    'NSFW inference duration in seconds'
)

INFERENCE_CONFIDENCE = Histogram(
    'nsfw_prediction_confidence',
    'NSFW prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# System metrics
MODEL_LOAD_TIME = Gauge(
    'nsfw_model_load_time_seconds',
    'Time taken to load the NSFW model'
)

MODEL_LOADED = Gauge(
    'nsfw_model_loaded',
    'Whether the NSFW model is loaded (1=loaded, 0=not loaded)'
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Number of rate limit hits',
    ['client_ip']
)


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = "POST" if "predict" in endpoint else "GET"
            start_time = time.time()
            
            ACTIVE_REQUESTS.inc()
            
            try:
                response = await func(*args, **kwargs)
                status_code = "200"
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                return response
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
                ACTIVE_REQUESTS.dec()
        
        return wrapper
    return decorator


def track_inference_metrics(prediction_result: dict):
    """Track inference-specific metrics."""
    label = prediction_result['label']
    is_safe = str(prediction_result['is_safe']).lower()
    confidence = prediction_result['confidence']
    processing_time_seconds = prediction_result['processing_time_ms'] / 1000.0
    
    INFERENCE_COUNT.labels(label=label, is_safe=is_safe).inc()
    INFERENCE_DURATION.observe(processing_time_seconds)
    INFERENCE_CONFIDENCE.observe(confidence)


def set_model_metrics(load_time: float, is_loaded: bool):
    """Set model-related metrics."""
    if load_time:
        MODEL_LOAD_TIME.set(load_time)
    MODEL_LOADED.set(1 if is_loaded else 0)


def track_rate_limit_hit(client_ip: str):
    """Track rate limit hits."""
    RATE_LIMIT_HITS.labels(client_ip=client_ip).inc()


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in the expected format."""
    return generate_latest()