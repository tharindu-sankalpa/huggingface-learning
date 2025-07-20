import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, Any

from api.core.config import Config

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe model manager for ONNX inference."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.session = None
            self.input_name = None
            self.output_name = None
            self.model_loaded = False
            self.load_time = None
            self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
            self._initialized = True
    
    def load_model(self):
        """Load ONNX model with optimization."""
        try:
            start_time = time.time()
            
            # Check if model file exists
            if not os.path.exists(Config.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")
            
            # Configure ONNX Runtime for production
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = Config.MAX_WORKERS
            sess_options.intra_op_num_threads = Config.MAX_WORKERS
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Choose providers based on available hardware
            providers = ['CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            
            # Add specialized providers if available
            if 'CoreMLExecutionProvider' in available_providers:
                providers.insert(0, 'CoreMLExecutionProvider')
            elif 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Load the model
            self.session = ort.InferenceSession(
                Config.MODEL_PATH,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            self.load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            logger.info(f"Using providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert to RGB and resize
        image = image.convert('RGB').resize((224, 224))
        
        # Convert to numpy array and normalize - ensure float32 throughout
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet normalization - use float32 arrays
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Change format from HWC to CHW and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Ensure final array is float32
        return img_array.astype(np.float32)
    
    async def predict_async(self, image: Image.Image) -> Dict[str, Any]:
        """Async prediction with thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, image)
    
    def _predict_sync(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous prediction logic."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            ort_inputs = {self.input_name: processed_image}
            ort_outputs = self.session.run([self.output_name], ort_inputs)
            
            # Process results
            logits = ort_outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            # Map to labels
            labels = {0: 'normal', 1: 'nsfw'}
            predicted_label = labels[predicted_class]
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'nsfw': float(probabilities[1])
                },
                'is_safe': predicted_label == 'normal',
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise