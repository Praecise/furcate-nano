# ============================================================================
# furcate_nano/edge_ml.py
"""Edge machine learning for environmental data analysis."""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    import tensorflow.lite as tflite
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("TensorFlow Lite not available - using simulation models")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of edge ML models."""
    ENVIRONMENTAL_CLASSIFIER = "env_classifier"
    ANOMALY_DETECTOR = "anomaly_detector"
    AIR_QUALITY_PREDICTOR = "air_quality_predictor"
    ECOSYSTEM_HEALTH = "ecosystem_health"

class EdgeMLEngine:
    """Edge machine learning engine for environmental analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize edge ML engine.
        
        Args:
            config: ML configuration
        """
        self.config = config
        self.simulation_mode = not ML_AVAILABLE or config.get("simulation", False)
        self.model_path = Path(config.get("model_path", "./models"))
        
        # TensorFlow Lite interpreters
        self.interpreters: Dict[ModelType, Any] = {}
        
        # Model configurations
        self.models_config = config.get("models", {
            ModelType.ENVIRONMENTAL_CLASSIFIER: {
                "file": "environmental_classifier.tflite",
                "input_shape": [1, 10],
                "output_classes": ["normal", "polluted", "extreme"]
            },
            ModelType.ANOMALY_DETECTOR: {
                "file": "anomaly_detector.tflite", 
                "input_shape": [1, 15],
                "threshold": 0.8
            }
        })
        
        # Feature processing
        self.feature_processors = {}
        
        logger.info(f"Edge ML engine initialized (simulation: {self.simulation_mode})")
    
    async def initialize(self) -> bool:
        """Initialize ML models and processors."""
        try:
            if self.simulation_mode:
                logger.info("🤖 Running ML in simulation mode")
                return True
            else:
                return await self._load_models()
                
        except Exception as e:
            logger.error(f"ML engine initialization failed: {e}")
            return False
    
    async def _load_models(self) -> bool:
        """Load TensorFlow Lite models."""
        for model_type, model_config in self.models_config.items():
            try:
                model_file = self.model_path / model_config["file"]
                
                if model_file.exists():
                    interpreter = tflite.Interpreter(model_path=str(model_file))
                    interpreter.allocate_tensors()
                    self.interpreters[model_type] = interpreter
                    logger.info(f"✅ Loaded model: {model_type.value}")
                else:
                    logger.warning(f"⚠️ Model file not found: {model_file}")
                    
            except Exception as e:
                logger.error(f"Failed to load {model_type.value}: {e}")
        
        return len(self.interpreters) > 0
    
    async def process_environmental_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process environmental data with edge ML models.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            ML analysis results
        """
        try:
            if self.simulation_mode:
                return self._simulate_ml_analysis(sensor_data)
            
            results = {}
            
            # Extract and normalize features
            features = self._extract_features(sensor_data)
            
            # Run environmental classification
            if ModelType.ENVIRONMENTAL_CLASSIFIER in self.interpreters:
                classification = await self._run_environmental_classifier(features)
                results["environmental_class"] = classification
            
            # Run anomaly detection
            if ModelType.ANOMALY_DETECTOR in self.interpreters:
                anomaly_score = await self._run_anomaly_detector(features)
                results["anomaly_score"] = anomaly_score
            
            # Calculate overall confidence
            results["confidence"] = self._calculate_confidence(results)
            results["timestamp"] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            logger.error(f"ML processing failed: {e}")
            return {"error": str(e)}
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize features from sensor data."""
        features = []
        
        # Standard feature extraction
        feature_map = {
            "temperature": ("temperature_humidity", "temperature"),
            "humidity": ("temperature_humidity", "humidity"),
            "pressure": ("pressure_temperature", "pressure"),
            "air_quality": ("air_quality", "aqi"),
            "soil_moisture": ("soil_moisture", "moisture")
        }
        
        for feature_name, (sensor_name, value_key) in feature_map.items():
            if sensor_name in sensor_data:
                sensor_reading = sensor_data[sensor_name]
                if isinstance(sensor_reading, dict) and "value" in sensor_reading:
                    value = sensor_reading["value"]
                    if isinstance(value, dict) and value_key in value:
                        features.append(float(value[value_key]))
                    elif isinstance(value, (int, float)):
                        features.append(float(value))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Pad or truncate to expected size
        expected_size = 10
        if len(features) < expected_size:
            features.extend([0.0] * (expected_size - len(features)))
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    async def _run_environmental_classifier(self, features: np.ndarray) -> str:
        """Run environmental classification model."""
        interpreter = self.interpreters[ModelType.ENVIRONMENTAL_CLASSIFIER]
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], features)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get class with highest probability
        class_idx = np.argmax(output)
        classes = self.models_config[ModelType.ENVIRONMENTAL_CLASSIFIER]["output_classes"]
        
        return classes[class_idx] if class_idx < len(classes) else "unknown"
    
    async def _run_anomaly_detector(self, features: np.ndarray) -> float:
        """Run anomaly detection model."""
        interpreter = self.interpreters[ModelType.ANOMALY_DETECTOR]
        
        # Adjust feature size for anomaly detector
        if features.shape[1] != 15:
            # Pad or truncate to match expected input
            padded_features = np.zeros((1, 15), dtype=np.float32)
            copy_size = min(features.shape[1], 15)
            padded_features[0, :copy_size] = features[0, :copy_size]
            features = padded_features
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Return anomaly score (0.0 = normal, 1.0 = highly anomalous)
        return float(output[0][0]) if output.size > 0 else 0.0
    
    def _simulate_ml_analysis(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ML analysis for development/testing."""
        import random
        
        # Simulate environmental classification
        classes = ["normal", "moderate", "polluted", "extreme"]
        environmental_class = random.choice(classes)
        
        # Simulate anomaly score based on data variability
        anomaly_score = random.random() * 0.3  # Usually low
        
        # Occasionally simulate anomalies
        if random.random() < 0.05:  # 5% chance
            anomaly_score = 0.8 + random.random() * 0.2
            environmental_class = "extreme"
        
        return {
            "environmental_class": environmental_class,
            "anomaly_score": anomaly_score,
            "confidence": 0.85 + random.random() * 0.1,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence in ML results."""
        confidences = []
        
        # Base confidence on anomaly score
        if "anomaly_score" in results:
            score = results["anomaly_score"]
            # Higher confidence for clear normal or clear anomaly
            confidence = 1.0 - abs(0.5 - score) * 2
            confidences.append(confidence)
        
        # Factor in classification confidence (would come from model probabilities)
        confidences.append(0.85)  # Placeholder
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    async def shutdown(self):
        """Shutdown ML engine."""
        self.interpreters.clear()
        logger.info("Edge ML engine shutdown complete")