# ============================================================================
# furcate_nano/ml_persistence.py
"""
Complete ML model persistence and management system supporting both 
TensorFlow Lite (Raspberry Pi) and PyTorch (Jetson) with cross-platform compatibility.
"""

import asyncio
import logging
import json
import pickle
import joblib
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# ML Framework imports with error handling
try:
    import torch
    import torch.nn as nn
    import torch.jit
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to full TensorFlow
            tflite = tf.lite
            TFLITE_AVAILABLE = True
        except:
            TFLITE_AVAILABLE = False
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TFLITE_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelFramework(Enum):
    """Supported ML frameworks."""
    TENSORFLOW = "tensorflow"
    TENSORFLOW_LITE = "tensorflow_lite"
    PYTORCH = "pytorch"
    PYTORCH_MOBILE = "pytorch_mobile"
    SKLEARN = "scikit_learn"
    ONNX = "onnx"

class ModelType(Enum):
    """Model types for different tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"

class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    name: str
    version: str
    framework: ModelFramework
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    file_path: str
    file_size: int
    checksum: str
    
    # Training metadata
    training_data_size: int = 0
    training_duration_seconds: float = 0.0
    epochs_trained: int = 0
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    
    # Model architecture
    input_shape: List[int] = None
    output_shape: List[int] = None
    num_parameters: int = 0
    
    # Deployment info
    target_platform: str = "unknown"
    optimization_level: str = "none"
    quantized: bool = False
    
    # Additional metadata
    description: str = ""
    tags: List[str] = None
    performance_notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.input_shape is None:
            self.input_shape = []
        if self.output_shape is None:
            self.output_shape = []

class ModelConverter:
    """Handles model conversion between frameworks."""
    
    @staticmethod
    def tensorflow_to_tflite(model_path: str, output_path: str, 
                           quantize: bool = True) -> bool:
        """Convert TensorFlow model to TensorFlow Lite."""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for conversion")
            return False
        
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Configure optimization
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # Optional: Add representative dataset for quantization
                # converter.representative_dataset = representative_data_gen
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"✅ Converted TensorFlow model to TFLite: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorFlow to TFLite conversion failed: {e}")
            return False
    
    @staticmethod
    def pytorch_to_torchscript(model, example_input, output_path: str, 
                              use_trace: bool = True) -> bool:
        """Convert PyTorch model to TorchScript."""
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available for conversion")
            return False
        
        try:
            if use_trace:
                # Trace the model
                traced_model = torch.jit.trace(model, example_input)
            else:
                # Script the model
                traced_model = torch.jit.script(model)
            
            # Save
            traced_model.save(output_path)
            
            logger.info(f"✅ Converted PyTorch model to TorchScript: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PyTorch to TorchScript conversion failed: {e}")
            return False
    
    @staticmethod
    def pytorch_to_mobile(model, example_input, output_path: str) -> bool:
        """Convert PyTorch model to mobile format."""
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available for conversion")
            return False
        
        try:
            # First convert to TorchScript
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize for mobile
            mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            # Save
            mobile_model._save_for_lite_interpreter(output_path)
            
            logger.info(f"✅ Converted PyTorch model to mobile: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PyTorch to mobile conversion failed: {e}")
            return False

class ModelValidator:
    """Validates model integrity and performance."""
    
    @staticmethod
    def validate_model_file(file_path: str, framework: ModelFramework) -> bool:
        """Validate model file integrity."""
        if not Path(file_path).exists():
            logger.error(f"Model file not found: {file_path}")
            return False
        
        try:
            if framework == ModelFramework.TENSORFLOW_LITE:
                return ModelValidator._validate_tflite_model(file_path)
            elif framework == ModelFramework.PYTORCH:
                return ModelValidator._validate_pytorch_model(file_path)
            elif framework == ModelFramework.SKLEARN:
                return ModelValidator._validate_sklearn_model(file_path)
            else:
                logger.warning(f"Validation not implemented for {framework.value}")
                return True
                
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    @staticmethod
    def _validate_tflite_model(file_path: str) -> bool:
        """Validate TensorFlow Lite model."""
        if not TFLITE_AVAILABLE:
            return False
        
        try:
            interpreter = tflite.Interpreter(model_path=file_path)
            interpreter.allocate_tensors()
            
            # Check input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            if not input_details or not output_details:
                logger.error("Invalid TFLite model: missing input/output tensors")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"TFLite model validation failed: {e}")
            return False
    
    @staticmethod
    def _validate_pytorch_model(file_path: str) -> bool:
        """Validate PyTorch model."""
        if not PYTORCH_AVAILABLE:
            return False
        
        try:
            # Try to load the model
            model = torch.jit.load(file_path, map_location='cpu')
            
            # Check if it's a valid ScriptModule
            if not isinstance(model, torch.jit.ScriptModule):
                logger.error("Invalid PyTorch model: not a ScriptModule")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"PyTorch model validation failed: {e}")
            return False
    
    @staticmethod
    def _validate_sklearn_model(file_path: str) -> bool:
        """Validate scikit-learn model."""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Try to load the model
            model = joblib.load(file_path)
            
            # Check if it has predict method
            if not hasattr(model, 'predict'):
                logger.error("Invalid sklearn model: missing predict method")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sklearn model validation failed: {e}")
            return False

class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "model_registry.json"
        self.models: Dict[str, ModelMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    # Convert datetime strings back to datetime objects
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    model_data['framework'] = ModelFramework(model_data['framework'])
                    model_data['model_type'] = ModelType(model_data['model_type'])
                    model_data['status'] = ModelStatus(model_data['status'])
                    
                    self.models[model_id] = ModelMetadata(**model_data)
                
                logger.info(f"📚 Loaded {len(self.models)} models from registry")
                
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                self.models = {}
    
    def save_registry(self):
        """Save model registry to file."""
        try:
            data = {}
            for model_id, metadata in self.models.items():
                model_data = asdict(metadata)
                # Convert datetime objects to ISO format strings
                model_data['created_at'] = metadata.created_at.isoformat()
                model_data['updated_at'] = metadata.updated_at.isoformat()
                model_data['framework'] = metadata.framework.value
                model_data['model_type'] = metadata.model_type.value
                model_data['status'] = metadata.status.value
                data[model_id] = model_data
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"📚 Saved model registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model."""
        try:
            self.models[metadata.model_id] = metadata
            self.save_registry()
            logger.info(f"📚 Registered model: {metadata.name} v{metadata.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def get_latest_model(self, name: str, framework: ModelFramework = None) -> Optional[ModelMetadata]:
        """Get the latest version of a model by name."""
        matching_models = []
        
        for metadata in self.models.values():
            if metadata.name == name:
                if framework is None or metadata.framework == framework:
                    matching_models.append(metadata)
        
        if not matching_models:
            return None
        
        # Sort by version (assuming semantic versioning)
        matching_models.sort(key=lambda x: x.created_at, reverse=True)
        return matching_models[0]
    
    def list_models(self, framework: ModelFramework = None, 
                   model_type: ModelType = None,
                   status: ModelStatus = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if framework:
            models = [m for m in models if m.framework == framework]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda x: x.updated_at, reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry and filesystem."""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            metadata = self.models[model_id]
            
            # Delete model file
            if Path(metadata.file_path).exists():
                Path(metadata.file_path).unlink()
            
            # Remove from registry
            del self.models[model_id]
            self.save_registry()
            
            logger.info(f"🗑️ Deleted model: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False

class ModelManager:
    """Complete ML model management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_path = Path(config.get("models_path", "./models"))
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.registry = ModelRegistry(str(self.models_path))
        self.converter = ModelConverter()
        self.validator = ModelValidator()
        
        # Loaded models cache
        self.loaded_models: Dict[str, Any] = {}
        
        # Platform detection
        self.platform = self._detect_platform()
        
        logger.info(f"🤖 Model Manager initialized for platform: {self.platform}")
    
    def _detect_platform(self) -> str:
        """Detect the current platform."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return 'raspberry_pi_5'
                elif 'jetson' in model.lower():
                    return 'jetson'
        except:
            pass
        return 'unknown'
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    async def save_model(self, model: Any, name: str, version: str,
                        framework: ModelFramework, model_type: ModelType,
                        metadata: Dict[str, Any] = None) -> Optional[str]:
        """Save a model with comprehensive metadata."""
        try:
            model_id = f"{name}_{version}_{int(datetime.now().timestamp())}"
            
            # Determine file extension based on framework
            extensions = {
                ModelFramework.TENSORFLOW: ".h5",
                ModelFramework.TENSORFLOW_LITE: ".tflite",
                ModelFramework.PYTORCH: ".pt",
                ModelFramework.PYTORCH_MOBILE: ".ptl",
                ModelFramework.SKLEARN: ".joblib"
            }
            
            file_name = f"{model_id}{extensions.get(framework, '.pkl')}"
            file_path = self.models_path / file_name
            
            # Save model based on framework
            success = await self._save_model_by_framework(
                model, str(file_path), framework
            )
            
            if not success:
                return None
            
            # Calculate file metadata
            file_size = file_path.stat().st_size
            checksum = self.calculate_checksum(str(file_path))
            
            # Create metadata object
            model_metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                framework=framework,
                model_type=model_type,
                status=ModelStatus.TRAINED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_path=str(file_path),
                file_size=file_size,
                checksum=checksum,
                target_platform=self.platform
            )
            
            # Add additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if hasattr(model_metadata, key):
                        setattr(model_metadata, key, value)
            
            # Validate model
            if self.validator.validate_model_file(str(file_path), framework):
                model_metadata.status = ModelStatus.VALIDATED
            else:
                logger.warning(f"Model validation failed for {model_id}")
            
            # Register model
            if self.registry.register_model(model_metadata):
                logger.info(f"✅ Saved model: {name} v{version} ({framework.value})")
                return model_id
            else:
                # Clean up file if registration failed
                file_path.unlink()
                return None
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    async def _save_model_by_framework(self, model: Any, file_path: str, 
                                     framework: ModelFramework) -> bool:
        """Save model using framework-specific method."""
        try:
            if framework == ModelFramework.TENSORFLOW and TENSORFLOW_AVAILABLE:
                model.save(file_path)
                
            elif framework == ModelFramework.PYTORCH and PYTORCH_AVAILABLE:
                if isinstance(model, torch.jit.ScriptModule):
                    model.save(file_path)
                else:
                    torch.save(model.state_dict(), file_path)
                
            elif framework == ModelFramework.SKLEARN and SKLEARN_AVAILABLE:
                joblib.dump(model, file_path)
                
            elif framework == ModelFramework.TENSORFLOW_LITE:
                # Assume model is already TFLite bytes
                with open(file_path, 'wb') as f:
                    f.write(model)
                    
            else:
                # Fallback to pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Framework-specific save failed: {e}")
            return False
    
    async def load_model(self, model_id: str, cache: bool = True) -> Optional[Any]:
        """Load a model by ID."""
        # Check cache first
        if cache and model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        metadata = self.registry.get_model(model_id)
        if not metadata:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        if not Path(metadata.file_path).exists():
            logger.error(f"Model file not found: {metadata.file_path}")
            return None
        
        try:
            model = await self._load_model_by_framework(
                metadata.file_path, metadata.framework
            )
            
            if model is not None and cache:
                self.loaded_models[model_id] = model
            
            logger.info(f"📖 Loaded model: {metadata.name} v{metadata.version}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    async def _load_model_by_framework(self, file_path: str, 
                                     framework: ModelFramework) -> Optional[Any]:
        """Load model using framework-specific method."""
        try:
            if framework == ModelFramework.TENSORFLOW and TENSORFLOW_AVAILABLE:
                return tf.keras.models.load_model(file_path)
                
            elif framework == ModelFramework.TENSORFLOW_LITE and TFLITE_AVAILABLE:
                interpreter = tflite.Interpreter(model_path=file_path)
                interpreter.allocate_tensors()
                return interpreter
                
            elif framework == ModelFramework.PYTORCH and PYTORCH_AVAILABLE:
                return torch.jit.load(file_path, map_location='cpu')
                
            elif framework == ModelFramework.SKLEARN and SKLEARN_AVAILABLE:
                return joblib.load(file_path)
                
            else:
                # Fallback to pickle
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Framework-specific load failed: {e}")
            return None
    
    async def convert_model(self, model_id: str, target_framework: ModelFramework,
                          optimize: bool = True) -> Optional[str]:
        """Convert model to different framework."""
        source_metadata = self.registry.get_model(model_id)
        if not source_metadata:
            logger.error(f"Source model {model_id} not found")
            return None
        
        if source_metadata.framework == target_framework:
            logger.warning(f"Model {model_id} already in target framework")
            return model_id
        
        try:
            # Load source model
            source_model = await self.load_model(model_id, cache=False)
            if source_model is None:
                return None
            
            # Perform conversion
            converted_model = None
            new_version = f"{source_metadata.version}_converted"
            
            if (source_metadata.framework == ModelFramework.TENSORFLOW and 
                target_framework == ModelFramework.TENSORFLOW_LITE):
                
                temp_path = self.models_path / f"temp_conversion_{int(datetime.now().timestamp())}.tflite"
                success = self.converter.tensorflow_to_tflite(
                    source_metadata.file_path, str(temp_path), quantize=optimize
                )
                
                if success:
                    with open(temp_path, 'rb') as f:
                        converted_model = f.read()
                    temp_path.unlink()  # Clean up temp file
            
            elif (source_metadata.framework == ModelFramework.PYTORCH and 
                  target_framework == ModelFramework.PYTORCH_MOBILE):
                
                # Need example input for tracing
                # This is a limitation - would need to store example inputs
                logger.error("PyTorch conversion requires example input")
                return None
            
            if converted_model is not None:
                # Save converted model
                converted_id = await self.save_model(
                    converted_model,
                    source_metadata.name,
                    new_version,
                    target_framework,
                    source_metadata.model_type,
                    {
                        'description': f"Converted from {source_metadata.framework.value}",
                        'optimization_level': 'optimized' if optimize else 'standard',
                        'quantized': optimize,
                        'tags': source_metadata.tags + ['converted']
                    }
                )
                
                return converted_id
            
            logger.error(f"Conversion from {source_metadata.framework.value} to {target_framework.value} not implemented")
            return None
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            return None
    
    async def benchmark_model(self, model_id: str, test_data: np.ndarray,
                            test_labels: np.ndarray = None) -> Dict[str, Any]:
        """Benchmark model performance."""
        model = await self.load_model(model_id)
        metadata = self.registry.get_model(model_id)
        
        if not model or not metadata:
            return {"error": "Model not found"}
        
        try:
            start_time = datetime.now()
            
            # Run inference
            if metadata.framework == ModelFramework.TENSORFLOW_LITE:
                predictions = self._benchmark_tflite(model, test_data)
            elif metadata.framework == ModelFramework.PYTORCH:
                predictions = self._benchmark_pytorch(model, test_data)
            elif metadata.framework == ModelFramework.SKLEARN:
                predictions = model.predict(test_data)
            else:
                return {"error": f"Benchmarking not implemented for {metadata.framework.value}"}
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            results = {
                "model_id": model_id,
                "inference_time_seconds": inference_time,
                "samples_processed": len(test_data),
                "samples_per_second": len(test_data) / inference_time,
                "predictions_shape": list(predictions.shape) if hasattr(predictions, 'shape') else len(predictions)
            }
            
            # Calculate accuracy metrics if labels provided
            if test_labels is not None:
                if metadata.model_type == ModelType.CLASSIFICATION:
                    accuracy = accuracy_score(test_labels, predictions)
                    results["accuracy"] = accuracy
                    
                    # Update model metadata
                    metadata.accuracy = accuracy
                    metadata.updated_at = datetime.now()
                    self.registry.register_model(metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return {"error": str(e)}
    
    def _benchmark_tflite(self, interpreter, test_data: np.ndarray) -> np.ndarray:
        """Benchmark TensorFlow Lite model."""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        
        for sample in test_data:
            # Prepare input
            input_data = np.expand_dims(sample, axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
        
        return np.array(predictions)
    
    def _benchmark_pytorch(self, model, test_data: np.ndarray) -> np.ndarray:
        """Benchmark PyTorch model."""
        model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(test_data)
            output = model(input_tensor)
            
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            else:
                return output
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model information."""
        metadata = self.registry.get_model(model_id)
        if not metadata:
            return {"error": "Model not found"}
        
        return {
            "metadata": asdict(metadata),
            "file_exists": Path(metadata.file_path).exists(),
            "is_loaded": model_id in self.loaded_models,
            "platform_compatible": self._check_platform_compatibility(metadata)
        }
    
    def _check_platform_compatibility(self, metadata: ModelMetadata) -> bool:
        """Check if model is compatible with current platform."""
        if self.platform == "raspberry_pi_5":
            return metadata.framework in [
                ModelFramework.TENSORFLOW_LITE,
                ModelFramework.SKLEARN
            ]
        elif self.platform == "jetson":
            return metadata.framework in [
                ModelFramework.PYTORCH,
                ModelFramework.PYTORCH_MOBILE,
                ModelFramework.TENSORFLOW,
                ModelFramework.SKLEARN
            ]
        else:
            return True  # Unknown platform, assume compatible
    
    def list_models(self, **filters) -> List[Dict[str, Any]]:
        """List all models with filtering options."""
        models = self.registry.list_models(**filters)
        
        return [
            {
                **asdict(metadata),
                "file_exists": Path(metadata.file_path).exists(),
                "is_loaded": metadata.model_id in self.loaded_models,
                "platform_compatible": self._check_platform_compatibility(metadata)
            }
            for metadata in models
        ]
    
    def cleanup_cache(self):
        """Clear loaded models cache."""
        self.loaded_models.clear()
        logger.info("🧹 Cleared model cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model management statistics."""
        all_models = self.registry.list_models()
        
        framework_counts = {}
        type_counts = {}
        status_counts = {}
        
        total_size = 0
        
        for metadata in all_models:
            # Count by framework
            framework = metadata.framework.value
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
            
            # Count by type
            model_type = metadata.model_type.value
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            # Count by status
            status = metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Sum file sizes
            total_size += metadata.file_size
        
        return {
            "total_models": len(all_models),
            "loaded_models": len(self.loaded_models),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "platform": self.platform,
            "framework_distribution": framework_counts,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "average_model_size_mb": round(total_size / len(all_models) / 1024 / 1024, 2) if all_models else 0
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_model_management():
        """Test the model management system."""
        config = {
            "models_path": "./test_models"
        }
        
        manager = ModelManager(config)
        
        print("Testing model management system...")
        
        # Test with a simple sklearn model
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # Create and train a simple model
            X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Save model
            model_id = await manager.save_model(
                model=model,
                name="test_classifier",
                version="1.0.0",
                framework=ModelFramework.SKLEARN,
                model_type=ModelType.CLASSIFICATION,
                metadata={
                    "description": "Test random forest classifier",
                    "training_data_size": len(X),
                    "tags": ["test", "classification"]
                }
            )
            
            if model_id:
                print(f"✅ Saved model with ID: {model_id}")
                
                # Load model
                loaded_model = await manager.load_model(model_id)
                if loaded_model:
                    print("✅ Successfully loaded model")
                    
                    # Benchmark model
                    benchmark_results = await manager.benchmark_model(model_id, X[:10], y[:10])
                    print(f"📊 Benchmark results: {benchmark_results}")
                
                # Get model info
                info = manager.get_model_info(model_id)
                print(f"📋 Model info: {json.dumps(info, indent=2, default=str)}")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"📈 Statistics: {json.dumps(stats, indent=2)}")
        
        # List models
        models = manager.list_models()
        print(f"📚 Found {len(models)} models")
        
        print("Model management test completed!")
    
    # Run test
    asyncio.run(test_model_management())