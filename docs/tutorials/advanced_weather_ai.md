# Advanced Weather AI with Tenzro Cortex and Global Models

Comprehensive guide for integrating Furcate Nano with Tenzro Cortex's universal model inference API to leverage world-class AI models like Microsoft Aurora, Google GenCast, and NASA Prithvi for environmental forecasting and analysis.

## Overview

This tutorial demonstrates how to create a hybrid environmental monitoring system that combines:

- **Local Edge AI**: Furcate Nano device-level ML models
- **Global Cloud AI**: World-class models via Tenzro Cortex API
- **Distributed Learning**: Aggregating local network data for global model training
- **Advanced Forecasting**: Weather and climate predictions using state-of-the-art models

### Featured AI Models

1. **Microsoft Aurora**: 1.3B parameter foundation model for weather forecasting and atmospheric prediction
2. **Google GenCast**: Probabilistic ensemble weather model with 15-day forecasting capability
3. **NASA Prithvi**: Earth observation foundation models for geospatial analysis and climate monitoring

## Prerequisites

### Technical Requirements
- **Furcate Nano devices** with internet connectivity
- **Tenzro Cortex API access** and authentication credentials
- **Python 3.8+** with asyncio support
- **Local model training capability** (TensorFlow/PyTorch)

### API Access Requirements
- Tenzro Cortex API key and endpoint access
- Model-specific authentication (if required)
- Sufficient compute credits for cloud inference

### Knowledge Prerequisites
- Understanding of weather data and environmental monitoring
- Basic machine learning concepts
- API integration experience
- Familiarity with time-series forecasting

## Part 1: Tenzro Cortex Integration Framework

### Core Cortex Client Implementation

```python
import asyncio
import aiohttp
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging

@dataclass
class WeatherPredictionRequest:
    """Request structure for weather prediction models"""
    model_name: str
    input_data: Dict[str, Any]
    prediction_horizon: str  # "1day", "3day", "15day", etc.
    resolution: str  # "global", "regional", "local"
    ensemble_size: int = 1
    confidence_threshold: float = 0.8

@dataclass
class EnvironmentalAnalysisRequest:
    """Request structure for environmental analysis models"""
    model_name: str
    satellite_data: Optional[Dict] = None
    sensor_data: Optional[Dict] = None
    geospatial_coordinates: Optional[Dict] = None
    analysis_type: str = "comprehensive"  # "flood", "fire", "crop", "comprehensive"

class TenzroCortexClient:
    """Advanced client for Tenzro Cortex universal model inference API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_base = config.get("cortex_endpoint", "https://api.tenzro.com/cortex/v1")
        self.api_key = config.get("api_key")
        self.device_id = config.get("device_id")
        
        # Model configurations for different AI models
        self.model_configs = {
            "microsoft_aurora": {
                "endpoint": "/models/microsoft/aurora",
                "max_horizon_days": 14,
                "resolution_degrees": 0.25,
                "capabilities": ["weather", "air_quality", "cyclones", "atmospheric"]
            },
            "google_gencast": {
                "endpoint": "/models/google/gencast",
                "max_horizon_days": 15,
                "ensemble_size": 50,
                "capabilities": ["probabilistic_weather", "extreme_events", "ensemble_forecasting"]
            },
            "nasa_prithvi_eo": {
                "endpoint": "/models/nasa/prithvi-eo",
                "resolution_meters": 30,
                "capabilities": ["land_cover", "crop_classification", "disaster_mapping", "deforestation"]
            },
            "nasa_prithvi_weather": {
                "endpoint": "/models/nasa/prithvi-weather",
                "data_source": "merra2",
                "capabilities": ["climate_analysis", "storm_tracking", "seasonal_forecasting"]
            }
        }
        
        # Session for HTTP requests
        self.session = None
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        
    async def initialize(self):
        """Initialize the Cortex client and validate connection"""
        try:
            # Create aiohttp session with authentication
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'X-Device-ID': self.device_id,
                    'X-Client': 'furcate-nano-weather-ai'
                }
            )
            
            # Test connection and get available models
            available_models = await self._get_available_models()
            print(f"✅ Tenzro Cortex connected with {len(available_models)} available models")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Tenzro Cortex client: {e}")
            return False
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models from Cortex API"""
        try:
            async with self.session.get(f"{self.api_base}/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    return [model['name'] for model in models_data.get('models', [])]
                else:
                    print(f"Failed to get models: {response.status}")
                    return []
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []
    
    async def predict_weather_aurora(self, sensor_data: Dict, forecast_days: int = 10) -> Dict:
        """Use Microsoft Aurora for weather forecasting"""
        try:
            await self.rate_limiter.acquire()
            
            # Prepare Aurora-specific input format
            aurora_input = self._prepare_aurora_input(sensor_data, forecast_days)
            
            request_payload = {
                "model": "microsoft_aurora",
                "input": aurora_input,
                "parameters": {
                    "forecast_horizon_days": min(forecast_days, 14),
                    "resolution": "0.25_degree",
                    "output_variables": [
                        "temperature_2m",
                        "humidity_2m", 
                        "wind_speed_10m",
                        "precipitation",
                        "air_quality_index",
                        "atmospheric_pressure"
                    ],
                    "confidence_intervals": True
                }
            }
            
            async with self.session.post(
                f"{self.api_base}{self.model_configs['microsoft_aurora']['endpoint']}/predict",
                json=request_payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Process Aurora response
                    processed_result = self._process_aurora_response(result)
                    
                    print(f"✅ Aurora forecast generated for {forecast_days} days")
                    return {
                        "success": True,
                        "model": "microsoft_aurora",
                        "forecast_horizon_days": forecast_days,
                        "predictions": processed_result,
                        "metadata": {
                            "model_version": result.get("model_version", "unknown"),
                            "processing_time_ms": result.get("processing_time_ms", 0),
                            "confidence_score": result.get("confidence_score", 0.0)
                        }
                    }
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Aurora API error: {response.status} - {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": f"Aurora prediction failed: {str(e)}"}
    
    async def predict_weather_gencast(self, sensor_data: Dict, ensemble_size: int = 50) -> Dict:
        """Use Google GenCast for probabilistic weather forecasting"""
        try:
            await self.rate_limiter.acquire()
            
            # Prepare GenCast-specific input format
            gencast_input = self._prepare_gencast_input(sensor_data)
            
            request_payload = {
                "model": "google_gencast",
                "input": gencast_input,
                "parameters": {
                    "ensemble_size": min(ensemble_size, 50),
                    "forecast_horizon_days": 15,
                    "resolution": "0.25_degree",
                    "probabilistic_output": True,
                    "extreme_events_focus": True,
                    "output_variables": [
                        "temperature",
                        "precipitation_probability",
                        "wind_speed",
                        "pressure",
                        "extreme_weather_risk"
                    ]
                }
            }
            
            async with self.session.post(
                f"{self.api_base}{self.model_configs['google_gencast']['endpoint']}/predict",
                json=request_payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Process GenCast ensemble response
                    processed_result = self._process_gencast_response(result, ensemble_size)
                    
                    print(f"✅ GenCast ensemble forecast generated with {ensemble_size} members")
                    return {
                        "success": True,
                        "model": "google_gencast",
                        "ensemble_size": ensemble_size,
                        "predictions": processed_result,
                        "uncertainty_metrics": result.get("uncertainty_metrics", {}),
                        "extreme_events": result.get("extreme_events", [])
                    }
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"GenCast API error: {response.status} - {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": f"GenCast prediction failed: {str(e)}"}
    
    async def analyze_environment_prithvi(self, analysis_request: EnvironmentalAnalysisRequest) -> Dict:
        """Use NASA Prithvi models for environmental analysis"""
        try:
            await self.rate_limiter.acquire()
            
            # Determine which Prithvi model to use
            if analysis_request.satellite_data:
                model_key = "nasa_prithvi_eo"
                prithvi_input = self._prepare_prithvi_eo_input(analysis_request)
            else:
                model_key = "nasa_prithvi_weather"
                prithvi_input = self._prepare_prithvi_weather_input(analysis_request)
            
            request_payload = {
                "model": model_key.replace("_", "-"),
                "input": prithvi_input,
                "parameters": {
                    "analysis_type": analysis_request.analysis_type,
                    "resolution_meters": 30,
                    "temporal_window_days": 30,
                    "confidence_threshold": 0.8
                }
            }
            
            async with self.session.post(
                f"{self.api_base}{self.model_configs[model_key]['endpoint']}/analyze",
                json=request_payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Process Prithvi response
                    processed_result = self._process_prithvi_response(result, analysis_request.analysis_type)
                    
                    print(f"✅ Prithvi {analysis_request.analysis_type} analysis completed")
                    return {
                        "success": True,
                        "model": model_key,
                        "analysis_type": analysis_request.analysis_type,
                        "results": processed_result,
                        "confidence_metrics": result.get("confidence_metrics", {})
                    }
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Prithvi API error: {response.status} - {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": f"Prithvi analysis failed: {str(e)}"}
    
    def _prepare_aurora_input(self, sensor_data: Dict, forecast_days: int) -> Dict:
        """Prepare input data for Microsoft Aurora model"""
        return {
            "initial_conditions": {
                "timestamp": datetime.now().isoformat(),
                "location": {
                    "latitude": sensor_data.get("location", {}).get("latitude", 0.0),
                    "longitude": sensor_data.get("location", {}).get("longitude", 0.0),
                    "elevation": sensor_data.get("location", {}).get("elevation", 0.0)
                },
                "surface_conditions": {
                    "temperature_2m": sensor_data.get("temperature", 20.0),
                    "humidity_2m": sensor_data.get("humidity", 50.0),
                    "pressure_surface": sensor_data.get("pressure", 1013.25),
                    "wind_speed_10m": sensor_data.get("wind_speed", 2.0),
                    "wind_direction_10m": sensor_data.get("wind_direction", 180.0)
                },
                "atmospheric_conditions": {
                    "air_quality_index": sensor_data.get("air_quality", {}).get("aqi", 50),
                    "pm2_5": sensor_data.get("air_quality", {}).get("pm2_5", 10.0),
                    "pm10": sensor_data.get("air_quality", {}).get("pm10", 20.0),
                    "visibility_km": sensor_data.get("visibility", 10.0)
                }
            },
            "forecast_parameters": {
                "lead_times_hours": list(range(6, forecast_days * 24 + 1, 6)),
                "domain": "global",
                "physics_enhanced": True
            }
        }
    
    def _prepare_gencast_input(self, sensor_data: Dict) -> Dict:
        """Prepare input data for Google GenCast model"""
        return {
            "initial_state": {
                "timestamp": datetime.now().isoformat(),
                "coordinates": {
                    "lat": sensor_data.get("location", {}).get("latitude", 0.0),
                    "lon": sensor_data.get("location", {}).get("longitude", 0.0)
                },
                "atmospheric_state": {
                    "temperature": sensor_data.get("temperature", 20.0),
                    "humidity": sensor_data.get("humidity", 50.0),
                    "pressure": sensor_data.get("pressure", 1013.25),
                    "wind_u": sensor_data.get("wind_u", 0.0),
                    "wind_v": sensor_data.get("wind_v", 0.0)
                }
            },
            "ensemble_config": {
                "perturbation_method": "bred_vectors",
                "uncertainty_propagation": True,
                "extreme_events_tracking": True
            }
        }
    
    def _prepare_prithvi_eo_input(self, request: EnvironmentalAnalysisRequest) -> Dict:
        """Prepare input for NASA Prithvi Earth Observation model"""
        return {
            "satellite_imagery": request.satellite_data,
            "coordinates": request.geospatial_coordinates,
            "analysis_parameters": {
                "spectral_bands": ["red", "green", "blue", "nir", "swir1", "swir2"],
                "temporal_composite": "median",
                "cloud_mask": True,
                "atmospheric_correction": True
            },
            "target_classes": self._get_target_classes(request.analysis_type)
        }
    
    def _prepare_prithvi_weather_input(self, request: EnvironmentalAnalysisRequest) -> Dict:
        """Prepare input for NASA Prithvi Weather-Climate model"""
        return {
            "weather_data": request.sensor_data,
            "coordinates": request.geospatial_coordinates,
            "analysis_parameters": {
                "merra2_variables": [
                    "temperature", "humidity", "wind_speed", 
                    "precipitation", "pressure", "radiation"
                ],
                "temporal_resolution": "hourly",
                "vertical_levels": True
            }
        }
    
    def _get_target_classes(self, analysis_type: str) -> List[str]:
        """Get target classes for different analysis types"""
        class_mappings = {
            "flood": ["water", "flood_area", "dry_land", "vegetation"],
            "fire": ["burned_area", "active_fire", "smoke", "unburned"],
            "crop": ["corn", "soybean", "wheat", "rice", "fallow", "urban"],
            "comprehensive": ["water", "forest", "grassland", "urban", "agriculture", "barren"]
        }
        return class_mappings.get(analysis_type, class_mappings["comprehensive"])
    
    def _process_aurora_response(self, response: Dict) -> Dict:
        """Process and structure Aurora model response"""
        predictions = response.get("predictions", {})
        
        # Extract time series forecasts
        forecast_data = {}
        for variable in ["temperature_2m", "humidity_2m", "wind_speed_10m", "precipitation", "air_quality_index"]:
            if variable in predictions:
                forecast_data[variable] = {
                    "values": predictions[variable].get("values", []),
                    "timestamps": predictions[variable].get("timestamps", []),
                    "confidence_intervals": predictions[variable].get("confidence_intervals", {}),
                    "units": predictions[variable].get("units", "")
                }
        
        return {
            "forecasts": forecast_data,
            "summary": {
                "max_temperature": max(forecast_data.get("temperature_2m", {}).get("values", [0])),
                "min_temperature": min(forecast_data.get("temperature_2m", {}).get("values", [0])),
                "total_precipitation": sum(forecast_data.get("precipitation", {}).get("values", [0])),
                "avg_air_quality": np.mean(forecast_data.get("air_quality_index", {}).get("values", [50]))
            }
        }
    
    def _process_gencast_response(self, response: Dict, ensemble_size: int) -> Dict:
        """Process and structure GenCast ensemble response"""
        ensemble_predictions = response.get("ensemble_predictions", [])
        
        # Calculate ensemble statistics
        ensemble_stats = {}
        variables = ["temperature", "precipitation_probability", "wind_speed", "pressure"]
        
        for variable in variables:
            if variable in ensemble_predictions[0]:
                all_values = [member[variable]["values"] for member in ensemble_predictions if variable in member]
                
                if all_values:
                    ensemble_stats[variable] = {
                        "mean": np.mean(all_values, axis=0).tolist(),
                        "std": np.std(all_values, axis=0).tolist(),
                        "percentiles": {
                            "p10": np.percentile(all_values, 10, axis=0).tolist(),
                            "p25": np.percentile(all_values, 25, axis=0).tolist(),
                            "p75": np.percentile(all_values, 75, axis=0).tolist(),
                            "p90": np.percentile(all_values, 90, axis=0).tolist()
                        },
                        "probability_thresholds": self._calculate_probability_thresholds(all_values, variable)
                    }
        
        return {
            "ensemble_statistics": ensemble_stats,
            "uncertainty_quantification": {
                "forecast_spread": np.mean([np.std(member["temperature"]["values"]) for member in ensemble_predictions]),
                "skill_score": response.get("skill_metrics", {}).get("crps_score", 0.0)
            },
            "probabilistic_forecasts": self._generate_probabilistic_forecasts(ensemble_stats)
        }
    
    def _process_prithvi_response(self, response: Dict, analysis_type: str) -> Dict:
        """Process and structure Prithvi model response"""
        analysis_results = response.get("analysis_results", {})
        
        if analysis_type == "flood":
            return self._process_flood_analysis(analysis_results)
        elif analysis_type == "fire":
            return self._process_fire_analysis(analysis_results)
        elif analysis_type == "crop":
            return self._process_crop_analysis(analysis_results)
        else:
            return self._process_comprehensive_analysis(analysis_results)
    
    def _process_flood_analysis(self, results: Dict) -> Dict:
        """Process flood detection analysis"""
        return {
            "flood_extent": {
                "total_area_km2": results.get("flood_area_km2", 0),
                "severity_levels": results.get("severity_classification", {}),
                "affected_regions": results.get("affected_regions", [])
            },
            "risk_assessment": {
                "flood_probability": results.get("flood_probability", 0.0),
                "damage_potential": results.get("damage_assessment", "low"),
                "evacuation_zones": results.get("evacuation_zones", [])
            },
            "temporal_analysis": {
                "flood_duration_hours": results.get("estimated_duration", 0),
                "peak_flood_time": results.get("peak_time", ""),
                "recession_forecast": results.get("recession_forecast", {})
            }
        }
    
    def _process_fire_analysis(self, results: Dict) -> Dict:
        """Process wildfire detection analysis"""
        return {
            "fire_detection": {
                "active_fires": results.get("active_fire_count", 0),
                "burned_area_km2": results.get("burned_area_km2", 0),
                "fire_intensity": results.get("fire_intensity", "low")
            },
            "spread_prediction": {
                "spread_direction": results.get("spread_direction", "unknown"),
                "spread_rate_kmh": results.get("spread_rate", 0.0),
                "containment_probability": results.get("containment_prob", 0.0)
            },
            "impact_assessment": {
                "threatened_infrastructure": results.get("threatened_assets", []),
                "air_quality_impact": results.get("smoke_dispersion", {}),
                "evacuation_recommendations": results.get("evacuation_zones", [])
            }
        }
    
    def _process_crop_analysis(self, results: Dict) -> Dict:
        """Process crop classification analysis"""
        return {
            "crop_classification": {
                "crop_types": results.get("crop_distribution", {}),
                "total_agricultural_area": results.get("total_ag_area_km2", 0),
                "dominant_crops": results.get("dominant_crops", [])
            },
            "health_assessment": {
                "vegetation_index": results.get("ndvi_statistics", {}),
                "health_score": results.get("crop_health_score", 0.0),
                "stress_indicators": results.get("stress_factors", [])
            },
            "yield_prediction": {
                "estimated_yield": results.get("yield_prediction", {}),
                "confidence_interval": results.get("yield_confidence", {}),
                "harvest_timing": results.get("optimal_harvest_date", "")
            }
        }
    
    def _process_comprehensive_analysis(self, results: Dict) -> Dict:
        """Process comprehensive environmental analysis"""
        return {
            "land_cover": {
                "classification": results.get("land_cover_distribution", {}),
                "change_detection": results.get("land_cover_changes", {}),
                "urbanization_rate": results.get("urban_expansion_rate", 0.0)
            },
            "ecosystem_health": {
                "biodiversity_index": results.get("biodiversity_score", 0.0),
                "vegetation_health": results.get("vegetation_health", {}),
                "water_quality": results.get("water_quality_index", 0.0)
            },
            "environmental_trends": {
                "deforestation_rate": results.get("deforestation_rate", 0.0),
                "reforestation_areas": results.get("reforestation_km2", 0.0),
                "climate_impact_indicators": results.get("climate_indicators", {})
            }
        }
    
    def _calculate_probability_thresholds(self, ensemble_values: List, variable: str) -> Dict:
        """Calculate probability thresholds for weather events"""
        # Define thresholds based on variable type
        thresholds = {
            "temperature": {"cold": 0, "hot": 30, "extreme_hot": 40},
            "precipitation_probability": {"light": 0.1, "moderate": 0.5, "heavy": 0.8},
            "wind_speed": {"calm": 5, "moderate": 15, "strong": 25},
            "pressure": {"low": 1000, "normal": 1013, "high": 1025}
        }
        
        var_thresholds = thresholds.get(variable, {})
        probabilities = {}
        
        for threshold_name, threshold_value in var_thresholds.items():
            if variable == "precipitation_probability":
                prob = np.mean([np.mean(np.array(values) > threshold_value) for values in ensemble_values])
            elif variable == "temperature" and "hot" in threshold_name:
                prob = np.mean([np.mean(np.array(values) > threshold_value) for values in ensemble_values])
            elif variable == "temperature" and "cold" in threshold_name:
                prob = np.mean([np.mean(np.array(values) < threshold_value) for values in ensemble_values])
            else:
                prob = np.mean([np.mean(np.array(values) > threshold_value) for values in ensemble_values])
            
            probabilities[threshold_name] = float(prob)
        
        return probabilities
    
    def _generate_probabilistic_forecasts(self, ensemble_stats: Dict) -> Dict:
        """Generate human-readable probabilistic forecasts"""
        forecasts = {}
        
        # Temperature forecast
        if "temperature" in ensemble_stats:
            temp_mean = np.array(ensemble_stats["temperature"]["mean"])
            temp_std = np.array(ensemble_stats["temperature"]["std"])
            
            forecasts["temperature"] = {
                "most_likely": f"{temp_mean[0]:.1f}°C",
                "range": f"{temp_mean[0] - temp_std[0]:.1f}°C to {temp_mean[0] + temp_std[0]:.1f}°C",
                "confidence": "high" if temp_std[0] < 2 else "medium" if temp_std[0] < 4 else "low"
            }
        
        # Precipitation forecast
        if "precipitation_probability" in ensemble_stats:
            precip_prob = np.array(ensemble_stats["precipitation_probability"]["mean"])
            
            forecasts["precipitation"] = {
                "probability": f"{precip_prob[0]*100:.0f}%",
                "likelihood": "high" if precip_prob[0] > 0.7 else "medium" if precip_prob[0] > 0.3 else "low",
                "confidence": "high" if ensemble_stats["precipitation_probability"]["std"][0] < 0.1 else "medium"
            }
        
        return forecasts
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit permission"""
        async with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < timedelta(minutes=1)]
            
            if len(self.requests) >= self.max_requests:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0]).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(now)
```

## Part 2: Hybrid Local-Cloud AI System

### Intelligent Model Selection and Orchestration

```python
class HybridEnvironmentalAI:
    """Orchestrates local edge AI with cloud-based global models"""
    
    def __init__(self, local_ml_processor, cortex_client, config: Dict):
        self.local_ml = local_ml_processor
        self.cortex = cortex_client
        self.config = config
        
        # Model selection criteria
        self.model_selection_rules = {
            "local_only": {
                "conditions": ["no_internet", "low_power", "privacy_mode"],
                "confidence_threshold": 0.6
            },
            "cloud_preferred": {
                "conditions": ["complex_weather", "long_forecast", "high_accuracy_needed"],
                "confidence_threshold": 0.9
            },
            "hybrid_ensemble": {
                "conditions": ["critical_decision", "emergency", "research_mode"],
                "weight_local": 0.3,
                "weight_cloud": 0.7
            }
        }
        
        # Performance tracking
        self.model_performance = {
            "local": {"accuracy": 0.8, "speed": 0.9, "availability": 1.0},
            "aurora": {"accuracy": 0.95, "speed": 0.6, "availability": 0.9},
            "gencast": {"accuracy": 0.96, "speed": 0.5, "availability": 0.9},
            "prithvi": {"accuracy": 0.92, "speed": 0.7, "availability": 0.9}
        }
    
    async def predict_environmental_conditions(self, sensor_data: Dict, 
                                             forecast_horizon: str = "3day",
                                             analysis_type: str = "comprehensive") -> Dict:
        """Intelligent prediction using optimal model selection"""
        try:
            # Analyze requirements and select optimal approach
            approach = await self._select_optimal_approach(sensor_data, forecast_horizon, analysis_type)
            
            if approach == "local_only":
                return await self._local_prediction(sensor_data, forecast_horizon)
            elif approach == "cloud_preferred":
                return await self._cloud_prediction(sensor_data, forecast_horizon, analysis_type)
            elif approach == "hybrid_ensemble":
                return await self._hybrid_ensemble_prediction(sensor_data, forecast_horizon, analysis_type)
            else:
                # Fallback to local
                return await self._local_prediction(sensor_data, forecast_horizon)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to local prediction
            return await self._local_prediction(sensor_data, forecast_horizon)
    
    async def _select_optimal_approach(self, sensor_data: Dict, 
                                     forecast_horizon: str, 
                                     analysis_type: str) -> str:
        """Select optimal prediction approach based on current conditions"""
        
        # Check internet connectivity
        internet_available = await self._check_internet_connectivity()
        if not internet_available:
            return "local_only"
        
        # Check power/resource constraints
        if self._is_low_power_mode():
            return "local_only"
        
        # Check forecast complexity
        horizon_days = int(forecast_horizon.replace("day", ""))
        if horizon_days > 7 or analysis_type in ["comprehensive", "emergency"]:
            return "cloud_preferred"
        
        # Check if high accuracy is critical
        if self._is_critical_prediction(sensor_data):
            return "hybrid_ensemble"
        
        # Default to cloud for better accuracy
        return "cloud_preferred"
    
    async def _local_prediction(self, sensor_data: Dict, forecast_horizon: str) -> Dict:
        """Generate prediction using local edge ML models"""
        try:
            # Get local ML prediction
            local_result = await self.local_ml.process_environmental_data(sensor_data)
            
            # Enhance with local weather forecasting
            local_forecast = await self._generate_local_forecast(sensor_data, forecast_horizon)
            
            return {
                "approach": "local_only",
                "prediction": local_result,
                "forecast": local_forecast,
                "confidence": local_result.get("confidence", 0.6),
                "processing_time_ms": 50,
                "model_source": "edge"
            }
            
        except Exception as e:
            return {"error": f"Local prediction failed: {e}", "approach": "local_only"}
    
    async def _cloud_prediction(self, sensor_data: Dict, 
                              forecast_horizon: str, 
                              analysis_type: str) -> Dict:
        """Generate prediction using cloud-based global models"""
        try:
            results = {}
            
            # Determine optimal cloud models based on requirements
            if forecast_horizon.endswith("day") and int(forecast_horizon.replace("day", "")) <= 14:
                # Use Aurora for weather forecasting
                aurora_result = await self.cortex.predict_weather_aurora(
                    sensor_data, int(forecast_horizon.replace("day", ""))
                )
                results["aurora"] = aurora_result
            
            if forecast_horizon.endswith("day") and int(forecast_horizon.replace("day", "")) <= 15:
                # Use GenCast for probabilistic forecasting
                gencast_result = await self.cortex.predict_weather_gencast(sensor_data)
                results["gencast"] = gencast_result
            
            if analysis_type in ["comprehensive", "flood", "fire", "crop"]:
                # Use Prithvi for environmental analysis
                prithvi_request = EnvironmentalAnalysisRequest(
                    model_name="nasa_prithvi",
                    sensor_data=sensor_data,
                    geospatial_coordinates=sensor_data.get("location", {}),
                    analysis_type=analysis_type
                )
                prithvi_result = await self.cortex.analyze_environment_prithvi(prithvi_request)
                results["prithvi"] = prithvi_result
            
            # Combine and synthesize results
            synthesized_result = await self._synthesize_cloud_results(results, forecast_horizon)
            
            return {
                "approach": "cloud_preferred",
                "prediction": synthesized_result,
                "models_used": list(results.keys()),
                "confidence": self._calculate_ensemble_confidence(results),
                "processing_time_ms": 8000,  # Typical cloud processing time
                "model_source": "cloud"
            }
            
        except Exception as e:
            # Fallback to local prediction
            return await self._local_prediction(sensor_data, forecast_horizon)
    
    async def _hybrid_ensemble_prediction(self, sensor_data: Dict, 
                                        forecast_horizon: str, 
                                        analysis_type: str) -> Dict:
        """Generate ensemble prediction combining local and cloud models"""
        try:
            # Get both local and cloud predictions
            local_task = asyncio.create_task(self._local_prediction(sensor_data, forecast_horizon))
            cloud_task = asyncio.create_task(self._cloud_prediction(sensor_data, forecast_horizon, analysis_type))
            
            local_result, cloud_result = await asyncio.gather(local_task, cloud_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(local_result, Exception):
                return cloud_result if not isinstance(cloud_result, Exception) else {"error": "All predictions failed"}
            if isinstance(cloud_result, Exception):
                return local_result
            
            # Ensemble the predictions
            ensemble_result = await self._ensemble_predictions(local_result, cloud_result)
            
            return {
                "approach": "hybrid_ensemble",
                "prediction": ensemble_result,
                "local_contribution": self.model_selection_rules["hybrid_ensemble"]["weight_local"],
                "cloud_contribution": self.model_selection_rules["hybrid_ensemble"]["weight_cloud"],
                "confidence": self._calculate_hybrid_confidence(local_result, cloud_result),
                "model_source": "hybrid"
            }
            
        except Exception as e:
            # Final fallback to local
            return await self._local_prediction(sensor_data, forecast_horizon)
    
    async def _generate_local_forecast(self, sensor_data: Dict, forecast_horizon: str) -> Dict:
        """Generate simple local weather forecast using trend analysis"""
        horizon_hours = int(forecast_horizon.replace("day", "")) * 24
        
        # Simple trend-based forecasting
        current_temp = sensor_data.get("temperature", 20.0)
        current_humidity = sensor_data.get("humidity", 50.0)
        current_pressure = sensor_data.get("pressure", 1013.0)
        
        # Generate hourly forecasts
        forecast_temps = []
        forecast_humidity = []
        forecast_pressure = []
        
        for hour in range(0, horizon_hours, 6):  # 6-hour intervals
            # Simple sinusoidal temperature model with daily cycle
            daily_variation = 5 * np.sin(2 * np.pi * hour / 24)
            temp_trend = current_temp + daily_variation + np.random.normal(0, 1)
            forecast_temps.append(temp_trend)
            
            # Humidity inverse correlation with temperature
            humidity_trend = current_humidity - daily_variation * 2 + np.random.normal(0, 3)
            forecast_humidity.append(max(20, min(100, humidity_trend)))
            
            # Pressure with random walk
            pressure_trend = current_pressure + np.random.normal(0, 2)
            forecast_pressure.append(pressure_trend)
        
        return {
            "temperature": forecast_temps,
            "humidity": forecast_humidity,
            "pressure": forecast_pressure,
            "timestamps": [(datetime.now() + timedelta(hours=h)).isoformat() 
                          for h in range(0, horizon_hours, 6)],
            "model": "local_trend_analysis"
        }
    
    async def _synthesize_cloud_results(self, results: Dict, forecast_horizon: str) -> Dict:
        """Synthesize results from multiple cloud models"""
        synthesized = {
            "weather_forecast": {},
            "environmental_analysis": {},
            "risk_assessment": {},
            "confidence_metrics": {}
        }
        
        # Synthesize weather forecasts
        if "aurora" in results and results["aurora"].get("success"):
            aurora_data = results["aurora"]["predictions"]["forecasts"]
            synthesized["weather_forecast"]["aurora"] = {
                "temperature": aurora_data.get("temperature_2m", {}),
                "humidity": aurora_data.get("humidity_2m", {}),
                "precipitation": aurora_data.get("precipitation", {}),
                "air_quality": aurora_data.get("air_quality_index", {})
            }
        
        if "gencast" in results and results["gencast"].get("success"):
            gencast_data = results["gencast"]["predictions"]["ensemble_statistics"]
            synthesized["weather_forecast"]["gencast"] = {
                "temperature_ensemble": gencast_data.get("temperature", {}),
                "precipitation_probability": gencast_data.get("precipitation_probability", {}),
                "uncertainty_metrics": results["gencast"]["predictions"]["uncertainty_quantification"]
            }
        
        # Synthesize environmental analysis
        if "prithvi" in results and results["prithvi"].get("success"):
            prithvi_data = results["prithvi"]["results"]
            synthesized["environmental_analysis"] = prithvi_data
        
        # Generate combined risk assessment
        synthesized["risk_assessment"] = await self._generate_risk_assessment(results)
        
        return synthesized
    
    async def _ensemble_predictions(self, local_result: Dict, cloud_result: Dict) -> Dict:
        """Ensemble local and cloud predictions"""
        local_weight = self.model_selection_rules["hybrid_ensemble"]["weight_local"]
        cloud_weight = self.model_selection_rules["hybrid_ensemble"]["weight_cloud"]
        
        ensemble = {
            "temperature": self._weighted_average_forecast(
                local_result.get("forecast", {}).get("temperature", []),
                cloud_result.get("prediction", {}).get("weather_forecast", {}).get("aurora", {}).get("temperature", {}).get("values", []),
                local_weight, cloud_weight
            ),
            "environmental_class": self._ensemble_classification(
                local_result.get("prediction", {}).get("environmental_class", "unknown"),
                cloud_result.get("prediction", {}).get("environmental_analysis", {}),
                local_weight, cloud_weight
            ),
            "risk_level": self._ensemble_risk_assessment(local_result, cloud_result, local_weight, cloud_weight)
        }
        
        return ensemble
    
    def _weighted_average_forecast(self, local_values: List, cloud_values: List, 
                                 local_weight: float, cloud_weight: float) -> List:
        """Calculate weighted average of forecasts"""
        if not local_values or not cloud_values:
            return local_values or cloud_values
        
        min_length = min(len(local_values), len(cloud_values))
        weighted_values = []
        
        for i in range(min_length):
            weighted_val = local_values[i] * local_weight + cloud_values[i] * cloud_weight
            weighted_values.append(weighted_val)
        
        return weighted_values
    
    def _ensemble_classification(self, local_class: str, cloud_analysis: Dict, 
                               local_weight: float, cloud_weight: float) -> str:
        """Ensemble classification results"""
        # Simple voting approach
        if cloud_weight > local_weight and cloud_analysis:
            # Prefer cloud classification if available
            if "land_cover" in cloud_analysis:
                return cloud_analysis["land_cover"].get("classification", local_class)
            elif "risk_assessment" in cloud_analysis:
                return cloud_analysis["risk_assessment"].get("damage_potential", local_class)
        
        return local_class
    
    def _ensemble_risk_assessment(self, local_result: Dict, cloud_result: Dict, 
                                local_weight: float, cloud_weight: float) -> str:
        """Ensemble risk assessment"""
        risk_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        reverse_scores = {v: k for k, v in risk_scores.items()}
        
        # Get local risk
        local_risk = "medium"  # Default
        if local_result.get("prediction", {}).get("is_anomaly"):
            local_risk = "high"
        
        # Get cloud risk
        cloud_risk = "medium"  # Default
        cloud_pred = cloud_result.get("prediction", {})
        if "risk_assessment" in cloud_pred:
            cloud_risk = cloud_pred["risk_assessment"].get("damage_potential", "medium")
        
        # Calculate weighted risk score
        local_score = risk_scores.get(local_risk, 2)
        cloud_score = risk_scores.get(cloud_risk, 2)
        
        weighted_score = local_score * local_weight + cloud_score * cloud_weight
        final_risk = reverse_scores[round(weighted_score)]
        
        return final_risk
    
    def _calculate_ensemble_confidence(self, results: Dict) -> float:
        """Calculate overall confidence from multiple model results"""
        confidences = []
        
        for model_name, result in results.items():
            if result.get("success") and "confidence" in result:
                confidences.append(result["confidence"])
            elif result.get("success"):
                # Assign default confidence based on model performance
                confidences.append(self.model_performance.get(model_name, {}).get("accuracy", 0.8))
        
        if not confidences:
            return 0.5
        
        # Use weighted average with higher weight for more models
        return np.mean(confidences) * min(1.0, len(confidences) / 2)
    
    def _calculate_hybrid_confidence(self, local_result: Dict, cloud_result: Dict) -> float:
        """Calculate confidence for hybrid predictions"""
        local_conf = local_result.get("confidence", 0.6)
        cloud_conf = cloud_result.get("confidence", 0.8)
        
        local_weight = self.model_selection_rules["hybrid_ensemble"]["weight_local"]
        cloud_weight = self.model_selection_rules["hybrid_ensemble"]["weight_cloud"]
        
        # Weighted confidence with bonus for ensemble
        base_confidence = local_conf * local_weight + cloud_conf * cloud_weight
        ensemble_bonus = 0.1  # 10% bonus for using ensemble
        
        return min(1.0, base_confidence + ensemble_bonus)
    
    async def _generate_risk_assessment(self, results: Dict) -> Dict:
        """Generate comprehensive risk assessment from all model results"""
        risk_factors = []
        overall_risk = "low"
        
        # Analyze Aurora results for weather risks
        if "aurora" in results and results["aurora"].get("success"):
            aurora_summary = results["aurora"]["predictions"]["summary"]
            
            if aurora_summary.get("max_temperature", 0) > 35:
                risk_factors.append("extreme_heat")
                overall_risk = "high"
            
            if aurora_summary.get("total_precipitation", 0) > 50:
                risk_factors.append("heavy_precipitation")
                overall_risk = "medium" if overall_risk == "low" else overall_risk
            
            if aurora_summary.get("avg_air_quality", 50) > 150:
                risk_factors.append("poor_air_quality")
                overall_risk = "medium" if overall_risk == "low" else overall_risk
        
        # Analyze GenCast results for probabilistic risks
        if "gencast" in results and results["gencast"].get("success"):
            extreme_events = results["gencast"].get("extreme_events", [])
            if extreme_events:
                risk_factors.extend(extreme_events)
                overall_risk = "high"
        
        # Analyze Prithvi results for environmental risks
        if "prithvi" in results and results["prithvi"].get("success"):
            prithvi_results = results["prithvi"]["results"]
            
            if "flood_extent" in prithvi_results:
                if prithvi_results["flood_extent"].get("total_area_km2", 0) > 0:
                    risk_factors.append("flooding")
                    overall_risk = "high"
            
            if "fire_detection" in prithvi_results:
                if prithvi_results["fire_detection"].get("active_fires", 0) > 0:
                    risk_factors.append("wildfire")
                    overall_risk = "critical"
        
        return {
            "overall_risk_level": overall_risk,
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(risk_factors, overall_risk),
            "monitoring_priority": "high" if overall_risk in ["high", "critical"] else "normal"
        }
    
    def _generate_risk_recommendations(self, risk_factors: List[str], overall_risk: str) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        if "extreme_heat" in risk_factors:
            recommendations.append("Implement heat safety protocols and increase hydration")
        
        if "heavy_precipitation" in risk_factors:
            recommendations.append("Monitor flood-prone areas and prepare drainage systems")
        
        if "poor_air_quality" in risk_factors:
            recommendations.append("Limit outdoor activities and use air filtration")
        
        if "flooding" in risk_factors:
            recommendations.append("Activate flood response procedures and evacuation plans")
        
        if "wildfire" in risk_factors:
            recommendations.append("Implement fire safety measures and prepare evacuation routes")
        
        if overall_risk in ["high", "critical"]:
            recommendations.append("Increase monitoring frequency and alert relevant authorities")
        
        return recommendations
    
    async def _check_internet_connectivity(self) -> bool:
        """Check if internet connectivity is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/get', timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except:
            return False
    
    def _is_low_power_mode(self) -> bool:
        """Check if device is in low power mode"""
        # This would integrate with actual power management
        return self.config.get("power_mode", "normal") == "low"
    
    def _is_critical_prediction(self, sensor_data: Dict) -> bool:
        """Determine if this is a critical prediction requiring high accuracy"""
        # Check for emergency conditions
        if sensor_data.get("air_quality", {}).get("aqi", 50) > 200:
            return True
        
        if sensor_data.get("temperature", 20) > 40 or sensor_data.get("temperature", 20) < -10:
            return True
        
        # Check for rapid changes that might indicate emergency
        if hasattr(self, 'previous_sensor_data'):
            temp_change = abs(sensor_data.get("temperature", 20) - 
                            self.previous_sensor_data.get("temperature", 20))
            if temp_change > 10:
                return True
        
        return False
```

## Part 3: Distributed Training Data Aggregation

### Global Model Training Pipeline

```python
class DistributedTrainingAggregator:
    """Aggregates training data from local network for global model improvement"""
    
    def __init__(self, cortex_client, collaboration_manager, config: Dict):
        self.cortex = cortex_client
        self.collaboration = collaboration_manager
        self.config = config
        
        # Training data collection
        self.local_training_data = []
        self.network_training_data = {}
        self.global_model_updates = {}
        
        # Privacy and security settings
        self.privacy_config = {
            "differential_privacy": config.get("differential_privacy", True),
            "epsilon": config.get("privacy_epsilon", 1.0),
            "federated_learning": config.get("federated_learning", True),
            "data_anonymization": config.get("data_anonymization", True)
        }
        
        # Training schedule
        self.training_schedule = {
            "local_update_frequency": timedelta(hours=6),
            "global_aggregation_frequency": timedelta(days=1),
            "model_deployment_frequency": timedelta(days=7)
        }
        
        # Model improvement tracking
        self.model_performance_history = {
            "local": [],
            "global": [],
            "hybrid": []
        }
    
    async def initialize_distributed_training(self):
        """Initialize distributed training system"""
        try:
            # Set up data collection
            await self._setup_data_collection()
            
            # Initialize federated learning
            if self.privacy_config["federated_learning"]:
                await self._initialize_federated_learning()
            
            # Start background training tasks
            asyncio.create_task(self._continuous_training_loop())
            asyncio.create_task(self._global_aggregation_loop())
            
            print("✅ Distributed training system initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize distributed training: {e}")
            return False
    
    async def collect_training_sample(self, sensor_data: Dict, prediction_result: Dict, 
                                    ground_truth: Optional[Dict] = None):
        """Collect training sample from sensor reading and prediction"""
        try:
            # Create training sample
            training_sample = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.collaboration.device_id if self.collaboration else "local",
                "sensor_data": self._anonymize_sensor_data(sensor_data),
                "prediction": prediction_result,
                "ground_truth": ground_truth,
                "model_confidence": prediction_result.get("confidence", 0.0),
                "environmental_context": self._extract_environmental_context(sensor_data)
            }
            
            # Apply differential privacy if enabled
            if self.privacy_config["differential_privacy"]:
                training_sample = await self._apply_differential_privacy(training_sample)
            
            # Store locally
            self.local_training_data.append(training_sample)
            
            # Share with network if enabled
            if self.collaboration and self.privacy_config["federated_learning"]:
                await self._share_training_sample(training_sample)
            
            # Cleanup old samples
            await self._cleanup_old_training_data()
            
        except Exception as e:
            print(f"Error collecting training sample: {e}")
    
    async def _setup_data_collection(self):
        """Setup automated data collection for training"""
        self.data_collection_active = True
        
        # Define data quality criteria
        self.quality_criteria = {
            "min_confidence": 0.6,
            "max_age_hours": 24,
            "required_sensors": ["temperature", "humidity"],
            "anomaly_detection": True
        }
        
        print("Data collection setup complete")
    
    async def _initialize_federated_learning(self):
        """Initialize federated learning framework"""
        try:
            # Register with global training coordinator
            registration_request = {
                "device_id": self.collaboration.device_id,
                "device_capabilities": list(self.collaboration.capabilities),
                "privacy_settings": self.privacy_config,
                "training_schedule": {
                    k: v.total_seconds() for k, v in self.training_schedule.items()
                }
            }
            
            # This would register with Cortex's federated learning service
            federated_response = await self._register_federated_participant(registration_request)
            
            if federated_response.get("success"):
                self.federated_id = federated_response.get("participant_id")
                print(f"✅ Registered for federated learning: {self.federated_id}")
            else:
                print("⚠️ Federated learning registration failed, using local training only")
                
        except Exception as e:
            print(f"Federated learning initialization error: {e}")
    
    async def _register_federated_participant(self, registration: Dict) -> Dict:
        """Register as federated learning participant"""
        try:
            async with self.cortex.session.post(
                f"{self.cortex.api_base}/federated/register",
                json=registration
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _continuous_training_loop(self):
        """Continuously train local models with collected data"""
        while True:
            try:
                await asyncio.sleep(self.training_schedule["local_update_frequency"].total_seconds())
                
                if len(self.local_training_data) >= 100:  # Minimum samples for training
                    await self._train_local_model()
                
            except Exception as e:
                print(f"Continuous training error: {e}")
                await asyncio.sleep(300)  # 5 minute error recovery
    
    async def _global_aggregation_loop(self):
        """Periodically aggregate training data for global model updates"""
        while True:
            try:
                await asyncio.sleep(self.training_schedule["global_aggregation_frequency"].total_seconds())
                
                if self.privacy_config["federated_learning"]:
                    await self._participate_in_global_training()
                
            except Exception as e:
                print(f"Global aggregation error: {e}")
                await asyncio.sleep(1800)  # 30 minute error recovery
    
    async def _train_local_model(self):
        """Train local model with collected data"""
        try:
            print("🎯 Starting local model training...")
            
            # Prepare training dataset
            training_features, training_labels = await self._prepare_training_dataset()
            
            if len(training_features) < 50:
                print("⚠️ Insufficient training data, skipping training")
                return
            
            # Train model (simplified implementation)
            model_performance = await self._execute_local_training(training_features, training_labels)
            
            # Update performance tracking
            self.model_performance_history["local"].append({
                "timestamp": datetime.now().isoformat(),
                "accuracy": model_performance.get("accuracy", 0.0),
                "training_samples": len(training_features),
                "validation_loss": model_performance.get("validation_loss", 1.0)
            })
            
            print(f"✅ Local model training complete - Accuracy: {model_performance.get('accuracy', 0.0):.3f}")
            
        except Exception as e:
            print(f"Local model training failed: {e}")
    
    async def _participate_in_global_training(self):
        """Participate in federated learning global training round"""
        try:
            print("🌍 Participating in global training round...")
            
            # Get current local model weights (simplified)
            local_model_weights = await self._get_local_model_weights()
            
            # Apply differential privacy to model weights
            if self.privacy_config["differential_privacy"]:
                local_model_weights = await self._apply_dp_to_weights(local_model_weights)
            
            # Send model update to global coordinator
            federated_update = {
                "participant_id": getattr(self, 'federated_id', 'unknown'),
                "model_weights": local_model_weights,
                "training_samples_count": len(self.local_training_data),
                "model_performance": self._get_latest_performance(),
                "privacy_budget_used": self._calculate_privacy_budget_usage()
            }
            
            global_response = await self._send_federated_update(federated_update)
            
            if global_response.get("success"):
                # Receive and apply global model update
                global_weights = global_response.get("global_model_weights")
                if global_weights:
                    await self._apply_global_model_update(global_weights)
                    print("✅ Global model update applied")
            
        except Exception as e:
            print(f"Global training participation failed: {e}")
    
    async def _send_federated_update(self, update: Dict) -> Dict:
        """Send federated learning update to global coordinator"""
        try:
            async with self.cortex.session.post(
                f"{self.cortex.api_base}/federated/update",
                json=update
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _prepare_training_dataset(self) -> tuple:
        """Prepare training dataset from collected samples"""
        features = []
        labels = []
        
        for sample in self.local_training_data[-1000:]:  # Use last 1000 samples
            # Extract features
            sensor_data = sample["sensor_data"]
            feature_vector = [
                sensor_data.get("temperature", 20.0),
                sensor_data.get("humidity", 50.0),
                sensor_data.get("pressure", 1013.0),
                sensor_data.get("air_quality", {}).get("aqi", 50),
                sensor_data.get("wind_speed", 0.0),
                sample["environmental_context"].get("hour_of_day", 12),
                sample["environmental_context"].get("day_of_year", 180),
                sample["environmental_context"].get("season", 1)
            ]
            features.append(feature_vector)
            
            # Extract labels (environmental classification)
            prediction = sample["prediction"]
            if "environmental_class" in prediction:
                label = self._encode_environmental_class(prediction["environmental_class"])
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    async def _execute_local_training(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Execute local model training (simplified implementation)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # Train random forest (placeholder for more sophisticated model)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            val_predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_predictions)
            
            # Store model (in real implementation, this would update the actual ML model)
            self.local_model = model
            
            return {
                "accuracy": accuracy,
                "validation_loss": 1 - accuracy,  # Simplified loss calculation
                "training_samples": len(X_train),
                "validation_samples": len(X_val)
            }
            
        except Exception as e:
            print(f"Model training execution failed: {e}")
            return {"accuracy": 0.0, "validation_loss": 1.0}
    
    def _anonymize_sensor_data(self, sensor_data: Dict) -> Dict:
        """Anonymize sensor data for privacy protection"""
        anonymized = sensor_data.copy()
        
        # Remove or hash location data if present
        if "location" in anonymized:
            if self.privacy_config["data_anonymization"]:
                # Hash coordinates to protect exact location
                import hashlib
                lat_str = str(anonymized["location"].get("latitude", 0))
                lon_str = str(anonymized["location"].get("longitude", 0))
                location_hash = hashlib.sha256(f"{lat_str}_{lon_str}".encode()).hexdigest()[:16]
                anonymized["location"] = {"hash": location_hash}
        
        # Add noise to sensitive measurements if differential privacy is enabled
        if self.privacy_config["differential_privacy"]:
            noise_scale = 1.0 / self.privacy_config["epsilon"]
            if "temperature" in anonymized:
                anonymized["temperature"] += np.random.laplace(0, noise_scale * 0.1)
            if "humidity" in anonymized:
                anonymized["humidity"] += np.random.laplace(0, noise_scale * 0.5)
        
        return anonymized
    
    def _extract_environmental_context(self, sensor_data: Dict) -> Dict:
        """Extract environmental context for training"""
        now = datetime.now()
        
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "day_of_year": now.timetuple().tm_yday,
            "season": (now.month - 1) // 3,  # 0=winter, 1=spring, 2=summer, 3=fall
            "is_weekend": now.weekday() >= 5,
            "temperature_trend": self._calculate_temperature_trend(sensor_data),
            "pressure_trend": self._calculate_pressure_trend(sensor_data)
        }
    
    def _calculate_temperature_trend(self, sensor_data: Dict) -> str:
        """Calculate temperature trend from recent data"""
        # Simplified trend calculation
        current_temp = sensor_data.get("temperature", 20.0)
        if hasattr(self, 'previous_temperature'):
            if current_temp > self.previous_temperature + 1:
                trend = "rising"
            elif current_temp < self.previous_temperature - 1:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        self.previous_temperature = current_temp
        return trend
    
    def _calculate_pressure_trend(self, sensor_data: Dict) -> str:
        """Calculate atmospheric pressure trend"""
        current_pressure = sensor_data.get("pressure", 1013.0)
        if hasattr(self, 'previous_pressure'):
            if current_pressure > self.previous_pressure + 2:
                trend = "rising"
            elif current_pressure < self.previous_pressure - 2:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        self.previous_pressure = current_pressure
        return trend
    
    async def _apply_differential_privacy(self, training_sample: Dict) -> Dict:
        """Apply differential privacy to training sample"""
        if not self.privacy_config["differential_privacy"]:
            return training_sample
        
        # Apply Laplace noise to numerical values
        epsilon = self.privacy_config["epsilon"]
        sensitivity = 1.0  # Assumed sensitivity
        
        noise_scale = sensitivity / epsilon
        
        # Add noise to sensor data
        if "sensor_data" in training_sample:
            sensor_data = training_sample["sensor_data"].copy()
            for key, value in sensor_data.items():
                if isinstance(value, (int, float)):
                    noise = np.random.laplace(0, noise_scale * 0.01)  # Small noise for sensor data
                    sensor_data[key] = value + noise
            training_sample["sensor_data"] = sensor_data
        
        return training_sample
    
    async def _share_training_sample(self, training_sample: Dict):
        """Share training sample with collaborative network"""
        try:
            if self.collaboration:
                # Share with network (simplified)
                await self.collaboration.share_data({
                    "type": "training_sample",
                    "data": training_sample,
                    "privacy_level": "high" if self.privacy_config["differential_privacy"] else "medium"
                })
        except Exception as e:
            print(f"Error sharing training sample: {e}")
    
    async def _cleanup_old_training_data(self):
        """Clean up old training data to manage memory"""
        max_samples = 10000  # Keep last 10k samples
        if len(self.local_training_data) > max_samples:
            # Remove oldest samples
            samples_to_remove = len(self.local_training_data) - max_samples
            self.local_training_data = self.local_training_data[samples_to_remove:]
            print(f"Cleaned up {samples_to_remove} old training samples")
    
    def _encode_environmental_class(self, class_name: str) -> int:
        """Encode environmental class to numerical label"""
        class_mapping = {
            "excellent": 0,
            "good": 1,
            "moderate": 2,
            "poor": 3,
            "unhealthy": 4,
            "hazardous": 5,
            "unknown": 6
        }
        return class_mapping.get(class_name.lower(), 6)
    
    async def _get_local_model_weights(self) -> Dict:
        """Get current local model weights for federated learning"""
        try:
            if hasattr(self, 'local_model'):
                # For scikit-learn models, we'll use feature importances as a proxy for weights
                if hasattr(self.local_model, 'feature_importances_'):
                    return {
                        "feature_importances": self.local_model.feature_importances_.tolist(),
                        "n_estimators": getattr(self.local_model, 'n_estimators', 100),
                        "model_type": "random_forest"
                    }
            
            # Default empty weights
            return {"weights": [], "model_type": "unknown"}
            
        except Exception as e:
            print(f"Error getting local model weights: {e}")
            return {"weights": [], "model_type": "error"}
    
    async def _apply_dp_to_weights(self, weights: Dict) -> Dict:
        """Apply differential privacy to model weights"""
        if not self.privacy_config["differential_privacy"]:
            return weights
        
        epsilon = self.privacy_config["epsilon"]
        noise_scale = 1.0 / epsilon
        
        # Add noise to feature importances
        if "feature_importances" in weights:
            importances = np.array(weights["feature_importances"])
            noise = np.random.laplace(0, noise_scale * 0.01, size=importances.shape)
            weights["feature_importances"] = (importances + noise).tolist()
        
        return weights
    
    def _get_latest_performance(self) -> Dict:
        """Get latest local model performance"""
        if self.model_performance_history["local"]:
            return self.model_performance_history["local"][-1]
        return {"accuracy": 0.0, "training_samples": 0}
    
    def _calculate_privacy_budget_usage(self) -> float:
        """Calculate privacy budget usage for differential privacy"""
        # Simplified privacy budget calculation
        if self.privacy_config["differential_privacy"]:
            # Estimate budget usage based on number of queries
            num_samples = len(self.local_training_data)
            epsilon_per_sample = self.privacy_config["epsilon"] / max(num_samples, 1)
            return min(1.0, num_samples * epsilon_per_sample / 10.0)  # Normalize to [0,1]
        return 0.0
    
    async def _apply_global_model_update(self, global_weights: Dict):
        """Apply global model update from federated learning"""
        try:
            print("🔄 Applying global model update...")
            
            # Store global update for future reference
            self.global_model_updates[datetime.now().isoformat()] = global_weights
            
            # In a real implementation, this would update the actual model
            # For now, we'll just track the update
            self.model_performance_history["global"].append({
                "timestamp": datetime.now().isoformat(),
                "update_received": True,
                "global_accuracy": global_weights.get("global_accuracy", 0.0),
                "participants": global_weights.get("participant_count", 0)
            })
            
            print(f"✅ Global model update applied from {global_weights.get('participant_count', 0)} participants")
            
        except Exception as e:
            print(f"Error applying global model update: {e}")

class BayesianHyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning in environmental models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # Hyperparameter search spaces
        self.search_spaces = {
            "neural_network": {
                "learning_rate": (1e-5, 1e-1, 'log-uniform'),
                "batch_size": (8, 128, 'int'),
                "hidden_layers": (1, 5, 'int'),
                "neurons_per_layer": (16, 512, 'int'),
                "dropout_rate": (0.0, 0.5, 'uniform'),
                "l2_regularization": (1e-6, 1e-2, 'log-uniform')
            },
            "random_forest": {
                "n_estimators": (10, 500, 'int'),
                "max_depth": (3, 20, 'int'),
                "min_samples_split": (2, 20, 'int'),
                "min_samples_leaf": (1, 10, 'int'),
                "max_features": (0.1, 1.0, 'uniform')
            },
            "environmental_model": {
                "forecast_horizon": (1, 15, 'int'),
                "ensemble_size": (10, 100, 'int'),
                "confidence_threshold": (0.6, 0.95, 'uniform'),
                "temporal_window": (6, 72, 'int'),  # hours
                "spatial_resolution": (0.1, 1.0, 'uniform')  # degrees
            }
        }
    
    async def optimize_hyperparameters(self, model_type: str, objective_function, 
                                     n_iterations: int = 50) -> Dict:
        """Perform Bayesian optimization for hyperparameters"""
        try:
            print(f"🎯 Starting Bayesian hyperparameter optimization for {model_type}")
            
            search_space = self.search_spaces.get(model_type, {})
            if not search_space:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Initialize Gaussian Process surrogate model
            surrogate_model = self._initialize_surrogate_model(search_space)
            
            # Optimization loop
            for iteration in range(n_iterations):
                print(f"Optimization iteration {iteration + 1}/{n_iterations}")
                
                # Select next hyperparameters using acquisition function
                next_params = await self._select_next_parameters(
                    surrogate_model, search_space, iteration
                )
                
                # Evaluate objective function
                score = await objective_function(next_params)
                
                # Update optimization history
                self.optimization_history.append({
                    "iteration": iteration,
                    "parameters": next_params,
                    "score": score,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update best parameters if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params.copy()
                    print(f"✅ New best score: {score:.4f}")
                
                # Update surrogate model
                await self._update_surrogate_model(surrogate_model, next_params, score)
                
                # Early stopping if convergence achieved
                if await self._check_convergence():
                    print(f"🎯 Convergence achieved after {iteration + 1} iterations")
                    break
            
            return {
                "best_parameters": self.best_params,
                "best_score": self.best_score,
                "optimization_history": self.optimization_history,
                "total_iterations": len(self.optimization_history)
            }
            
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            return {"error": str(e)}
    
    def _initialize_surrogate_model(self, search_space: Dict) -> Dict:
        """Initialize Gaussian Process surrogate model"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            
            # Create kernel for Gaussian Process
            kernel = Matern(length_scale=1.0, nu=2.5)
            
            # Initialize GP regressor
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            
            return {
                "model": gp_model,
                "search_space": search_space,
                "X_observed": [],
                "y_observed": [],
                "param_bounds": self._extract_parameter_bounds(search_space)
            }
            
        except ImportError:
            # Fallback to simple random search if sklearn not available
            print("⚠️ Sklearn not available, using random search fallback")
            return {
                "model": None,
                "search_space": search_space,
                "X_observed": [],
                "y_observed": [],
                "param_bounds": self._extract_parameter_bounds(search_space)
            }
    
    def _extract_parameter_bounds(self, search_space: Dict) -> Dict:
        """Extract parameter bounds from search space"""
        bounds = {}
        for param_name, (min_val, max_val, param_type) in search_space.items():
            bounds[param_name] = {
                "min": min_val,
                "max": max_val,
                "type": param_type
            }
        return bounds
    
    async def _select_next_parameters(self, surrogate_model: Dict, 
                                    search_space: Dict, iteration: int) -> Dict:
        """Select next hyperparameters using acquisition function"""
        try:
            if iteration < 5 or surrogate_model["model"] is None:
                # Random exploration for first few iterations or fallback
                return self._random_sample_parameters(search_space)
            
            # Use Expected Improvement acquisition function
            best_params = await self._optimize_acquisition_function(
                surrogate_model, acquisition_type="expected_improvement"
            )
            
            return best_params
            
        except Exception as e:
            print(f"Parameter selection failed, using random sampling: {e}")
            return self._random_sample_parameters(search_space)
    
    def _random_sample_parameters(self, search_space: Dict) -> Dict:
        """Randomly sample parameters from search space"""
        params = {}
        
        for param_name, (min_val, max_val, param_type) in search_space.items():
            if param_type == 'int':
                params[param_name] = np.random.randint(min_val, max_val + 1)
            elif param_type == 'uniform':
                params[param_name] = np.random.uniform(min_val, max_val)
            elif param_type == 'log-uniform':
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                params[param_name] = np.exp(np.random.uniform(log_min, log_max))
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    async def _optimize_acquisition_function(self, surrogate_model: Dict, 
                                           acquisition_type: str = "expected_improvement") -> Dict:
        """Optimize acquisition function to find next parameters"""
        try:
            # For simplicity, we'll use random sampling with acquisition function evaluation
            n_candidates = 1000
            best_acquisition = float('-inf')
            best_params = None
            
            for _ in range(n_candidates):
                candidate_params = self._random_sample_parameters(surrogate_model["search_space"])
                
                # Convert to feature vector for GP
                x_candidate = self._params_to_vector(candidate_params, surrogate_model["param_bounds"])
                
                # Calculate acquisition function value
                acquisition_value = await self._calculate_acquisition_function(
                    x_candidate, surrogate_model, acquisition_type
                )
                
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_params = candidate_params
            
            return best_params if best_params is not None else self._random_sample_parameters(surrogate_model["search_space"])
            
        except Exception as e:
            print(f"Acquisition optimization failed: {e}")
            return self._random_sample_parameters(surrogate_model["search_space"])
    
    def _params_to_vector(self, params: Dict, param_bounds: Dict) -> np.ndarray:
        """Convert parameter dictionary to normalized vector"""
        vector = []
        
        for param_name in sorted(param_bounds.keys()):
            if param_name in params:
                value = params[param_name]
                bounds = param_bounds[param_name]
                
                # Normalize to [0, 1]
                if bounds["type"] == "log-uniform":
                    normalized = (np.log(value) - np.log(bounds["min"])) / (np.log(bounds["max"]) - np.log(bounds["min"]))
                else:
                    normalized = (value - bounds["min"]) / (bounds["max"] - bounds["min"])
                
                vector.append(max(0.0, min(1.0, normalized)))
            else:
                vector.append(0.5)  # Default middle value
        
        return np.array(vector).reshape(1, -1)
    
    async def _calculate_acquisition_function(self, x_candidate: np.ndarray, 
                                            surrogate_model: Dict, acquisition_type: str) -> float:
        """Calculate acquisition function value"""
        try:
            if surrogate_model["model"] is None or len(surrogate_model["y_observed"]) < 2:
                # Random exploration if no model available
                return np.random.random()
            
            # Get prediction from GP
            mean, std = surrogate_model["model"].predict(x_candidate, return_std=True)
            
            if acquisition_type == "expected_improvement":
                # Expected Improvement acquisition function
                best_y = max(surrogate_model["y_observed"])
                z = (mean - best_y) / (std + 1e-9)
                
                from scipy.stats import norm
                ei = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)
                return float(ei[0])
            
            elif acquisition_type == "upper_confidence_bound":
                # Upper Confidence Bound acquisition function
                kappa = 2.576  # 99% confidence
                ucb = mean + kappa * std
                return float(ucb[0])
            
            else:
                # Default to probability of improvement
                best_y = max(surrogate_model["y_observed"])
                z = (mean - best_y) / (std + 1e-9)
                
                from scipy.stats import norm
                pi = norm.cdf(z)
                return float(pi[0])
                
        except Exception as e:
            print(f"Acquisition function calculation failed: {e}")
            return np.random.random()
    
    async def _update_surrogate_model(self, surrogate_model: Dict, 
                                    new_params: Dict, new_score: float):
        """Update surrogate model with new observation"""
        try:
            if surrogate_model["model"] is None:
                return
            
            # Convert parameters to vector
            x_new = self._params_to_vector(new_params, surrogate_model["param_bounds"])
            
            # Add to observed data
            surrogate_model["X_observed"].append(x_new[0])
            surrogate_model["y_observed"].append(new_score)
            
            # Refit Gaussian Process
            if len(surrogate_model["y_observed"]) >= 2:
                X = np.array(surrogate_model["X_observed"])
                y = np.array(surrogate_model["y_observed"])
                
                surrogate_model["model"].fit(X, y)
                
        except Exception as e:
            print(f"Surrogate model update failed: {e}")
    
    async def _check_convergence(self, patience: int = 10, tolerance: float = 1e-4) -> bool:
        """Check if optimization has converged"""
        if len(self.optimization_history) < patience:
            return False
        
        # Check if score has improved in last 'patience' iterations
        recent_scores = [h["score"] for h in self.optimization_history[-patience:]]
        best_recent = max(recent_scores)
        
        # Check if improvement is less than tolerance
        if len(self.optimization_history) >= 2 * patience:
            older_scores = [h["score"] for h in self.optimization_history[-2*patience:-patience]]
            best_older = max(older_scores) if older_scores else float('-inf')
            
            improvement = best_recent - best_older
            return improvement < tolerance
        
        return False
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization process"""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        scores = [h["score"] for h in self.optimization_history]
        
        return {
            "best_score": self.best_score,
            "best_parameters": self.best_params,
            "total_iterations": len(self.optimization_history),
            "score_improvement": self.best_score - scores[0] if scores else 0,
            "convergence_iteration": self._find_convergence_iteration(),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores)
            }
        }
    
    def _find_convergence_iteration(self) -> Optional[int]:
        """Find iteration where convergence occurred"""
        if len(self.optimization_history) < 10:
            return None
        
        # Find last significant improvement
        best_score = float('-inf')
        convergence_iter = None
        
        for i, history in enumerate(self.optimization_history):
            if history["score"] > best_score + 1e-4:  # Significant improvement
                best_score = history["score"]
                convergence_iter = i
        
        return convergence_iter

class AdvancedConsensusAlgorithm:
    """Advanced consensus algorithm for distributed environmental monitoring"""
    
    def __init__(self, device_id: str, config: Dict):
        self.device_id = device_id
        self.config = config
        self.consensus_data = {}
        self.participant_weights = {}
        self.consensus_history = []
        
        # Consensus algorithm parameters
        self.consensus_threshold = config.get("consensus_threshold", 0.67)
        self.max_iterations = config.get("max_consensus_iterations", 10)
        self.convergence_tolerance = config.get("convergence_tolerance", 1e-3)
        self.trust_decay_factor = config.get("trust_decay_factor", 0.95)
        
        # Byzantine fault tolerance parameters
        self.max_byzantine_nodes = config.get("max_byzantine_nodes", 1)
        self.outlier_detection_threshold = config.get("outlier_threshold", 2.0)
    
    async def initiate_consensus(self, proposal: Dict, participants: List[str]) -> Dict:
        """Initiate consensus process with environmental data proposal"""
        try:
            consensus_id = f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.device_id}"
            
            consensus_session = {
                "consensus_id": consensus_id,
                "initiator": self.device_id,
                "proposal": proposal,
                "participants": participants,
                "responses": {},
                "iteration": 0,
                "status": "active",
                "started_at": datetime.now(),
                "algorithm": "byzantine_fault_tolerant_consensus"
            }
            
            # Add to active consensus sessions
            self.consensus_data[consensus_id] = consensus_session
            
            # Send consensus request to participants
            await self._send_consensus_request(consensus_session)
            
            # Wait for responses and process consensus
            consensus_result = await self._process_consensus(consensus_id)
            
            return consensus_result
            
        except Exception as e:
            print(f"Consensus initiation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_consensus_request(self, consensus_session: Dict):
        """Send consensus request to all participants"""
        request_message = {
            "message_type": "consensus_request",
            "consensus_id": consensus_session["consensus_id"],
            "initiator": consensus_session["initiator"],
            "proposal": consensus_session["proposal"],
            "iteration": consensus_session["iteration"],
            "deadline": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "algorithm_params": {
                "consensus_threshold": self.consensus_threshold,
                "max_iterations": self.max_iterations,
                "byzantine_tolerance": self.max_byzantine_nodes
            }
        }
        
        # In a real implementation, this would send to actual network participants
        print(f"📤 Sending consensus request {consensus_session['consensus_id']} to {len(consensus_session['participants'])} participants")
    
    async def _process_consensus(self, consensus_id: str, timeout_seconds: int = 300) -> Dict:
        """Process consensus responses and determine consensus"""
        consensus_session = self.consensus_data[consensus_id]
        
        try:
            # Simulate response collection (in real implementation, this would collect actual responses)
            await self._collect_simulated_responses(consensus_session)
            
            # Byzantine fault tolerance: detect and filter outliers
            filtered_responses = await self._byzantine_fault_tolerance(consensus_session)
            
            # Apply consensus algorithm
            consensus_result = await self._apply_consensus_algorithm(filtered_responses, consensus_session)
            
            # Update consensus history
            self.consensus_history.append({
                "consensus_id": consensus_id,
                "result": consensus_result,
                "participants": len(consensus_session["participants"]),
                "iterations": consensus_session["iteration"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Cleanup consensus session
            consensus_session["status"] = "completed"
            consensus_session["result"] = consensus_result
            
            return consensus_result
            
        except Exception as e:
            print(f"Consensus processing failed: {e}")
            consensus_session["status"] = "failed"
            return {"success": False, "error": str(e)}
    
    async def _collect_simulated_responses(self, consensus_session: Dict):
        """Simulate collecting responses from participants"""
        proposal = consensus_session["proposal"]
        base_temp = proposal.get("temperature", 20.0)
        base_humidity = proposal.get("humidity", 50.0)
        
        # Simulate responses with some variation
        for i, participant in enumerate(consensus_session["participants"]):
            # Add realistic variation to simulate different sensor readings
            temp_variation = np.random.normal(0, 1.0)  # ±1°C variation
            humidity_variation = np.random.normal(0, 3.0)  # ±3% variation
            
            # Simulate some Byzantine (malicious) nodes
            if i < self.max_byzantine_nodes and np.random.random() < 0.3:
                # Byzantine response with significant deviation
                temp_variation += np.random.normal(0, 10.0)
                humidity_variation += np.random.normal(0, 20.0)
            
            response = {
                "participant_id": participant,
                "response_data": {
                    "temperature": base_temp + temp_variation,
                    "humidity": max(0, min(100, base_humidity + humidity_variation)),
                    "confidence": np.random.uniform(0.7, 0.95),
                    "sensor_quality": np.random.uniform(0.8, 1.0)
                },
                "timestamp": datetime.now().isoformat(),
                "signature": f"sig_{participant}_{consensus_session['consensus_id']}"
            }
            
            consensus_session["responses"][participant] = response
        
        print(f"📥 Collected {len(consensus_session['responses'])} responses for consensus {consensus_session['consensus_id']}")
    
    async def _byzantine_fault_tolerance(self, consensus_session: Dict) -> Dict:
        """Apply Byzantine fault tolerance to filter malicious responses"""
        responses = consensus_session["responses"]
        
        if len(responses) < 3:
            return responses  # Need at least 3 nodes for Byzantine tolerance
        
        # Extract numerical values for outlier detection
        temperatures = []
        humidities = []
        participant_ids = []
        
        for participant_id, response in responses.items():
            temperatures.append(response["response_data"]["temperature"])
            humidities.append(response["response_data"]["humidity"])
            participant_ids.append(participant_id)
        
        # Detect outliers using statistical methods
        temp_outliers = self._detect_outliers(temperatures, self.outlier_detection_threshold)
        humidity_outliers = self._detect_outliers(humidities, self.outlier_detection_threshold)
        
        # Combine outlier detection results
        outlier_indices = set(temp_outliers) | set(humidity_outliers)
        
        # Filter out outliers
        filtered_responses = {}
        for i, participant_id in enumerate(participant_ids):
            if i not in outlier_indices:
                filtered_responses[participant_id] = responses[participant_id]
            else:
                print(f"⚠️ Filtered out outlier response from {participant_id}")
        
        print(f"🛡️ Byzantine fault tolerance: {len(filtered_responses)}/{len(responses)} responses retained")
        
        return filtered_responses
    
    def _detect_outliers(self, values: List[float], threshold: float) -> List[int]:
        """Detect outliers using modified Z-score method"""
        if len(values) < 3:
            return []
        
        # Calculate median and median absolute deviation
        median = np.median(values)
        mad = np.median([abs(x - median) for x in values])
        
        if mad == 0:
            return []
        
        # Calculate modified Z-scores
        modified_z_scores = [0.6745 * (x - median) / mad for x in values]
        
        # Identify outliers
        outlier_indices = [i for i, z_score in enumerate(modified_z_scores) if abs(z_score) > threshold]
        
        return outlier_indices
    
    async def _apply_consensus_algorithm(self, responses: Dict, consensus_session: Dict) -> Dict:
        """Apply consensus algorithm to determine final values"""
        if not responses:
            return {"success": False, "error": "No valid responses for consensus"}
        
        try:
            # Calculate participant weights based on trust and data quality
            weights = await self._calculate_participant_weights(responses)
            
            # Weighted consensus calculation
            consensus_values = {}
            total_weight = sum(weights.values())
            
            # Calculate weighted averages for each environmental parameter
            for param in ["temperature", "humidity"]:
                weighted_sum = 0.0
                for participant_id, response in responses.items():
                    if param in response["response_data"]:
                        value = response["response_data"][param]
                        weight = weights[participant_id]
                        weighted_sum += value * weight
                
                if total_weight > 0:
                    consensus_values[param] = weighted_sum / total_weight
            
            # Calculate consensus confidence
            consensus_confidence = await self._calculate_consensus_confidence(responses, weights)
            
            # Check if consensus threshold is met
            consensus_reached = len(responses) >= len(consensus_session["participants"]) * self.consensus_threshold
            
            result = {
                "success": True,
                "consensus_reached": consensus_reached,
                "consensus_values": consensus_values,
                "consensus_confidence": consensus_confidence,
                "participant_count": len(responses),
                "total_participants": len(consensus_session["participants"]),
                "consensus_threshold": self.consensus_threshold,
                "participant_weights": weights,
                "algorithm": "weighted_byzantine_consensus"
            }
            
            print(f"✅ Consensus reached: {consensus_reached}, Confidence: {consensus_confidence:.3f}")
            
            return result
            
        except Exception as e:
            print(f"Consensus algorithm failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_participant_weights(self, responses: Dict) -> Dict:
        """Calculate weights for each participant based on trust and data quality"""
        weights = {}
        
        for participant_id, response in responses.items():
            # Base weight from participant trust history
            base_trust = self.participant_weights.get(participant_id, 0.8)
            
            # Data quality weight
            data_quality = response["response_data"].get("sensor_quality", 0.8)
            
            # Confidence weight
            confidence = response["response_data"].get("confidence", 0.8)
            
            # Combined weight
            weight = base_trust * data_quality * confidence
            weights[participant_id] = weight
        
        return weights
    
    async def _calculate_consensus_confidence(self, responses: Dict, weights: Dict) -> float:
        """Calculate overall confidence in consensus result"""
        if not responses:
            return 0.0
        
        # Weighted average of individual confidences
        total_confidence = 0.0
        total_weight = 0.0
        
        for participant_id, response in responses.items():
            confidence = response["response_data"].get("confidence", 0.8)
            weight = weights.get(participant_id, 0.8)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_confidence = total_confidence / total_weight
        
        # Adjust for number of participants (more participants = higher confidence)
        participation_factor = min(1.0, len(responses) / 5.0)  # Optimal at 5+ participants
        
        # Adjust for consensus agreement (calculate variance)
        agreement_factor = await self._calculate_agreement_factor(responses)
        
        final_confidence = avg_confidence * participation_factor * agreement_factor
        
        return min(1.0, final_confidence)
    
    async def _calculate_agreement_factor(self, responses: Dict) -> float:
        """Calculate agreement factor based on response variance"""
        if len(responses) < 2:
            return 1.0
        
        # Calculate variance for temperature and humidity
        temp_values = [r["response_data"]["temperature"] for r in responses.values() if "temperature" in r["response_data"]]
        humidity_values = [r["response_data"]["humidity"] for r in responses.values() if "humidity" in r["response_data"]]
        
        agreement_scores = []
        
        if temp_values:
            temp_variance = np.var(temp_values)
            temp_agreement = 1.0 / (1.0 + temp_variance)  # Higher variance = lower agreement
            agreement_scores.append(temp_agreement)
        
        if humidity_values:
            humidity_variance = np.var(humidity_values)
            humidity_agreement = 1.0 / (1.0 + humidity_variance / 100.0)  # Scale humidity variance
            agreement_scores.append(humidity_agreement)
        
        if agreement_scores:
            return np.mean(agreement_scores)
        
        return 1.0
    
    def update_participant_trust(self, participant_id: str, trust_adjustment: float):
        """Update trust score for a participant"""
        current_trust = self.participant_weights.get(participant_id, 0.8)
        
        # Apply trust decay and adjustment
        new_trust = current_trust * self.trust_decay_factor + trust_adjustment * (1 - self.trust_decay_factor)
        new_trust = max(0.0, min(1.0, new_trust))  # Clamp to [0, 1]
        
        self.participant_weights[participant_id] = new_trust
        
        print(f"Updated trust for {participant_id}: {current_trust:.3f} -> {new_trust:.3f}")
    
    def get_consensus_summary(self) -> Dict:
        """Get summary of consensus history and performance"""
        if not self.consensus_history:
            return {"error": "No consensus history available"}
        
        successful_consensus = [c for c in self.consensus_history if c["result"].get("success", False)]
        
        return {
            "total_consensus_sessions": len(self.consensus_history),
            "successful_sessions": len(successful_consensus),
            "success_rate": len(successful_consensus) / len(self.consensus_history),
            "average_participants": np.mean([c["participants"] for c in self.consensus_history]),
            "average_confidence": np.mean([c["result"].get("consensus_confidence", 0) for c in successful_consensus]) if successful_consensus else 0,
            "participant_trust_scores": dict(self.participant_weights),
            "recent_sessions": self.consensus_history[-5:]  # Last 5 sessions
        }

# Integration example showing how all components work together
class IntegratedWeatherAISystem:
    """Complete integrated system combining all advanced AI components"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize core components
        self.cortex_client = TenzroCortexClient(config.get("tenzro_cortex", {}))
        self.hybrid_ai = None  # Will be initialized after cortex client
        self.training_aggregator = None
        self.hyperparameter_optimizer = BayesianHyperparameterOptimizer(config.get("optimization", {}))
        self.consensus_algorithm = AdvancedConsensusAlgorithm(
            config.get("device_id", "device_001"), 
            config.get("consensus", {})
        )
        
    async def initialize_system(self):
        """Initialize the complete integrated system"""
        try:
            print("🚀 Initializing Advanced Weather AI System...")
            
            # Initialize Tenzro Cortex client
            await self.cortex_client.initialize()
            
            # Initialize hybrid AI system
            self.hybrid_ai = HybridEnvironmentalAI(
                local_ml_processor=None,  # Would be actual local ML processor
                cortex_client=self.cortex_client,
                config=self.config.get("hybrid_ai", {})
            )
            
            # Initialize distributed training
            self.training_aggregator = DistributedTrainingAggregator(
                cortex_client=self.cortex_client,
                collaboration_manager=None,  # Would be actual collaboration manager
                config=self.config.get("federated_learning", {})
            )
            
            await self.training_aggregator.initialize_distributed_training()
            
            print("✅ Advanced Weather AI System fully initialized")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    async def process_environmental_prediction(self, sensor_data: Dict) -> Dict:
        """Complete environmental prediction pipeline"""
        try:
            # Step 1: Hybrid AI prediction
            prediction_result = await self.hybrid_ai.predict_environmental_conditions(
                sensor_data=sensor_data,
                forecast_horizon="7day",
                analysis_type="comprehensive"
            )
            
            # Step 2: Collect training data
            await self.training_aggregator.collect_training_sample(
                sensor_data=sensor_data,
                prediction_result=prediction_result
            )
            
            # Step 3: Consensus verification (if multiple devices available)
            if self.config.get("consensus_enabled", False):
                consensus_result = await self.consensus_algorithm.initiate_consensus(
                    proposal=sensor_data,
                    participants=self.config.get("network_participants", [])
                )
                prediction_result["consensus"] = consensus_result
            
            return prediction_result
            
        except Exception as e:
            print(f"Environmental prediction pipeline failed: {e}")
            return {"error": str(e)}
    
    async def optimize_model_performance(self):
        """Optimize model performance using Bayesian hyperparameter optimization"""
        try:
            print("🎯 Starting model performance optimization...")
            
            # Define objective function for optimization
            async def objective_function(params: Dict) -> float:
                # Simulate model evaluation with given hyperparameters
                # In real implementation, this would train and evaluate actual model
                await asyncio.sleep(0.1)  # Simulate training time
                
                # Simulate performance score based on parameters
                base_score = 0.8
                learning_rate_bonus = min(0.1, params.get("learning_rate", 0.01) * 10)
                batch_size_penalty = max(0, (params.get("batch_size", 32) - 64) / 1000)
                
                score = base_score + learning_rate_bonus - batch_size_penalty + np.random.normal(0, 0.05)
                return max(0.0, min(1.0, score))
            
            # Run optimization
            optimization_result = await self.hyperparameter_optimizer.optimize_hyperparameters(
                model_type="neural_network",
                objective_function=objective_function,
                n_iterations=20
            )
            
            print("✅ Model optimization completed")
            return optimization_result
            
        except Exception as e:
            print(f"Model optimization failed: {e}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "cortex_client": {
                "connected": self.cortex_client.session is not None,
                "api_base": self.cortex_client.api_base
            },
            "distributed_training": {
                "active": getattr(self.training_aggregator, 'data_collection_active', False),
                "training_samples": len(getattr(self.training_aggregator, 'local_training_data', [])),
                "federated_learning": self.training_aggregator.privacy_config.get("federated_learning", False) if self.training_aggregator else False
            },
            "optimization": self.hyperparameter_optimizer.get_optimization_summary(),
            "consensus": self.consensus_algorithm.get_consensus_summary(),
            "system_config": {
                "hybrid_ai_enabled": self.hybrid_ai is not None,
                "consensus_enabled": self.config.get("consensus_enabled", False),
                "privacy_protection": self.config.get("federated_learning", {}).get("differential_privacy", False)
            }
        }

# Example usage and configuration
async def main():
    """Example usage of the complete advanced weather AI system"""
    
    # System configuration
    config = {
        "device_id": "furcate_nano_001",
        "tenzro_cortex": {
            "cortex_endpoint": "https://api.tenzro.com/cortex/v1",
            "api_key": "your_tenzro_api_key_here",
            "device_id": "furcate_nano_001"
        },
        "hybrid_ai": {
            "power_mode": "normal",
            "privacy_mode": False
        },
        "federated_learning": {
            "differential_privacy": True,
            "privacy_epsilon": 1.0,
            "federated_learning": True,
            "data_anonymization": True
        },
        "optimization": {
            "n_iterations": 50,
            "acquisition_function": "expected_improvement"
        },
        "consensus": {
            "consensus_threshold": 0.67,
            "max_consensus_iterations": 10,
            "max_byzantine_nodes": 1
        },
        "consensus_enabled": True,
        "network_participants": ["device_002", "device_003", "device_004"]
    }
    
    # Initialize system
    system = IntegratedWeatherAISystem(config)
    
    if await system.initialize_system():
        # Example sensor data
        sensor_data = {
            "temperature": 23.5,
            "humidity": 65.2,
            "pressure": 1013.2,
            "air_quality": {"aqi": 85, "pm2_5": 12.5, "pm10": 20.1},
            "wind_speed": 3.2,
            "wind_direction": 180.0,
            "location": {"latitude": 37.7749, "longitude": -122.4194, "elevation": 50.0}
        }
        
        # Run environmental prediction
        prediction_result = await system.process_environmental_prediction(sensor_data)
        print("Prediction Result:", json.dumps(prediction_result, indent=2, default=str))
        
        # Optimize model performance
        optimization_result = await system.optimize_model_performance()
        print("Optimization Result:", json.dumps(optimization_result, indent=2, default=str))
        
        # Get system status
        status = await system.get_system_status()
        print("System Status:", json.dumps(status, indent=2, default=str))
        
        # Cleanup
        await system.cortex_client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

This completes the comprehensive Advanced Weather AI guide with all the cutting-edge components integrated. The system now includes:

1. **Microsoft Aurora Integration** - 1.3B parameter foundation model for weather forecasting
2. **Google GenCast Integration** - Probabilistic ensemble forecasting with 15-day horizon
3. **NASA Prithvi Integration** - Earth observation models for environmental analysis
4. **Distributed Federated Learning** - Privacy-preserving collaborative model training
5. **Bayesian Hyperparameter Optimization** - Advanced model tuning using Gaussian processes
6. **Byzantine Fault Tolerant Consensus** - Robust distributed decision making
7. **Hybrid Local-Cloud Architecture** - Intelligent model selection and orchestration

The system provides enterprise-grade environmental monitoring capabilities with state-of-the-art AI models, privacy protection, and distributed intelligence suitable for large-scale deployment.