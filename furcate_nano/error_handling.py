# ============================================================================
# furcate_nano/error_handling.py
"""
Comprehensive error handling framework for Furcate Nano with asyncio support.
Implements retry mechanisms, circuit breakers, and graceful degradation.
"""

import asyncio
import logging
import traceback
import functools
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union, Type
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import inspect

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification."""
    HARDWARE = "hardware"
    NETWORK = "network" 
    SENSOR = "sensor"
    ML_MODEL = "ml_model"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    """Enhanced error context with comprehensive metadata."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    error_type: str
    message: str
    traceback_info: str
    metadata: Dict[str, Any]
    retry_count: int = 0
    recovery_attempted: bool = False
    user_impact: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value
        }

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    def __call__(self, func):
        """Decorator to apply circuit breaker pattern."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN for {func.__name__}"
                    )
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
                
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
            
        return delay

def retry_async(strategy: RetryStrategy = None, 
                exceptions: tuple = (Exception,),
                on_retry: Callable = None):
    """Async retry decorator with configurable strategy."""
    if strategy is None:
        strategy = RetryStrategy()
        
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(strategy.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < strategy.max_attempts - 1:
                        delay = strategy.get_delay(attempt)
                        
                        if on_retry:
                            await on_retry(e, attempt + 1, delay)
                            
                        logger.warning(
                            f"Retry {attempt + 1}/{strategy.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s due to: {str(e)}"
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {strategy.max_attempts} retry attempts failed for {func.__name__}"
                        )
            
            raise last_exception
            
        return wrapper
    return decorator

class ErrorHandler:
    """Comprehensive error handling system with context awareness."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.error_history: List[ErrorContext] = []
        self.error_counters: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.notification_handlers: List[Callable] = []
        
        # Configure based on config
        self.max_error_history = self.config.get('max_error_history', 1000)
        self.enable_recovery = self.config.get('enable_recovery', True)
        self.enable_notifications = self.config.get('enable_notifications', True)
        
        logger.info("🛡️ Error Handler initialized")
    
    def register_recovery_handler(self, category: ErrorCategory, handler: Callable):
        """Register recovery handler for specific error category."""
        self.recovery_handlers[category] = handler
        logger.info(f"Registered recovery handler for {category.value}")
    
    def register_notification_handler(self, handler: Callable):
        """Register notification handler for error alerts."""
        self.notification_handlers.append(handler)
        logger.info("Registered notification handler")
    
    async def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Main error handling entry point."""
        try:
            # Create error context
            error_context = self._create_error_context(exception, context)
            
            # Log error
            self._log_error(error_context)
            
            # Update statistics
            self._update_error_statistics(error_context)
            
            # Store in history
            self._store_error_context(error_context)
            
            # Attempt recovery if enabled
            if self.enable_recovery:
                await self._attempt_recovery(error_context)
            
            # Send notifications if enabled
            if self.enable_notifications:
                await self._send_notifications(error_context)
            
            return error_context
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
            # Fallback logging
            logger.error(f"Original error: {exception}")
            raise e
    
    def _create_error_context(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Create comprehensive error context."""
        import uuid
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        
        function_name = "unknown"
        component = "unknown"
        
        if caller_frame:
            function_name = caller_frame.f_code.co_name
            # Try to extract component from file path
            filename = caller_frame.f_code.co_filename
            if 'furcate_nano' in filename:
                component = filename.split('furcate_nano/')[-1].split('.')[0]
        
        # Determine error category and severity
        category = self._categorize_error(exception, context)
        severity = self._assess_severity(exception, category, context)
        
        return ErrorContext(
            error_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            function_name=function_name,
            error_type=type(exception).__name__,
            message=str(exception),
            traceback_info=traceback.format_exc(),
            metadata=context or {},
            user_impact=self._assess_user_impact(severity, category)
        )
    
    def _categorize_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Categorize error based on exception type and context."""
        error_type = type(exception).__name__
        
        # Network-related errors
        if any(term in error_type.lower() for term in 
               ['connection', 'timeout', 'network', 'socket', 'http']):
            return ErrorCategory.NETWORK
        
        # Hardware-related errors
        if any(term in error_type.lower() for term in 
               ['hardware', 'device', 'gpio', 'i2c', 'spi']):
            return ErrorCategory.HARDWARE
        
        # Sensor-related errors
        if any(term in error_type.lower() for term in 
               ['sensor', 'reading', 'calibration']):
            return ErrorCategory.SENSOR
        
        # ML model errors
        if any(term in error_type.lower() for term in 
               ['model', 'inference', 'prediction', 'tensor']):
            return ErrorCategory.ML_MODEL
        
        # Storage errors
        if any(term in error_type.lower() for term in 
               ['storage', 'database', 'file', 'permission']):
            return ErrorCategory.STORAGE
        
        # Configuration errors
        if any(term in error_type.lower() for term in 
               ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        # Integration errors
        if any(term in error_type.lower() for term in 
               ['integration', 'api', 'webhook', 'mqtt']):
            return ErrorCategory.INTEGRATION
        
        return ErrorCategory.SYSTEM
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory, 
                        context: Dict[str, Any] = None) -> ErrorSeverity:
        """Assess error severity based on various factors."""
        error_type = type(exception).__name__
        
        # Critical errors that affect core functionality
        if any(term in error_type.lower() for term in 
               ['critical', 'fatal', 'corruption', 'security']):
            return ErrorSeverity.CRITICAL
        
        # High severity for important components
        if category in [ErrorCategory.HARDWARE, ErrorCategory.STORAGE]:
            return ErrorSeverity.HIGH
        
        # Network and sensor issues are usually medium
        if category in [ErrorCategory.NETWORK, ErrorCategory.SENSOR]:
            return ErrorSeverity.MEDIUM
        
        # Configuration and integration issues are usually low
        return ErrorSeverity.LOW
    
    def _assess_user_impact(self, severity: ErrorSeverity, category: ErrorCategory) -> str:
        """Assess potential user impact."""
        if severity == ErrorSeverity.CRITICAL:
            return "system_unavailable"
        elif severity == ErrorSeverity.HIGH:
            return "functionality_degraded"
        elif severity == ErrorSeverity.MEDIUM:
            return "minor_disruption"
        else:
            return "minimal_impact"
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_msg = (
            f"[{error_context.error_id}] {error_context.category.value.upper()} ERROR "
            f"in {error_context.component}.{error_context.function_name}: {error_context.message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Log detailed traceback for high severity errors
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Traceback for {error_context.error_id}:\n{error_context.traceback_info}")
    
    def _update_error_statistics(self, error_context: ErrorContext):
        """Update error statistics and counters."""
        key = f"{error_context.category.value}_{error_context.error_type}"
        self.error_counters[key] = self.error_counters.get(key, 0) + 1
    
    def _store_error_context(self, error_context: ErrorContext):
        """Store error context in history."""
        self.error_history.append(error_context)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    async def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt automated recovery based on error category."""
        if error_context.category in self.recovery_handlers:
            try:
                recovery_handler = self.recovery_handlers[error_context.category]
                await recovery_handler(error_context)
                error_context.recovery_attempted = True
                logger.info(f"Recovery attempted for error {error_context.error_id}")
            except Exception as e:
                logger.error(f"Recovery failed for {error_context.error_id}: {e}")
    
    async def _send_notifications(self, error_context: ErrorContext):
        """Send error notifications to registered handlers."""
        if not self.notification_handlers:
            return
        
        # Only notify for medium+ severity errors
        if error_context.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            for handler in self.notification_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(error_context)
                    else:
                        handler(error_context)
                except Exception as e:
                    logger.error(f"Notification handler failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [
            e for e in self.error_history 
            if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        stats = {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "categories": {},
            "severities": {},
            "top_errors": {}
        }
        
        # Category breakdown
        for error in self.error_history:
            cat = error.category.value
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
        
        # Severity breakdown
        for error in self.error_history:
            sev = error.severity.value
            stats["severities"][sev] = stats["severities"].get(sev, 0) + 1
        
        # Top error types
        stats["top_errors"] = dict(
            sorted(self.error_counters.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return stats
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]

# Global error handler instance
error_handler = ErrorHandler()

# Convenience decorators
def handle_errors(category: ErrorCategory = ErrorCategory.SYSTEM):
    """Decorator to automatically handle errors in functions."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(e, {
                    'function': func.__name__,
                    'category': category,
                    'args': str(args)[:100],  # Truncate for privacy
                    'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                })
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create task for async error handling
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(error_handler.handle_error(e, {
                        'function': func.__name__,
                        'category': category,
                        'args': str(args)[:100],
                        'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
                    }))
                except Exception:
                    # Fallback to basic logging
                    logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Example recovery handlers
async def network_recovery_handler(error_context: ErrorContext):
    """Example recovery handler for network errors."""
    logger.info(f"Attempting network recovery for error {error_context.error_id}")
    
    # Wait a bit and try to re-establish connections
    await asyncio.sleep(2)
    
    # Could implement specific recovery logic here:
    # - Restart network connections
    # - Switch to backup endpoints
    # - Enable offline mode
    
    logger.info(f"Network recovery completed for {error_context.error_id}")

async def sensor_recovery_handler(error_context: ErrorContext):
    """Example recovery handler for sensor errors."""
    logger.info(f"Attempting sensor recovery for error {error_context.error_id}")
    
    # Could implement specific recovery logic here:
    # - Reinitialize sensor
    # - Recalibrate
    # - Switch to backup sensor
    
    logger.info(f"Sensor recovery completed for {error_context.error_id}")

# Register default recovery handlers
error_handler.register_recovery_handler(ErrorCategory.NETWORK, network_recovery_handler)
error_handler.register_recovery_handler(ErrorCategory.SENSOR, sensor_recovery_handler)

# Example usage and testing
if __name__ == "__main__":
    async def test_error_handling():
        """Test the error handling system."""
        
        @handle_errors(ErrorCategory.NETWORK)
        @retry_async(RetryStrategy(max_attempts=3, base_delay=0.1))
        async def failing_network_function():
            """Function that simulates network failures."""
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise ConnectionError("Network connection failed")
            return "Success!"
        
        @handle_errors(ErrorCategory.SENSOR)
        async def failing_sensor_function():
            """Function that simulates sensor failures."""
            raise RuntimeError("Sensor calibration error")
        
        print("Testing error handling system...")
        
        # Test network function with retries
        try:
            result = await failing_network_function()
            print(f"Network function result: {result}")
        except Exception as e:
            print(f"Network function failed: {e}")
        
        # Test sensor function
        try:
            await failing_sensor_function()
        except Exception as e:
            print(f"Sensor function failed: {e}")
        
        # Print error statistics
        stats = error_handler.get_error_statistics()
        print(f"\nError Statistics: {json.dumps(stats, indent=2)}")
    
    # Run test
    asyncio.run(test_error_handling())