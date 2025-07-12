# enhanced_csp/network/utils.py
"""
Utility functions for the Enhanced CSP Network.
Provides common functionality used across the network stack.
"""

import logging
import sys
import time
import socket
import ipaddress
from typing import Optional, Union, Dict, Any
from pathlib import Path
import asyncio


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the network stack.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers
    logging.getLogger('enhanced_csp.network').setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def validate_ip_address(address: str) -> bool:
    """
    Validate if a string is a valid IP address.
    
    Args:
        address: IP address string to validate
        
    Returns:
        True if valid IP address, False otherwise
    """
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return False


def validate_port_number(port: Union[int, str]) -> bool:
    """
    Validate if a port number is valid.
    
    Args:
        port: Port number to validate (int or string)
        
    Returns:
        True if valid port number, False otherwise
    """
    try:
        port_int = int(port)
        return 1 <= port_int <= 65535
    except (ValueError, TypeError):
        return False


def validate_message_size(message: Any, max_size: int = 10 * 1024 * 1024) -> bool:
    """
    Validate if a message is within size limits.
    
    Args:
        message: Message to validate
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if message is within limits, False otherwise
    """
    try:
        message_size = len(str(message).encode('utf-8'))
        return message_size <= max_size
    except Exception:
        return False


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes value into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes_value)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds into human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    if hours < 24:
        return f"{hours}h {remaining_minutes}m"
    
    days = int(hours // 24)
    remaining_hours = hours % 24
    
    return f"{days}d {remaining_hours}h"


def get_local_ip() -> str:
    """
    Get the local IP address of this machine.
    
    Returns:
        Local IP address as string
    """
    try:
        # Connect to a remote server to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available on the given host.
    
    Args:
        host: Host address
        port: Port number
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False


def find_available_port(host: str = "localhost", start_port: int = 30300, max_attempts: int = 100) -> Optional[int]:
    """
    Find an available port starting from the given port.
    
    Args:
        host: Host address to check
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number or None if not found
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return None


class Timer:
    """Simple timer context manager for measuring execution time"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    def __str__(self):
        if self.elapsed is not None:
            return f"{self.description}: {self.elapsed*1000:.2f}ms"
        return f"{self.description}: Not completed"


class AsyncTimer:
    """Async timer context manager for measuring execution time"""
    
    def __init__(self, description: str = "Async Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    def __str__(self):
        if self.elapsed is not None:
            return f"{self.description}: {self.elapsed*1000:.2f}ms"
        return f"{self.description}: Not completed"


class RateLimiter:
    """Simple rate limiter for controlling operation frequency"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> bool:
        """
        Acquire permission to make a call.
        
        Returns:
            True if call is allowed, False if rate limited
        """
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if we can make another call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    async def wait_and_acquire(self) -> None:
        """Wait until a call can be made"""
        while not await self.acquire():
            await asyncio.sleep(0.1)


def calculate_exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


class CircuitBreaker:
    """Simple circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """
        Call a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


def create_task_with_error_handling(coro, error_callback=None, task_name: str = "unnamed"):
    """
    Create an asyncio task with error handling.
    
    Args:
        coro: Coroutine to execute
        error_callback: Optional callback for handling errors
        task_name: Name for the task (for logging)
        
    Returns:
        asyncio.Task
    """
    async def wrapped_coro():
        try:
            return await coro
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Task '{task_name}' failed: {e}")
            
            if error_callback:
                try:
                    if asyncio.iscoroutinefunction(error_callback):
                        await error_callback(e)
                    else:
                        error_callback(e)
                except Exception as callback_error:
                    logger.error(f"Error callback for task '{task_name}' failed: {callback_error}")
            
            raise
    
    return asyncio.create_task(wrapped_coro(), name=task_name)


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    import json
    
    def default_serializer(o):
        if hasattr(o, '__dict__'):
            return o.__dict__
        elif hasattr(o, 'isoformat'):  # datetime objects
            return o.isoformat()
        else:
            return str(o)
    
    try:
        return json.dumps(obj, default=default_serializer, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}", "type": str(type(obj))})


def chunked(iterable, chunk_size: int):
    """
    Break an iterable into chunks of specified size.
    
    Args:
        iterable: Iterable to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the iterable
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


# Import for chunked function
from itertools import islice


class PerformanceProfiler:
    """Simple performance profiler for measuring function execution times"""
    
    def __init__(self):
        self.measurements = {}
    
    def measure(self, func_name: str):
        """Decorator for measuring function execution time"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    self._record_measurement(func_name, elapsed)
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    self._record_measurement(func_name, elapsed)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _record_measurement(self, func_name: str, elapsed: float):
        """Record a measurement"""
        if func_name not in self.measurements:
            self.measurements[func_name] = []
        
        self.measurements[func_name].append(elapsed)
        
        # Keep only last 1000 measurements
        if len(self.measurements[func_name]) > 1000:
            self.measurements[func_name].pop(0)
    
    def get_stats(self, func_name: str) -> Dict[str, float]:
        """Get statistics for a function"""
        measurements = self.measurements.get(func_name, [])
        if not measurements:
            return {}
        
        import statistics
        return {
            'count': len(measurements),
            'avg_ms': statistics.mean(measurements) * 1000,
            'min_ms': min(measurements) * 1000,
            'max_ms': max(measurements) * 1000,
            'median_ms': statistics.median(measurements) * 1000,
            'total_time_ms': sum(measurements) * 1000
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all measured functions"""
        return {func_name: self.get_stats(func_name) for func_name in self.measurements.keys()}


# Global profiler instance
profiler = PerformanceProfiler()


def measure_performance(func_name: str):
    """Decorator for measuring function performance"""
    return profiler.measure(func_name)