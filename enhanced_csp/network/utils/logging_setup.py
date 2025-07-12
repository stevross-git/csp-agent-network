"""
Logging Setup and Configuration
==============================

Centralized logging configuration with structured logging support.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if requested
        if self.include_extra:
            # Add any extra fields from the record
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                }
            }
            if extra_fields:
                log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_colors = use_colors and sys.stderr.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if enabled."""
        formatted = super().format(record)
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            if color:
                formatted = f"{color}{formatted}{self.RESET}"
        
        return formatted


class CorrelationFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        super().__init__()
        self.correlation_id = correlation_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the record."""
        if not hasattr(record, 'correlation_id') and self.correlation_id:
            record.correlation_id = self.correlation_id
        return True


class SamplingFilter(logging.Filter):
    """Filter to sample high-frequency log messages."""
    
    def __init__(self, sample_rate: float = 1.0):
        super().__init__()
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.message_counts: Dict[str, int] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Sample messages based on rate."""
        if self.sample_rate >= 1.0:
            return True
        
        message_key = f"{record.name}:{record.levelname}:{record.getMessage()}"
        self.message_counts[message_key] = self.message_counts.get(message_key, 0) + 1
        
        # Allow every nth message through
        sample_interval = int(1 / self.sample_rate) if self.sample_rate > 0 else float('inf')
        return self.message_counts[message_key] % sample_interval == 1


class CSPLogger(logging.LoggerAdapter):
    """Custom logger adapter with CSP-specific functionality."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
        self.component = extra.get('component', 'unknown') if extra else 'unknown'
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the logging call."""
        # Add component info to extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def with_context(self, **context) -> 'CSPLogger':
        """Create a new logger with additional context."""
        new_extra = {**self.extra, **context}
        return CSPLogger(self.logger, new_extra)
    
    def performance(self, msg: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.info(msg, extra={'duration_ms': duration * 1000, 'metric_type': 'performance', **kwargs})
    
    def security(self, msg: str, **kwargs):
        """Log security events."""
        self.warning(msg, extra={'event_type': 'security', **kwargs})
    
    def audit(self, msg: str, user: str = None, action: str = None, **kwargs):
        """Log audit events."""
        extra = {'event_type': 'audit', **kwargs}
        if user:
            extra['user'] = user
        if action:
            extra['action'] = action
        self.info(msg, extra=extra)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = 'structured',
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_colors: bool = True,
    correlation_id: Optional[str] = None,
    sample_rate: float = 1.0
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level
        format_type: 'structured', 'json', or 'text'
        log_file: Specific log file path
        log_dir: Log directory (used if log_file not specified)
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        enable_console: Enable console logging
        enable_colors: Enable colored console output
        correlation_id: Default correlation ID for filtering
        sample_rate: Sample rate for high-frequency messages
    
    Returns:
        Root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if format_type in ['structured', 'json']:
        file_formatter = StructuredFormatter()
        console_formatter = StructuredFormatter() if format_type == 'json' else ColoredFormatter(enable_colors)
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(enable_colors)
    
    # Setup filters
    filters = []
    
    if correlation_id:
        filters.append(CorrelationFilter(correlation_id))
    
    if sample_rate < 1.0:
        filters.append(SamplingFilter(sample_rate))
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        for f in filters:
            console_handler.addFilter(f)
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_dir and not log_file:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / 'csp_system.log'
        
        # Use rotating file handler for large log files
        if max_bytes > 0:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setFormatter(file_formatter)
        
        for f in filters:
            file_handler.addFilter(f)
        
        root_logger.addHandler(file_handler)
    
    # Setup specific loggers
    _setup_library_loggers()
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    
    return root_logger


def _setup_library_loggers():
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    library_levels = {
        'uvicorn': logging.WARNING,
        'uvicorn.access': logging.WARNING,
        'fastapi': logging.WARNING,
        'sqlalchemy': logging.WARNING,
        'asyncio': logging.WARNING,
        'aioredis': logging.WARNING,
        'websockets': logging.WARNING,
        'prometheus_client': logging.WARNING,
    }
    
    for library, level in library_levels.items():
        logging.getLogger(library).setLevel(level)


def get_logger(name: str, component: Optional[str] = None, **extra) -> CSPLogger:
    """
    Get a CSP logger with optional component context.
    
    Args:
        name: Logger name
        component: Component name for context
        **extra: Additional context fields
    
    Returns:
        CSP logger adapter
    """
    logger = logging.getLogger(name)
    
    context = extra.copy()
    if component:
        context['component'] = component
    
    return CSPLogger(logger, context)


def performance_logger(func):
    """Decorator to log function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(f"{func.__module__}.{func.__name__}", component='performance')
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.performance(f"Function {func.__name__} completed", duration)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(f"{func.__module__}.{func.__name__}", component='performance')
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.performance(f"Function {func.__name__} completed", duration)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def audit_logger(action: str):
    """Decorator to log audit events."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(f"{func.__module__}.{func.__name__}", component='audit')
            
            # Try to extract user from kwargs or args
            user = kwargs.get('user') or (args[0] if args and hasattr(args[0], 'get') and args[0].get('user') else 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                logger.audit(f"Action {action} completed successfully", user=str(user), action=action)
                return result
            except Exception as e:
                logger.audit(f"Action {action} failed: {e}", user=str(user), action=action, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(f"{func.__module__}.{func.__name__}", component='audit')
            
            # Try to extract user from kwargs or args
            user = kwargs.get('user') or (args[0] if args and hasattr(args[0], 'get') and args[0].get('user') else 'unknown')
            
            try:
                result = func(*args, **kwargs)
                logger.audit(f"Action {action} completed successfully", user=str(user), action=action)
                return result
            except Exception as e:
                logger.audit(f"Action {action} failed: {e}", user=str(user), action=action, error=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for temporary log level changes
class LogLevel:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: Union[str, logging.Logger], level: Union[str, int]):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger
        
        if isinstance(level, str):
            self.new_level = getattr(logging, level.upper())
        else:
            self.new_level = level
        
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import time
    
    # Test logging setup
    print("Testing logging setup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup logging
        setup_logging(
            level=logging.DEBUG,
            format_type='structured',
            log_dir=temp_dir,
            enable_console=True,
            enable_colors=True,
            sample_rate=1.0
        )
        
        # Get loggers
        logger = get_logger(__name__, component='test')
        perf_logger = get_logger('performance', component='performance')
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Test structured logging
        logger.info("Structured log", extra={'user_id': 123, 'action': 'test'})
        
        # Test performance logging
        start = time.time()
        time.sleep(0.1)
        duration = time.time() - start
        perf_logger.performance("Test operation", duration)
        
        # Test context logging
        context_logger = logger.with_context(request_id='req-123', user='testuser')
        context_logger.info("Message with context")
        
        # Test audit logging
        logger.audit("User login", user='testuser', action='login', ip='127.0.0.1')
        
        # Test security logging
        logger.security("Suspicious activity detected", user='attacker', ip='192.168.1.100')
        
        # Test exception logging
        try:
            raise ValueError("Test exception")
        except Exception:
            logger.exception("Exception occurred")
        
        # Test log level context manager
        with LogLevel(logger.logger, logging.ERROR):
            logger.info("This should not appear")
            logger.error("This should appear")
        
        logger.info("Back to normal level")
        
        print("✅ Logging test completed")
        print(f"Log files created in: {temp_dir}")
