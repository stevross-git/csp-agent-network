# enhanced_csp/network/utils.py
"""
Utility functions and helpers for Enhanced CSP Network.
Provides logging, validation, and common utilities.
"""

import logging
import ipaddress
import socket
import struct
import time
from typing import Optional, Any, Dict
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for network modules."""
    return logging.getLogger(f"enhanced_csp.network.{name}")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging for network modules."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger("enhanced_csp.network")
    root_logger.setLevel(log_level)
    root_logger.handlers = handlers


def validate_ip_address(ip_str: str) -> bool:
    """Validate IP address string."""
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def validate_port_number(port: int) -> bool:
    """Validate port number."""
    return 1 <= port <= 65535


def validate_message_size(size: int, max_size: int = 1024 * 1024) -> bool:
    """Validate message size."""
    return 0 < size <= max_size


class NetworkLogger:
    """Specialized logger for network events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def connection_established(self, peer: str, protocol: str):
        """Log connection establishment."""
        self.logger.info(f"Connection established to {peer} via {protocol}")
    
    def connection_failed(self, peer: str, error: str):
        """Log connection failure."""
        self.logger.warning(f"Connection failed to {peer}: {error}")
    
    def message_sent(self, peer: str, size: int, success: bool):
        """Log message send."""
        status = "success" if success else "failed"
        self.logger.debug(f"Message sent to {peer}: {size} bytes - {status}")
    
    def performance_alert(self, metric: str, value: float, threshold: float):
        """Log performance alert."""
        self.logger.warning(f"Performance alert: {metric} = {value:.2f} (threshold: {threshold:.2f})")


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.security")
    
    def authentication_success(self, peer: str):
        """Log successful authentication."""
        self.logger.info(f"Authentication successful for peer: {peer}")
    
    def authentication_failure(self, peer: str, reason: str):
        """Log authentication failure."""
        self.logger.warning(f"Authentication failed for peer {peer}: {reason}")
    
    def security_violation(self, peer: str, violation: str):
        """Log security violation."""
        self.logger.error(f"Security violation from {peer}: {violation}")


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
    
    def throughput_update(self, messages_per_sec: float, bytes_per_sec: float):
        """Log throughput metrics."""
        self.logger.debug(f"Throughput: {messages_per_sec:.1f} msg/s, {bytes_per_sec/1024/1024:.2f} MB/s")
    
    def latency_update(self, avg_latency_ms: float, p95_latency_ms: float):
        """Log latency metrics."""
        self.logger.debug(f"Latency: avg={avg_latency_ms:.1f}ms, p95={p95_latency_ms:.1f}ms")
    
    def optimization_result(self, optimization: str, improvement: float):
        """Log optimization results."""
        self.logger.info(f"Optimization '{optimization}' improved performance by {improvement:.2%}")


class AuditLogger:
    """Specialized logger for audit trails."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.audit")
        
        # Ensure audit logs go to separate file
        audit_handler = logging.FileHandler("csp_network_audit.log")
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(name)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
    
    def network_event(self, event: str, peer: str, details: Dict[str, Any]):
        """Log network audit event."""
        self.logger.info(f"Event: {event}, Peer: {peer}, Details: {details}")
    
    def configuration_change(self, parameter: str, old_value: Any, new_value: Any):
        """Log configuration changes."""
        self.logger.info(f"Config change: {parameter} changed from {old_value} to {new_value}")
    
    def security_event(self, event: str, peer: str, severity: str):
        """Log security audit event."""
        self.logger.warning(f"Security event: {event}, Peer: {peer}, Severity: {severity}")