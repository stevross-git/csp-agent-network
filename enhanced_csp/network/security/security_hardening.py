import importlib
import ssl
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

ALLOWED_PREFIX = "enhanced_csp"


def safe_import_class(module_path: str, class_name: str) -> Any:
    """Safely import a class from whitelisted modules."""
    if not module_path.startswith(ALLOWED_PREFIX):
        raise ImportError("Disallowed module path")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class MessageValidator:
    """Basic network message validator."""

    REQUIRED_FIELDS = {"type"}

    def validate_network_message(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if not self.REQUIRED_FIELDS.issubset(message.keys()):
            return False
        return True


class SecureTLSConfig:
    """Create secure TLS contexts."""

    def __init__(self, cert_path: Path | None = None, key_path: Path | None = None, ca_path: Path | None = None) -> None:
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path

    def create_server_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if self.cert_path and self.key_path:
            context.load_cert_chain(str(self.cert_path), str(self.key_path))
        if self.ca_path:
            context.load_verify_locations(str(self.ca_path))
            context.verify_mode = ssl.CERT_REQUIRED
        return context


class SecurityOrchestrator:
    """Minimal security orchestrator used for testing."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        self.logger.info("Security orchestrator initialized")

    async def shutdown(self) -> None:
        self.logger.info("Security orchestrator shutdown")
