"""Utility functions for basic CSP messages."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List


def generate_msg_id() -> str:
    return f"csp-{uuid.uuid4().hex[:8]}"


def current_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_csp_message(
    sender: str,
    recipient: str,
    type_: str,
    task: Dict[str, Any] | None = None,
    context_refs: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "protocol_version": "1.0",
        "msg_id": generate_msg_id(),
        "timestamp": current_timestamp(),
        "sender": sender,
        "recipient": recipient,
        "type": type_,
        "task": task or {},
        "context_refs": context_refs or [],
        "expect_response": True,
        "meta": {"compression": "none", "auth_token": "optional"},
    }
