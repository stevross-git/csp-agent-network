import uuid
from datetime import datetime

def generate_msg_id():
    return f"csp-{uuid.uuid4().hex[:8]}"

def current_timestamp():
    return datetime.utcnow().isoformat() + "Z"

def create_csp_message(sender, recipient, type_, task=None, context_refs=None):
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
        "meta": {
            "compression": "none",
            "auth_token": "optional"
        }
    }