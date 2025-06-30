import pytest
from enhanced_csp.network.core.types import NodeID
from cryptography.hazmat.primitives.asymmetric import ed25519


@pytest.mark.asyncio
async def test_node_id_base58_roundtrip():
    key = ed25519.Ed25519PrivateKey.generate()
    node_id = NodeID.from_public_key(key.public_key())
    b58 = node_id.to_base58()
    # Ensure we can decode base58 back to bytes
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    num = 0
    for char in b58:
        num = num * 58 + alphabet.index(char)
    decoded = num.to_bytes(len(node_id.raw_id), 'big')
    assert decoded == node_id.raw_id
