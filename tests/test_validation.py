import pytest

from enhanced_csp.network.utils import (
    validate_ip_address,
    validate_port_number,
    validate_node_id,
    validate_message_size,
    PeerRateLimiter,
)
from enhanced_csp.network.errors import ValidationError


def test_validate_ip_address():
    assert validate_ip_address("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValidationError):
        validate_ip_address("300.300.300.300")


def test_validate_port_number():
    assert validate_port_number(8080) == 8080
    with pytest.raises(ValidationError):
        validate_port_number(70000)


def test_validate_node_id():
    good = "Qm" + "a" * 44
    assert validate_node_id(good) == good
    with pytest.raises(ValidationError):
        validate_node_id("badid")


def test_validate_message_size():
    validate_message_size("x" * 10, 20)
    with pytest.raises(ValidationError):
        validate_message_size("x" * 30, 20)


def test_peer_rate_limiter():
    rl = PeerRateLimiter(2, 0.5)
    assert rl.is_allowed("peer")
    assert rl.is_allowed("peer")
    assert not rl.is_allowed("peer")
