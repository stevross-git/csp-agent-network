from time import sleep

from enhanced_csp.network.utils import RateLimiter


def test_token_bucket_basic():
    rl = RateLimiter(rate=2, burst=2, window=1)
    assert rl.is_allowed('peer')
    assert rl.is_allowed('peer')
    assert not rl.is_allowed('peer')
    sleep(0.6)
    assert rl.is_allowed('peer')

