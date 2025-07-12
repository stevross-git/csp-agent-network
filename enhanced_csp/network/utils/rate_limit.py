import time
import logging
from collections import defaultdict
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

class TokenBucket:
    """Simple token bucket implementation."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Token bucket rate limiter with per-peer buckets and cooldown."""

    def __init__(
        self,
        rate: float = 10.0,
        burst: float = 20.0,
        window: float = 1.0,
        whitelist: Optional[Set[str]] = None,
        block_threshold: int = 5,
        cooldown: float = 10.0,
    ) -> None:
        self.rate = rate / window
        self.capacity = burst
        self.peer_buckets: Dict[str, TokenBucket] = {}
        self.global_bucket = TokenBucket(self.rate, self.capacity)
        self.whitelist = set(whitelist or [])
        self.block_threshold = block_threshold
        self.cooldown = cooldown
        self.violations: Dict[str, int] = defaultdict(int)
        self.blocked_until: Dict[str, float] = {}

    def is_allowed(self, peer_id: str, tokens: float = 1.0) -> bool:
        if peer_id in self.whitelist:
            return True
        now = time.monotonic()
        if self.blocked_until.get(peer_id, 0) > now:
            return False
        if not self.global_bucket.consume(tokens):
            logger.debug("global rate limit exceeded")
            return False
        bucket = self.peer_buckets.get(peer_id)
        if bucket is None:
            bucket = TokenBucket(self.rate, self.capacity)
            self.peer_buckets[peer_id] = bucket
        if bucket.consume(tokens):
            return True
        # violation
        self.violations[peer_id] += 1
        if self.violations[peer_id] >= self.block_threshold:
            backoff = self.cooldown * (2 ** (self.violations[peer_id] - self.block_threshold))
            self.blocked_until[peer_id] = now + backoff
            logger.warning("peer %s temporarily blocked for %.1fs", peer_id, backoff)
        else:
            logger.debug("rate limit exceeded for peer %s", peer_id)
        return False

    def unblock_peer(self, peer_id: str) -> None:
        self.violations.pop(peer_id, None)
        self.blocked_until.pop(peer_id, None)

    @property
    def blocked_peers(self) -> Set[str]:
        now = time.monotonic()
        return {p for p, ts in self.blocked_until.items() if ts > now}

