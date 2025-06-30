import time
from dataclasses import dataclass

@dataclass
class RateLimiter:
    capacity: int
    period: float
    _allowance: float = 0
    _last_check: float = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_check
        self._last_check = now
        self._allowance += elapsed * (self.capacity / self.period)
        if self._allowance > self.capacity:
            self._allowance = self.capacity
        if self._allowance < 1.0:
            return False
        else:
            self._allowance -= 1.0
            return True
