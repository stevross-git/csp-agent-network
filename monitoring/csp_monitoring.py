class MonitoringSystem:
    """Placeholder monitoring system."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.is_active = False

    async def initialize(self):
        """Initialize monitoring hooks."""
        # TODO: integrate real metrics collection here
        self.is_active = True

    async def shutdown(self):
        """Shutdown monitoring system."""
        self.is_active = False


_default_monitor = MonitoringSystem()


def get_default() -> MonitoringSystem:
    """Return a singleton monitoring system instance."""
    return _default_monitor
