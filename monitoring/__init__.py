class DummyMonitor:
    def __init__(self):
        self.is_active = False

    async def initialize(self):
        self.is_active = True

    async def shutdown(self):
        self.is_active = False


def get_default():
    return DummyMonitor()

__all__ = ['get_default', 'DummyMonitor']
