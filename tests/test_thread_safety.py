import threading

from enhanced_csp.network.utils import ThreadSafeCounter
from enhanced_csp.network.compression import MessageCompressor


def test_threadsafe_counter():
    counter = ThreadSafeCounter()

    def worker():
        for _ in range(1000):
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.get() == 5000


def test_compressor_stats_thread_safe():
    compressor = MessageCompressor()

    payload = b"x" * 100

    def worker():
        for _ in range(200):
            compressor.compress(payload)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = compressor.export_stats()
    assert stats["compression_count"] == 600
