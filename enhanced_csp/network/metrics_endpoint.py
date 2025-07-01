# enhanced_csp/network/metrics_endpoint.py
from typing import Dict, Any
import time

class NetworkMetricsCollector:
    """Collect and expose network optimization metrics"""
    
    def __init__(self, channel: 'OptimizedNetworkChannel'):
        self.channel = channel
        self._start_time = time.time()
    
    def collect_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics"""
        stats = self.channel.get_stats()
        compression_stats = self.channel.compressor.export_stats()
        
        metrics = []
        
        # Channel metrics
        metrics.append(f'# HELP csp_messages_sent_total Total messages sent successfully')
        metrics.append(f'# TYPE csp_messages_sent_total counter')
        metrics.append(f'csp_messages_sent_total{{channel="{self.channel.channel_id}"}} {stats["messages_sent"]}')
        
        metrics.append(f'# HELP csp_messages_failed_total Total messages failed to send')
        metrics.append(f'# TYPE csp_messages_failed_total counter')
        metrics.append(f'csp_messages_failed_total{{channel="{self.channel.channel_id}"}} {stats["messages_failed"]}')
        
        # Compression metrics
        metrics.append(f'# HELP csp_compression_bytes_saved_total Total bytes saved by compression')
        metrics.append(f'# TYPE csp_compression_bytes_saved_total counter')
        metrics.append(f'csp_compression_bytes_saved_total {{channel="{self.channel.channel_id}"}} {compression_stats["space_saved_bytes"]}')
        
        metrics.append(f'# HELP csp_compression_ratio Average compression ratio')
        metrics.append(f'# TYPE csp_compression_ratio gauge')
        metrics.append(f'csp_compression_ratio{{channel="{self.channel.channel_id}"}} {compression_stats["average_compression_ratio"]:.3f}')
        
        # Batching metrics
        batch_metrics = stats["batch_metrics"]
        metrics.append(f'# HELP csp_batch_size_average Average batch size')
        metrics.append(f'# TYPE csp_batch_size_average gauge')
        metrics.append(f'csp_batch_size_average{{channel="{self.channel.channel_id}"}} {batch_metrics["average_batch_size"]:.1f}')
        
        # Connection pool metrics
        pool_stats = stats["connection_pool"]
        metrics.append(f'# HELP csp_connections_active Active connections in pool')
        metrics.append(f'# TYPE csp_connections_active gauge')
        metrics.append(f'csp_connections_active{{channel="{self.channel.channel_id}"}} {pool_stats["total_connections"]}')
        
        # Queue metrics
        metrics.append(f'# HELP csp_queue_size Current queue sizes')
        metrics.append(f'# TYPE csp_queue_size gauge')
        metrics.append(f'csp_queue_size{{channel="{self.channel.channel_id}",queue="batch"}} {stats["batch_queue_size"]}')
        metrics.append(f'csp_queue_size{{channel="{self.channel.channel_id}",queue="retry"}} {stats["retry_queue_size"]}')
        
        return '\n'.join(metrics)