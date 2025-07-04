# Fixes for enhanced_csp/network/main.py

# In main.py, replace the metrics collection section with:

async def collect_metrics(network):
    """Collect metrics safely"""
    try:
        if hasattr(network, 'metrics') and network.metrics:
            metrics_data = network.metrics
            if isinstance(metrics_data, dict):
                logger.info(f"Network metrics collected: {len(metrics_data)} metrics")
            else:
                logger.info("Network metrics available")
        else:
            logger.warning("Network metrics not available")
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")

# Replace the problematic metrics collection loop:
async def safe_metrics_loop(network):
    """Safe metrics collection loop"""
    while True:
        try:
            await collect_metrics(network)
            await asyncio.sleep(60)  # Collect every minute
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metrics loop: {e}")
            await asyncio.sleep(60)  # Continue despite errors
    