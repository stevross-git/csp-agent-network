import prometheus_client
from prometheus_client import CollectorRegistry, REGISTRY

# Clear the default registry
print("Clearing Prometheus registry...")
collectors = list(REGISTRY._collector_to_names.keys())
for collector in collectors:
    try:
        REGISTRY.unregister(collector)
        print(f"Unregistered: {collector}")
    except Exception as e:
        print(f"Could not unregister {collector}: {e}")

print("Registry cleared!")
