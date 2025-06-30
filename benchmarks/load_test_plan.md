# Load Test Plan

This document outlines a plan for benchmarking the Enhanced CSP network stack
with up to **10,000** concurrent peers.

## Metrics
- Latency (RTT)
- Throughput
- CPU and memory utilisation

## Approach
1. Use container based simulation to launch thousands of lightweight peers.
2. Measure connection establishment and message propagation time.
3. Capture metrics via Prometheus and export to CSV.
4. Visualise using Grafana dashboards.

