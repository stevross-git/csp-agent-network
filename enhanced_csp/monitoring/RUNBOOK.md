# Enhanced CSP Monitoring Runbook

## Overview
This runbook contains procedures for operating and troubleshooting the Enhanced CSP monitoring system.

## Architecture
- **Metrics Collection**: Prometheus scrapes metrics from all components
- **Visualization**: Grafana dashboards display real-time and historical data
- **Alerting**: Alertmanager routes alerts based on severity
- **Storage**: Prometheus stores metrics for 7 days by default

## Common Operations

### 1. Adding New Metrics

To add a new metric:

1. Define the metric in the appropriate module:
```python
from prometheus_client import Counter, Histogram, Gauge

my_metric = Counter(
    'csp_my_metric_total',
    'Description of my metric',
    ['label1', 'label2']
)
```

2. Instrument the code:
```python
my_metric.labels(label1='value1', label2='value2').inc()
```

3. Update Grafana dashboard to visualize the metric

### 2. Debugging Missing Metrics

If metrics are not appearing:

1. Check if monitoring is enabled:
```bash
grep MONITORING_ENABLED backend/config/settings.py
```

2. Verify the endpoint is working:
```bash
curl http://localhost:8000/metrics | grep my_metric
```

3. Check Prometheus targets:
```bash
curl http://localhost:9090/api/v1/targets | jq
```

4. Look for scrape errors in Prometheus:
```
http://localhost:9090/targets
```

### 3. Alert Response Procedures

#### High Error Rate Alert
1. Check recent deployments
2. Review error logs: `docker logs csp_api`
3. Check database connectivity
4. Review recent code changes

#### High Response Time Alert
1. Check CPU and memory usage
2. Review slow query logs
3. Check for blocking operations
4. Consider scaling if needed

#### Database Connection Pool Exhaustion
1. Check for connection leaks
2. Review long-running queries
3. Increase pool size if needed
4. Check for deadlocks

#### SLO Violation Alert
1. Review recent incidents
2. Check error budget consumption
3. Implement fixes for root causes
4. Update runbooks if needed

### 4. Performance Tuning

#### Prometheus Performance
- Increase `--storage.tsdb.retention.size` for more history
- Adjust `scrape_interval` for less frequent collection
- Use recording rules for expensive queries

#### Grafana Performance
- Use time range limits on dashboards
- Implement query caching
- Optimize PromQL queries

### 5. Maintenance Tasks

#### Weekly
- Review alert noise and tune thresholds
- Check disk usage for metrics storage
- Review SLO compliance

#### Monthly
- Update dashboards based on feedback
- Review and optimize expensive queries
- Update documentation

### 6. Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| No metrics appearing | Monitoring disabled | Set MONITORING_ENABLED=true |
| Gaps in metrics | Scrape failures | Check target health |
| High cardinality warnings | Too many label combinations | Review and reduce labels |
| Slow dashboards | Expensive queries | Optimize PromQL |
| Missing alerts | Alertmanager down | Check alertmanager logs |

## Emergency Procedures

### Complete Monitoring Failure
1. Verify Prometheus is running: `docker ps | grep prometheus`
2. Check disk space: `df -h`
3. Restart monitoring stack: `docker-compose -f monitoring/docker-compose.monitoring.yml restart`
4. Verify metrics collection resumed

### Metrics Explosion
1. Identify high cardinality metrics: Check Prometheus UI
2. Drop specific metrics: Use metric_relabel_configs
3. Increase resources if needed
4. Implement recording rules

## Contact Information
- On-call: Check PagerDuty
- Escalation: #monitoring Slack channel
- Documentation: /monitoring/README.md
