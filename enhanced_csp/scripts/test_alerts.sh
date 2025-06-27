#!/bin/bash
set -euo pipefail

echo "🧪 Testing Alert System..."

# Test critical alert
echo "Testing critical alert..."
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestCriticalAlert",
      "severity": "critical",
      "service": "test"
    },
    "annotations": {
      "summary": "Test critical alert",
      "description": "This is a test critical alert"
    }
  }]'

# Test warning alert
echo "Testing warning alert..."
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestWarningAlert",
      "severity": "warning",
      "service": "test"
    },
    "annotations": {
      "summary": "Test warning alert",
      "description": "This is a test warning alert"
    }
  }]'

echo "✅ Test alerts sent!"