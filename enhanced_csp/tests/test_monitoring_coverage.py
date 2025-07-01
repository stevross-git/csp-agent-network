"""
Test monitoring coverage and metrics
"""
import pytest
import asyncio
import aiohttp
from typing import Dict, List
import time

class TestMonitoringCoverage:
    """Test that all monitoring is properly wired"""
    
    @pytest.fixture
    async def api_client(self):
        """Create API client"""
        async with aiohttp.ClientSession() as session:
            yield session
    
    async def test_metrics_endpoint_exists(self, api_client):
        """Test that /metrics endpoint exists and returns data"""
        async with api_client.get('http://localhost:8000/metrics') as resp:
            assert resp.status == 200
            text = await resp.text()
            
            # Check for expected metrics
            assert 'csp_http_requests_total' in text
            assert 'csp_system_cpu_percent' in text
            assert 'csp_auth_login_attempts_total' in text
    
    async def test_auth_metrics_recorded(self, api_client):
        """Test that auth operations record metrics"""
        # Attempt login
        async with api_client.post('http://localhost:8000/api/auth/login',
                                 json={"username": "test", "password": "test"}) as resp:
            pass  # Don't care about result
        
        # Check metrics
        async with api_client.get('http://localhost:8000/metrics') as resp:
            text = await resp.text()
            assert 'csp_auth_login_attempts_total' in text
    
    async def test_file_upload_metrics(self, api_client):
        """Test file upload metrics"""
        # Create test file
        data = aiohttp.FormData()
        data.add_field('file',
                      b'test content',
                      filename='test.txt',
                      content_type='text/plain')
        
        # Upload file
        async with api_client.post('http://localhost:8000/api/files/upload',
                                 data=data) as resp:
            pass
        
        # Check metrics
        async with api_client.get('http://localhost:8000/metrics') as resp:
            text = await resp.text()
            assert 'csp_file_uploads_total' in text
            assert 'csp_file_upload_size_bytes' in text
    
    async def test_rate_limit_metrics(self, api_client):
        """Test rate limiting metrics"""
        # Make many requests to trigger rate limit
        for _ in range(150):
            async with api_client.get('http://localhost:8000/api/test') as resp:
                if resp.status == 429:
                    break
        
        # Check metrics
        async with api_client.get('http://localhost:8000/metrics') as resp:
            text = await resp.text()
            assert 'csp_rate_limit_hits_total' in text
    
    async def test_network_node_metrics(self, api_client):
        """Test network node metrics"""
        async with api_client.get('http://localhost:8080/metrics') as resp:
            assert resp.status == 200
            text = await resp.text()
            
            # Check for network metrics
            assert 'csp_peers_total' in text
            assert 'csp_uptime_seconds' in text
            assert 'csp_messages_sent_total' in text
    
    async def test_slo_calculation(self, api_client):
        """Test SLO calculation and metrics"""
        # Make some successful requests
        for _ in range(10):
            async with api_client.get('http://localhost:8000/health') as resp:
                assert resp.status == 200
        
        # Wait for SLO calculation
        await asyncio.sleep(2)
        
        # Check SLO metrics
        async with api_client.get('http://localhost:8000/metrics') as resp:
            text = await resp.text()
            assert 'csp_sli_availability' in text
            assert 'csp_slo_compliance' in text

def test_prometheus_targets():
    """Test that all Prometheus targets are scraped"""
    import requests
    
    # Query Prometheus targets
    resp = requests.get('http://localhost:9090/api/v1/targets')
    data = resp.json()
    
    # Check that all expected targets are up
    expected_jobs = [
        'csp-api',
        'csp-network-nodes',
        'postgres-exporter',
        'redis-exporter'
    ]
    
    active_jobs = set()
    for target in data['data']['activeTargets']:
        if target['health'] == 'up':
            active_jobs.add(target['labels']['job'])
    
    for job in expected_jobs:
        assert job in active_jobs, f"Job {job} is not being scraped"

def test_grafana_dashboards():
    """Test that Grafana dashboards have valid queries"""
    import requests
    
    # Get dashboards
    resp = requests.get('http://admin:admin@localhost:3000/api/search')
    dashboards = resp.json()
    
    for dashboard in dashboards:
        # Get dashboard details
        resp = requests.get(f'http://admin:admin@localhost:3000/api/dashboards/uid/{dashboard["uid"]}')
        data = resp.json()
        
        # Check that panels have valid queries
        for panel in data['dashboard'].get('panels', []):
            for target in panel.get('targets', []):
                expr = target.get('expr', '')
                # Verify no old metric names
                assert 'ecsp_' not in expr, f"Dashboard uses old metric name: {expr}"
                assert 'enhanced_csp_' not in expr, f"Dashboard uses old metric name: {expr}"

if __name__ == '__main__':
    # Run basic connectivity test
    print("Testing monitoring endpoints...")
    
    import requests
    
    endpoints = [
        ('http://localhost:8000/metrics', 'API metrics'),
        ('http://localhost:8080/metrics', 'Network node metrics'),
        ('http://localhost:9090/metrics', 'Prometheus metrics'),
        ('http://localhost:3000/api/health', 'Grafana health')
    ]
    
    for url, name in endpoints:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"✅ {name}: OK")
            else:
                print(f"❌ {name}: Status {resp.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
