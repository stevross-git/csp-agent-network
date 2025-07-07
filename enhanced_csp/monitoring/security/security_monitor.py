"""
Security Monitoring and Threat Detection System
"""
import asyncio
import re
import json
from typing import Dict, List, Pattern, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import hashlib

from prometheus_client import Counter, Gauge, Histogram
import aioredis

logger = logging.getLogger(__name__)

# Security metrics
security_events = Counter(
    'csp_security_events_total',
    'Total security events detected',
    ['event_type', 'severity', 'source', 'action']
)

threat_score = Gauge(
    'csp_threat_score',
    'Current threat score (0-100)',
    ['category']
)

blocked_requests = Counter(
    'csp_blocked_requests_total',
    'Total requests blocked',
    ['reason', 'source_ip']
)

security_scan_duration = Histogram(
    'csp_security_scan_duration_seconds',
    'Time taken for security scanning',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

class ThreatPattern:
    """Represents a threat detection pattern"""
    
    def __init__(self, name: str, pattern: str, severity: str, category: str):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.severity = severity
        self.category = category
        self.matches = 0

class SecurityMonitor:
    """Advanced security monitoring and threat detection"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
        # Threat patterns
        self.threat_patterns = self._load_threat_patterns()
        
        # IP reputation cache
        self.ip_reputation_cache = {}
        
        # Rate limiting
        self.rate_limits = {
            'api_calls': {'window': 60, 'limit': 100},
            'auth_attempts': {'window': 300, 'limit': 5},
            'file_uploads': {'window': 3600, 'limit': 50}
        }
        
        # Threat scoring weights
        self.threat_weights = {
            'sql_injection': 25,
            'xss': 20,
            'path_traversal': 20,
            'command_injection': 30,
            'auth_bypass': 25,
            'brute_force': 15,
            'dos_attack': 20,
            'data_exfiltration': 30
        }
    
    def _load_threat_patterns(self) -> List[ThreatPattern]:
        """Load threat detection patterns"""
        patterns = [
            # SQL Injection
            ThreatPattern(
                "sql_injection_union",
                r"(union|select|insert|update|delete|drop|create|alter|exec|execute).*?(from|where|table|database)",
                "high",
                "sql_injection"
            ),
            ThreatPattern(
                "sql_injection_comment",
                r"(--|#|\/\*|\*\/|@@|@)",
                "medium",
                "sql_injection"
            ),
            
            # XSS
            ThreatPattern(
                "xss_script_tag",
                r"<script[^>]*>.*?</script>",
                "high",
                "xss"
            ),
            ThreatPattern(
                "xss_event_handler",
                r"(onclick|onerror|onload|onmouseover|onfocus|onblur)=",
                "high",
                "xss"
            ),
            
            # Path Traversal
            ThreatPattern(
                "path_traversal",
                r"(\.\./|\.\.\\|%2e%2e%2f|%252e%252e%252f)",
                "high",
                "path_traversal"
            ),
            
            # Command Injection
            ThreatPattern(
                "command_injection",
                r"(;|\||&|`|\$\(|<\(|>\(|\$\{)",
                "critical",
                "command_injection"
            ),
            
            # Authentication Bypass
            ThreatPattern(
                "auth_bypass_null",
                r"(admin'--|' or '1'='1|' or 1=1--|\" or \"1\"=\"1)",
                "critical",
                "auth_bypass"
            ),
            
            # Suspicious User Agents
            ThreatPattern(
                "scanner_bot",
                r"(nikto|sqlmap|nmap|masscan|zap|burp|acunetix)",
                "medium",
                "scanner"
            ),
            
            # Data Exfiltration
            ThreatPattern(
                "base64_exfil",
                r"[A-Za-z0-9+/]{50,}={0,2}",
                "medium",
                "data_exfiltration"
            )
        ]
        
        return patterns
    
    async def initialize(self):
        """Initialize security monitor"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("Security monitor initialized")
    
    async def scan_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan a request for security threats"""
        start_time = asyncio.get_event_loop().time()
        threats_found = []
        
        try:
            # Extract data to scan
            scan_targets = {
                'path': request_data.get('path', ''),
                'query_params': json.dumps(request_data.get('query_params', {})),
                'body': json.dumps(request_data.get('body', {})),
                'headers': json.dumps(request_data.get('headers', {})),
                'user_agent': request_data.get('headers', {}).get('user-agent', '')
            }
            
            # Scan for threat patterns
            for target_name, target_value in scan_targets.items():
                for pattern in self.threat_patterns:
                    if pattern.pattern.search(str(target_value)):
                        threat = {
                            'pattern': pattern.name,
                            'category': pattern.category,
                            'severity': pattern.severity,
                            'location': target_name,
                            'matched_value': target_value[:100]  # Truncate for logging
                        }
                        threats_found.append(threat)
                        pattern.matches += 1
                        
                        # Log security event
                        security_events.labels(
                            event_type=pattern.category,
                            severity=pattern.severity,
                            source=request_data.get('source_ip', 'unknown'),
                            action='detected'
                        ).inc()
            
            # Check rate limits
            rate_limit_violations = await self._check_rate_limits(request_data)
            if rate_limit_violations:
                threats_found.extend(rate_limit_violations)
            
            # Check IP reputation
            ip_reputation = await self._check_ip_reputation(
                request_data.get('source_ip', '')
            )
            if ip_reputation and ip_reputation['risk_score'] > 0.7:
                threats_found.append({
                    'pattern': 'malicious_ip',
                    'category': 'reputation',
                    'severity': 'high',
                    'location': 'source_ip',
                    'risk_score': ip_reputation['risk_score']
                })
            
            # Calculate threat score
            total_score = self._calculate_threat_score(threats_found)
            
            # Update metrics
            threat_score.labels(category='overall').set(total_score)
            
            # Determine action
            action = 'allow'
            if total_score >= 70:
                action = 'block'
                blocked_requests.labels(
                    reason='high_threat_score',
                    source_ip=request_data.get('source_ip', 'unknown')
                ).inc()
            elif total_score >= 40:
                action = 'challenge'
            
            result = {
                'action': action,
                'threat_score': total_score,
                'threats': threats_found,
                'scan_time': asyncio.get_event_loop().time() - start_time
            }
            
            # Log high-risk requests
            if total_score >= 40:
                logger.warning(f"High-risk request detected: {result}")
            
            return result
            
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            security_scan_duration.observe(duration)
    
    async def _check_rate_limits(self, request_data: Dict[str, Any]) -> List[Dict]:
        """Check rate limits for the request"""
        violations = []
        source_ip = request_data.get('source_ip', 'unknown')
        
        for limit_type, config in self.rate_limits.items():
            key = f"rate_limit:{limit_type}:{source_ip}"
            
            # Increment counter
            current = await self.redis.incr(key)
            
            # Set expiry on first increment
            if current == 1:
                await self.redis.expire(key, config['window'])
            
            # Check if limit exceeded
            if current > config['limit']:
                violations.append({
                    'pattern': f'rate_limit_{limit_type}',
                    'category': 'dos_attack',
                    'severity': 'medium',
                    'location': 'rate_limit',
                    'current_rate': current,
                    'limit': config['limit']
                })
                
                security_events.labels(
                    event_type='rate_limit_exceeded',
                    severity='medium',
                    source=source_ip,
                    action='detected'
                ).inc()
        
        return violations
    
    async def _check_ip_reputation(self, ip: str) -> Optional[Dict]:
        """Check IP reputation"""
        if not ip or ip in self.ip_reputation_cache:
            return self.ip_reputation_cache.get(ip)
        
        # Check against known bad IPs (in production, use threat intelligence feeds)
        bad_ip_patterns = [
            r"^10\.0\.0\.",  # Example: internal testing
            r"^192\.168\.",  # Example: local network
        ]
        
        risk_score = 0.0
        reasons = []
        
        for pattern in bad_ip_patterns:
            if re.match(pattern, ip):
                risk_score += 0.5
                reasons.append(f"Matches pattern: {pattern}")
        
        # Check failed auth attempts
        auth_fails_key = f"auth_fails:{ip}"
        auth_fails = await self.redis.get(auth_fails_key)
        if auth_fails and int(auth_fails) > 5:
            risk_score += 0.3
            reasons.append(f"High auth failures: {auth_fails}")
        
        reputation = {
            'ip': ip,
            'risk_score': min(risk_score, 1.0),
            'reasons': reasons,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache for 1 hour
        self.ip_reputation_cache[ip] = reputation
        
        return reputation
    
    def _calculate_threat_score(self, threats: List[Dict]) -> float:
        """Calculate overall threat score"""
        if not threats:
            return 0.0
        
        score = 0.0
        severity_multipliers = {
            'critical': 1.5,
            'high': 1.0,
            'medium': 0.5,
            'low': 0.25
        }
        
        for threat in threats:
            category = threat.get('category', 'unknown')
            severity = threat.get('severity', 'medium')
            
            base_weight = self.threat_weights.get(category, 10)
            multiplier = severity_multipliers.get(severity, 0.5)
            
            score += base_weight * multiplier
        
        # Normalize to 0-100
        return min(score, 100.0)
    
    async def analyze_auth_attempt(self, username: str, success: bool, 
                                  source_ip: str, metadata: Dict = None):
        """Analyze authentication attempt for suspicious patterns"""
        # Track failed attempts
        if not success:
            fails_key = f"auth_fails:{source_ip}"
            fails = await self.redis.incr(fails_key)
            await self.redis.expire(fails_key, 3600)  # 1 hour
            
            if fails >= 5:
                security_events.labels(
                    event_type='brute_force',
                    severity='high',
                    source=source_ip,
                    action='detected'
                ).inc()
                
                # Add to temporary blacklist
                blacklist_key = f"blacklist:{source_ip}"
                await self.redis.setex(blacklist_key, 3600, "brute_force")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"admin.*test",
            r"test.*admin",
            r"root",
            r"administrator"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, username, re.I):
                security_events.labels(
                    event_type='suspicious_username',
                    severity='medium',
                    source=source_ip,
                    action='detected'
                ).inc()
                break
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        # Calculate threat levels
        threat_levels = {}
        for category in self.threat_weights.keys():
            level = threat_score.labels(category=category)._value.get() or 0
            threat_levels[category] = level
        
        # Get pattern match statistics
        pattern_stats = [
            {
                'name': p.name,
                'category': p.category,
                'matches': p.matches
            }
            for p in self.threat_patterns
        ]
        
        # Sort by matches
        pattern_stats.sort(key=lambda x: x['matches'], reverse=True)
        
        return {
            'overall_threat_score': threat_score.labels(category='overall')._value.get() or 0,
            'threat_levels': threat_levels,
            'top_patterns': pattern_stats[:10],
            'active_blacklists': await self._get_active_blacklists(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_active_blacklists(self) -> List[str]:
        """Get currently blacklisted IPs"""
        blacklists = []
        cursor = b'0'
        
        while cursor:
            cursor, keys = await self.redis.scan(
                cursor, match=b'blacklist:*', count=100
            )
            for key in keys:
                ip = key.decode('utf-8').split(':')[1]
                blacklists.append(ip)
            
            if cursor == b'0':
                break
        
        return blacklists

# API endpoint integration
from fastapi import Request, HTTPException
from functools import wraps

security_monitor = None

async def get_security_monitor() -> SecurityMonitor:
    """Get or create security monitor instance"""
    global security_monitor
    if security_monitor is None:
        security_monitor = SecurityMonitor()
        await security_monitor.initialize()
    return security_monitor

def secure_endpoint(severity_threshold: int = 40):
    """Decorator to add security scanning to endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            monitor = await get_security_monitor()
            
            # Prepare request data for scanning
            request_data = {
                'path': str(request.url.path),
                'query_params': dict(request.query_params),
                'headers': dict(request.headers),
                'source_ip': request.client.host if request.client else 'unknown',
                'method': request.method
            }
            
            # Get body if present
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    body = await request.json()
                    request_data['body'] = body
                except:
                    pass
            
            # Scan request
            scan_result = await monitor.scan_request(request_data)
            
            # Take action based on result
            if scan_result['action'] == 'block':
                raise HTTPException(
                    status_code=403,
                    detail="Request blocked due to security concerns"
                )
            elif scan_result['action'] == 'challenge':
                # In production, implement CAPTCHA or similar
                request.state.security_challenge = True
            
            # Add security context to request
            request.state.security_scan = scan_result
            
            # Execute original function
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator
