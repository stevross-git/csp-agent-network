"""
Alert Correlation and Deduplication Engine
"""
import asyncio
import json
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging

from prometheus_client import Counter, Gauge, Histogram
import aioredis
import networkx as nx

logger = logging.getLogger(__name__)

# Metrics
alerts_correlated = Counter(
    'csp_alerts_correlated_total',
    'Total alerts correlated',
    ['correlation_type']
)

incidents_created = Counter(
    'csp_incidents_created_total',
    'Total incidents created',
    ['severity', 'category']
)

alert_noise_reduction = Gauge(
    'csp_alert_noise_reduction_ratio',
    'Alert noise reduction ratio'
)

correlation_duration = Histogram(
    'csp_correlation_duration_seconds',
    'Time taken for alert correlation'
)

class Alert:
    """Represents an alert"""
    
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('fingerprint', hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest())
        self.name = data.get('alertname', 'unknown')
        self.severity = data.get('labels', {}).get('severity', 'medium')
        self.category = data.get('labels', {}).get('category', 'unknown')
        self.service = data.get('labels', {}).get('service', 'unknown')
        self.instance = data.get('labels', {}).get('instance', 'unknown')
        self.timestamp = datetime.fromisoformat(
            data.get('startsAt', datetime.utcnow().isoformat())
        )
        self.labels = data.get('labels', {})
        self.annotations = data.get('annotations', {})
        self.value = data.get('value', 0)
        self.raw_data = data

class Incident:
    """Represents a correlated incident"""
    
    def __init__(self, incident_id: str):
        self.id = incident_id
        self.alerts: List[Alert] = []
        self.severity = 'low'
        self.category = 'unknown'
        self.services: Set[str] = set()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.title = ""
        self.description = ""
        self.root_cause = None
        self.impact_score = 0
        self.correlation_confidence = 0

class AlertCorrelator:
    """Correlates alerts into incidents"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
        # Correlation rules
        self.correlation_window = timedelta(minutes=5)
        self.correlation_rules = self._load_correlation_rules()
        
        # Dependency graph
        self.service_graph = self._build_service_graph()
        
        # Alert history
        self.alert_history: Dict[str, List[Alert]] = defaultdict(list)
        self.active_incidents: Dict[str, Incident] = {}
    
    def _load_correlation_rules(self) -> List[Dict]:
        """Load correlation rules"""
        return [
            {
                'name': 'cascade_failure',
                'description': 'Correlate cascading failures across services',
                'conditions': [
                    {'field': 'category', 'value': 'availability'},
                    {'time_window': 60, 'min_services': 2}
                ],
                'priority': 1
            },
            {
                'name': 'resource_exhaustion',
                'description': 'Correlate resource exhaustion alerts',
                'conditions': [
                    {'field': 'name', 'pattern': '(memory|cpu|disk).*high'},
                    {'field': 'service', 'same': True},
                    {'time_window': 300}
                ],
                'priority': 2
            },
            {
                'name': 'security_attack',
                'description': 'Correlate security-related alerts',
                'conditions': [
                    {'field': 'category', 'value': 'security'},
                    {'time_window': 120}
                ],
                'priority': 1
            },
            {
                'name': 'performance_degradation',
                'description': 'Correlate performance issues',
                'conditions': [
                    {'field': 'name', 'pattern': '(latency|response_time).*high'},
                    {'correlation': 'upstream_downstream'},
                    {'time_window': 180}
                ],
                'priority': 3
            }
        ]
    
    def _build_service_graph(self) -> nx.DiGraph:
        """Build service dependency graph"""
        G = nx.DiGraph()
        
        # Define service dependencies
        dependencies = [
            ('frontend', 'api'),
            ('api', 'auth-service'),
            ('api', 'database'),
            ('api', 'cache'),
            ('api', 'ai-service'),
            ('ai-service', 'vector-db'),
            ('auth-service', 'database'),
            ('auth-service', 'cache')
        ]
        
        G.add_edges_from(dependencies)
        return G
    
    async def initialize(self):
        """Initialize correlator"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("Alert correlator initialized")
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Optional[Incident]:
        """Process incoming alert and correlate with existing alerts"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            alert = Alert(alert_data)
            
            # Add to history
            self.alert_history[alert.service].append(alert)
            
            # Find correlations
            correlated_alerts = await self._find_correlations(alert)
            
            if correlated_alerts:
                # Create or update incident
                incident = await self._create_or_update_incident(
                    alert, correlated_alerts
                )
                
                # Track metrics
                alerts_correlated.labels(
                    correlation_type=incident.category
                ).inc()
                
                # Calculate noise reduction
                total_alerts = len(correlated_alerts) + 1
                noise_reduction = 1 - (1 / total_alerts)
                alert_noise_reduction.set(noise_reduction)
                
                return incident
            
            return None
            
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            correlation_duration.observe(duration)
    
    async def _find_correlations(self, alert: Alert) -> List[Alert]:
        """Find alerts that correlate with the given alert"""
        correlated = []
        
        for rule in self.correlation_rules:
            matches = await self._evaluate_rule(alert, rule)
            correlated.extend(matches)
        
        # Remove duplicates
        seen = set()
        unique_correlated = []
        for a in correlated:
            if a.id not in seen:
                seen.add(a.id)
                unique_correlated.append(a)
        
        return unique_correlated
    
    async def _evaluate_rule(self, alert: Alert, rule: Dict) -> List[Alert]:
        """Evaluate a correlation rule"""
        matches = []
        
        # Time window check
        time_window = rule.get('conditions', [{}])[0].get('time_window', 300)
        cutoff_time = alert.timestamp - timedelta(seconds=time_window)
        
        # Check all recent alerts
        for service_alerts in self.alert_history.values():
            for historical_alert in service_alerts:
                if historical_alert.timestamp < cutoff_time:
                    continue
                
                if historical_alert.id == alert.id:
                    continue
                
                # Evaluate conditions
                if self._match_conditions(alert, historical_alert, rule['conditions']):
                    matches.append(historical_alert)
        
        # Apply correlation-specific logic
        if 'correlation' in rule:
            if rule['correlation'] == 'upstream_downstream':
                matches = self._filter_by_service_dependency(alert, matches)
        
        return matches
    
    def _match_conditions(self, alert: Alert, other: Alert, conditions: List[Dict]) -> bool:
        """Check if alerts match correlation conditions"""
        for condition in conditions:
            if 'field' in condition:
                field = condition['field']
                
                if 'value' in condition:
                    # Exact match
                    if getattr(alert, field, None) != condition['value']:
                        return False
                    if getattr(other, field, None) != condition['value']:
                        return False
                
                elif 'same' in condition and condition['same']:
                    # Same value check
                    if getattr(alert, field, None) != getattr(other, field, None):
                        return False
                
                elif 'pattern' in condition:
                    # Pattern match
                    import re
                    pattern = condition['pattern']
                    if not re.search(pattern, getattr(alert, field, ''), re.I):
                        return False
                    if not re.search(pattern, getattr(other, field, ''), re.I):
                        return False
        
        return True
    
    def _filter_by_service_dependency(self, alert: Alert, matches: List[Alert]) -> List[Alert]:
        """Filter alerts based on service dependencies"""
        filtered = []
        
        if alert.service in self.service_graph:
            # Get upstream and downstream services
            upstream = list(self.service_graph.predecessors(alert.service))
            downstream = list(self.service_graph.successors(alert.service))
            related_services = upstream + downstream + [alert.service]
            
            for match in matches:
                if match.service in related_services:
                    filtered.append(match)
        else:
            # If service not in graph, include all matches
            filtered = matches
        
        return filtered
    
    async def _create_or_update_incident(self, alert: Alert, 
                                       correlated: List[Alert]) -> Incident:
        """Create or update an incident"""
        # Check for existing incident
        incident_key = f"incident:{alert.service}:{alert.category}"
        existing_id = await self.redis.get(incident_key)
        
        if existing_id and existing_id.decode() in self.active_incidents:
            # Update existing incident
            incident = self.active_incidents[existing_id.decode()]
            incident.alerts.append(alert)
            incident.alerts.extend(correlated)
            incident.updated_at = datetime.utcnow()
        else:
            # Create new incident
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            incident = Incident(incident_id)
            incident.alerts = [alert] + correlated
            
            # Store incident
            self.active_incidents[incident_id] = incident
            await self.redis.setex(incident_key, 3600, incident_id)
            
            # Track metric
            incidents_created.labels(
                severity=incident.severity,
                category=incident.category
            ).inc()
        
        # Update incident properties
        incident.services = {a.service for a in incident.alerts}
        incident.severity = max(a.severity for a in incident.alerts)
        incident.category = self._determine_category(incident.alerts)
        incident.title = self._generate_title(incident)
        incident.description = self._generate_description(incident)
        incident.impact_score = self._calculate_impact(incident)
        incident.correlation_confidence = self._calculate_confidence(incident)
        
        # Attempt root cause analysis
        incident.root_cause = await self._analyze_root_cause(incident)
        
        return incident
    
    def _determine_category(self, alerts: List[Alert]) -> str:
        """Determine incident category from alerts"""
        categories = [a.category for a in alerts]
        # Return most common category
        return max(set(categories), key=categories.count)
    
    def _generate_title(self, incident: Incident) -> str:
        """Generate incident title"""
        if len(incident.services) == 1:
            return f"{incident.severity.upper()}: {list(incident.services)[0]} - {incident.category}"
        else:
            return f"{incident.severity.upper()}: Multiple services affected - {incident.category}"
    
    def _generate_description(self, incident: Incident) -> str:
        """Generate incident description"""
        alert_summary = defaultdict(int)
        for alert in incident.alerts:
            alert_summary[alert.name] += 1
        
        desc_parts = [
            f"Incident affecting {len(incident.services)} service(s).",
            f"Total alerts: {len(incident.alerts)}",
            "",
            "Alert breakdown:"
        ]
        
        for alert_name, count in sorted(alert_summary.items(), 
                                      key=lambda x: x[1], reverse=True):
            desc_parts.append(f"  - {alert_name}: {count}")
        
        return "\n".join(desc_parts)
    
    def _calculate_impact(self, incident: Incident) -> float:
        """Calculate incident impact score"""
        # Factors: severity, number of services, alert count
        severity_scores = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
        
        severity_score = severity_scores.get(incident.severity, 0.5)
        service_score = min(len(incident.services) / 5, 1.0)  # Normalize to 0-1
        alert_score = min(len(incident.alerts) / 10, 1.0)  # Normalize to 0-1
        
        # Weighted average
        impact = (severity_score * 0.5 + service_score * 0.3 + alert_score * 0.2)
        
        return round(impact, 2)
    
    def _calculate_confidence(self, incident: Incident) -> float:
        """Calculate correlation confidence"""
        # Factors: time proximity, service relationships, pattern matching
        confidence = 0.0
        
        # Time proximity
        if incident.alerts:
            time_range = max(a.timestamp for a in incident.alerts) - \
                        min(a.timestamp for a in incident.alerts)
            if time_range < timedelta(minutes=1):
                confidence += 0.4
            elif time_range < timedelta(minutes=5):
                confidence += 0.2
        
        # Service relationships
        if len(incident.services) > 1:
            # Check if services are related in dependency graph
            service_list = list(incident.services)
            for i in range(len(service_list)):
                for j in range(i + 1, len(service_list)):
                    if nx.has_path(self.service_graph, service_list[i], service_list[j]):
                        confidence += 0.2
                        break
        
        # Pattern matching
        alert_names = [a.name for a in incident.alerts]
        if len(set(alert_names)) < len(alert_names) * 0.5:
            # Many similar alerts
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def _analyze_root_cause(self, incident: Incident) -> Optional[str]:
        """Attempt to identify root cause"""
        # Simple heuristic-based root cause analysis
        
        # Check for cascade pattern
        if len(incident.services) > 2:
            # Sort alerts by time
            sorted_alerts = sorted(incident.alerts, key=lambda a: a.timestamp)
            first_service = sorted_alerts[0].service
            
            # Check if first service is upstream of others
            is_upstream = True
            for service in incident.services:
                if service != first_service:
                    if not nx.has_path(self.service_graph, first_service, service):
                        is_upstream = False
                        break
            
            if is_upstream:
                return f"Cascade failure originating from {first_service}"
        
        # Check for resource exhaustion
        resource_alerts = [a for a in incident.alerts 
                          if 'memory' in a.name or 'cpu' in a.name or 'disk' in a.name]
        if len(resource_alerts) > len(incident.alerts) * 0.7:
            return "Resource exhaustion across services"
        
        # Check for security attack pattern
        security_alerts = [a for a in incident.alerts if a.category == 'security']
        if len(security_alerts) > len(incident.alerts) * 0.8:
            return "Coordinated security attack"
        
        return None

# Intelligent alert routing
class AlertRouter:
    """Routes alerts based on content, time, and expertise"""
    
    def __init__(self):
        self.routing_rules = self._load_routing_rules()
        self.on_call_schedule = {}
        self.expertise_map = self._load_expertise_map()
    
    def _load_routing_rules(self) -> Dict:
        """Load routing rules"""
        return {
            'business_hours': {
                'start': 9,
                'end': 17,
                'timezone': 'UTC',
                'channels': ['slack', 'email']
            },
            'after_hours': {
                'channels': {
                    'critical': ['pagerduty', 'phone'],
                    'high': ['slack', 'email'],
                    'medium': ['email'],
                    'low': ['email']
                }
            },
            'escalation': {
                'critical': {
                    'initial_wait': 5,  # minutes
                    'escalation_levels': [
                        {'wait': 5, 'notify': ['team_lead']},
                        {'wait': 10, 'notify': ['manager']},
                        {'wait': 20, 'notify': ['director']}
                    ]
                }
            }
        }
    
    def _load_expertise_map(self) -> Dict:
        """Load expertise mapping"""
        return {
            'database': ['dba-team', 'john.doe@company.com'],
            'security': ['security-team', 'security-oncall@company.com'],
            'api': ['backend-team', 'api-oncall@company.com'],
            'ai': ['ml-team', 'ai-oncall@company.com'],
            'infrastructure': ['sre-team', 'sre-oncall@company.com']
        }
    
    async def route_incident(self, incident: Incident) -> Dict[str, Any]:
        """Route incident to appropriate channels and people"""
        routing_decision = {
            'incident_id': incident.id,
            'channels': [],
            'recipients': [],
            'escalation_plan': None
        }
        
        # Determine channels based on time and severity
        if self._is_business_hours():
            routing_decision['channels'] = self.routing_rules['business_hours']['channels']
        else:
            severity_channels = self.routing_rules['after_hours']['channels']
            routing_decision['channels'] = severity_channels.get(
                incident.severity, ['email']
            )
        
        # Add recipients based on expertise
        for service in incident.services:
            if service in self.expertise_map:
                routing_decision['recipients'].extend(self.expertise_map[service])
        
        # Remove duplicates
        routing_decision['recipients'] = list(set(routing_decision['recipients']))
        
        # Create escalation plan for critical incidents
        if incident.severity == 'critical':
            routing_decision['escalation_plan'] = self._create_escalation_plan(incident)
        
        return routing_decision
    
    def _is_business_hours(self) -> bool:
        """Check if current time is business hours"""
        from datetime import datetime
        import pytz
        
        tz = pytz.timezone(self.routing_rules['business_hours']['timezone'])
        now = datetime.now(tz)
        
        start_hour = self.routing_rules['business_hours']['start']
        end_hour = self.routing_rules['business_hours']['end']
        
        return start_hour <= now.hour < end_hour and now.weekday() < 5
    
    def _create_escalation_plan(self, incident: Incident) -> Dict:
        """Create escalation plan for incident"""
        escalation_config = self.routing_rules['escalation'][incident.severity]
        
        plan = {
            'initial_notification': datetime.utcnow().isoformat(),
            'levels': []
        }
        
        current_time = datetime.utcnow()
        for level in escalation_config['escalation_levels']:
            escalation_time = current_time + timedelta(minutes=level['wait'])
            plan['levels'].append({
                'time': escalation_time.isoformat(),
                'notify': level['notify']
            })
        
        return plan
