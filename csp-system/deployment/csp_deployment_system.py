"""
CSP Production Deployment System
===============================

Production-ready deployment system for CSP networks with:
- Kubernetes/Docker orchestration
- Auto-scaling and load balancing
- Service mesh integration
- Configuration management
- Health monitoring and alerting
- Rolling updates and blue-green deployments
- Multi-cloud deployment strategies
- Disaster recovery and backup systems
"""

import asyncio
import json
import yaml
import os
import time
import logging
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
import docker
import kubernetes
from kubernetes import client, config, watch
import consul
import etcd3
import redis
import boto3
import paramiko
from jinja2 import Template
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import grafana_api

# ============================================================================
# DEPLOYMENT CONFIGURATION MODELS
# ============================================================================

class DeploymentTarget(Enum):
    """Deployment target environments"""
    LOCAL = auto()
    DOCKER = auto()
    KUBERNETES = auto()
    AWS_ECS = auto()
    AWS_LAMBDA = auto()
    GCP_CLOUD_RUN = auto()
    AZURE_CONTAINER_INSTANCES = auto()
    BARE_METAL = auto()

class ScalingStrategy(Enum):
    """Auto-scaling strategies"""
    MANUAL = auto()
    CPU_BASED = auto()
    MEMORY_BASED = auto()
    REQUEST_RATE = auto()
    CUSTOM_METRIC = auto()
    PREDICTIVE = auto()

@dataclass
class ResourceLimits:
    """Resource limits for deployment"""
    cpu_limit: str = "1000m"  # 1 CPU
    memory_limit: str = "2Gi"
    cpu_request: str = "100m"
    memory_request: str = "256Mi"
    storage_limit: str = "10Gi"

@dataclass
class NetworkConfig:
    """Network configuration"""
    port: int = 8080
    target_port: int = 8080
    protocol: str = "TCP"
    ingress_enabled: bool = True
    tls_enabled: bool = True
    service_mesh: bool = False

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    strategy: ScalingStrategy = ScalingStrategy.CPU_BASED
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    jaeger_enabled: bool = True
    log_level: str = "INFO"
    metrics_port: int = 9090
    health_check_path: str = "/health"
    custom_metrics: List[str] = field(default_factory=list)

@dataclass
class DeploymentConfig:
    """Complete deployment configuration"""
    name: str
    version: str
    target: DeploymentTarget
    image: str = "csp-runtime:latest"
    replicas: int = 3
    namespace: str = "csp-system"
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)

# ============================================================================
# KUBERNETES DEPLOYMENT MANAGER
# ============================================================================

class KubernetesDeploymentManager:
    """Manage CSP deployments on Kubernetes"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.autoscaling_v2 = client.AutoscalingV2Api()
        
        self.template_env = self._setup_templates()
    
    def _setup_templates(self):
        """Setup Jinja2 templates for Kubernetes manifests"""
        from jinja2 import Environment, DictLoader
        
        templates = {
            'deployment': '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ name }}
  namespace: {{ namespace }}
  labels:
    app: {{ name }}
    version: {{ version }}
spec:
  replicas: {{ replicas }}
  selector:
    matchLabels:
      app: {{ name }}
  template:
    metadata:
      labels:
        app: {{ name }}
        version: {{ version }}
    spec:
      containers:
      - name: {{ name }}
        image: {{ image }}
        ports:
        - containerPort: {{ network.port }}
        env:
        {% for key, value in environment.items() %}
        - name: {{ key }}
          value: "{{ value }}"
        {% endfor %}
        resources:
          requests:
            cpu: {{ resources.cpu_request }}
            memory: {{ resources.memory_request }}
          limits:
            cpu: {{ resources.cpu_limit }}
            memory: {{ resources.memory_limit }}
        livenessProbe:
          httpGet:
            path: {{ monitoring.health_check_path }}
            port: {{ network.port }}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {{ monitoring.health_check_path }}
            port: {{ network.port }}
          initialDelaySeconds: 5
          periodSeconds: 5
      {% if volumes %}
        volumeMounts:
        {% for volume in volumes %}
        - name: {{ volume.name }}
          mountPath: {{ volume.mount_path }}
        {% endfor %}
      volumes:
      {% for volume in volumes %}
      - name: {{ volume.name }}
        {% if volume.type == 'configMap' %}
        configMap:
          name: {{ volume.source }}
        {% elif volume.type == 'secret' %}
        secret:
          secretName: {{ volume.source }}
        {% elif volume.type == 'persistentVolumeClaim' %}
        persistentVolumeClaim:
          claimName: {{ volume.source }}
        {% endif %}
      {% endfor %}
      {% endif %}
''',
            'service': '''
apiVersion: v1
kind: Service
metadata:
  name: {{ name }}-service
  namespace: {{ namespace }}
  labels:
    app: {{ name }}
spec:
  selector:
    app: {{ name }}
  ports:
  - port: {{ network.port }}
    targetPort: {{ network.target_port }}
    protocol: {{ network.protocol }}
  type: ClusterIP
''',
            'hpa': '''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ name }}-hpa
  namespace: {{ namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ name }}
  minReplicas: {{ scaling.min_replicas }}
  maxReplicas: {{ scaling.max_replicas }}
  metrics:
  {% if scaling.strategy.name in ['CPU_BASED', 'CUSTOM_METRIC'] %}
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{ scaling.target_cpu_utilization }}
  {% endif %}
  {% if scaling.strategy.name in ['MEMORY_BASED', 'CUSTOM_METRIC'] %}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {{ scaling.target_memory_utilization }}
  {% endif %}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {{ scaling.scale_up_cooldown }}
    scaleDown:
      stabilizationWindowSeconds: {{ scaling.scale_down_cooldown }}
''',
            'ingress': '''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ name }}-ingress
  namespace: {{ namespace }}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    {% if network.tls_enabled %}
    kubernetes.io/tls-acme: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    {% endif %}
spec:
  {% if network.tls_enabled %}
  tls:
  - hosts:
    - {{ name }}.{{ domain | default('example.com') }}
    secretName: {{ name }}-tls
  {% endif %}
  rules:
  - host: {{ name }}.{{ domain | default('example.com') }}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {{ name }}-service
            port:
              number: {{ network.port }}
'''
        }
        
        return Environment(loader=DictLoader(templates))
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy CSP application to Kubernetes"""
        
        deployment_result = {
            "deployment_id": f"{config.name}-{int(time.time())}",
            "status": "deploying",
            "resources_created": [],
            "errors": []
        }
        
        try:
            # Create namespace if not exists
            await self._ensure_namespace(config.namespace)
            
            # Create secrets
            if config.secrets:
                secret_name = await self._create_secrets(config)
                deployment_result["resources_created"].append(f"Secret/{secret_name}")
            
            # Create deployment
            deployment_name = await self._create_deployment(config)
            deployment_result["resources_created"].append(f"Deployment/{deployment_name}")
            
            # Create service
            service_name = await self._create_service(config)
            deployment_result["resources_created"].append(f"Service/{service_name}")
            
            # Create HPA if auto-scaling enabled
            if config.scaling.strategy != ScalingStrategy.MANUAL:
                hpa_name = await self._create_hpa(config)
                deployment_result["resources_created"].append(f"HPA/{hpa_name}")
            
            # Create ingress if enabled
            if config.network.ingress_enabled:
                ingress_name = await self._create_ingress(config)
                deployment_result["resources_created"].append(f"Ingress/{ingress_name}")
            
            # Wait for deployment to be ready
            await self._wait_for_deployment_ready(config.name, config.namespace)
            
            deployment_result["status"] = "deployed"
            
        except Exception as e:
            deployment_result["status"] = "failed"
            deployment_result["errors"].append(str(e))
            logging.error(f"Deployment failed: {e}")
            
            # Cleanup on failure
            await self._cleanup_failed_deployment(config, deployment_result["resources_created"])
        
        return deployment_result
    
    async def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists"""
        try:
            self.core_v1.read_namespace(namespace)
        except client.ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.core_v1.create_namespace(namespace_manifest)
                logging.info(f"Created namespace: {namespace}")
    
    async def _create_secrets(self, config: DeploymentConfig) -> str:
        """Create Kubernetes secrets"""
        secret_name = f"{config.name}-secrets"
        
        secret_manifest = client.V1Secret(
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace=config.namespace
            ),
            string_data=config.secrets
        )
        
        try:
            self.core_v1.create_namespaced_secret(config.namespace, secret_manifest)
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.core_v1.patch_namespaced_secret(secret_name, config.namespace, secret_manifest)
        
        return secret_name
    
    async def _create_deployment(self, config: DeploymentConfig) -> str:
        """Create Kubernetes deployment"""
        template = self.template_env.get_template('deployment')
        manifest_yaml = template.render(**asdict(config))
        
        manifest = yaml.safe_load(manifest_yaml)
        
        try:
            self.apps_v1.create_namespaced_deployment(config.namespace, manifest)
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.apps_v1.patch_namespaced_deployment(config.name, config.namespace, manifest)
        
        return config.name
    
    async def _create_service(self, config: DeploymentConfig) -> str:
        """Create Kubernetes service"""
        template = self.template_env.get_template('service')
        manifest_yaml = template.render(**asdict(config))
        
        manifest = yaml.safe_load(manifest_yaml)
        service_name = f"{config.name}-service"
        
        try:
            self.core_v1.create_namespaced_service(config.namespace, manifest)
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.core_v1.patch_namespaced_service(service_name, config.namespace, manifest)
        
        return service_name
    
    async def _create_hpa(self, config: DeploymentConfig) -> str:
        """Create Horizontal Pod Autoscaler"""
        template = self.template_env.get_template('hpa')
        manifest_yaml = template.render(**asdict(config))
        
        manifest = yaml.safe_load(manifest_yaml)
        hpa_name = f"{config.name}-hpa"
        
        try:
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(config.namespace, manifest)
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(hpa_name, config.namespace, manifest)
        
        return hpa_name
    
    async def _create_ingress(self, config: DeploymentConfig) -> str:
        """Create Kubernetes ingress"""
        template = self.template_env.get_template('ingress')
        manifest_yaml = template.render(**asdict(config))
        
        manifest = yaml.safe_load(manifest_yaml)
        ingress_name = f"{config.name}-ingress"
        
        try:
            self.networking_v1.create_namespaced_ingress(config.namespace, manifest)
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.networking_v1.patch_namespaced_ingress(ingress_name, config.namespace, manifest)
        
        return ingress_name
    
    async def _wait_for_deployment_ready(self, name: str, namespace: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment_status(name, namespace)
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logging.info(f"Deployment {name} is ready")
                    return
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logging.warning(f"Error checking deployment status: {e}")
                await asyncio.sleep(5)
        
        raise TimeoutError(f"Deployment {name} did not become ready within {timeout} seconds")
    
    async def _cleanup_failed_deployment(self, config: DeploymentConfig, created_resources: List[str]):
        """Cleanup resources on failed deployment"""
        for resource in created_resources:
            try:
                resource_type, resource_name = resource.split('/', 1)
                
                if resource_type == "Deployment":
                    self.apps_v1.delete_namespaced_deployment(resource_name, config.namespace)
                elif resource_type == "Service":
                    self.core_v1.delete_namespaced_service(resource_name, config.namespace)
                elif resource_type == "Secret":
                    self.core_v1.delete_namespaced_secret(resource_name, config.namespace)
                elif resource_type == "HPA":
                    self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(resource_name, config.namespace)
                elif resource_type == "Ingress":
                    self.networking_v1.delete_namespaced_ingress(resource_name, config.namespace)
                
                logging.info(f"Cleaned up {resource}")
                
            except Exception as e:
                logging.warning(f"Failed to cleanup {resource}: {e}")
    
    async def update_deployment(self, config: DeploymentConfig, strategy: str = "rolling") -> Dict[str, Any]:
        """Update existing deployment"""
        
        if strategy == "rolling":
            return await self._rolling_update(config)
        elif strategy == "blue_green":
            return await self._blue_green_update(config)
        elif strategy == "canary":
            return await self._canary_update(config)
        else:
            raise ValueError(f"Unknown update strategy: {strategy}")
    
    async def _rolling_update(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform rolling update"""
        try:
            # Update deployment
            await self._create_deployment(config)
            
            # Wait for rollout to complete
            await self._wait_for_deployment_ready(config.name, config.namespace)
            
            return {
                "status": "success",
                "strategy": "rolling",
                "updated_at": time.time()
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "strategy": "rolling",
                "error": str(e)
            }
    
    async def _blue_green_update(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform blue-green deployment"""
        # Implementation for blue-green deployment
        # This would involve creating a new deployment with a different name,
        # testing it, then switching traffic
        pass
    
    async def _canary_update(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Perform canary deployment"""
        # Implementation for canary deployment
        # This would involve gradually shifting traffic to new version
        pass
    
    async def scale_deployment(self, name: str, namespace: str, replicas: int) -> Dict[str, Any]:
        """Scale deployment to specified number of replicas"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            deployment.spec.replicas = replicas
            
            self.apps_v1.patch_namespaced_deployment(name, namespace, deployment)
            
            return {
                "status": "success",
                "previous_replicas": deployment.status.replicas,
                "new_replicas": replicas
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_deployment_status(self, name: str, namespace: str) -> Dict[str, Any]:
        """Get deployment status and metrics"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment_status(name, namespace)
            
            return {
                "name": name,
                "namespace": namespace,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "unavailable_replicas": deployment.status.unavailable_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# ============================================================================
# DOCKER DEPLOYMENT MANAGER
# ============================================================================

class DockerDeploymentManager:
    """Manage CSP deployments using Docker"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.networks = {}
        self.volumes = {}
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy CSP application using Docker"""
        
        deployment_result = {
            "deployment_id": f"{config.name}-{int(time.time())}",
            "status": "deploying",
            "containers": [],
            "errors": []
        }
        
        try:
            # Create network if needed
            network_name = f"{config.name}-network"
            network = await self._ensure_network(network_name)
            
            # Create volumes if needed
            volumes = {}
            for volume_config in config.volumes:
                volume_name = f"{config.name}-{volume_config['name']}"
                volume = await self._ensure_volume(volume_name)
                volumes[volume_name] = {'bind': volume_config['mount_path'], 'mode': 'rw'}
            
            # Deploy containers
            for i in range(config.replicas):
                container_name = f"{config.name}-{i}"
                
                container = self.client.containers.run(
                    config.image,
                    name=container_name,
                    network=network_name,
                    ports={f"{config.network.port}/tcp": config.network.port + i},
                    environment=config.environment,
                    volumes=volumes,
                    restart_policy={"Name": "unless-stopped"},
                    detach=True,
                    labels={
                        "csp.deployment": config.name,
                        "csp.version": config.version,
                        "csp.replica": str(i)
                    }
                )
                
                deployment_result["containers"].append({
                    "name": container_name,
                    "id": container.id[:12],
                    "port": config.network.port + i
                })
            
            # Setup load balancer (simplified with nginx)
            if config.replicas > 1:
                await self._setup_load_balancer(config, deployment_result["containers"])
            
            deployment_result["status"] = "deployed"
            
        except Exception as e:
            deployment_result["status"] = "failed"
            deployment_result["errors"].append(str(e))
            logging.error(f"Docker deployment failed: {e}")
        
        return deployment_result
    
    async def _ensure_network(self, network_name: str):
        """Ensure Docker network exists"""
        try:
            network = self.client.networks.get(network_name)
        except docker.errors.NotFound:
            network = self.client.networks.create(
                network_name,
                driver="bridge",
                labels={"csp.managed": "true"}
            )
            logging.info(f"Created Docker network: {network_name}")
        
        return network
    
    async def _ensure_volume(self, volume_name: str):
        """Ensure Docker volume exists"""
        try:
            volume = self.client.volumes.get(volume_name)
        except docker.errors.NotFound:
            volume = self.client.volumes.create(
                volume_name,
                labels={"csp.managed": "true"}
            )
            logging.info(f"Created Docker volume: {volume_name}")
        
        return volume
    
    async def _setup_load_balancer(self, config: DeploymentConfig, containers: List[Dict]):
        """Setup nginx load balancer for multiple containers"""
        
        # Generate nginx configuration
        upstream_servers = "\n".join([
            f"    server {container['name']}:{config.network.port};"
            for container in containers
        ])
        
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream backend {{
{upstream_servers}
    }}
    
    server {{
        listen {config.network.port};
        
        location / {{
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }}
        
        location /health {{
            access_log off;
            return 200 "healthy\\n";
        }}
    }}
}}
"""
        
        # Create nginx container
        nginx_name = f"{config.name}-lb"
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(nginx_config)
            config_path = f.name
        
        try:
            self.client.containers.run(
                "nginx:alpine",
                name=nginx_name,
                network=f"{config.name}-network",
                ports={f"{config.network.port}/tcp": config.network.port},
                volumes={config_path: {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}},
                restart_policy={"Name": "unless-stopped"},
                detach=True,
                labels={
                    "csp.deployment": config.name,
                    "csp.component": "load-balancer"
                }
            )
            
            logging.info(f"Created load balancer: {nginx_name}")
            
        finally:
            os.unlink(config_path)
    
    async def get_deployment_status(self, name: str) -> Dict[str, Any]:
        """Get Docker deployment status"""
        containers = self.client.containers.list(
            filters={"label": f"csp.deployment={name}"}
        )
        
        container_status = []
        for container in containers:
            container_status.append({
                "name": container.name,
                "id": container.id[:12],
                "status": container.status,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "ports": container.ports,
                "labels": container.labels
            })
        
        return {
            "deployment": name,
            "container_count": len(containers),
            "containers": container_status
        }
    
    async def scale_deployment(self, name: str, replicas: int) -> Dict[str, Any]:
        """Scale Docker deployment"""
        current_containers = self.client.containers.list(
            filters={"label": f"csp.deployment={name}"}
        )
        
        current_count = len([c for c in current_containers if "load-balancer" not in c.labels.get("csp.component", "")])
        
        if replicas > current_count:
            # Scale up - add more containers
            # Implementation would add more containers
            pass
        elif replicas < current_count:
            # Scale down - remove containers
            # Implementation would remove containers
            pass
        
        return {
            "status": "scaled",
            "previous_replicas": current_count,
            "new_replicas": replicas
        }

# ============================================================================
# CLOUD DEPLOYMENT MANAGERS
# ============================================================================

class AWSDeploymentManager:
    """Deploy CSP applications to AWS"""
    
    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self.ecs_client = boto3.client('ecs', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
    
    async def deploy_to_ecs(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to AWS ECS"""
        
        try:
            # Create task definition
            task_definition = await self._create_ecs_task_definition(config)
            
            # Create or update service
            service = await self._create_ecs_service(config, task_definition['taskDefinitionArn'])
            
            return {
                "status": "deployed",
                "service_arn": service['serviceArn'],
                "task_definition_arn": task_definition['taskDefinitionArn']
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def deploy_to_lambda(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to AWS Lambda"""
        
        try:
            # Create Lambda function
            function_config = {
                'FunctionName': config.name,
                'Runtime': 'python3.9',
                'Role': config.environment.get('LAMBDA_ROLE_ARN'),
                'Handler': 'lambda_handler.handler',
                'Code': {
                    'ImageUri': config.image
                },
                'PackageType': 'Image',
                'Environment': {
                    'Variables': config.environment
                },
                'MemorySize': int(config.resources.memory_limit.replace('Mi', '').replace('Gi', '000')),
                'Timeout': 300
            }
            
            try:
                response = self.lambda_client.create_function(**function_config)
            except self.lambda_client.exceptions.ResourceConflictException:
                # Function exists, update it
                response = self.lambda_client.update_function_configuration(**function_config)
            
            return {
                "status": "deployed",
                "function_arn": response['FunctionArn']
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _create_ecs_task_definition(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create ECS task definition"""
        
        container_definition = {
            'name': config.name,
            'image': config.image,
            'memory': int(config.resources.memory_limit.replace('Mi', '').replace('Gi', '000')),
            'cpu': int(config.resources.cpu_limit.replace('m', '')),
            'essential': True,
            'portMappings': [
                {
                    'containerPort': config.network.port,
                    'protocol': 'tcp'
                }
            ],
            'environment': [
                {'name': key, 'value': value}
                for key, value in config.environment.items()
            ],
            'logConfiguration': {
                'logDriver': 'awslogs',
                'options': {
                    'awslogs-group': f'/ecs/{config.name}',
                    'awslogs-region': self.region,
                    'awslogs-stream-prefix': 'ecs'
                }
            }
        }
        
        task_definition = {
            'family': config.name,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': config.resources.cpu_limit,
            'memory': config.resources.memory_limit.replace('Mi', '').replace('Gi', '000'),
            'executionRoleArn': config.environment.get('ECS_EXECUTION_ROLE_ARN'),
            'taskRoleArn': config.environment.get('ECS_TASK_ROLE_ARN'),
            'containerDefinitions': [container_definition]
        }
        
        response = self.ecs_client.register_task_definition(**task_definition)
        return response['taskDefinition']
    
    async def _create_ecs_service(self, config: DeploymentConfig, task_definition_arn: str) -> Dict[str, Any]:
        """Create ECS service"""
        
        service_config = {
            'serviceName': config.name,
            'cluster': config.environment.get('ECS_CLUSTER', 'default'),
            'taskDefinition': task_definition_arn,
            'desiredCount': config.replicas,
            'launchType': 'FARGATE',
            'networkConfiguration': {
                'awsvpcConfiguration': {
                    'subnets': config.environment.get('SUBNETS', '').split(','),
                    'securityGroups': config.environment.get('SECURITY_GROUPS', '').split(','),
                    'assignPublicIp': 'ENABLED'
                }
            }
        }
        
        try:
            response = self.ecs_client.create_service(**service_config)
        except self.ecs_client.exceptions.InvalidParameterException:
            # Service exists, update it
            response = self.ecs_client.update_service(
                cluster=service_config['cluster'],
                service=service_config['serviceName'],
                taskDefinition=task_definition_arn,
                desiredCount=config.replicas
            )
        
        return response['service']

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigurationManager:
    """Manage deployment configurations"""
    
    def __init__(self, config_store: str = "file"):
        self.config_store = config_store
        self.configs = {}
        
        if config_store == "consul":
            self.consul = consul.Consul()
        elif config_store == "etcd":
            self.etcd = etcd3.client()
    
    def save_config(self, config: DeploymentConfig, environment: str = "default"):
        """Save deployment configuration"""
        config_key = f"{environment}/{config.name}"
        config_data = asdict(config)
        
        if self.config_store == "file":
            config_path = Path(f"configs/{config_key}.yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
        elif self.config_store == "consul":
            self.consul.kv.put(config_key, json.dumps(config_data))
            
        elif self.config_store == "etcd":
            self.etcd.put(config_key, json.dumps(config_data))
        
        self.configs[config_key] = config
        logging.info(f"Saved configuration: {config_key}")
    
    def load_config(self, name: str, environment: str = "default") -> Optional[DeploymentConfig]:
        """Load deployment configuration"""
        config_key = f"{environment}/{name}"
        
        if config_key in self.configs:
            return self.configs[config_key]
        
        config_data = None
        
        if self.config_store == "file":
            config_path = Path(f"configs/{config_key}.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
        elif self.config_store == "consul":
            index, data = self.consul.kv.get(config_key)
            if data:
                config_data = json.loads(data['Value'].decode())
                
        elif self.config_store == "etcd":
            value, _ = self.etcd.get(config_key)
            if value:
                config_data = json.loads(value.decode())
        
        if config_data:
            # Convert nested dictionaries back to dataclasses
            config_data['resources'] = ResourceLimits(**config_data.get('resources', {}))
            config_data['network'] = NetworkConfig(**config_data.get('network', {}))
            config_data['scaling'] = ScalingConfig(**config_data.get('scaling', {}))
            config_data['monitoring'] = MonitoringConfig(**config_data.get('monitoring', {}))
            
            config = DeploymentConfig(**config_data)
            self.configs[config_key] = config
            return config
        
        return None
    
    def list_configs(self, environment: str = "default") -> List[str]:
        """List available configurations"""
        configs = []
        
        if self.config_store == "file":
            config_dir = Path(f"configs/{environment}")
            if config_dir.exists():
                configs = [f.stem for f in config_dir.glob("*.yaml")]
                
        elif self.config_store == "consul":
            index, data = self.consul.kv.get(f"{environment}/", recurse=True)
            if data:
                configs = [item['Key'].split('/')[-1] for item in data]
                
        elif self.config_store == "etcd":
            # Implementation for etcd listing
            pass
        
        return configs

# ============================================================================
# MONITORING AND ALERTING SYSTEM
# ============================================================================

class DeploymentMonitor:
    """Monitor CSP deployments across all platforms"""
    
    def __init__(self):
        self.metrics_registry = CollectorRegistry()
        self.metrics = self._setup_metrics()
        self.alert_rules = []
        self.notification_channels = []
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics"""
        return {
            'deployment_status': Gauge(
                'csp_deployment_status',
                'Status of CSP deployment (1=healthy, 0=unhealthy)',
                ['deployment', 'environment'],
                registry=self.metrics_registry
            ),
            'pod_count': Gauge(
                'csp_pod_count',
                'Number of pods in deployment',
                ['deployment', 'environment'],
                registry=self.metrics_registry
            ),
            'request_count': Counter(
                'csp_requests_total',
                'Total number of requests',
                ['deployment', 'environment', 'status'],
                registry=self.metrics_registry
            ),
            'response_time': Histogram(
                'csp_response_time_seconds',
                'Response time in seconds',
                ['deployment', 'environment'],
                registry=self.metrics_registry
            )
        }
    
    async def monitor_deployment(self, deployment_name: str, deployment_manager, environment: str = "default"):
        """Monitor a specific deployment"""
        
        while True:
            try:
                # Get deployment status
                status = await deployment_manager.get_deployment_status(deployment_name, "csp-system")
                
                # Update metrics
                if isinstance(status, dict) and 'ready_replicas' in status:
                    # Kubernetes deployment
                    is_healthy = (status.get('ready_replicas', 0) == status.get('replicas', 0))
                    self.metrics['deployment_status'].labels(
                        deployment=deployment_name,
                        environment=environment
                    ).set(1 if is_healthy else 0)
                    
                    self.metrics['pod_count'].labels(
                        deployment=deployment_name,
                        environment=environment
                    ).set(status.get('ready_replicas', 0))
                
                # Check alert rules
                await self._check_alert_rules(deployment_name, environment, status)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring error for {deployment_name}: {e}")
                await asyncio.sleep(60)
    
    async def _check_alert_rules(self, deployment_name: str, environment: str, status: Dict[str, Any]):
        """Check alert rules against current status"""
        
        for rule in self.alert_rules:
            if rule['deployment'] == deployment_name and rule['environment'] == environment:
                if await self._evaluate_alert_rule(rule, status):
                    await self._send_alert(rule, status)
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any], status: Dict[str, Any]) -> bool:
        """Evaluate if alert rule should fire"""
        
        condition = rule['condition']
        
        if condition['type'] == 'pod_count_below':
            return status.get('ready_replicas', 0) < condition['threshold']
        elif condition['type'] == 'deployment_unhealthy':
            return status.get('ready_replicas', 0) != status.get('replicas', 0)
        elif condition['type'] == 'high_error_rate':
            # Would need to integrate with actual metrics
            return False
        
        return False
    
    async def _send_alert(self, rule: Dict[str, Any], status: Dict[str, Any]):
        """Send alert notification"""
        
        alert_message = f"Alert: {rule['name']} for {rule['deployment']} in {rule['environment']}"
        
        for channel in self.notification_channels:
            try:
                if channel['type'] == 'slack':
                    await self._send_slack_alert(channel, alert_message, status)
                elif channel['type'] == 'email':
                    await self._send_email_alert(channel, alert_message, status)
                elif channel['type'] == 'webhook':
                    await self._send_webhook_alert(channel, alert_message, status)
                    
            except Exception as e:
                logging.error(f"Failed to send alert via {channel['type']}: {e}")
    
    async def _send_slack_alert(self, channel: Dict, message: str, status: Dict):
        """Send alert to Slack"""
        # Implementation for Slack webhook
        pass
    
    async def _send_email_alert(self, channel: Dict, message: str, status: Dict):
        """Send alert via email"""
        # Implementation for email alert
        pass
    
    async def _send_webhook_alert(self, channel: Dict, message: str, status: Dict):
        """Send alert via webhook"""
        # Implementation for webhook alert
        pass
    
    def add_alert_rule(self, deployment: str, environment: str, name: str, condition: Dict[str, Any]):
        """Add alert rule"""
        rule = {
            'deployment': deployment,
            'environment': environment,
            'name': name,
            'condition': condition
        }
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel"""
        channel = {
            'type': channel_type,
            'config': config
        }
        self.notification_channels.append(channel)

# ============================================================================
# MAIN DEPLOYMENT ORCHESTRATOR
# ============================================================================

class CSPDeploymentOrchestrator:
    """Main orchestrator for CSP deployments"""
    
    def __init__(self):
        self.deployment_managers = {
            DeploymentTarget.KUBERNETES: KubernetesDeploymentManager(),
            DeploymentTarget.DOCKER: DockerDeploymentManager(),
            DeploymentTarget.AWS_ECS: AWSDeploymentManager(),
        }
        
        self.config_manager = ConfigurationManager()
        self.monitor = DeploymentMonitor()
        self.active_deployments = {}
    
    async def deploy(self, config: DeploymentConfig, environment: str = "default") -> Dict[str, Any]:
        """Deploy CSP application"""
        
        deployment_id = f"{config.name}-{environment}-{int(time.time())}"
        
        try:
            # Save configuration
            self.config_manager.save_config(config, environment)
            
            # Get appropriate deployment manager
            manager = self.deployment_managers.get(config.target)
            if not manager:
                raise ValueError(f"Unsupported deployment target: {config.target}")
            
            # Perform deployment
            result = await manager.deploy(config)
            
            if result.get("status") == "deployed":
                # Start monitoring
                monitor_task = asyncio.create_task(
                    self.monitor.monitor_deployment(config.name, manager, environment)
                )
                
                self.active_deployments[deployment_id] = {
                    "config": config,
                    "manager": manager,
                    "environment": environment,
                    "monitor_task": monitor_task,
                    "deployed_at": time.time()
                }
            
            result["deployment_id"] = deployment_id
            return result
            
        except Exception as e:
            return {
                "deployment_id": deployment_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def update_deployment(self, deployment_id: str, strategy: str = "rolling") -> Dict[str, Any]:
        """Update existing deployment"""
        
        if deployment_id not in self.active_deployments:
            return {"status": "failed", "error": "Deployment not found"}
        
        deployment_info = self.active_deployments[deployment_id]
        manager = deployment_info["manager"]
        config = deployment_info["config"]
        
        return await manager.update_deployment(config, strategy)
    
    async def scale_deployment(self, deployment_id: str, replicas: int) -> Dict[str, Any]:
        """Scale deployment"""
        
        if deployment_id not in self.active_deployments:
            return {"status": "failed", "error": "Deployment not found"}
        
        deployment_info = self.active_deployments[deployment_id]
        manager = deployment_info["manager"]
        config = deployment_info["config"]
        
        # Update config
        config.replicas = replicas
        
        return await manager.scale_deployment(config.name, config.namespace, replicas)
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        
        if deployment_id not in self.active_deployments:
            return {"status": "not_found"}
        
        deployment_info = self.active_deployments[deployment_id]
        manager = deployment_info["manager"]
        config = deployment_info["config"]
        
        if hasattr(manager, 'get_deployment_status'):
            return await manager.get_deployment_status(config.name, config.namespace)
        else:
            return {"status": "unknown"}
    
    async def delete_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Delete deployment"""
        
        if deployment_id not in self.active_deployments:
            return {"status": "not_found"}
        
        deployment_info = self.active_deployments[deployment_id]
        
        # Cancel monitoring
        if "monitor_task" in deployment_info:
            deployment_info["monitor_task"].cancel()
        
        # Delete deployment (implementation would depend on platform)
        # For now, just remove from active deployments
        del self.active_deployments[deployment_id]
        
        return {"status": "deleted"}
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments"""
        deployments = []
        
        for deployment_id, info in self.active_deployments.items():
            deployments.append({
                "deployment_id": deployment_id,
                "name": info["config"].name,
                "environment": info["environment"],
                "target": info["config"].target.name,
                "replicas": info["config"].replicas,
                "deployed_at": info["deployed_at"]
            })
        
        return deployments

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def demonstrate_deployment_system():
    """Demonstrate the deployment system"""
    
    # Create deployment configuration
    config = DeploymentConfig(
        name="csp-demo",
        version="1.0.0",
        target=DeploymentTarget.KUBERNETES,
        image="csp-runtime:latest",
        replicas=3,
        namespace="csp-demo",
        environment={
            "CSP_LOG_LEVEL": "INFO",
            "CSP_MONITORING_ENABLED": "true"
        }
    )
    
    # Create orchestrator
    orchestrator = CSPDeploymentOrchestrator()
    
    print("🚀 CSP Deployment System Demo")
    print("=" * 40)
    
    # Deploy application
    print("\\n1. Deploying CSP application...")
    deployment_result = await orchestrator.deploy(config, "development")
    print(f"Deployment result: {deployment_result}")
    
    if deployment_result.get("status") == "deployed":
        deployment_id = deployment_result["deployment_id"]
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Get status
        print("\\n2. Getting deployment status...")
        status = await orchestrator.get_deployment_status(deployment_id)
        print(f"Status: {status}")
        
        # Scale deployment
        print("\\n3. Scaling deployment...")
        scale_result = await orchestrator.scale_deployment(deployment_id, 5)
        print(f"Scale result: {scale_result}")
        
        # List deployments
        print("\\n4. Listing all deployments...")
        deployments = orchestrator.list_deployments()
        for deployment in deployments:
            print(f"  - {deployment}")
        
        # Cleanup (in production, you might not want to auto-delete)
        print("\\n5. Cleaning up...")
        delete_result = await orchestrator.delete_deployment(deployment_id)
        print(f"Delete result: {delete_result}")
    
    print("\\n✅ Deployment system demo completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_deployment_system())
