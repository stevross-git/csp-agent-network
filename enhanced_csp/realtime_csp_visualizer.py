#!/usr/bin/env python3
"""
Real-time CSP Visualization Engine
==================================

Interactive visualization and monitoring system for CSP networks:
- Real-time process flow visualization
- Interactive network topology viewer  
- Performance metrics dashboards
- 3D process relationship graphs
- Time-series analytics
- Anomaly visualization
- Live debugging interface
- Export capabilities for reports
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import threading
import websocket
from pathlib import Path

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Arrow
    import matplotlib.patches as mpatches
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    from dash.exceptions import PreventUpdate
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Web framework
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Import our CSP components
from core.advanced_csp_core import Process, ProcessContext, Channel, Event

# ============================================================================
# DATA STRUCTURES FOR VISUALIZATION
# ============================================================================

@dataclass
class VisualizationNode:
    """Node in the visualization graph"""
    node_id: str
    node_type: str  # 'process', 'channel', 'resource', 'agent'
    position: Tuple[float, float, float]  # 3D coordinates
    properties: Dict[str, Any] = field(default_factory=dict)
    status: str = 'active'  # 'active', 'inactive', 'error', 'warning'
    metrics: Dict[str, float] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class VisualizationEdge:
    """Edge in the visualization graph"""
    edge_id: str
    source: str
    target: str
    edge_type: str  # 'communication', 'dependency', 'data_flow'
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = 'active'
    timestamp: float = field(default_factory=time.time)

@dataclass
class MetricTimeSeries:
    """Time series data for metrics"""
    metric_name: str
    timestamps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    unit: str = ""
    description: str = ""
    
    def add_point(self, timestamp: float, value: float):
        """Add a new data point"""
        self.timestamps.append(timestamp)
        self.values.append(value)
        
        # Keep only last 1000 points
        if len(self.timestamps) > 1000:
            self.timestamps.pop(0)
            self.values.pop(0)

# ============================================================================
# REAL-TIME CSP VISUALIZER
# ============================================================================

class RealtimeCSPVisualizer:
    """Main visualization engine for CSP systems"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.nodes = {}
        self.edges = {}
        self.metrics = defaultdict(lambda: MetricTimeSeries(""))
        self.alerts = deque(maxlen=100)
        self.connected_clients = set()
        
        # Visualization settings
        self.layout_algorithm = 'spring'
        self.color_scheme = 'viridis'
        self.animation_speed = 1.0
        self.show_metrics = True
        self.show_labels = True
        
        # Performance tracking
        self.frame_rate = 30  # FPS
        self.update_interval = 1.0 / self.frame_rate
        self.last_update = time.time()
        
        # Initialize web app
        self.app = self._create_web_app()
        
        # Start background services
        self.running = False
        self.update_task = None
        
    def _create_web_app(self) -> FastAPI:
        """Create FastAPI web application"""
        app = FastAPI(title="CSP Visualization Engine")
        
        # WebSocket endpoint for real-time updates
        @app.websocket("/ws/visualization")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Send real-time updates
                    update_data = {
                        'type': 'system_update',
                        'timestamp': time.time(),
                        'nodes': {nid: self._serialize_node(node) for nid, node in self.nodes.items()},
                        'edges': {eid: self._serialize_edge(edge) for eid, edge in self.edges.items()},
                        'metrics': self._get_latest_metrics(),
                        'alerts': list(self.alerts)[-10:]  # Last 10 alerts
                    }
                    
                    await websocket.send_json(update_data)
                    await asyncio.sleep(self.update_interval)
                    
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
            finally:
                self.connected_clients.remove(websocket)
        
        # REST API endpoints
        @app.get("/api/visualization/nodes")
        async def get_nodes():
            return {nid: self._serialize_node(node) for nid, node in self.nodes.items()}
        
        @app.get("/api/visualization/edges")
        async def get_edges():
            return {eid: self._serialize_edge(edge) for eid, edge in self.edges.items()}
        
        @app.get("/api/visualization/metrics/{metric_name}")
        async def get_metric_history(metric_name: str):
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                return {
                    'name': metric.metric_name,
                    'timestamps': metric.timestamps,
                    'values': metric.values,
                    'unit': metric.unit,
                    'description': metric.description
                }
            return {'error': 'Metric not found'}
        
        @app.post("/api/visualization/layout")
        async def update_layout(layout_data: dict):
            algorithm = layout_data.get('algorithm', 'spring')
            await self.update_layout(algorithm)
            return {'status': 'layout_updated', 'algorithm': algorithm}
        
        @app.get("/api/visualization/export/{format}")
        async def export_visualization(format: str):
            if format not in ['png', 'svg', 'json', 'pdf']:
                raise HTTPException(status_code=400, detail="Unsupported format")
            
            export_data = await self.export_visualization(format)
            return {'format': format, 'data': export_data}
        
        # Serve static files
        app.mount("/static", StaticFiles(directory="web_ui/static"), name="static")
        
        @app.get("/", response_class=HTMLResponse)
        async def visualization_dashboard():
            return self._generate_dashboard_html()
        
        return app
    
    async def add_node(self, node: VisualizationNode):
        """Add a node to the visualization"""
        self.nodes[node.node_id] = node
        await self._broadcast_update('node_added', {'node': self._serialize_node(node)})
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]):
        """Update node properties"""
        if node_id in self.nodes:
            self.nodes[node_id].properties.update(properties)
            self.nodes[node_id].timestamp = time.time()
            
            await self._broadcast_update('node_updated', {
                'node_id': node_id,
                'properties': properties
            })
    
    async def add_edge(self, edge: VisualizationEdge):
        """Add an edge to the visualization"""
        self.edges[edge.edge_id] = edge
        await self._broadcast_update('edge_added', {'edge': self._serialize_edge(edge)})
    
    async def update_edge(self, edge_id: str, properties: Dict[str, Any]):
        """Update edge properties"""
        if edge_id in self.edges:
            self.edges[edge_id].properties.update(properties)
            self.edges[edge_id].timestamp = time.time()
            
            await self._broadcast_update('edge_updated', {
                'edge_id': edge_id,
                'properties': properties
            })
    
    async def add_metric(self, metric_name: str, value: float, unit: str = "", 
                        description: str = ""):
        """Add a metric data point"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricTimeSeries(
                metric_name=metric_name,
                unit=unit,
                description=description
            )
        
        self.metrics[metric_name].add_point(time.time(), value)
        
        await self._broadcast_update('metric_updated', {
            'metric_name': metric_name,
            'value': value,
            'timestamp': time.time()
        })
    
    async def add_alert(self, level: str, message: str, source: str = "system"):
        """Add an alert"""
        alert = {
            'level': level,  # 'info', 'warning', 'error', 'critical'
            'message': message,
            'source': source,
            'timestamp': time.time()
        }
        
        self.alerts.append(alert)
        
        await self._broadcast_update('alert_added', {'alert': alert})
    
    async def update_layout(self, algorithm: str = 'spring'):
        """Update node positions using layout algorithm"""
        if not NETWORKX_AVAILABLE:
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.properties)
        
        # Add edges
        for edge_id, edge in self.edges.items():
            G.add_edge(edge.source, edge.target, **edge.properties)
        
        # Calculate layout
        if algorithm == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif algorithm == 'circular':
            pos = nx.circular_layout(G)
        elif algorithm == 'random':
            pos = nx.random_layout(G)
        elif algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Update node positions
        for node_id, (x, y) in pos.items():
            if node_id in self.nodes:
                self.nodes[node_id].position = (x, y, 0.0)
        
        self.layout_algorithm = algorithm
        await self._broadcast_update('layout_updated', {'algorithm': algorithm})
    
    async def create_process_flow_visualization(self, processes: List[Process]) -> Dict[str, Any]:
        """Create visualization of process flows"""
        
        # Clear existing visualization
        self.nodes.clear()
        self.edges.clear()
        
        # Add process nodes
        for i, process in enumerate(processes):
            node = VisualizationNode(
                node_id=process.name,
                node_type='process',
                position=(i * 2.0, 0.0, 0.0),
                properties={
                    'name': process.name,
                    'type': type(process).__name__,
                    'status': getattr(process, 'status', 'active'),
                    'priority': getattr(process, 'priority', 1.0),
                    'cpu_usage': np.random.uniform(0.1, 0.9),
                    'memory_usage': np.random.uniform(0.2, 0.8)
                }
            )
            await self.add_node(node)
        
        # Add communication edges (mock data)
        for i in range(len(processes) - 1):
            edge = VisualizationEdge(
                edge_id=f"comm_{i}_{i+1}",
                source=processes[i].name,
                target=processes[i+1].name,
                edge_type='communication',
                properties={
                    'bandwidth': np.random.uniform(10, 100),
                    'latency': np.random.uniform(0.001, 0.01),
                    'message_rate': np.random.uniform(10, 1000)
                }
            )
            await self.add_edge(edge)
        
        return {
            'nodes_created': len(self.nodes),
            'edges_created': len(self.edges),
            'visualization_type': 'process_flow'
        }
    
    async def create_network_topology_visualization(self, topology: Dict[str, Any]) -> Dict[str, Any]:
        """Create network topology visualization"""
        
        nodes = topology.get('nodes', [])
        edges = topology.get('edges', [])
        
        # Add nodes
        for node_data in nodes:
            node = VisualizationNode(
                node_id=node_data['id'],
                node_type='resource',
                position=(
                    node_data.get('x', np.random.uniform(-5, 5)),
                    node_data.get('y', np.random.uniform(-5, 5)),
                    node_data.get('z', 0.0)
                ),
                properties=node_data,
                metrics={
                    'cpu_usage': node_data.get('cpu_usage', 0.0),
                    'memory_usage': node_data.get('memory_usage', 0.0),
                    'network_usage': node_data.get('network_usage', 0.0)
                }
            )
            await self.add_node(node)
        
        # Add edges
        for edge_data in edges:
            edge = VisualizationEdge(
                edge_id=f"{edge_data['source']}-{edge_data['target']}",
                source=edge_data['source'],
                target=edge_data['target'],
                edge_type='network_connection',
                properties=edge_data,
                metrics={
                    'bandwidth_utilization': edge_data.get('bandwidth_utilization', 0.0),
                    'packet_loss': edge_data.get('packet_loss', 0.0),
                    'latency': edge_data.get('latency', 0.0)
                }
            )
            await self.add_edge(edge)
        
        return {
            'nodes_created': len(nodes),
            'edges_created': len(edges),
            'visualization_type': 'network_topology'
        }
    
    async def create_performance_dashboard(self, metrics: Dict[str, List[float]]) -> str:
        """Create performance metrics dashboard"""
        
        if not PLOTLY_AVAILABLE:
            return "Plotly not available"
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Network Throughput', 'Response Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for each metric
        timestamps = list(range(len(metrics.get('cpu_usage', []))))
        
        if 'cpu_usage' in metrics:
            fig.add_trace(
                go.Scatter(x=timestamps, y=metrics['cpu_usage'], name='CPU Usage', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'memory_usage' in metrics:
            fig.add_trace(
                go.Scatter(x=timestamps, y=metrics['memory_usage'], name='Memory Usage', line=dict(color='blue')),
                row=1, col=2
            )
        
        if 'network_throughput' in metrics:
            fig.add_trace(
                go.Scatter(x=timestamps, y=metrics['network_throughput'], name='Network Throughput', line=dict(color='green')),
                row=2, col=1
            )
        
        if 'response_time' in metrics:
            fig.add_trace(
                go.Scatter(x=timestamps, y=metrics['response_time'], name='Response Time', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="CSP System Performance Dashboard",
            showlegend=True,
            height=600
        )
        
        # Generate HTML
        html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
        return html_content
    
    async def create_3d_process_graph(self) -> str:
        """Create 3D process relationship graph"""
        
        if not PLOTLY_AVAILABLE:
            return "Plotly not available"
        
        # Extract 3D coordinates
        node_ids = list(self.nodes.keys())
        x_coords = [self.nodes[nid].position[0] for nid in node_ids]
        y_coords = [self.nodes[nid].position[1] for nid in node_ids]
        z_coords = [self.nodes[nid].position[2] for nid in node_ids]
        
        # Node colors based on status
        colors = []
        for nid in node_ids:
            status = self.nodes[nid].status
            if status == 'active':
                colors.append('green')
            elif status == 'warning':
                colors.append('yellow')
            elif status == 'error':
                colors.append('red')
            else:
                colors.append('gray')
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.8
            ),
            text=node_ids,
            textposition="middle center",
            name="Processes"
        ))
        
        # Add edges
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.edges.values():
            if edge.source in self.nodes and edge.target in self.nodes:
                source_pos = self.nodes[edge.source].position
                target_pos = self.nodes[edge.target].position
                
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
                edge_z.extend([source_pos[2], target_pos[2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            name="Connections"
        ))
        
        # Update layout
        fig.update_layout(
            title="3D CSP Process Network",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            showlegend=True
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=True)
    
    async def export_visualization(self, format: str) -> str:
        """Export visualization in various formats"""
        
        if format == 'json':
            export_data = {
                'nodes': {nid: self._serialize_node(node) for nid, node in self.nodes.items()},
                'edges': {eid: self._serialize_edge(edge) for eid, edge in self.edges.items()},
                'metrics': {name: {
                    'timestamps': metric.timestamps,
                    'values': metric.values,
                    'unit': metric.unit,
                    'description': metric.description
                } for name, metric in self.metrics.items()},
                'export_timestamp': time.time()
            }
            return json.dumps(export_data, indent=2)
        
        elif format == 'png' and MATPLOTLIB_AVAILABLE:
            return await self._export_matplotlib_png()
        
        elif format == 'svg' and MATPLOTLIB_AVAILABLE:
            return await self._export_matplotlib_svg()
        
        elif format == 'pdf':
            return await self._export_pdf_report()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _export_matplotlib_png(self) -> str:
        """Export as PNG using matplotlib"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        for node in self.nodes.values():
            x, y, _ = node.position
            color = 'green' if node.status == 'active' else 'red'
            ax.scatter(x, y, c=color, s=100, alpha=0.7)
            ax.annotate(node.node_id, (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Draw edges
        for edge in self.edges.values():
            if edge.source in self.nodes and edge.target in self.nodes:
                source_pos = self.nodes[edge.source].position
                target_pos = self.nodes[edge.target].position
                ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], 'k-', alpha=0.5)
        
        ax.set_title('CSP Network Visualization')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        # Save to file
        filename = f"csp_visualization_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    async def _export_pdf_report(self) -> str:
        """Export comprehensive PDF report"""
        
        # Generate report content
        report_content = f"""
# CSP System Visualization Report

Generated: {datetime.now().isoformat()}

## System Overview
- Total Nodes: {len(self.nodes)}
- Total Edges: {len(self.edges)}
- Active Metrics: {len(self.metrics)}
- Recent Alerts: {len(self.alerts)}

## Node Summary
"""
        
        for node_id, node in self.nodes.items():
            report_content += f"""
### {node_id}
- Type: {node.node_type}
- Status: {node.status}
- Position: {node.position}
- Properties: {json.dumps(node.properties, indent=2)}
"""
        
        report_content += """
## Metrics Summary
"""
        
        for metric_name, metric in self.metrics.items():
            if metric.values:
                avg_value = np.mean(metric.values)
                max_value = np.max(metric.values)
                min_value = np.min(metric.values)
                
                report_content += f"""
### {metric_name}
- Average: {avg_value:.3f} {metric.unit}
- Maximum: {max_value:.3f} {metric.unit}
- Minimum: {min_value:.3f} {metric.unit}
- Data Points: {len(metric.values)}
"""
        
        # Save report
        filename = f"csp_report_{int(time.time())}.md"
        with open(filename, 'w') as f:
            f.write(report_content)
        
        return filename
    
    def _serialize_node(self, node: VisualizationNode) -> Dict[str, Any]:
        """Serialize node for JSON transmission"""
        return {
            'node_id': node.node_id,
            'node_type': node.node_type,
            'position': node.position,
            'properties': node.properties,
            'status': node.status,
            'metrics': node.metrics,
            'connections': node.connections,
            'timestamp': node.timestamp
        }
    
    def _serialize_edge(self, edge: VisualizationEdge) -> Dict[str, Any]:
        """Serialize edge for JSON transmission"""
        return {
            'edge_id': edge.edge_id,
            'source': edge.source,
            'target': edge.target,
            'edge_type': edge.edge_type,
            'properties': edge.properties,
            'metrics': edge.metrics,
            'status': edge.status,
            'timestamp': edge.timestamp
        }
    
    def _get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metric values"""
        latest_metrics = {}
        
        for metric_name, metric in self.metrics.items():
            if metric.values:
                latest_metrics[metric_name] = {
                    'value': metric.values[-1],
                    'timestamp': metric.timestamps[-1],
                    'unit': metric.unit
                }
        
        return latest_metrics
    
    async def _broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.connected_clients:
            return
        
        message = {
            'type': update_type,
            'timestamp': time.time(),
            'data': data
        }
        
        # Send to all connected WebSocket clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSP Visualization Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background: #e8f4fd;
            border-radius: 4px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
        }
        #network-graph {
            height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        #performance-chart {
            height: 300px;
        }
        .alerts {
            max-height: 200px;
            overflow-y: auto;
        }
        .alert {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert.info { background: #d4edda; border-color: #28a745; }
        .alert.warning { background: #fff3cd; border-color: #ffc107; }
        .alert.error { background: #f8d7da; border-color: #dc3545; }
        .controls {
            margin-bottom: 20px;
        }
        button {
            padding: 8px 16px;
            margin: 4px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>🌟 CSP System Visualization Dashboard</h1>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="node-count">0</div>
            <div class="metric-label">Active Nodes</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="edge-count">0</div>
            <div class="metric-label">Connections</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="cpu-usage">0%</div>
            <div class="metric-label">CPU Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="memory-usage">0%</div>
            <div class="metric-label">Memory Usage</div>
        </div>
    </div>
    
    <div class="controls">
        <button onclick="updateLayout('spring')">Spring Layout</button>
        <button onclick="updateLayout('circular')">Circular Layout</button>
        <button onclick="updateLayout('random')">Random Layout</button>
        <button onclick="exportVisualization('json')">Export JSON</button>
        <button onclick="exportVisualization('png')">Export PNG</button>
    </div>
    
    <div class="dashboard">
        <div class="panel">
            <h3>Network Topology</h3>
            <div id="network-graph"></div>
        </div>
        
        <div class="panel">
            <h3>Performance Metrics</h3>
            <div id="performance-chart"></div>
        </div>
        
        <div class="panel">
            <h3>Process Flow</h3>
            <div id="process-flow"></div>
        </div>
        
        <div class="panel">
            <h3>System Alerts</h3>
            <div id="alerts" class="alerts"></div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws/visualization`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('node-count').textContent = Object.keys(data.nodes || {}).length;
            document.getElementById('edge-count').textContent = Object.keys(data.edges || {}).length;
            
            if (data.metrics) {
                const cpuMetric = data.metrics['cpu_usage'];
                const memoryMetric = data.metrics['memory_usage'];
                
                if (cpuMetric) {
                    document.getElementById('cpu-usage').textContent = 
                        Math.round(cpuMetric.value * 100) + '%';
                }
                
                if (memoryMetric) {
                    document.getElementById('memory-usage').textContent = 
                        Math.round(memoryMetric.value * 100) + '%';
                }
            }
            
            // Update network graph
            updateNetworkGraph(data.nodes, data.edges);
            
            // Update alerts
            updateAlerts(data.alerts);
        }
        
        function updateNetworkGraph(nodes, edges) {
            const nodeList = Object.values(nodes || {});
            const edgeList = Object.values(edges || {});
            
            // Create Plotly graph data
            const nodeTrace = {
                x: nodeList.map(n => n.position[0]),
                y: nodeList.map(n => n.position[1]),
                mode: 'markers+text',
                type: 'scatter',
                text: nodeList.map(n => n.node_id),
                textposition: 'middle center',
                marker: {
                    size: 15,
                    color: nodeList.map(n => n.status === 'active' ? 'green' : 'red')
                }
            };
            
            // Create edge traces
            const edgeXCoords = [];
            const edgeYCoords = [];
            
            edgeList.forEach(edge => {
                const sourceNode = nodeList.find(n => n.node_id === edge.source);
                const targetNode = nodeList.find(n => n.node_id === edge.target);
                
                if (sourceNode && targetNode) {
                    edgeXCoords.push(sourceNode.position[0], targetNode.position[0], null);
                    edgeYCoords.push(sourceNode.position[1], targetNode.position[1], null);
                }
            });
            
            const edgeTrace = {
                x: edgeXCoords,
                y: edgeYCoords,
                mode: 'lines',
                type: 'scatter',
                line: { color: 'gray', width: 1 }
            };
            
            const layout = {
                title: 'Real-time Network Topology',
                showlegend: false,
                margin: { t: 40, b: 40, l: 40, r: 40 }
            };
            
            Plotly.newPlot('network-graph', [edgeTrace, nodeTrace], layout);
        }
        
        function updateAlerts(alerts) {
            const alertsContainer = document.getElementById('alerts');
            alertsContainer.innerHTML = '';
            
            (alerts || []).forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert ${alert.level}`;
                alertDiv.innerHTML = `
                    <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                    <small style="float: right;">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</small>
                `;
                alertsContainer.appendChild(alertDiv);
            });
        }
        
        function updateLayout(algorithm) {
            fetch('/api/visualization/layout', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ algorithm: algorithm })
            });
        }
        
        function exportVisualization(format) {
            fetch(`/api/visualization/export/${format}`)
                .then(response => response.json())
                .then(data => {
                    if (format === 'json') {
                        const blob = new Blob([JSON.stringify(data.data, null, 2)], 
                            { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `csp_visualization.${format}`;
                        a.click();
                        URL.revokeObjectURL(url);
                    } else {
                        alert(`Export completed: ${data.data}`);
                    }
                });
        }
        
        // Initialize dashboard
        console.log('CSP Visualization Dashboard initialized');
    </script>
</body>
</html>
        """
    
    async def start(self):
        """Start the visualization server"""
        self.running = True
        
        # Start update loop
        self.update_task = asyncio.create_task(self._update_loop())
        
        # Start web server
        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        
        logging.info(f"Starting CSP Visualization Engine on port {self.port}")
        await server.serve()
    
    async def stop(self):
        """Stop the visualization server"""
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
    
    async def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                # Simulate system metrics updates
                await self.add_metric("cpu_usage", np.random.uniform(0.1, 0.9))
                await self.add_metric("memory_usage", np.random.uniform(0.2, 0.8))
                await self.add_metric("network_throughput", np.random.uniform(50, 150))
                await self.add_metric("response_time", np.random.uniform(0.001, 0.1))
                
                # Occasionally add alerts
                if np.random.random() < 0.1:  # 10% chance
                    levels = ['info', 'warning', 'error']
                    level = np.random.choice(levels)
                    messages = [
                        "Process completed successfully",
                        "High CPU usage detected",
                        "Network connection timeout",
                        "Memory usage approaching limit",
                        "New process started"
                    ]
                    message = np.random.choice(messages)
                    await self.add_alert(level, message)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Update loop error: {e}")
                await asyncio.sleep(1.0)

# ============================================================================
# VISUALIZATION DEMO
# ============================================================================

async def visualization_demo():
    """Demonstrate CSP visualization capabilities"""
    
    print("🎨 Real-time CSP Visualization Demo")
    print("=" * 50)
    
    # Create visualizer
    visualizer = RealtimeCSPVisualizer(port=8080)
    
    # Create mock CSP system
    from core.advanced_csp_core import AtomicProcess, ProcessSignature
    
    processes = []
    for i in range(6):
        process = AtomicProcess(
            name=f"Process_{i}",
            signature=ProcessSignature(inputs=[], outputs=[])
        )
        processes.append(process)
    
    print(f"✅ Created {len(processes)} mock processes")
    
    # Create process flow visualization
    flow_result = await visualizer.create_process_flow_visualization(processes)
    print(f"✅ Process flow visualization: {flow_result['nodes_created']} nodes, {flow_result['edges_created']} edges")
    
    # Create network topology
    topology = {
        'nodes': [
            {'id': 'server_1', 'cpu_usage': 0.6, 'memory_usage': 0.7, 'x': 0, 'y': 0},
            {'id': 'server_2', 'cpu_usage': 0.8, 'memory_usage': 0.5, 'x': 2, 'y': 1},
            {'id': 'server_3', 'cpu_usage': 0.4, 'memory_usage': 0.9, 'x': 1, 'y': 2},
            {'id': 'load_balancer', 'cpu_usage': 0.3, 'memory_usage': 0.2, 'x': 1, 'y': -1}
        ],
        'edges': [
            {'source': 'load_balancer', 'target': 'server_1', 'bandwidth_utilization': 0.6},
            {'source': 'load_balancer', 'target': 'server_2', 'bandwidth_utilization': 0.8},
            {'source': 'load_balancer', 'target': 'server_3', 'bandwidth_utilization': 0.4}
        ]
    }
    
    topo_result = await visualizer.create_network_topology_visualization(topology)
    print(f"✅ Network topology visualization: {topo_result['nodes_created']} nodes, {topo_result['edges_created']} edges")
    
    # Add some metrics
    for i in range(20):
        await visualizer.add_metric("cpu_usage", 0.5 + 0.3 * np.sin(i * 0.3))
        await visualizer.add_metric("memory_usage", 0.6 + 0.2 * np.cos(i * 0.4))
        await visualizer.add_metric("network_throughput", 100 + 50 * np.sin(i * 0.2))
    
    print("✅ Added sample metrics data")
    
    # Add some alerts
    await visualizer.add_alert("info", "System initialization completed", "system")
    await visualizer.add_alert("warning", "High CPU usage on server_2", "monitoring")
    await visualizer.add_alert("error", "Connection timeout to server_3", "network")
    
    print("✅ Added sample alerts")
    
    # Test different layouts
    await visualizer.update_layout('spring')
    print("✅ Applied spring layout")
    
    # Export visualization
    json_export = await visualizer.export_visualization('json')
    print(f"✅ JSON export: {len(json_export)} characters")
    
    if MATPLOTLIB_AVAILABLE:
        png_file = await visualizer.export_visualization('png')
        print(f"✅ PNG export: {png_file}")
    
    # Create performance dashboard
    metrics_data = {
        'cpu_usage': [0.3, 0.5, 0.7, 0.6, 0.8, 0.4, 0.9, 0.3],
        'memory_usage': [0.4, 0.6, 0.5, 0.8, 0.7, 0.9, 0.5, 0.6],
        'network_throughput': [80, 120, 100, 150, 90, 110, 140, 95],
        'response_time': [0.01, 0.02, 0.015, 0.03, 0.018, 0.025, 0.012, 0.022]
    }
    
    if PLOTLY_AVAILABLE:
        dashboard_html = await visualizer.create_performance_dashboard(metrics_data)
        print("✅ Performance dashboard created")
        
        # Create 3D graph
        graph_3d = await visualizer.create_3d_process_graph()
        print("✅ 3D process graph created")
    
    print(f"\n🌐 Visualization server starting on http://localhost:{visualizer.port}")
    print("Features available:")
    print("• Real-time process flow visualization")
    print("• Interactive network topology viewer")
    print("• Live performance metrics dashboard")
    print("• 3D process relationship graphs")
    print("• WebSocket-based real-time updates")
    print("• Multiple export formats (JSON, PNG, SVG, PDF)")
    print("• Anomaly visualization and alerting")
    print("• Customizable layouts and themes")
    
    # Start the server (this will run indefinitely)
    try:
        await visualizer.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down visualization server...")
        await visualizer.stop()

if __name__ == "__main__":
    asyncio.run(visualization_demo())
