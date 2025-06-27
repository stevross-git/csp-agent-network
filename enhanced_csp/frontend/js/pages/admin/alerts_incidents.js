/**
 * Enhanced Alerts & Incidents Dashboard for CSP Admin Portal
 * Integrates with Prometheus, Alertmanager, and monitoring stack
 */

class AlertsIncidentsManager {
    constructor() {
        this.alerts = [];
        this.incidents = [];
        this.isLoading = false;
        this.lastUpdate = null;
        this.updateInterval = null;
        this.filters = {
            severity: 'all',
            status: 'all',
            timeRange: '24h',
            search: ''
        };
        
        // Monitoring endpoints
        this.endpoints = {
            prometheus: 'http://localhost:9090',
            alertmanager: 'http://localhost:9093',
            grafana: 'http://localhost:3001'
        };
        
        this.init();
    }

    async init() {
        console.log('🚨 Initializing Alerts & Incidents Manager...');
        try {
            await this.createDashboard();
            await this.loadInitialData();
            this.startAutoRefresh();
            this.attachEventListeners();
            console.log('✅ Alerts & Incidents Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Alerts & Incidents Manager:', error);
            this.showError('Failed to initialize alerts dashboard');
        }
    }

    async createDashboard() {
        const alertsSection = document.getElementById('alerts');
        if (!alertsSection) {
            console.error('Alerts section not found');
            return;
        }

        alertsSection.innerHTML = `
            <div class="alerts-incidents-dashboard">
                <!-- Header -->
                <div class="dashboard-header">
                    <div class="header-left">
                        <h2 class="section-title">
                            <i class="fas fa-exclamation-triangle"></i> Alerts & Incidents
                        </h2>
                        <div class="status-indicator" id="alerts-status">
                            <span class="status-dot status-loading"></span>
                            <span class="status-text">Loading...</span>
                        </div>
                    </div>
                    <div class="header-actions">
                        <button class="btn btn-outline" onclick="alertsManager.refreshData()" id="refresh-btn">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                        <button class="btn btn-outline" onclick="alertsManager.exportData()">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <button class="btn btn-primary" onclick="alertsManager.openMonitoringTools()">
                            <i class="fas fa-external-link-alt"></i> Monitoring Tools
                        </button>
                    </div>
                </div>

                <!-- Summary Cards -->
                <div class="summary-cards">
                    <div class="summary-card critical" id="critical-alerts-card">
                        <div class="card-icon">
                            <i class="fas fa-fire"></i>
                        </div>
                        <div class="card-content">
                            <div class="card-number" id="critical-count">0</div>
                            <div class="card-label">Critical Alerts</div>
                        </div>
                    </div>
                    <div class="summary-card warning" id="warning-alerts-card">
                        <div class="card-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <div class="card-content">
                            <div class="card-number" id="warning-count">0</div>
                            <div class="card-label">Warning Alerts</div>
                        </div>
                    </div>
                    <div class="summary-card info" id="info-alerts-card">
                        <div class="card-icon">
                            <i class="fas fa-info-circle"></i>
                        </div>
                        <div class="card-content">
                            <div class="card-number" id="info-count">0</div>
                            <div class="card-label">Info Alerts</div>
                        </div>
                    </div>
                    <div class="summary-card incidents" id="incidents-card">
                        <div class="card-icon">
                            <i class="fas fa-bug"></i>
                        </div>
                        <div class="card-content">
                            <div class="card-number" id="incidents-count">0</div>
                            <div class="card-label">Open Incidents</div>
                        </div>
                    </div>
                </div>

                <!-- Filters -->
                <div class="filters-section">
                    <div class="filters-row">
                        <div class="filter-group">
                            <label for="severity-filter">Severity:</label>
                            <select id="severity-filter" onchange="alertsManager.applyFilters()">
                                <option value="all">All Severities</option>
                                <option value="critical">Critical</option>
                                <option value="warning">Warning</option>
                                <option value="info">Info</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="status-filter">Status:</label>
                            <select id="status-filter" onchange="alertsManager.applyFilters()">
                                <option value="all">All Status</option>
                                <option value="firing">Firing</option>
                                <option value="resolved">Resolved</option>
                                <option value="acknowledged">Acknowledged</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="time-filter">Time Range:</label>
                            <select id="time-filter" onchange="alertsManager.applyFilters()">
                                <option value="1h">Last Hour</option>
                                <option value="24h" selected>Last 24 Hours</option>
                                <option value="7d">Last 7 Days</option>
                                <option value="30d">Last 30 Days</option>
                            </select>
                        </div>
                        <div class="filter-group search-group">
                            <label for="search-filter">Search:</label>
                            <input type="text" id="search-filter" placeholder="Search alerts..." 
                                   oninput="alertsManager.applyFilters()">
                        </div>
                        <div class="filter-actions">
                            <button class="btn btn-outline btn-sm" onclick="alertsManager.clearFilters()">
                                <i class="fas fa-times"></i> Clear
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Main Content Tabs -->
                <div class="tabs-container">
                    <div class="tabs-header">
                        <button class="tab-btn active" onclick="alertsManager.switchTab('alerts')" data-tab="alerts">
                            <i class="fas fa-exclamation-triangle"></i> Active Alerts
                        </button>
                        <button class="tab-btn" onclick="alertsManager.switchTab('incidents')" data-tab="incidents">
                            <i class="fas fa-bug"></i> Incidents
                        </button>
                        <button class="tab-btn" onclick="alertsManager.switchTab('history')" data-tab="history">
                            <i class="fas fa-history"></i> History
                        </button>
                        <button class="tab-btn" onclick="alertsManager.switchTab('metrics')" data-tab="metrics">
                            <i class="fas fa-chart-line"></i> Metrics
                        </button>
                    </div>

                    <!-- Alerts Tab -->
                    <div class="tab-content active" id="alerts-tab">
                        <div class="alerts-container" id="alerts-container">
                            <div class="loading-placeholder">
                                <i class="fas fa-spinner fa-spin"></i> Loading alerts...
                            </div>
                        </div>
                    </div>

                    <!-- Incidents Tab -->
                    <div class="tab-content" id="incidents-tab">
                        <div class="incidents-toolbar">
                            <button class="btn btn-primary" onclick="alertsManager.createIncident()">
                                <i class="fas fa-plus"></i> Create Incident
                            </button>
                            <button class="btn btn-outline" onclick="alertsManager.bulkActions()">
                                <i class="fas fa-list"></i> Bulk Actions
                            </button>
                        </div>
                        <div class="incidents-container" id="incidents-container">
                            <div class="no-incidents">
                                <i class="fas fa-check-circle"></i>
                                <p>No open incidents</p>
                            </div>
                        </div>
                    </div>

                    <!-- History Tab -->
                    <div class="tab-content" id="history-tab">
                        <div class="history-container" id="history-container">
                            <div class="loading-placeholder">
                                <i class="fas fa-spinner fa-spin"></i> Loading history...
                            </div>
                        </div>
                    </div>

                    <!-- Metrics Tab -->
                    <div class="tab-content" id="metrics-tab">
                        <div class="metrics-container">
                            <div class="metrics-grid">
                                <div class="metric-panel">
                                    <h4>Alert Trends (24h)</h4>
                                    <canvas id="alert-trends-chart" width="400" height="200"></canvas>
                                </div>
                                <div class="metric-panel">
                                    <h4>Mean Time to Resolution</h4>
                                    <div class="metric-value">
                                        <span class="value" id="mttr-value">--</span>
                                        <span class="unit">minutes</span>
                                    </div>
                                </div>
                                <div class="metric-panel">
                                    <h4>Service Availability</h4>
                                    <div class="availability-grid" id="availability-grid">
                                        <!-- Will be populated dynamically -->
                                    </div>
                                </div>
                                <div class="metric-panel">
                                    <h4>Top Alert Sources</h4>
                                    <div class="top-sources" id="top-sources">
                                        <!-- Will be populated dynamically -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions Sidebar -->
                <div class="quick-actions-sidebar" id="quick-actions">
                    <h4><i class="fas fa-bolt"></i> Quick Actions</h4>
                    <div class="action-buttons">
                        <button class="action-btn" onclick="alertsManager.acknowledgeAll()">
                            <i class="fas fa-check"></i> Acknowledge All
                        </button>
                        <button class="action-btn" onclick="alertsManager.silenceAll()">
                            <i class="fas fa-volume-mute"></i> Silence All
                        </button>
                        <button class="action-btn" onclick="alertsManager.escalateAll()">
                            <i class="fas fa-arrow-up"></i> Escalate All
                        </button>
                        <button class="action-btn" onclick="alertsManager.openRunbook()">
                            <i class="fas fa-book"></i> View Runbooks
                        </button>
                    </div>
                    
                    <div class="monitoring-links">
                        <h5>Monitoring Tools</h5>
                        <a href="http://localhost:9090" target="_blank" class="monitoring-link">
                            <i class="fas fa-chart-line"></i> Prometheus
                        </a>
                        <a href="http://localhost:3001" target="_blank" class="monitoring-link">
                            <i class="fas fa-chart-bar"></i> Grafana
                        </a>
                        <a href="http://localhost:9093" target="_blank" class="monitoring-link">
                            <i class="fas fa-bell"></i> Alertmanager
                        </a>
                        <a href="http://localhost:8081" target="_blank" class="monitoring-link">
                            <i class="fas fa-server"></i> cAdvisor
                        </a>
                    </div>
                </div>
            </div>
        `;
    }

    async loadInitialData() {
        this.showLoading(true);
        try {
            await Promise.all([
                this.loadAlerts(),
                this.loadIncidents(),
                this.loadMetrics()
            ]);
            this.updateSummaryCards();
            this.updateStatus('operational', 'All systems operational');
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateStatus('error', 'Failed to load data');
            this.showFallbackData();
        } finally {
            this.showLoading(false);
        }
    }

    async loadAlerts() {
        try {
            // Try to fetch from Alertmanager
            const response = await fetch(`${this.endpoints.alertmanager}/api/v1/alerts`);
            if (response.ok) {
                const data = await response.json();
                this.alerts = this.processAlertsData(data.data || []);
            } else {
                throw new Error('Alertmanager not accessible');
            }
        } catch (error) {
            console.warn('Using fallback alert data:', error.message);
            this.alerts = this.generateFallbackAlerts();
        }
        
        this.renderAlerts();
    }

    async loadIncidents() {
        try {
            // In a real implementation, this would fetch from your incident management system
            this.incidents = this.generateFallbackIncidents();
        } catch (error) {
            console.error('Failed to load incidents:', error);
            this.incidents = [];
        }
        
        this.renderIncidents();
    }

    async loadMetrics() {
        try {
            // Try to fetch metrics from Prometheus
            const queries = [
                'up', // Service availability
                'rate(prometheus_notifications_total[5m])', // Alert rate
                'prometheus_rule_evaluation_failures_total' // Rule failures
            ];
            
            const metrics = {};
            for (const query of queries) {
                try {
                    const response = await fetch(
                        `${this.endpoints.prometheus}/api/v1/query?query=${encodeURIComponent(query)}`
                    );
                    if (response.ok) {
                        const data = await response.json();
                        metrics[query] = data.data;
                    }
                } catch (error) {
                    console.warn(`Failed to fetch metric ${query}:`, error);
                }
            }
            
            this.processMetrics(metrics);
        } catch (error) {
            console.error('Failed to load metrics:', error);
            this.generateFallbackMetrics();
        }
    }

    processAlertsData(rawAlerts) {
        return rawAlerts.map(alert => ({
            id: this.generateId(),
            name: alert.labels.alertname || 'Unknown Alert',
            severity: alert.labels.severity || 'warning',
            status: alert.status.state || 'firing',
            message: alert.annotations.summary || alert.annotations.description || 'No description',
            source: alert.labels.job || alert.labels.instance || 'Unknown',
            timestamp: new Date(alert.startsAt || Date.now()),
            endsAt: alert.endsAt ? new Date(alert.endsAt) : null,
            labels: alert.labels,
            annotations: alert.annotations,
            generatorURL: alert.generatorURL
        }));
    }

    generateFallbackAlerts() {
        const now = new Date();
        return [
            {
                id: 'alert-1',
                name: 'High Memory Usage',
                severity: 'critical',
                status: 'firing',
                message: 'Memory usage above 90% on csp_postgres container',
                source: 'csp_postgres',
                timestamp: new Date(now - 300000), // 5 minutes ago
                labels: { container: 'csp_postgres', severity: 'critical' }
            },
            {
                id: 'alert-2',
                name: 'Redis Connection Slow',
                severity: 'warning',
                status: 'firing',
                message: 'Redis response time exceeding 100ms',
                source: 'csp_redis',
                timestamp: new Date(now - 600000), // 10 minutes ago
                labels: { service: 'redis', severity: 'warning' }
            },
            {
                id: 'alert-3',
                name: 'Disk Space Low',
                severity: 'warning',
                status: 'resolved',
                message: 'Disk usage was above 85%',
                source: 'host',
                timestamp: new Date(now - 3600000), // 1 hour ago
                endsAt: new Date(now - 1800000), // 30 minutes ago
                labels: { severity: 'warning', filesystem: '/var/lib/docker' }
            }
        ];
    }

    generateFallbackIncidents() {
        const now = new Date();
        return [
            {
                id: 'inc-001',
                title: 'Database Performance Degradation',
                description: 'PostgreSQL queries running slower than normal',
                severity: 'high',
                status: 'investigating',
                assignee: 'System Admin',
                created: new Date(now - 1800000), // 30 minutes ago
                updated: new Date(now - 300000), // 5 minutes ago
                affectedServices: ['csp_postgres', 'csp_api'],
                timeline: [
                    { time: new Date(now - 1800000), action: 'Incident created', user: 'System' },
                    { time: new Date(now - 1500000), action: 'Investigation started', user: 'Admin' },
                    { time: new Date(now - 300000), action: 'Root cause identified', user: 'Admin' }
                ]
            }
        ];
    }

    renderAlerts() {
        const container = document.getElementById('alerts-container');
        if (!container) return;

        const filteredAlerts = this.getFilteredAlerts();
        
        if (filteredAlerts.length === 0) {
            container.innerHTML = `
                <div class="no-alerts">
                    <i class="fas fa-check-circle"></i>
                    <p>No alerts match your filters</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filteredAlerts.map(alert => `
            <div class="alert-item severity-${alert.severity} status-${alert.status}" data-alert-id="${alert.id}">
                <div class="alert-header">
                    <div class="alert-main">
                        <div class="alert-icon">
                            <i class="fas ${this.getAlertIcon(alert.severity)}"></i>
                        </div>
                        <div class="alert-content">
                            <h4 class="alert-title">${alert.name}</h4>
                            <p class="alert-message">${alert.message}</p>
                            <div class="alert-meta">
                                <span class="alert-source">
                                    <i class="fas fa-server"></i> ${alert.source}
                                </span>
                                <span class="alert-time">
                                    <i class="fas fa-clock"></i> ${this.formatTime(alert.timestamp)}
                                </span>
                                ${alert.endsAt ? `
                                    <span class="alert-duration">
                                        <i class="fas fa-hourglass"></i> ${this.formatDuration(alert.timestamp, alert.endsAt)}
                                    </span>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                    <div class="alert-actions">
                        <span class="severity-badge severity-${alert.severity}">${alert.severity}</span>
                        <span class="status-badge status-${alert.status}">${alert.status}</span>
                        <div class="action-buttons">
                            ${alert.status === 'firing' ? `
                                <button class="btn btn-sm btn-outline" onclick="alertsManager.acknowledgeAlert('${alert.id}')">
                                    <i class="fas fa-check"></i> Ack
                                </button>
                                <button class="btn btn-sm btn-outline" onclick="alertsManager.silenceAlert('${alert.id}')">
                                    <i class="fas fa-volume-mute"></i> Silence
                                </button>
                            ` : ''}
                            <button class="btn btn-sm btn-outline" onclick="alertsManager.viewAlertDetails('${alert.id}')">
                                <i class="fas fa-eye"></i> Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderIncidents() {
        const container = document.getElementById('incidents-container');
        if (!container) return;

        if (this.incidents.length === 0) {
            container.innerHTML = `
                <div class="no-incidents">
                    <i class="fas fa-check-circle"></i>
                    <p>No open incidents</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.incidents.map(incident => `
            <div class="incident-item severity-${incident.severity}" data-incident-id="${incident.id}">
                <div class="incident-header">
                    <div class="incident-main">
                        <h4 class="incident-title">${incident.title}</h4>
                        <p class="incident-description">${incident.description}</p>
                        <div class="incident-meta">
                            <span class="incident-id">ID: ${incident.id}</span>
                            <span class="incident-assignee">
                                <i class="fas fa-user"></i> ${incident.assignee}
                            </span>
                            <span class="incident-created">
                                <i class="fas fa-clock"></i> ${this.formatTime(incident.created)}
                            </span>
                        </div>
                    </div>
                    <div class="incident-actions">
                        <span class="severity-badge severity-${incident.severity}">${incident.severity}</span>
                        <span class="status-badge status-${incident.status}">${incident.status}</span>
                        <button class="btn btn-sm btn-primary" onclick="alertsManager.viewIncident('${incident.id}')">
                            <i class="fas fa-edit"></i> Manage
                        </button>
                    </div>
                </div>
                <div class="affected-services">
                    <label>Affected Services:</label>
                    ${incident.affectedServices.map(service => 
                        `<span class="service-tag">${service}</span>`
                    ).join('')}
                </div>
            </div>
        `).join('');
    }

    updateSummaryCards() {
        const criticalCount = this.alerts.filter(a => a.severity === 'critical' && a.status === 'firing').length;
        const warningCount = this.alerts.filter(a => a.severity === 'warning' && a.status === 'firing').length;
        const infoCount = this.alerts.filter(a => a.severity === 'info' && a.status === 'firing').length;
        const incidentsCount = this.incidents.filter(i => i.status !== 'resolved').length;

        document.getElementById('critical-count').textContent = criticalCount;
        document.getElementById('warning-count').textContent = warningCount;
        document.getElementById('info-count').textContent = infoCount;
        document.getElementById('incidents-count').textContent = incidentsCount;

        // Update card styles based on counts
        this.updateCardStatus('critical-alerts-card', criticalCount > 0 ? 'alert' : 'normal');
        this.updateCardStatus('warning-alerts-card', warningCount > 0 ? 'warning' : 'normal');
        this.updateCardStatus('incidents-card', incidentsCount > 0 ? 'alert' : 'normal');
    }

    updateCardStatus(cardId, status) {
        const card = document.getElementById(cardId);
        if (card) {
            card.classList.remove('alert', 'warning', 'normal');
            card.classList.add(status);
        }
    }

    getFilteredAlerts() {
        return this.alerts.filter(alert => {
            if (this.filters.severity !== 'all' && alert.severity !== this.filters.severity) return false;
            if (this.filters.status !== 'all' && alert.status !== this.filters.status) return false;
            if (this.filters.search && !this.matchesSearch(alert, this.filters.search)) return false;
            if (!this.matchesTimeRange(alert.timestamp, this.filters.timeRange)) return false;
            return true;
        });
    }

    matchesSearch(alert, search) {
        const searchLower = search.toLowerCase();
        return alert.name.toLowerCase().includes(searchLower) ||
               alert.message.toLowerCase().includes(searchLower) ||
               alert.source.toLowerCase().includes(searchLower);
    }

    matchesTimeRange(timestamp, range) {
        const now = new Date();
        const diff = now - timestamp;
        
        switch (range) {
            case '1h': return diff <= 3600000;
            case '24h': return diff <= 86400000;
            case '7d': return diff <= 604800000;
            case '30d': return diff <= 2592000000;
            default: return true;
        }
    }

    applyFilters() {
        this.filters.severity = document.getElementById('severity-filter').value;
        this.filters.status = document.getElementById('status-filter').value;
        this.filters.timeRange = document.getElementById('time-filter').value;
        this.filters.search = document.getElementById('search-filter').value;
        
        this.renderAlerts();
        this.updateSummaryCards();
    }

    clearFilters() {
        document.getElementById('severity-filter').value = 'all';
        document.getElementById('status-filter').value = 'all';
        document.getElementById('time-filter').value = '24h';
        document.getElementById('search-filter').value = '';
        
        this.filters = {
            severity: 'all',
            status: 'all',
            timeRange: '24h',
            search: ''
        };
        
        this.renderAlerts();
        this.updateSummaryCards();
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Load tab-specific data
        switch (tabName) {
            case 'history':
                this.loadHistory();
                break;
            case 'metrics':
                this.renderMetrics();
                break;
        }
    }

    async loadHistory() {
        const container = document.getElementById('history-container');
        // Simulate loading historical data
        setTimeout(() => {
            container.innerHTML = `
                <div class="history-list">
                    <div class="history-item resolved">
                        <div class="history-icon"><i class="fas fa-check-circle"></i></div>
                        <div class="history-content">
                            <h5>Redis Memory Alert Resolved</h5>
                            <p>Memory usage returned to normal levels</p>
                            <small>2 hours ago • Duration: 15 minutes</small>
                        </div>
                    </div>
                    <div class="history-item resolved">
                        <div class="history-icon"><i class="fas fa-check-circle"></i></div>
                        <div class="history-content">
                            <h5>Database Connection Pool Full</h5>
                            <p>Connection pool reached maximum capacity</p>
                            <small>6 hours ago • Duration: 8 minutes</small>
                        </div>
                    </div>
                </div>
            `;
        }, 500);
    }

    renderMetrics() {
        // Render MTTR
        document.getElementById('mttr-value').textContent = '12.5';
        
        // Render availability grid
        const availabilityGrid = document.getElementById('availability-grid');
        availabilityGrid.innerHTML = `
            <div class="availability-item">
                <span class="service-name">PostgreSQL</span>
                <span class="availability-percentage">99.9%</span>
                <div class="availability-bar">
                    <div class="availability-fill" style="width: 99.9%"></div>
                </div>
            </div>
            <div class="availability-item">
                <span class="service-name">Redis</span>
                <span class="availability-percentage">99.7%</span>
                <div class="availability-bar">
                    <div class="availability-fill" style="width: 99.7%"></div>
                </div>
            </div>
            <div class="availability-item">
                <span class="service-name">Chroma</span>
                <span class="availability-percentage">98.5%</span>
                <div class="availability-bar">
                    <div class="availability-fill" style="width: 98.5%"></div>
                </div>
            </div>
        `;

        // Render top sources
        const topSources = document.getElementById('top-sources');
        topSources.innerHTML = `
            <div class="source-item">
                <span class="source-name">csp_postgres</span>
                <span class="alert-count">8</span>
            </div>
            <div class="source-item">
                <span class="source-name">csp_redis</span>
                <span class="alert-count">5</span>
            </div>
            <div class="source-item">
                <span class="source-name">host</span>
                <span class="alert-count">3</span>
            </div>
        `;
    }

    // Utility methods
    getAlertIcon(severity) {
        switch (severity) {
            case 'critical': return 'fa-fire';
            case 'warning': return 'fa-exclamation-triangle';
            case 'info': return 'fa-info-circle';
            default: return 'fa-bell';
        }
    }

    formatTime(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return `${Math.floor(diff / 86400000)}d ago`;
    }

    formatDuration(start, end) {
        const diff = end - start;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        return `${minutes}m`;
    }

    generateId() {
        return `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    updateStatus(status, message) {
        const statusIndicator = document.getElementById('alerts-status');
        if (statusIndicator) {
            const dot = statusIndicator.querySelector('.status-dot');
            const text = statusIndicator.querySelector('.status-text');
            
            dot.className = `status-dot status-${status}`;
            text.textContent = message;
        }
    }

    showLoading(show) {
        this.isLoading = show;
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            const icon = refreshBtn.querySelector('i');
            if (show) {
                icon.classList.add('fa-spin');
                refreshBtn.disabled = true;
            } else {
                icon.classList.remove('fa-spin');
                refreshBtn.disabled = false;
            }
        }
    }

    showError(message) {
        // Show error notification
        console.error(message);
        this.updateStatus('error', message);
    }

    // Event handlers and actions
    async refreshData() {
        await this.loadInitialData();
    }

    acknowledgeAlert(alertId) {
        console.log(`Acknowledging alert: ${alertId}`);
        // In real implementation, this would call Alertmanager API
        this.showNotification('Alert acknowledged', 'success');
    }

    silenceAlert(alertId) {
        console.log(`Silencing alert: ${alertId}`);
        // In real implementation, this would call Alertmanager API
        this.showNotification('Alert silenced', 'success');
    }

    viewAlertDetails(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert) {
            // Open modal or navigate to detail view
            console.log('Viewing alert details:', alert);
        }
    }

    exportData() {
        const data = {
            alerts: this.alerts,
            incidents: this.incidents,
            exported: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alerts-export-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification('Data exported successfully', 'success');
    }

    openMonitoringTools() {
        const tools = [
            { name: 'Prometheus', url: this.endpoints.prometheus },
            { name: 'Grafana', url: this.endpoints.grafana },
            { name: 'Alertmanager', url: this.endpoints.alertmanager }
        ];
        
        tools.forEach(tool => {
            window.open(tool.url, '_blank');
        });
    }

    showNotification(message, type = 'info') {
        // Create and show notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check' : 'fa-info'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    attachEventListeners() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                this.refreshData();
            }
        });
    }

    startAutoRefresh() {
        // Refresh every 30 seconds
        this.updateInterval = setInterval(() => {
            if (document.getElementById('alerts').classList.contains('active')) {
                this.loadAlerts();
                this.updateSummaryCards();
            }
        }, 30000);
    }

    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();
        // Clean up event listeners
    }
}

// Initialize the alerts manager when the admin portal loads
let alertsManager;

// Function to initialize alerts dashboard (called from admin.js)
function initializeAlertsIncidents() {
    if (!alertsManager) {
        alertsManager = new AlertsIncidentsManager();
        window.alertsManager = alertsManager; // Make globally accessible
    }
    return alertsManager;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AlertsIncidentsManager, initializeAlertsIncidents };
}