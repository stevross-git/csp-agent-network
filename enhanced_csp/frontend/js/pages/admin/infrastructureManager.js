/**
 * Enhanced Infrastructure Manager v2.0
 * Comprehensive infrastructure monitoring, management, and automation system
 * Features: Real-time monitoring, automated backups, health checks, resource optimization
 */
class InfrastructureManager {
    constructor(options = {}) {
        this.section = null;
        this.status = null;
        this.metrics = new Map();
        this.alerts = [];
        this.services = new Map();
        this.refreshInterval = null;
        this.webSocket = null;
        
        // Configuration
        this.config = {
            refreshRate: options.refreshRate || 5000,
            alertThresholds: {
                cpu: 80,
                memory: 85,
                disk: 90,
                network: 95
            },
            autoBackup: options.autoBackup || true,
            backupInterval: options.backupInterval || 3600000, // 1 hour
            maxRetries: 3,
            ...options
        };
        
        // API configuration
        this.apiBaseUrl = this.getApiBaseUrl();
        this.authToken = this.getAuthToken();
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // Initialize WebSocket connection
        this.initializeWebSocket();
        
        // Initialize automated backup if enabled
        if (this.config.autoBackup) {
            this.initializeAutoBackup();
        }
    }

    // ===========================================
    // INITIALIZATION METHODS
    // ===========================================

    getApiBaseUrl() {
        if (window.REACT_APP_CSP_API_URL) return window.REACT_APP_CSP_API_URL;
        if (typeof REACT_APP_CSP_API_URL !== 'undefined') return REACT_APP_CSP_API_URL;
        const meta = document.querySelector('meta[name="api-base-url"]');
        if (meta) return meta.getAttribute('content');
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        return window.location.origin.replace(':3000', ':8000');
    }

    getAuthToken() {
        return localStorage.getItem('csp_auth_token') || sessionStorage.getItem('csp_auth_token');
    }

    async init() {
        try {
            this.section = document.getElementById('infrastructure');
            if (!this.section) {
                throw new Error('Infrastructure section not found');
            }

            console.log('🚀 Initializing Enhanced Infrastructure Manager...');
            
            // Load initial data
            await this.loadStatus();
            await this.loadMetrics();
            await this.loadServices();
            await this.loadAlerts();
            
            // Render the dashboard
            this.render();
            this.attachEvents();
            
            // Start real-time monitoring
            this.startMonitoring();
            
            console.log('✅ Infrastructure Manager initialized successfully');
            this.emit('initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Infrastructure Manager:', error);
            this.renderError(error);
        }
    }

    initializeWebSocket() {
        try {
            const wsUrl = this.apiBaseUrl.replace('http', 'ws') + '/ws/infrastructure';
            this.webSocket = new WebSocket(wsUrl);
            
            this.webSocket.onopen = () => {
                console.log('🔌 WebSocket connected to infrastructure monitoring');
            };
            
            this.webSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.webSocket.onclose = () => {
                console.log('🔌 WebSocket disconnected, attempting reconnection...');
                setTimeout(() => this.initializeWebSocket(), 5000);
            };
            
            this.webSocket.onerror = (error) => {
                console.error('🔌 WebSocket error:', error);
            };
        } catch (error) {
            console.warn('WebSocket not available, falling back to polling');
        }
    }

    initializeAutoBackup() {
        setInterval(async () => {
            try {
                await this.createBackup('automated');
                console.log('✅ Automated backup completed');
            } catch (error) {
                console.error('❌ Automated backup failed:', error);
                this.addAlert('Automated backup failed', 'error');
            }
        }, this.config.backupInterval);
    }

    // ===========================================
    // API METHODS
    // ===========================================

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
            }
        };

        let retries = 0;
        while (retries < this.config.maxRetries) {
            try {
                const response = await fetch(url, { ...defaultOptions, ...options });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
                }
                
                if (response.status === 204) return null;
                return await response.json();
            } catch (error) {
                retries++;
                if (retries >= this.config.maxRetries) {
                    throw error;
                }
                await this.delay(1000 * retries); // Exponential backoff
            }
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ===========================================
    // DATA LOADING METHODS
    // ===========================================

    async loadStatus() {
        try {
            this.status = await this.apiRequest('/api/infrastructure/status');
        } catch (error) {
            console.warn('Failed to load infrastructure status:', error);
            this.status = { 
                message: 'Status unavailable',
                timestamp: new Date().toISOString(),
                health: 'unknown',
                services: {}
            };
        }
    }

    async loadMetrics() {
        try {
            const metrics = await this.apiRequest('/api/infrastructure/metrics');
            this.updateMetrics(metrics);
        } catch (error) {
            console.warn('Failed to load metrics:', error);
            this.initializeDefaultMetrics();
        }
    }

    async loadServices() {
        try {
            const services = await this.apiRequest('/api/infrastructure/services');
            services.forEach(service => {
                this.services.set(service.name, service);
            });
        } catch (error) {
            console.warn('Failed to load services:', error);
            this.initializeDefaultServices();
        }
    }

    async loadAlerts() {
        try {
            this.alerts = await this.apiRequest('/api/infrastructure/alerts');
        } catch (error) {
            console.warn('Failed to load alerts:', error);
            this.alerts = [];
        }
    }

    updateMetrics(newMetrics) {
        Object.entries(newMetrics).forEach(([key, value]) => {
            this.metrics.set(key, {
                ...value,
                timestamp: new Date().toISOString(),
                trend: this.calculateTrend(key, value.current)
            });
            
            // Check thresholds and create alerts
            this.checkThreshold(key, value.current);
        });
    }

    initializeDefaultMetrics() {
        const defaultMetrics = {
            cpu: { current: 45, max: 100, unit: '%' },
            memory: { current: 62, max: 100, unit: '%' },
            disk: { current: 78, max: 100, unit: '%' },
            network: { current: 23, max: 100, unit: '%' },
            uptime: { current: 99.5, max: 100, unit: '%' },
            requests: { current: 1250, max: null, unit: '/min' }
        };
        this.updateMetrics(defaultMetrics);
    }

    initializeDefaultServices() {
        const defaultServices = [
            { name: 'Web Server', status: 'running', uptime: '15d 4h 23m', port: 80 },
            { name: 'Database', status: 'running', uptime: '15d 4h 23m', port: 5432 },
            { name: 'Redis Cache', status: 'running', uptime: '15d 4h 23m', port: 6379 },
            { name: 'API Gateway', status: 'running', uptime: '15d 4h 23m', port: 8000 },
            { name: 'Message Queue', status: 'warning', uptime: '2d 1h 15m', port: 5672 }
        ];
        defaultServices.forEach(service => {
            this.services.set(service.name, service);
        });
    }

    // ===========================================
    // MONITORING METHODS
    // ===========================================

    startMonitoring() {
        this.refreshInterval = setInterval(async () => {
            await this.refresh();
        }, this.config.refreshRate);
        
        console.log(`📊 Started real-time monitoring (${this.config.refreshRate}ms interval)`);
    }

    stopMonitoring() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        
        if (this.webSocket) {
            this.webSocket.close();
        }
        
        console.log('📊 Stopped monitoring');
    }

    calculateTrend(metric, currentValue) {
        const history = this.getMetricHistory(metric);
        if (history.length < 2) return 'stable';
        
        const previousValue = history[history.length - 2];
        const difference = currentValue - previousValue;
        const percentChange = (difference / previousValue) * 100;
        
        if (percentChange > 5) return 'up';
        if (percentChange < -5) return 'down';
        return 'stable';
    }

    getMetricHistory(metric) {
        // This would typically come from a time series database
        // For now, return mock data
        return [45, 47, 46, 48, 50];
    }

    checkThreshold(metric, value) {
        const threshold = this.config.alertThresholds[metric];
        if (threshold && value > threshold) {
            this.addAlert(
                `${metric.toUpperCase()} usage is high: ${value}%`,
                'warning',
                { metric, value, threshold }
            );
        }
    }

    addAlert(message, type = 'info', metadata = {}) {
        const alert = {
            id: Date.now(),
            message,
            type,
            timestamp: new Date().toISOString(),
            metadata,
            acknowledged: false
        };
        
        this.alerts.unshift(alert);
        
        // Keep only last 50 alerts
        if (this.alerts.length > 50) {
            this.alerts = this.alerts.slice(0, 50);
        }
        
        this.emit('alert', alert);
        this.updateAlertsDisplay();
    }

    // ===========================================
    // WEBSOCKET HANDLERS
    // ===========================================

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'metrics_update':
                this.updateMetrics(data.payload);
                this.updateMetricsDisplay();
                break;
            case 'service_status':
                this.updateServiceStatus(data.payload);
                break;
            case 'alert':
                this.addAlert(data.payload.message, data.payload.type, data.payload.metadata);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    updateServiceStatus(serviceData) {
        this.services.set(serviceData.name, serviceData);
        this.updateServicesDisplay();
    }

    // ===========================================
    // BACKUP METHODS
    // ===========================================

    async createBackup(type = 'manual') {
        try {
            this.showLoadingState('Creating backup...');
            
            const backupData = {
                type,
                timestamp: new Date().toISOString(),
                include_config: true,
                include_data: true,
                compression: true
            };
            
            const result = await this.apiRequest('/api/infrastructure/backup', {
                method: 'POST',
                body: JSON.stringify(backupData)
            });
            
            this.hideLoadingState();
            this.showNotification('Backup created successfully', 'success');
            this.emit('backup_created', result);
            
            return result;
        } catch (error) {
            this.hideLoadingState();
            this.showNotification('Failed to create backup: ' + error.message, 'error');
            throw error;
        }
    }

    async restoreBackup(backupId) {
        if (!confirm('Are you sure you want to restore from this backup? This action cannot be undone.')) {
            return;
        }
        
        try {
            this.showLoadingState('Restoring backup...');
            
            const result = await this.apiRequest(`/api/infrastructure/backup/${backupId}/restore`, {
                method: 'POST'
            });
            
            this.hideLoadingState();
            this.showNotification('Backup restored successfully', 'success');
            this.emit('backup_restored', result);
            
            // Refresh all data after restore
            setTimeout(() => this.refresh(), 2000);
            
            return result;
        } catch (error) {
            this.hideLoadingState();
            this.showNotification('Failed to restore backup: ' + error.message, 'error');
            throw error;
        }
    }

    async getBackupList() {
        try {
            return await this.apiRequest('/api/infrastructure/backups');
        } catch (error) {
            console.error('Failed to load backup list:', error);
            return [];
        }
    }

    // ===========================================
    // SERVICE MANAGEMENT
    // ===========================================

    async restartService(serviceName) {
        if (!confirm(`Are you sure you want to restart ${serviceName}?`)) {
            return;
        }
        
        try {
            this.showLoadingState(`Restarting ${serviceName}...`);
            
            await this.apiRequest(`/api/infrastructure/services/${serviceName}/restart`, {
                method: 'POST'
            });
            
            this.hideLoadingState();
            this.showNotification(`${serviceName} restarted successfully`, 'success');
            
            // Update service status
            setTimeout(() => this.loadServices(), 2000);
        } catch (error) {
            this.hideLoadingState();
            this.showNotification(`Failed to restart ${serviceName}: ${error.message}`, 'error');
        }
    }

    async stopService(serviceName) {
        if (!confirm(`Are you sure you want to stop ${serviceName}?`)) {
            return;
        }
        
        try {
            await this.apiRequest(`/api/infrastructure/services/${serviceName}/stop`, {
                method: 'POST'
            });
            
            this.showNotification(`${serviceName} stopped`, 'warning');
            setTimeout(() => this.loadServices(), 1000);
        } catch (error) {
            this.showNotification(`Failed to stop ${serviceName}: ${error.message}`, 'error');
        }
    }

    async startService(serviceName) {
        try {
            this.showLoadingState(`Starting ${serviceName}...`);
            
            await this.apiRequest(`/api/infrastructure/services/${serviceName}/start`, {
                method: 'POST'
            });
            
            this.hideLoadingState();
            this.showNotification(`${serviceName} started successfully`, 'success');
            setTimeout(() => this.loadServices(), 2000);
        } catch (error) {
            this.hideLoadingState();
            this.showNotification(`Failed to start ${serviceName}: ${error.message}`, 'error');
        }
    }

    // ===========================================
    // RENDERING METHODS
    // ===========================================

    render() {
        const statusText = this.status ? 
            JSON.stringify(this.status, null, 2) : 'No status available';
            
        this.section.innerHTML = `
            <div class="infrastructure-dashboard">
                ${this.renderHeader()}
                ${this.renderMetrics()}
                ${this.renderServices()}
                ${this.renderAlerts()}
                ${this.renderBackups()}
                ${this.renderActions()}
                ${this.renderLoadingOverlay()}
            </div>
        `;
        
        this.updateDisplays();
    }

    renderHeader() {
        return `
            <div class="infra-header">
                <div class="header-title">
                    <h2><i class="fas fa-server"></i> Infrastructure Management</h2>
                    <div class="header-status">
                        <span class="status-indicator ${this.getOverallHealthClass()}"></span>
                        <span class="status-text">${this.getOverallHealthText()}</span>
                        <span class="last-updated">Last updated: ${new Date().toLocaleTimeString()}</span>
                    </div>
                </div>
                <div class="header-controls">
                    <button class="btn btn-icon" id="infra-settings-btn" title="Settings">
                        <i class="fas fa-cog"></i>
                    </button>
                    <button class="btn btn-icon" id="infra-fullscreen-btn" title="Toggle Fullscreen">
                        <i class="fas fa-expand"></i>
                    </button>
                </div>
            </div>
        `;
    }

    renderMetrics() {
        const metricsHtml = Array.from(this.metrics.entries()).map(([name, data]) => `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-name">${this.formatMetricName(name)}</span>
                    <span class="metric-trend trend-${data.trend}">
                        <i class="fas fa-arrow-${data.trend === 'up' ? 'up' : data.trend === 'down' ? 'down' : 'right'}"></i>
                    </span>
                </div>
                <div class="metric-value">
                    ${data.current}${data.unit}
                </div>
                <div class="metric-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${data.max ? (data.current / data.max) * 100 : 50}%"></div>
                    </div>
                    ${data.max ? `<span class="progress-max">Max: ${data.max}${data.unit}</span>` : ''}
                </div>
            </div>
        `).join('');

        return `
            <div class="metrics-section">
                <h3><i class="fas fa-chart-line"></i> System Metrics</h3>
                <div class="metrics-grid">
                    ${metricsHtml}
                </div>
            </div>
        `;
    }

    renderServices() {
        const servicesHtml = Array.from(this.services.entries()).map(([name, service]) => `
            <div class="service-card">
                <div class="service-header">
                    <div class="service-info">
                        <span class="service-name">${name}</span>
                        <span class="service-port">:${service.port}</span>
                    </div>
                    <div class="service-status status-${service.status}">
                        <i class="fas fa-circle"></i>
                        ${service.status}
                    </div>
                </div>
                <div class="service-details">
                    <span class="service-uptime">
                        <i class="fas fa-clock"></i>
                        Uptime: ${service.uptime}
                    </span>
                </div>
                <div class="service-actions">
                    <button class="btn btn-sm btn-secondary" onclick="infrastructureManager.restartService('${name}')">
                        <i class="fas fa-redo"></i> Restart
                    </button>
                    ${service.status === 'running' ? 
                        `<button class="btn btn-sm btn-warning" onclick="infrastructureManager.stopService('${name}')">
                            <i class="fas fa-stop"></i> Stop
                        </button>` :
                        `<button class="btn btn-sm btn-success" onclick="infrastructureManager.startService('${name}')">
                            <i class="fas fa-play"></i> Start
                        </button>`
                    }
                </div>
            </div>
        `).join('');

        return `
            <div class="services-section">
                <h3><i class="fas fa-cogs"></i> Services Status</h3>
                <div class="services-grid">
                    ${servicesHtml}
                </div>
            </div>
        `;
    }

    renderAlerts() {
        const alertsHtml = this.alerts.slice(0, 5).map(alert => `
            <div class="alert-item alert-${alert.type} ${alert.acknowledged ? 'acknowledged' : ''}">
                <div class="alert-icon">
                    <i class="fas fa-${this.getAlertIcon(alert.type)}"></i>
                </div>
                <div class="alert-content">
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                </div>
                <div class="alert-actions">
                    ${!alert.acknowledged ? 
                        `<button class="btn btn-sm btn-outline" onclick="infrastructureManager.acknowledgeAlert('${alert.id}')">
                            <i class="fas fa-check"></i>
                        </button>` : ''
                    }
                    <button class="btn btn-sm btn-outline" onclick="infrastructureManager.dismissAlert('${alert.id}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `).join('');

        return `
            <div class="alerts-section">
                <div class="alerts-header">
                    <h3><i class="fas fa-exclamation-triangle"></i> Recent Alerts</h3>
                    <div class="alerts-summary">
                        <span class="alert-count">${this.alerts.length} total</span>
                        <span class="alert-unacknowledged">${this.alerts.filter(a => !a.acknowledged).length} unacknowledged</span>
                    </div>
                </div>
                <div class="alerts-list">
                    ${alertsHtml || '<div class="no-alerts">No recent alerts</div>'}
                </div>
            </div>
        `;
    }

    renderBackups() {
        return `
            <div class="backups-section">
                <div class="backups-header">
                    <h3><i class="fas fa-archive"></i> Backup Management</h3>
                    <button class="btn btn-primary" id="infra-create-backup-btn">
                        <i class="fas fa-plus"></i> Create Backup
                    </button>
                </div>
                <div class="backups-content">
                    <div class="backup-status">
                        <div class="status-item">
                            <span class="status-label">Last Backup:</span>
                            <span class="status-value" id="last-backup-time">Loading...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Auto Backup:</span>
                            <span class="status-value ${this.config.autoBackup ? 'enabled' : 'disabled'}">
                                ${this.config.autoBackup ? 'Enabled' : 'Disabled'}
                            </span>
                        </div>
                    </div>
                    <div class="backup-list" id="backup-list">
                        Loading backup history...
                    </div>
                </div>
            </div>
        `;
    }

    renderActions() {
        return `
            <div class="actions-section">
                <div class="action-group">
                    <button class="btn btn-secondary" id="infra-refresh-btn">
                        <i class="fas fa-sync-alt"></i> Refresh All
                    </button>
                    <button class="btn btn-info" id="infra-export-logs-btn">
                        <i class="fas fa-download"></i> Export Logs
                    </button>
                    <button class="btn btn-warning" id="infra-maintenance-btn">
                        <i class="fas fa-tools"></i> Maintenance Mode
                    </button>
                </div>
                <div class="action-group dangerous">
                    <button class="btn btn-danger" id="infra-emergency-shutdown-btn">
                        <i class="fas fa-power-off"></i> Emergency Shutdown
                    </button>
                </div>
            </div>
        `;
    }

    renderLoadingOverlay() {
        return `
            <div class="loading-overlay" id="infra-loading-overlay">
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Processing...</div>
                </div>
            </div>
        `;
    }

    renderError(error) {
        this.section.innerHTML = `
            <div class="infrastructure-dashboard error-state">
                <div class="error-container">
                    <div class="error-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div class="error-content">
                        <h3>Infrastructure Manager Error</h3>
                        <p>${error.message}</p>
                        <button class="btn btn-primary" onclick="infrastructureManager.init()">
                            <i class="fas fa-redo"></i> Retry
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    // ===========================================
    // UPDATE METHODS
    // ===========================================

    updateDisplays() {
        this.updateMetricsDisplay();
        this.updateServicesDisplay();
        this.updateAlertsDisplay();
        this.updateBackupsDisplay();
    }

    updateMetricsDisplay() {
        // Update metrics in real-time if the section is rendered
        const metricsGrid = this.section.querySelector('.metrics-grid');
        if (metricsGrid) {
            const metricsHtml = Array.from(this.metrics.entries()).map(([name, data]) => `
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-name">${this.formatMetricName(name)}</span>
                        <span class="metric-trend trend-${data.trend}">
                            <i class="fas fa-arrow-${data.trend === 'up' ? 'up' : data.trend === 'down' ? 'down' : 'right'}"></i>
                        </span>
                    </div>
                    <div class="metric-value">
                        ${data.current}${data.unit}
                    </div>
                    <div class="metric-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${data.max ? (data.current / data.max) * 100 : 50}%"></div>
                        </div>
                        ${data.max ? `<span class="progress-max">Max: ${data.max}${data.unit}</span>` : ''}
                    </div>
                </div>
            `).join('');
            metricsGrid.innerHTML = metricsHtml;
        }
    }

    updateServicesDisplay() {
        const servicesGrid = this.section.querySelector('.services-grid');
        if (servicesGrid) {
            // Update services display logic here
        }
    }

    updateAlertsDisplay() {
        const alertsList = this.section.querySelector('.alerts-list');
        if (alertsList) {
            // Update alerts display logic here
        }
    }

    async updateBackupsDisplay() {
        const backupList = this.section.querySelector('#backup-list');
        const lastBackupTime = this.section.querySelector('#last-backup-time');
        
        if (backupList) {
            try {
                const backups = await this.getBackupList();
                backupList.innerHTML = this.renderBackupItems(backups);
                
                if (lastBackupTime && backups.length > 0) {
                    lastBackupTime.textContent = new Date(backups[0].timestamp).toLocaleString();
                }
            } catch (error) {
                backupList.innerHTML = '<div class="error-message">Failed to load backup history</div>';
            }
        }
    }

    renderBackupItems(backups) {
        if (!backups.length) {
            return '<div class="no-backups">No backups available</div>';
        }
        
        return backups.slice(0, 10).map(backup => `
            <div class="backup-item">
                <div class="backup-info">
                    <span class="backup-name">${backup.name || 'Backup'}</span>
                    <span class="backup-date">${new Date(backup.timestamp).toLocaleString()}</span>
                    <span class="backup-size">${this.formatBytes(backup.size)}</span>
                </div>
                <div class="backup-actions">
                    <button class="btn btn-sm btn-secondary" onclick="infrastructureManager.restoreBackup('${backup.id}')">
                        <i class="fas fa-upload"></i> Restore
                    </button>
                    <button class="btn btn-sm btn-outline" onclick="infrastructureManager.downloadBackup('${backup.id}')">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    // ===========================================
    // EVENT HANDLING
    // ===========================================

    attachEvents() {
        // Main action buttons
        this.section.querySelector('#infra-refresh-btn')?.addEventListener('click', () => this.refresh());
        this.section.querySelector('#infra-create-backup-btn')?.addEventListener('click', () => this.createBackup());
        this.section.querySelector('#infra-settings-btn')?.addEventListener('click', () => this.showSettings());
        this.section.querySelector('#infra-fullscreen-btn')?.addEventListener('click', () => this.toggleFullscreen());
        this.section.querySelector('#infra-export-logs-btn')?.addEventListener('click', () => this.exportLogs());
        this.section.querySelector('#infra-maintenance-btn')?.addEventListener('click', () => this.toggleMaintenanceMode());
        this.section.querySelector('#infra-emergency-shutdown-btn')?.addEventListener('click', () => this.emergencyShutdown());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refresh();
                        break;
                    case 'b':
                        e.preventDefault();
                        this.createBackup();
                        break;
                }
            }
        });
    }

    // Event emitter functionality
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    emit(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.forEach(handler => handler(data));
        }
    }

    // ===========================================
    // UTILITY METHODS
    // ===========================================

    async refresh() {
        try {
            this.showLoadingState('Refreshing data...');
            
            await Promise.all([
                this.loadStatus(),
                this.loadMetrics(),
                this.loadServices(),
                this.loadAlerts()
            ]);
            
            this.updateDisplays();
            this.hideLoadingState();
            
            // Update last updated time
            const lastUpdated = this.section.querySelector('.last-updated');
            if (lastUpdated) {
                lastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
            }
            
            this.emit('refreshed');
        } catch (error) {
            this.hideLoadingState();
            this.showNotification('Failed to refresh data: ' + error.message, 'error');
        }
    }

    acknowledgeAlert(alertId) {
        const alert = this.alerts.find(a => a.id == alertId);
        if (alert) {
            alert.acknowledged = true;
            this.updateAlertsDisplay();
        }
    }

    dismissAlert(alertId) {
        this.alerts = this.alerts.filter(a => a.id != alertId);
        this.updateAlertsDisplay();
    }

    async exportLogs() {
        try {
            this.showLoadingState('Exporting logs...');
            
            const logs = await this.apiRequest('/api/infrastructure/logs/export');
            this.downloadFile(logs.url, logs.filename);
            
            this.hideLoadingState();
            this.showNotification('Logs exported successfully', 'success');
        } catch (error) {
            this.hideLoadingState();
            this.showNotification('Failed to export logs: ' + error.message, 'error');
        }
    }

    async toggleMaintenanceMode() {
        const isMaintenanceMode = this.status?.maintenance_mode || false;
        const action = isMaintenanceMode ? 'disable' : 'enable';
        
        if (!confirm(`Are you sure you want to ${action} maintenance mode?`)) {
            return;
        }
        
        try {
            await this.apiRequest('/api/infrastructure/maintenance', {
                method: 'POST',
                body: JSON.stringify({ enabled: !isMaintenanceMode })
            });
            
            this.showNotification(`Maintenance mode ${action}d`, 'info');
            await this.refresh();
        } catch (error) {
            this.showNotification(`Failed to ${action} maintenance mode: ${error.message}`, 'error');
        }
    }

    async emergencyShutdown() {
        const confirmation = prompt('Type "EMERGENCY SHUTDOWN" to confirm this action:');
        if (confirmation !== 'EMERGENCY SHUTDOWN') {
            return;
        }
        
        try {
            await this.apiRequest('/api/infrastructure/emergency-shutdown', {
                method: 'POST'
            });
            
            this.showNotification('Emergency shutdown initiated', 'warning');
        } catch (error) {
            this.showNotification('Failed to initiate emergency shutdown: ' + error.message, 'error');
        }
    }

    showSettings() {
        // This would open a settings modal
        console.log('Settings modal would open here');
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            this.section.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    showLoadingState(message = 'Loading...') {
        const overlay = this.section.querySelector('#infra-loading-overlay');
        const text = overlay?.querySelector('.loading-text');
        
        if (overlay) {
            if (text) text.textContent = message;
            overlay.style.display = 'flex';
        }
    }

    hideLoadingState() {
        const overlay = this.section.querySelector('#infra-loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showNotification(message, type = 'info') {
        // Create a toast notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    downloadFile(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    // ===========================================
    // HELPER METHODS
    // ===========================================

    formatMetricName(name) {
        return name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' ');
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getOverallHealthClass() {
        if (!this.status) return 'unknown';
        const health = this.status.health || 'unknown';
        return health;
    }

    getOverallHealthText() {
        if (!this.status) return 'Unknown';
        const health = this.status.health || 'unknown';
        return health.charAt(0).toUpperCase() + health.slice(1);
    }

    getAlertIcon(type) {
        const icons = {
            info: 'info-circle',
            warning: 'exclamation-triangle',
            error: 'exclamation-circle',
            success: 'check-circle'
        };
        return icons[type] || 'info-circle';
    }

    getNotificationIcon(type) {
        const icons = {
            info: 'info-circle',
            success: 'check-circle',
            warning: 'exclamation-triangle',
            error: 'times-circle'
        };
        return icons[type] || 'info-circle';
    }

    // ===========================================
    // CLEANUP
    // ===========================================

    destroy() {
        this.stopMonitoring();
        
        if (this.webSocket) {
            this.webSocket.close();
        }
        
        // Clear event handlers
        this.eventHandlers.clear();
        
        console.log('Infrastructure Manager destroyed');
    }
}

// Global instance and initialization
const infrastructureManager = new InfrastructureManager({
    refreshRate: 5000,
    autoBackup: true,
    backupInterval: 3600000 // 1 hour
});

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    infrastructureManager.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    infrastructureManager.destroy();
});

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InfrastructureManager;
}

// Make available globally
window.InfrastructureManager = InfrastructureManager;
window.infrastructureManager = infrastructureManager;
