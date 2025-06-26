// js/pages/admin/monitoringManager.js
// Enhanced CSP Admin Portal - Monitoring Dashboard Manager

/**
 * MonitoringManager - Handles all monitoring dashboard functionality
 * Manages real-time charts, metrics, alerts, and system health monitoring
 */
class MonitoringManager {
    constructor(adminPortal) {
        this.adminPortal = adminPortal;
        this.isInitialized = false;
        this.updateInterval = null;
        this.charts = new Map();
        this.components = new Map();
        
        // Monitoring data stores
        this.data = {
            systemMetrics: {
                cpu: { current: 0, history: [] },
                memory: { current: 0, history: [] },
                disk: { current: 0, history: [] },
                network: { current: 0, history: [] },
                uptime: 0,
                activeConnections: 0
            },
            performanceMetrics: {
                responseTime: { current: 0, history: [] },
                throughput: { current: 0, history: [] },
                errorRate: { current: 0, history: [] },
                availability: { current: 100, history: [] }
            },
            securityMetrics: {
                threatLevel: 'low',
                blockedRequests: 0,
                activeSessions: 0,
                failedLogins: 0
            },
            alerts: [],
            logs: []
        };
        
        // Update intervals and settings
        this.settings = {
            refreshInterval: 5000, // 5 seconds
            historyLimit: 50, // Keep last 50 data points
            chartOptions: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 }
            }
        };
    }

    /**
     * Initialize the monitoring manager
     */
    async init() {
        try {
            console.log('🔄 Initializing Monitoring Manager...');
            
            // Load monitoring components
            await this.loadComponents();
            
            // Initialize charts
            await this.initializeCharts();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Load initial data
            await this.loadInitialData();
            
            // Start real-time updates
            this.startRealTimeUpdates();
            
            this.isInitialized = true;
            console.log('✅ Monitoring Manager initialized');
            
        } catch (error) {
            console.error('❌ Failed to initialize Monitoring Manager:', error);
            throw error;
        }
    }

    /**
     * Load monitoring components
     */
    async loadComponents() {
        try {
            // Initialize metric cards component
            this.components.set('metricCards', new MonitoringMetricCards(this));
            
            // Initialize alert panel component
            this.components.set('alertPanel', new MonitoringAlertPanel(this));
            
            // Initialize log viewer component
            this.components.set('logViewer', new MonitoringLogViewer(this));
            
            console.log('✅ Monitoring components loaded');
            
        } catch (error) {
            console.error('❌ Failed to load monitoring components:', error);
            // Continue with basic functionality
        }
    }

    /**
     * Initialize monitoring charts
     */
    async initializeCharts() {
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not available - charts will be disabled');
            return;
        }

        try {
            // System Performance Chart
            await this.createSystemPerformanceChart();
            
            // Network Traffic Chart
            await this.createNetworkTrafficChart();
            
            // Response Time Chart
            await this.createResponseTimeChart();
            
            // Resource Usage Chart
            await this.createResourceUsageChart();
            
            console.log('✅ Monitoring charts initialized');
            
        } catch (error) {
            console.error('❌ Failed to initialize charts:', error);
        }
    }

    /**
     * Create system performance chart
     */
    async createSystemPerformanceChart() {
        const canvas = document.getElementById('system-performance-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(),
                datasets: [
                    {
                        label: 'CPU Usage (%)',
                        data: this.data.systemMetrics.cpu.history,
                        borderColor: '#ff6b35',
                        backgroundColor: 'rgba(255, 107, 53, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Memory Usage (%)',
                        data: this.data.systemMetrics.memory.history,
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Disk Usage (%)',
                        data: this.data.systemMetrics.disk.history,
                        borderColor: '#4ea5d9',
                        backgroundColor: 'rgba(78, 165, 217, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                ...this.settings.chartOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { 
                        labels: { color: '#94a3b8' }
                    },
                    title: {
                        display: true,
                        text: 'System Performance',
                        color: '#f8fafc'
                    }
                }
            }
        });

        this.charts.set('systemPerformance', chart);
    }

    /**
     * Create network traffic chart
     */
    async createNetworkTrafficChart() {
        const canvas = document.getElementById('network-traffic-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.generateTimeLabels(),
                datasets: [
                    {
                        label: 'Incoming (MB/s)',
                        data: this.generateRandomData(30, 100),
                        backgroundColor: 'rgba(255, 107, 53, 0.7)',
                        borderColor: '#ff6b35',
                        borderWidth: 1
                    },
                    {
                        label: 'Outgoing (MB/s)',
                        data: this.generateRandomData(20, 80),
                        backgroundColor: 'rgba(0, 212, 170, 0.7)',
                        borderColor: '#00d4aa',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                ...this.settings.chartOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { 
                        labels: { color: '#94a3b8' }
                    },
                    title: {
                        display: true,
                        text: 'Network Traffic',
                        color: '#f8fafc'
                    }
                }
            }
        });

        this.charts.set('networkTraffic', chart);
    }

    /**
     * Create response time chart
     */
    async createResponseTimeChart() {
        const canvas = document.getElementById('response-time-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(),
                datasets: [
                    {
                        label: 'Average Response Time (ms)',
                        data: this.data.performanceMetrics.responseTime.history,
                        borderColor: '#4ea5d9',
                        backgroundColor: 'rgba(78, 165, 217, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#4ea5d9',
                        pointBorderColor: '#ffffff',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                ...this.settings.chartOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { 
                        labels: { color: '#94a3b8' }
                    },
                    title: {
                        display: true,
                        text: 'Response Time Performance',
                        color: '#f8fafc'
                    }
                }
            }
        });

        this.charts.set('responseTime', chart);
    }

    /**
     * Create resource usage chart
     */
    async createResourceUsageChart() {
        const canvas = document.getElementById('resource-usage-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'Disk', 'Network'],
                datasets: [{
                    data: [
                        this.data.systemMetrics.cpu.current,
                        this.data.systemMetrics.memory.current,
                        this.data.systemMetrics.disk.current,
                        this.data.systemMetrics.network.current
                    ],
                    backgroundColor: [
                        '#ff6b35',
                        '#00d4aa',
                        '#4ea5d9',
                        '#f59e0b'
                    ],
                    borderColor: [
                        '#ffffff',
                        '#ffffff',
                        '#ffffff',
                        '#ffffff'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                ...this.settings.chartOptions,
                plugins: {
                    legend: { 
                        labels: { color: '#94a3b8' },
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: 'Resource Usage Distribution',
                        color: '#f8fafc'
                    }
                }
            }
        });

        this.charts.set('resourceUsage', chart);
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('monitoring-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Time range selector
        const timeRangeSelect = document.getElementById('monitoring-time-range');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', (e) => {
                this.updateTimeRange(e.target.value);
            });
        }

        // Export data button
        const exportBtn = document.getElementById('monitoring-export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }

        // Alert filter
        const alertFilter = document.getElementById('alert-severity-filter');
        if (alertFilter) {
            alertFilter.addEventListener('change', (e) => {
                this.filterAlerts(e.target.value);
            });
        }
    }

    /**
     * Load initial monitoring data
     */
    async loadInitialData() {
        try {
            // Generate initial metric history
            this.generateInitialHistory();
            
            // Load current system status
            await this.updateSystemMetrics();
            
            // Load performance data
            await this.updatePerformanceMetrics();
            
            // Load security metrics
            await this.updateSecurityMetrics();
            
            // Load alerts
            await this.loadAlerts();
            
            // Update UI
            this.updateMonitoringUI();
            
            console.log('✅ Initial monitoring data loaded');
            
        } catch (error) {
            console.error('❌ Failed to load initial monitoring data:', error);
        }
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        // Clear existing interval
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        // Start new update cycle
        this.updateInterval = setInterval(async () => {
            await this.updateRealTimeData();
        }, this.settings.refreshInterval);

        console.log('✅ Real-time monitoring updates started');
    }

    /**
     * Update real-time monitoring data
     */
    async updateRealTimeData() {
        try {
            // Update system metrics
            await this.updateSystemMetrics();
            
            // Update performance metrics  
            await this.updatePerformanceMetrics();
            
            // Update security metrics
            await this.updateSecurityMetrics();
            
            // Update charts
            this.updateCharts();
            
            // Update metric cards
            this.updateMetricCards();
            
            // Check for new alerts
            await this.checkForAlerts();
            
        } catch (error) {
            console.error('❌ Failed to update real-time data:', error);
        }
    }

    /**
     * Update system metrics
     */
    async updateSystemMetrics() {
        // Simulate realistic system metrics
        const metrics = this.data.systemMetrics;
        
        // CPU usage (20-80%)
        metrics.cpu.current = Math.max(20, Math.min(80, 
            metrics.cpu.current + (Math.random() - 0.5) * 10));
        
        // Memory usage (30-85%)
        metrics.memory.current = Math.max(30, Math.min(85, 
            metrics.memory.current + (Math.random() - 0.5) * 8));
        
        // Disk usage (gradually increasing)
        metrics.disk.current = Math.max(45, Math.min(90, 
            metrics.disk.current + Math.random() * 0.1));
        
        // Network usage (10-100 MB/s)
        metrics.network.current = Math.max(10, Math.min(100, 
            metrics.network.current + (Math.random() - 0.5) * 20));
        
        // Update history
        this.addToHistory(metrics.cpu, metrics.cpu.current);
        this.addToHistory(metrics.memory, metrics.memory.current);
        this.addToHistory(metrics.disk, metrics.disk.current);
        this.addToHistory(metrics.network, metrics.network.current);
        
        // Update other metrics
        metrics.uptime += this.settings.refreshInterval / 1000;
        metrics.activeConnections = Math.floor(Math.random() * 500) + 100;
    }

    /**
     * Update performance metrics
     */
    async updatePerformanceMetrics() {
        const metrics = this.data.performanceMetrics;
        
        // Response time (50-500ms)
        metrics.responseTime.current = Math.max(50, Math.min(500,
            metrics.responseTime.current + (Math.random() - 0.5) * 50));
        
        // Throughput (1000-5000 req/s)
        metrics.throughput.current = Math.max(1000, Math.min(5000,
            metrics.throughput.current + (Math.random() - 0.5) * 200));
        
        // Error rate (0-5%)
        metrics.errorRate.current = Math.max(0, Math.min(5,
            metrics.errorRate.current + (Math.random() - 0.5) * 0.5));
        
        // Availability (99-100%)
        metrics.availability.current = Math.max(99, Math.min(100,
            metrics.availability.current + (Math.random() - 0.5) * 0.1));
        
        // Update history
        this.addToHistory(metrics.responseTime, metrics.responseTime.current);
        this.addToHistory(metrics.throughput, metrics.throughput.current);
        this.addToHistory(metrics.errorRate, metrics.errorRate.current);
        this.addToHistory(metrics.availability, metrics.availability.current);
    }

    /**
     * Update security metrics
     */
    async updateSecurityMetrics() {
        const metrics = this.data.securityMetrics;
        
        // Threat level determination
        const cpuUsage = this.data.systemMetrics.cpu.current;
        const errorRate = this.data.performanceMetrics.errorRate.current;
        
        if (cpuUsage > 70 || errorRate > 3) {
            metrics.threatLevel = 'high';
        } else if (cpuUsage > 50 || errorRate > 1.5) {
            metrics.threatLevel = 'medium';
        } else {
            metrics.threatLevel = 'low';
        }
        
        // Update other security metrics
        metrics.blockedRequests += Math.floor(Math.random() * 5);
        metrics.activeSessions = Math.floor(Math.random() * 50) + 20;
        metrics.failedLogins += Math.random() < 0.1 ? 1 : 0;
    }

    /**
     * Load and check for alerts
     */
    async loadAlerts() {
        // Generate sample alerts based on current metrics
        this.data.alerts = [];
        
        const metrics = this.data.systemMetrics;
        const performance = this.data.performanceMetrics;
        
        // High CPU alert
        if (metrics.cpu.current > 75) {
            this.data.alerts.push({
                id: `cpu-${Date.now()}`,
                type: 'warning',
                severity: 'high',
                title: 'High CPU Usage',
                message: `CPU usage is at ${metrics.cpu.current.toFixed(1)}%`,
                timestamp: new Date(),
                acknowledged: false
            });
        }
        
        // High memory alert
        if (metrics.memory.current > 80) {
            this.data.alerts.push({
                id: `memory-${Date.now()}`,
                type: 'error',
                severity: 'critical',
                title: 'Critical Memory Usage',
                message: `Memory usage is at ${metrics.memory.current.toFixed(1)}%`,
                timestamp: new Date(),
                acknowledged: false
            });
        }
        
        // High response time alert
        if (performance.responseTime.current > 400) {
            this.data.alerts.push({
                id: `response-${Date.now()}`,
                type: 'warning',
                severity: 'medium',
                title: 'Slow Response Time',
                message: `Response time is ${performance.responseTime.current.toFixed(0)}ms`,
                timestamp: new Date(),
                acknowledged: false
            });
        }
    }

    /**
     * Check for new alerts
     */
    async checkForAlerts() {
        await this.loadAlerts();
        this.updateAlertPanel();
    }

    /**
     * Update monitoring UI
     */
    updateMonitoringUI() {
        this.updateMetricCards();
        this.updateCharts();
        this.updateAlertPanel();
        this.updateStatusIndicators();
    }

    /**
     * Update metric cards
     */
    updateMetricCards() {
        const metrics = this.data.systemMetrics;
        const performance = this.data.performanceMetrics;
        const security = this.data.securityMetrics;
        
        // System metrics
        this.updateMetricCard('cpu-usage', metrics.cpu.current, '%', this.getStatusClass(metrics.cpu.current, 70, 85));
        this.updateMetricCard('memory-usage', metrics.memory.current, '%', this.getStatusClass(metrics.memory.current, 70, 85));
        this.updateMetricCard('disk-usage', metrics.disk.current, '%', this.getStatusClass(metrics.disk.current, 80, 90));
        this.updateMetricCard('network-usage', metrics.network.current, 'MB/s');
        
        // Performance metrics
        this.updateMetricCard('response-time', performance.responseTime.current, 'ms', this.getStatusClass(performance.responseTime.current, 300, 450, true));
        this.updateMetricCard('throughput', performance.throughput.current, 'req/s');
        this.updateMetricCard('error-rate', performance.errorRate.current, '%', this.getStatusClass(performance.errorRate.current, 2, 4, true));
        this.updateMetricCard('availability', performance.availability.current, '%', this.getStatusClass(performance.availability.current, 99.5, 99, false));
        
        // Security metrics
        this.updateMetricCard('threat-level', security.threatLevel, '', this.getThreatLevelClass(security.threatLevel));
        this.updateMetricCard('blocked-requests', security.blockedRequests, '');
        this.updateMetricCard('active-sessions', security.activeSessions, '');
        this.updateMetricCard('failed-logins', security.failedLogins, '');
    }

    /**
     * Update individual metric card
     */
    updateMetricCard(id, value, unit, statusClass = '') {
        const element = document.getElementById(id);
        if (element) {
            if (typeof value === 'number') {
                element.textContent = value.toFixed(1) + unit;
            } else {
                element.textContent = value + unit;
            }
            
            // Update status class
            if (statusClass) {
                element.className = `metric-value ${statusClass}`;
            }
        }
    }

    /**
     * Update charts with new data
     */
    updateCharts() {
        // Update system performance chart
        const systemChart = this.charts.get('systemPerformance');
        if (systemChart) {
            systemChart.data.labels = this.generateTimeLabels();
            systemChart.data.datasets[0].data = this.data.systemMetrics.cpu.history;
            systemChart.data.datasets[1].data = this.data.systemMetrics.memory.history;
            systemChart.data.datasets[2].data = this.data.systemMetrics.disk.history;
            systemChart.update('none');
        }
        
        // Update response time chart
        const responseChart = this.charts.get('responseTime');
        if (responseChart) {
            responseChart.data.labels = this.generateTimeLabels();
            responseChart.data.datasets[0].data = this.data.performanceMetrics.responseTime.history;
            responseChart.update('none');
        }
        
        // Update resource usage chart
        const resourceChart = this.charts.get('resourceUsage');
        if (resourceChart) {
            resourceChart.data.datasets[0].data = [
                this.data.systemMetrics.cpu.current,
                this.data.systemMetrics.memory.current,
                this.data.systemMetrics.disk.current,
                this.data.systemMetrics.network.current
            ];
            resourceChart.update('none');
        }
    }

    /**
     * Update alert panel
     */
    updateAlertPanel() {
        const alertContainer = document.getElementById('monitoring-alerts');
        if (!alertContainer) return;
        
        const alerts = this.data.alerts.slice(0, 10); // Show last 10 alerts
        
        alertContainer.innerHTML = alerts.map(alert => `
            <div class="alert-item alert-${alert.severity}" data-alert-id="${alert.id}">
                <div class="alert-header">
                    <span class="alert-icon">
                        <i class="fas ${this.getAlertIcon(alert.type)}"></i>
                    </span>
                    <span class="alert-title">${alert.title}</span>
                    <span class="alert-time">${this.formatTime(alert.timestamp)}</span>
                </div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-actions">
                    <button onclick="acknowledgeAlert('${alert.id}')" class="btn btn-sm btn-outline">
                        Acknowledge
                    </button>
                </div>
            </div>
        `).join('');
        
        // Update alert count badge
        const alertBadge = document.getElementById('alert-count-badge');
        if (alertBadge) {
            const unacknowledged = alerts.filter(a => !a.acknowledged).length;
            alertBadge.textContent = unacknowledged;
            alertBadge.style.display = unacknowledged > 0 ? 'block' : 'none';
        }
    }

    /**
     * Update status indicators
     */
    updateStatusIndicators() {
        // Overall system status
        const systemStatus = document.getElementById('overall-system-status');
        if (systemStatus) {
            const metrics = this.data.systemMetrics;
            const performance = this.data.performanceMetrics;
            
            let status = 'operational';
            if (metrics.cpu.current > 85 || metrics.memory.current > 90 || performance.errorRate.current > 4) {
                status = 'critical';
            } else if (metrics.cpu.current > 70 || metrics.memory.current > 80 || performance.errorRate.current > 2) {
                status = 'warning';
            }
            
            systemStatus.className = `status-indicator status-${status}`;
            systemStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }
    }

    /**
     * Utility methods
     */
    generateTimeLabels(count = 20) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const time = new Date(now.getTime() - i * this.settings.refreshInterval);
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        }
        return labels;
    }

    generateRandomData(min, max, count = 20) {
        return Array.from({ length: count }, () => Math.floor(Math.random() * (max - min + 1)) + min);
    }

    generateInitialHistory() {
        const historyCount = this.settings.historyLimit;
        
        // Initialize CPU history
        this.data.systemMetrics.cpu.history = this.generateRandomData(20, 60, historyCount);
        this.data.systemMetrics.cpu.current = this.data.systemMetrics.cpu.history[historyCount - 1];
        
        // Initialize Memory history
        this.data.systemMetrics.memory.history = this.generateRandomData(30, 70, historyCount);
        this.data.systemMetrics.memory.current = this.data.systemMetrics.memory.history[historyCount - 1];
        
        // Initialize Disk history
        this.data.systemMetrics.disk.history = this.generateRandomData(45, 75, historyCount);
        this.data.systemMetrics.disk.current = this.data.systemMetrics.disk.history[historyCount - 1];
        
        // Initialize Network history
        this.data.systemMetrics.network.history = this.generateRandomData(20, 80, historyCount);
        this.data.systemMetrics.network.current = this.data.systemMetrics.network.history[historyCount - 1];
        
        // Initialize Response Time history
        this.data.performanceMetrics.responseTime.history = this.generateRandomData(100, 300, historyCount);
        this.data.performanceMetrics.responseTime.current = this.data.performanceMetrics.responseTime.history[historyCount - 1];
    }

    addToHistory(metric, value) {
        metric.history.push(value);
        if (metric.history.length > this.settings.historyLimit) {
            metric.history.shift();
        }
    }

    getStatusClass(value, warningThreshold, criticalThreshold, reverse = false) {
        if (reverse) {
            if (value > criticalThreshold) return 'status-critical';
            if (value > warningThreshold) return 'status-warning';
            return 'status-good';
        } else {
            if (value > criticalThreshold) return 'status-critical';
            if (value > warningThreshold) return 'status-warning';
            return 'status-good';
        }
    }

    getThreatLevelClass(level) {
        switch (level) {
            case 'high': return 'status-critical';
            case 'medium': return 'status-warning';
            default: return 'status-good';
        }
    }

    getAlertIcon(type) {
        switch (type) {
            case 'error': return 'fa-exclamation-circle';
            case 'warning': return 'fa-exclamation-triangle';
            case 'info': return 'fa-info-circle';
            default: return 'fa-bell';
        }
    }

    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Public methods for external interaction
     */
    refreshData() {
        console.log('🔄 Refreshing monitoring data...');
        this.updateRealTimeData();
    }

    updateTimeRange(range) {
        console.log(`📅 Updating time range to: ${range}`);
        // Implementation for different time ranges
    }

    exportData() {
        console.log('📊 Exporting monitoring data...');
        const data = {
            timestamp: new Date().toISOString(),
            systemMetrics: this.data.systemMetrics,
            performanceMetrics: this.data.performanceMetrics,
            securityMetrics: this.data.securityMetrics,
            alerts: this.data.alerts
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `monitoring-data-${new Date().toISOString().slice(0, 19)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    filterAlerts(severity) {
        console.log(`🔍 Filtering alerts by severity: ${severity}`);
        // Implementation for alert filtering
    }

    /**
     * Handle section changes
     */
    onSectionChange(newSection) {
        if (newSection === 'monitoring') {
            // Restart updates when monitoring section is active
            this.startRealTimeUpdates();
            this.updateMonitoringUI();
        } else {
            // Slow down updates when not on monitoring section
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
                this.updateInterval = setInterval(() => {
                    this.updateRealTimeData();
                }, 30000); // Update every 30 seconds
            }
        }
    }

    /**
     * Cleanup resources
     */
    destroy() {
        // Clear update interval
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Destroy charts
        this.charts.forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        this.charts.clear();
        
        // Cleanup components
        this.components.forEach(component => {
            if (component && component.destroy) {
                component.destroy();
            }
        });
        this.components.clear();
        
        console.log('✅ Monitoring Manager cleaned up');
    }

    /**
     * Get current monitoring data
     */
    getData() {
        return { ...this.data };
    }

    /**
     * Get monitoring settings
     */
    getSettings() {
        return { ...this.settings };
    }
}

// Global function for alert acknowledgment
function acknowledgeAlert(alertId) {
    console.log(`✅ Acknowledging alert: ${alertId}`);
    if (window.adminPortal) {
        const monitoringManager = window.adminPortal.getManager('monitoring');
        if (monitoringManager) {
            const alert = monitoringManager.data.alerts.find(a => a.id === alertId);
            if (alert) {
                alert.acknowledged = true;
                monitoringManager.updateAlertPanel();
            }
        }
    }
}

// Make function globally available
window.acknowledgeAlert = acknowledgeAlert;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringManager;
}