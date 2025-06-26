/**
 * Dashboard Manager - Handles real-time statistics and monitoring
 * Part of Enhanced CSP Admin Portal
 */

export class DashboardManager {
    constructor(adminPage) {
        this.adminPage = adminPage;
        this.updateInterval = null;
        this.metrics = new Map();
        this.lastUpdate = null;
    }

    /**
     * Initialize dashboard manager
     */
    async init() {
        try {
            console.log('📊 Initializing Dashboard Manager...');
            
            // Load initial metrics
            await this.loadInitialMetrics();
            
            // Set up metric cards
            this.setupMetricCards();
            
            // Start periodic updates
            this.startPeriodicUpdates();
            
            console.log('✅ Dashboard Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Dashboard Manager:', error);
            throw error;
        }
    }

    /**
     * Load initial metrics data
     */
    async loadInitialMetrics() {
        try {
            // Simulate API call - replace with actual endpoint
            this.metrics.set('activeAgents', {
                value: 8,
                status: 'success',
                trend: '+12%',
                label: 'Active AI Agents'
            });

            this.metrics.set('systemHealth', {
                value: 98.7,
                status: 'success',
                trend: '+0.3%',
                label: 'System Health',
                unit: '%'
            });

            this.metrics.set('requestsPerMinute', {
                value: 1247,
                status: 'info',
                trend: '+5%',
                label: 'Requests/Min'
            });

            this.metrics.set('securityThreats', {
                value: 3,
                status: 'warning',
                trend: '-2',
                label: 'Security Alerts'
            });

            this.metrics.set('responseTime', {
                value: 127,
                status: 'success',
                trend: '-8ms',
                label: 'Avg Response Time',
                unit: 'ms'
            });

            this.metrics.set('uptime', {
                value: 99.95,
                status: 'success',
                trend: 'stable',
                label: 'System Uptime',
                unit: '%'
            });

            this.lastUpdate = new Date();
            
        } catch (error) {
            console.error('❌ Failed to load initial metrics:', error);
            throw error;
        }
    }

    /**
     * Set up metric cards in the dashboard
     */
    setupMetricCards() {
        const dashboardGrid = document.querySelector('.dashboard-grid');
        if (!dashboardGrid) return;

        // Clear existing cards
        dashboardGrid.innerHTML = '';

        // Create metric cards
        this.metrics.forEach((metric, key) => {
            const card = this.createMetricCard(key, metric);
            dashboardGrid.appendChild(card);
        });
    }

    /**
     * Create a metric card element
     */
    createMetricCard(key, metric) {
        const card = document.createElement('div');
        card.className = 'admin-card metric-card';
        card.setAttribute('data-metric', key);

        const statusClass = this.getStatusClass(metric.status);
        const icon = this.getMetricIcon(key);
        const unit = metric.unit || '';

        card.innerHTML = `
            <h3>
                <i class="${icon}"></i>
                ${metric.label}
                <span class="status-badge ${statusClass}">
                    ${metric.status}
                </span>
            </h3>
            <div class="metric ${statusClass}">
                ${metric.value}${unit}
            </div>
            <div class="metric-trend">
                <span class="trend-indicator ${this.getTrendClass(metric.trend)}">
                    ${metric.trend}
                </span>
                <span class="last-update" title="Last updated: ${this.formatTime(this.lastUpdate)}">
                    ${this.getTimeAgo(this.lastUpdate)}
                </span>
            </div>
            <div class="metric-actions">
                <button class="btn btn-outline" onclick="window.adminPage.getManager('dashboard').refreshMetric('${key}')">
                    <i class="fas fa-refresh"></i>
                    Refresh
                </button>
                <button class="btn btn-outline" onclick="window.adminPage.getManager('dashboard').viewDetails('${key}')">
                    <i class="fas fa-chart-line"></i>
                    Details
                </button>
            </div>
        `;

        return card;
    }

    /**
     * Get icon for metric type
     */
    getMetricIcon(metricKey) {
        const icons = {
            activeAgents: 'fas fa-robot',
            systemHealth: 'fas fa-heartbeat',
            requestsPerMinute: 'fas fa-tachometer-alt',
            securityThreats: 'fas fa-shield-alt',
            responseTime: 'fas fa-clock',
            uptime: 'fas fa-server'
        };
        return icons[metricKey] || 'fas fa-chart-bar';
    }

    /**
     * Get CSS class for status
     */
    getStatusClass(status) {
        const statusMap = {
            success: 'success',
            warning: 'warning',
            danger: 'danger',
            info: 'info'
        };
        return statusMap[status] || 'info';
    }

    /**
     * Get CSS class for trend
     */
    getTrendClass(trend) {
        if (trend.includes('+') || trend.includes('up')) return 'trend-up';
        if (trend.includes('-') || trend.includes('down')) return 'trend-down';
        return 'trend-stable';
    }

    /**
     * Start periodic updates
     */
    startPeriodicUpdates() {
        // Update metrics every 10 seconds
        this.updateInterval = setInterval(() => {
            this.updateRealTimeData();
        }, 10000);
    }

    /**
     * Update real-time data
     */
    async updateRealTimeData() {
        try {
            // Simulate API call with random variations
            this.metrics.forEach((metric, key) => {
                const variation = (Math.random() - 0.5) * 0.1; // ±5% variation
                let newValue = metric.value * (1 + variation);

                // Apply constraints based on metric type
                switch (key) {
                    case 'activeAgents':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'systemHealth':
                    case 'uptime':
                        newValue = Math.min(100, Math.max(0, parseFloat(newValue.toFixed(2))));
                        break;
                    case 'requestsPerMinute':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'securityThreats':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'responseTime':
                        newValue = Math.max(10, Math.round(newValue));
                        break;
                }

                // Update the metric
                this.updateMetric(key, newValue);
            });

            this.lastUpdate = new Date();
            this.updateLastUpdateTimes();

        } catch (error) {
            console.error('❌ Failed to update real-time data:', error);
        }
    }

    /**
     * Update a specific metric
     */
    updateMetric(key, newValue) {
        const metric = this.metrics.get(key);
        if (!metric) return;

        const oldValue = metric.value;
        metric.value = newValue;

        // Calculate trend
        const change = newValue - oldValue;
        const changePercent = ((change / oldValue) * 100).toFixed(1);
        
        if (Math.abs(change) > 0.01) {
            metric.trend = change > 0 ? `+${changePercent}%` : `${changePercent}%`;
        }

        // Update status based on metric type and value
        metric.status = this.calculateStatus(key, newValue);

        // Update UI
        this.updateMetricCard(key, metric);
    }

    /**
     * Calculate status based on metric value
     */
    calculateStatus(key, value) {
        switch (key) {
            case 'systemHealth':
            case 'uptime':
                if (value >= 95) return 'success';
                if (value >= 85) return 'warning';
                return 'danger';
            
            case 'securityThreats':
                if (value === 0) return 'success';
                if (value <= 5) return 'warning';
                return 'danger';
            
            case 'responseTime':
                if (value <= 150) return 'success';
                if (value <= 300) return 'warning';
                return 'danger';
            
            default:
                return 'info';
        }
    }

    /**
     * Update metric card in UI
     */
    updateMetricCard(key, metric) {
        const card = document.querySelector(`[data-metric="${key}"]`);
        if (!card) return;

        const metricValue = card.querySelector('.metric');
        const trendIndicator = card.querySelector('.trend-indicator');
        const statusBadge = card.querySelector('.status-badge');

        if (metricValue) {
            const unit = metric.unit || '';
            metricValue.textContent = `${metric.value}${unit}`;
            metricValue.className = `metric ${this.getStatusClass(metric.status)}`;
        }

        if (trendIndicator) {
            trendIndicator.textContent = metric.trend;
            trendIndicator.className = `trend-indicator ${this.getTrendClass(metric.trend)}`;
        }

        if (statusBadge) {
            statusBadge.textContent = metric.status;
            statusBadge.className = `status-badge ${this.getStatusClass(metric.status)}`;
        }

        // Add update animation
        card.classList.add('updating');
        setTimeout(() => {
            card.classList.remove('updating');
        }, 300);
    }

    /**
     * Update last update times
     */
    updateLastUpdateTimes() {
        document.querySelectorAll('.last-update').forEach(element => {
            element.textContent = this.getTimeAgo(this.lastUpdate);
            element.title = `Last updated: ${this.formatTime(this.lastUpdate)}`;
        });
    }

    /**
     * Format time for display
     */
    formatTime(date) {
        return date.toLocaleTimeString();
    }

    /**
     * Get time ago string
     */
    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const seconds = Math.floor(diff / 1000);
        
        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    /**
     * Refresh a specific metric
     */
    async refreshMetric(key) {
        try {
            console.log(`🔄 Refreshing metric: ${key}`);
            
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Generate new value
            const metric = this.metrics.get(key);
            if (metric) {
                const variation = (Math.random() - 0.5) * 0.2; // ±10% variation
                let newValue = metric.value * (1 + variation);
                
                // Apply constraints
                switch (key) {
                    case 'activeAgents':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'systemHealth':
                    case 'uptime':
                        newValue = Math.min(100, Math.max(0, parseFloat(newValue.toFixed(2))));
                        break;
                    case 'requestsPerMinute':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'securityThreats':
                        newValue = Math.max(0, Math.round(newValue));
                        break;
                    case 'responseTime':
                        newValue = Math.max(10, Math.round(newValue));
                        break;
                }
                
                this.updateMetric(key, newValue);
            }
            
            this.adminPage.showSuccess('Metric Refreshed', `${key} has been updated`);
            
        } catch (error) {
            console.error(`❌ Failed to refresh metric ${key}:`, error);
            this.adminPage.showError('Refresh Failed', `Could not refresh ${key}`);
        }
    }

    /**
     * View metric details
     */
    viewDetails(key) {
        const metric = this.metrics.get(key);
        if (!metric) return;

        console.log(`📊 Viewing details for: ${key}`);
        
        // Create detailed view modal
        const modalManager = this.adminPage.getManager('modal');
        if (modalManager) {
            modalManager.showMetricDetails(key, metric);
        }
    }

    /**
     * Handle section change
     */
    onSectionChange(sectionId) {
        if (sectionId === 'dashboard') {
            // Refresh data when dashboard becomes active
            this.updateRealTimeData();
        }
    }

    /**
     * Handle window resize
     */
    onResize() {
        // Adjust dashboard grid if needed
        const dashboardGrid = document.querySelector('.dashboard-grid');
        if (dashboardGrid) {
            // Recalculate grid layout if necessary
            this.adjustGridLayout();
        }
    }

    /**
     * Adjust grid layout based on screen size
     */
    adjustGridLayout() {
        const dashboardGrid = document.querySelector('.dashboard-grid');
        if (!dashboardGrid) return;

        const screenWidth = window.innerWidth;
        let columns;

        if (screenWidth < 768) {
            columns = 1;
        } else if (screenWidth < 1024) {
            columns = 2;
        } else if (screenWidth < 1440) {
            columns = 3;
        } else {
            columns = 4;
        }

        dashboardGrid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    }

    /**
     * Get all metrics
     */
    getMetrics() {
        return new Map(this.metrics);
    }

    /**
     * Get specific metric
     */
    getMetric(key) {
        return this.metrics.get(key);
    }

    /**
     * Add custom metric
     */
    addMetric(key, metric) {
        this.metrics.set(key, metric);
        
        // Add to UI if dashboard is active
        if (this.adminPage.getState().currentSection === 'dashboard') {
            const dashboardGrid = document.querySelector('.dashboard-grid');
            if (dashboardGrid) {
                const card = this.createMetricCard(key, metric);
                dashboardGrid.appendChild(card);
            }
        }
    }

    /**
     * Remove metric
     */
    removeMetric(key) {
        this.metrics.delete(key);
        
        // Remove from UI
        const card = document.querySelector(`[data-metric="${key}"]`);
        if (card) {
            card.remove();
        }
    }

    /**
     * Export metrics data
     */
    exportMetrics() {
        const data = {};
        this.metrics.forEach((metric, key) => {
            data[key] = {
                ...metric,
                lastUpdate: this.lastUpdate
            };
        });
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `metrics-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.adminPage.showSuccess('Export Complete', 'Metrics data exported successfully');
    }

    /**
     * Reset all metrics
     */
    async resetMetrics() {
        try {
            console.log('🔄 Resetting all metrics...');
            
            // Reload initial metrics
            await this.loadInitialMetrics();
            
            // Update UI
            this.setupMetricCards();
            
            this.adminPage.showSuccess('Metrics Reset', 'All metrics have been reset to initial values');
            
        } catch (error) {
            console.error('❌ Failed to reset metrics:', error);
            this.adminPage.showError('Reset Failed', 'Could not reset metrics');
        }
    }

    /**
     * Get system overview
     */
    getSystemOverview() {
        const overview = {
            totalMetrics: this.metrics.size,
            healthyMetrics: 0,
            warningMetrics: 0,
            criticalMetrics: 0,
            lastUpdate: this.lastUpdate
        };

        this.metrics.forEach(metric => {
            switch (metric.status) {
                case 'success':
                    overview.healthyMetrics++;
                    break;
                case 'warning':
                    overview.warningMetrics++;
                    break;
                case 'danger':
                    overview.criticalMetrics++;
                    break;
            }
        });

        return overview;
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.metrics.clear();
        console.log('🧹 Dashboard Manager cleaned up');
    }
}

// Add CSS for dashboard animations
const style = document.createElement('style');
style.textContent = `
    .metric-card.updating {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
        transition: all 0.3s ease;
    }

    .trend-up {
        color: var(--success);
    }

    .trend-up::before {
        content: '↗ ';
    }

    .trend-down {
        color: var(--danger);
    }

    .trend-down::before {
        content: '↘ ';
    }

    .trend-stable {
        color: var(--text-muted);
    }

    .trend-stable::before {
        content: '→ ';
    }

    .metric-trend {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }

    .last-update {
        color: var(--text-muted);
        font-size: 0.75rem;
    }

    .metric-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover .metric-actions {
        opacity: 1;
    }

    .metric-actions .btn {
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
    }
`;

document.head.appendChild(style);