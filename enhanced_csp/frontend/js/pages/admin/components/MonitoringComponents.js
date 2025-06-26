// js/pages/admin/components/MonitoringComponents.js
// Enhanced CSP Admin Portal - Monitoring Dashboard Components

/**
 * MonitoringMetricCards - Handles metric card display and updates
 */
class MonitoringMetricCards {
    constructor(monitoringManager) {
        this.monitoringManager = monitoringManager;
        this.cards = new Map();
        this.init();
    }

    init() {
        this.setupMetricCards();
        console.log('✅ Monitoring Metric Cards initialized');
    }

    setupMetricCards() {
        const cardConfigs = [
            {
                id: 'cpu-usage-card',
                title: 'CPU Usage',
                icon: 'fas fa-microchip',
                valueId: 'cpu-usage',
                unit: '%',
                thresholds: { warning: 70, critical: 85 }
            },
            {
                id: 'memory-usage-card',
                title: 'Memory Usage',
                icon: 'fas fa-memory',
                valueId: 'memory-usage',
                unit: '%',
                thresholds: { warning: 70, critical: 85 }
            },
            {
                id: 'disk-usage-card',
                title: 'Disk Usage',
                icon: 'fas fa-hdd',
                valueId: 'disk-usage',
                unit: '%',
                thresholds: { warning: 80, critical: 90 }
            },
            {
                id: 'network-usage-card',
                title: 'Network Usage',
                icon: 'fas fa-network-wired',
                valueId: 'network-usage',
                unit: 'MB/s',
                thresholds: { warning: 80, critical: 95 }
            },
            {
                id: 'response-time-card',
                title: 'Response Time',
                icon: 'fas fa-clock',
                valueId: 'response-time',
                unit: 'ms',
                thresholds: { warning: 300, critical: 450 },
                reverse: true
            },
            {
                id: 'throughput-card',
                title: 'Throughput',
                icon: 'fas fa-tachometer-alt',
                valueId: 'throughput',
                unit: 'req/s',
                thresholds: { warning: 2000, critical: 1000 },
                reverse: true
            },
            {
                id: 'error-rate-card',
                title: 'Error Rate',
                icon: 'fas fa-exclamation-triangle',
                valueId: 'error-rate',
                unit: '%',
                thresholds: { warning: 2, critical: 4 },
                reverse: true
            },
            {
                id: 'availability-card',
                title: 'Availability',
                icon: 'fas fa-shield-alt',
                valueId: 'availability',
                unit: '%',
                thresholds: { warning: 99.5, critical: 99 },
                reverse: false
            }
        ];

        cardConfigs.forEach(config => {
            this.cards.set(config.id, config);
        });
    }

    updateCard(cardId, value, trend = null) {
        const config = this.cards.get(cardId);
        if (!config) return;

        const card = document.getElementById(cardId);
        if (!card) return;

        // Update value
        const valueElement = card.querySelector(`#${config.valueId}`);
        if (valueElement) {
            if (typeof value === 'number') {
                valueElement.textContent = value.toFixed(1) + config.unit;
            } else {
                valueElement.textContent = value + config.unit;
            }
        }

        // Update status class
        if (config.thresholds && typeof value === 'number') {
            const statusClass = this.getStatusClass(value, config.thresholds, config.reverse);
            valueElement.className = `metric-value ${statusClass}`;
        }

        // Update trend indicator
        if (trend !== null) {
            this.updateTrendIndicator(card, trend);
        }
    }

    getStatusClass(value, thresholds, reverse = false) {
        if (reverse) {
            if (value > thresholds.critical) return 'status-critical';
            if (value > thresholds.warning) return 'status-warning';
            return 'status-good';
        } else {
            if (value > thresholds.critical) return 'status-critical';
            if (value > thresholds.warning) return 'status-warning';
            return 'status-good';
        }
    }

    updateTrendIndicator(card, trend) {
        const trendElement = card.querySelector('.trend-indicator');
        if (!trendElement) return;

        let trendClass = 'trend-neutral';
        let trendIcon = 'fa-minus';
        let trendText = 'No change';

        if (trend > 0.1) {
            trendClass = 'trend-up';
            trendIcon = 'fa-arrow-up';
            trendText = `+${trend.toFixed(1)}%`;
        } else if (trend < -0.1) {
            trendClass = 'trend-down';
            trendIcon = 'fa-arrow-down';
            trendText = `${trend.toFixed(1)}%`;
        }

        trendElement.className = `trend-indicator ${trendClass}`;
        trendElement.innerHTML = `
            <i class="fas ${trendIcon}"></i>
            <span>${trendText}</span>
        `;
    }

    destroy() {
        this.cards.clear();
        console.log('✅ Monitoring Metric Cards cleaned up');
    }
}

/**
 * MonitoringAlertPanel - Handles alert display and management
 */
class MonitoringAlertPanel {
    constructor(monitoringManager) {
        this.monitoringManager = monitoringManager;
        this.alerts = [];
        this.filters = {
            severity: 'all',
            status: 'all'
        };
        this.init();
    }

    init() {
        this.setupEventListeners();
        console.log('✅ Monitoring Alert Panel initialized');
    }

    setupEventListeners() {
        // Severity filter
        const severityFilter = document.getElementById('alert-severity-filter');
        if (severityFilter) {
            severityFilter.addEventListener('change', (e) => {
                this.filters.severity = e.target.value;
                this.renderAlerts();
            });
        }

        // Status filter
        const statusFilter = document.getElementById('alert-status-filter');
        if (statusFilter) {
            statusFilter.addEventListener('change', (e) => {
                this.filters.status = e.target.value;
                this.renderAlerts();
            });
        }

        // Clear all alerts button
        const clearAllBtn = document.getElementById('clear-all-alerts-btn');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => this.clearAllAlerts());
        }
    }

    updateAlerts(alerts) {
        this.alerts = alerts || [];
        this.renderAlerts();
        this.updateAlertSummary();
    }

    renderAlerts() {
        const container = document.getElementById('monitoring-alerts');
        if (!container) return;

        const filteredAlerts = this.filterAlerts();
        
        if (filteredAlerts.length === 0) {
            container.innerHTML = `
                <div class="no-alerts">
                    <i class="fas fa-check-circle"></i>
                    <p>No alerts match your filters</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filteredAlerts.map(alert => this.renderAlert(alert)).join('');
    }

    renderAlert(alert) {
        const timeAgo = this.getTimeAgo(alert.timestamp);
        const statusClass = alert.acknowledged ? 'acknowledged' : 'active';

        return `
            <div class="alert-item alert-${alert.severity} alert-${statusClass}" data-alert-id="${alert.id}">
                <div class="alert-header">
                    <div class="alert-icon-container">
                        <i class="fas ${this.getAlertIcon(alert.type)} alert-icon"></i>
                        <span class="alert-severity-badge severity-${alert.severity}">${alert.severity}</span>
                    </div>
                    <div class="alert-content">
                        <h4 class="alert-title">${alert.title}</h4>
                        <p class="alert-message">${alert.message}</p>
                    </div>
                    <div class="alert-meta">
                        <span class="alert-time" title="${alert.timestamp.toLocaleString()}">${timeAgo}</span>
                        ${alert.acknowledged ? '<span class="alert-status">Acknowledged</span>' : ''}
                    </div>
                </div>
                <div class="alert-actions">
                    ${!alert.acknowledged ? `
                        <button onclick="acknowledgeAlert('${alert.id}')" class="btn btn-sm btn-outline">
                            <i class="fas fa-check"></i> Acknowledge
                        </button>
                    ` : ''}
                    <button onclick="dismissAlert('${alert.id}')" class="btn btn-sm btn-outline btn-danger">
                        <i class="fas fa-times"></i> Dismiss
                    </button>
                    <button onclick="viewAlertDetails('${alert.id}')" class="btn btn-sm btn-outline">
                        <i class="fas fa-info-circle"></i> Details
                    </button>
                </div>
            </div>
        `;
    }

    filterAlerts() {
        return this.alerts.filter(alert => {
            // Severity filter
            if (this.filters.severity !== 'all' && alert.severity !== this.filters.severity) {
                return false;
            }

            // Status filter
            if (this.filters.status === 'active' && alert.acknowledged) {
                return false;
            }
            if (this.filters.status === 'acknowledged' && !alert.acknowledged) {
                return false;
            }

            return true;
        });
    }

    updateAlertSummary() {
        const summary = this.getAlertSummary();
        
        // Update alert count badges
        const totalBadge = document.getElementById('total-alerts-badge');
        if (totalBadge) {
            totalBadge.textContent = summary.total;
        }

        const criticalBadge = document.getElementById('critical-alerts-badge');
        if (criticalBadge) {
            criticalBadge.textContent = summary.critical;
            criticalBadge.style.display = summary.critical > 0 ? 'inline' : 'none';
        }

        const activeBadge = document.getElementById('active-alerts-badge');
        if (activeBadge) {
            activeBadge.textContent = summary.active;
            activeBadge.style.display = summary.active > 0 ? 'inline' : 'none';
        }

        // Update alert summary text
        const summaryText = document.getElementById('alert-summary-text');
        if (summaryText) {
            summaryText.textContent = `${summary.active} active, ${summary.acknowledged} acknowledged`;
        }
    }

    getAlertSummary() {
        const summary = {
            total: this.alerts.length,
            active: 0,
            acknowledged: 0,
            critical: 0,
            high: 0,
            medium: 0,
            low: 0
        };

        this.alerts.forEach(alert => {
            if (alert.acknowledged) {
                summary.acknowledged++;
            } else {
                summary.active++;
            }

            summary[alert.severity]++;
        });

        return summary;
    }

    getAlertIcon(type) {
        switch (type) {
            case 'error': return 'fa-exclamation-circle';
            case 'warning': return 'fa-exclamation-triangle';
            case 'info': return 'fa-info-circle';
            case 'success': return 'fa-check-circle';
            default: return 'fa-bell';
        }
    }

    getTimeAgo(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return 'Just now';
    }

    acknowledgeAlert(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert) {
            alert.acknowledged = true;
            this.renderAlerts();
            this.updateAlertSummary();
            console.log(`✅ Alert acknowledged: ${alertId}`);
        }
    }

    dismissAlert(alertId) {
        this.alerts = this.alerts.filter(a => a.id !== alertId);
        this.renderAlerts();
        this.updateAlertSummary();
        console.log(`🗑️ Alert dismissed: ${alertId}`);
    }

    clearAllAlerts() {
        if (confirm('Are you sure you want to clear all alerts?')) {
            this.alerts = [];
            this.renderAlerts();
            this.updateAlertSummary();
            console.log('🗑️ All alerts cleared');
        }
    }

    destroy() {
        this.alerts = [];
        console.log('✅ Monitoring Alert Panel cleaned up');
    }
}

/**
 * MonitoringLogViewer - Handles system log display and filtering
 */
class MonitoringLogViewer {
    constructor(monitoringManager) {
        this.monitoringManager = monitoringManager;
        this.logs = [];
        this.filters = {
            level: 'all',
            source: 'all',
            search: ''
        };
        this.maxLogs = 1000; // Keep last 1000 logs
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.generateSampleLogs();
        console.log('✅ Monitoring Log Viewer initialized');
    }

    setupEventListeners() {
        // Log level filter
        const levelFilter = document.getElementById('log-level-filter');
        if (levelFilter) {
            levelFilter.addEventListener('change', (e) => {
                this.filters.level = e.target.value;
                this.renderLogs();
            });
        }

        // Log source filter
        const sourceFilter = document.getElementById('log-source-filter');
        if (sourceFilter) {
            sourceFilter.addEventListener('change', (e) => {
                this.filters.source = e.target.value;
                this.renderLogs();
            });
        }

        // Search input
        const searchInput = document.getElementById('log-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filters.search = e.target.value.toLowerCase();
                this.renderLogs();
            });
        }

        // Clear logs button
        const clearLogsBtn = document.getElementById('clear-logs-btn');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearLogs());
        }

        // Auto-scroll toggle
        const autoScrollToggle = document.getElementById('auto-scroll-toggle');
        if (autoScrollToggle) {
            autoScrollToggle.addEventListener('change', (e) => {
                this.autoScroll = e.target.checked;
            });
        }
    }

    generateSampleLogs() {
        const levels = ['info', 'warning', 'error', 'debug'];
        const sources = ['system', 'api', 'database', 'auth', 'monitoring'];
        const messages = [
            'System startup completed successfully',
            'User authentication successful',
            'Database connection established',
            'API request processed',
            'Memory usage threshold exceeded',
            'Backup operation completed',
            'Security scan finished',
            'Configuration updated',
            'Service restart initiated',
            'Performance optimization applied'
        ];

        // Generate initial logs
        for (let i = 0; i < 50; i++) {
            this.addLog({
                level: levels[Math.floor(Math.random() * levels.length)],
                source: sources[Math.floor(Math.random() * sources.length)],
                message: messages[Math.floor(Math.random() * messages.length)],
                timestamp: new Date(Date.now() - Math.random() * 3600000) // Random time in last hour
            });
        }

        this.renderLogs();
    }

    addLog(logEntry) {
        this.logs.unshift({
            id: Date.now() + Math.random(),
            timestamp: logEntry.timestamp || new Date(),
            level: logEntry.level,
            source: logEntry.source,
            message: logEntry.message
        });

        // Keep only the most recent logs
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(0, this.maxLogs);
        }

        // Re-render if monitoring section is active
        if (this.monitoringManager.adminPortal.getState().currentSection === 'monitoring') {
            this.renderLogs();
        }
    }

    renderLogs() {
        const container = document.getElementById('monitoring-logs');
        if (!container) return;

        const filteredLogs = this.filterLogs();
        
        if (filteredLogs.length === 0) {
            container.innerHTML = `
                <div class="no-logs">
                    <i class="fas fa-file-alt"></i>
                    <p>No logs match your filters</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filteredLogs.map(log => this.renderLogEntry(log)).join('');

        // Auto-scroll to bottom if enabled
        if (this.autoScroll) {
            container.scrollTop = container.scrollHeight;
        }
    }

    renderLogEntry(log) {
        return `
            <div class="log-entry log-${log.level}" data-log-id="${log.id}">
                <div class="log-timestamp">${log.timestamp.toLocaleTimeString()}</div>
                <div class="log-level">
                    <span class="log-level-badge level-${log.level}">${log.level.toUpperCase()}</span>
                </div>
                <div class="log-source">[${log.source}]</div>
                <div class="log-message">${log.message}</div>
            </div>
        `;
    }

    filterLogs() {
        return this.logs.filter(log => {
            // Level filter
            if (this.filters.level !== 'all' && log.level !== this.filters.level) {
                return false;
            }

            // Source filter
            if (this.filters.source !== 'all' && log.source !== this.filters.source) {
                return false;
            }

            // Search filter
            if (this.filters.search && !log.message.toLowerCase().includes(this.filters.search)) {
                return false;
            }

            return true;
        });
    }

    clearLogs() {
        if (confirm('Are you sure you want to clear all logs?')) {
            this.logs = [];
            this.renderLogs();
            console.log('🗑️ All logs cleared');
        }
    }

    exportLogs() {
        const data = {
            timestamp: new Date().toISOString(),
            logs: this.logs
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `system-logs-${new Date().toISOString().slice(0, 19)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    destroy() {
        this.logs = [];
        console.log('✅ Monitoring Log Viewer cleaned up');
    }
}

// Global functions for alert and log management
function dismissAlert(alertId) {
    console.log(`🗑️ Dismissing alert: ${alertId}`);
    if (window.adminPortal) {
        const monitoringManager = window.adminPortal.getManager('monitoring');
        if (monitoringManager && monitoringManager.components.has('alertPanel')) {
            monitoringManager.components.get('alertPanel').dismissAlert(alertId);
        }
    }
}

function viewAlertDetails(alertId) {
    console.log(`👁️ Viewing alert details: ${alertId}`);
    // Implementation for showing detailed alert information
}

function exportLogs() {
    if (window.adminPortal) {
        const monitoringManager = window.adminPortal.getManager('monitoring');
        if (monitoringManager && monitoringManager.components.has('logViewer')) {
            monitoringManager.components.get('logViewer').exportLogs();
        }
    }
}

// Make functions globally available
window.dismissAlert = dismissAlert;
window.viewAlertDetails = viewAlertDetails;
window.exportLogs = exportLogs;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MonitoringMetricCards,
        MonitoringAlertPanel,
        MonitoringLogViewer
    };
}