/**
 * System Manager - Handles system-level operations and controls
 * Part of Enhanced CSP Admin Portal
 */

export class SystemManager {
    constructor(adminPage) {
        this.adminPage = adminPage;
        this.systemStatus = {
            uptime: 0,
            lastRestart: null,
            maintenanceMode: false,
            emergencyMode: false
        };
        this.services = new Map();
        this.alerts = [];
    }

    /**
     * Initialize system manager
     */
    async init() {
        try {
            console.log('⚙️ Initializing System Manager...');
            
            // Load system status
            await this.loadSystemStatus();
            
            // Initialize services
            this.initializeServices();
            
            // Set up monitoring
            this.startSystemMonitoring();
            
            console.log('✅ System Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize System Manager:', error);
            throw error;
        }
    }

    /**
     * Load system status
     */
    async loadSystemStatus() {
        try {
            // Simulate API call to get system status
            this.systemStatus = {
                uptime: 432000000, // 5 days in milliseconds
                lastRestart: new Date(Date.now() - 432000000),
                maintenanceMode: false,
                emergencyMode: false,
                version: '2.1.3',
                buildDate: new Date('2024-06-15'),
                environment: 'production',
                clusterId: 'csp-cluster-001'
            };

        } catch (error) {
            console.error('❌ Failed to load system status:', error);
            throw error;
        }
    }

    /**
     * Initialize system services
     */
    initializeServices() {
        // Core system services
        this.services.set('database', {
            name: 'Database Service',
            status: 'running',
            uptime: 432000000,
            restartCount: 0,
            memoryUsage: 1024,
            critical: true
        });

        this.services.set('webserver', {
            name: 'Web Server',
            status: 'running',
            uptime: 432000000,
            restartCount: 1,
            memoryUsage: 512,
            critical: true
        });

        this.services.set('cache', {
            name: 'Cache Service',
            status: 'running',
            uptime: 432000000,
            restartCount: 0,
            memoryUsage: 256,
            critical: false
        });

        this.services.set('scheduler', {
            name: 'Task Scheduler',
            status: 'running',
            uptime: 432000000,
            restartCount: 2,
            memoryUsage: 128,
            critical: false
        });

        this.services.set('monitoring', {
            name: 'Monitoring Agent',
            status: 'running',
            uptime: 432000000,
            restartCount: 0,
            memoryUsage: 64,
            critical: true
        });
    }

    /**
     * Start system monitoring
     */
    startSystemMonitoring() {
        // Monitor system every 30 seconds
        this.monitoringInterval = setInterval(() => {
            this.updateSystemMetrics();
        }, 30000);
    }

    /**
     * Update system metrics
     */
    updateSystemMetrics() {
        // Update uptime
        this.systemStatus.uptime += 30000;

        // Simulate service status changes
        this.services.forEach((service, key) => {
            if (service.status === 'running') {
                service.uptime += 30000;
                
                // Random chance of service issues
                if (Math.random() < 0.001) { // 0.1% chance
                    this.simulateServiceIssue(key);
                }
                
                // Small memory fluctuations
                const variation = (Math.random() - 0.5) * 0.1;
                service.memoryUsage = Math.max(32, service.memoryUsage * (1 + variation));
            }
        });

        // Update dashboard if system monitoring section is active
        if (this.adminPage.getState().currentSection === 'monitoring') {
            this.updateMonitoringDisplay();
        }
    }

    /**
     * Simulate service issue
     */
    simulateServiceIssue(serviceKey) {
        const service = this.services.get(serviceKey);
        if (!service) return;

        const issues = ['high_memory', 'slow_response', 'connection_timeout'];
        const issue = issues[Math.floor(Math.random() * issues.length)];

        this.createAlert('warning', `Service Issue`, `${service.name} experiencing ${issue.replace('_', ' ')}`);
        
        console.warn(`⚠️ Service issue detected: ${service.name} - ${issue}`);
    }

    /**
     * Create system alert
     */
    createAlert(level, title, message) {
        const alert = {
            id: `alert-${Date.now()}`,
            level,
            title,
            message,
            timestamp: new Date(),
            acknowledged: false
        };

        this.alerts.unshift(alert);

        // Keep only last 50 alerts
        if (this.alerts.length > 50) {
            this.alerts = this.alerts.slice(0, 50);
        }

        // Show notification
        this.adminPage.showInfo(title, message);
    }

    /**
     * Emergency shutdown
     */
    async emergencyShutdown() {
        try {
            console.log('🚨 EMERGENCY SHUTDOWN INITIATED');

            const modalManager = this.adminPage.getManager('modal');
            if (modalManager) {
                modalManager.showConfirmation(
                    '🚨 Emergency Shutdown',
                    'This will immediately stop all agents and services. Are you sure you want to proceed?',
                    async () => {
                        await this.performEmergencyShutdown();
                    }
                );
            }

        } catch (error) {
            console.error('❌ Emergency shutdown failed:', error);
            this.adminPage.showError('Emergency Shutdown Failed', error.message);
        }
    }

    /**
     * Perform emergency shutdown
     */
    async performEmergencyShutdown() {
        try {
            this.systemStatus.emergencyMode = true;

            // Show emergency overlay
            this.showEmergencyOverlay();

            console.log('🛑 Stopping all agents...');
            const agentManager = this.adminPage.getManager('agent');
            if (agentManager) {
                const agents = Array.from(agentManager.agents.values());
                for (const agent of agents) {
                    if (agent.status === 'active') {
                        await agentManager.toggleAgent(agent.id);
                    }
                }
            }

            console.log('🛑 Stopping system services...');
            this.services.forEach(service => {
                if (service.status === 'running' && !service.critical) {
                    service.status = 'stopped';
                    service.uptime = 0;
                }
            });

            // Simulate shutdown delay
            await new Promise(resolve => setTimeout(resolve, 3000));

            this.createAlert('critical', 'Emergency Shutdown', 'Emergency shutdown completed. System is in safe mode.');
            
            this.hideEmergencyOverlay();
            this.systemStatus.emergencyMode = false;

            this.adminPage.showSuccess('Emergency Shutdown Complete', 'System is now in safe mode');

        } catch (error) {
            console.error('❌ Emergency shutdown process failed:', error);
            this.adminPage.showError('Shutdown Process Failed', error.message);
        }
    }

    /**
     * Show emergency overlay
     */
    showEmergencyOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'emergency-overlay';
        overlay.className = 'emergency-overlay';
        overlay.innerHTML = `
            <div class="emergency-content">
                <div class="emergency-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h2>Emergency Shutdown in Progress</h2>
                <p>Please wait while the system safely shuts down...</p>
                <div class="loading-spinner"></div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Add emergency styles
        const emergencyStyles = document.createElement('style');
        emergencyStyles.id = 'emergency-styles';
        emergencyStyles.textContent = `
            .emergency-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(231, 76, 60, 0.95);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                backdrop-filter: blur(10px);
            }

            .emergency-content {
                text-align: center;
                color: white;
                max-width: 400px;
                padding: 2rem;
            }

            .emergency-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
                animation: pulse 1s infinite;
            }

            .emergency-content h2 {
                font-size: 1.8rem;
                margin-bottom: 1rem;
            }

            .emergency-content p {
                font-size: 1.1rem;
                margin-bottom: 2rem;
                opacity: 0.9;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
        `;

        document.head.appendChild(emergencyStyles);
    }

    /**
     * Hide emergency overlay
     */
    hideEmergencyOverlay() {
        const overlay = document.getElementById('emergency-overlay');
        const styles = document.getElementById('emergency-styles');
        
        if (overlay) overlay.remove();
        if (styles) styles.remove();
    }

    /**
     * Restart system
     */
    async restartSystem() {
        try {
            console.log('🔄 System restart initiated');

            const modalManager = this.adminPage.getManager('modal');
            if (modalManager) {
                modalManager.showConfirmation(
                    'Restart System',
                    'This will restart all system services. This may take a few minutes. Continue?',
                    async () => {
                        await this.performSystemRestart();
                    }
                );
            }

        } catch (error) {
            console.error('❌ System restart failed:', error);
            this.adminPage.showError('Restart Failed', error.message);
        }
    }

    /**
     * Perform system restart
     */
    async performSystemRestart() {
        try {
            this.systemStatus.maintenanceMode = true;

            // Stop all services
            console.log('🛑 Stopping services...');
            this.services.forEach(service => {
                service.status = 'stopping';
            });

            await new Promise(resolve => setTimeout(resolve, 2000));

            // Start all services
            console.log('▶️ Starting services...');
            this.services.forEach(service => {
                service.status = 'running';
                service.uptime = 0;
                service.restartCount++;
            });

            await new Promise(resolve => setTimeout(resolve, 3000));

            this.systemStatus.maintenanceMode = false;
            this.systemStatus.lastRestart = new Date();
            this.systemStatus.uptime = 0;

            this.createAlert('info', 'System Restart', 'System restart completed successfully');
            this.adminPage.showSuccess('Restart Complete', 'All services are running normally');

        } catch (error) {
            console.error('❌ System restart process failed:', error);
            this.adminPage.showError('Restart Process Failed', error.message);
        }
    }

    /**
     * Toggle maintenance mode
     */
    toggleMaintenanceMode() {
        this.systemStatus.maintenanceMode = !this.systemStatus.maintenanceMode;
        
        const status = this.systemStatus.maintenanceMode ? 'enabled' : 'disabled';
        this.createAlert('info', 'Maintenance Mode', `Maintenance mode ${status}`);
        
        console.log(`🔧 Maintenance mode ${status}`);
        this.adminPage.showInfo('Maintenance Mode', `Maintenance mode ${status}`);
    }

    /**
     * Get system health
     */
    getSystemHealth() {
        const runningServices = Array.from(this.services.values()).filter(s => s.status === 'running').length;
        const totalServices = this.services.size;
        const healthPercentage = (runningServices / totalServices) * 100;

        let healthStatus;
        if (healthPercentage >= 90) healthStatus = 'excellent';
        else if (healthPercentage >= 75) healthStatus = 'good';
        else if (healthPercentage >= 50) healthStatus = 'fair';
        else healthStatus = 'poor';

        return {
            percentage: healthPercentage,
            status: healthStatus,
            runningServices,
            totalServices,
            criticalServicesDown: Array.from(this.services.values()).filter(s => s.critical && s.status !== 'running').length
        };
    }

    /**
     * Update monitoring display
     */
    updateMonitoringDisplay() {
        // Update service cards if monitoring section is visible
        const monitoringSection = document.getElementById('monitoring');
        if (!monitoringSection || !monitoringSection.classList.contains('active')) return;

        // This would update the monitoring dashboard
        console.log('📊 Updating monitoring display');
    }

    /**
     * Get system information
     */
    getSystemInfo() {
        return {
            ...this.systemStatus,
            services: Object.fromEntries(this.services),
            alerts: this.alerts.slice(0, 10), // Last 10 alerts
            health: this.getSystemHealth()
        };
    }

    /**
     * Format uptime
     */
    formatUptime(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) {
            return `${days}d ${hours % 24}h ${minutes % 60}m`;
        } else if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else {
            return `${minutes}m ${seconds % 60}s`;
        }
    }

    /**
     * Export system logs
     */
    exportSystemLogs() {
        const logData = {
            timestamp: new Date().toISOString(),
            systemStatus: this.systemStatus,
            services: Object.fromEntries(this.services),
            alerts: this.alerts,
            health: this.getSystemHealth()
        };

        const blob = new Blob([JSON.stringify(logData, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `system-logs-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.adminPage.showSuccess('Export Complete', 'System logs exported successfully');
    }

    /**
     * Check for updates
     */
    async checkForUpdates() {
        try {
            console.log('🔍 Checking for system updates...');

            // Simulate update check
            await new Promise(resolve => setTimeout(resolve, 2000));

            const hasUpdates = Math.random() < 0.3; // 30% chance of updates

            if (hasUpdates) {
                this.adminPage.showInfo('Updates Available', 'System updates are available. Please schedule maintenance.');
                this.createAlert('info', 'Updates Available', 'New system updates are ready for installation');
            } else {
                this.adminPage.showSuccess('System Up to Date', 'No updates available at this time');
            }

        } catch (error) {
            console.error('❌ Failed to check for updates:', error);
            this.adminPage.showError('Update Check Failed', error.message);
        }
    }

    /**
     * Handle section change
     */
    onSectionChange(sectionId) {
        if (sectionId === 'monitoring') {
            this.updateMonitoringDisplay();
        }
    }

    /**
     * Update real-time data
     */
    updateRealTimeData() {
        this.updateSystemMetrics();
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }

        this.services.clear();
        this.alerts = [];

        console.log('🧹 System Manager cleaned up');
    }
}

// Export for global access in backward compatibility functions
window.SystemManager = SystemManager;