/**
 * Enhanced CSP Admin Portal - Main Module
 * Extracted from frontend/pages/admin.html
 */

// Import helper modules
import { DashboardManager } from './dashboardManager.js';
import { ModalManager } from './modalManager.js';
import { NavigationManager } from './navigationManager.js';
import { AgentManager } from './agentManager.js';
import { SystemManager } from './systemManager.js';

/**
 * Main AdminPage class - Entry point for admin portal functionality
 */
class AdminPage {
    constructor() {
        this.initialized = false;
        this.managers = new Map();
        this.state = {
            currentSection: 'dashboard',
            loading: false,
            user: 'admin@csp.ai',
            isAdmin: true
        };
        
        // System statistics
        this.systemStats = {
            health: 98.5,
            users: 1247,
            agents: '22/24',
            throughput: '2.3M',
            storage: 78,
            alerts: 3
        };
        
        this.init();
    }

    /**
     * Initialize the admin page
     */
    async init() {
        try {
            console.log('🛡️ Enhanced CSP Admin Portal initialized');
            
            // Initialize managers
            await this.initializeManagers();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize UI
            this.initializeUI();
            
            // Start real-time updates
            this.startRealTimeUpdates();
            
            // Initialize range sliders
            this.initializeRangeSliders();
            
            this.initialized = true;
            console.log('✅ Admin Portal initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Admin Portal:', error);
            this.showNotification('Failed to initialize Admin Portal: ' + error.message, 'error');
        }
    }

    /**
     * Initialize all manager modules
     */
    async initializeManagers() {
        try {
            // Initialize dashboard manager
            this.managers.set('dashboard', new DashboardManager(this));
            
            // Initialize modal manager
            this.managers.set('modal', new ModalManager(this));
            
            // Initialize navigation manager
            this.managers.set('navigation', new NavigationManager(this));
            
            // Initialize agent manager
            this.managers.set('agent', new AgentManager(this));
            
            // Initialize system manager
            this.managers.set('system', new SystemManager(this));
            
            // Initialize all managers
            for (const [name, manager] of this.managers) {
                if (manager.init) {
                    await manager.init();
                    console.log(`✅ ${name} manager initialized`);
                }
            }
            
        } catch (error) {
            console.error('❌ Failed to initialize managers:', error);
            throw error;
        }
    }

    /**
     * Set up global event listeners
     */
    setupEventListeners() {
        // Navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const section = item.getAttribute('data-section');
                if (section) {
                    this.showSection(section);
                }
            });
            
            // Keyboard navigation
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const section = item.getAttribute('data-section');
                    if (section) {
                        this.showSection(section);
                    }
                }
            });
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'h':
                        e.preventDefault();
                        this.showSection('dashboard');
                        break;
                    case 'u':
                        e.preventDefault();
                        this.showSection('users');
                        break;
                    case 'm':
                        e.preventDefault();
                        this.showSection('monitoring');
                        break;
                    case 's':
                        e.preventDefault();
                        this.showSection('settings');
                        break;
                }
            }
        });

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('🔄 Tab hidden - reducing update frequency');
            } else {
                console.log('🔄 Tab visible - resuming normal updates');
                this.updateSystemStats();
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            const sidebar = document.querySelector('.sidebar');
            if (window.innerWidth > 768) {
                sidebar?.classList.remove('open');
            }
        });

        // Error handling
        window.addEventListener('error', (e) => {
            console.error('🚨 JavaScript Error:', e.error);
            this.showNotification('A system error occurred. Please refresh the page.', 'error');
        });

        // Unhandled promise rejection handling
        window.addEventListener('unhandledrejection', (e) => {
            console.error('🚨 Unhandled Promise Rejection:', e.reason);
            this.showNotification('An unexpected error occurred.', 'error');
        });
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        // Update user info in header
        this.updateUserInfo();
        
        // Set initial active section
        this.showSection(this.state.currentSection);
        
        // Load initial data
        this.loadSectionData(this.state.currentSection);
    }

    /**
     * Initialize range sliders in modals
     */
    initializeRangeSliders() {
        const cpuRange = document.getElementById('cpu-range');
        const memoryRange = document.getElementById('memory-range');
        const cpuDisplay = document.getElementById('cpu-display');
        const memoryDisplay = document.getElementById('memory-display');

        if (cpuRange && cpuDisplay) {
            cpuRange.addEventListener('input', function() {
                cpuDisplay.textContent = this.value + ' cores';
            });
        }

        if (memoryRange && memoryDisplay) {
            memoryRange.addEventListener('input', function() {
                memoryDisplay.textContent = this.value + ' GB';
            });
        }
    }

    /**
     * Update user information in the header
     */
    updateUserInfo() {
        const userElement = document.getElementById('admin-user');
        if (userElement) {
            userElement.textContent = this.state.user;
        }
    }

    /**
     * Show specific section
     */
    showSection(sectionId) {
        try {
            console.log(`📍 Navigating to section: ${sectionId}`);
            
            // Hide all sections
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from all nav items
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Show selected section
            const selectedSection = document.getElementById(sectionId);
            if (selectedSection) {
                selectedSection.classList.add('active');
            }
            
            // Add active class to corresponding nav item
            const activeNavItem = document.querySelector(`[data-section="${sectionId}"]`);
            if (activeNavItem) {
                activeNavItem.classList.add('active');
            }
            
            // Update state
            this.state.currentSection = sectionId;
            
            // Load section-specific data
            this.loadSectionData(sectionId);
            
            // Notify managers about section change
            this.managers.forEach(manager => {
                if (manager.onSectionChange) {
                    manager.onSectionChange(sectionId);
                }
            });

        } catch (error) {
            console.error('❌ Failed to show section:', error);
            this.showNotification('Failed to switch section: ' + error.message, 'error');
        }
    }

    /**
     * Load section-specific data
     */
    loadSectionData(sectionId) {
        console.log(`🔄 Loading data for section: ${sectionId}`);
        
        switch(sectionId) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'monitoring':
                this.updateMonitoringCharts();
                this.loadMonitoringMetrics();
                break;
            case 'alerts':
                this.loadAlertData();
                break;
            case 'users':
                this.loadUserData();
                break;
            case 'roles':
                this.loadRoleData();
                break;
            case 'auth':
                this.loadAuthenticationData();
                break;
            case 'ai-models':
                this.loadAIModelData();
                break;
            case 'agents':
                this.loadAgentData();
                break;
            case 'protocols':
                this.loadProtocolData();
                break;
            case 'settings':
                this.loadSystemSettings();
                break;
            case 'infrastructure':
                this.loadInfrastructureData();
                break;
            case 'integrations':
                this.loadIntegrationData();
                break;
            case 'security':
                this.loadSecurityData();
                break;
            case 'backups':
                this.loadBackupData();
                break;
            case 'logs':
                this.startLogTail();
                this.loadLogData();
                break;
            case 'maintenance':
                this.loadMaintenanceData();
                break;
            case 'licenses':
                this.loadLicenseData();
                break;
            case 'billing':
                this.loadBillingData();
                break;
            case 'audit':
                this.loadAuditData();
                break;
            default:
                console.log(`✅ Section ${sectionId} loaded (static content)`);
        }
    }

    /**
     * Update system statistics
     */
    updateSystemStats() {
        // Simulate real-time data updates
        this.systemStats.health = Math.max(95, Math.min(100, this.systemStats.health + (Math.random() - 0.5) * 2));
        this.systemStats.users = Math.floor(Math.random() * 100) + 1200;
        this.systemStats.throughput = (Math.random() * 0.5 + 2).toFixed(1) + 'M';
        this.systemStats.storage = Math.max(70, Math.min(90, this.systemStats.storage + (Math.random() - 0.5) * 5));
        this.systemStats.alerts = Math.max(0, Math.min(10, this.systemStats.alerts + Math.floor((Math.random() - 0.5) * 3)));
        
        // Update dashboard cards if visible
        if (this.state.currentSection === 'dashboard') {
            const healthElement = document.getElementById('system-health');
            const usersElement = document.getElementById('active-users');
            const agentsElement = document.getElementById('ai-agents');
            const throughputElement = document.getElementById('throughput');
            const storageElement = document.getElementById('storage');
            const alertsElement = document.getElementById('alerts');

            if (healthElement) healthElement.textContent = this.systemStats.health.toFixed(1) + '%';
            if (usersElement) usersElement.textContent = this.systemStats.users.toLocaleString();
            if (agentsElement) agentsElement.textContent = this.systemStats.agents;
            if (throughputElement) throughputElement.textContent = this.systemStats.throughput;
            if (storageElement) storageElement.textContent = this.systemStats.storage + '%';
            if (alertsElement) alertsElement.textContent = this.systemStats.alerts;

            // Update progress bars
            const storageProgress = document.querySelector('.progress-fill');
            if (storageProgress) {
                storageProgress.style.width = this.systemStats.storage + '%';
            }
        }
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        this.updateSystemStats();
        setInterval(() => this.updateSystemStats(), 5000);
        setInterval(() => this.updateSystemTime(), 1000);
        console.log('📡 Real-time updates started');
    }

    /**
     * Update system time
     */
    updateSystemTime() {
        const now = new Date();
        // Update any time displays if needed
    }

    /**
     * Data loading functions
     */
    loadDashboardData() {
        console.log('📊 Loading dashboard data...');
        this.updateSystemStats();
        this.updateRecentEvents();
        this.showNotification('Dashboard refreshed', 'info');
    }

    loadUserData() {
        console.log('👥 Loading user data...');
        setTimeout(() => {
            console.log('✅ User data loaded');
            this.showNotification('User data refreshed', 'success');
        }, 500);
    }

    loadAIModelData() {
        console.log('🧠 Loading AI model data...');
        setTimeout(() => {
            console.log('✅ AI model data loaded');
            this.showNotification('AI model data refreshed', 'success');
        }, 800);
    }

    loadAlertData() {
        console.log('🚨 Loading alert data...');
        setTimeout(() => {
            console.log('✅ Alert data loaded');
            this.updateAlertsCount();
        }, 600);
    }

    loadAgentData() {
        console.log('🤖 Loading agent data...');
        setTimeout(() => {
            console.log('✅ Agent data loaded');
            this.updateAgentStats();
        }, 700);
    }

    loadRoleData() {
        console.log('🔐 Loading role and permissions data...');
        setTimeout(() => {
            console.log('✅ Role data loaded');
            this.showNotification('Role permissions updated', 'info');
        }, 400);
    }

    loadAuthenticationData() {
        console.log('🔑 Loading authentication logs...');
        setTimeout(() => {
            console.log('✅ Authentication data loaded');
            this.updateAuthStats();
        }, 500);
    }

    loadProtocolData() {
        console.log('📋 Loading protocol templates...');
        setTimeout(() => {
            console.log('✅ Protocol data loaded');
            this.showNotification('Protocol templates loaded', 'info');
        }, 300);
    }

    loadSystemSettings() {
        console.log('⚙️ Loading system settings...');
        setTimeout(() => {
            console.log('✅ System settings loaded');
            this.validateSystemConfiguration();
        }, 400);
    }

    loadInfrastructureData() {
        console.log('🖥️ Loading infrastructure status...');
        setTimeout(() => {
            console.log('✅ Infrastructure data loaded');
            this.updateServerMetrics();
        }, 600);
    }

    loadIntegrationData() {
        console.log('🔌 Loading integration status...');
        setTimeout(() => {
            console.log('✅ Integration data loaded');
            this.testConnections();
        }, 800);
    }

    loadSecurityData() {
        console.log('🛡️ Loading security data...');
        setTimeout(() => {
            console.log('✅ Security data loaded');
            this.updateSecurityScore();
        }, 700);
    }

    loadBackupData() {
        console.log('💾 Loading backup status...');
        setTimeout(() => {
            console.log('✅ Backup data loaded');
            this.checkBackupHealth();
        }, 500);
    }

    loadLogData() {
        console.log('📄 Loading system logs...');
        setTimeout(() => {
            console.log('✅ Log data loaded');
            this.updateLogViewer();
        }, 300);
    }

    loadMaintenanceData() {
        console.log('🔧 Loading maintenance tasks...');
        setTimeout(() => {
            console.log('✅ Maintenance data loaded');
            this.checkMaintenanceTasks();
        }, 400);
    }

    loadLicenseData() {
        console.log('📜 Loading license information...');
        setTimeout(() => {
            console.log('✅ License data loaded');
            this.checkLicenseExpiry();
        }, 600);
    }

    loadBillingData() {
        console.log('💳 Loading billing information...');
        setTimeout(() => {
            console.log('✅ Billing data loaded');
            this.updateUsageMetrics();
        }, 800);
    }

    loadAuditData() {
        console.log('🔍 Loading audit trail...');
        setTimeout(() => {
            console.log('✅ Audit data loaded');
            this.updateAuditStats();
        }, 500);
    }

    loadMonitoringMetrics() {
        console.log('📈 Loading monitoring metrics...');
        setTimeout(() => {
            console.log('✅ Monitoring metrics loaded');
            this.updatePerformanceCharts();
        }, 900);
    }

    /**
     * Update functions for enhanced realism
     */
    updateRecentEvents() {
        console.log('📅 Updating recent events...');
    }

    updateAlertsCount() {
        this.systemStats.alerts = Math.max(0, Math.min(10, this.systemStats.alerts + Math.floor((Math.random() - 0.3) * 2)));
        const alertsElement = document.getElementById('alerts');
        if (alertsElement) {
            alertsElement.textContent = this.systemStats.alerts;
        }
        console.log(`🚨 Updated alerts count: ${this.systemStats.alerts}`);
    }

    updateAgentStats() {
        const activeAgents = Math.floor(Math.random() * 3) + 22;
        const totalAgents = 24;
        this.systemStats.agents = `${activeAgents}/${totalAgents}`;
        
        const agentsElement = document.getElementById('ai-agents');
        if (agentsElement) {
            agentsElement.textContent = this.systemStats.agents;
        }
        console.log(`🤖 Updated agent stats: ${this.systemStats.agents}`);
    }

    updateAuthStats() {
        console.log('🔑 Authentication stats updated');
        this.showNotification('Authentication data refreshed', 'info');
    }

    validateSystemConfiguration() {
        console.log('⚙️ Validating system configuration...');
        this.showNotification('System configuration validated', 'success');
    }

    updateServerMetrics() {
        console.log('🖥️ Server metrics updated');
        this.showNotification('Infrastructure status refreshed', 'info');
    }

    testConnections() {
        console.log('🔌 Testing integration connections...');
        setTimeout(() => {
            const successRate = Math.random();
            if (successRate > 0.8) {
                this.showNotification('All integrations healthy', 'success');
            } else if (successRate > 0.6) {
                this.showNotification('Some integration issues detected', 'warning');
            } else {
                this.showNotification('Integration failures detected', 'error');
            }
        }, 1000);
    }

    updateSecurityScore() {
        const newScore = Math.floor(Math.random() * 10) + 90;
        console.log(`🛡️ Security score updated: ${newScore}%`);
        this.showNotification(`Security score: ${newScore}%`, newScore > 95 ? 'success' : 'warning');
    }

    checkBackupHealth() {
        console.log('💾 Backup health checked');
        const backupStatus = Math.random() > 0.1 ? 'healthy' : 'needs attention';
        this.showNotification(`Backup status: ${backupStatus}`, backupStatus === 'healthy' ? 'success' : 'warning');
    }

    updateLogViewer() {
        console.log('📄 Log viewer updated');
    }

    checkMaintenanceTasks() {
        console.log('🔧 Maintenance tasks checked');
        const pendingTasks = Math.floor(Math.random() * 5);
        this.showNotification(`${pendingTasks} maintenance tasks pending`, pendingTasks === 0 ? 'success' : 'info');
    }

    checkLicenseExpiry() {
        console.log('📜 License expiry checked');
        const expiringLicenses = Math.floor(Math.random() * 3);
        if (expiringLicenses > 0) {
            this.showNotification(`${expiringLicenses} licenses expiring soon`, 'warning');
        } else {
            this.showNotification('All licenses current', 'success');
        }
    }

    updateUsageMetrics() {
        console.log('💳 Usage metrics updated');
        const currentUsage = (Math.random() * 2000 + 3000).toFixed(0);
        this.showNotification(`Current month usage: ${currentUsage}`, 'info');
    }

    updateAuditStats() {
        console.log('🔍 Audit statistics updated');
        this.showNotification('Audit trail refreshed', 'info');
    }

    updatePerformanceCharts() {
        console.log('📈 Performance charts updated');
    }

    updateMonitoringCharts() {
        console.log('📊 Updating monitoring charts...');
    }

    startLogTail() {
        console.log('📄 Starting log tail...');
    }

    /**
     * Mobile sidebar toggle
     */
    toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('open');
        }
    }

    /**
     * Emergency shutdown handler
     */
    async emergencyShutdown() {
        if (confirm('⚠️ WARNING: This will immediately shut down all CSP processes and AI agents. Continue?')) {
            if (confirm('🚨 FINAL WARNING: This action cannot be undone. All active sessions will be terminated. Proceed?')) {
                console.log('🚨 Emergency shutdown initiated...');
                this.showNotification('Emergency shutdown initiated!', 'error');
                
                const systemManager = this.managers.get('system');
                if (systemManager) {
                    await systemManager.emergencyShutdown();
                } else {
                    // Fallback implementation
                    setTimeout(() => {
                        this.showNotification('All systems stopped. Manual restart required.', 'error');
                    }, 3000);
                }
            }
        }
    }

    /**
     * Notification system
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                ${message}
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        let container = document.querySelector('.notification-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notification-container';
            document.body.appendChild(container);
        }

        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOutRight 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);

        console.log(`📢 Notification (${type}): ${message}`);
    }

    /**
     * Utility function for debouncing
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Get manager by name
     */
    getManager(name) {
        return this.managers.get(name);
    }

    /**
     * Update state
     */
    setState(newState) {
        this.state = { ...this.state, ...newState };
    }

    /**
     * Get current state
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Cleanup on page unload
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.managers.forEach(manager => {
            if (manager.destroy) {
                manager.destroy();
            }
        });

        this.managers.clear();
        console.log('🧹 Admin Portal cleaned up');
    }
}

// Global functions for backwards compatibility with existing HTML onclick handlers
window.showSection = function(sectionId) {
    if (window.adminPage) {
        window.adminPage.showSection(sectionId);
    } else {
        console.log(`Queued navigation to: ${sectionId}`);
        // Fallback manual section switching
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        document.querySelectorAll('.nav-item').forEach(navItem => {
            navItem.classList.remove('active');
        });
        
        const targetSection = document.getElementById(sectionId);
        const navItem = document.querySelector(`[data-section="${sectionId}"]`);
        if (targetSection) {
            targetSection.classList.add('active');
        }
        if (navItem) {
            navItem.classList.add('active');
        }
    }
};

window.toggleSidebar = function() {
    if (window.adminPage) {
        window.adminPage.toggleSidebar();
    } else {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('open');
        }
    }
};

window.emergencyShutdown = function() {
    if (window.adminPage) {
        window.adminPage.emergencyShutdown();
    } else {
        if (confirm('Emergency shutdown will stop all services immediately. Continue?')) {
            alert('Emergency shutdown initiated - Admin system is loading...');
        }
    }
};

// All the extracted functions from the original HTML
window.openAddUserModal = function() {
    if (window.adminPage) {
        const modalManager = window.adminPage.getManager('modal');
        if (modalManager && modalManager.openAddUserModal) {
            modalManager.openAddUserModal();
        } else {
            window.openModal('add-user-modal');
        }
    } else {
        window.openModal('add-user-modal');
    }
};

window.openDeployModelModal = function() {
    if (window.adminPage) {
        const modalManager = window.adminPage.getManager('modal');
        if (modalManager && modalManager.openDeployModelModal) {
            modalManager.openDeployModelModal();
        } else {
            window.openModal('deploy-model-modal');
        }
    } else {
        window.openModal('deploy-model-modal');
    }
};

window.openCreateAgentModal = function() {
    if (window.adminPage) {
        const modalManager = window.adminPage.getManager('modal');
        if (modalManager) {
            modalManager.openCreateAgentModal();
        }
    }
    window.openModal('create-agent-modal');
};

window.openModal = function(modalId) {
    console.log(`📂 Opening modal: ${modalId}`);
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }
};

window.closeModal = function(modalId) {
    console.log(`🚪 Closing modal: ${modalId}`);
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.style.display = 'none';
        }, 300);
        document.body.style.overflow = '';
    }
};

// Administrative action functions
window.createUser = function() {
    console.log('👤 Creating new user...');
    window.closeModal('add-user-modal');
    if (window.adminPage) {
        window.adminPage.showNotification('User created successfully!', 'success');
    }
};

window.deployModel = function() {
    console.log('🚀 Deploying AI model...');
    window.closeModal('deploy-model-modal');
    if (window.adminPage) {
        window.adminPage.showNotification('Model deployment started!', 'success');
    }
};

window.createAgent = function() {
    console.log('🤖 Creating new agent...');
    window.closeModal('create-agent-modal');
    if (window.adminPage) {
        window.adminPage.showNotification('Agent created successfully!', 'success');
    }
};

// Alert management functions
window.acknowledgeAlert = function(alertId) {
    console.log(`🔕 Acknowledging alert: ${alertId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('Alert acknowledged', 'info');
    }
};

window.resolveAlert = function(alertId) {
    console.log(`🔧 Resolving alert: ${alertId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('Alert resolved', 'success');
    }
};

window.acknowledgeAllAlerts = function() {
    console.log('🔕 Acknowledging all alerts...');
    if (window.adminPage) {
        window.adminPage.showNotification('All alerts acknowledged', 'info');
    }
};

// User management functions
window.editUser = function(userId) {
    console.log(`✏️ Editing user: ${userId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('User edit functionality coming soon', 'info');
    }
};

window.deactivateUser = function(userId) {
    if (confirm('Are you sure you want to deactivate this user?')) {
        console.log(`🚫 Deactivating user: ${userId}`);
        if (window.adminPage) {
            window.adminPage.showNotification('User deactivated', 'warning');
        }
    }
};

window.activateUser = function(userId) {
    console.log(`✅ Activating user: ${userId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('User activated', 'success');
    }
};

// Agent management functions (delegated to AgentManager)
window.viewAgentDetails = function(agentId) {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.viewAgent(agentId);
        }
    }
};

window.pauseAgent = function(agentId) {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.toggleAgent(agentId);
        }
    }
};

window.stopAgent = function(agentId) {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.deleteAgent(agentId);
        }
    }
};

window.startAgent = function(agentId) {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.toggleAgent(agentId);
        }
    }
};

window.deleteAgent = function(agentId) {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.deleteAgent(agentId);
        }
    }
};

window.startAllAgents = function() {
    if (window.adminPage) {
        const agentManager = window.adminPage.getManager('agent');
        if (agentManager) {
            agentManager.deployAllAgents();
        }
    }
};

window.pauseAllAgents = function() {
    if (confirm('Pause all active agents?')) {
        console.log('⏸️ Pausing all agents...');
        if (window.adminPage) {
            window.adminPage.showNotification('All agents paused', 'warning');
        }
    }
};

// Model management functions
window.configureModel = function(modelId) {
    console.log(`⚙️ Configuring model: ${modelId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('Model configuration functionality coming soon', 'info');
    }
};

window.pauseModel = function(modelId) {
    console.log(`⏸️ Pausing model: ${modelId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('Model paused', 'warning');
    }
};

// Security functions
window.banIP = function(ipAddress) {
    if (confirm(`Ban IP address ${ipAddress}?`)) {
        console.log(`🚫 Banning IP: ${ipAddress}`);
        if (window.adminPage) {
            window.adminPage.showNotification(`IP ${ipAddress} has been banned`, 'warning');
        }
    }
};

window.viewIncidentDetails = function(incidentId) {
    console.log(`🔍 Viewing incident details: ${incidentId}`);
    if (window.adminPage) {
        window.adminPage.showNotification('Incident details functionality coming soon', 'info');
    }
};

window.runSecurityScan = function() {
    console.log('🔍 Running security scan...');
    if (window.adminPage) {
        window.adminPage.showNotification('Security scan started', 'info');
    }
};

// Settings functions
window.toggleMaintenanceMode = function(enabled) {
    console.log(`🔧 Maintenance mode: ${enabled ? 'ON' : 'OFF'}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`Maintenance mode ${enabled ? 'enabled' : 'disabled'}`, enabled ? 'warning' : 'info');
    }
};

window.toggleDebugLogging = function(enabled) {
    console.log(`🐛 Debug logging: ${enabled ? 'ON' : 'OFF'}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`Debug logging ${enabled ? 'enabled' : 'disabled'}`, 'info');
    }
};

window.toggleAutoBackups = function(enabled) {
    console.log(`💾 Auto backups: ${enabled ? 'ON' : 'OFF'}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`Auto backups ${enabled ? 'enabled' : 'disabled'}`, 'info');
    }
};

window.toggleEnforce2FA = function(enabled) {
    console.log(`🔐 Enforce 2FA: ${enabled ? 'ON' : 'OFF'}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`2FA enforcement ${enabled ? 'enabled' : 'disabled'}`, enabled ? 'warning' : 'info');
    }
};

window.testDatabaseConnection = function() {
    console.log('🔌 Testing database connection...');
    if (window.adminPage) {
        window.adminPage.showNotification('Testing database connection...', 'info');
        
        setTimeout(() => {
            window.adminPage.showNotification('Database connection successful!', 'success');
        }, 2000);
    }
};

// Export functions
window.exportUsers = function() {
    console.log('📄 Exporting users...');
    if (window.adminPage) {
        window.adminPage.showNotification('User export started', 'info');
    }
};

window.exportSystemEvents = function() {
    console.log('📄 Exporting system events...');
    if (window.adminPage) {
        window.adminPage.showNotification('System events export started', 'info');
    }
};

window.exportLogs = function() {
    console.log('📄 Exporting logs...');
    if (window.adminPage) {
        window.adminPage.showNotification('Log export started', 'info');
    }
};

window.exportAuditLog = function() {
    console.log('📄 Exporting audit log...');
    if (window.adminPage) {
        window.adminPage.showNotification('Audit log export started', 'info');
    }
};

// Enhanced refresh functions
window.refreshUsers = function() {
    if (window.adminPage) {
        window.adminPage.loadUserData();
    }
};

window.refreshModels = function() {
    if (window.adminPage) {
        window.adminPage.loadAIModelData();
    }
};

window.refreshAlerts = function() {
    if (window.adminPage) {
        window.adminPage.loadAlertData();
    }
};

window.refreshSystemEvents = function() {
    if (window.adminPage) {
        window.adminPage.loadDashboardData();
    }
};

window.refreshInfrastructure = function() {
    if (window.adminPage) {
        window.adminPage.loadInfrastructureData();
    }
};

window.refreshAuditLog = function() {
    if (window.adminPage) {
        window.adminPage.loadAuditData();
    }
};

// Enhanced monitoring functions
window.updateMonitoringTimeRange = function(range) {
    console.log(`📊 Updating monitoring time range: ${range}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`Monitoring view updated: ${range}`, 'info');
        window.adminPage.updateMonitoringCharts();
    }
};

window.updateBillingChart = function(range) {
    console.log(`💳 Updating billing chart: ${range}`);
    if (window.adminPage) {
        window.adminPage.showNotification(`Billing view updated: ${range}`, 'info');
        window.adminPage.loadBillingData();
    }
};

// Filter functions
window.filterUsers = function() {
    const searchTerm = document.getElementById('user-search')?.value;
    console.log(`🔍 Filtering users: ${searchTerm}`);
};

window.filterAlerts = function() {
    const searchTerm = document.getElementById('alert-search')?.value;
    console.log(`🔍 Filtering alerts: ${searchTerm}`);
};

window.filterLogs = function() {
    const searchTerm = document.getElementById('log-search')?.value;
    console.log(`🔍 Filtering logs: ${searchTerm}`);
};

window.filterAuditLogs = function() {
    const searchTerm = document.getElementById('audit-search')?.value;
    console.log(`🔍 Filtering audit logs: ${searchTerm}`);
};

// Log management
window.pauseLogTail = function() {
    console.log('⏸️ Pausing log tail...');
    if (window.adminPage) {
        window.adminPage.showNotification('Log tail paused', 'info');
    }
};

window.clearLogs = function() {
    if (confirm('Clear all logs? This action cannot be undone.')) {
        console.log('🧹 Clearing logs...');
        if (window.adminPage) {
            window.adminPage.showNotification('Logs cleared', 'warning');
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('🔧 DOM loaded, initializing Admin Page...');
        window.adminPage = new AdminPage();
        
        // Set up fallback navigation immediately for any missed onclick handlers
        document.querySelectorAll('.nav-item[data-section]').forEach(item => {
            const existingOnclick = item.getAttribute('onclick');
            if (!existingOnclick) {
                item.addEventListener('click', (e) => {
                    const sectionId = item.getAttribute('data-section');
                    window.showSection(sectionId);
                });
            }
        });
        
    } catch (error) {
        console.error('❌ Failed to initialize Admin Page:', error);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.adminPage) {
        window.adminPage.destroy();
    }
});

// Export for module usage
export { AdminPage };

console.log('✅ Enhanced CSP Admin Portal scripts loaded successfully');/**
 * Enhanced CSP Admin Portal - Main Module
 * Extracted from frontend/pages/admin.html
 */

// Import helper modules
import { DashboardManager } from './dashboardManager.js';
import { ModalManager } from './modalManager.js';
import { NavigationManager } from './navigationManager.js';
import { AgentManager } from './agentManager.js';
import { SystemManager } from './systemManager.js';

/**
 * Main AdminPage class - Entry point for admin portal functionality
 */
class AdminPage {
    constructor() {
        this.initialized = false;
        this.managers = new Map();
        this.state = {
            currentSection: 'dashboard',
            loading: false,
            user: 'admin@csp.ai',
            isAdmin: true
        };
        
        this.init();
    }

    /**
     * Initialize the admin page
     */
    async init() {
        try {
            console.log('🚀 Initializing Enhanced CSP Admin Portal...');
            
            // Initialize managers
            await this.initializeManagers();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize UI
            this.initializeUI();
            
            // Start real-time updates
            this.startRealTimeUpdates();
            
            this.initialized = true;
            console.log('✅ Admin Portal initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Admin Portal:', error);
            this.showError('Failed to initialize Admin Portal', error.message);
        }
    }

    /**
     * Initialize all manager modules
     */
    async initializeManagers() {
        try {
            // Initialize dashboard manager
            this.managers.set('dashboard', new DashboardManager(this));
            
            // Initialize modal manager
            this.managers.set('modal', new ModalManager(this));
            
            // Initialize navigation manager
            this.managers.set('navigation', new NavigationManager(this));
            
            // Initialize agent manager
            this.managers.set('agent', new AgentManager(this));
            
            // Initialize system manager
            this.managers.set('system', new SystemManager(this));
            
            // Initialize all managers
            for (const [name, manager] of this.managers) {
                if (manager.init) {
                    await manager.init();
                    console.log(`✅ ${name} manager initialized`);
                }
            }
            
        } catch (error) {
            console.error('❌ Failed to initialize managers:', error);
            throw error;
        }
    }

    /**
     * Set up global event listeners
     */
    setupEventListeners() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseRealTimeUpdates();
            } else {
                this.resumeRealTimeUpdates();
            }
        });

        // Handle window resize
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 250));

        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Handle clicks outside modals
        document.addEventListener('click', (e) => {
            this.handleOutsideClick(e);
        });

        // Handle escape key for modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modalManager = this.managers.get('modal');
                if (modalManager) {
                    modalManager.closeActiveModal();
                }
            }
        });
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        // Update user info in header
        this.updateUserInfo();
        
        // Set initial active section
        this.showSection(this.state.currentSection);
        
        // Initialize real-time indicator
        this.updateRealtimeIndicator(true);
        
        // Set up mobile navigation
        this.setupMobileNavigation();
    }

    /**
     * Update user information in the header
     */
    updateUserInfo() {
        const userElement = document.getElementById('admin-user');
        if (userElement) {
            userElement.textContent = this.state.user;
        }
    }

    /**
     * Set up mobile navigation
     */
    setupMobileNavigation() {
        const mobileToggle = document.querySelector('.mobile-toggle');
        const sidebar = document.querySelector('.sidebar');
        
        if (mobileToggle && sidebar) {
            mobileToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });

            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 768) {
                    if (!sidebar.contains(e.target) && !mobileToggle.contains(e.target)) {
                        sidebar.classList.remove('open');
                    }
                }
            });
        }
    }

    /**
     * Show a specific section
     */
    showSection(sectionId) {
        try {
            // Update state
            this.state.currentSection = sectionId;
            
            // Hide all sections
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.remove('active');
            });

            // Show target section
            const targetSection = document.getElementById(sectionId);
            if (targetSection) {
                targetSection.classList.add('active');
            }

            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });

            const navItem = document.querySelector(`[data-section="${sectionId}"]`);
            if (navItem) {
                navItem.classList.add('active');
            }

            // Notify managers about section change
            this.managers.forEach(manager => {
                if (manager.onSectionChange) {
                    manager.onSectionChange(sectionId);
                }
            });

            console.log(`📄 Switched to section: ${sectionId}`);
            
        } catch (error) {
            console.error('❌ Failed to show section:', error);
            this.showError('Failed to switch section', error.message);
        }
    }

    /**
     * Toggle sidebar navigation (mobile)
     */
    toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('open');
        }
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        // Update every 5 seconds
        this.updateInterval = setInterval(() => {
            if (!document.hidden && this.initialized) {
                this.managers.forEach(manager => {
                    if (manager.updateRealTimeData) {
                        manager.updateRealTimeData();
                    }
                });
            }
        }, 5000);

        this.updateRealtimeIndicator(true);
        console.log('📡 Real-time updates started');
    }

    /**
     * Pause real-time updates
     */
    pauseRealTimeUpdates() {
        this.updateRealtimeIndicator(false);
        console.log('⏸️ Real-time updates paused');
    }

    /**
     * Resume real-time updates
     */
    resumeRealTimeUpdates() {
        this.updateRealtimeIndicator(true);
        console.log('▶️ Real-time updates resumed');
    }

    /**
     * Update real-time indicator
     */
    updateRealtimeIndicator(isActive) {
        const indicator = document.querySelector('.realtime-indicator');
        if (indicator) {
            if (isActive) {
                indicator.classList.add('active');
                indicator.querySelector('span').textContent = 'Live';
            } else {
                indicator.classList.remove('active');
                indicator.querySelector('span').textContent = 'Paused';
            }
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Notify managers about resize
        this.managers.forEach(manager => {
            if (manager.onResize) {
                manager.onResize();
            }
        });
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(e) {
        // Only handle shortcuts when no input is focused
        if (document.activeElement.tagName === 'INPUT' || 
            document.activeElement.tagName === 'TEXTAREA') {
            return;
        }

        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            this.focusSearch();
        }

        // Number keys for quick navigation
        if (e.key >= '1' && e.key <= '9') {
            const sectionIndex = parseInt(e.key) - 1;
            const navItems = document.querySelectorAll('.nav-item[data-section]');
            if (navItems[sectionIndex]) {
                const sectionId = navItems[sectionIndex].getAttribute('data-section');
                this.showSection(sectionId);
            }
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modalManager = this.managers.get('modal');
            if (modalManager) {
                modalManager.closeActiveModal();
            }
        }
    }

    /**
     * Handle clicks outside elements
     */
    handleOutsideClick(e) {
        // Close mobile sidebar if clicking outside
        if (window.innerWidth <= 768) {
            const sidebar = document.querySelector('.sidebar');
            const mobileToggle = document.querySelector('.mobile-toggle');
            
            if (sidebar && sidebar.classList.contains('open') && 
                !sidebar.contains(e.target) && 
                !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    }

    /**
     * Focus search input
     */
    focusSearch() {
        const searchInput = document.getElementById('agentSearch');
        if (searchInput) {
            searchInput.focus();
        }
    }

    /**
     * Show error message
     */
    showError(title, message) {
        console.error(`❌ ${title}:`, message);
        
        // Show toast notification if available
        if (window.showToast) {
            window.showToast(title, message, 'error');
        } else {
            // Fallback to alert
            alert(`${title}: ${message}`);
        }
    }

    /**
     * Show success message
     */
    showSuccess(title, message) {
        console.log(`✅ ${title}:`, message);
        
        // Show toast notification if available
        if (window.showToast) {
            window.showToast(title, message, 'success');
        }
    }

    /**
     * Show info message
     */
    showInfo(title, message) {
        console.log(`ℹ️ ${title}:`, message);
        
        // Show toast notification if available
        if (window.showToast) {
            window.showToast(title, message, 'info');
        }
    }

    /**
     * Emergency shutdown handler
     */
    async emergencyShutdown() {
        const systemManager = this.managers.get('system');
        if (systemManager) {
            await systemManager.emergencyShutdown();
        }
    }

    /**
     * Utility function for debouncing
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Get manager by name
     */
    getManager(name) {
        return this.managers.get(name);
    }

    /**
     * Update state
     */
    setState(newState) {
        this.state = { ...this.state, ...newState };
    }

    /**
     * Get current state
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Cleanup on page unload
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.managers.forEach(manager => {
            if (manager.destroy) {
                manager.destroy();
            }
        });

        this.managers.clear();
        console.log('🧹 Admin Portal cleaned up');
    }
}

// Global functions available immediately for backwards compatibility
window.showSection = function(sectionId) {
    if (window.adminPage) {
        window.adminPage.showSection(sectionId);
    } else {
        console.log(`Queued navigation to: ${sectionId}`);
    }
};

window.toggleSidebar = function() {
    if (window.adminPage) {
        window.adminPage.toggleSidebar();
    } else {
        // Fallback for immediate use
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('open');
        }
    }
};

window.emergencyShutdown = function() {
    if (window.adminPage) {
        window.adminPage.emergencyShutdown();
    } else {
        if (confirm('Emergency shutdown will stop all services immediately. Continue?')) {
            alert('Emergency shutdown initiated - Admin system is loading...');
        }
    }
};

window.filterAgentsByType = function(type) {
    const agentManager = window.adminPage?.getManager('agent');
    if (agentManager) {
        agentManager.filterByType(type);
    }
};

window.filterAgents = function() {
    const agentManager = window.adminPage?.getManager('agent');
    const searchTerm = document.getElementById('agentSearch')?.value || '';
    if (agentManager) {
        agentManager.setSearchTerm(searchTerm);
    }
};

window.openCreateAgentModal = function() {
    const modalManager = window.adminPage?.getManager('modal');
    if (modalManager) {
        modalManager.openCreateAgentModal();
    }
};

window.closeCreateAgentModal = function() {
    const modalManager = window.adminPage?.getManager('modal');
    if (modalManager) {
        modalManager.closeCreateAgentModal();
    }
};

window.closeAgentDetailsModal = function() {
    const modalManager = window.adminPage?.getManager('modal');
    if (modalManager) {
        modalManager.closeAgentDetailsModal();
    }
};

window.createAgent = function(event) {
    const agentManager = window.adminPage?.getManager('agent');
    if (agentManager) {
        agentManager.handleCreateAgent(event);
    }
};

window.deployAllAgents = function() {
    const agentManager = window.adminPage?.getManager('agent');
    if (agentManager) {
        agentManager.deployAllAgents();
    }
};

window.refreshAgents = function() {
    const agentManager = window.adminPage?.getManager('agent');
    if (agentManager) {
        agentManager.refreshAgents();
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('🔧 DOM loaded, initializing Admin Page...');
        window.adminPage = new AdminPage();
        
        // Set up fallback navigation immediately
        document.querySelectorAll('.nav-item[data-section]').forEach(item => {
            item.addEventListener('click', (e) => {
                const sectionId = item.getAttribute('data-section');
                if (window.adminPage) {
                    window.adminPage.showSection(sectionId);
                } else {
                    // Fallback manual section switching
                    document.querySelectorAll('.content-section').forEach(section => {
                        section.classList.remove('active');
                    });
                    document.querySelectorAll('.nav-item').forEach(navItem => {
                        navItem.classList.remove('active');
                    });
                    
                    const targetSection = document.getElementById(sectionId);
                    if (targetSection) {
                        targetSection.classList.add('active');
                        item.classList.add('active');
                    }
                }
            });
        });
        
    } catch (error) {
        console.error('❌ Failed to initialize Admin Page:', error);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.adminPage) {
        window.adminPage.destroy();
    }
});

// Export for module usage
export { AdminPage };