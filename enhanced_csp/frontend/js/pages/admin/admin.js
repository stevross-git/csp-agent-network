// js/pages/admin/admin.js - UPDATED with Monitoring Integration
// Enhanced CSP Admin Portal - Complete Navigation with MonitoringManager

// =============================================================================
// GLOBAL STATE - SIMPLE
// =============================================================================
let CURRENT_SECTION = 'dashboard';
let NAVIGATION_LOCKED = false;
let LAST_NAVIGATION = 0;

// Data storage
const SYSTEM_DATA = {
    agents: [],
    users: [],
    dashboard: {},
    monitoring: {}
};

// Monitoring state
let monitoringManager = null;
let monitoringLoaded = false;

// =============================================================================
// MAIN NAVIGATION FUNCTION - COMPLETELY ISOLATED
// =============================================================================

function showSection(sectionId) {
    const now = Date.now();
    
    // Rate limiting - prevent calls within 500ms
    if (now - LAST_NAVIGATION < 500) {
        console.log(`⏰ Rate limited: ${sectionId} (too fast)`);
        return false;
    }
    
    // Lock check
    if (NAVIGATION_LOCKED) {
        console.log(`🔒 Navigation locked: ${sectionId}`);
        return false;
    }
    
    // Same section check
    if (CURRENT_SECTION === sectionId) {
        console.log(`📍 Already on: ${sectionId}`);
        return false;
    }
    
    // Lock navigation
    NAVIGATION_LOCKED = true;
    LAST_NAVIGATION = now;
    
    console.log(`🔄 NAVIGATING: ${CURRENT_SECTION} → ${sectionId}`);
    
    try {
        // 1. Hide ALL sections
        const allSections = document.querySelectorAll('.content-section');
        allSections.forEach(section => {
            section.classList.remove('active');
        });
        
        // 2. Deactivate ALL nav items
        const allNavItems = document.querySelectorAll('.nav-item');
        allNavItems.forEach(item => {
            item.classList.remove('active');
            item.setAttribute('aria-current', 'false');
        });
        
        // 3. Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            console.log(`✅ Section shown: ${sectionId}`);
        } else {
            console.error(`❌ Section not found: ${sectionId}`);
            NAVIGATION_LOCKED = false;
            return false;
        }
        
        // 4. Activate nav item
        const navItem = document.querySelector(`[data-section="${sectionId}"]`);
        if (navItem) {
            navItem.classList.add('active');
            navItem.setAttribute('aria-current', 'page');
            console.log(`✅ Nav activated: ${sectionId}`);
        }
        
        // 5. Update state
        CURRENT_SECTION = sectionId;
        
        // 6. Section-specific initialization
        initializeSection(sectionId);
        
        console.log(`🎯 Navigation complete: ${sectionId}`);
        
    } catch (error) {
        console.error(`❌ Navigation error:`, error);
    } finally {
        // Always unlock navigation
        setTimeout(() => {
            NAVIGATION_LOCKED = false;
        }, 100);
    }
    
    return true;
}

// =============================================================================
// NAVIGATION EVENT LISTENERS
// =============================================================================

function attachNavigationListeners() {
    console.log('🔧 Attaching Navigation Event Listeners...');
    
    // Get all navigation items with data-section attributes
    const navItems = document.querySelectorAll('.nav-item[data-section]');
    
    console.log(`📍 Found ${navItems.length} navigation items`);
    
    navItems.forEach((item, index) => {
        const sectionId = item.getAttribute('data-section');
        
        // Remove any existing event listeners to avoid duplicates
        const newItem = item.cloneNode(true);
        item.parentNode.replaceChild(newItem, item);
        
        // Add click event listener
        newItem.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log(`🖱️ Navigation clicked: ${sectionId}`);
            showSection(sectionId);
        });
        
        // Add keyboard support
        newItem.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                console.log(`⌨️ Navigation keyboard: ${sectionId}`);
                showSection(sectionId);
            }
        });
        
        console.log(`  ${index + 1}. ✅ Event listeners added to: ${sectionId}`);
    });
    
    console.log('✅ All navigation event listeners attached!');
}

// =============================================================================
// MONITORING MANAGER INTEGRATION
// =============================================================================

/**
 * Load and initialize the monitoring manager
 */
async function loadMonitoringManager() {
    if (monitoringLoaded) {
        console.log('📊 MonitoringManager already loaded');
        return monitoringManager;
    }

    try {
        console.log('🔄 Loading MonitoringManager...');
        
        // Show loading indicator
        showLoadingIndicator('Loading monitoring dashboard...');
        
        // Load the monitoring manager script
        await loadScript('../js/pages/admin/monitoringManager.js');
        
        // Load monitoring components if they exist
        try {
            await loadScript('../js/pages/admin/components/MonitoringComponents.js');
            console.log('✅ Monitoring components loaded');
        } catch (error) {
            console.log('⚠️ Monitoring components not found, using basic implementation');
        }
        
        // Create monitoring manager instance
        const adminPortal = {
            getState: () => ({ currentSection: CURRENT_SECTION }),
            setState: (state) => { /* Implementation */ },
            getManager: (type) => {
                if (type === 'monitoring') return monitoringManager;
                return null;
            }
        };
        
        monitoringManager = new MonitoringManager(adminPortal);
        
        // Initialize the monitoring manager
        await monitoringManager.init();
        
        // Add monitoring manager to global scope
        window.monitoringManager = monitoringManager;
        
        monitoringLoaded = true;
        console.log('✅ MonitoringManager loaded and initialized');
        
        // Hide loading indicator
        hideLoadingIndicator();
        
        return monitoringManager;
        
    } catch (error) {
        console.error('❌ Failed to load MonitoringManager:', error);
        hideLoadingIndicator();
        showErrorMessage('Failed to load monitoring dashboard');
        
        // Fallback to basic monitoring
        console.log('🔄 Falling back to basic monitoring implementation...');
        await createBasicMonitoringDashboard();
        
        return null;
    }
}

/**
 * Initialize monitoring section
 */
async function initializeMonitoring() {
    try {
        console.log('🔄 Initializing monitoring section...');
        
        // Try to load full monitoring manager
        const manager = await loadMonitoringManager();
        
        // Create monitoring dashboard (either full or basic)
        if (manager) {
            await createAdvancedMonitoringDashboard();
            // Start real-time updates
            if (manager.startRealTimeUpdates) {
                manager.startRealTimeUpdates();
            }
        } else {
            await createBasicMonitoringDashboard();
            startBasicMonitoringUpdates();
        }
        
        console.log('✅ Monitoring section initialized');
        
    } catch (error) {
        console.error('❌ Failed to initialize monitoring:', error);
        await createBasicMonitoringDashboard();
    }
}

/**
 * Create basic monitoring dashboard (fallback)
 */
async function createBasicMonitoringDashboard() {
    const monitoringSection = document.getElementById('monitoring');
    if (!monitoringSection) return;

    monitoringSection.innerHTML = `
        <div class="monitoring-dashboard">
            <div class="monitoring-header">
                <h2 class="section-title">
                    <i class="fas fa-chart-line"></i> System Monitoring
                </h2>
                <div class="monitoring-controls">
                    <button class="btn btn-outline" onclick="refreshBasicMonitoring()">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                    <button class="btn btn-outline" onclick="exportBasicMonitoringData()">
                        <i class="fas fa-download"></i> Export
                    </button>
                </div>
            </div>

            <div class="status-overview">
                <div class="status-card">
                    <div class="status-indicator status-operational">
                        <i class="fas fa-check-circle"></i> Operational
                    </div>
                    <div class="status-details">
                        <h3>System Status</h3>
                        <p>All systems running normally</p>
                        <small>Last updated: <span id="basic-last-update">just now</span></small>
                    </div>
                </div>
            </div>

            <!-- Real-time Metrics -->
            <div class="metrics-grid">
                <div class="metric-category">
                    <h3><i class="fas fa-server"></i> System Resources</h3>
                    <div class="metric-cards">
                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-microchip"></i>
                                <span>CPU Usage</span>
                            </div>
                            <div id="basic-cpu" class="metric-value">45%</div>
                            <div class="progress-bar">
                                <div id="basic-cpu-bar" class="progress-fill" style="width: 45%; background: #27ae60;"></div>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-memory"></i>
                                <span>Memory Usage</span>
                            </div>
                            <div id="basic-memory" class="metric-value">62%</div>
                            <div class="progress-bar">
                                <div id="basic-memory-bar" class="progress-fill" style="width: 62%; background: #f39c12;"></div>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-hdd"></i>
                                <span>Disk Usage</span>
                            </div>
                            <div id="basic-disk" class="metric-value">78%</div>
                            <div class="progress-bar">
                                <div id="basic-disk-bar" class="progress-fill" style="width: 78%; background: #e74c3c;"></div>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-network-wired"></i>
                                <span>Network I/O</span>
                            </div>
                            <div id="basic-network" class="metric-value">1.2 GB/s</div>
                            <div class="metric-subtitle">Live data stream</div>
                        </div>
                    </div>
                </div>

                <div class="metric-category">
                    <h3><i class="fas fa-tachometer-alt"></i> Performance</h3>
                    <div class="metric-cards">
                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-clock"></i>
                                <span>Response Time</span>
                            </div>
                            <div id="basic-response" class="metric-value">245ms</div>
                            <div class="trend-indicator trend-good">
                                <i class="fas fa-arrow-down"></i>
                                <span>-12ms from avg</span>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-tachometer-alt"></i>
                                <span>Throughput</span>
                            </div>
                            <div id="basic-throughput" class="metric-value">2,347</div>
                            <div class="metric-subtitle">requests/minute</div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                <span>Error Rate</span>
                            </div>
                            <div id="basic-errors" class="metric-value">0.12%</div>
                            <div class="trend-indicator trend-good">
                                <i class="fas fa-arrow-down"></i>
                                <span>Improving</span>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <i class="fas fa-shield-alt"></i>
                                <span>Uptime</span>
                            </div>
                            <div id="basic-uptime" class="metric-value">99.97%</div>
                            <div class="metric-subtitle">Last 30 days</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Live Activity Feed -->
            <div class="activity-feed">
                <div class="panel-header">
                    <h3><i class="fas fa-activity"></i> Live Activity</h3>
                    <span class="realtime-indicator">
                        <div class="realtime-dot"></div>
                        <span>Live</span>
                    </span>
                </div>
                <div id="activity-log" class="activity-container">
                    <!-- Activity items will be added here -->
                </div>
            </div>
        </div>
    `;

    console.log('✅ Basic monitoring dashboard created');
}

/**
 * Start basic monitoring updates
 */
function startBasicMonitoringUpdates() {
    if (window.basicMonitoringInterval) {
        clearInterval(window.basicMonitoringInterval);
    }

    window.basicMonitoringInterval = setInterval(() => {
        updateBasicMetrics();
        addActivityLogEntry();
    }, 3000);

    console.log('✅ Basic monitoring updates started');
}

/**
 * Update basic metrics with simulated data
 */
function updateBasicMetrics() {
    // CPU Usage
    const cpu = Math.max(20, Math.min(85, 45 + (Math.random() - 0.5) * 20));
    updateElement('basic-cpu', `${cpu.toFixed(1)}%`);
    updateProgressBar('basic-cpu-bar', cpu);

    // Memory Usage
    const memory = Math.max(30, Math.min(90, 62 + (Math.random() - 0.5) * 15));
    updateElement('basic-memory', `${memory.toFixed(1)}%`);
    updateProgressBar('basic-memory-bar', memory);

    // Disk Usage
    const disk = Math.max(70, Math.min(95, 78 + Math.random() * 0.1));
    updateElement('basic-disk', `${disk.toFixed(1)}%`);
    updateProgressBar('basic-disk-bar', disk);

    // Network I/O
    const network = Math.max(0.5, Math.min(2.0, 1.2 + (Math.random() - 0.5) * 0.5));
    updateElement('basic-network', `${network.toFixed(1)} GB/s`);

    // Response Time
    const response = Math.max(100, Math.min(500, 245 + (Math.random() - 0.5) * 100));
    updateElement('basic-response', `${response.toFixed(0)}ms`);

    // Throughput
    const throughput = Math.max(1000, Math.min(5000, 2347 + (Math.random() - 0.5) * 500));
    updateElement('basic-throughput', throughput.toLocaleString());

    // Update timestamp
    updateElement('basic-last-update', 'just now');
}

/**
 * Add activity log entry
 */
function addActivityLogEntry() {
    const activities = [
        { icon: 'fa-check-circle', message: 'System health check completed', type: 'success' },
        { icon: 'fa-sync-alt', message: 'Database backup process started', type: 'info' },
        { icon: 'fa-user', message: 'New user authentication successful', type: 'info' },
        { icon: 'fa-exclamation-triangle', message: 'High memory usage detected', type: 'warning' },
        { icon: 'fa-shield-alt', message: 'Security scan completed successfully', type: 'success' },
        { icon: 'fa-cog', message: 'System configuration updated', type: 'info' }
    ];

    const activity = activities[Math.floor(Math.random() * activities.length)];
    const logContainer = document.getElementById('activity-log');
    
    if (logContainer) {
        const entry = document.createElement('div');
        entry.className = `activity-entry ${activity.type}`;
        entry.innerHTML = `
            <div class="activity-icon">
                <i class="fas ${activity.icon}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-message">${activity.message}</div>
                <div class="activity-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        logContainer.insertBefore(entry, logContainer.firstChild);
        
        // Keep only last 10 entries
        while (logContainer.children.length > 10) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }
}

// =============================================================================
// SECTION INITIALIZATION (UPDATED)
// =============================================================================

async function initializeSection(sectionId) {
    console.log(`🚀 Initializing section: ${sectionId}`);
    
    switch(sectionId) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'agents':
            loadAgentData();
            break;
        case 'users':
            loadUserData();
            break;
        case 'monitoring':
            await initializeMonitoring();
            break;
        default:
            console.log(`📄 Section ${sectionId} - basic initialization`);
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function loadScript(src) {
    return new Promise((resolve, reject) => {
        // Check if script is already loaded
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }

        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
        document.head.appendChild(script);
    });
}

function showLoadingIndicator(message = 'Loading...') {
    const indicator = document.createElement('div');
    indicator.id = 'monitoring-loading';
    indicator.className = 'loading-overlay';
    indicator.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        color: white;
        font-family: Inter, sans-serif;
    `;
    indicator.innerHTML = `
        <div style="text-align: center;">
            <div style="border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px;"></div>
            <p style="margin: 0; font-size: 16px;">${message}</p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    `;
    document.body.appendChild(indicator);
}

function hideLoadingIndicator() {
    const indicator = document.getElementById('monitoring-loading');
    if (indicator) {
        indicator.remove();
    }
}

function showErrorMessage(message) {
    console.error(message);
    // Simple alert for now - you can implement toast notifications
    alert(`Error: ${message}`);
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateProgressBar(id, percentage) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = `${percentage}%`;
        
        // Update color based on usage
        if (percentage > 85) {
            bar.style.backgroundColor = '#e74c3c'; // Red
        } else if (percentage > 70) {
            bar.style.backgroundColor = '#f39c12'; // Orange
        } else {
            bar.style.backgroundColor = '#27ae60'; // Green
        }
    }
}

/**
 * Basic monitoring action functions
 */
function refreshBasicMonitoring() {
    console.log('🔄 Refreshing basic monitoring...');
    updateBasicMetrics();
    addActivityLogEntry();
}

function exportBasicMonitoringData() {
    console.log('📊 Exporting basic monitoring data...');
    const data = {
        timestamp: new Date().toISOString(),
        metrics: {
            cpu: document.getElementById('basic-cpu')?.textContent,
            memory: document.getElementById('basic-memory')?.textContent,
            disk: document.getElementById('basic-disk')?.textContent,
            network: document.getElementById('basic-network')?.textContent,
            response: document.getElementById('basic-response')?.textContent,
            throughput: document.getElementById('basic-throughput')?.textContent
        }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `basic-monitoring-${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// =============================================================================
// EXISTING FUNCTIONS (keeping all your existing code)
// =============================================================================

function loadDashboardData() {
    console.log('📊 Loading dashboard data...');
    SYSTEM_DATA.dashboard = {
        systemHealth: '98.5%',
        activeUsers: 1247,
        aiAgents: '22/24',
        throughput: '2.3M',
        storage: '78%',
        alerts: 3
    };
    updateDashboardUI();
}

function loadAgentData() {
    console.log('🤖 Loading agent data...');
    SYSTEM_DATA.agents = [
        { id: 1, name: 'DataAnalyzer-v2', type: 'analysis', status: 'active' },
        { id: 2, name: 'SecurityMonitor', type: 'security', status: 'active' },
        { id: 3, name: 'BackupAgent', type: 'backup', status: 'inactive' }
    ];
    updateAgentUI();
}

function loadUserData() {
    console.log('👥 Loading user data...');
    SYSTEM_DATA.users = [
        { id: 1, name: 'John Doe', email: 'john.doe@example.com', role: 'Administrator', status: 'active' },
        { id: 2, name: 'Jane Smith', email: 'jane.smith@example.com', role: 'Operator', status: 'active' }
    ];
    updateUserUI();
}

function updateDashboardUI() {
    const data = SYSTEM_DATA.dashboard;
    const elements = {
        'system-health': data.systemHealth,
        'active-users': data.activeUsers,
        'ai-agents': data.aiAgents,
        'throughput': data.throughput,
        'storage': data.storage,
        'alerts': data.alerts
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

function updateAgentUI() {
    const agentGrid = document.getElementById('agent-grid');
    if (!agentGrid) return;
    
    agentGrid.innerHTML = SYSTEM_DATA.agents.map(agent => `
        <div class="agent-card ${agent.status}">
            <h3>${agent.name}</h3>
            <p>Type: ${agent.type}</p>
            <p>Status: <span class="status-badge ${agent.status}">${agent.status}</span></p>
            <div class="agent-actions">
                <button class="btn btn-sm btn-outline" onclick="viewAgent(${agent.id})">View</button>
                <button class="btn btn-sm btn-primary" onclick="toggleAgent(${agent.id})">${agent.status === 'active' ? 'Stop' : 'Start'}</button>
            </div>
        </div>
    `).join('');
}

function updateUserUI() {
    console.log('👥 User UI updated with', SYSTEM_DATA.users.length, 'users');
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
        console.log('📱 Sidebar toggled');
    }
}

function toggleAgent(agentId) {
    const agent = SYSTEM_DATA.agents.find(a => a.id === agentId);
    if (agent) {
        agent.status = agent.status === 'active' ? 'inactive' : 'active';
        updateAgentUI();
        console.log(`🤖 Agent ${agentId} toggled to ${agent.status}`);
    }
}

function viewAgent(agentId) {
    console.log(`👁️ Viewing agent ${agentId}`);
    alert(`Viewing agent ${agentId} details`);
}

function createAgent() {
    console.log('➕ Creating new agent');
    const newAgent = {
        id: SYSTEM_DATA.agents.length + 1,
        name: `Agent-${Date.now()}`,
        type: 'custom',
        status: 'inactive'
    };
    SYSTEM_DATA.agents.push(newAgent);
    updateAgentUI();
    closeModal('create-agent-modal');
}

function deployAllAgents() {
    console.log('🚀 Deploying all agents');
    SYSTEM_DATA.agents.forEach(agent => {
        agent.status = 'active';
    });
    updateAgentUI();
    alert('All agents deployed successfully');
}

function refreshAgents() {
    console.log('🔄 Refreshing agents');
    loadAgentData();
}

function editUser(userId) {
    console.log(`✏️ Editing user ${userId}`);
}

function deleteUser(userId) {
    console.log(`🗑️ Deleting user ${userId}`);
    if (confirm('Are you sure you want to delete this user?')) {
        SYSTEM_DATA.users = SYSTEM_DATA.users.filter(u => u.id !== userId);
        updateUserUI();
    }
}

function emergencyShutdown() {
    console.log('🚨 EMERGENCY SHUTDOWN INITIATED');
    if (confirm('Are you sure you want to perform an emergency shutdown?')) {
        SYSTEM_DATA.agents.forEach(agent => {
            agent.status = 'inactive';
        });
        updateAgentUI();
        alert('Emergency shutdown completed');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

function openCreateAgentModal() {
    const modal = document.getElementById('create-agent-modal');
    if (modal) {
        modal.style.display = 'block';
    }
}

function startAllAgents() { deployAllAgents(); }
function exportSystemEvents() { console.log('📤 Exporting system events'); }
function refreshSystemEvents() { console.log('🔄 Refreshing system events'); }
function filterAgentsByType(type) { console.log(`🔍 Filter by type: ${type}`); }
function filterAgents() { console.log('🔍 Filter agents'); }
function toggleChart(chartId) { console.log(`📊 Toggle chart: ${chartId}`); }
function downloadChart(chartId) { console.log(`💾 Download chart: ${chartId}`); }
function acknowledgeAlert(alertId) { console.log(`✅ Acknowledge alert: ${alertId}`); }
function dismissAlert(alertId) { console.log(`❌ Dismiss alert: ${alertId}`); }
function viewAlertDetails(alertId) { console.log(`👁️ View alert: ${alertId}`); }
function exportLogs() { console.log('📤 Export logs'); }

// =============================================================================
// GLOBAL EXPORTS
// =============================================================================

window.showSection = showSection;
window.toggleSidebar = toggleSidebar;
window.attachNavigationListeners = attachNavigationListeners;
window.loadMonitoringManager = loadMonitoringManager;
window.initializeMonitoring = initializeMonitoring;
window.refreshBasicMonitoring = refreshBasicMonitoring;
window.exportBasicMonitoringData = exportBasicMonitoringData;
window.toggleAgent = toggleAgent;
window.viewAgent = viewAgent;
window.createAgent = createAgent;
window.deployAllAgents = deployAllAgents;
window.startAllAgents = startAllAgents;
window.refreshAgents = refreshAgents;
window.editUser = editUser;
window.deleteUser = deleteUser;
window.closeModal = closeModal;
window.openCreateAgentModal = openCreateAgentModal;
window.emergencyShutdown = emergencyShutdown;
window.exportSystemEvents = exportSystemEvents;
window.refreshSystemEvents = refreshSystemEvents;
window.filterAgentsByType = filterAgentsByType;
window.filterAgents = filterAgents;
window.toggleChart = toggleChart;
window.downloadChart = downloadChart;
window.acknowledgeAlert = acknowledgeAlert;
window.dismissAlert = dismissAlert;
window.viewAlertDetails = viewAlertDetails;
window.exportLogs = exportLogs;

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Admin Portal - Nuclear Fix with Monitoring Integration Loading...');
    
    // CRITICAL: Attach navigation event listeners first
    setTimeout(() => {
        attachNavigationListeners();
    }, 100);
    
    // Load initial data
    loadDashboardData();
    loadAgentData();
    loadUserData();
    
    // Start periodic updates for dashboard
    setInterval(() => {
        if (CURRENT_SECTION === 'dashboard') loadDashboardData();
    }, 30000);
    
    console.log('✅ Admin Portal Ready with Monitoring Integration');
});

console.log('📄 Nuclear Fix Admin.js with Monitoring Integration Loaded');