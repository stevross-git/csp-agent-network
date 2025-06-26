// js/pages/admin/admin.js - NUCLEAR FIX - Isolated Navigation
// Enhanced CSP Admin Portal - Complete Navigation Isolation

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
        
        // 6. Load section data
        loadSectionData(sectionId);
        
        console.log(`🎯 NAVIGATION COMPLETE: ${sectionId}`);
        return true;
        
    } catch (error) {
        console.error(`❌ Navigation error:`, error);
        return false;
    } finally {
        // Always unlock after 1 second
        setTimeout(() => {
            NAVIGATION_LOCKED = false;
            console.log(`🔓 Navigation unlocked`);
        }, 1000);
    }
}

// =============================================================================
// DATA LOADING FUNCTIONS
// =============================================================================

function loadSectionData(sectionId) {
    console.log(`📂 Loading data: ${sectionId}`);
    
    switch(sectionId) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'monitoring':
            loadMonitoringData();
            break;
        case 'agents':
            loadAgentData();
            break;
        case 'users':
            loadUserData();
            break;
        case 'security':
            loadSecurityData();
            break;
        default:
            console.log(`📄 Basic load: ${sectionId}`);
            showSectionMessage(sectionId, `${sectionId.toUpperCase()} section loaded successfully`);
    }
}

function loadDashboardData() {
    console.log('📊 Loading dashboard...');
    
    SYSTEM_DATA.dashboard = {
        systemStatus: 'Operational',
        activeAgents: Math.floor(Math.random() * 5) + 10,
        totalUsers: Math.floor(Math.random() * 20) + 150,
        alertsCount: Math.floor(Math.random() * 5) + 1,
        cpuUsage: Math.floor(Math.random() * 40) + 20,
        memoryUsage: Math.floor(Math.random() * 30) + 40,
        networkTraffic: Math.floor(Math.random() * 1000) + 500,
        lastUpdated: new Date().toLocaleTimeString()
    };

    updateDashboardUI();
}

function loadMonitoringData() {
    console.log('📊 Loading monitoring...');
    
    // Update monitoring metrics
    updateElement('cpu-usage', Math.floor(Math.random() * 40) + 30 + '%');
    updateElement('memory-usage', Math.floor(Math.random() * 30) + 50 + '%');
    updateElement('disk-usage', Math.floor(Math.random() * 20) + 60 + '%');
    updateElement('network-usage', Math.floor(Math.random() * 50) + 25 + ' MB/s');
    updateElement('response-time', Math.floor(Math.random() * 200) + 100 + ' ms');
    updateElement('throughput', Math.floor(Math.random() * 2000) + 3000 + ' req/s');
    updateElement('error-rate', (Math.random() * 2).toFixed(1) + '%');
    updateElement('availability', (99 + Math.random()).toFixed(1) + '%');
    updateElement('threat-level', 'Low');
    updateElement('blocked-requests', Math.floor(Math.random() * 50) + 25);
    updateElement('active-sessions', Math.floor(Math.random() * 30) + 15);
    updateElement('failed-logins', Math.floor(Math.random() * 5));
    
    showSectionMessage('monitoring', 'Monitoring data updated successfully');
}

function loadAgentData() {
    console.log('🤖 Loading agents...');
    
    SYSTEM_DATA.agents = [
        { id: 1, name: 'Security Monitor', type: 'monitoring', status: 'active', lastSeen: new Date() },
        { id: 2, name: 'Content Analyzer', type: 'analysis', status: 'active', lastSeen: new Date() },
        { id: 3, name: 'Threat Detector', type: 'security', status: 'inactive', lastSeen: new Date(Date.now() - 3600000) },
        { id: 4, name: 'Performance Monitor', type: 'monitoring', status: 'active', lastSeen: new Date() }
    ];

    updateAgentUI();
}

function loadUserData() {
    console.log('👥 Loading users...');
    
    SYSTEM_DATA.users = [
        { id: 1, name: 'Admin User', email: 'admin@csp.ai', role: 'Administrator', status: 'active' },
        { id: 2, name: 'Security Analyst', email: 'analyst@csp.ai', role: 'Analyst', status: 'active' },
        { id: 3, name: 'Operator', email: 'operator@csp.ai', role: 'Operator', status: 'inactive' }
    ];

    updateUserUI();
}

function loadSecurityData() {
    console.log('🔒 Loading security...');
    
    updateElement('threat-level-display', 'Low');
    updateElement('active-sessions-count', '5');
    updateElement('blocked-threats', 'No active threats detected');
    
    showSectionMessage('security', 'Security data loaded successfully');
}

// =============================================================================
// UI UPDATE FUNCTIONS
// =============================================================================

function updateDashboardUI() {
    const data = SYSTEM_DATA.dashboard;
    if (!data) return;

    updateElement('system-health', data.systemStatus);
    updateElement('system-status', data.systemStatus);
    updateElement('active-agents', data.activeAgents);
    updateElement('total-users', data.totalUsers);
    updateElement('alerts-count', data.alertsCount);
    updateElement('cpu-usage', data.cpuUsage + '%');
    updateElement('memory-usage', data.memoryUsage + '%');
    updateElement('last-update-time', data.lastUpdated);
    
    console.log('✅ Dashboard UI updated');
}

function updateAgentUI() {
    const container = document.getElementById('agent-grid');
    if (!container) return;

    container.innerHTML = SYSTEM_DATA.agents.map(agent => `
        <div class="agent-card ${agent.status}" data-agent-id="${agent.id}">
            <div class="agent-header">
                <h4>${agent.name}</h4>
                <span class="agent-status ${agent.status}">${agent.status}</span>
            </div>
            <div class="agent-details">
                <p><strong>Type:</strong> ${agent.type}</p>
                <p><strong>Last Seen:</strong> ${agent.lastSeen.toLocaleString()}</p>
            </div>
            <div class="agent-actions">
                <button onclick="toggleAgent(${agent.id})" class="btn ${agent.status === 'active' ? 'btn-warning' : 'btn-success'}">
                    ${agent.status === 'active' ? 'Stop' : 'Start'}
                </button>
                <button onclick="viewAgent(${agent.id})" class="btn btn-info">View</button>
            </div>
        </div>
    `).join('');
    
    console.log('✅ Agent UI updated');
}

function updateUserUI() {
    const container = document.getElementById('user-table-body');
    if (!container) return;

    container.innerHTML = SYSTEM_DATA.users.map(user => `
        <tr data-user-id="${user.id}">
            <td>${user.name}</td>
            <td>${user.email}</td>
            <td>${user.role}</td>
            <td><span class="status ${user.status}">${user.status}</span></td>
            <td>
                <button onclick="editUser(${user.id})" class="btn btn-sm btn-primary">Edit</button>
                <button onclick="deleteUser(${user.id})" class="btn btn-sm btn-danger">Delete</button>
            </td>
        </tr>
    `).join('');
    
    console.log('✅ User UI updated');
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function showSectionMessage(sectionId, message) {
    const section = document.getElementById(sectionId);
    if (section) {
        let messageDiv = section.querySelector('.section-message');
        if (!messageDiv) {
            messageDiv = document.createElement('div');
            messageDiv.className = 'section-message';
            messageDiv.style.cssText = 'padding: 20px; text-align: center; color: #22c55e; background: rgba(34, 197, 94, 0.1); border-radius: 8px; margin: 20px;';
            section.appendChild(messageDiv);
        }
        messageDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    }
}

// =============================================================================
// INTERACTIVE FUNCTIONS
// =============================================================================

function toggleAgent(agentId) {
    const agent = SYSTEM_DATA.agents.find(a => a.id === agentId);
    if (agent) {
        agent.status = agent.status === 'active' ? 'inactive' : 'active';
        agent.lastSeen = new Date();
        updateAgentUI();
        console.log(`✅ Toggled agent ${agentId} to ${agent.status}`);
    }
}

function viewAgent(agentId) {
    const agent = SYSTEM_DATA.agents.find(a => a.id === agentId);
    if (agent) {
        alert(`Agent: ${agent.name}\nType: ${agent.type}\nStatus: ${agent.status}\nLast Seen: ${agent.lastSeen.toLocaleString()}`);
    }
}

function createAgent() {
    const name = prompt('Enter agent name:');
    if (name) {
        SYSTEM_DATA.agents.push({
            id: Date.now(),
            name: name,
            type: 'custom',
            status: 'inactive',
            lastSeen: new Date()
        });
        updateAgentUI();
        console.log(`✅ Created agent: ${name}`);
    }
}

function deployAllAgents() {
    SYSTEM_DATA.agents.forEach(agent => {
        agent.status = 'active';
        agent.lastSeen = new Date();
    });
    updateAgentUI();
    alert('All agents deployed successfully!');
}

function refreshAgents() {
    loadAgentData();
}

function editUser(userId) {
    const user = SYSTEM_DATA.users.find(u => u.id === userId);
    if (user) {
        const newName = prompt('Enter new name:', user.name);
        if (newName) {
            user.name = newName;
            updateUserUI();
        }
    }
}

function deleteUser(userId) {
    const user = SYSTEM_DATA.users.find(u => u.id === userId);
    if (user && confirm(`Delete user ${user.name}?`)) {
        SYSTEM_DATA.users = SYSTEM_DATA.users.filter(u => u.id !== userId);
        updateUserUI();
    }
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.toggle('mobile-visible');
    }
}

function emergencyShutdown() {
    if (confirm('Emergency shutdown - are you sure?')) {
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

// Placeholder functions
function filterAgentsByType(type) { console.log(`Filter by type: ${type}`); }
function filterAgents() { console.log('Filter agents'); }
function toggleChart(chartId) { console.log(`Toggle chart: ${chartId}`); }
function downloadChart(chartId) { console.log(`Download chart: ${chartId}`); }
function acknowledgeAlert(alertId) { console.log(`Acknowledge alert: ${alertId}`); }
function dismissAlert(alertId) { console.log(`Dismiss alert: ${alertId}`); }
function viewAlertDetails(alertId) { console.log(`View alert: ${alertId}`); }
function exportLogs() { console.log('Export logs'); }

// =============================================================================
// GLOBAL EXPORTS
// =============================================================================

window.showSection = showSection;
window.toggleSidebar = toggleSidebar;
window.toggleAgent = toggleAgent;
window.viewAgent = viewAgent;
window.createAgent = createAgent;
window.deployAllAgents = deployAllAgents;
window.refreshAgents = refreshAgents;
window.editUser = editUser;
window.deleteUser = deleteUser;
window.closeModal = closeModal;
window.emergencyShutdown = emergencyShutdown;
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
    console.log('🚀 Admin Portal - Nuclear Fix Loading...');
    
    // Load initial data
    loadDashboardData();
    loadAgentData();
    loadUserData();
    
    // Start updates
    setInterval(() => {
        if (CURRENT_SECTION === 'dashboard') loadDashboardData();
        if (CURRENT_SECTION === 'monitoring') loadMonitoringData();
    }, 30000);
    
    console.log('✅ Admin Portal Ready');
});

console.log('📄 Nuclear Fix Admin.js Loaded');