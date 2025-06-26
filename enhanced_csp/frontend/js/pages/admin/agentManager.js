/**
 * Agent Manager - Handles AI agent operations and management
 * Part of Enhanced CSP Admin Portal
 */

export class AgentManager {
    constructor(adminPage) {
        this.adminPage = adminPage;
        this.agents = new Map();
        this.selectedAgents = new Set();
        this.currentFilter = 'all';
        this.searchTerm = '';
    }

    /**
     * Initialize agent manager
     */
    async init() {
        try {
            console.log('🤖 Initializing Agent Manager...');
            
            // Load initial agent data
            await this.loadAgents();
            
            // Set up agent grid
            this.setupAgentGrid();
            
            // Set up event listeners
            this.setupEventListeners();
            
            console.log('✅ Agent Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Agent Manager:', error);
            throw error;
        }
    }

    /**
     * Load agents data
     */
    async loadAgents() {
        try {
            // Simulate API call - replace with actual endpoint
            const sampleAgents = [
                {
                    id: 'agent-001',
                    name: 'Security Monitor',
                    type: 'security',
                    status: 'active',
                    description: 'Monitors system security and detects threats',
                    priority: 'high',
                    created: new Date('2024-01-15'),
                    lastActivity: new Date(),
                    cpuUsage: 15,
                    memoryUsage: 32,
                    tasksCompleted: 1247,
                    uptime: '5d 12h 30m',
                    autoRestart: true,
                    logging: true
                },
                {
                    id: 'agent-002',
                    name: 'Performance Optimizer',
                    type: 'performance',
                    status: 'active',
                    description: 'Optimizes system performance and resource usage',
                    priority: 'medium',
                    created: new Date('2024-02-01'),
                    lastActivity: new Date(Date.now() - 300000),
                    cpuUsage: 8,
                    memoryUsage: 28,
                    tasksCompleted: 892,
                    uptime: '3d 8h 15m',
                    autoRestart: true,
                    logging: false
                },
                {
                    id: 'agent-003',
                    name: 'Backup Coordinator',
                    type: 'backup',
                    status: 'inactive',
                    description: 'Manages automated backup processes',
                    priority: 'low',
                    created: new Date('2024-01-20'),
                    lastActivity: new Date(Date.now() - 3600000),
                    cpuUsage: 0,
                    memoryUsage: 0,
                    tasksCompleted: 156,
                    uptime: '0h 0m',
                    autoRestart: false,
                    logging: true
                },
                {
                    id: 'agent-004',
                    name: 'Health Monitor',
                    type: 'monitoring',
                    status: 'active',
                    description: 'Monitors system health and performance metrics',
                    priority: 'high',
                    created: new Date('2024-01-10'),
                    lastActivity: new Date(Date.now() - 60000),
                    cpuUsage: 12,
                    memoryUsage: 24,
                    tasksCompleted: 2341,
                    uptime: '7d 14h 22m',
                    autoRestart: true,
                    logging: true
                }
            ];

            // Store agents
            sampleAgents.forEach(agent => {
                this.agents.set(agent.id, agent);
            });

        } catch (error) {
            console.error('❌ Failed to load agents:', error);
            throw error;
        }
    }

    /**
     * Set up agent grid
     */
    setupAgentGrid() {
        const agentGrid = document.getElementById('agent-grid');
        if (!agentGrid) return;

        this.renderAgents();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Agent search
        const searchInput = document.getElementById('agentSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.setSearchTerm(e.target.value);
            });
        }

        // Agent type filter
        const typeFilter = document.querySelector('.filter-dropdown');
        if (typeFilter) {
            typeFilter.addEventListener('change', (e) => {
                this.filterByType(e.target.value);
            });
        }
    }

    /**
     * Render agents in the grid
     */
    renderAgents() {
        const agentGrid = document.getElementById('agent-grid');
        if (!agentGrid) return;

        // Get filtered agents
        const filteredAgents = this.getFilteredAgents();

        if (filteredAgents.length === 0) {
            agentGrid.innerHTML = `
                <div class="no-agents">
                    <i class="fas fa-robot" style="font-size: 3rem; color: var(--text-muted); margin-bottom: 1rem;"></i>
                    <p>No agents found matching your criteria</p>
                    <button class="btn btn-primary" onclick="openCreateAgentModal()">
                        <i class="fas fa-plus"></i>
                        Create First Agent
                    </button>
                </div>
            `;
            return;
        }

        agentGrid.innerHTML = filteredAgents.map(agent => this.createAgentCard(agent)).join('');
    }

    /**
     * Create agent card HTML
     */
    createAgentCard(agent) {
        const statusClass = agent.status === 'active' ? 'online' : 'offline';
        const priorityClass = `priority-${agent.priority}`;

        return `
            <div class="agent-card" data-agent-id="${agent.id}">
                <div class="agent-header">
                    <div class="agent-info">
                        <h4 class="agent-name">${agent.name}</h4>
                        <span class="agent-type">${agent.type}</span>
                    </div>
                    <span class="status-badge ${statusClass}">
                        ${agent.status}
                    </span>
                </div>
                
                <div class="agent-description">
                    <p>${agent.description}</p>
                </div>
                
                <div class="agent-stats">
                    <div class="stat-item">
                        <label>CPU:</label>
                        <span>${agent.cpuUsage}%</span>
                    </div>
                    <div class="stat-item">
                        <label>Memory:</label>
                        <span>${agent.memoryUsage}%</span>
                    </div>
                    <div class="stat-item">
                        <label>Tasks:</label>
                        <span>${agent.tasksCompleted}</span>
                    </div>
                    <div class="stat-item">
                        <label>Priority:</label>
                        <span class="${priorityClass}">${agent.priority}</span>
                    </div>
                </div>
                
                <div class="agent-actions">
                    <button class="btn btn-outline btn-sm" onclick="window.adminPage.getManager('agent').viewAgent('${agent.id}')" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-primary btn-sm" onclick="window.adminPage.getManager('agent').toggleAgent('${agent.id}')" title="${agent.status === 'active' ? 'Stop' : 'Start'} Agent">
                        <i class="fas fa-${agent.status === 'active' ? 'stop' : 'play'}"></i>
                    </button>
                    <button class="btn btn-outline btn-sm" onclick="window.adminPage.getManager('agent').editAgent('${agent.id}')" title="Edit Agent">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="window.adminPage.getManager('agent').deleteAgent('${agent.id}')" title="Delete Agent">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Get filtered agents
     */
    getFilteredAgents() {
        let filtered = Array.from(this.agents.values());

        // Apply type filter
        if (this.currentFilter && this.currentFilter !== 'all' && this.currentFilter !== '') {
            filtered = filtered.filter(agent => agent.type === this.currentFilter);
        }

        // Apply search filter
        if (this.searchTerm) {
            const searchLower = this.searchTerm.toLowerCase();
            filtered = filtered.filter(agent => 
                agent.name.toLowerCase().includes(searchLower) ||
                agent.description.toLowerCase().includes(searchLower) ||
                agent.type.toLowerCase().includes(searchLower)
            );
        }

        return filtered;
    }

    /**
     * Filter agents by type
     */
    filterByType(type) {
        this.currentFilter = type;
        this.renderAgents();
    }

    /**
     * Set search term
     */
    setSearchTerm(term) {
        this.searchTerm = term;
        this.renderAgents();
    }

    /**
     * View agent details
     */
    viewAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return;

        const modalManager = this.adminPage.getManager('modal');
        if (modalManager) {
            modalManager.showAgentDetails(agent);
        }
    }

    /**
     * Toggle agent status
     */
    async toggleAgent(agentId) {
        try {
            const agent = this.agents.get(agentId);
            if (!agent) return;

            const newStatus = agent.status === 'active' ? 'inactive' : 'active';
            const action = newStatus === 'active' ? 'Starting' : 'Stopping';
            
            console.log(`${action} agent: ${agent.name}`);

            // Show loading state
            const agentCard = document.querySelector(`[data-agent-id="${agentId}"]`);
            if (agentCard) {
                agentCard.classList.add('loading');
            }

            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Update agent status
            agent.status = newStatus;
            agent.lastActivity = new Date();

            if (newStatus === 'active') {
                agent.cpuUsage = Math.floor(Math.random() * 20) + 5;
                agent.memoryUsage = Math.floor(Math.random() * 30) + 10;
            } else {
                agent.cpuUsage = 0;
                agent.memoryUsage = 0;
                agent.uptime = '0h 0m';
            }

            // Update UI
            this.renderAgents();

            this.adminPage.showSuccess(
                `Agent ${newStatus === 'active' ? 'Started' : 'Stopped'}`,
                `${agent.name} is now ${newStatus}`
            );

        } catch (error) {
            console.error(`❌ Failed to toggle agent ${agentId}:`, error);
            this.adminPage.showError('Agent Toggle Failed', error.message);
        }
    }

    /**
     * Edit agent
     */
    editAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return;

        console.log(`🖊️ Editing agent: ${agent.name}`);
        // TODO: Implement edit modal
        this.adminPage.showInfo('Edit Agent', 'Edit functionality coming soon!');
    }

    /**
     * Delete agent
     */
    async deleteAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return;

        const modalManager = this.adminPage.getManager('modal');
        if (modalManager) {
            modalManager.showConfirmation(
                'Delete Agent',
                `Are you sure you want to delete "${agent.name}"? This action cannot be undone.`,
                async () => {
                    try {
                        console.log(`🗑️ Deleting agent: ${agent.name}`);

                        // Simulate API call
                        await new Promise(resolve => setTimeout(resolve, 500));

                        // Remove from agents map
                        this.agents.delete(agentId);

                        // Update UI
                        this.renderAgents();

                        this.adminPage.showSuccess('Agent Deleted', `${agent.name} has been deleted`);

                    } catch (error) {
                        console.error(`❌ Failed to delete agent ${agentId}:`, error);
                        this.adminPage.showError('Delete Failed', error.message);
                    }
                }
            );
        }
    }

    /**
     * Add new agent
     */
    addAgent(agentData) {
        const newAgent = {
            id: `agent-${Date.now()}`,
            ...agentData,
            status: agentData.autoStart ? 'active' : 'inactive',
            created: new Date(),
            lastActivity: new Date(),
            cpuUsage: agentData.autoStart ? Math.floor(Math.random() * 10) + 2 : 0,
            memoryUsage: agentData.autoStart ? Math.floor(Math.random() * 15) + 5 : 0,
            tasksCompleted: 0,
            uptime: agentData.autoStart ? '0h 1m' : '0h 0m',
            autoRestart: false,
            logging: false
        };

        this.agents.set(newAgent.id, newAgent);
        this.renderAgents();

        console.log(`✅ Added new agent: ${newAgent.name}`);
    }

    /**
     * Deploy all agents
     */
    async deployAllAgents() {
        try {
            console.log('🚀 Deploying all agents...');

            const inactiveAgents = Array.from(this.agents.values()).filter(agent => agent.status === 'inactive');
            
            if (inactiveAgents.length === 0) {
                this.adminPage.showInfo('Deploy All', 'All agents are already active');
                return;
            }

            // Show loading state
            const deployBtn = document.querySelector('button[onclick="deployAllAgents()"]');
            if (deployBtn) {
                const originalText = deployBtn.innerHTML;
                deployBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deploying...';
                deployBtn.disabled = true;

                // Deploy each inactive agent
                for (const agent of inactiveAgents) {
                    await this.toggleAgent(agent.id);
                    await new Promise(resolve => setTimeout(resolve, 500)); // Stagger deployments
                }

                // Restore button
                deployBtn.innerHTML = originalText;
                deployBtn.disabled = false;
            }

            this.adminPage.showSuccess('Deploy Complete', `Deployed ${inactiveAgents.length} agents`);

        } catch (error) {
            console.error('❌ Failed to deploy all agents:', error);
            this.adminPage.showError('Deploy Failed', error.message);
        }
    }

    /**
     * Refresh agents
     */
    async refreshAgents() {
        try {
            console.log('🔄 Refreshing agents...');

            // Show loading state
            const refreshBtn = document.querySelector('button[onclick="refreshAgents()"]');
            if (refreshBtn) {
                const originalText = refreshBtn.innerHTML;
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
                refreshBtn.disabled = true;

                // Simulate refresh
                await new Promise(resolve => setTimeout(resolve, 1000));

                // Update agent stats
                this.agents.forEach(agent => {
                    if (agent.status === 'active') {
                        agent.lastActivity = new Date();
                        agent.cpuUsage = Math.floor(Math.random() * 20) + 5;
                        agent.memoryUsage = Math.floor(Math.random() * 30) + 10;
                        agent.tasksCompleted += Math.floor(Math.random() * 10) + 1;
                    }
                });

                // Re-render
                this.renderAgents();

                // Restore button
                refreshBtn.innerHTML = originalText;
                refreshBtn.disabled = false;
            }

            this.adminPage.showSuccess('Refresh Complete', 'Agent data updated');

        } catch (error) {
            console.error('❌ Failed to refresh agents:', error);
            this.adminPage.showError('Refresh Failed', error.message);
        }
    }

    /**
     * Handle create agent
     */
    handleCreateAgent(event) {
        // This is handled by modalManager, but we process the result
        console.log('🤖 Processing new agent creation...');
    }

    /**
     * Get agent statistics
     */
    getAgentStats() {
        const agents = Array.from(this.agents.values());
        return {
            total: agents.length,
            active: agents.filter(a => a.status === 'active').length,
            inactive: agents.filter(a => a.status === 'inactive').length,
            byType: agents.reduce((acc, agent) => {
                acc[agent.type] = (acc[agent.type] || 0) + 1;
                return acc;
            }, {}),
            avgCpuUsage: agents.reduce((sum, a) => sum + a.cpuUsage, 0) / agents.length,
            avgMemoryUsage: agents.reduce((sum, a) => sum + a.memoryUsage, 0) / agents.length
        };
    }

    /**
     * Handle section change
     */
    onSectionChange(sectionId) {
        if (sectionId === 'agents') {
            // Refresh agent data when section becomes active
            this.renderAgents();
        }
    }

    /**
     * Update real-time data
     */
    updateRealTimeData() {
        // Update active agent statistics
        this.agents.forEach(agent => {
            if (agent.status === 'active') {
                // Small random variations in usage
                const cpuVariation = (Math.random() - 0.5) * 4;
                const memVariation = (Math.random() - 0.5) * 6;
                
                agent.cpuUsage = Math.max(1, Math.min(100, agent.cpuUsage + cpuVariation));
                agent.memoryUsage = Math.max(1, Math.min(100, agent.memoryUsage + memVariation));
                agent.lastActivity = new Date();
            }
        });

        // Update UI if agents section is active
        if (this.adminPage.getState().currentSection === 'agents') {
            this.renderAgents();
        }
    }

    /**
     * Export agents data
     */
    exportAgents() {
        const data = Array.from(this.agents.values());
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `agents-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.adminPage.showSuccess('Export Complete', 'Agent data exported successfully');
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        this.agents.clear();
        this.selectedAgents.clear();
        console.log('🧹 Agent Manager cleaned up');
    }
}

// Add agent-specific CSS
const agentStyles = document.createElement('style');
agentStyles.textContent = `
    .no-agents {
        grid-column: 1 / -1;
        text-align: center;
        padding: 3rem;
        color: var(--text-muted);
    }

    .agent-card {
        position: relative;
        transition: var(--transition);
    }

    .agent-card.loading {
        opacity: 0.6;
        pointer-events: none;
    }

    .agent-card.loading::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 20px;
        height: 20px;
        border: 2px solid var(--border-color);
        border-top-color: var(--primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    .agent-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 1rem 0;
        font-size: 0.85rem;
    }

    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
    }

    .stat-item label {
        color: var(--text-muted);
        font-weight: 500;
    }

    .priority-low { color: var(--success); }
    .priority-medium { color: var(--warning); }
    .priority-high { color: var(--danger); }
    .priority-critical { 
        color: var(--danger); 
        font-weight: bold;
        text-transform: uppercase;
    }

    .agent-actions {
        display: flex;
        gap: 0.25rem;
        justify-content: center;
    }

    .btn-sm {
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        min-width: auto;
    }

    .search-filters {
        margin-bottom: 1.5rem;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }
`;

document.head.appendChild(agentStyles);