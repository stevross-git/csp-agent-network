// js/pages/admin/components/AgentManagement/AgentCard.js
class AgentCard extends BaseComponent {
    constructor(containerId, agentData) {
        super(containerId, { autoInit: false });
        this.agentData = agentData;
        this.state = { agent: agentData };
    }

    render() {
        const { agent } = this.state;
        this.container.innerHTML = `
            <div class="agent-card" data-agent-id="${agent.id}">
                <div class="priority-badge priority-${agent.priority}">${agent.priority}</div>
                
                <div class="agent-header">
                    <div class="agent-info">
                        <h4>${agent.name}</h4>
                        <span class="agent-type ${agent.type}">
                            ${this.getAgentTypeIcon(agent.type)} ${agent.type}
                        </span>
                    </div>
                </div>
                
                <div class="agent-status">
                    <span class="status-indicator ${agent.status}"></span>
                    <span>${agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                    <span class="agent-model">${agent.model}</span>
                </div>
                
                <div class="agent-description">${agent.description}</div>
                
                <div class="agent-metrics">
                    <div class="metric">
                        <div class="metric-value">${agent.tasksCompleted}</div>
                        <div class="metric-label">Tasks</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${agent.uptime}</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                </div>
                
                <div class="agent-last-activity">
                    Last activity: ${agent.lastActivity}
                </div>
                
                <div class="agent-actions">
                    <button class="btn btn-success toggle-btn" title="${agent.status === 'active' ? 'Pause' : 'Start'} agent">
                        ${agent.status === 'active' ? '⏸️' : '▶️'}
                    </button>
                    <button class="btn btn-primary details-btn" title="View details">👁️</button>
                    <button class="btn btn-secondary logs-btn" title="View logs">📄</button>
                    <button class="btn btn-warning duplicate-btn" title="Duplicate">📋</button>
                </div>
            </div>
        `;
    }

    bindEvents() {
        const card = this.container.querySelector('.agent-card');
        
        card.querySelector('.toggle-btn').addEventListener('click', () => {
            this.emit('agent:toggle', { agentId: this.state.agent.id });
        });
        
        card.querySelector('.details-btn').addEventListener('click', () => {
            this.emit('agent:details', { agentId: this.state.agent.id });
        });
        
        card.querySelector('.logs-btn').addEventListener('click', () => {
            this.emit('agent:logs', { agentId: this.state.agent.id });
        });
        
        card.querySelector('.duplicate-btn').addEventListener('click', () => {
            this.emit('agent:duplicate', { agentId: this.state.agent.id });
        });
    }

    updateAgent(newAgentData) {
        this.setState({ agent: newAgentData });
    }

    getAgentTypeIcon(type) {
        const icons = {
            'autonomous': '🧠',
            'collaborative': '🤝', 
            'specialized': '⚡',
            'monitoring': '👁️'
        };
        return icons[type] || '🤖';
    }
}