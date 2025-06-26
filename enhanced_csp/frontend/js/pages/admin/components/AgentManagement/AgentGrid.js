// js/pages/admin/components/AgentManagement/AgentGrid.js
class AgentGrid extends BaseComponent {
    constructor(containerId) {
        super(containerId);
        this.state = {
            agents: [],
            loading: false,
            filter: 'all',
            searchTerm: ''
        };
        this.agentCards = new Map();
    }

    async loadDependencies() {
        // Import AgentCard component
        if (!window.AgentCard) {
            await this.loadScript('../components/AgentManagement/AgentCard.js');
        }
    }

    render() {
        if (this.state.loading) {
            this.container.innerHTML = '<div class="loading-spinner">Loading agents...</div>';
            return;
        }

        const filteredAgents = this.getFilteredAgents();
        
        if (!filteredAgents.length) {
            this.renderEmptyState();
            return;
        }

        // Clear existing cards
        this.agentCards.clear();
        this.container.innerHTML = '';
        
        // Create grid container
        const grid = document.createElement('div');
        grid.className = 'agents-grid';
        this.container.appendChild(grid);
        
        // Create agent cards
        filteredAgents.forEach(agent => {
            const cardContainer = document.createElement('div');
            grid.appendChild(cardContainer);
            
            const agentCard = new AgentCard(cardContainer, agent);
            this.agentCards.set(agent.id, agentCard);
            
            // Listen to card events
            agentCard.on('agent:toggle', (data) => this.emit('agent:toggle', data));
            agentCard.on('agent:details', (data) => this.emit('agent:details', data));
            agentCard.on('agent:logs', (data) => this.emit('agent:logs', data));
            agentCard.on('agent:duplicate', (data) => this.emit('agent:duplicate', data));
            
            agentCard.init();
        });
    }

    renderEmptyState() {
        this.container.innerHTML = `
            <div class="empty-state">
                <h3>🔍 No agents found</h3>
                <p>Try adjusting your filters or create a new agent.</p>
                <button class="btn btn-primary create-agent-btn">➕ Create Agent</button>
            </div>
        `;
        
        this.container.querySelector('.create-agent-btn')
            .addEventListener('click', () => this.emit('agent:create'));
    }

    getFilteredAgents() {
        return this.state.agents.filter(agent => {
            // Type filter
            if (this.state.filter !== 'all' && agent.type !== this.state.filter) {
                return false;
            }
            
            // Search filter
            if (this.state.searchTerm) {
                const term = this.state.searchTerm.toLowerCase();
                return agent.name.toLowerCase().includes(term) ||
                       agent.description.toLowerCase().includes(term) ||
                       agent.id.toLowerCase().includes(term);
            }
            
            return true;
        });
    }

    setAgents(agents) {
        this.setState({ agents });
    }

    setFilter(filter) {
        this.setState({ filter });
    }

    setSearchTerm(searchTerm) {
        this.setState({ searchTerm });
    }

    setLoading(loading) {
        this.setState({ loading });
    }

    updateAgent(agentId, updatedData) {
        const agents = this.state.agents.map(agent => 
            agent.id === agentId ? { ...agent, ...updatedData } : agent
        );
        this.setState({ agents });
        
        // Update the specific card
        const card = this.agentCards.get(agentId);
        if (card) {
            card.updateAgent(agents.find(a => a.id === agentId));
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
}