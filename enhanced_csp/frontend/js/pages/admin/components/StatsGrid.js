// js/pages/admin/components/Dashboard/StatsGrid.js (Fixed Implementation)
class StatsGrid extends BaseComponent {
    constructor(containerId) {
        super(containerId);
        this.state = {
            stats: {
                totalAgents: 0,
                activeAgents: 0,
                totalExecutions: 0,
                successRate: 0,
                averageResponseTime: 0,
                systemUptime: 0
            },
            loading: false,
            autoRefresh: true,
            refreshInterval: 30000 // 30 seconds
        };
        
        this.refreshTimer = null;
        this.mockData = window.apiFallbackData.generateStatsGridData();
    }
    
    render() {
        if (this.state.loading) {
            this.container.innerHTML = '<div class="loading-spinner">Loading statistics...</div>';
            return;
        }
        
        const { stats } = this.state;
        
        this.container.innerHTML = `
            <div class="stats-grid">
                <div class="stats-header">
                    <h3>📊 System Statistics</h3>
                    <div class="stats-controls">
                        <button class="btn btn-sm btn-secondary refresh-btn" title="Refresh Stats">
                            🔄
                        </button>
                        <button class="btn btn-sm ${this.state.autoRefresh ? 'btn-success' : 'btn-outline-secondary'} auto-refresh-btn" title="Toggle Auto Refresh">
                            ${this.state.autoRefresh ? '⏸️' : '▶️'}
                        </button>
                    </div>
                </div>
                
                <div class="stats-cards">
                    <div class="stat-card">
                        <div class="stat-icon">🤖</div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.totalAgents}</div>
                            <div class="stat-label">Total Agents</div>
                            <div class="stat-change positive">↗️ +${Math.floor(Math.random() * 5) + 1}</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">⚡</div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.activeAgents}</div>
                            <div class="stat-label">Active Agents</div>
                            <div class="stat-change ${stats.activeAgents > stats.totalAgents * 0.7 ? 'positive' : 'neutral'}">
                                ${stats.activeAgents > stats.totalAgents * 0.7 ? '↗️' : '→'} 
                                ${((stats.activeAgents / stats.totalAgents) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">📈</div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.totalExecutions.toLocaleString()}</div>
                            <div class="stat-label">Total Executions</div>
                            <div class="stat-change positive">↗️ +${Math.floor(Math.random() * 100) + 50}</div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">✅</div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.successRate.toFixed(1)}%</div>
                            <div class="stat-label">Success Rate</div>
                            <div class="stat-change ${stats.successRate > 95 ? 'positive' : stats.successRate > 85 ? 'neutral' : 'negative'}">
                                ${stats.successRate > 95 ? '↗️' : stats.successRate > 85 ? '→' : '↘️'} 
                                ${stats.successRate > 90 ? 'Excellent' : stats.successRate > 80 ? 'Good' : 'Needs Attention'}
                            </div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">⏱️</div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.averageResponseTime.toFixed(0)}ms</div>
                            <div class="stat-label">Avg Response Time</div>
                            <div class="stat-change ${stats.averageResponseTime < 300 ? 'positive' : stats.averageResponseTime < 500 ? 'neutral' : 'negative'}">
                                ${stats.averageResponseTime < 300 ? '↗️ Fast' : stats.averageResponseTime < 500 ? '→ Normal' : '↘️ Slow'}
                            </div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-icon">⏰</div>
                        <div class="stat-content">
                            <div class="stat-value">${this.formatUptime(stats.systemUptime)}</div>
                            <div class="stat-label">System Uptime</div>
                            <div class="stat-change positive">↗️ Running</div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-footer">
                    <div class="last-updated">
                        Last updated: ${new Date().toLocaleTimeString()}
                    </div>
                    <div class="refresh-status">
                        ${this.state.autoRefresh ? `Auto-refresh: ${this.state.refreshInterval / 1000}s` : 'Auto-refresh: Off'}
                    </div>
                </div>
            </div>
        `;
        
        // Add some basic styling if not already present
        this.addStyles();
    }
    
    bindEvents() {
        const refreshBtn = this.find('.refresh-btn');
        const autoRefreshBtn = this.find('.auto-refresh-btn');
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshStats());
        }
        
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => this.toggleAutoRefresh());
        }
    }
    
    async refreshStats() {
        try {
            this.setState({ loading: true });
            
            // Simulate API call delay
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Update mock data
            this.mockData = window.apiFallbackData.generateStatsGridData();
            
            this.setState({
                stats: this.mockData,
                loading: false
            });
            
            // Emit event for parent components
            this.emit('stats:updated', this.mockData);
            
        } catch (error) {
            console.error('Failed to refresh stats:', error);
            this.emit('stats:error', error);
        }
    }
    
    toggleAutoRefresh() {
        const newAutoRefresh = !this.state.autoRefresh;
        this.setState({ autoRefresh: newAutoRefresh });
        
        if (newAutoRefresh) {
            this.startAutoRefresh();
        } else {
            this.stopAutoRefresh();
        }
        
        this.emit('stats:autoRefreshToggled', { autoRefresh: newAutoRefresh });
    }
    
    startAutoRefresh() {
        this.stopAutoRefresh(); // Clear any existing timer
        
        if (this.state.autoRefresh) {
            this.refreshTimer = setInterval(() => {
                this.refreshStats();
            }, this.state.refreshInterval);
        }
    }
    
    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }
    
    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.render();
        this.bindEvents();
    }
    
    formatUptime(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = Math.floor(minutes % 60);
        
        if (hours > 24) {
            const days = Math.floor(hours / 24);
            const remainingHours = hours % 24;
            return `${days}d ${remainingHours}h`;
        } else {
            return `${hours}h ${mins}m`;
        }
    }
    
    addStyles() {
        if (!document.getElementById('stats-grid-styles')) {
            const style = document.createElement('style');
            style.id = 'stats-grid-styles';
            style.textContent = `
                .stats-grid {
                    background: white;
                    border-radius: 8px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .stats-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid #eee;
                }
                
                .stats-controls {
                    display: flex;
                    gap: 0.5rem;
                }
                
                .stats-cards {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                }
                
                .stat-card {
                    background: #f8f9fa;
                    border-radius: 6px;
                    padding: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }
                
                .stat-icon {
                    font-size: 2rem;
                }
                
                .stat-content {
                    flex: 1;
                }
                
                .stat-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #333;
                }
                
                .stat-label {
                    color: #666;
                    font-size: 0.9rem;
                    margin: 0.25rem 0;
                }
                
                .stat-change {
                    font-size: 0.8rem;
                    font-weight: 500;
                }
                
                .stat-change.positive { color: #28a745; }
                .stat-change.negative { color: #dc3545; }
                .stat-change.neutral { color: #6c757d; }
                
                .stats-footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.85rem;
                    color: #666;
                    padding-top: 1rem;
                    border-top: 1px solid #eee;
                }
                
                .loading-spinner {
                    text-align: center;
                    padding: 2rem;
                    color: #666;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    onReady() {
        super.onReady();
        
        // Initial data load
        this.setState({ stats: this.mockData });
        
        // Start auto-refresh if enabled
        if (this.state.autoRefresh) {
            this.startAutoRefresh();
        }
    }
    
    destroy() {
        this.stopAutoRefresh();
        super.destroy();
    }
}