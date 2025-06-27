/**
 * Modal Manager - Handles all modal dialogs and popups
 * Part of Enhanced CSP Admin Portal
 */

class ModalManager {
    constructor(adminPage) {
        this.adminPage = adminPage;
        this.activeModals = new Set();
        this.modalContainer = null;
    }

    /**
     * Initialize modal manager
     */
    async init() {
        try {
            console.log('🪟 Initializing Modal Manager...');
            
            // Create modal container if it doesn't exist
            this.createModalContainer();
            
            // Set up event listeners
            this.setupEventListeners();
            
            console.log('✅ Modal Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Modal Manager:', error);
            throw error;
        }
    }

    /**
     * Create modal container
     */
    createModalContainer() {
        this.modalContainer = document.getElementById('modal-container');
        
        if (!this.modalContainer) {
            this.modalContainer = document.createElement('div');
            this.modalContainer.id = 'modal-container';
            this.modalContainer.className = 'modal-container';
            document.body.appendChild(this.modalContainer);
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.activeModals.size > 0) {
                this.closeTopModal();
            }
        });

        // Handle click outside modal
        this.modalContainer.addEventListener('click', (e) => {
            if (e.target === this.modalContainer || e.target.classList.contains('modal')) {
                this.closeTopModal();
            }
        });
    }

    /**
     * Create a modal element
     */
    createModal(options = {}) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-labelledby', `modal-title-${Date.now()}`);
        modal.setAttribute('aria-hidden', 'true');

        const {
            title = 'Modal',
            content = '',
            size = 'medium',
            closable = true,
            footer = null,
            className = ''
        } = options;

        const modalId = `modal-${Date.now()}`;
        modal.id = modalId;

        modal.innerHTML = `
            <div class="modal-content modal-${size} ${className}">
                <div class="modal-header">
                    <h2 class="modal-title" id="modal-title-${Date.now()}">${title}</h2>
                    ${closable ? '<button class="modal-close" aria-label="Close modal">&times;</button>' : ''}
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                ${footer ? `<div class="modal-footer">${footer}</div>` : ''}
            </div>
        `;

        // Add close button functionality
        if (closable) {
            const closeBtn = modal.querySelector('.modal-close');
            closeBtn.addEventListener('click', () => this.closeModal(modalId));
        }

        return { modal, modalId };
    }

    /**
     * Show modal
     */
    showModal(options = {}) {
        const { modal, modalId } = this.createModal(options);
        
        this.modalContainer.appendChild(modal);
        this.activeModals.add(modalId);

        // Trigger show animation
        requestAnimationFrame(() => {
            modal.classList.add('active');
            modal.setAttribute('aria-hidden', 'false');
        });

        // Focus management
        this.trapFocus(modal);

        return modalId;
    }

    /**
     * Close modal by ID
     */
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;

        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
        
        // Remove from DOM after animation
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
            this.activeModals.delete(modalId);
        }, 300);

        // Restore focus
        this.restoreFocus();
    }

    /**
     * Close top modal
     */
    closeTopModal() {
        if (this.activeModals.size > 0) {
            const lastModal = Array.from(this.activeModals).pop();
            this.closeModal(lastModal);
        }
    }

    /**
     * Close all modals
     */
    closeAllModals() {
        Array.from(this.activeModals).forEach(modalId => {
            this.closeModal(modalId);
        });
    }

    /**
     * Close active modal (backward compatibility)
     */
    closeActiveModal() {
        this.closeTopModal();
    }

    /**
     * Show create agent modal
     */
    openCreateAgentModal() {
        const content = `
            <form id="create-agent-form" class="form">
                <div class="form-group">
                    <label class="form-label" for="agent-name">Agent Name</label>
                    <input type="text" id="agent-name" class="form-input" required>
                </div>
                <div class="form-group">
                    <label class="form-label" for="agent-type">Agent Type</label>
                    <select id="agent-type" class="form-select" required>
                        <option value="">Select Type</option>
                        <option value="monitoring">Monitoring Agent</option>
                        <option value="security">Security Agent</option>
                        <option value="performance">Performance Agent</option>
                        <option value="backup">Backup Agent</option>
                        <option value="custom">Custom Agent</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label" for="agent-description">Description</label>
                    <textarea id="agent-description" class="form-input" rows="3" placeholder="Enter agent description..."></textarea>
                </div>
                <div class="form-group">
                    <label class="form-label" for="agent-priority">Priority Level</label>
                    <select id="agent-priority" class="form-select">
                        <option value="low">Low</option>
                        <option value="medium" selected>Medium</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">
                        <input type="checkbox" id="agent-auto-start" checked>
                        Auto-start agent after creation
                    </label>
                </div>
            </form>
        `;

        const footer = `
            <button type="button" class="btn btn-outline" onclick="window.adminPage.getManager('modal').closeActiveModal()">
                Cancel
            </button>
            <button type="submit" form="create-agent-form" class="btn btn-primary">
                <i class="fas fa-plus"></i>
                Create Agent
            </button>
        `;

        const modalId = this.showModal({
            title: '<i class="fas fa-robot"></i> Create New AI Agent',
            content,
            footer,
            size: 'medium'
        });

        // Add form submission handling
        const form = document.getElementById('create-agent-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleCreateAgent(e);
        });

        return modalId;
    }

    /**
     * Handle create agent form submission
     */
    async handleCreateAgent(event) {
        event.preventDefault();
        
        try {
            const formData = new FormData(event.target);
            const agentData = {
                name: document.getElementById('agent-name').value,
                type: document.getElementById('agent-type').value,
                description: document.getElementById('agent-description').value,
                priority: document.getElementById('agent-priority').value,
                autoStart: document.getElementById('agent-auto-start').checked
            };

            // Validate required fields
            if (!agentData.name || !agentData.type) {
                throw new Error('Name and Type are required fields');
            }

            // Show loading state
            const submitBtn = event.target.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
            submitBtn.disabled = true;

            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1500));

            // Close modal
            this.closeActiveModal();

            // Notify agent manager
            const agentManager = this.adminPage.getManager('agent');
            if (agentManager) {
                agentManager.addAgent(agentData);
            }

            this.adminPage.showSuccess('Agent Created', `${agentData.name} has been created successfully`);

        } catch (error) {
            console.error('❌ Failed to create agent:', error);
            this.adminPage.showError('Creation Failed', error.message);
            
            // Reset button state
            const submitBtn = event.target.querySelector('button[type="submit"]');
            submitBtn.innerHTML = '<i class="fas fa-plus"></i> Create Agent';
            submitBtn.disabled = false;
        }
    }

    /**
     * Close create agent modal
     */
    closeCreateAgentModal() {
        this.closeActiveModal();
    }

    /**
     * Show agent details modal
     */
    showAgentDetails(agent) {
        const content = `
            <div class="agent-details">
                <div class="agent-header">
                    <div class="agent-info">
                        <h3>${agent.name}</h3>
                        <span class="agent-type">${agent.type}</span>
                        <span class="status-badge ${agent.status === 'active' ? 'online' : 'offline'}">
                            ${agent.status}
                        </span>
                    </div>
                    <div class="agent-actions">
                        <button class="btn btn-primary" onclick="window.adminPage.getManager('agent').toggleAgent('${agent.id}')">
                            <i class="fas fa-${agent.status === 'active' ? 'stop' : 'play'}"></i>
                            ${agent.status === 'active' ? 'Stop' : 'Start'}
                        </button>
                        <button class="btn btn-outline" onclick="window.adminPage.getManager('agent').editAgent('${agent.id}')">
                            <i class="fas fa-edit"></i>
                            Edit
                        </button>
                        <button class="btn btn-danger" onclick="window.adminPage.getManager('agent').deleteAgent('${agent.id}')">
                            <i class="fas fa-trash"></i>
                            Delete
                        </button>
                    </div>
                </div>
                
                <div class="agent-tabs">
                    <div class="tab-nav">
                        <button class="tab-btn active" data-tab="overview">Overview</button>
                        <button class="tab-btn" data-tab="performance">Performance</button>
                        <button class="tab-btn" data-tab="logs">Logs</button>
                        <button class="tab-btn" data-tab="settings">Settings</button>
                    </div>
                    
                    <div class="tab-content">
                        <div class="tab-pane active" id="overview">
                            <div class="info-grid">
                                <div class="info-item">
                                    <label>Description:</label>
                                    <span>${agent.description || 'No description provided'}</span>
                                </div>
                                <div class="info-item">
                                    <label>Priority:</label>
                                    <span class="priority-${agent.priority}">${agent.priority}</span>
                                </div>
                                <div class="info-item">
                                    <label>Created:</label>
                                    <span>${new Date(agent.created).toLocaleString()}</span>
                                </div>
                                <div class="info-item">
                                    <label>Last Activity:</label>
                                    <span>${new Date(agent.lastActivity).toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane" id="performance">
                            <div class="performance-metrics">
                                <div class="metric-item">
                                    <label>CPU Usage:</label>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${agent.cpuUsage}%"></div>
                                    </div>
                                    <span>${agent.cpuUsage}%</span>
                                </div>
                                <div class="metric-item">
                                    <label>Memory Usage:</label>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${agent.memoryUsage}%"></div>
                                    </div>
                                    <span>${agent.memoryUsage}%</span>
                                </div>
                                <div class="metric-item">
                                    <label>Tasks Completed:</label>
                                    <span>${agent.tasksCompleted || 0}</span>
                                </div>
                                <div class="metric-item">
                                    <label>Uptime:</label>
                                    <span>${agent.uptime || '0h 0m'}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane" id="logs">
                            <div class="log-viewer">
                                <div class="log-controls">
                                    <button class="btn btn-outline" onclick="this.closest('.log-viewer').querySelector('.log-content').innerHTML = ''">
                                        Clear Logs
                                    </button>
                                    <button class="btn btn-outline">
                                        <i class="fas fa-download"></i>
                                        Download
                                    </button>
                                </div>
                                <div class="log-content">
                                    <div class="log-entry">
                                        <span class="log-time">${new Date().toLocaleTimeString()}</span>
                                        <span class="log-level info">INFO</span>
                                        <span class="log-message">Agent started successfully</span>
                                    </div>
                                    <div class="log-entry">
                                        <span class="log-time">${new Date(Date.now() - 60000).toLocaleTimeString()}</span>
                                        <span class="log-level success">SUCCESS</span>
                                        <span class="log-message">Task completed: System health check</span>
                                    </div>
                                    <div class="log-entry">
                                        <span class="log-time">${new Date(Date.now() - 120000).toLocaleTimeString()}</span>
                                        <span class="log-level warning">WARN</span>
                                        <span class="log-message">High CPU usage detected</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane" id="settings">
                            <div class="settings-form">
                                <div class="form-group">
                                    <label class="form-label">Agent Name</label>
                                    <input type="text" class="form-input" value="${agent.name}">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Priority Level</label>
                                    <select class="form-select">
                                        <option value="low" ${agent.priority === 'low' ? 'selected' : ''}>Low</option>
                                        <option value="medium" ${agent.priority === 'medium' ? 'selected' : ''}>Medium</option>
                                        <option value="high" ${agent.priority === 'high' ? 'selected' : ''}>High</option>
                                        <option value="critical" ${agent.priority === 'critical' ? 'selected' : ''}>Critical</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">
                                        <input type="checkbox" ${agent.autoRestart ? 'checked' : ''}>
                                        Auto-restart on failure
                                    </label>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">
                                        <input type="checkbox" ${agent.logging ? 'checked' : ''}>
                                        Enable detailed logging
                                    </label>
                                </div>
                                <button class="btn btn-primary">
                                    <i class="fas fa-save"></i>
                                    Save Changes
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const modalId = this.showModal({
            title: `<i class="fas fa-robot"></i> Agent Details - ${agent.name}`,
            content,
            size: 'large',
            className: 'agent-details-modal'
        });

        // Set up tab functionality
        setTimeout(() => {
            this.setupTabs(modalId);
        }, 100);

        return modalId;
    }

    /**
     * Close agent details modal
     */
    closeAgentDetailsModal() {
        this.closeActiveModal();
    }

    /**
     * Show metric details modal
     */
    showMetricDetails(key, metric) {
        const content = `
            <div class="metric-details">
                <div class="metric-overview">
                    <div class="metric-value ${this.getStatusClass(metric.status)}">
                        ${metric.value}${metric.unit || ''}
                    </div>
                    <div class="metric-info">
                        <h4>${metric.label}</h4>
                        <span class="status-badge ${this.getStatusClass(metric.status)}">
                            ${metric.status}
                        </span>
                    </div>
                </div>
                
                <div class="metric-chart">
                    <h5>Trend (Last 24 Hours)</h5>
                    <div class="chart-placeholder">
                        <canvas id="metric-chart-${key}" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="metric-stats">
                    <div class="stat-item">
                        <label>Current Trend:</label>
                        <span class="trend-indicator ${this.getTrendClass(metric.trend)}">
                            ${metric.trend}
                        </span>
                    </div>
                    <div class="stat-item">
                        <label>Last Updated:</label>
                        <span>${new Date().toLocaleString()}</span>
                    </div>
                    <div class="stat-item">
                        <label>Data Points:</label>
                        <span>144 (10min intervals)</span>
                    </div>
                </div>
                
                <div class="metric-actions">
                    <button class="btn btn-primary" onclick="window.adminPage.getManager('dashboard').refreshMetric('${key}')">
                        <i class="fas fa-refresh"></i>
                        Refresh Data
                    </button>
                    <button class="btn btn-outline" onclick="window.adminPage.getManager('dashboard').exportMetric('${key}')">
                        <i class="fas fa-download"></i>
                        Export Data
                    </button>
                    <button class="btn btn-outline" onclick="window.adminPage.getManager('modal').setupMetricAlert('${key}')">
                        <i class="fas fa-bell"></i>
                        Set Alert
                    </button>
                </div>
            </div>
        `;

        const modalId = this.showModal({
            title: `<i class="fas fa-chart-line"></i> ${metric.label} Details`,
            content,
            size: 'large',
            className: 'metric-details-modal'
        });

        // Draw sample chart
        setTimeout(() => {
            this.drawMetricChart(key);
        }, 100);

        return modalId;
    }

    /**
     * Set up tab functionality
     */
    setupTabs(modalId) {
        const modal = document.getElementById(modalId);
        if (!modal) return;

        const tabButtons = modal.querySelectorAll('.tab-btn');
        const tabPanes = modal.querySelectorAll('.tab-pane');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');

                // Remove active class from all buttons and panes
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanes.forEach(pane => pane.classList.remove('active'));

                // Add active class to clicked button and corresponding pane
                button.classList.add('active');
                const targetPane = modal.querySelector(`#${targetTab}`);
                if (targetPane) {
                    targetPane.classList.add('active');
                }
            });
        });
    }

    /**
     * Draw metric chart
     */
    drawMetricChart(key) {
        const canvas = document.getElementById(`metric-chart-${key}`);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Generate sample data
        const dataPoints = 24;
        const data = [];
        for (let i = 0; i < dataPoints; i++) {
            data.push(Math.random() * 100 + 50);
        }

        // Draw chart
        ctx.strokeStyle = '#ff6b35';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
            const x = (i / (data.length - 1)) * width;
            const y = height - (data[i] / 150) * height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Add grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    /**
     * Get status class
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
     * Get trend class
     */
    getTrendClass(trend) {
        if (trend.includes('+') || trend.includes('up')) return 'trend-up';
        if (trend.includes('-') || trend.includes('down')) return 'trend-down';
        return 'trend-stable';
    }

    /**
     * Trap focus within modal
     */
    trapFocus(modal) {
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        firstElement.focus();

        modal.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                if (e.shiftKey) {
                    if (document.activeElement === firstElement) {
                        e.preventDefault();
                        lastElement.focus();
                    }
                } else {
                    if (document.activeElement === lastElement) {
                        e.preventDefault();
                        firstElement.focus();
                    }
                }
            }
        });
    }

    /**
     * Restore focus
     */
    restoreFocus() {
        // Focus management after modal close
        const lastFocusedElement = document.querySelector('[data-last-focused]');
        if (lastFocusedElement) {
            lastFocusedElement.focus();
            lastFocusedElement.removeAttribute('data-last-focused');
        }
    }

    /**
     * Show confirmation dialog
     */
    showConfirmation(title, message, callback) {
        const content = `
            <div class="confirmation-dialog">
                <p>${message}</p>
            </div>
        `;

        const footer = `
            <button type="button" class="btn btn-outline" onclick="window.adminPage.getManager('modal').closeActiveModal()">
                Cancel
            </button>
            <button type="button" class="btn btn-danger" onclick="window.adminPage.getManager('modal').confirmAction()">
                Confirm
            </button>
        `;

        const modalId = this.showModal({
            title,
            content,
            footer,
            size: 'small',
            className: 'confirmation-modal'
        });

        // Store callback for confirmation
        this.confirmationCallback = callback;

        return modalId;
    }

    /**
     * Confirm action
     */
    confirmAction() {
        if (this.confirmationCallback) {
            this.confirmationCallback();
            this.confirmationCallback = null;
        }
        this.closeActiveModal();
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        this.closeAllModals();
        this.activeModals.clear();
        
        if (this.modalContainer && this.modalContainer.parentNode) {
            this.modalContainer.parentNode.removeChild(this.modalContainer);
        }
        
        console.log('🧹 Modal Manager cleaned up');
    }
}

// Add modal-specific CSS
const modalStyles = document.createElement('style');
modalStyles.textContent = `
    .modal-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 1000;
        pointer-events: none;
    }

    .modal {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }

    .modal.active {
        opacity: 1;
        pointer-events: all;
    }

    .modal-small { max-width: 400px; }
    .modal-medium { max-width: 600px; }
    .modal-large { max-width: 800px; }
    .modal-xl { max-width: 1200px; }

    .agent-details .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }

    .agent-details .agent-info h3 {
        margin: 0 0 0.5rem 0;
        color: var(--primary);
    }

    .agent-details .agent-actions {
        display: flex;
        gap: 0.5rem;
    }

    .tab-nav {
        display: flex;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .tab-btn {
        background: none;
        border: none;
        padding: 0.75rem 1rem;
        color: var(--text-muted);
        cursor: pointer;
        transition: var(--transition);
        border-bottom: 2px solid transparent;
    }

    .tab-btn.active,
    .tab-btn:hover {
        color: var(--primary);
        border-bottom-color: var(--primary);
    }

    .tab-pane {
        display: none;
    }

    .tab-pane.active {
        display: block;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .info-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .info-item label {
        font-weight: 600;
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .performance-metrics {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .metric-item {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .metric-item label {
        min-width: 120px;
        font-weight: 600;
    }

    .progress-bar {
        flex: 1;
        height: 8px;
        background: var(--bg-tertiary);
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: var(--primary);
        transition: width 0.3s ease;
    }

    .log-viewer {
        height: 300px;
        display: flex;
        flex-direction: column;
    }

    .log-controls {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .log-content {
        flex: 1;
        background: var(--bg-tertiary);
        border-radius: var(--border-radius);
        padding: 1rem;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }

    .log-entry {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        padding: 0.25rem;
        border-radius: 4px;
    }

    .log-entry:hover {
        background: rgba(255, 255, 255, 0.05);
    }

    .log-time {
        color: var(--text-muted);
        min-width: 80px;
    }

    .log-level {
        min-width: 60px;
        font-weight: 600;
        text-align: center;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }

    .log-level.info { background: var(--info); color: white; }
    .log-level.success { background: var(--success); color: white; }
    .log-level.warning { background: var(--warning); color: white; }
    .log-level.error { background: var(--danger); color: white; }

    .metric-details .metric-overview {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: var(--card-bg);
        border-radius: var(--border-radius);
    }

    .metric-details .metric-value {
        font-size: 3rem;
        font-weight: bold;
        min-width: 120px;
    }

    .chart-placeholder {
        background: var(--bg-tertiary);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 1rem 0;
    }

    .metric-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        background: var(--card-bg);
        border-radius: var(--border-radius);
    }

    .confirmation-dialog {
        text-align: center;
        padding: 2rem 1rem;
    }

    .confirmation-dialog p {
        font-size: 1.1rem;
        margin: 0;
    }
`;

document.head.appendChild(modalStyles);