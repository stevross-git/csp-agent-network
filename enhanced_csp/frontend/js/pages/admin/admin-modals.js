// Admin Portal Modal Functions Fix
// Add this to your admin portal JavaScript or create a new admin-modals.js file

const API_BASE_URL = 'http://localhost:8000';

// Enhanced Modal Management System
class AdminModalManager {
    constructor() {
        this.activeModals = new Set();
        this.initialized = false;
        this.init();
    }

    init() {
        if (this.initialized) return;
        
        console.log('🎛️ Initializing Admin Modal Manager...');
        
        // Bind global modal events
        this.bindGlobalEvents();
        
        // Fix existing buttons
        this.fixExistingButtons();
        
        this.initialized = true;
        console.log('✅ Admin Modal Manager initialized');
    }

    bindGlobalEvents() {
        // ESC key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });

        // Click outside modal to close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal') && e.target.classList.contains('active')) {
                this.closeModal(e.target.id);
            }
        });
    }

    fixExistingButtons() {
        // Fix "Add New User" button
        const addUserBtn = document.querySelector('button[onclick*="add-user-modal"]');
        if (addUserBtn) {
            addUserBtn.removeAttribute('onclick');
            addUserBtn.addEventListener('click', () => this.openAddUserModal());
        }

        // Fix "Add New Model" / "Deploy Model" button  
        const addModelBtn = document.querySelector('button[onclick*="deploy-model-modal"]');
        if (addModelBtn) {
            addModelBtn.removeAttribute('onclick');
            addModelBtn.addEventListener('click', () => this.openDeployModelModal());
        }

        // Fix any other buttons with onclick="openModal(...)"
        document.querySelectorAll('button[onclick^="openModal"]').forEach(btn => {
            const modalId = btn.getAttribute('onclick').match(/openModal\('([^']+)'\)/)?.[1];
            if (modalId) {
                btn.removeAttribute('onclick');
                btn.addEventListener('click', () => this.openModal(modalId));
            }
        });

        // Fix close buttons
        document.querySelectorAll('button[onclick^="closeModal"]').forEach(btn => {
            const modalId = btn.getAttribute('onclick').match(/closeModal\('([^']+)'\)/)?.[1];
            if (modalId) {
                btn.removeAttribute('onclick');
                btn.addEventListener('click', () => this.closeModal(modalId));
            }
        });

        console.log('🔧 Fixed existing modal buttons');
    }

    openModal(modalId) {
        console.log(`📂 Opening modal: ${modalId}`);
        
        const modal = document.getElementById(modalId);
        if (!modal) {
            console.error(`❌ Modal not found: ${modalId}`);
            return false;
        }

        // Close other modals first
        this.closeAllModals();

        // Show modal
        modal.classList.add('active');
        modal.style.display = 'flex';
        this.activeModals.add(modalId);

        // Prevent body scroll
        document.body.style.overflow = 'hidden';

        // Focus management
        this.focusModal(modal);

        // Add animation
        modal.style.opacity = '0';
        modal.style.transform = 'scale(0.9)';
        
        requestAnimationFrame(() => {
            modal.style.transition = 'all 0.3s ease-out';
            modal.style.opacity = '1';
            modal.style.transform = 'scale(1)';
        });

        return true;
    }

    closeModal(modalId) {
        console.log(`🚪 Closing modal: ${modalId}`);
        
        const modal = document.getElementById(modalId);
        if (!modal) {
            console.error(`❌ Modal not found: ${modalId}`);
            return false;
        }

        // Animation out
        modal.style.transition = 'all 0.2s ease-in';
        modal.style.opacity = '0';
        modal.style.transform = 'scale(0.9)';

        setTimeout(() => {
            modal.classList.remove('active');
            modal.style.display = 'none';
            modal.style.transition = '';
            modal.style.transform = '';
            modal.style.opacity = '';
        }, 200);

        this.activeModals.delete(modalId);

        // Re-enable body scroll if no modals are open
        if (this.activeModals.size === 0) {
            document.body.style.overflow = '';
        }

        return true;
    }

    closeAllModals() {
        this.activeModals.forEach(modalId => {
            this.closeModal(modalId);
        });
    }

    focusModal(modal) {
        // Focus first input or button in modal
        const focusable = modal.querySelector('input, select, textarea, button:not(.close-btn)');
        if (focusable) {
            setTimeout(() => focusable.focus(), 100);
        }
    }

    // Specific modal openers
    openAddUserModal() {
        console.log('👤 Opening Add User Modal...');
        if (this.openModal('add-user-modal')) {
            // Initialize form
            this.initializeAddUserForm();
        }
    }

    openDeployModelModal() {
        console.log('🚀 Opening Deploy Model Modal...');
        if (this.openModal('deploy-model-modal')) {
            // Initialize form
            this.initializeDeployModelForm();
        }
    }

    initializeAddUserForm() {
        const modal = document.getElementById('add-user-modal');
        if (!modal) return;

        // Clear form
        modal.querySelectorAll('input, select, textarea').forEach(field => {
            if (field.type === 'checkbox' || field.type === 'radio') {
                field.checked = false;
            } else {
                field.value = '';
            }
        });

        // Set default role
        const roleSelect = modal.querySelector('select');
        if (roleSelect) {
            roleSelect.value = 'user';
        }

        // Bind form submission
        this.bindAddUserFormSubmission(modal);

        console.log('📋 Add User form initialized');
    }

    initializeDeployModelForm() {
        const modal = document.getElementById('deploy-model-modal');
        if (!modal) return;

        // Clear form
        modal.querySelectorAll('input, select, textarea').forEach(field => {
            if (field.type === 'range') {
                field.value = field.getAttribute('value') || field.min;
            } else if (field.type !== 'checkbox' && field.type !== 'radio') {
                field.value = '';
            }
        });

        // Update range displays
        modal.querySelectorAll('input[type="range"]').forEach(range => {
            this.updateRangeDisplay(range);
            range.addEventListener('input', () => this.updateRangeDisplay(range));
        });

        // Bind form submission
        this.bindDeployModelFormSubmission(modal);

        console.log('🤖 Deploy Model form initialized');
    }

    updateRangeDisplay(rangeInput) {
        const display = rangeInput.parentNode.querySelector('span');
        if (display) {
            const unit = rangeInput.parentNode.textContent.includes('Memory') ? ' GB' : ' cores';
            display.textContent = rangeInput.value + unit;
        }
    }

    bindAddUserFormSubmission(modal) {
        // Remove existing listeners
        const createBtn = modal.querySelector('.btn-primary');
        if (createBtn) {
            // Clone to remove existing listeners
            const newBtn = createBtn.cloneNode(true);
            createBtn.parentNode.replaceChild(newBtn, createBtn);
            
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleCreateUser(modal);
            });
        }
    }

    bindDeployModelFormSubmission(modal) {
        // Remove existing listeners
        const deployBtn = modal.querySelector('.btn-primary');
        if (deployBtn) {
            // Clone to remove existing listeners
            const newBtn = deployBtn.cloneNode(true);
            deployBtn.parentNode.replaceChild(newBtn, deployBtn);
            
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleDeployModel(modal);
            });
        }
    }

    async handleCreateUser(modal) {
        console.log('👤 Processing new user creation...');
        
        // Get form data
        const formData = this.getFormData(modal);
        
        // Validate
        if (!this.validateUserForm(formData)) {
            return false;
        }

        // Show loading
        this.showButtonLoading(modal.querySelector('.btn-primary'), '⏳ Creating...');

        try {
            // Simulate API call
            await this.createUserAPI(formData);
            
            // Success
            this.showNotification('✅ User created successfully!', 'success');
            this.closeModal('add-user-modal');
            
            // Refresh user list if visible
            this.refreshUserList();
            
        } catch (error) {
            console.error('❌ Failed to create user:', error);
            this.showNotification('❌ Failed to create user: ' + error.message, 'error');
        } finally {
            this.hideButtonLoading(modal.querySelector('.btn-primary'), '👤 Create User');
        }
    }

    async handleDeployModel(modal) {
        console.log('🚀 Processing model deployment...');
        
        // Get form data
        const formData = this.getFormData(modal);
        
        // Validate
        if (!this.validateModelForm(formData)) {
            return false;
        }

        // Show loading
        this.showButtonLoading(modal.querySelector('.btn-primary'), '⏳ Deploying...');

        try {
            // Simulate API call
            await this.deployModelAPI(formData);
            
            // Success
            this.showNotification('✅ Model deployed successfully!', 'success');
            this.closeModal('deploy-model-modal');
            
            // Refresh model list if visible
            this.refreshModelList();
            
        } catch (error) {
            console.error('❌ Failed to deploy model:', error);
            this.showNotification('❌ Failed to deploy model: ' + error.message, 'error');
        } finally {
            this.hideButtonLoading(modal.querySelector('.btn-primary'), '🚀 Deploy Model');
        }
    }

    getFormData(modal) {
        const formData = {};
        modal.querySelectorAll('input, select, textarea').forEach(field => {
            if (field.name || field.id) {
                const key = field.name || field.id;
                if (field.type === 'checkbox') {
                    formData[key] = field.checked;
                } else {
                    formData[key] = field.value;
                }
            } else {
                // Fallback: use placeholder or label
                const label = modal.querySelector(`label[for="${field.id}"]`)?.textContent || 
                             field.previousElementSibling?.textContent || 
                             field.placeholder;
                if (label) {
                    const key = label.toLowerCase().replace(/[^a-z0-9]/g, '_');
                    formData[key] = field.value;
                }
            }
        });
        return formData;
    }

    validateUserForm(data) {
        const requiredFields = ['full_name', 'email_address'];
        
        for (const field of requiredFields) {
            if (!data[field] || data[field].trim() === '') {
                this.showNotification(`❌ Please fill in all required fields`, 'error');
                return false;
            }
        }

        // Email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(data.email_address)) {
            this.showNotification('❌ Please enter a valid email address', 'error');
            return false;
        }

        return true;
    }

    validateModelForm(data) {
        const requiredFields = ['model_name', 'model_type'];
        
        for (const field of requiredFields) {
            if (!data[field] || data[field].trim() === '') {
                this.showNotification(`❌ Please fill in all required fields`, 'error');
                return false;
            }
        }

        return true;
    }

    async createUserAPI(userData) {
        const payload = {
            email: userData.email_address,
            password: userData.initial_password || 'ChangeMe123!',
            confirm_password: userData.initial_password || 'ChangeMe123!',
            full_name: userData.full_name
        };

        const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'User creation failed');
        }

        return data;
    }

    async deployModelAPI(modelData) {
        // Simulate API call
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (Math.random() > 0.1) { // 90% success rate
                    resolve({ id: Date.now(), status: 'deploying', ...modelData });
                } else {
                    reject(new Error('Deployment failed'));
                }
            }, 2000);
        });
    }

    showButtonLoading(button, loadingText) {
        if (!button) return;
        button.dataset.originalText = button.textContent;
        button.textContent = loadingText;
        button.disabled = true;
        button.style.opacity = '0.7';
    }

    hideButtonLoading(button, originalText) {
        if (!button) return;
        button.textContent = button.dataset.originalText || originalText;
        button.disabled = false;
        button.style.opacity = '1';
        delete button.dataset.originalText;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                ${message}
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        // Add styles if not already present
        this.addNotificationStyles();

        // Add to page
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

        console.log(`📢 Notification: ${message}`);
    }

    addNotificationStyles() {
        if (document.getElementById('notification-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                pointer-events: none;
            }
            
            .notification {
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                margin-bottom: 10px;
                min-width: 300px;
                max-width: 500px;
                animation: slideInRight 0.3s ease-out;
                pointer-events: all;
            }
            
            .notification-success { border-left: 4px solid #00ff88; }
            .notification-error { border-left: 4px solid #ff4757; }
            .notification-info { border-left: 4px solid #3742fa; }
            
            .notification-content {
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .notification-close {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #666;
                padding: 0;
                margin-left: 15px;
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            @keyframes slideOutRight {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(styles);
    }

    refreshUserList() {
        console.log('🔄 Refreshing user list...');
        // Trigger user list refresh if on users page
        if (window.currentSection === 'users') {
            // This would call your actual user list refresh function
            if (typeof loadUserData === 'function') {
                loadUserData();
            }
        }
    }

    refreshModelList() {
        console.log('🔄 Refreshing model list...');
        // Trigger model list refresh if on AI models page
        if (window.currentSection === 'ai-models') {
            // This would call your actual model list refresh function
            if (typeof loadAIModelData === 'function') {
                loadAIModelData();
            }
        }
    }
}

// Global Functions (for backward compatibility)
function openModal(modalId) {
    if (window.adminModalManager) {
        return window.adminModalManager.openModal(modalId);
    }
    console.error('❌ AdminModalManager not initialized');
    return false;
}

function closeModal(modalId) {
    if (window.adminModalManager) {
        return window.adminModalManager.closeModal(modalId);
    }
    console.error('❌ AdminModalManager not initialized');
    return false;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.adminModalManager = new AdminModalManager();
    });
} else {
    window.adminModalManager = new AdminModalManager();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdminModalManager;
}