// Modal System
// Path: frontend/js/shared/Modal.js

class Modal {
    constructor(containerId = 'modal-container', options = {}) {
        this.containerId = containerId;
        this.options = {
            closeOnOverlayClick: true,
            closeOnEscape: true,
            showCloseButton: true,
            ...options
        };
        
        this.modals = new Map();
        this.modalCounter = 0;
        this.container = null;
        this.currentModal = null;
        
        this.init();
    }
    
    init() {
        this.container = document.getElementById(this.containerId);
        
        if (!this.container) {
            // Create container if it doesn't exist
            this.container = document.createElement('div');
            this.container.id = this.containerId;
            this.container.className = 'modal-container';
            document.body.appendChild(this.container);
        }
        
        // Set up global event listeners
        this.setupEventListeners();
        
        console.log('✅ Modal system initialized');
    }
    
    setupEventListeners() {
        // Escape key handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.options.closeOnEscape && this.currentModal) {
                this.close(this.currentModal);
            }
        });
    }
    
    show(content, options = {}) {
        const id = ++this.modalCounter;
        const config = { ...this.options, ...options };
        
        // Create modal overlay
        const overlay = this.createElement('div', {
            className: 'modal-overlay',
            'data-modal-id': id
        });
        
        // Create modal
        const modal = this.createElement('div', {
            className: `modal ${config.className || ''}`
        });
        
        // Create modal content
        let modalHTML = '';
        
        if (config.title || config.showCloseButton) {
            modalHTML += `
                <div class="modal-header">
                    <h3 class="modal-title">${config.title || ''}</h3>
                    ${config.showCloseButton ? `<button class="modal-close" onclick="window.Modal.close(${id})">×</button>` : ''}
                </div>
            `;
        }
        
        modalHTML += `<div class="modal-content">${content}</div>`;
        
        if (config.footer) {
            modalHTML += `<div class="modal-footer">${config.footer}</div>`;
        }
        
        modal.innerHTML = modalHTML;
        overlay.appendChild(modal);
        
        // Add overlay click handler
        if (config.closeOnOverlayClick) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.close(id);
                }
            });
        }
        
        // Add to container and show
        this.container.appendChild(overlay);
        this.modals.set(id, overlay);
        this.currentModal = id;
        
        // Show with animation
        setTimeout(() => {
            overlay.style.display = 'block';
            overlay.style.animation = 'fadeIn 0.3s ease-out';
        }, 10);
        
        // Disable body scroll
        document.body.style.overflow = 'hidden';
        
        return id;
    }
    
    close(id) {
        const modal = this.modals.get(id);
        if (modal) {
            modal.style.animation = 'fadeOut 0.3s ease-in-out';
            
            setTimeout(() => {
                if (modal.parentNode) {
                    modal.parentNode.removeChild(modal);
                }
                this.modals.delete(id);
                
                // Re-enable body scroll if no modals are open
                if (this.modals.size === 0) {
                    document.body.style.overflow = '';
                }
                
                // Update current modal
                if (this.currentModal === id) {
                    const remainingModals = Array.from(this.modals.keys());
                    this.currentModal = remainingModals.length > 0 ? remainingModals[remainingModals.length - 1] : null;
                }
            }, 300);
        }
    }
    
    closeAll() {
        this.modals.forEach((modal, id) => this.close(id));
    }
    
    confirm(message, options = {}) {
        return new Promise((resolve) => {
            const config = {
                title: 'Confirm',
                showCloseButton: false,
                closeOnOverlayClick: false,
                closeOnEscape: false,
                ...options
            };
            
            const content = `
                <p>${message}</p>
                <div class="modal-actions" style="text-align: right; margin-top: 20px;">
                    <button class="btn btn-secondary" onclick="window.Modal.resolveConfirm(false)">Cancel</button>
                    <button class="btn btn-primary" onclick="window.Modal.resolveConfirm(true)" style="margin-left: 10px;">Confirm</button>
                </div>
            `;
            
            this.confirmResolver = resolve;
            this.confirmModalId = this.show(content, config);
        });
    }
    
    resolveConfirm(result) {
        if (this.confirmResolver) {
            this.confirmResolver(result);
            this.confirmResolver = null;
        }
        if (this.confirmModalId) {
            this.close(this.confirmModalId);
            this.confirmModalId = null;
        }
    }
    
    alert(message, options = {}) {
        return new Promise((resolve) => {
            const config = {
                title: 'Alert',
                showCloseButton: false,
                closeOnOverlayClick: false,
                closeOnEscape: false,
                ...options
            };
            
            const content = `
                <p>${message}</p>
                <div class="modal-actions" style="text-align: right; margin-top: 20px;">
                    <button class="btn btn-primary" onclick="window.Modal.resolveAlert()">OK</button>
                </div>
            `;
            
            this.alertResolver = resolve;
            this.alertModalId = this.show(content, config);
        });
    }
    
    resolveAlert() {
        if (this.alertResolver) {
            this.alertResolver();
            this.alertResolver = null;
        }
        if (this.alertModalId) {
            this.close(this.alertModalId);
            this.alertModalId = null;
        }
    }
    
    createElement(tag, attributes = {}) {
        const element = document.createElement(tag);
        
        Object.keys(attributes).forEach(key => {
            if (key === 'className') {
                element.className = attributes[key];
            } else if (key.startsWith('data-')) {
                element.setAttribute(key, attributes[key]);
            } else {
                element[key] = attributes[key];
            }
        });
        
        return element;
    }
}

// Add CSS styles if not already present
if (!document.querySelector('#modal-styles')) {
    const style = document.createElement('style');
    style.id = 'modal-styles';
    style.textContent = `
        .modal-container {
            position: relative;
            z-index: 10000;
        }
        
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: none;
            z-index: 10000;
        }
        
        .modal {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        .modal-header {
            padding: 20px 20px 0 20px;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .modal-title {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            color: #111827;
        }
        
        .modal-content {
            padding: 20px;
        }
        
        .modal-footer {
            padding: 0 20px 20px 20px;
            border-top: 1px solid #e5e7eb;
            margin-top: 20px;
            padding-top: 20px;
        }
        
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6b7280;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
        }
        
        .modal-close:hover {
            background-color: #f3f4f6;
            color: #374151;
        }
        
        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid transparent;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-primary {
            background-color: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
        
        .btn-primary:hover {
            background-color: #2563eb;
            border-color: #2563eb;
        }
        
        .btn-secondary {
            background-color: #6b7280;
            color: white;
            border-color: #6b7280;
        }
        
        .btn-secondary:hover {
            background-color: #4b5563;
            border-color: #4b5563;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes fadeOut {
            from {
                opacity: 1;
            }
            to {
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// Create global instance
if (typeof window !== 'undefined') {
    window.Modal = new Modal();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Modal;
}