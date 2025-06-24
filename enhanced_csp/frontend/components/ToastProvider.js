/**
 * Enhanced CSP System - Toast Notification Provider
 * Global toast notification system for consistent UX
 */

class ToastProvider {
    constructor() {
        this.toasts = [];
        this.toastId = 0;
        this.container = null;
        this.init();
    }

    init() {
        // Create toast container
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.className = 'toast-container';
        
        // Add container styles
        const styles = `
            .toast-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                max-width: 400px;
                pointer-events: none;
            }
            
            .toast {
                background: white;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                border-left: 4px solid;
                pointer-events: auto;
                transform: translateX(100%);
                transition: transform 0.3s ease, opacity 0.3s ease;
                opacity: 0;
                max-height: 200px;
                overflow: hidden;
            }
            
            .toast.show {
                transform: translateX(0);
                opacity: 1;
            }
            
            .toast.success { border-left-color: #10b981; }
            .toast.error { border-left-color: #ef4444; }
            .toast.warning { border-left-color: #f59e0b; }
            .toast.info { border-left-color: #3b82f6; }
            
            .toast-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            
            .toast-title {
                font-weight: 600;
                font-size: 14px;
                color: #1f2937;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .toast-close {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #6b7280;
                padding: 0;
                line-height: 1;
            }
            
            .toast-close:hover { color: #374151; }
            
            .toast-message {
                font-size: 13px;
                color: #4b5563;
                line-height: 1.4;
            }
        `;
        
        // Add styles to document
        if (!document.getElementById('toast-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'toast-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
        
        // Add container to body
        document.body.appendChild(this.container);
        
        console.log('🔔 Toast Provider initialized');
    }

    showSuccess(title, message, options = {}) {
        return this.showToast('success', title, message, { icon: '✅', duration: 4000, ...options });
    }

    showError(title, message, options = {}) {
        return this.showToast('error', title, message, { icon: '❌', duration: 8000, ...options });
    }

    showWarning(title, message, options = {}) {
        return this.showToast('warning', title, message, { icon: '⚠️', duration: 6000, ...options });
    }

    showInfo(title, message, options = {}) {
        return this.showToast('info', title, message, { icon: 'ℹ️', duration: 5000, ...options });
    }

    showToast(type, title, message, options = {}) {
        const toastId = ++this.toastId;
        const { icon = '', duration = 5000, onClose = null } = options;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.setAttribute('data-toast-id', toastId);

        toast.innerHTML = `
            <div class="toast-header">
                <div class="toast-title">
                    ${icon ? `<span>${icon}</span>` : ''}
                    ${title}
                </div>
                <button class="toast-close" aria-label="Close">&times;</button>
            </div>
            <div class="toast-message">${message}</div>
        `;

        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            this.removeToast(toastId, onClose);
        });

        this.container.appendChild(toast);
        this.toasts.push({ id: toastId, element: toast, type });

        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        if (duration > 0) {
            setTimeout(() => {
                this.removeToast(toastId, onClose);
            }, duration);
        }

        return toastId;
    }

    removeToast(toastId, onClose = null) {
        const toastIndex = this.toasts.findIndex(t => t.id === toastId);
        if (toastIndex === -1) return;

        const toast = this.toasts[toastIndex];
        
        toast.element.classList.remove('show');
        
        setTimeout(() => {
            if (toast.element.parentNode) {
                toast.element.parentNode.removeChild(toast.element);
            }
            this.toasts.splice(toastIndex, 1);
            
            if (onClose) {
                onClose();
            }
        }, 300);
    }

    clearAll() {
        this.toasts.forEach(toast => {
            this.removeToast(toast.id);
        });
    }
}

// Create global instance
window.toastProvider = new ToastProvider();

// Global convenience methods
window.toast = {
    success: (title, message, options) => window.toastProvider.showSuccess(title, message, options),
    error: (title, message, options) => window.toastProvider.showError(title, message, options),
    warning: (title, message, options) => window.toastProvider.showWarning(title, message, options),
    info: (title, message, options) => window.toastProvider.showInfo(title, message, options),
    clear: () => window.toastProvider.clearAll()
};

export { ToastProvider };
export default window.toastProvider;
