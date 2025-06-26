// js/shared/Toast.js
class Toast {
    constructor(containerId = 'toast-container') {
        this.containerId = containerId;
        this.toastCounter = 0;
        this.toasts = new Map();
        this.init();
    }

    init() {
        // Create container if it doesn't exist
        if (!document.getElementById(this.containerId)) {
            const container = document.createElement('div');
            container.id = this.containerId;
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        this.container = document.getElementById(this.containerId);
    }

    show(message, type = 'info', duration = 5000) {
        const toastId = `toast-${this.toastCounter++}`;
        const toast = this.createToast(toastId, message, type);
        
        this.container.appendChild(toast);
        this.toasts.set(toastId, toast);
        
        // Animate in
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Auto-remove
        if (duration > 0) {
            setTimeout(() => this.remove(toastId), duration);
        }
        
        return toastId;
    }

    createToast(id, message, type) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.dataset.toastId = id;
        
        const icons = { 
            success: '✅', 
            danger: '❌', 
            error: '❌',
            warning: '⚠️', 
            info: 'ℹ️' 
        };
        
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
                <div class="toast-message">${message}</div>
                <button class="toast-close" onclick="window.toastSystem.remove('${id}')" aria-label="Close">&times;</button>
            </div>
        `;
        
        return toast;
    }

    remove(toastId) {
        const toast = this.toasts.get(toastId);
        if (toast) {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentElement) toast.remove();
                this.toasts.delete(toastId);
            }, 300);
        }
    }

    success(message, duration = 5000) {
        return this.show(message, 'success', duration);
    }

    error(message, duration = 7000) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration = 6000) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration = 5000) {
        return this.show(message, 'info', duration);
    }
}

// Create global instance
if (typeof window !== 'undefined') {
    window.toastSystem = new Toast();
    
    // Also create global helper functions
    window.showToast = (message, type, duration) => window.toastSystem.show(message, type, duration);
    window.showSuccess = (message, duration) => window.toastSystem.success(message, duration);
    window.showError = (message, duration) => window.toastSystem.error(message, duration);
    window.showWarning = (message, duration) => window.toastSystem.warning(message, duration);
    window.showInfo = (message, duration) => window.toastSystem.info(message, duration);
}