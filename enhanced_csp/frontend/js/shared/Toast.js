// js/shared/Toast.js
class Toast extends BaseComponent {
    constructor(containerId = 'toast-container') {
        super(containerId, { autoInit: true });
        this.toastCounter = 0;
        this.toasts = new Map();
    }

    render() {
        if (!this.container) {
            // Create container if it doesn't exist
            this.container = document.createElement('div');
            this.container.id = this.containerId;
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
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
        
        const icons = { success: '✅', danger: '❌', warning: '⚠️', info: 'ℹ️' };
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
}

// Global instance
window.toastSystem = new Toast();