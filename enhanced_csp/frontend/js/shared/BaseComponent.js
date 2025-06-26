// js/shared/BaseComponent.js
class BaseComponent {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.options = {
            autoInit: true,
            debounceDelay: 300,
            ...options
        };
        this.state = {};
        this.eventBus = window.EventBus || new EventBus();
        this.subscriptions = new Set();
        this.isDestroyed = false;
        
        if (!this.container) {
            console.warn(`Container ${containerId} not found`);
            return;
        }
        
        if (this.options.autoInit) {
            this.init();
        }
    }
    
    async init() {
        try {
            await this.loadDependencies();
            this.render();
            this.bindEvents();
            this.onInitialized();
        } catch (error) {
            console.error(`Failed to initialize ${this.constructor.name}:`, error);
            this.onError(error);
        }
    }
    
    async loadDependencies() {
        // Override in subclasses to load required components/services
        return Promise.resolve();
    }
    
    render() {
        // Override in subclasses
        if (this.container) {
            this.container.classList.add('component-initialized');
        }
    }
    
    bindEvents() {
        // Override in subclasses
        // Auto-bind methods that start with 'handle'
        this.autobindHandlers();
    }
    
    autobindHandlers() {
        const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(this));
        methods
            .filter(method => method.startsWith('handle') && typeof this[method] === 'function')
            .forEach(method => {
                this[method] = this[method].bind(this);
            });
    }
    
    // State management
    setState(newState, callback) {
        const oldState = { ...this.state };
        this.state = { ...this.state, ...newState };
        
        if (typeof callback === 'function') {
            callback(this.state, oldState);
        }
        
        this.onStateChange(this.state, oldState);
        this.eventBus.emit(`${this.containerId}:stateChange`, { 
            newState: this.state, 
            oldState 
        });
    }
    
    getState() {
        return { ...this.state };
    }
    
    // Event system
    subscribe(event, handler) {
        this.eventBus.on(event, handler);
        this.subscriptions.add({ event, handler });
    }
    
    emit(event, data) {
        this.eventBus.emit(event, data);
    }
    
    // Utility methods
    debounce(func, delay = this.options.debounceDelay) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }
    
    throttle(func, limit = this.options.debounceDelay) {
        let inThrottle;
        return (...args) => {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    // DOM utilities
    createElement(tag, attributes = {}, content = '') {
        const element = document.createElement(tag);
        
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'dataset') {
                Object.entries(value).forEach(([dataKey, dataValue]) => {
                    element.dataset[dataKey] = dataValue;
                });
            } else {
                element.setAttribute(key, value);
            }
        });
        
        if (content) {
            if (typeof content === 'string') {
                element.innerHTML = content;
            } else {
                element.appendChild(content);
            }
        }
        
        return element;
    }
    
    findElement(selector) {
        return this.container ? this.container.querySelector(selector) : null;
    }
    
    findElements(selector) {
        return this.container ? Array.from(this.container.querySelectorAll(selector)) : [];
    }
    
    // Loading states
    setLoading(isLoading = true) {
        if (!this.container) return;
        
        if (isLoading) {
            this.container.classList.add('loading');
            if (!this.container.querySelector('.loading-spinner')) {
                const spinner = this.createElement('div', { 
                    className: 'loading-spinner' 
                }, `
                    <div class="spinner"></div>
                    <span>Loading...</span>
                `);
                this.container.appendChild(spinner);
            }
        } else {
            this.container.classList.remove('loading');
            const spinner = this.container.querySelector('.loading-spinner');
            if (spinner) {
                spinner.remove();
            }
        }
    }
    
    // Error handling
    showError(message, details = null) {
        console.error(`${this.constructor.name} Error:`, message, details);
        
        if (window.Toast) {
            window.Toast.error(message);
        } else {
            alert(`Error: ${message}`);
        }
    }
    
    // Lifecycle hooks
    onInitialized() {
        // Override in subclasses
        console.log(`${this.constructor.name} initialized`);
    }
    
    onStateChange(newState, oldState) {
        // Override in subclasses
    }
    
    onError(error) {
        this.showError('Component initialization failed', error);
    }
    
    onDestroy() {
        // Override in subclasses for cleanup
    }
    
    // Cleanup
    destroy() {
        if (this.isDestroyed) return;
        
        // Remove event subscriptions
        this.subscriptions.forEach(({ event, handler }) => {
            this.eventBus.off(event, handler);
        });
        this.subscriptions.clear();
        
        // Remove loading state
        this.setLoading(false);
        
        // Call lifecycle hook
        this.onDestroy();
        
        // Mark as destroyed
        this.isDestroyed = true;
        
        console.log(`${this.constructor.name} destroyed`);
    }
    
    // Static helper for creating instances
    static create(containerId, options = {}) {
        return new this(containerId, options);
    }
}

// Simple EventBus implementation if not available globally
if (typeof EventBus === 'undefined') {
    window.EventBus = class EventBus {
        constructor() {
            this.events = {};
        }
        
        on(event, callback) {
            if (!this.events[event]) {
                this.events[event] = [];
            }
            this.events[event].push(callback);
        }
        
        off(event, callback) {
            if (!this.events[event]) return;
            
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
        
        emit(event, data) {
            if (!this.events[event]) return;
            
            this.events[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('EventBus callback error:', error);
                }
            });
        }
        
        clear() {
            this.events = {};
        }
    };
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BaseComponent;
}