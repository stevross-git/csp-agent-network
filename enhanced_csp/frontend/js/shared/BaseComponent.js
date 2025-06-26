// js/shared/BaseComponent.js
class BaseComponent {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = null;
        this.options = {
            autoInit: false,
            debounceDelay: 250,
            ...options
        };
        
        this.isInitialized = false;
        this.eventListeners = [];
        this.components = new Map();
        
        // Auto-initialize if requested
        if (this.options.autoInit) {
            this.init();
        }
    }
    
    async init() {
        try {
            this.container = document.getElementById(this.containerId);
            if (!this.container) {
                throw new Error(`Container element with ID '${this.containerId}' not found`);
            }
            
            // Load dependencies first
            await this.loadDependencies();
            
            // Then render
            this.render();
            
            // Finally bind events
            this.bindEvents();
            
            this.isInitialized = true;
            this.onReady();
            
        } catch (error) {
            console.error(`Failed to initialize ${this.constructor.name}:`, error);
            this.onError(error);
            throw error;
        }
    }
    
    async loadDependencies() {
        // Override in subclasses to load specific dependencies
    }
    
    render() {
        // Override in subclasses to implement rendering logic
    }
    
    bindEvents() {
        // Override in subclasses to bind event listeners
    }
    
    onReady() {
        // Override in subclasses for post-initialization logic
        console.log(`${this.constructor.name} initialized successfully`);
    }
    
    onError(error) {
        // Override in subclasses for error handling
        console.error(`${this.constructor.name} error:`, error);
        
        // Show user-friendly error message
        if (window.toastSystem) {
            window.toastSystem.error(`Component initialization failed: ${error.message}`);
        }
    }
    
    // Helper method to load external scripts
    async loadScript(src) {
        return new Promise((resolve, reject) => {
            // Check if script is already loaded
            const existingScript = document.querySelector(`script[src="${src}"]`);
            if (existingScript) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => resolve();
            script.onerror = (error) => {
                console.error(`Failed to load script: ${src}`, error);
                reject(new Error(`Failed to load script: ${src}`));
            };
            document.head.appendChild(script);
        });
    }
    
    // Helper method to load CSS files
    async loadCSS(href) {
        return new Promise((resolve, reject) => {
            // Check if CSS is already loaded
            const existingLink = document.querySelector(`link[href="${href}"]`);
            if (existingLink) {
                resolve();
                return;
            }
            
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = href;
            link.onload = () => resolve();
            link.onerror = (error) => {
                console.error(`Failed to load CSS: ${href}`, error);
                reject(new Error(`Failed to load CSS: ${href}`));
            };
            document.head.appendChild(link);
        });
    }
    
    // Event listener management
    addEventListener(element, event, handler, options = {}) {
        const boundHandler = handler.bind(this);
        element.addEventListener(event, boundHandler, options);
        
        // Store for cleanup
        this.eventListeners.push({
            element,
            event,
            handler: boundHandler,
            options
        });
        
        return boundHandler;
    }
    
    // Debounced event handler
    debounce(func, delay = this.options.debounceDelay) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }
    
    // Helper to create DOM elements
    createElement(tagName, attributes = {}, innerHTML = '') {
        const element = document.createElement(tagName);
        
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
        
        if (innerHTML) {
            element.innerHTML = innerHTML;
        }
        
        return element;
    }
    
    // Show/hide element
    show(element = this.container) {
        if (element) element.style.display = '';
    }
    
    hide(element = this.container) {
        if (element) element.style.display = 'none';
    }
    
    // Toggle visibility
    toggle(element = this.container) {
        if (element) {
            element.style.display = element.style.display === 'none' ? '' : 'none';
        }
    }
    
    // Find elements within the component
    find(selector) {
        return this.container ? this.container.querySelector(selector) : null;
    }
    
    findAll(selector) {
        return this.container ? this.container.querySelectorAll(selector) : [];
    }
    
    // Clean up resources
    destroy() {
        // Remove event listeners
        this.eventListeners.forEach(({ element, event, handler, options }) => {
            element.removeEventListener(event, handler, options);
        });
        this.eventListeners = [];
        
        // Destroy child components
        this.components.forEach(component => {
            if (component && typeof component.destroy === 'function') {
                component.destroy();
            }
        });
        this.components.clear();
        
        // Clean up container
        if (this.container) {
            this.container.innerHTML = '';
        }
        
        this.isInitialized = false;
        this.onDestroy();
    }
    
    onDestroy() {
        // Override in subclasses for cleanup logic
        console.log(`${this.constructor.name} destroyed`);
    }
}

// Global helper functions for logging
window.log_info = function(message) {
    if (console && console.log) {
        console.log(`[INFO] ${message}`);
    }
};

window.log_error = function(message) {
    if (console && console.error) {
        console.error(`[ERROR] ${message}`);
    }
};

window.log_success = function(message) {
    if (console && console.log) {
        console.log(`[SUCCESS] ✅ ${message}`);
    }
};

window.log_warning = function(message) {
    if (console && console.warn) {
        console.warn(`[WARNING] ⚠️ ${message}`);
    }
};