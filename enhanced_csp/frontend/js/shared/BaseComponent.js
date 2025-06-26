// js/shared/BaseComponent.js (Enhanced with EventEmitter)
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
        
        // EventEmitter functionality
        this.events = new Map();
        
        // Auto-initialize if requested
        if (this.options.autoInit) {
            this.init();
        }
    }
    
    // EventEmitter methods
    on(eventName, callback) {
        if (!this.events.has(eventName)) {
            this.events.set(eventName, []);
        }
        this.events.get(eventName).push(callback);
        return this;
    }
    
    off(eventName, callback) {
        if (this.events.has(eventName)) {
            const callbacks = this.events.get(eventName);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
        return this;
    }
    
    emit(eventName, ...args) {
        if (this.events.has(eventName)) {
            const callbacks = this.events.get(eventName);
            callbacks.forEach(callback => {
                try {
                    callback.apply(this, args);
                } catch (error) {
                    console.error(`Error in event handler for '${eventName}':`, error);
                }
            });
        }
        return this;
    }
    
    once(eventName, callback) {
        const onceWrapper = (...args) => {
            this.off(eventName, onceWrapper);
            callback.apply(this, args);
        };
        this.on(eventName, onceWrapper);
        return this;
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
    
    // Load external scripts
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
    
    // Cleanup
    destroy() {
        // Remove all event listeners
        this.eventListeners.forEach(({ element, event, handler, options }) => {
            element.removeEventListener(event, handler, options);
        });
        this.eventListeners = [];
        
        // Clear custom events
        this.events.clear();
        
        // Destroy child components
        this.components.forEach(component => {
            if (component.destroy) {
                component.destroy();
            }
        });
        this.components.clear();
        
        this.isInitialized = false;
    }
}