// js/utils/ApiClient.js
class ApiClient {
    constructor(options = {}) {
        this.baseURL = options.baseURL || '/api';
        this.timeout = options.timeout || 10000;
        this.retryAttempts = options.retryAttempts || 3;
        this.retryDelay = options.retryDelay || 1000;
        
        // Request interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];
        
        // Add default interceptors
        this.addDefaultInterceptors();
        
        // Auth token
        this.authToken = localStorage.getItem('authToken') || null;

        // Fallback data for failed requests
        this.fallbackData = options.fallbackData || {};
    }
    
    addDefaultInterceptors() {
        // Add auth token to requests
        this.requestInterceptors.push((config) => {
            if (this.authToken) {
                config.headers = {
                    ...config.headers,
                    'Authorization': `Bearer ${this.authToken}`
                };
            }
            return config;
        });
        
        // Handle auth errors
        this.responseInterceptors.push((response) => {
            if (response.status === 401) {
                this.clearAuth();
                window.location.href = '/login.html';
                return Promise.reject(new Error('Unauthorized'));
            }
            return response;
        });
    }
    
    setAuthToken(token) {
        this.authToken = token;
        if (token) {
            localStorage.setItem('authToken', token);
        } else {
            localStorage.removeItem('authToken');
        }
    }
    
    clearAuth() {
        this.setAuthToken(null);
    }
    
    // Add custom interceptors
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }
    
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    // Register fallback data for a specific endpoint
    setFallbackData(endpoint, data) {
        this.fallbackData[endpoint] = data;
    }
    
    // Apply request interceptors
    async applyRequestInterceptors(config) {
        let processedConfig = { ...config };
        
        for (const interceptor of this.requestInterceptors) {
            try {
                processedConfig = await interceptor(processedConfig);
            } catch (error) {
                console.error('Request interceptor error:', error);
            }
        }
        
        return processedConfig;
    }
    
    // Apply response interceptors
    async applyResponseInterceptors(response) {
        let processedResponse = response;
        
        for (const interceptor of this.responseInterceptors) {
            try {
                processedResponse = await interceptor(processedResponse);
            } catch (error) {
                console.error('Response interceptor error:', error);
                throw error;
            }
        }
        
        return processedResponse;
    }
    
    // Core request method with retry logic
    async request(endpoint, options = {}) {
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                ...options.headers
            },
            ...options
        };
        
        // Apply request interceptors
        const processedConfig = await this.applyRequestInterceptors(config);
        
        const url = `${this.baseURL}${endpoint}`;
        let lastError;
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const response = await this.fetchWithTimeout(url, processedConfig);
                
                // Apply response interceptors
                const processedResponse = await this.applyResponseInterceptors(response);
                
                // Parse JSON response
                if (processedResponse.ok) {
                    const contentType = processedResponse.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const data = await processedResponse.json();
                        return { success: true, data, status: processedResponse.status };
                    } else {
                        const text = await processedResponse.text();
                        return { success: true, data: text, status: processedResponse.status };
                    }
                } else if (processedResponse.status === 404) {
                    const key = endpoint.split('?')[0];
                    if (this.fallbackData[key]) {
                        console.warn(`Endpoint not found: ${endpoint}. Using fallback data.`);
                        return { success: true, data: this.fallbackData[key], status: 200, fallback: true };
                    }
                    const error = await this.parseErrorResponse(processedResponse);
                    throw error;
                } else {
                    const error = await this.parseErrorResponse(processedResponse);
                    throw error;
                }
                
            } catch (error) {
                lastError = error;
                
                // Don't retry on client errors (4xx)
                if (error.status >= 400 && error.status < 500) {
                    break;
                }
                
                if (attempt < this.retryAttempts) {
                    console.warn(`Request attempt ${attempt} failed, retrying...`, error.message);
                    await this.delay(this.retryDelay * attempt);
                }
            }
        }
        
        throw lastError;
    }
    
    async fetchWithTimeout(url, config) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        
        try {
            const response = await fetch(url, {
                ...config,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error(`Request timeout after ${this.timeout}ms`);
            }
            throw error;
        }
    }
    
    async parseErrorResponse(response) {
        try {
            const errorData = await response.json();
            return new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        } catch {
            return new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // Convenience methods
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    }
    
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
    
    async patch(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PATCH',
            body: JSON.stringify(data)
        });
    }
    
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
    
    // File upload
    async upload(endpoint, formData) {
        return this.request(endpoint, {
            method: 'POST',
            body: formData,
            headers: {} // Let browser set Content-Type for FormData
        });
    }
    
    // Batch requests
    async batch(requests) {
        const promises = requests.map(({ endpoint, options }) => 
            this.request(endpoint, options)
        );
        
        try {
            const results = await Promise.allSettled(promises);
            return results.map((result, index) => ({
                index,
                success: result.status === 'fulfilled',
                data: result.status === 'fulfilled' ? result.value : null,
                error: result.status === 'rejected' ? result.reason : null
            }));
        } catch (error) {
            throw new Error(`Batch request failed: ${error.message}`);
        }
    }
    
    // WebSocket connection helper
    createWebSocket(endpoint, protocols = []) {
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${location.host}${endpoint}`;
        
        return new WebSocket(wsUrl, protocols);
    }
    
    // Health check
    async healthCheck() {
        try {
            const response = await this.get('/health');
            return response.success;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }
    
    // Debug mode
    enableDebug() {
        this.addRequestInterceptor((config) => {
            console.group(`🚀 API Request: ${config.method} ${config.url || 'unknown'}`);
            console.log('Config:', config);
            console.groupEnd();
            return config;
        });
        
        this.addResponseInterceptor((response) => {
            console.group(`📥 API Response: ${response.status}`);
            console.log('Response:', response);
            console.groupEnd();
            return response;
        });
    }
}

// Create global instance with optional fallback data
window.ApiClient = new ApiClient({ fallbackData: window.apiFallbackData || {} });

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ApiClient;
}