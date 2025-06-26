// js/pages/admin/services/AdminPageService.js
class AdminPageService {
    constructor(apiClient = window.ApiClient) {
        this.api = apiClient;
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    async fetchData(useCache = true) {
        const cacheKey = 'admin_data';
        
        if (useCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        try {
            const response = await this.api.get('/admin');
            
            if (response.success) {
                this.cache.set(cacheKey, {
                    data: response.data,
                    timestamp: Date.now()
                });
                return response.data;
            }
            
            throw new Error(response.error || 'Failed to fetch data');
        } catch (error) {
            console.error('AdminPageService.fetchData error:', error);
            throw error;
        }
    }
    
    async saveData(data) {
        try {
            const response = await this.api.post('/admin', data);
            
            if (response.success) {
                // Invalidate cache
                this.cache.delete('admin_data');
                return response.data;
            }
            
            throw new Error(response.error || 'Failed to save data');
        } catch (error) {
            console.error('AdminPageService.saveData error:', error);
            throw error;
        }
    }
    
    clearCache() {
        this.cache.clear();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdminPageService;
}
