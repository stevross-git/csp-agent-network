// frontend/js/services/AIModelsService.js
/**
 * AI Models Service - Database Integration
 * =======================================
 * Service for managing AI models with real database backend
 */

class AIModelsService {
    constructor(apiBaseUrl = '/api/ai-models') {
        this.apiBaseUrl = apiBaseUrl;
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }

    // ========================================================================
    // HTTP UTILITY METHODS
    // ========================================================================

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                // Add authentication headers if needed
                // 'Authorization': `Bearer ${this.getAuthToken()}`
            }
        };

        const finalOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    // ========================================================================
    // MODEL MANAGEMENT METHODS
    // ========================================================================

    async getAllModels(useCache = true) {
        const cacheKey = 'all_models';
        
        // Check cache first
        if (useCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }

        try {
            const models = await this.apiRequest('/');
            
            // Cache the result
            this.cache.set(cacheKey, {
                data: models,
                timestamp: Date.now()
            });
            
            return models;
        } catch (error) {
            console.error('Failed to fetch models:', error);
            
            // Return cached data if available, otherwise empty array
            if (this.cache.has(cacheKey)) {
                return this.cache.get(cacheKey).data;
            }
            return [];
        }
    }

    async getModel(modelId) {
        try {
            return await this.apiRequest(`/${modelId}`);
        } catch (error) {
            console.error(`Failed to fetch model ${modelId}:`, error);
            throw error;
        }
    }

    async createModel(modelData) {
        try {
            const result = await this.apiRequest('/', {
                method: 'POST',
                body: JSON.stringify(modelData)
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error('Failed to create model:', error);
            throw error;
        }
    }

    async updateModel(modelId, updates) {
        try {
            const result = await this.apiRequest(`/${modelId}`, {
                method: 'PUT',
                body: JSON.stringify(updates)
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error(`Failed to update model ${modelId}:`, error);
            throw error;
        }
    }

    async deleteModel(modelId) {
        try {
            const result = await this.apiRequest(`/${modelId}`, {
                method: 'DELETE'
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error(`Failed to delete model ${modelId}:`, error);
            throw error;
        }
    }

    // ========================================================================
    // MODEL OPERATIONS
    // ========================================================================

    async activateModel(modelId) {
        try {
            const result = await this.apiRequest(`/${modelId}/activate`, {
                method: 'POST'
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error(`Failed to activate model ${modelId}:`, error);
            throw error;
        }
    }

    async pauseModel(modelId) {
        try {
            const result = await this.apiRequest(`/${modelId}/pause`, {
                method: 'POST'
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error(`Failed to pause model ${modelId}:`, error);
            throw error;
        }
    }

    async bulkActivateModels(modelIds) {
        try {
            const result = await this.apiRequest('/bulk/activate', {
                method: 'POST',
                body: JSON.stringify(modelIds)
            });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return result;
        } catch (error) {
            console.error('Failed to bulk activate models:', error);
            throw error;
        }
    }

    // ========================================================================
    // FILTERING AND SEARCH
    // ========================================================================

    async getModelsByStatus(status) {
        try {
            return await this.apiRequest(`/?status=${status}`);
        } catch (error) {
            console.error(`Failed to fetch models by status ${status}:`, error);
            return [];
        }
    }

    async getModelsByProvider(provider) {
        try {
            return await this.apiRequest(`/?provider=${encodeURIComponent(provider)}`);
        } catch (error) {
            console.error(`Failed to fetch models by provider ${provider}:`, error);
            return [];
        }
    }

    async getModelsByType(modelType) {
        try {
            return await this.apiRequest(`/?model_type=${modelType}`);
        } catch (error) {
            console.error(`Failed to fetch models by type ${modelType}:`, error);
            return [];
        }
    }

    async getProviders() {
        try {
            return await this.apiRequest('/providers');
        } catch (error) {
            console.error('Failed to fetch providers:', error);
            return [];
        }
    }

    async getModelTypes() {
        try {
            return await this.apiRequest('/types');
        } catch (error) {
            console.error('Failed to fetch model types:', error);
            return [];
        }
    }

    // ========================================================================
    // STATISTICS AND METRICS
    // ========================================================================

    async getModelStats() {
        try {
            return await this.apiRequest('/stats/overview');
        } catch (error) {
            console.error('Failed to fetch model stats:', error);
            // Return default stats on error
            return {
                total_models: 0,
                active_models: 0,
                total_requests: 0,
                requests_last_hour: 0,
                average_response_time: 0,
                average_success_rate: 100
            };
        }
    }

    async logModelUsage(modelId, usageData) {
        try {
            return await this.apiRequest('/usage-log', {
                method: 'POST',
                body: JSON.stringify({
                    model_id: modelId,
                    ...usageData
                })
            });
        } catch (error) {
            console.error('Failed to log model usage:', error);
            // Don't throw error for logging failures
            return null;
        }
    }

    // ========================================================================
    // CACHE MANAGEMENT
    // ========================================================================

    clearCache() {
        this.cache.clear();
    }

    clearCacheForKey(key) {
        this.cache.delete(key);
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    formatModelForDisplay(model) {
        return {
            id: model.id,
            model: model.name,
            type: model.model_type,
            status: model.status,
            provider: model.provider,
            version: model.version,
            requests: model.total_requests.toLocaleString(),
            responseTime: `${model.average_response_time.toFixed(1)}s`,
            successRate: `${model.success_rate.toFixed(1)}%`,
            lastUsed: model.last_used_at ? new Date(model.last_used_at).toLocaleString() : 'Never'
        };
    }

    async exportData() {
        try {
            return await this.apiRequest('/export');
        } catch (error) {
            console.error('Failed to export data:', error);
            throw error;
        }
    }

    async healthCheck() {
        try {
            return await this.apiRequest('/health');
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    // ========================================================================
    // REAL-TIME UPDATES (OPTIONAL)
    // ========================================================================

    startRealTimeUpdates(callback, interval = 30000) {
        this.updateInterval = setInterval(async () => {
            try {
                const models = await this.getAllModels(false); // Force refresh
                const stats = await this.getModelStats();
                callback({ models, stats });
            } catch (error) {
                console.error('Real-time update failed:', error);
            }
        }, interval);
    }

    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}

// ============================================================================
// UPDATED ADMIN PAGE INTEGRATION
// ============================================================================

// Enhanced AdminPage methods for database integration
class AdminPageDatabaseIntegration {
    constructor() {
        this.aiModelsService = new AIModelsService();
        this.isLoadingModels = false;
    }

    async initializeAIModels() {
        try {
            log_info("Initializing AI Models section with database");
            
            // Show loading state
            this.setModelsLoading(true);
            
            // Load models from database
            await this.loadModelsFromDatabase();
            
            // Load and update statistics
            await this.loadModelStats();
            
            // Set up event handlers
            this.bindAIModelsEvents();
            
            // Start real-time updates
            this.startModelUpdates();
            
            log_success("AI Models section initialized with database successfully");
        } catch (error) {
            log_error("Failed to initialize AI Models section: " + error.message);
            this.showError("Failed to load AI models. Please check your connection and try again.");
        } finally {
            this.setModelsLoading(false);
        }
    }

    async loadModelsFromDatabase() {
        try {
            const models = await this.aiModelsService.getAllModels();
            
            // Convert database models to display format
            this.modelsData = models.map(model => this.aiModelsService.formatModelForDisplay(model));
            
            // Populate the table
            this.populateAIModelsTable();
            
            log_info(`Loaded ${models.length} models from database`);
        } catch (error) {
            log_error("Failed to load models from database: " + error.message);
            throw error;
        }
    }

    async loadModelStats() {
        try {
            const stats = await this.aiModelsService.getModelStats();
            this.updateAIModelsStatsFromDatabase(stats);
        } catch (error) {
            log_error("Failed to load model stats: " + error.message);
        }
    }

    updateAIModelsStatsFromDatabase(stats) {
        const statsCards = document.querySelectorAll('#ai-models .stat-card');
        if (statsCards.length >= 4) {
            // Update stats cards with real data
            if (statsCards[0]) {
                statsCards[0].querySelector('.stat-value').textContent = stats.active_models;
            }
            if (statsCards[1]) {
                statsCards[1].querySelector('.stat-value').textContent = stats.total_requests.toLocaleString();
            }
            if (statsCards[2]) {
                statsCards[2].querySelector('.stat-value').textContent = `${stats.average_response_time.toFixed(1)}s`;
            }
            if (statsCards[3]) {
                statsCards[3].querySelector('.stat-value').textContent = `${stats.average_success_rate.toFixed(1)}%`;
            }
        }
    }

    async activateModel(modelId) {
        try {
            this.showToast(`Activating model...`, 'info');
            await this.aiModelsService.activateModel(modelId);
            await this.refreshModels();
            this.showToast(`Model activated successfully`, 'success');
        } catch (error) {
            this.showToast(`Failed to activate model: ${error.message}`, 'error');
        }
    }

    async pauseModel(modelId) {
        try {
            this.showToast(`Pausing model...`, 'info');
            await this.aiModelsService.pauseModel(modelId);
            await this.refreshModels();
            this.showToast(`Model paused successfully`, 'warning');
        } catch (error) {
            this.showToast(`Failed to pause model: ${error.message}`, 'error');
        }
    }

    async deleteModel(modelId) {
        const modelName = this.modelsData.find(m => m.id === modelId)?.model || modelId;
        
        if (confirm(`Are you sure you want to delete model "${modelName}"? This action cannot be undone.`)) {
            try {
                this.showToast(`Deleting model...`, 'info');
                await this.aiModelsService.deleteModel(modelId);
                await this.refreshModels();
                this.showToast(`Model "${modelName}" has been deleted`, 'success');
            } catch (error) {
                this.showToast(`Failed to delete model: ${error.message}`, 'error');
            }
        }
    }

    async refreshModels() {
        try {
            this.showToast('Refreshing models...', 'info');
            await this.loadModelsFromDatabase();
            await this.loadModelStats();
            this.showToast('Models refreshed successfully', 'success');
        } catch (error) {
            this.showToast(`Failed to refresh models: ${error.message}`, 'error');
        }
    }

    setModelsLoading(loading) {
        this.isLoadingModels = loading;
        const tbody = document.getElementById('ai-models-tbody');
        if (tbody) {
            if (loading) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align: center; padding: 2rem;">
                            <div class="loading-spinner">Loading models from database...</div>
                        </td>
                    </tr>
                `;
            }
        }
    }

    showError(message) {
        const tbody = document.getElementById('ai-models-tbody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 2rem; color: var(--danger);">
                        <div>❌ ${message}</div>
                        <button onclick="adminPage.refreshModels()" class="btn btn-primary mt-2">🔄 Retry</button>
                    </td>
                </tr>
            `;
        }
    }

    startModelUpdates() {
        // Start real-time updates every 30 seconds
        this.aiModelsService.startRealTimeUpdates(({ models, stats }) => {
            this.modelsData = models.map(model => this.aiModelsService.formatModelForDisplay(model));
            this.populateAIModelsTable();
            this.updateAIModelsStatsFromDatabase(stats);
        }, 30000);
    }

    stopModelUpdates() {
        this.aiModelsService.stopRealTimeUpdates();
    }

    destroy() {
        this.stopModelUpdates();
        super.destroy();
    }
}

// Global service instance
window.aiModelsService = new AIModelsService();