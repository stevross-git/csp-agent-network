/**
 * Enhanced CSP System - Backend API Service
 * Integration with CSP backend authentication and APIs
 */

import { authMiddleware } from '../middleware/authMiddleware';
import { cspApiConfig } from '../config/authConfig';

class CSPApiService {
    constructor() {
        this.baseUrl = cspApiConfig.baseUrl;
        this.endpoints = cspApiConfig.endpoints;
    }

    /**
     * Make authenticated API request
     */
    async apiRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        try {
            const response = await authMiddleware.secureApiCall(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                ...options
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `API request failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * Authentication methods
     */
    async loginWithAzure(azureToken) {
        return this.apiRequest(`${this.endpoints.auth}/azure-login`, {
            method: 'POST',
            body: JSON.stringify({ azure_token: azureToken })
        });
    }

    async refreshToken(refreshToken) {
        return this.apiRequest(`${this.endpoints.auth}/refresh`, {
            method: 'POST',
            body: JSON.stringify({ refresh_token: refreshToken })
        });
    }

    async logout() {
        return this.apiRequest(`${this.endpoints.auth}/logout`, {
            method: 'POST'
        });
    }

    async getCurrentUser() {
        return this.apiRequest(`${this.endpoints.auth}/me`);
    }

    /**
     * Design management methods
     */
    async getDesigns() {
        return this.apiRequest(this.endpoints.designs);
    }

    async getDesign(designId) {
        return this.apiRequest(`${this.endpoints.designs}/${designId}`);
    }

    async createDesign(designData) {
        return this.apiRequest(this.endpoints.designs, {
            method: 'POST',
            body: JSON.stringify(designData)
        });
    }

    async updateDesign(designId, designData) {
        return this.apiRequest(`${this.endpoints.designs}/${designId}`, {
            method: 'PUT',
            body: JSON.stringify(designData)
        });
    }

    async deleteDesign(designId) {
        return this.apiRequest(`${this.endpoints.designs}/${designId}`, {
            method: 'DELETE'
        });
    }

    /**
     * Component methods
     */
    async getComponents() {
        return this.apiRequest(this.endpoints.components);
    }

    async getComponent(componentType) {
        return this.apiRequest(`${this.endpoints.components}/${componentType}`);
    }

    /**
     * Execution methods
     */
    async executeDesign(designId, parameters = {}) {
        return this.apiRequest(`${this.endpoints.executions}/execute`, {
            method: 'POST',
            body: JSON.stringify({ design_id: designId, parameters })
        });
    }

    async getExecutionStatus(executionId) {
        return this.apiRequest(`${this.endpoints.executions}/${executionId}/status`);
    }

    async getExecutionResults(executionId) {
        return this.apiRequest(`${this.endpoints.executions}/${executionId}/results`);
    }

    /**
     * WebSocket connection
     */
    createWebSocketConnection() {
        const wsUrl = `${this.baseUrl.replace('http', 'ws')}${this.endpoints.websocket}`;
        
        return authMiddleware.getValidToken().then(tokenResponse => {
            const ws = new WebSocket(`${wsUrl}?token=${tokenResponse.accessToken}`);
            
            ws.addEventListener('open', () => {
                console.log('CSP WebSocket connected');
            });

            ws.addEventListener('error', (error) => {
                console.error('CSP WebSocket error:', error);
            });

            return ws;
        });
    }
}

export const cspApiService = new CSPApiService();
export default cspApiService;