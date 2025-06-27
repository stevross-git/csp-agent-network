/**
 * Infrastructure Manager
 * Handles infrastructure status and integrates backup operations.
 */
class InfrastructureManager {
    constructor() {
        this.section = null;
        this.status = null;
        this.apiBaseUrl = this.getApiBaseUrl();
        this.authToken = this.getAuthToken();
    }

    getApiBaseUrl() {
        if (window.REACT_APP_CSP_API_URL) return window.REACT_APP_CSP_API_URL;
        if (typeof REACT_APP_CSP_API_URL !== 'undefined') return REACT_APP_CSP_API_URL;
        const meta = document.querySelector('meta[name="api-base-url"]');
        if (meta) return meta.getAttribute('content');
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        return window.location.origin.replace(':3000', ':8000');
    }

    getAuthToken() {
        return localStorage.getItem('csp_auth_token') || sessionStorage.getItem('csp_auth_token');
    }

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
            }
        };
        const response = await fetch(url, { ...defaultOptions, ...options });
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        if (response.status === 204) return null;
        return response.json();
    }

    async init() {
        this.section = document.getElementById('infrastructure');
        if (!this.section) return;
        await this.loadStatus();
        this.render();
        this.attachEvents();
    }

    async loadStatus() {
        try {
            this.status = await this.apiRequest('/api/infrastructure/status');
        } catch (err) {
            console.warn('Failed to load infrastructure status', err);
            this.status = { message: 'Status unavailable' };
        }
    }

    render() {
        const statusText = this.status ? JSON.stringify(this.status, null, 2) : 'No status available';
        this.section.innerHTML = `
            <div class="infrastructure-dashboard">
                <div class="infra-actions">
                    <button class="btn btn-primary" id="infra-backup-btn">
                        <i class="fas fa-download"></i> Create Backup
                    </button>
                    <button class="btn btn-secondary" id="infra-refresh-btn">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
                <div class="infra-status">
                    <h3>Current Status</h3>
                    <pre>${statusText}</pre>
                </div>
            </div>
        `;
    }

    attachEvents() {
        this.section.querySelector('#infra-backup-btn')?.addEventListener('click', () => this.createBackup());
        this.section.querySelector('#infra-refresh-btn')?.addEventListener('click', () => this.refresh());
    }

    async refresh() {
        await this.loadStatus();
        this.render();
    }

    async createBackup() {
        try {
            await this.apiRequest('/api/backups', { method: 'POST' });
            alert('Backup created successfully');
        } catch (err) {
            console.error('Failed to create backup', err);
            alert('Failed to create backup');
        }
    }
}

const infrastructureManager = new InfrastructureManager();

document.addEventListener('DOMContentLoaded', () => infrastructureManager.init());
