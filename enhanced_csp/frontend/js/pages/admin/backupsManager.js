/**
 * Backup & Recovery Manager
 * Handles listing, creating and restoring backups via the backend API.
 */

class BackupsManager {
    constructor() {
        this.section = null;
        this.tableBody = null;
        this.apiBaseUrl = this.getApiBaseUrl();
        this.authToken = this.getAuthToken();
        this.backups = [];
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
        return response.json();
    }

    async init() {
        this.section = document.getElementById('backups');
        if (!this.section) return;
        await this.loadBackups();
        this.render();
        this.attachEvents();
    }

    async loadBackups() {
        try {
            this.backups = await this.apiRequest('/api/backups');
        } catch (err) {
            console.error('Failed to load backups', err);
            this.backups = [];
        }
    }

    render() {
        this.section.innerHTML = `
            <div class="backup-section">
                <div class="backup-actions">
                    <button class="btn btn-primary" id="create-backup-btn"><i class="fas fa-download"></i> Create Backup</button>
                    <button class="btn btn-secondary" id="refresh-backups-btn"><i class="fas fa-sync-alt"></i> Refresh</button>
                </div>
                <table class="backup-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Date</th>
                            <th>Size</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="backup-tbody">
                        ${this.backups.map(b => `
                            <tr>
                                <td>${b.name}</td>
                                <td>${new Date(b.created_at).toLocaleString()}</td>
                                <td>${(b.size / 1024 / 1024).toFixed(2)} MB</td>
                                <td><button class="btn btn-outline" data-restore="${b.id}">Restore</button></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        this.tableBody = this.section.querySelector('#backup-tbody');
    }

    attachEvents() {
        this.section.querySelector('#create-backup-btn')?.addEventListener('click', () => this.createBackup());
        this.section.querySelector('#refresh-backups-btn')?.addEventListener('click', () => this.refresh());
        this.section.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-restore]');
            if (btn) {
                const id = btn.getAttribute('data-restore');
                this.restoreBackup(id);
            }
        });
    }

    async refresh() {
        await this.loadBackups();
        this.render();
    }

    async createBackup() {
        try {
            await this.apiRequest('/api/backups', { method: 'POST' });
            await this.refresh();
        } catch (err) {
            console.error('Failed to create backup', err);
        }
    }

    async restoreBackup(id) {
        if (!confirm('Restore this backup? This will overwrite current data.')) return;
        try {
            await this.apiRequest(`/api/backups/${id}/restore`, { method: 'POST' });
            alert('Backup restored successfully');
        } catch (err) {
            console.error('Failed to restore backup', err);
        }
    }
}

const backupsManager = new BackupsManager();

document.addEventListener('DOMContentLoaded', () => backupsManager.init());
