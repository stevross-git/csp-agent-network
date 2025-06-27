// frontend/js/pages/admin/roleManager.js
/**
 * Role and Permission Management Module
 * Provides CRUD operations for roles and assignment of permissions.
 * Designed for the Enhanced CSP Admin Portal.
 */

class RoleManager {
    constructor() {
        this.roles = [];
        this.section = null;
        this.tbody = null;
        this.apiBaseUrl = this.getApiBaseUrl();
        this.authToken = this.getAuthToken();
        this.loading = false;
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
            if (response.status === 401) {
                this.handleAuthError();
                throw new Error('Authentication required');
            }
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        return response.json();
    }

    handleAuthError() {
        localStorage.removeItem('csp_auth_token');
        sessionStorage.removeItem('csp_auth_token');
        if (window.location.pathname !== '/login') {
            this.showNotification('Session expired. Please log in again.', 'warning');
        }
    }

    showNotification(message, type = 'info', duration = 4000) {
        const note = document.createElement('div');
        note.className = `notification notification-${type}`;
        note.textContent = message;
        document.body.appendChild(note);
        setTimeout(() => note.remove(), duration);
    }

    async init() {
        this.section = document.getElementById('roles');
        if (!this.section) return;
        await this.loadRoles();
        this.render();
        this.attachEvents();
    }

    async loadRoles() {
        this.loading = true;
        try {
            const data = await this.apiRequest('/api/roles');
            this.roles = data.roles || data.items || [];
        } catch (err) {
            console.warn('Failed to load roles, using defaults', err);
            this.roles = this.getDefaultRoles();
        }
        this.loading = false;
    }

    getDefaultRoles() {
        return [
            { id: 'admin', name: 'Administrator', permissions: ['manage_users', 'manage_system'] },
            { id: 'user', name: 'User', permissions: ['view_system'] }
        ];
    }

    render() {
        this.section.innerHTML = `
            <div class="roles-dashboard">
                <div class="table-header">
                    <div class="table-title">Roles</div>
                    <div class="table-actions">
                        <button class="btn btn-primary" id="add-role-btn"><i class="fas fa-plus"></i> Add Role</button>
                    </div>
                </div>
                <div class="table-content">
                    <table class="role-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Permissions</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="roles-tbody"></tbody>
                    </table>
                </div>
            </div>
        `;
        this.tbody = this.section.querySelector('#roles-tbody');
        this.renderRows();
    }

    renderRows() {
        if (!this.tbody) return;
        if (this.roles.length === 0) {
            this.tbody.innerHTML = `<tr><td colspan="3" class="empty-state">No roles defined</td></tr>`;
            return;
        }
        this.tbody.innerHTML = this.roles.map(role => `
            <tr data-role-id="${role.id}">
                <td>${role.name}</td>
                <td>${this.formatPermissions(role.permissions)}</td>
                <td>
                    <button class="btn btn-sm btn-secondary edit-role-btn" title="Edit"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-sm btn-danger delete-role-btn" title="Delete"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    }

    formatPermissions(perms = []) {
        return perms.map(p => `<span class="role-badge">${p}</span>`).join(' ');
    }

    attachEvents() {
        this.section.addEventListener('click', (e) => {
            const row = e.target.closest('tr[data-role-id]');
            if (!row) return;
            const roleId = row.getAttribute('data-role-id');
            if (e.target.closest('.delete-role-btn')) {
                this.deleteRole(roleId);
            } else if (e.target.closest('.edit-role-btn')) {
                this.editRole(roleId);
            }
        });
        const addBtn = this.section.querySelector('#add-role-btn');
        if (addBtn) addBtn.addEventListener('click', () => this.openAddRoleModal());
    }

    async addRole(data) {
        const res = await this.apiRequest('/api/roles', { method: 'POST', body: JSON.stringify(data) });
        this.showNotification('Role created', 'success');
        await this.loadRoles();
        this.renderRows();
        return res;
    }

    async updateRole(roleId, data) {
        const res = await this.apiRequest(`/api/roles/${roleId}`, { method: 'PUT', body: JSON.stringify(data) });
        this.showNotification('Role updated', 'success');
        await this.loadRoles();
        this.renderRows();
        return res;
    }

    async deleteRole(roleId) {
        if (!confirm('Delete this role?')) return;
        await this.apiRequest(`/api/roles/${roleId}`, { method: 'DELETE' });
        this.showNotification('Role deleted', 'success');
        await this.loadRoles();
        this.renderRows();
    }

    editRole(roleId) {
        const role = this.roles.find(r => r.id === roleId);
        if (!role) return;
        this.openEditRoleModal(role);
    }

    openAddRoleModal() {
        if (window.adminModalManager && window.adminModalManager.openAddRoleModal) {
            window.adminModalManager.openAddRoleModal();
        } else {
            alert('Add role modal not implemented');
        }
    }

    openEditRoleModal(role) {
        if (window.adminModalManager && window.adminModalManager.openEditRoleModal) {
            window.adminModalManager.openEditRoleModal(role);
        } else {
            alert('Edit role modal not implemented');
        }
    }
}

let roleManager;
function initializeRoleManager() {
    if (!roleManager) {
        roleManager = new RoleManager();
        roleManager.init();
        window.roleManager = roleManager;
    }
    return roleManager;
}

if (document.readyState !== 'loading') {
    initializeRoleManager();
} else {
    document.addEventListener('DOMContentLoaded', initializeRoleManager);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RoleManager, initializeRoleManager };
}
