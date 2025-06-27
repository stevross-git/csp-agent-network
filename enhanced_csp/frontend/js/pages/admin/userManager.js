// frontend/js/pages/admin/userManager.js
/**
 * Enhanced User Manager with Database Integration
 * Connects to the Enhanced CSP Backend API for user management
 */

class UserManager {
    constructor() {
        this.users = [];
        this.section = null;
        this.tbody = null;
        this.apiBaseUrl = this.getApiBaseUrl();
        this.authToken = this.getAuthToken();
        this.loading = false;
        this.pagination = {
            page: 1,
            limit: 50,
            total: 0,
            totalPages: 0
        };
        this.filters = {
            status: '',
            role: '',
            search: ''
        };
    }

    getApiBaseUrl() {
        // Get API URL from environment or fallback to default
        // Check window/global variables first
        if (window.REACT_APP_CSP_API_URL) {
            return window.REACT_APP_CSP_API_URL;
        }
        
        // Check if we're in a build environment with injected variables
        if (typeof REACT_APP_CSP_API_URL !== 'undefined') {
            return REACT_APP_CSP_API_URL;
        }
        
        // Check meta tags (often used for environment config)
        const metaApiUrl = document.querySelector('meta[name="api-base-url"]');
        if (metaApiUrl) {
            return metaApiUrl.getAttribute('content');
        }
        
        // Check if running locally (development)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }
        
        // Production fallback - same origin
        return window.location.origin.replace(':3000', ':8000');
    }

    getAuthToken() {
        // Get JWT token from localStorage or sessionStorage
        return localStorage.getItem('csp_auth_token') || 
               sessionStorage.getItem('csp_auth_token');
    }

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
            }
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                if (response.status === 401) {
                    this.handleAuthError();
                    throw new Error('Authentication required');
                }
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            this.showNotification('API request failed: ' + error.message, 'error');
            throw error;
        }
    }

    handleAuthError() {
        // Clear invalid token
        localStorage.removeItem('csp_auth_token');
        sessionStorage.removeItem('csp_auth_token');
        
        // Redirect to login or show login modal
        if (window.location.pathname !== '/login') {
            this.showNotification('Session expired. Please log in again.', 'warning');
            // window.location.href = '/login';
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        // Create or update notification system
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    async init() {
        this.section = document.getElementById('users');
        if (!this.section) {
            console.error('Users section not found');
            return;
        }
        
        await this.loadUsers();
        this.render();
        this.attachEvents();
    }

    async loadUsers(page = 1) {
        if (this.loading) return;
        
        this.loading = true;
        this.showLoadingState();

        try {
            // Build query parameters
            const params = new URLSearchParams({
                page: page.toString(),
                limit: this.pagination.limit.toString(),
                ...(this.filters.status && { status: this.filters.status }),
                ...(this.filters.role && { role: this.filters.role }),
                ...(this.filters.search && { search: this.filters.search })
            });

            const response = await this.apiRequest(`/api/admin/users?${params}`);
            
            this.users = response.users || response.items || [];
            this.pagination = {
                page: response.page || 1,
                limit: response.limit || 50,
                total: response.total || this.users.length,
                totalPages: response.totalPages || Math.ceil((response.total || this.users.length) / (response.limit || 50))
            };

            this.hideLoadingState();
            this.renderRows();
            this.renderPagination();
            
        } catch (error) {
            console.error('Failed to load users:', error);
            this.hideLoadingState();
            
            // Fallback to default users for demo purposes
            if (!this.authToken) {
                this.users = this.getDefaultUsers();
                this.renderRows();
                this.showNotification('Using demo data. Please log in for full functionality.', 'warning');
            }
        } finally {
            this.loading = false;
        }
    }

    getDefaultUsers() {
        return [
            { 
                id: '1', 
                full_name: 'John Smith', 
                email: 'john@company.com', 
                roles: ['admin'], 
                is_active: true, 
                last_login: '2024-01-15T09:30:00Z',
                created_at: '2024-01-01T00:00:00Z'
            },
            { 
                id: '2', 
                full_name: 'Sarah Johnson', 
                email: 'sarah@company.com', 
                roles: ['user'], 
                is_active: true, 
                last_login: '2024-01-15T14:22:00Z',
                created_at: '2024-01-02T00:00:00Z'
            },
            { 
                id: '3', 
                full_name: 'Mike Chen', 
                email: 'mike@company.com', 
                roles: ['developer'], 
                is_active: false, 
                last_login: '2024-01-10T16:30:00Z',
                created_at: '2024-01-03T00:00:00Z'
            }
        ];
    }

    showLoadingState() {
        if (this.tbody) {
            this.tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 2rem;">
                        <div class="loading-spinner">
                            <i class="fas fa-spinner fa-spin fa-2x"></i>
                            <p>Loading users...</p>
                        </div>
                    </td>
                </tr>
            `;
        }
    }

    hideLoadingState() {
        // Loading state will be replaced by renderRows()
    }

    render() {
        this.section.innerHTML = `
            <div class="user-management-container">
                <h2 class="page-title">
                    <i class="fas fa-users"></i> User Management
                    <span class="user-count">${this.pagination.total} users</span>
                </h2>
                
                <!-- Filters and Search -->
                <div class="user-filters">
                    <div class="filter-group">
                        <input type="text" id="user-search" placeholder="Search users..." 
                               value="${this.filters.search}" class="form-control">
                        <select id="status-filter" class="form-control">
                            <option value="">All Status</option>
                            <option value="active" ${this.filters.status === 'active' ? 'selected' : ''}>Active</option>
                            <option value="inactive" ${this.filters.status === 'inactive' ? 'selected' : ''}>Inactive</option>
                        </select>
                        <select id="role-filter" class="form-control">
                            <option value="">All Roles</option>
                            <option value="admin" ${this.filters.role === 'admin' ? 'selected' : ''}>Admin</option>
                            <option value="user" ${this.filters.role === 'user' ? 'selected' : ''}>User</option>
                            <option value="developer" ${this.filters.role === 'developer' ? 'selected' : ''}>Developer</option>
                        </select>
                        <button id="apply-filters-btn" class="btn btn-secondary">
                            <i class="fas fa-filter"></i> Apply
                        </button>
                        <button id="reset-filters-btn" class="btn btn-outline">
                            <i class="fas fa-undo"></i> Reset
                        </button>
                    </div>
                </div>

                <!-- User Table -->
                <div class="data-table">
                    <div class="table-header">
                        <div class="table-title">Users</div>
                        <div class="table-actions">
                            <button class="btn btn-primary" id="add-user-btn">
                                <i class="fas fa-user-plus"></i> Add User
                            </button>
                            <button class="btn btn-secondary" id="export-users-btn">
                                <i class="fas fa-download"></i> Export
                            </button>
                            <button class="btn btn-outline" id="refresh-users-btn">
                                <i class="fas fa-sync-alt"></i> Refresh
                            </button>
                        </div>
                    </div>
                    <div class="table-content">
                        <table class="user-table">
                            <thead>
                                <tr>
                                    <th>
                                        <input type="checkbox" id="select-all-users">
                                    </th>
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Role</th>
                                    <th>Status</th>
                                    <th>Last Login</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="users-tbody"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Pagination -->
                <div id="user-pagination" class="pagination-container"></div>
            </div>
        `;
        
        this.tbody = this.section.querySelector('#users-tbody');
        this.renderRows();
        this.renderPagination();
    }

    renderRows() {
        if (!this.tbody) return;
        
        if (this.users.length === 0) {
            this.tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="empty-state">
                        <div class="empty-state-content">
                            <i class="fas fa-users fa-3x"></i>
                            <h3>No users found</h3>
                            <p>Get started by adding your first user.</p>
                            <button class="btn btn-primary" onclick="userManager.openAddUserModal()">
                                <i class="fas fa-user-plus"></i> Add User
                            </button>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        this.tbody.innerHTML = this.users.map(user => `
            <tr data-user-id="${user.id}" class="user-row">
                <td>
                    <input type="checkbox" class="user-checkbox" value="${user.id}">
                </td>
                <td>
                    <div class="user-info">
                        <div class="user-avatar">
                            ${user.full_name ? user.full_name.charAt(0).toUpperCase() : 'U'}
                        </div>
                        <div class="user-details">
                            <div class="user-name">${user.full_name || 'Unknown User'}</div>
                            <div class="user-id">ID: ${user.id}</div>
                        </div>
                    </div>
                </td>
                <td>
                    <a href="mailto:${user.email}" class="user-email">${user.email}</a>
                    ${user.is_email_verified ? '<i class="fas fa-check-circle text-success" title="Email verified"></i>' : '<i class="fas fa-exclamation-triangle text-warning" title="Email not verified"></i>'}
                </td>
                <td>
                    <div class="user-roles">
                        ${this.renderUserRoles(user.roles || ['user'])}
                    </div>
                </td>
                <td>
                    <span class="status-badge ${user.is_active ? 'status-active' : 'status-inactive'}">
                        ${user.is_active ? 'Active' : 'Inactive'}
                    </span>
                </td>
                <td>
                    <span class="last-login" title="${user.last_login}">
                        ${user.last_login ? this.formatDateTime(user.last_login) : 'Never'}
                    </span>
                </td>
                <td>
                    <span class="created-date" title="${user.created_at}">
                        ${this.formatDate(user.created_at)}
                    </span>
                </td>
                <td>
                    <div class="action-buttons">
                        <button class="btn btn-sm btn-outline view-user-btn" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-secondary edit-user-btn" title="Edit User">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-sm btn-danger delete-user-btn" title="Delete User">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    renderUserRoles(roles) {
        return roles.map(role => `
            <span class="role-badge role-${role}">${role}</span>
        `).join('');
    }

    renderPagination() {
        const paginationContainer = this.section.querySelector('#user-pagination');
        if (!paginationContainer || this.pagination.totalPages <= 1) {
            if (paginationContainer) paginationContainer.innerHTML = '';
            return;
        }

        const { page, totalPages } = this.pagination;
        const maxVisible = 5;
        let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
        let endPage = Math.min(totalPages, startPage + maxVisible - 1);
        
        if (endPage - startPage + 1 < maxVisible) {
            startPage = Math.max(1, endPage - maxVisible + 1);
        }

        let paginationHTML = `
            <div class="pagination">
                <button class="btn btn-sm btn-outline" ${page === 1 ? 'disabled' : ''} 
                        onclick="userManager.loadUsers(1)">
                    <i class="fas fa-angle-double-left"></i>
                </button>
                <button class="btn btn-sm btn-outline" ${page === 1 ? 'disabled' : ''} 
                        onclick="userManager.loadUsers(${page - 1})">
                    <i class="fas fa-angle-left"></i>
                </button>
        `;

        for (let i = startPage; i <= endPage; i++) {
            paginationHTML += `
                <button class="btn btn-sm ${i === page ? 'btn-primary' : 'btn-outline'}" 
                        onclick="userManager.loadUsers(${i})">
                    ${i}
                </button>
            `;
        }

        paginationHTML += `
                <button class="btn btn-sm btn-outline" ${page === totalPages ? 'disabled' : ''} 
                        onclick="userManager.loadUsers(${page + 1})">
                    <i class="fas fa-angle-right"></i>
                </button>
                <button class="btn btn-sm btn-outline" ${page === totalPages ? 'disabled' : ''} 
                        onclick="userManager.loadUsers(${totalPages})">
                    <i class="fas fa-angle-double-right"></i>
                </button>
            </div>
            <div class="pagination-info">
                Showing ${((page - 1) * this.pagination.limit) + 1} to ${Math.min(page * this.pagination.limit, this.pagination.total)} of ${this.pagination.total} users
            </div>
        `;

        paginationContainer.innerHTML = paginationHTML;
    }

    attachEvents() {
        // Add user button
        const addBtn = this.section.querySelector('#add-user-btn');
        if (addBtn) {
            addBtn.addEventListener('click', () => this.openAddUserModal());
        }

        // Refresh button
        const refreshBtn = this.section.querySelector('#refresh-users-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadUsers(this.pagination.page));
        }

        // Export button
        const exportBtn = this.section.querySelector('#export-users-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportUsers());
        }

        // Filter controls
        const searchInput = this.section.querySelector('#user-search');
        const statusFilter = this.section.querySelector('#status-filter');
        const roleFilter = this.section.querySelector('#role-filter');
        const applyFiltersBtn = this.section.querySelector('#apply-filters-btn');
        const resetFiltersBtn = this.section.querySelector('#reset-filters-btn');

        if (applyFiltersBtn) {
            applyFiltersBtn.addEventListener('click', () => {
                this.filters.search = searchInput?.value || '';
                this.filters.status = statusFilter?.value || '';
                this.filters.role = roleFilter?.value || '';
                this.loadUsers(1);
            });
        }

        if (resetFiltersBtn) {
            resetFiltersBtn.addEventListener('click', () => {
                this.filters = { status: '', role: '', search: '' };
                if (searchInput) searchInput.value = '';
                if (statusFilter) statusFilter.value = '';
                if (roleFilter) roleFilter.value = '';
                this.loadUsers(1);
            });
        }

        // Search on Enter key
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.filters.search = e.target.value;
                    this.loadUsers(1);
                }
            });
        }

        // Row action buttons
        this.section.addEventListener('click', (e) => {
            const userId = e.target.closest('tr')?.dataset.userId;
            if (!userId) return;

            if (e.target.closest('.delete-user-btn')) {
                this.deleteUser(userId);
            } else if (e.target.closest('.edit-user-btn')) {
                this.editUser(userId);
            } else if (e.target.closest('.view-user-btn')) {
                this.viewUser(userId);
            }
        });

        // Select all checkbox
        const selectAllCheckbox = this.section.querySelector('#select-all-users');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => {
                const checkboxes = this.section.querySelectorAll('.user-checkbox');
                checkboxes.forEach(cb => cb.checked = e.target.checked);
            });
        }
    }

    async addUser(userData) {
        try {
            const response = await this.apiRequest('/api/auth/local/register', {
                method: 'POST',
                body: JSON.stringify(userData)
            });

            this.showNotification('User created successfully!', 'success');
            await this.loadUsers(this.pagination.page);
            return response;
        } catch (error) {
            console.error('Failed to create user:', error);
            throw error;
        }
    }

    async editUser(userId) {
        const user = this.users.find(u => u.id === userId);
        if (!user) {
            this.showNotification('User not found', 'error');
            return;
        }

        // Open edit modal with user data
        this.openEditUserModal(user);
    }

    async updateUser(userId, userData) {
        try {
            const response = await this.apiRequest(`/api/admin/users/${userId}`, {
                method: 'PUT',
                body: JSON.stringify(userData)
            });

            this.showNotification('User updated successfully!', 'success');
            await this.loadUsers(this.pagination.page);
            return response;
        } catch (error) {
            console.error('Failed to update user:', error);
            throw error;
        }
    }

    async deleteUser(userId) {
        const user = this.users.find(u => u.id === userId);
        if (!user) {
            this.showNotification('User not found', 'error');
            return;
        }

        if (!confirm(`Are you sure you want to delete user "${user.full_name}" (${user.email})?\n\nThis action cannot be undone.`)) {
            return;
        }

        try {
            await this.apiRequest(`/api/admin/users/${userId}`, {
                method: 'DELETE'
            });

            this.showNotification('User deleted successfully', 'success');
            await this.loadUsers(this.pagination.page);
        } catch (error) {
            console.error('Failed to delete user:', error);
        }
    }

    viewUser(userId) {
        const user = this.users.find(u => u.id === userId);
        if (!user) {
            this.showNotification('User not found', 'error');
            return;
        }

        // Open view modal with user details
        this.openUserDetailsModal(user);
    }

    openAddUserModal() {
        // Implementation depends on your modal system
        if (window.adminModalManager && window.adminModalManager.openAddUserModal) {
            window.adminModalManager.openAddUserModal();
        } else {
            console.log('Add user modal not implemented');
        }
    }

    openEditUserModal(user) {
        // Implementation depends on your modal system
        if (window.adminModalManager && window.adminModalManager.openEditUserModal) {
            window.adminModalManager.openEditUserModal(user);
        } else {
            console.log('Edit user modal not implemented', user);
        }
    }

    openUserDetailsModal(user) {
        // Implementation depends on your modal system
        if (window.adminModalManager && window.adminModalManager.openUserDetailsModal) {
            window.adminModalManager.openUserDetailsModal(user);
        } else {
            console.log('User details modal not implemented', user);
        }
    }

    async exportUsers() {
        try {
            const response = await this.apiRequest('/api/admin/users/export');
            
            // Create download link
            const blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `users_export_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            this.showNotification('Users exported successfully', 'success');
        } catch (error) {
            console.error('Failed to export users:', error);
        }
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleDateString();
    }

    formatDateTime(dateString) {
        if (!dateString) return 'Never';
        return new Date(dateString).toLocaleString();
    }
}

// Global instance and initialization
let userManager;

function initializeUserManager() {
    if (!userManager) {
        userManager = new UserManager();
        userManager.init();
        window.userManager = userManager;
    }
    return userManager;
}

// Auto-initialize when DOM is ready
if (document.readyState !== 'loading') {
    initializeUserManager();
} else {
    document.addEventListener('DOMContentLoaded', initializeUserManager);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UserManager, initializeUserManager };
}