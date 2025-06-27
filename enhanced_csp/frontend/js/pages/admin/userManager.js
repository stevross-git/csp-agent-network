class UserManager {
    constructor() {
        this.users = [];
        this.section = null;
        this.tbody = null;
    }

    init() {
        this.section = document.getElementById('users');
        if (!this.section) {
            console.error('Users section not found');
            return;
        }
        this.loadUsers();
        this.render();
        this.attachEvents();
    }

    defaultUsers() {
        return [
            { id: 1, name: 'John Smith', email: 'john@company.com', role: 'Admin', status: 'Active', lastLogin: '2024-01-15 09:30' },
            { id: 2, name: 'Sarah Johnson', email: 'sarah@company.com', role: 'User', status: 'Active', lastLogin: '2024-01-15 14:22' },
            { id: 3, name: 'Mike Chen', email: 'mike@company.com', role: 'Developer', status: 'Inactive', lastLogin: '2024-01-10 16:30' }
        ];
    }

    loadUsers() {
        try {
            const stored = localStorage.getItem('csp_users');
            if (stored) {
                this.users = JSON.parse(stored);
            } else {
                this.users = this.defaultUsers();
            }
        } catch (e) {
            console.warn('Failed to load stored users, using defaults');
            this.users = this.defaultUsers();
        }
    }

    saveUsers() {
        localStorage.setItem('csp_users', JSON.stringify(this.users));
    }

    render() {
        this.section.innerHTML = `
            <h2 class="mb-3"><i class="fas fa-users"></i> User Management</h2>
            <div class="data-table">
                <div class="table-header">
                    <div class="table-title">Users</div>
                    <div class="table-actions">
                        <button class="btn btn-primary" id="add-user-btn"><i class="fas fa-user-plus"></i> Add User</button>
                    </div>
                </div>
                <div class="table-content">
                    <table class="user-table">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Email</th>
                                <th>Role</th>
                                <th>Status</th>
                                <th>Last Login</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="users-tbody"></tbody>
                    </table>
                </div>
            </div>
        `;
        this.tbody = this.section.querySelector('#users-tbody');
        this.renderRows();
    }

    renderRows() {
        if (!this.tbody) return;
        if (this.users.length === 0) {
            this.tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:1rem;">No users found</td></tr>';
            return;
        }
        this.tbody.innerHTML = this.users.map(u => `
            <tr data-user-id="${u.id}">
                <td>${u.name}</td>
                <td>${u.email}</td>
                <td>${u.role}</td>
                <td><span class="status-badge ${u.status === 'Active' ? 'status-active' : 'status-inactive'}">${u.status}</span></td>
                <td>${u.lastLogin}</td>
                <td>
                    <button class="btn btn-sm btn-outline edit-user-btn">Edit</button>
                    <button class="btn btn-sm btn-danger delete-user-btn">Delete</button>
                </td>
            </tr>
        `).join('');
    }

    attachEvents() {
        const addBtn = this.section.querySelector('#add-user-btn');
        if (addBtn) {
            addBtn.addEventListener('click', () => {
                if (window.adminModalManager && window.adminModalManager.openAddUserModal) {
                    window.adminModalManager.openAddUserModal();
                } else if (typeof openModal === 'function') {
                    openModal('add-user-modal');
                }
            });
        }

        this.section.addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-user-btn')) {
                const id = parseInt(e.target.closest('tr').dataset.userId, 10);
                this.deleteUser(id);
            } else if (e.target.classList.contains('edit-user-btn')) {
                const id = parseInt(e.target.closest('tr').dataset.userId, 10);
                this.editUser(id);
            }
        });
    }

    addUser(data) {
        const id = this.users.length ? Math.max(...this.users.map(u => u.id)) + 1 : 1;
        const user = { id, ...data };
        this.users.push(user);
        this.saveUsers();
        this.renderRows();
    }

    editUser(id) {
        console.log('Edit user', id);
        // Implementation placeholder
    }

    deleteUser(id) {
        if (confirm('Delete this user?')) {
            this.users = this.users.filter(u => u.id !== id);
            this.saveUsers();
            this.renderRows();
        }
    }
}

let userManager;
function initializeUserManager() {
    if (!userManager) {
        userManager = new UserManager();
        userManager.init();
        window.userManager = userManager;
    }
    return userManager;
}

if (document.readyState !== 'loading') {
    initializeUserManager();
} else {
    document.addEventListener('DOMContentLoaded', initializeUserManager);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UserManager, initializeUserManager };
}


