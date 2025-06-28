// frontend/js/pages/admin/userManager.js
/**
 * UserManager - provides CRUD operations for admin users.
 * Falls back to in-memory mock data when API calls fail.
 */

class UserManager {
  constructor(apiClient = window.ApiClient) {
    this.api = apiClient;
    this.users = [];
    this.section = null;
    this.pagination = { page: 1, limit: 10, total: 0 };
    this.search = '';
  }

  /** Initialize manager and bind UI events. */
  async init() {
    this.section = document.getElementById('user-management');
    if (!this.section) return;
    await this.loadUsers();
    this.render();
    this.attachEvents();
  }

  /** Load users from the backend or mock store. */
  async loadUsers(page = 1) {
    this.pagination.page = page;
    try {
      const res = await this.api.get('/api/admin/users', {
        page,
        search: this.search,
        limit: this.pagination.limit
      });
      this.users = res.data.users || [];
      this.pagination.total = res.data.total || this.users.length;
    } catch (err) {
      console.warn('API failed, using mock users:', err);
      this.users = window.apiFallbackData.getUsers();
      this.pagination.total = this.users.length;
    }
  }

  /** Create a new user. */
  async createUser(data) {
    try {
      await this.api.post('/api/admin/users', data);
    } catch (err) {
      window.apiFallbackData.addUser(data);
    }
    this.showToast('User created');
    await this.loadUsers();
    this.renderRows();
  }

  /** Update user information. */
  async updateUser(id, data) {
    try {
      await this.api.put(`/api/admin/users/${id}`, data);
    } catch (err) {
      window.apiFallbackData.updateUser(id, data);
    }
    this.showToast('User updated');
    await this.loadUsers(this.pagination.page);
    this.renderRows();
  }

  /** Delete a user. */
  async deleteUser(id) {
    try {
      await this.api.delete(`/api/admin/users/${id}`);
    } catch (err) {
      window.apiFallbackData.deleteUser(id);
    }
    this.showToast('User deleted');
    await this.loadUsers(this.pagination.page);
    this.renderRows();
  }

  /** Change a user's primary role. */
  async changeRole(id, role) {
    await this.updateUser(id, { roles: [role] });
  }

  /** Trigger password reset for a user. */
  async resetPassword(id) {
    try {
      await this.api.post(`/api/admin/users/${id}/reset-password`);
    } catch (err) {
      console.warn('Password reset mock for', id, err);
    }
    this.showToast('Password reset email sent');
  }

  /** Render main structure. */
  render() {
    const table = `
      <table class="user-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Email</th>
            <th>Role</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="users-tbody"></tbody>
      </table>`;
    this.section.querySelector('.data-table').innerHTML = table;
    this.renderRows();
  }

  /** Render user rows to table body. */
  renderRows() {
    const tbody = this.section.querySelector('#users-tbody');
    if (!tbody) return;
    if (!this.users.length) {
      tbody.innerHTML = '<tr><td colspan="5">No users found</td></tr>';
      return;
    }
    tbody.innerHTML = this.users
      .map(
        u => `
      <tr data-user-id="${u.id}" class="user-row" data-cy="user-row">
        <td>${u.full_name}</td>
        <td>${u.email}</td>
        <td>${u.roles[0]}</td>
        <td>${u.is_active ? 'Active' : 'Inactive'}</td>
        <td>
          <button class="edit" data-id="${u.id}" aria-label="Edit user ${u.full_name}">✏️</button>
          <button class="delete" data-id="${u.id}" aria-label="Delete user ${u.full_name}">🗑️</button>
        </td>
      </tr>`
      )
      .join('');
  }

  /** Attach DOM event listeners. */
  attachEvents() {
    this.section.addEventListener('click', e => {
      const id = e.target.getAttribute('data-id');
      if (e.target.classList.contains('delete')) this.deleteUser(id);
      if (e.target.classList.contains('edit')) this.openEditModal(id);
    });

    const searchInput = this.section.querySelector('#user-search');
    if (searchInput) {
      searchInput.addEventListener('input', e => {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(async () => {
          this.search = e.target.value;
          await this.loadUsers(1);
          this.renderRows();
        }, 200);
      });
    }

    this.section.querySelector('#add-user-btn')?.addEventListener('click', () =>
      this.openCreateModal()
    );
  }

  /** Display a temporary toast message. */
  showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }

  openCreateModal() {
    // integrate with modal manager if available
    window.adminModalManager?.openAddUserModal?.();
  }

  openEditModal(id) {
    const user = this.users.find(u => u.id === id);
    window.adminModalManager?.openEditUserModal?.(user);
  }
}

const manager = new UserManager();
window.userManager = manager;

document.addEventListener('DOMContentLoaded', () => manager.init());

export default manager;
