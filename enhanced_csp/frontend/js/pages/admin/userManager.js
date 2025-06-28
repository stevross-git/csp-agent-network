/* --------------------------------------------------------------------------
 *  UserManager - complete CRUD, filters, pagination & mock-data fallback
 * --------------------------------------------------------------------------
 *  • Tries live API first; if any call fails (or no token), auto-falls back
 *    to the in-memory helpers in `apiFallbackData.js`.
 *  • Keeps every UI feature from the “codex” branch (advanced table, filters,
 *    pagination, toast / notification system) plus the cleaner dependency
 *    injection pattern and debounced search from “main”.
 * ------------------------------------------------------------------------ */

import {
  defaultUsersMock,
  getUsers as getMockUsers,
  addUser as addMockUser,
  updateUser as updateMockUser,
  deleteUser as deleteMockUser
} from "@/utils/apiFallbackData";

class UserManager {
  /* -------------------------------- Constructor ------------------------- */
  constructor(apiClient = null) {
    /* If you pass a fully-featured Axios-style client it will be used.
       Otherwise UserManager falls back to window.fetch. */
    this.api = apiClient;

    /* Data state */
    this.users = [];
    this.loading = false;

    /* UI state */
    this.section = null;
    this.tbody   = null;

    /* Pagination & filters */
    this.pagination = { page: 1, limit: 50, total: 0, totalPages: 1 };
    this.filters    = { status: "", role: "", search: "" };

    /* Environment */
    this.apiBaseUrl = this._getApiBaseUrl();
    this.authToken  = this._getAuthToken();

    /* Misc. */
    this.debounceTimer = null;
  }

  /* --------------------------- Environment helpers ---------------------- */
  _getApiBaseUrl() {
    if (window.REACT_APP_CSP_API_URL)            return window.REACT_APP_CSP_API_URL;
    if (typeof REACT_APP_CSP_API_URL !== "undefined") return REACT_APP_CSP_API_URL;

    const meta = document.querySelector('meta[name="api-base-url"]');
    if (meta) return meta.content;

    if (["localhost", "127.0.0.1"].includes(location.hostname))
      return "http://localhost:8000";

    return location.origin.replace(":3000", ":8000"); // prod default
  }

  _getAuthToken() {
    return (
      localStorage.getItem("csp_auth_token") ||
      sessionStorage.getItem("csp_auth_token")
    );
  }

  /* -------------------------- Low-level request ------------------------- */
  async _apiRequest(endpoint, opts = {}) {
    /* If an external client (e.g., Axios) was injected, use it */
    if (this.api) {
      return this.api.request({ url: endpoint, baseURL: this.apiBaseUrl, ...opts });
    }

    const url = `${this.apiBaseUrl}${endpoint}`;
    const defaultHeaders = {
      "Content-Type": "application/json",
      ...(this.authToken && { Authorization: `Bearer ${this.authToken}` })
    };

    const res = await fetch(url, { headers: defaultHeaders, ...opts });

    if (!res.ok) {
      if (res.status === 401) this._handleAuthError();
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}: ${res.statusText}`);
    }
    return res.json();
  }

  _handleAuthError() {
    localStorage.removeItem("csp_auth_token");
    sessionStorage.removeItem("csp_auth_token");
    this._notify("Session expired. Please log in again.", "warning");
    // Optionally redirect: window.location.href = "/login";
  }

  /* --------------------------- Public initialiser ---------------------- */
  async init() {
    this.section = document.getElementById("user-management") ||
                   document.getElementById("users");
    if (!this.section) {
      console.error("User manager section not found");
      return;
    }

    await this.loadUsers();
    this._render();
    this._attachEvents();
  }

  /* ---------------------------- CRUD methods --------------------------- */
  async loadUsers(page = 1) {
    if (this.loading) return;
    this.loading = true;
    this._showLoadingState();

    this.pagination.page = page;

    try {
      const params = new URLSearchParams({
        page, limit: this.pagination.limit,
        ...(this.filters.status && { status: this.filters.status }),
        ...(this.filters.role   && { role:   this.filters.role   }),
        ...(this.filters.search && { search: this.filters.search })
      });

      const res = await this._apiRequest(`/api/admin/users?${params}`);
      this.users             = res.users || res.items || [];
      this.pagination.total  = res.total || this.users.length;
      this.pagination.limit  = res.limit || this.pagination.limit;
      this.pagination.page   = res.page  || page;
      this.pagination.totalPages =
        res.totalPages || Math.ceil(this.pagination.total / this.pagination.limit);

    } catch (err) {
      console.warn("API failed, falling back to mock data:", err);
      this.users             = getMockUsers() || defaultUsersMock;
      this.pagination.total  = this.users.length;
      this.pagination.totalPages = Math.ceil(this.pagination.total / this.pagination.limit);

      /* Let the user know they're in offline/demo mode */
      this._notify("Using offline demo data – login for full functionality", "info", 7000);
    } finally {
      this.loading = false;
      this._hideLoadingState();
      this._renderRows();
      this._renderPagination();
    }
  }

  async createUser(data) {
    try {
      await this._apiRequest("/api/auth/local/register", {
        method: "POST",
        body: JSON.stringify(data)
      });
    } catch (e) {
      addMockUser(data);
    }
    this._notify("User created", "success");
    await this.loadUsers(this.pagination.page);
  }

  async updateUser(id, data) {
    try {
      await this._apiRequest(`/api/admin/users/${id}`, {
        method: "PUT",
        body: JSON.stringify(data)
      });
    } catch (e) {
      updateMockUser(id, data);
    }
    this._notify("User updated", "success");
    await this.loadUsers(this.pagination.page);
  }

  async deleteUser(id) {
    /* Confirm & delete */
    const user = this.users.find(u => u.id === id);
    if (!user) return this._notify("User not found", "error");

    if (!confirm(`Delete "${user.full_name}" (${user.email})? This cannot be undone.`))
      return;

    try {
      await this._apiRequest(`/api/admin/users/${id}`, { method: "DELETE" });
    } catch (e) {
      deleteMockUser(id);
    }
    this._notify("User deleted", "success");
    await this.loadUsers(this.pagination.page);
  }

  /* ------------------------- Rendering helpers ------------------------- */
  _render() {
    this.section.innerHTML = `
      <div class="user-management-container">
        <h2 class="page-title">
          <i class="fas fa-users"></i> User Management
          <span class="user-count">${this.pagination.total} users</span>
        </h2>

        <!-- Filters -->
        <div class="user-filters">
          <input  id="user-search"  class="form-control" type="text" placeholder="🔍 Search..."
                  value="${this.filters.search}">
          <select id="status-filter" class="form-control">
            <option value="">All Status</option>
            <option value="active"   ${this.filters.status==="active"  ?"selected":""}>Active</option>
            <option value="inactive" ${this.filters.status==="inactive"?"selected":""}>Inactive</option>
          </select>
          <select id="role-filter" class="form-control">
            <option value="">All Roles</option>
            <option value="admin"     ${this.filters.role==="admin"    ?"selected":""}>Admin</option>
            <option value="user"      ${this.filters.role==="user"     ?"selected":""}>User</option>
            <option value="developer" ${this.filters.role==="developer"?"selected":""}>Developer</option>
          </select>
          <button id="apply-filters-btn"  class="btn btn-secondary"><i class="fas fa-filter"></i> Apply</button>
          <button id="reset-filters-btn"  class="btn btn-outline">Reset</button>
        </div>

        <!-- Table -->
        <div class="data-table">
          <div class="table-header">
            <div class="table-title">Users</div>
            <div class="table-actions">
              <button id="add-user-btn"     class="btn btn-primary"><i class="fas fa-user-plus"></i> Add</button>
              <button id="export-users-btn" class="btn btn-secondary"><i class="fas fa-download"></i> Export</button>
              <button id="refresh-users-btn"class="btn btn-outline"><i class="fas fa-sync-alt"></i> Refresh</button>
            </div>
          </div>
          <div class="table-content">
            <table class="user-table">
              <thead>
                <tr>
                  <th><input type="checkbox" id="select-all-users"></th>
                  <th>Name</th><th>Email</th><th>Role</th>
                  <th>Status</th><th>Last Login</th><th>Created</th><th>Actions</th>
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

    this.tbody = this.section.querySelector("#users-tbody");
    this._renderRows();
    this._renderPagination();
  }

  _renderRows() {
    if (!this.tbody) return;

    /* No data */
    if (!this.users.length) {
      this.tbody.innerHTML = `
        <tr><td colspan="8" class="empty-state">
          <div class="empty-state-content">
            <i class="fas fa-users fa-3x"></i>
            <h3>No users found</h3>
            <button class="btn btn-primary" onclick="userManager.openAddUserModal()">
              <i class="fas fa-user-plus"></i> Add User
            </button>
          </div>
        </td></tr>`;
      return;
    }

    /* Data rows */
    this.tbody.innerHTML = this.users.map(u => `
      <tr data-user-id="${u.id}">
        <td><input type="checkbox" class="user-checkbox" value="${u.id}"></td>
        <td>
          <div class="user-info">
            <div class="user-avatar">${(u.full_name||"U").charAt(0).toUpperCase()}</div>
            <div class="user-details">
              <div class="user-name">${u.full_name||"Unknown"}</div>
              <div class="user-id">ID: ${u.id}</div>
            </div>
          </div>
        </td>
        <td>
          <a href="mailto:${u.email}" class="user-email">${u.email}</a>
          ${u.is_email_verified
            ? '<i class="fas fa-check-circle text-success" title="Verified"></i>'
            : '<i class="fas fa-exclamation-triangle text-warning" title="Unverified"></i>'}
        </td>
        <td>${this._renderRoles(u.roles||["user"])}</td>
        <td><span class="status-badge ${u.is_active?"status-active":"status-inactive"}">
          ${u.is_active?"Active":"Inactive"}
        </span></td>
        <td>${u.last_login ? this._formatDateTime(u.last_login) : "Never"}</td>
        <td>${this._formatDate(u.created_at)}</td>
        <td>
          <button class="btn btn-sm btn-outline view-user-btn"   title="View"><i class="fas fa-eye"></i></button>
          <button class="btn btn-sm btn-secondary edit-user-btn" title="Edit"><i class="fas fa-edit"></i></button>
          <button class="btn btn-sm btn-danger delete-user-btn"  title="Delete"><i class="fas fa-trash"></i></button>
        </td>
      </tr>`).join("");
  }

  _renderRoles(roles) {
    return roles.map(r => `<span class="role-badge role-${r}">${r}</span>`).join("");
  }

  _renderPagination() {
    const container = this.section.querySelector("#user-pagination");
    if (!container) return;
    if (this.pagination.totalPages <= 1) return container.innerHTML = "";

    const { page, totalPages, limit, total } = this.pagination;
    const maxVisible = 5;
    let start = Math.max(1, page - Math.floor(maxVisible / 2));
    let end   = Math.min(totalPages, start + maxVisible - 1);
    if (end - start < maxVisible - 1) start = Math.max(1, end - maxVisible + 1);

    const btn = (p, label, disabled=false) =>
      `<button class="btn btn-sm ${p===page?"btn-primary":"btn-outline"}"
               ${disabled?"disabled":""} onclick="userManager.loadUsers(${p})">${label}</button>`;

    let html = '<div class="pagination">';
    html += btn(1,  '«', page===1);
    html += btn(page-1,'‹', page===1);

    for (let i=start;i<=end;i++) html += btn(i,i);

    html += btn(page+1,'›', page===totalPages);
    html += btn(totalPages,'»', page===totalPages);
    html += '</div>';

    html += `<div class="pagination-info">
               Showing ${(page-1)*limit + 1}–${Math.min(page*limit, total)} of ${total}
             </div>`;
    container.innerHTML = html;
  }

  /* --------------------- UI helpers / notifications -------------------- */
  _showLoadingState() {
    if (this.tbody)
      this.tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem">
        <i class="fas fa-spinner fa-spin fa-2x"></i> Loading…
      </td></tr>`;
  }
  _hideLoadingState() { /* rows will overwrite */ }

  _notify(msg, type="info", dur=4000) {
    /* simple toast — you can replace with your own system */
    const notif = document.createElement("div");
    notif.className = `notification notification-${type}`;
    notif.innerHTML = `
      <div class="notification-content">
        <span>${msg}</span>
        <button class="notification-close"
                onclick="this.parentElement.parentElement.remove()">&times;</button>
      </div>`;
    document.body.appendChild(notif);
    setTimeout(()=>notif.remove(), dur);
  }

  /* ---------------------------- Event wiring --------------------------- */
  _attachEvents() {
    /* Add / Refresh / Export */
    this.section.querySelector("#add-user-btn")    ?.addEventListener("click", () => this.openAddUserModal());
    this.section.querySelector("#refresh-users-btn")?.addEventListener("click", () => this.loadUsers(this.pagination.page));
    this.section.querySelector("#export-users-btn") ?.addEventListener("click", () => this.exportUsers());

    /* Filters */
    const search  = this.section.querySelector("#user-search");
    const status  = this.section.querySelector("#status-filter");
    const roleSel = this.section.querySelector("#role-filter");

    this.section.querySelector("#apply-filters-btn")?.addEventListener("click", () => {
      this.filters = { search: search.value, status: status.value, role: roleSel.value };
      this.loadUsers(1);
    });
    this.section.querySelector("#reset-filters-btn")?.addEventListener("click", () => {
      this.filters = { search: "", status: "", role: "" };
      search.value = status.value = roleSel.value = "";
      this.loadUsers(1);
    });

    /* Debounced live search */
    search?.addEventListener("input", e => {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => {
        this.filters.search = e.target.value;
        this.loadUsers(1);
      }, 300);
    });

    /* Row-level actions */
    this.section.addEventListener("click", e => {
      const row   = e.target.closest("tr[data-user-id]");
      if (!row) return;
      const id = row.dataset.userId;

      if (e.target.closest(".delete-user-btn"))   this.deleteUser(id);
      if (e.target.closest(".edit-user-btn"))     this.openEditUserModal(this.users.find(u=>u.id===id));
      if (e.target.closest(".view-user-btn"))     this.openUserDetailsModal(this.users.find(u=>u.id===id));
    });

    /* Select-all checkbox */
    this.section.querySelector("#select-all-users")?.addEventListener("change", e => {
      this.section.querySelectorAll(".user-checkbox").forEach(cb => cb.checked = e.target.checked);
    });
  }

  /* ------------------------- Date formatting --------------------------- */
  _formatDate(d)     { return d ? new Date(d).toLocaleDateString()  : "N/A"; }
  _formatDateTime(d) { return d ? new Date(d).toLocaleString()      : "Never"; }

  /* ---------------------- Modal wrappers (delegates) ------------------- */
  openAddUserModal()     { window.adminModalManager?.openAddUserModal?.(); }
  openEditUserModal(u)   { window.adminModalManager?.openEditUserModal?.(u); }
  openUserDetailsModal(u){ window.adminModalManager?.openUserDetailsModal?.(u); }

  /* ---------------------------- Export JSON ---------------------------- */
  async exportUsers() {
    try {
      const data = await this._apiRequest("/api/admin/users/export");
      this._downloadBlob(JSON.stringify(data, null, 2),
        `users_export_${new Date().toISOString().split("T")[0]}.json`);
      this._notify("Users exported", "success");
    } catch (e) {
      /* Offline mode: export the currently loaded list */
      this._downloadBlob(JSON.stringify(this.users, null, 2),
        `users_export_${new Date().toISOString().split("T")[0]}_mock.json`);
      this._notify("Offline export", "info");
    }
  }

  _downloadBlob(text, filename) {
    const blob = new Blob([text], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement("a"), { href:url, download:filename });
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }
}

/* ------------------------- Global singleton --------------------------- */
const userManager = new UserManager(window.ApiClient);
window.userManager = userManager;

/* Auto-init when DOM is ready */
if (document.readyState !== "loading") {
  userManager.init();
} else {
  document.addEventListener("DOMContentLoaded", () => userManager.init());
}

/* For ES-module / Node unit tests */
export { UserManager };
export default userManager;
