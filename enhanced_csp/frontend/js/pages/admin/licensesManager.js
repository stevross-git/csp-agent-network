class LicensesManager {
    constructor() {
        this.section = null;
        this.licenses = [];
        this.api = window.ApiClient || new ApiClient();
    }

    async init() {
        this.section = document.getElementById('licenses');
        if (!this.section) return;
        await this.load();
        this.render();
        this.attachEvents();
    }

    async load() {
        try {
            const response = await this.api.get('/licenses');
            if (response.success) {
                this.licenses = response.data;
            } else {
                this.licenses = [];
            }
        } catch (err) {
            console.error('Failed to load licenses', err);
            this.licenses = [];
        }
    }

    render() {
        this.section.innerHTML = `
            <div class="licenses-dashboard">
                <h2><i class="fas fa-certificate"></i> License Management</h2>
                <button class="btn btn-primary" id="add-license-btn">Add License</button>
                <table class="license-table">
                    <thead><tr><th>Product</th><th>Key</th><th>Expires</th><th>Active</th><th>Actions</th></tr></thead>
                    <tbody id="licenses-tbody">
                        ${this.licenses.map(l => `
                            <tr data-id="${l.id}">
                                <td>${l.product}</td>
                                <td>${l.key}</td>
                                <td>${l.expires_at || ''}</td>
                                <td>${l.active ? 'Yes' : 'No'}</td>
                                <td><button class="delete-btn">Delete</button></td>
                            </tr>`).join('')}
                    </tbody>
                </table>
            </div>`;
    }

    attachEvents() {
        this.section.querySelector('#add-license-btn')?.addEventListener('click', () => this.create());
        this.section.addEventListener('click', (e) => {
            const btn = e.target.closest('.delete-btn');
            if (btn) {
                const tr = btn.closest('tr');
                const id = tr.getAttribute('data-id');
                this.remove(id);
            }
        });
    }

    async create() {
        const product = prompt('Product name?');
        if (!product) return;
        const key = prompt('License key?');
        if (!key) return;
        try {
            const response = await this.api.post('/licenses', { product, key });
            if (response.success) {
                await this.refresh();
            } else {
                console.error('Failed to add license', response.error);
            }
        } catch (err) {
            console.error('Failed to add license', err);
        }
    }

    async remove(id) {
        if (!confirm('Delete this license?')) return;
        try {
            const response = await this.api.delete(`/licenses/${id}`);
            if (response.success) {
                await this.refresh();
            } else {
                console.error('Failed to delete license', response.error);
            }
        } catch (err) {
            console.error('Failed to delete license', err);
        }
    }

    async refresh() {
        await this.load();
        this.render();
    }
}

let licensesManager;
function initializeLicensesManager() {
    if (!licensesManager) {
        licensesManager = new LicensesManager();
        licensesManager.init();
        window.licensesManager = licensesManager;
    }
    return licensesManager;
}

if (document.readyState !== 'loading') {
    initializeLicensesManager();
} else {
    document.addEventListener('DOMContentLoaded', initializeLicensesManager);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LicensesManager, initializeLicensesManager };
}
