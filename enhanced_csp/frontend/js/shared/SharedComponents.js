// js/shared/SharedComponents.js
// Shared Component Library for Enhanced CSP System

// =================================================================
// 1. NAVIGATION COMPONENT
// =================================================================
class Navigation extends BaseComponent {
    constructor(containerId = 'global-nav', options = {}) {
        super(containerId, {
            theme: 'dark',
            collapsible: true,
            activePageTracking: true,
            ...options
        });
        
        this.activePageName = this.getCurrentPageName();
        this.isCollapsed = false;
    }
    
    async loadDependencies() {
        // Load available pages data
        if (window.availablePages && window.pageCategories) {
            this.pages = window.availablePages;
            this.categories = window.pageCategories;
            this.icons = window.pageIcons || {};
        } else {
            await this.loadPageData();
        }
    }
    
    async loadPageData() {
        try {
            const response = await window.ApiClient.get('/pages/available');
            if (response.success) {
                this.pages = response.data.pages;
                this.categories = response.data.categories;
                this.icons = response.data.icons || {};
            }
        } catch (error) {
            console.warn('Failed to load page data, using defaults');
            this.setDefaultPages();
        }
    }
    
    setDefaultPages() {
        this.pages = [
            { name: 'dashboard', title: 'Dashboard', category: 'core' },
            { name: 'admin', title: 'Admin Portal', category: 'admin' },
            { name: 'designer', title: 'Visual Designer', category: 'core' },
            { name: 'monitoring', title: 'Monitoring', category: 'monitoring' },
            { name: 'ai-agents', title: 'AI Agents', category: 'ai' }
        ];
        this.categories = {
            core: ['dashboard', 'designer'],
            admin: ['admin'],
            monitoring: ['monitoring'],
            ai: ['ai-agents']
        };
    }
    
    render() {
        if (!this.container) return;
        
        this.container.innerHTML = `
            <nav class="main-navigation ${this.options.theme}">
                <div class="nav-header">
                    <div class="nav-brand">
                        <img src="../assets/logo.svg" alt="CSP System" class="nav-logo">
                        <span class="nav-title">Enhanced CSP</span>
                    </div>
                    <button class="nav-toggle" id="nav-toggle">
                        <span class="hamburger"></span>
                    </button>
                </div>
                
                <div class="nav-content" id="nav-content">
                    <div class="nav-search">
                        <input type="text" placeholder="Search pages..." id="nav-search">
                        <span class="search-icon">🔍</span>
                    </div>
                    
                    <div class="nav-categories" id="nav-categories">
                        ${this.renderCategories()}
                    </div>
                    
                    <div class="nav-footer">
                        <div class="user-info">
                            <div class="user-avatar">👤</div>
                            <span class="user-name">Administrator</span>
                        </div>
                        <button class="logout-btn" id="logout-btn">🚪 Logout</button>
                    </div>
                </div>
            </nav>
        `;
        
        super.render();
    }
    
    renderCategories() {
        return Object.entries(this.categories).map(([categoryName, pageNames]) => {
            const categoryPages = pageNames.map(pageName => 
                this.pages.find(p => p.name === pageName)
            ).filter(Boolean);
            
            if (categoryPages.length === 0) return '';
            
            return `
                <div class="nav-category">
                    <div class="category-header" data-category="${categoryName}">
                        <span class="category-name">${this.formatCategoryName(categoryName)}</span>
                        <span class="category-toggle">▼</span>
                    </div>
                    <div class="category-items ${this.isCategoryActive(categoryName) ? 'active' : ''}">
                        ${categoryPages.map(page => this.renderPageLink(page)).join('')}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    renderPageLink(page) {
        const icon = this.icons[page.name] || '📄';
        const isActive = page.name === this.activePageName;
        
        return `
            <a href="${page.name}.html" 
               class="nav-link ${isActive ? 'active' : ''}" 
               data-page="${page.name}">
                <span class="nav-icon">${icon}</span>
                <span class="nav-text">${page.title}</span>
                ${isActive ? '<span class="active-indicator"></span>' : ''}
            </a>
        `;
    }
    
    bindEvents() {
        super.bindEvents();
        
        // Toggle navigation
        const toggleBtn = this.findElement('#nav-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleNavigation());
        }
        
        // Search functionality
        const searchInput = this.findElement('#nav-search');
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce((e) => {
                this.filterPages(e.target.value);
            }));
        }
        
        // Category toggle
        this.findElements('.category-header').forEach(header => {
            header.addEventListener('click', (e) => {
                this.toggleCategory(e.currentTarget.dataset.category);
            });
        });
        
        // Page navigation
        this.findElements('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                this.handlePageNavigation(e);
            });
        });
        
        // Logout
        const logoutBtn = this.findElement('#logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.handleLogout());
        }
    }
    
    toggleNavigation() {
        this.isCollapsed = !this.isCollapsed;
        this.container.classList.toggle('collapsed', this.isCollapsed);
        this.emit('navigation:toggle', { collapsed: this.isCollapsed });
    }
    
    toggleCategory(categoryName) {
        const categoryItems = this.findElement(`[data-category="${categoryName}"] + .category-items`);
        if (categoryItems) {
            categoryItems.classList.toggle('active');
        }
    }
    
    filterPages(query) {
        const links = this.findElements('.nav-link');
        const lowerQuery = query.toLowerCase();
        
        links.forEach(link => {
            const text = link.textContent.toLowerCase();
            const isVisible = text.includes(lowerQuery);
            link.style.display = isVisible ? 'flex' : 'none';
        });
    }
    
    handlePageNavigation(e) {
        // Optional: Add smooth page transitions
        const pageName = e.currentTarget.dataset.page;
        this.emit('navigation:pageChange', { from: this.activePageName, to: pageName });
    }
    
    handleLogout() {
        if (confirm('Are you sure you want to logout?')) {
            window.ApiClient.clearAuth();
            window.location.href = '/login.html';
        }
    }
    
    getCurrentPageName() {
        const path = window.location.pathname;
        const filename = path.split('/').pop();
        return filename.replace('.html', '') || 'dashboard';
    }
    
    formatCategoryName(categoryName) {
        return categoryName.charAt(0).toUpperCase() + categoryName.slice(1);
    }
    
    isCategoryActive(categoryName) {
        return this.categories[categoryName]?.includes(this.activePageName) || false;
    }
    
    setActivePage(pageName) {
        // Remove active state from current
        const currentActive = this.findElement('.nav-link.active');
        if (currentActive) {
            currentActive.classList.remove('active');
            const indicator = currentActive.querySelector('.active-indicator');
            if (indicator) indicator.remove();
        }
        
        // Add active state to new page
        const newActive = this.findElement(`[data-page="${pageName}"]`);
        if (newActive) {
            newActive.classList.add('active');
            newActive.insertAdjacentHTML('beforeend', '<span class="active-indicator"></span>');
        }
        
        this.activePageName = pageName;
    }
}

// =================================================================
// 2. TOAST NOTIFICATION COMPONENT
// =================================================================
class Toast extends BaseComponent {
    constructor(containerId = 'toast-container', options = {}) {
        super(containerId, {
            position: 'top-right',
            autoClose: true,
            duration: 5000,
            maxToasts: 5,
            ...options
        });
        
        this.toasts = new Map();
        this.toastCounter = 0;
    }
    
    render() {
        if (!this.container) {
            // Create container if it doesn't exist
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.className = `toast-container ${this.options.position}`;
            document.body.appendChild(this.container);
        }
        
        this.container.className = `toast-container ${this.options.position}`;
        super.render();
    }
    
    show(message, type = 'info', options = {}) {
        const id = ++this.toastCounter;
        const config = { ...this.options, ...options };
        
        const toastElement = this.createElement('div', {
            className: `toast toast-${type}`,
            dataset: { toastId: id }
        }, `
            <div class="toast-content">
                <div class="toast-icon">${this.getToastIcon(type)}</div>
                <div class="toast-message">${message}</div>
                <button class="toast-close" data-action="close">×</button>
            </div>
            ${config.progress ? '<div class="toast-progress"></div>' : ''}
        `);
        
        // Add to container
        this.container.appendChild(toastElement);
        
        // Store reference
        this.toasts.set(id, {
            element: toastElement,
            type,
            message,
            timestamp: Date.now()
        });
        
        // Animate in
        requestAnimationFrame(() => {
            toastElement.classList.add('toast-show');
        });
        
        // Bind close event
        const closeBtn = toastElement.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.hide(id));
        
        // Auto close
        if (config.autoClose && config.duration > 0) {
            setTimeout(() => this.hide(id), config.duration);
            
            if (config.progress) {
                this.showProgress(toastElement, config.duration);
            }
        }
        
        // Limit number of toasts
        this.limitToasts();
        
        return id;
    }
    
    hide(id) {
        const toast = this.toasts.get(id);
        if (!toast) return;
        
        toast.element.classList.add('toast-hide');
        
        setTimeout(() => {
            if (toast.element.parentNode) {
                toast.element.parentNode.removeChild(toast.element);
            }
            this.toasts.delete(id);
        }, 300);
    }
    
    showProgress(element, duration) {
        const progressBar = element.querySelector('.toast-progress');
        if (!progressBar) return;
        
        progressBar.style.animation = `toast-progress ${duration}ms linear forwards`;
    }
    
    limitToasts() {
        if (this.toasts.size > this.options.maxToasts) {
            const oldestId = Math.min(...this.toasts.keys());
            this.hide(oldestId);
        }
    }
    
    getToastIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    // Convenience methods
    success(message, options = {}) {
        return this.show(message, 'success', options);
    }
    
    error(message, options = {}) {
        return this.show(message, 'error', { ...options, duration: 0 }); // Errors don't auto-close
    }
    
    warning(message, options = {}) {
        return this.show(message, 'warning', options);
    }
    
    info(message, options = {}) {
        return this.show(message, 'info', options);
    }
    
    clear() {
        this.toasts.forEach((_, id) => this.hide(id));
    }
}

// =================================================================
// 3. MODAL COMPONENT
// =================================================================
class Modal extends BaseComponent {
    constructor(containerId = 'modal-container', options = {}) {
        super(containerId, {
            closeOnEscape: true,
            closeOnOverlay: true,
            showCloseButton: true,
            animation: 'fade',
            ...options
        });
        
        this.isOpen = false;
        this.activeModal = null;
    }
    
    render() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'modal-container';
            document.body.appendChild(this.container);
        }
        
        super.render();
    }
    
    show(content, options = {}) {
        const config = { ...this.options, ...options };
        
        const modalId = 'modal-' + Date.now();
        const modalElement = this.createElement('div', {
            className: `modal-overlay ${config.animation}`,
            dataset: { modalId }
        }, `
            <div class="modal-dialog ${config.size || 'medium'}">
                <div class="modal-header">
                    <h3 class="modal-title">${config.title || 'Modal'}</h3>
                    ${config.showCloseButton ? '<button class="modal-close" data-action="close">×</button>' : ''}
                </div>
                <div class="modal-body">
                    ${typeof content === 'string' ? content : ''}
                </div>
                <div class="modal-footer">
                    ${this.renderModalButtons(config.buttons)}
                </div>
            </div>
        `);
        
        this.container.appendChild(modalElement);
        
        // Insert non-string content
        if (typeof content !== 'string') {
            const modalBody = modalElement.querySelector('.modal-body');
            modalBody.innerHTML = '';
            modalBody.appendChild(content);
        }
        
        this.bindModalEvents(modalElement, config);
        
        // Show modal
        requestAnimationFrame(() => {
            modalElement.classList.add('modal-show');
        });
        
        this.isOpen = true;
        this.activeModal = modalElement;
        
        // Prevent body scroll
        document.body.style.overflow = 'hidden';
        
        return modalId;
    }
    
    renderModalButtons(buttons = []) {
        if (!buttons.length) {
            buttons = [{ text: 'Close', action: 'close', variant: 'secondary' }];
        }
        
        return buttons.map(btn => `
            <button class="modal-btn btn-${btn.variant || 'primary'}" 
                    data-action="${btn.action || 'close'}">
                ${btn.text}
            </button>
        `).join('');
    }
    
    bindModalEvents(modalElement, config) {
        // Close button
        const closeBtn = modalElement.querySelector('.modal-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }
        
        // Footer buttons
        modalElement.querySelectorAll('.modal-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                if (action === 'close') {
                    this.hide();
                } else if (config.onAction) {
                    config.onAction(action, this);
                }
            });
        });
        
        // Overlay click
        if (config.closeOnOverlay) {
            modalElement.addEventListener('click', (e) => {
                if (e.target === modalElement) {
                    this.hide();
                }
            });
        }
        
        // Escape key
        if (config.closeOnEscape) {
            const escapeHandler = (e) => {
                if (e.key === 'Escape' && this.isOpen) {
                    this.hide();
                }
            };
            document.addEventListener('keydown', escapeHandler);
            modalElement._escapeHandler = escapeHandler;
        }
    }
    
    hide() {
        if (!this.activeModal) return;
        
        this.activeModal.classList.add('modal-hide');
        
        setTimeout(() => {
            if (this.activeModal && this.activeModal.parentNode) {
                // Remove escape handler
                if (this.activeModal._escapeHandler) {
                    document.removeEventListener('keydown', this.activeModal._escapeHandler);
                }
                
                this.activeModal.parentNode.removeChild(this.activeModal);
            }
            
            this.activeModal = null;
            this.isOpen = false;
            
            // Restore body scroll
            document.body.style.overflow = '';
        }, 300);
    }
    
    // Convenience methods
    confirm(message, options = {}) {
        return new Promise((resolve) => {
            this.show(message, {
                title: 'Confirm',
                buttons: [
                    { text: 'Cancel', action: 'cancel', variant: 'secondary' },
                    { text: 'Confirm', action: 'confirm', variant: 'primary' }
                ],
                onAction: (action) => {
                    this.hide();
                    resolve(action === 'confirm');
                },
                ...options
            });
        });
    }
    
    alert(message, options = {}) {
        return new Promise((resolve) => {
            this.show(message, {
                title: 'Alert',
                buttons: [{ text: 'OK', action: 'ok', variant: 'primary' }],
                onAction: () => {
                    this.hide();
                    resolve();
                },
                ...options
            });
        });
    }
}

// =================================================================
// 4. DATA TABLE COMPONENT
// =================================================================
class DataTable extends BaseComponent {
    constructor(containerId, options = {}) {
        super(containerId, {
            pagination: true,
            pageSize: 10,
            sorting: true,
            filtering: true,
            searchable: true,
            selectable: false,
            ...options
        });
        
        this.data = [];
        this.filteredData = [];
        this.currentPage = 1;
        this.sortColumn = null;
        this.sortDirection = 'asc';
        this.searchQuery = '';
        this.selectedRows = new Set();
    }
    
    setData(data, columns = null) {
        this.data = Array.isArray(data) ? data : [];
        this.filteredData = [...this.data];
        
        if (columns) {
            this.columns = columns;
        } else if (this.data.length > 0) {
            // Auto-detect columns
            this.columns = Object.keys(this.data[0]).map(key => ({
                key,
                title: this.formatColumnTitle(key),
                sortable: true,
                searchable: true
            }));
        }
        
        this.currentPage = 1;
        this.render();
    }
    
    render() {
        if (!this.container) return;
        
        this.container.innerHTML = `
            <div class="data-table-wrapper">
                ${this.options.searchable ? this.renderSearch() : ''}
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            ${this.renderHeader()}
                        </thead>
                        <tbody>
                            ${this.renderBody()}
                        </tbody>
                    </table>
                </div>
                ${this.options.pagination ? this.renderPagination() : ''}
            </div>
        `;
        
        super.render();
    }
    
    renderSearch() {
        return `
            <div class="table-search">
                <input type="text" 
                       placeholder="Search..." 
                       class="search-input"
                       value="${this.searchQuery}">
                <span class="search-icon">🔍</span>
            </div>
        `;
    }
    
    renderHeader() {
        if (!this.columns) return '';
        
        return `
            <tr>
                ${this.options.selectable ? '<th class="select-column"><input type="checkbox" class="select-all"></th>' : ''}
                ${this.columns.map(col => `
                    <th class="sortable-header ${col.sortable !== false ? 'sortable' : ''}" 
                        data-column="${col.key}">
                        <span class="column-title">${col.title}</span>
                        ${col.sortable !== false ? this.renderSortIndicator(col.key) : ''}
                    </th>
                `).join('')}
            </tr>
        `;
    }
    
    renderSortIndicator(columnKey) {
        if (this.sortColumn === columnKey) {
            return this.sortDirection === 'asc' ? '<span class="sort-indicator">▲</span>' : '<span class="sort-indicator">▼</span>';
        }
        return '<span class="sort-indicator">⇅</span>';
    }
    
    renderBody() {
        const paginatedData = this.getPaginatedData();
        
        if (paginatedData.length === 0) {
            return `
                <tr>
                    <td colspan="${this.getColumnCount()}" class="no-data">
                        No data available
                    </td>
                </tr>
            `;
        }
        
        return paginatedData.map((row, index) => `
            <tr class="data-row ${this.selectedRows.has(this.getRowId(row, index)) ? 'selected' : ''}" 
                data-row-id="${this.getRowId(row, index)}">
                ${this.options.selectable ? `<td class="select-column"><input type="checkbox" class="row-select"></td>` : ''}
                ${this.columns.map(col => `
                    <td class="data-cell" data-column="${col.key}">
                        ${this.formatCellValue(row[col.key], col, row)}
                    </td>
                `).join('')}
            </tr>
        `).join('');
    }
    
    renderPagination() {
        const totalPages = this.getTotalPages();
        const hasPages = totalPages > 1;
        
        if (!hasPages) return '';
        
        return `
            <div class="table-pagination">
                <div class="pagination-info">
                    Showing ${this.getStartIndex() + 1}-${this.getEndIndex()} of ${this.filteredData.length} entries
                </div>
                <div class="pagination-controls">
                    <button class="page-btn" data-action="first" ${this.currentPage === 1 ? 'disabled' : ''}>
                        ⏪
                    </button>
                    <button class="page-btn" data-action="prev" ${this.currentPage === 1 ? 'disabled' : ''}>
                        ◀
                    </button>
                    
                    ${this.renderPageNumbers()}
                    
                    <button class="page-btn" data-action="next" ${this.currentPage === totalPages ? 'disabled' : ''}>
                        ▶
                    </button>
                    <button class="page-btn" data-action="last" ${this.currentPage === totalPages ? 'disabled' : ''}>
                        ⏩
                    </button>
                </div>
            </div>
        `;
    }
    
    renderPageNumbers() {
        const totalPages = this.getTotalPages();
        const maxVisible = 5;
        let start = Math.max(1, this.currentPage - Math.floor(maxVisible / 2));
        let end = Math.min(totalPages, start + maxVisible - 1);
        
        if (end - start + 1 < maxVisible) {
            start = Math.max(1, end - maxVisible + 1);
        }
        
        const pages = [];
        for (let i = start; i <= end; i++) {
            pages.push(`
                <button class="page-btn page-number ${i === this.currentPage ? 'active' : ''}" 
                        data-page="${i}">
                    ${i}
                </button>
            `);
        }
        
        return pages.join('');
    }
    
    bindEvents() {
        super.bindEvents();
        
        // Search
        const searchInput = this.findElement('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce((e) => {
                this.search(e.target.value);
            }));
        }
        
        // Sorting
        this.findElements('.sortable-header.sortable').forEach(header => {
            header.addEventListener('click', (e) => {
                const column = e.currentTarget.dataset.column;
                this.sort(column);
            });
        });
        
        // Selection
        const selectAll = this.findElement('.select-all');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                this.selectAll(e.target.checked);
            });
        }
        
        this.findElements('.row-select').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const row = e.target.closest('.data-row');
                const rowId = row.dataset.rowId;
                this.selectRow(rowId, e.target.checked);
            });
        });
        
        // Pagination
        this.findElements('.page-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                const page = parseInt(e.target.dataset.page);
                
                if (page) {
                    this.goToPage(page);
                } else {
                    this.handlePaginationAction(action);
                }
            });
        });
    }
    
    // Data operations
    search(query) {
        this.searchQuery = query.toLowerCase();
        this.applyFilters();
    }
    
    sort(columnKey) {
        if (this.sortColumn === columnKey) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = columnKey;
            this.sortDirection = 'asc';
        }
        
        this.applyFilters();
    }
    
    applyFilters() {
        let filtered = [...this.data];
        
        // Apply search
        if (this.searchQuery) {
            filtered = filtered.filter(row => {
                return this.columns.some(col => {
                    if (col.searchable === false) return false;
                    const value = String(row[col.key] || '').toLowerCase();
                    return value.includes(this.searchQuery);
                });
            });
        }
        
        // Apply sorting
        if (this.sortColumn) {
            filtered.sort((a, b) => {
                const aVal = a[this.sortColumn];
                const bVal = b[this.sortColumn];
                
                let result = 0;
                if (aVal < bVal) result = -1;
                else if (aVal > bVal) result = 1;
                
                return this.sortDirection === 'desc' ? -result : result;
            });
        }
        
        this.filteredData = filtered;
        this.currentPage = 1;
        this.render();
    }
    
    // Selection methods
    selectRow(rowId, selected) {
        if (selected) {
            this.selectedRows.add(rowId);
        } else {
            this.selectedRows.delete(rowId);
        }
        
        this.emit('table:selectionChange', {
            selectedRows: Array.from(this.selectedRows),
            count: this.selectedRows.size
        });
    }
    
    selectAll(selected) {
        const visibleRows = this.getPaginatedData();
        
        visibleRows.forEach((row, index) => {
            const rowId = this.getRowId(row, index);
            if (selected) {
                this.selectedRows.add(rowId);
            } else {
                this.selectedRows.delete(rowId);
            }
        });
        
        this.render();
        this.emit('table:selectionChange', {
            selectedRows: Array.from(this.selectedRows),
            count: this.selectedRows.size
        });
    }
    
    // Pagination methods
    goToPage(page) {
        const totalPages = this.getTotalPages();
        this.currentPage = Math.max(1, Math.min(page, totalPages));
        this.render();
    }
    
    handlePaginationAction(action) {
        switch (action) {
            case 'first':
                this.goToPage(1);
                break;
            case 'prev':
                this.goToPage(this.currentPage - 1);
                break;
            case 'next':
                this.goToPage(this.currentPage + 1);
                break;
            case 'last':
                this.goToPage(this.getTotalPages());
                break;
        }
    }
    
    // Utility methods
    getPaginatedData() {
        if (!this.options.pagination) return this.filteredData;
        
        const start = this.getStartIndex();
        const end = this.getEndIndex();
        return this.filteredData.slice(start, end);
    }
    
    getStartIndex() {
        return (this.currentPage - 1) * this.options.pageSize;
    }
    
    getEndIndex() {
        return Math.min(this.getStartIndex() + this.options.pageSize, this.filteredData.length);
    }
    
    getTotalPages() {
        return Math.ceil(this.filteredData.length / this.options.pageSize);
    }
    
    getColumnCount() {
        let count = this.columns ? this.columns.length : 0;
        if (this.options.selectable) count++;
        return count;
    }
    
    getRowId(row, index) {
        return row.id || row._id || index;
    }
    
    formatColumnTitle(key) {
        return key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1');
    }
    
    formatCellValue(value, column, row) {
        if (column.formatter && typeof column.formatter === 'function') {
            return column.formatter(value, row);
        }
        
        if (value === null || value === undefined) {
            return '<span class="null-value">—</span>';
        }
        
        return String(value);
    }
    
    // Public API
    getSelectedRows() {
        return Array.from(this.selectedRows);
    }
    
    clearSelection() {
        this.selectedRows.clear();
        this.render();
    }
    
    refresh() {
        this.applyFilters();
    }
}



// =================================================================
// DATA TABLE COMPONENT
// =================================================================
class DataTable {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            selectable: false,
            sortable: true,
            filterable: false,
            pagination: false,
            pageSize: 10,
            ...options
        };
        
        this.data = [];
        this.filteredData = [];
        this.columns = [];
        this.selectedRows = new Set();
        this.sortColumn = null;
        this.sortDirection = 'asc';
        this.currentPage = 1;
        
        this.container = document.getElementById(containerId);
        this.init();
    }
    
    init() {
        if (this.container) {
            this.render();
        }
    }
    
    setData(data) {
        this.data = data;
        this.filteredData = [...data];
        this.render();
    }
    
    setColumns(columns) {
        this.columns = columns;
        this.render();
    }
    
    render() {
        if (!this.container) return;
        
        let html = '<div class="data-table-wrapper">';
        
        // Add filter if enabled
        if (this.options.filterable) {
            html += this.renderFilter();
        }
        
        // Add table
        html += '<table class="data-table">';
        html += this.renderHeader();
        html += this.renderBody();
        html += '</table>';
        
        // Add pagination if enabled
        if (this.options.pagination) {
            html += this.renderPagination();
        }
        
        html += '</div>';
        
        this.container.innerHTML = html;
        this.setupEventListeners();
    }
    
    renderFilter() {
        return `
            <div class="table-filter">
                <input type="text" 
                       class="filter-input" 
                       placeholder="Search..." 
                       onkeyup="this.closest('.data-table-wrapper').dispatchEvent(new CustomEvent('filter', {detail: this.value}))">
            </div>
        `;
    }
    
    renderHeader() {
        let html = '<thead><tr>';
        
        if (this.options.selectable) {
            html += '<th><input type="checkbox" class="select-all"></th>';
        }
        
        this.columns.forEach(column => {
            const sortable = this.options.sortable && column.sortable !== false;
            const sortClass = this.sortColumn === column.key ? `sorted-${this.sortDirection}` : '';
            
            html += `
                <th class="${sortable ? 'sortable' : ''} ${sortClass}" 
                    ${sortable ? `data-column="${column.key}"` : ''}>
                    ${column.title || this.formatColumnTitle(column.key)}
                    ${sortable ? '<span class="sort-indicator"></span>' : ''}
                </th>
            `;
        });
        
        html += '</tr></thead>';
        return html;
    }
    
    renderBody() {
        let html = '<tbody>';
        
        const dataToRender = this.options.pagination 
            ? this.getPaginatedData() 
            : this.filteredData;
        
        dataToRender.forEach((row, index) => {
            const rowId = this.getRowId(row, index);
            const selected = this.selectedRows.has(rowId);
            
            html += `<tr class="${selected ? 'selected' : ''}" data-row-id="${rowId}">`;
            
            if (this.options.selectable) {
                html += `<td><input type="checkbox" class="row-select" ${selected ? 'checked' : ''}></td>`;
            }
            
            this.columns.forEach(column => {
                const value = row[column.key];
                const formattedValue = this.formatCellValue(value, column, row);
                html += `<td>${formattedValue}</td>`;
            });
            
            html += '</tr>';
        });
        
        html += '</tbody>';
        return html;
    }
    
    renderPagination() {
        const totalPages = Math.ceil(this.filteredData.length / this.options.pageSize);
        
        if (totalPages <= 1) return '';
        
        let html = '<div class="table-pagination">';
        
        // Previous button
        html += `<button class="page-btn" ${this.currentPage === 1 ? 'disabled' : ''} 
                         onclick="this.closest('.data-table-wrapper').dispatchEvent(new CustomEvent('page', {detail: ${this.currentPage - 1}}))">
                    Previous
                 </button>`;
        
        // Page numbers
        for (let i = 1; i <= totalPages; i++) {
            html += `<button class="page-btn ${i === this.currentPage ? 'active' : ''}" 
                             onclick="this.closest('.data-table-wrapper').dispatchEvent(new CustomEvent('page', {detail: ${i}}))">
                        ${i}
                     </button>`;
        }
        
        // Next button
        html += `<button class="page-btn" ${this.currentPage === totalPages ? 'disabled' : ''} 
                         onclick="this.closest('.data-table-wrapper').dispatchEvent(new CustomEvent('page', {detail: ${this.currentPage + 1}}))">
                    Next
                 </button>`;
        
        html += '</div>';
        return html;
    }
    
    setupEventListeners() {
        // Sort handlers
        if (this.options.sortable) {
            this.container.querySelectorAll('th.sortable').forEach(th => {
                th.addEventListener('click', () => {
                    const column = th.dataset.column;
                    this.sort(column);
                });
            });
        }
        
        // Selection handlers
        if (this.options.selectable) {
            const selectAll = this.container.querySelector('.select-all');
            if (selectAll) {
                selectAll.addEventListener('change', (e) => {
                    this.selectAll(e.target.checked);
                });
            }
            
            this.container.querySelectorAll('.row-select').forEach(checkbox => {
                checkbox.addEventListener('change', (e) => {
                    const row = e.target.closest('tr');
                    const rowId = row.dataset.rowId;
                    this.selectRow(rowId, e.target.checked);
                });
            });
        }
        
        // Filter handler
        this.container.addEventListener('filter', (e) => {
            this.filter(e.detail);
        });
        
        // Pagination handler
        this.container.addEventListener('page', (e) => {
            this.goToPage(e.detail);
        });
    }
    
    sort(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'asc';
        }
        
        this.filteredData.sort((a, b) => {
            const aVal = a[column];
            const bVal = b[column];
            
            if (aVal < bVal) return this.sortDirection === 'asc' ? -1 : 1;
            if (aVal > bVal) return this.sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
        
        this.render();
    }
    
    filter(searchTerm) {
        if (!searchTerm) {
            this.filteredData = [...this.data];
        } else {
            this.filteredData = this.data.filter(row => {
                return Object.values(row).some(value => 
                    String(value).toLowerCase().includes(searchTerm.toLowerCase())
                );
            });
        }
        
        this.currentPage = 1;
        this.render();
    }
    
    selectRow(rowId, selected) {
        if (selected) {
            this.selectedRows.add(rowId);
        } else {
            this.selectedRows.delete(rowId);
        }
        
        this.updateRowSelection(rowId, selected);
    }
    
    selectAll(selected) {
        this.selectedRows.clear();
        
        if (selected) {
            this.filteredData.forEach((row, index) => {
                const rowId = this.getRowId(row, index);
                this.selectedRows.add(rowId);
            });
        }
        
        this.render();
    }
    
    updateRowSelection(rowId, selected) {
        const row = this.container.querySelector(`tr[data-row-id="${rowId}"]`);
        if (row) {
            row.classList.toggle('selected', selected);
            const checkbox = row.querySelector('.row-select');
            if (checkbox) {
                checkbox.checked = selected;
            }
        }
    }
    
    goToPage(page) {
        const totalPages = Math.ceil(this.filteredData.length / this.options.pageSize);
        
        if (page >= 1 && page <= totalPages) {
            this.currentPage = page;
            this.render();
        }
    }
    
    getPaginatedData() {
        const start = (this.currentPage - 1) * this.options.pageSize;
        const end = start + this.options.pageSize;
        return this.filteredData.slice(start, end);
    }
    
    // Helper methods
    getColumnCount() {
        return this.columns ? this.columns.length : 0;
    }
    
    getRowId(row, index) {
        return row.id || row._id || index;
    }
    
    formatColumnTitle(key) {
        return key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1');
    }
    
    formatCellValue(value, column, row) {
        if (column.formatter && typeof column.formatter === 'function') {
            return column.formatter(value, row);
        }
        
        if (value === null || value === undefined) {
            return '<span class="null-value">—</span>';
        }
        
        return String(value);
    }
    
    // Public API
    getSelectedRows() {
        return Array.from(this.selectedRows);
    }
    
    clearSelection() {
        this.selectedRows.clear();
        this.render();
    }
    
    refresh() {
        this.render();
    }
}

// =================================================================
// LOADING SPINNER COMPONENT
// =================================================================
class LoadingSpinner {
    static show(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <div class="loading-message">${message}</div>
                </div>
            `;
        }
    }
    
    static hide(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            const spinner = container.querySelector('.loading-spinner');
            if (spinner) {
                spinner.remove();
            }
        }
    }
}

// Add shared component styles
if (!document.querySelector('#shared-component-styles')) {
    const style = document.createElement('style');
    style.id = 'shared-component-styles';
    style.textContent = `
        /* Navigation Styles */
        .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
            background: #1f2937;
            color: white;
        }
        
        .nav-brand a {
            color: white;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
        }
        
        .nav-toggle {
            display: none;
            flex-direction: column;
            cursor: pointer;
        }
        
        .nav-toggle span {
            width: 25px;
            height: 3px;
            background: white;
            margin: 3px 0;
            transition: 0.3s;
        }
        
        .nav-menu {
            display: flex;
            gap: 20px;
        }
        
        .nav-link {
            color: #d1d5db;
            text-decoration: none;
            padding: 8px 12px;
            border-radius: 4px;
            transition: all 0.3s;
        }
        
        .nav-link:hover,
        .nav-link.active {
            background: #374151;
            color: white;
        }
        
        /* Data Table Styles */
        .data-table-wrapper {
            overflow-x: auto;
        }
        
        .table-filter {
            margin-bottom: 15px;
        }
        
        .filter-input {
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            width: 300px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        
        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .data-table th {
            background: #f9fafb;
            font-weight: 600;
            position: relative;
        }
        
        .data-table th.sortable {
            cursor: pointer;
            user-select: none;
        }
        
        .data-table th.sortable:hover {
            background: #f3f4f6;
        }
        
        .sort-indicator::after {
            content: '↕';
            position: absolute;
            right: 8px;
            opacity: 0.5;
        }
        
        .data-table th.sorted-asc .sort-indicator::after {
            content: '↑';
            opacity: 1;
        }
        
        .data-table th.sorted-desc .sort-indicator::after {
            content: '↓';
            opacity: 1;
        }
        
        .data-table tr.selected {
            background: #eff6ff;
        }
        
        .table-pagination {
            margin-top: 15px;
            display: flex;
            gap: 5px;
            justify-content: center;
        }
        
        .page-btn {
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            background: white;
            cursor: pointer;
            border-radius: 4px;
        }
        
        .page-btn:hover:not(:disabled) {
            background: #f3f4f6;
        }
        
        .page-btn.active {
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
        
        .page-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Loading Spinner Styles */
        .loading-spinner {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f4f6;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .loading-message {
            margin-top: 15px;
            color: #6b7280;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .nav-toggle {
                display: flex;
            }
            
            .nav-menu {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: #1f2937;
                flex-direction: column;
                padding: 20px;
            }
            
            .nav-menu:not(.collapsed) {
                display: flex;
            }
            
            .filter-input {
                width: 100%;
            }
        }
    `;
    document.head.appendChild(style);
}



// =================================================================
// INITIALIZE GLOBAL INSTANCES
// =================================================================

// Auto-initialize shared components when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize global navigation
    if (document.getElementById('global-nav')) {
        window.Navigation = new Navigation();
    }
    
    // Initialize toast system
    window.Toast = new Toast();
    
    // Initialize modal system
    window.Modal = new Modal();
    
    console.log('✅ Shared components initialized');
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Navigation, Toast, Modal, DataTable };
}