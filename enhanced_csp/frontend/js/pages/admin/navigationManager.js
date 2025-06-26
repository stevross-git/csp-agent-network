/**
 * Navigation Manager - Handles sidebar navigation and section switching
 * Part of Enhanced CSP Admin Portal
 */

export class NavigationManager {
    constructor(adminPage) {
        this.adminPage = adminPage;
        this.currentSection = 'dashboard';
        this.navigationItems = new Map();
        this.breadcrumbs = [];
    }

    /**
     * Initialize navigation manager
     */
    async init() {
        try {
            console.log('🧭 Initializing Navigation Manager...');
            
            // Set up navigation items
            this.setupNavigationItems();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize breadcrumbs
            this.updateBreadcrumbs();
            
            console.log('✅ Navigation Manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Navigation Manager:', error);
            throw error;
        }
    }

    /**
     * Set up navigation items
     */
    setupNavigationItems() {
        this.navigationItems.set('dashboard', {
            title: 'Dashboard',
            icon: 'fas fa-chart-line',
            section: 'dashboard',
            description: 'System overview and real-time metrics'
        });

        this.navigationItems.set('agents', {
            title: 'AI Agents',
            icon: 'fas fa-robot',
            section: 'agents',
            description: 'Manage and monitor AI agents'
        });

        this.navigationItems.set('monitoring', {
            title: 'System Monitoring',
            icon: 'fas fa-desktop',
            section: 'monitoring',
            description: 'System health and performance monitoring'
        });

        this.navigationItems.set('security', {
            title: 'Security Center',
            icon: 'fas fa-shield-alt',
            section: 'security',
            description: 'Security monitoring and threat detection'
        });

        this.navigationItems.set('logs', {
            title: 'System Logs',
            icon: 'fas fa-file-alt',
            section: 'logs',
            description: 'View and search system logs'
        });

        this.navigationItems.set('settings', {
            title: 'Settings',
            icon: 'fas fa-cog',
            section: 'settings',
            description: 'System configuration and preferences'
        });

        this.navigationItems.set('users', {
            title: 'User Management',
            icon: 'fas fa-users',
            section: 'users',
            description: 'Manage users and permissions'
        });

        this.navigationItems.set('backup', {
            title: 'Backup & Recovery',
            icon: 'fas fa-database',
            section: 'backup',
            description: 'Backup management and recovery'
        });
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Handle navigation item clicks
        document.querySelectorAll('.nav-item[data-section]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const sectionId = item.getAttribute('data-section');
                this.navigateToSection(sectionId);
            });
        });

        // Handle keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.altKey) {
                this.handleKeyboardNavigation(e);
            }
        });

        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.section) {
                this.navigateToSection(e.state.section, false);
            }
        });
    }

    /**
     * Navigate to a section
     */
    navigateToSection(sectionId, updateHistory = true) {
        try {
            // Validate section exists
            if (!this.navigationItems.has(sectionId) && !document.getElementById(sectionId)) {
                console.warn(`⚠️ Section '${sectionId}' not found`);
                return;
            }

            // Update current section
            const previousSection = this.currentSection;
            this.currentSection = sectionId;

            // Update admin page state
            this.adminPage.setState({ currentSection: sectionId });

            // Show the section
            this.adminPage.showSection(sectionId);

            // Update navigation UI
            this.updateNavigationUI(sectionId);

            // Update breadcrumbs
            this.updateBreadcrumbs(sectionId);

            // Update browser history
            if (updateHistory) {
                this.updateBrowserHistory(sectionId);
            }

            // Close mobile sidebar if open
            if (window.innerWidth <= 768) {
                const sidebar = document.querySelector('.sidebar');
                if (sidebar) {
                    sidebar.classList.remove('open');
                }
            }

            // Notify other managers
            this.notifyNavigationChange(sectionId, previousSection);

            console.log(`📄 Navigated to section: ${sectionId}`);

        } catch (error) {
            console.error('❌ Navigation failed:', error);
            this.adminPage.showError('Navigation Error', `Failed to navigate to ${sectionId}`);
        }
    }

    /**
     * Update navigation UI
     */
    updateNavigationUI(sectionId) {
        // Remove active class from all nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            item.setAttribute('aria-selected', 'false');
        });

        // Add active class to current nav item
        const activeItem = document.querySelector(`[data-section="${sectionId}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
            activeItem.setAttribute('aria-selected', 'true');

            // Scroll active item into view if needed
            this.scrollNavItemIntoView(activeItem);
        }

        // Update page title
        this.updatePageTitle(sectionId);
    }

    /**
     * Scroll navigation item into view
     */
    scrollNavItemIntoView(item) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar && item) {
            const itemRect = item.getBoundingClientRect();
            const sidebarRect = sidebar.getBoundingClientRect();

            if (itemRect.top < sidebarRect.top || itemRect.bottom > sidebarRect.bottom) {
                item.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest'
                });
            }
        }
    }

    /**
     * Update page title
     */
    updatePageTitle(sectionId) {
        const navItem = this.navigationItems.get(sectionId);
        const title = navItem ? navItem.title : sectionId;
        document.title = `${title} - Enhanced CSP Admin Portal`;
    }

    /**
     * Update breadcrumbs
     */
    updateBreadcrumbs(sectionId = this.currentSection) {
        this.breadcrumbs = [
            { title: 'Admin Portal', section: 'dashboard' }
        ];

        if (sectionId !== 'dashboard') {
            const navItem = this.navigationItems.get(sectionId);
            if (navItem) {
                this.breadcrumbs.push({
                    title: navItem.title,
                    section: sectionId
                });
            }
        }

        this.renderBreadcrumbs();
    }

    /**
     * Render breadcrumbs
     */
    renderBreadcrumbs() {
        let breadcrumbContainer = document.querySelector('.breadcrumbs');
        
        if (!breadcrumbContainer) {
            breadcrumbContainer = document.createElement('nav');
            breadcrumbContainer.className = 'breadcrumbs';
            breadcrumbContainer.setAttribute('aria-label', 'Breadcrumb navigation');

            const contentArea = document.querySelector('.content-area');
            if (contentArea) {
                contentArea.insertBefore(breadcrumbContainer, contentArea.firstChild);
            }
        }

        const breadcrumbHTML = this.breadcrumbs.map((crumb, index) => {
            const isLast = index === this.breadcrumbs.length - 1;
            return `
                <span class="breadcrumb-item ${isLast ? 'active' : ''}">
                    ${isLast ? 
                        crumb.title : 
                        `<a href="#" data-section="${crumb.section}" onclick="window.adminPage.getManager('navigation').navigateToSection('${crumb.section}')">${crumb.title}</a>`
                    }
                </span>
            `;
        }).join('<span class="breadcrumb-separator"> / </span>');

        breadcrumbContainer.innerHTML = breadcrumbHTML;
    }

    /**
     * Update browser history
     */
    updateBrowserHistory(sectionId) {
        const navItem = this.navigationItems.get(sectionId);
        const title = navItem ? navItem.title : sectionId;
        
        const state = { section: sectionId };
        const url = `#${sectionId}`;
        
        history.pushState(state, title, url);
    }

    /**
     * Handle keyboard navigation
     */
    handleKeyboardNavigation(e) {
        const navItems = Array.from(document.querySelectorAll('.nav-item[data-section]'));
        const currentIndex = navItems.findIndex(item => 
            item.getAttribute('data-section') === this.currentSection
        );

        let nextIndex;

        switch (e.key) {
            case 'ArrowUp':
                e.preventDefault();
                nextIndex = currentIndex > 0 ? currentIndex - 1 : navItems.length - 1;
                break;
            case 'ArrowDown':
                e.preventDefault();
                nextIndex = currentIndex < navItems.length - 1 ? currentIndex + 1 : 0;
                break;
            case 'Home':
                e.preventDefault();
                nextIndex = 0;
                break;
            case 'End':
                e.preventDefault();
                nextIndex = navItems.length - 1;
                break;
            default:
                return;
        }

        if (nextIndex !== undefined && navItems[nextIndex]) {
            const nextSection = navItems[nextIndex].getAttribute('data-section');
            this.navigateToSection(nextSection);
        }
    }

    /**
     * Notify other managers about navigation change
     */
    notifyNavigationChange(newSection, previousSection) {
        // Notify dashboard manager
        const dashboardManager = this.adminPage.getManager('dashboard');
        if (dashboardManager && dashboardManager.onSectionChange) {
            dashboardManager.onSectionChange(newSection, previousSection);
        }

        // Notify agent manager
        const agentManager = this.adminPage.getManager('agent');
        if (agentManager && agentManager.onSectionChange) {
            agentManager.onSectionChange(newSection, previousSection);
        }

        // Notify system manager
        const systemManager = this.adminPage.getManager('system');
        if (systemManager && systemManager.onSectionChange) {
            systemManager.onSectionChange(newSection, previousSection);
        }
    }

    /**
     * Get current section
     */
    getCurrentSection() {
        return this.currentSection;
    }

    /**
     * Get navigation items
     */
    getNavigationItems() {
        return new Map(this.navigationItems);
    }

    /**
     * Add navigation item
     */
    addNavigationItem(id, item) {
        this.navigationItems.set(id, item);
        this.renderNavigationItems();
    }

    /**
     * Remove navigation item
     */
    removeNavigationItem(id) {
        this.navigationItems.delete(id);
        this.renderNavigationItems();
    }

    /**
     * Render navigation items
     */
    renderNavigationItems() {
        const navSections = document.querySelectorAll('.nav-section');
        
        navSections.forEach(section => {
            const navItems = section.querySelectorAll('.nav-item[data-section]');
            navItems.forEach(item => {
                const sectionId = item.getAttribute('data-section');
                const navItem = this.navigationItems.get(sectionId);
                
                if (navItem) {
                    item.innerHTML = `
                        <i class="${navItem.icon} nav-icon"></i>
                        <span>${navItem.title}</span>
                    `;
                    item.title = navItem.description;
                }
            });
        });
    }

    /**
     * Toggle sidebar (mobile)
     */
    toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('open');
            
            // Update ARIA attributes
            const isOpen = sidebar.classList.contains('open');
            sidebar.setAttribute('aria-hidden', !isOpen);
            
            // Focus first nav item when opening
            if (isOpen) {
                const firstNavItem = sidebar.querySelector('.nav-item');
                if (firstNavItem) {
                    firstNavItem.focus();
                }
            }
        }
    }

    /**
     * Search navigation items
     */
    searchNavigation(query) {
        const results = [];
        
        this.navigationItems.forEach((item, id) => {
            const searchText = `${item.title} ${item.description}`.toLowerCase();
            if (searchText.includes(query.toLowerCase())) {
                results.push({
                    id,
                    ...item,
                    relevance: this.calculateRelevance(query, item)
                });
            }
        });

        return results.sort((a, b) => b.relevance - a.relevance);
    }

    /**
     * Calculate search relevance
     */
    calculateRelevance(query, item) {
        const lowerQuery = query.toLowerCase();
        const lowerTitle = item.title.toLowerCase();
        const lowerDescription = item.description.toLowerCase();

        let score = 0;

        // Exact title match
        if (lowerTitle === lowerQuery) score += 100;
        // Title starts with query
        else if (lowerTitle.startsWith(lowerQuery)) score += 80;
        // Title contains query
        else if (lowerTitle.includes(lowerQuery)) score += 60;
        // Description contains query
        else if (lowerDescription.includes(lowerQuery)) score += 40;

        return score;
    }

    /**
     * Get navigation history
     */
    getNavigationHistory() {
        return [...this.breadcrumbs];
    }

    /**
     * Go back in navigation
     */
    goBack() {
        if (this.breadcrumbs.length > 1) {
            const previousCrumb = this.breadcrumbs[this.breadcrumbs.length - 2];
            this.navigateToSection(previousCrumb.section);
        }
    }

    /**
     * Handle section change from admin page
     */
    onSectionChange(sectionId) {
        if (sectionId !== this.currentSection) {
            this.currentSection = sectionId;
            this.updateNavigationUI(sectionId);
            this.updateBreadcrumbs(sectionId);
        }
    }

    /**
     * Initialize from URL hash
     */
    initializeFromHash() {
        const hash = window.location.hash.substr(1);
        if (hash && this.navigationItems.has(hash)) {
            this.navigateToSection(hash, false);
        }
    }

    /**
     * Export navigation state
     */
    exportNavigationState() {
        return {
            currentSection: this.currentSection,
            breadcrumbs: [...this.breadcrumbs],
            navigationItems: Object.fromEntries(this.navigationItems)
        };
    }

    /**
     * Import navigation state
     */
    importNavigationState(state) {
        if (state.currentSection) {
            this.navigateToSection(state.currentSection, false);
        }
        
        if (state.breadcrumbs) {
            this.breadcrumbs = [...state.breadcrumbs];
            this.renderBreadcrumbs();
        }
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        // Remove event listeners
        document.removeEventListener('keydown', this.handleKeyboardNavigation);
        window.removeEventListener('popstate', this.handlePopState);
        
        // Clear navigation items
        this.navigationItems.clear();
        this.breadcrumbs = [];
        
        console.log('🧹 Navigation Manager cleaned up');
    }
}

// Add navigation-specific CSS
const navigationStyles = document.createElement('style');
navigationStyles.textContent = `
    .breadcrumbs {
        margin-bottom: 1.5rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.9rem;
    }

    .breadcrumb-item {
        color: var(--text-muted);
    }

    .breadcrumb-item.active {
        color: var(--text-primary);
        font-weight: 600;
    }

    .breadcrumb-item a {
        color: var(--primary);
        text-decoration: none;
        transition: var(--transition);
    }

    .breadcrumb-item a:hover {
        color: var(--primary-light);
        text-decoration: underline;
    }

    .breadcrumb-separator {
        margin: 0 0.5rem;
        color: var(--text-muted);
    }

    .nav-item {
        position: relative;
        outline: none;
    }

    .nav-item:focus {
        box-shadow: inset 0 0 0 2px var(--primary);
    }

    .nav-item::before {
        content: '';
        position: absolute;
        left: -1rem;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 0;
        background: var(--primary);
        transition: height 0.3s ease;
    }

    .nav-item.active::before {
        height: 60%;
    }

    .nav-section {
        position: relative;
    }

    .nav-section::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 1rem;
        right: 1rem;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    }

    @media (max-width: 768px) {
        .sidebar {
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }

        .sidebar.open {
            transform: translateX(0);
        }

        .breadcrumbs {
            font-size: 0.8rem;
            padding: 0.5rem 0;
        }

        .breadcrumb-separator {
            margin: 0 0.25rem;
        }
    }
`;

document.head.appendChild(navigationStyles);