/**
 * Enhanced CSP System - Unified Authentication Implementation
 * Combines Azure AD with universal page protection
 */

class UnifiedCSPAuth {
    constructor(options = {}) {
        this.options = {
            loginPage: '/pages/login.html',
            azureClientId: '53537e30-ae6b-48f7-9c7c-4db20fc27850',
            azureTenantId: '622a5fe0-fac1-4213-9cf7-d5f6defdf4c4',
            redirectUri: window.location.origin,
            sessionTimeout: 8 * 60 * 60 * 1000, // 8 hours
            ...options
        };

        this.msalInstance = null;
        this.currentUser = null;
        this.sessionTimer = null;
        this.isInitialized = false;

        this.USER_ROLES = {
            SUPER_ADMIN: 'super_admin',
            ADMIN: 'admin',
            DEVELOPER: 'developer',
            ANALYST: 'analyst',
            USER: 'user'
        };

        this.PERMISSIONS = {
            'system.view': ['super_admin', 'admin', 'developer', 'analyst', 'user'],
            'system.manage': ['super_admin', 'admin'],
            'user.manage': ['super_admin', 'admin'],
            'ai.manage': ['super_admin', 'admin', 'developer'],
            'security.view': ['super_admin', 'admin'],
            'settings.manage': ['super_admin', 'admin']
        };

        this.init();
    }

    async init() {
        console.log('🔐 Initializing Unified CSP Authentication...');
        
        // Skip auth on login page
        if (this.isLoginPage()) {
            console.log('📄 Login page detected - skipping auth check');
            return;
        }

        try {
            await this.initializeAzureAD();
            await this.performAuthCheck();
            this.setupEventListeners();
            this.isInitialized = true;
            
            // Dispatch ready event
            this.dispatchAuthEvent('cspAuthReady', { user: this.currentUser });
        } catch (error) {
            console.error('❌ Auth initialization failed:', error);
            this.handleAuthError(error);
        }
    }

    async initializeAzureAD() {
        if (typeof msal === 'undefined') {
            throw new Error('MSAL library not loaded');
        }

        const msalConfig = {
            auth: {
                clientId: this.options.azureClientId,
                authority: `https://login.microsoftonline.com/${this.options.azureTenantId}`,
                redirectUri: this.options.redirectUri,
                postLogoutRedirectUri: this.options.redirectUri,
                navigateToLoginRequestUrl: false
            },
            cache: {
                cacheLocation: "sessionStorage",
                storeAuthStateInCookie: false,
            }
        };

        this.msalInstance = new msal.PublicClientApplication(msalConfig);
        await this.msalInstance.initialize();

        // Handle redirect response
        const response = await this.msalInstance.handleRedirectPromise();
        if (response) {
            console.log('✅ Azure AD redirect response received');
            this.processAzureADResponse(response);
        }
    }

    async performAuthCheck() {
        console.log('🔍 Performing authentication check...');

        // First, check for existing Azure AD session
        const azureAccount = this.msalInstance.getActiveAccount();
        if (azureAccount) {
            console.log('✅ Active Azure AD session found');
            await this.setUserFromAzureAD(azureAccount);
            this.setupAuthenticatedEnvironment();
            return true;
        }

        // Check for stored session
        const storedSession = this.getStoredSession();
        if (storedSession && this.isValidSession(storedSession)) {
            console.log('✅ Valid stored session found');
            this.currentUser = storedSession;
            this.setupAuthenticatedEnvironment();
            return true;
        }

        // No valid authentication found
        console.log('❌ No valid authentication - redirecting to login');
        this.redirectToLogin();
        return false;
    }

    async setUserFromAzureAD(account) {
        const email = account.username || account.idTokenClaims?.preferred_username || '';
        const userRole = this.determineUserRole(email, account);

        this.currentUser = {
            id: account.homeAccountId,
            name: account.name || account.idTokenClaims?.name || 'Unknown',
            email: email,
            role: userRole,
            authMethod: 'azure_ad',
            loginTime: new Date().toISOString(),
            sessionId: this.generateSessionId(),
            azureAccount: account
        };

        // Store unified session
        this.storeSession(this.currentUser);
        console.log(`✅ User authenticated: ${this.currentUser.name} (${this.currentUser.role})`);
    }

    determineUserRole(email, account) {
        // Role determination logic - customize based on your needs
        const emailDomain = email.split('@')[1];
        const emailLocal = email.split('@')[0];

        // Admin users
        if (emailLocal.includes('admin') || emailDomain === 'admin.company.com') {
            return this.USER_ROLES.ADMIN;
        }
        
        // Developers
        if (emailLocal.includes('dev') || emailDomain === 'dev.company.com') {
            return this.USER_ROLES.DEVELOPER;
        }
        
        // Analysts
        if (emailLocal.includes('analyst') || emailDomain === 'analytics.company.com') {
            return this.USER_ROLES.ANALYST;
        }

        // Check Azure AD groups (if configured)
        const groups = account.idTokenClaims?.groups || [];
        if (groups.includes('admin-group-id')) return this.USER_ROLES.ADMIN;
        if (groups.includes('dev-group-id')) return this.USER_ROLES.DEVELOPER;

        // Default role
        return this.USER_ROLES.USER;
    }

    setupAuthenticatedEnvironment() {
        // Show page content
        document.body.style.opacity = '1';
        document.body.classList.add('authenticated');
        document.documentElement.setAttribute('data-user-role', this.currentUser.role);

        // Add auth header
        this.addAuthHeader();

        // Check page-specific access
        this.checkPageAccess();

        // Start session monitoring
        this.startSessionMonitoring();

        console.log('🎉 Authenticated environment setup complete');
    }

    addAuthHeader() {
        // Remove existing header
        const existingHeader = document.getElementById('csp-auth-header');
        if (existingHeader) existingHeader.remove();

        const header = document.createElement('div');
        header.id = 'csp-auth-header';
        header.innerHTML = `
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        color: white; padding: 0.75rem 1.5rem; font-size: 0.9rem;
                        display: flex; justify-content: space-between; align-items: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 1000;
                        position: fixed; top: 0; left: 0; right: 0;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-weight: 600;">🔐 CSP System</span>
                    <span style="opacity: 0.8;">Welcome, ${this.currentUser.name}</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; 
                                border-radius: 12px; font-size: 0.8rem;">
                        ${this.currentUser.role.replace('_', ' ').toUpperCase()}
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <button onclick="CSPAuth.logout()" 
                            style="background: rgba(220, 53, 69, 0.8); border: none;
                                   color: white; padding: 0.4rem 0.8rem; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.8rem;">
                        🚪 Logout
                    </button>
                </div>
            </div>
        `;

        document.body.insertAdjacentElement('afterbegin', header);
        
        // Adjust page content for header
        if (!document.body.style.paddingTop) {
            document.body.style.paddingTop = '60px';
        }
    }

    checkPageAccess() {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const userRole = this.currentUser.role;

        // Define page access rules
        const restrictedPages = {
            'security.html': ['super_admin', 'admin'],
            'settings.html': ['super_admin', 'admin'],
            'users.html': ['super_admin', 'admin'],
            'system.html': ['super_admin', 'admin'],
            'logs.html': ['super_admin', 'admin'],
            'ai-models.html': ['super_admin', 'admin', 'developer']
        };

        const allowedRoles = restrictedPages[currentPage];
        if (allowedRoles && !allowedRoles.includes(userRole)) {
            this.showAccessDenied(currentPage, userRole, allowedRoles);
            return false;
        }

        return true;
    }

    showAccessDenied(page, userRole, requiredRoles) {
        document.body.innerHTML = `
            <div style="display: flex; justify-content: center; align-items: center; 
                        height: 100vh; background: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <div style="text-align: center; background: white; padding: 3rem; 
                            border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 500px;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🚫</div>
                    <h2 style="color: #dc3545; margin-bottom: 1rem;">Access Denied</h2>
                    <p style="color: #6c757d; margin-bottom: 2rem;">
                        You don't have permission to access <strong>${page}</strong><br>
                        Your role: <span style="background: #e9ecef; padding: 0.25rem 0.5rem; border-radius: 4px;">${userRole}</span><br>
                        Required: <span style="background: #fff3cd; padding: 0.25rem 0.5rem; border-radius: 4px;">${requiredRoles.join(', ')}</span>
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center;">
                        <button onclick="history.back()" 
                                style="background: #6c757d; color: white; border: none; padding: 0.75rem 1.5rem; 
                                       border-radius: 6px; cursor: pointer;">← Go Back</button>
                        <button onclick="CSPAuth.logout()" 
                                style="background: #dc3545; color: white; border: none; padding: 0.75rem 1.5rem; 
                                       border-radius: 6px; cursor: pointer;">🚪 Logout</button>
                    </div>
                </div>
            </div>
        `;
    }

    // Permission checking methods
    hasPermission(permission) {
        const allowedRoles = this.PERMISSIONS[permission];
        return allowedRoles && allowedRoles.includes(this.currentUser?.role);
    }

    hasAnyPermission(permissions) {
        return permissions.some(permission => this.hasPermission(permission));
    }

    hasAllPermissions(permissions) {
        return permissions.every(permission => this.hasPermission(permission));
    }

    // Session management
    storeSession(user) {
        const sessionData = JSON.stringify(user);
        sessionStorage.setItem('csp_unified_session', sessionData);
        
        // Clear any old session formats
        ['csp_session', 'csp_azure_session', 'csp_demo_session'].forEach(key => {
            localStorage.removeItem(key);
            sessionStorage.removeItem(key);
        });
    }

    getStoredSession() {
        try {
            const sessionData = sessionStorage.getItem('csp_unified_session');
            return sessionData ? JSON.parse(sessionData) : null;
        } catch (error) {
            console.error('Error parsing stored session:', error);
            return null;
        }
    }

    isValidSession(session) {
        if (!session || !session.loginTime || !session.sessionId) {
            return false;
        }

        const loginTime = new Date(session.loginTime);
        const now = new Date();
        const sessionAge = now.getTime() - loginTime.getTime();

        return sessionAge < this.options.sessionTimeout;
    }

    startSessionMonitoring() {
        // Clear existing timer
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
        }

        // Check session every 5 minutes
        this.sessionTimer = setInterval(() => {
            if (!this.isValidSession(this.currentUser)) {
                console.log('⏰ Session expired');
                this.handleSessionExpiry();
            }
        }, 5 * 60 * 1000);
    }

    handleSessionExpiry() {
        alert('🔒 Your session has expired. Please log in again.');
        this.logout();
    }

    // Authentication actions
    async login() {
        if (!this.msalInstance) {
            throw new Error('Azure AD not initialized');
        }

        try {
            const loginRequest = {
                scopes: ["User.Read", "User.ReadBasic.All"],
                prompt: "select_account"
            };

            const response = await this.msalInstance.loginPopup(loginRequest);
            await this.setUserFromAzureAD(response.account);
            this.setupAuthenticatedEnvironment();
            
            // Redirect to intended page if stored
            const redirectUrl = sessionStorage.getItem('csp_redirect_after_login');
            if (redirectUrl) {
                sessionStorage.removeItem('csp_redirect_after_login');
                window.location.href = redirectUrl;
            }

            return response;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    async logout() {
        console.log('🚪 Logout initiated');

        try {
            // Clear session data
            this.clearSession();

            // Azure AD logout
            if (this.msalInstance && this.currentUser?.authMethod === 'azure_ad') {
                await this.msalInstance.logoutPopup({
                    postLogoutRedirectUri: this.options.redirectUri
                });
            } else {
                window.location.href = this.options.loginPage;
            }
        } catch (error) {
            console.error('Logout error:', error);
            // Force redirect even if Azure logout fails
            window.location.href = this.options.loginPage;
        }
    }

    clearSession() {
        // Clear all session storage
        ['csp_unified_session', 'csp_session', 'csp_azure_session', 'csp_demo_session'].forEach(key => {
            localStorage.removeItem(key);
            sessionStorage.removeItem(key);
        });

        // Clear timers
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
            this.sessionTimer = null;
        }

        this.currentUser = null;
        console.log('🗑️ Session cleared');
    }

    // Utility methods
    isLoginPage() {
        return window.location.pathname.includes('login.html');
    }

    redirectToLogin() {
        sessionStorage.setItem('csp_redirect_after_login', window.location.href);
        window.location.href = this.options.loginPage;
    }

    generateSessionId() {
        return 'csp_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
    }

    dispatchAuthEvent(eventName, data) {
        const event = new CustomEvent(eventName, { detail: data });
        document.dispatchEvent(event);
    }

    handleAuthError(error) {
        console.error('Authentication error:', error);
        // Could dispatch error event or show user-friendly error
        this.dispatchAuthEvent('cspAuthError', { error: error.message });
    }

    setupEventListeners() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.currentUser) {
                // Refresh auth check when page becomes visible
                this.performAuthCheck();
            }
        });

        // Handle storage changes (for multi-tab support)
        window.addEventListener('storage', (e) => {
            if (e.key === 'csp_unified_session' && e.newValue === null) {
                // Session was cleared in another tab
                console.log('🔄 Session cleared in another tab');
                this.logout();
            }
        });
    }

    // Public API
    isAuthenticated() {
        return this.currentUser !== null;
    }

    getCurrentUser() {
        return this.currentUser;
    }

    getUserRole() {
        return this.currentUser?.role || this.USER_ROLES.USER;
    }

    getUserPermissions() {
        const role = this.getUserRole();
        const permissions = [];
        
        Object.entries(this.PERMISSIONS).forEach(([permission, allowedRoles]) => {
            if (allowedRoles.includes(role)) {
                permissions.push(permission);
            }
        });
        
        return permissions;
    }
}

// Initialize authentication when DOM is ready
function initializeCSPAuth() {
    console.log('🔐 CSP Authentication script loaded');
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.CSPAuth = new UnifiedCSPAuth();
        });
    } else {
        window.CSPAuth = new UnifiedCSPAuth();
    }
}

// Auto-initialize
initializeCSPAuth();