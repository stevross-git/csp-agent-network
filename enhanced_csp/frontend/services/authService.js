/**
 * Enhanced CSP System - Authentication Service
 * Handles both Azure AD and local email/password authentication
 */

class UnifiedAuthService {
    constructor() {
        this.currentUser = null;
        this.authMethod = null;
        this.isInitialized = false;
        this.baseUrl = 'http://localhost:8000'; 
        this.msalInstance = null;
        
        this.authStateListeners = [];
        this.init();
    }

    async init() {
        console.log('🔐 Unified Auth Service initializing...');
        
        // Initialize MSAL for Azure AD if available
        if (typeof msal !== 'undefined') {
            try {
                this.msalInstance = new msal.PublicClientApplication({
                    auth: {
                        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
                        authority: "https://login.microsoftonline.com/622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
                        redirectUri: window.location.origin,
                    },
                    cache: {
                        cacheLocation: "sessionStorage",
                        storeAuthStateInCookie: false,
                    }
                });
                
                await this.msalInstance.initialize();
                console.log('✅ MSAL initialized');
            } catch (error) {
                console.error('❌ MSAL initialization failed:', error);
            }
        } else {
            console.warn('⚠️ MSAL library not loaded, using fallback authentication');
        }

        await this.checkExistingAuth();
        this.isInitialized = true;
        this.notifyAuthStateChange();
        
        console.log('✅ Unified Auth Service initialized');
    }

    async checkExistingAuth() {
        try {
            // Check Azure AD first
            if (this.msalInstance) {
                const accounts = this.msalInstance.getAllAccounts();
                if (accounts.length > 0) {
                    this.msalInstance.setActiveAccount(accounts[0]);
                    const azureUser = await this.getAzureUserInfo();
                    if (azureUser) {
                        this.currentUser = azureUser;
                        this.authMethod = 'azure';
                        console.log('✅ Found existing Azure AD session');
                        return true;
                    }
                }
            }

            // Check local auth tokens
            const localToken = localStorage.getItem('local_access_token');
            if (localToken) {
                const userInfo = await this.validateLocalToken(localToken);
                if (userInfo) {
                    this.currentUser = userInfo;
                    this.authMethod = 'local';
                    console.log('✅ Found existing local session');
                    return true;
                }
            }

        } catch (error) {
            console.error('Auth check failed:', error);
            this.clearStoredTokens();
        }

        return false;
    }

    async registerLocal(email, password, confirmPassword, fullName) {
        try {
            const response = await fetch(`${this.baseUrl}/api/auth/local/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email: email,
                    password: password,
                    confirm_password: confirmPassword,
                    full_name: fullName
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Registration failed');
            }

            window.toast.success('Registration Successful', 
                'Account created successfully! You can now sign in.');
            
            return data;

        } catch (error) {
            console.error('Local registration failed:', error);
            window.toast.error('Registration Failed', error.message);
            throw error;
        }
    }

    async loginLocal(email, password, rememberMe = false) {
        try {
            const response = await fetch(`${this.baseUrl}/api/auth/local/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email: email,
                    password: password,
                    remember_me: rememberMe
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Login failed');
            }

            // Store tokens
            localStorage.setItem('local_access_token', data.access_token);
            localStorage.setItem('local_refresh_token', data.refresh_token);
            
            this.currentUser = {
                ...data.user,
                token: data.access_token,
                auth_method: 'local'
            };
            this.authMethod = 'local';

            this.notifyAuthStateChange();
            window.toast.success('Login Successful', 'Welcome back!');

            return this.currentUser;

        } catch (error) {
            console.error('Local login failed:', error);
            window.toast.error('Login Failed', error.message);
            throw error;
        }
    }

    async validateLocalToken(token) {
        try {
            const response = await fetch(`${this.baseUrl}/api/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const userInfo = await response.json();
                return {
                    ...userInfo,
                    token: token
                };
            }
        } catch (error) {
            console.error('Token validation failed:', error);
        }

        return null;
    }

    async logout() {
        try {
            if (this.currentUser && this.currentUser.token) {
                await fetch(`${this.baseUrl}/api/auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.currentUser.token}`
                    }
                });
            }

            if (this.authMethod === 'azure' && this.msalInstance) {
                await this.msalInstance.logoutSilent();
            }

            this.clearStoredTokens();
            
            this.currentUser = null;
            this.authMethod = null;
            
            this.notifyAuthStateChange();
            window.toast.info('Signed Out', 'You have been successfully signed out');

        } catch (error) {
            console.error('Logout failed:', error);
            this.clearStoredTokens();
            this.currentUser = null;
            this.authMethod = null;
            this.notifyAuthStateChange();
        }
    }

    clearStoredTokens() {
        ['local_access_token', 'local_refresh_token', 'azure_access_token'].forEach(key => {
            localStorage.removeItem(key);
        });
    }

    onAuthStateChange(callback) {
        this.authStateListeners.push(callback);
        return () => {
            const index = this.authStateListeners.indexOf(callback);
            if (index > -1) {
                this.authStateListeners.splice(index, 1);
            }
        };
    }

    notifyAuthStateChange() {
        this.authStateListeners.forEach(callback => {
            try {
                callback(this.currentUser, this.authMethod);
            } catch (error) {
                console.error('Auth state listener error:', error);
            }
        });
    }

    isAuthenticated() {
        return this.currentUser !== null;
    }

    getCurrentUser() {
        return this.currentUser;
    }

    getAuthMethod() {
        return this.authMethod;
    }

    getToken() {
        return this.currentUser ? this.currentUser.token : null;
    }
}

// Create global instance
window.authService = new UnifiedAuthService();

export default window.authService;
