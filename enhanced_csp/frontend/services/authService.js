// services/authService.js
import { PublicClientApplication } from "@azure/msal-browser";
import { msalConfig, loginRequest, tokenRequest } from "../config/authConfig.js";
import { getUserRole } from "../config/roles.js";

class AuthService {
    constructor() {
        this.msalInstance = new PublicClientApplication(msalConfig);
        this.account = null;
        this.userRole = null;
        this.initializeAuth();
    }

    async initializeAuth() {
        await this.msalInstance.initialize();
        this.account = this.msalInstance.getActiveAccount();
        
        if (this.account) {
            await this.setUserRole();
        }
    }

    async login() {
        try {
            const loginResponse = await this.msalInstance.loginPopup(loginRequest);
            this.account = loginResponse.account;
            this.msalInstance.setActiveAccount(this.account);
            await this.setUserRole();
            return loginResponse;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    async loginRedirect() {
        try {
            await this.msalInstance.loginRedirect(loginRequest);
        } catch (error) {
            console.error('Login redirect failed:', error);
            throw error;
        }
    }

    async logout() {
        try {
            await this.msalInstance.logoutPopup();
            this.account = null;
            this.userRole = null;
        } catch (error) {
            console.error('Logout failed:', error);
            throw error;
        }
    }

    async getAccessToken() {
        if (!this.account) {
            throw new Error('No active account');
        }

        try {
            const tokenResponse = await this.msalInstance.acquireTokenSilent({
                ...tokenRequest,
                account: this.account
            });
            return tokenResponse.accessToken;
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError') {
                const tokenResponse = await this.msalInstance.acquireTokenPopup(tokenRequest);
                return tokenResponse.accessToken;
            }
            throw error;
        }
    }

    async setUserRole() {
        if (!this.account) return;

        try {
            // Get additional user info from Microsoft Graph
            const accessToken = await this.getAccessToken();
            const userInfo = await this.getUserInfo(accessToken);
            this.userRole = getUserRole(userInfo);
        } catch (error) {
            console.error('Failed to set user role:', error);
            this.userRole = 'user'; // Default role
        }
    }

    async getUserInfo(accessToken) {
        const response = await fetch('https://graph.microsoft.com/v1.0/me?$select=id,displayName,mail,userPrincipalName,jobTitle,department', {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch user info');
        }
        
        return response.json();
    }

    isAuthenticated() {
        return this.account !== null;
    }

    getUserInfo() {
        return this.account;
    }

    getUserRole() {
        return this.userRole;
    }

    hasPermission(permission) {
        const rolePermissions = ROLE_PERMISSIONS[this.userRole] || [];
        return rolePermissions.includes(permission);
    }
}

export const authService = new AuthService();