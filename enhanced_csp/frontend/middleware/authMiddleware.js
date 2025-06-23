/**
 * Enhanced CSP System - Authentication Middleware
 * Client-side authentication checks and token management
 */

import { msalInstance } from '../index';
import { loginRequest } from '../config/authConfig';

class AuthMiddleware {
    constructor() {
        this.tokenRefreshThreshold = 5 * 60 * 1000; // 5 minutes
        this.maxRetries = 3;
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated() {
        const accounts = msalInstance.getAllAccounts();
        return accounts.length > 0;
    }

    /**
     * Get current active account
     */
    getCurrentAccount() {
        return msalInstance.getActiveAccount() || msalInstance.getAllAccounts()[0];
    }

    /**
     * Get valid access token with automatic refresh
     */
    async getValidToken(scopes = ["User.Read"], retryCount = 0) {
        const account = this.getCurrentAccount();
        
        if (!account) {
            throw new Error('No authenticated account found');
        }

        try {
            const request = {
                scopes,
                account,
                forceRefresh: false
            };

            const response = await msalInstance.acquireTokenSilent(request);
            
            // Check if token expires soon
            const expiresIn = response.expiresOn.getTime() - Date.now();
            if (expiresIn < this.tokenRefreshThreshold) {
                console.log('Token expires soon, forcing refresh...');
                request.forceRefresh = true;
                return await msalInstance.acquireTokenSilent(request);
            }

            return response;
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError' && retryCount < this.maxRetries) {
                console.log(`Token acquisition failed, retrying... (${retryCount + 1}/${this.maxRetries})`);
                
                try {
                    // Try with interaction
                    const request = {
                        scopes,
                        account
                    };
                    
                    return await msalInstance.acquireTokenPopup(request);
                } catch (interactionError) {
                    if (retryCount < this.maxRetries - 1) {
                        return this.getValidToken(scopes, retryCount + 1);
                    }
                    throw interactionError;
                }
            }
            
            throw error;
        }
    }

    /**
     * Secure API request wrapper
     */
    async secureApiCall(url, options = {}, requiredScopes = ["User.Read"]) {
        try {
            const tokenResponse = await this.getValidToken(requiredScopes);
            
            const headers = {
                'Authorization': `Bearer ${tokenResponse.accessToken}`,
                'Content-Type': 'application/json',
                ...options.headers
            };

            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 401) {
                // Token might be invalid, try to refresh
                console.log('Received 401, attempting token refresh...');
                const refreshedToken = await this.getValidToken(requiredScopes);
                
                const retryHeaders = {
                    ...headers,
                    'Authorization': `Bearer ${refreshedToken.accessToken}`
                };

                return fetch(url, {
                    ...options,
                    headers: retryHeaders
                });
            }

            return response;
        } catch (error) {
            console.error('Secure API call failed:', error);
            throw error;
        }
    }

    /**
     * Logout current user
     */
    async logout() {
        const account = this.getCurrentAccount();
        
        if (account) {
            await msalInstance.logoutRedirect({
                account,
                postLogoutRedirectUri: window.location.origin
            });
        }
    }

    /**
     * Check token expiration
     */
    isTokenExpired(token) {
        if (!token || !token.expiresOn) {
            return true;
        }
        
        return Date.now() >= token.expiresOn.getTime();
    }

    /**
     * Get user claims from token
     */
    getUserClaims() {
        const account = this.getCurrentAccount();
        return account ? account.idTokenClaims : null;
    }

    /**
     * Validate session and redirect if necessary
     */
    validateSession(requiredPath = '/login') {
        if (!this.isAuthenticated()) {
            console.log('User not authenticated, redirecting to login...');
            window.location.href = requiredPath;
            return false;
        }
        return true;
    }
}

export const authMiddleware = new AuthMiddleware();

export default authMiddleware;