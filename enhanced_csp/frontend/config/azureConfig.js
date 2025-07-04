// Azure AD Configuration for Enhanced CSP System
// Your actual Azure AD app registration details

const azureConfig = {
    development: {
        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
        tenantId: "622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
        redirectUri: "http://localhost:3000",
        postLogoutRedirectUri: "http://localhost:3000",
        environment: "development"
    },
    production: {
        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
        tenantId: "622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
        redirectUri: "https://your-production-domain.com",
        postLogoutRedirectUri: "https://your-production-domain.com",
        environment: "production"
    }
};

// Get current environment configuration
function getAzureConfig() {
    const env = window.location.hostname === 'localhost' ? 'development' : 'production';
    return azureConfig[env];
}

// MSAL Configuration
export const msalConfig = {
    auth: {
        clientId: getAzureConfig().clientId,
        authority: `https://login.microsoftonline.com/${getAzureConfig().tenantId}`,
        redirectUri: getAzureConfig().redirectUri,
        postLogoutRedirectUri: getAzureConfig().postLogoutRedirectUri,
        navigateToLoginRequestUrl: false
    },
    cache: {
        cacheLocation: "sessionStorage",
        storeAuthStateInCookie: false,
    },
    system: {
        loggerOptions: {
            loggerCallback: (level, message, containsPii) => {
                if (containsPii) return;
                switch (level) {
                    case 0: console.error('[MSAL]', message); break;
                    case 1: console.warn('[MSAL]', message); break;
                    case 2: console.info('[MSAL]', message); break;
                    case 3: console.debug('[MSAL]', message); break;
                }
            },
            logLevel: 3
        }
    }
};

// Login request configuration
export const loginRequest = {
    scopes: ["User.Read", "User.ReadBasic.All", "Group.Read.All"],
    prompt: "select_account"
};

// Token request configuration for API calls
export const tokenRequest = {
    scopes: ["User.Read"],
    forceRefresh: false
};

// User roles configuration
export const USER_ROLES = {
    SUPER_ADMIN: 'super_admin',
    ADMIN: 'admin',
    DEVELOPER: 'developer',
    ANALYST: 'analyst',
    USER: 'user'
};

// Make available globally for backward compatibility
window.msalConfig = msalConfig;
window.loginRequest = loginRequest;
window.USER_ROLES = USER_ROLES;

export default {
    msalConfig,
    loginRequest,
    tokenRequest,
    USER_ROLES,
    getAzureConfig
};