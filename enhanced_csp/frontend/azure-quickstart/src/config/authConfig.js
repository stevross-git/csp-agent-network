```javascript
/**
 * Enhanced CSP System - Azure AD Configuration
 * Microsoft Official Implementation
 */

import { LogLevel } from "@azure/msal-browser";

// Azure AD B2C configuration
export const msalConfig = {
    auth: {
        clientId: process.env.REACT_APP_AZURE_CLIENT_ID || "your-client-id-here",
        authority: process.env.REACT_APP_AZURE_AUTHORITY || 
                  "https://login.microsoftonline.com/your-tenant-id",
        redirectUri: process.env.REACT_APP_REDIRECT_URI || 
                    window.location.origin + "/auth/callback",
        postLogoutRedirectUri: process.env.REACT_APP_POST_LOGOUT_REDIRECT_URI || 
                             window.location.origin,
        navigateToLoginRequestUrl: false,
    },
    cache: {
        cacheLocation: "localStorage", // or "sessionStorage"
        storeAuthStateInCookie: false, // Set to true for IE11 or Edge
    },
    system: {
        loggerOptions: {
            loggerCallback: (level, message, containsPii) => {
                if (containsPii) {
                    return;
                }
                switch (level) {
                    case LogLevel.Error:
                        console.error(message);
                        return;
                    case LogLevel.Info:
                        console.info(message);
                        return;
                    case LogLevel.Verbose:
                        console.debug(message);
                        return;
                    case LogLevel.Warning:
                        console.warn(message);
                        return;
                    default:
                        return;
                }
            }
        }
    }
};

// Graph API endpoints
export const graphConfig = {
    graphMeEndpoint: "https://graph.microsoft.com/v1.0/me",
    graphUsersEndpoint: "https://graph.microsoft.com/v1.0/users",
    graphGroupsEndpoint: "https://graph.microsoft.com/v1.0/groups",
};

// Scopes for API calls
export const loginRequest = {
    scopes: [
        "User.Read",
        "User.ReadBasic.All",
        "Directory.Read.All",
        "Group.Read.All"
    ]
};

export const graphRequest = {
    ...loginRequest,
    forceRefresh: false
};

// CSP Backend API configuration
export const cspApiConfig = {
    baseUrl: process.env.REACT_APP_CSP_API_URL || "http://localhost:8000",
    endpoints: {
        auth: "/api/auth",
        designs: "/api/designs",
        components: "/api/components",
        executions: "/api/executions",
        websocket: "/ws"
    }
};

// Role mappings between Azure AD groups and CSP roles
export const roleMappings = {
    "CSP-Administrators": "Administrator",
    "CSP-Security-Officers": "Security Officer",
    "CSP-Developers": "Developer",
    "CSP-Analysts": "Analyst",
    "CSP-Users": "User"
};

// Environment validation
export const validateConfig = () => {
    const requiredEnvVars = [
        'REACT_APP_AZURE_CLIENT_ID',
        'REACT_APP_AZURE_AUTHORITY',
        'REACT_APP_CSP_API_URL'
    ];

    const missing = requiredEnvVars.filter(envVar => !process.env[envVar]);
    
    if (missing.length > 0) {
        console.warn("Missing environment variables: " + missing.join(', '));
        console.warn('Using fallback configuration values');
    }

    return {
        isValid: missing.length === 0,
        missing,
        config: {
            clientId: process.env.REACT_APP_AZURE_CLIENT_ID,
            authority: process.env.REACT_APP_AZURE_AUTHORITY,
            apiUrl: process.env.REACT_APP_CSP_API_URL
        }
    };
};
```