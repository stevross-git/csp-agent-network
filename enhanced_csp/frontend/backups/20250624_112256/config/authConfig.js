// config/authConfig.js
export const msalConfig = {
    auth: {
        clientId: process.env.AZURE_CLIENT_ID,
        authority: `https://login.microsoftonline.com/${process.env.AZURE_TENANT_ID}`,
        redirectUri: process.env.AZURE_REDIRECT_URI,
        postLogoutRedirectUri: process.env.AZURE_POST_LOGOUT_REDIRECT_URI,
    },
    cache: {
        cacheLocation: "sessionStorage",
        storeAuthStateInCookie: false,
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
                }
            }
        }
    }
};

// Login request configuration
export const loginRequest = {
    scopes: ["User.Read", "User.ReadBasic.All"]
};

// Token request configuration for API calls
export const tokenRequest = {
    scopes: ["User.Read"],
    forceRefresh: false
};