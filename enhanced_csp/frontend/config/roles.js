// config/roles.js
export const USER_ROLES = {
    SUPER_ADMIN: 'super_admin',
    ADMIN: 'admin',
    DEVELOPER: 'developer',
    ANALYST: 'analyst',
    USER: 'user'
};

export const ROLE_PERMISSIONS = {
    [USER_ROLES.SUPER_ADMIN]: [
        'system.admin',
        'user.manage',
        'ai.manage',
        'quantum.manage',
        'blockchain.manage',
        'security.admin'
    ],
    [USER_ROLES.ADMIN]: [
        'system.view',
        'user.manage',
        'ai.manage',
        'quantum.view',
        'blockchain.view'
    ],
    [USER_ROLES.DEVELOPER]: [
        'system.view',
        'ai.manage',
        'quantum.view',
        'blockchain.view'
    ],
    [USER_ROLES.ANALYST]: [
        'system.view',
        'ai.view',
        'quantum.view',
        'reports.view'
    ],
    [USER_ROLES.USER]: [
        'system.view'
    ]
};

// Role assignment based on Azure AD groups or user attributes
export function getUserRole(userInfo) {
    const email = userInfo.mail || userInfo.upn;
    const groups = userInfo.groups || [];
    
    // Check Azure AD groups (configure these in Azure AD)
    if (groups.includes('CSP-SuperAdmins')) return USER_ROLES.SUPER_ADMIN;
    if (groups.includes('CSP-Admins')) return USER_ROLES.ADMIN;
    if (groups.includes('CSP-Developers')) return USER_ROLES.DEVELOPER;
    if (groups.includes('CSP-Analysts')) return USER_ROLES.ANALYST;
    
    // Fallback to email domain checking
    if (email?.endsWith('@admin.company.com')) return USER_ROLES.ADMIN;
    if (email?.endsWith('@dev.company.com')) return USER_ROLES.DEVELOPER;
    
    // Default role
    return USER_ROLES.USER;
}