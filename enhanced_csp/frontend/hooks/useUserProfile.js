/**
 * Enhanced CSP System - User Profile React Hook
 * Custom hook for managing user profile data
 */

import { useState, useEffect, useCallback } from 'react';
import { useMsal, useIsAuthenticated } from '@azure/msal-react';
import { callMsGraph, getUserGroups } from '../services/graphService';
import { cspApiService } from '../services/cspApiService';
import { roleMappings } from '../config/authConfig';

export const useUserProfile = () => {
    const { instance, accounts } = useMsal();
    const isAuthenticated = useIsAuthenticated();
    const [userProfile, setUserProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchUserProfile = useCallback(async () => {
        if (!isAuthenticated || !accounts.length) {
            setUserProfile(null);
            setLoading(false);
            return;
        }

        try {
            setLoading(true);
            setError(null);

            // Get access token
            const request = {
                scopes: ["User.Read", "Directory.Read.All"],
                account: accounts[0]
            };

            const response = await instance.acquireTokenSilent(request);

            // Fetch user data from Graph API
            const [graphProfile, userGroups] = await Promise.all([
                callMsGraph(response.accessToken),
                getUserGroups(response.accessToken).catch(() => ({ value: [] }))
            ]);

            // Map Azure groups to CSP roles
            const userRoles = [];
            const groupNames = userGroups.value?.map(group => group.displayName) || [];
            
            Object.entries(roleMappings).forEach(([groupName, role]) => {
                if (groupNames.includes(groupName)) {
                    userRoles.push(role);
                }
            });

            // Default role if no groups matched
            if (userRoles.length === 0) {
                userRoles.push('User');
            }

            // Fetch additional profile data from CSP backend
            let cspProfile = {};
            try {
                cspProfile = await cspApiService.getCurrentUser();
            } catch (cspError) {
                console.warn('Failed to fetch CSP profile data:', cspError);
            }

            // Combine profile data
            const combinedProfile = {
                id: graphProfile.id,
                username: graphProfile.userPrincipalName,
                email: graphProfile.mail || graphProfile.userPrincipalName,
                displayName: graphProfile.displayName,
                firstName: graphProfile.givenName,
                lastName: graphProfile.surname,
                jobTitle: graphProfile.jobTitle,
                department: graphProfile.department,
                officeLocation: graphProfile.officeLocation,
                mobilePhone: graphProfile.mobilePhone,
                businessPhones: graphProfile.businessPhones,
                roles: userRoles,
                groups: groupNames,
                isActive: true,
                lastLogin: new Date().toISOString(),
                createdAt: cspProfile.createdAt || new Date().toISOString(),
                permissions: derivePermissionsFromRoles(userRoles),
                sessionCount: cspProfile.sessionCount || 1,
                ...cspProfile // Merge any additional CSP-specific data
            };

            setUserProfile(combinedProfile);
        } catch (err) {
            console.error('Failed to fetch user profile:', err);
            setError(err.message || 'Failed to load user profile');
        } finally {
            setLoading(false);
        }
    }, [instance, accounts, isAuthenticated]);

    const refreshProfile = useCallback(() => {
        return fetchUserProfile();
    }, [fetchUserProfile]);

    useEffect(() => {
        fetchUserProfile();
    }, [fetchUserProfile]);

    return {
        userProfile,
        loading,
        error,
        refreshProfile
    };
};

/**
 * Derive permissions from user roles
 */
function derivePermissionsFromRoles(roles) {
    const rolePermissions = {
        'Administrator': [
            'view_all_designs',
            'create_design',
            'edit_design',
            'delete_design',
            'execute_design',
            'manage_users',
            'manage_system',
            'view_metrics',
            'manage_components'
        ],
        'Security Officer': [
            'view_all_designs',
            'create_design',
            'edit_design',
            'execute_design',
            'view_metrics',
            'manage_security'
        ],
        'Developer': [
            'view_own_designs',
            'create_design',
            'edit_own_design',
            'execute_design',
            'manage_components'
        ],
        'Analyst': [
            'view_shared_designs',
            'create_design',
            'execute_design',
            'view_metrics'
        ],
        'User': [
            'view_shared_designs',
            'execute_design'
        ]
    };

    const permissions = new Set();
    roles.forEach(role => {
        const rolePerms = rolePermissions[role] || [];
        rolePerms.forEach(perm => permissions.add(perm));
    });

    return Array.from(permissions);
}

export default useUserProfile;