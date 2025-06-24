/**
 * Enhanced CSP System - Microsoft Graph API Service
 * Official Microsoft Graph integration for user data
 */

import { graphConfig } from '../config/authConfig';

/**
 * Call Microsoft Graph API
 */
export async function callMsGraph(accessToken, endpoint = graphConfig.graphMeEndpoint) {
    const headers = new Headers();
    const bearer = `Bearer ${accessToken}`;

    headers.append("Authorization", bearer);

    const options = {
        method: "GET",
        headers: headers
    };

    try {
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            throw new Error(`Graph API request failed: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Graph API call failed:', error);
        throw error;
    }
}

/**
 * Get user profile from Graph API
 */
export async function getUserProfile(accessToken) {
    return await callMsGraph(accessToken, graphConfig.graphMeEndpoint);
}

/**
 * Get user's groups from Graph API
 */
export async function getUserGroups(accessToken) {
    const endpoint = `${graphConfig.graphMeEndpoint}/memberOf`;
    return await callMsGraph(accessToken, endpoint);
}

/**
 * Get user's manager from Graph API
 */
export async function getUserManager(accessToken) {
    const endpoint = `${graphConfig.graphMeEndpoint}/manager`;
    return await callMsGraph(accessToken, endpoint);
}

/**
 * Get user's direct reports from Graph API
 */
export async function getUserDirectReports(accessToken) {
    const endpoint = `${graphConfig.graphMeEndpoint}/directReports`;
    return await callMsGraph(accessToken, endpoint);
}

/**
 * Get organization details
 */
export async function getOrganization(accessToken) {
    const endpoint = "https://graph.microsoft.com/v1.0/organization";
    return await callMsGraph(accessToken, endpoint);
}