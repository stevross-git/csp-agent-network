/**
 * Enhanced CSP System - User Profile Component
 * Comprehensive user profile display with Azure AD integration
 */

import React, { useState, useEffect } from 'react';
import { 
    Container, 
    Row, 
    Col, 
    Card, 
    Badge, 
    Button,
    Alert,
    Spinner,
    ListGroup,
    Tab,
    Tabs
} from 'react-bootstrap';
import { 
    User, 
    Mail, 
    Building, 
    Shield, 
    Clock, 
    Activity,
    Settings,
    RefreshCw
} from 'lucide-react';

import { useUserProfile } from '../hooks/useUserProfile';
import { useMsal } from '@azure/msal-react';
import { callMsGraph } from '../services/graphService';

const UserProfile = () => {
    const { instance, accounts } = useMsal();
    const { userProfile, loading, error, refreshProfile } = useUserProfile();
    const [activeTab, setActiveTab] = useState('overview');
    const [graphData, setGraphData] = useState(null);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        if (userProfile) {
            fetchGraphData();
        }
    }, [userProfile]);

    const fetchGraphData = async () => {
        try {
            const request = {
                scopes: ["User.Read"],
                account: accounts[0]
            };

            const response = await instance.acquireTokenSilent(request);
            const data = await callMsGraph(response.accessToken);
            setGraphData(data);
        } catch (error) {
            console.error('Failed to fetch Graph data:', error);
        }
    };

    const handleRefresh = async () => {
        setRefreshing(true);
        await refreshProfile();
        await fetchGraphData();
        setRefreshing(false);
    };

    if (loading) {
        return (
            <Container className="mt-4">
                <div className="text-center">
                    <Spinner animation="border" variant="primary" className="mb-3" />
                    <div>Loading user profile...</div>
                </div>
            </Container>
        );
    }

    if (error) {
        return (
            <Container className="mt-4">
                <Alert variant="danger">
                    <strong>Error:</strong> {error}
                </Alert>
            </Container>
        );
    }

    return (
        <Container className="mt-4">
            <Row>
                <Col md={4}>
                    {/* Profile Card */}
                    <Card className="mb-4">
                        <Card.Body className="text-center">
                            <div className="profile-avatar mb-3">
                                <User size={64} className="text-primary" />
                            </div>
                            
                            <h4>{userProfile?.displayName || 'User'}</h4>
                            <p className="text-muted">{userProfile?.jobTitle || 'CSP System User'}</p>
                            
                            <div className="mb-3">
                                {userProfile?.roles?.map(role => (
                                    <Badge key={role} bg="primary" className="me-1 mb-1">
                                        <Shield size={12} className="me-1" />
                                        {role}
                                    </Badge>
                                ))}
                            </div>

                            <Button 
                                variant="outline-primary" 
                                onClick={handleRefresh}
                                disabled={refreshing}
                                className="w-100"
                            >
                                {refreshing ? (
                                    <>
                                        <Spinner size="sm" className="me-2" />
                                        Refreshing...
                                    </>
                                ) : (
                                    <>
                                        <RefreshCw size={16} className="me-2" />
                                        Refresh Profile
                                    </>
                                )}
                            </Button>
                        </Card.Body>
                    </Card>

                    {/* Quick Stats */}
                    <Card>
                        <Card.Header>
                            <Activity size={16} className="me-2" />
                            Quick Stats
                        </Card.Header>
                        <Card.Body>
                            <div className="d-flex justify-content-between mb-2">
                                <span>Last Login:</span>
                                <small className="text-muted">
                                    {userProfile?.lastLogin ? 
                                        new Date(userProfile.lastLogin).toLocaleString() : 
                                        'N/A'
                                    }
                                </small>
                            </div>
                            <div className="d-flex justify-content-between mb-2">
                                <span>Total Sessions:</span>
                                <Badge bg="info">{userProfile?.sessionCount || 0}</Badge>
                            </div>
                            <div className="d-flex justify-content-between">
                                <span>Account Status:</span>
                                <Badge bg={userProfile?.isActive ? 'success' : 'warning'}>
                                    {userProfile?.isActive ? 'Active' : 'Inactive'}
                                </Badge>
                            </div>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={8}>
                    <Tabs
                        activeKey={activeTab}
                        onSelect={(k) => setActiveTab(k)}
                        className="mb-3"
                    >
                        {/* Overview Tab */}
                        <Tab eventKey="overview" title="Overview">
                            <Card>
                                <Card.Header>
                                    <User size={16} className="me-2" />
                                    Profile Information
                                </Card.Header>
                                <Card.Body>
                                    <Row>
                                        <Col md={6}>
                                            <ListGroup variant="flush">
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>
                                                        <Mail size={16} className="me-2" />
                                                        Email:
                                                    </strong>
                                                    <span>{userProfile?.email || 'N/A'}</span>
                                                </ListGroup.Item>
                                                
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>
                                                        <User size={16} className="me-2" />
                                                        Username:
                                                    </strong>
                                                    <span>{userProfile?.username || 'N/A'}</span>
                                                </ListGroup.Item>
                                                
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>
                                                        <Building size={16} className="me-2" />
                                                        Department:
                                                    </strong>
                                                    <span>{graphData?.department || 'N/A'}</span>
                                                </ListGroup.Item>
                                            </ListGroup>
                                        </Col>
                                        
                                        <Col md={6}>
                                            <ListGroup variant="flush">
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>
                                                        <Clock size={16} className="me-2" />
                                                        Created:
                                                    </strong>
                                                    <span>
                                                        {userProfile?.createdAt ? 
                                                            new Date(userProfile.createdAt).toLocaleDateString() : 
                                                            'N/A'
                                                        }
                                                    </span>
                                                </ListGroup.Item>
                                                
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>
                                                        <Shield size={16} className="me-2" />
                                                        Role Count:
                                                    </strong>
                                                    <Badge bg="primary">{userProfile?.roles?.length || 0}</Badge>
                                                </ListGroup.Item>
                                                
                                                <ListGroup.Item className="d-flex justify-content-between">
                                                    <strong>Office Location:</strong>
                                                    <span>{graphData?.officeLocation || 'N/A'}</span>
                                                </ListGroup.Item>
                                            </ListGroup>
                                        </Col>
                                    </Row>
                                </Card.Body>
                            </Card>
                        </Tab>

                        {/* Permissions Tab */}
                        <Tab eventKey="permissions" title="Roles & Permissions">
                            <Card>
                                <Card.Header>
                                    <Shield size={16} className="me-2" />
                                    Access Control
                                </Card.Header>
                                <Card.Body>
                                    <h6>Assigned Roles:</h6>
                                    <div className="mb-3">
                                        {userProfile?.roles?.map(role => (
                                            <Badge key={role} bg="primary" className="me-2 mb-2">
                                                {role}
                                            </Badge>
                                        )) || <span className="text-muted">No roles assigned</span>}
                                    </div>

                                    <h6>Permissions:</h6>
                                    <div className="mb-3">
                                        {userProfile?.permissions?.map(permission => (
                                            <Badge key={permission} bg="success" className="me-2 mb-2">
                                                {permission}
                                            </Badge>
                                        )) || <span className="text-muted">No specific permissions</span>}
                                    </div>
                                </Card.Body>
                            </Card>
                        </Tab>

                        {/* Settings Tab */}
                        <Tab eventKey="settings" title="Settings">
                            <Card>
                                <Card.Header>
                                    <Settings size={16} className="me-2" />
                                    User Preferences
                                </Card.Header>
                                <Card.Body>
                                    <Alert variant="info">
                                        User settings and preferences will be implemented in the next version.
                                        This will include theme preferences, notification settings, and 
                                        workspace configurations.
                                    </Alert>
                                </Card.Body>
                            </Card>
                        </Tab>
                    </Tabs>
                </Col>
            </Row>
        </Container>
    );
};

export default UserProfile;