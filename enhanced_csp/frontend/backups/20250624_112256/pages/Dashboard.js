/**
 * Enhanced CSP System - Main Dashboard
 * Azure AD authenticated dashboard
 */

import React from 'react';
import { Container, Row, Col, Card, Badge } from 'react-bootstrap';
import { Activity, Users, Shield, Cpu } from 'lucide-react';

import { useUserProfile } from '../hooks/useUserProfile';
import RoleGuard from '../components/RoleGuard';

const Dashboard = () => {
    const { userProfile, loading } = useUserProfile();

    if (loading) {
        return (
            <Container className="mt-4">
                <div className="text-center">Loading dashboard...</div>
            </Container>
        );
    }

    return (
        <Container className="mt-4">
            <Row>
                <Col>
                    <h2>Welcome to Enhanced CSP System</h2>
                    <p className="text-muted">
                        Hello, {userProfile?.displayName || 'User'}! 
                        Your role: <Badge bg="primary">{userProfile?.roles?.[0] || 'User'}</Badge>
                    </p>
                </Col>
            </Row>

            <Row className="mt-4">
                <Col md={3}>
                    <Card className="text-center">
                        <Card.Body>
                            <Activity size={32} className="text-primary mb-2" />
                            <h5>Active Designs</h5>
                            <h3>12</h3>
                        </Card.Body>
                    </Card>
                </Col>

                <RoleGuard allowedRoles={['Administrator', 'Security Officer']}>
                    <Col md={3}>
                        <Card className="text-center">
                            <Card.Body>
                                <Users size={32} className="text-success mb-2" />
                                <h5>Total Users</h5>
                                <h3>48</h3>
                            </Card.Body>
                        </Card>
                    </Col>
                </RoleGuard>

                <Col md={3}>
                    <Card className="text-center">
                        <Card.Body>
                            <Cpu size={32} className="text-info mb-2" />
                            <h5>Components</h5>
                            <h3>156</h3>
                        </Card.Body>
                    </Card>
                </Col>

                <RoleGuard allowedRoles={['Administrator']}>
                    <Col md={3}>
                        <Card className="text-center">
                            <Card.Body>
                                <Shield size={32} className="text-warning mb-2" />
                                <h5>System Health</h5>
                                <h3>98%</h3>
                            </Card.Body>
                        </Card>
                    </Col>
                </RoleGuard>
            </Row>

            {/* Additional dashboard content would go here */}
        </Container>
    );
};

export default Dashboard;