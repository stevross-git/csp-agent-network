// admin-portal.test.js
// Jest + jsdom test suite for Enhanced CSP Admin Portal

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

// Load the HTML file
const html = fs.readFileSync(path.resolve(__dirname, 'admin.html'), 'utf8');

describe('Enhanced CSP Admin Portal', () => {
    let dom;
    let document;
    let window;

    beforeEach(() => {
        // Create fresh DOM for each test
        dom = new JSDOM(html, {
            runScripts: 'dangerously',
            resources: 'usable'
        });
        document = dom.window.document;
        window = dom.window;

        // Mock console methods to capture logs
        window.console.log = jest.fn();
        window.console.error = jest.fn();
        window.console.warn = jest.fn();

        // Wait for scripts to execute
        return new Promise(resolve => {
            window.addEventListener('DOMContentLoaded', resolve);
            if (document.readyState === 'complete') {
                resolve();
            }
        });
    });

    afterEach(() => {
        dom.window.close();
    });

    describe('Initial State', () => {
        test('should have dashboard section active on load', () => {
            const dashboardSection = document.getElementById('dashboard');
            const dashboardNav = document.querySelector('[data-section="dashboard"]');

            expect(dashboardSection).toBeTruthy();
            expect(dashboardSection.classList.contains('active')).toBe(true);
            expect(dashboardNav.classList.contains('active')).toBe(true);
            expect(dashboardNav.getAttribute('aria-current')).toBe('page');
        });

        test('should have only one active section at startup', () => {
            const activeSections = document.querySelectorAll('.content-section.active');
            expect(activeSections.length).toBe(1);
            expect(activeSections[0].id).toBe('dashboard');
        });

        test('should have only one active nav item at startup', () => {
            const activeNavItems = document.querySelectorAll('.nav-item.active');
            expect(activeNavItems.length).toBe(1);
            expect(activeNavItems[0].getAttribute('data-section')).toBe('dashboard');
        });
    });

    describe('Navigation Functionality', () => {
        const sectionIds = [
            'dashboard', 'monitoring', 'alerts', 'users', 'roles', 'auth',
            'ai-models', 'agents', 'protocols', 'settings', 'infrastructure',
            'integrations', 'backups', 'logs', 'maintenance', 'licenses',
            'billing', 'audit'
        ];

        test.each(sectionIds)('should switch to %s section correctly', (sectionId) => {
            // Trigger section switch
            window.cspAdmin.showSection(sectionId);

            // Wait for debounce to complete
            return new Promise(resolve => {
                setTimeout(() => {
                    // Check that target section is active
                    const targetSection = document.getElementById(sectionId);
                    expect(targetSection.classList.contains('active')).toBe(true);

                    // Check that only one section is active
                    const activeSections = document.querySelectorAll('.content-section.active');
                    expect(activeSections.length).toBe(1);

                    // Check that corresponding nav item is active
                    const activeNavItem = document.querySelector(`[data-section="${sectionId}"]`);
                    expect(activeNavItem.classList.contains('active')).toBe(true);
                    expect(activeNavItem.getAttribute('aria-current')).toBe('page');

                    // Check that only one nav item is active
                    const activeNavItems = document.querySelectorAll('.nav-item.active');
                    expect(activeNavItems.length).toBe(1);

                    resolve();
                }, 100);
            });
        });

        test('should handle non-existent section gracefully', () => {
            const consoleSpy = jest.spyOn(window.console, 'error');
            
            window.cspAdmin.showSection('non-existent-section');

            return new Promise(resolve => {
                setTimeout(() => {
                    expect(consoleSpy).toHaveBeenCalledWith(
                        expect.stringContaining("Section 'non-existent-section' not found!")
                    );

                    // Should maintain current state (dashboard)
                    const activeSections = document.querySelectorAll('.content-section.active');
                    expect(activeSections.length).toBe(1);
                    expect(activeSections[0].id).toBe('dashboard');

                    resolve();
                }, 100);
            });
        });
    });

    describe('Accessibility Features', () => {
        test('all nav items should have proper accessibility attributes', () => {
            const navItems = document.querySelectorAll('.nav-item[data-section]');
            
            navItems.forEach(item => {
                expect(item.getAttribute('role')).toBe('button');
                expect(item.getAttribute('tabindex')).toBe('0');
                expect(item.hasAttribute('data-section')).toBe(true);
            });
        });

        test('should handle keyboard navigation (Enter key)', () => {
            const usersNavItem = document.querySelector('[data-section="users"]');
            
            // Simulate Enter key press
            const event = new window.KeyboardEvent('keydown', {
                key: 'Enter',
                bubbles: true
            });
            
            usersNavItem.dispatchEvent(event);

            return new Promise(resolve => {
                setTimeout(() => {
                    const usersSection = document.getElementById('users');
                    expect(usersSection.classList.contains('active')).toBe(true);
                    expect(usersNavItem.classList.contains('active')).toBe(true);
                    resolve();
                }, 100);
            });
        });

        test('should handle keyboard navigation (Space key)', () => {
            const monitoringNavItem = document.querySelector('[data-section="monitoring"]');
            
            // Simulate Space key press
            const event = new window.KeyboardEvent('keydown', {
                key: ' ',
                bubbles: true
            });
            
            monitoringNavItem.dispatchEvent(event);

            return new Promise(resolve => {
                setTimeout(() => {
                    const monitoringSection = document.getElementById('monitoring');
                    expect(monitoringSection.classList.contains('active')).toBe(true);
                    expect(monitoringNavItem.classList.contains('active')).toBe(true);
                    resolve();
                }, 100);
            });
        });
    });

    describe('Mobile Functionality', () => {
        test('should toggle mobile sidebar', () => {
            const sidebar = document.getElementById('sidebar');
            
            // Initially should not have mobile-hidden class
            expect(sidebar.classList.contains('mobile-hidden')).toBe(false);
            
            // Toggle sidebar
            window.cspAdmin.toggleMobileSidebar();
            expect(sidebar.classList.contains('mobile-hidden')).toBe(true);
            
            // Toggle back
            window.cspAdmin.toggleMobileSidebar();
            expect(sidebar.classList.contains('mobile-hidden')).toBe(false);
        });
    });

    describe('Error Handling', () => {
        test('should not throw errors with rapid clicking', () => {
            const showSectionSpy = jest.spyOn(window.cspAdmin, 'showSection');
            
            // Rapidly call showSection multiple times
            for (let i = 0; i < 10; i++) {
                window.cspAdmin.showSection('users');
                window.cspAdmin.showSection('monitoring');
                window.cspAdmin.showSection('alerts');
            }

            expect(showSectionSpy).toHaveBeenCalledTimes(30);
            // Should not throw any errors
            expect(window.console.error).not.toHaveBeenCalledWith(
                expect.stringContaining('Error switching sections:')
            );
        });

        test('should maintain state consistency after errors', () => {
            // Trigger an error
            window.cspAdmin.showSection('invalid-section');
            
            return new Promise(resolve => {
                setTimeout(() => {
                    // Should still have exactly one active section
                    const activeSections = document.querySelectorAll('.content-section.active');
                    expect(activeSections.length).toBe(1);
                    
                    // Should still have exactly one active nav item
                    const activeNavItems = document.querySelectorAll('.nav-item.active');
                    expect(activeNavItems.length).toBe(1);
                    
                    resolve();
                }, 100);
            });
        });
    });

    describe('Performance & Debouncing', () => {
        test('should debounce rapid section switches', () => {
            const performSectionSwitchSpy = jest.spyOn(window, 'performSectionSwitch');
            
            // Rapidly trigger multiple switches
            window.cspAdmin.showSection('users');
            window.cspAdmin.showSection('monitoring');
            window.cspAdmin.showSection('alerts');

            return new Promise(resolve => {
                setTimeout(() => {
                    // Should only execute the last call due to debouncing
                    expect(performSectionSwitchSpy).toHaveBeenCalledTimes(1);
                    expect(performSectionSwitchSpy).toHaveBeenLastCalledWith('alerts');
                    resolve();
                }, 150);
            });
        });
    });

    describe('Integration Tests', () => {
        test('complete user journey: navigate through multiple sections', async () => {
            const journey = ['users', 'monitoring', 'settings', 'dashboard'];
            
            for (const sectionId of journey) {
                window.cspAdmin.showSection(sectionId);
                
                // Wait for transition
                await new Promise(resolve => setTimeout(resolve, 100));
                
                // Verify state
                const activeSection = document.querySelector('.content-section.active');
                const activeNavItem = document.querySelector('.nav-item.active');
                
                expect(activeSection.id).toBe(sectionId);
                expect(activeNavItem.getAttribute('data-section')).toBe(sectionId);
                
                // Verify only one active at a time
                expect(document.querySelectorAll('.content-section.active')).toHaveLength(1);
                expect(document.querySelectorAll('.nav-item.active')).toHaveLength(1);
            }
        });
    });
});

// Test runner script
if (require.main === module) {
    console.log('🧪 Running Enhanced CSP Admin Portal Tests...');
    console.log('📋 Install dependencies: npm install --save-dev jest jsdom');
    console.log('🚀 Run tests: npm test');
}