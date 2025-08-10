import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import HelpProvider from '../../../js/utils/HelpProvider.js';

describe('HelpProvider', () => {
    let helpProvider;
    let mockElement;
    let mockFetch;

    beforeEach(() => {
        // Setup DOM
        document.body.innerHTML = `
            <div id="test-container">
                <button id="test-button" data-help-tooltip="Test tooltip">Test</button>
                <div class="dashboard-header">Dashboard</div>
                <div class="agent-list">Agents</div>
            </div>
        `;

        // Mock fetch
        mockFetch = jest.fn();
        global.fetch = mockFetch;
        mockFetch.mockResolvedValue({
            ok: true,
            json: async () => ({
                helpConfiguration: {
                    views: {
                        dashboard: {
                            tooltips: {
                                testTooltip: "Test tooltip content"
                            },
                            contextualHelp: {
                                overview: {
                                    title: "Dashboard Overview",
                                    content: "Dashboard help content"
                                }
                            },
                            guidedTour: {
                                enabled: true,
                                steps: [
                                    {
                                        target: ".dashboard-header",
                                        title: "Welcome",
                                        content: "Welcome to the dashboard"
                                    }
                                ]
                            }
                        }
                    }
                }
            })
        });

        // Create instance
        helpProvider = new HelpProvider({
            configUrl: '/config/helpConfig.json'
        });

        mockElement = document.getElementById('test-button');
    });

    afterEach(() => {
        // Cleanup
        document.body.innerHTML = '';
        jest.clearAllMocks();
    });

    describe('Initialization', () => {
        it('should initialize with default options', () => {
            const provider = new HelpProvider();
            expect(provider.options.enableTooltips).toBe(true);
            expect(provider.options.enableContextualHelp).toBe(true);
            expect(provider.options.enableTours).toBe(true);
            expect(provider.options.tooltipDelay).toBe(200);
        });

        it('should merge custom options', () => {
            const provider = new HelpProvider({
                enableTooltips: false,
                tooltipDelay: 500
            });
            expect(provider.options.enableTooltips).toBe(false);
            expect(provider.options.tooltipDelay).toBe(500);
        });

        it('should load configuration on init', async () => {
            await helpProvider.init();
            expect(mockFetch).toHaveBeenCalledWith('/config/helpConfig.json');
            expect(helpProvider.helpConfig).toBeDefined();
        });

        it('should handle configuration load error', async () => {
            mockFetch.mockRejectedValue(new Error('Network error'));
            const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
            
            await helpProvider.init();
            
            expect(consoleSpy).toHaveBeenCalledWith(
                'Failed to load help configuration:',
                expect.any(Error)
            );
            consoleSpy.mockRestore();
        });
    });

    describe('Tooltips', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should create tooltip for element', () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Custom tooltip',
                position: 'bottom'
            });

            expect(tooltip).toBeDefined();
            expect(tooltip.element).toBe(mockElement);
            expect(tooltip.content).toBe('Custom tooltip');
            expect(tooltip.position).toBe('bottom');
        });

        it('should show tooltip on hover', async () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Hover tooltip',
                delay: 0
            });

            // Trigger mouseenter
            mockElement.dispatchEvent(new MouseEvent('mouseenter'));
            
            // Wait for delay
            await new Promise(resolve => setTimeout(resolve, 10));

            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement).toBeTruthy();
            expect(tooltipElement.textContent).toBe('Hover tooltip');
            expect(tooltipElement.classList.contains('show')).toBe(true);
        });

        it('should hide tooltip on mouse leave', async () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Leave tooltip',
                delay: 0
            });

            // Show tooltip
            mockElement.dispatchEvent(new MouseEvent('mouseenter'));
            await new Promise(resolve => setTimeout(resolve, 10));

            // Hide tooltip
            mockElement.dispatchEvent(new MouseEvent('mouseleave'));
            await new Promise(resolve => setTimeout(resolve, 10));

            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement.classList.contains('show')).toBe(false);
        });

        it('should position tooltip correctly', () => {
            // Mock getBoundingClientRect
            mockElement.getBoundingClientRect = jest.fn(() => ({
                top: 100,
                left: 200,
                bottom: 120,
                right: 250,
                width: 50,
                height: 20
            }));

            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Positioned tooltip',
                position: 'top'
            });

            tooltip.show();

            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement.style.top).toBeDefined();
            expect(tooltipElement.style.left).toBeDefined();
        });

        it('should handle click trigger', () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Click tooltip',
                trigger: 'click'
            });

            // Click to show
            mockElement.click();
            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement.classList.contains('show')).toBe(true);

            // Click again to hide
            mockElement.click();
            expect(tooltipElement.classList.contains('show')).toBe(false);
        });

        it('should auto-initialize tooltips from data attributes', () => {
            helpProvider.initializeTooltips();
            
            const tooltipIcon = mockElement.querySelector('.help-tooltip-icon');
            expect(tooltipIcon).toBeTruthy();
        });
    });

    describe('Contextual Help', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should show contextual help panel', () => {
            helpProvider.showContextualHelp('dashboard');

            const panel = document.querySelector('.contextual-help-panel');
            expect(panel).toBeTruthy();
            expect(panel.classList.contains('open')).toBe(true);
        });

        it('should display correct help content for view', () => {
            helpProvider.showContextualHelp('dashboard', 'overview');

            const title = document.querySelector('.contextual-help-title');
            const content = document.querySelector('.contextual-help-content');

            expect(title.textContent).toContain('Dashboard');
            expect(content).toBeTruthy();
        });

        it('should close help panel', () => {
            helpProvider.showContextualHelp('dashboard');
            helpProvider.hideContextualHelp();

            const panel = document.querySelector('.contextual-help-panel');
            expect(panel.classList.contains('open')).toBe(false);
        });

        it('should close panel on close button click', () => {
            helpProvider.showContextualHelp('dashboard');
            
            const closeButton = document.querySelector('.contextual-help-close');
            closeButton.click();

            const panel = document.querySelector('.contextual-help-panel');
            expect(panel.classList.contains('open')).toBe(false);
        });

        it('should update help content dynamically', () => {
            helpProvider.showContextualHelp();
            helpProvider.updateHelpContent({
                title: 'Custom Help',
                sections: [
                    {
                        title: 'Section 1',
                        content: 'Custom content'
                    }
                ]
            });

            const title = document.querySelector('.contextual-help-title');
            expect(title.textContent).toBe('Custom Help');
        });

        it('should handle missing help content gracefully', () => {
            helpProvider.showContextualHelp('nonexistent');

            const content = document.querySelector('.contextual-help-content');
            expect(content.textContent).toContain('No help available');
        });
    });

    describe('Guided Tours', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should start guided tour', () => {
            const tour = helpProvider.startGuidedTour('dashboard');

            expect(tour).toBeDefined();
            expect(tour.currentStep).toBe(0);
            expect(tour.isActive).toBe(true);
        });

        it('should highlight tour elements', () => {
            const tour = helpProvider.startGuidedTour('dashboard');

            const highlighted = document.querySelector('.guided-tour-highlight');
            const overlay = document.querySelector('.guided-tour-overlay');

            expect(highlighted).toBeTruthy();
            expect(overlay.classList.contains('active')).toBe(true);
        });

        it('should show tour popup with correct content', () => {
            const tour = helpProvider.startGuidedTour('dashboard');

            const popup = document.querySelector('.guided-tour-popup');
            const title = popup.querySelector('.guided-tour-title');
            const content = popup.querySelector('.guided-tour-content');

            expect(popup).toBeTruthy();
            expect(title.textContent).toBe('Welcome');
            expect(content.textContent).toBe('Welcome to the dashboard');
        });

        it('should navigate through tour steps', () => {
            const tour = helpProvider.startGuidedTour('dashboard');

            expect(tour.currentStep).toBe(0);

            tour.next();
            expect(tour.currentStep).toBe(1);

            tour.previous();
            expect(tour.currentStep).toBe(0);
        });

        it('should complete tour', () => {
            const completeSpy = jest.fn();
            const tour = helpProvider.startGuidedTour('dashboard');
            tour.on('complete', completeSpy);

            // Go to last step
            while (tour.hasNext()) {
                tour.next();
            }
            tour.next(); // Complete

            expect(completeSpy).toHaveBeenCalled();
            expect(tour.isActive).toBe(false);
        });

        it('should skip tour', () => {
            const skipSpy = jest.fn();
            const tour = helpProvider.startGuidedTour('dashboard');
            tour.on('skip', skipSpy);

            tour.skip();

            expect(skipSpy).toHaveBeenCalled();
            expect(tour.isActive).toBe(false);
        });

        it('should restart tour', () => {
            const tour = helpProvider.startGuidedTour('dashboard');
            
            tour.next();
            tour.next();
            expect(tour.currentStep).toBe(2);

            tour.restart();
            expect(tour.currentStep).toBe(0);
            expect(tour.isActive).toBe(true);
        });

        it('should persist tour completion', () => {
            const tour = helpProvider.startGuidedTour('dashboard');
            
            // Complete tour
            while (tour.hasNext()) {
                tour.next();
            }
            tour.next();

            // Check localStorage
            const completed = localStorage.getItem('help_tours_completed');
            expect(completed).toContain('dashboard');
        });

        it('should not start completed tour automatically', () => {
            // Mark as completed
            localStorage.setItem('help_tours_completed', JSON.stringify(['dashboard']));

            const tour = helpProvider.startGuidedTour('dashboard', { force: false });
            expect(tour).toBeNull();
        });
    });

    describe('Keyboard Shortcuts', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should show keyboard shortcuts modal', () => {
            helpProvider.showKeyboardShortcuts();

            const modal = document.querySelector('.keyboard-shortcuts-modal');
            expect(modal).toBeTruthy();
        });

        it('should display shortcuts list', () => {
            helpProvider.showKeyboardShortcuts();

            const shortcuts = document.querySelectorAll('.keyboard-shortcut-item');
            expect(shortcuts.length).toBeGreaterThan(0);
        });

        it('should close modal on escape', () => {
            helpProvider.showKeyboardShortcuts();

            const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape' });
            document.dispatchEvent(escapeEvent);

            const modal = document.querySelector('.keyboard-shortcuts-modal');
            expect(modal).toBeFalsy();
        });

        it('should register global keyboard shortcuts', () => {
            const spy = jest.spyOn(helpProvider, 'toggleHelp');

            // Trigger Ctrl+/
            const event = new KeyboardEvent('keydown', {
                key: '/',
                ctrlKey: true
            });
            document.dispatchEvent(event);

            expect(spy).toHaveBeenCalled();
        });
    });

    describe('Search', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should search help content', async () => {
            const results = await helpProvider.search('dashboard');

            expect(results).toBeDefined();
            expect(Array.isArray(results)).toBe(true);
            expect(results.length).toBeGreaterThan(0);
        });

        it('should return relevant search results', async () => {
            const results = await helpProvider.search('overview');

            expect(results[0]).toMatchObject({
                title: expect.stringContaining('Overview'),
                content: expect.any(String),
                path: expect.any(String)
            });
        });

        it('should handle empty search query', async () => {
            const results = await helpProvider.search('');
            expect(results).toEqual([]);
        });

        it('should debounce search input', async () => {
            const searchSpy = jest.spyOn(helpProvider, 'search');
            const input = document.createElement('input');
            
            helpProvider.enableSearchDebounce(input);

            // Type quickly
            input.value = 'd';
            input.dispatchEvent(new Event('input'));
            input.value = 'da';
            input.dispatchEvent(new Event('input'));
            input.value = 'das';
            input.dispatchEvent(new Event('input'));

            // Should not be called immediately
            expect(searchSpy).not.toHaveBeenCalled();

            // Wait for debounce
            await new Promise(resolve => setTimeout(resolve, 350));
            expect(searchSpy).toHaveBeenCalledTimes(1);
            expect(searchSpy).toHaveBeenCalledWith('das');
        });
    });

    describe('Event Handling', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should emit events', () => {
            const listener = jest.fn();
            helpProvider.on('help:opened', listener);

            helpProvider.showContextualHelp('dashboard');

            expect(listener).toHaveBeenCalledWith({
                view: 'dashboard',
                section: undefined
            });
        });

        it('should remove event listeners', () => {
            const listener = jest.fn();
            helpProvider.on('help:closed', listener);
            helpProvider.off('help:closed', listener);

            helpProvider.hideContextualHelp();

            expect(listener).not.toHaveBeenCalled();
        });

        it('should emit tour events', () => {
            const startListener = jest.fn();
            const completeListener = jest.fn();

            helpProvider.on('tour:started', startListener);
            helpProvider.on('tour:completed', completeListener);

            const tour = helpProvider.startGuidedTour('dashboard');
            expect(startListener).toHaveBeenCalledWith({ tourId: 'dashboard' });

            // Complete tour
            while (tour.hasNext()) {
                tour.next();
            }
            tour.next();

            expect(completeListener).toHaveBeenCalledWith({ tourId: 'dashboard' });
        });
    });

    describe('Digital Assistant Integration', () => {
        beforeEach(async () => {
            helpProvider = new HelpProvider({
                enableDigitalAssistant: true,
                digitalAssistantConfig: {
                    apiEndpoint: '/api/digital-assistant',
                    apiKey: 'test-key'
                }
            });
            await helpProvider.init();
        });

        it('should prepare digital assistant connector', () => {
            expect(helpProvider.digitalAssistantConnector).toBeDefined();
            expect(helpProvider.digitalAssistantConnector.isReady).toBe(true);
        });

        it('should handle digital assistant queries', async () => {
            const mockResponse = {
                answer: 'Here is the help you requested',
                confidence: 0.95
            };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse
            });

            const response = await helpProvider.queryDigitalAssistant('How do I create an agent?');

            expect(mockFetch).toHaveBeenCalledWith('/api/digital-assistant', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer test-key'
                },
                body: JSON.stringify({
                    query: 'How do I create an agent?',
                    context: expect.any(Object)
                })
            });

            expect(response).toEqual(mockResponse);
        });

        it('should handle digital assistant errors', async () => {
            mockFetch.mockRejectedValueOnce(new Error('API Error'));

            const response = await helpProvider.queryDigitalAssistant('Test query');

            expect(response).toEqual({
                error: 'Failed to query digital assistant',
                fallback: true
            });
        });
    });

    describe('Accessibility', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should add ARIA attributes to tooltips', () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Accessible tooltip'
            });

            expect(mockElement.getAttribute('aria-describedby')).toBeTruthy();
            
            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement.getAttribute('role')).toBe('tooltip');
        });

        it('should manage focus in help panel', () => {
            helpProvider.showContextualHelp('dashboard');

            const panel = document.querySelector('.contextual-help-panel');
            const focusableElements = panel.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );

            expect(focusableElements.length).toBeGreaterThan(0);
            expect(document.activeElement).toBe(focusableElements[0]);
        });

        it('should trap focus in guided tour', () => {
            const tour = helpProvider.startGuidedTour('dashboard');
            
            const popup = document.querySelector('.guided-tour-popup');
            const focusableElements = popup.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );

            // Simulate tab navigation
            const lastElement = focusableElements[focusableElements.length - 1];
            lastElement.focus();
            
            const tabEvent = new KeyboardEvent('keydown', {
                key: 'Tab',
                shiftKey: false
            });
            lastElement.dispatchEvent(tabEvent);

            // Focus should wrap to first element
            expect(document.activeElement).toBe(focusableElements[0]);
        });

        it('should announce help content changes', () => {
            helpProvider.showContextualHelp('dashboard');

            const liveRegion = document.querySelector('[aria-live="polite"]');
            expect(liveRegion).toBeTruthy();
            expect(liveRegion.textContent).toContain('Help panel opened');
        });
    });

    describe('Performance', () => {
        beforeEach(async () => {
            await helpProvider.init();
        });

        it('should lazy load help content', async () => {
            // Initial config should be minimal
            expect(helpProvider.helpConfig.views.agents).toBeUndefined();

            // Load agents help
            await helpProvider.loadHelpContent('agents');

            expect(mockFetch).toHaveBeenCalledWith(
                expect.stringContaining('agents')
            );
        });

        it('should cache help content', async () => {
            // First load
            await helpProvider.loadHelpContent('dashboard');
            expect(mockFetch).toHaveBeenCalledTimes(1);

            // Second load should use cache
            await helpProvider.loadHelpContent('dashboard');
            expect(mockFetch).toHaveBeenCalledTimes(1);
        });

        it('should clean up resources on destroy', () => {
            const tooltip = helpProvider.createTooltip(mockElement, {
                content: 'Test tooltip'
            });

            helpProvider.destroy();

            // Check that event listeners are removed
            mockElement.dispatchEvent(new MouseEvent('mouseenter'));
            const tooltipElement = document.querySelector('.help-tooltip-content');
            expect(tooltipElement).toBeFalsy();

            // Check that panels are removed
            const panel = document.querySelector('.contextual-help-panel');
            expect(panel).toBeFalsy();
        });
    });
});