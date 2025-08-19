/**
 * Visual Accessibility Tests
 * Based on testCases1Common.md - Visual Accessibility section
 */

describe('Visual Accessibility Tests', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  describe('Color Contrast Ratios', () => {
    it('should meet WCAG contrast requirements for normal text', () => {
      // Normal text (< 18pt) requires 4.5:1 contrast ratio
      cy.get('p, span, div').not('[class*="large"]').each(($el) => {
        if ($el.text().trim()) {
          cy.wrap($el).then(($element) => {
            const color = $element.css('color');
            const backgroundColor = getBackgroundColor($element);
            const ratio = getContrastRatio(color, backgroundColor);
            
            expect(ratio, `Text "${$element.text().substring(0, 20)}..." contrast ratio`).to.be.at.least(4.5);
          });
        }
      });
    });

    it('should meet WCAG contrast requirements for large text', () => {
      // Large text (≥ 18pt or 14pt bold) requires 3:1 contrast ratio
      cy.get('h1, h2, h3, h4, h5, h6, .large-text, [class*="heading"]').each(($el) => {
        if ($el.text().trim()) {
          cy.wrap($el).then(($element) => {
            const color = $element.css('color');
            const backgroundColor = getBackgroundColor($element);
            const ratio = getContrastRatio(color, backgroundColor);
            
            expect(ratio, `Large text "${$element.text().substring(0, 20)}..." contrast ratio`).to.be.at.least(3);
          });
        }
      });
    });

    it('should meet contrast requirements for UI components', () => {
      // UI components require 3:1 contrast ratio
      const uiComponents = [
        'button',
        'input',
        'select',
        'textarea',
        '[role="button"]',
        '[role="link"]'
      ];
      
      cy.get(uiComponents.join(', ')).each(($el) => {
        cy.wrap($el).then(($element) => {
          // Check border contrast
          const borderColor = $element.css('border-color');
          const backgroundColor = getBackgroundColor($element);
          const borderRatio = getContrastRatio(borderColor, backgroundColor);
          
          expect(borderRatio, 'UI component border contrast').to.be.at.least(3);
          
          // Check text contrast if applicable
          if ($element.text().trim()) {
            const textColor = $element.css('color');
            const textRatio = getContrastRatio(textColor, backgroundColor);
            expect(textRatio, 'UI component text contrast').to.be.at.least(4.5);
          }
        });
      });
    });

    it('should maintain contrast in different states', () => {
      // Test hover states
      cy.get('button, a').each(($el) => {
        cy.wrap($el).trigger('mouseover').then(($element) => {
          const color = $element.css('color');
          const backgroundColor = getBackgroundColor($element);
          const ratio = getContrastRatio(color, backgroundColor);
          
          expect(ratio, 'Hover state contrast').to.be.at.least(4.5);
        });
      });
      
      // Test focus states
      cy.get('button, a, input').each(($el) => {
        cy.wrap($el).focus().then(($element) => {
          const outlineColor = $element.css('outline-color');
          const backgroundColor = getBackgroundColor($element);
          const ratio = getContrastRatio(outlineColor, backgroundColor);
          
          expect(ratio, 'Focus outline contrast').to.be.at.least(3);
        });
      });
    });
  });

  describe('Text Readability at Zoom', () => {
    it('should remain readable at 200% zoom without horizontal scroll', () => {
      // Set viewport to simulate 200% zoom
      cy.viewport(640, 480);
      
      // Check no horizontal scroll
      cy.window().then((win) => {
        expect(win.document.documentElement.scrollWidth).to.be.lte(win.innerWidth);
      });
      
      // Text should still be visible
      cy.get('p, h1, h2, h3').should('be.visible');
      
      // Verify text doesn't overflow containers
      cy.get('p, div').each(($el) => {
        cy.wrap($el).then(($element) => {
          const elementWidth = $element.width();
          const parentWidth = $element.parent().width();
          expect(elementWidth).to.be.lte(parentWidth);
        });
      });
    });

    it('should support text spacing adjustments', () => {
      // Apply WCAG text spacing requirements
      cy.get('body').then($body => {
        $body.css({
          'line-height': '1.5',
          'letter-spacing': '0.12em',
          'word-spacing': '0.16em'
        });
        
        // Paragraphs should have 2x line height spacing
        $body.find('p').css('margin-bottom', '2em');
      });
      
      // Verify content is still readable and doesn't overflow
      cy.get('p, h1, h2, h3').should('be.visible');
      cy.get('.container, .content').each(($el) => {
        cy.wrap($el).should('not.have.css', 'overflow', 'hidden');
      });
    });

    it('should reflow text for mobile viewports', () => {
      // Test at 320px width (minimum mobile)
      cy.viewport(320, 568);
      
      // No horizontal scrolling except for data tables
      cy.window().then((win) => {
        const scrollWidth = win.document.documentElement.scrollWidth;
        const clientWidth = win.document.documentElement.clientWidth;
        expect(scrollWidth).to.be.lte(clientWidth + 20); // Allow small margin
      });
      
      // Text should reflow to single column
      cy.get('main p').each(($p) => {
        cy.wrap($p).should('have.css', 'width').and('be.lte', '320px');
      });
    });
  });

  describe('Error State Indicators', () => {
    beforeEach(() => {
      cy.visit('/forms/sample');
    });

    it('should use icons plus color for error states', () => {
      // Submit invalid form
      cy.get('form').submit();
      
      // Check error messages have both color and icon
      cy.get('.error, [class*="error"]').each(($error) => {
        // Should have error color (typically red)
        cy.wrap($error).should('have.css', 'color').and('match', /rgb\(2\d{2}, \d{1,2}, \d{1,2}\)/);
        
        // Should have error icon
        cy.wrap($error).find('svg, .icon, [class*="icon"]').should('exist');
      });
      
      // Form fields should have multiple error indicators
      cy.get('input:invalid').each(($input) => {
        // Border color change
        cy.wrap($input).should('have.css', 'border-color').and('not.equal', 'rgb(0, 0, 0)');
        
        // Error message below field
        const id = $input.attr('id');
        cy.get(`[id="${id}-error"]`).should('exist').and('be.visible');
        
        // Aria-invalid attribute
        cy.wrap($input).should('have.attr', 'aria-invalid', 'true');
      });
    });

    it('should not rely on color alone for status', () => {
      // Success states
      cy.get('.success, [class*="success"]').each(($success) => {
        // Should have success icon
        cy.wrap($success).find('svg, .icon, [class*="icon"]').should('exist');
      });
      
      // Warning states
      cy.get('.warning, [class*="warning"]').each(($warning) => {
        // Should have warning icon
        cy.wrap($warning).find('svg, .icon, [class*="icon"]').should('exist');
      });
      
      // Info states
      cy.get('.info, [class*="info"]').each(($info) => {
        // Should have info icon
        cy.wrap($info).find('svg, .icon, [class*="icon"]').should('exist');
      });
    });
  });

  describe('Focus Indicators', () => {
    it('should have visible focus indicators with 2px minimum border', () => {
      const focusableElements = [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])'
      ];
      
      cy.get(focusableElements.join(', ')).each(($el) => {
        cy.wrap($el).focus();
        
        // Check outline width
        cy.wrap($el).should('have.css', 'outline-width').then((outlineWidth) => {
          const width = parseInt(outlineWidth);
          expect(width).to.be.at.least(2);
        });
        
        // Check outline style
        cy.wrap($el).should('have.css', 'outline-style').and('not.equal', 'none');
        
        // Verify outline is visible (has color)
        cy.wrap($el).should('have.css', 'outline-color').and('not.equal', 'transparent');
      });
    });

    it('should maintain focus visibility on dark backgrounds', () => {
      cy.visit('/dark-theme');
      
      cy.get('button, a, input').each(($el) => {
        cy.wrap($el).focus();
        
        // Get colors
        cy.wrap($el).then(($element) => {
          const outlineColor = $element.css('outline-color');
          const backgroundColor = getBackgroundColor($element);
          const ratio = getContrastRatio(outlineColor, backgroundColor);
          
          expect(ratio, 'Focus indicator contrast on dark background').to.be.at.least(3);
        });
      });
    });
  });

  describe('Animation and Motion', () => {
    it('should respect prefers-reduced-motion', () => {
      // Set reduced motion preference
      cy.wrap(window).invoke('matchMedia', '(prefers-reduced-motion: reduce)').then((mediaQuery) => {
        // Mock the media query
        cy.stub(mediaQuery, 'matches').value(true);
      });
      
      // Check animations are disabled or reduced
      cy.get('[class*="animate"], [class*="transition"]').each(($el) => {
        cy.wrap($el).should('have.css', 'animation-duration', '0s')
          .or('have.css', 'transition-duration', '0s');
      });
    });

    it('should provide controls for auto-playing content', () => {
      cy.visit('/carousel');
      
      // Auto-playing content should have pause button
      cy.get('[data-autoplay="true"]').each(($el) => {
        cy.wrap($el).parent().find('button[aria-label*="pause"], button[aria-label*="stop"]').should('exist');
      });
      
      // Videos should not autoplay
      cy.get('video').each(($video) => {
        cy.wrap($video).should('not.have.attr', 'autoplay');
      });
    });
  });

  describe('Target Size', () => {
    it('should have minimum 44x44px touch targets', () => {
      const interactiveElements = [
        'button',
        'a',
        'input[type="checkbox"]',
        'input[type="radio"]',
        '[role="button"]',
        '[onclick]'
      ];
      
      cy.get(interactiveElements.join(', ')).each(($el) => {
        cy.wrap($el).then(($element) => {
          const width = $element.outerWidth();
          const height = $element.outerHeight();
          
          // Either dimension should be at least 44px
          expect(Math.max(width, height), 'Touch target size').to.be.at.least(44);
        });
      });
    });

    it('should have adequate spacing between targets', () => {
      cy.get('button').each(($button, index, $buttons) => {
        if (index < $buttons.length - 1) {
          const nextButton = $buttons[index + 1];
          const rect1 = $button[0].getBoundingClientRect();
          const rect2 = nextButton.getBoundingClientRect();
          
          // Calculate minimum distance
          const horizontalDistance = Math.abs(rect2.left - rect1.right);
          const verticalDistance = Math.abs(rect2.top - rect1.bottom);
          
          // At least 8px spacing
          expect(Math.min(horizontalDistance, verticalDistance), 'Target spacing').to.be.at.least(8);
        }
      });
    });
  });

  describe('Color Independence', () => {
    it('should not use color as the only indicator', () => {
      // Links should be underlined or have other indicators
      cy.get('a').not('nav a').each(($link) => {
        cy.wrap($link).should('satisfy', ($el) => {
          const textDecoration = $el.css('text-decoration');
          const fontWeight = $el.css('font-weight');
          const hasIcon = $el.find('svg, .icon').length > 0;
          
          // Should have underline, bold, or icon
          return textDecoration.includes('underline') || 
                 parseInt(fontWeight) >= 600 || 
                 hasIcon;
        });
      });
    });

    it('should provide patterns or labels for data visualization', () => {
      cy.visit('/charts');
      
      // Check chart elements have labels
      cy.get('[role="img"][aria-label*="chart"], canvas, svg.chart').each(($chart) => {
        // Should have accessible description
        cy.wrap($chart).should('have.attr', 'aria-label')
          .or('have.attr', 'aria-describedby');
      });
      
      // Legend items should have text labels
      cy.get('.legend-item, [class*="legend"] li').each(($item) => {
        cy.wrap($item).should('not.be.empty');
      });
    });
  });
});

// Helper function to get contrast ratio between two colors
function getContrastRatio(color1, color2) {
  const rgb1 = parseColor(color1);
  const rgb2 = parseColor(color2);
  
  const l1 = getLuminance(rgb1);
  const l2 = getLuminance(rgb2);
  
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  
  return (lighter + 0.05) / (darker + 0.05);
}

// Helper function to parse color string to RGB
function parseColor(color) {
  const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (match) {
    return {
      r: parseInt(match[1]),
      g: parseInt(match[2]),
      b: parseInt(match[3])
    };
  }
  return { r: 0, g: 0, b: 0 };
}

// Helper function to calculate relative luminance
function getLuminance(rgb) {
  const { r, g, b } = rgb;
  const [rs, gs, bs] = [r, g, b].map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

// Helper function to get computed background color
function getBackgroundColor($element) {
  let bgColor = $element.css('background-color');
  let parent = $element.parent();
  
  // Traverse up until we find a non-transparent background
  while (bgColor === 'transparent' || bgColor === 'rgba(0, 0, 0, 0)' && parent.length) {
    bgColor = parent.css('background-color');
    parent = parent.parent();
  }
  
  return bgColor || 'rgb(255, 255, 255)'; // Default to white
}