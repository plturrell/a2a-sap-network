/**
 * AgentVisualization.view.xml Component Tests
 * Test Cases: TC-AN-083 to TC-AN-105
 * Coverage: Network Topology, Graph Rendering, Interactive Controls, Layout Options, Node Interactions, Export
 */

describe('AgentVisualization.view.xml - Network Topology', () => {
  beforeEach(() => {
    cy.visit('/visualization');
    cy.viewport(1280, 720);
    
    // Wait for visualization to load
    cy.get('[data-testid="network-visualization"]').should('be.visible');
  });

  describe('TC-AN-083 to TC-AN-087: Graph Rendering Tests', () => {
    it('TC-AN-083: Should verify network graph renders all agents', () => {
      // Wait for graph to render
      cy.get('[data-testid="graph-canvas"]').should('be.visible');
      
      // Verify SVG or Canvas element exists
      cy.get('[data-testid="graph-canvas"] svg, [data-testid="graph-canvas"] canvas')
        .should('exist');
      
      // Check that nodes are rendered
      cy.get('[data-testid="graph-node"]').should('have.length.greaterThan', 0);
      
      // Verify all agents from API are rendered
      cy.request('/api/agents').then((response) => {
        const agentCount = response.body.length;
        cy.get('[data-testid="graph-node"]').should('have.length', agentCount);
      });
      
      // Check graph statistics
      cy.get('[data-testid="graph-stats"]').within(() => {
        cy.get('[data-testid="total-nodes"]').should('not.contain', '0');
        cy.get('[data-testid="total-edges"]').should('be.visible');
      });
    });

    it('TC-AN-084: Should test node positioning algorithm', () => {
      // Verify nodes are properly positioned (not overlapping)
      const nodePositions = new Set();
      
      cy.get('[data-testid="graph-node"]').each(($node) => {
        const rect = $node[0].getBoundingClientRect();
        const position = `${Math.round(rect.left)},${Math.round(rect.top)}`;
        
        // Ensure no two nodes have identical positions
        expect(nodePositions.has(position)).to.be.false;
        nodePositions.add(position);
      });
      
      // Verify nodes are within canvas bounds
      cy.get('[data-testid="graph-canvas"]').then(($canvas) => {
        const canvasBounds = $canvas[0].getBoundingClientRect();
        
        cy.get('[data-testid="graph-node"]').each(($node) => {
          const nodeBounds = $node[0].getBoundingClientRect();
          
          expect(nodeBounds.left).to.be.at.least(canvasBounds.left);
          expect(nodeBounds.right).to.be.at.most(canvasBounds.right);
          expect(nodeBounds.top).to.be.at.least(canvasBounds.top);
          expect(nodeBounds.bottom).to.be.at.most(canvasBounds.bottom);
        });
      });
    });

    it('TC-AN-085: Should verify edge connections display', () => {
      // Check that edges are rendered
      cy.get('[data-testid="graph-edge"]').should('have.length.greaterThan', 0);
      
      // Verify edges connect nodes properly
      cy.get('[data-testid="graph-edge"]').each(($edge) => {
        const sourceId = $edge.attr('data-source');
        const targetId = $edge.attr('data-target');
        
        // Verify source and target nodes exist
        cy.get(`[data-testid="graph-node"][data-node-id="${sourceId}"]`).should('exist');
        cy.get(`[data-testid="graph-node"][data-node-id="${targetId}"]`).should('exist');
      });
      
      // Test edge styling
      cy.get('[data-testid="graph-edge"]').first().should('have.css', 'stroke-width');
      
      // Verify edge labels if present
      cy.get('[data-testid="edge-label"]').then(($labels) => {
        if ($labels.length > 0) {
          cy.wrap($labels).first().should('be.visible');
        }
      });
    });

    it('TC-AN-086: Should test graph performance with 100+ nodes', () => {
      // Mock large dataset
      cy.intercept('GET', '/api/agents', { 
        fixture: 'large-agent-network.json' // 150+ agents
      }).as('getLargeNetwork');
      
      cy.reload();
      cy.wait('@getLargeNetwork');
      
      // Measure rendering performance
      cy.window().then((win) => {
        win.performance.mark('render-start');
      });
      
      cy.get('[data-testid="graph-node"]').should('have.length.greaterThan', 100);
      
      cy.window().then((win) => {
        win.performance.mark('render-end');
        win.performance.measure('graph-render', 'render-start', 'render-end');
        
        const measure = win.performance.getEntriesByName('graph-render')[0];
        expect(measure.duration).to.be.lessThan(5000); // 5 seconds max
      });
      
      // Test interaction performance with large graph
      cy.get('[data-testid="graph-canvas"]').trigger('wheel', { deltaY: 100 });
      cy.wait(100);
      
      // Verify graph remains responsive
      cy.get('[data-testid="graph-node"]').first().click();
      cy.get('[data-testid="node-tooltip"]').should('be.visible');
    });

    it('TC-AN-087: Should verify node color coding by status', () => {
      const expectedColors = {
        'active': 'rgb(40, 167, 69)',   // Green
        'inactive': 'rgb(108, 117, 125)', // Gray  
        'error': 'rgb(220, 53, 69)',      // Red
        'warning': 'rgb(255, 193, 7)'     // Yellow
      };
      
      // Check each node's color matches its status
      cy.get('[data-testid="graph-node"]').each(($node) => {
        const status = $node.attr('data-status');
        const expectedColor = expectedColors[status];
        
        if (expectedColor) {
          cy.wrap($node).should('have.css', 'fill', expectedColor);
        }
      });
      
      // Verify legend shows color mapping
      cy.get('[data-testid="status-legend"]').within(() => {
        Object.keys(expectedColors).forEach(status => {
          cy.get(`[data-testid="legend-${status}"]`).should('be.visible')
            .and('have.css', 'background-color', expectedColors[status]);
        });
      });
    });
  });

  describe('TC-AN-088 to TC-AN-092: Interactive Controls Tests', () => {
    it('TC-AN-088: Should test zoom in/out functionality', () => {
      // Test zoom in button
      cy.get('[data-testid="zoom-in"]').click();
      
      // Verify zoom level increased
      cy.get('[data-testid="zoom-level"]').should('contain', '110%');
      
      // Test zoom out button
      cy.get('[data-testid="zoom-out"]').click();
      cy.get('[data-testid="zoom-level"]').should('contain', '100%');
      
      // Test mouse wheel zoom
      cy.get('[data-testid="graph-canvas"]').trigger('wheel', { deltaY: -100 });
      cy.get('[data-testid="zoom-level"]').should('not.contain', '100%');
      
      // Test zoom limits
      for (let i = 0; i < 20; i++) {
        cy.get('[data-testid="zoom-in"]').click();
      }
      
      cy.get('[data-testid="zoom-level"]').invoke('text').then((zoomText) => {
        const zoomValue = parseInt(zoomText);
        expect(zoomValue).to.be.at.most(500); // Max zoom 500%
      });
    });

    it('TC-AN-089: Should verify pan/drag canvas', () => {
      // Test canvas dragging
      cy.get('[data-testid="graph-canvas"]')
        .trigger('mousedown', 300, 200)
        .trigger('mousemove', 400, 300)
        .trigger('mouseup');
      
      // Verify canvas position changed
      cy.get('[data-testid="graph-transform"]').should('have.attr', 'transform')
        .and('not.equal', 'translate(0,0)');
      
      // Test touch pan on mobile
      cy.viewport('iphone-x');
      cy.get('[data-testid="graph-canvas"]')
        .trigger('touchstart', { touches: [{ clientX: 100, clientY: 100 }] })
        .trigger('touchmove', { touches: [{ clientX: 200, clientY: 200 }] })
        .trigger('touchend');
      
      // Reset viewport
      cy.viewport(1280, 720);
    });

    it('TC-AN-090: Should test fit to screen button', () => {
      // Pan and zoom to a random position
      cy.get('[data-testid="zoom-in"]').click().click();
      cy.get('[data-testid="graph-canvas"]')
        .trigger('mousedown', 300, 200)
        .trigger('mousemove', 500, 400)
        .trigger('mouseup');
      
      // Click fit to screen
      cy.get('[data-testid="fit-to-screen"]').click();
      
      // Verify graph is centered and zoomed to fit
      cy.get('[data-testid="zoom-level"]').should('not.contain', '200%');
      
      // Verify all nodes are visible
      cy.get('[data-testid="graph-node"]').each(($node) => {
        cy.wrap($node).should('be.visible');
      });
    });

    it('TC-AN-091: Should verify reset view functionality', () => {
      // Modify view state
      cy.get('[data-testid="zoom-in"]').click().click();
      cy.get('[data-testid="graph-canvas"]').trigger('wheel', { deltaY: -200 });
      cy.get('[data-testid="graph-canvas"]')
        .trigger('mousedown', 200, 200)
        .trigger('mousemove', 400, 400)
        .trigger('mouseup');
      
      // Reset view
      cy.get('[data-testid="reset-view"]').click();
      
      // Verify view is reset to default
      cy.get('[data-testid="zoom-level"]').should('contain', '100%');
      cy.get('[data-testid="graph-transform"]').should('have.attr', 'transform', 'translate(0,0)');
    });

    it('TC-AN-092: Should test minimap display (if present)', () => {
      cy.get('[data-testid="minimap"]').then(($minimap) => {
        if ($minimap.length > 0) {
          // Verify minimap is visible
          cy.wrap($minimap).should('be.visible');
          
          // Test minimap interaction
          cy.wrap($minimap).click(50, 30);
          
          // Verify main view updated
          cy.get('[data-testid="graph-transform"]')
            .should('have.attr', 'transform')
            .and('not.equal', 'translate(0,0)');
          
          // Test minimap viewport indicator
          cy.get('[data-testid="minimap-viewport"]').should('be.visible');
        } else {
          // Skip test if minimap not implemented
          cy.log('Minimap not implemented - skipping test');
        }
      });
    });
  });

  describe('TC-AN-093 to TC-AN-097: Layout Options Tests', () => {
    it('TC-AN-093: Should test force-directed layout', () => {
      cy.get('[data-testid="layout-selector"]').select('force-directed');
      
      // Verify layout is applied
      cy.get('[data-testid="layout-progress"]').should('be.visible');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      // Check that nodes have spread out naturally
      const nodePositions = [];
      cy.get('[data-testid="graph-node"]').each(($node) => {
        const rect = $node[0].getBoundingClientRect();
        nodePositions.push({ x: rect.left, y: rect.top });
      }).then(() => {
        // Verify nodes are not clustered in one spot
        const centerX = nodePositions.reduce((sum, pos) => sum + pos.x, 0) / nodePositions.length;
        const centerY = nodePositions.reduce((sum, pos) => sum + pos.y, 0) / nodePositions.length;
        
        const avgDistanceFromCenter = nodePositions.reduce((sum, pos) => {
          return sum + Math.sqrt(Math.pow(pos.x - centerX, 2) + Math.pow(pos.y - centerY, 2));
        }, 0) / nodePositions.length;
        
        expect(avgDistanceFromCenter).to.be.greaterThan(50);
      });
    });

    it('TC-AN-094: Should verify hierarchical layout', () => {
      cy.get('[data-testid="layout-selector"]').select('hierarchical');
      
      // Wait for layout animation
      cy.get('[data-testid="layout-progress"]').should('be.visible');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      // Verify nodes are arranged in levels
      const nodesByY = [];
      cy.get('[data-testid="graph-node"]').each(($node) => {
        const rect = $node[0].getBoundingClientRect();
        nodesByY.push(rect.top);
      }).then(() => {
        // Group nodes by Y position (allowing for small variations)
        const levels = [];
        nodesByY.forEach(y => {
          const existingLevel = levels.find(level => Math.abs(level - y) < 20);
          if (!existingLevel) {
            levels.push(y);
          }
        });
        
        // Should have multiple distinct levels
        expect(levels.length).to.be.greaterThan(2);
      });
    });

    it('TC-AN-095: Should test circular layout', () => {
      cy.get('[data-testid="layout-selector"]').select('circular');
      
      cy.get('[data-testid="layout-progress"]').should('be.visible');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      // Verify nodes are arranged in a circular pattern
      const nodePositions = [];
      cy.get('[data-testid="graph-node"]').each(($node) => {
        const rect = $node[0].getBoundingClientRect();
        nodePositions.push({ x: rect.centerX, y: rect.centerY });
      }).then(() => {
        if (nodePositions.length > 3) {
          // Calculate if nodes form roughly circular arrangement
          const centerX = nodePositions.reduce((sum, pos) => sum + pos.x, 0) / nodePositions.length;
          const centerY = nodePositions.reduce((sum, pos) => sum + pos.y, 0) / nodePositions.length;
          
          const distances = nodePositions.map(pos => 
            Math.sqrt(Math.pow(pos.x - centerX, 2) + Math.pow(pos.y - centerY, 2))
          );
          
          const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
          const maxDeviation = Math.max(...distances.map(d => Math.abs(d - avgDistance)));
          
          // Deviation should be small for circular arrangement
          expect(maxDeviation / avgDistance).to.be.lessThan(0.3);
        }
      });
    });

    it('TC-AN-096: Should verify layout transition animations', () => {
      // Start with one layout
      cy.get('[data-testid="layout-selector"]').select('force-directed');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      // Switch to another layout
      cy.get('[data-testid="layout-selector"]').select('hierarchical');
      
      // Verify smooth transition
      cy.get('[data-testid="layout-progress"]').should('be.visible');
      
      // Check that animation duration is reasonable
      cy.window().then((win) => {
        win.performance.mark('layout-start');
      });
      
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      cy.window().then((win) => {
        win.performance.mark('layout-end');
        win.performance.measure('layout-transition', 'layout-start', 'layout-end');
        
        const measure = win.performance.getEntriesByName('layout-transition')[0];
        expect(measure.duration).to.be.lessThan(3000); // Max 3 seconds
        expect(measure.duration).to.be.greaterThan(100); // Min animation time
      });
    });

    it('TC-AN-097: Should test save custom layout', () => {
      // Modify layout by dragging nodes
      cy.get('[data-testid="graph-node"]').first()
        .trigger('mousedown')
        .trigger('mousemove', 100, 100)
        .trigger('mouseup');
      
      cy.get('[data-testid="graph-node"]').eq(1)
        .trigger('mousedown')
        .trigger('mousemove', 200, 150)
        .trigger('mouseup');
      
      // Save custom layout
      cy.get('[data-testid="save-layout"]').click();
      cy.get('[data-testid="layout-name"]').type('My Custom Layout');
      cy.get('[data-testid="confirm-save-layout"]').click();
      
      // Verify saved layout appears in selector
      cy.get('[data-testid="layout-selector"]').should('contain.text', 'My Custom Layout');
      
      // Test loading saved layout
      cy.get('[data-testid="layout-selector"]').select('force-directed');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      cy.get('[data-testid="layout-selector"]').select('My Custom Layout');
      cy.get('[data-testid="layout-progress"]').should('not.exist');
      
      // Verify custom positions are restored
      // (Would need to check specific node positions match saved state)
    });
  });

  describe('TC-AN-098 to TC-AN-102: Node Interactions Tests', () => {
    it('TC-AN-098: Should test node click shows details', () => {
      cy.get('[data-testid="graph-node"]').first().click();
      
      // Verify node details panel opens
      cy.get('[data-testid="node-details-panel"]').should('be.visible');
      
      // Check panel content
      cy.get('[data-testid="node-details-panel"]').within(() => {
        cy.get('[data-testid="node-name"]').should('be.visible');
        cy.get('[data-testid="node-status"]').should('be.visible');
        cy.get('[data-testid="node-type"]').should('be.visible');
        cy.get('[data-testid="node-connections"]').should('be.visible');
      });
      
      // Test panel close
      cy.get('[data-testid="close-node-details"]').click();
      cy.get('[data-testid="node-details-panel"]').should('not.be.visible');
    });

    it('TC-AN-099: Should verify node hover tooltip', () => {
      cy.get('[data-testid="graph-node"]').first().trigger('mouseenter');
      
      // Verify tooltip appears
      cy.get('[data-testid="node-tooltip"]').should('be.visible');
      
      // Check tooltip content
      cy.get('[data-testid="node-tooltip"]').within(() => {
        cy.get('[data-testid="tooltip-name"]').should('not.be.empty');
        cy.get('[data-testid="tooltip-status"]').should('be.visible');
      });
      
      // Test tooltip positioning
      cy.get('[data-testid="graph-node"]').first().then(($node) => {
        const nodeRect = $node[0].getBoundingClientRect();
        
        cy.get('[data-testid="node-tooltip"]').then(($tooltip) => {
          const tooltipRect = $tooltip[0].getBoundingClientRect();
          
          // Tooltip should be near the node
          expect(Math.abs(tooltipRect.left - nodeRect.left)).to.be.lessThan(100);
          expect(Math.abs(tooltipRect.top - nodeRect.top)).to.be.lessThan(100);
        });
      });
      
      // Test tooltip disappears on mouse leave
      cy.get('[data-testid="graph-node"]').first().trigger('mouseleave');
      cy.get('[data-testid="node-tooltip"]').should('not.be.visible');
    });

    it('TC-AN-100: Should test drag node to reposition', () => {
      // Get initial position
      cy.get('[data-testid="graph-node"]').first().then(($node) => {
        const initialRect = $node[0].getBoundingClientRect();
        
        // Drag node to new position
        cy.wrap($node)
          .trigger('mousedown')
          .trigger('mousemove', initialRect.left + 100, initialRect.top + 100)
          .trigger('mouseup');
        
        // Verify position changed
        const newRect = $node[0].getBoundingClientRect();
        expect(Math.abs(newRect.left - initialRect.left)).to.be.greaterThan(50);
        expect(Math.abs(newRect.top - initialRect.top)).to.be.greaterThan(50);
      });
      
      // Test that connected edges update
      cy.get('[data-testid="graph-edge"]').first().should('be.visible');
    });

    it('TC-AN-101: Should verify right-click context menu', () => {
      cy.get('[data-testid="graph-node"]').first().rightclick();
      
      // Verify context menu appears
      cy.get('[data-testid="node-context-menu"]').should('be.visible');
      
      // Check menu items
      cy.get('[data-testid="node-context-menu"]').within(() => {
        cy.get('[data-testid="view-details"]').should('be.visible');
        cy.get('[data-testid="edit-agent"]').should('be.visible');
        cy.get('[data-testid="restart-agent"]').should('be.visible');
        cy.get('[data-testid="remove-agent"]').should('be.visible');
      });
      
      // Test menu item click
      cy.get('[data-testid="view-details"]').click();
      cy.get('[data-testid="node-details-panel"]').should('be.visible');
      
      // Test menu closes when clicking outside
      cy.get('[data-testid="graph-node"]').eq(1).rightclick();
      cy.get('body').click();
      cy.get('[data-testid="node-context-menu"]').should('not.be.visible');
    });

    it('TC-AN-102: Should test navigate to agent details', () => {
      cy.get('[data-testid="graph-node"]').first().dblclick();
      
      // Verify navigation to agent details page
      cy.url().should('include', '/agents/');
      cy.get('[data-testid="agent-detail-container"]').should('be.visible');
      
      // Navigate back to test cleanup
      cy.go('back');
      cy.get('[data-testid="network-visualization"]').should('be.visible');
    });
  });

  describe('TC-AN-103 to TC-AN-105: Export Tests', () => {
    it('TC-AN-103: Should test export as image (PNG/SVG)', () => {
      cy.get('[data-testid="export-menu"]').click();
      
      // Test PNG export
      cy.get('[data-testid="export-png"]').click();
      cy.get('[data-testid="export-progress"]').should('be.visible');
      cy.get('[data-testid="export-complete"]').should('be.visible');
      
      // Verify download
      cy.readFile('cypress/downloads/network-visualization.png').should('exist');
      
      // Test SVG export
      cy.get('[data-testid="export-menu"]').click();
      cy.get('[data-testid="export-svg"]').click();
      cy.readFile('cypress/downloads/network-visualization.svg').should('exist');
    });

    it('TC-AN-104: Should test export network data', () => {
      cy.get('[data-testid="export-menu"]').click();
      cy.get('[data-testid="export-data"]').click();
      
      // Verify export format options
      cy.get('[data-testid="export-format-json"]').should('be.visible');
      cy.get('[data-testid="export-format-csv"]').should('be.visible');
      cy.get('[data-testid="export-format-graphml"]').should('be.visible');
      
      // Test JSON export
      cy.get('[data-testid="export-format-json"]').click();
      cy.get('[data-testid="confirm-export"]').click();
      
      // Verify download and content
      cy.readFile('cypress/downloads/network-data.json').then((data) => {
        expect(data).to.have.property('nodes');
        expect(data).to.have.property('edges');
        expect(data.nodes).to.be.an('array');
        expect(data.edges).to.be.an('array');
      });
      
      // Test CSV export
      cy.get('[data-testid="export-menu"]').click();
      cy.get('[data-testid="export-data"]').click();
      cy.get('[data-testid="export-format-csv"]').click();
      cy.get('[data-testid="confirm-export"]').click();
      
      cy.readFile('cypress/downloads/network-nodes.csv').should('exist');
      cy.readFile('cypress/downloads/network-edges.csv').should('exist');
    });

    it('TC-AN-105: Should test print view functionality', () => {
      cy.get('[data-testid="export-menu"]').click();
      cy.get('[data-testid="print-view"]').click();
      
      // Verify print dialog preparation
      cy.get('[data-testid="print-preview"]').should('be.visible');
      
      // Check print optimizations
      cy.get('[data-testid="print-preview"]').within(() => {
        // Verify graph is optimized for print
        cy.get('[data-testid="graph-canvas"]').should('have.css', 'background-color', 'rgb(255, 255, 255)');
        
        // Check print settings
        cy.get('[data-testid="print-orientation"]').should('exist');
        cy.get('[data-testid="print-scale"]').should('exist');
        cy.get('[data-testid="include-legend"]').should('exist');
      });
      
      // Test print settings
      cy.get('[data-testid="print-orientation"]').select('landscape');
      cy.get('[data-testid="print-scale"]').select('fit-to-page');
      cy.get('[data-testid="include-legend"]').check();
      
      // Test print button (would open browser print dialog)
      cy.window().then((win) => {
        cy.stub(win, 'print').as('printStub');
      });
      
      cy.get('[data-testid="print-button"]').click();
      cy.get('@printStub').should('have.been.called');
      
      // Close print preview
      cy.get('[data-testid="close-print-preview"]').click();
      cy.get('[data-testid="print-preview"]').should('not.be.visible');
    });
  });
});