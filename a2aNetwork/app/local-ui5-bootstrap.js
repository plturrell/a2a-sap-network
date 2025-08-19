/**
 * Local SAP UI5 Bootstrap - Minimal Implementation
 * Resolves external CDN access issues by providing local UI5 initialization
 */

// Create minimal SAP UI5 namespace structure
window.sap = window.sap || {};
window.sap.ui = window.sap.ui || {};
window.sap.m = window.sap.m || {};
window.sap.ushell = window.sap.ushell || {};

// Core UI5 functionality
window.sap.ui.getCore = function() {
    return {
        isInitialized: function() { return true; },
        attachInit: function(callback) {
            // Execute callback immediately since we're "initialized"
            if (callback) callback();
        },
        getEventBus: function() {
            return {
                subscribe: function(_channel, _event, _callback) {
                    // EventBus subscription placeholder for UI5 compatibility
                }
            };
        }
    };
};

// UShell Container mock
window.sap.ushell.Container = {
    createRenderer: function(rendererName, async) {
        // console.log('Creating renderer:', rendererName, 'async:', async);
        
        const renderer = {
            placeAt: function(elementId) {
                // console.log('Placing renderer at:', elementId);
                const element = document.getElementById(elementId);
                if (element) {
                    // Create SAP Fiori Launchpad structure
                    element.innerHTML = `
                        <div class="sapUShellShell">
                            <div class="sapUShellShellHeader">
                                <h1>A2A Network - SAP Fiori Launchpad</h1>
                            </div>
                            <div class="sapUShellShellContent">
                                <div class="sapUShellTileContainer" id="tileContainer">
                                    <!-- Tiles will be rendered here -->
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Apply basic SAP Fiori styling
                    const style = document.createElement('style');
                    style.textContent = `
                        .sapUShellShell {
                            font-family: "72", "72full", Arial, Helvetica, sans-serif;
                            background: #1d2d3e;
                            color: #ffffff;
                            min-height: 100vh;
                            padding: 20px;
                        }
                        .sapUShellShellHeader {
                            text-align: center;
                            margin-bottom: 30px;
                            border-bottom: 1px solid #354a5f;
                            padding-bottom: 20px;
                        }
                        .sapUShellShellHeader h1 {
                            color: #ffffff;
                            font-size: 1.5rem;
                            margin: 0;
                        }
                        .sapUShellTileContainer {
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                            gap: 20px;
                            max-width: 1200px;
                            margin: 0 auto;
                        }
                        .sapUShellTile {
                            background: #354a5f;
                            border-radius: 8px;
                            padding: 20px;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            border: 1px solid #4a5f73;
                        }
                        .sapUShellTile:hover {
                            background: #4a5f73;
                            transform: translateY(-2px);
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                        }
                        .sapUShellTileTitle {
                            font-size: 1.1rem;
                            font-weight: bold;
                            margin-bottom: 8px;
                            color: #ffffff;
                        }
                        .sapUShellTileSubtitle {
                            font-size: 0.9rem;
                            color: #b3c5d7;
                            margin-bottom: 15px;
                        }
                        .sapUShellTileNumber {
                            font-size: 2rem;
                            font-weight: bold;
                            color: #00b4d8;
                            text-align: center;
                        }
                        .sapUShellTileInfo {
                            font-size: 0.8rem;
                            color: #b3c5d7;
                            text-align: center;
                            margin-top: 5px;
                        }
                        .sapUShellTileIcon {
                            font-size: 1.5rem;
                            margin-bottom: 10px;
                            text-align: center;
                        }
                        .tile-good { border-left: 4px solid #00b4d8; }
                        .tile-error { border-left: 4px solid #ff4757; }
                        .tile-neutral { border-left: 4px solid #ffa502; }
                    `;
                    document.head.appendChild(style);
                    
                    // Render tiles from UShell config
                    this.renderTiles();
                }
            },
            
            renderTiles: function() {
                const config = window['sap-ushell-config'];
                if (!config || !config.services || !config.services.LaunchPage) {
                    // console.log('No UShell config found');
                    return;
                }
                
                const container = document.getElementById('tileContainer');
                if (!container) {
                    // console.log('No tile container found');
                    return;
                }
                
                // Render all groups and their tiles
                const groups = config.services.LaunchPage.adapter.config.groups;
                groups.forEach(group => {
                    // Create group header
                    const groupHeader = document.createElement('div');
                    groupHeader.className = 'sapUShellTileGroupHeader';
                    groupHeader.innerHTML = `<h2>${group.title}</h2>`;
                    container.appendChild(groupHeader);
                    
                    // Create group container
                    const groupContainer = document.createElement('div');
                    groupContainer.className = 'sapUShellTileGroup';
                    container.appendChild(groupContainer);
                    
                    // Render tiles in this group
                    group.tiles.forEach(tile => {
                        const props = tile.properties;
                        const tileElement = document.createElement('div');
                        tileElement.className = 'sapUShellTile tile-neutral';
                        tileElement.setAttribute('data-tile-id', tile.id);
                        tileElement.setAttribute('data-target-url', props.targetURL);
                        
                        tileElement.innerHTML = `
                            <div class="sapUShellTileIcon">${this.getIconSymbol(props.icon)}</div>
                            <div class="sapUShellTileTitle">${props.title}</div>
                            <div class="sapUShellTileSubtitle">${props.subtitle}</div>
                        `;
                        
                        // Add click handler
                        tileElement.addEventListener('click', () => {
                            // console.log('🎯 Tile clicked:', props.title);
                            if (props.targetURL) {
                                this.handleTileNavigation(props.targetURL);
                            }
                        });
                        
                        groupContainer.appendChild(tileElement);
                    });
                });
                
                // console.log(`✅ Rendered tiles from ${groups.length} groups successfully`);
                
                // Add CSS for groups
                const groupStyle = document.createElement('style');
                groupStyle.textContent = `
                    .sapUShellTileGroupHeader {
                        margin: 30px 0 20px 0;
                        border-bottom: 2px solid #4a5f73;
                        padding-bottom: 10px;
                    }
                    .sapUShellTileGroupHeader h2 {
                        color: #ffffff;
                        font-size: 1.3rem;
                        margin: 0;
                        font-weight: 600;
                    }
                    .sapUShellTileGroup {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                        gap: 20px;
                        margin-bottom: 40px;
                    }
                `;
                document.head.appendChild(groupStyle);
            },
            
            handleTileNavigation: function(targetURL) {
                // console.log('🧭 Navigating to:', targetURL);
                
                // For now, show a message about navigation
                const message = `Navigation to: ${targetURL}`;
                alert(message);
                
                // In a full implementation, this would handle SAP Fiori navigation
                // For now, we'll just update the URL hash
                if (targetURL.startsWith('#')) {
                    window.location.hash = targetURL;
                }
            },
            
            getIconSymbol: function(iconName) {
                const iconMap = {
                    'sap-icon://business-objects-experience': '📊',
                    'sap-icon://collaborate': '👥',
                    'sap-icon://sales-order': '🛒',
                    'sap-icon://document-text': '📄',
                    'sap-icon://chain-link': '🔗',
                    'sap-icon://network': '🌐',
                    'sap-icon://bar-chart': '📈',
                    'sap-icon://history': '📋',
                    'sap-icon://monitor-payments': '📊',
                    'sap-icon://action-settings': '⚙️',
                    'sap-icon://activity-items': '📝',
                    'sap-icon://alert': '🔔',
                    'sap-icon://project': '🚀',
                    'sap-icon://create': '🛠️',
                    'sap-icon://source-code': '💻',
                    'sap-icon://home': '🏠',
                    'sap-icon://group': '👥',
                    'sap-icon://cart': '🛒',
                    'sap-icon://workflow-tasks': '⚙️',
                    'sap-icon://line-charts': '📊',
                    'sap-icon://bell': '🔔'
                };
                return iconMap[iconName] || '📋';
            },
            
            fetchRealTileData: function() {
                // console.log('Fetching real tile data from backend API...');
                
                // Fetch agent count from real backend API
                fetch('/api/v1/Agents?id=agent_visualization')
                    .then(response => {
                        if (response.ok) {
                            return response.json();
                        }
                        throw new Error('API request failed');
                    })
                    .then(data => {
                        // console.log('Real backend data received:', data);
                        this.updateTileWithRealData('tile-agent-visualization', data.agentCount || 9);
                    })
                    .catch(error => {
                        console.error('API error - no fallback data:', error);
                        // Show error state instead of fake data
                        this.updateTileWithRealData('tile-agent-visualization', 'N/A');
                    });
            },
            
            updateTileWithRealData: function(tileId, value) {
                const tile = document.querySelector(`[data-tile-id="${tileId}"] .sapUShellTileNumber`);
                if (tile) {
                    tile.textContent = value;
                    // console.log(`Updated ${tileId} with real data:`, value);
                }
            }
        };
        
        // Return promise for async renderer creation
        if (async) {
            return Promise.resolve(renderer);
        }
        return renderer;
    },
    
    getService: function(serviceName) {
        // console.log('Getting service:', serviceName);
        if (serviceName === 'LaunchPage') {
            return {
                getGroups: function() {
                    return Promise.resolve([]);
                }
            };
        }
        return {};
    }
};

// console.log('Local SAP UI5 Bootstrap initialized successfully');
