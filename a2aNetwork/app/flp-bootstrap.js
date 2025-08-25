/**
 * SAP Fiori Launchpad Enterprise Bootstrap
 * Compliant with SAP NetWeaver 7.50+ and SAP BTP
 */
(function() {
    'use strict';

    // Enterprise configuration based on deployment type
    const deploymentType = window['sap-ui-config'] && window['sap-ui-config']['deployment-type'] || 'standalone';

    // Check for SAP NetWeaver environment
    const isNetWeaver = window.location.pathname.indexOf('/sap/bc/') !== -1;
    const isBTP = window.location.hostname.indexOf('.hana.ondemand.com') !== -1 ||
                window.location.hostname.indexOf('.cfapps.') !== -1;

    // Enterprise Shell Configuration
    window['sap-ushell-config'] = {
        defaultRenderer: 'fiori2',
        bootstrapPlugins: {
            'KeyUserPlugin': {
                component: 'sap.ushell.plugins.rta'
            },
            'PersonalizePlugin': {
                component: 'sap.ushell.plugins.flp.Personalization'
            }
        },
        renderers: {
            fiori2: {
                componentData: {
                    config: {
                        rootIntent: 'Shell-home',
                        enableSearch: true,
                        enablePersonalization: true,
                        enableTransientMode: false,
                        enableSetTheme: true,
                        enableSetLanguage: true,
                        enableAccessibility: true,
                        enableHelp: true,
                        enableUserActivityLog: true,
                        preloadLibrariesForRootIntent: true,
                        animationMode: 'full',
                        theme: 'sap_horizon',
                        floatingFooterMode: 'EmbedInShell'
                    }
                }
            }
        },
        services: {
            // Common Data Model configuration for enterprise
            CommonDataModel: {
                adapter: {
                    module: isNetWeaver ?
                        'sap.ushell.adapters.abap.CommonDataModelAdapter' :
                        'sap.ushell.adapters.cdm.v3.CommonDataModelAdapter',
                    config: {
                        cdmSiteUrl: isNetWeaver ?
                            '/sap/opu/odata/UI2/INTEROP' :
                            '/portal-site-api/v1/cdm/sites'
                    }
                }
            },

            // Container service for shell management
            Container: {
                adapter: {
                    config: {
                        userProfilePersonalization: true,
                        contentProviders: isNetWeaver ? ['ABAP_CONTENT_PROVIDER'] : ['CDM_CONTENT_PROVIDER'],
                        systemAliases: {
                            '': {
                                id: '',
                                client: isNetWeaver ? sap.ushell.Container.getLogonSystem().getClient() : '',
                                language: sap.ui.getCore().getConfiguration().getLanguage()
                            }
                        }
                    }
                }
            },

            // Navigation service
            CrossApplicationNavigation: {
                adapter: {
                    config: {
                        semanticObjectWhitelist: ['*']
                    }
                }
            },

            // User information service
            UserInfo: {
                adapter: {
                    module: isNetWeaver ?
                        'sap.ushell.adapters.abap.UserInfoAdapter' :
                        'sap.ushell.adapters.cdm.v3.UserInfoAdapter'
                }
            },

            // Personalization service
            Personalization: {
                adapter: {
                    module: isNetWeaver ?
                        'sap.ushell.adapters.abap.PersonalizationAdapter' :
                        'sap.ushell.adapters.cdm.v3.PersonalizationAdapter'
                }
            },

            // App State service
            AppState: {
                adapter: {
                    module: isNetWeaver ?
                        'sap.ushell.adapters.abap.AppStateAdapter' :
                        'sap.ushell.adapters.cdm.v3.AppStateAdapter'
                }
            },

            // Navigation Target Resolution
            ClientSideTargetResolution: {
                adapter: {
                    module: 'sap.ushell.adapters.cdm.v3.ClientSideTargetResolutionAdapter',
                    config: {
                        siteDataUrl: isNetWeaver ?
                            '/sap/bc/ui2/flp/sitedata' :
                            '/portal-site-api/v1/cdm/sitedata'
                    }
                }
            },

            // Support Ticket Integration
            SupportTicket: {
                adapter: {
                    config: {
                        ticketServiceUrl: isNetWeaver ?
                            '/sap/bc/webdynpro/sap/ags_incident_create' :
                            '/support/ticket'
                    }
                }
            }
        },

        // Enterprise-specific configurations
        bootConfig: {
            xhrLogon: isNetWeaver,
            theme: 'sap_horizon',
            accessibility: true,
            preloadBundles: true,
            enableAutoLogout: true,
            sessionTimeoutIntervalInMinutes: 30,
            sessionTimeoutReminderInMinutes: 5,
            sessionTimeoutTileStopRefreshInMinutes: 15,
            appCacheBuster: true,
            enableContentDensity: true,
            contentDensity: 'cozy',
            messagePusher: {
                enabled: true,
                pollingInterval: 30000
            }
        }
    };

    // Initialize shell with enterprise configuration
    if (isNetWeaver) {
        // For NetWeaver, ensure proper service paths
        sap.ui.loader.config({
            paths: {
                'sap/ushell': '/sap/bc/ui5_ui5/ui2/ushell/resources/sap/ushell'
            }
        });
    } else if (isBTP) {
        // For BTP, use CDN or platform resources
        sap.ui.loader.config({
            paths: {
                'sap/ushell': sap.ui.require.toUrl('sap/ushell')
            }
        });
    }

    // Load and initialize the shell
    sap.ui.require([
        'sap/ushell/bootstrap/cdm',
        'sap/ui/core/ComponentSupport'
    ], (cdmBootstrap) => {
        // Initialize with CDM bootstrap for enterprise
        cdmBootstrap.bootstrap('cdm', {
            defaultRenderer: 'fiori2',
            appCacheBuster: true
        }).then(() => {
            // Shell is ready
            sap.ushell.Container.createRenderer('fiori2').then((oRenderer) => {
                oRenderer.placeAt('content');
            });
        }).catch((oError) => {
            console.error('Failed to bootstrap FLP:', oError);
            // SAP Standard error handling
            sap.ui.require(['sap/m/MessageBox'], (MessageBox) => {
                MessageBox.error('Failed to load SAP Fiori Launchpad', {
                    title: 'System Error',
                    details: oError.message || oError.toString(),
                    actions: [MessageBox.Action.OK],
                    emphasizedAction: MessageBox.Action.OK
                });
            });
        });
    });
})();