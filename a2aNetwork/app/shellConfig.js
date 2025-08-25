/* global sap */
sap.ui.define([], () => {
    'use strict';

    return {
        // Production shell configuration
        getShellConfig: function(environment) {
            const baseConfig = {
                defaultRenderer: 'fiori2',
                renderers: {
                    fiori2: {
                        componentData: {
                            config: {
                                enableSearch: true,
                                enablePersonalization: true,
                                enableNotifications: true,
                                enableMeArea: true,
                                enableBackButton: true,
                                rootIntent: 'Shell-home',
                                applications: {}
                            }
                        }
                    }
                },
                services: {
                    LaunchPage: {
                        adapter: {
                            config: {
                                groups: [
                                    {
                                        id: 'agent_management',
                                        title: 'Agent Management',
                                        isPreset: true,
                                        isVisible: true,
                                        isDefaultGroup: true
                                    },
                                    {
                                        id: 'analytics_monitoring',
                                        title: 'Analytics & Monitoring',
                                        isPreset: true,
                                        isVisible: true
                                    },
                                    {
                                        id: 'services_workflow',
                                        title: 'Services & Workflows',
                                        isPreset: true,
                                        isVisible: true
                                    },
                                    {
                                        id: 'security_compliance',
                                        title: 'Security & Compliance',
                                        isPreset: true,
                                        isVisible: true
                                    }
                                ]
                            }
                        }
                    },
                    ClientSideTargetResolution: {
                        adapter: {
                            config: {
                                inbounds: {}
                            }
                        }
                    },
                    NavTargetResolution: {
                        config: {
                            enableClientSideTargetResolution: true
                        }
                    },
                    UserInfo: {
                        adapter: {
                            config: {
                                themes: [
                                    { id: 'sap_horizon', name: 'SAP Horizon' },
                                    { id: 'sap_horizon_dark', name: 'SAP Horizon Dark' },
                                    { id: 'sap_horizon_hcb', name: 'SAP Horizon High Contrast Black' },
                                    { id: 'sap_horizon_hcw', name: 'SAP Horizon High Contrast White' }
                                ]
                            }
                        }
                    },
                    Container: {
                        adapter: {
                            config: {
                                appState: 'lean',
                                enableContentDensity: true,
                                enableHelp: true,
                                allowShowHideGroups: true,
                                enableTileActionsIcon: true,
                                sessionTimeoutIntervalInMinutes: 30,
                                sessionTimeoutReminderInMinutes: 5,
                                sessionTimeoutTileStopRefreshInMinutes: 15,
                                enableAutomaticSignout: true,
                                themes: {
                                    default: 'sap_horizon',
                                    available: ['sap_horizon', 'sap_horizon_dark', 'sap_horizon_hcb', 'sap_horizon_hcw']
                                }
                            }
                        }
                    }
                },
                bootstrapPlugins: {
                    RuntimeAuthoringPlugin: {
                        component: 'sap.ushell.plugins.rta',
                        config: {
                            validateAppVersion: true
                        }
                    }
                }
            };

            // Environment-specific configurations
            if (environment === 'production') {
                baseConfig.services.Container.adapter.config.enableDevelopmentMode = false;
                baseConfig.services.Container.adapter.config.enableDebugMode = false;
                baseConfig.services.Container.adapter.config.disablePersonalization = false;

                // Add performance optimizations
                baseConfig.services.Container.adapter.config.enableAsyncComponentLoading = true;
                baseConfig.services.Container.adapter.config.enableComponentPreload = true;

                // Security settings
                baseConfig.services.Container.adapter.config.enableXFrameOptions = true;
                baseConfig.services.Container.adapter.config.xFrameOptions = 'SAMEORIGIN';
                baseConfig.services.Container.adapter.config.enableCSP = true;

            } else if (environment === 'development') {
                baseConfig.services.Container.adapter.config.enableDevelopmentMode = true;
                baseConfig.services.Container.adapter.config.enableDebugMode = true;
            }

            return baseConfig;
        },

        // Shell plugins configuration
        getPluginsConfig: function() {
            return {
                PerformanceMonitoringPlugin: {
                    enabled: true,
                    config: {
                        threshold: 3000,
                        reportingUrl: '/api/v1/performance'
                    }
                },
                UserActivityPlugin: {
                    enabled: true,
                    config: {
                        trackClicks: true,
                        trackNavigation: true,
                        reportingInterval: 60000
                    }
                },
                ErrorReportingPlugin: {
                    enabled: true,
                    config: {
                        captureErrors: true,
                        reportingUrl: '/api/v1/errors'
                    }
                }
            };
        }
    };
});