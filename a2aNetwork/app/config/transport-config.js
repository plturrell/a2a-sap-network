/* global sap, jQuery */
sap.ui.define([
    'sap/base/Log'
], (Log) => {
    'use strict';

    /**
     * Transport Management Configuration for SAP Enterprise
     * Handles CTS/CTS+ transport requests for Fiori content
     */
    return {
        /**
         * Transport configuration settings
         */
        config: {
            // Transport system settings
            transportSystem: {
                type: 'CTS_PLUS', // CTS or CTS_PLUS
                systemId: 'A2A',
                client: '100',
                transportLayer: 'ZA2A'
            },

            // Development package configuration
            packages: {
                root: 'ZA2A_FIORI',
                subPackages: {
                    apps: 'ZA2A_FIORI_APPS',
                    catalogs: 'ZA2A_FIORI_CATALOGS',
                    groups: 'ZA2A_FIORI_GROUPS',
                    roles: 'ZA2A_FIORI_ROLES',
                    tiles: 'ZA2A_FIORI_TILES'
                }
            },

            // Object types for transport
            objectTypes: {
                'CHIP': {
                    description: 'Fiori Tile/Chip',
                    package: 'tiles',
                    transportRequired: true
                },
                'PAGE': {
                    description: 'Fiori Page',
                    package: 'apps',
                    transportRequired: true
                },
                'CATA': {
                    description: 'Fiori Catalog',
                    package: 'catalogs',
                    transportRequired: true
                },
                'GRUP': {
                    description: 'Fiori Group',
                    package: 'groups',
                    transportRequired: true
                },
                'ROLE': {
                    description: 'PFCG Role',
                    package: 'roles',
                    transportRequired: true
                },
                'W3MI': {
                    description: 'Web Dynpro MIME Object',
                    package: 'apps',
                    transportRequired: true
                },
                'SICF': {
                    description: 'ICF Service',
                    package: 'apps',
                    transportRequired: true
                }
            },

            // Transport request types
            requestTypes: {
                'CUST': 'Customizing Request',
                'WORK': 'Workbench Request',
                'TRAN': 'Transport of Copies'
            },

            // Auto-transport settings
            autoTransport: {
                enabled: false,
                targetSystem: 'QA2',
                importImmediately: false
            }
        },

        /**
         * Create transport request
         * @param {Object} params - Transport parameters
         * @returns {Promise} Transport request number
         */
        createTransportRequest: function(params) {
            return new Promise((resolve, reject) => {
                const requestData = {
                    description: params.description || 'A2A Fiori Content Transport',
                    type: params.type || 'WORK',
                    targetSystem: params.targetSystem || this.config.autoTransport.targetSystem,
                    package: this._determinePackage(params.objectType)
                };

                jQuery.ajax({
                    url: '/sap/bc/adt/cts/transportrequests',
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/vnd.sap.adt.transportrequests.v1+xml',
                        'Accept': 'application/vnd.sap.adt.transportrequests.v1+xml'
                    },
                    data: this._buildTransportRequestXML(requestData),
                    success: (data) => {
                        const transportNumber = this._extractTransportNumber(data);
                        Log.info('Transport request created', transportNumber);
                        resolve(transportNumber);
                    },
                    error: (error) => {
                        Log.error('Failed to create transport request', error);
                        // Fallback for non-SAP environments
                        if (error.status === 404) {
                            resolve(this._createLocalTransportRequest(params));
                        } else {
                            reject(error);
                        }
                    }
                });
            });
        },

        /**
         * Add object to transport request
         * @param {string} transportNumber - Transport request number
         * @param {Object} object - Object to add
         * @returns {Promise} Success
         */
        addObjectToTransport: function(transportNumber, object) {
            return new Promise((resolve, reject) => {
                const objectData = {
                    pgmid: object.pgmid || 'R3TR',
                    object: object.type,
                    objName: object.name,
                    package: this._determinePackage(object.type)
                };

                jQuery.ajax({
                    url: `/sap/bc/adt/cts/transportrequests/${transportNumber}/tasks/objects`,
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/vnd.sap.adt.transportobjects.v1+xml'
                    },
                    data: this._buildTransportObjectXML(objectData),
                    success: () => {
                        Log.info('Object added to transport', { transport: transportNumber, object: object.name });
                        resolve();
                    },
                    error: (error) => {
                        Log.error('Failed to add object to transport', error);
                        reject(error);
                    }
                });
            });
        },

        /**
         * Release transport request
         * @param {string} transportNumber - Transport request number
         * @returns {Promise} Success
         */
        releaseTransportRequest: function(transportNumber) {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/sap/bc/adt/cts/transportrequests/${transportNumber}/release`,
                    method: 'POST',
                    success: () => {
                        Log.info('Transport request released', transportNumber);

                        if (this.config.autoTransport.enabled) {
                            this._triggerImport(transportNumber);
                        }

                        resolve();
                    },
                    error: (error) => {
                        Log.error('Failed to release transport request', error);
                        reject(error);
                    }
                });
            });
        },

        /**
         * Get transport history for object
         * @param {Object} object - Object to check
         * @returns {Promise} Transport history
         */
        getTransportHistory: function(object) {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: '/sap/bc/adt/vit/wb/object_requests',
                    method: 'GET',
                    data: {
                        pgmid: object.pgmid || 'R3TR',
                        object: object.type,
                        objName: object.name
                    },
                    success: (data) => {
                        const history = this._parseTransportHistory(data);
                        resolve(history);
                    },
                    error: (error) => {
                        Log.error('Failed to get transport history', error);
                        resolve([]); // Return empty history on error
                    }
                });
            });
        },

        /**
         * Determine package for object type
         * @private
         */
        _determinePackage: function(objectType) {
            const typeConfig = this.config.objectTypes[objectType];
            if (typeConfig && typeConfig.package) {
                return this.config.packages.subPackages[typeConfig.package] ||
                       this.config.packages.root;
            }
            return this.config.packages.root;
        },

        /**
         * Build transport request XML
         * @private
         */
        _buildTransportRequestXML: function(data) {
            return `<?xml version="1.0" encoding="UTF-8"?>
<asx:abap xmlns:asx="http://www.sap.com/abapxml" version="1.0">
  <asx:values>
    <DATA>
      <REQUESTHEADER>
        <TRFUNCTION>${data.type}</TRFUNCTION>
        <TRSTATUS>D</TRSTATUS>
        <TARSYSTEM>${data.targetSystem}</TARSYSTEM>
        <AS4TEXT>${data.description}</AS4TEXT>
        <CLIENT>${this.config.transportSystem.client}</CLIENT>
      </REQUESTHEADER>
    </DATA>
  </asx:values>
</asx:abap>`;
        },

        /**
         * Build transport object XML
         * @private
         */
        _buildTransportObjectXML: function(data) {
            return `<?xml version="1.0" encoding="UTF-8"?>
<asx:abap xmlns:asx="http://www.sap.com/abapxml" version="1.0">
  <asx:values>
    <DATA>
      <OBJECT>
        <PGMID>${data.pgmid}</PGMID>
        <OBJECT>${data.object}</OBJECT>
        <OBJ_NAME>${data.objName}</OBJ_NAME>
        <DEVCLASS>${data.package}</DEVCLASS>
      </OBJECT>
    </DATA>
  </asx:values>
</asx:abap>`;
        },

        /**
         * Extract transport number from response
         * @private
         */
        _extractTransportNumber: function(xmlData) {
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(xmlData, 'text/xml');
            const transportNode = xmlDoc.querySelector('TRKORR');
            return transportNode ? transportNode.textContent : null;
        },

        /**
         * Create local transport request for development
         * @private
         */
        _createLocalTransportRequest: function(params) {
            const transportNumber = `A2AK9${  Date.now().toString().substr(-5)}`;
            Log.info('Created local transport request', transportNumber);

            // Store in local storage for development
            const transports = JSON.parse(localStorage.getItem('a2a_transports') || '[]');
            transports.push({
                number: transportNumber,
                description: params.description,
                created: new Date().toISOString(),
                objects: []
            });
            localStorage.setItem('a2a_transports', JSON.stringify(transports));

            return transportNumber;
        },

        /**
         * Parse transport history from XML
         * @private
         */
        _parseTransportHistory: function(xmlData) {
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(xmlData, 'text/xml');
            const requests = xmlDoc.querySelectorAll('REQUEST');

            return Array.from(requests).map(req => ({
                number: req.querySelector('TRKORR')?.textContent,
                description: req.querySelector('AS4TEXT')?.textContent,
                owner: req.querySelector('AS4USER')?.textContent,
                date: req.querySelector('AS4DATE')?.textContent,
                time: req.querySelector('AS4TIME')?.textContent,
                status: req.querySelector('TRSTATUS')?.textContent
            }));
        },

        /**
         * Trigger import to target system
         * @private
         */
        _triggerImport: function(transportNumber) {
            if (!this.config.autoTransport.importImmediately) {
                return;
            }

            // Implementation would trigger TMS import
            Log.info('Auto-import triggered for transport', transportNumber);
        }
    };
});