sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (BaseObject, JSONModel, MessageToast, MessageBox) {
    "use strict";

    /**
     * Variant Management Service for A2A Agents
     * Provides user-specific view customization and variant saving capabilities
     */
    return BaseObject.extend("com.sap.a2a.developerportal.services.VariantManagementService", {

        constructor: function () {
            BaseObject.prototype.constructor.apply(this, arguments);
            
            // Initialize variant storage
            this._mVariants = {};
            this._sCurrentUser = this._getCurrentUser();
            
            // Load existing variants from backend or localStorage
            this._loadVariants();
        },

        /**
         * Save a variant for a specific control
         * @param {string} sControlId - Control identifier
         * @param {string} sVariantName - Variant name
         * @param {object} oVariantData - Variant data
         * @param {boolean} bIsDefault - Whether this is the default variant
         * @returns {Promise} Save promise
         */
        saveVariant: function (sControlId, sVariantName, oVariantData, bIsDefault) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    // Validate inputs
                    if (!sControlId || !sVariantName || !oVariantData) {
                        throw new Error("Control ID, variant name, and variant data are required");
                    }
                    
                    // Initialize control variants if not exists
                    if (!that._mVariants[sControlId]) {
                        that._mVariants[sControlId] = {};
                    }
                    
                    // Create variant object
                    var oVariant = {
                        id: that._generateVariantId(sControlId, sVariantName),
                        name: sVariantName,
                        data: oVariantData,
                        isDefault: bIsDefault || false,
                        userId: that._sCurrentUser,
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                        isPublic: false,
                        description: ""
                    };
                    
                    // Save variant
                    that._mVariants[sControlId][sVariantName] = oVariant;
                    
                    // Update default variant if specified
                    if (bIsDefault) {
                        that._setDefaultVariant(sControlId, sVariantName);
                    }
                    
                    // Persist to backend/localStorage
                    that._persistVariants().then(function () {
                        MessageToast.show("Variant '" + sVariantName + "' saved successfully");
                        resolve(oVariant);
                    }).catch(function (oError) {
                        reject(oError);
                    });
                    
                } catch (oError) {
                    console.error("Error saving variant:", oError);
                    MessageToast.show("Failed to save variant: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Load a variant for a specific control
         * @param {string} sControlId - Control identifier
         * @param {string} sVariantName - Variant name
         * @returns {object|null} Variant data or null if not found
         */
        loadVariant: function (sControlId, sVariantName) {
            var oControlVariants = this._mVariants[sControlId];
            
            if (oControlVariants && oControlVariants[sVariantName]) {
                return oControlVariants[sVariantName].data;
            }
            
            return null;
        },

        /**
         * Get all variants for a specific control
         * @param {string} sControlId - Control identifier
         * @returns {array} Array of variants
         */
        getVariants: function (sControlId) {
            var oControlVariants = this._mVariants[sControlId];
            var aVariants = [];
            
            if (oControlVariants) {
                for (var sVariantName in oControlVariants) {
                    aVariants.push(oControlVariants[sVariantName]);
                }
            }
            
            // Sort variants by name
            aVariants.sort(function (a, b) {
                return a.name.localeCompare(b.name);
            });
            
            return aVariants;
        },

        /**
         * Delete a variant
         * @param {string} sControlId - Control identifier
         * @param {string} sVariantName - Variant name
         * @returns {Promise} Delete promise
         */
        deleteVariant: function (sControlId, sVariantName) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    var oControlVariants = that._mVariants[sControlId];
                    
                    if (!oControlVariants || !oControlVariants[sVariantName]) {
                        throw new Error("Variant '" + sVariantName + "' not found");
                    }
                    
                    // Check if it's the default variant
                    var bIsDefault = oControlVariants[sVariantName].isDefault;
                    
                    // Delete variant
                    delete oControlVariants[sVariantName];
                    
                    // If deleted variant was default, set first available as default
                    if (bIsDefault) {
                        var aRemainingVariants = Object.keys(oControlVariants);
                        if (aRemainingVariants.length > 0) {
                            that._setDefaultVariant(sControlId, aRemainingVariants[0]);
                        }
                    }
                    
                    // Persist changes
                    that._persistVariants().then(function () {
                        MessageToast.show("Variant '" + sVariantName + "' deleted successfully");
                        resolve();
                    }).catch(function (oError) {
                        reject(oError);
                    });
                    
                } catch (oError) {
                    console.error("Error deleting variant:", oError);
                    MessageToast.show("Failed to delete variant: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Get the default variant for a control
         * @param {string} sControlId - Control identifier
         * @returns {object|null} Default variant or null
         */
        getDefaultVariant: function (sControlId) {
            var oControlVariants = this._mVariants[sControlId];
            
            if (oControlVariants) {
                for (var sVariantName in oControlVariants) {
                    if (oControlVariants[sVariantName].isDefault) {
                        return oControlVariants[sVariantName];
                    }
                }
            }
            
            return null;
        },

        /**
         * Set a variant as default
         * @param {string} sControlId - Control identifier
         * @param {string} sVariantName - Variant name
         * @returns {Promise} Set default promise
         */
        setDefaultVariant: function (sControlId, sVariantName) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    that._setDefaultVariant(sControlId, sVariantName);
                    
                    that._persistVariants().then(function () {
                        MessageToast.show("'" + sVariantName + "' set as default variant");
                        resolve();
                    }).catch(function (oError) {
                        reject(oError);
                    });
                    
                } catch (oError) {
                    console.error("Error setting default variant:", oError);
                    MessageToast.show("Failed to set default variant: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Update variant metadata
         * @param {string} sControlId - Control identifier
         * @param {string} sVariantName - Variant name
         * @param {object} oMetadata - Metadata to update
         * @returns {Promise} Update promise
         */
        updateVariantMetadata: function (sControlId, sVariantName, oMetadata) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    var oControlVariants = that._mVariants[sControlId];
                    
                    if (!oControlVariants || !oControlVariants[sVariantName]) {
                        throw new Error("Variant '" + sVariantName + "' not found");
                    }
                    
                    var oVariant = oControlVariants[sVariantName];
                    
                    // Update metadata
                    if (oMetadata.description !== undefined) {
                        oVariant.description = oMetadata.description;
                    }
                    if (oMetadata.isPublic !== undefined) {
                        oVariant.isPublic = oMetadata.isPublic;
                    }
                    
                    oVariant.updatedAt = new Date().toISOString();
                    
                    that._persistVariants().then(function () {
                        MessageToast.show("Variant metadata updated successfully");
                        resolve(oVariant);
                    }).catch(function (oError) {
                        reject(oError);
                    });
                    
                } catch (oError) {
                    console.error("Error updating variant metadata:", oError);
                    MessageToast.show("Failed to update variant metadata: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Export variants for backup
         * @param {string} sControlId - Control identifier (optional, exports all if not provided)
         * @returns {object} Exported variants
         */
        exportVariants: function (sControlId) {
            var oExportData = {
                exportDate: new Date().toISOString(),
                userId: this._sCurrentUser,
                variants: {}
            };
            
            if (sControlId) {
                if (this._mVariants[sControlId]) {
                    oExportData.variants[sControlId] = this._mVariants[sControlId];
                }
            } else {
                oExportData.variants = this._mVariants;
            }
            
            return oExportData;
        },

        /**
         * Import variants from backup
         * @param {object} oImportData - Import data
         * @param {boolean} bOverwrite - Whether to overwrite existing variants
         * @returns {Promise} Import promise
         */
        importVariants: function (oImportData, bOverwrite) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    if (!oImportData || !oImportData.variants) {
                        throw new Error("Invalid import data");
                    }
                    
                    var iImportedCount = 0;
                    var iSkippedCount = 0;
                    
                    for (var sControlId in oImportData.variants) {
                        var oControlVariants = oImportData.variants[sControlId];
                        
                        if (!that._mVariants[sControlId]) {
                            that._mVariants[sControlId] = {};
                        }
                        
                        for (var sVariantName in oControlVariants) {
                            if (!that._mVariants[sControlId][sVariantName] || bOverwrite) {
                                that._mVariants[sControlId][sVariantName] = oControlVariants[sVariantName];
                                iImportedCount++;
                            } else {
                                iSkippedCount++;
                            }
                        }
                    }
                    
                    that._persistVariants().then(function () {
                        var sMessage = "Import completed: " + iImportedCount + " variants imported";
                        if (iSkippedCount > 0) {
                            sMessage += ", " + iSkippedCount + " skipped";
                        }
                        MessageToast.show(sMessage);
                        resolve({ imported: iImportedCount, skipped: iSkippedCount });
                    }).catch(function (oError) {
                        reject(oError);
                    });
                    
                } catch (oError) {
                    console.error("Error importing variants:", oError);
                    MessageToast.show("Failed to import variants: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Get variant statistics
         * @returns {object} Variant statistics
         */
        getVariantStatistics: function () {
            var iTotalVariants = 0;
            var iControlsWithVariants = 0;
            var iPublicVariants = 0;
            var iDefaultVariants = 0;
            
            for (var sControlId in this._mVariants) {
                var oControlVariants = this._mVariants[sControlId];
                var iControlVariantCount = Object.keys(oControlVariants).length;
                
                if (iControlVariantCount > 0) {
                    iControlsWithVariants++;
                    iTotalVariants += iControlVariantCount;
                    
                    for (var sVariantName in oControlVariants) {
                        var oVariant = oControlVariants[sVariantName];
                        if (oVariant.isPublic) {
                            iPublicVariants++;
                        }
                        if (oVariant.isDefault) {
                            iDefaultVariants++;
                        }
                    }
                }
            }
            
            return {
                totalVariants: iTotalVariants,
                controlsWithVariants: iControlsWithVariants,
                publicVariants: iPublicVariants,
                defaultVariants: iDefaultVariants,
                userId: this._sCurrentUser
            };
        },

        // Private methods
        _getCurrentUser: function () {
            // In a real SAP BTP environment, get from XSUAA token
            // For now, use a fallback
            return "developer@company.com";
        },

        _generateVariantId: function (sControlId, sVariantName) {
            return sControlId + "_" + sVariantName + "_" + Date.now();
        },

        _setDefaultVariant: function (sControlId, sVariantName) {
            var oControlVariants = this._mVariants[sControlId];
            
            if (!oControlVariants || !oControlVariants[sVariantName]) {
                throw new Error("Variant '" + sVariantName + "' not found");
            }
            
            // Clear existing default
            for (var sExistingVariant in oControlVariants) {
                oControlVariants[sExistingVariant].isDefault = false;
            }
            
            // Set new default
            oControlVariants[sVariantName].isDefault = true;
            oControlVariants[sVariantName].updatedAt = new Date().toISOString();
        },

        _loadVariants: function () {
            var that = this;
            
            // Try to load from backend first
            jQuery.ajax({
                url: "/api/v2/user/variants",
                method: "GET",
                success: function (oData) {
                    that._mVariants = oData.variants || {};
                },
                error: function () {
                    // Fallback to localStorage
                    try {
                        var sStoredVariants = localStorage.getItem("a2a_variants_" + that._sCurrentUser);
                        if (sStoredVariants) {
                            that._mVariants = JSON.parse(sStoredVariants);
                        }
                    } catch (oError) {
                        console.warn("Failed to load variants from localStorage:", oError);
                        that._mVariants = {};
                    }
                }
            });
        },

        _persistVariants: function () {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                // Try to save to backend first
                jQuery.ajax({
                    url: "/api/v2/user/variants",
                    method: "PUT",
                    contentType: "application/json",
                    data: JSON.stringify({ variants: that._mVariants }),
                    success: function () {
                        resolve();
                    },
                    error: function () {
                        // Fallback to localStorage
                        try {
                            localStorage.setItem("a2a_variants_" + that._sCurrentUser, JSON.stringify(that._mVariants));
                            resolve();
                        } catch (oError) {
                            console.error("Failed to persist variants:", oError);
                            reject(oError);
                        }
                    }
                });
            });
        }
    });
});
