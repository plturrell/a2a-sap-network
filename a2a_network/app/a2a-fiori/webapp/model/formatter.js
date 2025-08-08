sap.ui.define([
    "sap/ui/core/format/DateFormat",
    "sap/ui/core/ValueState",
    "sap/base/Log"
], function(DateFormat, ValueState, Log) {
    "use strict";

    return {
        /**
         * Formats date/time values.
         * @public
         * @param {Date|string} vDate - Date object or date string
         * @returns {string} Formatted date/time string
         */
        formatDateTime: function(vDate) {
            if (!vDate) {
                return "";
            }
            
            try {
                var oDateFormat = DateFormat.getDateTimeInstance({
                    style: "medium"
                });
                
                var oDate = vDate instanceof Date ? vDate : new Date(vDate);
                return oDateFormat.format(oDate);
            } catch (e) {
                Log.error("Date formatting failed", e);
                return "";
            }
        },

        /**
         * Calculates duration between two dates.
         * @public
         * @param {Date|string} vStart - Start date
         * @param {Date|string} vEnd - End date
         * @returns {string} Duration string
         */
        calculateDuration: function(vStart, vEnd) {
            if (!vStart) {
                return "";
            }
            
            try {
                var iStart = new Date(vStart).getTime();
                var iEnd = vEnd ? new Date(vEnd).getTime() : Date.now();
                var iDuration = iEnd - iStart;
                
                // Convert to human readable format
                var iSeconds = Math.floor(iDuration / 1000);
                var iMinutes = Math.floor(iSeconds / 60);
                var iHours = Math.floor(iMinutes / 60);
                var iDays = Math.floor(iHours / 24);
                
                if (iDays > 0) {
                    return iDays + "d " + (iHours % 24) + "h";
                } else if (iHours > 0) {
                    return iHours + "h " + (iMinutes % 60) + "m";
                } else if (iMinutes > 0) {
                    return iMinutes + "m " + (iSeconds % 60) + "s";
                } else {
                    return iSeconds + "s";
                }
            } catch (e) {
                Log.error("Duration calculation failed", e);
                return "";
            }
        },

        /**
         * Formats Ethereum address for display.
         * @public
         * @param {string} sAddress - Full Ethereum address
         * @returns {string} Shortened address
         */
        formatAddress: function(sAddress) {
            if (!sAddress || typeof sAddress !== "string") {
                return "";
            }
            
            // Validate Ethereum address format
            if (!/^0x[a-fA-F0-9]{40}$/.test(sAddress)) {
                return sAddress; // Return as-is if not valid format
            }
            
            return sAddress.substring(0, 6) + "..." + sAddress.substring(38);
        },

        /**
         * Formats reputation score and returns appropriate value state.
         * @public
         * @param {number} iScore - Reputation score
         * @returns {sap.ui.core.ValueState} Value state for UI5 controls
         */
        formatReputationState: function(iScore) {
            if (typeof iScore !== "number") {
                return ValueState.None;
            }
            
            if (iScore >= 150) {
                return ValueState.Success;
            } else if (iScore >= 100) {
                return ValueState.Warning;
            } else if (iScore >= 50) {
                return ValueState.Error;
            } else {
                return ValueState.Error;
            }
        },

        /**
         * Formats reputation score text.
         * @public
         * @param {number} iScore - Reputation score
         * @returns {string} Formatted reputation text
         */
        formatReputationText: function(iScore) {
            if (typeof iScore !== "number") {
                return "Unknown";
            }
            
            if (iScore >= 150) {
                return iScore + " (Excellent)";
            } else if (iScore >= 100) {
                return iScore + " (Good)";
            } else if (iScore >= 50) {
                return iScore + " (Fair)";
            } else {
                return iScore + " (Poor)";
            }
        },

        /**
         * Formats service status for display.
         * @public
         * @param {string} sStatus - Service status
         * @returns {string} Human readable status
         */
        formatServiceStatus: function(sStatus) {
            var mStatusMap = {
                "pending": "Pending",
                "active": "Active",
                "completed": "Completed",
                "cancelled": "Cancelled",
                "disputed": "Disputed",
                "failed": "Failed"
            };
            
            return mStatusMap[sStatus] || sStatus || "";
        },

        /**
         * Returns value state for service status.
         * @public
         * @param {string} sStatus - Service status
         * @returns {sap.ui.core.ValueState} Value state
         */
        formatServiceStatusState: function(sStatus) {
            switch (sStatus) {
                case "active":
                case "completed":
                    return ValueState.Success;
                case "pending":
                    return ValueState.Warning;
                case "cancelled":
                case "failed":
                case "disputed":
                    return ValueState.Error;
                default:
                    return ValueState.None;
            }
        },

        /**
         * Formats gas price from Wei to Gwei.
         * @public
         * @param {string|number} vWei - Gas price in Wei
         * @returns {string} Gas price in Gwei
         */
        formatGasPrice: function(vWei) {
            if (!vWei) {
                return "0 Gwei";
            }
            
            try {
                // Convert Wei to Gwei (1 Gwei = 10^9 Wei)
                var fGwei = parseFloat(vWei) / 1000000000;
                return fGwei.toFixed(2) + " Gwei";
            } catch (e) {
                Log.error("Gas price formatting failed", e);
                return "0 Gwei";
            }
        },

        /**
         * Formats percentage values.
         * @public
         * @param {number} fValue - Decimal value (0-1)
         * @returns {string} Percentage string
         */
        formatPercentage: function(fValue) {
            if (typeof fValue !== "number" || isNaN(fValue)) {
                return "0%";
            }
            
            return (fValue * 100).toFixed(1) + "%";
        },

        /**
         * Formats capability category.
         * @public
         * @param {number} iCategory - Category enum value
         * @returns {string} Category name
         */
        formatCapabilityCategory: function(iCategory) {
            var mCategories = {
                0: "Computation",
                1: "Storage",
                2: "Analysis",
                3: "Communication",
                4: "Governance",
                5: "Security",
                6: "Integration"
            };
            
            return mCategories[iCategory] || "Other";
        },

        /**
         * Returns icon for workflow status.
         * @public
         * @param {string} sStatus - Workflow status
         * @returns {string} Icon name
         */
        getWorkflowStatusIcon: function(sStatus) {
            var mIcons = {
                "running": "sap-icon://process",
                "completed": "sap-icon://sys-enter-2",
                "failed": "sap-icon://error",
                "cancelled": "sap-icon://sys-cancel",
                "paused": "sap-icon://pause"
            };
            
            return mIcons[sStatus] || "sap-icon://question-mark";
        },

        /**
         * Returns value state for workflow status.
         * @public
         * @param {string} sStatus - Workflow status
         * @returns {sap.ui.core.ValueState} Value state
         */
        formatWorkflowStatusState: function(sStatus) {
            switch (sStatus) {
                case "completed":
                    return ValueState.Success;
                case "running":
                case "paused":
                    return ValueState.Warning;
                case "failed":
                case "cancelled":
                    return ValueState.Error;
                default:
                    return ValueState.None;
            }
        },

        /**
         * Formats large numbers with K, M, B suffixes.
         * @public
         * @param {number} iNumber - Large number
         * @returns {string} Formatted number (e.g., 1.2K, 3.4M)
         */
        formatLargeNumber: function(iNumber) {
            if (typeof iNumber !== "number" || isNaN(iNumber)) {
                return "0";
            }
            
            var absNumber = Math.abs(iNumber);
            var sign = iNumber < 0 ? "-" : "";
            
            if (absNumber >= 1000000000) {
                return sign + (absNumber / 1000000000).toFixed(1) + "B";
            } else if (absNumber >= 1000000) {
                return sign + (absNumber / 1000000).toFixed(1) + "M";
            } else if (absNumber >= 1000) {
                return sign + (absNumber / 1000).toFixed(1) + "K";
            }
            
            return sign + absNumber.toString();
        },

        /**
         * Formats currency values.
         * @public
         * @param {number} fValue - Currency value
         * @param {string} sCurrency - Currency code (default: USD)
         * @returns {string} Formatted currency string
         */
        formatCurrency: function(fValue, sCurrency) {
            if (typeof fValue !== "number" || isNaN(fValue)) {
                return "";
            }
            
            var oCurrencyFormat = sap.ui.core.format.NumberFormat.getCurrencyInstance({
                currencyCode: false
            });
            
            return oCurrencyFormat.format(fValue, sCurrency || "USD");
        },

        /**
         * Checks if value is empty.
         * @public
         * @param {any} vValue - Value to check
         * @returns {boolean} True if empty
         */
        isEmpty: function(vValue) {
            return !vValue || 
                   (Array.isArray(vValue) && vValue.length === 0) ||
                   (typeof vValue === "object" && Object.keys(vValue).length === 0);
        },

        /**
         * Checks if value is not empty.
         * @public
         * @param {any} vValue - Value to check
         * @returns {boolean} True if not empty
         */
        isNotEmpty: function(vValue) {
            return !this.isEmpty(vValue);
        }
    };
});