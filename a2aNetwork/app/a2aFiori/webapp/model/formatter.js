sap.ui.define([
    "sap/ui/core/format/DateFormat",
    "sap/ui/core/ValueState",
    "sap/base/Log"
], (DateFormat, ValueState, Log) => {
    "use strict";

    return {
        /**
         * Formats date/time values.
         * @public
         * @param {Date|string} vDate - Date object or date string
         * @returns {string} Formatted date/time string
         */
        formatDateTime(vDate) {
            if (!vDate) {
                return "";
            }

            try {
                const oDateFormat = DateFormat.getDateTimeInstance({
                    style: "medium"
                });

                const oDate = vDate instanceof Date ? vDate : new Date(vDate);
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
        calculateDuration(vStart, vEnd) {
            if (!vStart) {
                return "";
            }

            try {
                const iStart = new Date(vStart).getTime();
                const iEnd = vEnd ? new Date(vEnd).getTime() : Date.now();
                const iDuration = iEnd - iStart;

                // Convert to human readable format
                const iSeconds = Math.floor(iDuration / 1000);
                const iMinutes = Math.floor(iSeconds / 60);
                const iHours = Math.floor(iMinutes / 60);
                const iDays = Math.floor(iHours / 24);

                if (iDays > 0) {
                    return `${iDays }d ${ iHours % 24 }h`;
                } else if (iHours > 0) {
                    return `${iHours }h ${ iMinutes % 60 }m`;
                } else if (iMinutes > 0) {
                    return `${iMinutes }m ${ iSeconds % 60 }s`;
                }
                return `${iSeconds }s`;

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
        formatAddress(sAddress) {
            if (!sAddress || typeof sAddress !== "string") {
                return "";
            }

            // Validate Ethereum address format
            if (!/^0x[a-fA-F0-9]{40}$/.test(sAddress)) {
                return sAddress; // Return as-is if not valid format
            }

            return `${sAddress.substring(0, 6) }...${ sAddress.substring(38)}`;
        },

        /**
         * Formats reputation score and returns appropriate value state.
         * @public
         * @param {number} iScore - Reputation score
         * @returns {sap.ui.core.ValueState} Value state for UI5 controls
         */
        formatReputationState(iScore) {
            if (typeof iScore !== "number") {
                return ValueState.None;
            }

            if (iScore >= 150) {
                return ValueState.Success;
            } else if (iScore >= 100) {
                return ValueState.Warning;
            } else if (iScore >= 50) {
                return ValueState.Error;
            }
            return ValueState.Error;

        },

        /**
         * Formats reputation score text.
         * @public
         * @param {number} iScore - Reputation score
         * @returns {string} Formatted reputation text
         */
        formatReputationText(iScore) {
            if (typeof iScore !== "number") {
                return "Unknown";
            }

            if (iScore >= 150) {
                return `${iScore } (Excellent)`;
            } else if (iScore >= 100) {
                return `${iScore } (Good)`;
            } else if (iScore >= 50) {
                return `${iScore } (Fair)`;
            }
            return `${iScore } (Poor)`;

        },

        /**
         * Formats service status for display.
         * @public
         * @param {string} sStatus - Service status
         * @returns {string} Human readable status
         */
        formatServiceStatus(sStatus) {
            const mStatusMap = {
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
        formatServiceStatusState(sStatus) {
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
        formatGasPrice(vWei) {
            if (!vWei) {
                return "0 Gwei";
            }

            try {
                // Convert Wei to Gwei (1 Gwei = 10^9 Wei)
                const fGwei = parseFloat(vWei) / 1000000000;
                return `${fGwei.toFixed(2) } Gwei`;
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
        formatPercentage(fValue) {
            if (typeof fValue !== "number" || isNaN(fValue)) {
                return "0%";
            }

            return `${(fValue * 100).toFixed(1) }%`;
        },

        /**
         * Formats capability category.
         * @public
         * @param {number} iCategory - Category enum value
         * @returns {string} Category name
         */
        formatCapabilityCategory(iCategory) {
            const mCategories = {
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
        getWorkflowStatusIcon(sStatus) {
            const mIcons = {
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
        formatWorkflowStatusState(sStatus) {
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
        formatLargeNumber(iNumber) {
            if (typeof iNumber !== "number" || isNaN(iNumber)) {
                return "0";
            }

            const absNumber = Math.abs(iNumber);
            const sign = iNumber < 0 ? "-" : "";

            if (absNumber >= 1000000000) {
                return `${sign + (absNumber / 1000000000).toFixed(1) }B`;
            } else if (absNumber >= 1000000) {
                return `${sign + (absNumber / 1000000).toFixed(1) }M`;
            } else if (absNumber >= 1000) {
                return `${sign + (absNumber / 1000).toFixed(1) }K`;
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
        formatCurrency(fValue, sCurrency) {
            if (typeof fValue !== "number" || isNaN(fValue)) {
                return "";
            }

            const oCurrencyFormat = sap.ui.core.format.NumberFormat.getCurrencyInstance({
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
        isEmpty(vValue) {
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
        isNotEmpty(vValue) {
            return !this.isEmpty(vValue);
        }
    };
});