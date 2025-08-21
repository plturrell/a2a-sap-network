sap.ui.define([], function () {
    "use strict";

    return {
        /**
         * Format date to localized string
         * @param {Date|string|number} date - The date to format
         * @returns {string} Formatted date string
         */
        formatDate: function (date) {
            if (!date) {
                return "";
            }

            var oDate = date instanceof Date ? date : new Date(date);
            
            if (isNaN(oDate.getTime())) {
                return "";
            }

            // Use UI5 date formatter
            var oDateFormat = sap.ui.core.format.DateFormat.getDateTimeInstance({
                style: "medium"
            });

            return oDateFormat.format(oDate);
        },

        /**
         * Format short date
         * @param {Date|string|number} date - The date to format
         * @returns {string} Formatted date string (short format)
         */
        formatShortDate: function (date) {
            if (!date) {
                return "";
            }

            var oDate = date instanceof Date ? date : new Date(date);
            
            if (isNaN(oDate.getTime())) {
                return "";
            }

            var oDateFormat = sap.ui.core.format.DateFormat.getDateInstance({
                style: "short"
            });

            return oDateFormat.format(oDate);
        },

        /**
         * Format ethereum address for display
         * @param {string} address - The ethereum address
         * @returns {string} Shortened address
         */
        formatAddress: function (address) {
            if (!address || address.length < 10) {
                return address;
            }
            return address.substr(0, 6) + "..." + address.substr(-4);
        },

        /**
         * Format percentage
         * @param {number} value - The value to format as percentage
         * @returns {string} Formatted percentage
         */
        formatPercentage: function (value) {
            if (value === null || value === undefined || isNaN(value)) {
                return "0%";
            }
            return Math.round(value) + "%";
        },

        /**
         * Format rating state
         * @param {number} rating - The rating value
         * @returns {string} State for ObjectStatus
         */
        formatRatingState: function (rating) {
            if (rating >= 4) {
                return "Success";
            } else if (rating >= 3) {
                return "Warning";
            } else {
                return "Error";
            }
        },

        /**
         * Format status state
         * @param {string} status - The status value
         * @returns {string} State for ObjectStatus
         */
        formatStatusState: function (status) {
            switch (status) {
                case "Active":
                case "Validated":
                case "Success":
                    return "Success";
                case "Pending":
                case "Warning":
                    return "Warning";
                case "Inactive":
                case "Rejected":
                case "Failed":
                case "Error":
                    return "Error";
                default:
                    return "None";
            }
        },

        /**
         * Format time duration
         * @param {number} milliseconds - Duration in milliseconds
         * @returns {string} Formatted duration
         */
        formatDuration: function (milliseconds) {
            if (!milliseconds || isNaN(milliseconds)) {
                return "0 ms";
            }

            if (milliseconds < 1000) {
                return milliseconds + " ms";
            } else if (milliseconds < 60000) {
                return (milliseconds / 1000).toFixed(1) + " s";
            } else {
                return (milliseconds / 60000).toFixed(1) + " min";
            }
        }
    };
});