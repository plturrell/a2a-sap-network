/*!
 * SAP UI5 Custom Control: Consistent Loading Indicator
 * Provides standardized loading states across all views
 */

sap.ui.define([
    "sap/ui/core/Control",
    "sap/ui/core/HTML",
    "sap/m/Text",
    "sap/m/VBox",
    "sap/ui/core/Icon"
], function(
    Control,
    HTML,
    Text,
    VBox,
    Icon
) {
    "use strict";

    /**
     * Loading Indicator Control
     * Standardized loading states with SAP Fiori design compliance
     * @namespace com.sap.a2a.portal.control
     */
    return Control.extend("com.sap.a2a.portal.control.LoadingIndicator", {
        metadata: {
            properties: {
                /**
                 * Loading state type
                 */
                type: {
                    type: "string",
                    defaultValue: "spinner",
                    bindable: true
                },
                
                /**
                 * Loading message text
                 */
                message: {
                    type: "string",
                    defaultValue: "Loading...",
                    bindable: true
                },
                
                /**
                 * Size of the loading indicator
                 */
                size: {
                    type: "string",
                    defaultValue: "medium",
                    bindable: true
                },
                
                /**
                 * Show loading text
                 */
                showText: {
                    type: "boolean",
                    defaultValue: true,
                    bindable: true
                },
                
                /**
                 * Overlay the entire parent container
                 */
                overlay: {
                    type: "boolean",
                    defaultValue: false,
                    bindable: true
                },
                
                /**
                 * Progress percentage for progress type
                 */
                progress: {
                    type: "int",
                    defaultValue: 0,
                    bindable: true
                },
                
                /**
                 * Animation speed
                 */
                speed: {
                    type: "string",
                    defaultValue: "normal",
                    bindable: true
                }
            },
            
            aggregations: {
                /**
                 * Internal content aggregation
                 */
                _content: {
                    type: "sap.ui.core.Control",
                    multiple: false,
                    visibility: "hidden"
                }
            },
            
            events: {
                /**
                 * Fired when loading animation completes a cycle
                 */
                animationCycle: {
                    parameters: {
                        cycle: { type: "int" }
                    }
                }
            }
        },

        init: function() {
            this._animationCycle = 0;
            this._createContent();
        },

        _createContent: function() {
            const oContent = new VBox({
                alignItems: "Center",
                justifyContent: "Center",
                class: "sapUiLoadingIndicator"
            });

            this._updateContent(oContent);
            this.setAggregation("_content", oContent);
        },

        _updateContent: function(oContent) {
            oContent.removeAllItems();

            const sType = this.getType();
            const sSize = this.getSize();
            const bShowText = this.getShowText();
            const sMessage = this.getMessage();

            // Create loading element based on type
            let oLoadingElement;

            switch (sType) {
                case "spinner":
                    oLoadingElement = this._createSpinner(sSize);
                    break;
                case "dots":
                    oLoadingElement = this._createDots(sSize);
                    break;
                case "pulse":
                    oLoadingElement = this._createPulse(sSize);
                    break;
                case "skeleton":
                    oLoadingElement = this._createSkeleton(sSize);
                    break;
                case "progress":
                    oLoadingElement = this._createProgress(sSize);
                    break;
                case "shimmer":
                    oLoadingElement = this._createShimmer(sSize);
                    break;
                default:
                    oLoadingElement = this._createSpinner(sSize);
            }

            oContent.addItem(oLoadingElement);

            // Add text if enabled
            if (bShowText && sMessage) {
                const oText = new Text({
                    text: sMessage,
                    class: "sapUiLoadingText sapUiMediumMarginTop"
                });
                oContent.addItem(oText);
            }
        },

        _createSpinner: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            
            return new HTML({
                content: `
                    <div class="sapUiSpinner ${sSizeClass}" role="progressbar" aria-label="Loading">
                        <svg viewBox="0 0 50 50">
                            <circle 
                                cx="25" 
                                cy="25" 
                                r="20" 
                                fill="none" 
                                stroke="var(--sapBrandColor)" 
                                stroke-width="4"
                                stroke-linecap="round"
                                stroke-dasharray="31.416"
                                stroke-dashoffset="31.416"
                                class="sapUiSpinnerCircle">
                            </circle>
                        </svg>
                    </div>
                `
            });
        },

        _createDots: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            
            return new HTML({
                content: `
                    <div class="sapUiLoadingDots ${sSizeClass}" role="progressbar" aria-label="Loading">
                        <div class="sapUiLoadingDot"></div>
                        <div class="sapUiLoadingDot"></div>
                        <div class="sapUiLoadingDot"></div>
                    </div>
                `
            });
        },

        _createPulse: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            
            return new HTML({
                content: `
                    <div class="sapUiLoadingPulse ${sSizeClass}" role="progressbar" aria-label="Loading">
                        <div class="sapUiPulseRing"></div>
                        <div class="sapUiPulseRing"></div>
                        <div class="sapUiPulseRing"></div>
                    </div>
                `
            });
        },

        _createSkeleton: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            
            return new HTML({
                content: `
                    <div class="sapUiLoadingSkeleton ${sSizeClass}" role="progressbar" aria-label="Loading content">
                        <div class="sapUiSkeletonLine sapUiSkeletonLineTitle"></div>
                        <div class="sapUiSkeletonLine sapUiSkeletonLineSubtitle"></div>
                        <div class="sapUiSkeletonLine sapUiSkeletonLineText"></div>
                        <div class="sapUiSkeletonLine sapUiSkeletonLineTextShort"></div>
                    </div>
                `
            });
        },

        _createProgress: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            const iProgress = this.getProgress();
            
            return new HTML({
                content: `
                    <div class="sapUiLoadingProgress ${sSizeClass}" role="progressbar" aria-label="Loading progress" aria-valuenow="${iProgress}" aria-valuemin="0" aria-valuemax="100">
                        <div class="sapUiProgressTrack">
                            <div class="sapUiProgressFill" style="width: ${iProgress}%"></div>
                        </div>
                        <div class="sapUiProgressText">${iProgress}%</div>
                    </div>
                `
            });
        },

        _createShimmer: function(sSize) {
            const sSizeClass = this._getSizeClass(sSize);
            
            return new HTML({
                content: `
                    <div class="sapUiLoadingShimmer ${sSizeClass}" role="progressbar" aria-label="Loading">
                        <div class="sapUiShimmerBlock sapUiShimmerBlockLarge"></div>
                        <div class="sapUiShimmerBlock sapUiShimmerBlockMedium"></div>
                        <div class="sapUiShimmerBlock sapUiShimmerBlockSmall"></div>
                    </div>
                `
            });
        },

        _getSizeClass: function(sSize) {
            switch (sSize) {
                case "small":
                    return "sapUiLoadingSmall";
                case "large":
                    return "sapUiLoadingLarge";
                case "medium":
                default:
                    return "sapUiLoadingMedium";
            }
        },

        _getSpeedClass: function(sSpeed) {
            switch (sSpeed) {
                case "slow":
                    return "sapUiLoadingSlow";
                case "fast":
                    return "sapUiLoadingFast";
                case "normal":
                default:
                    return "sapUiLoadingNormal";
            }
        },

        onAfterRendering: function() {
            this._startAnimationTracking();
        },

        _startAnimationTracking: function() {
            // Clean up existing listener first
            this._stopAnimationTracking();
            
            const oDomRef = this.getDomRef();
            if (!oDomRef) return;

            // Track animation cycles for spinner
            const oSpinnerElement = oDomRef.querySelector('.sapUiSpinnerCircle');
            if (oSpinnerElement) {
                this._animationHandler = () => {
                    this._animationCycle++;
                    this.fireAnimationCycle({
                        cycle: this._animationCycle
                    });
                };
                
                oSpinnerElement.addEventListener('animationiteration', this._animationHandler);
                this._currentSpinnerElement = oSpinnerElement;
            }
        },
        
        _stopAnimationTracking: function() {
            if (this._currentSpinnerElement && this._animationHandler) {
                this._currentSpinnerElement.removeEventListener('animationiteration', this._animationHandler);
                this._currentSpinnerElement = null;
                this._animationHandler = null;
            }
        },
        
        exit: function() {
            this._stopAnimationTracking();
            Control.prototype.exit.apply(this, arguments);
        },

        renderer: {
            apiVersion: 2,
            
            render: function(oRm, oControl) {
                oRm.openStart("div", oControl);
                oRm.class("sapUiLoadingIndicator");
                oRm.class(oControl._getSpeedClass(oControl.getSpeed()));
                
                if (oControl.getOverlay()) {
                    oRm.class("sapUiLoadingOverlay");
                }
                
                // Accessibility attributes
                oRm.attr("role", "status");
                oRm.attr("aria-live", "polite");
                oRm.attr("aria-label", "Loading content");
                
                // Size and type specific classes
                oRm.class(oControl._getSizeClass(oControl.getSize()));
                
                oRm.openEnd();
                
                oRm.renderControl(oControl.getAggregation("_content"));
                
                oRm.close("div");
            }
        },

        // Property setters that trigger content update
        setType: function(sValue) {
            this.setProperty("type", sValue, true);
            this._updateContentIfRendered();
            return this;
        },

        setMessage: function(sValue) {
            this.setProperty("message", sValue, true);
            this._updateContentIfRendered();
            return this;
        },

        setSize: function(sValue) {
            this.setProperty("size", sValue, true);
            this._updateContentIfRendered();
            return this;
        },

        setShowText: function(bValue) {
            this.setProperty("showText", bValue, true);
            this._updateContentIfRendered();
            return this;
        },

        setProgress: function(iValue) {
            this.setProperty("progress", iValue, true);
            this._updateProgressIfRendered(iValue);
            return this;
        },

        _updateContentIfRendered: function() {
            if (this.getDomRef()) {
                const oContent = this.getAggregation("_content");
                if (oContent) {
                    this._updateContent(oContent);
                }
            }
        },

        _updateProgressIfRendered: function(iProgress) {
            if (this.getDomRef() && this.getType() === "progress") {
                const oProgressFill = this.getDomRef().querySelector('.sapUiProgressFill');
                const oProgressText = this.getDomRef().querySelector('.sapUiProgressText');
                
                if (oProgressFill) {
                    oProgressFill.style.width = iProgress + '%';
                }
                if (oProgressText) {
                    oProgressText.textContent = iProgress + '%';
                }
            }
        },

        // Static helper methods
        showGlobalLoading: function(sMessage) {
            sap.ui.core.BusyIndicator.show(0, sMessage);
        },

        hideGlobalLoading: function() {
            sap.ui.core.BusyIndicator.hide();
        }
    });
});