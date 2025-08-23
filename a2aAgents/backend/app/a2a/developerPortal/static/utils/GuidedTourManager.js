sap.ui.define([
    "sap/ui/base/Object",
    "sap/m/Popover",
    "sap/m/Button",
    "sap/m/Text",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Label",
    "sap/ui/core/HTML"
], (BaseObject, Popover, Button, Text, VBox, HBox, Label, HTML) => {
    "use strict";
/* global, $ */

    /**
     * Guided Tour Manager for managing interactive tours in the application.
     * @namespace a2a.developer.portal.utils
     * @extends sap.ui.base.Object
     * @description A comprehensive tour management system for creating interactive, step-by-step 
     * walkthroughs of UI features. Supports highlighting elements, showing contextual popovers,
     * and tracking user progress through multi-step tours.
     * 
     * @example
     * // Basic usage - Create and start a simple tour
     * const tourManager = new GuidedTourManager();
     * const tourConfig = {
     *     id: "getting-started",
     *     steps: [
     *         {
     *             target: "createButton",
     *             title: "Create New Item",
     *             content: "Click here to create a new item",
     *             placement: "bottom"
     *         },
     *         {
     *             target: "searchField", 
     *             title: "Search",
     *             content: "Use this field to search for items",
     *             placement: "top"
     *         }
     *     ]
     * };
     * tourManager.startTour(tourConfig, this.getView(), function() {
     // eslint-disable-next-line no-console
     *     console.log("Tour completed!");
     * });
     * 
     * @example
     * // Advanced usage - Tour with validation and callbacks
     * const advancedTour = {
     *     id: "advanced-features",
     *     steps: [
     *         {
     *             target: "dataInput",
     *             title: "Enter Data",
     *             content: "Please enter valid data before proceeding",
     *             placement: "right",
     *             validation: function(control) {
     *                 return control.getValue().length > 0;
     *             },
     *             onBeforeShow: function() {
     // eslint-disable-next-line no-console
     *                 console.log("Preparing to show step");
     *             },
     *             onAfterShow: function() {
     // eslint-disable-next-line no-console
     *                 console.log("Step is now visible");
     *             }
     *         }
     *     ]
     * };
     * 
     * @example
     * // Resumable tour that saves progress
     * const resumableTour = {
     *     id: "comprehensive-tour",
     *     resumable: true,
     *     steps: [...] // Many steps
     * };
     * 
     * // Check for saved progress
     * const savedStep = localStorage.getItem("tour_" + resumableTour.id);
     * if (savedStep) {
     *     tourManager._iCurrentStep = parseInt(savedStep);
     * }
     * 
     * // Save progress on each step
     * const originalShowStep = tourManager._showStep.bind(tourManager);
     * tourManager._showStep = function(stepIndex) {
     *     originalShowStep(stepIndex);
     *     localStorage.setItem("tour_" + resumableTour.id, stepIndex);
     * };
     */
    return BaseObject.extend("a2a.developer.portal.utils.GuidedTourManager", {
        
        /**
         * Constructor for the GuidedTourManager.
         * @public
         * @description Initializes the tour manager with default state. Sets up internal properties
         * for tracking the current tour, step index, and UI elements.
         * 
         * @example
         * // Create a new instance
         * const tourManager = new GuidedTourManager();
         * 
         * // Access in controller
         * onInit: function() {
         *     this._tourManager = new GuidedTourManager();
         * }
         */
        constructor: function () {
            this._oCurrentTour = null;
            this._iCurrentStep = 0;
            this._oPopover = null;
            this._oOverlay = null;
            this._fnEndCallback = null;
        },

        /**
         * Start a guided tour with the given configuration.
         * @param {object} oTourConfig - Tour configuration object
         * @param {string} oTourConfig.id - Unique identifier for the tour
         * @param {Array<object>} oTourConfig.steps - Array of step configurations
         * @param {string} oTourConfig.steps[].target - ID of the target control to highlight
         * @param {string} oTourConfig.steps[].title - Title text for the step
         * @param {string} oTourConfig.steps[].content - Description content for the step
         * @param {string} [oTourConfig.steps[].placement="auto"] - Popover placement (top|bottom|left|right|auto)
         * @param {function} [oTourConfig.steps[].validation] - Optional validation function
         * @param {sap.ui.core.mvc.View} oView - The view context containing target controls
         * @param {function} [fnEndCallback] - Optional callback when tour ends
         * @public
         * @fires tourStarted
         * @fires tourEnded
         * 
         * @example
         * // Simple tour
         * tourManager.startTour({
         *     id: "quick-start",
         *     steps: [
         *         {
         *             target: "mainButton",
         *             title: "Welcome!",
         *             content: "Let's take a quick tour",
         *             placement: "bottom"
         *         }
         *     ]
         * }, this.getView());
         * 
         * @example
         * // Tour with completion tracking
         * tourManager.startTour(tourConfig, this.getView(), function() {
         *     // Save completion status
         *     const completedTours = JSON.parse(localStorage.getItem("completedTours") || "[]");
         *     completedTours.push(tourConfig.id);
         *     localStorage.setItem("completedTours", JSON.stringify(completedTours));
         *     
         *     // Show reward
         *     sap.m.MessageToast.show("Tour completed! You earned a badge!");
         * });
         * 
         * @example 
         * // Conditional tour based on user level
         * const userLevel = this.getUserLevel();
         * const tourSteps = userLevel === "beginner" ? 
         *     this.getBeginnerSteps() : this.getAdvancedSteps();
         * 
         * tourManager.startTour({
         *     id: `tour-${userLevel}`,
         *     steps: tourSteps
         * }, this.getView());
         */
        startTour: function (oTourConfig, oView, fnEndCallback) {
            this._oCurrentTour = oTourConfig;
            this._oView = oView;
            this._iCurrentStep = 0;
            this._fnEndCallback = fnEndCallback;
            
            // Create overlay
            this._createOverlay();
            
            // Show first step
            this._showStep(0);
        },

        /**
         * Move to the next step in the tour.
         * @public
         * @returns {boolean} True if moved to next step, false if tour ended
         * 
         * @example
         * // Manual navigation
         * if (tourManager.nextStep()) {
         // eslint-disable-next-line no-console
         *     console.log("Moved to step " + tourManager.getCurrentStep());
         * } else {
         // eslint-disable-next-line no-console
         *     console.log("Tour completed");
         * }
         * 
         * @example
         * // With validation
         * const canProceed = validateCurrentStep();
         * if (canProceed) {
         *     tourManager.nextStep();
         * } else {
         *     sap.m.MessageToast.show("Please complete the current step");
         * }
         */
        nextStep: function () {
            if (this._iCurrentStep < this._oCurrentTour.steps.length - 1) {
                this._iCurrentStep++;
                this._showStep(this._iCurrentStep);
            } else {
                this.endTour();
            }
        },

        /**
         * Move to the previous step in the tour.
         * @public
         */
        previousStep: function () {
            if (this._iCurrentStep > 0) {
                this._iCurrentStep--;
                this._showStep(this._iCurrentStep);
            }
        },

        /**
         * End the current tour.
         * @public
         * @fires tourEnded
         * @fires tourCancelled - If tour was not completed
         * 
         * @example
         * // End tour with confirmation
         * sap.m.MessageBox.confirm("End the tour?", {
         *     onClose: function(action) {
         *         if (action === sap.m.MessageBox.Action.OK) {
         *             tourManager.endTour();
         *         }
         *     }
         * });
         * 
         * @example
         * // Clean up and save progress before ending
         * const currentStep = tourManager.getCurrentStep();
         * const totalSteps = tourManager.getTotalSteps();
         * 
         * // Save incomplete tour progress
         * if (currentStep < totalSteps - 1) {
         *     localStorage.setItem("tourProgress", JSON.stringify({
         *         tourId: currentTourId,
         *         step: currentStep,
         *         date: new Date().toISOString()
         *     }));
         * }
         * 
         * tourManager.endTour();
         */
        endTour: function () {
            if (this._oPopover) {
                this._oPopover.close();
                this._oPopover.destroy();
                this._oPopover = null;
            }
            
            if (this._oOverlay) {
                this._oOverlay.destroy();
                this._oOverlay = null;
            }
            
            this._oCurrentTour = null;
            this._iCurrentStep = 0;
            
            if (this._fnEndCallback) {
                this._fnEndCallback();
            }
        },

        /**
         * Create the overlay for the guided tour.
         * @private
         */
        _createOverlay: function () {
            this._oOverlay = new HTML({
                content: '<div class="guidedTourOverlay"></div>'
            });
            
            // Add custom CSS for overlay
            const sOverlayStyle = `
                <style>
                    .guidedTourOverlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.5);
                        z-index: 1000;
                        pointer-events: none;
                    }
                    .guidedTourHighlight {
                        position: relative;
                        z-index: 1001;
                        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
                        border-radius: 4px;
                        animation: pulse 2s infinite;
                    }
                    @keyframes pulse {
                        0% { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); }
                        50% { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.3); }
                        100% { box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); }
                    }
                </style>
            `;
            
            jQuery("head").append(sOverlayStyle);
            this._oOverlay.placeAt("content");
        },

        /**
         * Show a specific step in the tour.
         * @param {number} iStepIndex - The step index (0-based)
         * @private
         * @fires stepShown
         * @throws {Error} If step index is out of bounds
         * 
         * @example
         * // Override to add custom behavior
         * const originalShowStep = tourManager._showStep.bind(tourManager);
         * tourManager._showStep = function(stepIndex) {
         *     // Custom pre-show logic
         // eslint-disable-next-line no-console
         *     console.log("Showing step", stepIndex);
         *     
         *     // Track analytics
         *     analytics.track("tour_step_viewed", {
         *         tour: this._oCurrentTour.id,
         *         step: stepIndex
         *     });
         *     
         *     // Call original method
         *     originalShowStep(stepIndex);
         *     
         *     // Custom post-show logic
         *     this._updateProgressBar(stepIndex);
         * };
         */
        _showStep: function (iStepIndex) {
            const oStep = this._oCurrentTour.steps[iStepIndex];
            const oTarget = this._oView.byId(oStep.target);
            
            if (!oTarget) {
                console.error("Target element not found:", oStep.target);
                this.nextStep();
                return;
            }
            
            // Highlight target element
            this._highlightElement(oTarget);
            
            // Create or update popover
            if (this._oPopover) {
                this._oPopover.close();
                this._oPopover.destroy();
            }
            
            this._oPopover = this._createStepPopover(oStep, iStepIndex);
            this._oPopover.openBy(oTarget);
        },

        /**
         * Highlight a target element.
         * @param {sap.ui.core.Control} oElement - The element to highlight
         * @private
         */
        _highlightElement: function (oElement) {
            // Remove previous highlights
            jQuery(".guidedTourHighlight").removeClass("guidedTourHighlight");
            
            // Add highlight to current element
            const oDomRef = oElement.getDomRef();
            if (oDomRef) {
                jQuery(oDomRef).addClass("guidedTourHighlight");
                
                // Scroll element into view
                oDomRef.scrollIntoView({
                    behavior: "smooth",
                    block: "center",
                    inline: "center"
                });
            }
        },

        /**
         * Create a popover for a tour step.
         * @param {object} oStep - The step configuration
         * @param {number} iStepIndex - The step index
         * @returns {sap.m.Popover} The created popover
         * @private
         */
        _createStepPopover: function (oStep, iStepIndex) {
            const that = this;
            const iTotalSteps = this._oCurrentTour.steps.length;
            
            // Create step content
            const oContent = new VBox({
                items: [
                    new Label({
                        text: oStep.title,
                        design: "Bold",
                        wrapping: true
                    }).addStyleClass("sapUiSmallMarginBottom"),
                    new Text({
                        text: oStep.content,
                        wrapping: true
                    }).addStyleClass("sapUiSmallMarginBottom"),
                    new HBox({
                        justifyContent: "SpaceBetween",
                        items: [
                            new Text({
                                text: `Step ${iStepIndex + 1} of ${iTotalSteps}`
                            }),
                            new HBox({
                                items: [
                                    new Button({
                                        text: "Skip Tour",
                                        type: "Transparent",
                                        press: function () {
                                            that.endTour();
                                        }
                                    }),
                                    new Button({
                                        text: "Previous",
                                        enabled: iStepIndex > 0,
                                        press: function () {
                                            that.previousStep();
                                        }
                                    }).addStyleClass("sapUiTinyMarginBegin"),
                                    new Button({
                                        text: iStepIndex === iTotalSteps - 1 ? "Finish" : "Next",
                                        type: "Emphasized",
                                        press: function () {
                                            that.nextStep();
                                        }
                                    }).addStyleClass("sapUiTinyMarginBegin")
                                ]
                            })
                        ]
                    })
                ]
            }).addStyleClass("sapUiContentPadding");
            
            // Create popover
            const oPopover = new Popover({
                title: `${this._oCurrentTour.id} - Tour`,
                content: oContent,
                placement: this._determinePlacement(oStep.placement),
                showArrow: true,
                contentWidth: "350px",
                afterClose: function () {
                    // Remove highlight when popover closes
                    jQuery(".guidedTourHighlight").removeClass("guidedTourHighlight");
                }
            });
            
            // Add custom class for styling
            oPopover.addStyleClass("guidedTourPopover");
            
            return oPopover;
        },

        /**
         * Determine the best placement for the popover.
         * @param {string} sPreferredPlacement - The preferred placement
         * @returns {sap.m.PlacementType} The placement type
         * @private
         */
        _determinePlacement: function (sPreferredPlacement) {
            const oPlacementMap = {
                "top": sap.m.PlacementType.Top,
                "bottom": sap.m.PlacementType.Bottom,
                "left": sap.m.PlacementType.Left,
                "right": sap.m.PlacementType.Right,
                "auto": sap.m.PlacementType.Auto
            };
            
            return oPlacementMap[sPreferredPlacement] || sap.m.PlacementType.Auto;
        },

        /**
         * Get the current step index.
         * @returns {number} The current step index (0-based)
         * @public
         * 
         * @example
         * // Display progress
         * const current = tourManager.getCurrentStep();
         * const total = tourManager.getTotalSteps();
         * this.byId("progressText").setText(`Step ${current + 1} of ${total}`);
         * 
         * @example
         * // Enable/disable navigation buttons
         * const currentStep = tourManager.getCurrentStep();
         * this.byId("prevButton").setEnabled(currentStep > 0);
         * this.byId("nextButton").setEnabled(currentStep < tourManager.getTotalSteps() - 1);
         */
        getCurrentStep: function () {
            return this._iCurrentStep;
        },

        /**
         * Get the total number of steps.
         * @returns {number} The total number of steps
         * @public
         */
        getTotalSteps: function () {
            return this._oCurrentTour ? this._oCurrentTour.steps.length : 0;
        },

        /**
         * Check if a tour is currently active.
         * @returns {boolean} True if a tour is active
         * @public
         * 
         * @example
         * // Prevent navigation during tour
         * onNavigation: function(oEvent) {
         *     if (this._tourManager.isActive()) {
         *         oEvent.preventDefault();
         *         sap.m.MessageToast.show("Please complete the tour first");
         *     }
         * }
         * 
         * @example
         * // Show/hide tour indicator
         * setInterval(function() {
         *     const tourIndicator = this.byId("tourActiveIndicator");
         *     tourIndicator.setVisible(this._tourManager.isActive());
         *     
         *     if (this._tourManager.isActive()) {
         *         const step = this._tourManager.getCurrentStep() + 1;
         *         const total = this._tourManager.getTotalSteps();
         *         tourIndicator.setText(`Tour in progress: ${step}/${total}`);
         *     }
         * }.bind(this), 1000);
         */
        isActive: function () {
            return this._oCurrentTour !== null;
        }
    });
});