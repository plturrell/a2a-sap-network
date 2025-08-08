import UIComponent from "sap/ui/core/UIComponent";
import { support } from "sap/ui/Device";
import JSONModel from "sap/ui/model/json/JSONModel";
import models from "./model/models";

/**
 * @namespace com.sap.a2a.portal
 */
export default class Component extends UIComponent {
    public static metadata = {
        manifest: "json",
        interfaces: ["sap.ui.core.IAsyncContentCreation"]
    };

    private contentDensityClass: string;

    /**
     * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
     * @public
     * @override
     */
    public init(): void {
        // Call the base component's init function
        super.init();

        // Enable routing
        this.getRouter().initialize();

        // Set the device model
        this.setModel(models.createDeviceModel(), "device");

        // Initialize error handler
        this._initializeErrorHandler();

        // Initialize service worker for offline support
        this._initializeServiceWorker();

        // Set up CSRF token handling
        this._setupCSRFToken();

        // Initialize telemetry
        this._initializeTelemetry();
    }

    /**
     * Initialize error handler for better error management
     */
    private _initializeErrorHandler(): void {
        // Global error handler
        window.addEventListener("unhandledrejection", (event) => {
            console.error("Unhandled promise rejection:", event.reason);
            // Send to monitoring service
            this._logError(event.reason);
        });

        // UI5 error handler
        sap.ui.getCore().attachValidationError((oEvent) => {
            const oElement = oEvent.getParameter("element");
            const sMessage = oEvent.getParameter("message");
            console.error("Validation error:", sMessage, oElement);
        });
    }

    /**
     * Initialize service worker for PWA support
     */
    private _initializeServiceWorker(): void {
        if ("serviceWorker" in navigator && !support.edge) {
            navigator.serviceWorker
                .register("/service-worker.js")
                .then((registration) => {
                    console.log("Service Worker registered:", registration);
                })
                .catch((error) => {
                    console.error("Service Worker registration failed:", error);
                });
        }
    }

    /**
     * Set up CSRF token handling for secure API calls
     */
    private _setupCSRFToken(): void {
        // Fetch CSRF token on app initialization
        fetch("/api/v1/csrf-token", {
            method: "GET",
            credentials: "same-origin"
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.token) {
                    // Store CSRF token for subsequent requests
                    this._csrfToken = data.token;
                    
                    // Add to default headers
                    jQuery.ajaxSetup({
                        beforeSend: (xhr) => {
                            xhr.setRequestHeader("X-CSRF-Token", this._csrfToken);
                        }
                    });
                }
            })
            .catch((error) => {
                console.error("Failed to fetch CSRF token:", error);
            });
    }

    /**
     * Initialize telemetry for monitoring
     */
    private _initializeTelemetry(): void {
        // Initialize OpenTelemetry if available
        if (window.opentelemetry) {
            const { trace, context } = window.opentelemetry;
            const tracer = trace.getTracer("a2a-portal-ui", "1.0.0");

            // Instrument router
            const oRouter = this.getRouter();
            oRouter.attachRouteMatched((oEvent) => {
                const span = tracer.startSpan("route_navigation");
                span.setAttributes({
                    "route.name": oEvent.getParameter("name"),
                    "route.pattern": oEvent.getParameter("config").pattern
                });
                span.end();
            });
        }

        // SAP UI5 Performance API
        if (sap.ui.performance) {
            sap.ui.performance.setActive(true);
            
            // Log performance metrics periodically
            setInterval(() => {
                const measurements = sap.ui.performance.getInteractionMeasurements();
                if (measurements.length > 0) {
                    this._logPerformanceMetrics(measurements);
                }
            }, 30000); // Every 30 seconds
        }
    }

    /**
     * Log error to monitoring service
     */
    private _logError(error: any): void {
        const errorData = {
            message: error.message || error,
            stack: error.stack,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        // Send to monitoring endpoint
        fetch("/api/v1/monitoring/errors", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this._getAuthToken()}`
            },
            body: JSON.stringify(errorData)
        }).catch((err) => {
            console.error("Failed to log error:", err);
        });
    }

    /**
     * Log performance metrics
     */
    private _logPerformanceMetrics(measurements: any[]): void {
        const metrics = measurements.map((m) => ({
            interaction: m.event,
            duration: m.duration,
            timestamp: m.start,
            details: m.requests
        }));

        // Send to monitoring endpoint
        fetch("/api/v1/monitoring/performance", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this._getAuthToken()}`
            },
            body: JSON.stringify({ metrics })
        }).catch((err) => {
            console.error("Failed to log performance metrics:", err);
        });
    }

    /**
     * Get auth token from local storage
     */
    private _getAuthToken(): string {
        return localStorage.getItem("authToken") || "";
    }

    /**
     * This method can be called to determine whether the sapUiSizeCompact or sapUiSizeCozy
     * design mode class should be set, which influences the size appearance of some controls.
     * @public
     * @return {string} css class, either 'sapUiSizeCompact' or 'sapUiSizeCozy' - or an empty string if no css class should be set
     */
    public getContentDensityClass(): string {
        if (this.contentDensityClass === undefined) {
            // Check whether FLP has already set the content density class
            const oShell = sap.ushell?.Container?.getService("ShellUIService");
            if (oShell) {
                this.contentDensityClass = "";
            } else if (support.touch) {
                // Apply "cozy" mode if touch support is detected
                this.contentDensityClass = "sapUiSizeCozy";
            } else {
                // Apply "compact" mode for desktop
                this.contentDensityClass = "sapUiSizeCompact";
            }
        }
        return this.contentDensityClass;
    }

    /**
     * Convenience method for getting the router
     */
    public getRouter(): sap.ui.core.routing.Router {
        return UIComponent.prototype.getRouter.call(this);
    }

    /**
     * Convenience method for getting the event bus
     */
    public getEventBus(): sap.ui.core.EventBus {
        return sap.ui.getCore().getEventBus();
    }

    private _csrfToken: string;
}