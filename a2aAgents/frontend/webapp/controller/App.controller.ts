import Controller from "sap/ui/core/mvc/Controller";
import UIComponent from "sap/ui/core/UIComponent";
import { Route$MatchedEvent } from "sap/ui/core/routing/Route";
import MessageToast from "sap/m/MessageToast";
import JSONModel from "sap/ui/model/json/JSONModel";
import ResourceModel from "sap/ui/model/resource/ResourceModel";
import Device from "sap/ui/Device";
import SplitApp from "sap/m/SplitApp";
import { SearchField$SearchEvent, SearchField$SuggestEvent } from "sap/m/SearchField";
import { NavigationList$ItemSelectEvent } from "sap/tnt/NavigationList";
import { ComboBox$SelectionChangeEvent } from "sap/m/ComboBox";
import Log from "sap/base/Log";

/**
 * @namespace com.sap.a2a.portal.controller
 */
export default class App extends Controller {
    private _bExpanded: boolean = true;

    public onInit(): void {
        // Initialize models
        this._initializeModels();

        // Subscribe to route changes
        this.getOwnerComponent().getRouter().attachRouteMatched(this.onRouteMatched, this);

        // Initialize side navigation
        this._initializeSideNavigation();

        // Check authentication
        this._checkAuthentication();

        // Initialize keyboard shortcuts
        this._initializeKeyboardShortcuts();
    }

    private _initializeModels(): void {
        // User model
        const oUserModel = new JSONModel({
            name: "User Name",
            initials: "UN",
            role: "Developer",
            email: "user@sap.com"
        });
        this.getView().setModel(oUserModel, "user");

        // Notification model
        const oNotificationModel = new JSONModel({
            count: 3,
            items: []
        });
        this.getView().setModel(oNotificationModel, "notification");

        // App state model
        const oAppModel = new JSONModel({
            busy: false,
            delay: 0,
            sideExpanded: !Device.system.phone
        });
        this.getView().setModel(oAppModel, "app");

        // Workspace model
        const oWorkspaceModel = new JSONModel({
            current: "default",
            available: [
                { id: "default", name: "Default Workspace" },
                { id: "personal", name: "Personal Workspace" },
                { id: "team", name: "Team Workspace" },
                { id: "enterprise", name: "Enterprise Workspace" }
            ]
        });
        this.getView().setModel(oWorkspaceModel, "workspace");

        // Settings model
        const oSettingsModel = new JSONModel({
            theme: localStorage.getItem("userTheme") || "sap_horizon",
            language: localStorage.getItem("userLanguage") || "en",
            timezone: localStorage.getItem("userTimezone") || "UTC",
            autoSave: localStorage.getItem("userAutoSave") === "true",
            notifications: {
                email: localStorage.getItem("notifyEmail") === "true",
                push: localStorage.getItem("notifyPush") === "true", 
                deployment: localStorage.getItem("notifyDeployment") === "true",
                security: localStorage.getItem("notifySecurity") === "true",
                agents: localStorage.getItem("notifyAgents") === "true"
            },
            developer: {
                debugMode: localStorage.getItem("devDebugMode") === "true",
                consoleLogging: localStorage.getItem("devConsoleLogging") === "true",
                performanceMonitoring: localStorage.getItem("devPerformanceMonitoring") === "true",
                apiTimeout: parseInt(localStorage.getItem("devApiTimeout") || "30")
            }
        });
        this.getView().setModel(oSettingsModel, "settings");

        // Apply theme on initialization
        this._applyTheme(oSettingsModel.getProperty("/theme"));
    }

    private _initializeSideNavigation(): void {
        // Set default selection
        const oSideNavigation = this.byId("sideNavigation");
        if (oSideNavigation) {
            oSideNavigation.setSelectedKey("projects");
        }
    }

    private async _checkAuthentication(): Promise<void> {
        const authToken = this._getAuthToken();
        
        if (!authToken) {
            Log.warn("No authentication token found, redirecting to login", "com.sap.a2a.portal.controller.App");
            this._handleAuthenticationFailure("No authentication token");
            return;
        }

        try {
            const response = await fetch("/api/v1/users/me", {
                headers: {
                    "Authorization": `Bearer ${authToken}`,
                    "Content-Type": "application/json"
                },
                timeout: 10000 // 10 second timeout
            });

            if (response.ok) {
                const userData = await response.json();
                this._validateUserData(userData);
                this.getModel("user").setData(userData);
                Log.info("Authentication check successful", "com.sap.a2a.portal.controller.App");
            } else {
                await this._handleAuthenticationError(response);
            }
        } catch (error) {
            Log.error("Authentication check failed", error.message, "com.sap.a2a.portal.controller.App");
            this._handleNetworkError(error);
        }
    }

    private _validateUserData(userData: any): void {
        const requiredFields = ["name", "email", "role"];
        const missingFields = requiredFields.filter(field => !userData[field]);
        
        if (missingFields.length > 0) {
            Log.warn(`User data missing required fields: ${missingFields.join(", ")}`, "com.sap.a2a.portal.controller.App");
            // Set default values for missing fields
            userData.name = userData.name || "Unknown User";
            userData.email = userData.email || "";
            userData.role = userData.role || "User";
        }
    }

    private async _handleAuthenticationError(response: Response): Promise<void> {
        let errorMessage = "Authentication failed";
        
        try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorMessage;
        } catch {
            // Unable to parse error response
        }

        Log.warn(`Authentication failed with status ${response.status}: ${errorMessage}`, "com.sap.a2a.portal.controller.App");

        switch (response.status) {
            case 401:
                this._handleAuthenticationFailure("Invalid or expired token");
                break;
            case 403:
                MessageToast.show("Access denied. Please contact your administrator.");
                break;
            case 429:
                MessageToast.show("Too many requests. Please wait a moment and try again.");
                break;
            case 503:
                MessageToast.show("Authentication service temporarily unavailable. Please try again later.");
                break;
            default:
                MessageToast.show("Authentication failed. Please try logging in again.");
                this._handleAuthenticationFailure(errorMessage);
        }
    }

    private _handleAuthenticationFailure(reason: string): void {
        Log.warn(`Redirecting to login due to: ${reason}`, "com.sap.a2a.portal.controller.App");
        
        // Clear any stored authentication data
        localStorage.removeItem("authToken");
        localStorage.removeItem("refreshToken");
        
        // Show user-friendly message
        MessageToast.show("Your session has expired. Redirecting to login...");
        
        // Redirect after a short delay to allow user to see the message
        setTimeout(() => {
            window.location.href = "/login";
        }, 2000);
    }

    private _handleNetworkError(error: Error): void {
        if (error.name === "TimeoutError" || error.message.includes("timeout")) {
            MessageToast.show("Request timed out. Please check your connection and try again.");
        } else if (error.name === "NetworkError" || error.message.includes("fetch")) {
            MessageToast.show("Unable to connect to the server. Please check your internet connection.");
        } else {
            MessageToast.show("An unexpected error occurred. Please try refreshing the page.");
        }
        
        // Set user model to safe defaults
        this.getModel("user").setData({
            name: "Offline User",
            initials: "OU",
            role: "User",
            email: ""
        });
    }

    private _getAuthToken(): string {
        return this._getSecureToken("authToken");
    }

    private _getSecureToken(key: string): string {
        try {
            const token = sessionStorage.getItem(key) || localStorage.getItem(key);
            
            if (!token) {
                return "";
            }

            // Basic token validation - check format and expiration
            if (!this._isValidTokenFormat(token)) {
                Log.warn("Invalid token format detected, clearing token", "com.sap.a2a.portal.controller.App");
                this._clearAuthTokens();
                return "";
            }

            // Check if token is expired
            if (this._isTokenExpired(token)) {
                Log.warn("Expired token detected, clearing token", "com.sap.a2a.portal.controller.App");
                this._clearAuthTokens();
                return "";
            }

            return token;
        } catch (error) {
            Log.error("Error retrieving secure token", error.message, "com.sap.a2a.portal.controller.App");
            this._clearAuthTokens();
            return "";
        }
    }

    private _isValidTokenFormat(token: string): boolean {
        if (!token || typeof token !== "string") {
            return false;
        }

        // Basic JWT format validation (header.payload.signature)
        const parts = token.split(".");
        if (parts.length !== 3) {
            return false;
        }

        // Check that each part is base64 encoded
        try {
            parts.forEach(part => {
                if (!part || part.length === 0) {
                    throw new Error("Empty token part");
                }
                // Basic base64 character validation
                if (!/^[A-Za-z0-9+/=-]*$/.test(part)) {
                    throw new Error("Invalid base64 characters");
                }
            });
            return true;
        } catch {
            return false;
        }
    }

    private _isTokenExpired(token: string): boolean {
        try {
            const payload = JSON.parse(atob(token.split(".")[1]));
            const expirationTime = payload.exp;
            
            if (!expirationTime) {
                // No expiration time, consider it expired for security
                return true;
            }

            const currentTime = Math.floor(Date.now() / 1000);
            const buffer = 60; // 1 minute buffer before expiration
            
            return currentTime >= (expirationTime - buffer);
        } catch {
            // If we can't parse the token, consider it expired
            return true;
        }
    }

    private _clearAuthTokens(): void {
        try {
            localStorage.removeItem("authToken");
            localStorage.removeItem("refreshToken");
            sessionStorage.removeItem("authToken");
            sessionStorage.removeItem("refreshToken");
            Log.info("Authentication tokens cleared", "com.sap.a2a.portal.controller.App");
        } catch (error) {
            Log.error("Error clearing authentication tokens", error.message, "com.sap.a2a.portal.controller.App");
        }
    }

    public onRouteMatched(oEvent: Route$MatchedEvent): void {
        const sRouteName = oEvent.getParameter("name");
        const oSideNavigation = this.byId("sideNavigation");

        // Update navigation selection based on route
        if (oSideNavigation && sRouteName) {
            const mRouteToNav: Record<string, string> = {
                "ProjectsList": "projects",
                "ProjectDetail": "projects",
                "AgentsList": "agents",
                "AgentBuilder": "agentBuilder",
                "A2ANetwork": "network"
            };

            const sNavKey = mRouteToNav[sRouteName];
            if (sNavKey) {
                oSideNavigation.setSelectedKey(sNavKey);
            }
        }
    }

    public onNavigationItemSelect(oEvent: NavigationList$ItemSelectEvent): void {
        const sKey = oEvent.getParameter("item").getKey();
        const oRouter = this.getOwnerComponent().getRouter();

        // Route mapping
        const mNavToRoute: Record<string, string> = {
            "projects": "ProjectsList",
            "myProjects": "ProjectsList",
            "allProjects": "ProjectsList",
            "templates": "ProjectsList",
            "agents": "AgentsList",
            "agent0": "AgentDetail",
            "agent1": "AgentDetail",
            "agent2": "AgentDetail",
            "agent3": "AgentDetail",
            "agent4": "AgentDetail",
            "agent5": "AgentDetail",
            "agentManager": "AgentDetail",
            "dataManager": "AgentDetail",
            "catalogManager": "AgentDetail",
            "agentBuilder": "AgentBuilder",
            "workflowDesigner": "WorkflowDesigner",
            "testing": "Testing",
            "deployment": "Deployment",
            "dashboard": "Dashboard",
            "metrics": "Metrics",
            "logs": "Logs",
            "alerts": "Alerts",
            "network": "A2ANetwork",
            "ordRegistry": "ORDRegistry"
        };

        const sRoute = mNavToRoute[sKey];
        if (sRoute) {
            // Handle agent routes with parameters
            if (sKey.startsWith("agent") && sKey !== "agentBuilder") {
                oRouter.navTo(sRoute, { agentId: sKey });
            } else {
                oRouter.navTo(sRoute);
            }
        }
    }

    public onNavBack(): void {
        const oHistory = window.history;
        if (oHistory.length > 1) {
            oHistory.go(-1);
        } else {
            this.getOwnerComponent().getRouter().navTo("ProjectsList");
        }
    }

    public async onNotificationPress(): Promise<void> {
        // Load notifications
        try {
            const response = await fetch("/api/v1/notifications", {
                headers: {
                    "Authorization": `Bearer ${this._getAuthToken()}`
                }
            });

            if (response.ok) {
                const notifications = await response.json();
                this.getModel("notification").setProperty("/items", notifications);
                
                // Open notification popover
                if (!this._oNotificationPopover) {
                    this._oNotificationPopover = await this.loadFragment({
                        name: "com.sap.a2a.portal.view.NotificationPopover"
                    });
                }
                this._oNotificationPopover.openBy(oEvent.getSource());
            }
        } catch (error) {
            MessageToast.show(this.getResourceBundle().getText("notificationLoadError"));
        }
    }

    public onProductSwitcherPress(): void {
        MessageToast.show("Product Switcher - Coming Soon");
    }

    public onUserSettingsPress(): void {
        this.getOwnerComponent().getRouter().navTo("UserSettings");
    }

    public onSystemSettingsPress(): void {
        this.getOwnerComponent().getRouter().navTo("SystemSettings");
    }

    public onHelpPress(): void {
        // Open help documentation
        window.open("https://help.sap.com/a2a-portal", "_blank");
    }

    public async onLogoutPress(): Promise<void> {
        try {
            await fetch("/api/v1/auth/logout", {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${this._getAuthToken()}`
                }
            });

            // Clear local storage
            localStorage.removeItem("authToken");
            
            // Redirect to login
            window.location.href = "/login";
        } catch (error) {
            Log.error("Logout failed", error.message, "com.sap.a2a.portal.controller.App");
        }
    }

    public onSearch(oEvent: SearchField$SearchEvent): void {
        const sQuery = oEvent.getParameter("query");
        
        // Trigger global search
        this.getOwnerComponent().getEventBus().publish("app", "globalSearch", {
            query: sQuery
        });
    }

    public async onSearchSuggest(oEvent: SearchField$SuggestEvent): Promise<void> {
        const sValue = oEvent.getParameter("suggestValue");
        
        if (sValue.length < 2) {
            return;
        }

        try {
            const response = await fetch(`/api/v1/search/suggest?q=${encodeURIComponent(sValue)}`, {
                headers: {
                    "Authorization": `Bearer ${this._getAuthToken()}`
                }
            });

            if (response.ok) {
                const suggestions = await response.json();
                const oSearchField = oEvent.getSource();
                
                // Set suggestions
                oSearchField.suggest(sValue, suggestions.map((s: any) => s.text));
            }
        } catch (error) {
            Log.error("Search suggest failed", error.message, "com.sap.a2a.portal.controller.App");
        }
    }

    private getModel(sName?: string): JSONModel {
        return this.getView().getModel(sName) as JSONModel;
    }

    private getResourceBundle(): ResourceModel {
        return (this.getOwnerComponent().getModel("i18n") as ResourceModel).getResourceBundle();
    }

    private getOwnerComponent(): UIComponent {
        return super.getOwnerComponent() as UIComponent;
    }

    public onWorkspaceSelectionChange(oEvent: ComboBox$SelectionChangeEvent): void {
        const sSelectedKey = oEvent.getParameter("selectedItem")?.getKey();
        const oWorkspaceModel = this.getModel("workspace");
        
        if (sSelectedKey && oWorkspaceModel) {
            // Update current workspace
            oWorkspaceModel.setProperty("/current", sSelectedKey);
            
            // Show confirmation message
            const selectedWorkspace = oWorkspaceModel.getProperty("/available")
                .find((workspace: any) => workspace.id === sSelectedKey);
            
            if (selectedWorkspace) {
                MessageToast.show(`Switched to ${selectedWorkspace.name}`);
                
                // Trigger workspace change event
                this.getOwnerComponent().getEventBus().publish("app", "workspaceChanged", {
                    workspaceId: sSelectedKey,
                    workspaceName: selectedWorkspace.name
                });
                
                // Refresh current view data for new workspace
                this._refreshWorkspaceData(sSelectedKey);
            }
        }
    }

    private async _refreshWorkspaceData(workspaceId: string): Promise<void> {
        try {
            // Set app to busy state
            this.getModel("app").setProperty("/busy", true);
            
            // Fetch workspace-specific data
            const response = await fetch(`/api/v1/workspaces/${workspaceId}/data`, {
                headers: {
                    "Authorization": `Bearer ${this._getAuthToken()}`
                }
            });

            if (response.ok) {
                const workspaceData = await response.json();
                
                // Update relevant models with workspace data
                // This could include projects, agents, etc.
                Log.info("Workspace data loaded successfully", "com.sap.a2a.portal.controller.App");
                
                // Store workspace preference
                localStorage.setItem("selectedWorkspace", workspaceId);
            } else {
                MessageToast.show("Failed to load workspace data");
            }
        } catch (error) {
            Log.error("Error refreshing workspace data", error.message, "com.sap.a2a.portal.controller.App");
            MessageToast.show("Error loading workspace data");
        } finally {
            this.getModel("app").setProperty("/busy", false);
        }
    }

    private _applyTheme(sTheme: string): void {
        // Apply theme using SAP UI Core theme manager
        sap.ui.getCore().applyTheme(sTheme);
        
        // Store theme preference
        localStorage.setItem("userTheme", sTheme);
        
        // Log theme change for debugging
        Log.info(`Theme applied: ${sTheme}`, "com.sap.a2a.portal.controller.App");
    }

    public onSaveSettings(): void {
        const oSettingsModel = this.getModel("settings");
        const settings = oSettingsModel.getData();
        
        // Save theme and apply
        const currentTheme = localStorage.getItem("userTheme");
        const newTheme = settings.theme;
        
        if (currentTheme !== newTheme) {
            this._applyTheme(newTheme);
        }
        
        // Save all settings to localStorage
        localStorage.setItem("userLanguage", settings.language);
        localStorage.setItem("userTimezone", settings.timezone);
        localStorage.setItem("userAutoSave", settings.autoSave.toString());
        
        // Notification preferences
        localStorage.setItem("notifyEmail", settings.notifications.email.toString());
        localStorage.setItem("notifyPush", settings.notifications.push.toString());
        localStorage.setItem("notifyDeployment", settings.notifications.deployment.toString());
        localStorage.setItem("notifySecurity", settings.notifications.security.toString());
        localStorage.setItem("notifyAgents", settings.notifications.agents.toString());
        
        // Developer preferences
        localStorage.setItem("devDebugMode", settings.developer.debugMode.toString());
        localStorage.setItem("devConsoleLogging", settings.developer.consoleLogging.toString());
        localStorage.setItem("devPerformanceMonitoring", settings.developer.performanceMonitoring.toString());
        localStorage.setItem("devApiTimeout", settings.developer.apiTimeout.toString());
        
        MessageToast.show("Settings saved successfully");
        
        // Close settings dialog if it exists
        const oDialog = this.byId("settingsDialog");
        if (oDialog) {
            oDialog.close();
        }
    }

    public onResetSettings(): void {
        // Reset to default settings
        const oSettingsModel = this.getModel("settings");
        oSettingsModel.setData({
            theme: "sap_horizon",
            language: "en",
            timezone: "UTC", 
            autoSave: false,
            notifications: {
                email: false,
                push: false,
                deployment: true,
                security: true,
                agents: true
            },
            developer: {
                debugMode: false,
                consoleLogging: false,
                performanceMonitoring: false,
                apiTimeout: 30
            }
        });
        
        MessageToast.show("Settings reset to defaults");
    }

    public onCancelSettings(): void {
        // Close settings dialog without saving
        const oDialog = this.byId("settingsDialog");
        if (oDialog) {
            oDialog.close();
        }
    }

    private _initializeKeyboardShortcuts(): void {
        // Add global keyboard event listener
        document.addEventListener("keydown", this._handleKeyboardShortcut.bind(this), true);
        
        // Log keyboard shortcuts initialization
        Log.info("Keyboard shortcuts initialized", "com.sap.a2a.portal.controller.App");
    }

    private _handleKeyboardShortcut(oEvent: KeyboardEvent): void {
        // Check if user is typing in an input field
        const activeElement = document.activeElement as HTMLElement;
        const isInputElement = activeElement && (
            activeElement.tagName === "INPUT" ||
            activeElement.tagName === "TEXTAREA" ||
            activeElement.contentEditable === "true" ||
            activeElement.classList.contains("sapMInputBaseInner")
        );

        // Skip shortcuts if user is typing in input fields
        if (isInputElement && !oEvent.ctrlKey && !oEvent.metaKey) {
            return;
        }

        const key = oEvent.key.toLowerCase();
        const ctrlKey = oEvent.ctrlKey || oEvent.metaKey; // Support both Ctrl and Cmd
        const shiftKey = oEvent.shiftKey;
        const altKey = oEvent.altKey;

        // Handle keyboard shortcuts
        if (ctrlKey && !shiftKey && !altKey) {
            switch (key) {
                case "k": // Ctrl+K: Focus search
                    oEvent.preventDefault();
                    this._focusSearch();
                    break;
                case "h": // Ctrl+H: Go to home/projects
                    oEvent.preventDefault();
                    this._navigateToProjects();
                    break;
                case ",": // Ctrl+, : Open settings
                    oEvent.preventDefault();
                    this._openSettings();
                    break;
                case "/": // Ctrl+/: Show help/shortcuts
                    oEvent.preventDefault();
                    this._showKeyboardShortcuts();
                    break;
            }
        } else if (altKey && !ctrlKey && !shiftKey) {
            switch (key) {
                case "1": // Alt+1: Navigate to Projects
                    oEvent.preventDefault();
                    this._navigateToSection("projects");
                    break;
                case "2": // Alt+2: Navigate to Agents
                    oEvent.preventDefault();
                    this._navigateToSection("agents");
                    break;
                case "3": // Alt+3: Navigate to Tools
                    oEvent.preventDefault();
                    this._navigateToSection("agentBuilder");
                    break;
                case "4": // Alt+4: Navigate to Monitoring
                    oEvent.preventDefault();
                    this._navigateToSection("dashboard");
                    break;
                case "n": // Alt+N: Show notifications
                    oEvent.preventDefault();
                    this._openNotifications();
                    break;
            }
        } else if (!ctrlKey && !shiftKey && !altKey) {
            switch (key) {
                case "escape": // ESC: Close dialogs/modals
                    oEvent.preventDefault();
                    this._closeActiveDialog();
                    break;
                case "?": // ?: Show keyboard shortcuts help
                    if (oEvent.shiftKey) {
                        oEvent.preventDefault();
                        this._showKeyboardShortcuts();
                    }
                    break;
            }
        }
    }

    private _focusSearch(): void {
        const searchField = document.querySelector(".sapFShellBarSearch input") as HTMLInputElement;
        if (searchField) {
            searchField.focus();
            MessageToast.show("Search focused (Ctrl+K)");
        }
    }

    private _navigateToProjects(): void {
        this.getOwnerComponent().getRouter().navTo("ProjectsList");
        MessageToast.show("Navigated to Projects (Ctrl+H)");
    }

    private _openSettings(): void {
        // Trigger system settings navigation
        this.onSystemSettingsPress();
        MessageToast.show("Opening Settings (Ctrl+,)");
    }

    private _navigateToSection(sectionKey: string): void {
        const oSideNavigation = this.byId("sideNavigation");
        if (oSideNavigation) {
            // Set selection and trigger navigation
            oSideNavigation.setSelectedKey(sectionKey);
            oSideNavigation.fireItemSelect({
                item: oSideNavigation.getItems().find(item => item.getKey() === sectionKey)
            });
        }
    }

    private _openNotifications(): void {
        // Trigger notification press
        const notificationEvent = { getSource: () => document.querySelector(".sapFShellBarNotifications") };
        this.onNotificationPress.call(this);
        MessageToast.show("Notifications opened (Alt+N)");
    }

    private _closeActiveDialog(): void {
        // Find and close any open dialogs
        const dialogs = document.querySelectorAll(".sapMDialog");
        if (dialogs.length > 0) {
            const lastDialog = dialogs[dialogs.length - 1];
            const closeButton = lastDialog.querySelector(".sapMBtnIcon[data-sap-ui-icon-content='\\e1c7']");
            if (closeButton) {
                (closeButton as HTMLElement).click();
                MessageToast.show("Dialog closed (ESC)");
            }
        }
    }

    private _showKeyboardShortcuts(): void {
        MessageToast.show("Keyboard Shortcuts: Ctrl+K (Search), Ctrl+H (Home), Ctrl+, (Settings), Alt+1-4 (Navigation), ESC (Close)", {
            duration: 5000,
            width: "auto"
        });
    }
}