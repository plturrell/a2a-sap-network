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
    }

    private _initializeSideNavigation(): void {
        // Set default selection
        const oSideNavigation = this.byId("sideNavigation");
        if (oSideNavigation) {
            oSideNavigation.setSelectedKey("projects");
        }
    }

    private async _checkAuthentication(): Promise<void> {
        try {
            const response = await fetch("/api/v1/users/me", {
                headers: {
                    "Authorization": `Bearer ${this._getAuthToken()}`
                }
            });

            if (response.ok) {
                const userData = await response.json();
                this.getModel("user").setData(userData);
            } else if (response.status === 401) {
                // Redirect to login
                window.location.href = "/login";
            }
        } catch (error) {
            console.error("Authentication check failed:", error);
        }
    }

    private _getAuthToken(): string {
        return localStorage.getItem("authToken") || "";
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
            console.error("Logout failed:", error);
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
            console.error("Search suggest failed:", error);
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
}