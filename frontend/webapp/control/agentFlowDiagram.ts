import Control from "sap/ui/core/Control";
import RenderManager from "sap/ui/core/RenderManager";
import { ProcessFlow, ProcessFlowNode, ProcessFlowLaneHeader } from "sap/suite/ui/commons/ProcessFlow";
import JSONModel from "sap/ui/model/json/JSONModel";

/**
 * Agent Flow Diagram Control
 * Visualizes agent communication flow using SAP Fiori patterns
 * @namespace com.sap.a2a.portal.control
 */
export default class AgentFlowDiagram extends Control {
    private _processFlow: ProcessFlow;
    private _model: JSONModel;

    static readonly metadata = {
        properties: {
            /**
             * Width of the control
             */
            width: { type: "sap.ui.core.CSSSize", defaultValue: "100%" },
            
            /**
             * Height of the control
             */
            height: { type: "sap.ui.core.CSSSize", defaultValue: "600px" },
            
            /**
             * Show labels on connections
             */
            showLabels: { type: "boolean", defaultValue: true },
            
            /**
             * Enable zoom controls
             */
            zoomable: { type: "boolean", defaultValue: true }
        },
        
        aggregations: {
            /**
             * Process flow control
             */
            _processFlow: { 
                type: "sap.suite.ui.commons.ProcessFlow", 
                multiple: false, 
                visibility: "hidden" 
            }
        },
        
        events: {
            /**
             * Fired when a node is selected
             */
            nodePress: {
                parameters: {
                    nodeId: { type: "string" },
                    agentId: { type: "string" }
                }
            },
            
            /**
             * Fired when zoom level changes
             */
            zoomChanged: {
                parameters: {
                    zoomLevel: { type: "float" }
                }
            }
        }
    };

    constructor(sId?: string, mSettings?: any) {
        super(sId, mSettings);
        this._initializeProcessFlow();
    }

    public init(): void {
        this._model = new JSONModel({
            nodes: [],
            lanes: []
        });
        this.setModel(this._model, "flow");
    }

    private _initializeProcessFlow(): void {
        this._processFlow = new ProcessFlow({
            scrollable: false,
            wheelZoomable: "{= ${/zoomable} }",
            nodePress: this._onNodePress.bind(this)
        });

        this.setAggregation("_processFlow", this._processFlow);
    }

    private _onNodePress(oEvent: any): void {
        const oNode = oEvent.getParameter("node");
        const sNodeId = oNode.getNodeId();
        const oNodeData = this._getNodeData(sNodeId);

        this.fireNodePress({
            nodeId: sNodeId,
            agentId: oNodeData?.agentId || ""
        });
    }

    private _getNodeData(sNodeId: string): any {
        const aNodes = this._model.getProperty("/nodes") || [];
        return aNodes.find((node: any) => node.nodeId === sNodeId);
    }

    /**
     * Set the agent flow data
     */
    public setFlowData(oData: IAgentFlowData): void {
        const oFlowData = this._transformToProcessFlow(oData);
        this._model.setData(oFlowData);
        this._updateProcessFlow();
    }

    /**
     * Transform agent data to process flow format
     */
    private _transformToProcessFlow(oData: IAgentFlowData): any {
        const lanes: any[] = [];
        const nodes: any[] = [];
        const laneMap = new Map<string, number>();

        // Create lanes for each agent type
        const agentTypes = ["trigger", "agents", "processing", "output"];
        agentTypes.forEach((type, index) => {
            lanes.push({
                laneId: `lane_${index}`,
                iconSrc: this._getLaneIcon(type),
                text: this._getLaneText(type),
                state: [{ state: "Positive", value: 100 }],
                position: index
            });
            laneMap.set(type, index);
        });

        // Create nodes for each agent
        oData.agents.forEach((agent, index) => {
            const laneId = this._determineLane(agent.type);
            const node = {
                nodeId: `node_${agent.id}`,
                laneId: `lane_${laneMap.get(laneId)}`,
                title: agent.name,
                titleAbbreviation: agent.abbreviation || agent.id.toUpperCase(),
                texts: agent.description ? [agent.description] : [],
                highlighted: agent.active || false,
                state: this._getNodeState(agent.status),
                stateText: agent.statusText,
                focused: agent.focused || false,
                selected: agent.selected || false,
                children: agent.connections || [],
                agentId: agent.id,
                attributes: [
                    {
                        label: "Type",
                        value: agent.type
                    },
                    {
                        label: "Messages",
                        value: agent.messagesProcessed?.toString() || "0"
                    },
                    {
                        label: "Response Time",
                        value: `${agent.responseTime || 0}ms`
                    }
                ]
            };
            nodes.push(node);
        });

        return {
            nodes: nodes,
            lanes: lanes,
            zoomable: this.getZoomable()
        };
    }

    private _getLaneIcon(type: string): string {
        const iconMap: Record<string, string> = {
            trigger: "sap-icon://activate",
            agents: "sap-icon://group",
            processing: "sap-icon://process",
            output: "sap-icon://outbox"
        };
        return iconMap[type] || "sap-icon://question-mark";
    }

    private _getLaneText(type: string): string {
        const textMap: Record<string, string> = {
            trigger: "Input",
            agents: "Agent Processing",
            processing: "Data Processing",
            output: "Output"
        };
        return textMap[type] || type;
    }

    private _determineLane(agentType: string): string {
        if (agentType.includes("trigger") || agentType.includes("input")) {
            return "trigger";
        } else if (agentType.includes("output") || agentType.includes("result")) {
            return "output";
        } else if (agentType.includes("process") || agentType.includes("transform")) {
            return "processing";
        }
        return "agents";
    }

    private _getNodeState(status: string): string {
        const stateMap: Record<string, string> = {
            active: "Positive",
            success: "Positive",
            warning: "Critical",
            error: "Negative",
            inactive: "Neutral",
            processing: "Planned"
        };
        return stateMap[status] || "Neutral";
    }

    private _updateProcessFlow(): void {
        // Clear existing nodes
        this._processFlow.removeAllNodes();
        this._processFlow.removeAllLanes();

        // Add lanes
        const lanes = this._model.getProperty("/lanes") || [];
        lanes.forEach((laneData: any) => {
            const lane = new ProcessFlowLaneHeader({
                laneId: laneData.laneId,
                iconSrc: laneData.iconSrc,
                text: laneData.text,
                state: laneData.state,
                position: laneData.position
            });
            this._processFlow.addLane(lane);
        });

        // Add nodes
        const nodes = this._model.getProperty("/nodes") || [];
        nodes.forEach((nodeData: any) => {
            const node = new ProcessFlowNode({
                nodeId: nodeData.nodeId,
                laneId: nodeData.laneId,
                title: nodeData.title,
                titleAbbreviation: nodeData.titleAbbreviation,
                texts: nodeData.texts,
                highlighted: nodeData.highlighted,
                state: nodeData.state,
                stateText: nodeData.stateText,
                focused: nodeData.focused,
                selected: nodeData.selected,
                children: nodeData.children
            });

            // Add attributes
            if (nodeData.attributes) {
                nodeData.attributes.forEach((attr: any) => {
                    node.addAttribute({
                        label: attr.label,
                        value: attr.value
                    });
                });
            }

            this._processFlow.addNode(node);
        });
    }

    /**
     * Zoom in the flow diagram
     */
    public zoomIn(): void {
        const currentLevel = this._processFlow.getZoomLevel();
        const newLevel = Math.min(currentLevel + 0.1, 2);
        this._processFlow.setZoomLevel(newLevel);
        this.fireZoomChanged({ zoomLevel: newLevel });
    }

    /**
     * Zoom out the flow diagram
     */
    public zoomOut(): void {
        const currentLevel = this._processFlow.getZoomLevel();
        const newLevel = Math.max(currentLevel - 0.1, 0.5);
        this._processFlow.setZoomLevel(newLevel);
        this.fireZoomChanged({ zoomLevel: newLevel });
    }

    /**
     * Reset zoom to default
     */
    public resetZoom(): void {
        this._processFlow.setZoomLevel(1);
        this.fireZoomChanged({ zoomLevel: 1 });
    }

    /**
     * Highlight a specific path
     */
    public highlightPath(agentIds: string[]): void {
        const nodes = this._model.getProperty("/nodes") || [];
        
        // Reset all highlights
        nodes.forEach((node: any) => {
            node.highlighted = false;
            node.focused = false;
        });

        // Highlight specified path
        agentIds.forEach((agentId) => {
            const node = nodes.find((n: any) => n.agentId === agentId);
            if (node) {
                node.highlighted = true;
                node.focused = true;
            }
        });

        this._model.setProperty("/nodes", nodes);
        this._updateProcessFlow();
    }

    public renderer = {
        apiVersion: 2,
        
        render: function(oRm: RenderManager, oControl: AgentFlowDiagram) {
            oRm.openStart("div", oControl);
            oRm.class("sapUiAgentFlowDiagram");
            oRm.style("width", oControl.getWidth());
            oRm.style("height", oControl.getHeight());
            oRm.style("position", "relative");
            
            // Accessibility
            oRm.attr("role", "img");
            oRm.attr("aria-label", "Agent Communication Flow Diagram");
            
            oRm.openEnd();
            
            // Render process flow
            oRm.renderControl(oControl.getAggregation("_processFlow"));
            
            // Render zoom controls if enabled
            if (oControl.getZoomable()) {
                oRm.openStart("div");
                oRm.class("sapUiAgentFlowZoomControls");
                oRm.openEnd();
                
                // Zoom in button
                oRm.openStart("button");
                oRm.class("sapMBtn sapMBtnBase");
                oRm.attr("title", "Zoom In");
                oRm.attr("onclick", `sap.ui.getCore().byId('${oControl.getId()}').zoomIn()`);
                oRm.openEnd();
                oRm.icon("sap-icon://zoom-in");
                oRm.close("button");
                
                // Zoom out button
                oRm.openStart("button");
                oRm.class("sapMBtn sapMBtnBase");
                oRm.attr("title", "Zoom Out");
                oRm.attr("onclick", `sap.ui.getCore().byId('${oControl.getId()}').zoomOut()`);
                oRm.openEnd();
                oRm.icon("sap-icon://zoom-out");
                oRm.close("button");
                
                // Reset button
                oRm.openStart("button");
                oRm.class("sapMBtn sapMBtnBase");
                oRm.attr("title", "Reset Zoom");
                oRm.attr("onclick", `sap.ui.getCore().byId('${oControl.getId()}').resetZoom()`);
                oRm.openEnd();
                oRm.icon("sap-icon://reset");
                oRm.close("button");
                
                oRm.close("div");
            }
            
            oRm.close("div");
        }
    };
}

// Type definitions
interface IAgentFlowData {
    agents: IAgentNode[];
    connections?: IConnection[];
}

interface IAgentNode {
    id: string;
    name: string;
    type: string;
    abbreviation?: string;
    description?: string;
    status: string;
    statusText?: string;
    active?: boolean;
    focused?: boolean;
    selected?: boolean;
    connections?: string[];
    messagesProcessed?: number;
    responseTime?: number;
}

interface IConnection {
    from: string;
    to: string;
    label?: string;
}