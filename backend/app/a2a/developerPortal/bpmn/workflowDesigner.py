"""
BPMN Workflow Designer for A2A Developer Portal
Provides comprehensive workflow design, validation, and execution capabilities
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum
from uuid import uuid4
import logging

from pydantic import BaseModel, Field, validator
import httpx

from .workflow_engine import WorkflowExecutionEngine, WorkflowEngineConfig, ExecutionState

logger = logging.getLogger(__name__)


class WorkflowElementType(str, Enum):
    """BPMN workflow element types"""
    START_EVENT = "startEvent"
    END_EVENT = "endEvent"
    TASK = "task"
    USER_TASK = "userTask"
    SERVICE_TASK = "serviceTask"
    SCRIPT_TASK = "scriptTask"
    GATEWAY_EXCLUSIVE = "exclusiveGateway"
    GATEWAY_PARALLEL = "parallelGateway"
    GATEWAY_INCLUSIVE = "inclusiveGateway"
    SEQUENCE_FLOW = "sequenceFlow"
    MESSAGE_FLOW = "messageFlow"
    SUBPROCESS = "subProcess"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class WorkflowElement(BaseModel):
    """BPMN workflow element"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: WorkflowElementType
    description: str = ""
    
    # Position and dimensions for UI
    x: float = 0
    y: float = 0
    width: float = 100
    height: float = 80
    
    # Element-specific configuration
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Connections
    incoming: List[str] = Field(default_factory=list)  # IDs of incoming flows
    outgoing: List[str] = Field(default_factory=list)  # IDs of outgoing flows
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Element name cannot be empty')
        return v.strip()


class WorkflowConnection(BaseModel):
    """BPMN workflow connection (sequence flow)"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    source_id: str
    target_id: str
    condition: Optional[str] = None  # Conditional flow expression
    
    # Visual properties
    waypoints: List[Dict[str, float]] = Field(default_factory=list)  # [{"x": 100, "y": 200}, ...]
    
    @validator('source_id')
    def validate_source(cls, v):
        if not v:
            raise ValueError('Source ID is required')
        return v
    
    @validator('target_id')
    def validate_target(cls, v):
        if not v:
            raise ValueError('Target ID is required')
        return v


class WorkflowDefinition(BaseModel):
    """Complete BPMN workflow definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Workflow elements
    elements: List[WorkflowElement] = Field(default_factory=list)
    connections: List[WorkflowConnection] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    
    # Execution configuration
    variables: Dict[str, Any] = Field(default_factory=dict)
    listeners: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Deployment info
    deployment_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.DRAFT
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Workflow name cannot be empty')
        return v.strip()


class WorkflowValidationResult(BaseModel):
    """Workflow validation result"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    # Execution context
    variables: Dict[str, Any] = Field(default_factory=dict)
    current_activities: List[str] = Field(default_factory=list)
    completed_activities: List[str] = Field(default_factory=list)
    
    # Results and errors
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BPMNWorkflowDesigner:
    """BPMN Workflow Designer with comprehensive functionality"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Initialize workflow execution engine
        engine_config = WorkflowEngineConfig(
            persistence_path=str(self.storage_path / "executions"),
            enable_persistence=True,
            enable_metrics=True,
            enable_tracing=True,
            blockchain={
                "networks": {
                    "local": {
                        "provider_url": "http://localhost:8545",
                        "chain_id": 31337
                    }
                }
            }
        )
        self.execution_engine = WorkflowExecutionEngine(engine_config)
        
        # Load existing workflows
        self._load_workflows()
        
        logger.info(f"BPMN Workflow Designer initialized with {len(self.workflows)} workflows")
    
    def _load_workflows(self):
        """Load workflows from storage"""
        try:
            for workflow_file in self.storage_path.glob("*.json"):
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                    workflow = WorkflowDefinition(**workflow_data)
                    self.workflows[workflow.id] = workflow
                    
        except Exception as e:
            logger.error(f"Error loading workflows: {e}")
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowDefinition:
        """Create new workflow"""
        try:
            workflow = WorkflowDefinition(**workflow_data)
            
            # Save to storage
            await self._save_workflow(workflow)
            
            # Store in memory
            self.workflows[workflow.id] = workflow
            
            logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            raise
    
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    async def get_all_workflows(self) -> List[WorkflowDefinition]:
        """Get all workflows"""
        return list(self.workflows.values())
    
    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[WorkflowDefinition]:
        """Update existing workflow"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            
            workflow.last_modified = datetime.utcnow()
            
            # Save to storage
            await self._save_workflow(workflow)
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error updating workflow {workflow_id}: {e}")
            return None
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        try:
            if workflow_id not in self.workflows:
                return False
            
            # Remove from storage
            workflow_file = self.storage_path / f"{workflow_id}.json"
            if workflow_file.exists():
                workflow_file.unlink()
            
            # Remove from memory
            del self.workflows[workflow_id]
            
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting workflow {workflow_id}: {e}")
            return False
    
    async def validate_workflow(self, workflow: WorkflowDefinition) -> WorkflowValidationResult:
        """Validate workflow definition"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for start events
        start_events = [e for e in workflow.elements if e.type == WorkflowElementType.START_EVENT]
        if not start_events:
            errors.append("Workflow must have at least one start event")
        elif len(start_events) > 1:
            warnings.append("Multiple start events detected - ensure this is intentional")
        
        # Check for end events
        end_events = [e for e in workflow.elements if e.type == WorkflowElementType.END_EVENT]
        if not end_events:
            warnings.append("Workflow has no end events - it may run indefinitely")
        
        # Validate element connections
        element_ids = {e.id for e in workflow.elements}
        connection_sources = {c.source_id for c in workflow.connections}
        connection_targets = {c.target_id for c in workflow.connections}
        
        # Check for orphaned elements
        for element in workflow.elements:
            if element.type not in [WorkflowElementType.START_EVENT, WorkflowElementType.END_EVENT]:
                if element.id not in connection_targets and element.id not in connection_sources:
                    warnings.append(f"Element '{element.name}' is not connected to the workflow")
        
        # Check for invalid connections
        for connection in workflow.connections:
            if connection.source_id not in element_ids:
                errors.append(f"Connection references invalid source element: {connection.source_id}")
            if connection.target_id not in element_ids:
                errors.append(f"Connection references invalid target element: {connection.target_id}")
        
        # Check for unreachable elements
        reachable_elements = set()
        if start_events:
            await self._find_reachable_elements(start_events[0].id, workflow, reachable_elements)
        
        for element in workflow.elements:
            if element.id not in reachable_elements and element.type != WorkflowElementType.START_EVENT:
                warnings.append(f"Element '{element.name}' may be unreachable")
        
        # Performance suggestions
        if len(workflow.elements) > 50:
            suggestions.append("Consider breaking large workflows into smaller sub-processes")
        
        if len([e for e in workflow.elements if e.type == WorkflowElementType.GATEWAY_PARALLEL]) > 10:
            suggestions.append("Too many parallel gateways may impact performance")
        
        return WorkflowValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def _find_reachable_elements(
        self, 
        element_id: str, 
        workflow: WorkflowDefinition, 
        reachable: set
    ):
        """Recursively find reachable elements"""
        if element_id in reachable:
            return
        
        reachable.add(element_id)
        
        # Find outgoing connections
        for connection in workflow.connections:
            if connection.source_id == element_id:
                await self._find_reachable_elements(connection.target_id, workflow, reachable)
    
    async def generate_bpmn_xml(self, workflow: WorkflowDefinition) -> str:
        """Generate BPMN 2.0 XML from workflow definition"""
        try:
            # Create root element
            definitions = ET.Element("definitions")
            definitions.set("xmlns", "http://www.omg.org/spec/BPMN/20100524/MODEL")
            definitions.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
            definitions.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
            definitions.set("xmlns:dc", "http://www.omg.org/spec/DD/20100524/DC")
            definitions.set("xmlns:di", "http://www.omg.org/spec/DD/20100524/DI")
            definitions.set("id", f"definitions_{workflow.id}")
            definitions.set("targetNamespace", "http://a2a.dev/bpmn")
            
            # Create process element
            process = ET.SubElement(definitions, "process")
            process.set("id", workflow.id)
            process.set("name", workflow.name)
            process.set("isExecutable", "true")
            
            # Add workflow elements
            for element in workflow.elements:
                elem = ET.SubElement(process, element.type.value)
                elem.set("id", element.id)
                elem.set("name", element.name)
                
                # Add element-specific attributes
                if element.type == WorkflowElementType.SERVICE_TASK:
                    if "implementation" in element.properties:
                        elem.set("implementation", element.properties["implementation"])
                
                # Add incoming/outgoing references
                for incoming_id in element.incoming:
                    incoming_elem = ET.SubElement(elem, "incoming")
                    incoming_elem.text = incoming_id
                
                for outgoing_id in element.outgoing:
                    outgoing_elem = ET.SubElement(elem, "outgoing")
                    outgoing_elem.text = outgoing_id
            
            # Add connections (sequence flows)
            for connection in workflow.connections:
                flow = ET.SubElement(process, "sequenceFlow")
                flow.set("id", connection.id)
                flow.set("sourceRef", connection.source_id)
                flow.set("targetRef", connection.target_id)
                
                if connection.name:
                    flow.set("name", connection.name)
                
                if connection.condition:
                    condition_elem = ET.SubElement(flow, "conditionExpression")
                    condition_elem.set("xsi:type", "tFormalExpression")
                    condition_elem.text = connection.condition
            
            # Add diagram information
            diagram = ET.SubElement(definitions, "bpmndi:BPMNDiagram")
            diagram.set("id", f"diagram_{workflow.id}")
            
            plane = ET.SubElement(diagram, "bpmndi:BPMNPlane")
            plane.set("id", f"plane_{workflow.id}")
            plane.set("bpmnElement", workflow.id)
            
            # Add element shapes
            for element in workflow.elements:
                shape = ET.SubElement(plane, "bpmndi:BPMNShape")
                shape.set("id", f"shape_{element.id}")
                shape.set("bpmnElement", element.id)
                
                bounds = ET.SubElement(shape, "dc:Bounds")
                bounds.set("x", str(element.x))
                bounds.set("y", str(element.y))
                bounds.set("width", str(element.width))
                bounds.set("height", str(element.height))
            
            # Add connection edges
            for connection in workflow.connections:
                edge = ET.SubElement(plane, "bpmndi:BPMNEdge")
                edge.set("id", f"edge_{connection.id}")
                edge.set("bpmnElement", connection.id)
                
                # Add waypoints
                for waypoint in connection.waypoints:
                    wp = ET.SubElement(edge, "di:waypoint")
                    wp.set("x", str(waypoint["x"]))
                    wp.set("y", str(waypoint["y"]))
            
            # Convert to string
            ET.indent(definitions, space="  ")
            return ET.tostring(definitions, encoding='unicode')
            
        except Exception as e:
            logger.error(f"Error generating BPMN XML: {e}")
            raise
    
    async def parse_bpmn_xml(self, xml_content: str) -> WorkflowDefinition:
        """Parse BPMN 2.0 XML into workflow definition"""
        try:
            root = ET.fromstring(xml_content)
            
            # Find process element
            process = root.find(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}process")
            if process is None:
                raise ValueError("No process element found in BPMN XML")
            
            workflow_id = process.get("id", str(uuid4()))
            workflow_name = process.get("name", "Imported Workflow")
            
            elements = []
            connections = []
            
            # Parse elements
            for elem in process:
                if elem.tag.endswith(('Event', 'Task', 'Gateway')):
                    element_type = elem.tag.split('}')[-1]  # Remove namespace
                    
                    element = WorkflowElement(
                        id=elem.get("id"),
                        name=elem.get("name", ""),
                        type=WorkflowElementType(element_type)
                    )
                    
                    # Parse incoming/outgoing
                    for incoming in elem.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}incoming"):
                        if incoming.text:
                            element.incoming.append(incoming.text)
                    
                    for outgoing in elem.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}outgoing"):
                        if outgoing.text:
                            element.outgoing.append(outgoing.text)
                    
                    elements.append(element)
                
                elif elem.tag.endswith('sequenceFlow'):
                    connection = WorkflowConnection(
                        id=elem.get("id"),
                        name=elem.get("name", ""),
                        source_id=elem.get("sourceRef"),
                        target_id=elem.get("targetRef")
                    )
                    
                    # Parse condition
                    condition_elem = elem.find(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}conditionExpression")
                    if condition_elem is not None and condition_elem.text:
                        connection.condition = condition_elem.text
                    
                    connections.append(connection)
            
            # Parse diagram information for positioning
            diagram = root.find(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNDiagram")
            if diagram is not None:
                plane = diagram.find(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNPlane")
                if plane is not None:
                    # Parse shapes
                    for shape in plane.findall(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNShape"):
                        element_id = shape.get("bpmnElement")
                        bounds = shape.find(".//{http://www.omg.org/spec/DD/20100524/DC}Bounds")
                        
                        if bounds is not None:
                            # Find corresponding element
                            for element in elements:
                                if element.id == element_id:
                                    element.x = float(bounds.get("x", 0))
                                    element.y = float(bounds.get("y", 0))
                                    element.width = float(bounds.get("width", 100))
                                    element.height = float(bounds.get("height", 80))
                                    break
                    
                    # Parse edges
                    for edge in plane.findall(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNEdge"):
                        connection_id = edge.get("bpmnElement")
                        waypoints = []
                        
                        for waypoint in edge.findall(".//{http://www.omg.org/spec/DD/20100524/DI}waypoint"):
                            waypoints.append({
                                "x": float(waypoint.get("x", 0)),
                                "y": float(waypoint.get("y", 0))
                            })
                        
                        # Find corresponding connection
                        for connection in connections:
                            if connection.id == connection_id:
                                connection.waypoints = waypoints
                                break
            
            return WorkflowDefinition(
                id=workflow_id,
                name=workflow_name,
                elements=elements,
                connections=connections
            )
            
        except Exception as e:
            logger.error(f"Error parsing BPMN XML: {e}")
            raise
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute workflow using real execution engine"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            # Convert workflow definition to engine format
            engine_workflow = {
                "id": workflow.id,
                "name": workflow.name,
                "elements": [elem.dict() for elem in workflow.elements],
                "connections": [conn.dict() for conn in workflow.connections]
            }
            
            # Start execution in engine
            execution_id = await self.execution_engine.start_execution(
                engine_workflow,
                variables
            )
            
            # Create execution instance for tracking
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.ACTIVE,
                variables=variables or {}
            )
            
            # Store execution
            self.executions[execution.id] = execution
            
            # Start monitoring execution status
            asyncio.create_task(self._monitor_execution(execution))
            
            return execution
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            raise
    
    async def _monitor_execution(self, execution: WorkflowExecution):
        """Monitor workflow execution status"""
        try:
            while execution.status == WorkflowStatus.ACTIVE:
                # Get execution status from engine
                engine_status = await self.execution_engine.get_execution_status(execution.id)
                
                if not engine_status:
                    break
                
                # Update execution based on engine status
                engine_state = engine_status.get("state")
                
                if engine_state == ExecutionState.COMPLETED.value:
                    execution.status = WorkflowStatus.COMPLETED
                    execution.ended_at = datetime.utcnow()
                    execution.result = engine_status.get("variables", {})
                    execution.completed_activities = engine_status.get("completed_elements", [])
                    
                elif engine_state == ExecutionState.FAILED.value:
                    execution.status = WorkflowStatus.FAILED
                    execution.ended_at = datetime.utcnow()
                    execution.error_message = engine_status.get("error", "Unknown error")
                    
                elif engine_state == ExecutionState.SUSPENDED.value:
                    execution.status = WorkflowStatus.SUSPENDED
                    
                elif engine_state == ExecutionState.TERMINATED.value:
                    execution.status = WorkflowStatus.TERMINATED
                    execution.ended_at = datetime.utcnow()
                    
                elif engine_state == ExecutionState.WAITING.value:
                    # Update current activities
                    execution.current_activities = engine_status.get("active_elements", [])
                
                # Wait before next check
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error monitoring execution {execution.id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = f"Monitoring error: {str(e)}"
            execution.ended_at = datetime.utcnow()
    
    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        return self.executions.get(execution_id)
    
    async def get_workflow_executions(self, workflow_id: str) -> List[WorkflowExecution]:
        """Get all executions for a workflow"""
        return [e for e in self.executions.values() if e.workflow_id == workflow_id]
    
    async def get_user_tasks(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get pending user tasks from execution engine"""
        return await self.execution_engine.get_user_tasks(filters)
    
    async def complete_user_task(self, task_id: str, variables: Dict[str, Any]) -> bool:
        """Complete a user task"""
        return await self.execution_engine.complete_user_task(task_id, variables)
    
    async def terminate_execution(self, execution_id: str, reason: str = "") -> bool:
        """Terminate workflow execution"""
        try:
            await self.execution_engine.terminate_execution(execution_id, reason)
            
            # Update local execution record
            if execution_id in self.executions:
                self.executions[execution_id].status = WorkflowStatus.TERMINATED
                self.executions[execution_id].ended_at = datetime.utcnow()
            
            return True
        except Exception as e:
            logger.error(f"Failed to terminate execution {execution_id}: {e}")
            return False
    
    async def close(self):
        """Close the workflow designer and cleanup resources"""
        await self.execution_engine.close()
    
    async def _save_workflow(self, workflow: WorkflowDefinition):
        """Save workflow to storage"""
        try:
            workflow_file = self.storage_path / f"{workflow.id}.json"
            with open(workflow_file, 'w') as f:
                json.dump(workflow.dict(), f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving workflow {workflow.id}: {e}")
            raise
