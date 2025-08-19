"""
Context Engineering Orchestration Layer
Implements the complete BPMN process flow with all lanes, events, and gateways
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process execution states matching BPMN events"""
    STARTED = "started"
    PARSING = "parsing"
    ASSESSING = "assessing"
    OPTIMIZING = "optimizing"
    ENGINEERING = "engineering"
    COORDINATING = "coordinating"
    REASONING = "reasoning"
    COMPLETED = "completed"
    FAILED = "failed"


class GatewayType(Enum):
    """BPMN gateway types"""
    EXCLUSIVE = "exclusive"
    PARALLEL = "parallel"
    INCLUSIVE = "inclusive"
    EVENT_BASED = "event_based"


@dataclass
class ProcessEvent:
    """BPMN process event"""
    event_id: str
    event_type: str  # start, end, intermediate, timer
    timestamp: datetime
    data: Dict[str, Any]
    lane: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "lane": self.lane
        }


@dataclass
class ProcessTask:
    """BPMN task representation"""
    task_id: str
    task_type: str  # service, user, script, manual, send, receive
    name: str
    lane: str
    handler: Optional[Callable] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "name": self.name,
            "lane": self.lane,
            "status": self.status,
            "error": self.error
        }


@dataclass
class ProcessGateway:
    """BPMN gateway for flow control"""
    gateway_id: str
    gateway_type: GatewayType
    condition: Optional[Callable] = None
    outgoing_flows: List[str] = field(default_factory=list)
    
    async def evaluate(self, process_data: Dict[str, Any]) -> List[str]:
        """Evaluate gateway and return active flows"""
        if self.gateway_type == GatewayType.EXCLUSIVE:
            # Only one path
            if self.condition:
                return [await self.condition(process_data)]
            return [self.outgoing_flows[0]] if self.outgoing_flows else []
            
        elif self.gateway_type == GatewayType.PARALLEL:
            # All paths
            return self.outgoing_flows
            
        elif self.gateway_type == GatewayType.INCLUSIVE:
            # One or more paths based on conditions
            active_flows = []
            for flow in self.outgoing_flows:
                if await self._evaluate_flow_condition(flow, process_data):
                    active_flows.append(flow)
            return active_flows
            
        elif self.gateway_type == GatewayType.EVENT_BASED:
            # Wait for events
            return await self._wait_for_events(process_data)
        
        return []
    
    async def _evaluate_flow_condition(self, flow: str, data: Dict[str, Any]) -> bool:
        """Evaluate condition for a specific flow"""
        # Simplified - in production would have flow-specific conditions
        return True
    
    async def _wait_for_events(self, data: Dict[str, Any]) -> List[str]:
        """Wait for events in event-based gateway"""
        # Simplified - in production would wait for actual events
        await asyncio.sleep(0.1)
        return self.outgoing_flows[:1] if self.outgoing_flows else []


class ProcessOrchestrator:
    """Main orchestrator implementing the complete BPMN process"""
    
    def __init__(self, context_agent):
        self.context_agent = context_agent
        self.process_graph = nx.DiGraph()
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        self.process_events: Dict[str, List[ProcessEvent]] = defaultdict(list)
        self.lane_handlers = self._initialize_lane_handlers()
        
        # Build process graph from BPMN
        self._build_process_graph()
    
    def _initialize_lane_handlers(self) -> Dict[str, Dict[str, Callable]]:
        """Initialize handlers for each BPMN lane"""
        return {
            "main_orchestrator": {
                "initialize_context_state": self._initialize_context_state,
            },
            "context_understanding": {
                "parse_context": self._handle_parse_context,
                "assess_relevance": self._handle_assess_relevance,
                "domain_classification": self._handle_domain_classification,
            },
            "context_management": {
                "optimize_window": self._handle_optimize_window,
                "manage_memory": self._handle_manage_memory,
                "cache_management": self._handle_cache_management,
            },
            "context_engineering": {
                "generate_template": self._handle_generate_template,
                "assess_quality": self._handle_assess_quality,
                "improve_context": self._handle_improve_context,
            },
            "multi_agent_coordination": {
                "synchronize_contexts": self._handle_synchronize_contexts,
                "propagate_updates": self._handle_propagate_updates,
                "health_check": self._handle_health_check,
            },
            "context_aware_reasoning": {
                "coordinate_reasoning": self._handle_coordinate_reasoning,
                "synthesize_results": self._handle_synthesize_results,
            },
            "quality_feedback": {
                "monitor_performance": self._handle_monitor_performance,
                "collect_metrics": self._handle_collect_metrics,
                "update_models": self._handle_update_models,
            }
        }
    
    def _build_process_graph(self):
        """Build the complete process graph from BPMN specification"""
        # Start event
        self.process_graph.add_node("start", type="event", event_type="start")
        
        # Main orchestrator lane
        self.process_graph.add_node("initialize_state", type="task", lane="main_orchestrator")
        self.process_graph.add_node("fork_parallel", type="gateway", gateway_type=GatewayType.PARALLEL)
        
        # Context understanding lane
        self._add_subprocess("context_parsing", [
            ("ner_extraction", "service"),
            ("dependency_parsing", "service"),
            ("semantic_roles", "service"),
            ("calculate_relevance", "script")
        ], "context_understanding")
        
        self._add_subprocess("relevance_assessment", [
            ("bert_similarity", "service"),
            ("domain_scoring", "service"),
            ("temporal_analysis", "service"),
            ("relevance_decision", "gateway", GatewayType.EXCLUSIVE)
        ], "context_understanding")
        
        # Context management lane
        self._add_subprocess("window_optimization", [
            ("priority_ranking", "service"),
            ("token_counting", "service"),
            ("compression", "service"),
            ("window_decision", "gateway", GatewayType.EXCLUSIVE)
        ], "context_management")
        
        self._add_subprocess("memory_management", [
            ("working_memory", "manual"),
            ("semantic_memory", "manual"),
            ("episodic_memory", "manual"),
            ("memory_operations", "gateway", GatewayType.PARALLEL)
        ], "context_management")
        
        # Context engineering lane
        self._add_subprocess("template_generation", [
            ("task_analysis", "service"),
            ("template_library", "user"),
            ("optimization", "service"),
            ("template_decision", "gateway", GatewayType.EXCLUSIVE)
        ], "context_engineering")
        
        self._add_subprocess("quality_assessment", [
            ("coherence_analysis", "service"),
            ("completeness_check", "service"),
            ("accuracy_validation", "service"),
            ("bias_detection", "service"),
            ("quality_decision", "gateway", GatewayType.INCLUSIVE)
        ], "context_engineering")
        
        # Multi-agent coordination lane
        self._add_subprocess("synchronization", [
            ("detect_conflicts", "service"),
            ("version_control", "service"),
            ("state_merge", "service"),
            ("conflict_decision", "gateway", GatewayType.EXCLUSIVE)
        ], "multi_agent_coordination")
        
        self._add_subprocess("propagation", [
            ("generate_diff", "service"),
            ("route_planning", "service"),
            ("priority_queue", "service"),
            ("propagation_strategy", "gateway", GatewayType.INCLUSIVE)
        ], "multi_agent_coordination")
        
        # Context-aware reasoning lane
        self._add_subprocess("reasoning_coordination", [
            ("analyze_requirements", "service"),
            ("select_agents", "service"),
            ("optimize_chain", "service"),
            ("execute_reasoning", "service"),
            ("reasoning_loop", "gateway", GatewayType.EXCLUSIVE)
        ], "context_aware_reasoning")
        
        # End events
        self.process_graph.add_node("join_parallel", type="gateway", gateway_type=GatewayType.PARALLEL)
        self.process_graph.add_node("end", type="event", event_type="end")
        
        # Add main flow edges
        self._add_main_flows()
        
        # Add message flows to external systems
        self._add_message_flows()
        
        # Add feedback loops
        self._add_feedback_loops()
    
    def _add_subprocess(self, name: str, tasks: List[Tuple[str, str, ...]], lane: str):
        """Add a subprocess with its tasks to the graph"""
        subprocess_id = f"{lane}_{name}"
        self.process_graph.add_node(subprocess_id, type="subprocess", lane=lane)
        
        prev_task = subprocess_id
        for task_info in tasks:
            task_name = task_info[0]
            task_type = task_info[1]
            task_id = f"{subprocess_id}_{task_name}"
            
            if task_type == "gateway":
                gateway_type = task_info[2] if len(task_info) > 2 else GatewayType.EXCLUSIVE
                self.process_graph.add_node(
                    task_id, 
                    type="gateway", 
                    gateway_type=gateway_type,
                    lane=lane
                )
            else:
                self.process_graph.add_node(
                    task_id, 
                    type="task", 
                    task_type=task_type,
                    lane=lane
                )
            
            self.process_graph.add_edge(prev_task, task_id)
            prev_task = task_id
    
    def _add_main_flows(self):
        """Add main sequence flows between lanes"""
        # Start to initialization
        self.process_graph.add_edge("start", "initialize_state")
        self.process_graph.add_edge("initialize_state", "fork_parallel")
        
        # Parallel flows to different lanes
        self.process_graph.add_edge("fork_parallel", "context_understanding_context_parsing")
        self.process_graph.add_edge("fork_parallel", "context_management_window_optimization")
        self.process_graph.add_edge("fork_parallel", "context_engineering_template_generation")
        
        # Cross-lane dependencies
        self.process_graph.add_edge(
            "context_understanding_relevance_assessment_relevance_decision",
            "context_management_memory_management"
        )
        
        self.process_graph.add_edge(
            "context_management_memory_management_memory_operations",
            "context_engineering_quality_assessment"
        )
        
        self.process_graph.add_edge(
            "context_engineering_quality_assessment_quality_decision",
            "multi_agent_coordination_synchronization"
        )
        
        self.process_graph.add_edge(
            "multi_agent_coordination_propagation_propagation_strategy",
            "context_aware_reasoning_reasoning_coordination"
        )
        
        # Convergence to end
        self.process_graph.add_edge(
            "context_aware_reasoning_reasoning_coordination_reasoning_loop",
            "join_parallel"
        )
        self.process_graph.add_edge("join_parallel", "end")
    
    def _add_message_flows(self):
        """Add message flows to external systems"""
        external_systems = [
            ("vector_database", ["memory_management"]),
            ("knowledge_base", ["template_generation"]),
            ("llm_services", ["reasoning_coordination"]),
            ("embedding_models", ["context_parsing", "memory_management"]),
            ("domain_classifiers", ["relevance_assessment"]),
            ("monitoring_apis", ["quality_feedback"])
        ]
        
        for system, connected_tasks in external_systems:
            self.process_graph.add_node(
                f"external_{system}",
                type="external_system",
                system=system
            )
            
            for task in connected_tasks:
                # Add bidirectional message flow
                self.process_graph.add_edge(
                    f"context_management_{task}",
                    f"external_{system}",
                    flow_type="message"
                )
                self.process_graph.add_edge(
                    f"external_{system}",
                    f"context_management_{task}",
                    flow_type="message"
                )
    
    def _add_feedback_loops(self):
        """Add quality and performance feedback loops"""
        feedback_targets = [
            ("quality_feedback_collect_metrics", "context_understanding_context_parsing"),
            ("quality_feedback_monitor_performance", "context_management_window_optimization"),
            ("quality_feedback_update_models", "context_engineering_template_generation")
        ]
        
        for source, target in feedback_targets:
            self.process_graph.add_edge(
                source, target,
                flow_type="feedback",
                style="dotted"
            )
    
    async def execute_process(self, process_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete BPMN process"""
        # Initialize process state
        self.active_processes[process_id] = {
            "id": process_id,
            "state": ProcessState.STARTED,
            "data": initial_data,
            "start_time": datetime.now(),
            "current_tasks": set(),
            "completed_tasks": set(),
            "errors": []
        }
        
        # Record start event
        await self._record_event(process_id, "start", "start", initial_data, "main_orchestrator")
        
        try:
            # Execute process graph
            result = await self._execute_graph_from_node(
                process_id, "start", initial_data
            )
            
            # Record end event
            await self._record_event(process_id, "end", "end", result, "main_orchestrator")
            
            self.active_processes[process_id]["state"] = ProcessState.COMPLETED
            self.active_processes[process_id]["end_time"] = datetime.now()
            self.active_processes[process_id]["result"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Process {process_id} failed: {str(e)}")
            self.active_processes[process_id]["state"] = ProcessState.FAILED
            self.active_processes[process_id]["error"] = str(e)
            raise
    
    async def _execute_graph_from_node(
        self, 
        process_id: str, 
        node_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute process graph starting from a specific node"""
        node = self.process_graph.nodes[node_id]
        
        if node["type"] == "event":
            # Handle events
            return await self._handle_event(process_id, node_id, data)
            
        elif node["type"] == "task":
            # Execute task
            result = await self._execute_task(process_id, node_id, data)
            
            # Continue to next nodes
            next_nodes = list(self.process_graph.successors(node_id))
            if next_nodes:
                return await self._execute_graph_from_node(
                    process_id, next_nodes[0], result
                )
            return result
            
        elif node["type"] == "gateway":
            # Handle gateway logic
            return await self._handle_gateway(process_id, node_id, data)
            
        elif node["type"] == "subprocess":
            # Execute subprocess
            return await self._execute_subprocess(process_id, node_id, data)
        
        return data
    
    async def _handle_event(
        self, 
        process_id: str, 
        event_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle BPMN events"""
        node = self.process_graph.nodes[event_id]
        event_type = node.get("event_type", "intermediate")
        
        if event_type == "start":
            # Initialize process
            logger.info(f"Process {process_id} started")
            next_nodes = list(self.process_graph.successors(event_id))
            if next_nodes:
                return await self._execute_graph_from_node(
                    process_id, next_nodes[0], data
                )
                
        elif event_type == "end":
            # Finalize process
            logger.info(f"Process {process_id} completed")
            return data
            
        elif event_type == "timer":
            # Handle timer event
            delay = node.get("delay", 1.0)
            await asyncio.sleep(delay)
            return data
            
        elif event_type == "intermediate":
            # Handle intermediate events (validation, sync, etc.)
            event_handler = node.get("handler")
            if event_handler:
                return await event_handler(data)
        
        return data
    
    async def _execute_task(
        self, 
        process_id: str, 
        task_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific task"""
        node = self.process_graph.nodes[task_id]
        task_type = node.get("task_type", "service")
        lane = node.get("lane", "main_orchestrator")
        
        # Create task object
        task = ProcessTask(
            task_id=task_id,
            task_type=task_type,
            name=task_id.split("_")[-1],
            lane=lane,
            input_data=data
        )
        
        # Track active task
        self.active_processes[process_id]["current_tasks"].add(task_id)
        
        try:
            # Get handler from lane
            handler_name = task.name
            if lane in self.lane_handlers and handler_name in self.lane_handlers[lane]:
                handler = self.lane_handlers[lane][handler_name]
                task.output_data = await handler(process_id, data)
            else:
                # Default handler
                task.output_data = await self._default_task_handler(task, data)
            
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.active_processes[process_id]["errors"].append({
                "task": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
        
        finally:
            # Update tracking
            self.active_processes[process_id]["current_tasks"].discard(task_id)
            self.active_processes[process_id]["completed_tasks"].add(task_id)
        
        return task.output_data
    
    async def _handle_gateway(
        self, 
        process_id: str, 
        gateway_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle gateway branching logic"""
        node = self.process_graph.nodes[gateway_id]
        gateway_type = node.get("gateway_type", GatewayType.EXCLUSIVE)
        
        # Create gateway object
        outgoing = list(self.process_graph.successors(gateway_id))
        gateway = ProcessGateway(
            gateway_id=gateway_id,
            gateway_type=gateway_type,
            outgoing_flows=outgoing
        )
        
        # Add condition based on gateway type
        if "condition" in node:
            gateway.condition = node["condition"]
        
        # Evaluate gateway
        active_flows = await gateway.evaluate(data)
        
        if gateway_type == GatewayType.PARALLEL:
            # Execute all flows in parallel
            tasks = [
                self._execute_graph_from_node(process_id, flow, data.copy())
                for flow in active_flows
            ]
            results = await asyncio.gather(*tasks)
            
            # Merge results
            merged_data = data.copy()
            for result in results:
                merged_data.update(result)
            return merged_data
            
        elif gateway_type == GatewayType.EXCLUSIVE:
            # Execute only selected flow
            if active_flows:
                return await self._execute_graph_from_node(
                    process_id, active_flows[0], data
                )
            return data
            
        elif gateway_type == GatewayType.INCLUSIVE:
            # Execute selected flows
            if len(active_flows) == 1:
                return await self._execute_graph_from_node(
                    process_id, active_flows[0], data
                )
            else:
                # Multiple flows - execute in parallel
                tasks = [
                    self._execute_graph_from_node(process_id, flow, data.copy())
                    for flow in active_flows
                ]
                results = await asyncio.gather(*tasks)
                
                # Merge results
                merged_data = data.copy()
                for result in results:
                    merged_data.update(result)
                return merged_data
        
        return data
    
    async def _execute_subprocess(
        self, 
        process_id: str, 
        subprocess_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a subprocess"""
        # Get first task in subprocess
        subprocess_tasks = [
            n for n in self.process_graph.successors(subprocess_id)
        ]
        
        if subprocess_tasks:
            # Execute subprocess tasks
            current_data = data
            for task in subprocess_tasks:
                current_data = await self._execute_graph_from_node(
                    process_id, task, current_data
                )
            
            return current_data
        
        return data
    
    async def _record_event(
        self, 
        process_id: str, 
        event_id: str, 
        event_type: str, 
        data: Dict[str, Any], 
        lane: str
    ):
        """Record process event"""
        event = ProcessEvent(
            event_id=f"{process_id}_{event_id}_{datetime.now().timestamp()}",
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            lane=lane
        )
        
        self.process_events[process_id].append(event)
    
    # Lane-specific handlers
    
    async def _initialize_context_state(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize context state"""
        return {
            **data,
            "process_id": process_id,
            "initialized": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_parse_context(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context parsing"""
        result = await self.context_agent.handle_parse_context(
            data.get("message"),
            data.get("context_id", str(uuid.uuid4()))
        )
        return {**data, "parsed_context": result}
    
    async def _handle_assess_relevance(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle relevance assessment"""
        result = await self.context_agent.handle_assess_relevance(
            data.get("message"),
            data.get("context_id")
        )
        return {**data, "relevance_assessment": result}
    
    async def _handle_optimize_window(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle window optimization"""
        result = await self.context_agent.optimize_context_window(data)
        return {**data, "optimized_context": result}
    
    async def _handle_manage_memory(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory management"""
        result = await self.context_agent.semantic_memory_operations(data)
        return {**data, "memory_result": result}
    
    async def _handle_generate_template(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle template generation"""
        result = await self.context_agent.generate_context_template(data)
        return {**data, "template": result}
    
    async def _handle_assess_quality(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality assessment"""
        result = await self.context_agent.assess_context_quality(data)
        return {**data, "quality_assessment": result}
    
    async def _handle_synchronize_contexts(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context synchronization"""
        result = await self.context_agent.handle_coordinate_context(
            data.get("message"),
            data.get("context_id")
        )
        return {**data, "synchronization_result": result}
    
    async def _handle_coordinate_reasoning(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reasoning coordination"""
        # Simplified - would integrate with reasoning agents
        return {
            **data,
            "reasoning_result": {
                "status": "completed",
                "result": "Reasoning coordination completed"
            }
        }
    
    async def _handle_monitor_performance(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance monitoring"""
        metrics = {
            "response_time": (datetime.now() - self.active_processes[process_id]["start_time"]).total_seconds(),
            "tasks_completed": len(self.active_processes[process_id]["completed_tasks"]),
            "errors": len(self.active_processes[process_id]["errors"])
        }
        return {**data, "performance_metrics": metrics}
    
    # Default handlers for remaining tasks
    
    async def _handle_domain_classification(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "domain": "general"}
    
    async def _handle_cache_management(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "cache_updated": True}
    
    async def _handle_improve_context(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "improved": True}
    
    async def _handle_propagate_updates(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "propagated": True}
    
    async def _handle_health_check(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "health": "healthy"}
    
    async def _handle_synthesize_results(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "synthesized": True}
    
    async def _handle_collect_metrics(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "metrics_collected": True}
    
    async def _handle_update_models(self, process_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "models_updated": False}  # Not auto-updating in production
    
    async def _default_task_handler(self, task: ProcessTask, data: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler for tasks without specific implementation"""
        logger.warning(f"Using default handler for task: {task.task_id}")
        return {**data, f"{task.name}_completed": True}
    
    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """Get current status of a process"""
        if process_id not in self.active_processes:
            return {"error": "Process not found"}
        
        process = self.active_processes[process_id]
        duration = None
        
        if "start_time" in process:
            end_time = process.get("end_time", datetime.now())
            duration = (end_time - process["start_time"]).total_seconds()
        
        return {
            "process_id": process_id,
            "state": process["state"].value,
            "current_tasks": list(process["current_tasks"]),
            "completed_tasks": list(process["completed_tasks"]),
            "errors": process["errors"],
            "duration_seconds": duration,
            "events": [e.to_dict() for e in self.process_events.get(process_id, [])]
        }
    
    def visualize_process(self, process_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate process visualization data"""
        # Convert graph to visualization format
        nodes = []
        edges = []
        
        for node_id, node_data in self.process_graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "type": node_data.get("type"),
                "lane": node_data.get("lane"),
                "label": node_id.split("_")[-1],
                "status": self._get_node_status(process_id, node_id) if process_id else "inactive"
            })
        
        for source, target, edge_data in self.process_graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": edge_data.get("flow_type", "sequence"),
                "style": edge_data.get("style", "solid")
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "process_id": process_id
        }
    
    def _get_node_status(self, process_id: str, node_id: str) -> str:
        """Get execution status of a node"""
        if process_id not in self.active_processes:
            return "inactive"
        
        process = self.active_processes[process_id]
        if node_id in process["current_tasks"]:
            return "active"
        elif node_id in process["completed_tasks"]:
            return "completed"
        else:
            return "pending"


# Timer event handler for periodic tasks
class TimerEventHandler:
    """Handles timer events in the BPMN process"""
    
    def __init__(self, orchestrator: ProcessOrchestrator):
        self.orchestrator = orchestrator
        self.active_timers: Dict[str, asyncio.Task] = {}
    
    async def start_timer(
        self, 
        timer_id: str, 
        interval: float, 
        callback: Callable, 
        data: Dict[str, Any]
    ):
        """Start a timer event"""
        if timer_id in self.active_timers:
            await self.stop_timer(timer_id)
        
        async def timer_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Timer {timer_id} callback failed: {str(e)}")
        
        self.active_timers[timer_id] = asyncio.create_task(timer_loop())
    
    async def stop_timer(self, timer_id: str):
        """Stop a timer event"""
        if timer_id in self.active_timers:
            self.active_timers[timer_id].cancel()
            del self.active_timers[timer_id]
    
    async def stop_all_timers(self):
        """Stop all active timers"""
        for timer_id in list(self.active_timers.keys()):
            await self.stop_timer(timer_id)


# Example usage
if __name__ == "__main__":
    async def example_orchestration():
        # Create mock context agent
        class MockContextAgent:
            async def handle_parse_context(self, message, context_id):
                return {"status": "success", "parsed": True}
            
            async def handle_assess_relevance(self, message, context_id):
                return {"status": "success", "relevance": 0.85}
            
            async def optimize_context_window(self, data):
                return {"status": "success", "optimized": True}
            
            async def semantic_memory_operations(self, data):
                return {"status": "success", "stored": True}
            
            async def generate_context_template(self, data):
                return {"status": "success", "template": {}}
            
            async def assess_context_quality(self, data):
                return {"status": "success", "quality": 0.9}
            
            async def handle_coordinate_context(self, message, context_id):
                return {"status": "success", "synchronized": True}
        
        # Create orchestrator
        agent = MockContextAgent()
        orchestrator = ProcessOrchestrator(agent)
        
        # Execute process
        process_id = str(uuid.uuid4())
        initial_data = {
            "message": {"content": {"context": "Test context"}},
            "context_id": "test_context_123"
        }
        
        result = await orchestrator.execute_process(process_id, initial_data)
        
        # Get process status
        status = orchestrator.get_process_status(process_id)
        print(f"Process Status: {json.dumps(status, indent=2)}")
        
        # Visualize process
        visualization = orchestrator.visualize_process(process_id)
        print(f"Process has {len(visualization['nodes'])} nodes and {len(visualization['edges'])} edges")
    
    # Run example
    import json
    asyncio.run(example_orchestration())