"""
Autonomous Decision-Making Framework for A2A Agents
Part of Phase 1: Core AI Framework

This module provides autonomous decision-making capabilities including
planning, goal-oriented behavior, and self-directed actions.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)


# Helper functions for lambda replacements
def get_action_efficiency_ratio(a):
    """Get efficiency ratio for action selection"""
    return a.success_probability / a.cost

def get_goal_action_value(ag):
    """Get goal action value for sorting - ag is tuple (action, goal)"""
    # This is a simple replacement - the full calculation would need the instance method
    # For basic sorting, use action success probability
    return ag[0].success_probability

def get_goal_priority_score(g):
    """Get goal priority score"""
    return g.importance * g.urgency

def get_action_success_probability(a):
    """Get action success probability"""
    return a.success_probability

def get_action_safe_efficiency_ratio(a):
    """Get safe efficiency ratio for action selection"""
    return a.success_probability / (a.cost + 0.1)

def get_action_count(x):
    """Get action count for statistics"""
    return x[1]


class DecisionType(str, Enum):
    """Types of autonomous decisions"""

    REACTIVE = "reactive"  # Response to immediate stimuli
    PROACTIVE = "proactive"  # Anticipatory decisions
    STRATEGIC = "strategic"  # Long-term planning
    TACTICAL = "tactical"  # Short-term execution
    COLLABORATIVE = "collaborative"  # Multi-agent coordination
    ADAPTIVE = "adaptive"  # Learning-based adaptation


class PlanningAlgorithm(str, Enum):
    """Planning algorithms available"""

    HIERARCHICAL_TASK_NETWORK = "htn"
    GOAL_ORIENTED_ACTION_PLANNING = "goap"
    MONTE_CARLO_TREE_SEARCH = "mcts"
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BEHAVIOR_TREES = "behavior_trees"


class DecisionPriority(str, Enum):
    """Priority levels for decisions"""

    CRITICAL = "critical"  # Must be executed immediately
    HIGH = "high"  # Important, execute soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Execute when resources available
    BACKGROUND = "background"  # Execute in idle time


class GoalStatus(str, Enum):
    """Status of goals"""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class Goal:
    """Represents an autonomous goal"""

    goal_id: str
    description: str
    target_state: Dict[str, Any]
    priority: DecisionPriority
    deadline: Optional[datetime] = None
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Represents an autonomous action"""

    action_id: str
    name: str
    description: str
    preconditions: Dict[str, Any]
    effects: Dict[str, Any]
    cost: float
    duration: float
    success_probability: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_function: Optional[Callable] = None


@dataclass
class Plan:
    """Represents an execution plan"""

    plan_id: str
    goal_id: str
    actions: List[Action]
    total_cost: float
    estimated_duration: float
    success_probability: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    algorithm_used: PlanningAlgorithm = PlanningAlgorithm.FORWARD_CHAINING


@dataclass
class DecisionContext:
    """Context for decision making"""

    current_state: Dict[str, Any]
    available_actions: List[Action]
    active_goals: List[Goal]
    constraints: Dict[str, Any]
    resources: Dict[str, Any]
    time_horizon: float
    risk_tolerance: float


class AutonomousDecisionFramework:
    """
    Comprehensive autonomous decision-making framework
    Provides goal-oriented planning and execution capabilities
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Goal and plan management
        self.goals = {}
        self.active_plans = {}
        self.completed_plans = deque(maxlen=1000)
        self.action_library = {}

        # Decision making components
        self.planners = {
            PlanningAlgorithm.HIERARCHICAL_TASK_NETWORK: HTNPlanner(),
            PlanningAlgorithm.GOAL_ORIENTED_ACTION_PLANNING: GOAPPlanner(),
            PlanningAlgorithm.MONTE_CARLO_TREE_SEARCH: MCTSPlanner(),
            PlanningAlgorithm.FORWARD_CHAINING: ForwardChainingPlanner(),
            PlanningAlgorithm.BACKWARD_CHAINING: BackwardChainingPlanner(),
            PlanningAlgorithm.BEHAVIOR_TREES: BehaviorTreePlanner(),
        }

        # Execution engine
        self.execution_engine = ActionExecutionEngine(self)

        # Decision history
        self.decision_history = deque(maxlen=5000)

        # State management
        self.current_state = {}
        self.state_history = deque(maxlen=1000)

        # Resource management
        self.resources = {
            "cpu_budget": 1.0,
            "memory_budget": 1.0,
            "time_budget": 3600,  # 1 hour default
            "energy_budget": 1.0,
        }

        # Learning components
        self.action_success_rates = defaultdict(list)
        self.goal_achievement_patterns = defaultdict(list)

        # Monitoring
        self.performance_metrics = {
            "goals_completed": 0,
            "goals_failed": 0,
            "average_plan_success_rate": 0.0,
            "average_goal_completion_time": 0.0,
            "resource_utilization": 0.0,
        }

        logger.info(f"Initialized autonomous decision framework for agent {agent_id}")

    async def add_goal(
        self,
        description: str,
        target_state: Dict[str, Any],
        priority: DecisionPriority = DecisionPriority.MEDIUM,
        deadline: Optional[datetime] = None,
        prerequisites: Optional[List[str]] = None,
    ) -> Goal:
        """
        Add a new autonomous goal

        Args:
            description: Human-readable goal description
            target_state: Desired state to achieve
            priority: Goal priority level
            deadline: Optional deadline for goal completion
            prerequisites: Optional list of prerequisite goal IDs

        Returns:
            Created goal object
        """
        goal_id = str(uuid.uuid4())

        goal = Goal(
            goal_id=goal_id,
            description=description,
            target_state=target_state,
            priority=priority,
            deadline=deadline,
            prerequisites=prerequisites or [],
        )

        self.goals[goal_id] = goal

        # Automatically start planning if no prerequisites
        if not goal.prerequisites:
            await self._activate_goal(goal_id)

        logger.info(f"Added goal {goal_id}: {description}")
        return goal

    async def make_autonomous_decision(
        self,
        context: Optional[DecisionContext] = None,
        decision_type: DecisionType = DecisionType.REACTIVE,
    ) -> Dict[str, Any]:
        """
        Make an autonomous decision based on current context

        Args:
            context: Decision context (if None, will be inferred)
            decision_type: Type of decision to make

        Returns:
            Decision result with chosen actions
        """
        if context is None:
            context = await self._build_decision_context()

        decision_id = str(uuid.uuid4())
        decision_start = datetime.utcnow()

        try:
            # Select decision strategy based on type
            if decision_type == DecisionType.REACTIVE:
                decision = await self._make_reactive_decision(context)
            elif decision_type == DecisionType.PROACTIVE:
                decision = await self._make_proactive_decision(context)
            elif decision_type == DecisionType.STRATEGIC:
                decision = await self._make_strategic_decision(context)
            elif decision_type == DecisionType.TACTICAL:
                decision = await self._make_tactical_decision(context)
            elif decision_type == DecisionType.COLLABORATIVE:
                decision = await self._make_collaborative_decision(context)
            elif decision_type == DecisionType.ADAPTIVE:
                decision = await self._make_adaptive_decision(context)
            else:
                decision = await self._make_default_decision(context)

            decision_duration = (datetime.utcnow() - decision_start).total_seconds()

            # Record decision
            decision_record = {
                "decision_id": decision_id,
                "type": decision_type.value,
                "context": context.current_state,
                "decision": decision,
                "duration": decision_duration,
                "timestamp": decision_start,
            }
            self.decision_history.append(decision_record)

            logger.info(
                f"Made {decision_type.value} decision {decision_id} in {decision_duration:.3f}s"
            )
            return decision

        except Exception as e:
            logger.error(f"Failed to make autonomous decision: {e}")
            return {
                "success": False,
                "error": str(e),
                "decision_id": decision_id,
                "fallback_action": "maintain_current_state",
            }

    async def create_plan(
        self, goal_id: str, algorithm: PlanningAlgorithm = PlanningAlgorithm.FORWARD_CHAINING
    ) -> Plan:
        """
        Create execution plan for a goal

        Args:
            goal_id: ID of goal to plan for
            algorithm: Planning algorithm to use

        Returns:
            Generated execution plan
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")

        goal = self.goals[goal_id]
        planner = self.planners.get(algorithm)

        if not planner:
            raise ValueError(f"Unknown planning algorithm: {algorithm}")

        # Create planning context
        planning_context = {
            "current_state": self.current_state,
            "target_state": goal.target_state,
            "available_actions": list(self.action_library.values()),
            "constraints": {
                "deadline": goal.deadline,
                "resources": self.resources,
                "priority": goal.priority,
            },
        }

        # Generate plan
        plan = await planner.create_plan(goal, planning_context)

        if plan:
            self.active_plans[plan.plan_id] = plan
            logger.info(f"Created plan {plan.plan_id} for goal {goal_id} using {algorithm.value}")

        return plan

    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute a plan autonomously

        Args:
            plan_id: ID of plan to execute

        Returns:
            Execution result
        """
        if plan_id not in self.active_plans:
            raise ValueError(f"Plan {plan_id} not found in active plans")

        plan = self.active_plans[plan_id]

        execution_result = await self.execution_engine.execute_plan(plan)

        # Update goal progress
        goal = self.goals[plan.goal_id]
        if execution_result["success"]:
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            self.performance_metrics["goals_completed"] += 1
        else:
            goal.status = GoalStatus.FAILED
            self.performance_metrics["goals_failed"] += 1

        goal.updated_at = datetime.utcnow()

        # Move plan to completed
        self.completed_plans.append(plan)
        del self.active_plans[plan_id]

        return execution_result

    async def register_action(self, action: Action):
        """
        Register an action in the action library

        Args:
            action: Action to register
        """
        self.action_library[action.action_id] = action
        logger.debug(f"Registered action {action.action_id}: {action.name}")

    async def update_state(self, state_updates: Dict[str, Any]):
        """
        Update current state and trigger autonomous decisions if needed

        Args:
            state_updates: Updates to apply to current state
        """
        # Store previous state
        previous_state = self.current_state.copy()
        self.state_history.append({"state": previous_state, "timestamp": datetime.utcnow()})

        # Apply updates
        self.current_state.update(state_updates)

        # Check if any goals need attention
        await self._evaluate_goals_after_state_change(previous_state, self.current_state)

        # Consider autonomous actions
        await self._consider_autonomous_actions()

    async def optimize_decision_making(self) -> Dict[str, Any]:
        """
        Optimize decision-making based on historical performance

        Returns:
            Optimization results
        """
        optimization_results = {
            "actions_optimized": 0,
            "goals_rebalanced": 0,
            "resource_allocation_improved": False,
            "planning_algorithms_tuned": 0,
        }

        # Optimize action success rates
        for action_id, success_rates in self.action_success_rates.items():
            if len(success_rates) >= 10:
                current_rate = np.mean(success_rates[-10:])
                historical_rate = np.mean(success_rates)

                if current_rate < historical_rate * 0.8:
                    # Action performance degraded, adjust parameters
                    action = self.action_library.get(action_id)
                    if action:
                        action.success_probability = current_rate
                        optimization_results["actions_optimized"] += 1

        # Rebalance goal priorities based on success patterns
        priority_success_rates = defaultdict(list)
        for record in self.decision_history:
            if "goal_completed" in record.get("decision", {}):
                goal_id = record["decision"]["goal_id"]
                goal = self.goals.get(goal_id)
                if goal:
                    priority_success_rates[goal.priority.value].append(
                        1 if goal.status == GoalStatus.COMPLETED else 0
                    )

        # Adjust resource allocation
        resource_utilization = self._calculate_resource_utilization()
        if resource_utilization < 0.7:
            # Underutilizing resources, can take on more goals
            self.resources["time_budget"] *= 1.1
            optimization_results["resource_allocation_improved"] = True
        elif resource_utilization > 0.9:
            # Overutilizing resources, need to be more selective
            self.resources["time_budget"] *= 0.9
            optimization_results["resource_allocation_improved"] = True

        # Tune planning algorithms based on success rates
        algorithm_performance = defaultdict(list)
        for plan in self.completed_plans:
            success = (
                1
                if any(
                    g.status == GoalStatus.COMPLETED
                    for g in self.goals.values()
                    if g.goal_id == plan.goal_id
                )
                else 0
            )
            algorithm_performance[plan.algorithm_used.value].append(success)

        # Update planner preferences
        for algorithm, success_rates in algorithm_performance.items():
            if len(success_rates) >= 5:
                avg_success = np.mean(success_rates)
                planner = self.planners.get(PlanningAlgorithm(algorithm))
                if planner and hasattr(planner, "adjust_parameters"):
                    planner.adjust_parameters(avg_success)
                    optimization_results["planning_algorithms_tuned"] += 1

        logger.info(f"Optimization complete: {optimization_results}")
        return optimization_results

    def get_autonomy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive autonomy statistics"""
        total_goals = len(self.goals)
        completed_goals = sum(1 for g in self.goals.values() if g.status == GoalStatus.COMPLETED)

        stats = {
            "goal_statistics": {
                "total_goals": total_goals,
                "completed_goals": completed_goals,
                "active_goals": sum(
                    1 for g in self.goals.values() if g.status == GoalStatus.ACTIVE
                ),
                "failed_goals": sum(
                    1 for g in self.goals.values() if g.status == GoalStatus.FAILED
                ),
                "completion_rate": completed_goals / total_goals if total_goals > 0 else 0,
            },
            "planning_statistics": {
                "total_plans": len(self.active_plans) + len(self.completed_plans),
                "active_plans": len(self.active_plans),
                "completed_plans": len(self.completed_plans),
                "average_plan_success_rate": self.performance_metrics["average_plan_success_rate"],
            },
            "decision_statistics": {
                "total_decisions": len(self.decision_history),
                "decisions_per_hour": len(
                    [
                        d
                        for d in self.decision_history
                        if (datetime.utcnow() - d["timestamp"]).total_seconds() < 3600
                    ]
                ),
                "average_decision_time": (
                    np.mean([d["duration"] for d in self.decision_history])
                    if self.decision_history
                    else 0
                ),
            },
            "resource_utilization": self._calculate_resource_utilization(),
            "action_library_size": len(self.action_library),
            "state_complexity": len(self.current_state),
        }

        return stats

    async def _activate_goal(self, goal_id: str):
        """Activate a goal and start working on it"""
        goal = self.goals[goal_id]
        goal.status = GoalStatus.ACTIVE
        goal.updated_at = datetime.utcnow()

        # Create and execute plan
        try:
            plan = await self.create_plan(goal_id)
            if plan:
                # Execute plan asynchronously
                asyncio.create_task(self.execute_plan(plan.plan_id))
        except Exception as e:
            logger.error(f"Failed to activate goal {goal_id}: {e}")
            goal.status = GoalStatus.FAILED

    async def _build_decision_context(self) -> DecisionContext:
        """Build decision context from current state"""
        return DecisionContext(
            current_state=self.current_state,
            available_actions=list(self.action_library.values()),
            active_goals=[g for g in self.goals.values() if g.status == GoalStatus.ACTIVE],
            constraints={"deadline": datetime.utcnow() + timedelta(hours=1)},
            resources=self.resources,
            time_horizon=3600,  # 1 hour
            risk_tolerance=0.5,
        )

    async def _make_reactive_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make reactive decision to immediate stimuli"""
        # Find immediate threats or opportunities
        urgent_actions = []

        for action in context.available_actions:
            # Check if action addresses immediate needs
            if action.success_probability > 0.8 and action.cost < 0.5:
                urgent_actions.append(action)

        if urgent_actions:
            # Select best immediate action
            best_action = max(urgent_actions, key=get_action_efficiency_ratio)
            return {
                "type": "reactive",
                "selected_action": best_action.action_id,
                "reasoning": "Immediate response to stimuli",
                "success": True,
            }

        return {"type": "reactive", "action": "observe", "success": True}

    async def _make_proactive_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make proactive decision to anticipate future needs"""
        # Look for opportunities to prepare for future goals
        preparation_actions = []

        for goal in context.active_goals:
            for action in context.available_actions:
                # Check if action moves us toward goal
                if self._action_supports_goal(action, goal):
                    preparation_actions.append((action, goal))

        if preparation_actions:
            # Select action with best goal alignment
            def action_goal_value_func(ag):
                return self._calculate_goal_action_value(ag[1], ag[0])
            
            best_action, best_goal = max(
                preparation_actions, key=action_goal_value_func
            )
            return {
                "type": "proactive",
                "selected_action": best_action.action_id,
                "target_goal": best_goal.goal_id,
                "reasoning": "Proactive preparation for future goal",
                "success": True,
            }

        return {"type": "proactive", "action": "explore_opportunities", "success": True}

    async def _make_strategic_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make strategic long-term decision"""
        # Analyze long-term goal portfolio
        high_priority_goals = [
            g
            for g in context.active_goals
            if g.priority in [DecisionPriority.CRITICAL, DecisionPriority.HIGH]
        ]

        if high_priority_goals:
            # Focus on highest priority goals
            priority_goal = max(
                high_priority_goals, key=get_goal_priority_score
            )

            # Find actions that significantly advance this goal
            strategic_actions = [
                a
                for a in context.available_actions
                if self._action_supports_goal(a, priority_goal) and a.success_probability > 0.6
            ]

            if strategic_actions:
                best_action = max(strategic_actions, key=get_action_success_probability)
                return {
                    "type": "strategic",
                    "selected_action": best_action.action_id,
                    "strategic_goal": priority_goal.goal_id,
                    "reasoning": "Long-term strategic advancement",
                    "success": True,
                }

        return {"type": "strategic", "action": "reassess_priorities", "success": True}

    async def _make_tactical_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make tactical short-term execution decision"""
        # Focus on immediate execution steps
        executable_actions = [
            a
            for a in context.available_actions
            if a.cost
            <= context.resources.get("time_budget", 0) / 10  # Use at most 10% of time budget
        ]

        if executable_actions:
            # Select action with best immediate return
            best_action = max(
                executable_actions, key=get_action_safe_efficiency_ratio
            )
            return {
                "type": "tactical",
                "selected_action": best_action.action_id,
                "reasoning": "Tactical execution step",
                "success": True,
            }

        return {"type": "tactical", "action": "wait_for_resources", "success": True}

    async def _make_collaborative_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make collaborative decision involving other agents"""
        # Look for opportunities to collaborate
        collaboration_actions = [
            a
            for a in context.available_actions
            if "collaborate" in a.name.lower() or "coordinate" in a.name.lower()
        ]

        if collaboration_actions:
            best_collaboration = max(collaboration_actions, key=get_action_success_probability)
            return {
                "type": "collaborative",
                "selected_action": best_collaboration.action_id,
                "reasoning": "Collaborative coordination",
                "success": True,
            }

        return {"type": "collaborative", "action": "seek_collaboration", "success": True}

    async def _make_adaptive_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make adaptive decision based on learning"""
        # Use learning from past decisions
        recent_successes = [
            d for d in self.decision_history[-50:] if d.get("decision", {}).get("success", False)
        ]

        if recent_successes:
            # Find patterns in successful decisions
            successful_actions = [
                d["decision"].get("selected_action")
                for d in recent_successes
                if "selected_action" in d["decision"]
            ]
            action_counts = defaultdict(int)
            for action_id in successful_actions:
                action_counts[action_id] += 1

            # Prefer previously successful actions
            if action_counts:
                most_successful_action_id = max(action_counts.items(), key=get_action_count)[0]
                if most_successful_action_id in self.action_library:
                    return {
                        "type": "adaptive",
                        "selected_action": most_successful_action_id,
                        "reasoning": "Adapted from successful past decisions",
                        "success": True,
                    }

        return {"type": "adaptive", "action": "explore_new_strategies", "success": True}

    async def _make_default_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """Make default decision when no specific strategy applies"""
        return {
            "type": "default",
            "action": "maintain_status_quo",
            "reasoning": "Default fallback decision",
            "success": True,
        }

    async def _evaluate_goals_after_state_change(
        self, previous_state: Dict[str, Any], new_state: Dict[str, Any]
    ):
        """Evaluate goals after state change"""
        for goal in self.goals.values():
            if goal.status == GoalStatus.ACTIVE:
                # Check if goal is now achievable or if progress changed
                progress = self._calculate_goal_progress(goal, new_state)
                if progress != goal.progress:
                    goal.progress = progress
                    goal.updated_at = datetime.utcnow()

                    if progress >= 1.0:
                        goal.status = GoalStatus.COMPLETED
                        self.performance_metrics["goals_completed"] += 1

    async def _consider_autonomous_actions(self):
        """Consider taking autonomous actions based on current state"""
        # Check if any immediate actions are warranted
        decision = await self.make_autonomous_decision(decision_type=DecisionType.REACTIVE)

        if decision.get("success") and "selected_action" in decision:
            action_id = decision["selected_action"]
            if action_id in self.action_library:
                # Execute the action
                action = self.action_library[action_id]
                await self.execution_engine.execute_action(action)

    def _action_supports_goal(self, action: Action, goal: Goal) -> bool:
        """Check if action supports achieving a goal"""
        # Simplified check - in practice would be more sophisticated
        for effect_key, effect_value in action.effects.items():
            if effect_key in goal.target_state:
                target_value = goal.target_state[effect_key]
                if effect_value == target_value or (
                    isinstance(effect_value, (int, float))
                    and isinstance(target_value, (int, float))
                    and abs(effect_value - target_value) < 0.1
                ):
                    return True
        return False

    def _calculate_goal_action_value(self, goal: Goal, action: Action) -> float:
        """Calculate value of action for a goal"""
        base_value = action.success_probability / (action.cost + 0.1)

        # Adjust for goal priority
        priority_multiplier = {
            DecisionPriority.CRITICAL: 5.0,
            DecisionPriority.HIGH: 3.0,
            DecisionPriority.MEDIUM: 1.0,
            DecisionPriority.LOW: 0.5,
            DecisionPriority.BACKGROUND: 0.1,
        }.get(goal.priority, 1.0)

        return base_value * priority_multiplier

    def _calculate_goal_priority_score(self, goal: Goal) -> float:
        """Calculate priority score for a goal"""
        base_score = {
            DecisionPriority.CRITICAL: 100,
            DecisionPriority.HIGH: 75,
            DecisionPriority.MEDIUM: 50,
            DecisionPriority.LOW: 25,
            DecisionPriority.BACKGROUND: 10,
        }.get(goal.priority, 50)

        # Adjust for deadline urgency
        if goal.deadline:
            time_to_deadline = (goal.deadline - datetime.utcnow()).total_seconds()
            if time_to_deadline < 3600:  # Less than 1 hour
                base_score *= 2.0
            elif time_to_deadline < 86400:  # Less than 1 day
                base_score *= 1.5

        return base_score

    def _calculate_goal_progress(self, goal: Goal, current_state: Dict[str, Any]) -> float:
        """Calculate progress toward a goal"""
        if not goal.target_state:
            return 0.0

        matches = 0
        total = len(goal.target_state)

        for key, target_value in goal.target_state.items():
            current_value = current_state.get(key)
            if current_value == target_value:
                matches += 1
            elif isinstance(current_value, (int, float)) and isinstance(target_value, (int, float)):
                # Partial credit for numerical values
                if abs(current_value - target_value) < abs(target_value) * 0.1:
                    matches += 0.8
                elif abs(current_value - target_value) < abs(target_value) * 0.5:
                    matches += 0.5

        return matches / total

    def _calculate_resource_utilization(self) -> float:
        """Calculate current resource utilization"""
        # Simplified calculation
        active_plans_cost = sum(plan.total_cost for plan in self.active_plans.values())
        total_budget = sum(self.resources.values())

        return min(1.0, active_plans_cost / (total_budget + 0.1))


class ActionExecutionEngine:
    """Engine for executing actions and plans"""

    def __init__(self, framework: AutonomousDecisionFramework):
        self.framework = framework
        self.execution_queue = asyncio.PriorityQueue()
        self.executing_actions = {}

    async def execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute a single action"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Check preconditions
            if not self._check_preconditions(action):
                return {
                    "success": False,
                    "error": "Preconditions not met",
                    "action_id": action.action_id,
                    "execution_id": execution_id,
                }

            # Execute action
            if action.execution_function:
                result = await action.execution_function(action.parameters)
            else:
                # Simulate execution
                await asyncio.sleep(action.duration / 1000)  # Convert to seconds
                result = {"simulated": True, "effects_applied": action.effects}

            # Apply effects to state
            self.framework.current_state.update(action.effects)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Record success
            self.framework.action_success_rates[action.action_id].append(1.0)

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "execution_id": execution_id,
                "action_id": action.action_id,
            }

        except Exception as e:
            # Record failure
            self.framework.action_success_rates[action.action_id].append(0.0)

            return {"success": False, "error": str(e), "action_id": action.action_id}

    async def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a complete plan"""
        execution_results = []
        overall_success = True

        for action in plan.actions:
            result = await self.execute_action(action)
            execution_results.append(result)

            if not result["success"]:
                overall_success = False
                break  # Stop on first failure

        return {
            "success": overall_success,
            "plan_id": plan.plan_id,
            "execution_results": execution_results,
            "actions_completed": len([r for r in execution_results if r["success"]]),
        }

    def _check_preconditions(self, action: Action) -> bool:
        """Check if action preconditions are met"""
        current_state = self.framework.current_state

        for key, required_value in action.preconditions.items():
            current_value = current_state.get(key)
            if current_value != required_value:
                return False

        return True


# Planning Algorithm Implementations
class Planner(ABC):
    """Base class for planning algorithms"""

    @abstractmethod
    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        """Create a plan to achieve the goal"""


class ForwardChainingPlanner(Planner):
    """Forward chaining planner"""

    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        current_state = context["current_state"]
        target_state = goal.target_state
        available_actions = context["available_actions"]

        plan_actions = []
        current = current_state.copy()

        # Simple forward search
        for _ in range(10):  # Limit search depth
            # Find action that moves us closer to goal
            best_action = None
            best_score = -1

            for action in available_actions:
                if self._can_execute(action, current):
                    score = self._calculate_action_score(action, current, target_state)
                    if score > best_score:
                        best_score = score
                        best_action = action

            if best_action:
                plan_actions.append(best_action)
                # Apply action effects
                current.update(best_action.effects)

                # Check if goal achieved
                if self._goal_achieved(current, target_state):
                    break
            else:
                break

        if plan_actions:
            plan_id = str(uuid.uuid4())
            total_cost = sum(action.cost for action in plan_actions)
            total_duration = sum(action.duration for action in plan_actions)
            success_prob = np.prod([action.success_probability for action in plan_actions])

            return Plan(
                plan_id=plan_id,
                goal_id=goal.goal_id,
                actions=plan_actions,
                total_cost=total_cost,
                estimated_duration=total_duration,
                success_probability=success_prob,
                algorithm_used=PlanningAlgorithm.FORWARD_CHAINING,
            )

        return None

    def _can_execute(self, action: Action, state: Dict[str, Any]) -> bool:
        """Check if action can be executed in current state"""
        for key, value in action.preconditions.items():
            if state.get(key) != value:
                return False
        return True

    def _calculate_action_score(
        self, action: Action, current_state: Dict[str, Any], target_state: Dict[str, Any]
    ) -> float:
        """Calculate how good an action is for reaching target"""
        score = 0
        for key, target_value in target_state.items():
            if key in action.effects:
                effect_value = action.effects[key]
                if effect_value == target_value:
                    score += 1
                elif isinstance(effect_value, (int, float)) and isinstance(
                    target_value, (int, float)
                ):
                    # Closer is better
                    current_distance = abs(current_state.get(key, 0) - target_value)
                    effect_distance = abs(effect_value - target_value)
                    if effect_distance < current_distance:
                        score += 0.5

        return score / (action.cost + 0.1)

    def _goal_achieved(self, current_state: Dict[str, Any], target_state: Dict[str, Any]) -> bool:
        """Check if goal is achieved"""
        for key, target_value in target_state.items():
            if current_state.get(key) != target_value:
                return False
        return True


class BackwardChainingPlanner(Planner):
    """Backward chaining planner"""

    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        # Simplified backward chaining
        target_state = goal.target_state
        available_actions = context["available_actions"]

        # Find actions that achieve goal conditions
        relevant_actions = []
        for action in available_actions:
            for effect_key in action.effects:
                if effect_key in target_state:
                    relevant_actions.append(action)
                    break

        if relevant_actions:
            # Select best action
            best_action = max(relevant_actions, key=get_action_success_probability)

            plan_id = str(uuid.uuid4())
            return Plan(
                plan_id=plan_id,
                goal_id=goal.goal_id,
                actions=[best_action],
                total_cost=best_action.cost,
                estimated_duration=best_action.duration,
                success_probability=best_action.success_probability,
                algorithm_used=PlanningAlgorithm.BACKWARD_CHAINING,
            )

        return None


# Simplified implementations for other planners
class HTNPlanner(Planner):
    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        # Simplified HTN planning
        return None  # Not implemented in this example


class GOAPPlanner(Planner):
    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        # Simplified GOAP planning
        return None  # Not implemented in this example


class MCTSPlanner(Planner):
    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        # Simplified MCTS planning
        return None  # Not implemented in this example


class BehaviorTreePlanner(Planner):
    async def create_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Plan]:
        # Simplified behavior tree planning
        return None  # Not implemented in this example


# Utility functions
def create_autonomous_decision_framework(agent_id: str) -> AutonomousDecisionFramework:
    """Factory function to create autonomous decision framework"""
    return AutonomousDecisionFramework(agent_id)


async def add_autonomous_goal(
    framework: AutonomousDecisionFramework,
    description: str,
    target_state: Dict[str, Any],
    priority: DecisionPriority = DecisionPriority.MEDIUM,
) -> Goal:
    """Convenience function for adding autonomous goals"""
    return await framework.add_goal(description, target_state, priority)


async def make_decision(
    framework: AutonomousDecisionFramework, decision_type: DecisionType = DecisionType.REACTIVE
) -> Dict[str, Any]:
    """Convenience function for making autonomous decisions"""
    return await framework.make_autonomous_decision(decision_type=decision_type)
