"""
Swarm Intelligence Architecture
Implements collective reasoning using emergent behavior with MCP tools
"""

import asyncio
import random
import secrets
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# MCP decorators for inter-agent communication only
def mcp_tool(name: str, description: str):
    """MCP tool decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_tool = {"name": name, "description": description}
        return func
    return decorator

def mcp_resource(uri: str, description: str):
    """MCP resource decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_resource = {"uri": uri, "description": description}
        return func
    return decorator

def mcp_prompt(name: str, description: str):
    """MCP prompt decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_prompt = {"name": name, "description": description}
        return func
    return decorator

logger = logging.getLogger(__name__)


class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    EXPLORATION = "exploration"  # Agents explore solution space
    EXPLOITATION = "exploitation"  # Agents focus on promising areas
    MIGRATION = "migration"  # Agents move between solution clusters
    CONVERGENCE = "convergence"  # Agents converge on consensus


@dataclass
class Pheromone:
    """Pheromone trail for swarm communication"""
    trail_id: str
    strength: float
    source_agent: str
    content: Dict[str, Any]
    timestamp: datetime

    def evaporate(self, rate: float = 0.1):
        """Evaporate pheromone over time"""
        self.strength *= (1 - rate)

    def is_active(self) -> bool:
        """Check if pheromone is still active"""
        return self.strength > 0.1


class SwarmAgent:
    """Individual agent in the swarm"""

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        self.agent_id = agent_id
        self.position = position  # Position in solution space
        self.velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_fitness = 0.0
        self.current_fitness = 0.0
        self.memory: List[Dict[str, Any]] = []
        self.neighbors: Set[str] = set()

    def move(self, global_best: Tuple[float, float], inertia: float = 0.7):
        """Move agent in solution space"""
        # Particle swarm optimization movement
        r1, r2 = secrets.SystemRandom().random(), secrets.SystemRandom().random()
        cognitive_weight = 1.5
        social_weight = 1.5

        # Update velocity
        if self.best_solution:
            personal_best = self.best_solution.get("position", self.position)
        else:
            personal_best = self.position

        self.velocity = (
            inertia * self.velocity[0] +
            cognitive_weight * r1 * (personal_best[0] - self.position[0]) +
            social_weight * r2 * (global_best[0] - self.position[0]),

            inertia * self.velocity[1] +
            cognitive_weight * r1 * (personal_best[1] - self.position[1]) +
            social_weight * r2 * (global_best[1] - self.position[1])
        )

        # Update position
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )

    def evaluate_solution(self, question: str, context: Dict[str, Any]) -> float:
        """Evaluate fitness of current position as solution"""
        # Simple fitness based on position (in real implementation,
        # this would evaluate actual solution quality)
        base_fitness = 1.0 / (1.0 + abs(self.position[0]) + abs(self.position[1]))

        # Add randomness for exploration
        noise = random.uniform(0.8, 1.2)

        return base_fitness * noise

    def share_knowledge(self, pheromone: Pheromone):
        """Process pheromone from other agents"""
        if pheromone.strength > 0.5:
            self.memory.append({
                "source": pheromone.source_agent,
                "content": pheromone.content,
                "strength": pheromone.strength
            })

    def generate_solution(self, question: str) -> Dict[str, Any]:
        """Generate solution based on current position and memory"""
        # Combine position-based insights with memory
        insights = []

        # Position-based insight
        if abs(self.position[0]) < 1:
            insights.append("Balanced analytical approach")
        elif self.position[0] > 1:
            insights.append("Creative lateral thinking")
        else:
            insights.append("Critical systematic analysis")

        # Memory-based insights
        for mem in self.memory[-3:]:  # Last 3 memories
            if mem["strength"] > 0.7:
                insights.append(f"High-confidence insight from swarm")

        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "insights": insights,
            "confidence": self.current_fitness,
            "approach": self._determine_approach()
        }

    def _determine_approach(self) -> str:
        """Determine reasoning approach based on position"""
        x, y = self.position

        if x > 0 and y > 0:
            return "creative-analytical"
        elif x > 0 and y < 0:
            return "creative-practical"
        elif x < 0 and y > 0:
            return "systematic-theoretical"
        else:
            return "systematic-practical"


class SwarmIntelligenceCoordinator:
    """Coordinates swarm intelligence reasoning with direct communication"""

    def __init__(self, swarm_size: int = 20):
        self.swarm_size = swarm_size
        self.agents: Dict[str, SwarmAgent] = {}
        self.pheromone_trails: List[Pheromone] = []
        self.global_best_position = (0, 0)
        self.global_best_fitness = 0.0
        self.iteration_count = 0
        self.max_iterations = 50

        # Initialize swarm
        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initialize swarm agents with random positions"""
        for i in range(self.swarm_size):
            agent_id = f"swarm_agent_{i}"
            # Random position in solution space
            position = (
                random.uniform(-5, 5),
                random.uniform(-5, 5)
            )
            agent = SwarmAgent(agent_id, position)
            self.agents[agent_id] = agent

            # Create neighborhood connections
            if i > 0:
                # Connect to some previous agents
                num_neighbors = min(3, i)
                neighbor_indices = random.sample(range(i), num_neighbors)
                for idx in neighbor_indices:
                    neighbor_id = f"swarm_agent_{idx}"
                    agent.neighbors.add(neighbor_id)
                    self.agents[neighbor_id].neighbors.add(agent_id)

    @mcp_tool(
        name="swarm_intelligence_reasoning",
        description="Collective reasoning using swarm intelligence"
    )
    async def reason(
        self,
        question: str,
        context: Dict[str, Any] = None,
        behavior: SwarmBehavior = SwarmBehavior.EXPLORATION
    ) -> Dict[str, Any]:
        """Execute swarm intelligence reasoning"""
        start_time = datetime.utcnow()

        # Phase 1: Initialize swarm with question
        await self._initialize_reasoning(question, context)

        # Phase 2: Swarm iterations
        iteration_results = []
        for iteration in range(self.max_iterations):
            if iteration < self.max_iterations * 0.3:
                current_behavior = SwarmBehavior.EXPLORATION
            elif iteration < self.max_iterations * 0.7:
                current_behavior = SwarmBehavior.EXPLOITATION
            else:
                current_behavior = SwarmBehavior.CONVERGENCE

            iteration_result = await self._swarm_iteration(
                question, context, current_behavior, iteration
            )
            iteration_results.append(iteration_result)

            # Check for early convergence
            if self._check_convergence(iteration_results):
                break

        # Phase 3: Synthesize collective solution
        final_solution = await self._synthesize_swarm_solution(
            question, iteration_results
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "answer": final_solution["answer"],
            "reasoning_type": "swarm_intelligence",
            "swarm_size": self.swarm_size,
            "iterations": len(iteration_results),
            "convergence_achieved": self._check_convergence(iteration_results),
            "confidence": final_solution["confidence"],
            "execution_time": execution_time,
            "global_best_fitness": self.global_best_fitness,
            "diverse_solutions": final_solution["diverse_solutions"]
        }

    async def _initialize_reasoning(self, question: str, context: Dict[str, Any]):
        """Initialize swarm for reasoning task"""
        # Reset global best
        self.global_best_fitness = 0.0
        self.global_best_position = (0, 0)

        # Clear old pheromones
        self.pheromone_trails.clear()

        # Initialize agents with question context
        for agent in self.agents.values():
            agent.memory.clear()
            agent.best_solution = None
            agent.best_fitness = 0.0

            # Add initial context to memory
            if context:
                agent.memory.append({
                    "type": "context",
                    "content": context,
                    "strength": 1.0
                })

    async def _swarm_iteration(
        self,
        question: str,
        context: Dict[str, Any],
        behavior: SwarmBehavior,
        iteration: int
    ) -> Dict[str, Any]:
        """Execute one swarm iteration"""
        iteration_solutions = []

        # Each agent evaluates and moves
        agent_tasks = []
        for agent in self.agents.values():
            task = self._agent_iteration(agent, question, context, behavior)
            agent_tasks.append(task)

        # Execute all agents in parallel
        agent_results = await asyncio.gather(*agent_tasks)

        # Process results and update swarm state
        for agent, result in zip(self.agents.values(), agent_results):
            iteration_solutions.append(result["solution"])

            # Update personal best
            if result["fitness"] > agent.best_fitness:
                agent.best_fitness = result["fitness"]
                agent.best_solution = result["solution"]

            # Update global best
            if result["fitness"] > self.global_best_fitness:
                self.global_best_fitness = result["fitness"]
                self.global_best_position = agent.position

            # Create pheromone trail for good solutions
            if result["fitness"] > 0.7:
                pheromone = Pheromone(
                    trail_id=f"trail_{iteration}_{agent.agent_id}",
                    strength=result["fitness"],
                    source_agent=agent.agent_id,
                    content=result["solution"],
                    timestamp=datetime.utcnow()
                )
                self.pheromone_trails.append(pheromone)

        # Evaporate old pheromones
        for pheromone in self.pheromone_trails:
            pheromone.evaporate()
        self.pheromone_trails = [p for p in self.pheromone_trails if p.is_active()]

        # Share pheromones among neighbors
        await self._share_pheromones()

        return {
            "iteration": iteration,
            "behavior": behavior.value,
            "best_fitness": self.global_best_fitness,
            "active_pheromones": len(self.pheromone_trails),
            "solutions": iteration_solutions
        }

    async def _agent_iteration(
        self,
        agent: SwarmAgent,
        question: str,
        context: Dict[str, Any],
        behavior: SwarmBehavior
    ) -> Dict[str, Any]:
        """Single agent iteration"""
        # Evaluate current position
        fitness = agent.evaluate_solution(question, context)
        agent.current_fitness = fitness

        # Generate solution
        solution = agent.generate_solution(question)

        # Move based on behavior
        if behavior == SwarmBehavior.EXPLORATION:
            # More random movement
            agent.move(self.global_best_position, inertia=0.9)
        elif behavior == SwarmBehavior.EXPLOITATION:
            # Focus on best solutions
            agent.move(self.global_best_position, inertia=0.5)
        elif behavior == SwarmBehavior.CONVERGENCE:
            # Strong attraction to global best
            agent.move(self.global_best_position, inertia=0.3)

        return {
            "agent_id": agent.agent_id,
            "fitness": fitness,
            "solution": solution
        }

    async def _share_pheromones(self):
        """Share pheromone trails among neighboring agents"""
        for agent in self.agents.values():
            # Process pheromones from neighbors
            for neighbor_id in agent.neighbors:
                neighbor_pheromones = [
                    p for p in self.pheromone_trails
                    if p.source_agent == neighbor_id
                ]

                for pheromone in neighbor_pheromones:
                    agent.share_knowledge(pheromone)

    def _check_convergence(self, iteration_results: List[Dict[str, Any]]) -> bool:
        """Check if swarm has converged"""
        if len(iteration_results) < 5:
            return False

        # Check if fitness improvement has plateaued
        recent_fitness = [r["best_fitness"] for r in iteration_results[-5:]]
        fitness_variance = max(recent_fitness) - min(recent_fitness)

        return fitness_variance < 0.01

    async def _synthesize_swarm_solution(
        self,
        question: str,
        iteration_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize final solution from swarm intelligence"""
        # Collect all unique high-quality solutions
        all_solutions = []
        for iteration in iteration_results:
            for solution in iteration["solutions"]:
                if solution["confidence"] > 0.6:
                    all_solutions.append(solution)

        # Group by approach
        approach_groups = {}
        for solution in all_solutions:
            approach = solution.get("approach", "unknown")
            if approach not in approach_groups:
                approach_groups[approach] = []
            approach_groups[approach].append(solution)

        # Build comprehensive answer
        answer_parts = [f"Swarm intelligence analysis of '{question}':"]

        # Add insights from different approaches
        diverse_solutions = []
        for approach, solutions in approach_groups.items():
            if solutions:
                # Get best solution for this approach
                best_solution = max(solutions, key=lambda s: s["confidence"])
                diverse_solutions.append({
                    "approach": approach,
                    "insights": best_solution["insights"],
                    "confidence": best_solution["confidence"]
                })

                answer_parts.append(f"\n{approach.replace('-', ' ').title()} perspective:")
                for insight in best_solution["insights"][:2]:
                    answer_parts.append(f"- {insight}")

        # Add convergence summary
        answer_parts.append(f"\nSwarm convergence: {self.global_best_fitness:.2f}")
        answer_parts.append(f"Total iterations: {len(iteration_results)}")

        return {
            "answer": "\n".join(answer_parts),
            "confidence": self.global_best_fitness,
            "diverse_solutions": diverse_solutions
        }

    @mcp_resource(
        uri="swarm_state",
        description="Current state of the swarm"
    )
    async def get_swarm_state(self) -> Dict[str, Any]:
        """Get current swarm state"""
        agent_states = []

        for agent in self.agents.values():
            agent_states.append({
                "agent_id": agent.agent_id,
                "position": agent.position,
                "velocity": agent.velocity,
                "current_fitness": agent.current_fitness,
                "best_fitness": agent.best_fitness,
                "neighbors": list(agent.neighbors),
                "memory_size": len(agent.memory)
            })

        return {
            "swarm_size": self.swarm_size,
            "global_best_position": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "active_pheromones": len(self.pheromone_trails),
            "agents": agent_states
        }

    @mcp_prompt(
        name="visualize_swarm_positions",
        description="Visualize swarm agent positions in solution space"
    )
    async def visualize_positions(self) -> str:
        """Generate text visualization of swarm positions"""
        # Create simple ASCII visualization
        grid_size = 10
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        # Place agents on grid
        for agent in self.agents.values():
            x = int((agent.position[0] + 5) / 10 * (grid_size - 1))
            y = int((agent.position[1] + 5) / 10 * (grid_size - 1))

            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))

            if agent.best_fitness > 0.8:
                grid[y][x] = '*'  # High fitness
            elif agent.best_fitness > 0.5:
                grid[y][x] = 'o'  # Medium fitness
            else:
                grid[y][x] = '.'  # Low fitness

        # Build visualization
        viz_lines = ["Swarm Position Visualization (* = high fitness)"]
        viz_lines.append("+" + "-" * grid_size + "+")

        for row in grid:
            viz_lines.append("|" + "".join(row) + "|")

        viz_lines.append("+" + "-" * grid_size + "+")

        return "\n".join(viz_lines)


# Factory function
def create_swarm_intelligence_coordinator(swarm_size: int = 20) -> SwarmIntelligenceCoordinator:
    """Create a swarm intelligence coordinator"""
    return SwarmIntelligenceCoordinator(swarm_size)
