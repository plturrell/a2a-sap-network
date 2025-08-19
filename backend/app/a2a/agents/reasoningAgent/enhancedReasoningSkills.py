"""
Enhanced Reasoning Skills for A2A Reasoning Agent
Implements advanced internal reasoning capabilities including:
- Hierarchical reasoning with multiple internal reasoning engines
- Swarm intelligence algorithms for solution exploration
- Advanced debate mechanisms between reasoning chains
- Logical inference engine with knowledge representation
- Caching and parallel processing support
"""
import random

import asyncio
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import pickle
# Redis import - optional dependency
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task
from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
from app.a2a.core.trustIdentity import TrustIdentity

logger = logging.getLogger(__name__)


class InferenceRule:
    """Logical inference rule for reasoning"""
    def __init__(self, name: str, premises: List[str], conclusion: str, confidence: float = 1.0):
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.confidence = confidence
    
    def can_apply(self, facts: Set[str]) -> bool:
        """Check if rule can be applied given current facts"""
        return all(premise in facts for premise in self.premises)
    
    def apply(self, facts: Set[str]) -> Tuple[str, float]:
        """Apply rule and return conclusion with confidence"""
        if self.can_apply(facts):
            return self.conclusion, self.confidence
        return None, 0.0


@dataclass
class KnowledgeNode:
    """Node in knowledge representation graph"""
    node_id: str
    node_type: str  # concept, fact, rule, hypothesis
    content: Any
    relations: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [node_ids]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmAgent:
    """Individual agent in swarm intelligence system"""
    def __init__(self, agent_id: str, dimensions: int = 10, seed: int = None):
        self.agent_id = agent_id
        # Use deterministic initialization based on agent_id hash
        if seed is None:
            seed = abs(hash(agent_id)) % 10000
        np.random.seed(seed)
        self.position = np.random.rand(dimensions)
        self.velocity = np.random.rand(dimensions) * 0.1
        self.personal_best_position = self.position.copy()
        self.personal_best_score = -np.inf
        self.dimensions = dimensions
        self.seed = seed
        
    def update_velocity(self, global_best_position: np.ndarray, 
                       inertia: float = 0.7, cognitive: float = 1.5, social: float = 1.5):
        """Update velocity using PSO algorithm"""
        # Use deterministic random based on agent seed and iteration counter
        if not hasattr(self, '_iteration_counter'):
            self._iteration_counter = 0
        self._iteration_counter += 1
        
        np.random.seed(self.seed + self._iteration_counter)
        r1, r2 = np.random.rand(), np.random.rand()
        
        cognitive_component = cognitive * r1 * (self.personal_best_position - self.position)
        social_component = social * r2 * (global_best_position - self.position)
        
        self.velocity = inertia * self.velocity + cognitive_component + social_component
        
    def update_position(self):
        """Update position based on velocity"""
        self.position += self.velocity
        # Ensure position stays within bounds [0, 1]
        self.position = np.clip(self.position, 0, 1)


class EnhancedReasoningSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Enhanced reasoning skills with advanced algorithms and caching
    """
    
    def __init__(self, trust_identity: Optional[TrustIdentity] = None):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        
        # Knowledge representation graph
        self.knowledge_graph = nx.DiGraph()
        self.knowledge_nodes: Dict[str, KnowledgeNode] = {}
        
        # Inference engine
        self.inference_rules: List[InferenceRule] = self._initialize_inference_rules()
        self.facts: Set[str] = set()
        self.derived_facts: Set[Tuple[str, float]] = set()
        
        # Swarm intelligence
        self.swarm_agents: List[SwarmAgent] = []
        self.swarm_size = 20
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # Caching system
        self.reasoning_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=30)
        self.redis_client = None  # Will be initialized if Redis is available
        
        # Parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize swarm
        self._initialize_swarm()
        
    def _initialize_inference_rules(self) -> List[InferenceRule]:
        """Initialize basic inference rules"""
        return [
            # Modus Ponens
            InferenceRule("modus_ponens", ["A", "A->B"], "B", 0.95),
            # Modus Tollens
            InferenceRule("modus_tollens", ["!B", "A->B"], "!A", 0.95),
            # Hypothetical Syllogism
            InferenceRule("hypothetical_syllogism", ["A->B", "B->C"], "A->C", 0.9),
            # Disjunctive Syllogism
            InferenceRule("disjunctive_syllogism", ["A|B", "!A"], "B", 0.9),
            # Conjunction
            InferenceRule("conjunction", ["A", "B"], "A&B", 1.0),
            # Simplification
            InferenceRule("simplification", ["A&B"], "A", 1.0),
            # Addition
            InferenceRule("addition", ["A"], "A|B", 0.8),
        ]
    
    def _initialize_swarm(self):
        """Initialize swarm agents"""
        self.swarm_agents = [
            SwarmAgent(f"swarm_{i}", dimensions=10) 
            for i in range(self.swarm_size)
        ]
    
    async def initialize_redis(self, redis_url: str = "redis://localhost"):
        """Initialize Redis connection for distributed caching"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory cache only")
            self.redis_client = None
            return
            
        try:
            self.redis_client = await aioredis.create_redis_pool(redis_url)
            logger.info("Redis caching initialized")
        except Exception as e:
            logger.warning(f"Redis initialization failed, using in-memory cache: {e}")
            self.redis_client = None
    
    @a2a_skill(
        name="hierarchical_multi_engine_reasoning",
        description="Perform hierarchical reasoning with multiple internal reasoning engines",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "reasoning_engines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["logical", "probabilistic", "analogical", "causal"]
                },
                "max_depth": {"type": "integer", "default": 5}
            },
            "required": ["question"]
        }
    )
    async def hierarchical_multi_engine_reasoning(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement hierarchical reasoning with multiple internal reasoning engines
        Each engine processes the question independently, then results are synthesized
        """
        question = request_data["question"]
        context = request_data.get("context", {})
        engines = request_data.get("reasoning_engines", ["logical", "probabilistic", "analogical", "causal"])
        max_depth = request_data.get("max_depth", 5)
        
        # Check cache first
        cache_key = hashlib.md5(f"{question}:{json.dumps(engines)}".encode()).hexdigest()
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Run reasoning engines in parallel
        engine_tasks = []
        for engine in engines:
            if engine == "logical":
                task = self._logical_reasoning_engine(question, context, max_depth)
            elif engine == "probabilistic":
                task = self._probabilistic_reasoning_engine(question, context, max_depth)
            elif engine == "analogical":
                task = self._analogical_reasoning_engine(question, context, max_depth)
            elif engine == "causal":
                task = self._causal_reasoning_engine(question, context, max_depth)
            else:
                continue
            engine_tasks.append(task)
        
        # Execute engines in parallel
        engine_results = await asyncio.gather(*engine_tasks, return_exceptions=True)
        
        # Filter out exceptions and combine results
        valid_results = []
        for i, result in enumerate(engine_results):
            if not isinstance(result, Exception):
                valid_results.append({
                    "engine": engines[i],
                    "result": result
                })
        
        # Synthesize results using hierarchical aggregation
        synthesis = await self._hierarchical_synthesis(valid_results, question)
        
        # Build final result
        result = {
            "answer": synthesis["answer"],
            "confidence": synthesis["confidence"],
            "reasoning_paths": synthesis["reasoning_paths"],
            "engine_contributions": {
                r["engine"]: r["result"]["confidence"] 
                for r in valid_results
            },
            "hierarchical_depth": max_depth,
            "engines_used": len(valid_results),
            "synthesis_method": "hierarchical_weighted_aggregation"
        }
        
        # Cache result
        await self._cache_result(cache_key, result)
        
        return result
    
    async def _logical_reasoning_engine(
        self, 
        question: str, 
        context: Dict[str, Any], 
        max_depth: int
    ) -> Dict[str, Any]:
        """Logical inference engine using formal rules"""
        # Parse question to extract logical components
        logical_components = self._parse_logical_components(question)
        
        # Add facts from context
        self.facts.clear()
        self.facts.update(logical_components.get("facts", []))
        self.facts.update(context.get("known_facts", []))
        
        # Apply inference rules iteratively
        inference_chain = []
        depth = 0
        
        while depth < max_depth:
            new_facts_found = False
            
            for rule in self.inference_rules:
                conclusion, confidence = rule.apply(self.facts)
                if conclusion and conclusion not in self.facts:
                    self.facts.add(conclusion)
                    self.derived_facts.add((conclusion, confidence))
                    inference_chain.append({
                        "rule": rule.name,
                        "premises": rule.premises,
                        "conclusion": conclusion,
                        "confidence": confidence,
                        "depth": depth
                    })
                    new_facts_found = True
            
            if not new_facts_found:
                break
            depth += 1
        
        # Generate answer based on derived facts
        answer = self._generate_logical_answer(question, self.facts, inference_chain)
        
        return {
            "answer": answer["text"],
            "confidence": answer["confidence"],
            "inference_chain": inference_chain,
            "total_facts": len(self.facts),
            "derived_facts": len(self.derived_facts),
            "reasoning_type": "logical_inference"
        }
    
    async def _probabilistic_reasoning_engine(
        self, 
        question: str, 
        context: Dict[str, Any], 
        max_depth: int
    ) -> Dict[str, Any]:
        """Probabilistic reasoning using Bayesian networks"""
        # Build probabilistic model
        hypotheses = self._generate_hypotheses(question, context)
        
        # Initialize prior probabilities
        priors = {h["id"]: h["prior"] for h in hypotheses}
        
        # Collect evidence
        evidence = self._extract_evidence(question, context)
        
        # Update probabilities using Bayes' theorem
        posteriors = {}
        reasoning_steps = []
        
        for hypothesis in hypotheses:
            h_id = hypothesis["id"]
            prior = priors[h_id]
            
            # Calculate likelihood based on evidence
            likelihood = 1.0
            for e in evidence:
                # P(E|H)
                e_given_h = self._calculate_likelihood(e, hypothesis)
                likelihood *= e_given_h
                
                reasoning_steps.append({
                    "hypothesis": h_id,
                    "evidence": e["content"],
                    "likelihood": e_given_h,
                    "step": f"P({e['id']}|{h_id}) = {e_given_h:.3f}"
                })
            
            # Calculate posterior: P(H|E) = P(E|H) * P(H) / P(E)
            # Simplified - assuming P(E) is constant for comparison
            posterior = likelihood * prior
            posteriors[h_id] = posterior
        
        # Normalize posteriors
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {h: p/total for h, p in posteriors.items()}
        
        # Select best hypothesis
        best_hypothesis_id = max(posteriors, key=posteriors.get)
        best_hypothesis = next(h for h in hypotheses if h["id"] == best_hypothesis_id)
        
        return {
            "answer": best_hypothesis["content"],
            "confidence": posteriors[best_hypothesis_id],
            "hypotheses": [
                {
                    "content": h["content"],
                    "prior": priors[h["id"]],
                    "posterior": posteriors[h["id"]]
                }
                for h in hypotheses
            ],
            "evidence_count": len(evidence),
            "reasoning_steps": reasoning_steps,
            "reasoning_type": "probabilistic_bayesian"
        }
    
    async def _analogical_reasoning_engine(
        self, 
        question: str, 
        context: Dict[str, Any], 
        max_depth: int
    ) -> Dict[str, Any]:
        """Analogical reasoning by finding similar patterns"""
        # Extract key concepts from question
        concepts = self._extract_concepts(question)
        
        # Find analogies in knowledge base
        analogies = []
        for concept in concepts:
            similar_cases = self._find_similar_cases(concept, context)
            for case in similar_cases:
                analogy = {
                    "source": concept,
                    "target": case["concept"],
                    "similarity": case["similarity"],
                    "mapping": case["mapping"],
                    "inference": self._transfer_knowledge(concept, case)
                }
                analogies.append(analogy)
        
        # Rank analogies by relevance and similarity
        ranked_analogies = sorted(
            analogies, 
            key=lambda a: a["similarity"] * a["inference"]["confidence"], 
            reverse=True
        )
        
        # Generate answer from best analogies
        if ranked_analogies:
            best_analogy = ranked_analogies[0]
            answer = best_analogy["inference"]["conclusion"]
            confidence = best_analogy["similarity"] * best_analogy["inference"]["confidence"]
        else:
            answer = "No suitable analogies found"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "analogies": ranked_analogies[:3],  # Top 3 analogies
            "concepts_analyzed": len(concepts),
            "reasoning_type": "analogical"
        }
    
    async def _causal_reasoning_engine(
        self, 
        question: str, 
        context: Dict[str, Any], 
        max_depth: int
    ) -> Dict[str, Any]:
        """Causal reasoning using causal graphs"""
        # Build causal graph
        causal_graph = self._build_causal_graph(question, context)
        
        # Identify causal paths
        causal_paths = []
        nodes = list(causal_graph.nodes())
        
        for start in nodes:
            for end in nodes:
                if start != end:
                    try:
                        paths = list(nx.all_simple_paths(
                            causal_graph, start, end, cutoff=max_depth
                        ))
                        for path in paths:
                            strength = self._calculate_causal_strength(path, causal_graph)
                            causal_paths.append({
                                "path": path,
                                "strength": strength,
                                "length": len(path) - 1
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        # Sort by causal strength
        causal_paths.sort(key=lambda p: p["strength"], reverse=True)
        
        # Generate causal explanation
        if causal_paths:
            strongest_path = causal_paths[0]
            answer = self._generate_causal_explanation(
                strongest_path["path"], 
                causal_graph, 
                question
            )
            confidence = strongest_path["strength"]
        else:
            answer = "No clear causal relationships identified"
            confidence = 0.4
        
        return {
            "answer": answer,
            "confidence": confidence,
            "causal_paths": causal_paths[:5],  # Top 5 paths
            "graph_nodes": len(nodes),
            "graph_edges": len(causal_graph.edges()),
            "reasoning_type": "causal"
        }
    
    @a2a_skill(
        name="swarm_intelligence_reasoning",
        description="Use swarm intelligence algorithms for solution exploration",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "swarm_algorithm": {
                    "type": "string",
                    "enum": ["pso", "aco", "bee", "firefly"],
                    "default": "pso"
                },
                "iterations": {"type": "integer", "default": 50},
                "swarm_size": {"type": "integer", "default": 20}
            },
            "required": ["question"]
        }
    )
    async def swarm_intelligence_reasoning(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement swarm intelligence for solution exploration
        """
        question = request_data["question"]
        context = request_data.get("context", {})
        algorithm = request_data.get("swarm_algorithm", "pso")
        iterations = request_data.get("iterations", 50)
        swarm_size = request_data.get("swarm_size", 20)
        
        # Initialize or resize swarm if needed
        if len(self.swarm_agents) != swarm_size:
            self._initialize_swarm()
        
        # Reset swarm state
        self.global_best_score = -np.inf
        self.global_best_position = None
        
        # Define fitness function based on question
        fitness_fn = self._create_fitness_function(question, context)
        
        # Run swarm algorithm
        convergence_history = []
        
        for iteration in range(iterations):
            iteration_best_score = -np.inf
            
            # Evaluate each agent
            agent_evaluations = []
            for agent in self.swarm_agents:
                # Evaluate fitness
                score = await fitness_fn(agent.position)
                agent_evaluations.append((agent, score))
                
                # Update personal best
                if score > agent.personal_best_score:
                    agent.personal_best_score = score
                    agent.personal_best_position = agent.position.copy()
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = agent.position.copy()
                
                if score > iteration_best_score:
                    iteration_best_score = score
            
            # Update velocities and positions
            if algorithm == "pso":
                await self._update_pso_swarm()
            elif algorithm == "aco":
                await self._update_aco_swarm(agent_evaluations)
            elif algorithm == "bee":
                await self._update_bee_swarm(agent_evaluations)
            elif algorithm == "firefly":
                await self._update_firefly_swarm(agent_evaluations)
            
            convergence_history.append({
                "iteration": iteration,
                "best_score": self.global_best_score,
                "average_score": np.mean([s for _, s in agent_evaluations]),
                "diversity": np.std([s for _, s in agent_evaluations])
            })
            
            # Early stopping if converged
            if len(convergence_history) > 10:
                recent_scores = [h["best_score"] for h in convergence_history[-10:]]
                if np.std(recent_scores) < 0.001:
                    break
        
        # Generate answer from best position
        answer = await self._decode_swarm_solution(
            self.global_best_position, 
            question, 
            context
        )
        
        return {
            "answer": answer["text"],
            "confidence": min(1.0, self.global_best_score),
            "algorithm": algorithm,
            "iterations_run": len(convergence_history),
            "swarm_size": swarm_size,
            "convergence_history": convergence_history[-10:],  # Last 10 iterations
            "final_diversity": convergence_history[-1]["diversity"] if convergence_history else 0,
            "solution_quality": self.global_best_score
        }
    
    async def _update_pso_swarm(self):
        """Update swarm using Particle Swarm Optimization"""
        for agent in self.swarm_agents:
            agent.update_velocity(self.global_best_position)
            agent.update_position()
    
    async def _update_aco_swarm(self, evaluations: List[Tuple[SwarmAgent, float]]):
        """Update swarm using Ant Colony Optimization"""
        # Create pheromone trails based on good solutions
        pheromone_matrix = np.zeros((10, 10))
        
        for agent, score in evaluations:
            if score > 0:
                # Deposit pheromones along path
                path = (agent.position * 9).astype(int)  # Convert to discrete path
                for i in range(len(path) - 1):
                    pheromone_matrix[path[i], path[i+1]] += score
        
        # Update agent positions based on pheromones
        for agent in self.swarm_agents:
            # Follow pheromone trails with some randomness
            new_position = np.random.rand(agent.dimensions)
            for i in range(agent.dimensions):
                if i < 9:
                    curr = int(agent.position[i] * 9)
                    # Probabilistic selection based on pheromones
                    probs = pheromone_matrix[curr, :]
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        next_pos = np.random.choice(10, p=probs)
                        new_position[i] = next_pos / 9.0
            agent.position = new_position
    
    async def _update_bee_swarm(self, evaluations: List[Tuple[SwarmAgent, float]]):
        """Update swarm using Artificial Bee Colony algorithm"""
        # Sort agents by fitness
        sorted_agents = sorted(evaluations, key=lambda x: x[1], reverse=True)
        
        # Elite bees (top 20%) - exploit
        elite_count = max(1, len(sorted_agents) // 5)
        for i in range(elite_count):
            agent, score = sorted_agents[i]
            # Small local search
            agent.position += np.random.normal(0, 0.01, agent.dimensions)
            agent.position = np.clip(agent.position, 0, 1)
        
        # Selected bees (next 40%) - explore neighborhood
        selected_count = max(1, 2 * len(sorted_agents) // 5)
        for i in range(elite_count, elite_count + selected_count):
            agent, score = sorted_agents[i]
            # Larger exploration
            agent.position += np.random.normal(0, 0.1, agent.dimensions)
            agent.position = np.clip(agent.position, 0, 1)
        
        # Scout bees (remaining) - random search
        for i in range(elite_count + selected_count, len(sorted_agents)):
            agent, score = sorted_agents[i]
            # Random repositioning
            agent.position = np.random.rand(agent.dimensions)
    
    async def _update_firefly_swarm(self, evaluations: List[Tuple[SwarmAgent, float]]):
        """Update swarm using Firefly Algorithm"""
        agents_scores = [(a, s) for a, s in evaluations]
        
        for i, (agent_i, score_i) in enumerate(agents_scores):
            for j, (agent_j, score_j) in enumerate(agents_scores):
                if score_j > score_i:  # agent_j is brighter
                    # Calculate distance
                    distance = np.linalg.norm(agent_j.position - agent_i.position)
                    
                    # Attractiveness decreases with distance
                    beta = 1.0 * np.exp(-0.1 * distance**2)
                    
                    # Move towards brighter firefly
                    agent_i.position += beta * (agent_j.position - agent_i.position)
                    
                    # Add randomness
                    agent_i.position += 0.01 * np.random.randn(agent_i.dimensions)
                    
                    # Keep within bounds
                    agent_i.position = np.clip(agent_i.position, 0, 1)
    
    @a2a_skill(
        name="enhanced_debate_mechanism",
        description="Conduct sophisticated multi-perspective debate",
        input_schema={
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "perspective": {"type": "string"},
                            "argument": {"type": "string"},
                            "evidence": {"type": "array"},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "debate_structure": {
                    "type": "string",
                    "enum": ["dialectical", "deliberative", "adversarial", "collaborative"],
                    "default": "dialectical"
                },
                "max_rounds": {"type": "integer", "default": 5},
                "convergence_threshold": {"type": "number", "default": 0.8}
            },
            "required": ["positions"]
        }
    )
    async def enhanced_debate_mechanism(
        self, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement sophisticated debate mechanism with structured argumentation
        """
        positions = request_data["positions"]
        structure = request_data.get("debate_structure", "dialectical")
        max_rounds = request_data.get("max_rounds", 5)
        convergence_threshold = request_data.get("convergence_threshold", 0.8)
        
        # Initialize debate state
        debate_state = {
            "positions": {
                p["perspective"]: {
                    "current_argument": p["argument"],
                    "evidence": p["evidence"],
                    "confidence": p["confidence"],
                    "argument_history": [p["argument"]],
                    "rebuttals_received": [],
                    "concessions_made": []
                }
                for p in positions
            },
            "rounds": [],
            "consensus_tracking": []
        }
        
        # Run debate rounds
        for round_num in range(max_rounds):
            round_data = {
                "round": round_num + 1,
                "exchanges": [],
                "position_updates": [],
                "consensus_measure": 0.0
            }
            
            if structure == "dialectical":
                round_result = await self._dialectical_round(debate_state)
            elif structure == "deliberative":
                round_result = await self._deliberative_round(debate_state)
            elif structure == "adversarial":
                round_result = await self._adversarial_round(debate_state)
            elif structure == "collaborative":
                round_result = await self._collaborative_round(debate_state)
            
            round_data.update(round_result)
            debate_state["rounds"].append(round_data)
            
            # Check for consensus
            consensus = self._calculate_consensus(debate_state["positions"])
            debate_state["consensus_tracking"].append(consensus)
            
            if consensus >= convergence_threshold:
                break
        
        # Synthesize final position
        final_synthesis = self._synthesize_debate_outcome(debate_state)
        
        return {
            "final_position": final_synthesis["position"],
            "confidence": final_synthesis["confidence"],
            "debate_structure": structure,
            "rounds_conducted": len(debate_state["rounds"]),
            "consensus_achieved": debate_state["consensus_tracking"][-1] >= convergence_threshold,
            "consensus_progression": debate_state["consensus_tracking"],
            "key_arguments": final_synthesis["key_arguments"],
            "perspective_evolution": self._track_perspective_evolution(debate_state),
            "synthesis_method": final_synthesis["method"]
        }
    
    async def _dialectical_round(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Thesis-antithesis-synthesis dialectical reasoning"""
        exchanges = []
        position_updates = []
        
        perspectives = list(debate_state["positions"].keys())
        
        # Pair perspectives for dialectical exchange
        for i in range(0, len(perspectives), 2):
            if i + 1 < len(perspectives):
                thesis_persp = perspectives[i]
                antithesis_persp = perspectives[i + 1]
                
                thesis = debate_state["positions"][thesis_persp]
                antithesis = debate_state["positions"][antithesis_persp]
                
                # Generate dialectical exchange
                exchange = {
                    "type": "dialectical",
                    "thesis": {
                        "perspective": thesis_persp,
                        "argument": thesis["current_argument"],
                        "confidence": thesis["confidence"]
                    },
                    "antithesis": {
                        "perspective": antithesis_persp,
                        "argument": antithesis["current_argument"],
                        "confidence": antithesis["confidence"]
                    }
                }
                
                # Generate synthesis
                synthesis = self._generate_synthesis(
                    thesis["current_argument"],
                    antithesis["current_argument"],
                    thesis["evidence"] + antithesis["evidence"]
                )
                
                exchange["synthesis"] = synthesis
                exchanges.append(exchange)
                
                # Update positions based on synthesis
                if synthesis["confidence"] > 0.7:
                    # Both perspectives move towards synthesis
                    for persp in [thesis_persp, antithesis_persp]:
                        old_confidence = debate_state["positions"][persp]["confidence"]
                        new_confidence = (old_confidence + synthesis["confidence"]) / 2
                        
                        debate_state["positions"][persp]["current_argument"] = synthesis["argument"]
                        debate_state["positions"][persp]["confidence"] = new_confidence
                        debate_state["positions"][persp]["argument_history"].append(synthesis["argument"])
                        
                        position_updates.append({
                            "perspective": persp,
                            "update_type": "synthesis_adoption",
                            "new_confidence": new_confidence
                        })
        
        return {
            "exchanges": exchanges,
            "position_updates": position_updates,
            "consensus_measure": self._calculate_consensus(debate_state["positions"])
        }
    
    async def _deliberative_round(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deliberative democracy style reasoning"""
        exchanges = []
        position_updates = []
        
        # Each perspective considers all others
        for persp, position in debate_state["positions"].items():
            considerations = []
            
            for other_persp, other_position in debate_state["positions"].items():
                if persp != other_persp:
                    # Evaluate compatibility and strength
                    compatibility = self._evaluate_argument_compatibility(
                        position["current_argument"],
                        other_position["current_argument"]
                    )
                    
                    considerations.append({
                        "perspective": other_persp,
                        "compatibility": compatibility,
                        "strength": other_position["confidence"]
                    })
            
            # Update position based on deliberation
            if considerations:
                # Weight compatible arguments more
                weighted_update = self._deliberative_update(
                    position,
                    considerations,
                    debate_state["positions"]
                )
                
                if weighted_update["changed"]:
                    position["current_argument"] = weighted_update["new_argument"]
                    position["confidence"] = weighted_update["new_confidence"]
                    position["argument_history"].append(weighted_update["new_argument"])
                    
                    position_updates.append({
                        "perspective": persp,
                        "update_type": "deliberative_refinement",
                        "influenced_by": weighted_update["influences"],
                        "new_confidence": weighted_update["new_confidence"]
                    })
            
            exchanges.append({
                "type": "deliberative",
                "perspective": persp,
                "considerations": considerations,
                "update_made": weighted_update["changed"] if considerations else False
            })
        
        return {
            "exchanges": exchanges,
            "position_updates": position_updates,
            "consensus_measure": self._calculate_consensus(debate_state["positions"])
        }
    
    async def _adversarial_round(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adversarial debate with rebuttals and counter-arguments"""
        exchanges = []
        position_updates = []
        
        perspectives = list(debate_state["positions"].keys())
        
        # Each perspective attacks the weakest opposing argument
        for attacker_persp in perspectives:
            attacker = debate_state["positions"][attacker_persp]
            
            # Find weakest opponent
            weakest_opponent = None
            min_confidence = 1.0
            
            for defender_persp in perspectives:
                if defender_persp != attacker_persp:
                    defender = debate_state["positions"][defender_persp]
                    if defender["confidence"] < min_confidence:
                        min_confidence = defender["confidence"]
                        weakest_opponent = defender_persp
            
            if weakest_opponent:
                defender = debate_state["positions"][weakest_opponent]
                
                # Generate rebuttal
                rebuttal = self._generate_rebuttal(
                    attacker["current_argument"],
                    defender["current_argument"],
                    attacker["evidence"]
                )
                
                exchange = {
                    "type": "adversarial",
                    "attacker": attacker_persp,
                    "defender": weakest_opponent,
                    "rebuttal": rebuttal
                }
                
                # Defender responds
                if rebuttal["strength"] > 0.6:
                    # Strong rebuttal - defender must adapt or lose confidence
                    counter = self._generate_counter_argument(
                        defender["current_argument"],
                        rebuttal["argument"],
                        defender["evidence"]
                    )
                    
                    exchange["counter_argument"] = counter
                    
                    if counter["success"]:
                        # Successful counter - maintain position
                        defender["confidence"] = min(1.0, defender["confidence"] + 0.1)
                    else:
                        # Failed counter - lose confidence
                        defender["confidence"] *= 0.8
                        defender["rebuttals_received"].append(rebuttal)
                    
                    position_updates.append({
                        "perspective": weakest_opponent,
                        "update_type": "defense_outcome",
                        "new_confidence": defender["confidence"],
                        "defense_success": counter["success"]
                    })
                
                exchanges.append(exchange)
        
        return {
            "exchanges": exchanges,
            "position_updates": position_updates,
            "consensus_measure": self._calculate_consensus(debate_state["positions"])
        }
    
    async def _collaborative_round(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborative reasoning to build shared understanding"""
        exchanges = []
        position_updates = []
        
        # Identify common ground
        common_elements = self._find_common_ground(debate_state["positions"])
        
        # Build upon common ground
        collaborative_synthesis = self._build_collaborative_position(
            common_elements,
            debate_state["positions"]
        )
        
        exchanges.append({
            "type": "collaborative",
            "common_ground": common_elements,
            "collaborative_synthesis": collaborative_synthesis
        })
        
        # All perspectives move towards collaborative synthesis
        for persp, position in debate_state["positions"].items():
            alignment = self._calculate_alignment(
                position["current_argument"],
                collaborative_synthesis["argument"]
            )
            
            if alignment > 0.5:
                # Adopt collaborative position with confidence based on alignment
                new_confidence = position["confidence"] * alignment + \
                               collaborative_synthesis["confidence"] * (1 - alignment)
                
                position["current_argument"] = collaborative_synthesis["argument"]
                position["confidence"] = new_confidence
                position["argument_history"].append(collaborative_synthesis["argument"])
                
                position_updates.append({
                    "perspective": persp,
                    "update_type": "collaborative_convergence",
                    "alignment": alignment,
                    "new_confidence": new_confidence
                })
        
        return {
            "exchanges": exchanges,
            "position_updates": position_updates,
            "consensus_measure": self._calculate_consensus(debate_state["positions"])
        }
    
    # Helper methods for reasoning engines
    
    async def _parse_logical_components(self, question: str) -> Dict[str, Any]:
        """Parse question into logical components using Grok-4"""
        
        # Try Grok-4 first for intelligent parsing
        if hasattr(self, 'reasoning_agent') and hasattr(self.reasoning_agent, 'grok_client') and self.reasoning_agent.grok_client:
            try:
                from .grokReasoning import GrokReasoning
                grok = GrokReasoning()
                
                result = await grok.decompose_question(question, {
                    "task": "logical_component_parsing",
                    "focus": "facts, rules, queries, logical_relationships"
                })
                
                if result.get('success'):
                    decomposition = result.get('decomposition', {})
                    return {
                        "facts": decomposition.get('facts', []),
                        "rules": decomposition.get('rules', []),
                        "queries": decomposition.get('queries', []),
                        "logical_relationships": decomposition.get('logical_relationships', []),
                        "enhanced": True
                    }
            except Exception as e:
                logger.warning(f"Grok-4 logical parsing failed, using fallback: {e}")
        
        # Fallback to basic parsing
        components = {
            "facts": [],
            "rules": [],
            "queries": [],
            "enhanced": False
        }
        
        question_lower = question.lower()
        
        # Extract facts (statements of truth)
        if "given" in question_lower or "assume" in question_lower:
            fact_section = question_lower.split("given")[-1].split("assume")[-1]
            facts = [f.strip() for f in fact_section.split(",") if f.strip()]
            components["facts"] = facts
        
        # Extract logical connectives
        if "if" in question_lower and "then" in question_lower:
            components["rules"].append("implication")
        
        if "and" in question_lower or "or" in question_lower:
            components["rules"].append("conjunction" if "and" in question_lower else "disjunction")
        
        return components
    
    def _generate_logical_answer(
        self, 
        question: str, 
        facts: Set[str], 
        inference_chain: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer from logical inference"""
        if inference_chain:
            # Build answer from inference chain
            conclusion = inference_chain[-1]["conclusion"]
            confidence = min([step["confidence"] for step in inference_chain])
            
            explanation = f"Based on logical inference: {conclusion}"
            if len(inference_chain) > 1:
                explanation += f" (derived through {len(inference_chain)} inference steps)"
            
            return {
                "text": explanation,
                "confidence": confidence
            }
        else:
            return {
                "text": "No new conclusions can be derived from the given facts",
                "confidence": 0.5
            }
    
    def _generate_hypotheses(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses for probabilistic reasoning using semantic analysis"""
        hypotheses = []
        
        # Extract semantic components from question
        question_words = set(question.lower().split())
        context_entities = self._extract_entities_from_context(context)
        
        # Analyze question structure for hypothesis generation
        question_type = self._classify_question_type(question)
        key_concepts = self._extract_key_concepts(question, context)
        
        # Generate hypotheses based on semantic analysis
        if question_type == "causal":
            # Generate causal hypotheses based on identified concepts
            for i, concept in enumerate(key_concepts[:3]):
                # Look for potential causes in context
                potential_causes = self._find_related_concepts(concept, context_entities)
                if potential_causes:
                    cause = potential_causes[0]
                    hypothesis = {
                        "id": f"H_causal_{i}",
                        "content": f"{concept} is caused by {cause} through direct interaction",
                        "prior": self._calculate_concept_prior(concept, context),
                        "type": "causal",
                        "supporting_concepts": [concept, cause]
                    }
                else:
                    hypothesis = {
                        "id": f"H_causal_{i}",
                        "content": f"{concept} results from inherent properties and environmental factors",
                        "prior": 0.3,
                        "type": "causal",
                        "supporting_concepts": [concept]
                    }
                hypotheses.append(hypothesis)
        
        elif question_type == "process":
            # Generate process-based hypotheses
            for i, concept in enumerate(key_concepts[:3]):
                processes = self._infer_processes(concept, context)
                for j, process in enumerate(processes):
                    hypothesis = {
                        "id": f"H_process_{i}_{j}",
                        "content": f"{concept} is achieved through {process}",
                        "prior": self._calculate_process_likelihood(process, context),
                        "type": "process",
                        "supporting_concepts": [concept, process]
                    }
                    hypotheses.append(hypothesis)
        
        elif question_type == "definitional":
            # Generate definitional hypotheses based on semantic similarity
            for i, concept in enumerate(key_concepts[:2]):
                definitions = self._generate_concept_definitions(concept, context)
                for j, definition in enumerate(definitions):
                    hypothesis = {
                        "id": f"H_def_{i}_{j}",
                        "content": f"{concept} is defined as {definition}",
                        "prior": self._calculate_definition_confidence(definition, context),
                        "type": "definitional",
                        "supporting_concepts": [concept]
                    }
                    hypotheses.append(hypothesis)
        
        else:
            # Generate general hypotheses based on semantic relationships
            semantic_relations = self._analyze_semantic_relations(question, context)
            for i, relation in enumerate(semantic_relations[:4]):
                hypothesis = {
                    "id": f"H_general_{i}",
                    "content": self._generate_hypothesis_from_relation(relation, question),
                    "prior": relation.get("confidence", 0.5),
                    "type": "general",
                    "supporting_concepts": relation.get("concepts", [])
                }
                hypotheses.append(hypothesis)
        
        # Ensure at least one hypothesis exists
        if not hypotheses:
            fallback_concepts = key_concepts or ["unknown_factor"]
            hypothesis = {
                "id": "H_fallback",
                "content": f"The answer relates to {', '.join(fallback_concepts[:2])}",
                "prior": 0.4,
                "type": "fallback",
                "supporting_concepts": fallback_concepts
            }
            hypotheses.append(hypothesis)
        
        # Normalize priors
        total_prior = sum(h["prior"] for h in hypotheses)
        if total_prior > 0:
            for h in hypotheses:
                h["prior"] = h["prior"] / total_prior
        
        return hypotheses
    
    def _extract_evidence(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract evidence from context"""
        evidence = []
        
        # Extract from context
        for key, value in context.items():
            if isinstance(value, str):
                evidence.append({
                    "id": f"E_{key}",
                    "content": value,
                    "source": "context",
                    "reliability": 0.8
                })
        
        # Extract from question itself
        if "observed" in question.lower() or "seen" in question.lower():
            evidence.append({
                "id": "E_observation",
                "content": "Direct observation mentioned",
                "source": "question",
                "reliability": 0.9
            })
        
        return evidence
    
    async def _calculate_likelihood(self, evidence: Dict[str, Any], hypothesis: Dict[str, Any]) -> float:
        """Calculate P(Evidence|Hypothesis) using Grok-4 semantic analysis"""
        
        # Try Grok-4 first for intelligent likelihood calculation
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()
            
            combined_text = f"Evidence: {evidence['content']}\nHypothesis: {hypothesis['content']}"
            result = await grok.analyze_patterns(combined_text, [])
            
            if result.get('success'):
                patterns = result.get('patterns', {})
                # Look for semantic relationships in Grok analysis
                relationships = patterns.get('logical_relationships', [])
                for rel in relationships:
                    if 'likelihood' in rel or 'probability' in rel or 'support' in rel:
                        # Extract confidence score from Grok analysis
                        confidence = rel.get('confidence', 0.5)
                        return max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
        except Exception as e:
            logger.warning(f"Grok-4 likelihood calculation failed, using fallback: {e}")
        
        # Fallback to keyword matching
        evidence_words = set(evidence["content"].lower().split())
        hypothesis_words = set(hypothesis["content"].lower().split())
        
        overlap = len(evidence_words & hypothesis_words)
        total = len(evidence_words | hypothesis_words)
        
        if total == 0:
            return 0.5  # No information
        
        similarity = overlap / total
        return 0.5 + 0.5 * similarity  # Scale to [0.5, 1.0]
    
    async def _extract_concepts(self, question: str) -> List[Dict[str, Any]]:
        """Extract key concepts from question using Grok-4"""
        
        # Try Grok-4 first for intelligent concept extraction
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()
            
            result = await grok.analyze_patterns(question, [])
            
            if result.get('success'):
                patterns = result.get('patterns', {})
                key_concepts = patterns.get('key_concepts', [])
                
                if key_concepts:
                    concepts = []
                    for i, concept in enumerate(key_concepts[:10]):  # Top 10 from Grok-4
                        if isinstance(concept, str):
                            concepts.append({
                                "id": f"C{i}",
                                "term": concept,
                                "context": question,
                                "importance": 0.9 / (i + 1),
                                "enhanced": True
                            })
                        elif isinstance(concept, dict):
                            concepts.append({
                                "id": f"C{i}",
                                "term": concept.get('term', concept.get('concept', str(concept))),
                                "context": question,
                                "importance": concept.get('importance', 0.9 / (i + 1)),
                                "enhanced": True
                            })
                    return concepts
        except Exception as e:
            logger.warning(f"Grok-4 concept extraction failed, using fallback: {e}")
        
        # Fallback to simple extraction
        concepts = []
        words = question.lower().split()
        skip_words = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an"}
        
        concept_words = [w for w in words if w not in skip_words and len(w) > 2]
        
        for i, word in enumerate(concept_words):
            concepts.append({
                "id": f"C{i}",
                "term": word,
                "context": question,
                "importance": 1.0 / (i + 1),
                "enhanced": False
            })
        
        return concepts
    
    def _find_similar_cases(self, concept: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar cases for analogical reasoning"""
        similar_cases = []
        
        # Predefined analogical mappings for demonstration
        analogy_db = {
            "computer": ["brain", "calculator", "processor"],
            "network": ["web", "system", "connection"],
            "algorithm": ["recipe", "procedure", "method"],
            "data": ["information", "facts", "knowledge"]
        }
        
        term = concept["term"]
        if term in analogy_db:
            for analog in analogy_db[term]:
                similar_cases.append({
                    "concept": analog,
                    "similarity": 0.7 + 0.1 * np.random.random(),
                    "mapping": {term: analog},
                    "source_domain": "technology",
                    "target_domain": "general"
                })
        
        return similar_cases
    
    def _transfer_knowledge(self, source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source to target domain"""
        return {
            "conclusion": f"By analogy with {target['concept']}, {source['term']} can be understood as {target['mapping'].get(source['term'], 'similar concept')}",
            "confidence": target["similarity"] * 0.8
        }
    
    async def _build_causal_graph(self, question: str, context: Dict[str, Any]) -> nx.DiGraph:
        """Build causal graph from question and context using Grok-4 analysis"""
        G = nx.DiGraph()
        
        # Try Grok-4 first for intelligent causal relationship detection
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()
            
            result = await grok.analyze_patterns(question, [])
            
            if result.get('success'):
                patterns = result.get('patterns', {})
                causal_relations = patterns.get('causal_relationships', [])
                logical_relations = patterns.get('logical_relationships', [])
                
                # Add nodes and edges from Grok-4 analysis
                if causal_relations or logical_relations:
                    all_relations = causal_relations + logical_relations
                    for relation in all_relations:
                        if isinstance(relation, dict):
                            source = relation.get('source', relation.get('cause'))
                            target = relation.get('target', relation.get('effect'))
                            weight = relation.get('strength', relation.get('confidence', 0.8))
                            
                            if source and target:
                                G.add_node(source)
                                G.add_node(target)
                                G.add_edge(source, target, weight=weight, enhanced=True, 
                                          relation_type=relation.get('type', 'causal'))
                    
                    if G.number_of_nodes() > 0:
                        return G
        except Exception as e:
            logger.warning(f"Grok-4 causal graph building failed, using fallback: {e}")
        
        # Fallback to keyword-based approach
        causal_keywords = ["causes", "leads to", "results in", "because", "therefore", "hence"]
        
        # Add nodes from question terms
        terms = [t for t in question.lower().split() if len(t) > 3]
        for term in terms:
            G.add_node(term)
        
        # Add causal edges based on keywords
        for keyword in causal_keywords:
            if keyword in question.lower():
                # Simple heuristic: connect adjacent terms
                for i in range(len(terms) - 1):
                    G.add_edge(terms[i], terms[i+1], weight=0.7, keyword=keyword, enhanced=False)
        
        # Add edges from context
        if "causal_links" in context:
            for link in context["causal_links"]:
                G.add_edge(link["cause"], link["effect"], weight=link.get("strength", 0.8))
        
        return G
    
    def _calculate_causal_strength(self, path: List[str], graph: nx.DiGraph) -> float:
        """Calculate strength of causal path"""
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                strength *= edge_data.get("weight", 0.5)
            else:
                strength *= 0.3  # Weak connection
        
        return strength
    
    def _generate_causal_explanation(
        self, 
        path: List[str], 
        graph: nx.DiGraph, 
        question: str
    ) -> str:
        """Generate causal explanation from path"""
        if len(path) < 2:
            return "No clear causal relationship found"
        
        explanation = f"{path[0].capitalize()}"
        
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            connector = edge_data.get("keyword", "leads to") if edge_data else "may lead to"
            explanation += f" {connector} {path[i+1]}"
        
        return explanation
    
    def _create_fitness_function(self, question: str, context: Dict[str, Any]):
        """Create fitness function for swarm optimization"""
        async def fitness(position: np.ndarray) -> float:
            # Decode position to solution components
            # Each dimension represents a different aspect of the solution
            
            # Simple fitness based on question keywords
            keywords = [w for w in question.lower().split() if len(w) > 3]
            
            # Calculate fitness as alignment with question intent
            fitness_score = 0.0
            
            # Dimension 0-3: Relevance to keywords
            for i, keyword in enumerate(keywords[:4]):
                if i < len(position):
                    fitness_score += position[i] * (1.0 / (i + 1))
            
            # Dimension 4-6: Logical consistency
            consistency = np.mean(position[4:7]) if len(position) > 6 else 0.5
            fitness_score += consistency * 0.3
            
            # Dimension 7-9: Confidence alignment
            confidence = np.mean(position[7:10]) if len(position) > 9 else 0.5
            fitness_score += confidence * 0.2
            
            return fitness_score
        
        return fitness
    
    async def _decode_swarm_solution(
        self, 
        position: np.ndarray, 
        question: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decode swarm position to solution using dimensional mapping"""
        # Each dimension represents specific solution characteristics
        # Dimensions 0-2: Semantic alignment with question concepts
        # Dimensions 3-5: Logical coherence factors  
        # Dimensions 6-8: Evidence strength indicators
        # Dimension 9: Confidence modifier
        
        question_concepts = self._extract_key_concepts(question, context)
        
        # Calculate semantic alignment
        semantic_scores = position[:3]
        concept_alignments = []
        
        for i, concept in enumerate(question_concepts[:3]):
            if i < len(semantic_scores):
                # Map position value to concept relevance
                alignment = semantic_scores[i]
                concept_alignments.append((concept, alignment))
        
        # Calculate logical coherence
        logic_scores = position[3:6] if len(position) > 5 else [0.5, 0.5, 0.5]
        coherence_factors = [
            logic_scores[0],  # Internal consistency
            logic_scores[1],  # Causal validity
            logic_scores[2]   # Inferential strength
        ]
        
        # Calculate evidence strength
        evidence_scores = position[6:9] if len(position) > 8 else [0.5, 0.5, 0.5]
        evidence_strength = np.mean(evidence_scores)
        
        # Generate solution based on dimensional analysis
        answer_components = []
        
        # Add concept-based components
        for concept, alignment in concept_alignments:
            if alignment > 0.6:
                weight = "strongly" if alignment > 0.8 else "moderately"
                answer_components.append(f"{weight} involves {concept}")
        
        # Add logical reasoning component
        logic_strength = np.mean(coherence_factors)
        if logic_strength > 0.7:
            answer_components.append("follows logical principles")
        elif logic_strength > 0.4:
            answer_components.append("has reasonable logical basis")
        
        # Add evidence component  
        if evidence_strength > 0.6:
            answer_components.append("is supported by available evidence")
        
        # Construct final answer
        if answer_components:
            answer = "The solution " + ", ".join(answer_components[:3])
        else:
            # Fallback based on highest scoring dimension
            max_dim = np.argmax(position[:6])
            if max_dim < 3:
                answer = f"The solution primarily relates to {question_concepts[max_dim] if max_dim < len(question_concepts) else 'key concepts'}"
            else:
                answer = "The solution follows logical analysis patterns"
        
        # Calculate overall confidence from position geometry
        # Distance from optimal solution space
        optimal_region = np.array([0.7, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6, 0.8, 0.75])
        if len(position) < len(optimal_region):
            optimal_region = optimal_region[:len(position)]
        
        distance_from_optimal = np.linalg.norm(position[:len(optimal_region)] - optimal_region)
        max_distance = np.sqrt(len(optimal_region))  # Maximum possible distance
        
        confidence = max(0.1, 1.0 - (distance_from_optimal / max_distance))
        
        return {
            "text": answer,
            "relevance": np.mean(semantic_scores),
            "consistency": logic_strength, 
            "confidence": confidence,
            "evidence_strength": evidence_strength,
            "concept_alignments": concept_alignments,
            "dimensional_analysis": {
                "semantic_alignment": semantic_scores.tolist(),
                "logical_coherence": coherence_factors,
                "evidence_scores": evidence_scores.tolist()
            }
        }
    
    async def _hierarchical_synthesis(self, engine_results: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Hierarchically synthesize results from multiple reasoning engines using Grok-4"""
        if not engine_results:
            return {
                "answer": "No reasoning engines produced results",
                "confidence": 0.0,
                "reasoning_paths": []
            }
        
        # Weight results by confidence and engine type
        engine_weights = {
            "logical": 1.2,
            "probabilistic": 1.0,
            "analogical": 0.8,
            "causal": 1.1
        }
        
        weighted_answers = []
        reasoning_paths = []
        
        for engine_result in engine_results:
            engine = engine_result["engine"]
            result = engine_result["result"]
            
            weight = engine_weights.get(engine, 1.0)
            weighted_confidence = result["confidence"] * weight
            
            weighted_answers.append({
                "answer": result["answer"],
                "weighted_confidence": weighted_confidence,
                "engine": engine
            })
            
            reasoning_paths.append({
                "engine": engine,
                "path": result.get("inference_chain", result.get("reasoning_steps", [])),
                "confidence": result["confidence"]
            })
        
        # Try Grok-4 first for intelligent synthesis
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()
            
            # Prepare sub-answers for Grok-4
            sub_answers = [
                {
                    "content": wa["answer"],
                    "engine": wa["engine"],
                    "confidence": wa["weighted_confidence"],
                    "reasoning": reasoning_paths[i].get("path", [])
                }
                for i, wa in enumerate(weighted_answers)
            ]
            
            result = await grok.synthesize_answer(sub_answers, question)
            
            if result.get('success'):
                return {
                    "answer": result.get('synthesis'),
                    "confidence": min(1.0, max(wa["weighted_confidence"] for wa in weighted_answers)),
                    "reasoning_paths": reasoning_paths,
                    "enhanced": True
                }
        except Exception as e:
            logger.warning(f"Grok-4 hierarchical synthesis failed, using fallback: {e}")
        
        # Fallback to simple logic
        if len(weighted_answers) == 1:
            best = weighted_answers[0]
            return {
                "answer": best["answer"],
                "confidence": best["weighted_confidence"],
                "reasoning_paths": reasoning_paths,
                "enhanced": False
            }
        
        # Check for consensus
        answers_text = [wa["answer"] for wa in weighted_answers]
        unique_answers = list(set(answers_text))
        
        if len(unique_answers) == 1:
            # Full consensus
            total_confidence = sum(wa["weighted_confidence"] for wa in weighted_answers)
            avg_confidence = total_confidence / len(weighted_answers)
            
            return {
                "answer": unique_answers[0],
                "confidence": min(1.0, avg_confidence * 1.1),  # Boost for consensus
                "reasoning_paths": reasoning_paths,
                "enhanced": False
            }
        else:
            # Simple synthesis fallback
            best = max(weighted_answers, key=lambda x: x["weighted_confidence"])
            synthesis = best["answer"]
            
            for wa in weighted_answers:
                if wa != best and wa["weighted_confidence"] > 0.6:
                    synthesis += f" (Alternative perspective from {wa['engine']}: {wa['answer'][:50]}...)"
            
            return {
                "answer": synthesis,
                "confidence": best["weighted_confidence"],
                "reasoning_paths": reasoning_paths,
                "enhanced": False
            }
    
    # Debate helper methods
    
    async def _generate_synthesis(self, thesis: str, antithesis: str, evidence: List[Any]) -> Dict[str, Any]:
        """Generate dialectical synthesis using Grok-4"""
        
        # Try Grok-4 first for intelligent dialectical synthesis
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()
            
            # Prepare dialectical arguments for Grok-4
            sub_answers = [
                {"content": thesis, "position": "thesis", "evidence": evidence},
                {"content": antithesis, "position": "antithesis", "evidence": evidence}
            ]
            
            question = f"Synthesize these dialectical positions: thesis '{thesis}' and antithesis '{antithesis}'"
            result = await grok.synthesize_answer(sub_answers, question)
            
            if result.get('success'):
                return {
                    "synthesis": result.get('synthesis'),
                    "confidence": 0.85,
                    "enhanced": True
                }
        except Exception as e:
            logger.warning(f"Grok-4 dialectical synthesis failed, using fallback: {e}")
        
        # Fallback to word-based analysis
        thesis_words = set(thesis.lower().split())
        antithesis_words = set(antithesis.lower().split())
        
        common = thesis_words & antithesis_words
        unique_thesis = thesis_words - antithesis_words
        unique_antithesis = antithesis_words - thesis_words
        
        # Build synthesis
        if len(common) > len(unique_thesis) + len(unique_antithesis):
            # Strong overlap - merge positions
            synthesis = f"Both perspectives agree on {', '.join(list(common)[:3])}"
            confidence = 0.9
        else:
            # Weak overlap - find middle ground
            synthesis = f"Considering both {len(unique_thesis)} unique aspects from the first position and {len(unique_antithesis)} from the second"
            confidence = 0.7
        
        return {
            "argument": synthesis,
            "confidence": confidence,
            "method": "dialectical_synthesis"
        }
    
    def _evaluate_argument_compatibility(self, arg1: str, arg2: str) -> float:
        """Evaluate compatibility between arguments"""
        # Simple compatibility based on keyword overlap
        words1 = set(arg1.lower().split())
        words2 = set(arg2.lower().split())
        
        # Check for contradictory terms
        contradictions = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("support", "oppose"),
            ("agree", "disagree")
        ]
        
        for word1 in words1:
            for word2 in words2:
                for contra1, contra2 in contradictions:
                    if (word1 == contra1 and word2 == contra2) or \
                       (word1 == contra2 and word2 == contra1):
                        return 0.0  # Incompatible
        
        # Calculate similarity
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.5
    
    def _deliberative_update(
        self, 
        position: Dict[str, Any], 
        considerations: List[Dict[str, Any]], 
        all_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update position based on deliberative considerations"""
        
        # Find highly compatible positions
        compatible = [c for c in considerations if c["compatibility"] > 0.7]
        
        if not compatible:
            return {"changed": False}
        
        # Weight by compatibility and strength
        total_weight = sum(c["compatibility"] * c["strength"] for c in compatible)
        
        if total_weight > position["confidence"]:
            # Update position
            influences = [c["perspective"] for c in compatible]
            
            # Merge compatible arguments
            new_argument = position["current_argument"]
            for c in compatible:
                other_arg = all_positions[c["perspective"]]["current_argument"]
                # Add compatible elements
                new_argument += f" Additionally, {other_arg[:50]}..."
            
            new_confidence = min(1.0, (position["confidence"] + total_weight) / 2)
            
            return {
                "changed": True,
                "new_argument": new_argument,
                "new_confidence": new_confidence,
                "influences": influences
            }
        
        return {"changed": False}
    
    def _generate_rebuttal(
        self, 
        attacker_arg: str, 
        defender_arg: str, 
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Generate rebuttal to an argument"""
        # Identify weak points in defender's argument
        defender_words = defender_arg.lower().split()
        
        weak_indicators = ["maybe", "possibly", "might", "could", "perhaps", "seems"]
        weaknesses = [w for w in weak_indicators if w in defender_words]
        
        if weaknesses:
            rebuttal = f"The argument relies on uncertain terms like '{weaknesses[0]}', which undermines its strength"
            strength = 0.8
        else:
            # Attack based on missing evidence
            rebuttal = f"The position lacks concrete evidence to support its claims"
            strength = 0.6
        
        return {
            "argument": rebuttal,
            "strength": strength,
            "type": "weakness_identification"
        }
    
    def _generate_counter_argument(
        self, 
        original_arg: str, 
        rebuttal: str, 
        evidence: List[Any]
    ) -> Dict[str, Any]:
        """Generate counter-argument to rebuttal"""
        # Defend against rebuttal
        if "uncertain" in rebuttal or "lacks evidence" in rebuttal:
            if evidence:
                counter = f"The position is supported by {len(evidence)} pieces of evidence"
                return {"argument": counter, "success": True}
            else:
                counter = "While direct evidence is limited, the logical consistency remains strong"
                return {"argument": counter, "success": False}
        else:
            counter = "The criticism does not address the core argument"
            return {"argument": counter, "success": False}
    
    def _find_common_ground(self, positions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common elements across positions"""
        common_elements = []
        
        # Extract all arguments
        all_args = [p["current_argument"] for p in positions.values()]
        
        # Find common words/phrases
        if all_args:
            word_sets = [set(arg.lower().split()) for arg in all_args]
            common_words = set.intersection(*word_sets)
            
            for word in common_words:
                if len(word) > 3:  # Skip short words
                    common_elements.append({
                        "type": "shared_concept",
                        "content": word,
                        "frequency": sum(1 for arg in all_args if word in arg.lower())
                    })
        
        return common_elements
    
    def _build_collaborative_position(
        self, 
        common_elements: List[Dict[str, Any]], 
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build collaborative position from common ground"""
        if not common_elements:
            return {
                "argument": "Despite differences, all perspectives contribute valuable insights",
                "confidence": 0.5,
                "method": "inclusive_synthesis"
            }
        
        # Build on common ground
        shared_concepts = [e["content"] for e in common_elements if e["type"] == "shared_concept"]
        
        argument = f"Building on shared understanding of {', '.join(shared_concepts[:3])}"
        
        # Add unique contributions from each perspective
        unique_contributions = []
        for persp, position in positions.items():
            # Find unique valuable element
            arg_words = set(position["current_argument"].lower().split())
            unique_words = arg_words - set(shared_concepts)
            if unique_words:
                valuable_word = max(unique_words, key=len)  # Longest unique word
                unique_contributions.append(valuable_word)
        
        if unique_contributions:
            argument += f", integrating perspectives on {', '.join(unique_contributions[:3])}"
        
        # Confidence based on common ground strength
        confidence = min(1.0, 0.5 + 0.1 * len(common_elements))
        
        return {
            "argument": argument,
            "confidence": confidence,
            "method": "collaborative_building"
        }
    
    def _calculate_consensus(self, positions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consensus measure across positions"""
        if len(positions) < 2:
            return 1.0
        
        # Compare all pairs of positions
        similarities = []
        
        position_list = list(positions.values())
        for i in range(len(position_list)):
            for j in range(i + 1, len(position_list)):
                arg1 = position_list[i]["current_argument"]
                arg2 = position_list[j]["current_argument"]
                
                similarity = self._evaluate_argument_compatibility(arg1, arg2)
                similarities.append(similarity)
        
        # Average similarity as consensus measure
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_alignment(self, arg1: str, arg2: str) -> float:
        """Calculate alignment between arguments"""
        # Similar to compatibility but includes directional agreement
        words1 = set(arg1.lower().split())
        words2 = set(arg2.lower().split())
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        base_alignment = overlap / total if total > 0 else 0.0
        
        # Boost for same conclusion indicators
        conclusion_words = ["therefore", "thus", "hence", "conclude"]
        if any(w in words1 for w in conclusion_words) and \
           any(w in words2 for w in conclusion_words):
            base_alignment = min(1.0, base_alignment * 1.2)
        
        return base_alignment
    
    def _synthesize_debate_outcome(self, debate_state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final outcome from debate"""
        positions = debate_state["positions"]
        
        # Find position with highest final confidence
        best_position = max(
            positions.items(),
            key=lambda x: x[1]["confidence"]
        )
        
        perspective = best_position[0]
        position_data = best_position[1]
        
        # Extract key arguments from debate history
        key_arguments = []
        
        # Add strongest arguments from each round
        for round_data in debate_state["rounds"]:
            if "exchanges" in round_data:
                for exchange in round_data["exchanges"]:
                    if exchange.get("type") == "dialectical" and "synthesis" in exchange:
                        key_arguments.append({
                            "type": "synthesis",
                            "content": exchange["synthesis"]["argument"]
                        })
                    elif exchange.get("type") == "adversarial" and "rebuttal" in exchange:
                        if exchange["rebuttal"]["strength"] > 0.7:
                            key_arguments.append({
                                "type": "strong_rebuttal",
                                "content": exchange["rebuttal"]["argument"]
                            })
        
        # Determine synthesis method
        if debate_state["consensus_tracking"][-1] > 0.8:
            method = "consensus_based"
        elif len(set(p["current_argument"] for p in positions.values())) == 1:
            method = "convergence_based"
        else:
            method = "confidence_weighted"
        
        return {
            "position": position_data["current_argument"],
            "confidence": position_data["confidence"],
            "perspective": perspective,
            "key_arguments": key_arguments[:5],  # Top 5 key arguments
            "method": method
        }
    
    def _track_perspective_evolution(self, debate_state: Dict[str, Any]) -> Dict[str, List[float]]:
        """Track how perspectives evolved during debate"""
        evolution = {}
        
        for perspective, position in debate_state["positions"].items():
            # Track confidence over rounds
            confidence_history = [position["confidence"]]
            
            # Extract confidence from round updates
            for round_data in debate_state["rounds"]:
                for update in round_data.get("position_updates", []):
                    if update["perspective"] == perspective:
                        confidence_history.append(update["new_confidence"])
            
            evolution[perspective] = confidence_history
        
        return evolution
    
    # Caching methods
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached reasoning result"""
        # Check in-memory cache first
        if cache_key in self.reasoning_cache:
            cached = self.reasoning_cache[cache_key]
            if datetime.utcnow() - cached["timestamp"] < self.cache_ttl:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached["result"]
            else:
                # Expired
                del self.reasoning_cache[cache_key]
        
        # Check Redis if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"reasoning:{cache_key}")
                if cached_data:
                    cached = pickle.loads(cached_data)
                    if datetime.utcnow() - cached["timestamp"] < self.cache_ttl:
                        logger.info(f"Redis cache hit for key: {cache_key}")
                        return cached["result"]
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache reasoning result"""
        cached_entry = {
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
        # Store in memory
        self.reasoning_cache[cache_key] = cached_entry
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"reasoning:{cache_key}",
                    int(self.cache_ttl.total_seconds()),
                    pickle.dumps(cached_entry)
                )
                logger.info(f"Cached result in Redis for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
    
    # Semantic analysis helper methods for real reasoning
    
    def _extract_entities_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Extract named entities and key concepts from context"""
        entities = []
        
        for key, value in context.items():
            if isinstance(value, str):
                # Extract capitalized words (potential entities)
                words = value.split()
                entities.extend([w for w in words if w[0].isupper() and len(w) > 2])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 2:
                        entities.append(item)
        
        return list(set(entities))  # Remove duplicates
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question into reasoning categories"""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ["why", "because", "cause", "reason"]):
            return "causal"
        elif any(word in q_lower for word in ["how", "process", "method", "way"]):  
            return "process"
        elif any(word in q_lower for word in ["what", "define", "meaning", "is"]):
            return "definitional"
        elif any(word in q_lower for word in ["when", "where", "time", "location"]):
            return "contextual"
        else:
            return "general"
    
    def _extract_key_concepts(self, question: str, context: Dict[str, Any]) -> List[str]:
        """Extract key concepts using frequency and context analysis"""
        # Combine question and context text
        text = question + " "
        for key, value in context.items():
            if isinstance(value, str):
                text += value + " "
        
        words = text.lower().split()
        
        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "what", "how", "why", "when", "where", "which", "who", "whom"
        }
        
        # Filter and count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top concepts
        concepts = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)
        return concepts[:5]
    
    def _find_related_concepts(self, concept: str, entities: List[str]) -> List[str]:
        """Find related concepts using semantic similarity"""
        related = []
        
        # Semantic relationships based on domain knowledge
        concept_relations = {
            "system": ["component", "process", "function", "structure"],
            "process": ["method", "procedure", "algorithm", "workflow"],
            "problem": ["solution", "issue", "challenge", "difficulty"],
            "data": ["information", "knowledge", "facts", "evidence"],
            "model": ["framework", "structure", "template", "pattern"],
            "network": ["connection", "node", "link", "relationship"],
            "analysis": ["evaluation", "assessment", "examination", "study"]
        }
        
        # Check for direct relations
        for key, values in concept_relations.items():
            if concept in key or key in concept:
                related.extend(values)
            elif any(v in concept or concept in v for v in values):
                related.append(key)
        
        # Add entities that share semantic roots
        for entity in entities:
            if len(entity) > 3 and (concept[:3] == entity[:3] or concept[-3:] == entity[-3:]):
                related.append(entity)
        
        return related[:3]
    
    def _calculate_concept_prior(self, concept: str, context: Dict[str, Any]) -> float:
        """Calculate prior probability of concept based on context frequency"""
        total_words = 0
        concept_count = 0
        
        for key, value in context.items():
            if isinstance(value, str):
                words = value.lower().split()
                total_words += len(words)
                concept_count += words.count(concept.lower())
        
        if total_words == 0:
            return 0.3
        
        frequency = (concept_count + 1) / (total_words + 10)
        return min(0.8, max(0.1, frequency * 10))
    
    def _infer_processes(self, concept: str, context: Dict[str, Any]) -> List[str]:
        """Infer possible processes related to a concept"""
        processes = []
        
        if any(term in concept for term in ["system", "network", "structure"]):
            processes = ["initialization", "operation", "optimization", "maintenance"]
        elif any(term in concept for term in ["data", "information", "knowledge"]):
            processes = ["collection", "processing", "analysis", "storage"]
        elif any(term in concept for term in ["problem", "issue", "challenge"]):
            processes = ["identification", "analysis", "solution_design", "implementation"]
        elif any(term in concept for term in ["model", "algorithm", "method"]):
            processes = ["design", "implementation", "testing", "refinement"]
        else:
            processes = ["development", "implementation", "evaluation", "improvement"]
        
        return processes
    
    def _calculate_process_likelihood(self, process: str, context: Dict[str, Any]) -> float:
        """Calculate likelihood of a process being relevant"""
        process_indicators = {
            "initialization": ["start", "begin", "create", "setup"],
            "operation": ["run", "execute", "perform", "operate"],
            "optimization": ["improve", "enhance", "optimize", "refine"],
            "analysis": ["analyze", "examine", "study", "evaluate"],
            "design": ["create", "build", "construct", "develop"]
        }
        
        indicators = process_indicators.get(process, [process])
        
        score = 0
        total_text = ""
        for key, value in context.items():
            if isinstance(value, str):
                total_text += value.lower() + " "
        
        for indicator in indicators:
            if indicator in total_text:
                score += 0.2
        
        return min(0.9, max(0.2, 0.4 + score))
    
    def _generate_concept_definitions(self, concept: str, context: Dict[str, Any]) -> List[str]:
        """Generate possible definitions for a concept"""
        definitions = []
        
        context_text = ""
        for key, value in context.items():
            if isinstance(value, str):
                context_text += value + " "
        
        sentences = context_text.split(". ")
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                if any(pattern in sentence.lower() for pattern in ["is", "are", "means", "refers to"]):
                    parts = sentence.lower().split(concept.lower())
                    if len(parts) > 1:
                        definition_part = parts[1].strip()
                        if definition_part:
                            definitions.append(definition_part[:100])
        
        if not definitions:
            concept_type = self._infer_concept_type(concept)
            if concept_type == "process":
                definitions = [f"a systematic approach involving {concept}", f"a method that utilizes {concept}"]
            elif concept_type == "system":
                definitions = [f"a structured framework based on {concept}", f"an organized system utilizing {concept}"]
            else:
                definitions = [f"a concept fundamentally related to {concept}", f"an entity characterized by {concept}"]
        
        return definitions[:2]
    
    def _calculate_definition_confidence(self, definition: str, context: Dict[str, Any]) -> float:
        """Calculate confidence in a definition"""
        if any(isinstance(v, str) and definition[:20] in v for v in context.values()):
            return 0.8
        else:
            return 0.4
    
    def _analyze_semantic_relations(self, question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze semantic relationships"""
        relations = []
        question_concepts = self._extract_key_concepts(question, context)
        
        for i, concept1 in enumerate(question_concepts):
            for j, concept2 in enumerate(question_concepts[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(concept1, concept2)
                if similarity > 0.3:
                    relation = {
                        "concepts": [concept1, concept2],
                        "type": "similarity",
                        "confidence": similarity,
                        "relationship": f"{concept1} relates to {concept2}"
                    }
                    relations.append(relation)
        
        context_text = " ".join(str(v) for v in context.values() if isinstance(v, str))
        for concept in question_concepts:
            if concept in context_text.lower():
                relation = {
                    "concepts": [concept],
                    "type": "contextual",
                    "confidence": 0.7,
                    "relationship": f"{concept} is referenced in context"
                }
                relations.append(relation)
        
        return sorted(relations, key=lambda r: r["confidence"], reverse=True)
    
    def _calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity between words"""
        char_sim = len(set(word1.lower()) & set(word2.lower())) / len(set(word1.lower()) | set(word2.lower()))
        len_sim = 1 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
        return (char_sim * 0.6 + len_sim * 0.4)
    
    def _generate_hypothesis_from_relation(self, relation: Dict[str, Any], question: str) -> str:
        """Generate hypothesis text from semantic relation"""
        concepts = relation["concepts"]
        rel_type = relation["type"]
        
        if rel_type == "similarity" and len(concepts) >= 2:
            return f"The relationship between {concepts[0]} and {concepts[1]} provides the key to answering this question"
        elif rel_type == "contextual":
            return f"The question is answered through understanding {concepts[0]} within the given context"
        else:
            return f"The solution involves the interaction of {', '.join(concepts)}"
    
    def _infer_concept_type(self, concept: str) -> str:
        """Infer the type of a concept"""
        if any(suffix in concept for suffix in ["ing", "tion", "ment", "ance"]):
            return "process"
        elif any(prefix in concept for prefix in ["system", "network", "structure"]):
            return "system"
        elif concept.endswith("s") and len(concept) > 3:
            return "collection"
        else:
            return "entity"