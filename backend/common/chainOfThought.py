"""
Chain-of-thought reasoning implementation for A2A agents.
Provides step-by-step reasoning capabilities for complex QA tasks.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """Types of reasoning steps."""
    DECOMPOSITION = "decomposition"
    ANALYSIS = "analysis"
    INFERENCE = "inference"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"


class EvidenceType(Enum):
    """Types of evidence."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONTEXTUAL = "contextual"
    STATISTICAL = "statistical"
    COMPARATIVE = "comparative"


@dataclass
class ThoughtNode:
    """Individual thought in reasoning chain."""
    step_id: str
    step_type: ReasoningStep
    content: str
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sub_thoughts: List['ThoughtNode'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete chain of reasoning."""
    question: str
    root_thought: ThoughtNode
    conclusion: Optional[str] = None
    confidence: float = 0.0
    total_steps: int = 0
    reasoning_time: float = 0.0
    evidence_summary: Dict[str, Any] = field(default_factory=dict)


class ChainOfThoughtReasoner:
    """
    Implements chain-of-thought reasoning for complex QA tasks.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        min_confidence: float = 0.7,
        enable_parallel: bool = True
    ):
        """
        Initialize chain-of-thought reasoner.
        
        Args:
            max_depth: Maximum reasoning depth
            min_confidence: Minimum confidence threshold
            enable_parallel: Enable parallel reasoning branches
        """
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.enable_parallel = enable_parallel
        
        # Reasoning strategies
        self.reasoning_strategies = {
            "factual": self._factual_reasoning,
            "comparative": self._comparative_reasoning,
            "analytical": self._analytical_reasoning,
            "synthetic": self._synthetic_reasoning
        }
        
        # Reasoning history
        self.reasoning_history = []
    
    async def reason(
        self,
        question: str,
        context: Dict[str, Any],
        strategy: str = "analytical"
    ) -> ReasoningChain:
        """
        Perform chain-of-thought reasoning on a question.
        
        Args:
            question: Question to reason about
            context: Context information
            strategy: Reasoning strategy to use
            
        Returns:
            Complete reasoning chain
        """
        start_time = datetime.now()
        
        # Initialize root thought
        root_thought = ThoughtNode(
            step_id="root",
            step_type=ReasoningStep.DECOMPOSITION,
            content=f"Understanding question: {question}",
            confidence=1.0,
            metadata={"strategy": strategy}
        )
        
        # Create reasoning chain
        chain = ReasoningChain(
            question=question,
            root_thought=root_thought
        )
        
        try:
            # Select reasoning strategy
            strategy_func = self.reasoning_strategies.get(
                strategy,
                self._analytical_reasoning
            )
            
            # Execute reasoning
            await strategy_func(root_thought, question, context, depth=0)
            
            # Synthesize conclusion
            chain.conclusion = await self._synthesize_conclusion(root_thought)
            chain.confidence = self._calculate_chain_confidence(root_thought)
            chain.total_steps = self._count_steps(root_thought)
            chain.reasoning_time = (datetime.now() - start_time).total_seconds()
            chain.evidence_summary = self._summarize_evidence(root_thought)
            
            # Record in history
            self.reasoning_history.append({
                'timestamp': start_time,
                'question': question,
                'strategy': strategy,
                'conclusion': chain.conclusion,
                'confidence': chain.confidence,
                'steps': chain.total_steps,
                'time': chain.reasoning_time
            })
            
            return chain
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            chain.conclusion = f"Unable to reason: {str(e)}"
            chain.confidence = 0.0
            return chain
    
    async def _analytical_reasoning(
        self,
        node: ThoughtNode,
        question: str,
        context: Dict[str, Any],
        depth: int
    ):
        """Analytical reasoning strategy."""
        if depth >= self.max_depth:
            return
        
        # Step 1: Decompose question
        decomposition = await self._decompose_question(question, context)
        for i, component in enumerate(decomposition):
            sub_thought = ThoughtNode(
                step_id=f"{node.step_id}.decomp.{i}",
                step_type=ReasoningStep.DECOMPOSITION,
                content=f"Component {i+1}: {component['question']}",
                confidence=component['confidence'],
                metadata={"component_type": component['type']}
            )
            node.sub_thoughts.append(sub_thought)
            
            # Step 2: Analyze each component
            await self._analyze_component(
                sub_thought,
                component,
                context,
                depth + 1
            )
        
        # Step 3: Validate findings
        validation_thought = ThoughtNode(
            step_id=f"{node.step_id}.validate",
            step_type=ReasoningStep.VALIDATION,
            content="Validating analytical findings",
            confidence=0.9
        )
        node.sub_thoughts.append(validation_thought)
        
        await self._validate_findings(validation_thought, node.sub_thoughts, context)
    
    async def _decompose_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose question into components."""
        components = []
        
        # Identify question type
        question_lower = question.lower()
        
        if "what" in question_lower:
            components.append({
                "type": "definition",
                "question": f"Define the key concepts in: {question}",
                "confidence": 0.9
            })
        
        if "why" in question_lower:
            components.append({
                "type": "causal",
                "question": f"Identify causal relationships in: {question}",
                "confidence": 0.85
            })
        
        if "how" in question_lower:
            components.append({
                "type": "procedural",
                "question": f"Describe the process/mechanism in: {question}",
                "confidence": 0.85
            })
        
        if "when" in question_lower or "where" in question_lower:
            components.append({
                "type": "contextual",
                "question": f"Identify temporal/spatial context in: {question}",
                "confidence": 0.9
            })
        
        # Add factual verification component
        components.append({
            "type": "factual",
            "question": f"Verify factual claims in: {question}",
            "confidence": 0.95
        })
        
        return components
    
    async def _analyze_component(
        self,
        node: ThoughtNode,
        component: Dict[str, Any],
        context: Dict[str, Any],
        depth: int
    ):
        """Analyze a question component."""
        analysis_type = component['type']
        
        # Create analysis thought
        analysis = ThoughtNode(
            step_id=f"{node.step_id}.analysis",
            step_type=ReasoningStep.ANALYSIS,
            content=f"Analyzing {analysis_type} aspects",
            confidence=component['confidence']
        )
        node.sub_thoughts.append(analysis)
        
        # Gather evidence
        evidence = await self._gather_evidence(
            component['question'],
            context,
            analysis_type
        )
        
        for i, ev in enumerate(evidence):
            evidence_thought = ThoughtNode(
                step_id=f"{analysis.step_id}.evidence.{i}",
                step_type=ReasoningStep.INFERENCE,
                content=f"Evidence: {ev['content']}",
                confidence=ev['confidence'],
                evidence=[ev],
                metadata={"evidence_type": ev['type']}
            )
            analysis.sub_thoughts.append(evidence_thought)
            
            # Make inferences if confidence is high enough
            if ev['confidence'] >= self.min_confidence:
                inference = await self._make_inference(ev, context)
                if inference:
                    inference_thought = ThoughtNode(
                        step_id=f"{evidence_thought.step_id}.inference",
                        step_type=ReasoningStep.INFERENCE,
                        content=inference['content'],
                        confidence=inference['confidence'],
                        metadata={"inference_type": inference['type']}
                    )
                    evidence_thought.sub_thoughts.append(inference_thought)
    
    async def _gather_evidence(
        self,
        question: str,
        context: Dict[str, Any],
        evidence_type: str
    ) -> List[Dict[str, Any]]:
        """Gather evidence for reasoning."""
        evidence = []
        
        # Direct evidence from context
        if 'data' in context:
            for key, value in context['data'].items():
                if self._is_relevant(key, question):
                    evidence.append({
                        'type': EvidenceType.DIRECT.value,
                        'content': f"{key}: {value}",
                        'confidence': 0.95,
                        'source': 'context_data'
                    })
        
        # Contextual evidence
        if 'metadata' in context:
            relevant_metadata = self._extract_relevant_metadata(
                context['metadata'],
                question
            )
            for meta in relevant_metadata:
                evidence.append({
                    'type': EvidenceType.CONTEXTUAL.value,
                    'content': meta['content'],
                    'confidence': meta['confidence'],
                    'source': 'metadata'
                })
        
        # Statistical evidence if available
        if 'statistics' in context:
            stats = self._analyze_statistics(context['statistics'], question)
            for stat in stats:
                evidence.append({
                    'type': EvidenceType.STATISTICAL.value,
                    'content': stat['content'],
                    'confidence': stat['confidence'],
                    'source': 'statistics'
                })
        
        return evidence
    
    def _is_relevant(self, key: str, question: str) -> bool:
        """Check if data key is relevant to question."""
        # Simple keyword matching (can be enhanced)
        key_words = key.lower().split('_')
        question_words = question.lower().split()
        
        return any(kw in question_words for kw in key_words)
    
    def _extract_relevant_metadata(
        self,
        metadata: Dict[str, Any],
        question: str
    ) -> List[Dict[str, Any]]:
        """Extract relevant metadata."""
        relevant = []
        
        for key, value in metadata.items():
            if self._is_relevant(key, question):
                relevant.append({
                    'content': f"Metadata {key}: {value}",
                    'confidence': 0.8
                })
        
        return relevant
    
    def _analyze_statistics(
        self,
        statistics: Dict[str, Any],
        question: str
    ) -> List[Dict[str, Any]]:
        """Analyze statistical information."""
        stats = []
        
        for stat_name, stat_value in statistics.items():
            if isinstance(stat_value, (int, float)):
                stats.append({
                    'content': f"Statistical measure {stat_name}: {stat_value}",
                    'confidence': 0.85
                })
        
        return stats
    
    async def _make_inference(
        self,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make inference from evidence."""
        evidence_type = evidence.get('type', '')
        content = evidence.get('content', '')
        
        # Simple inference rules (can be enhanced with ML)
        if evidence_type == EvidenceType.DIRECT.value:
            return {
                'type': 'direct_inference',
                'content': f"Based on direct evidence: {content}",
                'confidence': evidence['confidence'] * 0.9
            }
        elif evidence_type == EvidenceType.STATISTICAL.value:
            return {
                'type': 'statistical_inference',
                'content': f"Statistical pattern suggests: {content}",
                'confidence': evidence['confidence'] * 0.85
            }
        
        return None
    
    async def _validate_findings(
        self,
        node: ThoughtNode,
        findings: List[ThoughtNode],
        context: Dict[str, Any]
    ):
        """Validate reasoning findings."""
        # Cross-validate findings
        validations = []
        
        for i, finding in enumerate(findings):
            if finding.step_type == ReasoningStep.DECOMPOSITION:
                # Validate decomposition completeness
                validation = self._validate_decomposition(finding, context)
                validations.append(validation)
        
        # Calculate overall validation confidence
        if validations:
            avg_confidence = sum(v['confidence'] for v in validations) / len(validations)
            node.confidence = avg_confidence
            node.content = f"Validation complete with {avg_confidence:.2f} confidence"
    
    def _validate_decomposition(
        self,
        node: ThoughtNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate decomposition completeness."""
        # Check if all aspects are covered
        covered_aspects = set()
        for sub in node.sub_thoughts:
            if 'component_type' in sub.metadata:
                covered_aspects.add(sub.metadata['component_type'])
        
        expected_aspects = {'definition', 'factual', 'contextual'}
        coverage = len(covered_aspects & expected_aspects) / len(expected_aspects)
        
        return {
            'type': 'decomposition_validation',
            'confidence': coverage,
            'details': f"Covered {len(covered_aspects)} aspects"
        }
    
    async def _synthesize_conclusion(self, root: ThoughtNode) -> str:
        """Synthesize conclusion from reasoning chain."""
        # Collect all inference nodes
        inferences = []
        self._collect_nodes_by_type(root, ReasoningStep.INFERENCE, inferences)
        
        if not inferences:
            return "Unable to reach conclusion due to insufficient evidence"
        
        # Sort by confidence
        inferences.sort(key=lambda n: n.confidence, reverse=True)
        
        # Build conclusion from top inferences
        conclusion_parts = []
        for inf in inferences[:3]:  # Top 3 inferences
            if inf.confidence >= self.min_confidence:
                conclusion_parts.append(inf.content)
        
        if conclusion_parts:
            return f"Based on chain-of-thought reasoning: {' Additionally, '.join(conclusion_parts)}"
        else:
            return "Confidence too low to form definitive conclusion"
    
    def _collect_nodes_by_type(
        self,
        node: ThoughtNode,
        step_type: ReasoningStep,
        collection: List[ThoughtNode]
    ):
        """Recursively collect nodes of specific type."""
        if node.step_type == step_type:
            collection.append(node)
        
        for sub in node.sub_thoughts:
            self._collect_nodes_by_type(sub, step_type, collection)
    
    def _calculate_chain_confidence(self, root: ThoughtNode) -> float:
        """Calculate overall chain confidence."""
        all_nodes = []
        self._collect_all_nodes(root, all_nodes)
        
        if not all_nodes:
            return 0.0
        
        # Weighted average based on step type
        weights = {
            ReasoningStep.DECOMPOSITION: 0.1,
            ReasoningStep.ANALYSIS: 0.2,
            ReasoningStep.INFERENCE: 0.3,
            ReasoningStep.VALIDATION: 0.2,
            ReasoningStep.SYNTHESIS: 0.15,
            ReasoningStep.CONCLUSION: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for node in all_nodes:
            weight = weights.get(node.step_type, 0.1)
            weighted_sum += node.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _collect_all_nodes(
        self,
        node: ThoughtNode,
        collection: List[ThoughtNode]
    ):
        """Recursively collect all nodes."""
        collection.append(node)
        for sub in node.sub_thoughts:
            self._collect_all_nodes(sub, collection)
    
    def _count_steps(self, root: ThoughtNode) -> int:
        """Count total reasoning steps."""
        count = 1
        for sub in root.sub_thoughts:
            count += self._count_steps(sub)
        return count
    
    def _summarize_evidence(self, root: ThoughtNode) -> Dict[str, Any]:
        """Summarize evidence used in reasoning."""
        evidence_nodes = []
        self._collect_nodes_with_evidence(root, evidence_nodes)
        
        summary = {
            'total_evidence': sum(len(n.evidence) for n in evidence_nodes),
            'evidence_types': {},
            'high_confidence_evidence': [],
            'sources': set()
        }
        
        for node in evidence_nodes:
            for ev in node.evidence:
                ev_type = ev.get('type', 'unknown')
                summary['evidence_types'][ev_type] = summary['evidence_types'].get(ev_type, 0) + 1
                
                if ev.get('confidence', 0) >= 0.9:
                    summary['high_confidence_evidence'].append({
                        'content': ev.get('content', '')[:100],
                        'confidence': ev.get('confidence', 0)
                    })
                
                if 'source' in ev:
                    summary['sources'].add(ev['source'])
        
        summary['sources'] = list(summary['sources'])
        return summary
    
    def _collect_nodes_with_evidence(
        self,
        node: ThoughtNode,
        collection: List[ThoughtNode]
    ):
        """Collect nodes that have evidence."""
        if node.evidence:
            collection.append(node)
        
        for sub in node.sub_thoughts:
            self._collect_nodes_with_evidence(sub, collection)
    
    async def _factual_reasoning(
        self,
        node: ThoughtNode,
        question: str,
        context: Dict[str, Any],
        depth: int
    ):
        """Factual reasoning strategy."""
        # Focus on verifying facts
        fact_check = ThoughtNode(
            step_id=f"{node.step_id}.factcheck",
            step_type=ReasoningStep.VALIDATION,
            content="Verifying factual claims",
            confidence=0.95
        )
        node.sub_thoughts.append(fact_check)
        
        # Extract and verify each fact
        facts = self._extract_facts(question, context)
        for i, fact in enumerate(facts):
            verification = await self._verify_fact(fact, context)
            fact_node = ThoughtNode(
                step_id=f"{fact_check.step_id}.fact.{i}",
                step_type=ReasoningStep.ANALYSIS,
                content=f"Fact: {fact['claim']} - {verification['result']}",
                confidence=verification['confidence'],
                evidence=[verification]
            )
            fact_check.sub_thoughts.append(fact_node)
    
    async def _comparative_reasoning(
        self,
        node: ThoughtNode,
        question: str,
        context: Dict[str, Any],
        depth: int
    ):
        """Comparative reasoning strategy."""
        # Focus on comparisons and relationships
        comparison = ThoughtNode(
            step_id=f"{node.step_id}.compare",
            step_type=ReasoningStep.ANALYSIS,
            content="Analyzing comparisons and relationships",
            confidence=0.85
        )
        node.sub_thoughts.append(comparison)
        
        # Identify entities to compare
        entities = self._extract_entities(question, context)
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                for j in range(i + 1, len(entities)):
                    comp_result = await self._compare_entities(
                        entities[i],
                        entities[j],
                        context
                    )
                    comp_node = ThoughtNode(
                        step_id=f"{comparison.step_id}.comp.{i}.{j}",
                        step_type=ReasoningStep.INFERENCE,
                        content=comp_result['comparison'],
                        confidence=comp_result['confidence']
                    )
                    comparison.sub_thoughts.append(comp_node)
    
    async def _synthetic_reasoning(
        self,
        node: ThoughtNode,
        question: str,
        context: Dict[str, Any],
        depth: int
    ):
        """Synthetic reasoning strategy."""
        # Focus on combining information
        synthesis = ThoughtNode(
            step_id=f"{node.step_id}.synthesis",
            step_type=ReasoningStep.SYNTHESIS,
            content="Synthesizing information from multiple sources",
            confidence=0.8
        )
        node.sub_thoughts.append(synthesis)
        
        # Gather all available information
        info_sources = self._identify_information_sources(context)
        for source_name, source_data in info_sources.items():
            source_node = ThoughtNode(
                step_id=f"{synthesis.step_id}.source.{source_name}",
                step_type=ReasoningStep.ANALYSIS,
                content=f"Information from {source_name}",
                confidence=0.85,
                metadata={"source": source_name}
            )
            synthesis.sub_thoughts.append(source_node)
    
    def _extract_facts(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract factual claims from question."""
        # Simple extraction (can be enhanced with NLP)
        facts = []
        
        # Look for statements that can be verified
        if "is" in question or "are" in question:
            facts.append({
                'claim': question,
                'type': 'existence'
            })
        
        return facts
    
    async def _verify_fact(
        self,
        fact: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify a factual claim."""
        # Simple verification (can be enhanced)
        return {
            'result': 'verified',
            'confidence': 0.85,
            'evidence': context.get('data', {})
        }
    
    def _extract_entities(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract entities from question."""
        # Simple entity extraction (can be enhanced with NER)
        entities = []
        
        # Extract capitalized words as potential entities
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.append({
                    'name': word,
                    'type': 'unknown'
                })
        
        return entities
    
    async def _compare_entities(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two entities."""
        return {
            'comparison': f"{entity1['name']} vs {entity2['name']}: similar",
            'confidence': 0.7
        }
    
    def _identify_information_sources(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify available information sources."""
        sources = {}
        
        for key in ['data', 'metadata', 'statistics', 'history']:
            if key in context and context[key]:
                sources[key] = context[key]
        
        return sources
    
    def visualize_chain(self, chain: ReasoningChain) -> str:
        """
        Generate a text visualization of the reasoning chain.
        
        Args:
            chain: Reasoning chain to visualize
            
        Returns:
            Text representation of the chain
        """
        lines = []
        lines.append(f"Question: {chain.question}")
        lines.append(f"Conclusion: {chain.conclusion}")
        lines.append(f"Confidence: {chain.confidence:.2f}")
        lines.append(f"Total Steps: {chain.total_steps}")
        lines.append(f"Reasoning Time: {chain.reasoning_time:.2f}s")
        lines.append("\nReasoning Chain:")
        
        self._visualize_node(chain.root_thought, lines, indent=0)
        
        return '\n'.join(lines)
    
    def _visualize_node(
        self,
        node: ThoughtNode,
        lines: List[str],
        indent: int
    ):
        """Recursively visualize node."""
        prefix = "  " * indent + "→ "
        lines.append(
            f"{prefix}[{node.step_type.value}] {node.content} "
            f"(confidence: {node.confidence:.2f})"
        )
        
        if node.evidence:
            lines.append(f"{'  ' * (indent + 1)}Evidence: {len(node.evidence)} pieces")
        
        for sub in node.sub_thoughts:
            self._visualize_node(sub, lines, indent + 1)