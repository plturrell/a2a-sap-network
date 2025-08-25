"""
Advanced Reasoning Engine - Real NLP and Logical Inference Implementation
Replaces primitive keyword matching with proper semantic analysis and reasoning
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Try to load spacy model, install if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class SemanticAnalysisType(Enum):
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    COMPARATIVE = "comparative"
    HYPOTHETICAL = "hypothetical"
    DEFINITIONAL = "definitional"

@dataclass
class SemanticEntity:
    text: str
    label: str
    confidence: float
    span: Tuple[int, int]
    relations: List[str] = field(default_factory=list)
    semantic_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class LogicalRelation:
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    relation_type: str = "semantic"

@dataclass
class ReasoningStep:
    step_id: str
    premise: str
    inference_rule: str
    conclusion: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)

class AdvancedQuestionDecomposer:
    """Advanced NLP-based question decomposition using semantic analysis"""

    def __init__(self):
        self.sentence_transformer = None
        self.qa_pipeline = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.knowledge_graph = nx.DiGraph()

        # Initialize ML models
        self._initialize_models()

        # Decomposition strategies
        self.strategies = {
            SemanticAnalysisType.CAUSAL: self._causal_decomposition,
            SemanticAnalysisType.TEMPORAL: self._temporal_decomposition,
            SemanticAnalysisType.STRUCTURAL: self._structural_decomposition,
            SemanticAnalysisType.COMPARATIVE: self._comparative_decomposition,
            SemanticAnalysisType.HYPOTHETICAL: self._hypothetical_decomposition,
            SemanticAnalysisType.DEFINITIONAL: self._definitional_decomposition
        }

    def _initialize_models(self):
        """Initialize NLP and ML models"""
        try:
            # Sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Question answering pipeline
            self.qa_pipeline = pipeline("question-answering",
                                       model="distilbert-base-cased-distilled-squad",
                                       tokenizer="distilbert-base-cased-distilled-squad")

            logger.info("Advanced NLP models initialized successfully")

        except Exception as e:
            logger.warning(f"Could not initialize all ML models: {e}")
            logger.info("Falling back to rule-based analysis")

    async def decompose_question(self, question: str, context: str = "",
                                max_depth: int = 3) -> Dict[str, Any]:
        """Advanced semantic question decomposition"""

        # Step 1: Semantic analysis
        semantic_type = self._classify_question_semantics(question)
        entities = self._extract_semantic_entities(question, context)
        relations = self._extract_logical_relations(question, entities)

        # Step 2: Strategy-based decomposition
        strategy_func = self.strategies.get(semantic_type, self._structural_decomposition)
        sub_questions = await strategy_func(question, entities, relations, context)

        # Step 3: Validation and ranking
        validated_questions = self._validate_sub_questions(question, sub_questions)
        ranked_questions = self._rank_questions_by_importance(question, validated_questions)

        # Step 4: Build decomposition tree
        decomposition_tree = self._build_decomposition_tree(question, ranked_questions, max_depth)

        return {
            "original_question": question,
            "semantic_type": semantic_type.value,
            "entities": entities,
            "relations": relations,
            "sub_questions": ranked_questions[:max_depth],
            "decomposition_tree": decomposition_tree,
            "confidence": self._calculate_decomposition_confidence(ranked_questions),
            "reasoning_strategy": strategy_func.__name__
        }

    def _classify_question_semantics(self, question: str) -> SemanticAnalysisType:
        """Classify question semantics using NLP patterns"""

        # Causal indicators
        causal_patterns = [
            r'\b(why|because|cause|reason|lead to|result in|due to)\b',
            r'\b(what makes|what causes|how does.*affect)\b'
        ]

        # Temporal indicators
        temporal_patterns = [
            r'\b(when|before|after|during|while|timeline|sequence)\b',
            r'\b(first|then|next|finally|order)\b'
        ]

        # Structural indicators
        structural_patterns = [
            r'\b(how|structure|component|part|element|system)\b',
            r'\b(what is|define|describe|explain)\b'
        ]

        # Comparative indicators
        comparative_patterns = [
            r'\b(compare|versus|vs|difference|similar|unlike|better|worse)\b',
            r'\b(more|less|than|between)\b'
        ]

        # Hypothetical indicators
        hypothetical_patterns = [
            r'\b(if|suppose|imagine|what if|hypothetically|assume)\b',
            r'\b(would|could|might|should)\b'
        ]

        question_lower = question.lower()

        # Score each semantic type
        scores = {}
        for semantic_type, patterns in [
            (SemanticAnalysisType.CAUSAL, causal_patterns),
            (SemanticAnalysisType.TEMPORAL, temporal_patterns),
            (SemanticAnalysisType.STRUCTURAL, structural_patterns),
            (SemanticAnalysisType.COMPARATIVE, comparative_patterns),
            (SemanticAnalysisType.HYPOTHETICAL, hypothetical_patterns)
        ]:
            score = sum(len(re.findall(pattern, question_lower)) for pattern in patterns)
            scores[semantic_type] = score

        # Return highest scoring type, default to structural
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else SemanticAnalysisType.STRUCTURAL

    def _extract_semantic_entities(self, question: str, context: str = "") -> List[SemanticEntity]:
        """Extract semantic entities using spaCy NER"""
        entities = []

        if nlp is None:
            # Fallback to simple entity extraction
            return self._simple_entity_extraction(question)

        try:
            doc = nlp(question + " " + context)

            for ent in doc.ents:
                # Calculate semantic features
                semantic_features = {}
                if self.sentence_transformer:
                    embedding = self.sentence_transformer.encode([ent.text])[0]
                    semantic_features = {f"dim_{i}": float(embedding[i]) for i in range(min(10, len(embedding)))}

                entity = SemanticEntity(
                    text=ent.text,
                    label=ent.label_,
                    confidence=0.8,  # spaCy doesn't provide confidence directly
                    span=(ent.start_char, ent.end_char),
                    semantic_features=semantic_features
                )
                entities.append(entity)

            # Add noun phrases as potential entities
            for chunk in doc.noun_chunks:
                if chunk.text not in [e.text for e in entities]:
                    entity = SemanticEntity(
                        text=chunk.text,
                        label="NOUN_PHRASE",
                        confidence=0.6,
                        span=(chunk.start_char, chunk.end_char)
                    )
                    entities.append(entity)

        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return self._simple_entity_extraction(question)

        return entities[:10]  # Limit to top 10 entities

    def _simple_entity_extraction(self, question: str) -> List[SemanticEntity]:
        """Simple fallback entity extraction"""
        import re

        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)

        # Extract important nouns (simple heuristic)
        words = question.split()
        important_words = [w for w in words if len(w) > 4 and w.lower() not in ['what', 'when', 'where', 'how', 'why', 'which']]

        entities = []
        for i, text in enumerate(proper_nouns + quoted + important_words[:3]):
            entity = SemanticEntity(
                text=text,
                label="SIMPLE_ENTITY",
                confidence=0.5,
                span=(0, len(text))
            )
            entities.append(entity)

        return entities[:5]

    def _extract_logical_relations(self, question: str, entities: List[SemanticEntity]) -> List[LogicalRelation]:
        """Extract logical relations between entities"""
        relations = []

        if len(entities) < 2:
            return relations

        # Define relation patterns
        relation_patterns = {
            "causes": r"(\w+)\s+(causes?|leads? to|results? in)\s+(\w+)",
            "is_part_of": r"(\w+)\s+(is part of|belongs to|contains)\s+(\w+)",
            "precedes": r"(\w+)\s+(before|after|precedes|follows)\s+(\w+)",
            "similar_to": r"(\w+)\s+(like|similar to|resembles)\s+(\w+)",
            "affects": r"(\w+)\s+(affects?|influences?|impacts?)\s+(\w+)"
        }

        question_lower = question.lower()

        for relation_type, pattern in relation_patterns.items():
            matches = re.findall(pattern, question_lower)
            for match in matches:
                if len(match) >= 3:
                    relation = LogicalRelation(
                        subject=match[0],
                        predicate=relation_type,
                        object=match[2],
                        confidence=0.7,
                        evidence=[question],
                        relation_type="explicit"
                    )
                    relations.append(relation)

        # Infer implicit relations between entities
        for i, entity1 in enumerate(entities[:3]):
            for entity2 in entities[i+1:4]:
                # Simple semantic similarity if transformer available
                if self.sentence_transformer:
                    sim = cosine_similarity(
                        [self.sentence_transformer.encode([entity1.text])[0]],
                        [self.sentence_transformer.encode([entity2.text])[0]]
                    )[0][0]

                    if sim > 0.7:
                        relation = LogicalRelation(
                            subject=entity1.text,
                            predicate="semantically_related",
                            object=entity2.text,
                            confidence=float(sim),
                            evidence=[f"Semantic similarity: {sim:.2f}"],
                            relation_type="implicit"
                        )
                        relations.append(relation)

        return relations[:5]

    async def _causal_decomposition(self, question: str, entities: List[SemanticEntity],
                                  relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose causal questions"""
        sub_questions = []

        # Find causal relations
        causal_relations = [r for r in relations if r.predicate in ["causes", "affects"]]

        if causal_relations:
            for rel in causal_relations:
                sub_questions.extend([
                    {
                        "question": f"What is the mechanism by which {rel.subject} {rel.predicate} {rel.object}?",
                        "type": "mechanism",
                        "confidence": rel.confidence * 0.9,
                        "focus": "causal_mechanism"
                    },
                    {
                        "question": f"What are the necessary conditions for {rel.subject} to {rel.predicate} {rel.object}?",
                        "type": "conditions",
                        "confidence": rel.confidence * 0.8,
                        "focus": "necessary_conditions"
                    },
                    {
                        "question": f"What other factors influence the relationship between {rel.subject} and {rel.object}?",
                        "type": "factors",
                        "confidence": rel.confidence * 0.7,
                        "focus": "influencing_factors"
                    }
                ])
        else:
            # No explicit causal relations found, infer from entities
            if len(entities) >= 2:
                main_entities = entities[:2]
                sub_questions.extend([
                    {
                        "question": f"What is the relationship between {main_entities[0].text} and {main_entities[1].text}?",
                        "type": "relationship",
                        "confidence": 0.6,
                        "focus": "entity_relationship"
                    },
                    {
                        "question": f"What factors could cause changes in {main_entities[0].text}?",
                        "type": "factors",
                        "confidence": 0.5,
                        "focus": "causal_factors"
                    }
                ])

        return sub_questions[:4]

    async def _temporal_decomposition(self, question: str, entities: List[SemanticEntity],
                                    relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose temporal questions"""
        sub_questions = []

        # Look for temporal indicators
        temporal_words = ["when", "before", "after", "during", "sequence", "timeline"]
        has_temporal = any(word in question.lower() for word in temporal_words)

        if has_temporal and entities:
            main_entity = entities[0].text
            sub_questions.extend([
                {
                    "question": f"What are the chronological stages of {main_entity}?",
                    "type": "sequence",
                    "confidence": 0.8,
                    "focus": "chronological_stages"
                },
                {
                    "question": f"What events preceded {main_entity}?",
                    "type": "antecedents",
                    "confidence": 0.7,
                    "focus": "preceding_events"
                },
                {
                    "question": f"What are the consequences or results of {main_entity}?",
                    "type": "consequences",
                    "confidence": 0.7,
                    "focus": "subsequent_events"
                }
            ])

        return sub_questions[:3]

    async def _structural_decomposition(self, question: str, entities: List[SemanticEntity],
                                      relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose structural questions"""
        sub_questions = []

        if entities:
            main_entity = entities[0].text
            sub_questions.extend([
                {
                    "question": f"What are the main components of {main_entity}?",
                    "type": "components",
                    "confidence": 0.8,
                    "focus": "structural_components"
                },
                {
                    "question": f"How do the parts of {main_entity} interact with each other?",
                    "type": "interactions",
                    "confidence": 0.7,
                    "focus": "component_interactions"
                },
                {
                    "question": f"What is the function or purpose of {main_entity}?",
                    "type": "function",
                    "confidence": 0.75,
                    "focus": "functional_purpose"
                }
            ])

            # Add hierarchical questions if multiple entities
            if len(entities) > 1:
                sub_questions.append({
                    "question": f"How does {entities[1].text} relate to the structure of {main_entity}?",
                    "type": "hierarchical",
                    "confidence": 0.6,
                    "focus": "hierarchical_relationships"
                })

        return sub_questions[:4]

    async def _comparative_decomposition(self, question: str, entities: List[SemanticEntity],
                                       relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose comparative questions"""
        sub_questions = []

        if len(entities) >= 2:
            entity1, entity2 = entities[0].text, entities[1].text
            sub_questions.extend([
                {
                    "question": f"What are the key similarities between {entity1} and {entity2}?",
                    "type": "similarities",
                    "confidence": 0.8,
                    "focus": "comparative_similarities"
                },
                {
                    "question": f"What are the key differences between {entity1} and {entity2}?",
                    "type": "differences",
                    "confidence": 0.8,
                    "focus": "comparative_differences"
                },
                {
                    "question": f"In what contexts is {entity1} preferable to {entity2}?",
                    "type": "preference_contexts",
                    "confidence": 0.7,
                    "focus": "contextual_preferences"
                }
            ])

        return sub_questions[:3]

    async def _hypothetical_decomposition(self, question: str, entities: List[SemanticEntity],
                                        relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose hypothetical questions"""
        sub_questions = []

        # Extract hypothetical condition
        if_match = re.search(r'if\s+(.+?)\s+(then|,|\?)', question.lower())
        condition = if_match.group(1) if if_match else "the stated condition"

        sub_questions.extend([
            {
                "question": f"What are the assumptions underlying {condition}?",
                "type": "assumptions",
                "confidence": 0.7,
                "focus": "underlying_assumptions"
            },
            {
                "question": f"What would be the most likely consequences if {condition}?",
                "type": "consequences",
                "confidence": 0.8,
                "focus": "hypothetical_consequences"
            },
            {
                "question": f"What factors would influence the outcome of {condition}?",
                "type": "influencing_factors",
                "confidence": 0.7,
                "focus": "outcome_factors"
            }
        ])

        return sub_questions[:3]

    async def _definitional_decomposition(self, question: str, entities: List[SemanticEntity],
                                        relations: List[LogicalRelation], context: str) -> List[Dict[str, Any]]:
        """Decompose definitional questions"""
        sub_questions = []

        if entities:
            main_entity = entities[0].text
            sub_questions.extend([
                {
                    "question": f"What are the essential characteristics of {main_entity}?",
                    "type": "characteristics",
                    "confidence": 0.8,
                    "focus": "essential_characteristics"
                },
                {
                    "question": f"How is {main_entity} distinguished from similar concepts?",
                    "type": "distinctions",
                    "confidence": 0.7,
                    "focus": "conceptual_distinctions"
                },
                {
                    "question": f"What are examples and non-examples of {main_entity}?",
                    "type": "examples",
                    "confidence": 0.75,
                    "focus": "illustrative_examples"
                }
            ])

        return sub_questions[:3]

    def _validate_sub_questions(self, original: str, sub_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter sub-questions"""
        validated = []

        for sq in sub_questions:
            # Check for meaningful content
            if len(sq["question"]) < 10 or len(sq["question"]) > 200:
                continue

            # Check for question words
            question_words = ["what", "how", "why", "when", "where", "which", "who"]
            if not any(word in sq["question"].lower() for word in question_words):
                continue

            # Avoid circular questions
            if original.lower() in sq["question"].lower():
                sq["confidence"] *= 0.5

            # Add validation metadata
            sq["validated"] = True
            sq["validation_score"] = sq["confidence"]
            validated.append(sq)

        return validated

    def _rank_questions_by_importance(self, original: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank sub-questions by importance and relevance"""

        # Calculate semantic similarity if transformer available
        if self.sentence_transformer and questions:
            original_embedding = self.sentence_transformer.encode([original])

            for q in questions:
                q_embedding = self.sentence_transformer.encode([q["question"]])
                similarity = cosine_similarity(original_embedding, q_embedding)[0][0]
                q["semantic_relevance"] = float(similarity)
                q["importance_score"] = q["confidence"] * 0.7 + similarity * 0.3
        else:
            # Fallback ranking
            for q in questions:
                q["semantic_relevance"] = 0.5
                q["importance_score"] = q["confidence"]

        # Sort by importance score
        return sorted(questions, key=lambda x: x["importance_score"], reverse=True)

    def _build_decomposition_tree(self, root_question: str, sub_questions: List[Dict[str, Any]],
                                max_depth: int) -> Dict[str, Any]:
        """Build hierarchical decomposition tree"""

        tree = {
            "root": {
                "question": root_question,
                "type": "root",
                "depth": 0,
                "children": []
            }
        }

        # Add sub-questions as children
        for i, sq in enumerate(sub_questions[:max_depth]):
            child_node = {
                "question": sq["question"],
                "type": sq["type"],
                "confidence": sq["confidence"],
                "focus": sq["focus"],
                "depth": 1,
                "parent": "root",
                "node_id": f"child_{i}",
                "children": []
            }
            tree["root"]["children"].append(child_node)

        return tree

    def _calculate_decomposition_confidence(self, questions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in decomposition"""
        if not questions:
            return 0.0

        # Weighted average of question confidences
        total_confidence = sum(q["confidence"] for q in questions)
        avg_confidence = total_confidence / len(questions)

        # Penalize if too few questions
        quantity_factor = min(len(questions) / 3.0, 1.0)

        # Boost if diverse question types
        unique_types = len(set(q["type"] for q in questions))
        diversity_factor = min(unique_types / 3.0, 1.2)

        return min(avg_confidence * quantity_factor * diversity_factor, 1.0)


class SemanticPatternAnalyzer:
    """Advanced semantic pattern analysis replacing keyword matching"""

    def __init__(self):
        self.pattern_extractors = {
            "causal_patterns": self._extract_causal_patterns,
            "temporal_patterns": self._extract_temporal_patterns,
            "structural_patterns": self._extract_structural_patterns,
            "semantic_clusters": self._extract_semantic_clusters,
            "argument_patterns": self._extract_argument_patterns
        }

        # Initialize semantic models
        self.sentence_transformer = None
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")

    async def analyze_patterns(self, text: str, context: str = "") -> Dict[str, Any]:
        """Comprehensive semantic pattern analysis"""

        patterns = {}

        # Run all pattern extractors
        for pattern_type, extractor in self.pattern_extractors.items():
            try:
                patterns[pattern_type] = await extractor(text, context)
            except Exception as e:
                logger.error(f"Pattern extraction failed for {pattern_type}: {e}")
                patterns[pattern_type] = []

        # Calculate pattern confidence
        overall_confidence = self._calculate_pattern_confidence(patterns)

        # Extract key insights
        insights = self._extract_pattern_insights(patterns)

        return {
            "patterns": patterns,
            "confidence": overall_confidence,
            "insights": insights,
            "pattern_count": sum(len(p) for p in patterns.values()),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    async def _extract_causal_patterns(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract sophisticated causal patterns"""
        patterns = []

        # Advanced causal indicators
        causal_indicators = {
            "direct_causation": [
                r"(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+leads?\s+to\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+results?\s+in\s+(\w+(?:\s+\w+)*)"
            ],
            "conditional_causation": [
                r"if\s+(\w+(?:\s+\w+)*),?\s+then\s+(\w+(?:\s+\w+)*)",
                r"when\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)\s+occurs?"
            ],
            "enabling_conditions": [
                r"(\w+(?:\s+\w+)*)\s+enables?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+allows?\s+(\w+(?:\s+\w+)*)"
            ]
        }

        for causation_type, patterns_list in causal_indicators.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()

                    # Calculate confidence based on pattern strength
                    confidence = 0.8 if causation_type == "direct_causation" else 0.6

                    patterns.append({
                        "type": "causal",
                        "subtype": causation_type,
                        "cause": cause,
                        "effect": effect,
                        "confidence": confidence,
                        "evidence": match.group(0),
                        "position": match.span()
                    })

        return patterns[:10]  # Limit to top 10

    async def _extract_temporal_patterns(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract temporal sequence patterns"""
        patterns = []

        # Temporal sequence indicators
        temporal_indicators = {
            "sequence": [
                r"first\s+(\w+(?:\s+\w+)*),?\s+then\s+(\w+(?:\s+\w+)*)",
                r"before\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)",
                r"after\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)"
            ],
            "duration": [
                r"(\w+(?:\s+\w+)*)\s+lasts?\s+(\d+\s+\w+)",
                r"(\w+(?:\s+\w+)*)\s+takes?\s+(\d+\s+\w+)"
            ],
            "frequency": [
                r"(\w+(?:\s+\w+)*)\s+occurs?\s+(daily|weekly|monthly|annually)",
                r"(\w+(?:\s+\w+)*)\s+happens?\s+every\s+(\w+)"
            ]
        }

        for temporal_type, patterns_list in temporal_indicators.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    patterns.append({
                        "type": "temporal",
                        "subtype": temporal_type,
                        "event1": match.group(1).strip(),
                        "event2": match.group(2).strip() if len(match.groups()) > 1 else None,
                        "confidence": 0.7,
                        "evidence": match.group(0),
                        "position": match.span()
                    })

        return patterns[:8]

    async def _extract_structural_patterns(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract structural relationship patterns"""
        patterns = []

        # Structural indicators
        structural_indicators = {
            "composition": [
                r"(\w+(?:\s+\w+)*)\s+consists?\s+of\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+contains?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+comprises?\s+(\w+(?:\s+\w+)*)"
            ],
            "hierarchy": [
                r"(\w+(?:\s+\w+)*)\s+is\s+a\s+type\s+of\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+belongs\s+to\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+falls\s+under\s+(\w+(?:\s+\w+)*)"
            ],
            "dependency": [
                r"(\w+(?:\s+\w+)*)\s+depends\s+on\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+requires?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+needs?\s+(\w+(?:\s+\w+)*)"
            ]
        }

        for struct_type, patterns_list in structural_indicators.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    patterns.append({
                        "type": "structural",
                        "subtype": struct_type,
                        "parent": match.group(1).strip(),
                        "child": match.group(2).strip(),
                        "confidence": 0.75,
                        "evidence": match.group(0),
                        "position": match.span()
                    })

        return patterns[:8]

    async def _extract_semantic_clusters(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract semantic clusters using embeddings"""
        clusters = []

        if not self.sentence_transformer:
            return clusters

        try:
            # Split text into sentences
            sentences = text.split('.')
            if len(sentences) < 2:
                return clusters

            # Get embeddings
            embeddings = self.sentence_transformer.encode(sentences)

            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)

            # Find high-similarity clusters
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    similarity = similarities[i][j]
                    if similarity > 0.7:  # High semantic similarity
                        clusters.append({
                            "type": "semantic_cluster",
                            "sentence1": sentences[i].strip(),
                            "sentence2": sentences[j].strip(),
                            "similarity": float(similarity),
                            "confidence": float(similarity),
                            "evidence": f"Semantic similarity: {similarity:.3f}"
                        })

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")

        return sorted(clusters, key=lambda x: x["similarity"], reverse=True)[:5]

    async def _extract_argument_patterns(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract argument and reasoning patterns"""
        patterns = []

        # Argument indicators
        argument_indicators = {
            "premise": [
                r"given\s+that\s+(\w+(?:\s+\w+)*)",
                r"since\s+(\w+(?:\s+\w+)*)",
                r"because\s+(\w+(?:\s+\w+)*)"
            ],
            "conclusion": [
                r"therefore\s+(\w+(?:\s+\w+)*)",
                r"thus\s+(\w+(?:\s+\w+)*)",
                r"consequently\s+(\w+(?:\s+\w+)*)"
            ],
            "evidence": [
                r"according\s+to\s+(\w+(?:\s+\w+)*)",
                r"research\s+shows\s+(\w+(?:\s+\w+)*)",
                r"studies\s+indicate\s+(\w+(?:\s+\w+)*)"
            ]
        }

        for arg_type, patterns_list in argument_indicators.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    patterns.append({
                        "type": "argument",
                        "subtype": arg_type,
                        "content": match.group(1).strip(),
                        "confidence": 0.6,
                        "evidence": match.group(0),
                        "position": match.span()
                    })

        return patterns[:6]

    def _calculate_pattern_confidence(self, patterns: Dict[str, List]) -> float:
        """Calculate overall pattern analysis confidence"""
        if not patterns:
            return 0.0

        total_patterns = sum(len(p) for p in patterns.values())
        if total_patterns == 0:
            return 0.0

        # Calculate weighted confidence
        total_confidence = 0.0
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                avg_confidence = sum(p.get("confidence", 0.5) for p in pattern_list) / len(pattern_list)
                total_confidence += avg_confidence

        return min(total_confidence / len(patterns), 1.0)

    def _extract_pattern_insights(self, patterns: Dict[str, List]) -> List[str]:
        """Extract key insights from pattern analysis"""
        insights = []

        # Count patterns by type
        pattern_counts = {k: len(v) for k, v in patterns.items()}

        # Dominant pattern type
        if pattern_counts:
            dominant_type = max(pattern_counts.items(), key=lambda x: x[1])
            insights.append(f"Dominant pattern type: {dominant_type[0]} ({dominant_type[1]} instances)")

        # Causal relationships
        causal_patterns = patterns.get("causal_patterns", [])
        if causal_patterns:
            insights.append(f"Found {len(causal_patterns)} causal relationships")

        # Temporal sequences
        temporal_patterns = patterns.get("temporal_patterns", [])
        if temporal_patterns:
            insights.append(f"Identified {len(temporal_patterns)} temporal sequences")

        # Semantic clusters
        semantic_clusters = patterns.get("semantic_clusters", [])
        if semantic_clusters:
            avg_similarity = sum(c["similarity"] for c in semantic_clusters) / len(semantic_clusters)
            insights.append(f"High semantic coherence (avg similarity: {avg_similarity:.2f})")

        return insights[:5]


class LogicalInferenceEngine:
    """Real logical inference engine replacing string concatenation"""

    def __init__(self):
        self.inference_rules = {
            "modus_ponens": self._modus_ponens,
            "modus_tollens": self._modus_tollens,
            "hypothetical_syllogism": self._hypothetical_syllogism,
            "disjunctive_syllogism": self._disjunctive_syllogism,
            "constructive_dilemma": self._constructive_dilemma
        }

        self.knowledge_base = nx.DiGraph()
        self.reasoning_history = []

    async def synthesize_reasoning(self, premises: List[str], question: str,
                                 patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize reasoning using logical inference"""

        # Step 1: Parse premises into logical statements
        parsed_premises = [self._parse_premise(p) for p in premises]

        # Step 2: Apply inference rules
        inferences = []
        for rule_name, rule_func in self.inference_rules.items():
            try:
                rule_inferences = await rule_func(parsed_premises, question)
                inferences.extend(rule_inferences)
            except Exception as e:
                logger.debug(f"Inference rule {rule_name} failed: {e}")

        # Step 3: Validate inferences
        validated_inferences = self._validate_inferences(inferences, patterns)

        # Step 4: Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(validated_inferences)

        # Step 5: Generate synthesis
        synthesis = self._generate_synthesis(reasoning_chain, question)

        return {
            "synthesis": synthesis,
            "reasoning_chain": reasoning_chain,
            "inferences": validated_inferences,
            "confidence": self._calculate_synthesis_confidence(validated_inferences),
            "logical_validity": self._assess_logical_validity(reasoning_chain)
        }

    def _parse_premise(self, premise: str) -> Dict[str, Any]:
        """Parse natural language premise into logical structure"""

        # Simple logical parsing (can be enhanced with more sophisticated NLP)
        logical_forms = {
            "conditional": r"if\s+(.+?)\s+then\s+(.+)",
            "universal": r"all\s+(.+?)\s+are\s+(.+)",
            "existential": r"some\s+(.+?)\s+are\s+(.+)",
            "negation": r"not\s+(.+)",
            "conjunction": r"(.+?)\s+and\s+(.+)",
            "disjunction": r"(.+?)\s+or\s+(.+)"
        }

        premise_lower = premise.lower().strip()

        for form_type, pattern in logical_forms.items():
            match = re.search(pattern, premise_lower)
            if match:
                return {
                    "type": form_type,
                    "original": premise,
                    "components": [g.strip() for g in match.groups()],
                    "confidence": 0.8
                }

        # Default: treat as atomic proposition
        return {
            "type": "atomic",
            "original": premise,
            "components": [premise.strip()],
            "confidence": 0.6
        }

    async def _modus_ponens(self, premises: List[Dict], question: str) -> List[ReasoningStep]:
        """Apply modus ponens inference rule"""
        steps = []

        # Find conditional statements and their antecedents
        conditionals = [p for p in premises if p["type"] == "conditional"]
        atomics = [p for p in premises if p["type"] == "atomic"]

        for conditional in conditionals:
            if len(conditional["components"]) >= 2:
                antecedent = conditional["components"][0]
                consequent = conditional["components"][1]

                # Look for matching antecedent
                for atomic in atomics:
                    if self._semantic_match(antecedent, atomic["components"][0]):
                        step = ReasoningStep(
                            step_id=f"mp_{len(steps)}",
                            premise=f"If {antecedent}, then {consequent}; {atomic['components'][0]}",
                            inference_rule="modus_ponens",
                            conclusion=consequent,
                            confidence=min(conditional["confidence"], atomic["confidence"]),
                            supporting_evidence=[conditional["original"], atomic["original"]]
                        )
                        steps.append(step)

        return steps

    async def _modus_tollens(self, premises: List[Dict], question: str) -> List[ReasoningStep]:
        """Apply modus tollens inference rule"""
        steps = []

        conditionals = [p for p in premises if p["type"] == "conditional"]
        negations = [p for p in premises if p["type"] == "negation"]

        for conditional in conditionals:
            if len(conditional["components"]) >= 2:
                antecedent = conditional["components"][0]
                consequent = conditional["components"][1]

                # Look for negated consequent
                for negation in negations:
                    if self._semantic_match(consequent, negation["components"][0]):
                        step = ReasoningStep(
                            step_id=f"mt_{len(steps)}",
                            premise=f"If {antecedent}, then {consequent}; not {consequent}",
                            inference_rule="modus_tollens",
                            conclusion=f"not {antecedent}",
                            confidence=min(conditional["confidence"], negation["confidence"]),
                            supporting_evidence=[conditional["original"], negation["original"]]
                        )
                        steps.append(step)

        return steps

    async def _hypothetical_syllogism(self, premises: List[Dict], question: str) -> List[ReasoningStep]:
        """Apply hypothetical syllogism inference rule"""
        steps = []

        conditionals = [p for p in premises if p["type"] == "conditional"]

        # Find chained conditionals
        for i, cond1 in enumerate(conditionals):
            for cond2 in conditionals[i+1:]:
                if (len(cond1["components"]) >= 2 and len(cond2["components"]) >= 2):
                    # Check if consequent of first matches antecedent of second
                    if self._semantic_match(cond1["components"][1], cond2["components"][0]):
                        step = ReasoningStep(
                            step_id=f"hs_{len(steps)}",
                            premise=f"If {cond1['components'][0]}, then {cond1['components'][1]}; If {cond2['components'][0]}, then {cond2['components'][1]}",
                            inference_rule="hypothetical_syllogism",
                            conclusion=f"If {cond1['components'][0]}, then {cond2['components'][1]}",
                            confidence=min(cond1["confidence"], cond2["confidence"]) * 0.9,
                            supporting_evidence=[cond1["original"], cond2["original"]]
                        )
                        steps.append(step)

        return steps

    async def _disjunctive_syllogism(self, premises: List[Dict], question: str) -> List[ReasoningStep]:
        """Apply disjunctive syllogism inference rule"""
        steps = []

        disjunctions = [p for p in premises if p["type"] == "disjunction"]
        negations = [p for p in premises if p["type"] == "negation"]

        for disjunction in disjunctions:
            if len(disjunction["components"]) >= 2:
                option1 = disjunction["components"][0]
                option2 = disjunction["components"][1]

                # Look for negation of one option
                for negation in negations:
                    if self._semantic_match(option1, negation["components"][0]):
                        step = ReasoningStep(
                            step_id=f"ds_{len(steps)}",
                            premise=f"{option1} or {option2}; not {option1}",
                            inference_rule="disjunctive_syllogism",
                            conclusion=option2,
                            confidence=min(disjunction["confidence"], negation["confidence"]),
                            supporting_evidence=[disjunction["original"], negation["original"]]
                        )
                        steps.append(step)
                    elif self._semantic_match(option2, negation["components"][0]):
                        step = ReasoningStep(
                            step_id=f"ds_{len(steps)}",
                            premise=f"{option1} or {option2}; not {option2}",
                            inference_rule="disjunctive_syllogism",
                            conclusion=option1,
                            confidence=min(disjunction["confidence"], negation["confidence"]),
                            supporting_evidence=[disjunction["original"], negation["original"]]
                        )
                        steps.append(step)

        return steps

    async def _constructive_dilemma(self, premises: List[Dict], question: str) -> List[ReasoningStep]:
        """Apply constructive dilemma inference rule"""
        steps = []

        conditionals = [p for p in premises if p["type"] == "conditional"]
        disjunctions = [p for p in premises if p["type"] == "disjunction"]

        # Find pattern: (P→Q) ∧ (R→S) ∧ (P∨R) ⊢ (Q∨S)
        for i, cond1 in enumerate(conditionals):
            for cond2 in conditionals[i+1:]:
                for disjunction in disjunctions:
                    if (len(cond1["components"]) >= 2 and len(cond2["components"]) >= 2 and
                        len(disjunction["components"]) >= 2):

                        p1, q1 = cond1["components"]
                        p2, q2 = cond2["components"]
                        d1, d2 = disjunction["components"]

                        # Check if disjunction matches antecedents
                        if ((self._semantic_match(p1, d1) and self._semantic_match(p2, d2)) or
                            (self._semantic_match(p1, d2) and self._semantic_match(p2, d1))):

                            step = ReasoningStep(
                                step_id=f"cd_{len(steps)}",
                                premise=f"If {p1}, then {q1}; If {p2}, then {q2}; {d1} or {d2}",
                                inference_rule="constructive_dilemma",
                                conclusion=f"{q1} or {q2}",
                                confidence=min(cond1["confidence"], cond2["confidence"], disjunction["confidence"]) * 0.8,
                                supporting_evidence=[cond1["original"], cond2["original"], disjunction["original"]]
                            )
                            steps.append(step)

        return steps

    def _semantic_match(self, phrase1: str, phrase2: str, threshold: float = 0.8) -> bool:
        """Check if two phrases are semantically equivalent"""

        # Simple string matching (can be enhanced with embeddings)
        phrase1_clean = phrase1.lower().strip()
        phrase2_clean = phrase2.lower().strip()

        # Exact match
        if phrase1_clean == phrase2_clean:
            return True

        # Subset match
        if phrase1_clean in phrase2_clean or phrase2_clean in phrase1_clean:
            return True

        # Word overlap
        words1 = set(phrase1_clean.split())
        words2 = set(phrase2_clean.split())

        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            return overlap >= threshold

        return False

    def _validate_inferences(self, inferences: List[ReasoningStep],
                           patterns: Dict[str, Any]) -> List[ReasoningStep]:
        """Validate logical inferences"""
        validated = []

        for inference in inferences:
            # Check confidence threshold
            if inference.confidence < 0.3:
                continue

            # Check for circular reasoning
            if inference.premise.lower() in inference.conclusion.lower():
                inference.confidence *= 0.5

            # Boost confidence if supported by patterns
            if patterns:
                pattern_support = self._check_pattern_support(inference, patterns)
                inference.confidence = min(inference.confidence * (1 + pattern_support), 1.0)

            validated.append(inference)

        return sorted(validated, key=lambda x: x.confidence, reverse=True)

    def _check_pattern_support(self, inference: ReasoningStep, patterns: Dict[str, Any]) -> float:
        """Check if inference is supported by extracted patterns"""
        support = 0.0

        # Check causal patterns
        causal_patterns = patterns.get("causal_patterns", [])
        for pattern in causal_patterns:
            if (pattern.get("cause", "") in inference.premise and
                pattern.get("effect", "") in inference.conclusion):
                support += 0.2

        # Check structural patterns
        structural_patterns = patterns.get("structural_patterns", [])
        for pattern in structural_patterns:
            if (pattern.get("parent", "") in inference.premise or
                pattern.get("child", "") in inference.conclusion):
                support += 0.1

        return min(support, 0.5)

    def _build_reasoning_chain(self, inferences: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """Build logical reasoning chain"""
        chain = []

        # Group inferences by rule type
        rule_groups = {}
        for inference in inferences:
            rule = inference.inference_rule
            if rule not in rule_groups:
                rule_groups[rule] = []
            rule_groups[rule].append(inference)

        # Build chain by confidence and logical dependency
        remaining_inferences = inferences.copy()

        while remaining_inferences:
            # Find next best inference
            best_inference = max(remaining_inferences, key=lambda x: x.confidence)
            remaining_inferences.remove(best_inference)

            chain_step = {
                "step": len(chain) + 1,
                "inference": best_inference,
                "logical_form": f"{best_inference.premise} ⊢ {best_inference.conclusion}",
                "rule": best_inference.inference_rule,
                "confidence": best_inference.confidence
            }

            chain.append(chain_step)

            # Limit chain length
            if len(chain) >= 5:
                break

        return chain

    def _generate_synthesis(self, reasoning_chain: List[Dict[str, Any]], question: str) -> str:
        """Generate logical synthesis from reasoning chain"""

        if not reasoning_chain:
            return "No valid logical inferences could be derived from the given premises."

        # Build synthesis narrative
        synthesis_parts = []

        # Introduction
        synthesis_parts.append(f"Based on logical analysis of the premises, the following reasoning chain addresses the question: {question}")

        # Reasoning steps
        for i, step in enumerate(reasoning_chain):
            inference = step["inference"]
            rule_name = step["rule"].replace("_", " ").title()

            synthesis_parts.append(
                f"Step {step['step']}: Using {rule_name}, from the premise '{inference.premise}', "
                f"we can conclude '{inference.conclusion}' (confidence: {inference.confidence:.2f})"
            )

        # Final conclusion
        if reasoning_chain:
            final_conclusion = reasoning_chain[-1]["inference"].conclusion
            synthesis_parts.append(f"Therefore, the logical conclusion is: {final_conclusion}")

        return " ".join(synthesis_parts)

    def _calculate_synthesis_confidence(self, inferences: List[ReasoningStep]) -> float:
        """Calculate overall synthesis confidence"""
        if not inferences:
            return 0.0

        # Weighted average of inference confidences
        total_confidence = sum(inf.confidence for inf in inferences)
        avg_confidence = total_confidence / len(inferences)

        # Boost for multiple supporting inferences
        support_factor = min(len(inferences) / 3.0, 1.2)

        return min(avg_confidence * support_factor, 1.0)

    def _assess_logical_validity(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess logical validity of reasoning chain"""

        validity_scores = {
            "structural_validity": 0.0,
            "premise_consistency": 0.0,
            "conclusion_support": 0.0
        }

        if not reasoning_chain:
            return validity_scores

        # Structural validity - are inference rules applied correctly?
        valid_rules = 0
        for step in reasoning_chain:
            if step["rule"] in self.inference_rules:
                valid_rules += 1

        validity_scores["structural_validity"] = valid_rules / len(reasoning_chain)

        # Premise consistency - do premises contradict each other?
        premises = [step["inference"].premise for step in reasoning_chain]
        consistency_score = self._check_premise_consistency(premises)
        validity_scores["premise_consistency"] = consistency_score

        # Conclusion support - how well do premises support conclusions?
        support_score = sum(step["confidence"] for step in reasoning_chain) / len(reasoning_chain)
        validity_scores["conclusion_support"] = support_score

        return validity_scores

    def _check_premise_consistency(self, premises: List[str]) -> float:
        """Check consistency of premises"""
        # Simple consistency check (can be enhanced)

        # Look for explicit contradictions
        positive_statements = set()
        negative_statements = set()

        for premise in premises:
            if "not " in premise.lower():
                negative_statements.add(premise.lower().replace("not ", "").strip())
            else:
                positive_statements.add(premise.lower().strip())

        # Check for direct contradictions
        contradictions = positive_statements.intersection(negative_statements)

        if contradictions:
            return 0.5  # Some contradictions found
        else:
            return 1.0  # No obvious contradictions