"""
NLP Pattern Matcher
Real natural language processing for pattern matching using Grok-4
"""

import re
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class NLPPatternMatcher:
    """Advanced NLP pattern matching using Grok-4 and linguistic analysis"""

    def __init__(self, grok_client=None):
        self.grok_client = grok_client
        self.pattern_cache = {}
        self.linguistic_patterns = self._initialize_linguistic_patterns()

    def _initialize_linguistic_patterns(self) -> Dict[str, Any]:
        """Initialize linguistic pattern definitions"""
        return {
            "question_types": {
                "what": {"type": "definition", "expects": "explanation"},
                "how": {"type": "process", "expects": "steps"},
                "why": {"type": "causation", "expects": "reasons"},
                "when": {"type": "temporal", "expects": "time"},
                "where": {"type": "location", "expects": "place"},
                "who": {"type": "agent", "expects": "entity"},
                "which": {"type": "selection", "expects": "choice"},
                "can": {"type": "possibility", "expects": "yes/no + explanation"},
                "should": {"type": "recommendation", "expects": "advice"},
                "would": {"type": "hypothetical", "expects": "conditional"},
            },
            "semantic_patterns": {
                "comparison": ["versus", "vs", "compared to", "difference between", "similar to"],
                "causation": ["because", "therefore", "thus", "consequently", "as a result"],
                "condition": ["if", "when", "unless", "provided that", "assuming"],
                "sequence": ["first", "then", "next", "finally", "subsequently"],
                "example": ["for instance", "such as", "like", "including", "e.g."],
                "contrast": ["however", "but", "although", "despite", "whereas"],
                "emphasis": ["especially", "particularly", "notably", "importantly"],
            },
            "domain_indicators": {
                "technical": ["algorithm", "system", "protocol", "architecture", "implementation"],
                "scientific": ["hypothesis", "experiment", "theory", "evidence", "analysis"],
                "business": ["strategy", "market", "revenue", "customer", "growth"],
                "philosophical": ["ethics", "morality", "existence", "consciousness", "meaning"],
                "practical": ["how to", "steps", "guide", "tutorial", "process"],
            },
            "complexity_markers": {
                "simple": ["basic", "simple", "easy", "straightforward", "elementary"],
                "moderate": ["explain", "describe", "discuss", "analyze", "compare"],
                "complex": ["evaluate", "synthesize", "critique", "design", "optimize"],
                "expert": ["prove", "derive", "formulate", "construct", "demonstrate"],
            }
        }

    async def analyze_patterns(self, text: str, use_grok: bool = True) -> Dict[str, Any]:
        """Analyze patterns in text using NLP and optionally Grok-4"""
        # Local linguistic analysis
        local_patterns = self._analyze_linguistic_patterns(text)

        # Grok-4 enhanced analysis if available
        if use_grok and self.grok_client:
            try:
                grok_patterns = await self._analyze_with_grok(text)
                # Merge patterns
                return self._merge_pattern_analyses(local_patterns, grok_patterns)
            except Exception as e:
                logger.warning(f"Grok-4 pattern analysis failed, using local only: {e}")

        return local_patterns

    def _analyze_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns locally"""
        text_lower = text.lower()
        words = text_lower.split()

        # Detect question type
        question_type = None
        for word, info in self.linguistic_patterns["question_types"].items():
            if text_lower.startswith(word) or f" {word} " in text_lower:
                question_type = info
                break

        # Detect semantic patterns
        detected_patterns = []
        for pattern_type, indicators in self.linguistic_patterns["semantic_patterns"].items():
            for indicator in indicators:
                if indicator in text_lower:
                    detected_patterns.append({
                        "type": pattern_type,
                        "indicator": indicator,
                        "position": text_lower.find(indicator)
                    })

        # Detect domain
        domain_scores = {}
        for domain, keywords in self.linguistic_patterns["domain_indicators"].items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score

        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"

        # Detect complexity
        complexity = "moderate"  # default
        for level, markers in self.linguistic_patterns["complexity_markers"].items():
            if any(marker in text_lower for marker in markers):
                complexity = level
                break

        # Extract key entities (simple noun phrase detection)
        key_entities = self._extract_key_entities(text)

        # Analyze structure
        structure_analysis = {
            "sentence_count": len(re.split(r'[.!?]+', text.strip())),
            "word_count": len(words),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "has_subclauses": ',' in text or 'which' in text_lower or 'that' in text_lower,
            "is_compound": ' and ' in text_lower or ' or ' in text_lower,
        }

        return {
            "question_type": question_type,
            "semantic_patterns": detected_patterns,
            "domain": primary_domain,
            "domain_scores": domain_scores,
            "complexity": complexity,
            "key_entities": key_entities,
            "structure": structure_analysis,
            "linguistic_features": {
                "has_negation": any(neg in text_lower for neg in ['not', 'no', 'never', 'neither']),
                "has_quantifiers": any(q in text_lower for q in ['all', 'some', 'many', 'few', 'most']),
                "has_modals": any(m in text_lower for m in ['can', 'could', 'should', 'would', 'might']),
                "tense": self._detect_tense(text_lower),
            }
        }

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities using simple heuristics"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        words = text.lower().split()

        # Extract potential entities (capitalized words, multi-word phrases)
        entities = []

        # Find capitalized words (proper nouns)
        for word in text.split():
            if word[0].isupper() and word.lower() not in stop_words:
                entities.append(word)

        # Find noun phrases (simple pattern: adj + noun)
        word_pairs = zip(words[:-1], words[1:])
        for w1, w2 in word_pairs:
            if w1 not in stop_words and w2 not in stop_words:
                if w1.endswith('ing') or w1.endswith('ed') or w1.endswith('al'):
                    entities.append(f"{w1} {w2}")

        # Deduplicate and return top entities
        entity_counts = Counter(entities)
        return [entity for entity, _ in entity_counts.most_common(5)]

    def _detect_tense(self, text: str) -> str:
        """Detect the primary tense of the text"""
        past_indicators = ['was', 'were', 'had', 'did', 'went', 'came', 'saw']
        present_indicators = ['is', 'are', 'am', 'have', 'has', 'do', 'does']
        future_indicators = ['will', 'shall', 'going to', 'would', 'could']

        past_count = sum(1 for ind in past_indicators if ind in text)
        present_count = sum(1 for ind in present_indicators if ind in text)
        future_count = sum(1 for ind in future_indicators if ind in text)

        if future_count > past_count and future_count > present_count:
            return "future"
        elif past_count > present_count:
            return "past"
        else:
            return "present"

    async def _analyze_with_grok(self, text: str) -> Dict[str, Any]:
        """Use Grok-4 for advanced pattern analysis"""
        if not self.grok_client:
            return {}

        try:
            # Use Grok's pattern analysis
            result = await self.grok_client.analyze_patterns(text)

            if result.get('success'):
                return result.get('patterns', {})
            else:
                return {}

        except Exception as e:
            logger.error(f"Grok-4 pattern analysis error: {e}")
            return {}

    def _merge_pattern_analyses(
        self,
        local_patterns: Dict[str, Any],
        grok_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge local and Grok-4 pattern analyses"""
        merged = local_patterns.copy()

        # Enhance with Grok insights
        if "semantic_patterns" in grok_patterns:
            merged["grok_semantic_patterns"] = grok_patterns["semantic_patterns"]

        if "entities" in grok_patterns:
            # Merge entity lists
            local_entities = set(merged.get("key_entities", []))
            grok_entities = set(grok_patterns["entities"])
            merged["key_entities"] = list(local_entities | grok_entities)

        if "complexity_score" in grok_patterns:
            merged["grok_complexity"] = grok_patterns["complexity_score"]

        if "intent" in grok_patterns:
            merged["grok_intent"] = grok_patterns["intent"]

        # Add confidence based on agreement
        local_complexity = merged.get("complexity", "moderate")
        grok_complexity = grok_patterns.get("complexity", "moderate")

        if local_complexity == grok_complexity:
            merged["pattern_confidence"] = 0.9
        else:
            merged["pattern_confidence"] = 0.7

        merged["enhanced_with_grok"] = True

        return merged

    def match_patterns(self, text: str, pattern_list: List[str]) -> List[Dict[str, Any]]:
        """Match text against a list of patterns"""
        matches = []

        for pattern in pattern_list:
            # Try regex matching
            try:
                regex_matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in regex_matches:
                    matches.append({
                        "pattern": pattern,
                        "type": "regex",
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end()
                    })
            except re.error:
                # Not a valid regex, try substring matching
                if pattern.lower() in text.lower():
                    start = text.lower().find(pattern.lower())
                    matches.append({
                        "pattern": pattern,
                        "type": "substring",
                        "match": text[start:start+len(pattern)],
                        "start": start,
                        "end": start + len(pattern)
                    })

        return matches

    async def find_semantic_similarity(
        self,
        text1: str,
        text2: str,
        use_grok: bool = True
    ) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple approach: shared key entities and patterns
        patterns1 = await self.analyze_patterns(text1, use_grok)
        patterns2 = await self.analyze_patterns(text2, use_grok)

        # Compare key entities
        entities1 = set(patterns1.get("key_entities", []))
        entities2 = set(patterns2.get("key_entities", []))

        if not entities1 and not entities2:
            entity_similarity = 0.5  # No entities to compare
        elif not entities1 or not entities2:
            entity_similarity = 0.0
        else:
            entity_similarity = len(entities1 & entities2) / len(entities1 | entities2)

        # Compare domains
        domain_similarity = 1.0 if patterns1.get("domain") == patterns2.get("domain") else 0.5

        # Compare complexity
        complexity_similarity = 1.0 if patterns1.get("complexity") == patterns2.get("complexity") else 0.7

        # Compare question types
        type_similarity = 1.0
        if patterns1.get("question_type") and patterns2.get("question_type"):
            type_similarity = 1.0 if patterns1["question_type"]["type"] == patterns2["question_type"]["type"] else 0.5

        # Weighted average
        similarity = (
            entity_similarity * 0.4 +
            domain_similarity * 0.2 +
            complexity_similarity * 0.1 +
            type_similarity * 0.3
        )

        return min(1.0, max(0.0, similarity))


# Factory function
def create_nlp_pattern_matcher(grok_client=None) -> NLPPatternMatcher:
    """Create an NLP pattern matcher"""
    return NLPPatternMatcher(grok_client)