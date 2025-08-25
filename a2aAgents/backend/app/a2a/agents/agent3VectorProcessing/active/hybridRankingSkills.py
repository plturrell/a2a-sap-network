from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import logging
import json
import math
from collections import Counter, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.a2a.core.security_base import SecureA2AAgent
"""
Hybrid Ranking Skills for Agent 3 (Vector Processing)
Implements advanced ranking combining BM25, vector similarity, and PageRank
Following SAP naming conventions and best practices
"""

logger = logging.getLogger(__name__)


class HybridRankingSkills(SecureA2AAgent):
    """Advanced hybrid ranking for vector search results"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, hanaConnection=None):
        super().__init__()
        self.hanaConnection = hanaConnection
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.rankingWeights = {
            'vectorSimilarity': 0.4,
            'bm25Score': 0.3,
            'pageRankScore': 0.2,
            'contextualRelevance': 0.1
        }
        self.documentFrequencyCache = {}
        self.pageRankCache = {}

    async def hybridRankingSearch(self,
                                query: str,
                                queryVector: List[float],
                                candidateDocuments: List[Dict[str, Any]],
                                searchContext: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform hybrid ranking combining multiple relevance signals
        """
        try:
            if not candidateDocuments:
                # Return empty result with proper structure
                return {
                    'ranked_documents': [],
                    'ranking_scores': {},
                    'total_candidates': 0,
                    'processing_time': 0.0
                }

            logger.info(f"Hybrid ranking {len(candidateDocuments)} documents")

            # Parallel computation of different ranking components
            tasks = [
                self._computeVectorSimilarityScores(queryVector, candidateDocuments),
                self._computeBM25Scores(query, candidateDocuments),
                self._computePageRankScores(candidateDocuments),
                self._computeContextualRelevanceScores(candidateDocuments, searchContext)
            ]

            rankings = await asyncio.gather(*tasks)

            vectorScores = rankings[0]
            bm25Scores = rankings[1]
            pageRankScores = rankings[2]
            contextualScores = rankings[3]

            # Combine rankings using weighted scoring
            finalScores = await self._combineRankings(
                candidateDocuments,
                vectorScores,
                bm25Scores,
                pageRankScores,
                contextualScores,
                searchContext
            )

            # Sort by final score and add ranking metadata
            rankedResults = []
            for docId, score in sorted(finalScores.items(), key=lambda x: x[1], reverse=True):
                doc = next((d for d in candidateDocuments if d.get('docId') == docId), None)
                if doc:
                    enhancedDoc = doc.copy()
                    enhancedDoc.update({
                        'hybridScore': score,
                        'rankingComponents': {
                            'vectorSimilarity': vectorScores.get(docId, 0.0),
                            'bm25Score': bm25Scores.get(docId, 0.0),
                            'pageRankScore': pageRankScores.get(docId, 0.0),
                            'contextualRelevance': contextualScores.get(docId, 0.0)
                        },
                        'rankingExplanation': self._generateRankingExplanation(
                            docId, vectorScores, bm25Scores, pageRankScores, contextualScores
                        )
                    })
                    rankedResults.append(enhancedDoc)

            # Post-process for diversity and freshness
            diversifiedResults = await self._applyDiversification(rankedResults, searchContext)

            return diversifiedResults

        except Exception as e:
            logger.error(f"Hybrid ranking failed: {e}")
            # Fallback to vector similarity only
            return sorted(candidateDocuments,
                         key=lambda x: x.get('similarityScore', 0),
                         reverse=True)

    async def _computeVectorSimilarityScores(self,
                                           queryVector: List[float],
                                           documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute vector similarity scores (already available from initial search)
        """
        vectorScores = {}

        for doc in documents:
            docId = doc.get('docId')
            if docId:
                # Use existing similarity score or recompute if needed
                if 'similarityScore' in doc:
                    vectorScores[docId] = float(doc['similarityScore'])
                elif 'vector' in doc:
                    # Recompute similarity
                    docVector = doc['vector']
                    similarity = self._cosineSimilarity(queryVector, docVector)
                    vectorScores[docId] = similarity
                else:
                    vectorScores[docId] = 0.0

        return vectorScores

    async def _computeBM25Scores(self,
                               query: str,
                               documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute BM25 scores for text relevance
        """
        try:
            # BM25 parameters
            k1 = 1.2  # Term frequency saturation parameter
            b = 0.75  # Length normalization parameter

            # Tokenize query
            queryTerms = self._tokenizeQuery(query)

            if not queryTerms:
                return {doc.get('docId'): 0.0 for doc in documents}

            # Get corpus statistics
            corpusStats = await self._getCorpusStatistics(documents)
            totalDocs = corpusStats['totalDocuments']
            avgDocLength = corpusStats['avgDocLength']

            bm25Scores = {}

            for doc in documents:
                docId = doc.get('docId')
                content = doc.get('content', '') or doc.get('text', '')

                if not content:
                    bm25Scores[docId] = 0.0
                    continue

                # Tokenize document content
                docTerms = self._tokenizeDocument(content)
                docLength = len(docTerms)
                termFreqs = Counter(docTerms)

                score = 0.0

                for term in queryTerms:
                    if term in termFreqs:
                        tf = termFreqs[term]
                        df = await self._getDocumentFrequency(term, totalDocs)
                        idf = math.log((totalDocs - df + 0.5) / (df + 0.5))

                        # BM25 formula
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength))
                        score += idf * (numerator / denominator)

                bm25Scores[docId] = max(0.0, score)

            # Normalize BM25 scores to 0-1 range
            if bm25Scores:
                maxScore = max(bm25Scores.values())
                if maxScore > 0:
                    bm25Scores = {docId: score/maxScore for docId, score in bm25Scores.items()}

            return bm25Scores

        except Exception as e:
            logger.error(f"BM25 computation failed: {e}")
            return {doc.get('docId'): 0.0 for doc in documents}

    async def _computePageRankScores(self,
                                   documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute PageRank-style authority scores based on entity relationships
        """
        try:
            if not self.hanaConnection:
                # Return uniform scores if no HANA connection
                return {doc.get('docId'): 0.5 for doc in documents}

            docIds = [doc.get('docId') for doc in documents if doc.get('docId')]

            if not docIds:
                # Return empty scores with proper structure
                return {
                    'scores': {},
                    'total_documents': 0,
                    'algorithm': 'pagerank',
                    'iterations': 0
                }

            # Check cache first
            cachedScores = {}
            uncachedIds = []

            for docId in docIds:
                if docId in self.pageRankCache:
                    cachedScores[docId] = self.pageRankCache[docId]
                else:
                    uncachedIds.append(docId)

            # Compute PageRank for uncached documents
            if uncachedIds:
                newScores = await self._computeGraphBasedAuthority(uncachedIds)
                cachedScores.update(newScores)
                self.pageRankCache.update(newScores)

            return cachedScores

        except Exception as e:
            logger.error(f"PageRank computation failed: {e}")
            return {doc.get('docId'): 0.5 for doc in documents}

    async def _computeGraphBasedAuthority(self, docIds: List[str]) -> Dict[str, float]:
        """
        Compute authority scores using HANA Graph algorithms
        """
        try:
            # Query for document relationships
            relationshipQuery = """
            WITH DOCUMENT_GRAPH AS (
                SELECT
                    d1.DOC_ID as SOURCE_DOC,
                    d2.DOC_ID as TARGET_DOC,
                    COUNT(*) as RELATIONSHIP_STRENGTH
                FROM A2A_VECTORS d1
                JOIN A2A_GRAPH_EDGES e ON d1.ENTITY_ID = e.SOURCE_NODE_ID
                JOIN A2A_VECTORS d2 ON e.TARGET_NODE_ID = d2.ENTITY_ID
                WHERE d1.DOC_ID IN ({})
                   OR d2.DOC_ID IN ({})
                GROUP BY d1.DOC_ID, d2.DOC_ID
            ),
            PAGERANK_SCORES AS (
                -- Simplified PageRank calculation
                SELECT
                    DOC_ID,
                    -- Authority score based on incoming relationships
                    COALESCE(
                        (SELECT SUM(RELATIONSHIP_STRENGTH)
                         FROM DOCUMENT_GRAPH dg
                         WHERE dg.TARGET_DOC = v.DOC_ID), 0
                    ) as AUTHORITY_SCORE
                FROM A2A_VECTORS v
                WHERE v.DOC_ID IN ({})
            )
            SELECT
                DOC_ID,
                CASE
                    WHEN MAX(AUTHORITY_SCORE) OVER() > 0
                    THEN AUTHORITY_SCORE / MAX(AUTHORITY_SCORE) OVER()
                    ELSE 0.5
                END as NORMALIZED_AUTHORITY
            FROM PAGERANK_SCORES
            """.format(
                ','.join([f"'{docId}'" for docId in docIds]),
                ','.join([f"'{docId}'" for docId in docIds]),
                ','.join([f"'{docId}'" for docId in docIds])
            )

            results = await self.hanaConnection.execute(relationshipQuery)

            authorityScores = {}
            for result in results:
                authorityScores[result['DOC_ID']] = float(result['NORMALIZED_AUTHORITY'])

            # Fill in missing documents with default score
            for docId in docIds:
                if docId not in authorityScores:
                    authorityScores[docId] = 0.5

            return authorityScores

        except Exception as e:
            logger.error(f"Graph-based authority computation failed: {e}")
            return {docId: 0.5 for docId in docIds}

    async def _computeContextualRelevanceScores(self,
                                              documents: List[Dict[str, Any]],
                                              searchContext: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute contextual relevance based on search context
        """
        contextualScores = {}

        try:
            userContext = searchContext.get('userContext', {})
            temporalContext = searchContext.get('temporalContext', {})
            domainContext = searchContext.get('domainContext', {})

            for doc in documents:
                docId = doc.get('docId')
                if not docId:
                    continue

                score = 0.0

                # User preference alignment
                if 'userPreferences' in userContext:
                    userScore = self._calculateUserRelevance(doc, userContext['userPreferences'])
                    score += userScore * 0.4

                # Temporal relevance (fresher content scores higher)
                if 'preferRecent' in temporalContext and temporalContext['preferRecent']:
                    temporalScore = self._calculateTemporalRelevance(doc)
                    score += temporalScore * 0.3

                # Domain-specific relevance
                if 'domain' in domainContext:
                    domainScore = self._calculateDomainRelevance(doc, domainContext['domain'])
                    score += domainScore * 0.3

                # Normalize to 0-1 range
                contextualScores[docId] = min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Contextual relevance computation failed: {e}")
            return {doc.get('docId'): 0.5 for doc in documents}

        return contextualScores

    async def _combineRankings(self,
                             documents: List[Dict[str, Any]],
                             vectorScores: Dict[str, float],
                             bm25Scores: Dict[str, float],
                             pageRankScores: Dict[str, float],
                             contextualScores: Dict[str, float],
                             searchContext: Dict[str, Any]) -> Dict[str, float]:
        """
        Combine multiple ranking signals using weighted scoring
        """
        finalScores = {}

        # Adaptive weight adjustment based on search context
        adaptiveWeights = self._adjustWeightsForContext(searchContext)

        for doc in documents:
            docId = doc.get('docId')
            if not docId:
                continue

            # Get individual scores
            vectorScore = vectorScores.get(docId, 0.0)
            bm25Score = bm25Scores.get(docId, 0.0)
            pageRankScore = pageRankScores.get(docId, 0.0)
            contextualScore = contextualScores.get(docId, 0.0)

            # Weighted combination
            finalScore = (
                vectorScore * adaptiveWeights['vectorSimilarity'] +
                bm25Score * adaptiveWeights['bm25Score'] +
                pageRankScore * adaptiveWeights['pageRankScore'] +
                contextualScore * adaptiveWeights['contextualRelevance']
            )

            # Apply quality boost based on document metadata
            qualityBoost = self._calculateQualityBoost(doc)
            finalScore = finalScore * qualityBoost

            finalScores[docId] = finalScore

        return finalScores

    def _adjustWeightsForContext(self, searchContext: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjust ranking weights based on search context
        """
        weights = self.rankingWeights.copy()

        searchType = searchContext.get('searchType', 'general')

        if searchType == 'semantic':
            # Emphasize vector similarity for semantic search
            weights['vectorSimilarity'] = 0.6
            weights['bm25Score'] = 0.2
            weights['pageRankScore'] = 0.1
            weights['contextualRelevance'] = 0.1
        elif searchType == 'keyword':
            # Emphasize BM25 for keyword search
            weights['vectorSimilarity'] = 0.2
            weights['bm25Score'] = 0.5
            weights['pageRankScore'] = 0.2
            weights['contextualRelevance'] = 0.1
        elif searchType == 'authoritative':
            # Emphasize PageRank for authoritative results
            weights['vectorSimilarity'] = 0.3
            weights['bm25Score'] = 0.2
            weights['pageRankScore'] = 0.4
            weights['contextualRelevance'] = 0.1

        return weights

    async def _applyDiversification(self,
                                  rankedResults: List[Dict[str, Any]],
                                  searchContext: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply diversity constraints to avoid result clustering
        """
        try:
            diversityThreshold = searchContext.get('diversityThreshold', 0.85)
            maxResults = searchContext.get('maxResults', 10)

            if not rankedResults or len(rankedResults) <= 1:
                return rankedResults

            diversifiedResults = [rankedResults[0]]  # Always include top result

            for candidate in rankedResults[1:]:
                if len(diversifiedResults) >= maxResults:
                    break

                # Check diversity against already selected results
                isDiverse = True
                candidateVector = candidate.get('vector', [])

                if candidateVector:
                    for selected in diversifiedResults:
                        selectedVector = selected.get('vector', [])
                        if selectedVector:
                            similarity = self._cosineSimilarity(candidateVector, selectedVector)
                            if similarity > diversityThreshold:
                                isDiverse = False
                                break

                if isDiverse:
                    diversifiedResults.append(candidate)

            # Add position-based ranking information
            for i, result in enumerate(diversifiedResults):
                result['finalRank'] = i + 1
                result['diversityApplied'] = len(rankedResults) != len(diversifiedResults)

            return diversifiedResults

        except Exception as e:
            logger.error(f"Diversification failed: {e}")
            return rankedResults[:searchContext.get('maxResults', 10)]

    def _tokenizeQuery(self, query: str) -> List[str]:
        """Simple query tokenization"""
        import re
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        return [term.strip() for term in query.split() if term.strip()]

    def _tokenizeDocument(self, content: str) -> List[str]:
        """Simple document tokenization"""
        import re
        # Remove punctuation and convert to lowercase
        content = re.sub(r'[^\w\s]', ' ', content.lower())
        return [term.strip() for term in content.split() if term.strip()]

    async def _getCorpusStatistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get corpus-level statistics for BM25"""
        totalDocs = len(documents)
        totalLength = 0

        for doc in documents:
            content = doc.get('content', '') or doc.get('text', '')
            docTerms = self._tokenizeDocument(content)
            totalLength += len(docTerms)

        avgDocLength = totalLength / totalDocs if totalDocs > 0 else 0

        return {
            'totalDocuments': totalDocs,
            'avgDocLength': avgDocLength
        }

    async def _getDocumentFrequency(self, term: str, totalDocs: int) -> int:
        """Get document frequency for a term (with caching)"""
        if term in self.documentFrequencyCache:
            return self.documentFrequencyCache[term]

        # In a real implementation, this would query the corpus
        # For now, estimate based on term characteristics
        if len(term) <= 3:
            df = max(1, int(totalDocs * 0.1))  # Common short terms
        elif term in ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
            df = max(1, int(totalDocs * 0.8))  # Stop words
        else:
            df = max(1, int(totalDocs * 0.05))  # Regular terms

        self.documentFrequencyCache[term] = df
        return df

    def _calculateUserRelevance(self, doc: Dict[str, Any], userPreferences: Dict[str, Any]) -> float:
        """Calculate relevance based on user preferences"""
        score = 0.0

        # Domain preference
        docDomain = doc.get('domain') or doc.get('entityType', '')
        preferredDomains = userPreferences.get('domains', [])
        if docDomain in preferredDomains:
            score += 0.5

        # Content type preference
        docType = doc.get('contentType', '')
        preferredTypes = userPreferences.get('contentTypes', [])
        if docType in preferredTypes:
            score += 0.3

        # Source preference
        docSource = doc.get('source', '')
        preferredSources = userPreferences.get('sources', [])
        if docSource in preferredSources:
            score += 0.2

        return min(score, 1.0)

    def _calculateTemporalRelevance(self, doc: Dict[str, Any]) -> float:
        """Calculate temporal relevance (recency bias)"""
        try:
            createdAt = doc.get('createdAt') or doc.get('timestamp')
            if not createdAt:
                return 0.5

            if isinstance(createdAt, str):
                from dateutil import parser
                docDate = parser.parse(createdAt)
            else:
                docDate = createdAt

            # Calculate days since creation
            daysSinceCreation = (datetime.now() - docDate).days

            # Exponential decay with half-life of 30 days
            temporalScore = math.exp(-daysSinceCreation / 30.0)

            return min(temporalScore, 1.0)

        except Exception as e:
            logger.error(f"Temporal relevance calculation failed: {e}")
            return 0.5

    def _calculateDomainRelevance(self, doc: Dict[str, Any], targetDomain: str) -> float:
        """Calculate domain-specific relevance"""
        docDomain = doc.get('domain') or doc.get('entityType', '').lower()

        if docDomain == targetDomain.lower():
            return 1.0

        # Related domains mapping
        domainRelations = {
            'financial': ['accounting', 'investment', 'banking'],
            'legal': ['compliance', 'regulatory', 'contract'],
            'medical': ['healthcare', 'clinical', 'pharmaceutical'],
            'technical': ['engineering', 'software', 'technology']
        }

        targetRelated = domainRelations.get(targetDomain.lower(), [])
        if docDomain in targetRelated:
            return 0.8

        return 0.3  # Default relevance for unrelated domains

    def _calculateQualityBoost(self, doc: Dict[str, Any]) -> float:
        """Calculate quality boost multiplier"""
        boost = 1.0

        # AI readiness score boost
        aiReadinessScore = doc.get('aiReadinessScore', 0)
        if aiReadinessScore > 0.8:
            boost *= 1.1
        elif aiReadinessScore < 0.3:
            boost *= 0.9

        # Metadata completeness boost
        metadata = doc.get('metadata', {})
        if isinstance(metadata, dict) and len(metadata) > 3:
            boost *= 1.05

        return boost

    def _generateRankingExplanation(self,
                                  docId: str,
                                  vectorScores: Dict[str, float],
                                  bm25Scores: Dict[str, float],
                                  pageRankScores: Dict[str, float],
                                  contextualScores: Dict[str, float]) -> str:
        """Generate human-readable ranking explanation"""
        explanations = []

        vectorScore = vectorScores.get(docId, 0)
        if vectorScore > 0.8:
            explanations.append("high semantic similarity")
        elif vectorScore > 0.6:
            explanations.append("good semantic match")

        bm25Score = bm25Scores.get(docId, 0)
        if bm25Score > 0.7:
            explanations.append("strong keyword relevance")

        pageRankScore = pageRankScores.get(docId, 0)
        if pageRankScore > 0.7:
            explanations.append("high authority")

        contextualScore = contextualScores.get(docId, 0)
        if contextualScore > 0.7:
            explanations.append("contextually relevant")

        if not explanations:
            return "ranked by combined relevance signals"

        return f"Ranked by: {', '.join(explanations)}"

    def _cosineSimilarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except:
            return 0.0