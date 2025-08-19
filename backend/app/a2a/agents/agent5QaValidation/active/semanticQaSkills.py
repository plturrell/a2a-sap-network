"""
Semantic QA Skills for Agent 5 (QA Testing) - SAP HANA Knowledge Engine Integration
Following SAP naming conventions and best practices
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import logging
import json
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionComplexity(str, Enum):
    SIMPLE = "simple"  # Direct fact retrieval
    MODERATE = "moderate"  # Single-hop reasoning
    COMPLEX = "complex"  # Multi-hop reasoning
    EXPERT = "expert"  # Domain expertise required


class SemanticQASkills:
    """Enhanced QA testing skills leveraging SAP HANA Knowledge Engine"""
    
    def __init__(self, hanaClient=None, vectorServiceUrl=None):
        self.hanaClient = hanaClient
        self.vectorServiceUrl = vectorServiceUrl
        self.knowledgeGraphCache = {}
        self.questionTemplates = self._loadQuestionTemplates()
        
    def _loadQuestionTemplates(self) -> Dict[str, List[Dict]]:
        """Load semantic question templates"""
        return {
            'factual': [
                {
                    'template': 'What is the {attribute} of {entity}?',
                    'complexity': QuestionComplexity.SIMPLE,
                    'answerSource': 'direct_attribute'
                },
                {
                    'template': 'Which {entity_type} has {attribute} equal to {value}?',
                    'complexity': QuestionComplexity.SIMPLE,
                    'answerSource': 'filtered_search'
                }
            ],
            'relationship': [
                {
                    'template': 'How is {entity1} related to {entity2}?',
                    'complexity': QuestionComplexity.MODERATE,
                    'answerSource': 'graph_traversal'
                },
                {
                    'template': 'What {entity_type} are connected to {entity} through {relationship}?',
                    'complexity': QuestionComplexity.MODERATE,
                    'answerSource': 'relationship_query'
                }
            ],
            'aggregation': [
                {
                    'template': 'What is the total {metric} for all {entity_type} in {category}?',
                    'complexity': QuestionComplexity.MODERATE,
                    'answerSource': 'aggregation_query'
                },
                {
                    'template': 'Which {entity_type} has the highest {metric} in {timeframe}?',
                    'complexity': QuestionComplexity.COMPLEX,
                    'answerSource': 'ranking_query'
                }
            ],
            'reasoning': [
                {
                    'template': 'Given that {condition}, what would be the impact on {entity}?',
                    'complexity': QuestionComplexity.COMPLEX,
                    'answerSource': 'inference_chain'
                },
                {
                    'template': 'Why does {entity1} have different {attribute} than {entity2}?',
                    'complexity': QuestionComplexity.EXPERT,
                    'answerSource': 'comparative_analysis'
                }
            ]
        }
    
    async def semanticQuestionGeneration(self, 
                                       productMetadata: Dict[str, Any],
                                       knowledgeContext: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate semantic questions from knowledge graph relationships
        """
        generatedQuestions = []
        
        try:
            # Extract entities and relationships from product metadata
            entities = await self._extractEntitiesFromMetadata(productMetadata)
            
            # Query knowledge graph for entity relationships
            for entity in entities:
                relationships = await self._queryEntityRelationships(entity)
                
                # Generate questions based on relationships
                for relationship in relationships:
                    questions = await self._generateQuestionsFromRelationship(
                        entity,
                        relationship,
                        productMetadata
                    )
                    generatedQuestions.extend(questions)
            
            # Generate multi-hop reasoning questions
            multiHopQuestions = await self._generateMultiHopQuestions(
                entities,
                knowledgeContext
            )
            generatedQuestions.extend(multiHopQuestions)
            
            # Add domain-specific questions
            domainQuestions = await self._generateDomainSpecificQuestions(
                productMetadata,
                knowledgeContext.get('domain', 'general')
            )
            generatedQuestions.extend(domainQuestions)
            
            # Score and rank questions by quality
            rankedQuestions = await self._rankQuestionsByQuality(generatedQuestions)
            
        except Exception as e:
            logger.error(f"Semantic question generation failed: {e}")
            rankedQuestions = []
            
        return rankedQuestions[:50]  # Return top 50 questions
    
    async def knowledgeBasedValidation(self, 
                                     answer: str,
                                     questionContext: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate answers against knowledge graph facts and semantic understanding
        """
        validationResult = {
            'isCorrect': False,
            'confidence': 0.0,
            'supportingFacts': [],
            'contradictions': [],
            'semanticScore': 0.0,
            'explanation': ''
        }
        
        try:
            # Retrieve ground truth from knowledge graph
            groundTruth = await self._retrieveGroundTruth(questionContext)
            
            if not groundTruth:
                validationResult['explanation'] = 'No ground truth found in knowledge base'
                return validationResult
            
            # Exact match validation
            if self._exactMatch(answer, groundTruth['answer']):
                validationResult['isCorrect'] = True
                validationResult['confidence'] = 1.0
                validationResult['supportingFacts'] = groundTruth['facts']
            else:
                # Semantic similarity validation
                semanticValidation = await self._validateSemanticSimilarity(
                    answer,
                    groundTruth['answer'],
                    questionContext
                )
                validationResult['semanticScore'] = semanticValidation['score']
                
                # Check if semantically equivalent
                if semanticValidation['score'] >= 0.85:
                    validationResult['isCorrect'] = True
                    validationResult['confidence'] = semanticValidation['score']
                    validationResult['explanation'] = 'Answer is semantically equivalent'
                else:
                    # Check for partial correctness
                    partialValidation = await self._validatePartialCorrectness(
                        answer,
                        groundTruth,
                        questionContext
                    )
                    
                    if partialValidation['isPartiallyCorrect']:
                        validationResult['confidence'] = partialValidation['score']
                        validationResult['explanation'] = partialValidation['explanation']
                        validationResult['supportingFacts'] = partialValidation['supportingFacts']
            
            # Check for contradictions with knowledge base
            contradictions = await self._checkContradictions(answer, questionContext)
            validationResult['contradictions'] = contradictions
            
            # Generate comprehensive explanation
            validationResult['explanation'] = await self._generateValidationExplanation(
                validationResult,
                groundTruth,
                questionContext
            )
            
        except Exception as e:
            logger.error(f"Knowledge-based validation failed: {e}")
            validationResult['explanation'] = f'Validation error: {str(e)}'
            
        return validationResult
    
    async def continuousLearning(self, 
                               validatedFacts: List[Dict[str, Any]],
                               testResults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update knowledge graph with validated facts and improve test effectiveness
        """
        learningResult = {
            'factsAdded': 0,
            'factsUpdated': 0,
            'embeddingsRetrained': False,
            'testEffectivenessMetrics': {},
            'improvements': []
        }
        
        try:
            # Update knowledge graph with validated facts
            for fact in validatedFacts:
                if fact['confidence'] >= 0.9:  # High confidence threshold
                    updateResult = await self._updateKnowledgeGraph(fact)
                    
                    if updateResult['isNew']:
                        learningResult['factsAdded'] += 1
                    else:
                        learningResult['factsUpdated'] += 1
            
            # Analyze test effectiveness
            effectiveness = await self._analyzeTestEffectiveness(testResults)
            learningResult['testEffectivenessMetrics'] = effectiveness
            
            # Identify areas for improvement
            if effectiveness['accuracy'] < 0.8:
                improvements = await self._identifyImprovements(
                    testResults,
                    effectiveness
                )
                learningResult['improvements'] = improvements
            
            # Trigger embedding retraining if needed
            if self._shouldRetrainEmbeddings(learningResult):
                retrainResult = await self._triggerEmbeddingRetraining(validatedFacts)
                learningResult['embeddingsRetrained'] = retrainResult['success']
            
            # Update test generation patterns
            await self._updateTestGenerationPatterns(
                testResults,
                effectiveness
            )
            
        except Exception as e:
            logger.error(f"Continuous learning failed: {e}")
            
        return learningResult
    
    async def _extractEntitiesFromMetadata(self, 
                                         metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from product metadata
        """
        entities = []
        
        try:
            # Direct entity extraction
            if metadata.get('entities'):
                entities.extend(metadata['entities'])
            
            # Extract from Dublin Core metadata
            if metadata.get('dublinCore'):
                dc = metadata['dublinCore']
                if dc.get('subject'):
                    entities.append({
                        'id': f"subject_{dc['subject']}",
                        'type': 'subject',
                        'value': dc['subject']
                    })
                if dc.get('creator'):
                    entities.append({
                        'id': f"creator_{dc['creator']}",
                        'type': 'creator',
                        'value': dc['creator']
                    })
            
            # Extract from technical metadata
            if metadata.get('technicalMetadata'):
                tech = metadata['technicalMetadata']
                for key, value in tech.items():
                    if isinstance(value, str) and len(value) > 3:
                        entities.append({
                            'id': f"tech_{key}_{value}",
                            'type': f'technical_{key}',
                            'value': value
                        })
                        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            
        return entities
    
    async def _queryEntityRelationships(self, 
                                      entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query knowledge graph for entity relationships
        """
        relationships = []
        
        try:
            if not self.hanaClient:
                return relationships
                
            relationshipQuery = """
            SELECT 
                e.EDGE_ID,
                e.SOURCE_NODE_ID,
                e.TARGET_NODE_ID,
                e.RELATIONSHIP_TYPE,
                e.PROPERTIES,
                e.CONFIDENCE_SCORE,
                n1.ENTITY_TYPE as SOURCE_TYPE,
                n2.ENTITY_TYPE as TARGET_TYPE,
                JSON_VALUE(n1.PROPERTIES, '$.name') as SOURCE_NAME,
                JSON_VALUE(n2.PROPERTIES, '$.name') as TARGET_NAME
            FROM A2A_GRAPH_EDGES e
            JOIN A2A_GRAPH_NODES n1 ON e.SOURCE_NODE_ID = n1.NODE_ID
            JOIN A2A_GRAPH_NODES n2 ON e.TARGET_NODE_ID = n2.NODE_ID
            WHERE n1.ENTITY_ID = :entityId
               OR n2.ENTITY_ID = :entityId
            ORDER BY e.CONFIDENCE_SCORE DESC
            LIMIT 20
            """
            
            results = await self.hanaClient.execute(relationshipQuery, {
                'entityId': entity['id']
            })
            
            for row in results:
                relationship = {
                    'edgeId': row['EDGE_ID'],
                    'sourceId': row['SOURCE_NODE_ID'],
                    'targetId': row['TARGET_NODE_ID'],
                    'type': row['RELATIONSHIP_TYPE'],
                    'properties': json.loads(row['PROPERTIES']) if row['PROPERTIES'] else {},
                    'confidence': row['CONFIDENCE_SCORE'],
                    'sourceType': row['SOURCE_TYPE'],
                    'targetType': row['TARGET_TYPE'],
                    'sourceName': row['SOURCE_NAME'],
                    'targetName': row['TARGET_NAME']
                }
                relationships.append(relationship)
                
        except Exception as e:
            logger.error(f"Failed to query entity relationships: {e}")
            
        return relationships
    
    async def _generateQuestionsFromRelationship(self, 
                                               entity: Dict[str, Any],
                                               relationship: Dict[str, Any],
                                               productMetadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate questions based on entity relationships
        """
        questions = []
        
        try:
            # Select appropriate templates based on relationship type
            templates = self._selectTemplatesForRelationship(relationship['type'])
            
            for template in templates:
                # Instantiate template with entity and relationship data
                question = self._instantiateTemplate(
                    template,
                    entity,
                    relationship,
                    productMetadata
                )
                
                if question:
                    # Generate answer from knowledge graph
                    answer = await self._generateAnswerFromKnowledge(
                        question,
                        entity,
                        relationship
                    )
                    
                    questions.append({
                        'testId': f"qa_{entity['id']}_{relationship['edgeId']}_{len(questions)}",
                        'question': question['text'],
                        'answer': answer['text'],
                        'complexity': question['complexity'],
                        'sourceProduct': productMetadata['ordId'],
                        'testType': 'relationship',
                        'groundTruthSource': answer['source'],
                        'confidence': answer['confidence'],
                        'metadata': {
                            'entityId': entity['id'],
                            'relationshipId': relationship['edgeId'],
                            'template': template['template']
                        }
                    })
                    
        except Exception as e:
            logger.error(f"Failed to generate questions from relationship: {e}")
            
        return questions
    
    async def _generateMultiHopQuestions(self, 
                                       entities: List[Dict[str, Any]],
                                       knowledgeContext: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multi-hop reasoning questions
        """
        multiHopQuestions = []
        
        try:
            if not self.hanaClient or len(entities) < 2:
                return multiHopQuestions
                
            # Find paths between entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    paths = await self._findPathsBetweenEntities(
                        entity1['id'],
                        entity2['id']
                    )
                    
                    for path in paths[:3]:  # Limit to 3 paths per pair
                        if len(path) >= 3:  # At least 2 hops
                            question = await self._generateMultiHopQuestion(
                                entity1,
                                entity2,
                                path,
                                knowledgeContext
                            )
                            
                            if question:
                                multiHopQuestions.append(question)
                                
        except Exception as e:
            logger.error(f"Failed to generate multi-hop questions: {e}")
            
        return multiHopQuestions
    
    async def _findPathsBetweenEntities(self, 
                                      entityId1: str,
                                      entityId2: str) -> List[List[Dict]]:
        """
        Find paths between two entities in the knowledge graph
        """
        paths = []
        
        try:
            if not self.hanaClient:
                return paths
                
            # Use HANA Graph shortest path algorithm
            pathQuery = """
            DO BEGIN
                DECLARE GRAPH g = GRAPH("FINANCIAL_KNOWLEDGE_GRAPH");
                
                -- Find paths between entities
                PATHS = SELECT * FROM GRAPH_SHORTEST_PATHS_ONE_TO_ONE(
                    :g,
                    START WHERE ENTITY_ID = :entity1,
                    END WHERE ENTITY_ID = :entity2,
                    PARAMETERS (
                        'maximumHops' = 4,
                        'maximumPaths' = 5
                    )
                );
                
                -- Return path details
                SELECT 
                    PATH_ID,
                    PATH_LENGTH,
                    NODES,
                    EDGES
                FROM PATHS
                ORDER BY PATH_LENGTH ASC;
            END;
            """
            
            pathResults = await self.hanaClient.execute(pathQuery, {
                'entity1': entityId1,
                'entity2': entityId2
            })
            
            for pathData in pathResults:
                path = self._parseGraphPath(pathData)
                if path:
                    paths.append(path)
                    
        except Exception as e:
            logger.error(f"Failed to find paths between entities: {e}")
            
        return paths
    
    async def _validateSemanticSimilarity(self, 
                                        answer: str,
                                        groundTruth: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate semantic similarity between answer and ground truth
        """
        validation = {
            'score': 0.0,
            'explanation': '',
            'similarityComponents': {}
        }
        
        try:
            # Generate embeddings for both answers
            if self.vectorServiceUrl:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.vectorServiceUrl}/generate_embeddings",
                        json={
                            'texts': [answer, groundTruth],
                            'model': 'sentence'
                        }
                    )
                    response.raise_for_status()
                    
                embeddings = response.json()['embeddings']
                
                # Calculate cosine similarity
                answerEmb = np.array(embeddings[0])
                truthEmb = np.array(embeddings[1])
                
                cosineSim = np.dot(answerEmb, truthEmb) / (
                    np.linalg.norm(answerEmb) * np.linalg.norm(truthEmb)
                )
                validation['score'] = float(cosineSim)
                
                # Additional similarity measures
                validation['similarityComponents'] = {
                    'cosine': float(cosineSim),
                    'tokenOverlap': self._calculateTokenOverlap(answer, groundTruth),
                    'editDistance': self._calculateNormalizedEditDistance(answer, groundTruth)
                }
                
                # Generate explanation
                if cosineSim >= 0.95:
                    validation['explanation'] = 'Nearly identical answers'
                elif cosineSim >= 0.85:
                    validation['explanation'] = 'Highly similar with minor variations'
                elif cosineSim >= 0.75:
                    validation['explanation'] = 'Moderately similar with some differences'
                else:
                    validation['explanation'] = 'Significantly different answers'
                    
        except Exception as e:
            logger.error(f"Semantic similarity validation failed: {e}")
            validation['explanation'] = f'Validation error: {str(e)}'
            
        return validation
    
    async def _updateKnowledgeGraph(self, 
                                  fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update knowledge graph with validated fact
        """
        updateResult = {
            'isNew': False,
            'nodeId': None,
            'edgeIds': []
        }
        
        try:
            if not self.hanaClient:
                return updateResult
                
            # Check if fact already exists
            existingFact = await self._checkFactExists(fact)
            
            if not existingFact:
                # Create new node for the fact
                nodeData = {
                    'nodeId': f"fact_{fact['id']}",
                    'entityId': fact['entityId'],
                    'entityType': 'validated_fact',
                    'properties': json.dumps({
                        'fact': fact['content'],
                        'confidence': fact['confidence'],
                        'source': fact['source'],
                        'validatedAt': datetime.utcnow().isoformat()
                    })
                }
                
                insertNodeQuery = """
                INSERT INTO A2A_GRAPH_NODES (
                    NODE_ID, ENTITY_ID, ENTITY_TYPE, PROPERTIES
                ) VALUES (
                    :nodeId, :entityId, :entityType, :properties
                )
                """
                
                await self.hanaClient.execute(insertNodeQuery, nodeData)
                updateResult['isNew'] = True
                updateResult['nodeId'] = nodeData['nodeId']
                
                # Create relationships to related entities
                for relatedEntity in fact.get('relatedEntities', []):
                    edgeData = {
                        'edgeId': f"validates_{fact['id']}_{relatedEntity}",
                        'sourceNodeId': nodeData['nodeId'],
                        'targetNodeId': f"node_{relatedEntity}",
                        'relationshipType': 'validates',
                        'properties': json.dumps({
                            'validationType': fact.get('validationType', 'qa_test'),
                            'confidence': fact['confidence']
                        }),
                        'confidenceScore': fact['confidence']
                    }
                    
                    insertEdgeQuery = """
                    INSERT INTO A2A_GRAPH_EDGES (
                        EDGE_ID, SOURCE_NODE_ID, TARGET_NODE_ID,
                        RELATIONSHIP_TYPE, PROPERTIES, CONFIDENCE_SCORE
                    ) VALUES (
                        :edgeId, :sourceNodeId, :targetNodeId,
                        :relationshipType, :properties, :confidenceScore
                    )
                    """
                    
                    await self.hanaClient.execute(insertEdgeQuery, edgeData)
                    updateResult['edgeIds'].append(edgeData['edgeId'])
            else:
                # Update existing fact confidence if higher
                if fact['confidence'] > existingFact['confidence']:
                    updateNodeQuery = """
                    UPDATE A2A_GRAPH_NODES
                    SET PROPERTIES = JSON_MERGE_PATCH(
                        PROPERTIES,
                        :updates
                    ),
                    UPDATED_AT = CURRENT_TIMESTAMP
                    WHERE NODE_ID = :nodeId
                    """
                    
                    await self.hanaClient.execute(updateNodeQuery, {
                        'nodeId': existingFact['nodeId'],
                        'updates': json.dumps({
                            'confidence': fact['confidence'],
                            'lastValidated': datetime.utcnow().isoformat()
                        })
                    })
                    
                updateResult['nodeId'] = existingFact['nodeId']
                
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
            
        return updateResult