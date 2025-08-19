"""
Enhanced Skills for Agent 2 (AI Preparation) - SAP HANA Knowledge Engine Integration
Following SAP naming conventions and best practices
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging
import json
from sentence_transformers import SentenceTransformer
import asyncio
from .domainSpecificEmbeddingSkills import DomainSpecificEmbeddingSkills

logger = logging.getLogger(__name__)


class EnhancedAIPreparationSkills:
    """Enhanced skills for AI Preparation Agent leveraging SAP HANA Knowledge Engine"""
    
    def __init__(self, hanaClient=None):
        self.hanaClient = hanaClient
        self.embeddingModels = {}
        self._initializeModels()
        # Initialize domain-specific embedding skills
        self.domainEmbeddingSkills = DomainSpecificEmbeddingSkills(hanaClient)
    
    def _initializeModels(self):
        """Initialize multiple embedding models for different purposes"""
        try:
            # Document-level embeddings (768-dim)
            self.embeddingModels['document'] = SentenceTransformer('all-mpnet-base-v2')
            
            # Sentence-level embeddings (384-dim, faster)
            self.embeddingModels['sentence'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Financial domain model (if available)
            try:
                self.embeddingModels['financial'] = SentenceTransformer('ProsusAI/finbert')
            except:
                logger.warning("Financial domain model not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
    
    async def advancedSemanticEnrichment(self, entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced semantic enrichment using HANA Text Analysis
        """
        enrichedData = {
            'originalData': entityData,
            'semanticLayers': {},
            'extractedEntities': {},
            'domainConcepts': {},
            'temporalPatterns': {}
        }
        
        try:
            # Extract text content from entity
            textContent = self._extractTextContent(entityData)
            
            if self.hanaClient:
                # Use HANA Text Analysis for entity extraction
                entities = await self._hanaTextAnalysis(textContent)
                enrichedData['extractedEntities'] = entities
                
                # Extract financial domain concepts
                domainConcepts = await self._extractDomainConcepts(textContent, entities)
                enrichedData['domainConcepts'] = domainConcepts
                
                # Identify temporal patterns
                temporalPatterns = await self._extractTemporalPatterns(entityData)
                enrichedData['temporalPatterns'] = temporalPatterns
            
            # Generate semantic layers
            enrichedData['semanticLayers'] = {
                'surface': self._extractSurfaceSemantics(textContent),
                'syntactic': self._extractSyntacticFeatures(textContent),
                'semantic': self._extractSemanticFeatures(textContent),
                'pragmatic': self._extractPragmaticContext(entityData)
            }
            
            # Calculate semantic richness score
            enrichedData['semanticRichnessScore'] = self._calculateSemanticRichness(enrichedData)
            
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            
        return enrichedData
    
    async def multiModelEmbeddingGeneration(self, 
                                          text: str, 
                                          context: Dict[str, Any],
                                          granularity: List[str] = ['document', 'paragraph', 'sentence']) -> Dict[str, Any]:
        """
        Generate embeddings at multiple granularities using different models
        """
        embeddings = {
            'metadata': {
                'generatedAt': datetime.utcnow().isoformat(),
                'modelsUsed': {},
                'dimensions': {}
            },
            'embeddings': {}
        }
        
        try:
            # Document-level embedding (768-dim)
            if 'document' in granularity and 'document' in self.embeddingModels:
                docEmbedding = self.embeddingModels['document'].encode(
                    text, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings['embeddings']['document'] = docEmbedding.tolist()
                embeddings['metadata']['modelsUsed']['document'] = 'all-mpnet-base-v2'
                embeddings['metadata']['dimensions']['document'] = len(docEmbedding)
            
            # Paragraph-level embeddings
            if 'paragraph' in granularity:
                paragraphs = text.split('\n\n')
                paraEmbeddings = []
                for para in paragraphs:
                    if para.strip():
                        paraEmb = self.embeddingModels['sentence'].encode(
                            para.strip(),
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        paraEmbeddings.append(paraEmb.tolist())
                embeddings['embeddings']['paragraphs'] = paraEmbeddings
                embeddings['metadata']['modelsUsed']['paragraph'] = 'all-MiniLM-L6-v2'
                
            # Sentence-level embeddings
            if 'sentence' in granularity:
                sentences = self._splitIntoSentences(text)
                sentEmbeddings = []
                for sent in sentences:
                    if sent.strip():
                        sentEmb = self.embeddingModels['sentence'].encode(
                            sent.strip(),
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        sentEmbeddings.append({
                            'text': sent.strip(),
                            'embedding': sentEmb.tolist()
                        })
                embeddings['embeddings']['sentences'] = sentEmbeddings
                embeddings['metadata']['modelsUsed']['sentence'] = 'all-MiniLM-L6-v2'
            
            # Financial domain-specific embeddings
            if 'financial' in self.embeddingModels and self._isFinancialContent(text):
                finEmbedding = self.embeddingModels['financial'].encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings['embeddings']['financialDomain'] = finEmbedding.tolist()
                embeddings['metadata']['modelsUsed']['financial'] = 'finbert'
                
        except Exception as e:
            logger.error(f"Multi-model embedding generation failed: {e}")
            
        return embeddings
    
    async def generateDomainOptimizedEmbeddings(self,
                                               text: str,
                                               entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings optimized for specific domains using specialized models
        """
        try:
            # Determine the domain based on entity data and content
            domain = await self._detectDomain(text, entityData)
            
            # Prepare context for domain-specific embedding
            context = {
                'entityType': entityData.get('entityType', 'unknown'),
                'contentType': entityData.get('contentType', 'general'),
                'language': entityData.get('language', 'en'),
                'requireCrossDomain': entityData.get('requireCrossDomain', False),
                'metadata': entityData.get('metadata', {})
            }
            
            # Generate domain-specific embeddings
            domainEmbeddings = await self.domainEmbeddingSkills.generateDomainSpecificEmbeddings(
                text,
                domain,
                context
            )
            
            # If successful, enhance with additional features
            if domainEmbeddings.get('status') == 'success':
                # Add quality evaluation
                if 'embeddings' in domainEmbeddings:
                    qualityEvaluation = await self.domainEmbeddingSkills.evaluateEmbeddingQuality(
                        domainEmbeddings['embeddings'],
                        []  # Reference data would come from historical embeddings
                    )
                    domainEmbeddings['qualityEvaluation'] = qualityEvaluation
                
                # Store the domain-optimized embeddings
                storageResult = await self._storeDomainOptimizedEmbeddings(
                    domainEmbeddings,
                    entityData
                )
                domainEmbeddings['storageResult'] = storageResult
            
            return domainEmbeddings
            
        except Exception as e:
            logger.error(f"Domain-optimized embedding generation failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'fallback': await self.multiModelEmbeddingGeneration(text, entityData)
            }
    
    async def _detectDomain(self, text: str, entityData: Dict[str, Any]) -> str:
        """
        Detect the domain of the content using heuristics and metadata
        """
        # Check entity type first
        entityType = entityData.get('entityType', '').lower()
        
        domainMapping = {
            'financial_report': 'financial',
            'earnings_call': 'financial',
            'contract': 'legal',
            'patent': 'legal',
            'medical_record': 'medical',
            'clinical_trial': 'medical',
            'source_code': 'technical',
            'technical_doc': 'technical',
            'manufacturing_spec': 'manufacturing',
            'quality_report': 'manufacturing'
        }
        
        # Check if entity type maps to a domain
        for key, domain in domainMapping.items():
            if key in entityType:
                return domain
        
        # Analyze text content for domain indicators
        textLower = text.lower()
        
        # Financial indicators
        if any(term in textLower for term in ['revenue', 'earnings', 'financial', 'profit', 'investment']):
            return 'financial'
        
        # Legal indicators
        elif any(term in textLower for term in ['agreement', 'contract', 'legal', 'party', 'clause']):
            return 'legal'
        
        # Medical indicators
        elif any(term in textLower for term in ['patient', 'diagnosis', 'treatment', 'medical', 'clinical']):
            return 'medical'
        
        # Technical indicators
        elif any(term in textLower for term in ['function', 'class', 'api', 'code', 'algorithm']):
            return 'technical'
        
        # Manufacturing indicators
        elif any(term in textLower for term in ['production', 'quality', 'manufacturing', 'assembly', 'specification']):
            return 'manufacturing'
        
        # Check language for multilingual
        if entityData.get('language') and entityData['language'] != 'en':
            return 'multilingual'
        
        return 'general'
    
    async def _storeDomainOptimizedEmbeddings(self,
                                            embeddings: Dict[str, Any],
                                            entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store domain-optimized embeddings with enhanced metadata
        """
        try:
            if not self.hanaClient:
                return {'status': 'skipped', 'reason': 'No HANA client available'}
            
            # Prepare storage data
            storageData = {
                'entityId': entityData['entityId'],
                'entityType': entityData['entityType'],
                'domain': embeddings.get('domain', 'general'),
                'embeddings': json.dumps(embeddings.get('embeddings', {})),
                'modelInfo': json.dumps(embeddings.get('modelUsed', {})),
                'qualityScore': embeddings.get('qualityEvaluation', {}).get('overallQuality', 0.0),
                'metadata': json.dumps({
                    'dimensions': embeddings.get('dimensions', {}),
                    'timestamp': datetime.now().isoformat(),
                    'entityMetadata': entityData.get('metadata', {})
                })
            }
            
            # Store in HANA
            insertQuery = """
            INSERT INTO A2A_DOMAIN_EMBEDDINGS (
                ENTITY_ID, ENTITY_TYPE, DOMAIN,
                EMBEDDINGS, MODEL_INFO, QUALITY_SCORE,
                METADATA, CREATED_AT
            ) VALUES (
                :entityId, :entityType, :domain,
                :embeddings, :modelInfo, :qualityScore,
                :metadata, CURRENT_TIMESTAMP
            )
            """
            
            await self.hanaClient.execute(insertQuery, storageData)
            
            return {
                'status': 'success',
                'stored': True,
                'domain': storageData['domain'],
                'qualityScore': storageData['qualityScore']
            }
            
        except Exception as e:
            logger.error(f"Failed to store domain-optimized embeddings: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def contextualRelationshipDiscovery(self, 
                                            entityData: Dict[str, Any],
                                            knowledgeGraphContext: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Discover implicit relationships using HANA Graph capabilities
        """
        relationships = {
            'explicitRelationships': [],
            'implicitRelationships': [],
            'temporalRelationships': [],
            'hierarchicalRelationships': [],
            'semanticRelationships': []
        }
        
        try:
            # Extract explicit relationships from entity data
            explicitRels = self._extractExplicitRelationships(entityData)
            relationships['explicitRelationships'] = explicitRels
            
            if self.hanaClient:
                # Query HANA Graph for implicit relationships
                entityId = entityData.get('entityId')
                
                # Find co-occurring entities
                coOccurrenceQuery = """
                SELECT 
                    e2.ENTITY_ID,
                    e2.ENTITY_TYPE,
                    COUNT(*) as CO_OCCURRENCE_COUNT,
                    AVG(DAYS_BETWEEN(e1.CREATED_AT, e2.CREATED_AT)) as TEMPORAL_DISTANCE
                FROM ENTITIES e1
                JOIN ENTITY_RELATIONSHIPS er ON e1.ENTITY_ID = er.SOURCE_ID
                JOIN ENTITIES e2 ON er.TARGET_ID = e2.ENTITY_ID
                WHERE e1.ENTITY_ID = :entityId
                GROUP BY e2.ENTITY_ID, e2.ENTITY_TYPE
                HAVING CO_OCCURRENCE_COUNT > :minCount
                """
                
                implicitRels = await self.hanaClient.execute(coOccurrenceQuery, {
                    'entityId': entityId,
                    'minCount': 3
                })
                relationships['implicitRelationships'] = implicitRels
                
                # Discover temporal relationships
                temporalRels = await self._discoverTemporalRelationships(entityData)
                relationships['temporalRelationships'] = temporalRels
                
                # Find hierarchical relationships using graph algorithms
                hierarchyQuery = """
                CALL GRAPH_SHORTEST_PATH(
                    GRAPH WORKSPACE "FINANCIAL_KNOWLEDGE_GRAPH",
                    SOURCE :entityId,
                    TARGET '*',
                    PARAMETERS ('maxDistance' = 3)
                )
                """
                hierarchicalRels = await self.hanaClient.execute(hierarchyQuery, {
                    'entityId': entityId
                })
                relationships['hierarchicalRelationships'] = hierarchicalRels
            
            # Discover semantic relationships using embeddings
            if knowledgeGraphContext:
                semanticRels = await self._discoverSemanticRelationships(
                    entityData, 
                    knowledgeGraphContext
                )
                relationships['semanticRelationships'] = semanticRels
                
        except Exception as e:
            logger.error(f"Relationship discovery failed: {e}")
            
        return relationships
    
    async def _hanaTextAnalysis(self, text: str) -> Dict[str, List[Dict]]:
        """Use HANA Text Analysis for entity extraction"""
        if not self.hanaClient:
            return {}
            
        try:
            # Create temporary table for text analysis
            await self.hanaClient.execute("""
                CREATE LOCAL TEMPORARY TABLE #TEXT_ANALYSIS_TEMP (
                    DOC_ID INT,
                    TEXT_CONTENT NCLOB
                )
            """)
            
            # Insert text for analysis
            await self.hanaClient.execute("""
                INSERT INTO #TEXT_ANALYSIS_TEMP VALUES (1, :text)
            """, {'text': text})
            
            # Run text analysis
            await self.hanaClient.execute("""
                CREATE FULLTEXT INDEX TEMP_IDX ON #TEXT_ANALYSIS_TEMP(TEXT_CONTENT)
                TEXT ANALYSIS ON
                CONFIGURATION 'EXTRACTION_CORE_VOICEOFCUSTOMER'
            """)
            
            # Extract entities
            entities = await self.hanaClient.execute("""
                SELECT 
                    TA_TOKEN as ENTITY,
                    TA_TYPE as ENTITY_TYPE,
                    TA_NORMALIZED as NORMALIZED_FORM,
                    TA_STEM as STEM_FORM,
                    TA_PARAGRAPH as PARAGRAPH_NUM,
                    TA_SENTENCE as SENTENCE_NUM,
                    TA_OFFSET as POSITION
                FROM "$TA_TEMP_IDX"
                WHERE TA_TYPE IN ('PERSON', 'ORGANIZATION', 'LOCATION', 'MONEY', 'DATE', 'PRODUCT')
                ORDER BY TA_OFFSET
            """)
            
            # Group entities by type
            groupedEntities = {}
            for entity in entities:
                entityType = entity['ENTITY_TYPE']
                if entityType not in groupedEntities:
                    groupedEntities[entityType] = []
                groupedEntities[entityType].append(entity)
                
            return groupedEntities
            
        except Exception as e:
            logger.error(f"HANA text analysis failed: {e}")
            return {}
            
    async def _extractDomainConcepts(self, text: str, entities: Dict) -> Dict[str, Any]:
        """Extract financial domain-specific concepts"""
        domainConcepts = {
            'financialInstruments': [],
            'riskIndicators': [],
            'complianceTerms': [],
            'marketIndicators': [],
            'businessMetrics': []
        }
        
        # Financial instrument patterns
        instrumentKeywords = ['bond', 'equity', 'derivative', 'option', 'future', 'swap', 'security']
        riskKeywords = ['var', 'volatility', 'exposure', 'credit risk', 'market risk', 'operational risk']
        complianceKeywords = ['basel', 'mifid', 'gdpr', 'sox', 'aml', 'kyc', 'regulatory']
        
        textLower = text.lower()
        
        # Extract based on keywords and context
        for keyword in instrumentKeywords:
            if keyword in textLower:
                context = self._extractKeywordContext(text, keyword, window=50)
                domainConcepts['financialInstruments'].append({
                    'concept': keyword,
                    'context': context,
                    'confidence': 0.85
                })
                
        # Similar extraction for other categories...
        
        return domainConcepts
    
    async def _extractTemporalPatterns(self, entityData: Dict) -> Dict[str, Any]:
        """Extract temporal patterns from entity data"""
        temporalPatterns = {
            'periodicity': None,
            'trends': [],
            'seasonality': {},
            'anomalies': []
        }
        
        # Extract time-series data if available
        if 'timeSeries' in entityData:
            tsData = entityData['timeSeries']
            
            # Detect periodicity
            periodicity = self._detectPeriodicity(tsData)
            temporalPatterns['periodicity'] = periodicity
            
            # Identify trends
            trends = self._identifyTrends(tsData)
            temporalPatterns['trends'] = trends
            
            # Detect seasonality
            seasonality = self._detectSeasonality(tsData)
            temporalPatterns['seasonality'] = seasonality
            
        return temporalPatterns
    
    def _extractTextContent(self, entityData: Dict) -> str:
        """Extract text content from entity data"""
        textParts = []
        
        # Common text fields
        textFields = ['description', 'content', 'text', 'summary', 'notes']
        for field in textFields:
            if field in entityData and entityData[field]:
                textParts.append(str(entityData[field]))
                
        # Concatenate all text parts
        return ' '.join(textParts)
    
    def _splitIntoSentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be enhanced with NLTK or spaCy)
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _isFinancialContent(self, text: str) -> bool:
        """Check if text contains financial content"""
        financialIndicators = [
            'financial', 'investment', 'portfolio', 'risk', 'return',
            'asset', 'liability', 'equity', 'revenue', 'profit'
        ]
        textLower = text.lower()
        return any(indicator in textLower for indicator in financialIndicators)
    
    def _calculateSemanticRichness(self, enrichedData: Dict) -> float:
        """Calculate semantic richness score"""
        score = 0.0
        weights = {
            'extractedEntities': 0.3,
            'domainConcepts': 0.3,
            'semanticLayers': 0.2,
            'temporalPatterns': 0.2
        }
        
        # Score based on entity extraction
        if enrichedData.get('extractedEntities'):
            entityScore = min(len(enrichedData['extractedEntities']) / 10, 1.0)
            score += weights['extractedEntities'] * entityScore
            
        # Score based on domain concepts
        if enrichedData.get('domainConcepts'):
            conceptCount = sum(len(v) for v in enrichedData['domainConcepts'].values())
            conceptScore = min(conceptCount / 20, 1.0)
            score += weights['domainConcepts'] * conceptScore
            
        # Score based on semantic layers
        if enrichedData.get('semanticLayers'):
            layerScore = len(enrichedData['semanticLayers']) / 4
            score += weights['semanticLayers'] * layerScore
            
        # Score based on temporal patterns
        if enrichedData.get('temporalPatterns'):
            temporalScore = 0.5 if enrichedData['temporalPatterns'].get('periodicity') else 0.0
            score += weights['temporalPatterns'] * temporalScore
            
        return round(score, 3)
    
    async def generateAdaptiveEmbeddings(self,
                                       text: str,
                                       entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate adaptive embeddings that automatically select the best domain model
        """
        adaptiveResult = {
            'primaryEmbedding': None,
            'domainEmbeddings': {},
            'selectedDomain': None,
            'embeddingStrategy': None,
            'metadata': {}
        }
        
        try:
            # Analyze text to determine relevant domains
            relevantDomains = await self._determinRelevantDomains(text, entityData)
            
            # Generate embeddings for relevant domains
            if relevantDomains:
                domainResults = await self.domainEmbeddingSkills.generateMultiDomainEmbeddings(
                    text, relevantDomains, entityData
                )
                
                adaptiveResult['domainEmbeddings'] = domainResults.get('embeddings', {})
                adaptiveResult['selectedDomain'] = domainResults.get('recommendedDomain')
                
                # Use the best domain embedding as primary
                if adaptiveResult['selectedDomain'] and adaptiveResult['selectedDomain'] in domainResults['embeddings']:
                    selectedEmbedding = domainResults['embeddings'][adaptiveResult['selectedDomain']]
                    adaptiveResult['primaryEmbedding'] = selectedEmbedding.get('embedding')
                    adaptiveResult['embeddingStrategy'] = 'domain_specific'
                elif domainResults.get('fusedEmbedding'):
                    adaptiveResult['primaryEmbedding'] = domainResults['fusedEmbedding']
                    adaptiveResult['embeddingStrategy'] = 'multi_domain_fusion'
            
            # Fallback to general embedding if no domain match
            if not adaptiveResult['primaryEmbedding']:
                generalEmbedding = await self.multiModelEmbeddingGeneration(
                    text, entityData, ['document']
                )
                adaptiveResult['primaryEmbedding'] = generalEmbedding['embeddings'].get('document')
                adaptiveResult['embeddingStrategy'] = 'general'
                adaptiveResult['selectedDomain'] = 'general'
            
            # Add metadata
            adaptiveResult['metadata'] = {
                'textLength': len(text),
                'identifiedDomains': relevantDomains,
                'generatedAt': datetime.utcnow().isoformat(),
                'embeddingDimensions': len(adaptiveResult['primaryEmbedding']) if adaptiveResult['primaryEmbedding'] else 0
            }
            
        except Exception as e:
            logger.error(f"Adaptive embedding generation failed: {e}")
            adaptiveResult['error'] = str(e)
            
        return adaptiveResult
    
    async def enhancedDocumentProcessing(self,
                                       document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process documents with domain-aware chunking and embedding
        """
        processingResult = {
            'documentId': document.get('id'),
            'chunks': [],
            'documentEmbedding': None,
            'domainAnalysis': {},
            'processingMetrics': {}
        }
        
        try:
            startTime = datetime.utcnow()
            
            # Extract and analyze document content
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            # Determine document domain
            domains = await self._determinRelevantDomains(content, metadata)
            primaryDomain = domains[0] if domains else 'general'
            
            # Apply domain-specific chunking strategy
            chunks = await self._domainAwareChunking(content, primaryDomain)
            
            # Process each chunk
            for idx, chunk in enumerate(chunks):
                # Generate adaptive embeddings for chunk
                chunkEmbedding = await self.generateAdaptiveEmbeddings(chunk['text'], {
                    'chunkIndex': idx,
                    'documentDomain': primaryDomain,
                    **metadata
                })
                
                # Enhance chunk with semantic information
                enhancedChunk = {
                    'chunkId': f"{document.get('id')}_{idx}",
                    'text': chunk['text'],
                    'embedding': chunkEmbedding['primaryEmbedding'],
                    'domain': chunkEmbedding['selectedDomain'],
                    'position': chunk.get('position', {}),
                    'semanticTags': await self._generateSemanticTags(chunk['text'], primaryDomain),
                    'importance': chunk.get('importance', 1.0)
                }
                
                processingResult['chunks'].append(enhancedChunk)
            
            # Generate document-level embedding
            docEmbeddingResult = await self.domainEmbeddingSkills.generateDomainEmbedding(
                content, primaryDomain, metadata
            )
            processingResult['documentEmbedding'] = docEmbeddingResult.get('embedding')
            
            # Domain analysis
            processingResult['domainAnalysis'] = {
                'primaryDomain': primaryDomain,
                'identifiedDomains': domains,
                'domainInsights': docEmbeddingResult.get('domainInsights', {})
            }
            
            # Processing metrics
            processingTime = (datetime.utcnow() - startTime).total_seconds()
            processingResult['processingMetrics'] = {
                'processingTime': processingTime,
                'chunkCount': len(chunks),
                'averageChunkSize': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0,
                'embeddingDimensions': len(processingResult['documentEmbedding']) if processingResult['documentEmbedding'] else 0
            }
            
            # Store processing results in HANA
            if self.hanaClient:
                await self._storeProcessingResults(processingResult)
                
        except Exception as e:
            logger.error(f"Enhanced document processing failed: {e}")
            processingResult['error'] = str(e)
            
        return processingResult
    
    async def _determinRelevantDomains(self, 
                                     text: str, 
                                     metadata: Dict[str, Any]) -> List[str]:
        """
        Determine relevant domains for the given text
        """
        relevantDomains = []
        
        # Domain indicators
        domainIndicators = {
            'financial': ['investment', 'portfolio', 'trading', 'market', 'finance', 'equity', 'bond'],
            'legal': ['contract', 'agreement', 'law', 'legal', 'compliance', 'regulation'],
            'medical': ['patient', 'diagnosis', 'treatment', 'medical', 'health', 'clinical'],
            'technical': ['code', 'software', 'api', 'system', 'algorithm', 'database'],
            'scientific': ['research', 'study', 'hypothesis', 'experiment', 'analysis', 'data']
        }
        
        textLower = text.lower()
        
        # Check for domain indicators
        for domain, indicators in domainIndicators.items():
            score = sum(1 for indicator in indicators if indicator in textLower)
            if score >= 2:  # At least 2 indicators present
                relevantDomains.append(domain)
        
        # Check metadata for domain hints
        if metadata.get('domain'):
            domain = metadata['domain'].lower()
            if domain in domainIndicators and domain not in relevantDomains:
                relevantDomains.append(domain)
        
        # Always include multilingual if non-English characters detected
        if any(ord(char) > 127 for char in text):
            relevantDomains.append('multilingual')
            
        return relevantDomains[:3]  # Limit to top 3 domains
    
    async def _domainAwareChunking(self, 
                                 content: str, 
                                 domain: str) -> List[Dict[str, Any]]:
        """
        Apply domain-specific chunking strategies
        """
        chunks = []
        
        if domain == 'legal':
            # Legal documents: chunk by sections/clauses
            chunks = self._chunkLegalDocument(content)
        elif domain == 'financial':
            # Financial documents: chunk by statements/periods
            chunks = self._chunkFinancialDocument(content)
        elif domain == 'technical':
            # Technical documents: chunk by code blocks/functions
            chunks = self._chunkTechnicalDocument(content)
        else:
            # Default: semantic chunking
            chunks = self._semanticChunking(content)
            
        return chunks
    
    def _chunkLegalDocument(self, content: str) -> List[Dict[str, Any]]:
        """Chunk legal documents by sections and clauses"""
        import re
        chunks = []
        
        # Split by section headers
        sectionPattern = r'(?:^|\n)(?:Section|Article|Clause)\s+\d+[.\s]'
        sections = re.split(sectionPattern, content)
        
        for idx, section in enumerate(sections):
            if section.strip():
                chunks.append({
                    'text': section.strip(),
                    'position': {'section': idx},
                    'importance': 1.0 if idx == 0 else 0.8  # First section often most important
                })
                
        return chunks if chunks else [{'text': content, 'position': {'section': 0}, 'importance': 1.0}]
    
    def _chunkFinancialDocument(self, content: str) -> List[Dict[str, Any]]:
        """Chunk financial documents by logical sections"""
        chunks = []
        
        # Common financial document sections
        sectionKeywords = ['executive summary', 'financial highlights', 'revenue', 
                          'expenses', 'balance sheet', 'cash flow', 'notes']
        
        currentChunk = []
        currentSection = None
        
        for line in content.split('\n'):
            lineLower = line.lower()
            
            # Check if line starts a new section
            newSection = False
            for keyword in sectionKeywords:
                if keyword in lineLower:
                    newSection = True
                    if currentChunk:
                        chunks.append({
                            'text': '\n'.join(currentChunk),
                            'position': {'section': currentSection},
                            'importance': 1.0 if currentSection in ['executive summary', 'financial highlights'] else 0.8
                        })
                    currentChunk = [line]
                    currentSection = keyword
                    break
                    
            if not newSection:
                currentChunk.append(line)
                
        # Add last chunk
        if currentChunk:
            chunks.append({
                'text': '\n'.join(currentChunk),
                'position': {'section': currentSection},
                'importance': 0.8
            })
            
        return chunks if chunks else [{'text': content, 'position': {'section': 'full'}, 'importance': 1.0}]
    
    def _chunkTechnicalDocument(self, content: str) -> List[Dict[str, Any]]:
        """Chunk technical documents by code blocks and sections"""
        chunks = []
        
        # Split by code blocks
        import re
        codeBlockPattern = r'```[\s\S]*?```'
        parts = re.split(codeBlockPattern, content)
        codeBlocks = re.findall(codeBlockPattern, content)
        
        # Interleave text and code blocks
        for i, part in enumerate(parts):
            if part.strip():
                chunks.append({
                    'text': part.strip(),
                    'position': {'index': i * 2},
                    'importance': 0.8
                })
            if i < len(codeBlocks):
                chunks.append({
                    'text': codeBlocks[i],
                    'position': {'index': i * 2 + 1},
                    'importance': 1.0  # Code blocks are usually important
                })
                
        return chunks if chunks else [{'text': content, 'position': {'index': 0}, 'importance': 1.0}]
    
    def _semanticChunking(self, content: str) -> List[Dict[str, Any]]:
        """Default semantic chunking based on paragraphs and length"""
        chunks = []
        maxChunkSize = 1000
        
        paragraphs = content.split('\n\n')
        currentChunk = []
        currentSize = 0
        
        for para in paragraphs:
            paraSize = len(para)
            
            if currentSize + paraSize > maxChunkSize and currentChunk:
                chunks.append({
                    'text': '\n\n'.join(currentChunk),
                    'position': {'start': len(chunks)},
                    'importance': 1.0 if len(chunks) == 0 else 0.8
                })
                currentChunk = [para]
                currentSize = paraSize
            else:
                currentChunk.append(para)
                currentSize += paraSize
                
        # Add last chunk
        if currentChunk:
            chunks.append({
                'text': '\n\n'.join(currentChunk),
                'position': {'start': len(chunks)},
                'importance': 0.8
            })
            
        return chunks if chunks else [{'text': content, 'position': {'start': 0}, 'importance': 1.0}]
    
    async def _generateSemanticTags(self, text: str, domain: str) -> List[str]:
        """Generate semantic tags based on domain and content"""
        tags = []
        
        # Domain-specific tag generation
        if domain == 'financial':
            if any(term in text.lower() for term in ['risk', 'volatility', 'exposure']):
                tags.append('risk_analysis')
            if any(term in text.lower() for term in ['revenue', 'profit', 'earnings']):
                tags.append('financial_performance')
        elif domain == 'legal':
            if any(term in text.lower() for term in ['liability', 'obligation', 'responsibility']):
                tags.append('legal_obligations')
            if any(term in text.lower() for term in ['compliance', 'regulation', 'requirement']):
                tags.append('regulatory_compliance')
                
        # General semantic tags
        if len(text) > 500:
            tags.append('detailed_content')
        if any(char.isdigit() for char in text):
            tags.append('contains_numbers')
            
        return tags
    
    async def _storeProcessingResults(self, results: Dict[str, Any]):
        """Store document processing results in HANA"""
        if not self.hanaClient:
            return
            
        try:
            # Store document-level results
            docInsertQuery = """
            INSERT INTO A2A_PROCESSED_DOCUMENTS (
                DOCUMENT_ID,
                PRIMARY_DOMAIN,
                CHUNK_COUNT,
                PROCESSING_TIME,
                DOCUMENT_EMBEDDING,
                DOMAIN_ANALYSIS,
                PROCESSED_AT
            ) VALUES (
                :documentId,
                :primaryDomain,
                :chunkCount,
                :processingTime,
                TO_REAL_VECTOR(:documentEmbedding),
                :domainAnalysis,
                CURRENT_TIMESTAMP
            )
            """
            
            await self.hanaClient.execute(docInsertQuery, {
                'documentId': results['documentId'],
                'primaryDomain': results['domainAnalysis']['primaryDomain'],
                'chunkCount': len(results['chunks']),
                'processingTime': results['processingMetrics']['processingTime'],
                'documentEmbedding': results['documentEmbedding'],
                'domainAnalysis': json.dumps(results['domainAnalysis'])
            })
            
            # Store chunk-level results
            for chunk in results['chunks']:
                chunkInsertQuery = """
                INSERT INTO A2A_DOCUMENT_CHUNKS (
                    CHUNK_ID,
                    DOCUMENT_ID,
                    CHUNK_TEXT,
                    CHUNK_EMBEDDING,
                    DOMAIN,
                    SEMANTIC_TAGS,
                    IMPORTANCE_SCORE,
                    POSITION_INFO
                ) VALUES (
                    :chunkId,
                    :documentId,
                    :chunkText,
                    TO_REAL_VECTOR(:chunkEmbedding),
                    :domain,
                    :semanticTags,
                    :importance,
                    :positionInfo
                )
                """
                
                await self.hanaClient.execute(chunkInsertQuery, {
                    'chunkId': chunk['chunkId'],
                    'documentId': results['documentId'],
                    'chunkText': chunk['text'][:5000],  # Limit text size
                    'chunkEmbedding': chunk['embedding'],
                    'domain': chunk['domain'],
                    'semanticTags': json.dumps(chunk['semanticTags']),
                    'importance': chunk['importance'],
                    'positionInfo': json.dumps(chunk['position'])
                })
                
        except Exception as e:
            logger.error(f"Failed to store processing results: {e}")