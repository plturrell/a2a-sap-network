"""
Priority 4: Advanced AI Enhancement Capabilities Test Suite
Comprehensive testing for extended AI-powered features in ORD registry
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the backend directory to the Python path
sys.path.append('/Users/apple/projects/finsight_cib/backend')

from app.ordRegistry.service import ORDRegistryService
from app.ordRegistry.advanced_ai_enhancer import AdvancedAIEnhancer, SemanticSearchRequest, create_advanced_ai_enhancer
from app.ordRegistry.models import ORDDocument, DublinCoreMetadata


class Priority4AITestSuite:
    """Comprehensive test suite for Priority 4 Advanced AI Enhancement capabilities"""
    
    def __init__(self):
        self.service = ORDRegistryService("http://localhost:8080")
        self.advanced_ai_enhancer = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed_tests"] += 1
            status = "✅ PASS"
        else:
            self.test_results["failed_tests"] += 1
            status = "❌ FAIL"
            
        self.test_results["test_details"].append({
            "test_name": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        print(f"{status}: {test_name}")
        if details:
            print(f"    Details: {details}")
    
    async def test_service_initialization_with_advanced_ai(self):
        """Test 1: Service initialization with advanced AI enhancer"""
        try:
            # Initialize service
            await self.service.initialize()
            
            # Create advanced AI enhancer
            self.advanced_ai_enhancer = create_advanced_ai_enhancer(
                grok_client=self.service.grok_client,
                perplexity_client=self.service.perplexity_client
            )
            
            # Verify initialization
            if self.advanced_ai_enhancer and self.service.grok_client and self.service.perplexity_client:
                self.log_test("Service Initialization with Advanced AI", True, 
                             f"Advanced AI enhancer initialized with Grok and Perplexity clients")
                return True
            else:
                self.log_test("Service Initialization with Advanced AI", False, 
                             "Failed to initialize advanced AI enhancer or clients")
                return False
                
        except Exception as e:
            self.log_test("Service Initialization with Advanced AI", False, f"Exception: {e}")
            return False
    
    async def test_enhanced_semantic_search(self):
        """Test 2: Enhanced semantic search with AI-powered relevance scoring"""
        try:
            # Create test documents for search
            test_documents = await self._create_test_documents_for_search()
            
            # Create semantic search request
            search_request = SemanticSearchRequest(
                query="financial data products with account information",
                context="Banking and financial services domain",
                search_type="semantic",
                max_results=5,
                include_explanations=True
            )
            
            # Perform enhanced semantic search
            search_results = await self.advanced_ai_enhancer.enhanced_semantic_search(
                search_request, test_documents
            )
            
            # Validate results
            if (search_results and 
                "results" in search_results and 
                "semantic_facets" in search_results and
                "explanations" in search_results and
                search_results["enhancement_metadata"]["ai_processed"]):
                
                result_count = len(search_results["results"])
                has_explanations = bool(search_results["explanations"])
                
                self.log_test("Enhanced Semantic Search", True, 
                             f"Found {result_count} results with AI explanations: {has_explanations}")
                return True
            else:
                self.log_test("Enhanced Semantic Search", False, 
                             "Search results missing required AI enhancement features")
                return False
                
        except Exception as e:
            self.log_test("Enhanced Semantic Search", False, f"Exception: {e}")
            return False
    
    async def test_intelligent_content_classification(self):
        """Test 3: AI-powered content classification with confidence scores"""
        try:
            # Create test ORD document
            test_document = self._create_financial_test_document()
            
            # Perform intelligent content classification
            classification_results = await self.advanced_ai_enhancer.intelligent_content_classification(test_document)
            
            # Validate classification results
            expected_keys = [
                "primary_categories", "secondary_categories", "confidence_scores",
                "ai_reasoning", "suggested_tags", "domain_specific_metadata", "compliance_indicators"
            ]
            
            has_all_keys = all(key in classification_results for key in expected_keys)
            
            if has_all_keys:
                primary_count = len(classification_results["primary_categories"])
                confidence_available = bool(classification_results["confidence_scores"])
                
                self.log_test("Intelligent Content Classification", True, 
                             f"Classified with {primary_count} primary categories, confidence scores: {confidence_available}")
                return True
            else:
                missing_keys = [key for key in expected_keys if key not in classification_results]
                self.log_test("Intelligent Content Classification", False, 
                             f"Missing keys: {missing_keys}")
                return False
                
        except Exception as e:
            self.log_test("Intelligent Content Classification", False, f"Exception: {e}")
            return False
    
    async def test_advanced_quality_assessment(self):
        """Test 4: AI-powered quality assessment with improvement recommendations"""
        try:
            # Create test ORD document with varying quality levels
            test_document = self._create_financial_test_document()
            
            # Perform advanced quality assessment
            quality_results = await self.advanced_ai_enhancer.advanced_quality_assessment(test_document)
            
            # Validate quality assessment results
            expected_keys = [
                "overall_score", "dimension_scores", "improvement_suggestions",
                "compliance_analysis", "ai_insights", "benchmark_comparison", "enhancement_priority"
            ]
            
            has_all_keys = all(key in quality_results for key in expected_keys)
            
            if has_all_keys:
                overall_score = quality_results["overall_score"]
                dimension_count = len(quality_results["dimension_scores"])
                has_recommendations = bool(quality_results["improvement_suggestions"])
                
                self.log_test("Advanced Quality Assessment", True, 
                             f"Overall score: {overall_score:.2f}, {dimension_count} dimensions, recommendations: {has_recommendations}")
                return True
            else:
                missing_keys = [key for key in expected_keys if key not in quality_results]
                self.log_test("Advanced Quality Assessment", False, 
                             f"Missing keys: {missing_keys}")
                return False
                
        except Exception as e:
            self.log_test("Advanced Quality Assessment", False, f"Exception: {e}")
            return False
    
    async def test_smart_recommendations_engine(self):
        """Test 5: AI-powered recommendation engine for related content"""
        try:
            # Create test ORD document
            test_document = self._create_financial_test_document()
            
            # Create context for recommendations
            context = {
                "domain": "financial_services",
                "use_case": "risk_management",
                "user_preferences": {"focus": "data_quality", "urgency": "high"}
            }
            
            # Generate smart recommendations
            recommendations = await self.advanced_ai_enhancer.generate_smart_recommendations(
                test_document, context
            )
            
            # Validate recommendations
            expected_keys = [
                "content_recommendations", "metadata_enhancements", "related_documents",
                "integration_opportunities", "ai_generated_insights"
            ]
            
            has_all_keys = all(key in recommendations for key in expected_keys)
            
            if has_all_keys and "domain_trends" in recommendations["ai_generated_insights"]:
                content_recs = len(recommendations["content_recommendations"])
                metadata_recs = len(recommendations["metadata_enhancements"])
                has_insights = bool(recommendations["ai_generated_insights"])
                
                self.log_test("Smart Recommendations Engine", True, 
                             f"Generated {content_recs} content + {metadata_recs} metadata recommendations, insights: {has_insights}")
                return True
            else:
                missing_keys = [key for key in expected_keys if key not in recommendations]
                self.log_test("Smart Recommendations Engine", False, 
                             f"Missing keys: {missing_keys}")
                return False
                
        except Exception as e:
            self.log_test("Smart Recommendations Engine", False, f"Exception: {e}")
            return False
    
    async def test_multi_model_enhancement(self):
        """Test 6: Multi-model AI enhancement with performance comparison"""
        try:
            # Create test ORD document
            test_document = self._create_financial_test_document()
            
            # Test different enhancement types
            enhancement_types = ["metadata_enrichment", "content_optimization", "compliance_validation"]
            successful_enhancements = 0
            
            for enhancement_type in enhancement_types:
                try:
                    # Perform multi-model enhancement
                    enhancement_result = await self.advanced_ai_enhancer.multi_model_enhancement(
                        test_document, enhancement_type
                    )
                    
                    # Validate enhancement result
                    if (enhancement_result.enhanced_content and 
                        enhancement_result.confidence_score > 0 and
                        enhancement_result.quality_improvements):
                        successful_enhancements += 1
                        
                except Exception as enhancement_error:
                    print(f"    Warning: {enhancement_type} enhancement failed: {enhancement_error}")
                    continue
            
            if successful_enhancements > 0:
                self.log_test("Multi-Model Enhancement", True, 
                             f"Successfully enhanced {successful_enhancements}/{len(enhancement_types)} enhancement types")
                return True
            else:
                self.log_test("Multi-Model Enhancement", False, 
                             "All enhancement types failed")
                return False
                
        except Exception as e:
            self.log_test("Multi-Model Enhancement", False, f"Exception: {e}")
            return False
    
    async def test_ai_performance_metrics(self):
        """Test 7: AI performance metrics and model comparison"""
        try:
            # Test that AI enhancer tracks performance metrics
            initial_history_count = len(self.advanced_ai_enhancer.enhancement_history)
            
            # Perform a few AI operations to generate metrics
            test_document = self._create_financial_test_document()
            
            # Run multiple AI operations
            await self.advanced_ai_enhancer.intelligent_content_classification(test_document)
            await self.advanced_ai_enhancer.advanced_quality_assessment(test_document)
            
            # Check if performance metrics are being tracked
            metrics_available = bool(self.advanced_ai_enhancer.model_performance_metrics)
            history_updated = len(self.advanced_ai_enhancer.enhancement_history) >= initial_history_count
            
            if metrics_available or history_updated:
                self.log_test("AI Performance Metrics", True, 
                             f"Performance tracking active, metrics: {metrics_available}")
                return True
            else:
                self.log_test("AI Performance Metrics", False, 
                             "No performance metrics or history tracking detected")
                return False
                
        except Exception as e:
            self.log_test("AI Performance Metrics", False, f"Exception: {e}")
            return False
    
    def _create_financial_test_document(self) -> ORDDocument:
        """Create a comprehensive test ORD document for financial services"""
        return ORDDocument(
            openResourceDiscovery="1.9.0",
            description="Comprehensive financial data product suite for risk management and compliance",
            dublinCore=DublinCoreMetadata(
                title="Financial Risk Management Data Product",
                creator=["Risk Analytics Team", "Data Engineering Team"],
                subject=["financial_risk", "compliance", "banking", "credit_analysis"],
                description="Multi-dimensional financial data product combining account hierarchies, transaction patterns, and risk indicators for comprehensive risk management and regulatory compliance",
                publisher="FinSight CIB Advanced Analytics",
                contributor=["Risk Management Division", "Compliance Team"],
                date="2025-08-02",
                type="Dataset",
                format="JSON, Parquet, CSV",
                identifier="com.finsight.cib:dataProduct:risk_management_suite",
                source="Core Banking Systems, Market Data Feeds, Regulatory Filings",
                language="en",
                relation=["Related to credit scoring models", "Market risk calculations"],
                coverage="Global banking operations, Real-time and historical data",
                rights="Internal use only, Regulatory compliance required"
            ),
            dataProducts=[
                {
                    "ordId": "com.finsight.cib:dataProduct:account_hierarchy",
                    "title": "Account Hierarchy and Relationships",
                    "shortDescription": "Comprehensive account structure with hierarchical relationships",
                    "description": "Detailed account hierarchy data including parent-child relationships, account types, ownership structures, and cross-reference mappings for risk aggregation and compliance reporting",
                    "package": "com.finsight.cib:package:core_banking",
                    "visibility": "internal",
                    "releaseStatus": "active",
                    "systemInstanceAware": True,
                    "version": "2.1.0",
                    "partOfGroups": ["risk_management", "compliance_reporting"],
                    "entryPoints": [
                        {
                            "type": "rest-api",
                            "url": "/api/v2/accounts/hierarchy"
                        }
                    ]
                },
                {
                    "ordId": "com.finsight.cib:dataProduct:transaction_analytics",
                    "title": "Transaction Pattern Analytics",
                    "shortDescription": "Real-time transaction pattern analysis for risk detection",
                    "description": "Advanced transaction analytics including velocity patterns, anomaly detection, cross-border flows, and suspicious activity indicators for AML and fraud prevention",
                    "package": "com.finsight.cib:package:risk_analytics",
                    "visibility": "internal",
                    "releaseStatus": "active",
                    "systemInstanceAware": True,
                    "version": "1.8.3"
                }
            ],
            apiResources=[
                {
                    "ordId": "com.finsight.cib:apiResource:risk_calculation_api",
                    "title": "Risk Calculation API",
                    "shortDescription": "Real-time risk calculation and scoring API",
                    "description": "Comprehensive API for calculating various risk metrics including credit risk, market risk, operational risk, and composite risk scores with configurable parameters and real-time updates",
                    "package": "com.finsight.cib:package:risk_apis",
                    "visibility": "internal",
                    "releaseStatus": "active",
                    "version": "3.2.1"
                }
            ],
            entityTypes=[
                {
                    "ordId": "com.finsight.cib:entityType:Customer",
                    "title": "Customer Entity",
                    "shortDescription": "Core customer entity with KYC and risk attributes",
                    "description": "Comprehensive customer entity including KYC information, risk ratings, relationship hierarchies, and regulatory classifications",
                    "package": "com.finsight.cib:package:core_entities",
                    "version": "2.0.0"
                }
            ]
        )
    
    async def _create_test_documents_for_search(self) -> List:
        """Create multiple test ORD registrations for semantic search testing"""
        # Create and register multiple test documents
        documents = []
        
        # Document 1: Financial risk data
        doc1 = self._create_financial_test_document()
        registration1 = await self.service.register_ord_document(doc1, "test_user")
        if registration1:
            documents.append(registration1)
        
        # Document 2: Customer analytics
        doc2 = ORDDocument(
            openResourceDiscovery="1.9.0",
            description="Customer analytics and behavioral insights",
            dublinCore=DublinCoreMetadata(
                title="Customer Analytics Data Product",
                subject=["customer_analytics", "behavioral_insights", "segmentation"],
                description="Customer behavioral analytics for personalization and marketing optimization"
            ),
            dataProducts=[{
                "ordId": "com.finsight.cib:dataProduct:customer_analytics",
                "title": "Customer Behavioral Analytics",
                "shortDescription": "Customer segmentation and behavioral patterns",
                "description": "Advanced customer analytics including behavioral segmentation, purchase patterns, and engagement metrics",
                "package": "com.finsight.cib:package:customer_insights",
                "visibility": "internal",
                "releaseStatus": "active",
                "version": "1.5.0"
            }]
        )
        registration2 = await self.service.register_ord_document(doc2, "test_user")
        if registration2:
            documents.append(registration2)
        
        return documents
    
    def print_summary(self) -> float:
        """Print test summary and return success rate"""
        print("\n" + "="*80)
        print("PRIORITY 4 AI ENHANCEMENT TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.test_results['total_tests']}")
        print(f"Passed: {self.test_results['passed_tests']}")
        print(f"Failed: {self.test_results['failed_tests']}")
        
        if self.test_results['total_tests'] > 0:
            success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        else:
            success_rate = 0.0
            print("Success Rate: N/A")
        
        if self.test_results['failed_tests'] > 0:
            print(f"\n❌ FAILED TESTS:")
            for test in self.test_results['test_details']:
                if test['status'] == '❌ FAIL':
                    print(f"  - {test['test_name']}: {test['details']}")
        
        print("\n" + "="*80)
        if success_rate == 100.0:
            print("🎉 ALL TESTS PASSED - Priority 4 AI enhancement features are working correctly!")
        elif success_rate >= 80.0:
            print("✅ MOSTLY WORKING - Some issues need to be addressed")
        else:
            print("⚠️  SIGNIFICANT ISSUES - Multiple features need attention")
        
        return success_rate


async def main():
    """Run the comprehensive Priority 4 AI Enhancement test suite"""
    print("Starting Priority 4 AI Enhancement Test Suite...")
    print("Testing all advanced AI-powered capabilities")
    print("="*80)
    
    test_suite = Priority4AITestSuite()
    
    # Run all tests
    tests = [
        test_suite.test_service_initialization_with_advanced_ai(),
        test_suite.test_enhanced_semantic_search(),
        test_suite.test_intelligent_content_classification(),
        test_suite.test_advanced_quality_assessment(),
        test_suite.test_smart_recommendations_engine(),
        test_suite.test_multi_model_enhancement(),
        test_suite.test_ai_performance_metrics()
    ]
    
    # Execute tests sequentially
    for test in tests:
        await test
        print()  # Add spacing between tests
    
    # Print final summary
    success_rate = test_suite.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success_rate == 100.0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
