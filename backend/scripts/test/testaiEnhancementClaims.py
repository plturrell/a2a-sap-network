"""
AI Enhancement Claims Testing
Concrete before/after comparison to validate or disprove AI enhancement effectiveness
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to the Python path
sys.path.append('/Users/apple/projects/finsight_cib/backend')

from app.clients.grokClient import get_grok_client
from app.clients.perplexityClient import get_perplexity_client
from app.ordRegistry.advancedAiEnhancer import create_advanced_ai_enhancer
from app.ordRegistry.models import ORDDocument, DublinCoreMetadata


class AIEnhancementClaimsTest:
    """Test actual AI enhancement effects with quantitative measurements"""
    
    def __init__(self):
        self.grok_client = None
        self.perplexity_client = None
        self.enhancer = None
        
    async def initialize_clients(self):
        """Initialize AI clients"""
        try:
            self.grok_client = get_grok_client()
            self.perplexity_client = get_perplexity_client()
            self.enhancer = create_advanced_ai_enhancer(self.grok_client, self.perplexity_client)
            return True
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False
    
    def create_minimal_test_document(self) -> ORDDocument:
        """Create a minimal ORD document for testing"""
        return ORDDocument(
            openResourceDiscovery="1.9.0",
            description="Basic financial API",
            dublinCore=DublinCoreMetadata(
                title="Payment API",
                creator=["Dev Team"],
                subject=["payments"],
                description="Basic payment processing API"
            ),
            dataProducts=[{
                "ordId": "com.bank.payment.api",
                "title": "Payment API",
                "shortDescription": "Payment processing",
                "description": "Basic payment API",
                "package": "com.bank.apis",
                "visibility": "internal",
                "releaseStatus": "active",
                "version": "1.0.0"
            }]
        )
    
    def measure_document_complexity(self, doc_data: Dict) -> Dict[str, int]:
        """Measure document complexity with concrete metrics"""
        measurements = {
            "total_fields": 0,
            "populated_fields": 0,
            "nested_objects": 0,
            "array_fields": 0,
            "total_characters": 0,
            "dublin_core_fields": 0,
            "data_products_count": 0,
            "api_resources_count": 0,
            "entity_types_count": 0,
            "event_resources_count": 0
        }
        
        # Convert to JSON string for analysis
        doc_str = json.dumps(doc_data, indent=2)
        measurements["total_characters"] = len(doc_str)
        
        # Count top-level fields
        if isinstance(doc_data, dict):
            measurements["total_fields"] = len(doc_data.keys())
            measurements["populated_fields"] = len([k for k, v in doc_data.items() if v is not None and v != ""])
            
            # Count specific sections
            measurements["data_products_count"] = len(doc_data.get("dataProducts", []))
            measurements["api_resources_count"] = len(doc_data.get("apiResources", []))
            measurements["entity_types_count"] = len(doc_data.get("entityTypes", []))
            measurements["event_resources_count"] = len(doc_data.get("eventResources", []))
            
            # Dublin Core analysis
            dublin_core = doc_data.get("dublinCore", {})
            if dublin_core:
                measurements["dublin_core_fields"] = len([k for k, v in dublin_core.items() if v is not None and v != ""])
            
            # Count nested objects and arrays
            for value in doc_data.values():
                if isinstance(value, dict):
                    measurements["nested_objects"] += 1
                elif isinstance(value, list):
                    measurements["array_fields"] += 1
        
        return measurements
    
    async def test_grok_enhancement_claims(self) -> Dict[str, Any]:
        """Test Grok enhancement with before/after comparison"""
        print("\n🔍 TESTING GROK ENHANCEMENT CLAIMS")
        print("-" * 50)
        
        # Create baseline document
        baseline_doc = self.create_minimal_test_document()
        baseline_data = baseline_doc.model_dump()
        baseline_metrics = self.measure_document_complexity(baseline_data)
        
        print("📊 BASELINE DOCUMENT METRICS:")
        for metric, value in baseline_metrics.items():
            print(f"   {metric}: {value}")
        
        # Apply Grok enhancement
        try:
            enhanced_result = await self.enhancer._enhance_with_grok(baseline_doc, "metadata_enrichment")
            enhanced_data = enhanced_result["enhanced_content"]
            
            # Handle case where Grok returns JSON string
            if isinstance(enhanced_data, str):
                try:
                    enhanced_data = json.loads(enhanced_data.replace("```json\n", "").replace("\n```", ""))
                except:
                    print("⚠️  Grok returned non-JSON string, using original data")
                    enhanced_data = baseline_data
            
            enhanced_metrics = self.measure_document_complexity(enhanced_data)
            
            print("\n📊 ENHANCED DOCUMENT METRICS:")
            for metric, value in enhanced_metrics.items():
                print(f"   {metric}: {value}")
            
            print("\n📈 ENHANCEMENT IMPACT:")
            impact_analysis = {}
            for metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                diff = enhanced_val - baseline_val
                percent_change = (diff / baseline_val * 100) if baseline_val > 0 else 0
                impact_analysis[metric] = {
                    "before": baseline_val,
                    "after": enhanced_val,
                    "difference": diff,
                    "percent_change": round(percent_change, 1)
                }
                print(f"   {metric}: {baseline_val} → {enhanced_val} ({diff:+d}, {percent_change:+.1f}%)")
            
            return {
                "success": True,
                "baseline_metrics": baseline_metrics,
                "enhanced_metrics": enhanced_metrics,
                "impact_analysis": impact_analysis,
                "enhancement_result": enhanced_result
            }
            
        except Exception as e:
            print(f"❌ Grok enhancement failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_perplexity_research_claims(self) -> Dict[str, Any]:
        """Test Perplexity research enhancement claims"""
        print("\n🔍 TESTING PERPLEXITY RESEARCH CLAIMS")
        print("-" * 50)
        
        try:
            # Test research query
            query = "Best practices for financial API metadata in Open Resource Discovery"
            enhanced_result = await self.enhancer._enhance_with_perplexity(
                self.create_minimal_test_document(), 
                "content_optimization"
            )
            
            enhanced_data = enhanced_result["enhanced_content"]
            ai_enhancement = enhanced_data.get("ai_enhancement", {})
            
            # Analyze research quality
            research_analysis = {
                "has_perplexity_insights": "perplexity_insights" in ai_enhancement,
                "insights_length": len(str(ai_enhancement.get("perplexity_insights", ""))),
                "has_citations": "citations" in ai_enhancement,
                "citation_count": len(ai_enhancement.get("citations", [])),
                "enhancement_type_matches": ai_enhancement.get("enhancement_type") == "content_optimization",
                "confidence_score": enhanced_result.get("confidence", 0)
            }
            
            print("📊 PERPLEXITY RESEARCH ANALYSIS:")
            for metric, value in research_analysis.items():
                print(f"   {metric}: {value}")
            
            # Show actual research content (truncated)
            insights = ai_enhancement.get("perplexity_insights", "")
            if insights:
                print(f"\n📄 RESEARCH SAMPLE (first 200 chars):")
                print(f"   {insights[:200]}...")
            
            return {
                "success": True,
                "research_analysis": research_analysis,
                "enhancement_result": enhanced_result
            }
            
        except Exception as e:
            print(f"❌ Perplexity research failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_multi_model_enhancement_claims(self) -> Dict[str, Any]:
        """Test multi-model enhancement comparison claims"""
        print("\n🔍 TESTING MULTI-MODEL ENHANCEMENT CLAIMS")
        print("-" * 50)
        
        try:
            baseline_doc = self.create_minimal_test_document()
            
            # Test all three enhancement types from Priority 4 test
            enhancement_types = ["metadata_enrichment", "content_optimization", "compliance_validation"]
            results = {}
            
            for enhancement_type in enhancement_types:
                try:
                    result = await self.enhancer.multi_model_enhancement(baseline_doc, enhancement_type)
                    
                    # Analyze the result
                    analysis = {
                        "success": True,
                        "model_used": result.model_used,
                        "confidence_score": result.confidence_score,
                        "processing_time": result.processing_time,
                        "has_quality_improvements": bool(result.quality_improvements),
                        "enhancement_type_matches": result.enhancement_type == enhancement_type
                    }
                    
                    results[enhancement_type] = analysis
                    print(f"   {enhancement_type}: ✅ Success (model: {result.model_used}, confidence: {result.confidence_score})")
                    
                except Exception as e:
                    results[enhancement_type] = {"success": False, "error": str(e)}
                    print(f"   {enhancement_type}: ❌ Failed ({e})")
            
            return {
                "success": True,
                "enhancement_results": results,
                "total_successful": len([r for r in results.values() if r.get("success")])
            }
            
        except Exception as e:
            print(f"❌ Multi-model enhancement test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_claims_verification_report(self, grok_results: Dict, perplexity_results: Dict, multi_model_results: Dict):
        """Generate final verification report"""
        print("\n" + "="*80)
        print("AI ENHANCEMENT CLAIMS VERIFICATION REPORT")
        print("="*80)
        
        # Verify Grok claims
        print("\n1. GROK METADATA ENRICHMENT CLAIMS:")
        if grok_results.get("success"):
            impact = grok_results["impact_analysis"]
            total_fields_added = impact["total_fields"]["difference"]
            characters_added = impact["total_characters"]["difference"]
            new_sections = impact["api_resources_count"]["difference"] + impact["entity_types_count"]["difference"]
            
            print(f"   ✅ Fields added: {total_fields_added}")
            print(f"   ✅ Content expansion: {characters_added} characters")
            print(f"   ✅ New sections created: {new_sections}")
            
            if total_fields_added > 5 and characters_added > 1000:
                print("   ✅ CLAIM VERIFIED: Significant metadata enrichment")
            else:
                print("   ❌ CLAIM QUESTIONABLE: Minimal enhancement")
        else:
            print("   ❌ CLAIM FAILED: Grok enhancement not working")
        
        # Verify Perplexity claims
        print("\n2. PERPLEXITY RESEARCH CLAIMS:")
        if perplexity_results.get("success"):
            analysis = perplexity_results["research_analysis"]
            insights_length = analysis["insights_length"]
            
            print(f"   ✅ Research insights: {insights_length} characters")
            print(f"   ✅ Has citations: {analysis['has_citations']}")
            print(f"   ✅ Confidence: {analysis['confidence_score']}")
            
            if insights_length > 500 and analysis["has_citations"]:
                print("   ✅ CLAIM VERIFIED: Comprehensive research capability")
            else:
                print("   ❌ CLAIM QUESTIONABLE: Limited research quality")
        else:
            print("   ❌ CLAIM FAILED: Perplexity research not working")
        
        # Verify Multi-model claims
        print("\n3. MULTI-MODEL ENHANCEMENT CLAIMS:")
        if multi_model_results.get("success"):
            successful = multi_model_results["total_successful"]
            total_types = len(multi_model_results["enhancement_results"])
            
            print(f"   ✅ Successful enhancements: {successful}/{total_types}")
            
            if successful == total_types:
                print("   ✅ CLAIM VERIFIED: Multi-model enhancement working")
            else:
                print("   ❌ CLAIM QUESTIONABLE: Partial multi-model success")
        else:
            print("   ❌ CLAIM FAILED: Multi-model enhancement not working")
        
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT:")
        claims_verified = sum([
            grok_results.get("success", False),
            perplexity_results.get("success", False),
            multi_model_results.get("success", False)
        ])
        
        if claims_verified == 3:
            print("🎉 CLAIMS LARGELY VERIFIED: AI enhancement is delivering real value")
        elif claims_verified == 2:
            print("⚠️  CLAIMS PARTIALLY VERIFIED: Some AI features working, others questionable")
        elif claims_verified == 1:
            print("❌ CLAIMS MOSTLY FALSE: Limited AI enhancement effectiveness")
        else:
            print("❌ CLAIMS COMPLETELY FALSE: No working AI enhancement")


async def main():
    """Run the comprehensive AI enhancement claims test"""
    print("Starting AI Enhancement Claims Verification Test...")
    print("Testing actual before/after effects with concrete measurements")
    
    tester = AIEnhancementClaimsTest()
    
    # Initialize clients
    if not await tester.initialize_clients():
        print("❌ Failed to initialize AI clients")
        return
    
    # Run all tests
    grok_results = await tester.test_grok_enhancement_claims()
    perplexity_results = await tester.test_perplexity_research_claims()
    multi_model_results = await tester.test_multi_model_enhancement_claims()
    
    # Generate verification report
    tester.generate_claims_verification_report(grok_results, perplexity_results, multi_model_results)


if __name__ == "__main__":
    asyncio.run(main())
