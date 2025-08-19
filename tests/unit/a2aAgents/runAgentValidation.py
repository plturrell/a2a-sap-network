#!/usr/bin/env python3
"""
Comprehensive Validation Suite for All 15 Enhanced A2A Agents
Validates AI Intelligence (90+ requirement) and A2A Protocol v0.2.9 Compliance

This script performs full validation and quality assurance testing.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the test directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossAgentCommunicationTest import (
    CrossAgentCommunicationTester, TestScenario, AgentEndpoint,
    AIReasoningQualityMetrics, SecurityTestMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'agent_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class A2AAgentValidator:
    """
    Comprehensive validator for all 15 enhanced A2A agents
    Ensures 90+ AI intelligence and full protocol compliance
    """
    
    def __init__(self):
        self.tester = CrossAgentCommunicationTester(environment="development")
        self.validation_results = {}
        self.start_time = None
        self.end_time = None
        
        # Define the 15 enhanced agents with expected capabilities
        self.enhanced_agents = [
            {
                "id": "enhanced_embedding_fine_tuner_agent",
                "name": "Enhanced Embedding Fine-tuner Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["contrastive_learning", "domain_adaptation", "performance_prediction"],
                "priority": "high"
            },
            {
                "id": "enhanced_calc_validation_agent",
                "name": "Enhanced Calculation Validation Agent", 
                "expected_ai_rating": 90,
                "key_capabilities": ["formal_verification", "error_prediction", "multi_step_validation"],
                "priority": "high"
            },
            {
                "id": "enhanced_data_manager_agent",
                "name": "Enhanced Data Manager Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["intelligent_organization", "predictive_access", "autonomous_operations"],
                "priority": "critical"
            },
            {
                "id": "enhanced_catalog_manager_agent",
                "name": "Enhanced Catalog Manager Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["semantic_understanding", "ml_recommendations", "adaptive_cataloging"],
                "priority": "high"
            },
            {
                "id": "enhanced_agent_builder_agent",
                "name": "Enhanced Agent Builder Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["intelligent_design", "learning_from_history", "adaptive_generation"],
                "priority": "high"
            },
            {
                "id": "enhanced_data_product_agent",
                "name": "Enhanced Data Product Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["intelligent_management", "impact_analysis", "autonomous_operations"],
                "priority": "high"
            },
            {
                "id": "enhanced_sql_agent",
                "name": "Enhanced SQL Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["multi_dialect_generation", "ml_optimization", "autonomous_tuning"],
                "priority": "high"
            },
            {
                "id": "enhanced_data_standardization_agent",
                "name": "Enhanced Data Standardization Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["adaptive_standardization", "pattern_recognition", "continuous_learning"],
                "priority": "high"
            },
            {
                "id": "enhanced_calculation_agent",
                "name": "Enhanced Calculation Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["advanced_problem_solving", "solution_optimization", "explainable_computation"],
                "priority": "high"
            },
            {
                "id": "enhanced_ai_preparation_agent",
                "name": "Enhanced AI Preparation Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["intelligent_preprocessing", "meta_learning", "autonomous_optimization"],
                "priority": "critical"
            },
            {
                "id": "enhanced_vector_processing_agent",
                "name": "Enhanced Vector Processing Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["graph_reasoning", "dynamic_embeddings", "autonomous_optimization"],
                "priority": "critical"
            },
            {
                "id": "enhanced_qa_validation_agent",
                "name": "Enhanced QA Validation Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["multi_perspective_validation", "adversarial_testing", "confidence_calibration"],
                "priority": "critical"
            },
            {
                "id": "enhanced_context_engineering_agent",
                "name": "Enhanced Context Engineering Agent",
                "expected_ai_rating": 90,
                "key_capabilities": ["hierarchical_modeling", "intelligent_adaptation", "pattern_mining"],
                "priority": "critical"
            },
            {
                "id": "enhanced_agent_manager_agent",
                "name": "Enhanced Agent Manager Agent",
                "expected_ai_rating": 92,
                "key_capabilities": ["strategic_planning", "swarm_intelligence", "emergent_behavior_management"],
                "priority": "critical"
            },
            {
                "id": "enhanced_reasoning_agent",
                "name": "Enhanced Reasoning Agent",
                "expected_ai_rating": 97,
                "key_capabilities": ["multi_strategy_reasoning", "metacognition", "advanced_problem_solving"],
                "priority": "critical"
            }
        ]
    
    async def run_comprehensive_validation(self):
        """Run comprehensive validation for all 15 enhanced agents"""
        self.start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("Starting Comprehensive A2A Agent Validation")
        logger.info(f"Timestamp: {self.start_time.isoformat()}")
        logger.info(f"Total Agents to Validate: {len(self.enhanced_agents)}")
        logger.info("=" * 80)
        
        try:
            # Initialize the testing framework
            logger.info("\n1. Initializing Testing Framework...")
            await self.tester.initialize()
            
            # Phase 1: Agent Discovery and Basic Health Checks
            logger.info("\n2. Phase 1: Agent Discovery and Health Assessment")
            discovery_results = await self._validate_agent_discovery()
            
            # Phase 2: A2A Protocol Compliance Validation
            logger.info("\n3. Phase 2: A2A Protocol v0.2.9 Compliance Testing")
            protocol_results = await self._validate_protocol_compliance()
            
            # Phase 3: AI Intelligence Validation (90+ requirement)
            logger.info("\n4. Phase 3: AI Intelligence Validation (90+ Requirement)")
            ai_results = await self._validate_ai_intelligence()
            
            # Phase 4: Cross-Agent Communication Testing
            logger.info("\n5. Phase 4: Cross-Agent Communication and Collaboration")
            communication_results = await self._validate_cross_agent_communication()
            
            # Phase 5: Security and Authentication Testing
            logger.info("\n6. Phase 5: Security and Authentication Validation")
            security_results = await self._validate_security()
            
            # Phase 6: Performance and Stress Testing
            logger.info("\n7. Phase 6: Performance and Stress Testing")
            performance_results = await self._validate_performance()
            
            # Phase 7: Complex Workflow Validation
            logger.info("\n8. Phase 7: Complex Multi-Agent Workflow Testing")
            workflow_results = await self._validate_complex_workflows()
            
            # Generate Comprehensive Report
            logger.info("\n9. Generating Comprehensive Validation Report...")
            report = await self._generate_validation_report()
            
            # Save report to file
            report_file = f"a2a_agent_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"\nValidation Report saved to: {report_file}")
            
            # Print summary
            self._print_validation_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            raise
        finally:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            logger.info(f"\nTotal Validation Duration: {duration:.2f} seconds")
            await self.tester.cleanup()
    
    async def _validate_agent_discovery(self) -> dict:
        """Validate agent discovery and availability"""
        logger.info("\nValidating Agent Discovery...")
        discovery_results = {"discovered": 0, "active": 0, "inactive": [], "details": {}}
        
        for agent in self.enhanced_agents:
            agent_id = agent["id"]
            if agent_id in self.tester.agent_endpoints:
                endpoint = self.tester.agent_endpoints[agent_id]
                discovery_results["discovered"] += 1
                
                if endpoint.status == "active":
                    discovery_results["active"] += 1
                    logger.info(f"✓ {agent['name']}: ACTIVE (Port: {endpoint.port})")
                else:
                    discovery_results["inactive"].append(agent_id)
                    logger.warning(f"✗ {agent['name']}: INACTIVE")
                
                discovery_results["details"][agent_id] = {
                    "status": endpoint.status,
                    "skills_count": len(endpoint.skills),
                    "ai_rating": endpoint.ai_intelligence_rating
                }
            else:
                logger.error(f"✗ {agent['name']}: NOT CONFIGURED")
                discovery_results["inactive"].append(agent_id)
        
        self.validation_results["discovery"] = discovery_results
        return discovery_results
    
    async def _validate_protocol_compliance(self) -> dict:
        """Validate A2A Protocol v0.2.9 compliance for all agents"""
        logger.info("\nValidating A2A Protocol Compliance...")
        
        # Run basic communication tests
        basic_results = await self.tester.run_basic_communication_tests()
        
        compliance_results = {
            "total_tests": len(basic_results),
            "compliant": 0,
            "non_compliant": 0,
            "agent_compliance": {}
        }
        
        for result in basic_results:
            if result.a2a_protocol_compliance:
                compliance_results["compliant"] += 1
            else:
                compliance_results["non_compliant"] += 1
            
            # Track per-agent compliance
            for agent_id in result.agents_involved:
                if agent_id not in compliance_results["agent_compliance"]:
                    compliance_results["agent_compliance"][agent_id] = {
                        "tests": 0, "compliant": 0, "success_rate": 0.0
                    }
                
                compliance_results["agent_compliance"][agent_id]["tests"] += 1
                if result.a2a_protocol_compliance:
                    compliance_results["agent_compliance"][agent_id]["compliant"] += 1
        
        # Calculate success rates
        for agent_id, stats in compliance_results["agent_compliance"].items():
            stats["success_rate"] = stats["compliant"] / stats["tests"] if stats["tests"] > 0 else 0.0
            
            agent_name = next((a["name"] for a in self.enhanced_agents if a["id"] == agent_id), agent_id)
            if stats["success_rate"] >= 0.9:
                logger.info(f"✓ {agent_name}: Protocol Compliance {stats['success_rate']*100:.0f}%")
            else:
                logger.warning(f"✗ {agent_name}: Protocol Compliance {stats['success_rate']*100:.0f}%")
        
        self.validation_results["protocol_compliance"] = compliance_results
        return compliance_results
    
    async def _validate_ai_intelligence(self) -> dict:
        """Validate AI intelligence meets 90+ requirement"""
        logger.info("\nValidating AI Intelligence Levels (90+ Requirement)...")
        
        # Run deep AI reasoning quality tests
        ai_test_results = await self.tester.run_deep_ai_reasoning_quality_tests()
        
        ai_validation = {
            "total_agents_tested": 0,
            "meeting_requirement": 0,
            "below_requirement": 0,
            "average_score": 0.0,
            "agent_scores": {}
        }
        
        scores = []
        for result in ai_test_results:
            if result.ai_reasoning_quality_score > 0:
                ai_validation["total_agents_tested"] += 1
                score = result.ai_reasoning_quality_score
                scores.append(score)
                
                agent_id = result.agents_involved[0] if result.agents_involved else "unknown"
                agent_info = next((a for a in self.enhanced_agents if a["id"] == agent_id), None)
                
                if agent_info:
                    expected = agent_info["expected_ai_rating"]
                    meets_requirement = score >= expected
                    
                    ai_validation["agent_scores"][agent_id] = {
                        "name": agent_info["name"],
                        "measured_score": score,
                        "expected_score": expected,
                        "meets_requirement": meets_requirement,
                        "reasoning_quality": {
                            "logical_consistency": result.reasoning_traces.get("logical_consistency", 0),
                            "reasoning_depth": result.reasoning_traces.get("reasoning_depth", 0),
                            "creativity_score": result.reasoning_traces.get("creativity_score", 0)
                        }
                    }
                    
                    if meets_requirement:
                        ai_validation["meeting_requirement"] += 1
                        logger.info(f"✓ {agent_info['name']}: AI Score {score:.1f}/100 (Required: {expected})")
                    else:
                        ai_validation["below_requirement"] += 1
                        logger.warning(f"✗ {agent_info['name']}: AI Score {score:.1f}/100 (Required: {expected})")
        
        if scores:
            ai_validation["average_score"] = sum(scores) / len(scores)
        
        self.validation_results["ai_intelligence"] = ai_validation
        return ai_validation
    
    async def _validate_cross_agent_communication(self) -> dict:
        """Validate cross-agent communication and collaboration"""
        logger.info("\nValidating Cross-Agent Communication...")
        
        # Run AI reasoning collaboration tests
        collab_results = await self.tester.run_ai_reasoning_collaboration_tests()
        
        # Run multi-agent workflow tests
        workflow_results = await self.tester.run_multi_agent_workflow_tests()
        
        communication_validation = {
            "collaboration_tests": len(collab_results),
            "successful_collaborations": sum(1 for r in collab_results if r.success),
            "workflow_tests": len(workflow_results),
            "successful_workflows": sum(1 for r in workflow_results if r.success),
            "agent_interaction_matrix": {}
        }
        
        # Build interaction matrix
        for result in collab_results + workflow_results:
            if len(result.agents_involved) >= 2:
                for i, agent1 in enumerate(result.agents_involved):
                    for agent2 in result.agents_involved[i+1:]:
                        pair = tuple(sorted([agent1, agent2]))
                        if pair not in communication_validation["agent_interaction_matrix"]:
                            communication_validation["agent_interaction_matrix"][pair] = {
                                "interactions": 0, "successful": 0
                            }
                        
                        communication_validation["agent_interaction_matrix"][pair]["interactions"] += 1
                        if result.success:
                            communication_validation["agent_interaction_matrix"][pair]["successful"] += 1
        
        logger.info(f"✓ Collaboration Success Rate: {communication_validation['successful_collaborations']}/{communication_validation['collaboration_tests']}")
        logger.info(f"✓ Workflow Success Rate: {communication_validation['successful_workflows']}/{communication_validation['workflow_tests']}")
        
        self.validation_results["communication"] = communication_validation
        return communication_validation
    
    async def _validate_security(self) -> dict:
        """Validate security and authentication mechanisms"""
        logger.info("\nValidating Security and Authentication...")
        
        # Run security tests
        security_test_results = await self.tester.run_security_authentication_tests()
        
        security_validation = {
            "total_agents_tested": len(security_test_results),
            "secure_agents": 0,
            "vulnerable_agents": 0,
            "average_security_score": 0.0,
            "security_details": {}
        }
        
        scores = []
        for result in security_test_results:
            if result.security_compliance_score > 0:
                scores.append(result.security_compliance_score)
                
                agent_id = result.agents_involved[0] if result.agents_involved else "unknown"
                agent_name = next((a["name"] for a in self.enhanced_agents if a["id"] == agent_id), agent_id)
                
                is_secure = result.security_compliance_score >= 80
                security_validation["security_details"][agent_id] = {
                    "name": agent_name,
                    "security_score": result.security_compliance_score,
                    "is_secure": is_secure,
                    "authentication_valid": result.collaborative_metrics.get("authentication_valid", False),
                    "rate_limiting": result.collaborative_metrics.get("rate_limiting_active", False),
                    "input_validation": result.collaborative_metrics.get("input_validation", False)
                }
                
                if is_secure:
                    security_validation["secure_agents"] += 1
                    logger.info(f"✓ {agent_name}: Security Score {result.security_compliance_score:.0f}%")
                else:
                    security_validation["vulnerable_agents"] += 1
                    logger.warning(f"✗ {agent_name}: Security Score {result.security_compliance_score:.0f}%")
        
        if scores:
            security_validation["average_security_score"] = sum(scores) / len(scores)
        
        self.validation_results["security"] = security_validation
        return security_validation
    
    async def _validate_performance(self) -> dict:
        """Validate performance under stress conditions"""
        logger.info("\nValidating Performance and Stress Handling...")
        
        # Run performance stress tests
        stress_results = await self.tester.run_performance_stress_tests()
        
        performance_validation = {
            "stress_tests_run": len(stress_results),
            "stress_tests_passed": sum(1 for r in stress_results if r.success),
            "average_response_time_ms": 0.0,
            "performance_metrics": {}
        }
        
        response_times = []
        for result in stress_results:
            response_times.append(result.response_time_ms)
            
            scenario_metrics = result.collaborative_metrics
            performance_validation["performance_metrics"][result.test_id] = {
                "concurrent_messages": scenario_metrics.get("concurrent_messages", 0),
                "success_rate": scenario_metrics.get("success_rate", 0),
                "avg_response_time_ms": scenario_metrics.get("avg_response_time_ms", 0),
                "passed": result.success
            }
        
        if response_times:
            performance_validation["average_response_time_ms"] = sum(response_times) / len(response_times)
        
        logger.info(f"✓ Stress Test Success Rate: {performance_validation['stress_tests_passed']}/{performance_validation['stress_tests_run']}")
        logger.info(f"✓ Average Response Time: {performance_validation['average_response_time_ms']:.0f}ms")
        
        self.validation_results["performance"] = performance_validation
        return performance_validation
    
    async def _validate_complex_workflows(self) -> dict:
        """Validate complex multi-agent workflows"""
        logger.info("\nValidating Complex Multi-Agent Workflows...")
        
        # Run complex orchestration tests
        orchestration_results = await self.tester.run_complex_orchestration_tests()
        
        workflow_validation = {
            "complex_workflows_tested": len(orchestration_results),
            "successful_workflows": sum(1 for r in orchestration_results if r.success),
            "workflow_details": {}
        }
        
        for result in orchestration_results:
            workflow_name = result.collaborative_metrics.get("workflow_name", "unknown")
            workflow_validation["workflow_details"][workflow_name] = {
                "success": result.success,
                "agents_involved": len(result.agents_involved),
                "parallel_branches": result.collaborative_metrics.get("parallel_branches", 1),
                "steps_completed": result.collaborative_metrics.get("steps_completed", 0),
                "response_time_ms": result.response_time_ms
            }
            
            status = "✓" if result.success else "✗"
            logger.info(f"{status} {workflow_name}: {result.collaborative_metrics.get('steps_completed', 0)} steps completed")
        
        self.validation_results["workflows"] = workflow_validation
        return workflow_validation
    
    async def _generate_validation_report(self) -> dict:
        """Generate comprehensive validation report"""
        
        # Calculate overall metrics
        total_agents = len(self.enhanced_agents)
        active_agents = self.validation_results["discovery"]["active"]
        
        # AI Intelligence compliance
        ai_results = self.validation_results.get("ai_intelligence", {})
        agents_meeting_ai_requirement = ai_results.get("meeting_requirement", 0)
        
        # Protocol compliance
        protocol_results = self.validation_results.get("protocol_compliance", {})
        protocol_compliance_rate = (protocol_results.get("compliant", 0) / 
                                   max(1, protocol_results.get("total_tests", 1))) * 100
        
        # Security assessment
        security_results = self.validation_results.get("security", {})
        secure_agents = security_results.get("secure_agents", 0)
        
        # Performance metrics
        performance_results = self.validation_results.get("performance", {})
        stress_test_success_rate = (performance_results.get("stress_tests_passed", 0) / 
                                   max(1, performance_results.get("stress_tests_run", 1))) * 100
        
        # Calculate overall system score
        system_score = await self._calculate_overall_system_score()
        
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
                "environment": "development",
                "a2a_protocol_version": "0.2.9"
            },
            "agent_overview": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "inactive_agents": total_agents - active_agents,
                "availability_rate": (active_agents / total_agents) * 100
            },
            "ai_intelligence_validation": {
                "agents_tested": ai_results.get("total_agents_tested", 0),
                "meeting_90_plus_requirement": agents_meeting_ai_requirement,
                "below_requirement": ai_results.get("below_requirement", 0),
                "average_ai_score": ai_results.get("average_score", 0),
                "compliance_rate": (agents_meeting_ai_requirement / max(1, ai_results.get("total_agents_tested", 1))) * 100
            },
            "protocol_compliance": {
                "compliance_rate": protocol_compliance_rate,
                "total_tests": protocol_results.get("total_tests", 0),
                "compliant_tests": protocol_results.get("compliant", 0),
                "non_compliant_tests": protocol_results.get("non_compliant", 0)
            },
            "security_assessment": {
                "secure_agents": secure_agents,
                "vulnerable_agents": security_results.get("vulnerable_agents", 0),
                "average_security_score": security_results.get("average_security_score", 0),
                "security_compliance_rate": (secure_agents / max(1, security_results.get("total_agents_tested", 1))) * 100
            },
            "performance_metrics": {
                "stress_test_success_rate": stress_test_success_rate,
                "average_response_time_ms": performance_results.get("average_response_time_ms", 0),
                "tests_passed": performance_results.get("stress_tests_passed", 0),
                "tests_run": performance_results.get("stress_tests_run", 0)
            },
            "communication_validation": {
                "collaboration_success_rate": (self.validation_results.get("communication", {}).get("successful_collaborations", 0) /
                                             max(1, self.validation_results.get("communication", {}).get("collaboration_tests", 1))) * 100,
                "workflow_success_rate": (self.validation_results.get("communication", {}).get("successful_workflows", 0) /
                                        max(1, self.validation_results.get("communication", {}).get("workflow_tests", 1))) * 100
            },
            "system_score": system_score,
            "detailed_results": self.validation_results,
            "recommendations": self._generate_recommendations(),
            "critical_issues": self._identify_critical_issues()
        }
        
        return report
    
    async def _calculate_overall_system_score(self) -> dict:
        """Calculate overall system score based on all validations"""
        
        weights = {
            "availability": 0.15,
            "ai_intelligence": 0.30,
            "protocol_compliance": 0.20,
            "security": 0.15,
            "performance": 0.10,
            "communication": 0.10
        }
        
        scores = {}
        
        # Availability score
        discovery = self.validation_results.get("discovery", {})
        scores["availability"] = (discovery.get("active", 0) / len(self.enhanced_agents)) * 100
        
        # AI Intelligence score
        ai_results = self.validation_results.get("ai_intelligence", {})
        scores["ai_intelligence"] = (ai_results.get("meeting_requirement", 0) / 
                                   max(1, ai_results.get("total_agents_tested", 1))) * 100
        
        # Protocol compliance score
        protocol = self.validation_results.get("protocol_compliance", {})
        scores["protocol_compliance"] = (protocol.get("compliant", 0) / 
                                       max(1, protocol.get("total_tests", 1))) * 100
        
        # Security score
        security = self.validation_results.get("security", {})
        scores["security"] = security.get("average_security_score", 0)
        
        # Performance score
        performance = self.validation_results.get("performance", {})
        scores["performance"] = (performance.get("stress_tests_passed", 0) / 
                               max(1, performance.get("stress_tests_run", 1))) * 100
        
        # Communication score
        communication = self.validation_results.get("communication", {})
        collab_rate = (communication.get("successful_collaborations", 0) / 
                      max(1, communication.get("collaboration_tests", 1))) * 100
        workflow_rate = (communication.get("successful_workflows", 0) / 
                        max(1, communication.get("workflow_tests", 1))) * 100
        scores["communication"] = (collab_rate + workflow_rate) / 2
        
        # Calculate weighted overall score
        overall_score = sum(scores[aspect] * weight for aspect, weight in weights.items())
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": scores,
            "weights": weights,
            "rating": self._get_rating_label(overall_score)
        }
    
    def _get_rating_label(self, score: float) -> str:
        """Get rating label based on score"""
        if score >= 95:
            return "Exceptional (Production Ready)"
        elif score >= 90:
            return "Excellent (Near Production Ready)"
        elif score >= 80:
            return "Good (Minor Improvements Needed)"
        elif score >= 70:
            return "Satisfactory (Significant Improvements Needed)"
        else:
            return "Needs Improvement (Major Work Required)"
    
    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Check AI Intelligence compliance
        ai_results = self.validation_results.get("ai_intelligence", {})
        if ai_results.get("below_requirement", 0) > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "AI Intelligence",
                "recommendation": f"Enhance AI capabilities for {ai_results['below_requirement']} agents not meeting 90+ requirement",
                "impact": "Critical for system intelligence"
            })
        
        # Check inactive agents
        inactive = self.validation_results.get("discovery", {}).get("inactive", [])
        if inactive:
            recommendations.append({
                "priority": "HIGH",
                "category": "Availability",
                "recommendation": f"Investigate and activate {len(inactive)} inactive agents: {', '.join(inactive[:3])}...",
                "impact": "System functionality incomplete"
            })
        
        # Check security vulnerabilities
        security = self.validation_results.get("security", {})
        if security.get("vulnerable_agents", 0) > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Security",
                "recommendation": f"Address security vulnerabilities in {security['vulnerable_agents']} agents",
                "impact": "Security risk to system"
            })
        
        # Check performance issues
        performance = self.validation_results.get("performance", {})
        if performance.get("average_response_time_ms", 0) > 5000:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Performance",
                "recommendation": "Optimize response times - current average exceeds 5 seconds",
                "impact": "User experience degradation"
            })
        
        # If everything is good, add optimization recommendations
        if len(recommendations) == 0:
            recommendations.extend([
                {
                    "priority": "LOW",
                    "category": "Enhancement",
                    "recommendation": "Consider implementing advanced AI features like federated learning",
                    "impact": "Future system capabilities"
                },
                {
                    "priority": "LOW", 
                    "category": "Monitoring",
                    "recommendation": "Add comprehensive APM and observability tools",
                    "impact": "Operational excellence"
                }
            ])
        
        return recommendations
    
    def _identify_critical_issues(self) -> list:
        """Identify any critical issues that must be addressed immediately"""
        critical_issues = []
        
        # Check for complete agent failures
        discovery = self.validation_results.get("discovery", {})
        if discovery.get("active", 0) < len(self.enhanced_agents) * 0.5:
            critical_issues.append({
                "severity": "CRITICAL",
                "issue": "Less than 50% of agents are active",
                "impact": "System non-functional",
                "action": "Immediate investigation and activation required"
            })
        
        # Check for AI intelligence failures
        ai_results = self.validation_results.get("ai_intelligence", {})
        if ai_results.get("average_score", 0) < 80:
            critical_issues.append({
                "severity": "HIGH",
                "issue": f"Average AI intelligence score {ai_results.get('average_score', 0):.1f} below 80",
                "impact": "AI capabilities insufficient",
                "action": "Review and enhance AI implementations"
            })
        
        # Check for protocol compliance issues
        protocol = self.validation_results.get("protocol_compliance", {})
        compliance_rate = (protocol.get("compliant", 0) / max(1, protocol.get("total_tests", 1))) * 100
        if compliance_rate < 90:
            critical_issues.append({
                "severity": "HIGH",
                "issue": f"A2A protocol compliance only {compliance_rate:.1f}%",
                "impact": "Interoperability issues",
                "action": "Fix protocol implementation issues"
            })
        
        return critical_issues
    
    def _print_validation_summary(self, report: dict):
        """Print a formatted validation summary"""
        print("\n" + "=" * 80)
        print("A2A AGENT VALIDATION SUMMARY")
        print("=" * 80)
        
        system_score = report.get("system_score", {})
        print(f"\nOVERALL SYSTEM SCORE: {system_score.get('overall_score', 0):.1f}/100")
        print(f"RATING: {system_score.get('rating', 'Unknown')}")
        
        print("\nKEY METRICS:")
        print(f"  • Active Agents: {report['agent_overview']['active_agents']}/{report['agent_overview']['total_agents']}")
        print(f"  • AI Intelligence Compliance: {report['ai_intelligence_validation']['compliance_rate']:.1f}%")
        print(f"  • Protocol Compliance: {report['protocol_compliance']['compliance_rate']:.1f}%")
        print(f"  • Security Score: {report['security_assessment']['average_security_score']:.1f}%")
        print(f"  • Performance Success: {report['performance_metrics']['stress_test_success_rate']:.1f}%")
        print(f"  • Communication Success: {report['communication_validation']['collaboration_success_rate']:.1f}%")
        
        critical_issues = report.get("critical_issues", [])
        if critical_issues:
            print(f"\nCRITICAL ISSUES: {len(critical_issues)}")
            for issue in critical_issues[:3]:  # Show top 3
                print(f"  • [{issue['severity']}] {issue['issue']}")
        else:
            print("\nNO CRITICAL ISSUES FOUND ✓")
        
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\nTOP RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  • [{rec['priority']}] {rec['recommendation']}")
        
        print("\n" + "=" * 80)


async def main():
    """Run the comprehensive A2A agent validation"""
    validator = A2AAgentValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Exit with appropriate code based on validation results
        system_score = report.get("system_score", {}).get("overall_score", 0)
        if system_score >= 90:
            logger.info("\n✅ VALIDATION PASSED - System meets all requirements")
            exit(0)
        else:
            logger.warning(f"\n⚠️  VALIDATION NEEDS ATTENTION - System score: {system_score}/100")
            exit(1)
            
    except Exception as e:
        logger.error(f"\n❌ VALIDATION FAILED - Error: {e}")
        exit(2)


if __name__ == "__main__":
    asyncio.run(main())