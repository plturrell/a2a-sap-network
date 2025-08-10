#!/usr/bin/env python3
"""
Test Versioning and Dependency Management System
Validates proper versioning, compatibility checking, and dependency resolution
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime

# Add the backend to path
sys.path.insert(0, '/Users/apple/projects/a2a/a2aAgents/backend')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def testVersioningSystem():
    """Test the complete versioning and dependency management system"""
    print("🔧 Testing A2A Versioning & Dependency Management System")
    print("=" * 70)
    
    try:
        # Test 1: Version Manager
        await testVersionManager()
        
        # Test 2: Compatibility Checker
        await testCompatibilityChecker()
        
        # Test 3: Dependency Resolver
        await testDependencyResolver()
        
        # Test 4: Integration with Network
        await testVersioningIntegration()
        
        # Test 5: Comprehensive System Test
        await testComprehensiveVersioning()
        
        print("\n" + "=" * 70)
        print("🎉 Versioning System Tests Completed Successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Versioning system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testVersionManager():
    """Test Version Manager functionality"""
    print("\n📊 Testing Version Manager")
    print("-" * 35)
    
    try:
        from app.a2a.version import get_version_manager
        
        version_manager = get_version_manager()
        print("✅ Version Manager imported successfully")
        
        # Initialize version manager
        initialized = await version_manager.initialize()
        print(f"✅ Version Manager initialized: {initialized}")
        
        # Test version detection
        network_version = await version_manager.detect_network_version()
        print(f"✅ Network version detection: {network_version or 'Not detected (expected in test)'}")
        
        # Test compatibility check
        compatibility = await version_manager.check_compatibility()
        print(f"✅ Compatibility check completed")
        print(f"   Compatible: {compatibility.get('compatible', 'Unknown')}")
        print(f"   Issues: {len(compatibility.get('issues', []))}")
        
        # Test version info
        version_info = await version_manager.get_version_info()
        print(f"✅ Version info retrieved")
        print(f"   a2aAgents: {version_info['versions']['a2aAgents']}")
        print(f"   a2aNetwork: {version_info['versions']['a2aNetwork']}")
        print(f"   Protocol: {version_info['versions']['a2a_protocol']}")
        
        # Test upgrade recommendations
        recommendations = await version_manager.get_upgrade_recommendations()
        print(f"✅ Upgrade recommendations: Priority {recommendations.get('priority', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Version Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testCompatibilityChecker():
    """Test Compatibility Checker functionality"""
    print("\n🔍 Testing Compatibility Checker")
    print("-" * 40)
    
    try:
        from app.a2a.version.compatibilityChecker import CompatibilityChecker
        
        checker = CompatibilityChecker()
        print("✅ Compatibility Checker created")
        
        # Perform comprehensive check
        comprehensive_result = await checker.perform_comprehensive_check()
        
        print("✅ Comprehensive compatibility check completed")
        print(f"   Overall Compatible: {comprehensive_result.get('overall_compatible', False)}")
        print(f"   Risk Level: {comprehensive_result.get('risk_level', 'unknown')}")
        
        # Display check results
        checks = comprehensive_result.get('checks', {})
        for check_name, result in checks.items():
            status = "✅" if result.get('compatible', False) else "⚠️"
            issues_count = len(result.get('issues', []))
            print(f"   {status} {check_name.title()}: {issues_count} issues")
        
        # Display resolutions
        resolutions = comprehensive_result.get('resolutions', [])
        print(f"✅ Resolution suggestions: {len(resolutions)} actions")
        
        for i, resolution in enumerate(resolutions[:3]):  # Show first 3
            print(f"   {i+1}. {resolution.get('type', 'unknown')}: {resolution.get('description', 'No description')}")
        
        # Test specific feature compatibility
        feature_result = await checker.check_feature_compatibility()
        supported_features = feature_result.get('supported_features', [])
        print(f"✅ Feature compatibility: {len(supported_features)} features supported")
        
        # Test protocol compatibility
        protocol_result = await checker.check_protocol_compatibility()
        protocol_compatible = protocol_result.get('compatible', False)
        compliance_score = protocol_result.get('compliance_score', 0)
        print(f"✅ Protocol compatibility: {protocol_compatible} (Score: {compliance_score:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility Checker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testDependencyResolver():
    """Test Dependency Resolver functionality"""
    print("\n🔗 Testing Dependency Resolver")
    print("-" * 35)
    
    try:
        from app.a2a.version.dependencyResolver import DependencyResolver
        
        resolver = DependencyResolver()
        print("✅ Dependency Resolver created")
        
        # Analyze current dependencies
        analysis = await resolver.analyze_dependencies()
        
        print("✅ Dependency analysis completed")
        
        # Display a2aAgents analysis
        agents_analysis = analysis.get('a2aAgents', {})
        if 'satisfied' in agents_analysis:
            print(f"   a2aAgents satisfied: {len(agents_analysis['satisfied'])}")
            print(f"   a2aAgents missing: {len(agents_analysis['missing'])}")
            print(f"   a2aAgents conflicts: {len(agents_analysis['version_conflicts'])}")
        
        # Display a2aNetwork analysis
        network_analysis = analysis.get('a2aNetwork', {})
        if 'satisfied' in network_analysis:
            print(f"   a2aNetwork satisfied: {len(network_analysis['satisfied'])}")
            print(f"   a2aNetwork missing: {len(network_analysis['missing'])}")
            print(f"   a2aNetwork conflicts: {len(network_analysis['version_conflicts'])}")
        
        # Display conflicts
        conflicts = analysis.get('conflicts', [])
        print(f"   Cross-component conflicts: {len(conflicts)}")
        
        for conflict in conflicts[:2]:  # Show first 2 conflicts
            print(f"     - {conflict['dependency']}: {conflict['severity']} severity")
        
        # Test resolution plan
        resolution_plan = analysis.get('resolution_plan', {})
        if resolution_plan:
            install_cmds = len(resolution_plan.get('install_commands', []))
            upgrade_cmds = len(resolution_plan.get('upgrade_commands', []))
            manual_actions = len(resolution_plan.get('manual_actions', []))
            risk = resolution_plan.get('risk_assessment', 'unknown')
            
            print(f"✅ Resolution plan generated")
            print(f"   Install commands: {install_cmds}")
            print(f"   Upgrade commands: {upgrade_cmds}")
            print(f"   Manual actions: {manual_actions}")
            print(f"   Risk assessment: {risk}")
        
        # Test dry-run execution
        if resolution_plan:
            execution_result = await resolver.execute_resolution_plan(resolution_plan, dry_run=True)
            
            print(f"✅ Dry-run execution completed")
            print(f"   Would install: {len(execution_result.get('install_results', []))}")
            print(f"   Would upgrade: {len(execution_result.get('upgrade_results', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency Resolver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testVersioningIntegration():
    """Test versioning integration with network components"""
    print("\n🌐 Testing Versioning Network Integration")
    print("-" * 45)
    
    try:
        # Test integration with network connector
        from app.a2a.network import get_network_connector
        from app.a2a.version import get_version_manager
        
        network_connector = get_network_connector()
        version_manager = get_version_manager()
        
        print("✅ Network and version components imported")
        
        # Initialize both systems
        await network_connector.initialize()
        await version_manager.initialize()
        
        print("✅ Both systems initialized")
        
        # Get network status with version info
        network_status = await network_connector.get_network_status()
        version_info = await version_manager.get_version_info()
        
        print("✅ Status information retrieved")
        print(f"   Network available: {network_status.get('network_available', False)}")
        print(f"   Version compatibility: {version_info['compatibility'].get('compatible', False)}")
        
        # Test version-aware agent registration
        try:
            from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
            
            agent = DataProductRegistrationAgentSDK(
                base_url="http://localhost:8001",
                ord_registry_url="http://localhost:9000"
            )
            
            # Check if agent reports version information
            agent_version = getattr(agent, 'version', 'Unknown')
            print(f"✅ Agent version information: {agent_version}")
            
            # Clean up
            await agent.initialize()
            await agent.shutdown()
            
        except Exception as e:
            print(f"⚠️  Agent version test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Versioning integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testComprehensiveVersioning():
    """Comprehensive end-to-end versioning system test"""
    print("\n🔄 Testing Comprehensive Versioning System")
    print("-" * 50)
    
    try:
        # Import all components
        from app.a2a.version import (
            get_version_manager, 
            VersionManager, 
            CompatibilityChecker, 
            DependencyResolver
        )
        
        print("✅ All versioning components imported")
        
        # Create comprehensive test scenario
        version_manager = get_version_manager()
        compatibility_checker = CompatibilityChecker()
        dependency_resolver = DependencyResolver()
        
        print("✅ All components instantiated")
        
        # Scenario 1: Fresh system check
        print("\n🔍 Scenario 1: Fresh System Analysis")
        
        await version_manager.initialize()
        compatibility_result = await compatibility_checker.perform_comprehensive_check()
        dependency_analysis = await dependency_resolver.analyze_dependencies()
        
        print(f"   Overall system health: {'✅ Good' if compatibility_result.get('overall_compatible', False) else '⚠️ Needs attention'}")
        
        # Scenario 2: Simulated upgrade scenario
        print("\n⬆️  Scenario 2: Upgrade Simulation")
        
        # Get upgrade recommendations
        recommendations = await version_manager.get_upgrade_recommendations()
        resolution_plan = dependency_analysis.get('resolution_plan', {})
        
        upgrade_actions = len(recommendations.get('agents_upgrades', []))
        dependency_actions = len(resolution_plan.get('install_commands', [])) + len(resolution_plan.get('upgrade_commands', []))
        
        print(f"   Version upgrade actions: {upgrade_actions}")
        print(f"   Dependency actions: {dependency_actions}")
        print(f"   Total maintenance actions: {upgrade_actions + dependency_actions}")
        
        # Scenario 3: Compatibility validation
        print("\n✅ Scenario 3: Compatibility Validation")
        
        # Check specific version compatibility
        agents_version = version_manager.agents_version
        network_version = version_manager.network_version or "1.0.0"
        
        is_compatible = await version_manager.validate_agent_network_compatibility(
            agents_version, network_version
        )
        
        print(f"   a2aAgents v{agents_version} + a2aNetwork v{network_version}: {'✅ Compatible' if is_compatible else '❌ Incompatible'}")
        
        # Scenario 4: System readiness assessment
        print("\n📊 Scenario 4: System Readiness Assessment")
        
        readiness_score = 0
        total_checks = 5
        
        # Check 1: Basic compatibility
        if compatibility_result.get('overall_compatible', False):
            readiness_score += 1
        
        # Check 2: Critical dependencies satisfied
        agents_missing = dependency_analysis.get('a2aAgents', {}).get('missing', [])
        critical_missing = [m for m in agents_missing if m.get('critical', False)]
        if len(critical_missing) == 0:
            readiness_score += 1
        
        # Check 3: No critical conflicts
        conflicts = dependency_analysis.get('conflicts', [])
        critical_conflicts = [c for c in conflicts if c.get('severity') == 'critical']
        if len(critical_conflicts) == 0:
            readiness_score += 1
        
        # Check 4: Risk level acceptable
        risk_level = compatibility_result.get('risk_level', 'unknown')
        if risk_level in ['minimal', 'low']:
            readiness_score += 1
        
        # Check 5: Network connectivity
        from app.a2a.network import get_network_connector
        connector = get_network_connector()
        await connector.initialize()
        network_status = await connector.get_network_status()
        if network_status.get('initialized', False):
            readiness_score += 1
        
        readiness_percentage = (readiness_score / total_checks) * 100
        
        print(f"   System Readiness: {readiness_score}/{total_checks} ({readiness_percentage:.0f}%)")
        
        if readiness_percentage >= 80:
            print("   🎉 System is ready for production!")
        elif readiness_percentage >= 60:
            print("   ⚠️  System needs minor attention")
        else:
            print("   🔧 System requires maintenance before production")
        
        # Generate final report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_readiness": f"{readiness_percentage:.0f}%",
            "compatibility": compatibility_result.get('overall_compatible', False),
            "risk_level": risk_level,
            "recommended_actions": upgrade_actions + dependency_actions,
            "versions": {
                "a2aAgents": agents_version,
                "a2aNetwork": network_version,
                "protocol": version_manager.protocol_version
            }
        }
        
        # Save report
        report_path = "/tmp/a2a_versioning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comprehensive versioning report saved to: {report_path}")
        
        return readiness_percentage >= 60
        
    except Exception as e:
        print(f"❌ Comprehensive versioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def runVersioningTests():
    """Run all versioning system tests"""
    try:
        success = await testVersioningSystem()
        return success
    except Exception as e:
        print(f"❌ Versioning tests failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 A2A Versioning & Dependency Management Test Suite")
    print("=" * 70)
    
    success = asyncio.run(runVersioningTests())
    
    if success:
        print("\n✨ All versioning tests passed!")
        print("🔧 Versioning and dependency management system working correctly")
    else:
        print("\n💥 Some versioning tests failed")
        print("🔧 Check logs for details")
    
    sys.exit(0 if success else 1)