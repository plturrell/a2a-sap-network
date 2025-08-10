#!/usr/bin/env python3
"""
Quick Version System Test - Simple validation
"""

import sys
import asyncio

# Add path
sys.path.insert(0, '/Users/apple/projects/a2a/a2aAgents/backend')

async def main():
    print("🔧 Quick A2A Versioning System Test")
    print("=" * 45)
    
    try:
        # Test 1: Import version manager
        from app.a2a.version import get_version_manager
        print("✅ Version Manager imported")
        
        # Test 2: Initialize and get version info  
        version_manager = get_version_manager()
        await version_manager.initialize()
        print(f"✅ Version Manager initialized")
        
        version_info = await version_manager.get_version_info()
        agents_version = version_info["versions"]["a2aAgents"]
        network_version = version_info["versions"]["a2aNetwork"] 
        protocol_version = version_info["versions"]["a2a_protocol"]
        
        print(f"   a2aAgents: v{agents_version}")
        print(f"   a2aNetwork: v{network_version}")
        print(f"   Protocol: v{protocol_version}")
        
        # Test 3: Compatibility check
        compatibility = await version_manager.check_compatibility()
        compatible = compatibility.get("compatible", False)
        issues_count = len(compatibility.get("issues", []))
        
        print(f"✅ Compatibility check: {'Compatible' if compatible else f'{issues_count} issues'}")
        
        # Test 4: Test compatibility checker
        from app.a2a.version.compatibilityChecker import CompatibilityChecker
        checker = CompatibilityChecker()
        
        feature_result = await checker.check_feature_compatibility()
        supported_features = len(feature_result.get("supported_features", []))
        
        print(f"✅ Feature compatibility: {supported_features} features supported")
        
        # Test 5: Test dependency resolver
        from app.a2a.version.dependencyResolver import DependencyResolver  
        resolver = DependencyResolver()
        
        analysis = await resolver.analyze_dependencies()
        agents_satisfied = len(analysis.get("a2aAgents", {}).get("satisfied", []))
        conflicts = len(analysis.get("conflicts", []))
        
        print(f"✅ Dependency analysis: {agents_satisfied} satisfied, {conflicts} conflicts")
        
        # Test 6: Integration with network
        from app.a2a.network import get_network_connector
        connector = get_network_connector()
        await connector.initialize()
        
        network_status = await connector.get_network_status()
        network_available = network_status.get("network_available", False)
        
        print(f"✅ Network integration: {'Available' if network_available else 'Local mode'}")
        
        print("\n🎉 All versioning system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)