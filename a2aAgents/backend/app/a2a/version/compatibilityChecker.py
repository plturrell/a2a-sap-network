"""
Compatibility Checker - Advanced compatibility validation and resolution
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
import semantic_version

from .versionManager import get_version_manager

logger = logging.getLogger(__name__)


class CompatibilityChecker:
    """
    Advanced compatibility checking between a2aAgents and a2aNetwork
    Provides detailed analysis and resolution suggestions
    """
    
    def __init__(self):
        self.version_manager = get_version_manager()
        self.compatibility_cache = {}
        self.feature_matrix = {
            # Core features
            "agent_registration": {
                "introduced": {"agents": "1.0.0", "network": "1.0.0"},
                "deprecated": None,
                "breaking_changes": []
            },
            "network_messaging": {
                "introduced": {"agents": "1.0.0", "network": "1.0.0"}, 
                "deprecated": None,
                "breaking_changes": []
            },
            "trust_system": {
                "introduced": {"agents": "1.0.0", "network": "1.0.0"},
                "deprecated": None,
                "breaking_changes": []
            },
            "dublin_core_compliance": {
                "introduced": {"agents": "1.0.0", "network": "1.0.0"},
                "deprecated": None,
                "breaking_changes": []
            },
            # Advanced features  
            "ord_integration": {
                "introduced": {"agents": "1.0.0", "network": "1.0.0"},
                "deprecated": None,
                "breaking_changes": []
            },
            "blockchain_trust": {
                "introduced": {"agents": "1.1.0", "network": "1.1.0"},
                "deprecated": None,
                "breaking_changes": []
            },
            "cross_chain_messaging": {
                "introduced": {"agents": "2.0.0", "network": "2.0.0"},
                "deprecated": None,
                "breaking_changes": []
            }
        }
        
        logger.info("CompatibilityChecker initialized")
    
    async def perform_comprehensive_check(self) -> Dict[str, Any]:
        """Perform comprehensive compatibility analysis"""
        try:
            logger.info("🔍 Performing comprehensive compatibility check")
            
            # Get version information
            await self.version_manager.initialize()
            
            # Basic compatibility
            basic_compat = await self.version_manager.check_compatibility()
            
            # Feature compatibility
            feature_compat = await self.check_feature_compatibility()
            
            # Protocol compatibility
            protocol_compat = await self.check_protocol_compatibility()
            
            # API compatibility
            api_compat = await self.check_api_compatibility()
            
            # Dependency compatibility
            dep_compat = await self.check_dependency_compatibility()
            
            # Overall assessment
            overall_compatible = (
                basic_compat.get("compatible", False) and
                feature_compat.get("compatible", False) and
                protocol_compat.get("compatible", False) and
                api_compat.get("compatible", False) and
                dep_compat.get("compatible", False)
            )
            
            # Risk assessment
            risk_level = self._calculate_risk_level([
                basic_compat, feature_compat, protocol_compat, api_compat, dep_compat
            ])
            
            # Resolution suggestions
            resolutions = await self._generate_resolutions([
                basic_compat, feature_compat, protocol_compat, api_compat, dep_compat
            ])
            
            result = {
                "overall_compatible": overall_compatible,
                "risk_level": risk_level,
                "checks": {
                    "basic": basic_compat,
                    "features": feature_compat,
                    "protocol": protocol_compat,
                    "api": api_compat,
                    "dependencies": dep_compat
                },
                "resolutions": resolutions,
                "checked_at": datetime.utcnow().isoformat(),
                "summary": self._generate_summary(overall_compatible, risk_level, resolutions)
            }
            
            # Cache result
            self.compatibility_cache[datetime.utcnow().isoformat()[:10]] = result
            
            logger.info(f"✅ Comprehensive compatibility check completed - Compatible: {overall_compatible}")
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive compatibility check failed: {e}")
            return {
                "overall_compatible": False,
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    async def check_feature_compatibility(self) -> Dict[str, Any]:
        """Check feature-level compatibility"""
        try:
            issues = []
            warnings = []
            supported_features = []
            unsupported_features = []
            deprecated_features = []
            
            agents_version = self.version_manager.agents_version
            network_version = self.version_manager.network_version or "1.0.0"
            
            for feature, info in self.feature_matrix.items():
                # Check if feature is supported in current versions
                agents_min = info["introduced"]["agents"]
                network_min = info["introduced"]["network"]
                
                agents_supports = self._version_gte(agents_version, agents_min)
                network_supports = self._version_gte(network_version, network_min)
                
                if agents_supports and network_supports:
                    supported_features.append(feature)
                    
                    # Check for deprecation
                    if info.get("deprecated"):
                        dep_agents = info["deprecated"].get("agents")
                        dep_network = info["deprecated"].get("network")
                        
                        if (dep_agents and self._version_gte(agents_version, dep_agents)) or \
                           (dep_network and self._version_gte(network_version, dep_network)):
                            deprecated_features.append(feature)
                            warnings.append(f"Feature '{feature}' is deprecated")
                else:
                    unsupported_features.append(feature)
                    if not agents_supports:
                        issues.append(f"Feature '{feature}' requires a2aAgents v{agents_min}+")
                    if not network_supports:
                        issues.append(f"Feature '{feature}' requires a2aNetwork v{network_min}+")
                
                # Check for breaking changes
                for breaking_change in info.get("breaking_changes", []):
                    change_version = breaking_change.get("version")
                    if change_version and self._version_gte(agents_version, change_version):
                        issues.append(f"Breaking change in '{feature}' at v{change_version}: {breaking_change.get('description')}")
            
            compatible = len(issues) == 0
            
            return {
                "compatible": compatible,
                "supported_features": supported_features,
                "unsupported_features": unsupported_features,
                "deprecated_features": deprecated_features,
                "issues": issues,
                "warnings": warnings,
                "feature_coverage": len(supported_features) / len(self.feature_matrix) * 100
            }
            
        except Exception as e:
            logger.error(f"Feature compatibility check failed: {e}")
            return {
                "compatible": False,
                "error": str(e)
            }
    
    async def check_protocol_compatibility(self) -> Dict[str, Any]:
        """Check A2A protocol compatibility"""
        try:
            protocol_version = self.version_manager.protocol_version
            issues = []
            warnings = []
            
            # Define protocol compatibility rules
            protocol_rules = {
                "0.2.9": {
                    "required_features": ["agent_registration", "network_messaging", "dublin_core_compliance"],
                    "optional_features": ["trust_system", "ord_integration"],
                    "message_format": "json",
                    "security_level": "basic"
                }
            }
            
            if protocol_version not in protocol_rules:
                issues.append(f"Unknown A2A protocol version: {protocol_version}")
                return {
                    "compatible": False,
                    "issues": issues
                }
            
            rules = protocol_rules[protocol_version]
            
            # Check required features are supported
            feature_compat = await self.check_feature_compatibility()
            supported = set(feature_compat.get("supported_features", []))
            
            for required_feature in rules["required_features"]:
                if required_feature not in supported:
                    issues.append(f"Protocol v{protocol_version} requires feature '{required_feature}'")
            
            # Check optional features
            optional_supported = []
            for optional_feature in rules["optional_features"]:
                if optional_feature in supported:
                    optional_supported.append(optional_feature)
            
            compatible = len(issues) == 0
            
            return {
                "compatible": compatible,
                "protocol_version": protocol_version,
                "required_features_supported": len(issues) == 0,
                "optional_features_supported": optional_supported,
                "issues": issues,
                "warnings": warnings,
                "compliance_score": (len(rules["required_features"]) - len(issues)) / len(rules["required_features"]) * 100
            }
            
        except Exception as e:
            logger.error(f"Protocol compatibility check failed: {e}")
            return {
                "compatible": False,
                "error": str(e)
            }
    
    async def check_api_compatibility(self) -> Dict[str, Any]:
        """Check API compatibility between components"""
        try:
            issues = []
            warnings = []
            
            # Define API compatibility matrix
            api_endpoints = {
                "registry": {
                    "/api/v1/agents/register": {"min_version": "1.0.0", "deprecated": None},
                    "/api/v1/agents/{agent_id}": {"min_version": "1.0.0", "deprecated": None},
                    "/api/v1/ord/register": {"min_version": "1.0.0", "deprecated": None}
                },
                "trust": {
                    "/api/v1/trust/initialize": {"min_version": "1.0.0", "deprecated": None},
                    "/api/v1/trust/verify": {"min_version": "1.0.0", "deprecated": None}
                }
            }
            
            network_version = self.version_manager.network_version or "1.0.0"
            
            # Check API endpoint compatibility
            compatible_endpoints = []
            incompatible_endpoints = []
            
            for service, endpoints in api_endpoints.items():
                for endpoint, info in endpoints.items():
                    min_version = info["min_version"]
                    
                    if self._version_gte(network_version, min_version):
                        compatible_endpoints.append(f"{service}{endpoint}")
                        
                        # Check for deprecation
                        if info.get("deprecated") and self._version_gte(network_version, info["deprecated"]):
                            warnings.append(f"API endpoint {service}{endpoint} is deprecated")
                    else:
                        incompatible_endpoints.append(f"{service}{endpoint}")
                        issues.append(f"API endpoint {service}{endpoint} requires a2aNetwork v{min_version}+")
            
            compatible = len(issues) == 0
            
            return {
                "compatible": compatible,
                "compatible_endpoints": compatible_endpoints,
                "incompatible_endpoints": incompatible_endpoints,
                "issues": issues,
                "warnings": warnings,
                "api_coverage": len(compatible_endpoints) / (len(compatible_endpoints) + len(incompatible_endpoints)) * 100 if (len(compatible_endpoints) + len(incompatible_endpoints)) > 0 else 100
            }
            
        except Exception as e:
            logger.error(f"API compatibility check failed: {e}")
            return {
                "compatible": False,
                "error": str(e)
            }
    
    async def check_dependency_compatibility(self) -> Dict[str, Any]:
        """Check dependency compatibility"""
        try:
            issues = []
            warnings = []
            
            # Check Python version compatibility
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            if sys.version_info < (3, 8):
                issues.append("Python 3.8+ required for a2aAgents")
            elif sys.version_info >= (3, 12):
                warnings.append("Python 3.12+ not fully tested")
            
            # Check critical dependencies
            critical_deps = {
                "fastapi": {"min_version": "0.104.0", "package": "fastapi"},
                "pydantic": {"min_version": "2.5.0", "package": "pydantic"},
                "httpx": {"min_version": "0.25.0", "package": "httpx"},
                "asyncio": {"min_version": "3.8.0", "package": None}  # Built-in
            }
            
            missing_deps = []
            version_conflicts = []
            
            for dep_name, dep_info in critical_deps.items():
                try:
                    if dep_info["package"]:
                        import importlib
                        module = importlib.import_module(dep_info["package"])
                        
                        if hasattr(module, "__version__"):
                            version = module.__version__
                            min_version = dep_info["min_version"]
                            
                            if not self._version_gte(version, min_version):
                                version_conflicts.append(f"{dep_name} v{version} < required v{min_version}")
                        else:
                            warnings.append(f"Could not determine {dep_name} version")
                    
                except ImportError:
                    missing_deps.append(dep_name)
                    issues.append(f"Missing required dependency: {dep_name}")
            
            issues.extend(version_conflicts)
            
            compatible = len(issues) == 0
            
            return {
                "compatible": compatible,
                "python_version": python_version,
                "missing_dependencies": missing_deps,
                "version_conflicts": version_conflicts,
                "issues": issues,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Dependency compatibility check failed: {e}")
            return {
                "compatible": False,
                "error": str(e)
            }
    
    def _version_gte(self, version1: str, version2: str) -> bool:
        """Check if version1 >= version2"""
        try:
            return semantic_version.Version(version1) >= semantic_version.Version(version2)
        except Exception:
            return version1 >= version2
    
    def _calculate_risk_level(self, check_results: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level"""
        try:
            issue_count = sum(len(result.get("issues", [])) for result in check_results)
            warning_count = sum(len(result.get("warnings", [])) for result in check_results)
            
            if issue_count >= 5:
                return "critical"
            elif issue_count >= 2:
                return "high"  
            elif issue_count >= 1 or warning_count >= 3:
                return "medium"
            elif warning_count >= 1:
                return "low"
            else:
                return "minimal"
                
        except Exception:
            return "unknown"
    
    async def _generate_resolutions(self, check_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate resolution suggestions"""
        try:
            resolutions = []
            
            # Collect all issues
            all_issues = []
            for result in check_results:
                all_issues.extend(result.get("issues", []))
            
            # Generate specific resolutions
            for issue in all_issues:
                resolution = None
                
                if "requires a2aAgents" in issue:
                    resolution = {
                        "type": "upgrade",
                        "component": "a2aAgents",
                        "description": "Upgrade a2aAgents to latest compatible version",
                        "priority": "high"
                    }
                elif "requires a2aNetwork" in issue:
                    resolution = {
                        "type": "upgrade",
                        "component": "a2aNetwork", 
                        "description": "Upgrade a2aNetwork to latest compatible version",
                        "priority": "high"
                    }
                elif "Missing required dependency" in issue:
                    dep_name = issue.split(": ")[1]
                    resolution = {
                        "type": "install",
                        "component": dep_name,
                        "description": f"Install missing dependency: {dep_name}",
                        "priority": "critical"
                    }
                elif "Breaking change" in issue:
                    resolution = {
                        "type": "code_update",
                        "component": "application",
                        "description": "Update application code to handle breaking changes",
                        "priority": "high"
                    }
                
                if resolution and resolution not in resolutions:
                    resolutions.append(resolution)
            
            return resolutions
            
        except Exception as e:
            logger.error(f"Failed to generate resolutions: {e}")
            return []
    
    def _generate_summary(self, compatible: bool, risk_level: str, resolutions: List[Dict[str, Any]]) -> str:
        """Generate human-readable summary"""
        try:
            if compatible and risk_level == "minimal":
                return "✅ All systems compatible - no action required"
            elif compatible and risk_level == "low":
                return "⚠️  Systems compatible with minor warnings - monitor for updates"
            elif not compatible and risk_level == "critical":
                return f"❌ Critical compatibility issues - {len(resolutions)} actions required immediately"
            elif not compatible and risk_level == "high":
                return f"⚠️  High compatibility risk - {len(resolutions)} actions recommended"
            elif not compatible and risk_level == "medium":
                return f"⚠️  Moderate compatibility issues - {len(resolutions)} actions suggested"
            else:
                return f"⚠️  Compatibility status unclear - risk level: {risk_level}"
                
        except Exception:
            return "❓ Unable to generate compatibility summary"