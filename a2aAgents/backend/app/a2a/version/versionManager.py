"""
Version Manager - Centralized version and compatibility management
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import semantic_version

logger = logging.getLogger(__name__)

# Global version manager instance
_version_manager = None


class VersionManager:
    """
    Manages versions and compatibility between a2aAgents and a2aNetwork
    Ensures seamless integration and upgrade paths
    """
    
    def __init__(self):
        # Current versions
        self.agents_version = "1.0.0"
        self.network_version = None  # Will be detected
        self.protocol_version = "0.2.9"
        
        # Compatibility matrices
        self.compatibility_matrix = {
            "a2aAgents": {
                "1.0.0": {
                    "a2aNetwork": ["1.0.0", "1.0.1", "1.1.0"],
                    "protocol": ["0.2.9"]
                }
            },
            "a2aNetwork": {
                "1.0.0": {
                    "a2aAgents": ["1.0.0"],
                    "protocol": ["0.2.9"]
                }
            }
        }
        
        # Feature compatibility
        self.feature_compatibility = {
            "network_registration": {
                "min_agents_version": "1.0.0",
                "min_network_version": "1.0.0"
            },
            "trust_system": {
                "min_agents_version": "1.0.0", 
                "min_network_version": "1.0.0"
            },
            "dublin_core_compliance": {
                "min_agents_version": "1.0.0",
                "min_network_version": "1.0.0"
            },
            "ord_registration": {
                "min_agents_version": "1.0.0",
                "min_network_version": "1.0.0"
            }
        }
        
        # Version cache
        self.version_cache = {}
        self.last_network_check = None
        
        logger.info(f"VersionManager initialized - a2aAgents v{self.agents_version}")
    
    async def initialize(self) -> bool:
        """Initialize version manager and detect network version"""
        try:
            # Detect network version
            await self.detect_network_version()
            
            # Validate compatibility
            compatibility = await self.check_compatibility()
            
            if compatibility["compatible"]:
                logger.info("✅ Version compatibility verified")
                return True
            else:
                logger.warning(f"⚠️  Version compatibility issues: {compatibility['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Version manager initialization failed: {e}")
            return False
    
    async def detect_network_version(self) -> Optional[str]:
        """Detect a2aNetwork version"""
        try:
            # Try to import and get version from a2aNetwork
            network_version = await self._detect_from_network_module()
            
            if not network_version:
                # Try to get version from network API
                network_version = await self._detect_from_network_api()
            
            if network_version:
                self.network_version = network_version
                self.last_network_check = datetime.utcnow()
                logger.info(f"📡 Detected a2aNetwork version: {network_version}")
            else:
                logger.warning("⚠️  Could not detect a2aNetwork version")
                
            return network_version
            
        except Exception as e:
            logger.error(f"Failed to detect network version: {e}")
            return None
    
    async def _detect_from_network_module(self) -> Optional[str]:
        """Try to detect version from a2aNetwork module"""
        try:
            # The package is now installed, so we can import it directly
            import importlib.metadata
            version = importlib.metadata.version('a2a-network')
            return f"a2aNetwork v{version}"
        except importlib.metadata.PackageNotFoundError:
            return "a2aNetwork (unknown version)"
            logger.debug(f"Could not detect version from module: {e}")
            
        return None
    
    async def _detect_from_network_api(self) -> Optional[str]:
        """Try to detect version from network API"""
        try:
            from app.a2a.network import get_network_connector
            
            connector = get_network_connector()
            await connector.initialize()
            
            status = await connector.get_network_status()
            
            # Look for version in status response
            return status.get("version", status.get("network_version"))
            
        except Exception as e:
            logger.debug(f"Could not detect version from API: {e}")
            
        return None
    
    async def check_compatibility(self) -> Dict[str, Any]:
        """Check version compatibility between components"""
        try:
            issues = []
            compatible = True
            
            # Check if network version is detected
            if not self.network_version:
                await self.detect_network_version()
            
            if self.network_version:
                # Check agents-network compatibility
                agents_compat = self.compatibility_matrix.get("a2aAgents", {}).get(self.agents_version, {})
                network_compatible = agents_compat.get("a2aNetwork", [])
                
                if self.network_version not in network_compatible:
                    compatible = False
                    issues.append(f"a2aAgents v{self.agents_version} not compatible with a2aNetwork v{self.network_version}")
                
                # Check protocol compatibility
                protocol_compatible = agents_compat.get("protocol", [])
                if self.protocol_version not in protocol_compatible:
                    compatible = False
                    issues.append(f"A2A Protocol v{self.protocol_version} not supported")
            else:
                # Network version unknown - assume compatible but warn
                issues.append("a2aNetwork version could not be determined")
            
            # Check feature compatibility
            feature_issues = await self._check_feature_compatibility()
            issues.extend(feature_issues)
            
            if feature_issues and compatible:
                compatible = len([i for i in feature_issues if "critical" in i.lower()]) == 0
            
            return {
                "compatible": compatible,
                "issues": issues,
                "agents_version": self.agents_version,
                "network_version": self.network_version,
                "protocol_version": self.protocol_version,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return {
                "compatible": False,
                "issues": [f"Compatibility check error: {str(e)}"],
                "agents_version": self.agents_version,
                "network_version": self.network_version
            }
    
    async def _check_feature_compatibility(self) -> List[str]:
        """Check feature-specific compatibility"""
        issues = []
        
        try:
            for feature, requirements in self.feature_compatibility.items():
                min_agents = requirements.get("min_agents_version")
                min_network = requirements.get("min_network_version")
                
                # Check agents version
                if min_agents and self._version_less_than(self.agents_version, min_agents):
                    issues.append(f"Feature '{feature}' requires a2aAgents v{min_agents}+")
                
                # Check network version 
                if min_network and self.network_version and self._version_less_than(self.network_version, min_network):
                    issues.append(f"Feature '{feature}' requires a2aNetwork v{min_network}+")
                    
        except Exception as e:
            logger.error(f"Feature compatibility check failed: {e}")
            issues.append(f"Feature compatibility check error: {str(e)}")
        
        return issues
    
    def _version_less_than(self, version1: str, version2: str) -> bool:
        """Compare semantic versions"""
        try:
            return semantic_version.Version(version1) < semantic_version.Version(version2)
        except Exception:
            # Fallback to string comparison
            return version1 < version2
    
    async def get_upgrade_recommendations(self) -> Dict[str, Any]:
        """Get upgrade recommendations for better compatibility"""
        try:
            recommendations = {
                "agents_upgrades": [],
                "network_upgrades": [],
                "protocol_upgrades": [],
                "priority": "low"
            }
            
            compatibility = await self.check_compatibility()
            
            if not compatibility["compatible"]:
                recommendations["priority"] = "high"
                
                # Analyze issues and suggest upgrades
                for issue in compatibility["issues"]:
                    if "a2aAgents" in issue and "not compatible" in issue:
                        # Suggest agents upgrade
                        latest_agents = self._get_latest_compatible_agents_version()
                        if latest_agents and latest_agents != self.agents_version:
                            recommendations["agents_upgrades"].append({
                                "from": self.agents_version,
                                "to": latest_agents,
                                "reason": "Improve a2aNetwork compatibility"
                            })
                    
                    elif "a2aNetwork" in issue and "not compatible" in issue:
                        # Suggest network upgrade
                        latest_network = self._get_latest_compatible_network_version()
                        if latest_network and latest_network != self.network_version:
                            recommendations["network_upgrades"].append({
                                "from": self.network_version,
                                "to": latest_network, 
                                "reason": "Improve a2aAgents compatibility"
                            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate upgrade recommendations: {e}")
            return {
                "error": str(e),
                "priority": "unknown"
            }
    
    def _get_latest_compatible_agents_version(self) -> Optional[str]:
        """Get latest agents version compatible with current network"""
        try:
            if not self.network_version:
                return None
                
            # Find latest agents version that supports current network
            compatible_versions = []
            
            for agents_ver, compat in self.compatibility_matrix.get("a2aAgents", {}).items():
                network_compat = compat.get("a2aNetwork", [])
                if self.network_version in network_compat:
                    compatible_versions.append(agents_ver)
            
            if compatible_versions:
                return max(compatible_versions, key=lambda v: semantic_version.Version(v))
                
        except Exception as e:
            logger.error(f"Error finding latest compatible agents version: {e}")
            
        return None
    
    def _get_latest_compatible_network_version(self) -> Optional[str]:
        """Get latest network version compatible with current agents"""
        try:
            # Find latest network version that supports current agents
            agents_compat = self.compatibility_matrix.get("a2aAgents", {}).get(self.agents_version, {})
            network_versions = agents_compat.get("a2aNetwork", [])
            
            if network_versions:
                return max(network_versions, key=lambda v: semantic_version.Version(v))
                
        except Exception as e:
            logger.error(f"Error finding latest compatible network version: {e}")
            
        return None
    
    async def validate_agent_network_compatibility(self, agent_version: str, 
                                                  network_version: str) -> bool:
        """Validate compatibility between specific agent and network versions"""
        try:
            agents_compat = self.compatibility_matrix.get("a2aAgents", {}).get(agent_version, {})
            compatible_networks = agents_compat.get("a2aNetwork", [])
            
            return network_version in compatible_networks
            
        except Exception as e:
            logger.error(f"Version validation failed: {e}")
            return False
    
    async def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information"""
        try:
            compatibility = await self.check_compatibility()
            recommendations = await self.get_upgrade_recommendations()
            
            return {
                "versions": {
                    "a2aAgents": self.agents_version,
                    "a2aNetwork": self.network_version,
                    "a2a_protocol": self.protocol_version
                },
                "compatibility": compatibility,
                "recommendations": recommendations,
                "last_check": self.last_network_check.isoformat() if self.last_network_check else None,
                "supported_features": list(self.feature_compatibility.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get version info: {e}")
            return {
                "error": str(e),
                "versions": {
                    "a2aAgents": self.agents_version,
                    "a2aNetwork": self.network_version,
                    "a2a_protocol": self.protocol_version
                }
            }
    
    async def update_compatibility_matrix(self, new_matrix: Dict[str, Any]) -> bool:
        """Update compatibility matrix with new version information"""
        try:
            # Validate new matrix structure
            if not self._validate_compatibility_matrix(new_matrix):
                logger.error("Invalid compatibility matrix structure")
                return False
            
            # Merge with existing matrix
            self.compatibility_matrix.update(new_matrix)
            
            # Save to persistent storage
            await self._save_compatibility_matrix()
            
            logger.info("Compatibility matrix updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update compatibility matrix: {e}")
            return False
    
    def _validate_compatibility_matrix(self, matrix: Dict[str, Any]) -> bool:
        """Validate compatibility matrix structure"""
        try:
            required_keys = ["a2aAgents", "a2aNetwork"]
            
            for key in required_keys:
                if key not in matrix:
                    return False
                    
                for version, compat in matrix[key].items():
                    if not isinstance(compat, dict):
                        return False
                    if "protocol" not in compat:
                        return False
                        
            return True
            
        except Exception:
            return False
    
    async def _save_compatibility_matrix(self):
        """Save compatibility matrix to persistent storage"""
        try:
            matrix_file = "/tmp/a2a_compatibility_matrix.json"
            
            with open(matrix_file, 'w') as f:
                json.dump(self.compatibility_matrix, f, indent=2)
                
            logger.debug("Compatibility matrix saved")
            
        except Exception as e:
            logger.error(f"Failed to save compatibility matrix: {e}")


def get_version_manager() -> VersionManager:
    """Get global version manager instance"""
    global _version_manager
    
    if _version_manager is None:
        _version_manager = VersionManager()
    
    return _version_manager