"""
Dependency Resolver - Intelligent dependency management and resolution
"""

import asyncio
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import semantic_version

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Intelligent dependency resolver for a2aAgents and a2aNetwork integration
    Handles version conflicts, missing dependencies, and upgrade paths
    """

    def __init__(self):
        self.dependency_graph = {
            "a2aAgents": {
                "core_dependencies": {
                    "fastapi": {"version": ">=0.104.0", "critical": True},
                    "uvicorn": {"version": ">=0.24.0", "critical": True},
                    "pydantic": {"version": ">=2.5.0,<3.0.0", "critical": True},
                    "httpx": {"version": ">=0.25.0", "critical": True},
                    "pandas": {"version": ">=1.5.0", "critical": False},
                    "numpy": {"version": ">=1.21.0", "critical": False},
                    "sentence-transformers": {"version": ">=2.2.0", "critical": False},
                    "networkx": {"version": ">=3.0", "critical": False}
                },
                "optional_dependencies": {
                    "prometheus_client": {"version": ">=0.15.0", "feature": "metrics"},
                    "opentelemetry-api": {"version": ">=1.20.0", "feature": "telemetry"},
                    "jinja2": {"version": ">=3.0.0", "feature": "templating"},
                    "aiosqlite": {"version": ">=0.17.0", "feature": "sqlite_storage"}
                }
            },
            "a2aNetwork": {
                "core_dependencies": {
                    "fastapi": {"version": ">=0.104.0", "critical": True},
                    "uvicorn": {"version": ">=0.24.0", "critical": True},
                    "pydantic": {"version": ">=2.5.0", "critical": True},
                    "httpx": {"version": ">=0.25.0", "critical": True},
                    "sqlalchemy": {"version": ">=2.0.0", "critical": True},
                    "alembic": {"version": ">=1.12.0", "critical": True}
                },
                "optional_dependencies": {
                    "redis": {"version": ">=5.0.0", "feature": "caching"},
                    "celery": {"version": ">=5.3.0", "feature": "task_queue"},
                    "web3": {"version": ">=6.10.0", "feature": "blockchain"}
                }
            }
        }

        self.conflict_resolution_rules = {
            "pydantic": {
                "agents_requires": ">=2.5.0,<3.0.0",
                "network_requires": ">=2.5.0",
                "resolution": ">=2.7.4,<3.0.0"
            },
            "fastapi": {
                "agents_requires": ">=0.104.0",
                "network_requires": ">=0.104.0",
                "resolution": ">=0.104.0"
            }
        }

        logger.info("DependencyResolver initialized")

    async def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze current dependency state"""
        try:
            logger.info("🔍 Analyzing dependency state")

            # Get currently installed packages
            installed_packages = await self._get_installed_packages()

            # Check a2aAgents dependencies
            agents_analysis = await self._analyze_component_dependencies("a2aAgents", installed_packages)

            # Check a2aNetwork dependencies
            network_analysis = await self._analyze_component_dependencies("a2aNetwork", installed_packages)

            # Detect conflicts
            conflicts = await self._detect_conflicts(agents_analysis, network_analysis)

            # Generate resolution plan
            resolution_plan = await self._generate_resolution_plan(conflicts, agents_analysis, network_analysis)

            return {
                "a2aAgents": agents_analysis,
                "a2aNetwork": network_analysis,
                "conflicts": conflicts,
                "resolution_plan": resolution_plan,
                "analyzed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {
                "error": str(e),
                "analyzed_at": datetime.utcnow().isoformat()
            }

    async def _get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed Python packages"""
        try:
            import importlib.metadata

            installed = {}

            # Get all installed packages
            for dist in importlib.metadata.distributions():
                package_name = dist.metadata['Name'].lower()
                version = dist.version
                installed[package_name] = version

            return installed

        except Exception as e:
            logger.error(f"Failed to get installed packages: {e}")
            return {}

    async def _analyze_component_dependencies(self, component: str,
                                           installed_packages: Dict[str, str]) -> Dict[str, Any]:
        """Analyze dependencies for a specific component"""
        try:
            component_deps = self.dependency_graph.get(component, {})
            core_deps = component_deps.get("core_dependencies", {})
            optional_deps = component_deps.get("optional_dependencies", {})

            analysis = {
                "satisfied": [],
                "missing": [],
                "version_conflicts": [],
                "optional_available": [],
                "optional_missing": []
            }

            # Check core dependencies
            for dep_name, dep_info in core_deps.items():
                required_version = dep_info["version"]
                installed_version = installed_packages.get(dep_name.lower())

                if installed_version:
                    if self._version_satisfies(installed_version, required_version):
                        analysis["satisfied"].append({
                            "name": dep_name,
                            "required": required_version,
                            "installed": installed_version,
                            "critical": dep_info["critical"]
                        })
                    else:
                        analysis["version_conflicts"].append({
                            "name": dep_name,
                            "required": required_version,
                            "installed": installed_version,
                            "critical": dep_info["critical"]
                        })
                else:
                    analysis["missing"].append({
                        "name": dep_name,
                        "required": required_version,
                        "critical": dep_info["critical"]
                    })

            # Check optional dependencies
            for dep_name, dep_info in optional_deps.items():
                required_version = dep_info["version"]
                installed_version = installed_packages.get(dep_name.lower())

                if installed_version:
                    if self._version_satisfies(installed_version, required_version):
                        analysis["optional_available"].append({
                            "name": dep_name,
                            "required": required_version,
                            "installed": installed_version,
                            "feature": dep_info["feature"]
                        })
                    else:
                        analysis["version_conflicts"].append({
                            "name": dep_name,
                            "required": required_version,
                            "installed": installed_version,
                            "critical": False,
                            "feature": dep_info["feature"]
                        })
                else:
                    analysis["optional_missing"].append({
                        "name": dep_name,
                        "required": required_version,
                        "feature": dep_info["feature"]
                    })

            return analysis

        except Exception as e:
            logger.error(f"Component dependency analysis failed for {component}: {e}")
            return {
                "error": str(e)
            }

    def _version_satisfies(self, installed: str, required: str) -> bool:
        """Check if installed version satisfies requirement"""
        try:
            # Handle different requirement formats
            if ">=" in required:
                min_version = required.split(">=")[1].split(",")[0].strip()
                if not semantic_version.Version(installed) >= semantic_version.Version(min_version):
                    return False

            if "<" in required and not ">=" in required:
                max_version = required.split("<")[1].strip()
                if not semantic_version.Version(installed) < semantic_version.Version(max_version):
                    return False
            elif "," in required and "<" in required.split(",")[1]:
                max_version = required.split("<")[1].strip()
                if not semantic_version.Version(installed) < semantic_version.Version(max_version):
                    return False

            return True

        except Exception as e:
            logger.debug(f"Version comparison failed: {e}")
            return installed >= required  # Fallback comparison

    async def _detect_conflicts(self, agents_analysis: Dict[str, Any],
                               network_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between a2aAgents and a2aNetwork dependencies"""
        try:
            conflicts = []

            # Get all dependencies from both components
            agents_deps = {}
            network_deps = {}

            for dep in agents_analysis.get("satisfied", []):
                agents_deps[dep["name"]] = dep
            for dep in agents_analysis.get("version_conflicts", []):
                agents_deps[dep["name"]] = dep

            for dep in network_analysis.get("satisfied", []):
                network_deps[dep["name"]] = dep
            for dep in network_analysis.get("version_conflicts", []):
                network_deps[dep["name"]] = dep

            # Find common dependencies with different requirements
            common_deps = set(agents_deps.keys()) & set(network_deps.keys())

            for dep_name in common_deps:
                agents_req = agents_deps[dep_name]
                network_req = network_deps[dep_name]

                # Check if requirements are incompatible
                if agents_req["required"] != network_req["required"]:
                    # Check if there's a resolution rule
                    resolution = self.conflict_resolution_rules.get(dep_name, {}).get("resolution")

                    conflict = {
                        "dependency": dep_name,
                        "agents_requirement": agents_req["required"],
                        "network_requirement": network_req["required"],
                        "installed_version": agents_req.get("installed"),
                        "resolution_available": resolution is not None,
                        "suggested_resolution": resolution,
                        "severity": "critical" if agents_req.get("critical", False) or network_req.get("critical", False) else "moderate"
                    }

                    conflicts.append(conflict)

            return conflicts

        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []

    async def _generate_resolution_plan(self, conflicts: List[Dict[str, Any]],
                                       agents_analysis: Dict[str, Any],
                                       network_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan to resolve dependency conflicts"""
        try:
            plan = {
                "install_commands": [],
                "upgrade_commands": [],
                "manual_actions": [],
                "risk_assessment": "low"
            }

            # Handle missing dependencies
            for component, analysis in [("a2aAgents", agents_analysis), ("a2aNetwork", network_analysis)]:
                for missing in analysis.get("missing", []):
                    if missing["critical"]:
                        cmd = f"pip install \"{missing['name']}{missing['required']}\""
                        plan["install_commands"].append({
                            "command": cmd,
                            "component": component,
                            "dependency": missing["name"],
                            "reason": "Missing critical dependency"
                        })

                # Handle version conflicts
                for conflict in analysis.get("version_conflicts", []):
                    if conflict["critical"]:
                        cmd = f"pip install --upgrade \"{conflict['name']}{conflict['required']}\""
                        plan["upgrade_commands"].append({
                            "command": cmd,
                            "component": component,
                            "dependency": conflict["name"],
                            "reason": "Version conflict for critical dependency"
                        })

            # Handle cross-component conflicts
            for conflict in conflicts:
                if conflict["resolution_available"]:
                    cmd = f"pip install --upgrade \"{conflict['dependency']}{conflict['suggested_resolution']}\""
                    plan["upgrade_commands"].append({
                        "command": cmd,
                        "component": "both",
                        "dependency": conflict["dependency"],
                        "reason": f"Resolve conflict between components"
                    })
                else:
                    plan["manual_actions"].append({
                        "dependency": conflict["dependency"],
                        "issue": f"Incompatible requirements: a2aAgents needs {conflict['agents_requirement']}, a2aNetwork needs {conflict['network_requirement']}",
                        "suggestion": "Manual resolution required - check component documentation"
                    })

            # Assess risk
            if len(plan["manual_actions"]) > 0:
                plan["risk_assessment"] = "high"
            elif len(conflicts) > 2:
                plan["risk_assessment"] = "medium"
            elif len(plan["install_commands"]) + len(plan["upgrade_commands"]) > 5:
                plan["risk_assessment"] = "medium"

            return plan

        except Exception as e:
            logger.error(f"Resolution plan generation failed: {e}")
            return {
                "error": str(e)
            }

    async def execute_resolution_plan(self, plan: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Execute dependency resolution plan"""
        try:
            logger.info(f"{'🔍 Dry run:' if dry_run else '⚡ Executing:'} dependency resolution plan")

            results = {
                "install_results": [],
                "upgrade_results": [],
                "manual_actions": plan.get("manual_actions", []),
                "overall_success": True
            }

            if dry_run:
                # Just validate commands
                for cmd_info in plan.get("install_commands", []):
                    results["install_results"].append({
                        "command": cmd_info["command"],
                        "status": "would_execute",
                        "dependency": cmd_info["dependency"]
                    })

                for cmd_info in plan.get("upgrade_commands", []):
                    results["upgrade_results"].append({
                        "command": cmd_info["command"],
                        "status": "would_execute",
                        "dependency": cmd_info["dependency"]
                    })

                logger.info(f"✅ Dry run completed - {len(results['install_results'])} installs, {len(results['upgrade_results'])} upgrades")

            else:
                # Actually execute commands
                for cmd_info in plan.get("install_commands", []):
                    success = await self._execute_pip_command(cmd_info["command"])
                    results["install_results"].append({
                        "command": cmd_info["command"],
                        "status": "success" if success else "failed",
                        "dependency": cmd_info["dependency"]
                    })
                    if not success:
                        results["overall_success"] = False

                for cmd_info in plan.get("upgrade_commands", []):
                    success = await self._execute_pip_command(cmd_info["command"])
                    results["upgrade_results"].append({
                        "command": cmd_info["command"],
                        "status": "success" if success else "failed",
                        "dependency": cmd_info["dependency"]
                    })
                    if not success:
                        results["overall_success"] = False

                logger.info(f"✅ Resolution plan executed - Success: {results['overall_success']}")

            return results

        except Exception as e:
            logger.error(f"Resolution plan execution failed: {e}")
            return {
                "error": str(e),
                "overall_success": False
            }

    async def _execute_pip_command(self, command: str) -> bool:
        """Execute pip command safely"""
        try:
            logger.info(f"Executing: {command}")

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"✅ Command succeeded: {command}")
                return True
            else:
                logger.error(f"❌ Command failed: {command}")
                logger.error(f"Error: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Failed to execute pip command: {e}")
            return False

    async def validate_post_resolution(self) -> Dict[str, Any]:
        """Validate dependency state after resolution"""
        try:
            logger.info("🔍 Validating post-resolution state")

            # Re-analyze dependencies
            analysis = await self.analyze_dependencies()

            # Check if all critical dependencies are satisfied
            agents_missing = analysis.get("a2aAgents", {}).get("missing", [])
            network_missing = analysis.get("a2aNetwork", {}).get("missing", [])

            critical_missing = []
            for missing in agents_missing + network_missing:
                if missing.get("critical", False):
                    critical_missing.append(missing)

            # Check remaining conflicts
            remaining_conflicts = analysis.get("conflicts", [])
            critical_conflicts = [c for c in remaining_conflicts if c.get("severity") == "critical"]

            validation = {
                "validation_passed": len(critical_missing) == 0 and len(critical_conflicts) == 0,
                "critical_missing": critical_missing,
                "critical_conflicts": critical_conflicts,
                "total_dependencies_satisfied": {
                    "a2aAgents": len(analysis.get("a2aAgents", {}).get("satisfied", [])),
                    "a2aNetwork": len(analysis.get("a2aNetwork", {}).get("satisfied", []))
                },
                "validated_at": datetime.utcnow().isoformat()
            }

            if validation["validation_passed"]:
                logger.info("✅ Post-resolution validation passed")
            else:
                logger.warning("⚠️  Post-resolution validation found remaining issues")

            return validation

        except Exception as e:
            logger.error(f"Post-resolution validation failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e)
            }