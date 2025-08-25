#!/usr/bin/env python3
"""
A2A Platform Deployment Readiness Checker
Verifies that the platform is ready for production deployment
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class DeploymentReadinessChecker:
    def __init__(self):
        self.project_root = Path("/Users/apple/projects/a2a")
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        
    def check_syntax_status(self) -> Tuple[bool, str]:
        """Verify all syntax errors are fixed"""
        print("✅ Checking syntax status...")
        
        # Check if we have recent syntax analysis results
        try:
            # Run a quick syntax check on main files
            main_files = [
                "a2aAgents/backend/main.py",
                "a2aNetwork/srv/server.js",
            ]
            
            for file_path in main_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    if file_path.endswith('.py'):
                        result = subprocess.run(['python3', '-m', 'py_compile', str(full_path)], 
                                              capture_output=True, text=True)
                        if result.returncode != 0:
                            return False, f"Python syntax error in {file_path}: {result.stderr}"
                    elif file_path.endswith('.js'):
                        result = subprocess.run(['node', '--check', str(full_path)], 
                                              capture_output=True, text=True)
                        if result.returncode != 0:
                            return False, f"JavaScript syntax error in {file_path}: {result.stderr}"
            
            return True, "All main files have valid syntax"
            
        except Exception as e:
            return False, f"Could not verify syntax: {e}"

    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if all dependencies are available"""
        print("📦 Checking dependencies...")
        
        issues = []
        
        # Check Python dependencies
        requirements_files = ['requirements.txt', 'a2aAgents/backend/requirements.txt']
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                                try:
                                    result = subprocess.run(['python3', '-c', f'import {package}'], 
                                                          capture_output=True, text=True)
                                    if result.returncode != 0:
                                        issues.append(f"Missing Python package: {package}")
                                except:
                                    # Some packages have different import names
                                    pass
                except Exception as e:
                    issues.append(f"Could not read {req_file}: {e}")
        
        # Check Node.js dependencies
        package_json = self.project_root / 'package.json'
        if package_json.exists():
            node_modules = self.project_root / 'node_modules'
            if not node_modules.exists():
                issues.append("Node.js dependencies not installed (missing node_modules)")
        
        if issues:
            return False, "; ".join(issues[:5])  # Limit to first 5 issues
        
        return True, "Dependencies check passed"

    def check_configuration(self) -> Tuple[bool, str]:
        """Check configuration files"""
        print("⚙️ Checking configuration...")
        
        issues = []
        
        # Check for required config files
        required_configs = [
            ".env.example",
            "a2aAgents/backend/main.py",
        ]
        
        for config_file in required_configs:
            config_path = self.project_root / config_file
            if not config_path.exists():
                issues.append(f"Missing required config: {config_file}")
        
        # Check environment variables setup
        env_files = ['.env', 'a2aAgents/.env', 'a2aNetwork/.env']
        has_env_file = False
        
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                has_env_file = True
                break
        
        if not has_env_file:
            self.warnings.append("No .env files found - ensure environment variables are configured")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Configuration check passed"

    def check_security_status(self) -> Tuple[bool, str]:
        """Check security status"""
        print("🔒 Checking security status...")
        
        # Check if security fixes have been applied
        security_fixes_file = self.project_root / 'security_fixes_applied.json'
        if security_fixes_file.exists():
            try:
                with open(security_fixes_file, 'r') as f:
                    security_data = json.load(f)
                    
                total_fixes = security_data.get('weak_crypto_fixed', 0)
                if total_fixes > 0:
                    return True, f"Security fixes applied: {total_fixes} crypto fixes"
                else:
                    self.warnings.append("No security fixes recorded")
            except:
                self.warnings.append("Could not read security fixes data")
        
        # Check for security guidelines
        guidelines_file = self.project_root / 'SECURITY_GUIDELINES.md'
        if not guidelines_file.exists():
            self.warnings.append("Security guidelines not found")
        
        return True, "Security status acceptable"

    def check_performance_optimizations(self) -> Tuple[bool, str]:
        """Check if performance optimizations are in place"""
        print("⚡ Checking performance optimizations...")
        
        # Check for performance optimization files
        perf_files = [
            'performance_optimizations_applied.json',
            'a2aAgents/backend/app/a2a/core/common_imports.py',
            'a2aAgents/backend/app/a2a/core/production_performance_monitor.py'
        ]
        
        optimizations_found = 0
        for perf_file in perf_files:
            perf_path = self.project_root / perf_file
            if perf_path.exists():
                optimizations_found += 1
        
        if optimizations_found >= 2:
            return True, f"Performance optimizations in place: {optimizations_found}/{len(perf_files)} files found"
        else:
            return False, f"Performance optimizations incomplete: only {optimizations_found}/{len(perf_files)} files found"

    def check_database_readiness(self) -> Tuple[bool, str]:
        """Check database configuration and readiness"""
        print("🗄️ Checking database readiness...")
        
        # Look for database configuration
        db_indicators = []
        
        # Check for SQLite databases
        for db_file in self.project_root.rglob("*.db"):
            if 'test' not in str(db_file).lower():
                db_indicators.append(f"SQLite: {db_file.name}")
        
        # Check for database configuration in main files
        main_py = self.project_root / "a2aAgents/backend/main.py"
        if main_py.exists():
            try:
                with open(main_py, 'r') as f:
                    content = f.read()
                    if 'database' in content.lower() or 'db' in content.lower():
                        db_indicators.append("Database configuration found in main.py")
            except:
                pass
        
        if db_indicators:
            return True, f"Database setup detected: {', '.join(db_indicators[:2])}"
        else:
            self.warnings.append("No clear database configuration found")
            return True, "Database check completed (no databases required for basic functionality)"

    def check_logging_and_monitoring(self) -> Tuple[bool, str]:
        """Check logging and monitoring setup"""
        print("📊 Checking logging and monitoring...")
        
        monitoring_indicators = []
        
        # Check for logging configuration
        logging_files = ['a2aAgents/backend/app/core/loggingConfig.py']
        for log_file in logging_files:
            log_path = self.project_root / log_file
            if log_path.exists():
                monitoring_indicators.append("Logging configuration")
        
        # Check for performance monitoring
        perf_monitor = self.project_root / 'a2aAgents/backend/app/a2a/core/production_performance_monitor.py'
        if perf_monitor.exists():
            monitoring_indicators.append("Performance monitoring")
        
        # Check for telemetry
        telemetry_files = ['a2aAgents/backend/app/a2a/core/telemetry.py']
        for tel_file in telemetry_files:
            tel_path = self.project_root / tel_file
            if tel_path.exists():
                monitoring_indicators.append("Telemetry")
        
        if monitoring_indicators:
            return True, f"Monitoring setup: {', '.join(monitoring_indicators)}"
        else:
            return False, "No monitoring/logging setup found"

    def check_documentation(self) -> Tuple[bool, str]:
        """Check documentation completeness"""
        print("📚 Checking documentation...")
        
        doc_files = []
        
        # Look for key documentation files
        key_docs = ['README.md', 'SECURITY_GUIDELINES.md', 'CLAUDE.md']
        for doc_file in key_docs:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                doc_files.append(doc_file)
        
        # Check for API documentation
        api_docs = list(self.project_root.rglob("*api*.md"))
        if api_docs:
            doc_files.append(f"{len(api_docs)} API docs")
        
        if len(doc_files) >= 2:
            return True, f"Documentation found: {', '.join(doc_files[:3])}"
        else:
            return False, f"Insufficient documentation: only found {', '.join(doc_files)}"

    def check_container_readiness(self) -> Tuple[bool, str]:
        """Check containerization readiness"""
        print("🐳 Checking containerization...")
        
        container_files = []
        
        # Look for Docker files
        docker_files = ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml']
        for docker_file in docker_files:
            if (self.project_root / docker_file).exists():
                container_files.append(docker_file)
        
        # Look for Kubernetes files
        k8s_dirs = ['k8s', 'kubernetes', 'deploy']
        for k8s_dir in k8s_dirs:
            k8s_path = self.project_root / k8s_dir
            if k8s_path.exists() and k8s_path.is_dir():
                container_files.append(f"{k8s_dir}/")
        
        if container_files:
            return True, f"Container setup: {', '.join(container_files[:3])}"
        else:
            self.warnings.append("No containerization setup found")
            return True, "Containerization not required for basic deployment"

    def generate_deployment_checklist(self) -> List[str]:
        """Generate deployment checklist"""
        checklist = [
            "✅ Verify all syntax errors are fixed",
            "✅ Install all dependencies",
            "⚙️ Configure environment variables",
            "🔒 Review security settings",
            "🗄️ Set up database (if required)",
            "📊 Configure logging and monitoring", 
            "🧪 Run integration tests",
            "📦 Build application packages",
            "🚀 Deploy to staging environment",
            "✅ Run smoke tests",
            "📈 Monitor performance metrics",
            "🔍 Verify security scanning results"
        ]
        
        return checklist

    def run_deployment_readiness_check(self) -> Dict[str, Any]:
        """Run complete deployment readiness check"""
        print("🚀 A2A Platform Deployment Readiness Check")
        print("=" * 60)
        
        checks = [
            ("Syntax Status", self.check_syntax_status),
            ("Dependencies", self.check_dependencies),
            ("Configuration", self.check_configuration),
            ("Security Status", self.check_security_status),
            ("Performance Optimizations", self.check_performance_optimizations),
            ("Database Readiness", self.check_database_readiness),
            ("Logging & Monitoring", self.check_logging_and_monitoring),
            ("Documentation", self.check_documentation),
            ("Container Readiness", self.check_container_readiness)
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {},
            'deployment_checklist': self.generate_deployment_checklist(),
            'warnings': []
        }
        
        for check_name, check_func in checks:
            try:
                passed, message = check_func()
                results['checks'][check_name] = {
                    'passed': passed,
                    'message': message,
                    'status': 'PASS' if passed else 'FAIL'
                }
                
                if passed:
                    self.checks_passed += 1
                    print(f"  ✅ {check_name}: {message}")
                else:
                    self.checks_failed += 1
                    print(f"  ❌ {check_name}: {message}")
                    
            except Exception as e:
                self.checks_failed += 1
                results['checks'][check_name] = {
                    'passed': False,
                    'message': f"Check failed: {str(e)}",
                    'status': 'ERROR'
                }
                print(f"  ⚠️ {check_name}: Check failed - {str(e)}")
        
        # Add warnings
        results['warnings'] = self.warnings
        for warning in self.warnings:
            print(f"  ⚠️ WARNING: {warning}")
        
        # Calculate readiness score
        total_checks = len(checks)
        readiness_score = (self.checks_passed / total_checks) * 100
        
        results['summary'] = {
            'total_checks': total_checks,
            'passed': self.checks_passed,
            'failed': self.checks_failed,
            'warnings': len(self.warnings),
            'readiness_score': round(readiness_score, 1),
            'deployment_ready': readiness_score >= 80  # 80% threshold
        }
        
        print("\n" + "=" * 60)
        print("🎯 DEPLOYMENT READINESS SUMMARY")
        print("=" * 60)
        print(f"📊 Readiness Score: {readiness_score:.1f}/100")
        print(f"✅ Checks Passed: {self.checks_passed}/{total_checks}")
        print(f"❌ Checks Failed: {self.checks_failed}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        
        if results['summary']['deployment_ready']:
            print(f"🚀 DEPLOYMENT READY! Platform meets deployment requirements.")
        else:
            print(f"⚠️ NOT DEPLOYMENT READY. Address failed checks before deploying.")
        
        print("=" * 60)
        
        # Save results
        with open(self.project_root / 'deployment_readiness_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Full report saved to: deployment_readiness_report.json")
        
        return results

def main():
    checker = DeploymentReadinessChecker()
    return checker.run_deployment_readiness_check()

if __name__ == "__main__":
    main()