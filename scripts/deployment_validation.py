#!/usr/bin/env python3
"""
A2A Platform Deployment Validation Script
Comprehensive pre-deployment checks for A2A compliance and system readiness
"""

import os
import sys
import json
import asyncio
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, manually load .env
    def load_env_file():
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentValidator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'warnings': [],
            'errors': [],
            'deployment_ready': False
        }
        
    def run_check(self, check_name: str, check_function, critical: bool = True):
        """Run a validation check and record results"""
        logger.info(f"🔍 Running check: {check_name}")
        
        try:
            start_time = time.time()
            result = check_function()
            duration = time.time() - start_time
            
            self.results['checks'][check_name] = {
                'status': 'PASS' if result['success'] else 'FAIL',
                'duration': round(duration, 2),
                'critical': critical,
                'details': result.get('details', {}),
                'message': result.get('message', ''),
                'warnings': result.get('warnings', [])
            }
            
            if result['success']:
                logger.info(f"✅ {check_name}: PASSED")
            else:
                logger.error(f"❌ {check_name}: FAILED - {result.get('message', '')}")
                if critical:
                    self.results['errors'].append(f"{check_name}: {result.get('message', '')}")
                else:
                    self.results['warnings'].append(f"{check_name}: {result.get('message', '')}")
                    
            # Add warnings from the check
            for warning in result.get('warnings', []):
                self.results['warnings'].append(f"{check_name}: {warning}")
                
        except Exception as e:
            logger.error(f"💥 {check_name}: ERROR - {str(e)}")
            self.results['checks'][check_name] = {
                'status': 'ERROR',
                'critical': critical,
                'message': str(e),
                'details': {}
            }
            if critical:
                self.results['errors'].append(f"{check_name}: {str(e)}")
            else:
                self.results['warnings'].append(f"{check_name}: {str(e)}")
    
    def check_a2a_compliance(self) -> Dict[str, Any]:
        """Validate A2A protocol compliance"""
        try:
            # Run A2A compliance validator
            result = subprocess.run([
                sys.executable, 'scripts/a2a_compliance_validator.py'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                # Parse compliance report
                try:
                    with open('a2a_compliance_report.json', 'r') as f:
                        compliance_data = json.load(f)
                    
                    total_files = compliance_data['summary']['total_files']
                    compliant_files = compliance_data['summary']['compliant_files']
                    violations = compliance_data['summary']['total_violations']
                    compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0
                    
                    # We consider 95%+ compliance as acceptable for deployment
                    success = compliance_rate >= 95.0
                    
                    details = {
                        'compliance_rate': round(compliance_rate, 1),
                        'total_files': total_files,
                        'compliant_files': compliant_files,
                        'violations': violations,
                        'violations_by_category': compliance_data['violations_by_category']
                    }
                    
                    message = f"A2A compliance: {compliance_rate:.1f}% ({violations} violations)"
                    warnings = []
                    
                    if compliance_rate < 100:
                        warnings.append(f"{violations} A2A protocol violations remain")
                    
                    return {
                        'success': success,
                        'message': message,
                        'details': details,
                        'warnings': warnings
                    }
                    
                except FileNotFoundError:
                    return {
                        'success': False,
                        'message': 'A2A compliance report not found'
                    }
            else:
                return {
                    'success': False,
                    'message': f'A2A compliance check failed: {result.stderr}'
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'A2A compliance check error: {str(e)}'
            }
    
    def check_blockchain_readiness(self) -> Dict[str, Any]:
        """Check blockchain infrastructure readiness"""
        details = {}
        warnings = []
        
        # Check environment variables
        required_env_vars = [
            'BLOCKCHAIN_URL',
            'A2A_CONTRACT_ADDRESS', 
            'A2A_PRIVATE_KEY'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            warnings.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        details['environment_variables'] = {
            var: '✓ Set' if os.getenv(var) else '✗ Missing'
            for var in required_env_vars
        }
        
        # Check for blockchain client files
        blockchain_files = [
            'shared/core/blockchain-client.js',
            'shared/blockchain/blockchain-event-server.js',
            'a2aAgents/backend/app/a2a/core/network_client.py'
        ]
        
        missing_files = [f for f in blockchain_files if not Path(f).exists()]
        if missing_files:
            warnings.append(f"Missing blockchain files: {', '.join(missing_files)}")
        
        details['blockchain_files'] = {
            f: '✓ Exists' if Path(f).exists() else '✗ Missing'
            for f in blockchain_files
        }
        
        # Success if no critical issues
        success = len(missing_vars) == 0 and len(missing_files) == 0
        message = "Blockchain infrastructure ready" if success else "Blockchain infrastructure issues found"
        
        return {
            'success': success,
            'message': message,
            'details': details,
            'warnings': warnings
        }
    
    def check_database_schema(self) -> Dict[str, Any]:
        """Validate database schema compilation"""
        try:
            # Check both possible schema locations
            schema_locations = [
                Path('a2aNetwork/db/schema.cds'),
                Path('db/schema.cds')  # Alternative location
            ]
            
            schema_file = None
            for location in schema_locations:
                if location.exists():
                    schema_file = location
                    break
            
            if not schema_file:
                return {
                    'success': False,
                    'message': 'CDS schema file not found in any expected location'
                }
            
            # Try to compile CDS schema - use project root for compilation
            result = subprocess.run([
                'npx', 'cds', 'compile', str(schema_file)
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            success = result.returncode == 0
            details = {
                'schema_size': schema_file.stat().st_size,
                'compile_output': result.stdout if success else result.stderr
            }
            
            message = "CDS schema compiles successfully" if success else f"CDS compilation failed: {result.stderr[:200]}..."
            
            return {
                'success': success,
                'message': message,
                'details': details
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Schema validation error: {str(e)}'
            }
    
    def check_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration"""
        details = {}
        warnings = []
        
        # Check for security files
        security_files = [
            'a2aAgents/backend/app/a2a/core/security_middleware.py',
            'a2aAgents/backend/app/a2a/core/security_base.py'
        ]
        
        missing_files = [f for f in security_files if not Path(f).exists()]
        if missing_files:
            warnings.append(f"Missing security files: {', '.join(missing_files)}")
        
        details['security_files'] = {
            f: '✓ Exists' if Path(f).exists() else '✗ Missing'
            for f in security_files
        }
        
        # Check security environment variables
        security_env_vars = [
            'JWT_SECRET_KEY',
            'ENABLE_RATE_LIMITING',
            'ENABLE_AUDIT_LOGGING'
        ]
        
        missing_security_vars = [var for var in security_env_vars if not os.getenv(var)]
        if missing_security_vars:
            warnings.append(f"Missing security env vars: {', '.join(missing_security_vars)}")
        
        details['security_env_vars'] = {
            var: '✓ Set' if os.getenv(var) else '✗ Missing'
            for var in security_env_vars
        }
        
        success = len(missing_files) == 0
        message = "Security configuration complete" if success else "Security configuration issues found"
        
        return {
            'success': success,
            'message': message,
            'details': details,
            'warnings': warnings
        }
    
    def check_agent_migration(self) -> Dict[str, Any]:
        """Check agent migration to SecureA2AAgent"""
        try:
            agent_dirs = Path('a2aAgents/backend/app/a2a/agents').glob('*/active')
            migrated_agents = 0
            total_agents = 0
            unmigrated_agents = []
            
            for agent_dir in agent_dirs:
                for agent_file in agent_dir.glob('*.py'):
                    # Skip __init__.py files and test utilities
                    if agent_file.name in ['__init__.py'] or agent_file.name.startswith('test_') or 'test' in agent_file.name.lower():
                        continue
                    
                    # Only count files that actually define agent classes
                    content = agent_file.read_text()
                    if not re.search(r'class\s+\w+.*:', content):
                        continue
                    
                    total_agents += 1
                    
                    if 'SecureA2AAgent' in content or 'A2AAgentBase' in content:
                        migrated_agents += 1
                    else:
                        unmigrated_agents.append(str(agent_file))
            
            migration_rate = (migrated_agents / total_agents * 100) if total_agents > 0 else 100
            success = migration_rate >= 90  # 90% migration rate acceptable
            
            details = {
                'total_agents': total_agents,
                'migrated_agents': migrated_agents,
                'migration_rate': round(migration_rate, 1),
                'unmigrated_agents': unmigrated_agents[:10]  # Limit list
            }
            
            message = f"Agent migration: {migration_rate:.1f}% ({migrated_agents}/{total_agents})"
            warnings = []
            
            if migration_rate < 100:
                warnings.append(f"{len(unmigrated_agents)} agents not migrated to SecureA2AAgent")
            
            return {
                'success': success,
                'message': message,
                'details': details,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Agent migration check error: {str(e)}'
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check for required dependencies"""
        details = {}
        warnings = []
        
        # Check Python dependencies  
        python_deps = [
            ('fastapi', 'fastapi'),
            ('uvicorn', 'uvicorn'), 
            ('websockets', 'websockets'),
            ('aiohttp', 'aiohttp'),
            ('cryptography', 'cryptography'),
            ('jwt', 'PyJWT'),  # Import name vs package name
            ('web3', 'web3')
        ]
        
        missing_python_deps = []
        for import_name, package_name in python_deps:
            try:
                __import__(import_name)
                details[f'python_{package_name}'] = '✓ Installed'
            except ImportError:
                missing_python_deps.append(package_name)
                details[f'python_{package_name}'] = '✗ Missing'
        
        if missing_python_deps:
            warnings.append(f"Missing Python dependencies: {', '.join(missing_python_deps)}")
        
        # Check Node.js dependencies (if package.json exists)
        package_json = Path('a2aNetwork/package.json')
        if package_json.exists():
            node_modules = Path('a2aNetwork/node_modules')
            details['nodejs_dependencies'] = '✓ Installed' if node_modules.exists() else '✗ Missing'
            if not node_modules.exists():
                warnings.append("Node.js dependencies not installed (run npm install)")
        
        success = len(missing_python_deps) == 0
        message = "Dependencies satisfied" if success else f"Missing {len(missing_python_deps)} dependencies"
        
        return {
            'success': success,
            'message': message,
            'details': details,
            'warnings': warnings
        }
    
    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment report"""
        
        # Calculate overall status
        critical_failures = sum(1 for check in self.results['checks'].values() 
                               if check.get('critical', True) and check['status'] != 'PASS')
        
        total_warnings = len(self.results['warnings'])
        total_errors = len(self.results['errors'])
        
        if critical_failures == 0 and total_errors == 0:
            self.results['overall_status'] = 'READY'
            self.results['deployment_ready'] = True
        elif critical_failures == 0:
            self.results['overall_status'] = 'READY_WITH_WARNINGS' 
            self.results['deployment_ready'] = True
        else:
            self.results['overall_status'] = 'NOT_READY'
            self.results['deployment_ready'] = False
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("A2A PLATFORM DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Status: {self.results['overall_status']}")
        report.append(f"Deployment Ready: {'✅ YES' if self.results['deployment_ready'] else '❌ NO'}")
        report.append("")
        
        # Summary
        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for check in self.results['checks'].values() if check['status'] == 'PASS')
        
        report.append(f"📊 SUMMARY")
        report.append(f"   Checks Run: {total_checks}")
        report.append(f"   Passed: {passed_checks}")
        report.append(f"   Failed: {total_checks - passed_checks}")
        report.append(f"   Warnings: {total_warnings}")
        report.append(f"   Critical Errors: {critical_failures}")
        report.append("")
        
        # Check results
        report.append("🔍 CHECK RESULTS:")
        for check_name, check_result in self.results['checks'].items():
            status_icon = "✅" if check_result['status'] == 'PASS' else "❌"
            critical_flag = " [CRITICAL]" if check_result.get('critical', True) else ""
            duration = check_result.get('duration', 0)
            
            report.append(f"   {status_icon} {check_name}{critical_flag} ({duration}s)")
            if check_result.get('message'):
                report.append(f"      {check_result['message']}")
        report.append("")
        
        # Warnings
        if self.results['warnings']:
            report.append("⚠️  WARNINGS:")
            for warning in self.results['warnings']:
                report.append(f"   • {warning}")
            report.append("")
        
        # Errors
        if self.results['errors']:
            report.append("❌ CRITICAL ERRORS:")
            for error in self.results['errors']:
                report.append(f"   • {error}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if self.results['deployment_ready']:
            report.append("   • Platform is ready for deployment")
            if self.results['warnings']:
                report.append("   • Address warnings before production deployment")
            report.append("   • Run final integration tests")
            report.append("   • Monitor blockchain network connectivity")
        else:
            report.append("   • Fix all critical errors before deployment")
            report.append("   • Re-run validation after fixes")
            report.append("   • Consider staging environment deployment first")
        
        report.append("")
        report.append("=" * 80)
        
        return "\\n".join(report)
    
    async def run_all_checks(self):
        """Run all deployment validation checks"""
        logger.info("🚀 Starting A2A Platform Deployment Validation")
        
        # Run all checks
        self.run_check("A2A Protocol Compliance", self.check_a2a_compliance, critical=True)
        self.run_check("Blockchain Readiness", self.check_blockchain_readiness, critical=True)
        self.run_check("Database Schema", self.check_database_schema, critical=True)
        self.run_check("Security Configuration", self.check_security_configuration, critical=True)
        self.run_check("Agent Migration", self.check_agent_migration, critical=False)
        self.run_check("Dependencies", self.check_dependencies, critical=True)
        
        # Generate and display report
        report = self.generate_deployment_report()
        print(report)
        
        # Save detailed results
        with open('deployment_validation_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("📄 Detailed report saved to: deployment_validation_report.json")
        
        return self.results['deployment_ready']

async def main():
    """Main function"""
    validator = DeploymentValidator()
    deployment_ready = await validator.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if deployment_ready else 1)

if __name__ == "__main__":
    asyncio.run(main())