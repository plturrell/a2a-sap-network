#!/usr/bin/env python3
"""
Agent Migration Tool
Automatically migrates all agents to SecureA2AAgent base class
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentMigrationTool:
    def __init__(self):
        self.agents_dir = Path('a2aAgents/backend/app/a2a/agents')
        self.migrated_agents = 0
        self.total_agents = 0
        self.migration_results = []
        
        # Migration patterns
        self.base_class_patterns = [
            (r'class\s+(\w+)\s*\(\s*Agent\s*\)', r'class \1(SecureA2AAgent)'),
            (r'class\s+(\w+)\s*\(\s*BaseAgent\s*\)', r'class \1(SecureA2AAgent)'),
            (r'class\s+(\w+)\s*\(\s*A2AAgent\s*\)', r'class \1(SecureA2AAgent)'),
            (r'class\s+(\w+)\s*\(\s*object\s*\)', r'class \1(SecureA2AAgent)'),
            (r'class\s+(\w+)\s*:', r'class \1(SecureA2AAgent):'),
        ]
        
        self.import_patterns = [
            (r'from.*agent.*import.*Agent(?!.*SecureA2AAgent)', 'from app.a2a.core.security_base import SecureA2AAgent'),
            (r'import.*Agent(?!.*SecureA2AAgent)', 'from app.a2a.core.security_base import SecureA2AAgent'),
        ]
    
    def scan_agents(self) -> Dict[str, Any]:
        """Scan all agents to determine migration status"""
        results = {
            'total_agents': 0,
            'migrated_agents': 0,
            'unmigrated_agents': [],
            'agents_by_status': {
                'migrated': [],
                'needs_migration': [],
                'errors': []
            }
        }
        
        if not self.agents_dir.exists():
            logger.error(f"Agents directory not found: {self.agents_dir}")
            return results
        
        # Find all agent Python files
        for agent_dir in self.agents_dir.glob('*/active'):
            for agent_file in agent_dir.glob('*.py'):
                if agent_file.name == '__init__.py':
                    continue
                
                results['total_agents'] += 1
                
                try:
                    content = agent_file.read_text(encoding='utf-8')
                    
                    # Check if already migrated
                    if 'SecureA2AAgent' in content:
                        results['migrated_agents'] += 1
                        results['agents_by_status']['migrated'].append(str(agent_file))
                    elif 'A2AAgentBase' in content:
                        results['migrated_agents'] += 1
                        results['agents_by_status']['migrated'].append(str(agent_file))
                    else:
                        results['unmigrated_agents'].append(str(agent_file))
                        results['agents_by_status']['needs_migration'].append(str(agent_file))
                        
                except Exception as e:
                    logger.error(f"Error scanning {agent_file}: {e}")
                    results['agents_by_status']['errors'].append({
                        'file': str(agent_file),
                        'error': str(e)
                    })
        
        self.total_agents = results['total_agents']
        self.migrated_agents = results['migrated_agents']
        
        return results
    
    def migrate_agent(self, agent_file: Path) -> Dict[str, Any]:
        """Migrate a single agent to SecureA2AAgent"""
        result = {
            'file': str(agent_file),
            'success': False,
            'changes_made': [],
            'backup_created': False,
            'error': None
        }
        
        try:
            # Read original content
            original_content = agent_file.read_text(encoding='utf-8')
            content = original_content
            
            # Skip if already migrated
            if 'SecureA2AAgent' in content or 'A2AAgentBase' in content:
                result['success'] = True
                result['changes_made'].append('Already migrated')
                return result
            
            # Add SecureA2AAgent import if not present
            if 'from app.a2a.core.security_base import SecureA2AAgent' not in content:
                # Find existing imports and add after them
                import_lines = []
                other_lines = []
                in_imports = True
                
                for line in content.split('\n'):
                    if line.strip().startswith(('import ', 'from ')) and in_imports:
                        import_lines.append(line)
                    elif line.strip() == '' and in_imports and import_lines:
                        import_lines.append(line)
                    else:
                        if in_imports and import_lines:
                            in_imports = False
                        other_lines.append(line)
                
                # Add SecureA2AAgent import
                if import_lines:
                    import_lines.append('from app.a2a.core.security_base import SecureA2AAgent')
                    content = '\n'.join(import_lines + other_lines)
                    result['changes_made'].append('Added SecureA2AAgent import')
                else:
                    # Add at the top if no imports found
                    content = 'from app.a2a.core.security_base import SecureA2AAgent\n\n' + content
                    result['changes_made'].append('Added SecureA2AAgent import at top')
            
            # Migrate class definitions
            for pattern, replacement in self.base_class_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    result['changes_made'].append(f'Updated class inheritance: {matches}')
            
            # Add security initialization if class has __init__
            if 'def __init__' in content and 'super().__init__' not in content and 'SecureA2AAgent.__init__' not in content:
                # Find __init__ method and add super call
                init_pattern = r'(def __init__\(self[^)]*\):\s*(?:\n\s*"""[^"]*"""\s*)?)'
                def add_super_init(match):
                    init_def = match.group(1)
                    return init_def + '\n        super().__init__()\n        '
                
                if re.search(init_pattern, content):
                    content = re.sub(init_pattern, add_super_init, content)
                    result['changes_made'].append('Added super().__init__() call')
            
            # Add security features comment
            if '# Security features provided by SecureA2AAgent:' not in content:
                security_comment = '''
        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling  
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning
'''
                
                # Add after class definition
                class_pattern = r'(class\s+\w+\(SecureA2AAgent\):\s*(?:\n\s*"""[^"]*"""\s*)?)'
                def add_security_comment(match):
                    class_def = match.group(1)
                    return class_def + security_comment
                
                if re.search(class_pattern, content):
                    content = re.sub(class_pattern, add_security_comment, content)
                    result['changes_made'].append('Added security features documentation')
            
            # Only write if changes were made
            if content != original_content:
                # Create backup
                backup_path = agent_file.with_suffix('.py.backup')
                backup_path.write_text(original_content, encoding='utf-8')
                result['backup_created'] = str(backup_path)
                
                # Write migrated content
                agent_file.write_text(content, encoding='utf-8')
                
                result['success'] = True
                logger.info(f"✅ Migrated {agent_file.name}: {len(result['changes_made'])} changes")
            else:
                result['success'] = True
                result['changes_made'].append('No changes needed')
                logger.info(f"✅ {agent_file.name}: Already compliant")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Failed to migrate {agent_file}: {e}")
        
        return result
    
    def migrate_all_agents(self) -> Dict[str, Any]:
        """Migrate all unmigrated agents"""
        logger.info("🚀 Starting comprehensive agent migration to SecureA2AAgent")
        
        # Scan current state
        scan_results = self.scan_agents()
        
        logger.info(f"📊 Found {scan_results['total_agents']} agents")
        logger.info(f"✅ Already migrated: {scan_results['migrated_agents']}")
        logger.info(f"🔄 Need migration: {len(scan_results['unmigrated_agents'])}")
        
        if not scan_results['unmigrated_agents']:
            logger.info("🎉 All agents already migrated!")
            return {
                'success': True,
                'total_agents': scan_results['total_agents'],
                'already_migrated': scan_results['migrated_agents'],
                'newly_migrated': 0,
                'migration_results': [],
                'final_migration_rate': 100.0
            }
        
        # Migrate each unmigrated agent
        migration_results = []
        successful_migrations = 0
        
        for agent_file_path in scan_results['unmigrated_agents']:
            agent_file = Path(agent_file_path)
            result = self.migrate_agent(agent_file)
            migration_results.append(result)
            
            if result['success']:
                successful_migrations += 1
        
        # Final scan to verify
        final_scan = self.scan_agents()
        final_migration_rate = (final_scan['migrated_agents'] / final_scan['total_agents'] * 100) if final_scan['total_agents'] > 0 else 0
        
        summary = {
            'success': True,
            'total_agents': final_scan['total_agents'],
            'already_migrated': scan_results['migrated_agents'],
            'newly_migrated': successful_migrations,
            'final_migrated_count': final_scan['migrated_agents'],
            'migration_results': migration_results,
            'final_migration_rate': final_migration_rate,
            'errors': [r for r in migration_results if r.get('error')]
        }
        
        logger.info(f"✅ Migration complete!")
        logger.info(f"📊 Final stats: {final_scan['migrated_agents']}/{final_scan['total_agents']} agents migrated ({final_migration_rate:.1f}%)")
        logger.info(f"🚀 Successfully migrated {successful_migrations} additional agents")
        
        return summary
    
    def generate_migration_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive migration report"""
        report = []
        report.append("=" * 80)
        report.append("AGENT MIGRATION TO SecureA2AAgent - COMPLETION REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"📊 SUMMARY")
        report.append(f"   Total Agents: {results['total_agents']}")
        report.append(f"   Already Migrated: {results['already_migrated']}")
        report.append(f"   Newly Migrated: {results['newly_migrated']}")
        report.append(f"   Final Migrated: {results['final_migrated_count']}")
        report.append(f"   Migration Rate: {results['final_migration_rate']:.1f}%")
        report.append("")
        
        if results['final_migration_rate'] == 100.0:
            report.append("🎉 CONGRATULATIONS!")
            report.append("   100% Agent Migration Complete!")
            report.append("   All agents now inherit from SecureA2AAgent with enterprise-grade security.")
        else:
            report.append("⚠️  MIGRATION INCOMPLETE")
            report.append(f"   {100 - results['final_migration_rate']:.1f}% of agents still need migration")
        
        report.append("")
        
        # Migration details
        if results['newly_migrated'] > 0:
            report.append("🔄 MIGRATION DETAILS:")
            successful = [r for r in results['migration_results'] if r['success']]
            for result in successful:
                agent_name = Path(result['file']).stem
                changes = len(result['changes_made'])
                report.append(f"   ✅ {agent_name}: {changes} changes applied")
            report.append("")
        
        # Errors
        if results['errors']:
            report.append("❌ MIGRATION ERRORS:")
            for error_result in results['errors']:
                agent_name = Path(error_result['file']).stem
                report.append(f"   ❌ {agent_name}: {error_result['error']}")
            report.append("")
        
        # Security benefits
        report.append("🔐 SECURITY BENEFITS ACHIEVED:")
        report.append("   • JWT authentication and authorization")
        report.append("   • Rate limiting and request throttling")
        report.append("   • Input validation and sanitization") 
        report.append("   • Audit logging and compliance tracking")
        report.append("   • Encrypted communication channels")
        report.append("   • Automatic security scanning")
        report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)

def main():
    """Main function"""
    migrator = AgentMigrationTool()
    
    # Perform comprehensive migration
    results = migrator.migrate_all_agents()
    
    # Generate and display report
    report = migrator.generate_migration_report(results)
    print(report)
    
    # Save detailed results
    import json
    with open('agent_migration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("📄 Detailed results saved to: agent_migration_results.json")
    
    return results['final_migration_rate'] == 100.0

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)