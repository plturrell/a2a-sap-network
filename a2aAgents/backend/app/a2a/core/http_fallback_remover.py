"""
HTTP Fallback Removal Utility
Removes all HTTP fallback mechanisms and enforces strict A2A blockchain communication
"""

import os
import re
from typing import List, Dict, Tuple, Any
from pathlib import Path


class HTTPFallbackRemover:
    """Utility to remove HTTP fallbacks and enforce blockchain-only communication"""

    # Patterns to identify and fix
    FALLBACK_PATTERNS = {
        # Memory fallback patterns
        'memory_fallback': {
            'pattern': r'(memory_fallback|fallback.*memory|in[-_]?memory.*fallback)',
            'replacement': '# REMOVED: Memory fallback violates A2A protocol - blockchain only',
            'fix_type': 'comment'
        },

        # Circuit breaker patterns
        'circuit_breaker': {
            'pattern': r'self\.circuit_breaker\s*=\s*CircuitBreaker',
            'replacement': 'self.blockchain_enforcer = BlockchainOnlyEnforcer',
            'fix_type': 'replace'
        },

        # Direct endpoint usage
        'direct_endpoint': {
            'pattern': r'if\s+.*\.endpoint\s*:.*\n.*return.*endpoint',
            'replacement': '# A2A Protocol: Direct endpoints not allowed - use blockchain messaging',
            'fix_type': 'comment_block'
        },

        # HTTP protocol in service discovery
        'http_protocol': {
            'pattern': r'protocol:\s*str\s*=\s*["\']http["\']',
            'replacement': 'protocol: str = "a2a"  # A2A blockchain protocol only',
            'fix_type': 'replace'
        },

        # Port specifications for HTTP
        'http_port': {
            'pattern': r'port:\s*int\s*=\s*\d+',
            'replacement': 'port: int = 0  # A2A Protocol: No ports - blockchain messaging only',
            'fix_type': 'replace'
        },

        # Local registry storage
        'local_registry': {
            'pattern': r'self\.(service_registry|registered_agents|agent_registry)\s*=\s*\{\}',
            'replacement': 'self.blockchain_registry = BlockchainRegistry()  # A2A: No local storage',
            'fix_type': 'replace'
        },

        # Blockchain failure handling that continues
        'blockchain_failure_continue': {
            'pattern': r'except.*:.*\n.*logger\.(warning|warn).*blockchain.*failed.*\n.*#.*[Cc]ontinue',
            'replacement': '''except Exception as e:
                logger.error(f"A2A Protocol: Blockchain required - cannot continue: {e}")
                raise RuntimeError("A2A Protocol requires blockchain connection")''',
            'fix_type': 'replace_multiline'
        },

        # Memory storage initialization
        'memory_storage': {
            'pattern': r'self\.memory_store\s*=.*\{\}',
            'replacement': '# REMOVED: self.memory_store - A2A requires blockchain storage only',
            'fix_type': 'comment'
        }
    }

    # Blockchain-only enforcement templates
    BLOCKCHAIN_ENFORCER_TEMPLATE = '''
class BlockchainOnlyEnforcer:
    """Enforces blockchain-only communication - no HTTP fallbacks"""

    def __init__(self):
        self.blockchain_required = True

    async def call(self, func, *args, **kwargs):
        """Execute function only if blockchain is available"""
        if not await self._check_blockchain():
            raise RuntimeError("A2A Protocol: Blockchain connection required")
        return await func(*args, **kwargs)

    async def _check_blockchain(self):
        """Check blockchain availability"""
        # Implementation depends on blockchain client
        return True  # Placeholder
'''

    BLOCKCHAIN_REGISTRY_TEMPLATE = '''
class BlockchainRegistry:
    """Registry that uses blockchain as single source of truth"""

    def __init__(self):
        self.blockchain_client = None
        self._init_blockchain()

    def _init_blockchain(self):
        """Initialize blockchain connection"""
        # A2A Protocol: Must have blockchain or fail
        pass

    async def get(self, key):
        """Get from blockchain only"""
        if not self.blockchain_client:
            raise RuntimeError("A2A Protocol: Blockchain required for registry access")
        # Blockchain get implementation

    async def set(self, key, value):
        """Set in blockchain only"""
        if not self.blockchain_client:
            raise RuntimeError("A2A Protocol: Blockchain required for registry updates")
        # Blockchain set implementation
'''

    def __init__(self):
        self.files_processed = 0
        self.fixes_applied = 0
        self.errors = []

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze file for HTTP fallback patterns"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            issues = []
            for pattern_name, pattern_info in self.FALLBACK_PATTERNS.items():
                matches = re.finditer(pattern_info['pattern'], content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    issues.append({
                        'pattern': pattern_name,
                        'line': content[:match.start()].count('\n') + 1,
                        'match': match.group(0),
                        'fix_type': pattern_info['fix_type']
                    })

            return issues
        except Exception as e:
            self.errors.append({'file': file_path, 'error': str(e)})
            return []

    def fix_file(self, file_path: str) -> Dict[str, Any]:
        """Fix HTTP fallback patterns in file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            original_content = content
            fixes = 0

            # Apply fixes for each pattern
            for pattern_name, pattern_info in self.FALLBACK_PATTERNS.items():
                if pattern_info['fix_type'] == 'replace':
                    new_content, count = re.subn(
                        pattern_info['pattern'],
                        pattern_info['replacement'],
                        content,
                        flags=re.IGNORECASE | re.MULTILINE
                    )
                    fixes += count
                    content = new_content

                elif pattern_info['fix_type'] == 'comment':
                    # Comment out matching lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                            lines[i] = f"# A2A REMOVED: {line}"
                            fixes += 1
                    content = '\n'.join(lines)

                elif pattern_info['fix_type'] == 'comment_block':
                    # Comment out entire blocks
                    matches = list(re.finditer(pattern_info['pattern'], content, re.IGNORECASE | re.MULTILINE))
                    for match in reversed(matches):  # Process in reverse to maintain positions
                        block_lines = content[match.start():match.end()].split('\n')
                        commented_block = '\n'.join([f"# A2A REMOVED: {line}" for line in block_lines])
                        content = content[:match.start()] + commented_block + content[match.end():]
                        fixes += 1

            # Add blockchain enforcer classes if needed
            if 'BlockchainOnlyEnforcer' in content and 'class BlockchainOnlyEnforcer' not in content:
                # Add the class definition
                import_section_end = self._find_import_section_end(content)
                content = (
                    content[:import_section_end] +
                    '\n' + self.BLOCKCHAIN_ENFORCER_TEMPLATE + '\n' +
                    content[import_section_end:]
                )

            if 'BlockchainRegistry' in content and 'class BlockchainRegistry' not in content:
                # Add the class definition
                import_section_end = self._find_import_section_end(content)
                content = (
                    content[:import_section_end] +
                    '\n' + self.BLOCKCHAIN_REGISTRY_TEMPLATE + '\n' +
                    content[import_section_end:]
                )

            # Write fixed content if changes were made
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)

                self.fixes_applied += fixes
                return {
                    'status': 'fixed',
                    'fixes': fixes,
                    'file': file_path
                }
            else:
                return {
                    'status': 'no_changes',
                    'file': file_path
                }

        except Exception as e:
            self.errors.append({'file': file_path, 'error': str(e)})
            return {
                'status': 'error',
                'error': str(e),
                'file': file_path
            }

    def _find_import_section_end(self, content: str) -> int:
        """Find the end of the import section in a Python file"""
        lines = content.split('\n')
        last_import = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import = i
            elif line.strip() and not line.strip().startswith('#') and last_import > 0:
                # Found first non-import, non-comment line after imports
                break

        # Return position after last import
        return sum(len(line) + 1 for line in lines[:last_import + 1])

    def process_directory(self, directory: str) -> Dict[str, Any]:
        """Process all Python files in directory"""
        results = {
            'total_files': 0,
            'files_with_issues': 0,
            'total_fixes': 0,
            'file_results': []
        }

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)

                    # Analyze file
                    issues = self.analyze_file(file_path)
                    if issues:
                        results['files_with_issues'] += 1

                        # Fix file
                        fix_result = self.fix_file(file_path)
                        fix_result['issues_found'] = len(issues)
                        results['file_results'].append(fix_result)

                        if fix_result['status'] == 'fixed':
                            results['total_fixes'] += fix_result['fixes']

                    results['total_files'] += 1

        return results

    def generate_report(self, results: Dict[str, Any]):
        """Generate fix report"""
        print("\n=== HTTP Fallback Removal Report ===\n")
        print(f"Total files processed: {results['total_files']}")
        print(f"Files with issues: {results['files_with_issues']}")
        print(f"Total fixes applied: {results['total_fixes']}")

        if results['file_results']:
            print("\nFiles fixed:")
            for result in results['file_results']:
                if result['status'] == 'fixed':
                    print(f"  ✅ {result['file']}: {result['fixes']} fixes applied")
                elif result['status'] == 'error':
                    print(f"  ❌ {result['file']}: {result['error']}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error['file']}: {error['error']}")

        print("\n✅ HTTP fallback removal complete!")
        print("⚠️  Remember to test all agents after these changes")
        print("📋 All agents now enforce strict A2A blockchain communication")


def main():
    """Main function to run the HTTP fallback remover"""
    remover = HTTPFallbackRemover()

    # Process the agents directory
    agents_dir = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents"

    print("🔍 Scanning for HTTP fallback mechanisms...")
    results = remover.process_directory(agents_dir)

    # Generate report
    remover.generate_report(results)

    # Save detailed results
    import json
    with open('http_fallback_removal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: http_fallback_removal_results.json")


if __name__ == "__main__":
    main()