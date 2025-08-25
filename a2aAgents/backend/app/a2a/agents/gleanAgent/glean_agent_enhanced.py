"""
Main Enhanced Glean Agent Integration
Brings together all capabilities: error handling, logging, performance tracking, security scanning, and monitoring
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import traceback

from .enhanced_glean_agent import EnhancedGleanAgent
from .security_scanner import SecurityScanner, SecurityScanResult
from .monitoring_dashboard import MonitoringDashboard
from .base_agent import BaseAgent, A2AError, ErrorCode, ErrorSeverity
from app.a2a.core.security_base import SecureA2AAgent


class GleanAgentEnhanced(SecureA2AAgent):
    """
    Comprehensive Enhanced Glean Agent with all enterprise capabilities:
    - Structured logging with correlation IDs
    - Performance tracking and metrics
    - Security vulnerability scanning
    - Error handling with categorization
    - Real-time monitoring dashboard
    - Health checks and alerting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('glean-agent-enhanced', config)
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()


        # Initialize components
        self.glean_agent = EnhancedGleanAgent(config)
        self.security_scanner = SecurityScanner(config)
        self.monitoring_dashboard = MonitoringDashboard(config)

        # Service registry
        self.services = {
            'glean': self.glean_agent,
            'security': self.security_scanner,
            'monitoring': self.monitoring_dashboard
        }

        # Global configuration
        self.config = {
            'enable_auto_monitoring': True,
            'enable_security_alerts': True,
            'performance_alert_threshold': 5000,  # ms
            'max_concurrent_scans': 5,
            **config
        } if config else {
            'enable_auto_monitoring': True,
            'enable_security_alerts': True,
            'performance_alert_threshold': 5000,
            'max_concurrent_scans': 5
        }

        # Semaphore for concurrent operations
        self.scan_semaphore = asyncio.Semaphore(self.config['max_concurrent_scans'])

        # Track active operations
        self.active_operations = {}

        self.logger.info('Comprehensive Enhanced Glean Agent initialized',
                        config=self.config)

    @BaseAgent.track_performance('comprehensive_analysis')
    async def analyze_codebase_comprehensive(self,
                                           directory: str,
                                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive codebase analysis including:
        - Semantic analysis
        - Security scanning
        - Performance profiling
        - Quality metrics
        """
        correlation_id = self.correlation_id or str(uuid.uuid4())

        try:
            # Validate input
            self.validate_input({'directory': directory}, {'required': ['directory']})

            analysis_options = {
                'include_semantic': True,
                'include_security': True,
                'include_performance': True,
                'include_quality': True,
                'file_patterns': ['*.py', '*.js', '*.ts', '*.java', '*.php'],
                **(options or {})
            }

            self.logger.info(
                'Starting comprehensive codebase analysis',
                directory=directory,
                correlation_id=correlation_id,
                options=analysis_options
            )

            # Record start metrics
            start_time = time.time()
            self.monitoring_dashboard.record_metric(
                'analysis.started',
                1,
                'count',
                {'type': 'comprehensive', 'directory': directory}
            )

            # Initialize results
            results = {
                'directory': directory,
                'correlation_id': correlation_id,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'components': {}
            }

            # Track operation
            self.active_operations[correlation_id] = {
                'type': 'comprehensive_analysis',
                'directory': directory,
                'started_at': start_time,
                'status': 'running'
            }

            # Run analysis components in parallel
            tasks = []

            if analysis_options['include_semantic']:
                tasks.append(self._run_semantic_analysis(directory, correlation_id))

            if analysis_options['include_security']:
                tasks.append(self._run_security_analysis(directory, correlation_id))

            if analysis_options['include_performance']:
                tasks.append(self._run_performance_analysis(directory, correlation_id))

            if analysis_options['include_quality']:
                tasks.append(self._run_quality_analysis(directory, correlation_id))

            # Wait for all components to complete
            component_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            component_names = []
            if analysis_options['include_semantic']:
                component_names.append('semantic')
            if analysis_options['include_security']:
                component_names.append('security')
            if analysis_options['include_performance']:
                component_names.append('performance')
            if analysis_options['include_quality']:
                component_names.append('quality')

            for i, result in enumerate(component_results):
                component_name = component_names[i]

                if isinstance(result, Exception):
                    self.logger.error(
                        f'{component_name} analysis failed',
                        error=str(result),
                        correlation_id=correlation_id
                    )
                    results['components'][component_name] = {
                        'status': 'failed',
                        'error': str(result)
                    }

                    # Create alert for failed component
                    if self.config['enable_auto_monitoring']:
                        self.monitoring_dashboard.create_alert(
                            f'{component_name}_analysis_failed',
                            ErrorSeverity.HIGH,
                            f'{component_name} analysis failed for {directory}',
                            {'directory': directory, 'error': str(result)}
                        )
                else:
                    results['components'][component_name] = {
                        'status': 'completed',
                        'data': result
                    }

            # Calculate final metrics
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to ms

            results.update({
                'completed_at': datetime.utcnow().isoformat(),
                'duration_ms': duration,
                'status': 'completed'
            })

            # Record completion metrics
            self.monitoring_dashboard.record_operation_result(
                'comprehensive_analysis',
                duration,
                True,
                {'directory': directory}
            )

            # Check for performance alerts
            if duration > self.config['performance_alert_threshold']:
                self.monitoring_dashboard.create_alert(
                    'slow_analysis',
                    ErrorSeverity.MEDIUM,
                    f'Comprehensive analysis took {duration:.0f}ms (threshold: {self.config["performance_alert_threshold"]}ms)',
                    {'duration': duration, 'directory': directory}
                )

            # Clean up operation tracking
            if correlation_id in self.active_operations:
                del self.active_operations[correlation_id]

            self.logger.info(
                'Comprehensive analysis completed',
                correlation_id=correlation_id,
                duration_ms=duration,
                components_completed=len([c for c in results['components'].values() if c['status'] == 'completed']),
                components_failed=len([c for c in results['components'].values() if c['status'] == 'failed'])
            )

            return results

        except A2AError:
            # Clean up on error
            if correlation_id in self.active_operations:
                del self.active_operations[correlation_id]
            raise
        except Exception as e:
            # Clean up on error
            if correlation_id in self.active_operations:
                del self.active_operations[correlation_id]
            raise self.handle_error(e, 'comprehensive_analysis')

    async def _run_semantic_analysis(self, directory: str, correlation_id: str) -> Dict[str, Any]:
        """Run semantic analysis component"""
        async with self.scan_semaphore:
            try:
                result = await self.glean_agent.analyze_directory_batch(
                    directory,
                    file_patterns=['*.py', '*.js', '*.ts'],
                    analysis_types=['semantic']
                )

                return {
                    'files_analyzed': result.get('files_analyzed', 0),
                    'successful_analyses': result.get('successful', 0),
                    'failed_analyses': result.get('failed', 0),
                    'symbols_found': self._count_symbols_in_results(result.get('results', {}))
                }

            except Exception as e:
                self.logger.error(
                    'Semantic analysis component failed',
                    directory=directory,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    async def _run_security_analysis(self, directory: str, correlation_id: str) -> Dict[str, Any]:
        """Run security analysis component"""
        async with self.scan_semaphore:
            try:
                # Get all Python files in directory
                import os
                files_to_scan = []
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith(('.py', '.js', '.ts', '.php', '.java')):
                            files_to_scan.append(os.path.join(root, file))

                # Scan files concurrently (in smaller batches)
                scan_results = []
                batch_size = 10

                for i in range(0, len(files_to_scan), batch_size):
                    batch = files_to_scan[i:i + batch_size]
                    batch_tasks = [self.security_scanner.scan_file(f) for f in batch]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    for result in batch_results:
                        if not isinstance(result, Exception):
                            scan_results.append(result)

                # Aggregate results
                total_vulnerabilities = sum(len(r.vulnerabilities) for r in scan_results)
                critical_vulns = sum(len([v for v in r.vulnerabilities
                                        if v.severity.value == 'critical']) for r in scan_results)
                high_vulns = sum(len([v for v in r.vulnerabilities
                                    if v.severity.value == 'high']) for r in scan_results)

                # Create security alert if critical vulnerabilities found
                if critical_vulns > 0 and self.config['enable_security_alerts']:
                    self.monitoring_dashboard.create_alert(
                        'critical_vulnerabilities',
                        ErrorSeverity.CRITICAL,
                        f'Found {critical_vulns} critical security vulnerabilities in {directory}',
                        {
                            'directory': directory,
                            'critical_count': critical_vulns,
                            'total_vulnerabilities': total_vulnerabilities
                        }
                    )

                return {
                    'files_scanned': len(scan_results),
                    'total_vulnerabilities': total_vulnerabilities,
                    'critical_vulnerabilities': critical_vulns,
                    'high_vulnerabilities': high_vulns,
                    'scan_results': [r.to_dict() for r in scan_results[:10]]  # Limit for response size
                }

            except Exception as e:
                self.logger.error(
                    'Security analysis component failed',
                    directory=directory,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    async def _run_performance_analysis(self, directory: str, correlation_id: str) -> Dict[str, Any]:
        """Run performance analysis component"""
        async with self.scan_semaphore:
            try:
                # Get performance metrics from monitoring dashboard
                metrics = self.monitoring_dashboard.get_dashboard_data()

                # Analyze recent operation performance
                operation_metrics = metrics.get('operations', {})
                slow_operations = []

                for op_name, stats in operation_metrics.items():
                    if stats['avg_response_time'] > 1000:  # Operations slower than 1s
                        slow_operations.append({
                            'operation': op_name,
                            'avg_response_time': stats['avg_response_time'],
                            'success_rate': stats['success_rate']
                        })

                return {
                    'total_operations': sum(stats['total_operations'] for stats in operation_metrics.values()),
                    'overall_success_rate': metrics.get('performance_summary', {}).get('overall_success_rate', 0),
                    'average_response_time': metrics.get('performance_summary', {}).get('average_response_time', 0),
                    'slow_operations': slow_operations,
                    'system_health': metrics.get('health_checks', {})
                }

            except Exception as e:
                self.logger.error(
                    'Performance analysis component failed',
                    directory=directory,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    async def _run_quality_analysis(self, directory: str, correlation_id: str) -> Dict[str, Any]:
        """Run code quality analysis component"""
        async with self.scan_semaphore:
            try:
                # Use enhanced Glean agent for linting
                result = await self.glean_agent.analyze_directory_batch(
                    directory,
                    file_patterns=['*.py', '*.js', '*.ts'],
                    analysis_types=['lint']
                )

                # Count quality issues
                total_issues = 0
                error_count = 0
                warning_count = 0

                for file_result in result.get('results', {}).values():
                    if 'lint' in file_result and isinstance(file_result['lint'], dict):
                        for linter_result in file_result['lint'].values():
                            if isinstance(linter_result, dict) and 'issues' in linter_result:
                                issues = linter_result['issues']
                                total_issues += len(issues)

                                for issue in issues:
                                    severity = issue.get('severity', '').lower()
                                    if severity == 'error':
                                        error_count += 1
                                    elif severity == 'warning':
                                        warning_count += 1

                return {
                    'files_analyzed': result.get('files_analyzed', 0),
                    'total_issues': total_issues,
                    'errors': error_count,
                    'warnings': warning_count,
                    'quality_score': self._calculate_quality_score(total_issues, error_count, result.get('files_analyzed', 1))
                }

            except Exception as e:
                self.logger.error(
                    'Quality analysis component failed',
                    directory=directory,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    def _count_symbols_in_results(self, results: Dict[str, Any]) -> int:
        """Count symbols found in semantic analysis results"""
        total_symbols = 0
        for file_result in results.values():
            if isinstance(file_result, dict) and 'semantic' in file_result:
                semantic_result = file_result['semantic']
                if hasattr(semantic_result, 'symbols'):
                    total_symbols += len(semantic_result.symbols)
        return total_symbols

    def _calculate_quality_score(self, total_issues: int, error_count: int, files_count: int) -> float:
        """Calculate a quality score from 0-100"""
        if files_count == 0:
            return 100.0

        # Base score
        score = 100.0

        # Penalize errors more than warnings
        issues_per_file = total_issues / files_count
        errors_per_file = error_count / files_count

        # Subtract points for issues
        score -= (issues_per_file * 5)  # 5 points per issue per file
        score -= (errors_per_file * 10)  # Additional 10 points per error per file

        return max(0.0, min(100.0, score))

    async def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time status of all components"""
        try:
            # Get status from all components
            dashboard_data = self.monitoring_dashboard.get_dashboard_data()
            agent_status = self.glean_agent.get_agent_status()
            scanner_stats = self.security_scanner.get_scan_statistics()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': dashboard_data['system_status']['status'],
                'active_operations': len(self.active_operations),
                'components': {
                    'glean_agent': {
                        'status': 'healthy' if agent_status['status'] == 'healthy' else 'unhealthy',
                        'scan_metrics': agent_status.get('scan_metrics', {}),
                        'performance_metrics': agent_status.get('performance_metrics', {})
                    },
                    'security_scanner': {
                        'status': 'healthy',
                        'stats': scanner_stats,
                        'external_tools': scanner_stats.get('external_tools_available', {})
                    },
                    'monitoring_dashboard': {
                        'status': 'healthy',
                        'active_alerts': dashboard_data['alerts']['active_count'],
                        'total_alerts': dashboard_data['alerts']['total_count']
                    }
                },
                'system_metrics': dashboard_data['health_checks'],
                'active_alerts': dashboard_data['alerts']['active']
            }

        except Exception as e:
            self.logger.error('Failed to get real-time status', error=str(e))
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check across all components"""
        try:
            health_results = {}

            # Check each component
            health_results['glean_agent'] = await self.glean_agent.run_health_check()
            health_results['monitoring_dashboard'] = self.monitoring_dashboard._run_health_checks()

            # Security scanner health
            scanner_stats = self.security_scanner.get_scan_statistics()
            health_results['security_scanner'] = {
                'status': 'healthy',
                'external_tools': scanner_stats.get('external_tools_available', {}),
                'patterns_loaded': scanner_stats.get('patterns_loaded', 0)
            }

            # Overall health determination
            component_statuses = []
            for component, health in health_results.items():
                if isinstance(health, dict) and 'status' in health:
                    component_statuses.append(health['status'])

            if 'critical' in component_statuses:
                overall_status = 'critical'
            elif 'unhealthy' in component_statuses or 'warning' in component_statuses:
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': overall_status,
                'components': health_results,
                'active_operations_count': len(self.active_operations)
            }

        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }


# Factory function
def create_enhanced_glean_agent(config: Optional[Dict[str, Any]] = None) -> GleanAgentEnhanced:
    """Create a fully enhanced Glean Agent with all capabilities"""
    return GleanAgentEnhanced(config)


# Quick start function
async def quick_analyze(directory: str,
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick analysis function for immediate use"""
    agent = create_enhanced_glean_agent()

    with agent:  # Use context manager for proper cleanup
        return await agent.analyze_codebase_comprehensive(directory, options)
