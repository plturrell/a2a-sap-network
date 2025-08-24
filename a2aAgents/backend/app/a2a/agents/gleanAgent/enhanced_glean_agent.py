"""
Enhanced Glean Agent with Error Handling, Logging, and Performance Tracking
Extends the base Glean Agent with enterprise-grade capabilities
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import aiohttp
import time

from .base_agent import BaseAgent, A2AError, ErrorCode, track_performance
from .gleanAgentSdk import GleanAgentSDK, AnalysisResult, SecurityScanResult, RefactoringResult
from .intelligentScanManager import IntelligentScanManager
from app.a2a.core.security_base import SecureA2AAgent


class EnhancedGleanAgent(SecureA2AAgent):
    """Enhanced Glean Agent with comprehensive error handling and monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize base agent
        super().__init__('glean-agent', config)
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        
        # Initialize original Glean SDK
        self.glean_sdk = GleanAgentSDK()
        
        # Initialize intelligent scan manager
        self.scan_manager = IntelligentScanManager()
        
        # Configure performance thresholds
        self.performance_thresholds = {
            'semantic_analysis': 5000,      # 5 seconds
            'lint_file': 2000,              # 2 seconds
            'security_scan': 10000,         # 10 seconds
            'test_execution': 30000,        # 30 seconds
            'refactoring_analysis': 3000,   # 3 seconds
            'dependency_analysis': 5000     # 5 seconds
        }
        
        # Initialize metrics
        self.scan_metrics = {
            'total_scans': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'security_issues_found': 0,
            'performance_issues_found': 0
        }
        
        self.logger.info('Enhanced Glean Agent initialized')
    
    @track_performance('semantic_analysis')
    async def analyze_code_semantic(self, 
                                  file_path: str, 
                                  context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze code semantics with enhanced error handling"""
        try:
            # Validate input
            self.validate_input({'file_path': file_path}, {'required': ['file_path']})
            
            if not os.path.exists(file_path):
                raise A2AError(
                    code=ErrorCode.DATA_NOT_FOUND,
                    message=f'File not found: {file_path}',
                    details={'file_path': file_path}
                )
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise A2AError(
                    code=ErrorCode.DATA_SIZE_LIMIT_EXCEEDED,
                    message='File size exceeds limit',
                    details={'file_size': file_size, 'limit': 10 * 1024 * 1024}
                )
            
            # Log analysis start
            self.logger.info(
                'Starting semantic analysis',
                file_path=file_path,
                file_size=file_size,
                context=context
            )
            
            # Perform analysis with timeout
            try:
                result = await asyncio.wait_for(
                    self.glean_sdk.analyze_code_semantic(file_path, context),
                    timeout=self.performance_thresholds['semantic_analysis'] / 1000
                )
                
                # Validate result
                if not result or not hasattr(result, 'symbols'):
                    raise A2AError(
                        code=ErrorCode.DATA_QUALITY_CHECK_FAILED,
                        message='Invalid analysis result',
                        details={'result_type': type(result).__name__}
                    )
                
                # Update metrics
                self.scan_metrics['total_scans'] += 1
                self.scan_metrics['successful_scans'] += 1
                
                # Log success
                self.logger.info(
                    'Semantic analysis completed',
                    file_path=file_path,
                    symbols_found=len(result.symbols) if hasattr(result, 'symbols') else 0,
                    issues_found=len(result.issues) if hasattr(result, 'issues') else 0
                )
                
                return result
                
            except asyncio.TimeoutError:
                raise A2AError(
                    code=ErrorCode.AGENT_TIMEOUT,
                    message='Semantic analysis timeout',
                    details={
                        'file_path': file_path,
                        'timeout': self.performance_thresholds['semantic_analysis']
                    }
                )
                
        except A2AError:
            self.scan_metrics['failed_scans'] += 1
            raise
        except Exception as e:
            self.scan_metrics['failed_scans'] += 1
            raise self.handle_error(e, 'semantic_analysis')
    
    @track_performance('security_scan')
    async def run_security_scan(self, 
                              directory: str,
                              scan_options: Optional[Dict[str, Any]] = None) -> List[SecurityScanResult]:
        """Run comprehensive security scan with monitoring"""
        try:
            # Validate input
            self.validate_input({'directory': directory}, {'required': ['directory']})
            
            if not os.path.isdir(directory):
                raise A2AError(
                    code=ErrorCode.DATA_NOT_FOUND,
                    message=f'Directory not found: {directory}',
                    details={'directory': directory}
                )
            
            # Use intelligent scan manager
            scan_config = self.scan_manager.optimize_scan_schedule({
                'directory': directory,
                'type': 'security',
                'options': scan_options or {}
            })
            
            self.logger.info(
                'Starting security scan',
                directory=directory,
                scan_config=scan_config
            )
            
            # Track scan in manager
            scan_id = self.scan_manager.track_scan({
                'type': 'security',
                'target': directory,
                'started_at': datetime.utcnow(),
                'config': scan_config
            })
            
            # Run scan with timeout
            try:
                results = await asyncio.wait_for(
                    self.glean_sdk.run_security_scan(directory, scan_options),
                    timeout=self.performance_thresholds['security_scan'] / 1000
                )
                
                # Process results
                security_issues = 0
                critical_issues = 0
                
                for result in results:
                    if hasattr(result, 'vulnerabilities'):
                        security_issues += len(result.vulnerabilities)
                        critical_issues += sum(
                            1 for v in result.vulnerabilities 
                            if getattr(v, 'severity', '').lower() == 'critical'
                        )
                
                # Update scan tracking
                self.scan_manager.complete_scan(scan_id, {
                    'completed_at': datetime.utcnow(),
                    'status': 'success',
                    'issues_found': security_issues,
                    'critical_issues': critical_issues
                })
                
                # Update metrics
                self.scan_metrics['security_issues_found'] += security_issues
                
                # Log completion
                self.logger.info(
                    'Security scan completed',
                    directory=directory,
                    files_scanned=len(results),
                    total_issues=security_issues,
                    critical_issues=critical_issues
                )
                
                # Alert on critical issues
                if critical_issues > 0:
                    self.logger.warning(
                        'Critical security issues found',
                        count=critical_issues,
                        directory=directory
                    )
                
                return results
                
            except asyncio.TimeoutError:
                self.scan_manager.complete_scan(scan_id, {
                    'completed_at': datetime.utcnow(),
                    'status': 'timeout'
                })
                raise A2AError(
                    code=ErrorCode.AGENT_TIMEOUT,
                    message='Security scan timeout',
                    details={
                        'directory': directory,
                        'timeout': self.performance_thresholds['security_scan']
                    }
                )
                
        except A2AError:
            raise
        except Exception as e:
            raise self.handle_error(e, 'security_scan')
    
    @track_performance('lint_file')
    async def lint_file_with_tracking(self,
                                    file_path: str,
                                    linters: Optional[List[str]] = None) -> Dict[str, Any]:
        """Lint file with performance tracking and error handling"""
        try:
            # Validate input
            self.validate_input({'file_path': file_path}, {'required': ['file_path']})
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise A2AError(
                    code=ErrorCode.DATA_NOT_FOUND,
                    message=f'File not found: {file_path}',
                    details={'file_path': file_path}
                )
            
            self.logger.debug(
                'Starting file linting',
                file_path=file_path,
                linters=linters
            )
            
            # Run linting
            result = await self.glean_sdk.lint_file(file_path, linters)
            
            # Count issues
            total_issues = 0
            error_count = 0
            warning_count = 0
            
            if isinstance(result, dict):
                for linter_name, linter_result in result.items():
                    if isinstance(linter_result, dict) and 'issues' in linter_result:
                        issues = linter_result['issues']
                        total_issues += len(issues)
                        
                        for issue in issues:
                            severity = issue.get('severity', '').lower()
                            if severity == 'error':
                                error_count += 1
                            elif severity == 'warning':
                                warning_count += 1
            
            # Log results
            self.logger.info(
                'Linting completed',
                file_path=file_path,
                total_issues=total_issues,
                errors=error_count,
                warnings=warning_count
            )
            
            # Check quality threshold
            if error_count > 0:
                self.logger.warning(
                    'Linting errors found',
                    file_path=file_path,
                    error_count=error_count
                )
            
            return result
            
        except A2AError:
            raise
        except Exception as e:
            raise self.handle_error(e, 'lint_file')
    
    @track_performance('batch_analysis')
    async def analyze_directory_batch(self,
                                    directory: str,
                                    file_patterns: Optional[List[str]] = None,
                                    analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze directory with batch processing and progress tracking"""
        try:
            # Validate input
            self.validate_input({'directory': directory}, {'required': ['directory']})
            
            if not os.path.isdir(directory):
                raise A2AError(
                    code=ErrorCode.DATA_NOT_FOUND,
                    message=f'Directory not found: {directory}',
                    details={'directory': directory}
                )
            
            # Default analysis types
            if not analysis_types:
                analysis_types = ['semantic', 'lint', 'security']
            
            self.logger.info(
                'Starting batch directory analysis',
                directory=directory,
                file_patterns=file_patterns,
                analysis_types=analysis_types
            )
            
            # Collect files
            files_to_analyze = []
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check patterns
                    if file_patterns:
                        if not any(file.endswith(pattern) for pattern in file_patterns):
                            continue
                    
                    files_to_analyze.append(file_path)
            
            if not files_to_analyze:
                self.logger.warning(
                    'No files found to analyze',
                    directory=directory,
                    patterns=file_patterns
                )
                return {'files_analyzed': 0, 'results': {}}
            
            self.logger.info(
                f'Found {len(files_to_analyze)} files to analyze',
                file_count=len(files_to_analyze)
            )
            
            # Process files in batches
            batch_size = 10
            results = {}
            errors = []
            
            for i in range(0, len(files_to_analyze), batch_size):
                batch = files_to_analyze[i:i + batch_size]
                
                # Log progress
                progress = (i + len(batch)) / len(files_to_analyze) * 100
                self.logger.info(
                    f'Processing batch {i//batch_size + 1}',
                    progress=f'{progress:.1f}%',
                    batch_size=len(batch)
                )
                
                # Process batch concurrently
                batch_tasks = []
                for file_path in batch:
                    if 'semantic' in analysis_types:
                        batch_tasks.append(self.analyze_code_semantic(file_path))
                    if 'lint' in analysis_types:
                        batch_tasks.append(self.lint_file_with_tracking(file_path))
                    if 'security' in analysis_types and file_path.endswith('.py'):
                        batch_tasks.append(self._scan_file_security(file_path))
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for idx, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        errors.append({
                            'file': batch[idx % len(batch)],
                            'error': str(result)
                        })
                    else:
                        file_idx = idx % len(batch)
                        if batch[file_idx] not in results:
                            results[batch[file_idx]] = {}
                        
                        analysis_type = analysis_types[idx // len(batch)]
                        results[batch[file_idx]][analysis_type] = result
            
            # Generate summary
            summary = {
                'files_analyzed': len(files_to_analyze),
                'successful': len(results),
                'failed': len(errors),
                'results': results,
                'errors': errors,
                'metrics': self.performance_tracker.get_metrics()
            }
            
            self.logger.info(
                'Batch analysis completed',
                summary=summary
            )
            
            return summary
            
        except A2AError:
            raise
        except Exception as e:
            raise self.handle_error(e, 'batch_analysis')
    
    async def _scan_file_security(self, file_path: str) -> Dict[str, Any]:
        """Helper method to scan individual file for security issues"""
        try:
            result = await self.glean_sdk.run_security_scan(
                os.path.dirname(file_path),
                {'include_patterns': [os.path.basename(file_path)]}
            )
            return result[0] if result else {}
        except Exception as e:
            self.logger.error(
                'Failed to scan file for security',
                file_path=file_path,
                error=str(e)
            )
            return {'error': str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        base_status = self.get_health_status()
        
        # Add Glean-specific metrics
        base_status.update({
            'scan_metrics': self.scan_metrics,
            'scan_history': self.scan_manager.get_scan_history(limit=10),
            'performance_thresholds': self.performance_thresholds,
            'active_scans': len(self.scan_manager.active_scans)
        })
        
        return base_status
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check Glean service connectivity
        try:
            # Simple connectivity test
            test_result = await self.glean_sdk.analyze_code_semantic(__file__)
            health_status['checks']['glean_service'] = {
                'status': 'healthy',
                'message': 'Glean service is accessible'
            }
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['glean_service'] = {
                'status': 'unhealthy',
                'message': f'Glean service error: {str(e)}'
            }
        
        # Check scan manager
        try:
            scan_stats = self.scan_manager.get_analytics()
            health_status['checks']['scan_manager'] = {
                'status': 'healthy',
                'stats': scan_stats
            }
        except Exception as e:
            health_status['checks']['scan_manager'] = {
                'status': 'unhealthy',
                'message': f'Scan manager error: {str(e)}'
            }
        
        # Check performance metrics
        metrics = self.performance_tracker.get_metrics()
        slow_operations = []
        
        for op_name, op_metrics in metrics.items():
            if op_name in self.performance_thresholds:
                threshold = self.performance_thresholds[op_name]
                if op_metrics.get('p95', 0) > threshold:
                    slow_operations.append({
                        'operation': op_name,
                        'p95': op_metrics['p95'],
                        'threshold': threshold
                    })
        
        if slow_operations:
            health_status['checks']['performance'] = {
                'status': 'warning',
                'slow_operations': slow_operations
            }
        else:
            health_status['checks']['performance'] = {
                'status': 'healthy',
                'message': 'All operations within thresholds'
            }
        
        return health_status


# Factory function
def create_enhanced_glean_agent(config: Optional[Dict[str, Any]] = None) -> EnhancedGleanAgent:
    """Factory function to create enhanced Glean agent"""
    return EnhancedGleanAgent(config)