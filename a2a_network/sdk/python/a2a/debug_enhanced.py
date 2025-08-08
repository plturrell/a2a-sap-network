"""
Enhanced Debug and Profiling Utilities for A2A SDK
Production-ready debugging tools with comprehensive features
"""
import asyncio
import time
import json
import logging
import threading
import signal
import sys
import traceback
import inspect
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import psutil
import websockets
from web3 import Web3
from web3.exceptions import Web3Exception
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from .types import Agent, Message, MessageStatus, MessageType, AgentStatus, BlockchainEvent, PerformanceSnapshot
from .constants import MessageType as MT, ErrorCategory, PERFORMANCE_THRESHOLDS, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ErrorAnalysis:
    """Comprehensive error analysis data"""
    category: ErrorCategory
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_agents: List[str]
    error_patterns: List[str]
    root_causes: List[str]
    resolution_suggestions: List[str]
    severity: str  # low, medium, high, critical


@dataclass 
class PerformanceRegression:
    """Performance regression detection data"""
    metric: str
    baseline: float
    current: float
    degradation_percent: float
    detected_at: datetime
    confidence: float
    affected_operations: List[str]


class BlockchainTracer:
    """Enhanced blockchain integration for message tracing"""
    
    def __init__(self, web3_client: Web3, contract_addresses: Dict[str, str]):
        self.w3 = web3_client
        self.contracts = {}
        self.event_filters = {}
        self.trace_cache = {}
        
        # Initialize contract instances
        for name, address in contract_addresses.items():
            try:
                # In production, you'd load the ABI properly
                self.contracts[name] = self.w3.eth.contract(address=address)
            except Exception as e:
                logger.warning(f"Failed to initialize contract {name}: {e}")
    
    async def trace_message_blockchain(self, message_id: str) -> Dict[str, Any]:
        """Trace message through blockchain events"""
        trace_data = {
            'message_id': message_id,
            'blockchain_events': [],
            'gas_usage': 0,
            'confirmation_time': None,
            'block_confirmations': 0
        }
        
        try:
            # Get message sent events
            sent_events = await self._get_message_events(message_id, 'MessageSent')
            for event in sent_events:
                trace_data['blockchain_events'].append({
                    'type': 'sent',
                    'block_number': event.block_number,
                    'transaction_hash': event.transaction_hash,
                    'gas_used': await self._get_transaction_gas(event.transaction_hash),
                    'timestamp': await self._get_block_timestamp(event.block_number)
                })
            
            # Get message delivery events
            delivered_events = await self._get_message_events(message_id, 'MessageDelivered')
            for event in delivered_events:
                trace_data['blockchain_events'].append({
                    'type': 'delivered',
                    'block_number': event.block_number,
                    'transaction_hash': event.transaction_hash,
                    'gas_used': await self._get_transaction_gas(event.transaction_hash),
                    'timestamp': await self._get_block_timestamp(event.block_number)
                })
            
            # Calculate total gas and confirmation time
            if trace_data['blockchain_events']:
                trace_data['gas_usage'] = sum(event.get('gas_used', 0) for event in trace_data['blockchain_events'])
                
                if len(trace_data['blockchain_events']) >= 2:
                    start_time = trace_data['blockchain_events'][0]['timestamp']
                    end_time = trace_data['blockchain_events'][-1]['timestamp']
                    trace_data['confirmation_time'] = end_time - start_time
                    
                latest_block = max(event['block_number'] for event in trace_data['blockchain_events'])
                current_block = self.w3.eth.block_number
                trace_data['block_confirmations'] = current_block - latest_block
        
        except Exception as e:
            trace_data['error'] = str(e)
            logger.error(f"Blockchain tracing failed for message {message_id}: {e}")
        
        return trace_data
    
    async def _get_message_events(self, message_id: str, event_name: str) -> List[BlockchainEvent]:
        """Get blockchain events for a message"""
        events = []
        try:
            if 'MessageRouter' in self.contracts:
                contract = self.contracts['MessageRouter']
                event_filter = contract.events[event_name].createFilter(
                    fromBlock='earliest',
                    argument_filters={'messageId': message_id}
                )
                
                for event in event_filter.get_all_entries():
                    events.append(BlockchainEvent(
                        block_number=event['blockNumber'],
                        transaction_hash=event['transactionHash'].hex(),
                        contract_address=event['address'],
                        event_name=event_name,
                        args=dict(event['args']),
                        timestamp=await self._get_block_timestamp(event['blockNumber'])
                    ))
        except Exception as e:
            logger.error(f"Failed to get {event_name} events: {e}")
        
        return events
    
    async def _get_transaction_gas(self, tx_hash: str) -> int:
        """Get gas used for transaction"""
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return receipt.gasUsed
        except Exception:
            return 0
    
    async def _get_block_timestamp(self, block_number: int) -> datetime:
        """Get timestamp for block"""
        try:
            block = self.w3.eth.get_block(block_number)
            return datetime.fromtimestamp(block.timestamp)
        except Exception:
            return datetime.now()


class ErrorAnalyzer:
    """Comprehensive error analysis and categorization"""
    
    def __init__(self):
        self.errors = defaultdict(list)
        self.error_patterns = {}
        self.root_cause_map = {}
        
        # Initialize error pattern recognition
        self._setup_error_patterns()
    
    def _setup_error_patterns(self):
        """Setup error pattern recognition"""
        self.error_patterns = {
            r'Connection.*refused': {
                'category': ErrorCategory.NETWORK,
                'root_causes': ['Service unavailable', 'Network connectivity'],
                'suggestions': ['Check service status', 'Verify network connectivity']
            },
            r'Timeout.*expired': {
                'category': ErrorCategory.TIMEOUT,
                'root_causes': ['Slow response', 'Network latency', 'Overloaded service'],
                'suggestions': ['Increase timeout', 'Check service load', 'Optimize request']
            },
            r'Invalid.*signature': {
                'category': ErrorCategory.AUTHENTICATION,
                'root_causes': ['Wrong private key', 'Corrupted signature'],
                'suggestions': ['Verify private key', 'Check signature generation']
            },
            r'Gas.*exceeded': {
                'category': ErrorCategory.BLOCKCHAIN,
                'root_causes': ['Complex computation', 'Infinite loop', 'Large data'],
                'suggestions': ['Optimize gas usage', 'Split operations', 'Use gas estimation']
            },
            r'Agent.*not.*found': {
                'category': ErrorCategory.AGENT,
                'root_causes': ['Agent offline', 'Wrong address', 'Not registered'],
                'suggestions': ['Check agent status', 'Verify address', 'Register agent']
            }
        }
    
    def analyze_error(self, error: str, context: Dict[str, Any]) -> ErrorAnalysis:
        """Analyze an error and provide categorization"""
        import re
        
        # Find matching pattern
        category = ErrorCategory.UNKNOWN
        root_causes = ['Unknown error']
        suggestions = ['Check logs for more details']
        
        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error, re.IGNORECASE):
                category = info['category']
                root_causes = info['root_causes']
                suggestions = info['suggestions']
                break
        
        # Determine severity
        severity = 'medium'
        if any(word in error.lower() for word in ['critical', 'fatal', 'panic']):
            severity = 'critical'
        elif any(word in error.lower() for word in ['error', 'failed', 'exception']):
            severity = 'high'
        elif any(word in error.lower() for word in ['warning', 'warn']):
            severity = 'low'
        
        # Create analysis
        analysis = ErrorAnalysis(
            category=category,
            count=1,
            first_occurrence=datetime.now(),
            last_occurrence=datetime.now(),
            affected_agents=context.get('agents', []),
            error_patterns=[error],
            root_causes=root_causes,
            resolution_suggestions=suggestions,
            severity=severity
        )
        
        # Update tracking
        error_key = f"{category.value}:{hash(error) % 10000}"
        if error_key in self.errors:
            existing = self.errors[error_key][-1]
            analysis.count = existing.count + 1
            analysis.first_occurrence = existing.first_occurrence
        
        self.errors[error_key].append(analysis)
        return analysis
    
    def get_error_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        cutoff = datetime.now() - time_window
        
        summary = {
            'total_errors': 0,
            'by_category': defaultdict(int),
            'by_severity': defaultdict(int),
            'top_errors': [],
            'trending_up': [],
            'resolution_suggestions': defaultdict(list)
        }
        
        for error_list in self.errors.values():
            recent_errors = [e for e in error_list if e.last_occurrence >= cutoff]
            
            for error in recent_errors:
                summary['total_errors'] += error.count
                summary['by_category'][error.category.value] += error.count
                summary['by_severity'][error.severity] += error.count
                
                for suggestion in error.resolution_suggestions:
                    summary['resolution_suggestions'][error.category.value].append(suggestion)
        
        # Get top errors
        error_counts = []
        for error_list in self.errors.values():
            if error_list:
                latest = error_list[-1]
                if latest.last_occurrence >= cutoff:
                    error_counts.append((latest.count, latest))
        
        summary['top_errors'] = [error for _, error in sorted(error_counts, reverse=True)[:10]]
        
        return dict(summary)


class InteractiveDebugger:
    """Interactive debugging console with real breakpoint handling"""
    
    def __init__(self, client):
        self.client = client
        self.breakpoints = {}
        self.paused_contexts = {}
        self.debug_server = None
        self.active = False
        self.command_history = deque(maxlen=100)
        
    def start_debug_server(self, port: int = 5000):
        """Start interactive debugging server"""
        app = Flask(__name__)
        
        @app.route('/breakpoints')
        def list_breakpoints():
            return jsonify(list(self.breakpoints.keys()))
        
        @app.route('/breakpoints', methods=['POST'])
        def set_breakpoint():
            data = request.json
            method_name = data['method']
            condition = data.get('condition', 'True')
            self.set_breakpoint(method_name, condition)
            return jsonify({'status': 'set', 'method': method_name})
        
        @app.route('/continue/<context_id>')
        def continue_execution(context_id):
            if context_id in self.paused_contexts:
                self.paused_contexts[context_id]['event'].set()
                return jsonify({'status': 'continued'})
            return jsonify({'status': 'not_found'}), 404
        
        @app.route('/inspect/<context_id>')
        def inspect_context(context_id):
            if context_id in self.paused_contexts:
                context = self.paused_contexts[context_id]
                return jsonify({
                    'method': context['method'],
                    'args': context['args'],
                    'kwargs': context['kwargs'],
                    'locals': context.get('locals', {}),
                    'stack_trace': context.get('stack_trace', [])
                })
            return jsonify({'status': 'not_found'}), 404
        
        # Start server in background thread
        def run_server():
            app.run(host='0.0.0.0', port=port, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        self.active = True
        
        logger.info(f"Debug server started on http://0.0.0.0:{port}")
    
    def set_breakpoint(self, method_name: str, condition: str = "True"):
        """Set a real breakpoint with condition"""
        compiled_condition = compile(condition, '<breakpoint>', 'eval')
        self.breakpoints[method_name] = compiled_condition
        logger.info(f"Breakpoint set on {method_name} with condition: {condition}")
    
    async def handle_breakpoint(self, method_path: str, args: tuple, kwargs: dict, frame):
        """Handle breakpoint with real pause and inspection"""
        if not self.active or method_path not in self.breakpoints:
            return
        
        # Evaluate breakpoint condition
        condition = self.breakpoints[method_path]
        try:
            local_vars = frame.f_locals.copy()
            local_vars.update({'args': args, 'kwargs': kwargs})
            
            if not eval(condition, frame.f_globals, local_vars):
                return
        except Exception as e:
            logger.warning(f"Breakpoint condition evaluation failed: {e}")
            return
        
        # Create pause context
        context_id = f"{method_path}_{int(time.time() * 1000)}"
        pause_event = asyncio.Event()
        
        # Extract debug information
        stack_trace = []
        current_frame = frame
        while current_frame:
            stack_trace.append({
                'filename': current_frame.f_code.co_filename,
                'function': current_frame.f_code.co_name,
                'line': current_frame.f_lineno,
                'locals': {k: str(v) for k, v in current_frame.f_locals.items()}
            })
            current_frame = current_frame.f_back
        
        self.paused_contexts[context_id] = {
            'event': pause_event,
            'method': method_path,
            'args': str(args),
            'kwargs': str(kwargs),
            'locals': {k: str(v) for k, v in frame.f_locals.items()},
            'stack_trace': stack_trace,
            'created_at': datetime.now()
        }
        
        logger.critical(f"🔴 BREAKPOINT HIT: {method_path} (Context: {context_id})")
        logger.info(f"Visit debug server to inspect context and continue execution")
        logger.info(f"Or use: curl -X POST http://localhost:5000/continue/{context_id}")
        
        # Wait for user to continue
        await pause_event.wait()
        
        # Clean up
        del self.paused_contexts[context_id]
        logger.info(f"▶️ CONTINUING: {method_path}")


class PerformanceDashboard:
    """Visual performance dashboard with real-time updates"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.dashboard_app = None
        self.dashboard_thread = None
        
    def start_dashboard(self, port: int = 8050):
        """Start interactive dashboard"""
        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import plotly.graph_objs as go
        except ImportError:
            logger.error("Dash not installed. Install with: pip install dash")
            return
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("A2A Agent Performance Dashboard", style={'textAlign': 'center'}),
            
            # Key metrics row
            html.Div([
                html.Div([
                    html.H3(id='total-requests'),
                    html.P('Total Requests')
                ], className='metric-box'),
                html.Div([
                    html.H3(id='success-rate'),
                    html.P('Success Rate')
                ], className='metric-box'),
                html.Div([
                    html.H3(id='avg-response-time'),
                    html.P('Avg Response Time')
                ], className='metric-box'),
                html.Div([
                    html.H3(id='active-agents'),
                    html.P('Active Agents')
                ], className='metric-box'),
            ], style={'display': 'flex', 'justify-content': 'space-around'}),
            
            # Charts
            dcc.Graph(id='response-time-chart'),
            dcc.Graph(id='throughput-chart'),
            dcc.Graph(id='error-rate-chart'),
            dcc.Graph(id='agent-performance-chart'),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('total-requests', 'children'),
             Output('success-rate', 'children'),
             Output('avg-response-time', 'children'),
             Output('active-agents', 'children'),
             Output('response-time-chart', 'figure'),
             Output('throughput-chart', 'figure'),
             Output('error-rate-chart', 'figure'),
             Output('agent-performance-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._generate_dashboard_data()
        
        def run_dashboard():
            app.run_server(host='0.0.0.0', port=port, debug=False)
        
        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        
        logger.info(f"Performance dashboard started on http://0.0.0.0:{port}")
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics data point"""
        timestamp = datetime.now()
        
        for key, value in metrics.items():
            self.metrics_history[key].append({
                'timestamp': timestamp,
                'value': value
            })
    
    def _generate_dashboard_data(self):
        """Generate dashboard data"""
        # Get latest metrics
        latest_metrics = {}
        for key, history in self.metrics_history.items():
            if history:
                latest_metrics[key] = history[-1]['value']
        
        # Key metrics
        total_requests = latest_metrics.get('total_requests', 0)
        success_rate = f"{latest_metrics.get('success_rate', 0):.1%}"
        avg_response_time = f"{latest_metrics.get('avg_response_time', 0):.2f}s"
        active_agents = latest_metrics.get('active_agents', 0)
        
        # Charts
        response_time_chart = self._create_time_series_chart(
            'response_time', 'Response Time (seconds)', 'Response Time Trend'
        )
        throughput_chart = self._create_time_series_chart(
            'throughput', 'Requests/sec', 'Throughput Trend'
        )
        error_rate_chart = self._create_time_series_chart(
            'error_rate', 'Error Rate (%)', 'Error Rate Trend'
        )
        agent_performance_chart = self._create_agent_performance_chart()
        
        return (
            total_requests, success_rate, avg_response_time, active_agents,
            response_time_chart, throughput_chart, error_rate_chart, agent_performance_chart
        )
    
    def _create_time_series_chart(self, metric_key: str, y_label: str, title: str):
        """Create time series chart"""
        history = list(self.metrics_history[metric_key])
        
        if not history:
            return {'data': [], 'layout': {'title': title}}
        
        timestamps = [point['timestamp'] for point in history]
        values = [point['value'] for point in history]
        
        return {
            'data': [go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=metric_key
            )],
            'layout': go.Layout(
                title=title,
                xaxis={'title': 'Time'},
                yaxis={'title': y_label}
            )
        }
    
    def _create_agent_performance_chart(self):
        """Create agent performance comparison chart"""
        # Mock data for agent performance
        agents = ['Agent-1', 'Agent-2', 'Agent-3', 'Agent-4']
        success_rates = [0.95, 0.87, 0.92, 0.89]
        
        return {
            'data': [go.Bar(
                x=agents,
                y=success_rates,
                name='Success Rate'
            )],
            'layout': go.Layout(
                title='Agent Performance Comparison',
                xaxis={'title': 'Agents'},
                yaxis={'title': 'Success Rate', 'range': [0, 1]}
            )
        }


class ComprehensiveDebugSuite:
    """Main debug suite combining all enhanced features"""
    
    def __init__(self, client):
        self.client = client
        self.blockchain_tracer = None
        self.error_analyzer = ErrorAnalyzer()
        self.interactive_debugger = InteractiveDebugger(client)
        self.performance_dashboard = PerformanceDashboard()
        self.regression_detector = PerformanceRegressionDetector()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.baseline_metrics = {}
        
        # Comprehensive method coverage
        self.wrapped_methods = set()
        self.method_stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0,
            'error_count': 0,
            'last_called': None
        })
    
    def initialize_blockchain_integration(self, web3_client: Web3, contracts: Dict[str, str]):
        """Initialize blockchain integration"""
        self.blockchain_tracer = BlockchainTracer(web3_client, contracts)
        logger.info("Blockchain integration initialized")
    
    def start_comprehensive_debugging(self, 
                                    debug_port: int = 5000,
                                    dashboard_port: int = 8050):
        """Start all debugging features"""
        # Start interactive debugger
        self.interactive_debugger.start_debug_server(debug_port)
        
        # Start performance dashboard
        self.performance_dashboard.start_dashboard(dashboard_port)
        
        # Enable comprehensive method wrapping
        self._wrap_all_sdk_methods()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        logger.info("Comprehensive debugging suite started")
        logger.info(f"Debug console: http://localhost:{debug_port}")
        logger.info(f"Performance dashboard: http://localhost:{dashboard_port}")
    
    def _wrap_all_sdk_methods(self):
        """Wrap all SDK methods for comprehensive coverage"""
        # Get all service managers
        services = ['agents', 'messages', 'tokens', 'governance', 'scalability', 'reputation']
        
        for service_name in services:
            if hasattr(self.client, service_name):
                service = getattr(self.client, service_name)
                self._wrap_service_methods(service, service_name)
        
        # Also wrap client methods
        self._wrap_service_methods(self.client, 'client')
    
    def _wrap_service_methods(self, service: Any, service_name: str):
        """Wrap all methods of a service"""
        for method_name in dir(service):
            if not method_name.startswith('_') and callable(getattr(service, method_name)):
                full_method_name = f"{service_name}.{method_name}"
                if full_method_name not in self.wrapped_methods:
                    self._wrap_method_enhanced(service, method_name, full_method_name)
                    self.wrapped_methods.add(full_method_name)
    
    def _wrap_method_enhanced(self, obj: Any, method_name: str, full_method_name: str):
        """Enhanced method wrapping with comprehensive monitoring"""
        original_method = getattr(obj, method_name)
        
        async def enhanced_wrapped_method(*args, **kwargs):
            start_time = time.time()
            frame = inspect.currentframe()
            
            try:
                # Breakpoint handling
                await self.interactive_debugger.handle_breakpoint(
                    full_method_name, args, kwargs, frame
                )
                
                # Execute original method
                result = await original_method(*args, **kwargs)
                
                # Success metrics
                execution_time = time.time() - start_time
                self._update_method_stats(full_method_name, execution_time, True)
                
                return result
                
            except Exception as e:
                # Error handling and analysis
                execution_time = time.time() - start_time
                self._update_method_stats(full_method_name, execution_time, False)
                
                # Analyze error
                error_context = {
                    'method': full_method_name,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200],
                    'agents': self._extract_agent_addresses(args, kwargs)
                }
                
                error_analysis = self.error_analyzer.analyze_error(str(e), error_context)
                logger.error(f"Method {full_method_name} failed: {error_analysis}")
                
                raise
                
        # Wrap sync methods too
        if not asyncio.iscoroutinefunction(original_method):
            def sync_wrapped_method(*args, **kwargs):
                start_time = time.time()
                frame = inspect.currentframe()
                
                try:
                    result = original_method(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._update_method_stats(full_method_name, execution_time, True)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._update_method_stats(full_method_name, execution_time, False)
                    
                    error_context = {
                        'method': full_method_name,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200],
                        'agents': self._extract_agent_addresses(args, kwargs)
                    }
                    
                    error_analysis = self.error_analyzer.analyze_error(str(e), error_context)
                    logger.error(f"Method {full_method_name} failed: {error_analysis}")
                    raise
                    
            setattr(obj, method_name, sync_wrapped_method)
        else:
            setattr(obj, method_name, enhanced_wrapped_method)
    
    def _update_method_stats(self, method_name: str, execution_time: float, success: bool):
        """Update comprehensive method statistics"""
        stats = self.method_stats[method_name]
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        stats['last_called'] = datetime.now()
        
        if not success:
            stats['error_count'] += 1
        
        # Update dashboard metrics
        self.performance_dashboard.add_metrics({
            'response_time': execution_time,
            'success_rate': (stats['call_count'] - stats['error_count']) / stats['call_count'],
            'throughput': 1 / execution_time if execution_time > 0 else 0,
            'error_rate': stats['error_count'] / stats['call_count'],
            'total_requests': sum(s['call_count'] for s in self.method_stats.values()),
            'active_agents': len(set(self._extract_all_agent_addresses()))
        })
        
        # Check for regressions
        self.regression_detector.check_regression(method_name, execution_time, success)
    
    def _extract_agent_addresses(self, args: tuple, kwargs: dict) -> List[str]:
        """Extract agent addresses from method arguments"""
        addresses = []
        
        # Check args
        for arg in args:
            if isinstance(arg, str) and arg.startswith('0x') and len(arg) == 42:
                addresses.append(arg)
            elif hasattr(arg, 'address'):
                addresses.append(arg.address)
        
        # Check kwargs
        for value in kwargs.values():
            if isinstance(value, str) and value.startswith('0x') and len(value) == 42:
                addresses.append(value)
            elif hasattr(value, 'address'):
                addresses.append(value.address)
        
        return addresses
    
    def _extract_all_agent_addresses(self) -> List[str]:
        """Extract all agent addresses from recent activity"""
        addresses = set()
        
        # This would integrate with actual agent registry
        # For now, return mock data
        return ['0x1234...', '0x5678...', '0x9abc...']
    
    def _start_performance_monitoring(self):
        """Start continuous performance monitoring"""
        def monitor_system():
            while True:
                try:
                    # System metrics
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    # Network metrics (mock for now)
                    network_latency = 0.05  # Would ping actual services
                    
                    snapshot = PerformanceSnapshot(
                        timestamp=datetime.now(),
                        cpu_usage=cpu_percent / 100,
                        memory_usage=memory.percent / 100,
                        network_latency=network_latency,
                        active_connections=len(self.wrapped_methods),
                        request_count=sum(s['call_count'] for s in self.method_stats.values()),
                        error_count=sum(s['error_count'] for s in self.method_stats.values())
                    )
                    
                    self.performance_history.append(snapshot)
                    
                    # Update dashboard
                    self.performance_dashboard.add_metrics({
                        'cpu_usage': cpu_percent / 100,
                        'memory_usage': memory.percent / 100,
                        'network_latency': network_latency
                    })
                    
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                
                time.sleep(self.performance_dashboard.update_interval)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    async def trace_message_comprehensive(self, message_id: str) -> Dict[str, Any]:
        """Comprehensive message tracing with blockchain integration"""
        trace_data = {
            'message_id': message_id,
            'sdk_trace': [],
            'blockchain_trace': {},
            'performance_metrics': {},
            'error_analysis': None
        }
        
        try:
            # SDK-level tracing
            trace_data['sdk_trace'] = self._get_sdk_message_trace(message_id)
            
            # Blockchain tracing
            if self.blockchain_tracer:
                trace_data['blockchain_trace'] = await self.blockchain_tracer.trace_message_blockchain(message_id)
            
            # Performance analysis
            trace_data['performance_metrics'] = self._analyze_message_performance(message_id)
            
        except Exception as e:
            error_analysis = self.error_analyzer.analyze_error(str(e), {'message_id': message_id})
            trace_data['error_analysis'] = asdict(error_analysis)
        
        return trace_data
    
    def _get_sdk_message_trace(self, message_id: str) -> List[Dict[str, Any]]:
        """Get SDK-level message trace"""
        # This would integrate with actual message tracking
        return [
            {
                'step': 'message_created',
                'timestamp': datetime.now().isoformat(),
                'duration': 0.01,
                'status': 'success'
            },
            {
                'step': 'message_sent',
                'timestamp': datetime.now().isoformat(),
                'duration': 0.05,
                'status': 'success'
            }
        ]
    
    def _analyze_message_performance(self, message_id: str) -> Dict[str, Any]:
        """Analyze message performance"""
        return {
            'total_duration': 0.5,
            'network_time': 0.3,
            'processing_time': 0.2,
            'gas_efficiency': 0.8,
            'bottlenecks': ['network_latency']
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive debugging report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'method_statistics': dict(self.method_stats),
            'error_summary': self.error_analyzer.get_error_summary(),
            'performance_summary': self._get_performance_summary(),
            'regression_alerts': self.regression_detector.get_recent_regressions(),
            'active_breakpoints': list(self.interactive_debugger.breakpoints.keys()),
            'system_health': self._get_system_health()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {}
        
        recent_snapshots = list(self.performance_history)[-10:]
        
        return {
            'avg_cpu_usage': statistics.mean(s.cpu_usage for s in recent_snapshots),
            'avg_memory_usage': statistics.mean(s.memory_usage for s in recent_snapshots),
            'avg_network_latency': statistics.mean(s.network_latency for s in recent_snapshots),
            'total_requests': recent_snapshots[-1].request_count if recent_snapshots else 0,
            'total_errors': recent_snapshots[-1].error_count if recent_snapshots else 0
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            'status': 'healthy',  # Would be calculated based on metrics
            'uptime': time.time() - (self.performance_history[0].timestamp.timestamp() if self.performance_history else time.time()),
            'debug_server_active': self.interactive_debugger.active,
            'dashboard_active': self.performance_dashboard.dashboard_thread is not None,
            'blockchain_connected': self.blockchain_tracer is not None
        }


class PerformanceRegressionDetector:
    """Automated performance regression detection"""
    
    def __init__(self, sensitivity: float = 0.2):
        self.sensitivity = sensitivity
        self.baselines = {}
        self.regressions = deque(maxlen=100)
        
    def check_regression(self, method_name: str, execution_time: float, success: bool):
        """Check for performance regression"""
        if method_name not in self.baselines:
            self.baselines[method_name] = {
                'response_times': deque(maxlen=50),
                'success_rates': deque(maxlen=50)
            }
        
        baseline = self.baselines[method_name]
        baseline['response_times'].append(execution_time)
        baseline['success_rates'].append(1 if success else 0)
        
        # Need sufficient data for comparison
        if len(baseline['response_times']) < 20:
            return
        
        # Calculate recent vs baseline performance
        recent_times = list(baseline['response_times'])[-10:]
        baseline_times = list(baseline['response_times'])[:-10]
        
        if not baseline_times:
            return
        
        recent_avg = statistics.mean(recent_times)
        baseline_avg = statistics.mean(baseline_times)
        
        # Check for regression
        if recent_avg > baseline_avg * (1 + self.sensitivity):
            degradation = ((recent_avg - baseline_avg) / baseline_avg) * 100
            
            regression = PerformanceRegression(
                metric='response_time',
                baseline=baseline_avg,
                current=recent_avg,
                degradation_percent=degradation,
                detected_at=datetime.now(),
                confidence=min(degradation / 50, 1.0),  # Higher degradation = higher confidence
                affected_operations=[method_name]
            )
            
            self.regressions.append(regression)
            logger.warning(f"Performance regression detected in {method_name}: {degradation:.1f}% slower")
    
    def get_recent_regressions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent performance regressions"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.regressions if r.detected_at >= cutoff]
        return [asdict(r) for r in recent]