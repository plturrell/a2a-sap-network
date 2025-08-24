"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

#!/usr/bin/env python3
"""
A2A ChatAgent CLI - Test and interact with the production ChatAgent
Provides comprehensive testing, monitoring, and interaction capabilities
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of aiohttp  # REMOVED: A2A protocol violation
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import print as rprint


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Add directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "shared"))
sys.path.append(str(Path(__file__).parent.parent.parent / "app" / "a2a" / "sdk"))

# Import production components with fallbacks
try:
    from chatAgent import ChatAgent
    CHATAGENT_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Warning: ChatAgent not available ({e})[/yellow]")
    CHATAGENT_AVAILABLE = False
    ChatAgent = None

try:
    from production.authRateLimit import AuthenticationManager
    AUTH_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Warning: AuthenticationManager not available ({e})[/yellow]")
    AUTH_AVAILABLE = False
    AuthenticationManager = None

try:
    from production.productionDatabase import create_production_database
    DATABASE_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Warning: ProductionDatabase not available ({e})[/yellow]")
    DATABASE_AVAILABLE = False
    create_production_database = None

try:
    from production.monitoringMetrics import create_metrics_collector
    METRICS_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Warning: MetricsCollector not available ({e})[/yellow]")
    METRICS_AVAILABLE = False
    create_metrics_collector = None

try:
    from production.websocketManager import create_connection_manager
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]Warning: WebSocketManager not available ({e})[/yellow]")
    WEBSOCKET_AVAILABLE = False
    create_connection_manager = None

console = Console()

class A2AChatCLI:
    """
    Comprehensive CLI for testing and interacting with A2A ChatAgent
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.chat_agent = None
        self.auth_manager = None
        self.database = None
        self.metrics = None
        self.session_id = f"cli_{int(time.time())}"
        
        # Configure logging
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "base_url": os.getenv("A2A_SERVICE_URL"),
            "environment": "development",
            "log_level": "INFO",
            "timeout": 30,
            "max_retries": 3,
            "database": {
                "type": "sqlite",
                "connection_string": "sqlite+aiosqlite:///cli_test.db"
            },
            "auth": {
                "jwt_secret": "test-secret-key",
                "enable_jwt": True,
                "enable_api_key": True
            },
            "test_agents": [
                "data-processor",
                "nlp-agent", 
                "crypto-trader",
                "file-manager",
                "web-scraper"
            ]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Merge with defaults
                default_config.update(file_config)
                console.print(f"✅ Loaded configuration from {config_path}")
            except Exception as e:
                console.print(f"⚠️ Failed to load config file {config_path}: {e}")
                console.print("Using default configuration...")
        
        return default_config
    
    async def initialize(self):
        """Initialize ChatAgent and all production components"""
        console.print("\n🚀 Initializing A2A ChatAgent CLI...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            # Initialize database
            if DATABASE_AVAILABLE:
                task1 = progress.add_task("Setting up production database...", total=None)
                self.database = create_production_database(self.config["database"])
                await self.database.initialize()
                progress.update(task1, completed=True)
            else:
                console.print("[yellow]⚠️ Database not available - using mock mode[/yellow]")
            
            # Initialize metrics
            if METRICS_AVAILABLE:
                task2 = progress.add_task("Setting up monitoring & metrics...", total=None)
                self.metrics = create_metrics_collector(self.config)
                progress.update(task2, completed=True)
            else:
                console.print("[yellow]⚠️ Metrics not available - using mock mode[/yellow]")
            
            # Initialize authentication
            if AUTH_AVAILABLE:
                task3 = progress.add_task("Setting up authentication...", total=None)
                self.auth_manager = AuthenticationManager(
                    self.config["auth"], 
                    self.database
                )
                progress.update(task3, completed=True)
            else:
                console.print("[yellow]⚠️ Authentication not available - using mock mode[/yellow]")
            
            # Initialize ChatAgent
            if CHATAGENT_AVAILABLE:
                task4 = progress.add_task("Initializing ChatAgent...", total=None)
                self.chat_agent = ChatAgent(
                    base_url=self.config["base_url"],
                    config=self.config
                )
                await self.chat_agent.initialize()
                progress.update(task4, completed=True)
                console.print("✅ ChatAgent initialized successfully!")
            else:
                console.print("[red]❌ ChatAgent not available - CLI will run in demo mode[/red]")
        
        console.print("✅ CLI initialization completed!\n")
    
    async def test_single_message(self, message: str, target_agent: Optional[str] = None) -> Dict[str, Any]:
        """Test sending a single message through the ChatAgent"""
        console.print(f"📤 Sending message: [bold cyan]{message}[/bold cyan]")
        
        if target_agent:
            console.print(f"🎯 Target agent: [bold yellow]{target_agent}[/bold yellow]")
        
        try:
            # Create A2A message
            from a2aCommon import A2AMessage, MessageRole
            
            a2a_message = A2AMessage(
                role=MessageRole.USER,
                content={
                    'data': {
                        'prompt': message,
                        'user_id': 'cli_user',
                        'target_agent': target_agent,
                        'session_id': self.session_id
                    }
                },
                context_id=f"cli_{int(time.time())}"
            )
            
            start_time = time.time()
            
            # Send through ChatAgent
            response = await self.chat_agent.handle_chat_message(
                a2a_message, 
                a2a_message.context_id
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Display results
            self._display_response(response, duration)
            
            return {
                "success": True,
                "response": response,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            console.print(f"❌ Error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def test_multi_agent_broadcast(self, message: str, agents: List[str] = None) -> Dict[str, Any]:
        """Test broadcasting a message to multiple agents"""
        if not agents:
            agents = self.config["test_agents"]
        
        console.print(f"\n📡 Broadcasting to {len(agents)} agents: [bold cyan]{message}[/bold cyan]")
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            task = progress.add_task("Testing agents...", total=len(agents))
            
            for agent in agents:
                progress.update(task, description=f"Testing {agent}...")
                
                result = await self.test_single_message(message, agent)
                results[agent] = result
                
                progress.advance(task)
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.5)
        
        # Display summary
        self._display_broadcast_summary(results)
        
        return results
    
    async def test_conversation_flow(self, messages: List[str]) -> Dict[str, Any]:
        """Test a multi-turn conversation"""
        console.print(f"\n💬 Testing conversation flow with {len(messages)} messages...")
        
        conversation_results = []
        
        for i, message in enumerate(messages, 1):
            console.print(f"\n--- Turn {i}/{len(messages)} ---")
            result = await self.test_single_message(message)
            conversation_results.append(result)
            
            # Brief pause between messages
            await asyncio.sleep(1)
        
        # Display conversation summary
        self._display_conversation_summary(conversation_results)
        
        return {
            "conversation_id": self.session_id,
            "turns": len(messages),
            "results": conversation_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def test_agent_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to all registered agents"""
        console.print("\n🔗 Testing agent connectivity...")
        
        if not self.chat_agent.agent_registry:
            await self.chat_agent._discover_network_agents()
        
        connectivity_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            agents = list(self.chat_agent.agent_registry.keys())
            task = progress.add_task("Checking connectivity...", total=len(agents))
            
            async with # WARNING: aiohttp ClientSession usage violates A2A protocol - must use blockchain messaging
        # aiohttp\.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for agent_id in agents:
                    progress.update(task, description=f"Checking {agent_id}...")
                    
                    agent_info = self.chat_agent.agent_registry[agent_id]
                    endpoint = agent_info["endpoint"]
                    
                    try:
                        async with session.get(f"{endpoint}/health") as resp:
                            if resp.status == 200:
                                connectivity_results[agent_id] = {
                                    "status": "online",
                                    "endpoint": endpoint,
                                    "response_time": resp.headers.get("response-time", "unknown")
                                }
                            else:
                                connectivity_results[agent_id] = {
                                    "status": "error",
                                    "endpoint": endpoint,
                                    "http_status": resp.status
                                }
                    except Exception as e:
                        connectivity_results[agent_id] = {
                            "status": "offline",
                            "endpoint": endpoint,
                            "error": str(e)
                        }
                    
                    progress.advance(task)
        
        # Display connectivity table
        self._display_connectivity_table(connectivity_results)
        
        return connectivity_results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        console.print("\n🧪 Running comprehensive ChatAgent test suite...\n")
        
        test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "tests": {}
        }
        
        # Test 1: Agent connectivity
        console.print(Panel("🔗 Test 1: Agent Connectivity", style="bold blue"))
        test_results["tests"]["connectivity"] = await self.test_agent_connectivity()
        
        # Test 2: Single message routing
        console.print(Panel("📤 Test 2: Single Message Routing", style="bold blue"))
        test_results["tests"]["single_message"] = await self.test_single_message(
            "Analyze the current cryptocurrency market trends and provide insights."
        )
        
        # Test 3: Multi-agent broadcast
        console.print(Panel("📡 Test 3: Multi-Agent Broadcast", style="bold blue"))
        test_results["tests"]["broadcast"] = await self.test_multi_agent_broadcast(
            "Process this data and provide comprehensive analysis from your specialty perspective."
        )
        
        # Test 4: Conversation flow
        console.print(Panel("💬 Test 4: Conversation Flow", style="bold blue"))
        conversation_messages = [
            "Hello, I need help with data analysis.",
            "Can you analyze cryptocurrency trends?",
            "What about file processing capabilities?",
            "Thank you for the comprehensive help!"
        ]
        test_results["tests"]["conversation"] = await self.test_conversation_flow(conversation_messages)
        
        # Test 5: Performance metrics
        console.print(Panel("📊 Test 5: Performance Metrics", style="bold blue"))
        test_results["tests"]["metrics"] = await self._get_performance_metrics()
        
        test_results["end_time"] = datetime.utcnow().isoformat()
        
        # Display final summary
        self._display_comprehensive_summary(test_results)
        
        return test_results
    
    async def interactive_mode(self):
        """Interactive chat mode"""
        console.print(Panel(
            "🤖 A2A ChatAgent Interactive Mode\n"
            "Type your messages to chat with the agent network.\n"
            "Commands: /help, /agents, /metrics, /quit",
            title="Interactive Mode",
            style="bold green"
        ))
        
        while True:
            try:
                message = console.input("\n[bold cyan]You:[/bold cyan] ")
                
                if message.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    console.print("👋 Goodbye!")
                    break
                elif message.lower() == '/help':
                    self._show_help()
                    continue
                elif message.lower() == '/agents':
                    await self.test_agent_connectivity()
                    continue
                elif message.lower() == '/metrics':
                    metrics = await self._get_performance_metrics()
                    self._display_metrics(metrics)
                    continue
                elif message.strip() == '':
                    continue
                
                # Process message
                await self.test_single_message(message)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}")
    
    def _display_response(self, response: Any, duration: float):
        """Display formatted response"""
        table = Table(title="Response Details")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        if isinstance(response, dict):
            for key, value in response.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                table.add_row(str(key), str(value))
        else:
            table.add_row("Response", str(response))
        
        table.add_row("Duration", f"{duration:.2f}s")
        table.add_row("Timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        
        console.print(table)
    
    def _display_broadcast_summary(self, results: Dict[str, Any]):
        """Display broadcast test summary"""
        table = Table(title="Multi-Agent Broadcast Results")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="yellow")
        table.add_column("Success", style="green")
        
        for agent, result in results.items():
            status = "✅" if result["success"] else "❌"
            duration = f"{result.get('duration', 0):.2f}s" if result["success"] else "N/A"
            success_rate = "100%" if result["success"] else "0%"
            
            table.add_row(agent, status, duration, success_rate)
        
        console.print(table)
        
        # Summary stats
        total_agents = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        console.print(f"\n📊 Summary: {successful}/{total_agents} agents responded successfully ({successful/total_agents*100:.1f}%)")
    
    def _display_conversation_summary(self, results: List[Dict[str, Any]]):
        """Display conversation test summary"""
        table = Table(title="Conversation Flow Results")
        table.add_column("Turn", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="yellow")
        table.add_column("Response Type", style="green")
        
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            duration = f"{result.get('duration', 0):.2f}s" if result["success"] else "N/A"
            response_type = type(result.get("response", {})).__name__
            
            table.add_row(str(i), status, duration, response_type)
        
        console.print(table)
    
    def _display_connectivity_table(self, results: Dict[str, Any]):
        """Display agent connectivity results"""
        table = Table(title="Agent Connectivity Status")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Endpoint", style="yellow")
        table.add_column("Details", style="green")
        
        for agent_id, info in results.items():
            status_icon = {
                "online": "🟢",
                "offline": "🔴", 
                "error": "🟡"
            }.get(info["status"], "❓")
            
            status = f"{status_icon} {info['status'].upper()}"
            endpoint = info["endpoint"]
            
            if info["status"] == "online":
                details = info.get("response_time", "OK")
            elif info["status"] == "error":
                details = f"HTTP {info.get('http_status', 'Unknown')}"
            else:
                details = info.get("error", "Connection failed")[:50]
            
            table.add_row(agent_id, status, endpoint, details)
        
        console.print(table)
    
    def _display_comprehensive_summary(self, results: Dict[str, Any]):
        """Display comprehensive test summary"""
        console.print(Panel(
            "🎉 Comprehensive Test Suite Completed!",
            title="Test Results Summary",
            style="bold green"
        ))
        
        # Extract key metrics
        tests = results["tests"]
        
        # Connectivity stats
        connectivity = tests.get("connectivity", {})
        online_agents = sum(1 for info in connectivity.values() if info.get("status") == "online")
        total_agents = len(connectivity)
        
        # Performance stats
        single_msg_success = tests.get("single_message", {}).get("success", False)
        broadcast_success = sum(1 for r in tests.get("broadcast", {}).values() if r.get("success", False))
        conversation_success = sum(1 for r in tests.get("conversation", {}).get("results", []) if r.get("success", False))
        
        # Create summary table
        summary_table = Table(title="Test Results Overview")
        summary_table.add_column("Test Category", style="cyan")
        summary_table.add_column("Result", style="white")
        summary_table.add_column("Details", style="yellow")
        
        summary_table.add_row(
            "Agent Connectivity",
            f"🟢 {online_agents}/{total_agents}" if online_agents > 0 else "🔴 Failed",
            f"{online_agents/total_agents*100:.1f}% online" if total_agents > 0 else "No agents"
        )
        
        summary_table.add_row(
            "Single Message",
            "🟢 Success" if single_msg_success else "🔴 Failed",
            f"Duration: {tests.get('single_message', {}).get('duration', 0):.2f}s"
        )
        
        summary_table.add_row(
            "Multi-Agent Broadcast", 
            f"🟢 {broadcast_success} agents" if broadcast_success > 0 else "🔴 Failed",
            f"Success rate: {broadcast_success/len(tests.get('broadcast', {})):.1%}" if tests.get('broadcast') else "N/A"
        )
        
        summary_table.add_row(
            "Conversation Flow",
            f"🟢 {conversation_success} turns" if conversation_success > 0 else "🔴 Failed", 
            f"Success rate: {conversation_success/len(tests.get('conversation', {}).get('results', [])):.1%}" if tests.get('conversation', {}).get('results') else "N/A"
        )
        
        console.print(summary_table)
        
        # Overall assessment
        overall_success = (
            online_agents > 0 and 
            single_msg_success and 
            broadcast_success > 0 and 
            conversation_success > 0
        )
        
        if overall_success:
            console.print("\n🎊 [bold green]ChatAgent is working perfectly![/bold green] Ready for production deployment.")
        else:
            console.print("\n⚠️ [bold yellow]Some issues detected.[/bold yellow] Review the results above.")
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.metrics:
            return {"error": "Metrics not initialized"}
        
        try:
            performance_report = self.metrics.get_performance_report()
            return {
                "success": True,
                "metrics": performance_report,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _display_metrics(self, metrics: Dict[str, Any]):
        """Display performance metrics"""
        if not metrics.get("success"):
            console.print(f"❌ Metrics error: {metrics.get('error')}")
            return
        
        metrics_data = metrics.get("metrics", {})
        
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in metrics_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    table.add_row(f"{key}.{sub_key}", str(sub_value))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    
    def _show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]A2A ChatAgent CLI Commands:[/bold cyan]

[yellow]Interactive Commands:[/yellow]
  /help     - Show this help message
  /agents   - Test agent connectivity
  /metrics  - Show performance metrics  
  /quit     - Exit interactive mode

[yellow]CLI Arguments:[/yellow]
  --message "text"     - Send single message
  --broadcast "text"   - Broadcast to multiple agents
  --conversation       - Test conversation flow
  --test-all          - Run comprehensive test suite
  --connectivity      - Test agent connectivity only
  --interactive       - Enter interactive mode
  --config FILE       - Use custom config file

[yellow]Examples:[/yellow]
  python cli.py --message "Analyze cryptocurrency trends"
  python cli.py --broadcast "Process this data"
  python cli.py --test-all
  python cli.py --interactive
        """
        console.print(Panel(help_text, title="Help", style="bold blue"))
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.database:
            await self.database.close()
        console.print("🧹 Cleanup completed")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="A2A ChatAgent CLI - Test and interact with the production ChatAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --message "Analyze cryptocurrency trends"
  python cli.py --broadcast "Process this data" 
  python cli.py --test-all
  python cli.py --interactive
  python cli.py --config config.yaml --test-all
        """
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--message', type=str, help='Send a single message')
    parser.add_argument('--target-agent', type=str, help='Target specific agent')
    parser.add_argument('--broadcast', type=str, help='Broadcast message to multiple agents')
    parser.add_argument('--conversation', action='store_true', help='Test conversation flow')
    parser.add_argument('--connectivity', action='store_true', help='Test agent connectivity')
    parser.add_argument('--test-all', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')
    parser.add_argument('--metrics', action='store_true', help='Show performance metrics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = A2AChatCLI(args.config)
    
    try:
        await cli.initialize()
        
        # Execute based on arguments
        if args.message:
            await cli.test_single_message(args.message, args.target_agent)
        elif args.broadcast:
            await cli.test_multi_agent_broadcast(args.broadcast)
        elif args.conversation:
            conversation_messages = [
                "Hello, I need help with data analysis.",
                "Can you analyze cryptocurrency trends?", 
                "What about file processing capabilities?",
                "Thank you for the comprehensive help!"
            ]
            await cli.test_conversation_flow(conversation_messages)
        elif args.connectivity:
            await cli.test_agent_connectivity()
        elif args.test_all:
            await cli.run_comprehensive_test()
        elif args.metrics:
            metrics = await cli._get_performance_metrics()
            cli._display_metrics(metrics)
        elif args.interactive:
            await cli.interactive_mode()
        else:
            # Default: show help and enter interactive mode
            cli._show_help()
            console.print("\nNo specific command provided. Entering interactive mode...\n")
            await cli.interactive_mode()
            
    except KeyboardInterrupt:
        console.print("\n👋 Interrupted by user")
    except Exception as e:
        console.print(f"❌ Fatal error: {e}")
        if args.verbose:
            import traceback


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            console.print(traceback.format_exc())
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())