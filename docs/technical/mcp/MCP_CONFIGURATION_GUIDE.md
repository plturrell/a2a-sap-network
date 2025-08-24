# MCP Configuration Guide and Setup Templates
## A2A Network Model Context Protocol Setup

### Quick Start Configuration

#### 1. Environment Setup

Create a `.env` file in your project root:

```bash
# GitHub MCP Configuration
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_github_personal_access_token_here
GITHUB_DEFAULT_ORG=a2a-network
GITHUB_DEFAULT_REPO=a2a-agents

# Supabase MCP Configuration
SUPABASE_ACCESS_TOKEN=sbp_your_supabase_access_token_here
SUPABASE_DEFAULT_PROJECT_ID=your_project_id_here
SUPABASE_DEFAULT_ORG_ID=your_org_id_here

# Perplexity MCP Configuration
PERPLEXITY_API_KEY=pplx_your_perplexity_api_key_here

# Puppeteer MCP Configuration
PUPPETEER_HEADLESS=true
PUPPETEER_TIMEOUT=30000
PUPPETEER_VIEWPORT_WIDTH=1920
PUPPETEER_VIEWPORT_HEIGHT=1080

# A2A Network Specific
A2A_MCP_LOG_LEVEL=INFO
A2A_MCP_RETRY_ATTEMPTS=3
A2A_MCP_TIMEOUT=60000
```

#### 2. MCP Server Configuration

Create `mcp-config.json` in your project root:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "supabase": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-supabase"
      ],
      "env": {
        "SUPABASE_ACCESS_TOKEN": "${SUPABASE_ACCESS_TOKEN}"
      }
    },
    "puppeteer": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-puppeteer"
      ],
      "env": {
        "PUPPETEER_HEADLESS": "${PUPPETEER_HEADLESS}",
        "PUPPETEER_TIMEOUT": "${PUPPETEER_TIMEOUT}"
      }
    },
    "perplexity": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-perplexity"
      ],
      "env": {
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
      }
    }
  }
}
```

### A2A Agent MCP Integration Template

#### Base Agent with MCP Support

```python
# a2a_mcp_base.py
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MCPToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    duration: float = 0.0
    tool_name: str = ""

class MCPClientBase(ABC):
    """Base class for all MCP clients in A2A Network"""
    
    def __init__(self, client_name: str, config: Dict[str, Any] = None):
        self.client_name = client_name
        self.config = config or {}
        self.logger = logging.getLogger(f"a2a.mcp.{client_name}")
        self.metrics = []
        
    async def call_tool(self, tool_name: str, **kwargs) -> MCPToolResult:
        """Generic tool calling with monitoring and error handling"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Calling MCP tool: {tool_name}", extra={
                "tool": tool_name,
                "client": self.client_name,
                "parameters": kwargs
            })
            
            result = await self._execute_tool(tool_name, **kwargs)
            duration = time.time() - start_time
            
            tool_result = MCPToolResult(
                success=True,
                data=result,
                duration=duration,
                tool_name=tool_name
            )
            
            self.metrics.append(tool_result)
            
            self.logger.info(f"MCP tool completed: {tool_name}", extra={
                "tool": tool_name,
                "client": self.client_name,
                "duration": duration,
                "success": True
            })
            
            return tool_result
            
        except Exception as e:
            duration = time.time() - start_time
            
            tool_result = MCPToolResult(
                success=False,
                data=None,
                error=str(e),
                duration=duration,
                tool_name=tool_name
            )
            
            self.metrics.append(tool_result)
            
            self.logger.error(f"MCP tool failed: {tool_name}", extra={
                "tool": tool_name,
                "client": self.client_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    @abstractmethod
    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Implement specific tool execution logic"""
        pass

class A2AAgentWithMCP:
    """Base class for A2A agents with MCP integration"""
    
    def __init__(self, agent_name: str, mcp_config: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.mcp_config = mcp_config or {}
        self.mcp_clients = {}
        self.logger = logging.getLogger(f"a2a.agent.{agent_name}")
        
        # Initialize MCP clients based on configuration
        self._initialize_mcp_clients()
    
    def _initialize_mcp_clients(self):
        """Initialize MCP clients based on configuration"""
        if "github" in self.mcp_config:
            self.mcp_clients["github"] = GitHubMCPClient(
                "github", 
                self.mcp_config["github"]
            )
        
        if "supabase" in self.mcp_config:
            self.mcp_clients["supabase"] = SupabaseMCPClient(
                "supabase", 
                self.mcp_config["supabase"]
            )
        
        if "puppeteer" in self.mcp_config:
            self.mcp_clients["puppeteer"] = PuppeteerMCPClient(
                "puppeteer", 
                self.mcp_config["puppeteer"]
            )
        
        if "perplexity" in self.mcp_config:
            self.mcp_clients["perplexity"] = PerplexityMCPClient(
                "perplexity", 
                self.mcp_config["perplexity"]
            )
    
    async def call_mcp_tool(self, client_name: str, tool_name: str, **kwargs) -> MCPToolResult:
        """Call MCP tool with agent context"""
        if client_name not in self.mcp_clients:
            raise ValueError(f"MCP client '{client_name}' not configured for agent '{self.agent_name}'")
        
        return await self.mcp_clients[client_name].call_tool(tool_name, **kwargs)
    
    def get_mcp_metrics(self) -> Dict[str, Any]:
        """Get aggregated MCP metrics for this agent"""
        all_metrics = []
        for client in self.mcp_clients.values():
            all_metrics.extend(client.metrics)
        
        if not all_metrics:
            return {"message": "No MCP metrics available"}
        
        total_calls = len(all_metrics)
        successful_calls = len([m for m in all_metrics if m.success])
        avg_duration = sum(m.duration for m in all_metrics) / total_calls
        
        tool_stats = {}
        for metric in all_metrics:
            if metric.tool_name not in tool_stats:
                tool_stats[metric.tool_name] = {
                    "calls": 0,
                    "successes": 0,
                    "total_duration": 0,
                    "errors": []
                }
            
            stats = tool_stats[metric.tool_name]
            stats["calls"] += 1
            if metric.success:
                stats["successes"] += 1
            stats["total_duration"] += metric.duration
            if metric.error:
                stats["errors"].append(metric.error)
        
        return {
            "agent": self.agent_name,
            "total_calls": total_calls,
            "success_rate": successful_calls / total_calls,
            "avg_duration": avg_duration,
            "tool_statistics": tool_stats
        }
```

#### Specific MCP Client Implementations

```python
# github_mcp_client.py
class GitHubMCPClient(MCPClientBase):
    """GitHub MCP client for A2A Network"""
    
    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute GitHub MCP tools"""
        # Map tool names to actual MCP functions
        tool_mapping = {
            "create_repository": self._create_repository,
            "create_branch": self._create_branch,
            "create_pull_request": self._create_pull_request,
            "push_files": self._push_files,
            "search_code": self._search_code,
            "list_commits": self._list_commits,
            "merge_pull_request": self._merge_pull_request
        }
        
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown GitHub MCP tool: {tool_name}")
        
        return await tool_mapping[tool_name](**kwargs)
    
    async def _create_repository(self, name: str, description: str = "", private: bool = False, **kwargs):
        # Implementation using actual MCP tool
        return await mcp1_create_repository(
            name=name,
            description=description,
            private=private,
            autoInit=kwargs.get("autoInit", True)
        )
    
    async def _create_branch(self, owner: str, repo: str, branch: str, from_branch: str = "main", **kwargs):
        return await mcp1_create_branch(
            owner=owner,
            repo=repo,
            branch=branch,
            from_branch=from_branch
        )
    
    async def _create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str, body: str = "", **kwargs):
        return await mcp1_create_pull_request(
            owner=owner,
            repo=repo,
            title=title,
            head=head,
            base=base,
            body=body,
            draft=kwargs.get("draft", False)
        )
    
    async def _push_files(self, owner: str, repo: str, branch: str, files: List[Dict], message: str, **kwargs):
        return await mcp1_push_files(
            owner=owner,
            repo=repo,
            branch=branch,
            files=files,
            message=message
        )
    
    async def _search_code(self, q: str, **kwargs):
        return await mcp1_search_code(
            q=q,
            per_page=kwargs.get("per_page", 30),
            page=kwargs.get("page", 1)
        )
    
    async def _list_commits(self, owner: str, repo: str, sha: str = "main", **kwargs):
        return await mcp1_list_commits(
            owner=owner,
            repo=repo,
            sha=sha,
            page=kwargs.get("page", 1),
            perPage=kwargs.get("perPage", 30)
        )
    
    async def _merge_pull_request(self, owner: str, repo: str, pull_number: int, **kwargs):
        return await mcp1_merge_pull_request(
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            commit_title=kwargs.get("commit_title", ""),
            commit_message=kwargs.get("commit_message", ""),
            merge_method=kwargs.get("merge_method", "merge")
        )

# puppeteer_mcp_client.py
class PuppeteerMCPClient(MCPClientBase):
    """Puppeteer MCP client for A2A Network"""

    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        tool_mapping = {
            "navigate": self._navigate,
            "screenshot": self._screenshot,
            "click": self._click,
            "fill": self._fill,
            "evaluate": self._evaluate
        }
        
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown Puppeteer MCP tool: {tool_name}")
        
        return await tool_mapping[tool_name](**kwargs)

    async def _navigate(self, url: str, **kwargs):
        return await mcp4_puppeteer_navigate(url=url)

    async def _screenshot(self, name: str, **kwargs):
        return await mcp4_puppeteer_screenshot(name=name, **kwargs)

    async def _click(self, selector: str, **kwargs):
        return await mcp4_puppeteer_click(selector=selector)

    async def _fill(self, selector: str, value: str, **kwargs):
        return await mcp4_puppeteer_fill(selector=selector, value=value)

    async def _evaluate(self, script: str, **kwargs):
        return await mcp4_puppeteer_evaluate(script=script)

# perplexity_mcp_client.py
class PerplexityMCPClient(MCPClientBase):
    """Perplexity MCP client for A2A Network"""

    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        tool_mapping = {
            "ask": self._ask
        }
        
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown Perplexity MCP tool: {tool_name}")
        
        return await tool_mapping[tool_name](**kwargs)

    async def _ask(self, messages: List[Dict[str, str]], **kwargs):
        return await mcp2_perplexity_ask(messages=messages)


# supabase_mcp_client.py
class SupabaseMCPClient(MCPClientBase):
    """Supabase MCP client for A2A Network"""
    
    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        tool_mapping = {
            "execute_sql": self._execute_sql,
            "apply_migration": self._apply_migration,
            "create_project": self._create_project,
            "list_projects": self._list_projects,
            "get_advisors": self._get_advisors,
            "deploy_edge_function": self._deploy_edge_function,
            "generate_typescript_types": self._generate_typescript_types
        }
        
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown Supabase MCP tool: {tool_name}")
        
        return await tool_mapping[tool_name](**kwargs)
    
    async def _execute_sql(self, project_id: str, query: str, **kwargs):
        return await mcp6_execute_sql(
            project_id=project_id,
            query=query
        )
    
    async def _apply_migration(self, project_id: str, name: str, query: str, **kwargs):
        return await mcp6_apply_migration(
            project_id=project_id,
            name=name,
            query=query
        )
    
    async def _create_project(self, name: str, organization_id: str, confirm_cost_id: str, **kwargs):
        return await mcp6_create_project(
            name=name,
            organization_id=organization_id,
            confirm_cost_id=confirm_cost_id,
            region=kwargs.get("region", "us-east-1")
        )
    
    async def _list_projects(self, **kwargs):
        return await mcp6_list_projects()
    
    async def _get_advisors(self, project_id: str, type: str = "security", **kwargs):
        return await mcp6_get_advisors(
            project_id=project_id,
            type=type
        )
    
    async def _deploy_edge_function(self, project_id: str, name: str, files: List[Dict], **kwargs):
        return await mcp6_deploy_edge_function(
            project_id=project_id,
            name=name,
            files=files,
            entrypoint_path=kwargs.get("entrypoint_path", "index.ts")
        )
    
    async def _generate_typescript_types(self, project_id: str, **kwargs):
        return await mcp6_generate_typescript_types(
            project_id=project_id
        )
```

### Agent Configuration Templates

#### 1. Data Processing Agent Configuration

```python
# data_processing_agent_config.py
DATA_PROCESSING_AGENT_CONFIG = {
    "agent_name": "data_processing_agent",
    "mcp_config": {
        "supabase": {
            "project_id": "${SUPABASE_DEFAULT_PROJECT_ID}",
            "timeout": 30000,
            "retry_attempts": 3
        },
        "github": {
            "default_owner": "${GITHUB_DEFAULT_ORG}",
            "default_repo": "a2a-data",
            "timeout": 15000
        }
    },
    "skills": [
        "process_data_pipeline",
        "validate_data_quality",
        "transform_data_format"
    ],
    "dependencies": [
        "quality_assessment_agent",
        "validation_agent"
    ]
}

class DataProcessingAgent(A2AAgentWithMCP):
    def __init__(self):
        super().__init__(
            agent_name="data_processing_agent",
            mcp_config=DATA_PROCESSING_AGENT_CONFIG["mcp_config"]
        )
    
    @a2a_skill("process_data_pipeline")
    async def process_data_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        # Extract data from Supabase
        extraction_result = await self.call_mcp_tool(
            "supabase",
            "execute_sql",
            project_id=self.mcp_config["supabase"]["project_id"],
            query=pipeline_config["extraction_query"]
        )
        
        # Process and validate data
        processed_data = self._process_raw_data(extraction_result.data["data"])
        
        # Store results back to Supabase
        storage_result = await self.call_mcp_tool(
            "supabase",
            "execute_sql",
            project_id=self.mcp_config["supabase"]["project_id"],
            query=self._generate_insert_query(processed_data, pipeline_config["target_table"])
        )
        
        return {
            "status": "completed",
            "records_processed": len(processed_data),
            "storage_result": storage_result.data
        }
```

#### 2. Deployment Agent Configuration

```python
# deployment_agent_config.py
DEPLOYMENT_AGENT_CONFIG = {
    "agent_name": "deployment_agent",
    "mcp_config": {
        "github": {
            "default_owner": "${GITHUB_DEFAULT_ORG}",
            "default_repo": "${GITHUB_DEFAULT_REPO}",
            "timeout": 60000,
            "retry_attempts": 2
        },
        "supabase": {
            "project_id": "${SUPABASE_DEFAULT_PROJECT_ID}",
            "timeout": 45000
        },
        "puppeteer": {
            "headless": True,
            "timeout": 30000,
            "viewport": {
                "width": 1920,
                "height": 1080
            }
        }
    },
    "deployment_environments": {
        "development": {
            "branch": "develop",
            "auto_deploy": True,
            "run_tests": True
        },
        "staging": {
            "branch": "staging",
            "auto_deploy": False,
            "run_tests": True,
            "require_approval": True
        },
        "production": {
            "branch": "main",
            "auto_deploy": False,
            "run_tests": True,
            "require_approval": True,
            "backup_required": True
        }
    }
}
```

#### 3. Testing Agent Configuration

```python
# testing_agent_config.py
TESTING_AGENT_CONFIG = {
    "agent_name": "testing_agent",
    "mcp_config": {
        "puppeteer": {
            "headless": "${PUPPETEER_HEADLESS}",
            "timeout": "${PUPPETEER_TIMEOUT}",
            "viewport": {
                "width": "${PUPPETEER_VIEWPORT_WIDTH}",
                "height": "${PUPPETEER_VIEWPORT_HEIGHT}"
            }
        },
        "supabase": {
            "project_id": "${SUPABASE_DEFAULT_PROJECT_ID}",
            "timeout": 30000
        },
        "perplexity": {
            "model": "llama-3.1-sonar-small-128k-online",
            "timeout": 20000
        }
    },
    "test_suites": {
        "ui_tests": {
            "enabled": True,
            "test_urls": [
                "/app/fioriLaunchpad.html",
                "/app/a2a-fiori/webapp/index.html"
            ],
            "test_scenarios": [
                "tile_interaction",
                "navigation_flow",
                "data_display"
            ]
        },
        "integration_tests": {
            "enabled": True,
            "test_apis": [
                "/api/v1/agents",
                "/api/v1/services",
                "/api/v1/workflows"
            ]
        },
        "performance_tests": {
            "enabled": True,
            "thresholds": {
                "page_load_time": 5000,
                "api_response_time": 1000,
                "memory_usage": "512MB"
            }
        }
    }
}
```

### Docker Configuration

#### Dockerfile for MCP-enabled A2A Agent

```dockerfile
# Dockerfile.a2a-agent-mcp
FROM python:3.9-slim

# Install Node.js for MCP servers
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY package.json .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies (MCP servers)
RUN npm install

# Copy application code
COPY . .

# Install MCP servers globally
RUN npm install -g \
    @modelcontextprotocol/server-github \
    @modelcontextprotocol/server-supabase \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-perplexity

# Create MCP configuration directory
RUN mkdir -p /app/config/mcp

# Copy MCP configuration
COPY mcp-config.json /app/config/mcp/

# Set environment variables
ENV PYTHONPATH=/app
ENV MCP_CONFIG_PATH=/app/config/mcp/mcp-config.json

# Expose ports
EXPOSE 8000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "a2a.agents.start_agent", "--agent-type", "mcp_enabled"]
```

#### Docker Compose for A2A Network with MCP

```yaml
# docker-compose.mcp.yml
version: '3.8'

services:
  a2a-agent-data-processing:
    build:
      context: .
      dockerfile: Dockerfile.a2a-agent-mcp
    environment:
      - AGENT_TYPE=data_processing
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - SUPABASE_ACCESS_TOKEN=${SUPABASE_ACCESS_TOKEN}
      - SUPABASE_DEFAULT_PROJECT_ID=${SUPABASE_DEFAULT_PROJECT_ID}
      - A2A_MCP_LOG_LEVEL=${A2A_MCP_LOG_LEVEL}
    volumes:
      - ./config/mcp:/app/config/mcp:ro
      - ./logs:/app/logs
    ports:
      - "8001:8000"
    networks:
      - a2a-network

  a2a-agent-deployment:
    build:
      context: .
      dockerfile: Dockerfile.a2a-agent-mcp
    environment:
      - AGENT_TYPE=deployment
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - SUPABASE_ACCESS_TOKEN=${SUPABASE_ACCESS_TOKEN}
      - PUPPETEER_HEADLESS=${PUPPETEER_HEADLESS}
    volumes:
      - ./config/mcp:/app/config/mcp:ro
      - ./logs:/app/logs
    ports:
      - "8002:8000"
    networks:
      - a2a-network

  a2a-agent-testing:
    build:
      context: .
      dockerfile: Dockerfile.a2a-agent-mcp
    environment:
      - AGENT_TYPE=testing
      - PUPPETEER_HEADLESS=${PUPPETEER_HEADLESS}
      - PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
      - SUPABASE_ACCESS_TOKEN=${SUPABASE_ACCESS_TOKEN}
    volumes:
      - ./config/mcp:/app/config/mcp:ro
      - ./logs:/app/logs
      - ./test-results:/app/test-results
    ports:
      - "8003:8000"
    networks:
      - a2a-network

  mcp-monitoring:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    networks:
      - a2a-network

networks:
  a2a-network:
    driver: bridge

volumes:
  grafana-storage:
```

### Monitoring and Logging Configuration

#### Logging Configuration

```python
# logging_config.py
import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "extra": %(extra)s}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/a2a-mcp.log',
            'mode': 'a'
        },
        'mcp_file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/mcp-operations.log',
            'mode': 'a'
        }
    },
    'loggers': {
        'a2a': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'a2a.mcp': {
            'level': 'DEBUG',
            'handlers': ['console', 'mcp_file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

def setup_logging():
    """Setup logging configuration for A2A MCP integration"""
    logging.config.dictConfig(LOGGING_CONFIG)
```

### Installation Script

```bash
#!/bin/bash
# install-mcp.sh

echo "🚀 Installing A2A Network MCP Integration..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Aborting." >&2; exit 1; }

# Install MCP servers
echo "📦 Installing MCP servers..."
npm install -g \
    @modelcontextprotocol/server-github \
    @modelcontextprotocol/server-supabase \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-perplexity

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

# Create configuration directories
echo "📁 Creating configuration directories..."
mkdir -p config/mcp
mkdir -p logs
mkdir -p test-results

# Create configuration files from templates
echo "⚙️ Setting up configuration files..."

# Create .env file
cat << 'EOF' > .env
# GitHub MCP Configuration
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_github_personal_access_token_here
GITHUB_DEFAULT_ORG=a2a-network
GITHUB_DEFAULT_REPO=a2a-agents

# Supabase MCP Configuration
SUPABASE_ACCESS_TOKEN=sbp_your_supabase_access_token_here
SUPABASE_DEFAULT_PROJECT_ID=your_project_id_here
SUPABASE_DEFAULT_ORG_ID=your_org_id_here

# Perplexity MCP Configuration
PERPLEXITY_API_KEY=pplx_your_perplexity_api_key_here

# Puppeteer MCP Configuration
PUPPETEER_HEADLESS=true
PUPPETEER_TIMEOUT=30000
PUPPETEER_VIEWPORT_WIDTH=1920
PUPPETEER_VIEWPORT_HEIGHT=1080

# A2A Network Specific
A2A_MCP_LOG_LEVEL=INFO
A2A_MCP_RETRY_ATTEMPTS=3
A2A_MCP_TIMEOUT=60000
EOF

# Create mcp-config.json
cat << EOF > config/mcp/mcp-config.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "\${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "supabase": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-supabase"
      ],
      "env": {
        "SUPABASE_ACCESS_TOKEN": "\${SUPABASE_ACCESS_TOKEN}"
      }
    },
    "puppeteer": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-puppeteer"
      ],
      "env": {
        "PUPPETEER_HEADLESS": "\${PUPPETEER_HEADLESS}",
        "PUPPETEER_TIMEOUT": "\${PUPPETEER_TIMEOUT}"
      }
    },
    "perplexity": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-perplexity"
      ],
      "env": {
        "PERPLEXITY_API_KEY": "\${PERPLEXITY_API_KEY}"
      }
    }
  }
}
EOF

# Set permissions
chmod +x scripts/*.sh
chmod 600 .env

echo "✅ A2A Network MCP integration installed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Review config/mcp/mcp-config.json"
echo "3. Run: python -m a2a.agents.start_agent --agent-type mcp_enabled"
echo ""
echo "📚 Documentation: docs/mcp/"
```

This comprehensive configuration guide provides everything needed to set up and configure MCP integration in the A2A Network project, from basic environment setup to advanced Docker deployments and monitoring configurations.
