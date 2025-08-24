# MCP Usage Patterns and Best Practices
## A2A Network Implementation Guide

### Core Usage Patterns

#### 1. Agent-to-Agent Communication via MCP
**Pattern**: Cross-agent skill invocation using MCP protocol
**Use Case**: When Agent A needs to leverage capabilities from Agent B

```python
# In Agent A (e.g., Data Standardization Agent)
class DataStandardizationAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.mcp_client = MCPClient("data_standardization")
        
    @a2a_skill("standardize_with_quality_check")
    async def standardize_with_quality_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Use MCP to call Quality Assessment Agent
        quality_result = await self.mcp_client.call_tool(
            agent="quality_assessment_agent",
            tool="assess_data_quality",
            parameters={
                "dataset": [data],
                "quality_dimensions": ["completeness", "accuracy", "consistency"]
            }
        )
        
        if quality_result["overall_score"] < 0.7:
            # Use MCP to call AI Preparation Agent for enhancement
            enhancement_result = await self.mcp_client.call_tool(
                agent="ai_preparation_agent", 
                tool="enhance_data_quality",
                parameters={"data": data, "quality_issues": quality_result["issues"]}
            )
            data = enhancement_result["enhanced_data"]
        
        # Proceed with standardization
        return await self._perform_standardization(data)
```

#### 2. External Service Integration Pattern
**Pattern**: Integrating external MCP servers for specialized functionality
**Use Case**: Using GitHub MCP for automated deployment processes

```python
# In Deployment Agent
class DeploymentAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.github_mcp = GitHubMCPClient()
        self.supabase_mcp = SupabaseMCPClient()
        
    @a2a_skill("automated_deployment")
    async def automated_deployment(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Create deployment branch
            branch_name = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            branch_result = await self.github_mcp.create_branch(
                owner=deployment_config["repo_owner"],
                repo=deployment_config["repo_name"],
                branch=branch_name,
                from_branch="main"
            )
            
            # Apply database migrations via Supabase MCP
            if deployment_config.get("migrations"):
                for migration in deployment_config["migrations"]:
                    await self.supabase_mcp.apply_migration(
                        project_id=deployment_config["supabase_project"],
                        name=migration["name"],
                        query=migration["sql"]
                    )
            
            # Push deployment files
            files = self._prepare_deployment_files(deployment_config)
            await self.github_mcp.push_files(
                owner=deployment_config["repo_owner"],
                repo=deployment_config["repo_name"],
                branch=branch_name,
                files=files,
                message=f"Automated deployment: {deployment_config['version']}"
            )
            
            # Create pull request
            pr_result = await self.github_mcp.create_pull_request(
                owner=deployment_config["repo_owner"],
                repo=deployment_config["repo_name"],
                title=f"Deployment v{deployment_config['version']}",
                head=branch_name,
                base="main",
                body=self._generate_pr_description(deployment_config)
            )
            
            return {
                "status": "success",
                "branch": branch_name,
                "pull_request": pr_result["html_url"],
                "deployment_id": deployment_config["version"]
            }
            
        except Exception as e:
            await self._handle_deployment_failure(e, deployment_config)
            raise
```

#### 3. Quality Assurance Pipeline Pattern
**Pattern**: Multi-stage validation using multiple MCP tools
**Use Case**: Comprehensive quality checks before data processing

```python
# In QA Validation Agent
class QAValidationAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.puppeteer_mcp = PuppeteerMCPClient()
        self.perplexity_mcp = PerplexityMCPClient()
        
    @a2a_skill("comprehensive_qa_validation")
    async def comprehensive_qa_validation(self, target_url: str, test_scenarios: List[Dict]) -> Dict[str, Any]:
        validation_results = {
            "ui_tests": [],
            "content_validation": [],
            "performance_metrics": {},
            "overall_score": 0.0
        }
        
        # Stage 1: UI Testing with Puppeteer MCP
        for scenario in test_scenarios:
            try:
                # Navigate to test page
                await self.puppeteer_mcp.navigate(
                    url=f"{target_url}/{scenario['path']}",
                    allowDangerous=False
                )
                
                # Take baseline screenshot
                screenshot = await self.puppeteer_mcp.screenshot(
                    name=f"test_{scenario['name']}",
                    width=1920,
                    height=1080
                )
                
                # Execute test interactions
                for action in scenario["actions"]:
                    if action["type"] == "click":
                        await self.puppeteer_mcp.click(selector=action["selector"])
                    elif action["type"] == "fill":
                        await self.puppeteer_mcp.fill(
                            selector=action["selector"],
                            value=action["value"]
                        )
                    elif action["type"] == "evaluate":
                        result = await self.puppeteer_mcp.evaluate(script=action["script"])
                        scenario["results"] = result
                
                validation_results["ui_tests"].append({
                    "scenario": scenario["name"],
                    "status": "passed",
                    "screenshot": screenshot,
                    "results": scenario.get("results", {})
                })
                
            except Exception as e:
                validation_results["ui_tests"].append({
                    "scenario": scenario["name"],
                    "status": "failed",
                    "error": str(e)
                })
        
        # Stage 2: Content Validation with Perplexity MCP
        for test in validation_results["ui_tests"]:
            if test["status"] == "passed" and "content" in test.get("results", {}):
                content_check = await self.perplexity_mcp.perplexity_ask(
                    messages=[{
                        "role": "user",
                        "content": f"Validate the accuracy of this content: {test['results']['content']}"
                    }]
                )
                
                validation_results["content_validation"].append({
                    "scenario": test["scenario"],
                    "validation": content_check,
                    "accuracy_score": self._extract_accuracy_score(content_check)
                })
        
        # Calculate overall score
        validation_results["overall_score"] = self._calculate_overall_qa_score(validation_results)
        
        return validation_results
```

#### 4. Data Pipeline Integration Pattern
**Pattern**: Chaining multiple MCP tools for data processing workflows
**Use Case**: End-to-end data processing with quality gates

```python
# In Data Processing Pipeline Agent
class DataPipelineAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.supabase_mcp = SupabaseMCPClient()
        self.quality_mcp = QualityAssessmentMCPClient()
        
    @a2a_skill("process_data_pipeline")
    async def process_data_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        pipeline_results = {
            "stages": [],
            "total_records": 0,
            "success_rate": 0.0,
            "quality_metrics": {}
        }
        
        # Stage 1: Data Extraction
        extraction_result = await self.supabase_mcp.execute_sql(
            project_id=pipeline_config["source_project"],
            query=pipeline_config["extraction_query"]
        )
        
        raw_data = extraction_result["data"]
        pipeline_results["total_records"] = len(raw_data)
        pipeline_results["stages"].append({
            "name": "extraction",
            "status": "completed",
            "records_processed": len(raw_data)
        })
        
        # Stage 2: Quality Assessment
        quality_result = await self.quality_mcp.assess_data_quality(
            dataset=raw_data,
            quality_dimensions=["completeness", "accuracy", "consistency", "validity"]
        )
        
        pipeline_results["quality_metrics"] = quality_result
        pipeline_results["stages"].append({
            "name": "quality_assessment",
            "status": "completed",
            "overall_score": quality_result["overall_score"]
        })
        
        # Quality Gate: Only proceed if quality score > threshold
        if quality_result["overall_score"] < pipeline_config.get("quality_threshold", 0.8):
            # Attempt data cleaning
            cleaned_data = await self._clean_data_based_on_quality_issues(
                raw_data, 
                quality_result["issues"]
            )
            
            # Re-assess quality
            quality_result = await self.quality_mcp.assess_data_quality(
                dataset=cleaned_data,
                quality_dimensions=["completeness", "accuracy", "consistency", "validity"]
            )
            
            if quality_result["overall_score"] < pipeline_config.get("quality_threshold", 0.8):
                raise ValueError(f"Data quality below threshold: {quality_result['overall_score']}")
            
            raw_data = cleaned_data
        
        # Stage 3: Data Transformation
        transformed_data = []
        for record in raw_data:
            try:
                transformed_record = await self._transform_record(record, pipeline_config["transformations"])
                transformed_data.append(transformed_record)
            except Exception as e:
                pipeline_results["stages"].append({
                    "name": "transformation_error",
                    "record_id": record.get("id", "unknown"),
                    "error": str(e)
                })
        
        pipeline_results["stages"].append({
            "name": "transformation",
            "status": "completed",
            "records_processed": len(transformed_data)
        })
        
        # Stage 4: Data Loading
        if transformed_data:
            # Create target table if needed
            if pipeline_config.get("create_table"):
                await self.supabase_mcp.apply_migration(
                    project_id=pipeline_config["target_project"],
                    name=f"create_table_{pipeline_config['target_table']}",
                    query=pipeline_config["table_schema"]
                )
            
            # Batch insert data
            insert_query = self._generate_batch_insert_query(
                transformed_data, 
                pipeline_config["target_table"]
            )
            
            load_result = await self.supabase_mcp.execute_sql(
                project_id=pipeline_config["target_project"],
                query=insert_query
            )
            
            pipeline_results["stages"].append({
                "name": "loading",
                "status": "completed",
                "records_loaded": len(transformed_data)
            })
        
        # Calculate success rate
        successful_stages = len([s for s in pipeline_results["stages"] if s["status"] == "completed"])
        total_stages = len(pipeline_results["stages"])
        pipeline_results["success_rate"] = successful_stages / total_stages if total_stages > 0 else 0.0
        
        return pipeline_results
```

### Advanced Patterns

#### 5. Event-Driven MCP Integration
**Pattern**: Reactive processing based on external events
**Use Case**: Automated response to GitHub webhooks

```python
# Event-driven GitHub integration
class GitHubEventProcessor(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.github_mcp = GitHubMCPClient()
        self.supabase_mcp = SupabaseMCPClient()
        
    async def handle_pull_request_event(self, event_data: Dict[str, Any]):
        if event_data["action"] == "opened":
            await self._process_new_pull_request(event_data["pull_request"])
        elif event_data["action"] == "closed" and event_data["pull_request"]["merged"]:
            await self._process_merged_pull_request(event_data["pull_request"])
    
    async def _process_new_pull_request(self, pr_data: Dict[str, Any]):
        # Get PR files and analyze changes
        files = await self.github_mcp.get_pull_request_files(
            owner=pr_data["base"]["repo"]["owner"]["login"],
            repo=pr_data["base"]["repo"]["name"],
            pull_number=pr_data["number"]
        )
        
        # Run automated code review
        review_comments = []
        for file in files:
            if file["filename"].endswith((".py", ".js", ".ts")):
                # Analyze code quality
                analysis = await self._analyze_code_quality(file["patch"])
                if analysis["issues"]:
                    review_comments.append({
                        "path": file["filename"],
                        "line": analysis["line"],
                        "body": f"Code quality issue: {analysis['issues'][0]['message']}"
                    })
        
        # Submit review if issues found
        if review_comments:
            await self.github_mcp.create_pull_request_review(
                owner=pr_data["base"]["repo"]["owner"]["login"],
                repo=pr_data["base"]["repo"]["name"],
                pull_number=pr_data["number"],
                event="REQUEST_CHANGES",
                body="Automated code review found issues that need attention.",
                comments=review_comments
            )
```

#### 6. Multi-Agent Coordination Pattern
**Pattern**: Orchestrating multiple agents via MCP
**Use Case**: Complex workflow requiring multiple specialized agents

```python
# Workflow orchestration agent
class WorkflowOrchestrator(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.agent_registry = {
            "data_standardization": MCPClient("data_standardization_agent"),
            "ai_preparation": MCPClient("ai_preparation_agent"),
            "vector_processing": MCPClient("vector_processing_agent"),
            "qa_validation": MCPClient("qa_validation_agent")
        }
    
    @a2a_skill("orchestrate_data_ingestion_workflow")
    async def orchestrate_data_ingestion_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        workflow_state = {
            "workflow_id": str(uuid.uuid4()),
            "stages": [],
            "current_stage": 0,
            "data": workflow_config["input_data"]
        }
        
        # Stage 1: Data Standardization
        standardization_result = await self.agent_registry["data_standardization"].call_tool(
            tool="l4_standardization_with_mcp",
            parameters={"entity_data": workflow_state["data"]}
        )
        
        if standardization_result["status"] != "success":
            raise WorkflowException("Data standardization failed", standardization_result)
        
        workflow_state["data"] = standardization_result["standardized_data"]
        workflow_state["stages"].append({
            "name": "standardization",
            "status": "completed",
            "quality_score": standardization_result["quality_score"]
        })
        
        # Stage 2: AI Preparation
        ai_prep_result = await self.agent_registry["ai_preparation"].call_tool(
            tool="ai_readiness_assessment",
            parameters={"entity_data": workflow_state["data"]}
        )
        
        if ai_prep_result["ai_readiness_score"] < 0.7:
            # Enhance data for AI readiness
            enhancement_result = await self.agent_registry["ai_preparation"].call_tool(
                tool="enhance_for_ai_readiness",
                parameters={
                    "entity_data": workflow_state["data"],
                    "readiness_issues": ai_prep_result["recommendations"]
                }
            )
            workflow_state["data"] = enhancement_result["enhanced_data"]
        
        workflow_state["stages"].append({
            "name": "ai_preparation",
            "status": "completed",
            "readiness_score": ai_prep_result["ai_readiness_score"]
        })
        
        # Stage 3: Vector Processing
        vector_result = await self.agent_registry["vector_processing"].call_tool(
            tool="enhanced_vector_ingestion",
            parameters={"ai_entity": workflow_state["data"]}
        )
        
        workflow_state["stages"].append({
            "name": "vector_processing",
            "status": "completed",
            "vector_id": vector_result.get("vector_id")
        })
        
        # Stage 4: QA Validation
        qa_result = await self.agent_registry["qa_validation"].call_tool(
            tool="validate_processed_data",
            parameters={
                "original_data": workflow_config["input_data"],
                "processed_data": workflow_state["data"],
                "vector_id": vector_result.get("vector_id")
            }
        )
        
        workflow_state["stages"].append({
            "name": "qa_validation",
            "status": "completed",
            "validation_score": qa_result["validation_score"]
        })
        
        # Calculate overall workflow success
        workflow_state["overall_success"] = all(
            stage["status"] == "completed" for stage in workflow_state["stages"]
        )
        
        return workflow_state
```

### Error Handling Patterns

#### 7. Resilient MCP Integration
**Pattern**: Robust error handling and recovery for MCP operations

```python
class ResilientMCPClient:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker = CircuitBreaker()
    
    async def call_with_retry(self, mcp_client, tool_name: str, **kwargs):
        """Call MCP tool with retry logic and circuit breaker"""
        
        @self.circuit_breaker
        async def _call():
            for attempt in range(self.max_retries):
                try:
                    result = await getattr(mcp_client, tool_name)(**kwargs)
                    return result
                except MCPTimeoutError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.backoff_factor ** attempt)
                except MCPRateLimitError as e:
                    # Exponential backoff for rate limits
                    wait_time = e.retry_after or (self.backoff_factor ** attempt)
                    await asyncio.sleep(wait_time)
                except MCPAuthenticationError as e:
                    # Don't retry auth errors
                    raise
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.backoff_factor ** attempt)
        
        return await _call()

# Usage in agent
class ResilientAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.resilient_client = ResilientMCPClient()
        self.github_mcp = GitHubMCPClient()
    
    @a2a_skill("resilient_github_operation")
    async def resilient_github_operation(self, operation_config: Dict[str, Any]):
        try:
            result = await self.resilient_client.call_with_retry(
                self.github_mcp,
                "create_pull_request",
                **operation_config
            )
            return {"status": "success", "result": result}
        except Exception as e:
            # Fallback to alternative approach
            return await self._fallback_operation(operation_config, error=str(e))
```

### Performance Optimization Patterns

#### 8. Batch Processing with MCP
**Pattern**: Efficient batch operations to minimize API calls

```python
class BatchMCPProcessor:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
    
    async def batch_process_github_operations(self, operations: List[Dict[str, Any]]):
        """Process multiple GitHub operations in optimized batches"""
        results = []
        
        # Group operations by type
        grouped_ops = {}
        for op in operations:
            op_type = op["type"]
            if op_type not in grouped_ops:
                grouped_ops[op_type] = []
            grouped_ops[op_type].append(op)
        
        # Process each type in batches
        for op_type, ops in grouped_ops.items():
            for i in range(0, len(ops), self.batch_size):
                batch = ops[i:i + self.batch_size]
                batch_results = await self._process_batch(op_type, batch)
                results.extend(batch_results)
        
        return results
    
    async def _process_batch(self, op_type: str, batch: List[Dict[str, Any]]):
        """Process a batch of operations concurrently"""
        if op_type == "create_issue":
            tasks = [
                self.github_mcp.create_issue(**op["params"])
                for op in batch
            ]
        elif op_type == "update_file":
            # Combine multiple file updates into single push
            files = [{"path": op["params"]["path"], "content": op["params"]["content"]} 
                    for op in batch]
            return [await self.github_mcp.push_files(
                owner=batch[0]["params"]["owner"],
                repo=batch[0]["params"]["repo"],
                branch=batch[0]["params"]["branch"],
                files=files,
                message="Batch file update"
            )]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### Testing Patterns

#### 9. MCP Integration Testing
**Pattern**: Comprehensive testing of MCP integrations

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestMCPIntegrations:
    @pytest.fixture
    async def mock_github_mcp(self):
        mock = AsyncMock()
        mock.create_pull_request.return_value = {
            "id": 123,
            "html_url": "https://github.com/test/repo/pull/123",
            "number": 123
        }
        return mock
    
    @pytest.mark.asyncio
    async def test_deployment_agent_success_flow(self, mock_github_mcp):
        """Test successful deployment flow"""
        with patch('deployment_agent.GitHubMCPClient', return_value=mock_github_mcp):
            agent = DeploymentAgent()
            
            config = {
                "repo_owner": "test-org",
                "repo_name": "test-repo",
                "version": "1.0.0",
                "migrations": []
            }
            
            result = await agent.automated_deployment(config)
            
            assert result["status"] == "success"
            assert "pull_request" in result
            mock_github_mcp.create_branch.assert_called_once()
            mock_github_mcp.create_pull_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling(self, mock_github_mcp):
        """Test MCP error handling"""
        mock_github_mcp.create_branch.side_effect = Exception("API Error")
        
        with patch('deployment_agent.GitHubMCPClient', return_value=mock_github_mcp):
            agent = DeploymentAgent()
            
            config = {"repo_owner": "test-org", "repo_name": "test-repo", "version": "1.0.0"}
            
            with pytest.raises(Exception):
                await agent.automated_deployment(config)
```

### Monitoring and Observability

#### 10. MCP Operations Monitoring
**Pattern**: Comprehensive monitoring of MCP tool usage

```python
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class MCPMetrics:
    tool_name: str
    duration: float
    success: bool
    error: Optional[str] = None
    response_size: Optional[int] = None

class MCPMonitor:
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger("a2a.mcp.monitor")
    
    async def monitored_call(self, mcp_client, tool_name: str, **kwargs):
        """Wrapper for MCP calls with monitoring"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting MCP call: {tool_name}", extra={
                "tool": tool_name,
                "parameters": kwargs
            })
            
            result = await getattr(mcp_client, tool_name)(**kwargs)
            
            duration = time.time() - start_time
            response_size = len(str(result)) if result else 0
            
            metric = MCPMetrics(
                tool_name=tool_name,
                duration=duration,
                success=True,
                response_size=response_size
            )
            
            self.metrics.append(metric)
            
            self.logger.info(f"MCP call completed: {tool_name}", extra={
                "tool": tool_name,
                "duration": duration,
                "success": True,
                "response_size": response_size
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            metric = MCPMetrics(
                tool_name=tool_name,
                duration=duration,
                success=False,
                error=str(e)
            )
            
            self.metrics.append(metric)
            
            self.logger.error(f"MCP call failed: {tool_name}", extra={
                "tool": tool_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from collected metrics"""
        if not self.metrics:
            return {"message": "No metrics collected"}
        
        total_calls = len(self.metrics)
        successful_calls = len([m for m in self.metrics if m.success])
        
        avg_duration = sum(m.duration for m in self.metrics) / total_calls
        
        tool_stats = {}
        for metric in self.metrics:
            if metric.tool_name not in tool_stats:
                tool_stats[metric.tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "total_duration": 0,
                    "errors": []
                }
            
            stats = tool_stats[metric.tool_name]
            stats["total_calls"] += 1
            if metric.success:
                stats["successful_calls"] += 1
            stats["total_duration"] += metric.duration
            if metric.error:
                stats["errors"].append(metric.error)
        
        # Calculate averages
        for tool, stats in tool_stats.items():
            stats["avg_duration"] = stats["total_duration"] / stats["total_calls"]
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
        
        return {
            "total_calls": total_calls,
            "success_rate": successful_calls / total_calls,
            "avg_duration": avg_duration,
            "tool_statistics": tool_stats
        }
```

These patterns provide a comprehensive foundation for implementing MCP integrations in the A2A Network project, covering everything from basic tool usage to advanced orchestration, error handling, and monitoring.
