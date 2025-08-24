# MCP Integration Examples for A2A Network
## Real-World Implementation Scenarios

### 1. SAP Fiori Launchpad Testing with MCP

Based on the successful A2A Network launchpad testing mentioned in the memories, here's how to implement comprehensive UI testing using MCP tools:

```python
# SAP Fiori Launchpad Testing Agent
class FioriLaunchpadTestingAgent(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.puppeteer_mcp = PuppeteerMCPClient()
        self.supabase_mcp = SupabaseMCPClient()
        
    @a2a_skill("comprehensive_launchpad_testing")
    async def comprehensive_launchpad_testing(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive testing of SAP Fiori Launchpad functionality
        Based on successful A2A Network implementation
        """
        test_results = {
            "ui_validation": {},
            "backend_integration": {},
            "performance_metrics": {},
            "tile_functionality": {},
            "overall_score": 0.0
        }
        
        base_url = test_config.get("base_url", "http://localhost:4005")
        
        # Stage 1: Navigate to launchpad and validate loading
        await self.puppeteer_mcp.navigate(
            url=f"{base_url}/app/fioriLaunchpad.html",
            allowDangerous=False
        )
        
        # Wait for SAP UI5 to load
        await asyncio.sleep(3)
        
        # Take baseline screenshot
        baseline_screenshot = await self.puppeteer_mcp.screenshot(
            name="launchpad_baseline",
            width=1920,
            height=1080
        )
        
        # Stage 2: Validate tile presence and data
        tile_validation = await self.puppeteer_mcp.evaluate(
            script="""
            return {
                tiles_count: document.querySelectorAll('.sapMGT').length,
                agent_count: document.querySelector('[data-tile="agent-management"] .sapMGTNumValue')?.textContent || '0',
                service_count: document.querySelector('[data-tile="service-marketplace"] .sapMGTNumValue')?.textContent || '0',
                workflow_count: document.querySelector('[data-tile="workflow-designer"] .sapMGTNumValue')?.textContent || '0',
                sap_loaded: !!window.sap,
                ushell_loaded: !!window['sap-ushell-config']
            }
            """
        )
        
        test_results["ui_validation"] = {
            "tiles_present": tile_validation["tiles_count"] >= 6,
            "sap_framework_loaded": tile_validation["sap_loaded"],
            "ushell_configured": tile_validation["ushell_loaded"],
            "screenshot": baseline_screenshot
        }
        
        # Stage 3: Test backend data integration
        # Verify real data is displayed (not hardcoded values)
        backend_data = await self.supabase_mcp.execute_sql(
            project_id=test_config["project_id"],
            query="SELECT COUNT(*) as agent_count FROM agents WHERE status = 'active'"
        )
        
        actual_agent_count = str(backend_data["data"][0]["agent_count"])
        displayed_agent_count = tile_validation["agent_count"]
        
        test_results["backend_integration"] = {
            "data_sync_accurate": actual_agent_count == displayed_agent_count,
            "actual_count": actual_agent_count,
            "displayed_count": displayed_agent_count,
            "api_responsive": True
        }
        
        # Stage 4: Test tile interactions
        tile_tests = []
        test_tiles = [
            {"selector": "[data-tile='agent-management']", "name": "Agent Management"},
            {"selector": "[data-tile='service-marketplace']", "name": "Service Marketplace"},
            {"selector": "[data-tile='workflow-designer']", "name": "Workflow Designer"}
        ]
        
        for tile in test_tiles:
            try:
                # Test tile click
                await self.puppeteer_mcp.click(selector=tile["selector"])
                
                # Verify navigation or modal opening
                navigation_result = await self.puppeteer_mcp.evaluate(
                    script="return { url: window.location.href, modals: document.querySelectorAll('.sapMDialog').length }"
                )
                
                tile_tests.append({
                    "tile": tile["name"],
                    "clickable": True,
                    "navigation_changed": navigation_result["url"] != f"{base_url}/app/fioriLaunchpad.html",
                    "modal_opened": navigation_result["modals"] > 0
                })
                
                # Return to launchpad if navigated away
                if navigation_result["url"] != f"{base_url}/app/fioriLaunchpad.html":
                    await self.puppeteer_mcp.navigate(url=f"{base_url}/app/fioriLaunchpad.html")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                tile_tests.append({
                    "tile": tile["name"],
                    "clickable": False,
                    "error": str(e)
                })
        
        test_results["tile_functionality"] = tile_tests
        
        # Stage 5: Performance testing
        performance_metrics = await self.puppeteer_mcp.evaluate(
            script="""
            return {
                load_time: performance.timing.loadEventEnd - performance.timing.navigationStart,
                dom_ready: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                tiles_rendered: document.querySelectorAll('.sapMGTStateLoaded').length,
                interactive_elements: document.querySelectorAll('[tabindex], button, a, input').length
            }
            """
        )
        
        test_results["performance_metrics"] = performance_metrics
        
        # Calculate overall score
        ui_score = 1.0 if test_results["ui_validation"]["tiles_present"] and test_results["ui_validation"]["sap_framework_loaded"] else 0.0
        backend_score = 1.0 if test_results["backend_integration"]["data_sync_accurate"] else 0.0
        tile_score = len([t for t in tile_tests if t.get("clickable", False)]) / len(tile_tests) if tile_tests else 0.0
        performance_score = 1.0 if performance_metrics["load_time"] < 5000 else 0.5
        
        test_results["overall_score"] = (ui_score + backend_score + tile_score + performance_score) / 4
        
        return test_results
```

### 2. Automated A2A Agent Deployment Pipeline

```python
# A2A Agent Deployment Pipeline using GitHub and Supabase MCP
class A2ADeploymentPipeline(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.github_mcp = GitHubMCPClient()
        self.supabase_mcp = SupabaseMCPClient()
        self.puppeteer_mcp = PuppeteerMCPClient()
        
    @a2a_skill("deploy_a2a_agent_update")
    async def deploy_a2a_agent_update(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete A2A agent deployment pipeline with testing and validation
        """
        deployment_results = {
            "deployment_id": str(uuid.uuid4()),
            "stages": [],
            "status": "in_progress",
            "rollback_info": {}
        }
        
        try:
            # Stage 1: Create deployment branch
            branch_name = f"deploy-agent-{deployment_config['agent_name']}-{int(time.time())}"
            
            branch_result = await self.github_mcp.create_branch(
                owner="a2a-network",
                repo="a2a-agents",
                branch=branch_name,
                from_branch="main"
            )
            
            deployment_results["stages"].append({
                "name": "branch_creation",
                "status": "completed",
                "branch_name": branch_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Stage 2: Apply database migrations if needed
            if deployment_config.get("migrations"):
                for migration in deployment_config["migrations"]:
                    migration_result = await self.supabase_mcp.apply_migration(
                        project_id=deployment_config["supabase_project"],
                        name=f"{deployment_config['agent_name']}_{migration['name']}",
                        query=migration["sql"]
                    )
                    
                    deployment_results["stages"].append({
                        "name": f"migration_{migration['name']}",
                        "status": "completed",
                        "migration_id": migration_result.get("id"),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Stage 3: Update agent files
            agent_files = []
            for file_update in deployment_config["file_updates"]:
                agent_files.append({
                    "path": file_update["path"],
                    "content": file_update["content"]
                })
            
            push_result = await self.github_mcp.push_files(
                owner="a2a-network",
                repo="a2a-agents",
                branch=branch_name,
                files=agent_files,
                message=f"Deploy {deployment_config['agent_name']} v{deployment_config['version']}"
            )
            
            deployment_results["stages"].append({
                "name": "file_updates",
                "status": "completed",
                "files_updated": len(agent_files),
                "commit_sha": push_result.get("commit", {}).get("sha"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Stage 4: Create pull request
            pr_body = f"""
# A2A Agent Deployment: {deployment_config['agent_name']} v{deployment_config['version']}

## Changes
{deployment_config.get('description', 'Automated agent deployment')}

## Files Modified
{chr(10).join(f"- {f['path']}" for f in agent_files)}

## Database Migrations
{chr(10).join(f"- {m['name']}: {m['description']}" for m in deployment_config.get('migrations', []))}

## Testing
- [ ] Automated tests passed
- [ ] Integration tests completed
- [ ] Performance validation completed

## Deployment Checklist
- [x] Branch created: {branch_name}
- [x] Files updated
- [x] Database migrations applied
- [ ] Pull request reviewed
- [ ] Deployment approved
            """
            
            pr_result = await self.github_mcp.create_pull_request(
                owner="a2a-network",
                repo="a2a-agents",
                title=f"Deploy {deployment_config['agent_name']} v{deployment_config['version']}",
                head=branch_name,
                base="main",
                body=pr_body,
                draft=False
            )
            
            deployment_results["stages"].append({
                "name": "pull_request_creation",
                "status": "completed",
                "pr_number": pr_result["number"],
                "pr_url": pr_result["html_url"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Stage 5: Automated testing
            if deployment_config.get("run_tests", True):
                test_results = await self._run_automated_tests(
                    deployment_config,
                    branch_name,
                    pr_result["number"]
                )
                
                deployment_results["stages"].append({
                    "name": "automated_testing",
                    "status": "completed" if test_results["success"] else "failed",
                    "test_results": test_results,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Add test results as PR comment
                test_comment = f"""
## Automated Test Results

**Overall Status**: {'✅ PASSED' if test_results['success'] else '❌ FAILED'}

### Test Summary
- **Unit Tests**: {test_results.get('unit_tests', {}).get('status', 'Not Run')}
- **Integration Tests**: {test_results.get('integration_tests', {}).get('status', 'Not Run')}
- **Performance Tests**: {test_results.get('performance_tests', {}).get('status', 'Not Run')}

### Details
```json
{json.dumps(test_results, indent=2)}
```
                """
                
                await self.github_mcp.add_issue_comment(
                    owner="a2a-network",
                    repo="a2a-agents",
                    issue_number=pr_result["number"],
                    body=test_comment
                )
            
            # Stage 6: Auto-merge if all tests pass and auto-deploy is enabled
            if (deployment_config.get("auto_deploy", False) and 
                test_results.get("success", False)):
                
                merge_result = await self.github_mcp.merge_pull_request(
                    owner="a2a-network",
                    repo="a2a-agents",
                    pull_number=pr_result["number"],
                    commit_title=f"Deploy {deployment_config['agent_name']} v{deployment_config['version']}",
                    commit_message="Automated deployment with successful test validation",
                    merge_method="squash"
                )
                
                deployment_results["stages"].append({
                    "name": "auto_merge",
                    "status": "completed",
                    "merge_sha": merge_result.get("sha"),
                    "timestamp": datetime.now().isoformat()
                })
                
                deployment_results["status"] = "completed"
            else:
                deployment_results["status"] = "pending_review"
            
            return deployment_results
            
        except Exception as e:
            # Rollback on failure
            deployment_results["status"] = "failed"
            deployment_results["error"] = str(e)
            
            # Attempt to close PR and delete branch
            if "pr_number" in [s.get("pr_number") for s in deployment_results["stages"] if s.get("pr_number")]:
                try:
                    await self.github_mcp.update_issue(
                        owner="a2a-network",
                        repo="a2a-agents",
                        issue_number=pr_result["number"],
                        state="closed"
                    )
                except:
                    pass
            
            return deployment_results
    
    async def _run_automated_tests(self, deployment_config: Dict[str, Any], branch_name: str, pr_number: int) -> Dict[str, Any]:
        """Run comprehensive automated tests for the deployment"""
        test_results = {
            "success": False,
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {}
        }
        
        # Unit tests via GitHub Actions or direct execution
        # This would trigger the CI/CD pipeline
        
        # Integration tests using Puppeteer MCP
        if deployment_config.get("test_ui", False):
            ui_test_results = await self._run_ui_integration_tests(deployment_config)
            test_results["integration_tests"]["ui"] = ui_test_results
        
        # Database integration tests
        if deployment_config.get("test_database", False):
            db_test_results = await self._run_database_tests(deployment_config)
            test_results["integration_tests"]["database"] = db_test_results
        
        # Performance tests
        if deployment_config.get("test_performance", False):
            perf_test_results = await self._run_performance_tests(deployment_config)
            test_results["performance_tests"] = perf_test_results
        
        # Determine overall success
        test_results["success"] = all([
            test_results["integration_tests"].get("ui", {}).get("success", True),
            test_results["integration_tests"].get("database", {}).get("success", True),
            test_results["performance_tests"].get("success", True)
        ])
        
        return test_results
```

### 3. A2A Network Analytics Dashboard Integration

```python
# Analytics Dashboard with Real-time Data via MCP
class A2AAnalyticsDashboard(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.supabase_mcp = SupabaseMCPClient()
        self.github_mcp = GitHubMCPClient()
        self.perplexity_mcp = PerplexityMCPClient()
        
    @a2a_skill("generate_network_analytics")
    async def generate_network_analytics(self, analytics_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive A2A Network analytics using multiple MCP sources
        """
        analytics_data = {
            "timestamp": datetime.now().isoformat(),
            "network_health": {},
            "agent_performance": {},
            "development_metrics": {},
            "user_engagement": {},
            "insights": []
        }
        
        # Stage 1: Network Health Metrics from Supabase
        network_health_query = """
        SELECT 
            COUNT(CASE WHEN status = 'active' THEN 1 END) as active_agents,
            COUNT(CASE WHEN status = 'inactive' THEN 1 END) as inactive_agents,
            AVG(performance_score) as avg_performance,
            COUNT(CASE WHEN last_activity > NOW() - INTERVAL '1 hour' THEN 1 END) as recently_active
        FROM agents
        """
        
        health_result = await self.supabase_mcp.execute_sql(
            project_id=analytics_config["project_id"],
            query=network_health_query
        )
        
        health_data = health_result["data"][0]
        analytics_data["network_health"] = {
            "total_agents": health_data["active_agents"] + health_data["inactive_agents"],
            "active_agents": health_data["active_agents"],
            "inactive_agents": health_data["inactive_agents"],
            "avg_performance": float(health_data["avg_performance"] or 0),
            "recently_active": health_data["recently_active"],
            "health_score": self._calculate_health_score(health_data)
        }
        
        # Stage 2: Agent Performance Analytics
        performance_query = """
        SELECT 
            agent_name,
            performance_score,
            success_rate,
            avg_response_time,
            total_requests,
            error_count,
            last_activity
        FROM agents 
        WHERE status = 'active'
        ORDER BY performance_score DESC
        """
        
        performance_result = await self.supabase_mcp.execute_sql(
            project_id=analytics_config["project_id"],
            query=performance_query
        )
        
        analytics_data["agent_performance"] = {
            "top_performers": performance_result["data"][:5],
            "performance_distribution": self._analyze_performance_distribution(performance_result["data"]),
            "bottlenecks": self._identify_performance_bottlenecks(performance_result["data"])
        }
        
        # Stage 3: Development Metrics from GitHub
        # Get recent commits and PR activity
        commits_result = await self.github_mcp.list_commits(
            owner="a2a-network",
            repo="a2a-agents",
            sha="main",
            page=1,
            perPage=50
        )
        
        prs_result = await self.github_mcp.list_pull_requests(
            owner="a2a-network",
            repo="a2a-agents",
            state="all",
            page=1,
            per_page=20
        )
        
        analytics_data["development_metrics"] = {
            "recent_commits": len(commits_result),
            "active_prs": len([pr for pr in prs_result if pr["state"] == "open"]),
            "merged_prs_last_week": len([
                pr for pr in prs_result 
                if pr["state"] == "closed" and pr["merged_at"] and
                datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00')) > 
                datetime.now(timezone.utc) - timedelta(days=7)
            ]),
            "development_velocity": self._calculate_development_velocity(commits_result, prs_result)
        }
        
        # Stage 4: Generate AI-powered insights
        insights_prompt = f"""
        Analyze the following A2A Network metrics and provide actionable insights:
        
        Network Health:
        - Active Agents: {analytics_data['network_health']['active_agents']}
        - Average Performance: {analytics_data['network_health']['avg_performance']:.2f}
        - Health Score: {analytics_data['network_health']['health_score']:.2f}
        
        Development Activity:
        - Recent Commits: {analytics_data['development_metrics']['recent_commits']}
        - Active PRs: {analytics_data['development_metrics']['active_prs']}
        - Development Velocity: {analytics_data['development_metrics']['development_velocity']:.2f}
        
        Provide 3-5 specific, actionable insights for improving the A2A Network.
        """
        
        insights_response = await self.perplexity_mcp.perplexity_ask(
            messages=[{
                "role": "user",
                "content": insights_prompt
            }]
        )
        
        analytics_data["insights"] = self._parse_insights_from_response(insights_response)
        
        # Stage 5: Store analytics in database for historical tracking
        analytics_insert_query = """
        INSERT INTO analytics_snapshots (
            timestamp, network_health, agent_performance, 
            development_metrics, insights
        ) VALUES ($1, $2, $3, $4, $5)
        """
        
        await self.supabase_mcp.execute_sql(
            project_id=analytics_config["project_id"],
            query=analytics_insert_query,
            params=[
                analytics_data["timestamp"],
                json.dumps(analytics_data["network_health"]),
                json.dumps(analytics_data["agent_performance"]),
                json.dumps(analytics_data["development_metrics"]),
                json.dumps(analytics_data["insights"])
            ]
        )
        
        return analytics_data
    
    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall network health score"""
        total_agents = health_data["active_agents"] + health_data["inactive_agents"]
        if total_agents == 0:
            return 0.0
        
        active_ratio = health_data["active_agents"] / total_agents
        performance_score = (health_data["avg_performance"] or 0) / 100
        activity_ratio = health_data["recently_active"] / max(health_data["active_agents"], 1)
        
        return (active_ratio * 0.4 + performance_score * 0.4 + activity_ratio * 0.2) * 100
```

### 4. Automated Security Scanning Pipeline

```python
# Security scanning using multiple MCP tools
class A2ASecurityScanner(A2AAgentBase):
    def __init__(self):
        super().__init__()
        self.github_mcp = GitHubMCPClient()
        self.supabase_mcp = SupabaseMCPClient()
        
    @a2a_skill("comprehensive_security_scan")
    async def comprehensive_security_scan(self, scan_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive security scanning of A2A Network components
        """
        scan_results = {
            "scan_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "code_security": {},
            "database_security": {},
            "infrastructure_security": {},
            "vulnerability_summary": {},
            "remediation_plan": []
        }
        
        # Stage 1: Code Security Analysis via GitHub
        # Search for potential security issues in code
        security_patterns = [
            "password",
            "secret",
            "api_key",
            "private_key",
            "token",
            "credential"
        ]
        
        code_vulnerabilities = []
        for pattern in security_patterns:
            search_result = await self.github_mcp.search_code(
                q=f"{pattern} repo:a2a-network/a2a-agents",
                per_page=10
            )
            
            for item in search_result.get("items", []):
                code_vulnerabilities.append({
                    "file": item["path"],
                    "pattern": pattern,
                    "repository": item["repository"]["name"],
                    "url": item["html_url"],
                    "severity": self._assess_vulnerability_severity(pattern, item)
                })
        
        scan_results["code_security"] = {
            "vulnerabilities_found": len(code_vulnerabilities),
            "high_severity": len([v for v in code_vulnerabilities if v["severity"] == "high"]),
            "medium_severity": len([v for v in code_vulnerabilities if v["severity"] == "medium"]),
            "low_severity": len([v for v in code_vulnerabilities if v["severity"] == "low"]),
            "details": code_vulnerabilities
        }
        
        # Stage 2: Database Security Analysis via Supabase
        # Get security advisors
        security_advisors = await self.supabase_mcp.get_advisors(
            project_id=scan_config["project_id"],
            type="security"
        )
        
        # Check for missing RLS policies
        rls_check_query = """
        SELECT schemaname, tablename, rowsecurity 
        FROM pg_tables 
        WHERE schemaname = 'public' AND rowsecurity = false
        """
        
        rls_result = await self.supabase_mcp.execute_sql(
            project_id=scan_config["project_id"],
            query=rls_check_query
        )
        
        scan_results["database_security"] = {
            "advisor_recommendations": len(security_advisors.get("advisors", [])),
            "tables_without_rls": len(rls_result["data"]),
            "rls_violations": rls_result["data"],
            "security_score": self._calculate_db_security_score(security_advisors, rls_result)
        }
        
        # Stage 3: Generate remediation plan
        remediation_tasks = []
        
        # Code security remediations
        for vuln in code_vulnerabilities:
            if vuln["severity"] in ["high", "medium"]:
                remediation_tasks.append({
                    "priority": vuln["severity"],
                    "type": "code_security",
                    "description": f"Review and secure {vuln['pattern']} in {vuln['file']}",
                    "file": vuln["file"],
                    "action": "manual_review"
                })
        
        # Database security remediations
        for table in rls_result["data"]:
            remediation_tasks.append({
                "priority": "high",
                "type": "database_security",
                "description": f"Enable RLS for table {table['tablename']}",
                "table": table["tablename"],
                "action": "enable_rls"
            })
        
        scan_results["remediation_plan"] = sorted(
            remediation_tasks, 
            key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], 
            reverse=True
        )
        
        # Stage 4: Create GitHub issues for high-priority vulnerabilities
        if scan_config.get("create_issues", False):
            high_priority_issues = [
                task for task in remediation_tasks 
                if task["priority"] == "high"
            ]
            
            for task in high_priority_issues[:5]:  # Limit to top 5
                issue_title = f"Security: {task['description']}"
                issue_body = f"""
# Security Vulnerability

**Priority**: {task['priority'].upper()}
**Type**: {task['type']}
**Scan ID**: {scan_results['scan_id']}

## Description
{task['description']}

## Recommended Action
{task['action']}

## Additional Context
This issue was automatically created by the A2A Security Scanner.
Scan timestamp: {scan_results['timestamp']}
                """
                
                await self.github_mcp.create_issue(
                    owner="a2a-network",
                    repo="a2a-agents",
                    title=issue_title,
                    body=issue_body,
                    labels=["security", "automated", task["priority"]]
                )
        
        return scan_results
```

These integration examples demonstrate real-world usage of MCP tools within the A2A Network project, covering testing, deployment, analytics, and security scenarios based on the successful implementations mentioned in the memories.
