"""
Unit tests for SQL Agent
Tests NL2SQL and SQL2NL conversion capabilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the SQL Agent
from a2aAgents.backend.app.a2a.agents.sqlAgent.active.sqlAgentSdk import SQLAgentSDK
from a2aAgents.backend.app.a2a.sdk import A2AMessage, MessagePart, MessageRole


class TestSQLAgent:
    """Test suite for SQL Agent"""
    
    @pytest.fixture
    async def sql_agent(self):
        """Create SQL agent instance for testing"""
        agent = SQLAgentSDK(
            base_url="http://localhost:8150",
            enable_monitoring=False  # Disable monitoring for tests
        )
        # Mock trust system
        agent.trust_identity = {"mock": True}
        agent.trusted_agents = {"test_agent"}
        
        # Initialize without blockchain
        agent._initialize_blockchain_integration = AsyncMock()
        await agent.initialize()
        
        yield agent
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, sql_agent):
        """Test agent initializes correctly"""
        assert sql_agent.agent_id == "sql_agent"
        assert sql_agent.name == "A2A SQL Agent"
        assert "nl2sql" in [h for h in sql_agent.handlers.keys()]
        assert "sql2nl" in [h for h in sql_agent.handlers.keys()]
        assert len(sql_agent.skills) > 0
    
    @pytest.mark.asyncio
    async def test_relational_nl2sql_simple(self, sql_agent):
        """Test simple relational NL to SQL conversion"""
        # Create test message
        message = A2AMessage(
            messageId="test_1",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "show all customers",
                        "query_type": "relational"
                    }
                )
            ]
        )
        
        # Process message
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        assert "sql_query" in result["data"]["result"]
        assert "SELECT" in result["data"]["result"]["sql_query"]
        assert "FROM customers" in result["data"]["result"]["sql_query"]
    
    @pytest.mark.asyncio
    async def test_relational_nl2sql_with_filter(self, sql_agent):
        """Test NL to SQL with WHERE clause"""
        message = A2AMessage(
            messageId="test_2",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "find orders where status is completed",
                        "query_type": "relational"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "WHERE status = 'completed'" in sql_query
    
    @pytest.mark.asyncio
    async def test_relational_nl2sql_aggregation(self, sql_agent):
        """Test NL to SQL with aggregation"""
        message = A2AMessage(
            messageId="test_3",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "count of products",
                        "query_type": "relational"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "COUNT(*)" in sql_query
        assert "FROM products" in sql_query
    
    @pytest.mark.asyncio
    async def test_graph_nl2sql_path(self, sql_agent):
        """Test graph query for path finding"""
        message = A2AMessage(
            messageId="test_4",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "find path from Alice to Bob",
                        "query_type": "graph"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "MATCH" in sql_query
        assert "Alice" in sql_query
        assert "Bob" in sql_query
    
    @pytest.mark.asyncio
    async def test_graph_nl2sql_neighbors(self, sql_agent):
        """Test graph query for neighbors"""
        message = A2AMessage(
            messageId="test_5",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "find neighbors of Node123",
                        "query_type": "graph"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "MATCH" in sql_query
        assert "Node123" in sql_query
        assert "-[r]-(neighbor)" in sql_query
    
    @pytest.mark.asyncio
    async def test_vector_nl2sql_similarity(self, sql_agent):
        """Test vector similarity search"""
        message = A2AMessage(
            messageId="test_6",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "find 5 most similar documents to machine learning",
                        "query_type": "vector",
                        "schema": {
                            "default_table": "documents",
                            "vector_column": "embedding"
                        }
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "SELECT TOP 5" in sql_query
        assert "COSINE_SIMILARITY" in sql_query
        assert "embedding" in sql_query
        assert "machine learning" in sql_query
    
    @pytest.mark.asyncio
    async def test_vector_nl2sql_knn(self, sql_agent):
        """Test k-nearest neighbors search"""
        message = A2AMessage(
            messageId="test_7",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "3 nearest products to iPhone",
                        "query_type": "vector"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        
        assert result["success"] is True
        sql_query = result["data"]["result"]["sql_query"]
        assert "SELECT TOP 3" in sql_query
        assert "L2DISTANCE" in sql_query
        assert "iPhone" in sql_query
    
    @pytest.mark.asyncio
    async def test_sql2nl_simple_select(self, sql_agent):
        """Test SQL to natural language for simple SELECT"""
        message = A2AMessage(
            messageId="test_8",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "sql2nl",
                        "query": "SELECT * FROM customers WHERE country = 'USA'"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_sql2nl_request(message, "test_context")
        
        assert result["success"] is True
        nl_text = result["data"]["result"]["natural_language"]
        assert "retrieves data" in nl_text.lower()
        assert "customers" in nl_text.lower()
        assert "filtering" in nl_text.lower() or "conditions" in nl_text.lower()
    
    @pytest.mark.asyncio
    async def test_sql2nl_join_query(self, sql_agent):
        """Test SQL to natural language for JOIN query"""
        message = A2AMessage(
            messageId="test_9",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "sql2nl",
                        "query": "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_sql2nl_request(message, "test_context")
        
        assert result["success"] is True
        nl_text = result["data"]["result"]["natural_language"]
        assert "combining" in nl_text.lower() or "join" in nl_text.lower()
        assert "orders" in nl_text.lower()
        assert "customers" in nl_text.lower()
    
    @pytest.mark.asyncio
    async def test_sql2nl_graph_query(self, sql_agent):
        """Test SQL to natural language for graph query"""
        message = A2AMessage(
            messageId="test_10",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "sql2nl",
                        "query": "MATCH p = shortestPath((a {id: '1'})-[*]-(b {id: '2'})) RETURN p"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_sql2nl_request(message, "test_context")
        
        assert result["success"] is True
        nl_text = result["data"]["result"]["natural_language"]
        assert "graph" in nl_text.lower()
        assert "shortest path" in nl_text.lower()
    
    @pytest.mark.asyncio
    async def test_auto_detect_query_type(self, sql_agent):
        """Test automatic query type detection"""
        # Test vector detection
        message = A2AMessage(
            messageId="test_11",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "find similar products to laptop",
                        "query_type": "auto"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        assert result["success"] is True
        assert result["data"]["result"]["query_type"] == "vector"
        
        # Test graph detection
        message.parts[0].data["query"] = "find shortest path from A to B"
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        assert result["success"] is True
        assert result["data"]["result"]["query_type"] == "graph"
        
        # Test relational detection
        message.parts[0].data["query"] = "show all orders from last month"
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        assert result["success"] is True
        assert result["data"]["result"]["query_type"] == "relational"
    
    @pytest.mark.asyncio
    async def test_skill_execution(self, sql_agent):
        """Test direct skill execution"""
        # Test relational skill
        result = await sql_agent.execute_skill("relational_nl2sql", {
            "nl_query": "count customers by country",
            "schema_info": {}
        })
        
        assert result["success"] is True
        assert "COUNT" in result["result"]["sql_query"]
        assert "GROUP BY" in result["result"]["sql_query"]
    
    @pytest.mark.asyncio
    async def test_batch_conversions(self, sql_agent):
        """Test batch SQL conversions"""
        conversions = [
            {
                "type": "nl2sql",
                "query": "show all products",
                "query_type": "relational"
            },
            {
                "type": "nl2sql",
                "query": "find similar items to book",
                "query_type": "vector"
            },
            {
                "type": "sql2nl",
                "query": "SELECT COUNT(*) FROM orders"
            }
        ]
        
        result = await sql_agent.process_batch_conversions(conversions, "test_batch")
        
        assert result["summary"]["total"] == 3
        assert result["summary"]["successful"] == 3
        assert result["summary"]["failed"] == 0
        assert len(result["conversions"]) == 3
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sql_agent):
        """Test error handling for invalid queries"""
        # Test empty query
        message = A2AMessage(
            messageId="test_error",
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "nl2sql",
                        "query": "",
                        "query_type": "relational"
                    }
                )
            ]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        assert result["success"] is True  # Handler succeeds but conversion may have issues
        
        # Test missing data
        message = A2AMessage(
            messageId="test_error2",
            role=MessageRole.USER,
            parts=[]
        )
        
        result = await sql_agent.handle_nl2sql_request(message, "test_context")
        assert result["success"] is False
        assert "error" in result
    
    def test_pattern_matching(self, sql_agent):
        """Test pattern matching for various NL queries"""
        # Test relational patterns
        assert sql_agent._detect_query_type("show all customers") == "relational"
        assert sql_agent._detect_query_type("count of orders") == "relational"
        assert sql_agent._detect_query_type("average price by category") == "relational"
        
        # Test vector patterns
        assert sql_agent._detect_query_type("find similar products") == "vector"
        assert sql_agent._detect_query_type("nearest items to query") == "vector"
        assert sql_agent._detect_query_type("documents like this one") == "vector"
        
        # Test graph patterns
        assert sql_agent._detect_query_type("shortest path between nodes") == "graph"
        assert sql_agent._detect_query_type("find all connections") == "graph"
        assert sql_agent._detect_query_type("neighbors of node X") == "graph"
    
    def test_sql_complexity_assessment(self, sql_agent):
        """Test SQL query complexity assessment"""
        # Simple query
        simple = "SELECT * FROM users"
        assert sql_agent._assess_query_complexity(simple) == "simple"
        
        # Moderate query
        moderate = "SELECT * FROM users WHERE age > 18 ORDER BY name"
        assert sql_agent._assess_query_complexity(moderate) == "moderate"
        
        # Complex query
        complex_query = """
        SELECT u.name, COUNT(o.id), AVG(o.total)
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.date > '2024-01-01'
        GROUP BY u.name
        HAVING COUNT(o.id) > 5
        """
        assert sql_agent._assess_query_complexity(complex_query) == "complex"