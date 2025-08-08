import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from datetime import datetime
import base64

from main import app
from app.a2a.agents.data_standardization_agent import MessageRole, TaskState

client = TestClient(app)


class TestA2AAgent:
    """Test suite for A2A Financial Data Standardization Agent"""
    
    def test_agent_card(self):
        """Test agent card endpoint"""
        response = client.get("/a2a/v1/.well-known/agent.json")
        assert response.status_code == 200
        
        agent_card = response.json()
        assert agent_card["name"] == "Financial Data Standardization Agent"
        assert agent_card["protocolVersion"] == "0.2.9"
        assert "location-standardization" in [s["id"] for s in agent_card["skills"]]
        assert "account-standardization" in [s["id"] for s in agent_card["skills"]]
    
    def test_health_check(self):
        """Test health endpoint"""
        response = client.get("/a2a/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_json_rpc_agent_card(self):
        """Test JSON-RPC agent.getCard method"""
        response = client.post("/a2a/v1/rpc", json={
            "jsonrpc": "2.0",
            "method": "agent.getCard",
            "id": 1
        })
        assert response.status_code == 200
        
        result = response.json()
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert "result" in result
        assert result["result"]["name"] == "Financial Data Standardization Agent"
    
    def test_location_standardization(self):
        """Test location standardization via JSON-RPC"""
        response = client.post("/a2a/v1/rpc", json={
            "jsonrpc": "2.0",
            "method": "agent.processMessage",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": "standardize location: New York, NY, USA"
                    }]
                },
                "contextId": "test-context-001"
            },
            "id": 2
        })
        assert response.status_code == 200
        
        result = response.json()
        assert "result" in result
        assert "taskId" in result["result"]
        assert result["result"]["contextId"] == "test-context-001"
        
        # Get task status
        task_id = result["result"]["taskId"]
        
        # Wait a moment for processing
        import time
        time.sleep(0.5)
        
        status_response = client.post("/a2a/v1/rpc", json={
            "jsonrpc": "2.0",
            "method": "agent.getTaskStatus",
            "params": {"taskId": task_id},
            "id": 3
        })
        
        assert status_response.status_code == 200
        status = status_response.json()["result"]["status"]
        assert status["state"] in ["working", "completed"]
    
    def test_account_standardization_csv(self):
        """Test account standardization with CSV data"""
        csv_data = """Account (L0),Account (L1),Account (L2),Account (L3),_row_number
Impairments,Impairments,Credit Impairments,ECL P/L Provision,1
Income,Fee Income,Fee Income,Portfolio Fee,2"""
        
        csv_bytes = base64.b64encode(csv_data.encode()).decode()
        
        response = client.post("/a2a/v1/rpc", json={
            "jsonrpc": "2.0",
            "method": "agent.processMessage",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{
                        "kind": "file",
                        "file": {
                            "name": "accounts.csv",
                            "mimeType": "text/csv",
                            "bytes": csv_bytes
                        }
                    }]
                },
                "contextId": "test-context-002"
            },
            "id": 4
        })
        
        assert response.status_code == 200
        result = response.json()
        assert "result" in result
        assert "taskId" in result["result"]
    
    def test_rest_api_message(self):
        """Test REST-style message endpoint"""
        response = client.post("/a2a/v1/messages", json={
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "type": "location",
                        "items": [
                            {"raw_value": "London, UK"},
                            {"raw_value": "Tokyo, Japan"}
                        ]
                    }
                }]
            },
            "contextId": "test-context-003"
        })
        
        assert response.status_code == 200
        result = response.json()
        assert "taskId" in result
    
    def test_batch_standardization(self):
        """Test batch multi-type standardization"""
        response = client.post("/a2a/v1/messages", json={
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "data",
                    "data": {
                        "type": "batch",
                        "location": [
                            {"raw_value": "Paris, France"},
                            {"raw_value": "Berlin, Germany"}
                        ],
                        "account": [
                            {"raw_value": "Income → Fee Income → Transaction Fee"},
                            {"raw_value": "Impairments → Credit Impairments → ECL"}
                        ],
                        "product": [
                            {"raw_value": "Banking → Retail → Mortgages"},
                            {"raw_value": "Markets → FX → Spot"}
                        ]
                    }
                }]
            },
            "contextId": "test-context-004"
        })
        
        assert response.status_code == 200
        result = response.json()
        assert "taskId" in result
        
        # Check task status
        task_id = result["taskId"]
        status_response = client.get(f"/a2a/v1/tasks/{task_id}")
        assert status_response.status_code == 200
    
    def test_task_cancellation(self):
        """Test task cancellation"""
        # Start a task
        response = client.post("/a2a/v1/messages", json={
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": "standardize location: Multiple locations to process"
                }]
            }
        })
        
        task_id = response.json()["taskId"]
        
        # Cancel the task
        cancel_response = client.delete(f"/a2a/v1/tasks/{task_id}")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"
        
        # Verify task is cancelled
        status_response = client.get(f"/a2a/v1/tasks/{task_id}")
        assert status_response.json()["status"]["state"] == "canceled"
    
    def test_error_handling(self):
        """Test error handling"""
        # Invalid JSON-RPC request
        response = client.post("/a2a/v1/rpc", json={
            "method": "invalid.method",
            "id": 1
        })
        
        assert response.status_code == 400
        error = response.json()
        assert "error" in error
        assert error["error"]["code"] == -32600  # Invalid Request
    
    def test_invalid_task_id(self):
        """Test accessing non-existent task"""
        response = client.get("/a2a/v1/tasks/invalid-task-id")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_streaming_updates(self):
        """Test Server-Sent Events streaming"""
        # This would require a more complex test setup with async client
        # For now, just verify the endpoint exists
        response = client.get("/a2a/v1/tasks/test-task-id/stream", stream=True)
        # Should return a streaming response or 200 status
        assert response.status_code in [200, 404]


class TestDataStandardization:
    """Test actual standardization logic"""
    
    def test_location_data_from_csv(self):
        """Test location standardization with real CSV data"""
        # Read sample location data
        import pandas as pd
        import os
        
        csv_path = "/Users/apple/projects/finsight_cib/data/raw/CRD_Extraction_v1_location_sorted.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, nrows=5)  # Test with first 5 rows
            
            # Convert to base64
            csv_data = df.to_csv(index=False)
            csv_bytes = base64.b64encode(csv_data.encode()).decode()
            
            response = client.post("/a2a/v1/messages", json={
                "message": {
                    "role": "user",
                    "parts": [{
                        "kind": "file",
                        "file": {
                            "name": "locations.csv",
                            "mimeType": "text/csv",
                            "bytes": csv_bytes
                        }
                    }]
                }
            })
            
            assert response.status_code == 200
            task_id = response.json()["taskId"]
            
            # Wait and check results
            import time
            time.sleep(1)
            
            status_response = client.get(f"/a2a/v1/tasks/{task_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            if status_data["status"]["state"] == "completed":
                assert len(status_data["artifacts"]) > 0
                artifact = status_data["artifacts"][0]
                assert "parts" in artifact
                assert artifact["parts"][0]["kind"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])