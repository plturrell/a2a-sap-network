import os
import sys
import logging

from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Minimal test for Enhanced AI Preparation Agent with MCP Integration
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
os.environ['AGENT_PRIVATE_KEY'] = 'test_key_12345'
os.environ['AI_PREPARATION_OUTPUT_DIR'] = '/tmp/ai_preparation_data'
os.environ['AI_PREP_PROMETHEUS_PORT'] = '8014'

# Database configuration
os.environ['SQLITE_DATABASE_PATH'] = '/tmp/test_agent.db'
os.environ['DATABASE_URL'] = 'sqlite:///tmp/test_agent.db'
os.environ['POSTGRES_DATABASE_URL'] = 'sqlite:///tmp/test_agent.db'

# Create temp directory
os.makedirs('/tmp/ai_preparation_data', exist_ok=True)

def test_import():
    """Test basic import functionality"""
    try:
        # Test import
        from app.a2a.agents.agent2AiPreparation.active.enhancedAiPreparationAgentMcp import (
            EnhancedAIPreparationAgentMCP,
            EmbeddingMode,
            ConfidenceMetric,
            SophisticatedEmbeddingGenerator,
            AdvancedConfidenceScorer
        )
        print("✅ Import successful!")
        
        # Test enum creation
        mode = EmbeddingMode.HYBRID
        print(f"✅ EmbeddingMode enum works: {mode}")
        
        metric = ConfidenceMetric.SEMANTIC_COHERENCE
        print(f"✅ ConfidenceMetric enum works: {metric}")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_import()
    sys.exit(0 if result else 1)