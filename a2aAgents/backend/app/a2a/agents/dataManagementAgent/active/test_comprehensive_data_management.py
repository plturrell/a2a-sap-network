#!/usr/bin/env python3
"""
Comprehensive Test Suite for Data Management Agent

Tests all major functionality including:
- Data quality assessment
- Pipeline management
- Data cataloging
- Integrity validation
- Data archival
- Performance monitoring
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agent
from .comprehensiveDataManagementAgentSdk import (
    ComprehensiveDataManagementAgent,
    create_data_management_agent,
    DataQualityIssue,
    DataPipelineStatus
)

class TestComprehensiveDataManagementAgent:
    """Test suite for Comprehensive Data Management Agent"""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing"""
        config = {
            "test_mode": True,
            "temp_storage": True
        }
        agent = create_data_management_agent(config)
        yield agent
        await agent.cleanup()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = {
            'id': range(1, 1001),
            'name': [f'User_{i}' for i in range(1, 1001)],
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.uniform(30000, 150000, 1000),
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
            'join_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000]
        }
        
        # Introduce some quality issues for testing
        df = pd.DataFrame(data)
        
        # Add missing values (5% of salary)
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'salary'] = None
        
        # Add duplicates (2% of data)
        duplicate_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
        
        # Add outliers (some very high salaries)
        outlier_indices = np.random.choice(df.index, size=10, replace=False)
        df.loc[outlier_indices, 'salary'] = np.random.uniform(500000, 1000000, 10)
        
        return df
    
    @pytest.fixture
    def sample_file(self, sample_data):
        """Create temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            return f.name
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert agent.agent_id is not None
        assert len(agent.storage_backends) > 0
        assert 'local_fs' in agent.storage_backends
        logger.info("✓ Agent initialization test passed")
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, agent, sample_file):
        """Test data quality assessment functionality"""
        # Perform quality assessment
        result = await agent.assess_data_quality(sample_file)
        
        assert result is not None
        assert 0 <= result.overall_score <= 100
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.metrics, dict)
        
        # Check for expected issues
        issue_types = [issue['type'] for issue in result.issues]
        assert DataQualityIssue.MISSING_VALUES.value in issue_types
        assert DataQualityIssue.DUPLICATES.value in issue_types
        
        # Check metrics
        assert 'missing_values_ratio' in result.metrics
        assert 'duplicate_ratio' in result.metrics
        assert result.metrics['missing_values_ratio'] > 0
        assert result.metrics['duplicate_ratio'] > 0
        
        logger.info(f"✓ Data quality assessment test passed - Score: {result.overall_score:.1f}")
    
    @pytest.mark.asyncio
    async def test_pipeline_creation_and_execution(self, agent, sample_file):
        """Test data pipeline creation and execution"""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Create pipeline
            pipeline_id = await agent.create_data_pipeline(
                name="Test Pipeline",
                description="Test data transformation pipeline",
                source_config={
                    "type": "file",
                    "path": sample_file
                },
                target_config={
                    "type": "file",
                    "path": output_path,
                    "format": "csv"
                },
                transformation_rules=[
                    {
                        "type": "filter",
                        "condition": "age >= 25"
                    },
                    {
                        "type": "rename",
                        "mapping": {"name": "full_name"}
                    }
                ]
            )
            
            assert pipeline_id is not None
            assert pipeline_id in agent.active_pipelines
            assert agent.active_pipelines[pipeline_id].status == DataPipelineStatus.PENDING
            
            # Execute pipeline
            result = await agent.execute_pipeline(pipeline_id)
            
            assert result['status'] == DataPipelineStatus.COMPLETED.value
            assert result['records_processed'] > 0
            assert os.path.exists(output_path)
            
            # Verify output
            output_df = pd.read_csv(output_path)
            assert 'full_name' in output_df.columns
            assert 'name' not in output_df.columns
            assert all(output_df['age'] >= 25)
            
            logger.info(f"✓ Pipeline test passed - Processed {result['records_processed']} records")
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_data_cataloging(self, agent, sample_file):
        """Test data cataloging functionality"""
        catalog_id = await agent.catalog_dataset(
            name="Test Dataset",
            description="Sample dataset for testing",
            data_location=sample_file,
            tags=["test", "sample", "employee_data"]
        )
        
        assert catalog_id is not None
        assert catalog_id in agent.data_catalog
        
        entry = agent.data_catalog[catalog_id]
        assert entry.name == "Test Dataset"
        assert entry.data_location == sample_file
        assert entry.size_bytes > 0
        assert entry.record_count > 0
        assert "test" in entry.tags
        assert "schema" in entry.schema or len(entry.schema) > 0
        
        logger.info(f"✓ Data cataloging test passed - Cataloged {entry.record_count} records")
    
    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, agent, sample_file):
        """Test data integrity validation"""
        validation_rules = {
            "ranges": {
                "age": {"min": 18, "max": 100},
                "salary": {"min": 0, "max": 200000}
            },
            "patterns": {
                "department": r"^(IT|HR|Finance|Marketing)$"
            }
        }
        
        result = await agent.validate_data_integrity(
            data_location=sample_file,
            validation_rules=validation_rules
        )
        
        assert result is not None
        assert 'integrity_score' in result
        assert 'checksum' in result
        assert 'record_count' in result
        assert 'validation_errors' in result
        assert 0 <= result['integrity_score'] <= 100
        
        # Should have some validation errors due to outliers
        assert len(result['validation_errors']) > 0
        
        logger.info(f"✓ Data integrity validation test passed - Score: {result['integrity_score']:.1f}")
    
    @pytest.mark.asyncio
    async def test_data_archival(self, agent, sample_file):
        """Test data archival functionality"""
        archive_config = {
            "compression": "gzip",
            "encryption": True
        }
        
        result = await agent.archive_data(
            data_location=sample_file,
            archive_config=archive_config
        )
        
        assert result is not None
        assert 'archive_id' in result
        assert 'archive_location' in result
        assert 'compression_ratio' in result
        assert result['compression_ratio'] > 0  # Should have some compression
        assert os.path.exists(result['archive_location'])
        
        # Archived file should be smaller than original
        original_size = result['original_size_bytes']
        archived_size = result['archived_size_bytes']
        assert archived_size < original_size
        
        logger.info(f"✓ Data archival test passed - Compression: {result['compression_ratio']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, agent):
        """Test performance monitoring functionality"""
        # First, create some activity
        sample_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.rand(100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Perform some operations
            await agent.assess_data_quality(temp_file)
            await agent.catalog_dataset("Test", "Test dataset", temp_file)
            
            # Get performance metrics
            metrics = await agent.monitor_data_performance("1h")
            
            assert metrics is not None
            assert 'timestamp' in metrics
            assert 'overall_performance' in metrics
            assert 'pipeline_metrics' in metrics
            assert 'quality_metrics' in metrics
            assert 'storage_metrics' in metrics
            assert metrics['catalog_entries'] > 0
            assert metrics['quality_assessments'] > 0
            
            logger.info("✓ Performance monitoring test passed")
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling for invalid inputs"""
        # Test with non-existent file
        try:
            await agent.assess_data_quality("/nonexistent/file.csv")
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "Could not load data" in str(e)
        
        # Test with invalid pipeline
        try:
            await agent.execute_pipeline("nonexistent_pipeline")
            assert False, "Should have raised an exception"
        except ValueError as e:
            assert "Pipeline not found" in str(e)
        
        logger.info("✓ Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_schema_validation(self, agent, sample_file):
        """Test schema validation functionality"""
        schema = {
            "required_columns": ["id", "name", "age", "salary"],
            "column_types": {
                "id": "int64",
                "name": "object",
                "age": "int64"
            }
        }
        
        result = await agent.assess_data_quality(sample_file, schema)
        
        # Should pass basic schema validation
        schema_issues = [issue for issue in result.issues 
                        if issue['type'] == 'inconsistent_schema']
        
        # With our test data, schema should be mostly consistent
        assert len(schema_issues) == 0 or all(issue['severity'] != 'high' for issue in schema_issues)
        
        logger.info("✓ Schema validation test passed")
    
    @pytest.mark.asyncio  
    async def test_ml_capabilities(self, agent, sample_file):
        """Test machine learning capabilities"""
        # Test quality assessment (which uses ML internally)
        result = await agent.assess_data_quality(sample_file)
        
        # Should detect outliers using ML
        assert any(issue['type'] == DataQualityIssue.OUTLIERS.value for issue in result.issues)
        
        # Check if ML models are initialized
        assert agent.quality_classifier is not None
        assert agent.anomaly_detector is not None
        assert agent.clustering_model is not None
        
        logger.info("✓ ML capabilities test passed")

def run_tests():
    """Run all tests"""
    logger.info("Starting Comprehensive Data Management Agent Tests...")
    
    # Set environment variables for A2A compliance
    os.environ.setdefault("A2A_SERVICE_URL", "http://localhost:3000")
    os.environ.setdefault("A2A_SERVICE_HOST", "localhost:3000") 
    os.environ.setdefault("A2A_BASE_URL", "http://localhost:3000")
    
    pytest.main([__file__, "-v", "-s"])

if __name__ == "__main__":
    run_tests()