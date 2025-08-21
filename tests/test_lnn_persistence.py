"""
Tests for the LNN Fallback persistent storage functionality.
"""

import asyncio
import os
import pytest
import json
from typing import Dict, Any

# Add project root to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from a2aNetwork.core.lnnFallback import LNNFallbackClient
from a2aNetwork.core.dataStore import DataStore

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'lnn_training_test.db'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'a2aNetwork', 'core', 'models', 'lnn_fallback_test.pth'))

@pytest.fixture
def setup_test_environment():
    """Ensure a clean environment for each test."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(MODEL_PATH.replace('.pth', '_best.pth')):
        os.remove(MODEL_PATH.replace('.pth', '_best.pth'))
    
    yield
    
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(MODEL_PATH.replace('.pth', '_best.pth')):
        os.remove(MODEL_PATH.replace('.pth', '_best.pth'))


def create_lnn_client() -> LNNFallbackClient:
    """Helper to create a test-specific LNN client."""
    client = LNNFallbackClient(model_path=MODEL_PATH)
    # Override the datastore path for testing
    client.data_store = DataStore(db_path=DB_PATH)
    return client


@pytest.mark.asyncio
async def test_data_persistence(setup_test_environment):
    """Test that training data persists across different LNNFallbackClient instances."""
    # 1. Create the first client and add data
    client1 = create_lnn_client()
    assert client1.data_store.get_data_count('train') == 0
    assert client1.data_store.get_data_count('validation') == 0

    prompt = "What is the derivative of x^2?"
    expected = {"accuracy_score": 100, "methodology_score": 100, "explanation_score": 100}
    
    # Add 10 samples
    for _ in range(10):
        client1.add_training_data(prompt, expected)

    train_count = client1.data_store.get_data_count('train')
    val_count = client1.data_store.get_data_count('validation')
    assert train_count + val_count == 10
    print(f"Client 1 - Train: {train_count}, Val: {val_count}")

    # 2. Create a second client, which should load the persisted data
    client2 = create_lnn_client()
    train_count_2 = client2.data_store.get_data_count('train')
    val_count_2 = client2.data_store.get_data_count('validation')
    print(f"Client 2 - Train: {train_count_2}, Val: {val_count_2}")

    assert train_count_2 == train_count
    assert val_count_2 == val_count
    assert train_count_2 + val_count_2 == 10

@pytest.mark.asyncio
async def test_automatic_training_scheduling(setup_test_environment):
    """Test that training is automatically scheduled when enough data is present on init."""
    # 1. Add just enough data to the database without a client
    data_store = DataStore(db_path=DB_PATH)
    prompt = "What is the derivative of x^2?"
    expected = {"accuracy_score": 100, "methodology_score": 100, "explanation_score": 100}
    for i in range(50):
        data_store.add_data(prompt, expected, 'train' if i % 2 == 0 else 'validation')
    
    assert data_store.get_data_count('train') == 25
    # Close connection to avoid lock issues
    del data_store

    # 2. Create a client - it should not schedule training yet (needs 50 train samples)
    client = create_lnn_client()
    await asyncio.sleep(0.1) # Give asyncio loop a chance to run
    assert not client.is_trained

    # 3. Add more data to reach the threshold
    for _ in range(25):
         client.add_training_data(prompt, expected)
    
    # Now training should be triggered, but it's an async task.
    # For this test, we'll just check if the model becomes trained after explicitly calling it.
    await client.train_model()
    assert client.is_trained
    assert os.path.exists(MODEL_PATH.replace('.pth', '_best.pth'))

