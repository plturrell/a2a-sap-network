# Embedding Model Fine-Tuning for Agent 2

This document describes the embedding model fine-tuning capability added to Agent 2 (AI Preparation Agent).

## Overview

Agent 2 now has the ability to fine-tune its embedding model based on user feedback data. This creates a continuous learning loop where the model improves based on actual usage patterns.

## How It Works

### 1. Feedback Collection
- User searches and selections are tracked in the adaptive learning database
- Each selection creates a positive training example (query → selected result)
- Non-selected results become negative examples for contrastive learning

### 2. Training Data Generation
```python
# Positive pair: query matches selected result
("Apple stock price", "AAPL - Apple Inc. Common Stock")

# Negative pair: query doesn't match ignored result  
("Apple stock price", "APLE - Apple Hospitality REIT")
```

### 3. Fine-Tuning Process
- Uses **contrastive learning** with triplet loss
- Learns to bring similar financial concepts closer in embedding space
- Preserves base model knowledge while adapting to domain

### 4. Model Evaluation
- Evaluates on held-out feedback data
- Requires 70% accuracy to replace current model
- Maintains model versioning for rollback

## Implementation Details

### Files Added:
- `embeddingFinetuner.py` - Core fine-tuning logic
- `testFineTuning.py` - Test suite
- `requirements_finetuning.txt` - Dependencies

### Key Classes:
- `EmbeddingFineTuner` - Handles the fine-tuning process
- `Agent2EmbeddingSkill` - Integrates fine-tuning into agent

### Agent Skills Added:
- `fine_tune_embeddings` - Manually trigger fine-tuning
- `check_fine_tuning_status` - Check if fine-tuning is needed

## Usage

### Automatic Fine-Tuning
The agent automatically checks every hour if enough feedback has accumulated (default: 100 events).

### Manual Fine-Tuning
```python
# Via agent skill
result = await agent.execute_embedding_fine_tuning()
```

### Check Status
```python
status = await agent.check_fine_tuning_status()
# Returns:
# {
#   "should_fine_tune": true,
#   "feedback_count": 150,
#   "threshold": 100,
#   "current_model": "sentence-transformers/all-mpnet-base-v2"
# }
```

## Benefits

1. **Domain Adaptation**: Model learns financial terminology relationships
2. **Continuous Improvement**: Gets better with more usage
3. **User-Driven**: Learns from actual search patterns
4. **Efficient**: Only fine-tunes when sufficient data available

## Technical Approach

### Why Contrastive Learning?
- Efficient for embedding models
- Learns from relative similarities
- Preserves semantic structure
- Works with limited labeled data

### Training Strategy
1. Collect user feedback pairs
2. Add domain-specific examples
3. Use triplet loss for training
4. Evaluate on validation set
5. Deploy if accuracy threshold met

## Configuration

Environment variables:
- `EMBEDDING_MODEL`: Base model name (default: all-mpnet-base-v2)
- `LEARNING_STORAGE_PATH`: Where to store learning data
- `ADAPTIVE_LEARNING_DB`: SQLite database path

## Future Enhancements

1. **Active Learning**: Query users for ambiguous cases
2. **Multi-Task Learning**: Train on multiple objectives
3. **Federated Learning**: Learn from multiple agents
4. **Model Compression**: Optimize for inference speed