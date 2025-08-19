"""
Test script for embedding model fine-tuning functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from src.embeddingFinetuner import EmbeddingFineTuner, TrainingPair, Agent2EmbeddingSkill
from src.adaptive_learning import AdaptiveLearningStorage, FeedbackEvent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_feedback_data():
    """Create sample feedback data for testing"""
    storage = AdaptiveLearningStorage("/tmp/test_adaptive_learning")
    
    # Create sample feedback events (simulating user selections)
    feedback_events = [
        # Financial entity searches
        ("Apple stock price", "AAPL - Apple Inc. Common Stock", ["AAPL - Apple Inc. Common Stock", "APLE - Apple Hospitality REIT", "APP - AppLovin Corporation"]),
        ("Goldman Sachs", "GS - The Goldman Sachs Group, Inc.", ["GS - The Goldman Sachs Group, Inc.", "MS - Morgan Stanley", "JPM - JPMorgan Chase"]),
        ("risk assessment tools", "Risk Assessment and Analytics Platform", ["Risk Assessment and Analytics Platform", "Marketing Analytics Tools", "Profit Analysis Dashboard"]),
        ("ESG investing", "Environmental, Social, and Governance Investment Strategies", ["Environmental, Social, and Governance Investment Strategies", "Day Trading Strategies", "Value Investing Guide"]),
        ("market capitalization", "Market Cap - Total value of company shares", ["Market Cap - Total value of company shares", "Market Share Analysis", "Marketing Budget Calculator"]),
        ("P/E ratio analysis", "Price-to-Earnings Ratio Calculator", ["Price-to-Earnings Ratio Calculator", "Debt Ratio Analysis", "Return on Investment Calculator"]),
        ("regulatory compliance", "Financial Regulatory Compliance Framework", ["Financial Regulatory Compliance Framework", "Marketing Compliance Guidelines", "HR Compliance Checklist"]),
        ("mergers and acquisitions", "M&A Advisory Services - Corporate Finance", ["M&A Advisory Services - Corporate Finance", "Marketing & Advertising Services", "Business Development Strategies"]),
        ("derivative instruments", "Financial Derivatives - Options, Futures, Swaps", ["Financial Derivatives - Options, Futures, Swaps", "Marketing Derivatives", "Product Development Tools"]),
        ("portfolio optimization", "Investment Portfolio Optimization Tools", ["Investment Portfolio Optimization Tools", "Website Optimization Services", "Process Optimization Consulting"]),
    ]
    
    # Store feedback events
    for query, selected, all_results in feedback_events:
        event = FeedbackEvent(
            event_type="search_selection",
            search_query=query,
            search_results=all_results,
            selected_entity=selected,
            effectiveness_score=0.9
        )
        storage.record_feedback(event)
    
    logger.info(f"Created {len(feedback_events)} sample feedback events")
    return storage


async def test_fine_tuning():
    """Test the fine-tuning process"""
    logger.info("Starting fine-tuning test...")
    
    # Create sample data
    storage = await create_sample_feedback_data()
    
    # Initialize fine-tuner with test database
    finetuner = EmbeddingFineTuner()
    finetuner.db_path = storage.db_path
    
    # Collect training data
    training_pairs = finetuner.collect_training_data_from_feedback()
    logger.info(f"Collected {len(training_pairs)} training pairs")
    
    # Display sample pairs
    logger.info("\nSample training pairs:")
    for i, pair in enumerate(training_pairs[:3]):
        logger.info(f"\nPair {i+1}:")
        logger.info(f"  Anchor: {pair.anchor_text}")
        logger.info(f"  Positive: {pair.positive_text}")
        logger.info(f"  Negative: {pair.negative_text}")
    
    # Add domain-specific pairs
    domain_pairs = finetuner.create_domain_specific_pairs()
    all_pairs = training_pairs + domain_pairs
    logger.info(f"\nTotal pairs with domain-specific examples: {len(all_pairs)}")
    
    # Test fine-tuning with reduced epochs for faster testing
    finetuner.num_epochs = 1  # Reduce for testing
    finetuner.batch_size = 4  # Smaller batch for testing
    
    logger.info("\nStarting fine-tuning process...")
    model_path = finetuner.fine_tune(custom_pairs=all_pairs)
    logger.info(f"Fine-tuning completed. Model saved to: {model_path}")
    
    # Test the fine-tuned model
    if model_path != finetuner.base_model_name:
        logger.info("\nEvaluating fine-tuned model...")
        evaluation = finetuner.evaluate_model(model_path, all_pairs[-5:])
        logger.info(f"Evaluation results: {evaluation}")
        
        # Compare embeddings before and after
        test_queries = [
            "Apple stock",
            "risk management",
            "M&A deals"
        ]
        
        logger.info("\nComparing embeddings before and after fine-tuning:")
        base_model = finetuner.model
        fine_tuned_model = finetuner.load_fine_tuned_model(model_path)
        
        for query in test_queries:
            base_embedding = base_model.encode(query)
            fine_tuned_embedding = fine_tuned_model.encode(query)
            
            # Calculate similarity to financial terms
            financial_term = query + " financial analysis"
            base_sim = base_model.encode([query, financial_term])
            fine_tuned_sim = fine_tuned_model.encode([query, financial_term])
            
            import torch
            import torch.nn.functional as F
            
            base_similarity = F.cosine_similarity(
                torch.tensor(base_sim[0]), 
                torch.tensor(base_sim[1]), 
                dim=0
            ).item()
            
            fine_tuned_similarity = F.cosine_similarity(
                torch.tensor(fine_tuned_sim[0]), 
                torch.tensor(fine_tuned_sim[1]), 
                dim=0
            ).item()
            
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"  Base model similarity to financial context: {base_similarity:.4f}")
            logger.info(f"  Fine-tuned similarity to financial context: {fine_tuned_similarity:.4f}")
            logger.info(f"  Improvement: {fine_tuned_similarity - base_similarity:.4f}")


async def test_agent_skill():
    """Test the Agent2EmbeddingSkill integration"""
    logger.info("\n\nTesting Agent2EmbeddingSkill...")
    
    # Create sample data
    storage = await create_sample_feedback_data()
    
    # Initialize skill
    skill = Agent2EmbeddingSkill()
    skill.finetuner.db_path = storage.db_path
    skill.fine_tune_threshold = 5  # Lower threshold for testing
    
    # Check status
    should_fine_tune = skill.should_fine_tune()
    logger.info(f"Should fine-tune: {should_fine_tune}")
    
    if should_fine_tune:
        # Execute fine-tuning
        result = skill.execute_fine_tuning()
        logger.info(f"\nFine-tuning result: {json.dumps(result, indent=2)}")
        
        # Check if model was updated
        if result["status"] == "success":
            logger.info(f"Model successfully updated to: {skill.current_model_path}")


if __name__ == "__main__":
    asyncio.run(test_fine_tuning())
    asyncio.run(test_agent_skill())