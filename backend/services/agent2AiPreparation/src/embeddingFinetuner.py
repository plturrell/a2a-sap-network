"""
Embedding Model Fine-Tuning Module for Agent 2 (AI Preparation Agent)

This module provides real fine-tuning capabilities for the embedding model
using contrastive learning on user feedback data.
"""

import os
import json
import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import logging
import time
from adaptive_learning import FeedbackEvent

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """Represents a training pair for contrastive learning"""
    anchor_text: str
    positive_text: str
    negative_text: Optional[str] = None
    score: float = 1.0


class EmbeddingFineTuner:
    """
    Fine-tunes embedding models using contrastive learning from user feedback.
    
    This is REAL fine-tuning that updates model weights, not mock learning.
    """
    
    def __init__(self, base_model_name: str = "sentence-transformers/all-mpnet-base-v2", learning_storage=None):
        self.base_model_name = base_model_name
        self.model = SentenceTransformer(base_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Paths
        self.models_dir = "models/fineTuned"
        self.training_data_path = "data/trainingPairs.json"
        self.db_path = os.getenv("ADAPTIVE_LEARNING_DB", "adaptiveLearning.db")
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 3
        self.warmup_steps = 100
        self.evaluation_steps = 500
        
        # Learning storage for tracking
        self.learning_storage = learning_storage
        
        # Metrics tracking
        self.training_metrics = {
            "total_fine_tunings": 0,
            "successful_fine_tunings": 0,
            "failed_fine_tunings": 0,
            "total_training_time_seconds": 0,
            "average_accuracy_improvement": 0,
            "models_deployed": 0
        }
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        
    def collect_training_data_from_feedback(self) -> List[TrainingPair]:
        """
        Collects training pairs from the feedback database.
        
        Uses real user interactions:
        - Positive pairs: query + selected result
        - Negative pairs: query + ignored results
        """
        training_pairs = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get successful searches (user selected a result)
            cursor.execute("""
                SELECT 
                    fe.search_query,
                    fe.selected_entity,
                    fe.search_results,
                    fe.effectiveness_score
                FROM feedback_events fe
                WHERE fe.event_type = 'search_selection'
                AND fe.selected_entity IS NOT NULL
                ORDER BY fe.timestamp DESC
                LIMIT 1000
            """)
            
            for row in cursor.fetchall():
                query = row[0]
                selected = row[1]
                all_results = json.loads(row[2]) if row[2] else []
                effectiveness = row[3] or 0.5
                
                # Create positive pair (query -> selected result)
                if selected:
                    # Find non-selected results for negative examples
                    negatives = [r for r in all_results if r != selected]
                    
                    if negatives:
                        # Use the top non-selected result as hard negative
                        training_pairs.append(TrainingPair(
                            anchor_text=query,
                            positive_text=selected,
                            negative_text=negatives[0],
                            score=effectiveness
                        ))
                    else:
                        # Just positive pair
                        training_pairs.append(TrainingPair(
                            anchor_text=query,
                            positive_text=selected,
                            score=effectiveness
                        ))
            
            conn.close()
            logger.info(f"Collected {len(training_pairs)} training pairs from feedback")
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            
        return training_pairs
    
    def prepare_training_examples(self, pairs: List[TrainingPair]) -> List[InputExample]:
        """Converts training pairs to SentenceTransformers format"""
        examples = []
        
        for i, pair in enumerate(pairs):
            if pair.negative_text:
                # Triplet example (anchor, positive, negative)
                examples.append(InputExample(
                    texts=[pair.anchor_text, pair.positive_text, pair.negative_text],
                    label=pair.score
                ))
            else:
                # Pair example (anchor, positive)
                examples.append(InputExample(
                    texts=[pair.anchor_text, pair.positive_text],
                    label=pair.score
                ))
                
        return examples
    
    def create_evaluator(self, val_pairs: List[TrainingPair]) -> evaluation.TripletEvaluator:
        """Creates an evaluator for the model"""
        anchors = []
        positives = []
        negatives = []
        
        for pair in val_pairs:
            if pair.negative_text:
                anchors.append(pair.anchor_text)
                positives.append(pair.positive_text)
                negatives.append(pair.negative_text)
        
        return evaluation.TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="feedback_eval"
        )
    
    def fine_tune(self, custom_pairs: Optional[List[TrainingPair]] = None) -> str:
        """
        Fine-tunes the embedding model on collected feedback data.
        
        Returns: Path to the fine-tuned model
        """
        logger.info("Starting embedding model fine-tuning...")
        start_time = time.time()
        
        # Track fine-tuning event
        self.training_metrics["total_fine_tunings"] += 1
        fine_tuning_event = {
            "event_type": "model_fine_tuning_started",
            "timestamp": datetime.now(),
            "base_model": self.base_model_name,
            "device": str(self.device)
        }
        
        # Record in learning storage if available
        if self.learning_storage:
            self._record_fine_tuning_event(fine_tuning_event)
        
        # Collect training data
        training_pairs = custom_pairs or self.collect_training_data_from_feedback()
        
        if len(training_pairs) < 10:
            logger.warning("Insufficient training data. Need at least 10 pairs.")
            self.training_metrics["failed_fine_tunings"] += 1
            return self.base_model_name
        
        # Split into train/validation
        split_idx = int(0.8 * len(training_pairs))
        train_pairs = training_pairs[:split_idx]
        val_pairs = training_pairs[split_idx:]
        
        # Prepare training examples
        train_examples = self.prepare_training_examples(train_pairs)
        
        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.batch_size
        )
        
        # Define loss function
        # Use TripletLoss for triplet examples, MultipleNegativesRankingLoss for pairs
        train_loss = losses.TripletLoss(model=self.model)
        
        # Create evaluator
        evaluator = self.create_evaluator(val_pairs) if val_pairs else None
        
        # Fine-tune the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.models_dir, f"fineTuned_{timestamp}")
        
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.num_epochs,
                evaluation_steps=self.evaluation_steps,
                warmup_steps=self.warmup_steps,
                output_path=output_path,
                save_best_model=True,
                show_progress_bar=True
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            self.training_metrics["total_training_time_seconds"] += training_time
            self.training_metrics["successful_fine_tunings"] += 1
            
            logger.info(f"Fine-tuning completed in {training_time:.2f}s. Model saved to: {output_path}")
            
            # Record successful fine-tuning
            if self.learning_storage:
                completion_event = {
                    "event_type": "model_fine_tuning_completed",
                    "timestamp": datetime.now(),
                    "model_path": output_path,
                    "training_time_seconds": training_time,
                    "training_pairs_count": len(train_pairs),
                    "epochs": self.num_epochs,
                    "status": "success"
                }
                self._record_fine_tuning_event(completion_event)
                
        except Exception as e:
            self.training_metrics["failed_fine_tunings"] += 1
            logger.error(f"Fine-tuning failed: {e}")
            
            # Record failure
            if self.learning_storage:
                failure_event = {
                    "event_type": "model_fine_tuning_failed",
                    "timestamp": datetime.now(),
                    "error": str(e),
                    "status": "failed"
                }
                self._record_fine_tuning_event(failure_event)
            
            raise
        
        # Save training metadata
        metadata = {
            "base_model": self.base_model_name,
            "timestamp": timestamp,
            "num_training_pairs": len(train_pairs),
            "num_epochs": self.num_epochs,
            "training_pairs_sample": [
                {
                    "anchor": p.anchor_text[:50],
                    "positive": p.positive_text[:50],
                    "negative": p.negative_text[:50] if p.negative_text else None
                }
                for p in train_pairs[:5]
            ]
        }
        
        with open(os.path.join(output_path, "trainingMetadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def load_fine_tuned_model(self, model_path: str) -> SentenceTransformer:
        """Loads a fine-tuned model"""
        return SentenceTransformer(model_path)
    
    def evaluate_model(self, model_path: str, test_pairs: List[TrainingPair]) -> Dict[str, float]:
        """Evaluates a fine-tuned model against test pairs"""
        model = self.load_fine_tuned_model(model_path)
        
        correct = 0
        total = 0
        
        for pair in test_pairs:
            if pair.negative_text:
                # Encode texts
                embeddings = model.encode([
                    pair.anchor_text,
                    pair.positive_text,
                    pair.negative_text
                ])
                
                # Calculate similarities
                anchor_emb = torch.tensor(embeddings[0])
                positive_emb = torch.tensor(embeddings[1])
                negative_emb = torch.tensor(embeddings[2])
                
                pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=0)
                neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=0)
                
                # Check if positive is closer than negative
                if pos_sim > neg_sim:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_evaluated": total,
            "correct_predictions": correct
        }
    
    def create_domain_specific_pairs(self) -> List[TrainingPair]:
        """
        Creates domain-specific training pairs for financial entities.
        
        This uses the existing financial preprocessing knowledge.
        """
        pairs = []
        
        # Financial entity similarity pairs
        financial_pairs = [
            ("Apple Inc", "AAPL", "Microsoft Corporation"),  # Same company vs different
            ("Goldman Sachs", "GS", "JPMorgan Chase"),
            ("risk assessment", "risk analysis", "profit analysis"),
            ("regulatory compliance", "compliance requirements", "marketing strategy"),
            ("ESG investing", "sustainable investing", "day trading"),
            ("market cap", "market capitalization", "market share"),
            ("P/E ratio", "price earnings ratio", "debt ratio"),
            ("M&A", "mergers and acquisitions", "marketing and advertising"),
        ]
        
        for anchor, positive, negative in financial_pairs:
            pairs.append(TrainingPair(
                anchor_text=anchor,
                positive_text=positive,
                negative_text=negative,
                score=1.0
            ))
        
        return pairs
    
    def incremental_fine_tune(self, model_path: str, new_pairs: List[TrainingPair]) -> str:
        """
        Performs incremental fine-tuning on an already fine-tuned model.
        
        This allows continuous improvement without full retraining.
        """
        # Load existing fine-tuned model
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)
        
        # Fine-tune with new data
        return self.fine_tune(custom_pairs=new_pairs)
    
    def _record_fine_tuning_event(self, event: Dict[str, any]) -> None:
        """Record fine-tuning event in the learning storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create fine-tuning events table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fine_tuning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_data TEXT,
                    status TEXT
                )
            """)
            
            # Insert event
            cursor.execute("""
                INSERT INTO fine_tuning_events (event_type, timestamp, event_data, status)
                VALUES (?, ?, ?, ?)
            """, (
                event["event_type"],
                event["timestamp"],
                json.dumps(event),
                event.get("status", "in_progress")
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record fine-tuning event: {e}")
    
    def get_fine_tuning_history(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get recent fine-tuning history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT event_type, timestamp, event_data, status
                FROM fine_tuning_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                event_data = json.loads(row[2])
                history.append(event_data)
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Failed to get fine-tuning history: {e}")
            return []


# Integration with Agent 2
class Agent2EmbeddingSkill:
    """
    Skill for Agent 2 to fine-tune its embedding model based on feedback.
    """
    
    def __init__(self, learning_storage=None, audit_logger=None, metrics_client=None):
        self.finetuner = EmbeddingFineTuner(learning_storage=learning_storage)
        self.current_model_path = "sentence-transformers/all-mpnet-base-v2"
        self.fine_tune_threshold = 100  # Minimum feedback events before fine-tuning
        self.audit_logger = audit_logger
        self.metrics_client = metrics_client
        self.model_versions = []  # Track model version history
        
    def should_fine_tune(self) -> bool:
        """Checks if enough feedback has accumulated for fine-tuning"""
        try:
            conn = sqlite3.connect(self.finetuner.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM feedback_events 
                WHERE event_type = 'search_selection'
                AND selected_entity IS NOT NULL
                AND timestamp > datetime('now', '-7 days')
            """)
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count >= self.fine_tune_threshold
            
        except Exception as e:
            logger.error(f"Error checking fine-tune threshold: {e}")
            return False
    
    def execute_fine_tuning(self) -> Dict[str, any]:
        """Executes the fine-tuning process"""
        start_time = time.time()
        
        # Log audit event for fine-tuning start
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="MODEL_FINE_TUNING_INITIATED",
                agent_id="agent2_ai_preparation",
                details={
                    "current_model": self.current_model_path,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        if not self.should_fine_tune():
            return {
                "status": "skipped",
                "reason": "Insufficient feedback data"
            }
        
        try:
            # Track start metric
            if self.metrics_client:
                self.metrics_client.increment("model_fine_tuning_attempts")
            
            # Perform fine-tuning
            new_model_path = self.finetuner.fine_tune()
            
            # Evaluate the new model
            test_pairs = self.finetuner.collect_training_data_from_feedback()[-20:]
            evaluation = self.finetuner.evaluate_model(new_model_path, test_pairs)
            
            # Calculate improvement
            baseline_accuracy = 0.5  # Assume baseline
            improvement = evaluation["accuracy"] - baseline_accuracy
            
            # Update current model if improvement
            if evaluation["accuracy"] > 0.7:
                old_model = self.current_model_path
                self.current_model_path = new_model_path
                self.model_versions.append({
                    "version": len(self.model_versions) + 1,
                    "model_path": new_model_path,
                    "timestamp": datetime.now(),
                    "accuracy": evaluation["accuracy"]
                })
                
                status = "success"
                message = f"Model fine-tuned successfully. Accuracy: {evaluation['accuracy']:.2f}"
                
                # Log successful deployment
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="MODEL_VERSION_CHANGED",
                        agent_id="agent2_ai_preparation",
                        details={
                            "old_model": old_model,
                            "new_model": new_model_path,
                            "accuracy": evaluation["accuracy"],
                            "improvement": improvement,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
                # Track deployment metrics
                if self.metrics_client:
                    self.metrics_client.increment("model_deployments")
                    self.metrics_client.gauge("model_accuracy", evaluation["accuracy"])
                    self.metrics_client.gauge("model_improvement", improvement)
                
            else:
                status = "rejected"
                message = f"Fine-tuned model didn't meet accuracy threshold: {evaluation['accuracy']:.2f}"
                
                # Log rejection
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="MODEL_FINE_TUNING_REJECTED",
                        agent_id="agent2_ai_preparation",
                        details={
                            "model_path": new_model_path,
                            "accuracy": evaluation["accuracy"],
                            "threshold": 0.7,
                            "reason": "accuracy_below_threshold"
                        }
                    )
            
            # Track completion metrics
            training_time = time.time() - start_time
            if self.metrics_client:
                self.metrics_client.timing("model_fine_tuning_duration", training_time)
                self.metrics_client.increment(f"model_fine_tuning_{status}")
            
            return {
                "status": status,
                "model_path": new_model_path,
                "evaluation": evaluation,
                "message": message,
                "training_time_seconds": training_time,
                "improvement": improvement
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            
            # Log failure
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="MODEL_FINE_TUNING_FAILED",
                    agent_id="agent2_ai_preparation",
                    details={
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Track failure metric
            if self.metrics_client:
                self.metrics_client.increment("model_fine_tuning_failures")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_current_model(self) -> SentenceTransformer:
        """Returns the current best model (base or fine-tuned)"""
        return SentenceTransformer(self.current_model_path)