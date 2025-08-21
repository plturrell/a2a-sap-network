"""
Liquid Neural Network (LNN) Fallback for A2A AI Reasoning
Provides intelligent local reasoning when Grok API is unavailable
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path
from .dataStore import DataStore

logger = logging.getLogger(__name__)

class LiquidTimeConstantCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Neural Network Cell
    Implements continuous-time recurrent dynamics with adaptive time constants
    """
    
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Learnable time constants (tau) for each neuron
        self.tau = nn.Parameter(torch.ones(hidden_size) * 20.0)  # Initial tau = 20ms
        
        # Input transformation
        self.input_linear = nn.Linear(input_size, hidden_size)
        
        # Recurrent weights
        self.recurrent_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Activation function (tanh for stability)
        self.activation = nn.Tanh()
        self.ode_solver = self._solve_rk4
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability"""
        nn.init.xavier_normal_(self.input_linear.weight)
        nn.init.orthogonal_(self.recurrent_linear.weight)
        # Ensure tau stays positive
        with torch.no_grad():
            self.tau.data = torch.abs(self.tau.data) + 1.0
    
    def _solve_rk4(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Solve the LTC ODE with 4th-order Runge-Kutta"""
        
        def _dhdt(h, activated_input, tau):
            return (-h + activated_input) / tau

        tau_pos = torch.abs(self.tau) + 1e-6
        
        # Compute activated input once
        input_contrib = self.input_linear(x)
        recurrent_contrib = self.recurrent_linear(hidden)
        activated = self.activation(input_contrib + recurrent_contrib)

        # RK4 steps
        k1 = self.dt * _dhdt(hidden, activated, tau_pos)
        
        # For k2, we need to compute a new activated input based on h + k1/2
        recurrent_contrib_k2 = self.recurrent_linear(hidden + 0.5 * k1)
        activated_k2 = self.activation(input_contrib + recurrent_contrib_k2)
        k2 = self.dt * _dhdt(hidden + 0.5 * k1, activated_k2, tau_pos)

        # For k3, based on h + k2/2
        recurrent_contrib_k3 = self.recurrent_linear(hidden + 0.5 * k2)
        activated_k3 = self.activation(input_contrib + recurrent_contrib_k3)
        k3 = self.dt * _dhdt(hidden + 0.5 * k2, activated_k3, tau_pos)

        # For k4, based on h + k3
        recurrent_contrib_k4 = self.recurrent_linear(hidden + k3)
        activated_k4 = self.activation(input_contrib + recurrent_contrib_k4)
        k4 = self.dt * _dhdt(hidden + k3, activated_k4, tau_pos)

        # Combine steps
        new_hidden = hidden + (k1 + 2*k2 + 2*k3 + k4) / 6
        return new_hidden

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing ODE with a chosen solver
        """
        return self.ode_solver(x, hidden)


class LiquidNeuralNetwork(nn.Module):
    """
    Complete Liquid Neural Network for A2A reasoning tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.get('vocab_size', 1000)
        self.embed_dim = config.get('embed_dim', 128)
        self.hidden_size = config.get('hidden_size', 256)  # Increased capacity
        self.num_layers = config.get('num_layers', 3)      # Deeper model
        self.output_size = config.get('output_size', 512)   # Larger output space
        self.dt = config.get('dt', 0.1)
        
        # Embedding layer for text input
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # LTC layers
        self.ltc_layers = nn.ModuleList([
            LiquidTimeConstantCell(
                input_size=self.embed_dim if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
                dt=self.dt
            ) for i in range(self.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_size, self.output_size)
        )
        
        # Classification heads for different reasoning tasks
        self.accuracy_head = nn.Linear(self.output_size, 101)  # 0-100 score
        self.methodology_head = nn.Linear(self.output_size, 101)
        self.explanation_head = nn.Linear(self.output_size, 101)
        self.confidence_head = nn.Linear(self.output_size, 1)  # 0-1 confidence
        
        # Initialize hidden states
        self.hidden_states = None
        
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states for all layers"""
        self.hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
    
    def forward(self, input_ids: torch.Tensor, sequence_length: int = 10) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the LNN
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            sequence_length: Number of time steps to run LTC dynamics
            
        Returns:
            Dict with reasoning outputs
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize hidden states if needed
        if self.hidden_states is None or self.hidden_states[0].shape[0] != batch_size:
            self.init_hidden(batch_size, device)
        else:
            # Detach hidden states to prevent backprop through previous sequences
            self.hidden_states = [h.detach() for h in self.hidden_states]
        
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Process sequence token by token
        for t in range(embedded.size(1)):
            x_t = embedded[:, t, :]
            layer_input = x_t
            for i, ltc_layer in enumerate(self.ltc_layers):
                self.hidden_states[i] = ltc_layer(layer_input, self.hidden_states[i])
                layer_input = self.hidden_states[i]
        
        # Final hidden state
        final_hidden = self.hidden_states[-1]
        
        # Project to output space
        output_features = self.output_proj(final_hidden)
        
        # Generate predictions
        accuracy_logits = self.accuracy_head(output_features)
        methodology_logits = self.methodology_head(output_features)
        explanation_logits = self.explanation_head(output_features)
        confidence = torch.sigmoid(self.confidence_head(output_features))
        
        return {
            'accuracy_logits': accuracy_logits,
            'methodology_logits': methodology_logits,
            'explanation_logits': explanation_logits,
            'confidence': confidence,
            'features': output_features
        }


class LNNFallbackClient:
    """
    LNN-based fallback client for A2A AI reasoning
    Integrates with existing GrokClient architecture
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'models', 'lnn_fallback.pth'
        )
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = LiquidNeuralNetwork(self.config)
        self.model.to(self.device)
        
        # Tokenizer (simple word-based for now)
        self.vocab = self._build_vocab()
        
        # Persistent data store
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lnn_training.db')
        self.data_store = DataStore(db_path=db_path)

        # Training state
        self.is_trained = False
        
        # Load pre-trained model if available
        self._load_model()

        # If the model is not trained, check if there's enough data to start
        if not self.is_trained and self.data_store.get_data_count('train') >= 50:
            logger.info("Sufficient data found in DataStore. Scheduling initial model training.")
            asyncio.create_task(self.train_model())
        
        logger.info(f"LNN Fallback Client initialized on {self.device}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for LNN"""
        return {
            'vocab_size': 2000,
            'embed_dim': 256,
            'hidden_size': 256,
            'num_layers': 3,
            'output_size': 512,
            'dt': 0.1,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50,
            'sequence_length': 10
        }
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build a character-level vocabulary"""
        chars = ['<pad>', '<unk>', '<sos>', '<eos>'] + list("\"!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ \t\n\r")
        self.config['vocab_size'] = len(chars)
        return {char: idx for idx, char in enumerate(chars)}
    
    def tokenize(self, text: str, max_length: int = 256) -> List[int]:
        """Character-level tokenization"""
        token_ids = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        token_ids = [self.vocab['<sos>']] + token_ids + [self.vocab['<eos>']]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['<pad>']] * (max_length - len(token_ids)))
        return token_ids
    
    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Main analysis method - compatible with GrokClient interface"""
        try:
            if not self.is_trained:
                logger.warning("LNN model not trained, using basic pattern matching")
                return await self._pattern_based_analysis(prompt, context)
            
            token_ids = self.tokenize(prompt)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor, sequence_length=self.config['sequence_length'])
            
            accuracy_score = torch.argmax(outputs['accuracy_logits'], dim=1).item()
            methodology_score = torch.argmax(outputs['methodology_logits'], dim=1).item()
            explanation_score = torch.argmax(outputs['explanation_logits'], dim=1).item()
            confidence = outputs['confidence'].item()
            overall_score = accuracy_score * 0.5 + methodology_score * 0.3 + explanation_score * 0.2
            feedback = self._generate_feedback(accuracy_score, methodology_score, explanation_score)
            
            result = {
                "accuracy_score": accuracy_score,
                "methodology_score": methodology_score,
                "explanation_score": explanation_score,
                "overall_score": round(overall_score, 1),
                "feedback": feedback,
                "passed": overall_score >= 70,
                "confidence": round(confidence, 3),
                "analysis_type": "lnn_neural",
            }
            return json.dumps(result)
        except Exception as e:
            logger.error(f"LNN analysis failed: {e}", exc_info=True)
            return await self._pattern_based_analysis(prompt, context)
    
    async def _pattern_based_analysis(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Fallback pattern-based analysis when LNN is unavailable"""
        try:
            prompt_lower = prompt.lower()
            accuracy_score, methodology_score, explanation_score = 65, 65, 65
            
            if sum(1 for p in ['derivative', 'integral', 'equation', 'solve', 'calculate', '=', 'formula'] if p in prompt_lower) >= 3:
                accuracy_score += 15; methodology_score += 10
            if sum(1 for p in ['step', 'method', 'approach', 'because', 'therefore', 'thus'] if p in prompt_lower) >= 2:
                explanation_score += 15; methodology_score += 10
            if any(p in prompt_lower for p in ['error', 'wrong', 'incorrect', 'failed', 'undefined']):
                accuracy_score -= 25; methodology_score -= 15
            if len(prompt) > 300:
                explanation_score += 10; methodology_score += 5

            accuracy_score = max(0, min(100, accuracy_score))
            methodology_score = max(0, min(100, methodology_score))
            explanation_score = max(0, min(100, explanation_score))
            
            overall_score = accuracy_score * 0.5 + methodology_score * 0.3 + explanation_score * 0.2
            feedback = self._generate_feedback(accuracy_score, methodology_score, explanation_score)
            
            return json.dumps({
                "accuracy_score": accuracy_score, "methodology_score": methodology_score,
                "explanation_score": explanation_score, "overall_score": round(overall_score, 1),
                "feedback": feedback, "passed": overall_score >= 70, "confidence": 0.6,
                "analysis_type": "lnn_pattern_fallback",
            })
        except Exception as e:
            logger.error(f"Pattern-based analysis failed: {e}", exc_info=True)
            return json.dumps({
                "accuracy_score": 50, "methodology_score": 50, "explanation_score": 50,
                "overall_score": 50, "feedback": "Analysis unavailable - LNN fallback error",
                "passed": False, "confidence": 0.3, "analysis_type": "lnn_error_fallback",
            })
    
    def _generate_feedback(self, accuracy: int, methodology: int, explanation: int) -> str:
        """Generate human-readable feedback"""
        parts = []
        if accuracy >= 85: parts.append("Excellent mathematical accuracy")
        elif accuracy >= 70: parts.append("Good mathematical accuracy")
        elif accuracy >= 55: parts.append("Acceptable mathematical accuracy")
        else: parts.append("Mathematical accuracy needs improvement")
        
        if methodology >= 85: parts.append("well-structured methodology")
        elif methodology >= 70: parts.append("clear methodology")
        elif methodology >= 55: parts.append("adequate methodology")
        else: parts.append("methodology needs clarification")
        
        if explanation >= 85: parts.append("excellent detailed explanation")
        elif explanation >= 70: parts.append("good step-by-step explanation")
        elif explanation >= 55: parts.append("sufficient explanation")
        else: parts.append("explanation needs more detail")
        return ". ".join(parts).capitalize() + "."
        
    def add_training_data(self, prompt: str, expected_result: Dict[str, Any]):
        """Add training data for continuous learning"""
        train_count = self.data_store.get_data_count('train')
        validation_count = self.data_store.get_data_count('validation')
        data_type = 'train' if np.random.rand() < 0.8 or validation_count > (train_count * 0.25) else 'validation'
        self.data_store.add_data(prompt, expected_result, data_type)
        if self.data_store.get_data_count('train') >= 50 and not self.is_trained:
            logger.info("Reached sufficient data for initial training. Scheduling...")
            asyncio.create_task(self.train_model())

    async def train_model(self, epochs: Optional[int] = None):
        """Train the LNN model on collected data"""
        training_data = self.data_store.get_data('train')
        if len(training_data) < 10:
            logger.warning("Insufficient training data for LNN in DataStore. Need at least 10 samples.")
            return

        validation_data = self.data_store.get_data('validation')
        logger.info(f"Starting LNN model training on {len(training_data)} samples ({len(validation_data)} validation) from DataStore")

        try:
            epochs = epochs or self.config['epochs']
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            criterion = nn.CrossEntropyLoss()
            mse_criterion = nn.MSELoss()
            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in range(epochs):
                self.model.train()
                np.random.shuffle(training_data)
                train_losses = []
                for i in range(0, len(training_data), self.config['batch_size']):
                    optimizer.zero_grad()
                    batch = training_data[i:i+self.config['batch_size']]
                    if not batch: continue
                    loss = self._process_batch(batch, criterion, mse_criterion, optimizer)
                    train_losses.append(loss.detach())
                avg_train_loss = sum(l.item() for l in train_losses) / len(train_losses)

                self.model.eval()
                avg_val_loss = 0
                if validation_data:
                    with torch.no_grad():
                        val_losses = [self._process_batch(validation_data[i:i+self.config['batch_size']], criterion, mse_criterion) for i in range(0, len(validation_data), self.config['batch_size'])]
                        avg_val_loss = sum(l.item() for l in val_losses) / len(val_losses)
                
                scheduler.step(avg_val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    await self._save_model(is_best=True)
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= 10:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

            self.is_trained = True
            self._load_model(is_best=True)
            logger.info("LNN model training completed successfully.")
        except Exception as e:
            logger.error(f"LNN training failed: {e}", exc_info=True)
    
    def _process_batch(self, batch: List[Dict], criterion, mse_criterion, optimizer: Optional[optim.Optimizer] = None) -> torch.Tensor:
        """Helper to process a batch of data and return the loss. Performs backprop if optimizer is provided."""
        inputs, acc_targets, meth_targets, exp_targets, conf_targets = [], [], [], [], []
        for sample in batch:
            inputs.append(self.tokenize(sample['prompt']))
            expected = sample['expected']
            acc_targets.append(expected.get('accuracy_score', 70))
            meth_targets.append(expected.get('methodology_score', 70))
            exp_targets.append(expected.get('explanation_score', 70))
            conf_targets.append(expected.get('confidence', 0.8))

        input_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device)
        acc_tensor = torch.tensor(acc_targets, dtype=torch.long, device=self.device)
        meth_tensor = torch.tensor(meth_targets, dtype=torch.long, device=self.device)
        exp_tensor = torch.tensor(exp_targets, dtype=torch.long, device=self.device)
        conf_tensor = torch.tensor(conf_targets, dtype=torch.float, device=self.device).unsqueeze(1)

        outputs = self.model(input_tensor)
        
        acc_loss = criterion(outputs['accuracy_logits'], acc_tensor)
        meth_loss = criterion(outputs['methodology_logits'], meth_tensor)
        exp_loss = criterion(outputs['explanation_logits'], exp_tensor)
        conf_loss = mse_criterion(outputs['confidence'], conf_tensor)
        
        total_loss = acc_loss + meth_loss + exp_loss + conf_loss

        if optimizer: # Training mode
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return total_loss

    async def _save_model(self, is_best: bool = False):
        """Save trained model weights and metadata"""
        path = self.model_path.replace('.pth', '_best.pth') if is_best else self.model_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'vocab': self.vocab,
                'is_trained': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            torch.save(checkpoint, path)
            logger.info(f"LNN model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save LNN model to {path}: {e}")

    def _load_model(self, is_best: bool = False):
        """Load pre-trained model weights and metadata"""
        path = self.model_path.replace('.pth', '_best.pth') if is_best else self.model_path
        if not os.path.exists(path) and is_best:
            path = self.model_path

        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.config.update(checkpoint.get('config', {}))
                self.vocab = checkpoint.get('vocab', self.vocab)
                self.is_trained = checkpoint.get('is_trained', False)
                logger.info(f"LNN model loaded from {path} (trained: {self.is_trained})")
            except Exception as e:
                logger.warning(f"Could not load or parse LNN model from {path}: {e}. Starting fresh.")
                self.is_trained = False
        else:
            logger.info("No pre-trained LNN model found. Ready for initial training.")
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model state"""
        return {
            'is_trained': self.is_trained,
            'training_data_size': self.data_store.get_data_count('train'),
            'validation_data_size': self.data_store.get_data_count('validation'),
            'model_path': self.model_path,
            'device': str(self.device),
            'config': self.config,
            'vocab_size': len(self.vocab),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }