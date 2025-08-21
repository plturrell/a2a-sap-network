"""
Liquid Neural Network (LNN) Fallback for A2A AI Reasoning
Provides intelligent local reasoning when Grok API is unavailable
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from collections import Counter
import regex as re

logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for self-supervised learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss
        Args:
            features: L2-normalized features [batch_size, feature_dim]
            labels: Optional labels for supervised contrastive learning
        """
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pair mask
        if labels is not None:
            # Supervised contrastive
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
        else:
            # Self-supervised (adjacent samples are positive pairs)
            mask = torch.eye(batch_size, dtype=torch.float32, device=features.device)
            mask = torch.roll(mask, shifts=1, dims=0)
        
        # Remove diagonal
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # Compute log probabilities
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        return loss

class BPETokenizer:
    """Byte Pair Encoding tokenizer for improved text representation"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word_tokenizer = re.compile(r'\b\w+\b|[^\w\s]')
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
            '<mask>': 4, '<cls>': 5, '<sep>': 6
        }
        self.inverse_vocab = {}
        self.initialized = False
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train BPE on a corpus of texts"""
        # Get word frequencies
        word_freqs = Counter()
        for text in texts:
            words = self.word_tokenizer.findall(text.lower())
            for word in words:
                word_freqs[word] += 1
        
        # Initialize vocabulary with characters
        self.vocab = self.special_tokens.copy()
        vocab_idx = len(self.special_tokens)
        
        # Add single characters
        chars = set()
        for word in word_freqs:
            chars.update(word)
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = vocab_idx
                vocab_idx += 1
        
        # Split words into characters
        word_splits = {}
        for word, freq in word_freqs.items():
            if freq >= min_frequency:
                word_splits[word] = list(word)
        
        # Learn merges
        while len(self.vocab) < self.vocab_size:
            # Count pair frequencies
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                if word in word_splits:
                    split = word_splits[word]
                    for i in range(len(split) - 1):
                        pair = (split[i], split[i + 1])
                        pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)
            
            # Create new token
            new_token = ''.join(best_pair)
            self.vocab[new_token] = vocab_idx
            vocab_idx += 1
            
            # Update word splits
            for word in word_splits:
                split = word_splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = new_split
        
        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.initialized = True
        logger.info(f"BPE tokenizer trained with vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        if not self.initialized:
            # Fallback to character encoding
            return self._char_encode(text, max_length)
        
        words = self.word_tokenizer.findall(text.lower())
        tokens = [self.special_tokens['<sos>']]
        
        for word in words:
            # Apply BPE merges
            word_tokens = list(word)
            for merge in self.merges:
                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge:
                        new_word_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens
            
            # Convert to IDs
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.special_tokens['<unk>']))
        
        tokens.append(self.special_tokens['<eos>'])
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.special_tokens['<pad>']] * (max_length - len(tokens)))
        
        return tokens
    
    def _char_encode(self, text: str, max_length: int) -> List[int]:
        """Fallback character-level encoding"""
        tokens = [self.special_tokens['<sos>']]
        for char in text[:max_length - 2]:
            tokens.append(self.vocab.get(char, self.special_tokens['<unk>']))
        tokens.append(self.special_tokens['<eos>'])
        tokens.extend([self.special_tokens['<pad>']] * (max_length - len(tokens)))
        return tokens

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for long-range dependencies"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # Output projection
        output = self.out_linear(context)
        return output

class LiquidTimeConstantCell(nn.Module):
    """
    Enhanced Liquid Time-Constant (LTC) Neural Network Cell
    Implements continuous-time recurrent dynamics with adaptive time constants and attention
    """
    
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.use_attention = use_attention
        
        # Learnable time constants (tau) for each neuron
        self.tau = nn.Parameter(torch.ones(hidden_size) * 20.0)  # Initial tau = 20ms
        
        # Input transformation with layer normalization
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Recurrent weights with gating mechanism
        self.recurrent_linear = nn.Linear(hidden_size, hidden_size * 3, bias=False)  # For gates
        
        # Optional attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads=4, dropout=0.1)
            self.attn_norm = nn.LayerNorm(hidden_size)
        
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
    
    def _solve_rk4(self, x: torch.Tensor, hidden: torch.Tensor, hidden_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve the LTC ODE with 4th-order Runge-Kutta and gating"""
        
        def _dhdt(h, activated_input, tau, gates):
            # Apply gating mechanism
            update_gate, reset_gate = gates
            return (-h * reset_gate + activated_input * update_gate) / tau

        tau_pos = torch.abs(self.tau) + 1e-6
        
        # Compute input with normalization
        input_contrib = self.input_norm(self.input_linear(x))
        
        # Apply attention if enabled and we have history
        if self.use_attention and hidden_history is not None:
            # Self-attention over hidden states
            attn_out = self.attention(hidden.unsqueeze(1), hidden_history, hidden_history)
            input_contrib = input_contrib + self.attn_norm(attn_out.squeeze(1))
        
        # Compute gates
        gate_values = self.recurrent_linear(hidden)
        update_gate = torch.sigmoid(gate_values[:, :self.hidden_size])
        reset_gate = torch.sigmoid(gate_values[:, self.hidden_size:2*self.hidden_size])
        candidate = self.activation(gate_values[:, 2*self.hidden_size:])
        
        activated = candidate + input_contrib
        gates = (update_gate, reset_gate)

        # RK4 steps with gating
        k1 = self.dt * _dhdt(hidden, activated, tau_pos, gates)
        
        # For k2
        gate_values_k2 = self.recurrent_linear(hidden + 0.5 * k1)
        gates_k2 = (torch.sigmoid(gate_values_k2[:, :self.hidden_size]), 
                    torch.sigmoid(gate_values_k2[:, self.hidden_size:2*self.hidden_size]))
        activated_k2 = self.activation(gate_values_k2[:, 2*self.hidden_size:]) + input_contrib
        k2 = self.dt * _dhdt(hidden + 0.5 * k1, activated_k2, tau_pos, gates_k2)

        # For k3
        gate_values_k3 = self.recurrent_linear(hidden + 0.5 * k2)
        gates_k3 = (torch.sigmoid(gate_values_k3[:, :self.hidden_size]), 
                    torch.sigmoid(gate_values_k3[:, self.hidden_size:2*self.hidden_size]))
        activated_k3 = self.activation(gate_values_k3[:, 2*self.hidden_size:]) + input_contrib
        k3 = self.dt * _dhdt(hidden + 0.5 * k2, activated_k3, tau_pos, gates_k3)

        # For k4
        gate_values_k4 = self.recurrent_linear(hidden + k3)
        gates_k4 = (torch.sigmoid(gate_values_k4[:, :self.hidden_size]), 
                    torch.sigmoid(gate_values_k4[:, self.hidden_size:2*self.hidden_size]))
        activated_k4 = self.activation(gate_values_k4[:, 2*self.hidden_size:]) + input_contrib
        k4 = self.dt * _dhdt(hidden + k3, activated_k4, tau_pos, gates_k4)

        # Combine steps
        new_hidden = hidden + (k1 + 2*k2 + 2*k3 + k4) / 6
        return new_hidden

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, hidden_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass implementing ODE with a chosen solver
        """
        return self.ode_solver(x, hidden, hidden_history)


class LiquidNeuralNetwork(nn.Module):
    """
    Enhanced Liquid Neural Network for A2A reasoning tasks
    Features: Larger capacity, attention mechanisms, and advanced training support
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.get('vocab_size', 8000)  # Increased for BPE
        self.embed_dim = config.get('embed_dim', 512)     # Larger embeddings
        self.hidden_size = config.get('hidden_size', 768)  # Much larger capacity
        self.num_layers = config.get('num_layers', 6)      # Deeper model
        self.output_size = config.get('output_size', 1024) # Larger output space
        self.dt = config.get('dt', 0.1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.dropout = config.get('dropout', 0.1)
        
        # Embedding layer with positional encoding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoding = self._create_positional_encoding()
        self.embed_dropout = nn.Dropout(self.dropout)
        
        # Layer normalization for embeddings
        self.embed_norm = nn.LayerNorm(self.embed_dim)
        
        # Projection layer to match hidden size
        self.input_projection = nn.Linear(self.embed_dim, self.hidden_size)
        
        # LTC layers with attention
        self.ltc_layers = nn.ModuleList([
            LiquidTimeConstantCell(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                dt=self.dt,
                use_attention=True
            ) for i in range(self.num_layers)
        ])
        
        # Cross-layer connections (residual)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = MultiHeadAttention(self.hidden_size, num_heads=12, dropout=self.dropout)
        self.global_norm = nn.LayerNorm(self.hidden_size)
        
        # Output projection with deeper network
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Task-specific heads with more capacity
        self.accuracy_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 101)
        )
        self.methodology_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 101)
        )
        self.explanation_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 101)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(self.output_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1)
        )
        
        # Contrastive learning projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Initialize hidden states and history
        self.hidden_states = None
        self.hidden_history = None
        
    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(self.max_seq_length, self.embed_dim)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           -(np.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states for all layers"""
        self.hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        self.hidden_history = []
    
    def forward(self, input_ids: torch.Tensor, sequence_length: int = 10,
                return_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass through the LNN with attention and residual connections
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            sequence_length: Number of time steps to run LTC dynamics
            return_contrastive: Whether to return contrastive embeddings
            
        Returns:
            Dict with reasoning outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize hidden states if needed
        if self.hidden_states is None or self.hidden_states[0].shape[0] != batch_size:
            self.init_hidden(batch_size, device)
        else:
            # Detach hidden states to prevent backprop through previous sequences
            self.hidden_states = [h.detach() for h in self.hidden_states]
            if self.hidden_history:
                self.hidden_history = [h.detach() for h in self.hidden_history]
        
        # Embed input tokens with positional encoding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        embedded = self.embed_dropout(self.embed_norm(embedded))
        
        # Project to hidden size
        embedded = self.input_projection(embedded)  # [batch_size, seq_len, hidden_size]
        
        # Process sequence token by token with attention
        all_hidden_states = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            # Stack previous hidden states for attention
            if self.hidden_history:
                history_tensor = torch.stack(self.hidden_history, dim=1)  # [batch_size, history_len, hidden_size]
            else:
                history_tensor = None
            
            # Process through LTC layers with residual connections
            layer_input = x_t
            for i, (ltc_layer, layer_norm) in enumerate(zip(self.ltc_layers, self.layer_norms)):
                # LTC forward with attention
                new_hidden = ltc_layer(layer_input, self.hidden_states[i], history_tensor)
                
                # Residual connection and layer norm
                if i > 0:  # Skip residual for first layer
                    new_hidden = layer_norm(new_hidden + self.hidden_states[i])
                else:
                    new_hidden = layer_norm(new_hidden)
                
                self.hidden_states[i] = new_hidden
                layer_input = new_hidden
            
            # Store final layer hidden state in history
            all_hidden_states.append(self.hidden_states[-1].clone())
            
            # Keep limited history (last 32 timesteps)
            if len(self.hidden_history) >= 32:
                self.hidden_history.pop(0)
            self.hidden_history.append(self.hidden_states[-1].clone())
        
        # Stack all hidden states
        all_hidden = torch.stack(all_hidden_states, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # Global attention pooling
        global_attn = self.global_attention(
            all_hidden[:, -1:, :],  # Query with last hidden state
            all_hidden,             # Keys and values from all states
            all_hidden
        )
        final_hidden = self.global_norm(global_attn.squeeze(1) + self.hidden_states[-1])
        
        # Project to output space
        output_features = self.output_proj(final_hidden)
        
        # Generate predictions
        accuracy_logits = self.accuracy_head(output_features)
        methodology_logits = self.methodology_head(output_features)
        explanation_logits = self.explanation_head(output_features)
        confidence = torch.sigmoid(self.confidence_head(output_features))
        
        results = {
            'accuracy_logits': accuracy_logits,
            'methodology_logits': methodology_logits,
            'explanation_logits': explanation_logits,
            'confidence': confidence,
            'features': output_features
        }
        
        # Add contrastive embeddings if requested
        if return_contrastive:
            results['contrastive_features'] = F.normalize(
                self.contrastive_proj(output_features), p=2, dim=1
            )
        
        return results


class LNNFallbackClient:
    """
    Enhanced LNN-based fallback client for A2A AI reasoning
    Features: BPE tokenization, larger models, advanced training
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'models', 'lnn_fallback.pth'
        )
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BPE tokenizer
        self.tokenizer = BPETokenizer(vocab_size=self.config['vocab_size'])
        
        # Initialize model
        self.model = LiquidNeuralNetwork(self.config)
        self.model.to(self.device)
        
        # Persistent data store
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lnn_training.db')
        self.data_store = DataStore(db_path=db_path)

        # Training state
        self.is_trained = False
        self.training_step = 0
        
        # Load pre-trained model if available
        self._load_model()
        
        # Initialize tokenizer if we have training data
        if not self.tokenizer.initialized and self.data_store.get_data_count('train') >= 10:
            self._initialize_tokenizer()

        # If the model is not trained, check if there's enough data to start
        if not self.is_trained and self.data_store.get_data_count('train') >= 50:
            logger.info("Sufficient data found in DataStore. Scheduling initial model training.")
            asyncio.create_task(self.train_model())
        
        logger.info(f"Enhanced LNN Fallback Client initialized on {self.device}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Enhanced default configuration for LNN"""
        return {
            'vocab_size': 8000,          # Increased for BPE
            'embed_dim': 512,            # Larger embeddings
            'hidden_size': 768,          # Much larger hidden size
            'num_layers': 6,             # Deeper model
            'output_size': 1024,         # Larger output
            'dt': 0.1,
            'learning_rate': 0.0001,     # Lower LR for larger model
            'batch_size': 32,            # Larger batches
            'epochs': 100,               # More epochs
            'sequence_length': 10,
            'max_seq_length': 512,       # For positional encoding
            'dropout': 0.1,
            'warmup_steps': 1000,        # LR warmup
            'gradient_clip': 1.0,        # Gradient clipping
            'contrastive_temp': 0.07,    # Temperature for contrastive loss
            'contrastive_weight': 0.1    # Weight for contrastive loss
        }
    
    def _initialize_tokenizer(self):
        """Initialize BPE tokenizer with training data"""
        try:
            training_data = self.data_store.get_data('train')
            if len(training_data) < 10:
                logger.warning("Not enough data to train BPE tokenizer")
                return
            
            # Extract prompts for tokenizer training
            texts = [sample['prompt'] for sample in training_data]
            
            # Train BPE tokenizer
            self.tokenizer.train(texts)
            logger.info("BPE tokenizer initialized with training data")
            
        except Exception as e:
            logger.error(f"Failed to initialize BPE tokenizer: {e}")
    
    def tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """Tokenize text using BPE"""
        return self.tokenizer.encode(text, max_length)
    
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

    async def train_model(self, epochs: Optional[int] = None, pretrain: bool = True):
        """Enhanced training with contrastive learning and self-supervised pretraining"""
        training_data = self.data_store.get_data('train')
        if len(training_data) < 10:
            logger.warning("Insufficient training data for LNN in DataStore. Need at least 10 samples.")
            return

        validation_data = self.data_store.get_data('validation')
        logger.info(f"Starting enhanced LNN training on {len(training_data)} samples ({len(validation_data)} validation)")

        try:
            # Initialize tokenizer if needed
            if not self.tokenizer.initialized:
                self._initialize_tokenizer()
            
            epochs = epochs or self.config['epochs']
            
            # Optimizer with different LR for different parts
            param_groups = [
                {'params': self.model.embedding.parameters(), 'lr': self.config['learning_rate'] * 0.1},
                {'params': self.model.ltc_layers.parameters(), 'lr': self.config['learning_rate']},
                {'params': self.model.output_proj.parameters(), 'lr': self.config['learning_rate']},
                {'params': [p for name, p in self.model.named_parameters() 
                           if 'head' in name], 'lr': self.config['learning_rate'] * 2}
            ]
            
            optimizer = optim.AdamW(param_groups, lr=self.config['learning_rate'], 
                                   weight_decay=0.01)
            
            # Learning rate scheduler with warmup
            def lr_lambda(step):
                if step < self.config['warmup_steps']:
                    return step / self.config['warmup_steps']
                return 1.0
            
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=5, factor=0.5
            )
            
            # Loss functions
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            mse_criterion = nn.MSELoss()
            contrastive_criterion = ContrastiveLoss(temperature=self.config['contrastive_temp'])
            
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            # Self-supervised pretraining phase
            if pretrain and epochs > 20:
                logger.info("Starting self-supervised pretraining phase...")
                pretrain_epochs = min(20, epochs // 5)
                
                for epoch in range(pretrain_epochs):
                    self.model.train()
                    np.random.shuffle(training_data)
                    
                    for i in range(0, len(training_data), self.config['batch_size']):
                        batch = training_data[i:i+self.config['batch_size']]
                        if len(batch) < 4: continue  # Need at least 4 samples for contrastive
                        
                        # Prepare augmented batch (simple: add slight noise to prompts)
                        augmented_batch = []
                        for sample in batch:
                            # Create positive pair by slightly modifying prompt
                            aug_prompt = self._augment_text(sample['prompt'])
                            augmented_batch.append({'prompt': aug_prompt, 'expected': sample['expected']})
                        
                        # Process both original and augmented
                        optimizer.zero_grad()
                        loss = self._process_contrastive_batch(
                            batch + augmented_batch, contrastive_criterion
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                        optimizer.step()
                        
                        self.training_step += 1
                        if self.training_step < self.config['warmup_steps']:
                            warmup_scheduler.step()
                    
                    logger.info(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} completed")
            
            # Main training phase
            logger.info("Starting main training phase...")
            for epoch in range(epochs):
                self.model.train()
                np.random.shuffle(training_data)
                train_losses = []
                
                for i in range(0, len(training_data), self.config['batch_size']):
                    optimizer.zero_grad()
                    batch = training_data[i:i+self.config['batch_size']]
                    if not batch: continue
                    
                    # Combined loss: task loss + contrastive loss
                    task_loss = self._process_batch(batch, criterion, mse_criterion, optimizer,
                                                   include_contrastive=True, contrastive_criterion=contrastive_criterion)
                    train_losses.append(task_loss.detach())
                    
                    self.training_step += 1
                    if self.training_step < self.config['warmup_steps']:
                        warmup_scheduler.step()
                
                avg_train_loss = sum(l.item() for l in train_losses) / len(train_losses)

                # Validation
                self.model.eval()
                avg_val_loss = 0
                if validation_data:
                    with torch.no_grad():
                        val_losses = []
                        for i in range(0, len(validation_data), self.config['batch_size']):
                            batch = validation_data[i:i+self.config['batch_size']]
                            if batch:
                                loss = self._process_batch(batch, criterion, mse_criterion)
                                val_losses.append(loss)
                        avg_val_loss = sum(l.item() for l in val_losses) / len(val_losses) if val_losses else 0
                
                # Learning rate scheduling
                if self.training_step >= self.config['warmup_steps']:
                    plateau_scheduler.step(avg_val_loss)
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

                # Early stopping and checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    await self._save_model(is_best=True)
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= 15:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
                
                # Periodic checkpointing
                if (epoch + 1) % 10 == 0:
                    await self._save_model(is_best=False)

            self.is_trained = True
            self._load_model(is_best=True)
            logger.info("Enhanced LNN model training completed successfully.")
            
        except Exception as e:
            logger.error(f"Enhanced LNN training failed: {e}", exc_info=True)
    
    def _process_batch(self, batch: List[Dict], criterion, mse_criterion, 
                      optimizer: Optional[optim.Optimizer] = None,
                      include_contrastive: bool = False,
                      contrastive_criterion: Optional[nn.Module] = None) -> torch.Tensor:
        """Enhanced batch processing with optional contrastive learning"""
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

        # Forward pass with contrastive features if needed
        outputs = self.model(input_tensor, return_contrastive=include_contrastive)
        
        # Task-specific losses
        acc_loss = criterion(outputs['accuracy_logits'], acc_tensor)
        meth_loss = criterion(outputs['methodology_logits'], meth_tensor)
        exp_loss = criterion(outputs['explanation_logits'], exp_tensor)
        conf_loss = mse_criterion(outputs['confidence'], conf_tensor)
        
        total_loss = acc_loss + meth_loss + exp_loss + conf_loss
        
        # Add contrastive loss if enabled
        if include_contrastive and contrastive_criterion and 'contrastive_features' in outputs:
            # Create pseudo-labels based on overall scores
            overall_scores = (acc_targets + meth_targets + exp_targets) / 3
            score_labels = torch.tensor([int(s // 10) for s in overall_scores], 
                                       device=self.device)
            
            contrastive_loss = contrastive_criterion(outputs['contrastive_features'], score_labels)
            total_loss = total_loss + self.config['contrastive_weight'] * contrastive_loss

        if optimizer:  # Training mode
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            optimizer.step()

        return total_loss
    
    def _process_contrastive_batch(self, batch: List[Dict], contrastive_criterion: nn.Module) -> torch.Tensor:
        """Process batch for contrastive pretraining"""
        inputs = []
        for sample in batch:
            inputs.append(self.tokenize(sample['prompt']))
        
        input_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device)
        outputs = self.model(input_tensor, return_contrastive=True)
        
        # Contrastive loss with self-supervised learning
        # First half are originals, second half are augmented
        if 'contrastive_features' in outputs:
            features = outputs['contrastive_features']
            loss = contrastive_criterion(features)
            return loss
        else:
            # Fallback to feature-based loss
            features = F.normalize(outputs['features'], p=2, dim=1)
            loss = contrastive_criterion(features)
            return loss
    
    def _augment_text(self, text: str) -> str:
        """Simple text augmentation for contrastive learning"""
        import random
        
        # Simple augmentations
        augmentations = [
            lambda t: t.replace('.', '!'),  # Change punctuation
            lambda t: t.replace(',', ';'),
            lambda t: t.lower() if random.random() > 0.5 else t.upper(),
            lambda t: ' '.join(t.split()[::-1]) if len(t.split()) < 10 else t,  # Reverse short texts
            lambda t: t + ' ' + random.choice(['Indeed.', 'Right?', 'Clearly.']),
            lambda t: random.choice(['Well, ', 'So, ', 'Now, ']) + t
        ]
        
        # Apply 1-2 random augmentations
        num_augs = random.randint(1, 2)
        augmented = text
        for _ in range(num_augs):
            aug_func = random.choice(augmentations)
            augmented = aug_func(augmented)
        
        return augmented

    async def _save_model(self, is_best: bool = False):
        """Save trained model weights and metadata"""
        path = self.model_path.replace('.pth', '_best.pth') if is_best else self.model_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'tokenizer_vocab': self.tokenizer.vocab,
                'tokenizer_merges': self.tokenizer.merges,
                'tokenizer_initialized': self.tokenizer.initialized,
                'is_trained': True,
                'training_step': self.training_step,
                'timestamp': datetime.utcnow().isoformat()
            }
            torch.save(checkpoint, path)
            logger.info(f"Enhanced LNN model saved to {path}")
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
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.config.update(checkpoint.get('config', {}))
                
                # Load tokenizer state
                if 'tokenizer_vocab' in checkpoint:
                    self.tokenizer.vocab = checkpoint['tokenizer_vocab']
                    self.tokenizer.merges = checkpoint.get('tokenizer_merges', [])
                    self.tokenizer.initialized = checkpoint.get('tokenizer_initialized', False)
                    self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
                
                self.is_trained = checkpoint.get('is_trained', False)
                self.training_step = checkpoint.get('training_step', 0)
                
                logger.info(f"Enhanced LNN model loaded from {path} "
                          f"(trained: {self.is_trained}, step: {self.training_step})")
            except Exception as e:
                logger.warning(f"Could not load LNN model from {path}: {e}. Starting fresh.")
                self.is_trained = False
        else:
            logger.info("No pre-trained LNN model found. Ready for initial training.")
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the enhanced model state"""
        return {
            'is_trained': self.is_trained,
            'training_step': self.training_step,
            'training_data_size': self.data_store.get_data_count('train'),
            'validation_data_size': self.data_store.get_data_count('validation'),
            'model_path': self.model_path,
            'device': str(self.device),
            'config': self.config,
            'tokenizer_type': 'BPE',
            'tokenizer_initialized': self.tokenizer.initialized,
            'vocab_size': len(self.tokenizer.vocab) if self.tokenizer.initialized else self.config['vocab_size'],
            'model_architecture': {
                'type': 'Enhanced LNN with Attention',
                'hidden_size': self.config['hidden_size'],
                'num_layers': self.config['num_layers'],
                'embed_dim': self.config['embed_dim'],
                'output_size': self.config['output_size'],
                'has_attention': True,
                'has_contrastive': True
            },
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }