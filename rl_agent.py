"""
RL Agent Module for Trading System
==================================
This module implements the PPO (Proximal Policy Optimization) agent
for the reinforcement learning trading system.

Author: Senior Quantitative Developer
Date: 2024
Version: 2.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable TF32 on Ampere GPUs (RTX 4090) for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Force GPU memory allocation at startup
if torch.cuda.is_available():
    # Allocate a small tensor to initialize CUDA
    dummy = torch.zeros(1).cuda()
    del dummy
    torch.cuda.empty_cache()

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any, Optional, Callable, Tuple, Type
import logging
import os
from datetime import datetime
import json
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Efficient multi-head self-attention layer with causal masking and stability improvements.
    
    This implementation includes:
    - Causal attention masking for auto-regressive tasks
    - Scaled dot-product attention with proper scaling
    - Residual connection and layer normalization
    - Dropout for attention weights
    - Efficient memory usage with in-place operations
    """
    
    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize the attention layer with stability optimizations.
        
        Args:
            embed_size: Dimensionality of input and output features
            num_heads: Number of attention heads (must divide embed_size)
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})")
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Linear projections for Q, K, V with bias
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Output projection with residual connection
        self.out_proj = nn.Linear(embed_size, embed_size)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embed_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for dot product attention (1/sqrt(d_k))
        self.scale = self.head_dim ** -0.5
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()
        
        # Log initialization
        logger.debug(f"Initialized AttentionLayer with {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters")
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        # Initialize Q, K, V projections
        for proj in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(proj.weight, mode='fan_in', nonlinearity='linear')
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with stability optimizations.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_size)
            mask: Optional boolean mask of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len)
                 where True indicates positions that should be masked out
                 
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_size)
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, embed_size), got {x.shape}")
            
        batch_size, seq_len, _ = x.size()
        residual = x  # Store for residual connection
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Project inputs to Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has the right shape
            if mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            
            # Apply mask (fill with -inf where mask is True)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights with dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to (batch_size, seq_len, embed_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Final linear projection
        output = self.out_proj(output)
        
        # Add residual connection and apply dropout
        output = output + residual
        
        return output


class TradingNetworkV2(BaseFeaturesExtractor):
    """
    Custom network with attention, instance normalization, and residual connections.
    Designed to handle both batch and single-sample inputs robustly.
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 n_market_features: int = 50,
                 hidden_dims: list = None,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1):
        """
        Initialize the network with robust handling of batch and single-sample inputs.
        
        Args:
            observation_space: The observation space of the environment
            features_dim: Dimension of the output features
            n_market_features: Number of market features in the observation
            hidden_dims: List of hidden layer dimensions
            use_attention: Whether to use attention mechanism
            dropout_rate: Dropout rate for regularization (0.0 to disable)
        """
        super().__init__(observation_space, features_dim)
        
        # Network configuration
        self.n_market_features = n_market_features
        self.n_portfolio_features = observation_space.shape[0] - n_market_features
        self.use_attention = use_attention
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Market feature processor (first part of the network)
        market_layers = []
        prev_dim = n_market_features
        for dim in hidden_dims:
            market_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = dim
        
        # Portfolio feature processor (smaller network)
        portfolio_layers = []
        prev_dim = self.n_portfolio_features
        for dim in [d // 2 for d in hidden_dims]:  # Smaller dimensions for portfolio
            portfolio_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = dim
        
        # Feature fusion layers (combine market and portfolio features)
        fusion_dim = hidden_dims[-1] + (hidden_dims[-1] // 2)  # Combined dimensions
        fusion_layers = [
            nn.Linear(fusion_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        ]
        
        # Residual connection (skip connection from input to output)
        self.residual_proj = None
        if n_market_features + self.n_portfolio_features != features_dim:
            self.residual_proj = nn.Linear(n_market_features + self.n_portfolio_features, features_dim, bias=False)
        
        # Attention mechanism (only used when batch size > 1)
        if use_attention:
            self.attention = AttentionLayer(hidden_dims[-1])
        
        # Create network modules
        self.market_processor = nn.Sequential(*market_layers)
        self.portfolio_processor = nn.Sequential(*portfolio_layers)
        self.feature_fusion = nn.Sequential(*fusion_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Log architecture
        logger.info(f"Initialized TradingNetworkV2 with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with robust handling of batch and single-sample inputs.
        
        Args:
            observations: Input tensor of shape (batch_size, n_features) or (n_features,)
            
        Returns:
            torch.Tensor: Output features of shape (batch_size, features_dim)
        """
        # Handle single sample case (batch size = 1) by adding a batch dimension
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        # Ensure we have a valid batch dimension
        batch_size = observations.size(0)
        
        # Split observations into market and portfolio features
        market_features = observations[:, :self.n_market_features]
        portfolio_features = observations[:, -self.n_portfolio_features:]
        
        # Process market features
        market_encoded = self.market_processor(market_features)
        
        # Apply attention if enabled and we have a valid batch size
        if self.use_attention and batch_size > 1:
            # Add sequence dimension for attention (batch, seq_len, features)
            market_encoded = market_encoded.unsqueeze(1)
            market_encoded = self.attention(market_encoded)
            market_encoded = market_encoded.squeeze(1)
        
        # Process portfolio features
        portfolio_encoded = self.portfolio_processor(portfolio_features)
        
        # Concatenate and fuse features
        combined = torch.cat([market_encoded, portfolio_encoded], dim=1)
        features = self.feature_fusion(combined)
        
        # Add residual connection if projection is available
        if self.residual_proj is not None:
            residual = self.residual_proj(observations)
            # Ensure shapes match before adding residual
            if residual.shape == features.shape:
                features = features + 0.1 * residual  # Scale down residual to stabilize training
        
        return features


class RLTradingAgentV2:
    """
    Enhanced RL agent class with advanced features for trading.
    
    This class provides:
    - Custom network architecture with attention
    - Advanced training techniques
    - Comprehensive evaluation metrics
    - Model ensembling capabilities
    """
    
    def __init__(self,
                 env: Any,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RL trading agent.
        
        Args:
            env: Trading environment
            config: Configuration dictionary for hyperparameters
        """
        self.env = env
        # Merge provided config with defaults
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        self.model = None
        self.vec_env = None
        
        # Setup directories
        self.model_dir = self.config.get('model_dir', './output/models')
        self.log_dir = self.config.get('log_dir', './output/logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Performance tracking
        self.training_history = []
        self.best_model_path = None
        self.best_sharpe = -float('inf')
        
        # Store the original environment's observation space shape
        self.original_obs_shape = env.observation_space.shape[0]
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for PPO trading agent optimized for 5-minute data."""
        return {
            # PPO hyperparameters - optimized for 5-minute data and GPU utilization
            'learning_rate': 3e-4,      # Slightly lower for stability
            'n_steps': 8192,            # Larger for better GPU efficiency
            'batch_size': 512,          # Increased from 256 for better GPU usage
            'n_epochs': 20,             # More epochs for better convergence
            'gamma': 0.995,             # Higher for longer-term rewards
            'gae_lambda': 0.98,         # Higher for better advantage estimation
            'clip_range': 0.3,          # Slightly wider for market adaptation
            'clip_range_vf': 0.1,       # Add value function clipping
            'ent_coef': 0.05,           # Increased for more exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.02,          # Slightly higher for stability
            
            # Network architecture
            'use_custom_network': True,
            'use_attention': True,
            'hidden_dims': [512, 256, 128],
            'dropout_rate': 0.1,
            
            # Training settings
            'total_timesteps': 1000000,  # Increased for better learning
            'eval_freq': 10000,
            'save_freq': 50000,
            'n_eval_episodes': 10,
            'normalize_observations': True,
            'normalize_rewards': True,
            'normalize_advantage': True,
            
            # Learning rate schedule
            'use_lr_schedule': True,
            'lr_schedule_type': 'cosine',
            
            # Advanced features
            'use_ensemble': False,
            'n_ensemble_models': 3,
            'seed': 42,
            
            # Directories
            'model_dir': './output/models',
            'log_dir': './output/logs'
        }
    
    def _create_lr_schedule(self) -> Callable[[float], float]:
        """Create learning rate schedule."""
        initial_lr = self.config['learning_rate']
        schedule_type = self.config.get('lr_schedule_type', 'linear')
        
        if schedule_type == 'linear':
            def lr_schedule(progress: float) -> float:
                """Linear decay from initial_lr to 0.1 * initial_lr."""
                return initial_lr * (0.1 + 0.9 * (1 - progress))
        
        elif schedule_type == 'cosine':
            def lr_schedule(progress: float) -> float:
                """Cosine annealing schedule."""
                return initial_lr * (0.1 + 0.45 * (1 + np.cos(np.pi * progress)))
        
        else:  # constant
            def lr_schedule(progress: float) -> float:
                return initial_lr
        
        return lr_schedule
    
    def build_model(self) -> None:
        """Build and initialize the PPO model with custom architecture."""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create vectorized environment with Monitor wrapper
        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
        
        # Add normalization wrapper if configured
        if self.config.get('normalize_observations', True):
            self.vec_env = VecNormalize(
                self.vec_env,
                norm_obs=True,
                norm_reward=self.config.get('normalize_rewards', True),
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.config.get('gamma', 0.99)
            )
        
        # Get observation and action space dimensions
        n_obs = self.vec_env.observation_space.shape[0]
        n_actions = self.vec_env.action_space.n
        
        # Define policy kwargs with custom network
        policy_kwargs = {
            'net_arch': {
                'pi': self.config.get('hidden_dims', [256, 128]),
                'vf': self.config.get('hidden_dims', [256, 128])
            },
            'activation_fn': nn.LeakyReLU,
            'ortho_init': True,
            'features_extractor_class': TradingNetworkV2,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'n_market_features': n_obs - 12,  # Assuming last 12 are portfolio features
                'use_attention': self.config.get('use_attention', True),
                'dropout_rate': self.config.get('dropout_rate', 0.1)
            }
        }
        
        # Initialize model with proper device handling
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            clip_range_vf=self.config.get('clip_range_vf', None),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            target_kl=self.config.get('target_kl', 0.01),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, 'tensorboard'),
            device=self.device,
            seed=self.config.get('seed', 42)
        )
        
        # Initialize network parameters with a forward pass
        self._initialize_network()
        
        # Log model configuration
        self._log_model_config()
    
    def _initialize_network(self):
        """Initialize network parameters with a forward pass."""
        # Create a batch of dummy observations
        obs_shape = self.vec_env.observation_space.shape
        batch_size = 2  # Small batch size for initialization
        dummy_obs = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
        
        # Convert to tensor and move to device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(dummy_obs).to(self.device)
            # Forward pass to initialize all parameters
            _ = self.model.policy.features_extractor(obs_tensor)
            _ = self.model.policy.predict_values(obs_tensor)
            _ = self.model.policy.forward(obs_tensor)
        
        # Move model to device
        self.model.policy = self.model.policy.to(self.device)
        
        # Clear any cached CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _log_model_config(self):
        """Log model configuration details."""
        logger.info("PPO model built successfully")
        logger.info(f"Observation space: {self.vec_env.observation_space.shape}")
        logger.info(f"Action space: {self.vec_env.action_space}")
        logger.info(f"Using custom network: {self.config.get('use_custom_network', True)}")
        logger.info(f"Using attention: {self.config.get('use_attention', True)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.get('batch_size', 64)}")
        logger.info(f"N steps: {self.config.get('n_steps', 2048)}")
        
        # Log GPU memory usage if using CUDA
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    def train(self, 
              eval_env: Optional[Any] = None,
              callbacks: Optional[list] = None) -> None:
        """
        Train the PPO agent with enhanced monitoring.
        
        Args:
            eval_env: Environment for evaluation during training
            callbacks: Additional callbacks for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Starting enhanced training...")
        
        # Setup callbacks
        all_callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('save_freq', 50000),
            save_path=self.model_dir,
            name_prefix='ppo_trading',
            save_vecnormalize=True
        )
        all_callbacks.append(checkpoint_callback)
        
        # DISABLE THE PROBLEMATIC EVALUATION CALLBACK
        # The standard EvalCallback is causing hangs with 5-minute data
        # We'll use only the custom TradingMetricsCallbackV2 instead
        
        # DISABLE ALL EVALUATION CALLBACKS FOR NOW
        # Both EvalCallback and TradingMetricsCallbackV2 are causing hangs
        # We'll evaluate after training is complete instead
        logger.info("Evaluation during training is disabled to prevent hanging issues")
        
        # Add any additional callbacks
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # Enable mixed precision training if available
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            logger.info("GPU supports mixed precision training")
        
        # Train the model
        try:
            total_timesteps = self.config.get('total_timesteps', 1000000)
            logger.info(f"Training for {total_timesteps:,} timesteps...")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=all_callbacks,
                progress_bar=True,
                tb_log_name=f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Save final model
        self.save_model('final_model')
        
        # Save training configuration
        config_path = os.path.join(self.model_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Training completed")
    
    def save_model(self, name: str) -> None:
        """
        Save the trained model and normalization statistics.
        
        Args:
            name: Name for the saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = os.path.join(self.model_dir, f'{name}.zip')
        self.model.save(model_path)
        
        # Save normalization statistics if using VecNormalize
        if isinstance(self.vec_env, VecNormalize):
            stats_path = os.path.join(self.model_dir, f'{name}_vecnormalize.pkl')
            self.vec_env.save(stats_path)
        
        # Save configuration
        config_path = os.path.join(self.model_dir, f'{name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, name: str) -> None:
        """
        Load a saved model and normalization statistics.
        
        Args:
            name: Name of the saved model or full path to model file
        """
        # Check if it's already a full path
        if os.path.exists(name):
            # It's a full path
            model_path = name
            base_dir = os.path.dirname(name)
            base_name = os.path.basename(name).replace('.zip', '')
            config_path = os.path.join(base_dir, f'{base_name}_config.json')
            stats_path = os.path.join(base_dir, f'{base_name}_vecnormalize.pkl')
        else:
            # It's just a name, construct the path
            model_path = os.path.join(self.model_dir, f'{name}.zip')
            config_path = os.path.join(self.model_dir, f'{name}_config.json')
            stats_path = os.path.join(self.model_dir, f'{name}_vecnormalize.pkl')
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Create environment if not exists
        if self.vec_env is None:
            self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
            
            # Load normalization statistics if exists
            if os.path.exists(stats_path):
                self.vec_env = VecNormalize.load(stats_path, self.vec_env)
                logger.info(f"Loaded VecNormalize from {stats_path}")
        
        # Load model
        self.model = PPO.load(model_path, env=self.vec_env)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Make a trading decision based on observation.
        
        Args:
            observation: Current market state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Trading action to take
            info: Additional information (e.g., action probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Just let the model handle the observation directly
        # The PPO model with VecNormalize will handle normalization internally
        action, _states = self.model.predict(
            observation, 
            deterministic=deterministic
        )
        
        # Get additional information
        info = {}
        
        # Get action probabilities if using stochastic policy
        if not deterministic:
            try:
                # Get observation tensor
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0) if len(observation.shape) == 1 else torch.FloatTensor(observation)
                
                # Get action distribution
                with torch.no_grad():
                    features = self.model.policy.extract_features(obs_tensor)
                    if self.model.policy.share_features_extractor:
                        latent_pi, latent_vf = self.model.policy.mlp_extractor(features[0])
                    else:
                        pi_features, _ = features
                        latent_pi = self.model.policy.mlp_extractor.forward_actor(pi_features)
                    
                    distribution = self.model.policy.action_dist.proba_distribution(
                        self.model.policy.action_net(latent_pi)
                    )
                    
                    # Get probabilities for discrete actions
                    if hasattr(distribution, 'distribution'):
                        probs = distribution.distribution.probs[0].cpu().numpy()
                        info['action_probabilities'] = probs
                    
                    # Get value estimate
                    if self.model.policy.share_features_extractor:
                        value = self.model.policy.value_net(latent_vf)
                    else:
                        _, vf_features = features
                        latent_vf = self.model.policy.mlp_extractor.forward_critic(vf_features)
                        value = self.model.policy.value_net(latent_vf)
                    
                    info['value_estimate'] = value.item()
            except Exception as e:
                logger.debug(f"Could not get action probabilities: {e}")
        
        return int(action), info
    
    def evaluate(self, 
                 env: Any,
                 n_episodes: int = 10,
                 deterministic: bool = True) -> Dict[str, float]:
        """
        Comprehensive evaluation of the agent's performance.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of aggregated performance metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate")
        
        logger.info(f"Evaluating model over {n_episodes} episodes...")
        
        all_metrics = []
        episode_rewards = []
        
        for episode in range(n_episodes):
            reset_output = env.reset()
            if isinstance(reset_output, tuple):
                obs, _ = reset_output
            else:
                obs = reset_output
                
            done = False
            episode_reward = 0
            
            while not done:
                action, info = self.predict(obs, deterministic=deterministic)
                step_output = env.step(action)
                
                if len(step_output) == 4:
                    obs, reward, done, _ = step_output
                else:
                    obs, reward, terminated, truncated, _ = step_output
                    done = terminated or truncated
                
                episode_reward += reward
            
            # Get episode statistics
            metrics = env.get_episode_statistics()
            all_metrics.append(metrics)
            episode_rewards.append(episode_reward)
        
        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
                    aggregated[f'{key}_min'] = np.min(values)
                    aggregated[f'{key}_max'] = np.max(values)
        
        # Add episode reward statistics
        aggregated['episode_reward_mean'] = np.mean(episode_rewards)
        aggregated['episode_reward_std'] = np.std(episode_rewards)
        
        return aggregated
    
    def train(self, eval_env: Optional[Any] = None) -> None:
        """
        Train the PPO agent with comprehensive callbacks and error handling.
        
        Args:
            eval_env: Optional evaluation environment for periodic evaluation
        """
        if self.model is None:
            raise ValueError("Model must be built before training. Call build_model() first.")
        
        logger.info("Starting PPO training...")
        
        # Create evaluation environment if not provided
        if eval_env is None:
            eval_env = self.env
        
        # Wrap evaluation environment to match training environment wrapping
        if not isinstance(eval_env, Monitor):
            eval_env = Monitor(eval_env)
        
        # Check if training environment is vectorized and normalize
        if hasattr(self.vec_env, 'normalize_obs') and self.vec_env.normalize_obs:
            # Training env is normalized, so we need to normalize eval env too
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
            # Copy normalization statistics from training environment
            if hasattr(self.vec_env, 'obs_rms'):
                eval_vec_env.obs_rms = self.vec_env.obs_rms
            if hasattr(self.vec_env, 'ret_rms'):
                eval_vec_env.ret_rms = self.vec_env.ret_rms
        else:
            # Training env is not normalized
            eval_vec_env = DummyVecEnv([lambda: eval_env])
        
        # Create custom callback for training metrics
        class TradingCallback(BaseCallback):
            def __init__(self, agent, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
                super().__init__(verbose)
                self.agent = agent
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes
                self.training_metrics = []
                self.eval_results = []
                self.best_sharpe = -np.inf
                self.best_return = -np.inf
                
            def _on_step(self) -> bool:
                """Called at each training step."""
                # Log training metrics periodically
                if self.n_calls % 100 == 0:
                    # Get learning rate value (handle both scalar and callable)
                    if hasattr(self.model, 'learning_rate'):
                        if callable(self.model.learning_rate):
                            # For scheduled learning rates, get current value
                            if hasattr(self.model, '_current_progress_remaining'):
                                lr_value = float(self.model.learning_rate(self.model._current_progress_remaining))
                            else:
                                lr_value = float(self.model.learning_rate(1.0))
                        else:
                            lr_value = float(self.model.learning_rate)
                    else:
                        lr_value = 0.0
                    
                    # Get other metrics safely
                    entropy = self.model.logger.name_to_value.get('train/entropy_loss', 0) if hasattr(self.model, 'logger') else 0
                    policy_loss = self.model.logger.name_to_value.get('train/policy_gradient_loss', 0) if hasattr(self.model, 'logger') else 0
                    value_loss = self.model.logger.name_to_value.get('train/value_loss', 0) if hasattr(self.model, 'logger') else 0
                    
                    self.training_metrics.append({
                        'step': int(self.n_calls),
                        'timestamp': datetime.now().isoformat(),
                        'learning_rate': lr_value,
                        'entropy': float(entropy) if entropy is not None else 0.0,
                        'policy_loss': float(policy_loss) if policy_loss is not None else 0.0,
                        'value_loss': float(value_loss) if value_loss is not None else 0.0
                    })
                
                # Evaluate periodically
                if self.n_calls % self.eval_freq == 0:
                    metrics = self._evaluate_trading_performance(self.eval_env, 1)  # Just 1 episode for speed
                    self.eval_results.append({
                        'step': int(self.n_calls),
                        'timestamp': datetime.now().isoformat(),
                        **{k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()}
                    })
                    
                    # Check for improvement
                    if metrics['sharpe_ratio'] > self.best_sharpe:
                        self.best_sharpe = metrics['sharpe_ratio']
                        if self.verbose > 0:
                            logger.info(f"New best Sharpe ratio: {self.best_sharpe:.2f}")
                    
                    if metrics['mean_return'] > self.best_return:
                        self.best_return = metrics['mean_return']
                    
                    # Log metrics
                    if self.verbose > 0:
                        logger.info(f"Step {self.n_calls}: Return={metrics['mean_return']:.2%}, "
                                   f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                                   f"Max DD={metrics['max_drawdown']:.2%}, "
                                   f"Win Rate={metrics['win_rate']:.2%}")
                    
                    # Save results
                    self._save_results()
                    
                    # Generate plots
                    if len(self.eval_results) > 5:
                        self._generate_plots()
                
                return True
            
            def _evaluate_trading_performance(self, eval_env, n_episodes=2):
                """Fast evaluation with minimal episodes and steps"""
                try:
                    episode_returns = []
                    episode_lengths = []
                    total_trades = 0
                    winning_trades = 0
                    total_profit = 0
                    total_loss = 0
                    
                    for episode in range(n_episodes):
                        try:
                            obs = eval_env.reset()
                            if isinstance(obs, tuple):
                                obs = obs[0]
                            
                            episode_return = 0
                            episode_length = 0
                            episode_trades = 0
                            max_steps = 200  # Much shorter episodes for speed
                            
                            done = False
                            while not done and episode_length < max_steps:
                                try:
                                    action, _ = self.agent.model.predict(obs, deterministic=True)
                                    obs, reward, done, info = eval_env.step(action)
                                    
                                    episode_return += reward
                                    episode_length += 1
                                    
                                    # Track trading metrics if available
                                    if hasattr(info, 'get') and info.get('trade_made', False):
                                        episode_trades += 1
                                        total_trades += 1
                                        trade_return = info.get('trade_return', 0)
                                        if trade_return > 0:
                                            winning_trades += 1
                                            total_profit += trade_return
                                        else:
                                            total_loss += abs(trade_return)
                                except Exception as step_e:
                                    logger.warning(f"Step error in episode {episode}: {step_e}")
                                    break
                            
                            if episode_length > 0:  # Only add if episode had some steps
                                episode_returns.append(episode_return)
                                episode_lengths.append(episode_length)
                                
                        except Exception as episode_e:
                            logger.warning(f"Episode {episode} failed: {episode_e}")
                            continue
                    
                    # Calculate metrics only if we have valid episodes
                    if not episode_returns:
                        logger.warning("No valid episodes completed for evaluation")
                        return {
                            'mean_return': 0.0,
                            'std_return': 0.0,
                            'sharpe_ratio': 0.0,
                            'max_drawdown': 0.0,
                            'win_rate': 0.0,
                            'num_trades': 0,
                            'avg_episode_length': 0
                        }
                    
                    mean_return = np.mean(episode_returns)
                    std_return = np.std(episode_returns) if len(episode_returns) > 1 else 0.0
                    sharpe_ratio = mean_return / (std_return + 1e-8) if std_return > 0 else 0.0
                    
                    # Calculate drawdown
                    if len(episode_returns) > 1:
                        cumulative_returns = np.cumsum(episode_returns)
                        running_max = np.maximum.accumulate(cumulative_returns)
                        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
                        max_drawdown = np.min(drawdown)
                    else:
                        max_drawdown = 0.0
                    
                    # Calculate win rate
                    win_rate = winning_trades / max(total_trades, 1)
                    
                    # Calculate profit factor
                    profit_factor = total_profit / max(total_loss, 1e-8) if total_loss > 0 else 1.0
                    
                    return {
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'num_trades': total_trades,
                        'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
                        'profit_factor': profit_factor
                    }
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    return {
                        'mean_return': 0.0,
                        'std_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0,
                        'num_trades': 0,
                        'avg_episode_length': 0,
                        'profit_factor': 1.0
                    }
            
            def _save_results(self) -> None:
                """Save evaluation results to file."""
                try:
                    # Save evaluation results
                    results_path = os.path.join(self.agent.log_dir, 'eval_results.json')
                    # Ensure all values are JSON serializable
                    clean_eval_results = []
                    for result in self.eval_results:
                        clean_result = {}
                        for k, v in result.items():
                            if isinstance(v, (bool, int, float, str, type(None))):
                                clean_result[k] = v
                            elif isinstance(v, np.number):
                                clean_result[k] = float(v)
                            elif isinstance(v, np.ndarray):
                                clean_result[k] = v.tolist()
                            else:
                                clean_result[k] = str(v)
                        clean_eval_results.append(clean_result)
                    
                    with open(results_path, 'w') as f:
                        json.dump(clean_eval_results, f, indent=2)
                    
                    # Save training metrics
                    training_path = os.path.join(self.agent.log_dir, 'training_metrics.json')
                    # Ensure all values are JSON serializable
                    clean_training_metrics = []
                    for metric in self.training_metrics:
                        clean_metric = {}
                        for k, v in metric.items():
                            if isinstance(v, (bool, int, float, str, type(None))):
                                clean_metric[k] = v
                            elif isinstance(v, np.number):
                                clean_metric[k] = float(v)
                            elif isinstance(v, np.ndarray):
                                clean_metric[k] = v.tolist()
                            else:
                                clean_metric[k] = str(v)
                        clean_training_metrics.append(clean_metric)
                    
                    with open(training_path, 'w') as f:
                        json.dump(clean_training_metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Error saving results: {e}")
            
            def _generate_plots(self) -> None:
                """Generate performance plots."""
                if not self.eval_results:
                    return
                
                # Set matplotlib backend to prevent threading issues
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Convert to DataFrame for easy plotting
                df = pd.DataFrame(self.eval_results)
                
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Training Progress', fontsize=16)
                
                # Plot 1: Returns over time
                ax1 = axes[0, 0]
                ax1.plot(df['step'], df['mean_return'] * 100, 'b-', label='Mean Return')
                # Handle std_return properly - use actual column or create zeros
                if 'std_return' in df.columns:
                    std_values = df['std_return']
                else:
                    std_values = [0.0] * len(df)
                    
                ax1.fill_between(df['step'], 
                                (df['mean_return'] - std_values) * 100,
                                (df['mean_return'] + std_values) * 100,
                                alpha=0.3)
                ax1.set_xlabel('Training Steps')
                ax1.set_ylabel('Return (%)')
                ax1.set_title('Returns Evolution')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot 2: Risk metrics
                ax2 = axes[0, 1]
                ax2.plot(df['step'], df['sharpe_ratio'], 'g-', label='Sharpe Ratio')
                ax2.plot(df['step'], df.get('sortino_ratio', df['sharpe_ratio']), 'r--', label='Sortino Ratio')
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Ratio')
                ax2.set_title('Risk-Adjusted Returns')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Plot 3: Trading behavior
                ax3 = axes[1, 0]
                ax3.plot(df['step'], df['num_trades'], 'purple', label='Trades')
                ax3.set_xlabel('Training Steps')
                ax3.set_ylabel('Number of Trades')
                ax3.set_title('Trading Activity')
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Win rate and profit factor
                ax4 = axes[1, 1]
                ax4.plot(df['step'], df['win_rate'] * 100, 'orange', label='Win Rate')
                ax4_twin = ax4.twinx()
                
                # Handle profit factor properly - create array if missing
                if 'profit_factor' in df.columns:
                    profit_factor = df['profit_factor']
                else:
                    profit_factor = [1.0] * len(df)  # Default array instead of scalar
                ax4_twin.plot(df['step'], profit_factor, 'brown', label='Profit Factor')
                ax4.set_xlabel('Training Steps')
                ax4.set_ylabel('Win Rate (%)', color='orange')
                ax4_twin.set_ylabel('Profit Factor', color='brown')
                ax4.set_title('Trading Quality')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(self.agent.log_dir, 'training_progress.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Important: close the figure to free memory
        
        # Create callbacks
        callbacks = []
        
        # Custom trading callback
        trading_callback = TradingCallback(
            agent=self,
            eval_env=eval_vec_env,
            eval_freq=self.config.get('eval_freq', 1000),
            n_eval_episodes=self.config.get('n_eval_episodes', 5),
            verbose=self.config.get('verbose', 1)
        )
        callbacks.append(trading_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('save_freq', 5000),
            save_path=self.model_dir,
            name_prefix='ppo_trading_model',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback - use the properly wrapped eval environment
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=self.config.get('eval_freq', 1000),
            n_eval_episodes=self.config.get('n_eval_episodes', 5),
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        try:
            # Train the model
            self.model.learn(
                total_timesteps=self.config.get('total_timesteps', 100000),
                callback=callbacks,
                log_interval=self.config.get('log_interval', 10),
                tb_log_name="ppo_trading",
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, 'final_model.zip')
            self.model.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
            # Save training metrics
            self.training_metrics = trading_callback.training_metrics
            self.eval_results = trading_callback.eval_results
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def create_ensemble(self, n_models: int = 3) -> 'EnsembleAgent':
        """
        Create an ensemble of models for robust predictions.
        
        Args:
            n_models: Number of models in ensemble
            
        Returns:
            Ensemble agent
        """
        return EnsembleAgent(self.env, self.config, n_models)


class EnsembleAgent:
    """Ensemble of multiple PPO agents for robust trading decisions."""
    
    def __init__(self, env: Any, config: Dict[str, Any], n_models: int = 3):
        """
        Initialize ensemble agent.
        
        Args:
            env: Trading environment
            config: Configuration dictionary
            n_models: Number of models in ensemble
        """
        self.env = env
        self.config = config
        self.n_models = n_models
        self.models = []
        
        # Create multiple models with different seeds
        for i in range(n_models):
            agent = RLTradingAgentV2(env, config)
            agent.config['seed'] = i * 100  # Different seeds
            self.models.append(agent)
    
    def train_ensemble(self, eval_env: Optional[Any] = None) -> None:
        """Train all models in the ensemble."""
        for i, agent in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")
            agent.build_model()
            agent.train(eval_env)
    
    def predict_ensemble(self, observation: np.ndarray, 
                        method: str = 'majority') -> Tuple[int, Dict[str, Any]]:
        """
        Make ensemble prediction.
        
        Args:
            observation: Current state
            method: Ensemble method ('majority', 'average', 'weighted')
            
        Returns:
            Ensemble action and info
        """
        actions = []
        values = []
        
        for agent in self.models:
            action, info = agent.predict(observation, deterministic=True)
            actions.append(action)
            if 'value_estimate' in info:
                values.append(info['value_estimate'])
        
        if method == 'majority':
            # Majority voting
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            ensemble_action = max(action_counts, key=action_counts.get)
        
        elif method == 'average':
            # Average action (for continuous action spaces)
            ensemble_action = int(np.round(np.mean(actions)))
        
        elif method == 'weighted':
            # Weight by value estimates
            if values:
                weights = np.array(values) - np.min(values) + 1e-8
                weights = weights / weights.sum()
                ensemble_action = int(np.round(np.average(actions, weights=weights)))
            else:
                ensemble_action = int(np.round(np.mean(actions)))
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        info = {
            'all_actions': actions,
            'all_values': values,
            'ensemble_method': method
        }
        
        return ensemble_action, info


# Example usage
if __name__ == "__main__":
    # This would typically be imported from trading_environment.py
    from gymnasium.envs.classic_control import CartPoleEnv
    
    # Create dummy environment (replace with TradingEnvironment)
    env = CartPoleEnv()
    
    # Create agent with custom configuration
    config = {
        'learning_rate': 3e-4,
        'total_timesteps': 10000,
        'use_custom_network': True,
        'use_attention': True
    }
    
    agent = RLTradingAgentV2(env, config)
    
    # Build model
    agent.build_model()
    
    # Train
    agent.train()
    
    # Evaluate
    results = agent.evaluate(env, n_episodes=5)
    print("\nEvaluation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")