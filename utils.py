"""
Utilities Module for RL Trading System
=====================================
This module provides helper functions, constants, and utilities
for the reinforcement learning trading system.

Author: Senior Quantitative Developer
Date: 2024
Version: 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import json
import yaml
import os
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import pickle
import hashlib
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import joblib
from collections import defaultdict
import time
import psutil
import GPUtil

warnings.filterwarnings('ignore')

# Configure logging
def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> None:
    """
    Configure logging for the entire application with enhanced formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Optional custom log format
    """
    if log_format is None:
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    
    # Create formatter with colors for console output
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels."""
        
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
    
    # Configure handlers
    handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format))
    handlers.append(console_handler)
    
    # File handler without colors
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    # Set levels for specific loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# Enhanced Constants
TRADING_DAYS_PER_YEAR = 252
SECONDS_PER_DAY = 86400
DEFAULT_RISK_FREE_RATE = 0.02

# Market hours (Eastern Time)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
PRE_MARKET_OPEN = "04:00"
AFTER_MARKET_CLOSE = "20:00"

# Technical indicator periods
DEFAULT_PERIODS = {
    'SMA_SHORT': 20,
    'SMA_LONG': 50,
    'SMA_ULTRA_LONG': 200,
    'RSI': 14,
    'RSI_FAST': 7,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_STD': 2,
    'ATR': 14,
    'ADX': 14,
    'STOCHASTIC': 14,
    'VOLUME_MA': 20,
    'MOMENTUM': 10
}

# Model hyperparameter ranges for optimization
HYPERPARAMETER_RANGES = {
    'learning_rate': (1e-5, 1e-3),
    'batch_size': [32, 64, 128, 256],
    'gamma': (0.9, 0.999),
    'gae_lambda': (0.9, 0.99),
    'clip_range': (0.1, 0.3),
    'clip_range_vf': (None, 0.1, 0.2),
    'ent_coef': (0.0, 0.1),
    'vf_coef': (0.25, 0.75),
    'n_steps': [512, 1024, 2048, 4096],
    'n_epochs': [5, 10, 20],
    'target_kl': (0.005, 0.05)
}

# Market regimes
MARKET_REGIMES = {
    'BULL': {'volatility': 'low', 'trend': 'up'},
    'BEAR': {'volatility': 'high', 'trend': 'down'},
    'VOLATILE': {'volatility': 'high', 'trend': 'sideways'},
    'QUIET': {'volatility': 'low', 'trend': 'sideways'}
}


class PerformanceTracker:
    """Enhanced performance tracking with real-time monitoring."""
    
    def __init__(self, metrics_file: str = 'performance_metrics.json'):
        """
        Initialize performance tracker.
        
        Args:
            metrics_file: Path to store performance metrics
        """
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
        self.real_time_metrics = defaultdict(list)
        self.start_time = time.time()
    
    def _load_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load existing metrics from file."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def record_training_metrics(self, 
                              model_id: str,
                              epoch: int,
                              metrics: Dict[str, float],
                              system_metrics: bool = True) -> None:
        """
        Record training metrics with system resource usage.
        
        Args:
            model_id: Unique identifier for the model
            epoch: Training epoch
            metrics: Dictionary of metrics to record
            system_metrics: Whether to record system metrics
        """
        if model_id not in self.metrics:
            self.metrics[model_id] = []
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        # Add system metrics
        if system_metrics:
            record.update(self._get_system_metrics())
        
        self.metrics[model_id].append(record)
        self._save_metrics()
        
        # Update real-time metrics
        for key, value in metrics.items():
            self.real_time_metrics[key].append(value)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        # Add GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                metrics.update({
                    'gpu_percent': gpu.load * 100,
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_temperature': gpu.temperature
                })
        except:
            pass
        
        return metrics
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_best_model(self, 
                      metric: str = 'sharpe_ratio',
                      minimize: bool = False) -> Tuple[str, float, Dict[str, Any]]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            minimize: Whether to minimize the metric (e.g., for loss)
            
        Returns:
            Tuple of (model_id, best_metric_value, full_record)
        """
        best_model = None
        best_value = float('inf') if minimize else -float('inf')
        best_record = None
        
        for model_id, records in self.metrics.items():
            if records:
                # Get best record for this model
                if minimize:
                    model_best = min(records, key=lambda x: x.get(metric, float('inf')))
                else:
                    model_best = max(records, key=lambda x: x.get(metric, -float('inf')))
                
                if metric in model_best:
                    if (minimize and model_best[metric] < best_value) or \
                       (not minimize and model_best[metric] > best_value):
                        best_value = model_best[metric]
                        best_model = model_id
                        best_record = model_best
        
        return best_model, best_value, best_record
    
    def plot_training_history(self, 
                            model_id: str,
                            metrics: List[str] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot enhanced training history with multiple metrics.
        
        Args:
            model_id: Model identifier
            metrics: List of metrics to plot (None for default)
            save_path: Optional path to save plot
        """
        if model_id not in self.metrics:
            raise ValueError(f"No metrics found for model {model_id}")
        
        records = self.metrics[model_id]
        df = pd.DataFrame(records)
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            # Plot metric
            ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=6)
            
            # Add rolling average
            if len(df) > 10:
                rolling_avg = df[metric].rolling(window=5, center=True).mean()
                ax.plot(df['epoch'], rolling_avg, '--', alpha=0.7, label='5-epoch avg')
            
            # Highlight best value
            if metric in ['loss', 'max_drawdown']:  # Minimize these
                best_idx = df[metric].idxmin()
            else:  # Maximize others
                best_idx = df[metric].idxmax()
            
            ax.scatter(df.loc[best_idx, 'epoch'], df.loc[best_idx, metric], 
                      color='red', s=100, zorder=5)
            ax.annotate(f'Best: {df.loc[best_idx, metric]:.3f}',
                       xy=(df.loc[best_idx, 'epoch'], df.loc[best_idx, metric]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric} over Training')
            ax.grid(True, alpha=0.3)
            if len(df) > 10:
                ax.legend()
        
        plt.suptitle(f'Training History for {model_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary report of all models."""
        summary = []
        
        for model_id, records in self.metrics.items():
            if records:
                # Get final metrics
                final_record = records[-1]
                
                # Get best metrics
                best_sharpe = max(r.get('sharpe_ratio', -np.inf) for r in records)
                best_return = max(r.get('total_return', -np.inf) for r in records)
                
                summary.append({
                    'model_id': model_id,
                    'total_epochs': len(records),
                    'training_time_hours': final_record.get('elapsed_time', 0) / 3600,
                    'final_sharpe': final_record.get('sharpe_ratio', 0),
                    'best_sharpe': best_sharpe,
                    'final_return': final_record.get('total_return', 0),
                    'best_return': best_return,
                    'final_drawdown': final_record.get('max_drawdown', 0),
                    'convergence_epoch': self._find_convergence_epoch(records)
                })
        
        return pd.DataFrame(summary)
    
    def _find_convergence_epoch(self, records: List[Dict], 
                               metric: str = 'sharpe_ratio',
                               patience: int = 10) -> int:
        """Find epoch where model converged."""
        if len(records) < patience:
            return len(records)
        
        values = [r.get(metric, 0) for r in records]
        best_value = -np.inf
        patience_counter = 0
        
        for i, value in enumerate(values):
            if value > best_value:
                best_value = value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    return i - patience + 1
        
        return len(records)


class ConfigManager:
    """Enhanced configuration management with validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file with validation.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
        
        # Validate configuration
        ConfigManager.validate_config(config)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration for required fields and types."""
        required_fields = {
            'data': ['symbol', 'start_date', 'end_date'],
            'environment': ['initial_capital', 'transaction_cost'],
            'agent': ['learning_rate', 'total_timesteps']
        }
        
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}")
        
        # Validate types and ranges
        if config['environment']['transaction_cost'] < 0 or config['environment']['transaction_cost'] > 0.1:
            raise ValueError("Transaction cost must be between 0 and 0.1")
        
        if config['agent']['learning_rate'] <= 0 or config['agent']['learning_rate'] > 1:
            raise ValueError("Learning rate must be between 0 and 1")
    
    @staticmethod
    def save_config(config: Dict[str, Any], 
                   config_path: str,
                   backup: bool = True) -> None:
        """
        Save configuration to file with optional backup.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
            backup: Whether to create backup of existing config
        """
        # Create backup if file exists
        if backup and os.path.exists(config_path):
            backup_path = f"{config_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
            os.rename(config_path, backup_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif config_path.endswith('.json'):
                json.dump(config, f, indent=2)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any],
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
            
        Returns:
            Merged configuration
        """
        import copy
        merged = copy.deepcopy(base_config)
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        return deep_update(merged, override_config)
    
    @staticmethod
    def generate_experiment_configs(base_config: Dict[str, Any],
                                  param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate multiple experiment configurations from parameter grid.
        
        Args:
            base_config: Base configuration
            param_grid: Dictionary of parameters to vary
            
        Returns:
            List of experiment configurations
        """
        from itertools import product
        
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate all combinations
        configs = []
        for values in product(*param_values):
            # Create config for this combination
            config = base_config.copy()
            
            for name, value in zip(param_names, values):
                # Handle nested parameters (e.g., 'agent.learning_rate')
                parts = name.split('.')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            
            configs.append(config)
        
        return configs


class DataValidator:
    """Enhanced data validation and cleaning utilities."""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive OHLCV data validation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check price consistency
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # High >= Low
            invalid_hl = df['High'] < df['Low']
            if invalid_hl.any():
                issues.append(f"High < Low in {invalid_hl.sum()} rows")
            
            # High >= Open, Close
            invalid_high = (df['High'] < df['Open']) | (df['High'] < df['Close'])
            if invalid_high.any():
                issues.append(f"High < Open/Close in {invalid_high.sum()} rows")
            
            # Low <= Open, Close
            invalid_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            if invalid_low.any():
                issues.append(f"Low > Open/Close in {invalid_low.sum()} rows")
        
        # Check for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = df[col] <= 0
                if invalid_prices.any():
                    issues.append(f"Invalid prices in {col}: {invalid_prices.sum()} rows")
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            issues.append(f"Duplicate dates found: {df.index.duplicated().sum()}")
        
        # Check date continuity
        if len(df) > 1:
            date_diffs = pd.Series(df.index).diff()
            
            # Check for backwards dates
            if (date_diffs < pd.Timedelta(0)).any():
                issues.append("Dates not in ascending order")
            
            # Check for large gaps (more than 10 days)
            large_gaps = date_diffs[date_diffs > pd.Timedelta(days=10)]
            if len(large_gaps) > 0:
                issues.append(f"Large date gaps found: {len(large_gaps)} gaps > 10 days")
        
        # Check for suspicious volume patterns
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            if zero_volume > len(df) * 0.1:  # More than 10% zero volume
                issues.append(f"Excessive zero volume days: {zero_volume} ({zero_volume/len(df)*100:.1f}%)")
        
        # Check for data anomalies
        if len(df) > 20:
            for col in price_cols:
                if col in df.columns:
                    returns = df[col].pct_change().dropna()
                    extreme_moves = (returns.abs() > 0.2).sum()  # 20% moves
                    if extreme_moves > 0:
                        issues.append(f"Extreme price moves in {col}: {extreme_moves} days with >20% change")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                  method: str = 'forward_fill',
                  remove_outliers: bool = True,
                  outlier_threshold: float = 10.0) -> pd.DataFrame:
        """
        Enhanced data cleaning with outlier detection.
        
        Args:
            df: DataFrame to clean
            method: Cleaning method ('forward_fill', 'interpolate', 'drop')
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        if method == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        elif method == 'drop':
            df_clean = df_clean.dropna()
        
        # Ensure price consistency
        if all(col in df_clean.columns for col in ['High', 'Low', 'Open', 'Close']):
            # Ensure High is highest
            df_clean['High'] = df_clean[['High', 'Open', 'Close']].max(axis=1)
            # Ensure Low is lowest
            df_clean['Low'] = df_clean[['Low', 'Open', 'Close']].min(axis=1)
        
        # Remove outliers
        if remove_outliers and len(df_clean) > 100:
            price_cols = ['Open', 'High', 'Low', 'Close']
            
            for col in price_cols:
                if col in df_clean.columns:
                    # Calculate returns
                    returns = df_clean[col].pct_change()
                    
                    # Calculate z-scores
                    z_scores = np.abs(stats.zscore(returns.dropna()))
                    
                    # Identify outliers
                    outlier_mask = z_scores > outlier_threshold
                    
                    if outlier_mask.any():
                        # Replace outliers with interpolated values
                        df_clean.loc[returns.index[1:][outlier_mask], col] = np.nan
                        df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Ensure no negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=0.01)
        
        return df_clean
    
    @staticmethod
    def validate_features(df: pd.DataFrame, 
                         feature_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate feature data quality."""
        issues = []
        
        for col in feature_columns:
            if col not in df.columns:
                issues.append(f"Missing feature column: {col}")
                continue
            
            # Check for NaN values
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"NaN values in {col}: {nan_count}")
            
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Infinite values in {col}: {inf_count}")
            
            # Check for constant features
            if df[col].std() == 0:
                issues.append(f"Constant feature: {col}")
        
        return len(issues) == 0, issues


class ModelCheckpointer:
    """Enhanced model checkpointing with versioning and metadata."""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        """
        Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.checkpoint_dir / 'models').mkdir(exist_ok=True)
        (self.checkpoint_dir / 'metadata').mkdir(exist_ok=True)
        (self.checkpoint_dir / 'configs').mkdir(exist_ok=True)
    
    def save_checkpoint(self,
                       model: Any,
                       metadata: Dict[str, Any],
                       checkpoint_name: Optional[str] = None,
                       save_optimizer: bool = True) -> str:
        """
        Save enhanced model checkpoint with comprehensive metadata.
        
        Args:
            model: Model to save
            metadata: Metadata about the model
            checkpoint_name: Optional checkpoint name
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            # Generate unique checkpoint name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:8]
            checkpoint_name = f"checkpoint_{timestamp}_{model_hash}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / 'model.pkl'
        if hasattr(model, 'save'):
            model.save(str(model_path))
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save optimizer state if applicable
        if save_optimizer and hasattr(model, 'optimizer'):
            optimizer_path = checkpoint_path / 'optimizer.pkl'
            torch.save(model.optimizer.state_dict(), optimizer_path)
        
        # Enhanced metadata
        metadata['checkpoint_info'] = {
            'checkpoint_time': datetime.now().isoformat(),
            'checkpoint_name': checkpoint_name,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'python_version': sys.version,
            'pytorch_version': torch.__version__ if 'torch' in sys.modules else None,
            'system_info': {
                'platform': os.name,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        # Save metadata
        metadata_path = checkpoint_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save config if provided
        if 'config' in metadata:
            config_path = checkpoint_path / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(metadata['config'], f, default_flow_style=False)
        
        # Create summary file
        self._create_checkpoint_summary(checkpoint_path, metadata)
        
        return str(checkpoint_path)
    
    def _create_checkpoint_summary(self, checkpoint_path: Path, metadata: Dict[str, Any]) -> None:
        """Create human-readable summary of checkpoint."""
        summary_path = checkpoint_path / 'README.md'
        
        summary = f"""# Checkpoint Summary

## Basic Information
- **Created**: {metadata['checkpoint_info']['checkpoint_time']}
- **Name**: {metadata['checkpoint_info']['checkpoint_name']}
- **Model Size**: {metadata['checkpoint_info']['model_size_mb']:.2f} MB

## Performance Metrics
"""
        
        # Add key metrics if available
        metrics_to_show = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        for metric in metrics_to_show:
            if metric in metadata.get('metrics', {}):
                summary += f"- **{metric.replace('_', ' ').title()}**: {metadata['metrics'][metric]:.4f}\n"
        
        summary += f"""
## Training Information
- **Total Epochs**: {metadata.get('epoch', 'N/A')}
- **Training Time**: {metadata.get('training_time_hours', 'N/A'):.2f} hours

## Configuration
See `config.yaml` for full configuration details.

## Usage
```python
from utils import ModelCheckpointer

checkpointer = ModelCheckpointer()
model, metadata = checkpointer.load_checkpoint('{checkpoint_path}')
```
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model checkpoint with validation.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Tuple of (model, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load metadata
        metadata_path = checkpoint_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = checkpoint_path / 'model.pkl'
        try:
            # Try loading with pickle first
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            # Try other loading methods based on metadata
            if 'model_type' in metadata:
                if metadata['model_type'] == 'stable_baselines3':
                    from stable_baselines3 import PPO
                    model = PPO.load(str(model_path))
                else:
                    raise ValueError(f"Unknown model type: {metadata['model_type']}")
            else:
                raise
        
        return model, metadata
    
    def list_checkpoints(self, 
                        sort_by: str = 'created',
                        filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with filtering.
        
        Args:
            sort_by: Field to sort by ('created', 'sharpe_ratio', 'return')
            filter_func: Optional function to filter checkpoints
            
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint_'):
                metadata_path = checkpoint_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoint_info = {
                        'name': checkpoint_dir.name,
                        'path': str(checkpoint_dir),
                        'created': metadata.get('checkpoint_info', {}).get('checkpoint_time', 'Unknown'),
                        'metrics': metadata.get('metrics', {}),
                        'size_mb': metadata.get('checkpoint_info', {}).get('model_size_mb', 0)
                    }
                    
                    # Apply filter if provided
                    if filter_func is None or filter_func(checkpoint_info):
                        checkpoints.append(checkpoint_info)
        
        # Sort checkpoints
        if sort_by == 'created':
            checkpoints.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by in ['sharpe_ratio', 'total_return']:
            checkpoints.sort(key=lambda x: x['metrics'].get(sort_by, -np.inf), reverse=True)
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_n: int = 10, keep_best: int = 3) -> None:
        """
        Clean up old checkpoints, keeping the most recent and best performing.
        
        Args:
            keep_n: Number of most recent checkpoints to keep
            keep_best: Number of best performing checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_n + keep_best:
            return  # Nothing to clean up
        
        # Get checkpoints to keep
        recent_checkpoints = checkpoints[:keep_n]
        best_checkpoints = sorted(
            checkpoints,
            key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf),
            reverse=True
        )[:keep_best]
        
        # Combine and get unique paths
        keep_paths = set()
        for ckpt in recent_checkpoints + best_checkpoints:
            keep_paths.add(ckpt['path'])
        
        # Remove others
        for ckpt in checkpoints:
            if ckpt['path'] not in keep_paths:
                import shutil
                shutil.rmtree(ckpt['path'])
                logger.info(f"Removed old checkpoint: {ckpt['name']}")


# Market Analysis Functions
def calculate_market_regime(prices: pd.Series, 
                          volatility_window: int = 20,
                          trend_window: int = 50) -> pd.Series:
    """
    Calculate market regime based on volatility and trend.
    
    Args:
        prices: Series of prices
        volatility_window: Window for volatility calculation
        trend_window: Window for trend calculation
        
    Returns:
        Series with market regime labels
    """
    # Calculate returns and volatility
    returns = prices.pct_change()
    volatility = returns.rolling(volatility_window).std() * np.sqrt(252)
    
    # Calculate trend using linear regression slope
    def calculate_trend(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window.values, 1)
        return slope / window.mean()  # Normalize by mean
    
    trend = prices.rolling(trend_window).apply(calculate_trend)
    
    # Define regime thresholds
    vol_median = volatility.median()
    
    # Classify regimes
    regimes = pd.Series(index=prices.index, dtype='object')
    
    # Bull: Low volatility, positive trend
    bull_mask = (volatility < vol_median) & (trend > 0.001)
    regimes[bull_mask] = 'BULL'
    
    # Bear: High volatility, negative trend
    bear_mask = (volatility > vol_median) & (trend < -0.001)
    regimes[bear_mask] = 'BEAR'
    
    # Volatile: High volatility, sideways trend
    volatile_mask = (volatility > vol_median) & (trend.abs() <= 0.001)
    regimes[volatile_mask] = 'VOLATILE'
    
    # Quiet: Low volatility, sideways trend
    quiet_mask = (volatility < vol_median) & (trend.abs() <= 0.001)
    regimes[quiet_mask] = 'QUIET'
    
    # Fill NaN values
    regimes = regimes.fillna('UNKNOWN')
    
    return regimes


def calculate_position_size(capital: float,
                          risk_per_trade: float,
                          stop_loss_pct: float,
                          price: float,
                          volatility: float = None,
                          method: str = 'fixed') -> int:
    """
    Calculate optimal position size using various methods.
    
    Args:
        capital: Available capital
        risk_per_trade: Maximum risk per trade (as fraction)
        stop_loss_pct: Stop loss percentage
        price: Current asset price
        volatility: Current volatility (for volatility-based sizing)
        method: Sizing method ('fixed', 'volatility', 'kelly')
        
    Returns:
        Number of shares to trade
    """
    if method == 'fixed':
        # Fixed fractional position sizing
        risk_amount = capital * risk_per_trade
        shares = risk_amount / (price * stop_loss_pct)
    
    elif method == 'volatility' and volatility is not None:
        # Volatility-based position sizing
        target_volatility = 0.15  # 15% target
        position_fraction = min(target_volatility / volatility, 1.0)
        shares = (capital * position_fraction * risk_per_trade) / price
    
    elif method == 'kelly':
        # Simplified Kelly criterion
        # Requires win rate and payoff ratio (simplified here)
        win_rate = 0.55  # Assumed
        payoff_ratio = 1.5  # Assumed
        
        kelly_fraction = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        kelly_fraction = min(kelly_fraction * 0.25, risk_per_trade)  # Conservative Kelly
        
        shares = (capital * kelly_fraction) / price
    
    else:
        # Default to fixed
        risk_amount = capital * risk_per_trade
        shares = risk_amount / (price * stop_loss_pct)
    
    return max(int(shares), 0)


def calculate_rolling_metrics(returns: pd.Series,
                            window: int = 252,
                            min_periods: int = 20) -> pd.DataFrame:
    """
    Calculate comprehensive rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        min_periods: Minimum periods required
        
    Returns:
        DataFrame with rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    metrics['rolling_return'] = returns.rolling(window, min_periods=min_periods).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Rolling volatility
    metrics['rolling_volatility'] = returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    rf_rate = DEFAULT_RISK_FREE_RATE / 252  # Daily risk-free rate
    metrics['rolling_sharpe'] = returns.rolling(window, min_periods=min_periods).apply(
        lambda x: np.sqrt(252) * (x.mean() - rf_rate) / x.std() if x.std() > 0 else 0
    )
    
    # Rolling Sortino ratio
    def calculate_sortino(returns_window):
        downside_returns = returns_window[returns_window < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                return np.sqrt(252) * (returns_window.mean() - rf_rate) / downside_std
        return 0
    
    metrics['rolling_sortino'] = returns.rolling(window, min_periods=min_periods).apply(calculate_sortino)
    
    # Rolling maximum drawdown
    def calculate_max_dd(returns_window):
        cumulative = (1 + returns_window).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    metrics['rolling_max_drawdown'] = returns.rolling(window, min_periods=min_periods).apply(calculate_max_dd)
    
    # Rolling Calmar ratio
    metrics['rolling_calmar'] = metrics['rolling_return'] / abs(metrics['rolling_max_drawdown'])
    metrics['rolling_calmar'] = metrics['rolling_calmar'].replace([np.inf, -np.inf], 0)
    
    # Rolling win rate
    metrics['rolling_win_rate'] = returns.rolling(window, min_periods=min_periods).apply(
        lambda x: (x > 0).mean()
    )
    
    # Rolling skewness and kurtosis
    metrics['rolling_skew'] = returns.rolling(window, min_periods=min_periods).skew()
    metrics['rolling_kurtosis'] = returns.rolling(window, min_periods=min_periods).kurt()
    
    return metrics


def plot_correlation_matrix(returns_dict: Dict[str, pd.Series],
                          save_path: Optional[str] = None,
                          method: str = 'pearson') -> None:
    """
    Enhanced correlation matrix plot with multiple correlation methods.
    
    Args:
        returns_dict: Dictionary of return series
        save_path: Optional path to save plot
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    # Create DataFrame from returns
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = returns_df.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = returns_df.corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = returns_df.corr(method='kendall')
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax1)
    
    ax1.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14)
    
    # Plot 2: Correlation clustering (with NaN handling)
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    try:
        # Handle NaN values in correlation matrix
        # Convert correlation to distance, handling NaN
        distance_matrix = 1 - corr_matrix.fillna(0)
        
        # Ensure all values are finite
        distance_matrix = distance_matrix.replace([np.inf, -np.inf], 1)
        
        # Calculate linkage only if we have valid values
        if not distance_matrix.isnull().all().all() and len(distance_matrix) > 1:
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Create dendrogram
            dendrogram(linkage_matrix, labels=corr_matrix.index, ax=ax2)
            ax2.set_title('Hierarchical Clustering of Strategies', fontsize=14)
            ax2.set_xlabel('Strategy')
            ax2.set_ylabel('Distance')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for clustering', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hierarchical Clustering (Not Available)', fontsize=14)
    except Exception as e:
        # If clustering fails for any reason, show a message
        ax2.text(0.5, 0.5, f'Clustering not available:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes, wrap=True)
        ax2.set_title('Hierarchical Clustering (Error)', fontsize=14)
    
    plt.suptitle('Strategy Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_report(results: Dict[str, pd.DataFrame],
                            output_path: str,
                            format: str = 'html') -> None:
    """
    Create comprehensive performance report in HTML or PDF format.
    
    Args:
        results: Dictionary of strategy results
        output_path: Path to save report
        format: Output format ('html' or 'pdf')
    """
    # This would use a template engine like Jinja2
    # For now, create a simple HTML report
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Strategy Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Trading Strategy Performance Report</h1>
        <p>Generated: {date}</p>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
            </tr>
            {performance_rows}
        </table>
        
        <h2>Risk Analysis</h2>
        {risk_analysis}
        
        <h2>Trade Statistics</h2>
        {trade_stats}
    </body>
    </html>
    """
    
    # Generate content (simplified)
    performance_rows = ""
    for strategy, data in results.items():
        performance_rows += f"""
        <tr>
            <td class="metric">{strategy}</td>
            <td>{data.get('total_return', 0):.2%}</td>
            <td>{data.get('sharpe_ratio', 0):.2f}</td>
            <td>{data.get('max_drawdown', 0):.2%}</td>
            <td>{data.get('win_rate', 0):.2%}</td>
        </tr>
        """
    
    # Fill template
    html_content = html_content.format(
        date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        performance_rows=performance_rows,
        risk_analysis="<p>Detailed risk analysis would go here...</p>",
        trade_stats="<p>Trade statistics would go here...</p>"
    )
    
    # Save report
    if format == 'html':
        with open(output_path, 'w') as f:
            f.write(html_content)
    elif format == 'pdf':
        # Would use a library like weasyprint or pdfkit
        logger.warning("PDF output not implemented, saving as HTML")
        with open(output_path.replace('.pdf', '.html'), 'w') as f:
            f.write(html_content)


# Feature Engineering Utilities
class FeatureEngineer:
    """Advanced feature engineering utilities."""
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, 
                                  feature_cols: List[str],
                                  max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between columns."""
        interactions = []
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                if len(interactions) >= max_interactions:
                    break
                
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
                interactions.append(interaction_name)
                
                # Ratio interaction (with protection against division by zero)
                if df[col2].abs().min() > 1e-8:
                    ratio_name = f"{col1}_div_{col2}"
                    df[ratio_name] = df[col1] / df[col2]
                    interactions.append(ratio_name)
        
        return df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame,
                          columns: List[str],
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features."""
        for col in columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                              columns: List[str],
                              windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create rolling statistical features."""
        for col in columns:
            for window in windows:
                # Rolling mean
                df[f"{col}_sma_{window}"] = df[col].rolling(window).mean()
                
                # Rolling std
                df[f"{col}_std_{window}"] = df[col].rolling(window).std()
                
                # Rolling min/max
                df[f"{col}_min_{window}"] = df[col].rolling(window).min()
                df[f"{col}_max_{window}"] = df[col].rolling(window).max()
                
                # Rolling quantiles
                df[f"{col}_q25_{window}"] = df[col].rolling(window).quantile(0.25)
                df[f"{col}_q75_{window}"] = df[col].rolling(window).quantile(0.75)
        
        return df
    
    @staticmethod
    def apply_pca(df: pd.DataFrame,
                 feature_cols: List[str],
                 n_components: int = None,
                 variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """Apply PCA for dimensionality reduction."""
        # Prepare data
        X = df[feature_cols].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        if n_components is None:
            # Determine n_components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        pca_cols = [f'pca_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Add to original DataFrame
        df = pd.concat([df, pca_df], axis=1)
        
        return df, pca


# Experiment Management
class ExperimentTracker:
    """Track and manage multiple experiments."""
    
    def __init__(self, experiment_dir: str = './experiments'):
        """Initialize experiment tracker."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment = None
    
    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Create new experiment."""
        # Generate experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{name}_{timestamp}"
        
        # Create experiment directory
        exp_path = self.experiment_dir / experiment_id
        exp_path.mkdir(exist_ok=True)
        
        # Save initial config
        config_path = exp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'config': config
        }
        
        metadata_path = exp_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.current_experiment = experiment_id
        
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics for current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        exp_path = self.experiment_dir / self.current_experiment
        metrics_file = exp_path / 'metrics.jsonl'
        
        # Add timestamp and step
        record = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        # Append to metrics file
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def compare_experiments(self, experiment_ids: List[str] = None) -> pd.DataFrame:
        """Compare multiple experiments."""
        if experiment_ids is None:
            # Get all experiments
            experiment_ids = [d.name for d in self.experiment_dir.iterdir() if d.is_dir()]
        
        comparison = []
        
        for exp_id in experiment_ids:
            exp_path = self.experiment_dir / exp_id
            
            # Load metadata
            metadata_path = exp_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load final metrics
                metrics_file = exp_path / 'metrics.jsonl'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            final_metrics = json.loads(lines[-1])
                        else:
                            final_metrics = {}
                else:
                    final_metrics = {}
                
                comparison.append({
                    'experiment_id': exp_id,
                    'name': metadata.get('name'),
                    'created_at': metadata.get('created_at'),
                    'status': metadata.get('status'),
                    **final_metrics
                })
        
        return pd.DataFrame(comparison)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Test performance tracker
    tracker = PerformanceTracker()
    tracker.record_training_metrics(
        'model_001',
        epoch=1,
        metrics={'sharpe_ratio': 1.5, 'total_return': 0.15}
    )
    
    # Test config manager
    config = {
        'model': {
            'learning_rate': 0.001,
            'batch_size': 64
        },
        'data': {
            'symbol': 'AAPL',
            'period': '10Y'
        }
    }
    
    ConfigManager.save_config(config, 'test_config.json')
    loaded_config = ConfigManager.load_config('test_config.json')
    print(f"Loaded config: {loaded_config}")
    
    # Test model checkpointer
    checkpointer = ModelCheckpointer()
    
    # Create dummy model
    class DummyModel:
        def __init__(self):
            self.weights = np.random.randn(100, 50)
    
    model = DummyModel()
    checkpoint_path = checkpointer.save_checkpoint(
        model,
        metadata={
            'epoch': 100,
            'metrics': {'sharpe_ratio': 1.8, 'total_return': 0.25}
        }
    )
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # List checkpoints
    checkpoints = checkpointer.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    # Clean up
    os.remove('test_config.json')