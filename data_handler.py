"""
Data Handler Module for RL Trading System
========================================
This module handles all data collection, preprocessing, and feature engineering
for the reinforcement learning trading system.

Author: Senior Quantitative Developer
Date: 2024
Version: 2.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles data collection, preprocessing, and feature engineering for RL trading.
    
    This class provides methods to:
    - Load and validate historical price data
    - Calculate technical indicators
    - Normalize features for neural network input
    - Split data into train/validation/test sets
    """
    
    def __init__(self, symbol: str = 'AAPL', 
                 start_date: str = '2010-01-01',
                 end_date: str = '2020-12-31',
                 csv_path: Optional[str] = None):
        """
        Initialize DataHandler with configuration parameters.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            csv_path: Optional path to CSV file with historical data
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical price data from CSV or yfinance.
        
        Returns:
            DataFrame with OHLCV data indexed by date
            
        Raises:
            ValueError: If data cannot be loaded or validated
        """
        try:
            if self.csv_path:
                logger.info(f"Loading data from CSV: {self.csv_path}")
                self.raw_data = self._load_csv_data()
            else:
                logger.info(f"Downloading data from yfinance for {self.symbol}")
                self.raw_data = self._download_yfinance_data()
            
            # Validate data
            self._validate_data()
            logger.info(f"Successfully loaded {len(self.raw_data)} rows of data")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load and parse CSV data with proper formatting."""
        df = pd.read_csv(self.csv_path)
        
        # Clean column names (remove leading spaces)
        df.columns = df.columns.str.strip()
        
        # Parse date - handle both datetime and date formats
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("No 'Date' column found in CSV")
        
        # Handle price columns - check if they're already numeric or need cleaning
        price_columns = ['Close/Last', 'Open', 'High', 'Low']
        
        for col in price_columns:
            if col in df.columns:
                # Check if column contains string values with $ signs
                if df[col].dtype == 'object':
                    # Clean string columns
                    df[col] = df[col].str.replace('$', '').str.strip().astype(float)
                else:
                    # Already numeric, just ensure it's float
                    df[col] = df[col].astype(float)
        
        # Rename columns to standard format
        df = df.rename(columns={
            'Close/Last': 'Close',
            'Date': 'Date'
        })
        
        # Sort by date (oldest first)
        df = df.sort_values('Date')
        
        # Set date as index
        df.set_index('Date', inplace=True)
        
        # Select date range
        mask = (df.index >= self.start_date) & (df.index <= self.end_date)
        df = df.loc[mask]
        
        return df
    
    def _download_yfinance_data(self) -> pd.DataFrame:
        """Download data from yfinance."""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in yfinance data")
            
        return df
    
    def _validate_data(self) -> None:
        """Validate loaded data for completeness and quality."""
        if self.raw_data is None or len(self.raw_data) == 0:
            raise ValueError("No data loaded")
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if self.raw_data[required_cols].isnull().any().any():
            logger.warning("Found missing values, forward filling...")
            self.raw_data.fillna(method='ffill', inplace=True)
        
        # Check for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (self.raw_data[price_cols] <= 0).any().any():
            raise ValueError("Found zero or negative prices")
        
        # Check data consistency (High >= Low, etc.)
        if not (self.raw_data['High'] >= self.raw_data['Low']).all():
            raise ValueError("Data inconsistency: High < Low")
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators and features for the RL model.
        
        Returns:
            DataFrame with all features calculated
        """
        logger.info("Calculating technical features...")
        
        df = self.raw_data.copy()
        
        # Price-based features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)
        
        # Log returns for better statistical properties
        df['log_returns_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_returns_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        # Volatility measures
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()
        df['realized_vol'] = df['returns_1d'].rolling(20).std() * np.sqrt(252)
        
        # Volatility regime
        df['vol_percentile'] = df['realized_vol'].rolling(252).rank(pct=True)
        
        # Simple Moving Averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # SMA ratios
        df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
        df['sma_ratio_50_200'] = df['sma_50'] / df['sma_200']
        
        # Price relative to SMAs
        df['price_to_sma20'] = df['Close'] / df['sma_20'] - 1
        df['price_to_sma50'] = df['Close'] / df['sma_50'] - 1
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['Close'], period=14)
        df['rsi_7'] = self._calculate_rsi(df['Close'], period=7)
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(20).mean() / df['Volume'].rolling(60).mean()
        
        # Price-Volume correlation
        df['price_volume_corr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Order flow imbalance (using high-low as proxy)
        df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
        df['close_location'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        # Price momentum
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_30d'] = df['Close'] / df['Close'].shift(30) - 1
        df['momentum_60d'] = df['Close'] / df['Close'].shift(60) - 1
        
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(df, period=14)
        
        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, period=14)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek / 4.0 - 1.0  # Normalize to [-1, 1]
        df['month_of_year'] = (df.index.month - 6.5) / 5.5  # Normalize to ~[-1, 1]
        df['quarter'] = (df.index.quarter - 2.5) / 1.5  # Normalize to ~[-1, 1]
        
        # For 5-minute data, add intraday features
        if hasattr(df.index, 'hour'):
            df['hour_of_day'] = (df.index.hour - 12) / 12  # Normalize to ~[-1, 1]
            df['minute_of_hour'] = (df.index.minute - 30) / 30  # Normalize to ~[-1, 1]
            
            # Market session indicators
            df['pre_market'] = ((df.index.hour >= 4) & (df.index.hour < 9.5)).astype(float)
            df['regular_market'] = ((df.index.hour >= 9.5) & (df.index.hour < 16)).astype(float)
            df['after_market'] = ((df.index.hour >= 16) & (df.index.hour < 20)).astype(float)
        
        # Market microstructure
        df['daily_range'] = (df['High'] - df['Low']) / df['Open']
        df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Define feature columns for the model
        self.feature_columns = [
            # Returns
            'returns_1d', 'returns_5d', 'returns_20d',
            'log_returns_1d', 'log_returns_5d',
            
            # Volatility
            'volatility_20d', 'volatility_60d', 'vol_percentile',
            
            # Moving averages
            'sma_ratio_20_50', 'sma_ratio_50_200',
            'price_to_sma20', 'price_to_sma50',
            
            # Technical indicators
            'rsi_14', 'rsi_7',
            'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width',
            'adx', 'atr_ratio',
            'stoch_k', 'stoch_d',
            
            # Volume
            'volume_ratio', 'volume_trend', 'price_volume_corr',
            
            # Market microstructure
            'high_low_spread', 'close_location',
            'daily_range', 'overnight_gap',
            
            # Momentum
            'momentum_10d', 'momentum_30d', 'momentum_60d',
            
            # Time features
            'day_of_week', 'month_of_year', 'quarter'
        ]
        
        # Add intraday features if available
        if 'hour_of_day' in df.columns:
            self.feature_columns.extend([
                'hour_of_day', 'minute_of_hour',
                'pre_market', 'regular_market', 'after_market'
            ])
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        self.processed_data = df
        logger.info(f"Features calculated. Dataset size: {len(df)}")
        logger.info(f"Feature columns set to: {self.feature_columns}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast_period: int = 12, 
                       slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, 
                                  std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate directional movements
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        # Calculate true range
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        
        # Calculate ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, 
                            period: int = 14, 
                            smooth_k: int = 3,
                            smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return k_percent, d_percent
    
    def normalize_features(self, method: str = 'robust') -> pd.DataFrame:
        """
        Normalize features for neural network input.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run calculate_features() first.")
        
        df = self.processed_data.copy()
        
        # Store original feature columns before normalization
        original_feature_columns = self.feature_columns.copy()
        
        logger.info(f"Normalizing features using {method} method: {original_feature_columns}")
        
        if method == 'standard':
            # Z-score normalization
            for col in original_feature_columns:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-8)
                else:
                    logger.warning(f"Column {col} not found in DataFrame")
        
        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            for col in original_feature_columns:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f'{col}_norm'] = 2 * (df[col] - min_val) / (max_val - min_val + 1e-8) - 1
                else:
                    logger.warning(f"Column {col} not found in DataFrame")
        
        elif method == 'robust':
            # Robust scaling using median and IQR (better for outliers)
            for col in original_feature_columns:
                if col in df.columns:
                    median = df[col].median()
                    q75 = df[col].quantile(0.75)
                    q25 = df[col].quantile(0.25)
                    iqr = q75 - q25
                    df[f'{col}_norm'] = (df[col] - median) / (iqr + 1e-8)
                    # Clip to reasonable range
                    df[f'{col}_norm'] = df[f'{col}_norm'].clip(-3, 3)
                else:
                    logger.warning(f"Column {col} not found in DataFrame")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Update feature columns to normalized versions
        self.feature_columns = [f'{col}_norm' for col in original_feature_columns]
        
        # Update processed_data with normalized features
        self.processed_data = df
        
        logger.info(f"Created normalized features: {self.feature_columns}")
        
        return df
    
    def split_data(self, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with temporal ordering.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        n = len(self.processed_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': self.processed_data.iloc[:train_end].copy(),
            'val': self.processed_data.iloc[train_end:val_end].copy(),
            'test': self.processed_data.iloc[val_end:].copy()
        }
        
        # Log split information
        for split_name, df in splits.items():
            logger.info(f"{split_name.capitalize()} set: {len(df)} samples "
                       f"({df.index[0].strftime('%Y-%m-%d %H:%M:%S')} to "
                       f"{df.index[-1].strftime('%Y-%m-%d %H:%M:%S')})")
        
        return splits
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for all features (useful for monitoring).
        
        Returns:
            DataFrame with feature statistics
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        stats = []
        for col in self.feature_columns:
            if col in self.processed_data.columns:
                series = self.processed_data[col]
                stats.append({
                    'feature': col,
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'skew': series.skew(),
                    'kurtosis': series.kurtosis()
                })
        
        return pd.DataFrame(stats)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate basic feature importance using correlation with returns.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        importance_scores = []
        
        # Calculate correlation with forward returns
        forward_returns = self.processed_data['returns_1d'].shift(-1)
        
        for col in self.feature_columns:
            if col in self.processed_data.columns:
                # Pearson correlation
                corr = self.processed_data[col].corr(forward_returns)
                
                # Information coefficient (IC)
                ic = self.processed_data[col].rolling(20).corr(forward_returns).mean()
                
                # Mutual information (would require sklearn in practice)
                # For now, use absolute correlation as proxy
                mi_proxy = abs(corr)
                
                importance_scores.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'information_coefficient': ic,
                    'mutual_info_proxy': mi_proxy
                })
        
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        return importance_df


# Example usage and testing
if __name__ == "__main__":
    # Initialize data handler for 5-minute data
    handler = DataHandler(
        symbol='AAPL',
        start_date='2022-01-01',  # Adjust based on your data
        end_date='2024-12-31',
        csv_path='AAPL_HistoricalQuotes.csv'  # Your 5-minute data file
    )
    
    # Load and process data
    handler.load_data()
    handler.calculate_features()
    handler.normalize_features(method='robust')  # Using robust scaling
    
    # Split data
    splits = handler.split_data(train_ratio=0.7, val_ratio=0.15)
    
    # Display feature statistics
    print("\nFeature Statistics:")
    print(handler.get_feature_stats().head())
    
    # Display feature importance
    print("\nFeature Importance:")
    print(handler.get_feature_importance().head(10))
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(handler.processed_data[handler.feature_columns].head())