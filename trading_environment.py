"""
Trading Environment Module for RL Trading System
==============================================
This module implements a custom OpenAI Gym environment for training
reinforcement learning agents in financial trading.

Author: Senior Quantitative Developer
Date: 2024
Version: 2.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any, List
import logging
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class Actions(IntEnum):
    """Trading action enumeration for discrete action space with long/short support.
    
    Actions represent the following position targets:
    - SELL_STRONG: Short position (-0.5 to -1.0)
    - SELL_WEAK: Light short position (-0.1 to -0.5)
    - HOLD: Neutral position (0.0)
    - BUY_WEAK: Light long position (0.1 to 0.5)
    - BUY_STRONG: Strong long position (0.5 to 1.0)
    """
    SELL_STRONG = 0
    SELL_WEAK = 1
    HOLD = 2
    BUY_WEAK = 3
    BUY_STRONG = 4


@dataclass
class TradingState:
    """Represents the current state of the trading environment."""
    position: float  # Position size (0.0 to 1.0)
    entry_price: float
    current_price: float
    cash: float
    portfolio_value: float
    step_count: int
    max_drawdown: float
    cumulative_return: float
    realized_pnl: float
    unrealized_pnl: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int


class TradingEnvironment(gym.Env):
    """
    Custom OpenAI Gym environment for stock trading with RL.
    
    This environment simulates realistic trading conditions including:
    - Transaction costs and slippage
    - Position sizing constraints
    - Risk-adjusted reward calculation
    - Proper handling of market dynamics
    - Advanced position management
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 feature_columns: list,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 volatility_penalty: float = 0.005,  # REDUCED from 0.02
                 drawdown_penalty: float = 0.01,     # REDUCED from 0.05
                 max_position_size: float = 1.0,
                 lookback_window: int = 1,
                 position_sizing: str = 'fixed',
                 use_discrete_actions: bool = True,
                 min_position_change: float = 0.02):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with price data and features
            feature_columns: List of column names to use as features
            initial_capital: Starting capital in dollars
            transaction_cost: Transaction cost as fraction (0.001 = 10 bps)
            slippage: Slippage as fraction
            volatility_penalty: Penalty factor for volatility in reward (REDUCED)
            drawdown_penalty: Penalty factor for drawdown in reward (REDUCED)
            max_position_size: Maximum position size as fraction of capital
            lookback_window: Number of historical steps to include in state
            position_sizing: Position sizing method ('fixed', 'kelly', 'volatility')
            use_discrete_actions: Whether to use discrete or continuous actions
        """
        super().__init__()
        
        # Store configuration
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.volatility_penalty = volatility_penalty
        self.drawdown_penalty = drawdown_penalty
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.position_sizing = position_sizing
        self.use_discrete_actions = use_discrete_actions
        self.min_position_change = min_position_change
        
        # Validate feature columns exist in dataframe
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features in DataFrame: {missing_features}")
            # Remove missing features from feature_columns
            self.feature_columns = [col for col in feature_columns if col in df.columns]
            logger.info(f"Using {len(self.feature_columns)} available features")
        
        # Action space - updated for 5 discrete actions supporting long/short
        if use_discrete_actions:
            self.action_space = spaces.Discrete(5)  # 5 discrete actions for position sizing
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )# Continuous action: position size from -1 to 1
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        
        # State includes: features + position indicator + portfolio metrics
        n_features = len(self.feature_columns) * lookback_window
        n_portfolio_features = 12  # Fixed portfolio features
        self.n_total_features = n_features + n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_total_features,),
            dtype=np.float32
        )
        
        logger.info(f"TradingEnvironment initialized with {len(self.feature_columns)} features")
        logger.info(f"Total observation space: {self.n_total_features} dimensions")
        
        # Trading state
        self.state = None
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Trading history
        self.trades = []
        self.positions = []
        self.portfolio_values = []
        self.returns_history = []
        self.actions_history = []
        self.vol_history = []  # Track volatility history
        
        # Step tracking
        self.current_step = 0
        self.prev_position = 0.0
        
        # Calculate returns for reward computation
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['Close'].pct_change().fillna(0)
        
        # Pre-calculate volatility regimes
        self._calculate_market_regimes()
        
    def _calculate_market_regimes(self) -> None:
        """Pre-calculate market volatility regimes."""
        # Calculate rolling volatility
        self.df['rolling_vol'] = self.df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Define volatility regimes using percentiles
        vol_percentiles = self.df['rolling_vol'].quantile([0.33, 0.67])
        self.df['vol_regime'] = pd.cut(
            self.df['rolling_vol'],
            bins=[-np.inf, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Calculate trend strength
        self.df['trend_strength'] = self.df['Close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 1 else 0
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (initial observation, info dict)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset to random starting point (ensuring enough history)
        min_start = max(self.lookback_window, 200)  # Need some history for indicators
        max_start = len(self.df) - 100  # Leave some data for episode
        
        self.current_step = np.random.randint(min_start, max_start)
        
        # Initialize state
        self.state = TradingState(
            position=0.0,
            entry_price=0.0,
            current_price=self.df.iloc[self.current_step]['Close'],
            cash=self.initial_capital,
            portfolio_value=self.initial_capital,
            step_count=0,
            max_drawdown=0.0,
            cumulative_return=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            total_trades=0,
            winning_trades=0
        )
        
        # Reset history
        self.trades = []
        self.positions = []
        self.portfolio_values = [self.initial_capital]
        self.returns_history = []
        self.actions_history = []
        self.vol_history = []
        
        # Reset step counter - start after lookback window
        self.current_step = max(self.lookback_window, 1)
        self.prev_position = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.
        
        Args:
            action: Trading action to take
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Check if already terminated
        if self.current_step >= len(self.df) - 2:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")
        
        # Get current and next prices
        current_price = self.df.iloc[self.current_step]['Close']
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        next_price = self.df.iloc[self.current_step]['Close']
        daily_return = self.df.iloc[self.current_step]['returns']
        
        # Process action based on action space type
        if self.use_discrete_actions:
            target_position = self._process_discrete_action(action)
        else:
            target_position = float(action[0])  # Continuous action
            target_position = np.clip(target_position, 0.0, 1.0)  # Long only for now
        
        # Apply position sizing method
        target_position = self._apply_position_sizing(target_position)
        
        # Clip to max position size
        target_position = np.clip(target_position, -self.max_position_size, self.max_position_size)
        
        # Store previous position before trade
        self.prev_position = self.state.position
        
        # Execute trade with improved position handling
        trade_executed, trade_cost = self._execute_trade(
            self.state.position,
            target_position,
            self.state.current_price
        )
        
        # Track if we just made a trade
        trade_return = 0.0
        if trade_executed and self.state.total_trades > 0:
            # Calculate trade return if we closed a position
            if len(self.trades) > 0:
                last_trade = self.trades[-1]
                if 'realized_pnl' in last_trade:
                    trade_return = last_trade['realized_pnl'] - (self.state.realized_pnl - last_trade['realized_pnl'])
        
        # Update portfolio value
        if self.state.position > 0:
            # Long position
            position_return = (next_price - current_price) / current_price
            position_value = self.state.position * self.state.portfolio_value * (1 + position_return)
            cash_value = (1 - self.state.position) * self.state.portfolio_value
            self.state.portfolio_value = cash_value + position_value
        else:
            # No position
            self.state.portfolio_value = self.state.cash
        
        # Update state
        self.state.current_price = next_price
        self.state.step_count += 1
        
        # Update performance metrics
        self._update_performance_metrics(daily_return if self.state.position > 0 else 0, action)
        
        # Calculate reward using balanced function for better learning
        reward = self._calculate_balanced_reward(
            daily_return=daily_return,
            trade_cost=trade_cost,
            trade_executed=trade_executed,
            current_drawdown=self._calculate_current_drawdown()
        )
        
        # Get next observation
        observation = self._get_observation()
        
        # Store position
        self.positions.append(self.state.position)
        
        # Prepare info dict
        info = {
            'portfolio_value': self.state.portfolio_value,
            'position': self.state.position,
            'cumulative_return': self.state.cumulative_return,
            'max_drawdown': self.state.max_drawdown,
            'sharpe_ratio': self.state.sharpe_ratio,
            'win_rate': self.state.win_rate,
            'trades_count': self.state.total_trades,
            'current_price': self.state.current_price,
            'unrealized_pnl': self.state.unrealized_pnl,
            'trade_made': trade_executed,
            'trade_return': trade_return,
            'trade_cost': trade_cost if trade_executed else 0.0
        }
        
        # Check if episode is terminated or truncated
        terminated = self.current_step >= len(self.df) - 2
        truncated = False  # Can be set based on max episode length if needed
        
        return observation, reward, terminated, truncated, info
    
    def _apply_position_sizing(self, target_position: float) -> float:
        """
        Apply position sizing based on the selected method.
        
        Args:
            target_position: Raw target position (-1.0 to 1.0)
            
        Returns:
            Adjusted position size based on position sizing method
        """
        if target_position == 0.0 or self.position_sizing == 'fixed':
            return target_position
            
        if self.position_sizing == 'kelly':
            kelly_fraction = self._calculate_kelly_fraction()
            return target_position * kelly_fraction
            
        elif self.position_sizing == 'volatility':
            vol_scalar = self._calculate_volatility_scalar()
            return target_position * vol_scalar
            
        return target_position
        
    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Returns:
            Kelly fraction (0.0 to 1.0)
        """
        if len(self.returns_history) < 30:
            return 0.5  # Default to 50%
        
        returns = np.array(self.returns_history[-60:])  # Last 60 trades
        
        # Calculate win rate and average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.5
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula with safety factor
        kelly = (win_rate * avg_loss - (1 - win_rate) * avg_win) / (avg_win * avg_loss)
        kelly = kelly * 0.25  # Conservative Kelly (25%)
        
        return np.clip(kelly, 0.1, 0.6)
    
    def _calculate_volatility_scalar(self) -> float:
        """
        Calculate position scalar based on current volatility.
        
        Returns:
            Volatility scalar (0.0 to 1.0)
        """
        if len(self.returns_history) < 20:
            return 0.8  # Less conservative default for early periods
        
        # Calculate recent volatility (20-period)
        recent_vol = np.std(self.returns_history[-20:])
        
        # More aggressive volatility scaling for 5-minute data
        # Use percentile-based scaling
        if len(self.vol_history) >= 100:
            vol_percentile = np.percentile(self.vol_history[-100:], [20, 50, 80])
            
            if recent_vol <= vol_percentile[0]:  # Low volatility
                return 1.0  # Full position
            elif recent_vol <= vol_percentile[1]:  # Medium volatility
                return 0.8
            elif recent_vol <= vol_percentile[2]:  # High volatility
                return 0.6
            else:  # Very high volatility
                return 0.4
        else:
            # Fallback to simple scaling
            target_vol = 0.003  # Adjusted target for 5-min data
            vol_scalar = np.clip(target_vol / (recent_vol + 1e-8), 0.3, 1.0)
            return vol_scalar
    
    def _process_discrete_action(self, action) -> float:
        """
        Map discrete action to target position with improved scaling.
        
        Args:
            action: Discrete action index (0-4) or numpy array
            
        Returns:
            Target position (-1.0 to 1.0)
        """
        # More aggressive action mapping to encourage trading
        action_map = {
            int(Actions.SELL_STRONG): -1.0,   # Full short position
            int(Actions.SELL_WEAK): -0.5,     # Half short position
            int(Actions.HOLD): 0.0,           # Flat
            int(Actions.BUY_WEAK): 0.5,       # Half long position
            int(Actions.BUY_STRONG): 1.0      # Full long position
        }
        
        # Handle numpy array input
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
            
        # Get base position from action
        target_position = action_map.get(action, 0.0)
        
        # Apply position sizing with less aggressive scaling
        if self.position_sizing == 'volatility' and target_position != 0:
            vol_scalar = self._calculate_volatility_scalar()
            # Use square root of volatility scalar for less aggressive reduction
            target_position *= np.sqrt(vol_scalar)
        elif self.position_sizing == 'kelly' and target_position != 0:
            kelly_fraction = self._calculate_kelly_fraction()
            target_position *= kelly_fraction
        
        # Apply max position size limit
        return np.clip(target_position, -self.max_position_size, self.max_position_size)
    
    def _execute_trade(self, 
                      current_position: float,
                      target_position: float,
                      current_price: float) -> Tuple[bool, float]:
        """
        Execute trade with realistic constraints for 5-minute data.
        Supports both long and short positions.
        
        Args:
            current_position: Current position size (-1.0 to 1.0)
            target_position: Target position size (-1.0 to 1.0)
            current_price: Current asset price
            
        Returns:
            Tuple of (trade_executed, trade_cost)
        """
        position_change = target_position - current_position
        trade_executed = False
        trade_cost = 0.0
        
        # Only trade if position change is significant
        if abs(position_change) > self.min_position_change:
            # Calculate trade value
            trade_value = abs(position_change) * self.state.portfolio_value
            
            # Apply transaction cost and slippage
            trade_cost = trade_value * self.transaction_cost
            slippage_cost = trade_value * self.slippage
            total_cost = trade_cost + slippage_cost
            
            # Check if we're closing a position or opening a new one
            is_closing = (current_position > 0 and target_position <= 0) or \
                        (current_position < 0 and target_position >= 0) or \
                        (current_position != 0 and target_position == 0)
            
            # Calculate P&L if closing a position
            if is_closing and self.state.entry_price != 0:
                exit_price = current_price * (1 - np.sign(current_position) * self.slippage)
                
                if current_position > 0:  # Closing long position
                    pnl = (exit_price - self.state.entry_price) / self.state.entry_price
                else:  # Closing short position
                    pnl = (self.state.entry_price - exit_price) / self.state.entry_price
                
                self.state.realized_pnl += pnl * abs(current_position)  # Scale by position size
                self.returns_history.append(pnl)
                
                # Update win rate
                self.state.total_trades += 1
                if pnl > 0:
                    self.state.winning_trades += 1
            
            # Update position and entry price
            self.state.position = target_position
            
            # Set new entry price if opening a position
            if target_position != 0 and (current_position * target_position <= 0):
                # Opening a new position (long or short)
                self.state.entry_price = current_price * (1 + np.sign(target_position) * self.slippage)
            elif target_position == 0:
                # Fully closed position
                self.state.entry_price = 0.0
            
            # Apply costs
            self.state.cash -= total_cost
            self.state.portfolio_value -= total_cost
            
            # Determine trade action type
            if current_position * target_position < 0:  # Reversing position
                action_type = 'REVERSE_TO_SHORT' if target_position < 0 else 'REVERSE_TO_LONG'
            elif position_change > 0:
                action_type = 'BUY' if target_position > 0 else 'COVER_SHORT'
            else:
                action_type = 'SELL' if current_position > 0 else 'SHORT'
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': action_type,
                'price': current_price,
                'position_change': position_change,
                'new_position': target_position,
                'cost': total_cost,
                'portfolio_value': self.state.portfolio_value,
                'size': abs(position_change) * self.state.portfolio_value / current_price,
                'entry_price': self.state.entry_price,
                'realized_pnl': self.state.realized_pnl
            })
            
            trade_executed = True
            trade_cost = total_cost
        
        return trade_executed, trade_cost
    
    def _update_performance_metrics(self, period_return: float, action: int) -> None:
        """Update running performance metrics."""
        # Update cumulative return
        self.state.cumulative_return = (self.state.portfolio_value - self.initial_capital) / self.initial_capital
        # Update histories
        self.positions.append(self.state.position)
        self.portfolio_values.append(self.state.portfolio_value)
        self.returns_history.append(period_return)
        self.actions_history.append(action)
        
        # Update volatility history
        if len(self.returns_history) >= 20:
            recent_vol = np.std(self.returns_history[-20:])
            self.vol_history.append(recent_vol)
        
        # Update max drawdown
        self.state.max_drawdown = max(self.state.max_drawdown, self._calculate_current_drawdown())
        
        # Update unrealized P&L
        if self.state.position > 0 and self.state.entry_price > 0:
            self.state.unrealized_pnl = (self.state.current_price - self.state.entry_price) / self.state.entry_price
        else:
            self.state.unrealized_pnl = 0.0
        
        # Update Sharpe ratio (simplified rolling calculation)
        if len(self.portfolio_values) > 20:
            recent_returns = pd.Series(self.portfolio_values[-20:]).pct_change().dropna()
            if len(recent_returns) > 0 and recent_returns.std() > 0:
                self.state.sharpe_ratio = np.sqrt(252) * recent_returns.mean() / recent_returns.std()
        
        # Update win rate
        if self.state.total_trades > 0:
            self.state.win_rate = self.state.winning_trades / self.state.total_trades
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if len(self.portfolio_values) == 0:
            return 0.0
        
        peak = max(self.portfolio_values)
        current = self.state.portfolio_value
        return (peak - current) / peak
    
    def _calculate_aggressive_reward(self, 
                                   daily_return: float,
                                   trade_cost: float,
                                   trade_executed: bool,
                                   current_drawdown: float) -> float:
        """
        Balanced reward function that encourages trading with reasonable scaling.
        """
        reward = 0.0
        
        # 1. Incentive for being in the market with reasonable scaling
        if self.state.position != 0:
            # Base reward for position-weighted returns
            position_weighted_return = daily_return * abs(self.state.position)
            reward += position_weighted_return * 50  # Reduced from 200
            
            # Extra reward for profitable positions
            if position_weighted_return > 0:
                reward += position_weighted_return * 25  # Reduced from 100
        else:
            # Moderate penalty for being out of the market
            reward -= 2.0  # Reduced from 10.0
            
            # Penalty for missing opportunities
            if abs(daily_return) > 0.001:  # Any meaningful move
                reward -= abs(daily_return) * 50  # Reduced from 200
        
        # 2. Reward for executing trades
        if trade_executed:
            reward += 5.0  # Reduced from 20.0
            
            # Extra reward for opening positions
            if self.prev_position == 0 and self.state.position != 0:
                reward += 10.0  # Reduced from 50.0
            
            # Transaction cost (reduced impact)
            reward -= trade_cost * 0.5  # Minimal transaction penalty
        else:
            # Small penalty for not trading when we should
            if self.state.step_count > 10:  # After warmup
                reward -= 1.0  # Reduced from 5.0
        
        # 3. Volatility-based trading encouragement
        if len(self.returns_history) >= 20:
            recent_vol = np.std(self.returns_history[-20:])
            if recent_vol > 0.002 and self.state.position == 0:
                # High volatility but not trading - moderate penalty
                reward -= 5.0  # Reduced from 30.0
        
        # 4. Trend following bonus (simplified to avoid errors)
        if self.current_step > 20 and len(self.returns_history) >= 20:
            # Use returns instead of prices to avoid column issues
            recent_returns = self.returns_history[-10:]
            if len(recent_returns) >= 10:
                trend = np.mean(recent_returns[-5:]) - np.mean(recent_returns[-10:-5])
                if self.state.position * trend > 0:  # Position aligned with trend
                    reward += min(abs(trend) * 20, 5)  # Capped bonus
        
        # 5. Quick profit-taking reward
        if trade_executed and self.state.unrealized_pnl > 0.005:  # 0.5% profit
            if abs(self.state.position) < abs(self.prev_position):
                reward += 5.0  # Reduced from 30.0
        
        # 6. Minimal drawdown penalty (don't discourage trading)
        if current_drawdown > 0.1:  # Only penalize severe drawdowns
            reward -= current_drawdown * 20  # Reduced from complex calculation
        
        # 7. Clip reward to reasonable range
        reward = np.clip(reward, -100, 100)  # Prevent extreme values
        
        return reward
    
    def _calculate_simple_reward(self,
                                daily_return: float,
                                trade_cost: float,
                                trade_executed: bool,
                                current_drawdown: float) -> float:
        """
        Simple reward function focused on profitable trading.
        """
        # Start with position-weighted returns
        if self.state.position != 0:
            position_weighted_return = daily_return * abs(self.state.position)
            reward = position_weighted_return * 100  # Scale returns
        else:
            reward = 0  # No penalty for being out of market
        
        # Small bonus for trading to encourage activity
        if trade_executed:
            reward += 1.0
            # Subtract actual transaction costs
            reward -= trade_cost
        
        # Small drawdown penalty
        if current_drawdown > 0.05:  # 5% drawdown
            reward -= current_drawdown * 10
        
        # Clip to reasonable range
        reward = np.clip(reward, -50, 50)
        
        return reward
    
    def _calculate_balanced_reward(self,
                                    daily_return: float,
                                    trade_cost: float,
                                    trade_executed: bool,
                                    current_drawdown: float) -> float:
        """
        Balanced reward function with clear signals for profitable trading.
        """
        reward = 0.0
        
        # 1. Core: Position-weighted returns with moderate scaling
        if self.state.position != 0:
            position_weighted_return = daily_return * abs(self.state.position)
            reward += position_weighted_return * 100  # Scale returns
            
            # Bonus for profitable positions
            if position_weighted_return > 0:
                reward += position_weighted_return * 50  # Extra reward for profits
            else:
                # Small penalty for losses to encourage learning
                reward += position_weighted_return * 20  # Less penalty than reward
        
        # 2. Small penalty for being out of market during opportunities
        if self.state.position == 0 and abs(daily_return) > 0.001:
            reward -= abs(daily_return) * 10  # Opportunity cost
        
        # 3. Trade execution with smart incentives
        if trade_executed:
            # Small bonus for trading activity
            reward += 0.5
            
            # Reward good entries (entering when volatility is higher)
            if self.prev_position == 0 and self.state.position != 0:
                if len(self.returns_history) >= 20:
                    recent_vol = np.std(self.returns_history[-10:])
                    avg_vol = np.std(self.returns_history[-20:])
                    if recent_vol > avg_vol * 1.2:  # Higher than average volatility
                        reward += 2.0  # Good timing bonus
            
            # Reward profit taking
            if self.state.position == 0 and self.prev_position != 0:
                # Check if we just closed a profitable position
                if len(self.trades) > 0:
                    last_trade = self.trades[-1]
                    if 'realized_pnl' in last_trade and last_trade['realized_pnl'] > 0:
                        reward += min(last_trade['realized_pnl'] * 50, 10)  # Capped reward for profit taking
            
            # Subtract actual costs
            reward -= trade_cost
        
        # 4. Risk management
        if current_drawdown > 0.05:  # 5% drawdown
            reward -= current_drawdown * 20  # Progressive penalty
        
        # 5. Consistency bonus (reduce variance in returns)
        if len(self.returns_history) >= 30:
            recent_returns = self.returns_history[-10:]
            longer_returns = self.returns_history[-30:]
            if np.std(recent_returns) < np.std(longer_returns) * 0.7:
                reward += 1.0  # Reward stable returns
        
        # 6. Clip to reasonable range
        reward = np.clip(reward, -50, 50)
        
        return reward
    
    def _calculate_improved_reward(self, 
                             daily_return: float,
                             trade_cost: float,
                             trade_executed: bool,
                             current_drawdown: float) -> float:
        """
        Calculate balanced reward that encourages profitable trading without forcing action.
        """
        reward = 0.0
        
        # 1. Base reward: Position-weighted returns (realistic scaling)
        if self.state.position != 0:
            position_weighted_return = daily_return * self.state.position
            # Scale returns appropriately for 5-minute bars
            reward += position_weighted_return * 100  # Balanced scaling
            
            # Asymmetric risk-reward to encourage cutting losses
            if position_weighted_return > 0:
                reward += position_weighted_return * 20  # Bonus for profits
            else:
                # Slightly larger penalty for losses to encourage risk management
                reward += position_weighted_return * 10
        
        # 2. Transaction costs (realistic)
        if trade_executed:
            reward -= trade_cost * 2.0  # Realistic transaction penalty
        
        # 3. Trading activity incentives (balanced)
        if trade_executed:
            position_change = abs(self.state.position - self.prev_position)
            
            # Small reward for trading, but only if the trade size is meaningful
            if position_change >= 0.05:  # 5%+ position change
                reward += 2.0  # Small bonus for meaningful trades
                
                # Extra bonus for entering positions when volatility is right
                if self.prev_position == 0 and self.state.position != 0:
                    current_vol = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.01
                    if 0.001 < current_vol < 0.02:  # Goldilocks volatility zone
                        reward += 5.0
        
        # 4. Opportunity cost (gentle nudge, not punishment)
        if self.state.position == 0 and abs(daily_return) > 0.002:  # Missing large moves
            reward -= abs(daily_return) * 50  # Gentle penalty for missing opportunities
        
        # 5. Risk management rewards
        # Reward for exiting losing positions quickly
        if self.state.position != 0 and self.state.unrealized_pnl < -0.02:  # 2% loss
            if trade_executed and abs(self.state.position) < abs(self.prev_position):
                reward += 10.0  # Reward for cutting losses
        
        # 6. Trend alignment bonus (moderate)
        if self.current_step > 10:
            recent_returns = self.returns_history[-10:]
            if len(recent_returns) >= 10:
                trend = np.mean(recent_returns[-5:]) - np.mean(recent_returns[-10:-5])
                if self.state.position * trend > 0:  # Aligned with trend
                    reward += min(abs(trend) * 50, 10)  # Capped bonus
        
        # 7. Drawdown penalty (progressive)
        if current_drawdown > 0.05:  # 5%+ drawdown
            # Progressive penalty that increases with drawdown
            penalty_multiplier = min(current_drawdown / 0.05, 3.0)  # Cap at 3x
            reward -= current_drawdown * self.drawdown_penalty * penalty_multiplier * 10
        
        # 8. Sharpe ratio improvement bonus
        if len(self.returns_history) >= 20:
            recent_sharpe = self._calculate_rolling_sharpe(20)
            if recent_sharpe > 0.5:  # Good risk-adjusted returns
                reward += min(recent_sharpe * 2, 10)  # Capped bonus
        
        return reward
    
    def _calculate_rolling_sharpe(self, window: int) -> float:
        """Calculate rolling Sharpe ratio for reward calculation."""
        if len(self.returns_history) < window:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-window:])
        if recent_returns.std() > 0:
            # Annualize for 5-minute bars (78 bars per day * 252 days)
            return np.sqrt(78 * 252) * recent_returns.mean() / recent_returns.std()
        return 0.0
    
    def _calculate_reward(self, 
                        daily_return: float,
                        trade_cost: float,
                        trade_executed: bool,
                        current_drawdown: float) -> float:
        """
        Original reward function - kept for backward compatibility.
        This method now calls the improved reward function.
        """
        return self._calculate_improved_reward(
            daily_return=daily_return,
            trade_cost=trade_cost,
            trade_executed=trade_executed,
            current_drawdown=current_drawdown
        )
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Observation array
        """
        # Get feature history
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1
        
        feature_history = []
        for i in range(start_idx, end_idx):
            if i < len(self.df):
                # Only use features that exist in the DataFrame
                row_features = []
                for col in self.feature_columns:
                    if col in self.df.columns:
                        row_features.append(self.df.iloc[i][col])
                    else:
                        row_features.append(0.0)  # Use 0 for missing features
                feature_history.extend(row_features)
            else:
                # Pad with zeros if not enough history
                feature_history.extend([0] * len(self.feature_columns))
        
        # Enhanced portfolio state features (always 12 features)
        portfolio_features = [
            float(self.state.position),  # Current position
            float(self.state.cumulative_return),  # Cumulative return
            float(self.state.max_drawdown),  # Max drawdown
            float(self.state.unrealized_pnl),  # Unrealized P&L
            float(self.state.step_count / len(self.df)),  # Progress through dataset
            float(self.state.sharpe_ratio),  # Current Sharpe ratio
            float(self.state.win_rate),  # Win rate
            float(self.state.total_trades / 100),  # Normalized trade count
            float(self.df.iloc[self.current_step]['rolling_vol']) if 'rolling_vol' in self.df.columns else 0.0,
            float(self.df.iloc[self.current_step]['trend_strength']) if 'trend_strength' in self.df.columns else 0.0,
            float(self.portfolio_values[-1] / self.initial_capital) if self.portfolio_values else 1.0,
            float(len(self.trades) / (self.state.step_count + 1))  # Trading frequency
        ]
        
        # Ensure no NaN values
        portfolio_features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in portfolio_features]
        
        # Combine all features
        observation = np.array(feature_history + portfolio_features, dtype=np.float32)
        
        # Handle any remaining NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Ensure observation matches expected shape
        if observation.shape[0] != self.n_total_features:
            logger.warning(f"Observation shape mismatch: {observation.shape[0]} vs expected {self.n_total_features}")
            # Pad or trim as needed
            if observation.shape[0] < self.n_total_features:
                # Pad with zeros
                padding = self.n_total_features - observation.shape[0]
                observation = np.pad(observation, (0, padding), 'constant', constant_values=0.0)
            else:
                # Trim excess
                observation = observation[:self.n_total_features]
        
        # Clip to observation space bounds
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        
        return observation
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment (print current state).
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"\n=== Step {self.state.step_count} ===")
            print(f"Date: {self.df.index[self.current_step].strftime('%Y-%m-%d')}")
            print(f"Price: ${self.state.current_price:.2f}")
            print(f"Position: {self.state.position:.1%}")
            print(f"Portfolio Value: ${self.state.portfolio_value:,.2f}")
            print(f"Cumulative Return: {self.state.cumulative_return:.2%}")
            print(f"Max Drawdown: {self.state.max_drawdown:.2%}")
            print(f"Sharpe Ratio: {self.state.sharpe_ratio:.2f}")
            print(f"Win Rate: {self.state.win_rate:.2%}")
            print(f"Trades: {self.state.total_trades}")
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for the completed episode.
        
        Returns:
            Dictionary of performance metrics
        """
        # Check if episode is complete by checking if we've reached the end of the data
        episode_complete = self.current_step >= len(self.df) - 2
        
        if not episode_complete and len(self.portfolio_values) < 2:
            logger.warning("Episode not complete or too short for statistics")
            return {}
        
        portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        # Calculate metrics
        total_return = self.state.cumulative_return
        
        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = np.sqrt(252) * portfolio_returns.mean() / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio  # If no downside, use Sharpe
        
        # Calmar ratio
        calmar_ratio = total_return / self.state.max_drawdown if self.state.max_drawdown > 0 else 0
        
        # Average position duration
        position_durations = []
        in_position = False
        position_start = 0
        
        for i, pos in enumerate(self.positions):
            if pos > 0 and not in_position:
                position_start = i
                in_position = True
            elif pos == 0 and in_position:
                position_durations.append(i - position_start)
                in_position = False
        
        avg_position_duration = np.mean(position_durations) if position_durations else 0
        
        # Profit factor
        if len(self.returns_history) > 0:
            returns_array = np.array(self.returns_history)
            gross_profits = returns_array[returns_array > 0].sum()
            gross_losses = abs(returns_array[returns_array < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        else:
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': self.state.max_drawdown,
            'win_rate': self.state.win_rate,
            'total_trades': self.state.total_trades,
            'avg_position_duration': avg_position_duration,
            'profit_factor': profit_factor,
            'final_portfolio_value': self.state.portfolio_value
        }


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 1000)))
    
    df = pd.DataFrame({
        'Close': prices,
        'returns': np.random.normal(0, 0.02, 1000),
        'feature1_norm': np.random.normal(0, 1, 1000),
        'feature2_norm': np.random.normal(0, 1, 1000),
        'feature3_norm': np.random.normal(0, 1, 1000)
    }, index=dates)
    
    # Create environment
    env = TradingEnvironment(
        df=df,
        feature_columns=['feature1_norm', 'feature2_norm', 'feature3_norm'],
        initial_capital=100000.0,
        position_sizing='volatility'
    )
    
    # Test environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    # Get episode statistics
    stats = env.get_episode_statistics()
    print("\nEpisode Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")