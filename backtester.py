"""
Backtester Module for RL Trading System
======================================
This module provides comprehensive backtesting and performance evaluation
capabilities for the trained RL trading agent.

Author: Senior Quantitative Developer
Date: 2024
Version: 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import os
from scipy import stats
import warnings
from dataclasses import dataclass
from collections import defaultdict
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Detailed trade record for analysis."""
    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'ADJUST'
    price: float
    size: float
    position_before: float
    position_after: float
    portfolio_value: float
    cost: float
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_period: int = 0


class Backtester:
    """
    Enhanced backtesting engine for evaluating trading strategies.
    
    Features:
    - Comprehensive performance metrics (30+ metrics)
    - Benchmark comparison with multiple strategies
    - Risk analysis and attribution
    - Trade analysis with detailed breakdown
    - Monte Carlo simulations
    - Walk-forward analysis support
    - Advanced visualization suite
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95):
        """
        Initialize the enhanced backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            confidence_level: Confidence level for VaR/CVaR calculations
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.results = {}
        self.trade_records = defaultdict(list)
        
    def backtest_agent(self,
                    agent: Any,
                    test_env: Any,
                    test_data: pd.DataFrame,
                    deterministic: bool = True,
                    n_runs: int = 1) -> Dict[str, Any]:
        """
        Enhanced backtest of the trained RL agent.
        
        Args:
            agent: Trained RL agent
            test_env: Testing environment
            test_data: DataFrame with test period data
            deterministic: Whether to use deterministic policy
            n_runs: Number of runs for Monte Carlo analysis
            
        Returns:
            Dictionary containing comprehensive backtest results
        """
        logger.info(f"Starting enhanced agent backtest ({n_runs} runs)...")
        
        all_run_results = []
        
        for run in range(n_runs):
            if n_runs > 1:
                logger.info(f"Run {run + 1}/{n_runs}")
            
            # Run agent through test period
            reset_output = test_env.reset()
            if isinstance(reset_output, tuple):
                obs, _ = reset_output
            else:
                obs = reset_output
                
            done = False
            
            actions = []
            positions = []
            portfolio_values = [self.initial_capital]
            rewards = []
            step_count = 0
            
            # Run the episode
            while not done:
                # Get action from agent
                # IMPORTANT: Don't reshape the observation here - let the agent handle it
                action, action_info = agent.predict(obs, deterministic=deterministic)
                
                # Step environment
                step_output = test_env.step(action)
                
                if len(step_output) == 4:
                    obs, reward, done, info = step_output
                else:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                
                actions.append(action)
                positions.append(info['position'])
                portfolio_values.append(info['portfolio_value'])
                rewards.append(reward)
                step_count += 1
            
            # Get final statistics from environment
            env_stats = test_env.get_episode_statistics()
            
            # Create results DataFrame with proper length
            # Use step_count + 1 to include initial state
            actual_length = min(step_count + 1, len(test_data))
            
            results_df = pd.DataFrame({
                'date': test_data.index[:actual_length],
                'portfolio_value': portfolio_values[:actual_length],
                'position': [0] + positions[:actual_length-1],  # Include initial position of 0
                'action': [0] + actions[:actual_length-1],      # Include initial action of 0
                'price': test_data['Close'].iloc[:actual_length].values,
                'reward': [0] + rewards[:actual_length-1]       # Include initial reward of 0
            })
            
            # Calculate returns
            results_df['returns'] = results_df['portfolio_value'].pct_change()
            results_df['log_returns'] = np.log(results_df['portfolio_value'] / results_df['portfolio_value'].shift(1))
            results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
            
            # Process trades - FIXED: ensure trades are within bounds
            raw_trades = test_env.trades if hasattr(test_env, 'trades') else []
            valid_trades = [t for t in raw_trades if t.get('step', 0) < actual_length]
            trades = self._process_trades(results_df, valid_trades)
            
            all_run_results.append({
                'data': results_df,
                'env_stats': env_stats,
                'trades': trades
            })
        
        # Aggregate results across runs
        if n_runs > 1:
            aggregated_results = self._aggregate_monte_carlo_results(all_run_results)
            self.results['agent'] = aggregated_results
        else:
            self.results['agent'] = all_run_results[0]
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            self.results['agent']['data'], 
            'RL Agent',
            self.results['agent']['trades']
        )
        
        logger.info(f"Agent backtest complete. Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def backtest_buy_hold(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced buy-and-hold strategy backtest.
        
        Args:
            test_data: DataFrame with test period data
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Running buy-and-hold backtest...")
        
        # Calculate buy-and-hold returns with costs
        initial_price = test_data['Close'].iloc[0]
        
        # Apply initial transaction cost
        initial_cost = self.initial_capital * self.transaction_cost
        invested_capital = self.initial_capital - initial_cost
        
        # Calculate portfolio values
        portfolio_values = invested_capital * (test_data['Close'] / initial_price)
        
        results_df = pd.DataFrame({
            'date': test_data.index,
            'portfolio_value': portfolio_values,
            'position': 1,
            'price': test_data['Close']
        })
        
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['log_returns'] = np.log(results_df['portfolio_value'] / results_df['portfolio_value'].shift(1))
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        # Create single trade record
        trades = [
            TradeRecord(
                timestamp=test_data.index[0],
                action='BUY',
                price=initial_price,
                size=invested_capital / initial_price,
                position_before=0,
                position_after=1,
                portfolio_value=self.initial_capital,
                cost=initial_cost
            )
        ]
        
        # Store results
        self.results['buy_hold'] = {
            'data': results_df,
            'trades': trades
        }
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(results_df, 'Buy & Hold', trades)
        
        logger.info(f"Buy & Hold backtest complete. Return: {metrics['total_return']:.2%}")
        
        return metrics
    
    def backtest_sma_crossover(self, 
                              test_data: pd.DataFrame,
                              fast_period: int = 20,
                              slow_period: int = 50) -> Dict[str, Any]:
        """
        Enhanced SMA crossover strategy backtest.
        
        Args:
            test_data: DataFrame with test period data
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Running SMA crossover backtest ({fast_period}/{slow_period})...")
        
        # Calculate SMAs
        sma_fast = test_data['Close'].rolling(fast_period).mean()
        sma_slow = test_data['Close'].rolling(slow_period).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=test_data.index)
        signals['signal'] = 0
        signals.loc[sma_fast > sma_slow, 'signal'] = 1
        signals['position'] = signals['signal'].diff()
        
        # Simulate trading with realistic execution
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        entry_price = None
        entry_date = None
        
        for i, (date, row) in enumerate(test_data.iterrows()):
            # Check for trade signal
            if i < slow_period:
                portfolio_value = cash
            elif signals.iloc[i]['position'] == 1:  # Buy signal
                if shares == 0:
                    # Apply slippage and transaction costs
                    execution_price = row['Close'] * (1 + self.slippage)
                    trade_cost = cash * self.transaction_cost
                    
                    shares = (cash - trade_cost) / execution_price
                    cash = 0
                    entry_price = execution_price
                    entry_date = date
                    
                    trades.append(TradeRecord(
                        timestamp=date,
                        action='BUY',
                        price=execution_price,
                        size=shares,
                        position_before=0,
                        position_after=1,
                        portfolio_value=portfolio_value,
                        cost=trade_cost
                    ))
                portfolio_value = shares * row['Close']
                
            elif signals.iloc[i]['position'] == -1:  # Sell signal
                if shares > 0:
                    # Apply slippage and transaction costs
                    execution_price = row['Close'] * (1 - self.slippage)
                    gross_proceeds = shares * execution_price
                    trade_cost = gross_proceeds * self.transaction_cost
                    cash = gross_proceeds - trade_cost
                    
                    # Calculate trade return
                    if entry_price:
                        trade_return = (execution_price - entry_price) / entry_price
                        holding_period = (date - entry_date).days if entry_date else 0
                    else:
                        trade_return = 0
                        holding_period = 0
                    
                    trades.append(TradeRecord(
                        timestamp=date,
                        action='SELL',
                        price=execution_price,
                        size=shares,
                        position_before=1,
                        position_after=0,
                        portfolio_value=portfolio_value,
                        cost=trade_cost,
                        pnl=gross_proceeds - shares * entry_price if entry_price else 0,
                        return_pct=trade_return,
                        holding_period=holding_period
                    ))
                    
                    shares = 0
                    entry_price = None
                    entry_date = None
                portfolio_value = cash
            else:
                # Hold position
                portfolio_value = cash if shares == 0 else shares * row['Close']
            
            portfolio_values.append(portfolio_value)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': test_data.index,
            'portfolio_value': portfolio_values,
            'position': signals['signal'],
            'price': test_data['Close'],
            'sma_fast': sma_fast,
            'sma_slow': sma_slow
        })
        
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['log_returns'] = np.log(results_df['portfolio_value'] / results_df['portfolio_value'].shift(1))
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        # Store results
        self.results['sma_crossover'] = {
            'data': results_df,
            'trades': trades
        }
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(results_df, 'SMA Crossover', trades)
        
        logger.info(f"SMA Crossover backtest complete. Return: {metrics['total_return']:.2%}")
        
        return metrics
    
    def backtest_momentum(self, 
                         test_data: pd.DataFrame,
                         lookback_period: int = 60,
                         holding_period: int = 20) -> Dict[str, Any]:
        """
        Backtest momentum strategy.
        
        Args:
            test_data: DataFrame with test period data
            lookback_period: Lookback period for momentum calculation
            holding_period: Holding period for positions
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Running momentum strategy backtest (lookback={lookback_period})...")
        
        # Calculate momentum
        momentum = test_data['Close'].pct_change(lookback_period)
        
        # Generate signals (long if positive momentum)
        signals = pd.DataFrame(index=test_data.index)
        signals['momentum'] = momentum
        signals['signal'] = (momentum > 0).astype(int)
        
        # Rebalance every holding_period days
        signals['rebalance'] = 0
        for i in range(lookback_period, len(signals), holding_period):
            if i < len(signals):
                signals.iloc[i, signals.columns.get_loc('rebalance')] = 1
        
        # Simulate trading
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        
        for i, (date, row) in enumerate(test_data.iterrows()):
            if i < lookback_period:
                portfolio_value = cash
            elif signals.iloc[i]['rebalance'] == 1:
                # Rebalance position
                target_position = signals.iloc[i]['signal']
                
                if target_position == 1 and shares == 0:
                    # Buy
                    execution_price = row['Close'] * (1 + self.slippage)
                    trade_cost = cash * self.transaction_cost
                    shares = (cash - trade_cost) / execution_price
                    cash = 0
                    
                    trades.append(TradeRecord(
                        timestamp=date,
                        action='BUY',
                        price=execution_price,
                        size=shares,
                        position_before=0,
                        position_after=1,
                        portfolio_value=portfolio_value,
                        cost=trade_cost
                    ))
                    
                elif target_position == 0 and shares > 0:
                    # Sell
                    execution_price = row['Close'] * (1 - self.slippage)
                    gross_proceeds = shares * execution_price
                    trade_cost = gross_proceeds * self.transaction_cost
                    cash = gross_proceeds - trade_cost
                    
                    trades.append(TradeRecord(
                        timestamp=date,
                        action='SELL',
                        price=execution_price,
                        size=shares,
                        position_before=1,
                        position_after=0,
                        portfolio_value=portfolio_value,
                        cost=trade_cost
                    ))
                    
                    shares = 0
                
                portfolio_value = cash if shares == 0 else shares * row['Close']
            else:
                # Hold position
                portfolio_value = cash if shares == 0 else shares * row['Close']
            
            portfolio_values.append(portfolio_value)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': test_data.index,
            'portfolio_value': portfolio_values,
            'position': signals['signal'],
            'price': test_data['Close'],
            'momentum': momentum
        })
        
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['log_returns'] = np.log(results_df['portfolio_value'] / results_df['portfolio_value'].shift(1))
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        # Store results
        self.results['momentum'] = {
            'data': results_df,
            'trades': trades
        }
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(results_df, 'Momentum', trades)
        
        logger.info(f"Momentum backtest complete. Return: {metrics['total_return']:.2%}")
        
        return metrics
    
    def _process_trades(self, results_df: pd.DataFrame, raw_trades: List[Dict]) -> List[TradeRecord]:
        """Process raw trades into TradeRecord objects."""
        processed_trades = []
        results_length = len(results_df)
        
        for i, trade in enumerate(raw_trades):
            # Find position before and after
            trade_step = trade.get('step', 0)
            
            # Skip if trade step is out of bounds
            if trade_step >= results_length:
                continue
                
            # Get positions safely
            position_before = 0
            position_after = 0
            
            if trade_step > 0 and trade_step - 1 < results_length:
                position_before = results_df.iloc[trade_step - 1]['position']
            
            if trade_step < results_length:
                position_after = results_df.iloc[trade_step]['position']
            
            # Get timestamp
            timestamp = results_df.iloc[trade_step]['date']
            
            # Create trade record
            processed_trades.append(TradeRecord(
                timestamp=timestamp,
                action=trade.get('action', 'UNKNOWN'),
                price=trade.get('price', 0),
                size=trade.get('size', 0),
                position_before=position_before,
                position_after=position_after,
                portfolio_value=trade.get('portfolio_value', 0),
                cost=trade.get('cost', 0)
            ))
        
        return processed_trades
    
    def _aggregate_monte_carlo_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple Monte Carlo runs."""
        # Aggregate portfolio values
        all_portfolio_values = [r['data']['portfolio_value'] for r in all_results]
        mean_portfolio_values = pd.concat(all_portfolio_values, axis=1).mean(axis=1)
        std_portfolio_values = pd.concat(all_portfolio_values, axis=1).std(axis=1)
        
        # Create aggregated results
        aggregated_df = all_results[0]['data'].copy()
        aggregated_df['portfolio_value'] = mean_portfolio_values
        aggregated_df['portfolio_value_std'] = std_portfolio_values
        aggregated_df['returns'] = aggregated_df['portfolio_value'].pct_change()
        aggregated_df['cumulative_returns'] = (1 + aggregated_df['returns']).cumprod() - 1
        
        # Aggregate trades
        all_trades = []
        for r in all_results:
            all_trades.extend(r['trades'])
        
        return {
            'data': aggregated_df,
            'trades': all_trades,
            'n_runs': len(all_results),
            'all_results': all_results
        }
    
    def _calculate_comprehensive_metrics(self, 
                                       results_df: pd.DataFrame, 
                                       strategy_name: str,
                                       trades: List[TradeRecord] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results_df: DataFrame with portfolio values and returns
            strategy_name: Name of the strategy
            trades: List of trade records
            
        Returns:
            Dictionary of performance metrics
        """
        # Filter out NaN values
        returns = results_df['returns'].dropna()
        log_returns = results_df['log_returns'].dropna()
        
        # Basic return metrics
        total_return = (results_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        n_days = len(results_df)
        
        # For 5-minute data, convert to years properly
        # Assuming 6.5 hours of trading per day, 78 bars per day
        bars_per_day = 78  # 6.5 hours * 12 five-minute bars
        bars_per_year = bars_per_day * 252
        n_years = n_days / bars_per_year
        
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics - adjust for 5-minute data
        volatility = returns.std() * np.sqrt(bars_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(bars_per_year) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted returns
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration (in bars)
        drawdown_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_dd_duration = i - drawdown_start
            else:
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                drawdown_start = None
                current_dd_duration = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_percentile = (1 - self.confidence_level) * 100
        var_daily = np.percentile(returns, var_percentile)
        var_annual = var_daily * np.sqrt(bars_per_year)
        
        cvar_daily = returns[returns <= var_daily].mean() if len(returns[returns <= var_daily]) > 0 else var_daily
        cvar_annual = cvar_daily * np.sqrt(bars_per_year)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Trade analysis - FIXED for 5-minute data
        if trades:
            trade_metrics = self._analyze_trades_5min(trades, bars_per_day, bars_per_year)
        else:
            trade_metrics = {
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_period': 0,
                'trades_per_year': 0
            }
        
        # Information ratio (if we have a benchmark)
        if 'buy_hold' in self.results and strategy_name != 'Buy & Hold':
            benchmark_returns = self.results['buy_hold']['data']['returns'].dropna()
            active_returns = returns - benchmark_returns.reindex(returns.index, fill_value=0)
            tracking_error = active_returns.std() * np.sqrt(bars_per_year)
            information_ratio = (active_returns.mean() * bars_per_year) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        # Time in market
        positions = results_df.get('position', pd.Series(1, index=results_df.index))
        time_in_market = (positions > 0).mean()
        
        # Stability metrics
        cumulative_returns = results_df['cumulative_returns']
        if len(cumulative_returns) > 2:
            # Fit linear regression to log cumulative returns
            x = np.arange(len(cumulative_returns))
            y = np.log1p(cumulative_returns.fillna(0))
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            stability = r_value ** 2  # R-squared as stability metric
        else:
            stability = 0
        
        # Compile all metrics
        metrics = {
            'strategy': strategy_name,
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0,
            
            # Risk metrics
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'var_95': var_annual,
            'cvar_95': cvar_annual,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            'stability': stability,
            
            # Trading metrics
            'time_in_market': time_in_market,
            'final_value': results_df['portfolio_value'].iloc[-1],
            
            # Trade metrics
            **trade_metrics
        }
        
        return metrics
    
    def _analyze_trades_5min(self, trades: List[TradeRecord], bars_per_day: int, bars_per_year: int) -> Dict[str, float]:
        """Analyze trade records for 5-minute data."""
        if not trades:
            return {}
        
        # Separate buys and sells
        buys = [t for t in trades if t.action == 'BUY']
        sells = [t for t in trades if t.action == 'SELL']
        
        # Calculate roundtrips (complete trades)
        roundtrips = []
        buy_stack = []
        
        for trade in trades:
            if trade.action == 'BUY':
                buy_stack.append(trade)
            elif trade.action == 'SELL' and buy_stack:
                buy_trade = buy_stack.pop(0)  # FIFO matching
                sell_trade = trade
                
                # Calculate holding period in bars (5-minute intervals)
                time_diff = sell_trade.timestamp - buy_trade.timestamp
                holding_period_bars = int(time_diff.total_seconds() / 300)  # 300 seconds = 5 minutes
                
                trade_return = (sell_trade.price - buy_trade.price) / buy_trade.price
                trade_pnl = (sell_trade.price - buy_trade.price) * buy_trade.size
                
                roundtrips.append({
                    'buy_date': buy_trade.timestamp,
                    'sell_date': sell_trade.timestamp,
                    'holding_period': holding_period_bars,
                    'return': trade_return,
                    'pnl': trade_pnl
                })
        
        # Calculate metrics from roundtrips
        if roundtrips:
            returns = [rt['return'] for rt in roundtrips]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]
            
            win_rate = len(wins) / len(returns) if returns else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average holding period in bars
            avg_holding_period = np.mean([rt['holding_period'] for rt in roundtrips])
            
            # Payoff ratio
            payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Expected return per trade
            expected_return = win_rate * avg_win + (1 - win_rate) * avg_loss
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_holding_period = payoff_ratio = expected_return = 0
        
        # Trades per year
        if len(trades) > 1:
            time_span = trades[-1].timestamp - trades[0].timestamp
            total_bars = time_span.total_seconds() / 300  # Convert to 5-minute bars
            trades_per_year = len(trades) * bars_per_year / total_bars if total_bars > 0 else 0
        else:
            trades_per_year = 0
        
        return {
            'num_trades': len(trades),
            'num_roundtrips': len(roundtrips),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'payoff_ratio': payoff_ratio,
            'expected_return': expected_return,
            'avg_holding_period': avg_holding_period,  # In 5-minute bars
            'trades_per_year': trades_per_year
        }
    
    def _analyze_trades(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Original trade analysis method - redirect to 5-minute version."""
        bars_per_day = 78
        bars_per_year = bars_per_day * 252
        return self._analyze_trades_5min(trades, bars_per_day, bars_per_year)
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all backtested strategies with comprehensive metrics.
        
        Returns:
            DataFrame with comparison of all strategies
        """
        comparison = []
        
        for strategy_key, results in self.results.items():
            if 'data' in results:
                metrics = self._calculate_comprehensive_metrics(
                    results['data'], 
                    strategy_key.replace('_', ' ').title(),
                    results.get('trades', [])
                )
                comparison.append(metrics)
        
        comparison_df = pd.DataFrame(comparison).set_index('strategy')
        
        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Create enhanced visualization of backtest results.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :2])
        for strategy_key, results in self.results.items():
            if 'data' in results:
                data = results['data']
                label = strategy_key.replace('_', ' ').title()
                ax1.plot(data['date'], data['cumulative_returns'] * 100, 
                        label=label, linewidth=2)
        
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Sharpe Ratio
        ax2 = fig.add_subplot(gs[0, 2])
        for strategy_key, results in self.results.items():
            if 'data' in results:
                data = results['data']
                returns = data['returns'].dropna()
                rolling_sharpe = returns.rolling(60).apply(
                    lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
                )
                label = strategy_key.replace('_', ' ').title()
                ax2.plot(data['date'][1:], rolling_sharpe, label=label, alpha=0.7)
        
        ax2.set_title('Rolling Sharpe Ratio (60-day)', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Underwater Plot (Drawdowns)
        ax3 = fig.add_subplot(gs[1, :])
        for strategy_key, results in self.results.items():
            if 'data' in results:
                data = results['data']
                returns = data['returns'].dropna()
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                label = strategy_key.replace('_', ' ').title()
                ax3.fill_between(data['date'][1:], drawdown * 100, 0, 
                               alpha=0.3, label=label)
        
        ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        for strategy_key, results in self.results.items():
            if 'data' in results:
                data = results['data']
                returns = data['returns'].dropna() * 100
                label = strategy_key.replace('_', ' ').title()
                ax4.hist(returns, bins=50, alpha=0.5, label=label, density=True)
        
        ax4.set_title('Returns Distribution', fontsize=12)
        ax4.set_xlabel('Daily Return (%)')
        ax4.set_ylabel('Density')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Risk-Return Scatter
        ax5 = fig.add_subplot(gs[2, 1])
        comparison = self.compare_strategies()
        
        for strategy in comparison.index:
            metrics = comparison.loc[strategy]
            ax5.scatter(metrics['volatility'] * 100, 
                       metrics['annualized_return'] * 100,
                       s=200, alpha=0.7, label=strategy)
            ax5.annotate(strategy, 
                        (metrics['volatility'] * 100, metrics['annualized_return'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add efficient frontier indication
        ax5.plot([0, comparison['volatility'].max() * 100], 
                [self.risk_free_rate * 100, 
                 self.risk_free_rate * 100 + comparison['sharpe_ratio'].max() * comparison['volatility'].max() * 100],
                'k--', alpha=0.5, label='Efficient Frontier')
        
        ax5.set_title('Risk-Return Profile', fontsize=12)
        ax5.set_xlabel('Volatility (%)')
        ax5.set_ylabel('Annual Return (%)')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance Metrics Heatmap
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Select key metrics for heatmap
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
                          'win_rate', 'profit_factor', 'stability']
        heatmap_data = comparison[metrics_to_plot].T
        
        # Normalize metrics to [0, 1] for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min(axis=1).values.reshape(-1, 1)) / \
                            (heatmap_data.max(axis=1) - heatmap_data.min(axis=1)).values.reshape(-1, 1)
        
        im = ax6.imshow(heatmap_normalized, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax6.set_xticks(np.arange(len(heatmap_data.columns)))
        ax6.set_yticks(np.arange(len(heatmap_data.index)))
        ax6.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax6.set_yticklabels(heatmap_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Normalized Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                text = ax6.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        ax6.set_title('Performance Metrics Comparison', fontsize=12)
        
        # Add overall title
        fig.suptitle('Comprehensive Backtest Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: str) -> None:
        """
        Generate a comprehensive backtest report with enhanced analytics.
        
        Args:
            save_path: Path to save the report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'configuration': {
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage,
                'risk_free_rate': self.risk_free_rate,
                'confidence_level': self.confidence_level
            },
            'results': {}
        }
        
        # Add detailed results for each strategy
        for strategy_key, results in self.results.items():
            if 'data' in results:
                metrics = self._calculate_comprehensive_metrics(
                    results['data'],
                    strategy_key.replace('_', ' ').title(),
                    results.get('trades', [])
                )
                
                # Add additional analytics
                strategy_report = {
                    'metrics': metrics,
                    'summary_statistics': {
                        'start_date': str(results['data']['date'].iloc[0]),
                        'end_date': str(results['data']['date'].iloc[-1]),
                        'trading_days': len(results['data']),
                        'best_day': {
                            'date': str(results['data'].loc[results['data']['returns'].idxmax(), 'date']),
                            'return': float(results['data']['returns'].max())
                        },
                        'worst_day': {
                            'date': str(results['data'].loc[results['data']['returns'].idxmin(), 'date']),
                            'return': float(results['data']['returns'].min())
                        }
                    }
                }
                
                # Add trade analysis if available
                if 'trades' in results and results['trades']:
                    strategy_report['trade_analysis'] = self._generate_trade_report(results['trades'])
                
                # Add Monte Carlo results if available
                if 'n_runs' in results:
                    strategy_report['monte_carlo'] = {
                        'n_runs': results['n_runs'],
                        'return_confidence_interval': self._calculate_confidence_interval(results)
                    }
                
                report['results'][strategy_key] = strategy_report
        
        # Add strategy comparison
        comparison = self.compare_strategies()
        report['comparison'] = comparison.to_dict()
        
        # Add correlation analysis
        report['correlation_analysis'] = self._calculate_strategy_correlations()
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Enhanced report saved to {save_path}")
    
    def _generate_trade_report(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Generate detailed trade report."""
        if not trades:
            return {}
        
        trade_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'action': t.action,
            'price': t.price,
            'size': t.size,
            'cost': t.cost,
            'position_after': t.position_after
        } for t in trades])
        
        # Group by action type
        action_summary = trade_df.groupby('action').agg({
            'price': ['count', 'mean'],
            'size': 'mean',
            'cost': 'sum'
        })
        
        return {
            'total_trades': len(trades),
            'first_trade': str(trades[0].timestamp),
            'last_trade': str(trades[-1].timestamp),
            'action_summary': action_summary.to_dict(),
            'total_costs': float(trade_df['cost'].sum()),
            'avg_trade_size': float(trade_df['size'].mean())
        }
    
    def _calculate_confidence_interval(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence intervals for Monte Carlo results."""
        if 'all_results' not in results:
            return {}
        
        final_returns = [r['data']['cumulative_returns'].iloc[-1] 
                        for r in results['all_results']]
        
        return {
            'mean': float(np.mean(final_returns)),
            'std': float(np.std(final_returns)),
            'lower_95': float(np.percentile(final_returns, 2.5)),
            'upper_95': float(np.percentile(final_returns, 97.5))
        }
    
    def _calculate_strategy_correlations(self) -> Dict[str, Any]:
        """Calculate correlations between strategy returns."""
        returns_dict = {}
        
        for strategy_key, results in self.results.items():
            if 'data' in results:
                returns_dict[strategy_key] = results['data']['returns']
        
        if len(returns_dict) < 2:
            return {}
        
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'average_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        }
    
    def calculate_alpha_beta(self, 
                           strategy_returns: pd.Series, 
                           benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate alpha and beta relative to benchmark using CAPM.
        
        Args:
            strategy_returns: Returns of the strategy
            benchmark_returns: Returns of the benchmark
            
        Returns:
            Tuple of (alpha, beta)
        """
        # Align series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 20:  # Need sufficient data
            return 0.0, 1.0
        
        # Calculate excess returns
        strategy_excess = aligned['strategy'] - self.risk_free_rate / 252
        benchmark_excess = aligned['benchmark'] - self.risk_free_rate / 252
        
        # Calculate beta using covariance
        covariance = np.cov(strategy_excess, benchmark_excess)[0, 1]
        benchmark_variance = np.var(benchmark_excess)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha
        alpha = strategy_excess.mean() - beta * benchmark_excess.mean()
        alpha_annualized = alpha * 252
        
        return alpha_annualized, beta
    
    def run_monte_carlo_analysis(self, 
                               strategy_data: pd.DataFrame,
                               n_simulations: int = 1000,
                               forecast_days: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for future performance estimation.
        
        Args:
            strategy_data: Historical strategy data
            n_simulations: Number of Monte Carlo simulations
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        # Calculate historical parameters
        returns = strategy_data['returns'].dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        # Run simulations
        final_values = []
        paths = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(mu, sigma, forecast_days)
            
            # Calculate price path
            price_path = [strategy_data['portfolio_value'].iloc[-1]]
            for r in random_returns:
                price_path.append(price_path[-1] * (1 + r))
            
            paths.append(price_path)
            final_values.append(price_path[-1])
        
        # Calculate statistics
        final_values = np.array(final_values)
        
        return {
            'expected_value': float(np.mean(final_values)),
            'std_dev': float(np.std(final_values)),
            'percentiles': {
                '5th': float(np.percentile(final_values, 5)),
                '25th': float(np.percentile(final_values, 25)),
                '50th': float(np.percentile(final_values, 50)),
                '75th': float(np.percentile(final_values, 75)),
                '95th': float(np.percentile(final_values, 95))
            },
            'probability_of_profit': float((final_values > self.initial_capital).mean()),
            'expected_return': float((np.mean(final_values) - self.initial_capital) / self.initial_capital),
            'var_95': float(self.initial_capital - np.percentile(final_values, 5)),
            'paths_sample': paths[:10]  # Store sample paths for visualization
        }


# Example usage
if __name__ == "__main__":
    # Create dummy test data
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
    
    test_data = pd.DataFrame({
        'Close': prices,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    # Initialize enhanced backtester
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Run backtests
    bh_metrics = backtester.backtest_buy_hold(test_data)
    sma_metrics = backtester.backtest_sma_crossover(test_data)
    momentum_metrics = backtester.backtest_momentum(test_data)
    
    # Compare strategies
    comparison = backtester.compare_strategies()
    print("\nEnhanced Strategy Comparison:")
    print(comparison)
    
    # Generate visualizations
    backtester.plot_results()
    
    # Generate report
    backtester.generate_report('enhanced_backtest_report.json')
    
    # Run Monte Carlo analysis
    if 'buy_hold' in backtester.results:
        mc_results = backtester.run_monte_carlo_analysis(
            backtester.results['buy_hold']['data'],
            n_simulations=1000
        )
        print("\nMonte Carlo Analysis:")
        print(f"Expected value in 1 year: ${mc_results['expected_value']:,.2f}")
        print(f"Probability of profit: {mc_results['probability_of_profit']:.1%}")