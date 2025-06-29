"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""

import math
import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path
from .impact_models import ACImpactModel
from typing import Dict, List, Optional, Tuple

matplotlib.use("Agg")
from stable_baselines3.common.vec_env import DummyVecEnv
import quantstats as qs


class PortfolioOptimizationImpactEnv(gym.Env):
    """
    A portfolio optimization environment that incorporates a market impact model.
    The agent learns to allocate capital across a portfolio of assets, where
    rebalancing actions incur costs based on a realistic market impact model.
    This environment is based on the Portfolio Optimization Environment (POE)
    proposed by Costa and Costa (2023).
    Reference:
        - Costa, C., & Costa, A. (2023). POE: A General Portfolio
          Optimization Environment for FinRL. In Anais do II Brazilian
          Workshop on Artificial Intelligence in Finance (pp. 132–143). SBC.
          https://doi.org/10.5753/bwaif.2023.231144
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float,
        order_df: bool = True,
        normalize_df: str = "by_previous_time",
        reward_scaling: float = 1.0,
        features: List[str] = ["close", "high", "low"],
        valuation_feature: str = "close",
        time_column: str = "date",
        tic_column: str = "tic",
        time_window: int = 1,
        cwd: str = "./",
        new_gym_api: bool = False,
        impact_Y: float = 0.6,
        impact_perm_fraction: float = 0.25,
    ) -> None:
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._reward_scaling = reward_scaling
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api

        self._results_file = self._cwd / "results" / "rl_impact"
        self._results_file.mkdir(parents=True, exist_ok=True)

        self._df_price_variation: Optional[pd.DataFrame] = None
        self._preprocess_data(order_df, normalize_df)

        self._tic_list = self._df[self._tic_column].unique()
        self._stock_dim = len(self._tic_list)
        self._stock_symbols = [f"STOCK_{i}" for i in range(self._stock_dim)]
        action_space_dim = 1 + self._stock_dim  # Cash + Stocks

        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        self.action_space = spaces.Box(low=0, high=1, shape=(action_space_dim,))
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self._features), self._stock_dim, self._time_window),
            ),
            "last_action": spaces.Box(low=0, high=1, shape=(action_space_dim,)),
            "permanent_impact": spaces.Box(low=-np.inf, high=np.inf, shape=(self._stock_dim,))
        })

        self.impact_model = ACImpactModel(Y=impact_Y, perm_fraction=impact_perm_fraction)
        
        self.reset()

    def step(self, actions: np.ndarray) -> Tuple:
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            self._generate_performance_report()
            info = self._get_info()
            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, info
            return self._state, self._reward, self._terminal, info

        weights = self._softmax_normalization(actions)
        self._actions_memory.append(weights)
        last_weights = self._final_weights[-1]

        self._time_index += 1
        self._update_state()
        
        portfolio_pre_rebalance = self._portfolio_value * (last_weights * self._price_variation)
        value_pre_rebalance = np.sum(portfolio_pre_rebalance)
        weights_pre_rebalance = portfolio_pre_rebalance / value_pre_rebalance if value_pre_rebalance != 0 else last_weights

        current_prices = self._info["current_prices"]
        trades_in_value = value_pre_rebalance * (weights - weights_pre_rebalance)
        trades_in_shares = np.divide(trades_in_value[1:], current_prices, out=np.zeros_like(trades_in_value[1:]), where=current_prices!=0)

        total_impact_cost, price_shifts = self._calculate_impact(trades_in_shares, current_prices)
        self._permanent_impact_per_stock += price_shifts

        self._asset_memory["initial"].append(value_pre_rebalance)
        self._portfolio_value = value_pre_rebalance - total_impact_cost
        
        final_portfolio_distribution = self._portfolio_value * weights
        self._final_weights.append(final_portfolio_distribution / self._portfolio_value if self._portfolio_value !=0 else weights)
        self._asset_memory["final"].append(self._portfolio_value)

        self._calculate_reward()

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def reset(self) -> Tuple[Dict, Dict]:
        self._time_index = self._time_window - 1
        self._reset_memory()
        self._portfolio_value = self._initial_amount
        self._terminal = False
        self.impact_model.reset_impact_history()
        self._permanent_impact_per_stock.fill(0)
        
        self._update_state()

        if self._new_gym_api:
            return self._state, self._get_info()
        return self._state

    def _update_state(self) -> None:
        """Updates the environment state and info for the current time index."""
        end_time = self._sorted_times[self._time_index]
        start_time = self._sorted_times[self._time_index - self._time_window + 1]

        data_window = self._df[(self._df[self._time_column] >= start_time) & (self._df[self._time_column] <= end_time)]
        self._data = data_window[[self._time_column, self._tic_column] + self._features]
        
        current_data = self._df[self._df[self._time_column] == end_time]

        self._price_variation = self._df_price_variation[self._df_price_variation[self._time_column] == end_time][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        state = np.array([
            self._data[self._data[self._tic_column] == tic][self._features].values.T
            for tic in self._tic_list
        ]).transpose(1, 0, 2)
        
        self._info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "end_time": end_time,
            "price_variation": self._price_variation,
            "current_prices": current_data[self._valuation_feature].values,
            "current_volatility": current_data["volatility"].values,
            "current_volume": current_data["volume"].values,
        }
        self._state = self._standardize_state(state)

    def _calculate_impact(self, trades_in_shares: np.ndarray, current_prices: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculates the total impact cost and price shifts for a set of trades."""
        total_impact_cost = 0.0
        price_shifts = np.zeros(self._stock_dim)
        
        vol = self._info["current_volatility"]
        volm = self._info["current_volume"]

        for i in range(self.stock_dim):
            if not math.isclose(trades_in_shares[i], 0, abs_tol=1e-6):
                impact = self.impact_model.apply_trade(
                    trades_in_shares[i], current_prices[i], vol[i], volm[i], self._stock_symbols[i]
                )
                total_impact_cost += impact.cost
                price_shifts[i] = impact.price_shift
        return total_impact_cost, price_shifts

    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        last_value = self._asset_memory["final"][-2]
        current_value = self._asset_memory["final"][-1]
        
        rate_of_return = current_value / last_value if last_value != 0 else 1.0
        portfolio_return = rate_of_return - 1
        log_return = np.log(rate_of_return)

        self._portfolio_return_memory.append(portfolio_return)
        self._portfolio_reward_memory.append(log_return)
        self._reward = log_return * self._reward_scaling

    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """Normalizes the action vector using the softmax function."""
        exp_actions = np.exp(actions - np.max(actions))
        return exp_actions / np.sum(exp_actions)

    def _preprocess_data(self, order: bool, normalize: str) -> None:
        """Preprocesses the input dataframe."""
        for f in ["volatility", "volume"]:
            if f not in self._features and f in self._df.columns:
                 self._features.append(f)

        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        
        self._df_price_variation = self._temporal_variation_df()
        
        if normalize:
            self._normalize_dataframe(normalize)

        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        if self._df_price_variation is not None:
            self._df_price_variation[self._time_column] = pd.to_datetime(self._df_price_variation[self._time_column])
        
        self._df[self._features] = self._df[self._features].astype("float32")

    def _reset_memory(self) -> None:
        """Resets the environment's internal memory."""
        date_time = self._sorted_times[self._time_index]
        self._asset_memory = {"initial": [self._initial_amount], "final": [self._initial_amount]}
        self._portfolio_return_memory = [0.0]
        self._portfolio_reward_memory = [0.0]
        self._actions_memory = [np.array([1.0] + [0.0] * self._stock_dim, dtype=np.float32)]
        self._final_weights = [np.array([1.0] + [0.0] * self._stock_dim, dtype=np.float32)]
        self._date_memory = [date_time]
        self._permanent_impact_per_stock = np.zeros(self._stock_dim, dtype=np.float32)

    def _standardize_state(self, state: np.ndarray) -> Dict:
        """Standardizes the state into the dictionary format for the observation space."""
        return {
            "state": state.astype(np.float32),
            "last_action": self._actions_memory[-1].astype(np.float32),
            "permanent_impact": self._permanent_impact_per_stock.astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        return self._info

    def _temporal_variation_df(self, periods: int = 1) -> pd.DataFrame:
        """Calculates the temporal variation of the valuation feature."""
        df_temp = self._df.copy()
        feature = self._valuation_feature
        prev_col = f"prev_{feature}"
        df_temp[prev_col] = df_temp.groupby(self._tic_column)[feature].shift(periods)
        df_temp[feature] = df_temp[feature] / df_temp[prev_col]
        return df_temp.drop(columns=[prev_col]).fillna(1).reset_index(drop=True)

    def _normalize_dataframe(self, method: str) -> None:
        """Normalizes features in the dataframe."""
        if method == "by_previous_time":
            temp_df = self._df.copy()
            for feature in self._features:
                if feature in temp_df.columns:
                    prev_col = f"prev_{feature}"
                    temp_df[prev_col] = temp_df.groupby(self._tic_column)[feature].shift(1)
                    temp_df[feature] /= temp_df[prev_col]
                    temp_df = temp_df.drop(columns=[prev_col])
            self._df = temp_df.fillna(1.0)
        else:
            print(f"Normalization method '{method}' not implemented. Skipping.")
    
    def _generate_performance_report(self) -> None:
        """Calculates and saves performance metrics and plots."""
        metrics_df = pd.DataFrame({
            "date": self._date_memory[1:], # Skip initial state
            "returns": self._portfolio_return_memory[1:],
            "rewards": self._portfolio_reward_memory[1:],
            "portfolio_values": self._asset_memory["final"][1:],
        }).set_index("date")

        if not metrics_df.empty:
            print("=" * 50)
            print(f"Initial portfolio value: {self._asset_memory['final'][0]:,.2f}")
            print(f"Final portfolio value: {self._portfolio_value:,.2f}")
            print(f"Final portfolio return: {self._portfolio_value / self._asset_memory['final'][0] - 1:.2%}")
            try:
                print(f"Max DrawDown: {qs.stats.max_drawdown(metrics_df['portfolio_values']):.2%}")
                print(f"Sharpe ratio: {qs.stats.sharpe(metrics_df['returns']):.3f}")
            except Exception as e:
                print(f"Could not calculate QuantStats metrics: {e}")
            print("=" * 50)
            
            qs.plots.snapshot(
                metrics_df["returns"],
                show=False,
                savefig=self._results_file / "portfolio_summary.png",
            )

    def get_sb_env(self, env_number: int = 1) -> Tuple[DummyVecEnv, Dict]:
        """Generates an environment compatible with Stable Baselines 3. The
        generated environment is a vectorized version of the current one.

        Returns:
            A tuple with the generated environment and an initial observation.
        """
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs

    @property
    def results_file(self):
        return self._results_file
