from datetime import date
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy import random as rd

from .impact_models import ACImpactModel


class StockTradingEnvImpact(gym.Env):
    """
    A stock trading environment that incorporates a market impact model.
    This environment simulates single-stock trading, where the agent's actions
    (buying or selling shares) incur costs based on a realistic market
    impact model, affecting the portfolio's value.
    Attributes:
        observation_space (gym.spaces.Box): The observation space.
        action_space (gym.spaces.Box): The action space.
        stock_dim (int): The number of stocks in the environment.
    """

    def __init__(
        self,
        config: Dict,
        initial_capital: float = 1e6,
        gamma: float = 0.99,
        turbulence_thresh: float = 99,
        min_stock_rate: float = 0.1,
        max_stock_pct: float = 0.02,
        reward_scaling: float = 2**-11,
        initial_stocks: Optional[np.ndarray] = None,
        impact_Y: float = 0.6,
        impact_perm_fraction: float = 0.25,
    ) -> None:
        """
        Initializes the StockTradingEnvImpact.
        Args:
            config: A dictionary containing the market data arrays.
                Expected keys: "price_array", "tech_array", "turbulence_array",
                "volatility_array", "volume_array".
            initial_capital: The initial cash balance.
            gamma: The discount factor for future rewards.
            turbulence_thresh: The threshold for market turbulence.
            min_stock_rate: The minimum trade size relative to max position.
            max_stock_pct: The maximum percentage of portfolio value for one stock.
            reward_scaling: Scaling factor for the reward.
            initial_stocks: Initial holdings of stocks.
            impact_Y: The 'Y' parameter for the ACImpactModel.
            impact_perm_fraction: The permanent fraction for the ACImpactModel.
        """
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        turbulence_array = config["turbulence_array"]
        if_train = config.get("if_train", True)

        self.volatility_array = config.get(
            "volatility_array", np.ones_like(price_array) * 0.02
        )
        self.volume_array = config.get(
            "volume_array", np.ones_like(price_array) * 1e6
        )

        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32) * (2**-7)
        self.turbulence_bool = (turbulence_array > turbulence_thresh).astype(np.float32)
        self.turbulence_array = (
            self.sigmoid_sign(turbulence_array, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        self.stock_dim = self.price_array.shape[1]
        self.gamma = gamma
        self.max_stock_pct = max_stock_pct
        self.min_stock_rate = min_stock_rate
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(self.stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )
        self.if_train = if_train

        self.impact_model = ACImpactModel(
            Y=impact_Y, perm_fraction=impact_perm_fraction
        )
        self.stock_symbols = [f"STOCK_{i}" for i in range(self.stock_dim)]

        # State: [cash, turbulence, turbulence_bool, price, stocks, perm_impact, cooldown, tech]
        self.state_dim = 1 + 2 + (4 * self.stock_dim) + self.tech_array.shape[1]
        self.action_dim = self.stock_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self.max_step = self.price_array.shape[0] - 1
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment to its initial state."""
        self.time = 0
        price = self.price_array[self.time]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.cash = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.cash = self.initial_capital

        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.permanent_impact_per_stock = np.zeros(self.stock_dim, dtype=np.float32)
        self.impact_model.reset_impact_history()

        self.total_asset = self.cash + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.episode_return = 0.0

        return self.get_state(price)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        Args:
            actions: An action provided by the agent.
        Returns:
            A tuple containing the new state, reward, done flag, and info dict.
        """
        self.time += 1
        done = self.time >= self.max_step

        price = self.price_array[self.time - 1]
        volatility = self.volatility_array[self.time - 1]
        volume = self.volume_array[self.time - 1]

        max_stocks_per_position = self._calculate_max_stock_per_position(price)
        trade_shares = (actions * max_stocks_per_position).astype(int)

        self.stocks_cool_down += 1

        if self.turbulence_bool[self.time - 1] == 0:
            min_action = (
                int(np.mean(max_stocks_per_position) * self.min_stock_rate)
                if np.mean(max_stocks_per_position) > 0
                else 0
            )

            for index in np.where(trade_shares < -min_action)[0]:
                self._sell_stock(index, -trade_shares[index], price, volatility, volume)

            for index in np.where(trade_shares > min_action)[0]:
                self._buy_stock(index, trade_shares[index], price, volatility, volume)
        else:
            self._liquidate_positions(price, volatility, volume)

        new_price = self.price_array[self.time]
        state = self.get_state(new_price)

        end_total_asset = self._calculate_total_asset(new_price)
        reward = (end_total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = end_total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        if done:
            reward = self.gamma_reward
            self.episode_return = self.total_asset / self.initial_capital
            self.impact_model.end_day(date.today())

        return state, reward, done, {}

    def _sell_stock(
        self, index: int, sell_shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray
    ) -> None:
        """Executes a sell trade."""
        if price[index] > 0 and sell_shares > 0:
            sell_num_shares = min(self.stocks[index], sell_shares)
            if sell_num_shares > 0:
                impact_result = self.impact_model.apply_trade(
                    -sell_num_shares,
                    price[index],
                    volatility[index],
                    volume[index],
                    self.stock_symbols[index],
                )
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares - impact_result.cost
                self.permanent_impact_per_stock[index] += impact_result.price_shift
                self.stocks_cool_down[index] = 0

    def _buy_stock(
        self, index: int, buy_shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray
    ) -> None:
        """Executes a buy trade."""
        if price[index] > 0 and buy_shares > 0:
            impact_result = self.impact_model.apply_trade(
                buy_shares,
                price[index],
                volatility[index],
                volume[index],
                self.stock_symbols[index],
            )
            total_cost = price[index] * buy_shares + impact_result.cost
            if total_cost <= self.cash:
                self.stocks[index] += buy_shares
                self.cash -= total_cost
                self.permanent_impact_per_stock[index] += impact_result.price_shift
                self.stocks_cool_down[index] = 0

    def _liquidate_positions(self, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Sells all held assets."""
        for index in range(self.stock_dim):
            if self.stocks[index] > 0:
                self._sell_stock(index, self.stocks[index], price, volatility, volume)
        self.stocks_cool_down[:] = 0

    def _calculate_total_asset(self, price: np.ndarray) -> float:
        """Calculates the total portfolio value."""
        adjusted_prices = price + self.permanent_impact_per_stock
        return self.cash + (self.stocks * adjusted_prices).sum()

    def _calculate_max_stock_per_position(self, current_prices: np.ndarray) -> np.ndarray:
        """Calculates the max number of shares per stock based on portfolio percentage."""
        portfolio_value = self._calculate_total_asset(current_prices)
        max_position_value = portfolio_value * self.max_stock_pct
        return np.where(current_prices > 0, max_position_value / current_prices, 0).astype(int)

    def get_state(self, price: np.ndarray) -> np.ndarray:
        """Returns the current state of the environment."""
        scale = 2**-12
        state_components = [
            np.array([self.cash]) * scale,
            self.turbulence_array[self.time],
            self.turbulence_bool[self.time],
            price * scale,
            self.stocks * scale,
            self.permanent_impact_per_stock * scale,
            self.stocks_cool_down,
            self.tech_array[self.time],
        ]
        return np.hstack(state_components).astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary: np.ndarray, thresh: float) -> np.ndarray:
        """A sigmoid function to bound the turbulence values."""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        return sigmoid(ary / thresh) * thresh
