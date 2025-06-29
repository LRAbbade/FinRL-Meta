"""
Margin Trading Environment with Market Impact
This environment adapts the "Margin Trader" paper by Gu et al. (2023) to the
FinRL-Meta framework, incorporating a realistic market impact model.
The agent learns to manage a portfolio with both long and short positions,
handle leverage, and adhere to margin constraints, all while considering
the execution costs of its trades.
Reference:
Gu, J., Du, W., Rahman, A. M. M., & Wang, G. (2023).
Margin Trader: A Reinforcement Learning Framework for Portfolio Management with Margin and Constraints.
In Proceedings of the Fourth ACM International Conference on AI in Finance (pp. 610–618).
"""
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np

from .impact_models import ACImpactModel


class MarginTraderImpactEnv(gym.Env):
    """
    A margin trading environment that incorporates a market impact model.
    This environment simulates trading with leverage, where the agent can
    hold both long and short positions and must manage margin requirements.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        initial_capital: float = 1e6,
        gamma: float = 0.99,
        turbulence_thresh: float = 99,
        max_stock_pct: float = 0.02,
        reward_scaling: float = 2**-11,
        initial_stocks: Optional[np.ndarray] = None,
        margin_rate: float = 2.0,
        maintenance_margin: float = 0.3, # Common retail maintenance margin
        impact_Y: float = 0.6,
        impact_perm_fraction: float = 0.25,
    ) -> None:
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        turbulence_array = config["turbulence_array"]
        self.volatility_array = config.get("volatility_array", np.ones_like(price_array) * 0.02)
        self.volume_array = config.get("volume_array", np.ones_like(price_array) * 1e6)
        self.if_train = config.get("if_train", True)

        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32) * 2**-7
        self.turbulence_bool = (turbulence_array > turbulence_thresh).astype(np.float32)
        self.turbulence_array = (self.sigmoid_sign(turbulence_array, turbulence_thresh) * 2**-5).astype(np.float32)

        self.stock_dim = self.price_array.shape[1]
        self.stock_symbols = [f"STOCK_{i}" for i in range(self.stock_dim)]
        self.gamma = gamma
        self.max_stock_pct = max_stock_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(self.stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        self.margin_rate = margin_rate
        self.maintenance_margin = maintenance_margin
        self.impact_model = ACImpactModel(Y=impact_Y, perm_fraction=impact_perm_fraction)

        # State: [long_equity, loan, short_equity, short_credit_balance, turbulence, turbulence_bool, prices, holdings, perm_impact, cooldown, tech]
        self.state_dim = 4 + 2 + (4 * self.stock_dim) + self.tech_array.shape[1]
        self.action_dim = self.stock_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.max_step = self.price_array.shape[0] - 1
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment to its initial state."""
        self.time = 0
        price = self.price_array[self.time]

        self.long_equity = self.initial_capital / 2
        self.loan = self.long_equity * (self.margin_rate - 1)
        self.short_equity = self.initial_capital / 2
        self.short_credit_balance = 0.0

        self.stocks = self.initial_stocks.copy() # Positive for long, negative for short
        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.permanent_impact_per_stock = np.zeros(self.stock_dim, dtype=np.float32)
        self.impact_model.reset_impact_history()

        self.total_asset = self._calculate_total_equity(price)
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.episode_return = 0.0
        
        return self.get_state(price)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        self.time += 1
        done = self.time >= self.max_step
        
        price = self.price_array[self.time - 1]
        volatility = self.volatility_array[self.time - 1]
        volume = self.volume_array[self.time - 1]

        total_equity = self._calculate_total_equity(price)
        max_position_value = total_equity * self.max_stock_pct
        max_stocks_per_position = np.where(price > 0, max_position_value / price, 0).astype(int)
        trade_shares = (actions * max_stocks_per_position).astype(int)

        for i in range(self.stock_dim):
            self._execute_trade(i, trade_shares[i], price, volatility, volume)

        new_price = self.price_array[self.time]
        self._update_equities(new_price)
        self._handle_margin_calls(new_price, volatility, volume)
        
        current_total_equity = self._calculate_total_equity(new_price)
        reward = (current_total_equity - self.total_asset) * self.reward_scaling
        self.total_asset = current_total_equity
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        if done:
            reward = self.gamma_reward
            self.episode_return = self.total_asset / self.initial_capital

        return self.get_state(new_price), reward, done, {}

    def _execute_trade(self, i: int, trade_size: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Executes a trade for a single stock, handling long, short, cover, and sell."""
        if trade_size == 0:
            return

        current_holding = self.stocks[i]
        
        if trade_size > 0:  # Buy or cover
            if current_holding < 0:
                shares_to_cover = min(trade_size, -current_holding)
                self._cover_short(i, shares_to_cover, price, volatility, volume)
                trade_size -= shares_to_cover
            if trade_size > 0:
                self._buy_long(i, trade_size, price, volatility, volume)
        else:  # Sell or short
            trade_size = abs(trade_size)
            if current_holding > 0:
                shares_to_sell = min(trade_size, current_holding)
                self._sell_long(i, shares_to_sell, price, volatility, volume)
                trade_size -= shares_to_sell
            if trade_size > 0:
                self._sell_short(i, trade_size, price, volatility, volume)

    def _buy_long(self, i: int, shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Opens or increases a long position."""
        impact = self.impact_model.apply_trade(shares, price[i], volatility[i], volume[i], self.stock_symbols[i])
        cost = shares * price[i] + impact.cost

        available_buying_power = self.long_equity * (self.margin_rate -1)
        if cost <= available_buying_power:
            self.stocks[i] += shares
            self.loan += cost
            self.permanent_impact_per_stock[i] += impact.price_shift

    def _sell_long(self, i: int, shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Closes or reduces a long position."""
        impact = self.impact_model.apply_trade(-shares, price[i], volatility[i], volume[i], self.stock_symbols[i])
        proceeds = shares * price[i] - impact.cost
        self.stocks[i] -= shares
        self.loan -= proceeds
        if self.loan < 0:
            self.long_equity += -self.loan
            self.loan = 0
        self.permanent_impact_per_stock[i] += impact.price_shift

    def _sell_short(self, i: int, shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Opens or increases a short position."""
        impact = self.impact_model.apply_trade(-shares, price[i], volatility[i], volume[i], self.stock_symbols[i])
        proceeds = shares * price[i] - impact.cost
        
        required_equity = (self.short_credit_balance + proceeds) * self.maintenance_margin
        if self.short_equity >= required_equity:
            self.stocks[i] -= shares
            self.short_credit_balance += proceeds
            self.permanent_impact_per_stock[i] += impact.price_shift

    def _cover_short(self, i: int, shares: int, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Closes or reduces a short position."""
        impact = self.impact_model.apply_trade(shares, price[i], volatility[i], volume[i], self.stock_symbols[i])
        cost = shares * price[i] + impact.cost
        self.stocks[i] += shares
        self.short_credit_balance -= cost
        if self.short_credit_balance < 0:
            self.short_equity += -self.short_credit_balance
            self.short_credit_balance = 0
        self.permanent_impact_per_stock[i] += impact.price_shift
    
    def _update_equities(self, price: np.ndarray) -> None:
        """Updates long and short equity values based on current market prices."""
        adjusted_prices = price + self.permanent_impact_per_stock
        long_market_value = (self.stocks[self.stocks > 0] * adjusted_prices[self.stocks > 0]).sum()
        self.long_equity = long_market_value - self.loan

        short_market_value = abs((self.stocks[self.stocks < 0] * adjusted_prices[self.stocks < 0]).sum())
        # Short equity is credit balance (cash from shorting) minus the current cost to buy back
        self.short_equity = self.short_credit_balance - short_market_value

    def _handle_margin_calls(self, price: np.ndarray, volatility: np.ndarray, volume: np.ndarray) -> None:
        """Checks for and handles margin calls by liquidating positions."""
        # Long account margin call
        long_market_value = (self.stocks[self.stocks > 0] * price[self.stocks > 0]).sum()
        if long_market_value > 0 and (self.long_equity / long_market_value) < self.maintenance_margin:
            if np.any(self.stocks > 0):
                largest_pos_idx = np.argmax(self.stocks * price)
                shares_to_sell = self.stocks[largest_pos_idx] * 0.2  # Liquidate 20%
                self._sell_long(largest_pos_idx, shares_to_sell, price, volatility, volume)

        # Short account margin call
        short_market_value = abs((self.stocks[self.stocks < 0] * price[self.stocks < 0]).sum())
        if short_market_value > 0 and (self.short_equity / short_market_value) < self.maintenance_margin:
            if np.any(self.stocks < 0):
                largest_short_idx = np.argmin(self.stocks * price)
                shares_to_cover = abs(self.stocks[largest_short_idx] * 0.2)  # Cover 20%
                self._cover_short(largest_short_idx, shares_to_cover, price, volatility, volume)

    def _calculate_total_equity(self, price: np.ndarray) -> float:
        """Calculates the total net equity across long and short accounts."""
        self._update_equities(price)
        return self.long_equity + self.short_equity

    def get_state(self, price: np.ndarray) -> np.ndarray:
        """Returns the current state of the environment."""
        self._update_equities(price)
        scale = 2**-12
        state_components = [
            np.array([self.long_equity, self.loan, self.short_equity, self.short_credit_balance]) * scale,
            self.turbulence_array[self.time], self.turbulence_bool[self.time],
            price * scale, self.stocks * scale, self.permanent_impact_per_stock * scale,
            self.stocks_cool_down, self.tech_array[self.time]
        ]
        return np.hstack(state_components).astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary: np.ndarray, thresh: float) -> np.ndarray:
        """A sigmoid function to bound the turbulence values."""
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        return sigmoid(ary / thresh) * thresh
