from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import date


@dataclass
class TradeImpact:
    """
    Encapsulates the result of an impact model's trade.
    
    Attributes:
    -----------
    cost : float
        Total notional cost for executing the trade (impact + commission).
    price_shift : float
        Permanent price shift ΔP to apply to the mid-price for subsequent trades.
    """
    cost: float
    price_shift: float


class ImpactModel(ABC):
    def __init__(self):
        # Track cumulative permanent price shift per symbol
        self._perm_state = defaultdict(float)
        # Create dataframe to store impact history
        self._impact_history = pd.DataFrame(columns=['date', 'symbol', 'permanent_impact'])
    
    @abstractmethod
    def apply_trade(self,
                    trade_size: float,
                    price: float,
                    volatility: float,
                    volume: float,
                    symbol: str) -> TradeImpact:
        """
        Execute a trade and return its impact.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def end_day(self, date_obj: date) -> None:
        """
        Record the permanent impact state at the end of a trading day.
        
        Parameters
        ----------
        date_obj : datetime.date
            The date of the trading day being ended.
            
        Notes
        -----
        This method records the current permanent impact state for each symbol
        into the impact history dataframe for later analysis.
        """
        # Create records for each symbol's permanent impact
        new_records = []
        for symbol, impact in self._perm_state.items():
            new_records.append({
                'date': date_obj.isoformat(),
                'symbol': symbol,
                'permanent_impact': impact
            })

        # Append to the impact history dataframe
        if new_records:
            # Convert to DataFrame with explicit dtypes to avoid FutureWarning
            new_df = pd.DataFrame(new_records)

            # If impact_history is empty, just use the new DataFrame
            if self._impact_history.empty:
                self._impact_history = new_df
            else:
                # Otherwise append to existing DataFrame with consistent dtypes
                self._impact_history = pd.concat(
                    [self._impact_history, new_df],
                    ignore_index=True,
                    axis=0
                )

    def get_impact_history(self) -> pd.DataFrame:
        """
        Get the recorded impact history dataframe.
        
        Returns
        -------
        pd.DataFrame
            The impact history with columns ['date', 'symbol', 'permanent_impact'].
        """
        return self._impact_history.copy()
    
    def reset_impact_history(self) -> None:
        """
        Reset the impact history dataframe.
        """
        self._impact_history = pd.DataFrame(columns=['date', 'symbol', 'permanent_impact'])


class ACImpactModel(ImpactModel):
    """
    Almgren–Chriss market impact model for transaction cost estimation.

    The total execution cost is decomposed into a permanent (quadratic) impact
    and a temporary (linear) impact:

        C(x) = (0.5 * η * x² / V + γ * |x|) * P

    where
    -------
    x : float
        Signed trade size in shares.
    V : float
        Available daily volume in shares.
    P : float
        Execution price per share (e.g. VWAP).
    η : float
        Permanent impact coefficient.
    γ : float
        Temporary impact coefficient.

    In practice, one sets these coefficients relative to daily volatility σ and
    volume V via dimensionless scaling factors α and β:

        η = α * σ / V
        γ = β * σ / √V

    Typical values for large‐cap equities are α ≈ 0.01 and β ≈ 0.1,
    which yield execution costs of a few basis points for trades of 1% ADV—
    see Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and High‑Frequency Trading. Cambridge University Press.

    This model uses the empirically robust square-root law for market impact,
    consistent with extensive research literature (Gatheral 2010, Tóth et al. 2011).

    Parameters
    ----------
    Y : float (default=0.6)
        Empirical square-root impact coefficient, typically 0.5–1.0.
    perm_fraction : float (default=0.25)
        Fraction of total impact assumed permanent, ~25% empirically.

    Methods
    -------
    cost_notional(trade_size, price, volatility, volume) -> float
        Compute total notional cost (impact + commission) for trading
        `trade_size` shares at `price`, given `volume`. Volatility is
        unused in this static implementation but kept for interface consistency.

    References
    ----------
    Almgren, R. & Chriss, N. (2000).
        Optimal execution of portfolio transactions.
        Journal of Risk, 3(2), 5–39.
    Cartea, Á., Jaimungal, S., & Penalva, J. (2015).
        Algorithmic and High‑Frequency Trading. Cambridge University Press.
    Gatheral, J. & Schied, A. (2013).
        Dynamical Models of Market Impact and Algorithms for Order Execution,
        in *Handbook on Systemic Risk* (J.‑P. Fouque & J. Langsam, eds.),
        pp. 579–599. Cambridge University Press. doi:10.2139/ssrn.2034178
    """

    def __init__(self, Y=0.6, perm_fraction=0.25):
        super().__init__()
        self.Y = Y
        self.perm_fraction = perm_fraction

    def apply_trade(self,
                    trade_size: float,
                    price: float,
                    volatility: float,
                    volume: float,
                    symbol: str) -> TradeImpact:
        participation_rate = abs(trade_size) / volume

        # Square-root total impact fraction (empirically robust)
        total_impact_fraction = self.Y * volatility * np.sqrt(participation_rate)
        gamma = total_impact_fraction
        eta = self.perm_fraction * gamma

        # Compute costs (permanent + immediate)
        perm_cost = eta * abs(trade_size)
        immediate_cost = (gamma - eta) * abs(trade_size)
        cost = (perm_cost + immediate_cost) * price

        # Permanent price shift for future trades
        price_shift = eta * np.sign(trade_size) * volatility * np.sqrt(participation_rate)
        self._perm_state[symbol] += price_shift

        return TradeImpact(cost=cost, price_shift=price_shift)


class OWImpactModel(ImpactModel):
    """
    Obizhaeva–Wang market impact model for transaction cost estimation.

    This model captures three components of execution impact:
      1. Permanent impact (quadratic in trade size).
      2. Immediate (temporary) impact (linear in trade size).
      3. Transient impact from past trades that decays exponentially.

    The notional cost for a trade of size x_t at price P_t is:

        C_t(x) = (0.5 * η * x_t² / V_t
                + γ * |x_t|
                + γ * ∑_{s<t} x_s * exp(-κ * (t - s))
               ) * P_t

    where
    -----
    x_t : float
        Signed trade size in shares at time t.
    V_t : float
        Available daily volume in shares.
    P_t : float
        Execution price per share (e.g., VWAP).
    η : float
        Permanent impact coefficient.
    γ : float
        Immediate (temporary) impact coefficient.
    κ : float
        Transient decay rate per time step.

     On each trade we set:

      η = α · σ / V  
      γ = β · σ / √V  
      κ = ln(2) / h  

    where:
    - σ is the asset’s daily return vol (from any reliable source).  
    - V is the asset’s average daily volume.  
    - α = 0.01, β = 0.1 (Cartea et al. 2015 – typical for large‑cap equities) :contentReference[oaicite:0]{index=0}.  
    - h = 30 minutes ≃ 30/390 days ⇒ κ ≃ ln(2)/(30/390) ≃ 9 (Gatheral & Schied 2013 on resilience decay) :contentReference[oaicite:1]{index=1}.  
    - Obizhaeva & Wang (2013) show that this exponential‑decay form captures real LOB resilience dynamics :contentReference[oaicite:2]{index=2}.

    The model calculates impact dynamically, based on:
        - Permanent impact (fraction of total impact, typically ~25%)
        - Temporary/transient impact following the square-root law

    Empirical references:
        - Square-root impact law: Gatheral (2010), Tóth et al. (2011), Bouchaud et al. (2018)
        - Transient impact half-life: Gatheral & Schied (2013)

    Parameters
    ----------
    Y : float (default=0.6)
        Empirical square-root impact coefficient (dimensionless), 
        typically between 0.5 and 1.0 in empirical studies.
    perm_fraction : float (default=0.25)
        Fraction of the total impact that's permanent (long-run), 
        typically ~25% empirically.
    half_life_minutes : float (default=30.0)
        Half-life of transient impact in minutes (typical values: 15–60 mins).

    Methods
    -------
    apply_trade(trade_size, price, volatility, volume, symbol) -> TradeImpact
        Execute a trade and return:
            - cost: total notional impact cost for that trade
            - price_shift: permanent mid-price shift ΔP for subsequent valuation

    References
    ----------
    Obizhaeva, A. & Wang, J. (2013).
        Optimal trading strategy and supply/demand dynamics.
        Journal of Financial Markets, 16(1), 1–32.
    Cartea, Á., Jaimungal, S., & Penalva, J. (2015).
        Algorithmic and High‑Frequency Trading.
        Cambridge University Press.
    Gatheral, J. & Schied, A. (2013).
        Dynamical Models of Market Impact and Algorithms for Order Execution,
        in *Handbook on Systemic Risk* (J.‑P. Fouque & J. Langsam, eds.),
        pp. 579–599. Cambridge University Press. doi:10.2139/ssrn.2034178
    """
    def __init__(self, Y=0.6, perm_fraction=0.25, half_life_minutes=30.0):
        super().__init__()
        self.Y = Y
        self.perm_fraction = perm_fraction
        # Convert half-life (in trading days) to decay rate
        self.kappa = np.log(2) / (half_life_minutes / 390)
        # Internal state trackers per symbol
        self._transient_states = defaultdict(float)

    def apply_trade(self,
                    trade_size: float,
                    price: float,
                    volatility: float,
                    volume: float,
                    symbol: str) -> TradeImpact:
        """
        Execute a trade and compute both cost and permanent price shift.

        Parameters
        ----------
        trade_size : float
            Signed number of shares to trade.
        price : float
            Execution price per share (e.g., VWAP).
        volatility : float
            Asset volatility (unused here; for interface consistency).
        volume : float
            Available daily volume in shares.
        symbol : str
            Asset identifier to track per-symbol state.

        Returns
        -------
        TradeImpact
            cost : total notional cost (impact * price)
            price_shift : permanent mid-price shift for subsequent valuation
        """
        participation_rate = abs(trade_size) / volume

        total_impact_fraction = self.Y * volatility * np.sqrt(participation_rate)
        gamma = total_impact_fraction
        eta = self.perm_fraction * gamma

        # decay previous transient state
        prev_transient = self._transient_states[symbol] * np.exp(-self.kappa)

        # costs calculation
        perm_cost = eta * abs(trade_size)
        immediate_cost = (gamma - eta) * abs(trade_size)
        trans_cost = gamma * prev_transient * volume  # scaled by volume to be meaningful

        cost = (perm_cost + immediate_cost + trans_cost) * price

        # update transient state properly
        self._transient_states[symbol] = prev_transient + participation_rate

        price_shift = eta * np.sign(trade_size) * volatility * np.sqrt(participation_rate)

        return TradeImpact(cost=cost, price_shift=price_shift)
