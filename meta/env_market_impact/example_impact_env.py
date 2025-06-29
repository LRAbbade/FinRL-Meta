"""
Example usage of StockTradingEnvImpact with real market data.
This script demonstrates how to:
1. Download real market data for NASDAQ 100 stocks using FinRL's YahooDownloader.
2. Preprocess the data to include prices, technical indicators, volatility, and volume.
3. Instantiate the StockTradingEnvImpact environment.
4. Run a simple simulation with random actions to showcase its functionality.
"""
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config_tickers import NDX_100_TICKER
from env_stock_trading_impact import StockTradingEnvImpact
import sys
import os

# Ensure the custom environment is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def prepare_market_data() -> pd.DataFrame:
    """Download and preprocess NASDAQ 100 data."""
    tickers = NDX_100_TICKER
    # Download data
    df = YahooDownloader(
        start_date="2010-01-01", end_date="2021-12-31", ticker_list=tickers
    ).fetch_data()

    # Preprocess data
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed_df = fe.preprocess_data(df)
    
    # Add volume and calculate daily volatility
    processed_df['volume'] = processed_df['volume'].astype(float)
    # Simple daily volatility: (high - low) / close
    processed_df['volatility'] = (processed_df['high'] - processed_df['low']) / processed_df['close']
    processed_df = processed_df.fillna(0)

    return processed_df


def create_env_config(df: pd.DataFrame) -> dict:
    """Create the configuration dictionary for the environment."""
    
    # Sort by date and ticker
    df = df.sort_values(["date", "tic"], ignore_index=True)
    
    # Get lists of dates and tickers
    date_list = df["date"].unique()
    tic_list = df["tic"].unique()
    
    # Create arrays for price, tech, volatility, and volume
    price_array = []
    tech_array = []
    volatility_array = []
    volume_array = []
    turbulence_array = []

    for date in date_list:
        date_df = df[df["date"] == date]
        
        # Prices and tech indicators
        price_array.append(date_df["close"].values)
        tech_array.append(date_df[["macd", "rsi_30", "cci_30", "dx_30"]].values.flatten())
        
        # Volatility, Volume, and Turbulence
        volatility_array.append(date_df["volatility"].values)
        volume_array.append(date_df["volume"].values)
        turbulence_array.append(date_df["turbulence"].values[0]) # Turbulence is the same for all stocks on a given day

    return {
        "price_array": np.array(price_array),
        "tech_array": np.array(tech_array),
        "volatility_array": np.array(volatility_array),
        "volume_array": np.array(volume_array),
        "turbulence_array": np.array(turbulence_array),
        "if_train": True,
    }


def run_example():
    """Run a simple example with the impact environment using real data."""
    print("Preparing NASDAQ 100 data...")
    market_df = prepare_market_data()
    train_df = data_split(market_df, "2010-01-01", "2020-12-31")
    
    print("Creating environment configuration...")
    config = create_env_config(train_df)
    
    # Create environment
    env = StockTradingEnvImpact(
        config=config,
        initial_capital=1_000_000,
        max_stock_pct=0.02,
    )
    
    print(f"Environment created with {env.stock_dim} stocks.")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Reset environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial cash: ${env.cash:,.2f}")
    
    # Run a few steps with random actions
    for step in range(10):
        actions = np.random.uniform(-0.5, 0.5, env.action_dim)
        next_state, reward, done, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        print(f"  Portfolio value: ${env.total_asset:,.2f}")
        print(f"  Reward: {reward:.6f}")
        
        if done:
            print("Episode finished!")
            break
            
    print("\nExample run complete.")
    print("This demonstrates the environment running with real data.")


if __name__ == "__main__":
    run_example()
