# Market Impact Environments for FinRL

This directory contains enhanced versions of FinRL's trading environments that incorporate realistic market impact models. These environments are designed to provide a more accurate simulation of real-world trading by accounting for the costs associated with executing large orders.

## Key Features

1.  **Dynamic Position Sizing**: Instead of fixed share limits, the environments can dynamically calculate position sizes based on a percentage of the current portfolio value. This leads to more realistic capital allocation.

2.  **Market Impact Modeling**: Transaction costs are not based on a fixed percentage. Instead, they use the Almgren-Chriss (AC) impact model to calculate costs based on:
    *   **Trade Size**: Larger trades incur larger costs.
    *   **Market-Volume**: Costs are relative to the available liquidity (daily volume).
    *   **Volatility**: Higher volatility increases impact costs.

3.  **Permanent vs. Temporary Impact**: The model distinguishes between two types of impact:
    *   **Temporary Impact**: The immediate cost of demanding liquidity.
    *   **Permanent Impact**: The persistent effect on the stock's price caused by the trade. This is tracked over time and affects future mark-to-market calculations.

## Environments

-   `env_stock_trading_impact.py`: An enhanced version of the stock trading environment. The agent's actions are to buy or sell a certain number of shares, and it learns to manage trades to minimize impact costs.

-   `env_portfolio_optimization_impact.py`: An enhanced version of the portfolio optimization environment. The agent's actions are to define target portfolio weights, and it learns to rebalance the portfolio while considering the impact costs of the required trades.

-   `env_margin_trader_impact.py`: This environment adapts the concepts from the "Margin Trader" paper to the FinRL-Meta framework. It allows the agent to manage a portfolio with both long and short positions, use leverage, and handle margin constraints, all while accounting for the market impact of its trades.

-   `impact_models.py`: Contains the implementation of the market impact models (e.g., `ACImpactModel`).

## Motivation

By incorporating market impact, these environments encourage the reinforcement learning agent to develop more sophisticated strategies. The agent must learn to:
- Break up large orders to reduce costs.
- Time trades to coincide with higher liquidity.
- Balance the alpha signal of a trade against its potential impact cost.
- Understand that its own actions can affect market prices.

This leads to more robust and realistic trading algorithms.

## References

The environments in this folder are based on concepts from the following academic papers:

-   **Portfolio Optimization**: The portfolio optimization environments are based on the POE (Portfolio Optimization Environment) framework.
    -   *Costa, C., & Costa, A. (2023). POE: A General Portfolio Optimization Environment for FinRL.*

-   **Market Impact**: The market impact model is based on the Almgren-Chriss model, a standard in quantitative finance for modeling execution costs.
    -   *Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions. Journal of Risk, 3(2), 5–39.*

-   **Margin Trading**: The margin trading environment adapts concepts from the "Margin Trader" paper, which specifically addresses portfolio management with leverage and constraints.
    -   *Gu, J., Du, W., Rahman, A. M. M., & Wang, G. (2023). Margin Trader: A Reinforcement Learning Framework for Portfolio Management with Margin and Constraints.*

### Citations
```
@inproceedings{bwaif,
 author = {Caio Costa and Anna Costa},
 title = {POE: A General Portfolio Optimization Environment for FinRL},
 booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
 location = {João Pessoa/PB},
 year = {2023},
 keywords = {},
 issn = {0000-0000},
 pages = {132--143},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/bwaif.2023.231144},
 url = {https://sol.sbc.org.br/index.php/bwaif/article/view/24959}
}
```
```
@inproceedings{gu2023margin,
  title={Margin Trader: A Reinforcement Learning Framework for Portfolio Management with Margin and Constraints},
  author={Gu, Jingyi and Du, Wenlu and Rahman, AM Muntasir and Wang, Guiling},
  booktitle={Proceedings of the Fourth ACM International Conference on AI in Finance},
  pages={610--618},
  year={2023}
}
```
