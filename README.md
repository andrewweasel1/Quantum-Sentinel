# 🛡️ Quantum Sentinel V4

Quantum Sentinel is an institutional-grade, multi-agent quantitative trading architecture. It is designed to rigorously backtest, evaluate, and execute machine learning-based trading strategies while actively mitigating common quantitative pitfalls such as survivorship bias, look-ahead bias, and multiple testing bias.

## 🏗️ Architecture Overview

The system is decoupled into isolated layers for data acquisition, out-of-core feature compilation, statistical evaluation, and live execution.

*   **`config.py`**: Centralized state manager for all operational modes, hyperparameters, and PyArrow schema definitions.
*   **`data_ingestion.py`**: Uses Point-in-Time (PiT) data queries (via EODHD) to construct a survivorship-bias-free historical universe. 
*   **`feature_compiler.py`**: Leverages Numba JIT-compilation and Dask out-of-core memory mapping to generate market microstructure sensors (e.g., Amihud Illiquidity, Roll's Spread) and Tail-Risk metrics. It utilizes a Friction-Adjusted Triple Barrier Method for label generation.
*   **`tournament.py`**: Runs a massive hyperparameter grid search across XGBoost models using Combinatorial Purged Cross-Validation (CPCV) to prevent data leakage and generates multi-path returns matrices.
*   **`evaluator.py`**: The statistical gatekeeper. Evaluates candidate models by computing the Probability of Backtest Overfitting (PBO), controls for the False Discovery Rate (FDR) using the Deflated Sharpe Ratio (DSR), and generates HTML tearsheets.
*   **`live_trader.py`**: An asynchronous execution sandbox that routes signals from the champion models directly to the Alpaca Trading API, managing dynamic position sizing and logging to a partitioned PyArrow ledger.
*   **`dashboard.py`**: An interactive Streamlit application for monitoring active models, live telemetry, and reviewing quantitative tearsheets.

## 🚀 Quick Start

**1. Set Environment Variables**
Ensure you have the necessary API keys exported in your environment:
```bash
export EODHD_API_KEY="your_api_key"
export ALPACA_PAPER_API_KEY="your_paper_key"
export ALPACA_PAPER_SECRET_KEY="your_paper_secret"
# Add Live Alpaca keys if trading with live capital
2. Install Dependencies
pip install -r requirements.txt
3. Run the Pipeline Execute the main orchestrator to download raw data, compile offline feature matrices, and initiate the out-of-core XGBoost tournament:
python main.py --refresh-raw
4. Evaluate Champions Once the tournament generates the CPCV returns matrices, run the evaluator to mathematically prove the alpha and promote candidates to production champions:
python evaluator.py
5. Launch the Dashboard Monitor your live execution ledger and view institutional tearsheets:
streamlit run dashboard.py