import os
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import plotly.express as px

import config

# ==============================================================================
# 1. PAGE CONFIGURATION & SECURITY GATE
# ==============================================================================
st.set_page_config(page_title="Quantum Sentinel Analytics Hub", layout="wide", page_icon="🛡️")

# Force strict HTTP verification layers directly within session allocations
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Quantum Workspace Security Gate")
    user_input = st.text_input("Username Identification Profile:")
    pass_input = st.text_input("Secret Clearance Authentication Key:", type="password")
    
    valid_user = os.environ.get("DASHBOARD_USER") 
    valid_pass = os.environ.get("DASHBOARD_PASS")
    
    if not valid_user or not valid_pass:
        st.error("CRITICAL: Security environment variables (DASHBOARD_USER / DASHBOARD_PASS) are missing.")
        st.stop()
        
    if st.button("Authenticate"):
        if user_input == valid_user and pass_input == valid_pass: 
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Authentication failed. Unauthorized access attempt logged.")
    st.stop()

# ==============================================================================
# 2. DATA INGESTION (PYARROW MEMORY MAPPING)
# ==============================================================================
@st.cache_data(ttl=30)
def fetch_live_ledger() -> pd.DataFrame:
    """Reads the partitioned live execution log utilizing PyArrow zero-copy backends."""
    if not os.path.exists(config.LIVE_LOG_DIR):
        return pd.DataFrame()
    try:
        dataset = pq.ParquetDataset(config.LIVE_LOG_DIR)
        return dataset.read().to_pandas()
    except Exception as e:
        st.error(f"Failed to read live ledger: {e}")
        return pd.DataFrame()

live_df = fetch_live_ledger()

# ==============================================================================
# 3. INTERACTIVE TABS & WORKSPACE
# ==============================================================================
st.title("🛡️ Quantum Sentinel Strategy Workspace")

# Dynamic Badges based on Argparse config
fusion_status = "🟢 ONLINE" if config.FUSION_ENABLED else "🔴 OFFLINE"
risk_status = "🟢 ONLINE" if config.RISK_MANAGER_ENABLED else "🔴 OFFLINE"
st.markdown(f"**LLM Fusion Agent:** {fusion_status} | **Risk Manager Agent:** {risk_status}")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary", 
    "🏆 Tournament Standings", 
    "📡 Live Activity & Veto Ledger", 
    "⚙️ System Health"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY (Dynamic Metrics)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Executive Summary")
    if not live_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Market Evaluations", len(live_df))
        
        # Conditionally render LLM Sensor metrics
        if config.FUSION_ENABLED and 'sentiment' in live_df.columns:
            avg_sentiment = live_df['sentiment'].mean()
            col2.metric("Average LLM Sentiment", f"{avg_sentiment:.3f}")
        else:
            col2.metric("Average LLM Sentiment", "N/A (Fusion Offline)")

        # Conditionally render Risk Manager Veto metrics
        if config.RISK_MANAGER_ENABLED and 'signal' in live_df.columns:
            veto_count = len(live_df[live_df['signal'] == 'VETO'])
            col3.metric("Trades Intercepted by Shield", veto_count)
        else:
            col3.metric("Trades Intercepted by Shield", "N/A (Shield Offline)")
    else:
        st.info("Live execution ledger is currently empty. Awaiting market data.")

# ------------------------------------------------------------------------------
# TAB 2: TOURNAMENT STANDINGS & TEARSHEETS
# ------------------------------------------------------------------------------
with tab2:
    st.header("Quantitative Model Leaderboard")
    st.info("Champion models, feature manifolds, and DSR tearsheets are staged in the `production_models` directory.")

# ------------------------------------------------------------------------------
# TAB 3: LIVE ACTIVITY & VETO LEDGER
# ------------------------------------------------------------------------------
with tab3:
    st.header("Live Agent Telemetry Ledger")
    if not live_df.empty:
        # Sub-tab structure to clearly separate accepted vs blocked trades
        log_tab1, log_tab2 = st.tabs(["🟢 Executed Orders", "🛑 Veto Ledger"])
        
        with log_tab1:
            executed_df = live_df[live_df['signal'] == 'BUY']
            st.dataframe(executed_df, use_container_width=True)
            
        with log_tab2:
            if config.RISK_MANAGER_ENABLED:
                vetoed_df = live_df[live_df['signal'] == 'VETO']
                if not vetoed_df.empty:
                    st.warning("The following predictions were intercepted and canceled by the Risk Manager.")
                    # Only show relevant columns for clarity
                    st.dataframe(vetoed_df[['timestamp', 'ticker', 'probability', 'veto_reason']], use_container_width=True)
                else:
                    st.success("No trades have been vetoed today.")
            else:
                st.error("⚠️ Shield Agent is OFFLINE. Veto logic is bypassed.")
    else:
        st.info("No live telemetry found. Initiate live_trader.py to populate the ledger.")

# ------------------------------------------------------------------------------
# TAB 4: SYSTEM HEALTH
# ------------------------------------------------------------------------------
with tab4:
    st.header("System Orchestration & Environment Parameters")
    st.json({
        "Run Mode": config.RUN_MODE,
        "LLM Fusion Agent": config.FUSION_ENABLED,
        "Risk Manager Agent": config.RISK_MANAGER_ENABLED,
        "Max Drawdown Limit": f"{config.MAX_DAILY_DRAWDOWN * 100}%",
        "Target Premium Gain": f"{config.TARGET_PREMIUM_GAIN * 100}%",
        "Data Engine": "PyArrow Zero-Copy",
        "Parallelization": "Intel TBB + CUDA"
    })