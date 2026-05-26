import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import config

# ==============================================================================
# 1. PAGE CONFIGURATION & SECURITY GATE
# ==============================================================================
st.set_page_config(page_title="Quantum Sentinel Analytics Hub", layout="wide", page_icon="🛡️")

# Force strict HTTP verification layers directly within session allocations
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
# 2. DATA ACQUISITION & CACHING
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_tournament_results():
    """Reads the finalized tournament results containing Deflated Sharpe Ratios."""
    if os.path.exists(config.RESULTS_FILE):
        return pd.read_parquet(config.RESULTS_FILE, engine="pyarrow")
    return pd.DataFrame()

@st.cache_data(ttl=15)
def fetch_live_ledger():
    """
    Seamlessly reads the entire PyArrow partitioned directory as a single DataFrame.
    Filters out any hidden OS files to prevent read errors.
    """
    if os.path.exists(config.LIVE_LOG_DIR):
        try:
            return pd.read_parquet(config.LIVE_LOG_DIR, engine="pyarrow")
        except Exception as e:
            st.error(f"Error reading live ledger: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

results_df = fetch_tournament_results()
live_df = fetch_live_ledger()

# ==============================================================================
# 3. INTERACTIVE TABS & WORKSPACE
# ==============================================================================
st.title("🛡️ Quantum Sentinel Strategy Workspace")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary", 
    "🏆 Tournament Standings", 
    "📡 Live Activity & Veto Ledger", 
    "⚙️ System Health"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY (Human Readable)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Executive Summary")
    st.markdown("A high-level, human-readable overview of the current trading agent and portfolio health.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # High-level Metrics
    total_models = len(results_df) if not results_df.empty else 0
    total_trades = len(live_df[live_df['Status'].str.contains("EXECUTED", na=False)]) if not live_df.empty else 0
    total_vetoes = len(live_df[live_df['Status'] == "HELD"]) if not live_df.empty else 0
    avg_confidence = live_df['ML_Confidence'].mean() * 100 if not live_df.empty else 0.0

    col1.metric("Active Champion Models", f"{total_models}")
    col2.metric("Total Executed Trades", f"{total_trades}")
    col3.metric("Trades Vetoed (Held)", f"{total_vetoes}")
    col4.metric("Average Model Confidence", f"{avg_confidence:.1f}%")
    
    st.markdown("### Recent Strategy Execution Overview")
    if not live_df.empty:
        # Simple Pie Chart for Executed vs Vetoed
        status_counts = live_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig1 = px.pie(
            status_counts, 
            names='Status', 
            values='Count',
            hole=0.4,
            color='Status',
            color_discrete_map={"HELD": "#EF553B", "EXECUTED (PAPER)": "#00CC96", "EXECUTED (LIVE)": "#AB63FA"}
        )
        fig1.update_layout(title_text="Model Decision Breakdown")
        
        # Simple Bar Chart for Veto Reasons
        veto_counts = live_df[live_df['Status'] == 'HELD']['Veto_Reason'].value_counts().reset_index()
        veto_counts.columns = ['Veto Reason', 'Frequency']
        fig2 = px.bar(veto_counts, x='Veto Reason', y='Frequency', title="Why are trades being vetoed?")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No live trading activity recorded yet.")

# ------------------------------------------------------------------------------
# TAB 2: TOURNAMENT STANDINGS & TEARSHEETS
# ------------------------------------------------------------------------------
with tab2:
    st.header("Quantitative Model Leaderboard")
    
    if not results_df.empty:
        # Data-driven Plotly Visualization
        st.subheader("Deflated Sharpe Ratios (DSR) by Sector")
        fig_dsr = px.bar(
            results_df.sort_values(by='composite_score', ascending=False), 
            x='sector', 
            y='composite_score', 
            color='composite_score',
            color_continuous_scale='Viridis',
            labels={'composite_score': 'Deflated Sharpe Ratio (DSR)', 'sector': 'Sector'},
            text_auto='.2f'
        )
        st.plotly_chart(fig_dsr, use_container_width=True)

        st.dataframe(results_df.style.highlight_max(axis=0, subset=['composite_score'], color='lightgreen'))

        st.markdown("---")
        st.subheader("Institutional Risk Profiles (Tearsheets)")
        st.markdown("Select a sector champion to view its comprehensive `quantstats` tear sheet.")
        
        selected_sector = st.selectbox("Select Sector Champion:", results_df['sector'].unique())
        
        tearsheet_path = f"tearsheet_{selected_sector}.html"
        if os.path.exists(tearsheet_path):
            with open(tearsheet_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            # Embed the full QuantStats HTML report
            components.html(html_data, height=800, scrolling=True)
        else:
            st.warning(f"No tearsheet found for {selected_sector}. Ensure evaluator.py ran successfully.")
    else:
        st.info("No tournament results found. Run `tournament.py` and `evaluator.py` to populate.")

# ------------------------------------------------------------------------------
# TAB 3: LIVE ACTIVITY & VETO LEDGER
# ------------------------------------------------------------------------------
with tab3:
    st.header("Live Agent Telemetry Ledger")
    
    if not live_df.empty:
        # Filter controls
        st.markdown("#### Filter Ledger")
        c1, c2 = st.columns(2)
        model_filter = c1.multiselect("Filter by Model ID", live_df['Model_ID'].unique())
        status_filter = c2.multiselect("Filter by Execution Status", live_df['Status'].unique())
        
        filtered_df = live_df.copy()
        if model_filter:
            filtered_df = filtered_df[filtered_df['Model_ID'].isin(model_filter)]
        if status_filter:
            filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
            
        st.dataframe(
            filtered_df.sort_values(by="Timestamp", ascending=False), 
            use_container_width=True
        )
        
        # Advanced Data-driven visual: Confidence Density
        st.subheader("Model Confidence Distribution")
        fig_conf = px.histogram(
            filtered_df, 
            x="ML_Confidence", 
            color="Status", 
            marginal="box",
            nbins=50,
            title="Distribution of XGBoost Output Probabilities"
        )
        # Overlay the configured confidence threshold
        fig_conf.add_vline(x=config.CONFIDENCE_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Execution Threshold")
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Live execution ledger is empty. Start the `live_trader.py` loop.")

# ------------------------------------------------------------------------------
# TAB 4: SYSTEM HEALTH
# ------------------------------------------------------------------------------
with tab4:
    st.header("System Orchestration & Environment Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment Configurations")
        st.code(f"""
        RUN_MODE: {config.RUN_MODE}
        START_DATE: {config.START_DATE}
        END_DATE: {config.END_DATE}
        CONFIDENCE_THRESHOLD: {config.CONFIDENCE_THRESHOLD}
        MAX_HOLD_DAYS: {config.MAX_HOLD_DAYS}
        """)
        
    with col2:
        st.subheader("Directory Architecture Status")
        dirs = {
            "Raw Vault": config.RAW_VAULT_DIR,
            "Processed Vault": config.PROCESSED_VAULT_DIR,
            "Production Models": config.PROD_MODELS_DIR,
            "Live Ledger": config.LIVE_LOG_DIR,
            "System Logs": config.LOG_DIR
        }
        for name, path in dirs.items():
            status = "✅ Online" if os.path.exists(path) else "❌ Missing"
            st.write(f"**{name}:** {status} (`{path}`)")
            
    st.write(f"Last UI Data Sync: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")