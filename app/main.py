import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
from src.llm_engine import LLMEngine
from src.ml_engine import MLEngine

# --- SETUP ---
st.set_page_config(page_title="RetailPulse", layout="wide", page_icon="⚡")
load_dotenv()

# --- LOAD ENGINES ---
@st.cache_resource
def load_engines():
    try:
        return LLMEngine(), MLEngine()
    except Exception as e:
        st.error(f"❌ Critical Error: Engines failed to load.\n{e}")
        return None, None

llm_engine, ml_engine = load_engines()

# --- DATABASE CONNECTION ---
try:
    conn = sqlite3.connect("retailpulse.db", check_same_thread=False)
except Exception as e:
    st.error(f"❌ Database Connection Failed: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("RetailPulse")
    st.markdown(
        "<h2 style='margin-bottom: 0.5rem;'>Workflow</h2>",
        unsafe_allow_html=True
    )

    page = st.radio(
        label="Workflow",
        options=["1. Data Blender", "2. Visual Insights", "3. Forecast Engine"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🧠 AI Context")
    
    data_context = st.selectbox("Data Domain", ["Retail", "Financial", "Generic"])
    user_rules = st.text_area(
        "SOPs / Business Rules:",
        height=100,
        placeholder="e.g. 'Logistics are closed on Sundays'..."
    )
    
    if st.button("💾 Update Memory"):
        if llm_engine:
            with st.spinner("Updating Knowledge Base..."):
                msg = llm_engine.update_knowledge_base(user_rules)
                st.success(msg)

# --- SESSION STATE ---
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'forecast_results' not in st.session_state:
    st.session_state['forecast_results'] = None

# ==========================================
# PAGE 1: DATA BLENDER
# ==========================================
if page == "1. Data Blender":
    st.title("📂 Data Blender")
    
    with st.expander("📤 Upload Raw Data", expanded=True):
        uploaded_files = st.file_uploader("Upload CSVs", accept_multiple_files=True)
        if uploaded_files:
            success_count = 0
            for f in uploaded_files:
                table_name = f.name.split('.')[0].replace(" ", "_").lower()
                try:
                    df = pd.read_csv(f)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Error loading {f.name}: {e}")
            
            if success_count > 0:
                st.success(f"Successfully uploaded {success_count} files to Database.")

    st.markdown("---")
    st.subheader("🔗 Merge Tables")
    
    try:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        table_list = tables['name'].tolist() if not tables.empty else []
    except Exception:
        table_list = []
    
    if not table_list:
        st.info("No data found. Please upload files above.")
    else:
        c1, c2, c3 = st.columns(3)
        left_table = c1.selectbox("Left Table", table_list)
        right_table = c2.selectbox("Right Table", ["(None)"] + table_list)
        
        join_col = None
        if right_table != "(None)":
            try:
                l_cols = pd.read_sql(f"SELECT * FROM {left_table} LIMIT 0", conn).columns.tolist()
                r_cols = pd.read_sql(f"SELECT * FROM {right_table} LIMIT 0", conn).columns.tolist()
                common = list(set(l_cols) & set(r_cols))
                if common:
                    join_col = c3.selectbox("Join Key", common)
                else:
                    st.warning("⚠️ No common columns found.")
            except Exception:
                pass

        col_btn_1, col_btn_2, _ = st.columns([1, 1, 3])

        if col_btn_1.button("🔎 Preview"):
            try:
                if right_table == "(None)":
                    merged_df = pd.read_sql(f"SELECT * FROM {left_table}", conn)
                else:
                    if not join_col:
                        st.error("Select a Join Key.")
                        st.stop()
                    df_left = pd.read_sql(f"SELECT * FROM {left_table}", conn)
                    df_right = pd.read_sql(f"SELECT * FROM {right_table}", conn)
                    merged_df = pd.merge(df_left, df_right, on=join_col, how='inner')
                
                if not merged_df.empty:
                    st.session_state['preview_df'] = merged_df
                    st.dataframe(merged_df.head(5))
                    st.success(f"Preview generated: {len(merged_df)} rows")
            except Exception as e:
                st.error(f"Merge Failed: {e}")

        if 'preview_df' in st.session_state:
            if col_btn_2.button("💾 Save as Master"):
                try:
                    st.session_state['preview_df'].to_sql(
                        "master_view", conn, if_exists='replace', index=False
                    )
                    st.session_state['master_df'] = st.session_state['preview_df']
                    st.success("Master View Saved! Proceed to Visualization.")
                except Exception as e:
                    st.error(f"Save Failed: {e}")

# ==========================================
# PAGE 2: VISUAL INSIGHTS
# ==========================================
elif page == "2. Visual Insights":
    st.title("📊 Visual Insights")
    df = st.session_state['master_df']
    
    if df is None:
        st.warning("⚠️ No Master Data set.")
        st.stop()
        
    c1, c2, c3 = st.columns(3)
    col_date = c1.selectbox("Date Column", df.columns)
    col_target = c2.selectbox("Metric", df.select_dtypes(include=np.number).columns)
    chart_type = c3.selectbox("Chart Type", ["Line Trend", "Bar Chart", "Distribution"])
    
    try:
        df['dt_mapped'] = pd.to_datetime(df[col_date], errors='coerce')
        daily = df.groupby('dt_mapped')[col_target].sum().reset_index()
        
        st.markdown("---")
        if chart_type == "Line Trend":
            fig = px.line(daily, x='dt_mapped', y=col_target)
        elif chart_type == "Bar Chart":
            fig = px.bar(daily, x='dt_mapped', y=col_target)
        else:
            fig = px.histogram(df, x=col_target, nbins=50)
            
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ⚡ Quick Stats")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total", f"{daily[col_target].sum():,.0f}")
        k2.metric("Average", f"{daily[col_target].mean():,.2f}")
        k3.metric("Max Day", f"{daily[col_target].max():,.2f}")

        st.markdown("---")
        st.subheader("AI Trend Analyst")
        if st.button("✨ Analyze Trends"):
            with st.spinner("Analyzing..."):
                summary = f"Metric: {col_target}, Total: {daily[col_target].sum()}"
                st.info(llm_engine.analyze_visuals(summary))

    except Exception as e:
        st.error(f"Visualization Error: {e}")

# ==========================================
# PAGE 3: FORECAST ENGINE
# ==========================================
elif page == "3. Forecast Engine":
    st.title("Forecaster")
    df = st.session_state['master_df']
    
    if df is None:
        st.warning("⚠️ No Master Data set.")
        st.stop()
        
    c1, c2, c3, c4 = st.columns(4)
    col_date = c1.selectbox("Date", df.columns)
    col_target = c2.selectbox("Target", df.select_dtypes(include=np.number).columns)
    model_choice = c3.selectbox("Model", ["XGBoost", "Prophet"])
    mode_choice = c4.radio("Mode", ["Validation", "Future (30 Days)"])
    
    if st.button("🚀 Run Prediction"):
        with st.spinner("Calculating..."):
            train, test, forecast, metrics = ml_engine.run_forecast(
                df,
                col_date,
                col_target,
                model_choice,
                "Validation" if "Validation" in mode_choice else "Future",
                data_context
            )

            st.session_state['forecast_results'] = {
                'train': train,
                'test': test,
                'forecast': forecast,
                'metrics': metrics,
                'model': model_choice,
                'mode': mode_choice
            }

    res = st.session_state['forecast_results']
    if res:
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['train']['ds'], y=res['train']['y'],
            name='History', line=dict(color='#95a5a6')
        ))
        
        if "Validation" in res['mode']:
            fig.add_trace(go.Scatter(
                x=res['test']['ds'], y=res['test']['y'],
                name='Actual', line=dict(color='#2980b9')
            ))
            fig.add_trace(go.Scatter(
                x=res['forecast']['ds'], y=res['forecast']['yhat'],
                name='AI', line=dict(color='#e74c3c', dash='dot')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=res['forecast']['ds'], y=res['forecast']['yhat'],
                name='Future', line=dict(color='#27ae60')
            ))
        
        st.plotly_chart(fig, use_container_width=True)

        # ---- METRICS (FINAL, DOMAIN-AWARE) ----
        metrics = res['metrics']
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                if k == "Direction":
                    col.metric("Trend Signal", f"{v:.1f}%")
                elif k == "SMAPE":
                    col.metric("SMAPE", f"{v:.1f}%")
                else:
                    col.metric(k, f"{v:.2f}")
        else:
            st.info("Future forecast — no validation metrics available.")

        if data_context == "Financial":
            st.caption("⚠ Financial forecasts are statistical signals, not trading advice.")
