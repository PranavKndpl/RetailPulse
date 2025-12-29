import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os 
from dotenv import load_dotenv
from src.llm_engine import LLMEngine
from src.ml_engine import MLEngine
from src.data_ingestor import DataIngestor
from src.data_repository import DataRepository

# --- SETUP ---
st.set_page_config(page_title="RetailPulse", layout="wide")
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

if "llm_engine" not in st.session_state:
    st.session_state["llm_engine"] = llm_engine

@st.cache_data(show_spinner=False)
def cached_llm_call(summary: str, domain: str):
    engine = st.session_state.get("llm_engine")
    if engine is None:
        return "LLM engine is not available."
    return engine.generate_strategy(summary)

@st.cache_data(show_spinner=False)
def ingest_cached(file):
    return ingestor.ingest(file)
    


# --- POSTGRES DATABASE CONNECTION ---
@st.cache_resource
def get_db_engine():
    try:
        user = os.getenv("DB_USER", "retail_admin")
        password = os.getenv("DB_PASSWORD", "password")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        dbname = os.getenv("DB_NAME", "retailpulse")
        
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(url)
        
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        st.error(f"❌ Database Connection Failed: {e}")
        return None

db_engine = get_db_engine()
ingestor = DataIngestor()
repo = DataRepository(db_engine)

if not db_engine:
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("RetailPulse")
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            overflow: visible !important;
            height: auto !important;
        }
        </style>
        """
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




# --- database flush ---
    st.markdown("---")
    if st.button("🗑️ Reset Project", type="primary"):
        with st.spinner("Flushing Database & Memory..."):
            try:
                # Clear Postgres Tables
                with db_engine.connect() as conn:
                    result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
                    tables = [row[0] for row in result]
                    
                    # Drop each table
                    for t in tables:
                        conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                    conn.commit()
                
                # Clear AI Memory
                if llm_engine:
                    llm_engine.clear_knowledge_base()
                
                # Clear Session State
                cached_llm_call.clear()
                st.session_state.clear()
                
                
                st.toast("Project Reset Successfully!", icon="🧹")
                st.rerun()
                
            except Exception as e:
                st.error(f"Reset Failed: {e}")

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
                table_name = f.name.split(".")[0].replace(" ", "_").lower()

                try:
                    df = ingest_cached(f)
                    repo.save_table(df, table_name)
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Error loading {f.name}: {e}")

            if success_count:
                st.success(f"Successfully uploaded {success_count} files to Database.")

    st.markdown("---")
    st.subheader("🔗 Merge Tables")

    table_list = repo.list_tables()

    if not table_list:
        st.info("No data found. Please upload files above.")
    else:
        c1, c2, c3 = st.columns(3)
        left_table = c1.selectbox("Left Table", table_list)
        right_table = c2.selectbox("Right Table", ["(None)"] + table_list)

        join_col = None
        if right_table != "(None)" and left_table:
            try:
                l_cols = repo.get_columns(left_table)
                r_cols = repo.get_columns(right_table)
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
                if not left_table:
                    st.error("Select a Left Table.")
                    st.stop()
                if right_table == "(None)":
                    merged_df = repo.load_table(left_table)
                else:
                    if not join_col:
                        st.error("Select a Join Key.")
                        st.stop()
                    merged_df = repo.merge_tables(left_table, right_table, join_col)

                if not merged_df.empty:
                    st.session_state["preview_df"] = merged_df
                    st.dataframe(merged_df.head(5))
                    st.success(f"Preview generated: {len(merged_df)} rows")
            except Exception as e:
                st.error(f"Merge Failed: {e}")

        if "preview_df" in st.session_state:
            if col_btn_2.button("💾 Save as Master"):
                try:
                    repo.save_table(st.session_state["preview_df"], "master_view")
                    st.session_state["master_df"] = st.session_state["preview_df"]
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
                if llm_engine:
                    st.info(llm_engine.analyze_visuals(summary))
                else:
                    st.error("❌ LLM Engine not available.")

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
    mode_choice = c4.radio("Mode", ["Validation", "Prediction (30 Days)"])
    
    if st.button("🚀 Run Prediction"):
        if not ml_engine:
            st.error("ML Engine not available.")
            st.stop()
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
                'mode': mode_choice,
                'context': data_context,
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

        result_context = res.get('context', 'Retail')

        if data_context == "Financial":
            title_suffix = "Log Returns (Relative Change)"
            y_axis_label = "Log Return"
        else:
            title_suffix = "Absolute Values"
            y_axis_label = "Value"

        fig.update_layout(
            title=f"Forecast Output — {title_suffix}",
            yaxis_title=y_axis_label,
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # ---- METRICS ----
        metrics = res['metrics']
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                if k == "Direction":
                    col.metric("Trend Signal", f"{v:.1f}%")
                if k == "SMAPE":
                    if v < 0.1:
                        col.metric("SMAPE", f"{v:.4f}%")
                    else:
                        col.metric("SMAPE", f"{v:.1f}%")
                elif k == "MAE":
                    col.metric("MAE", f"{v:.6f}")

                else:
                    # col.metric(k, f"{v:.2f}")
                    if abs(v) < 0.01:
                        col.metric(k, f"{v:.6f}")
                    else:
                        col.metric(k, f"{v:.2f}")
                                        
        else:
            st.info("Future forecast — no validation metrics available.")

        if data_context == "Financial":
            st.caption("⚠ Financial forecasts are statistical signals, not trading advice.")
        
        st.subheader("AI Strategist")
        if st.button("✨ Analyze Forecast"):
            with st.spinner("Consulting..."):
                metrics_text = "N/A (Future Forecast)"
                if res['metrics']:
                    metrics_text = ", ".join([f"{k}: {v:.2f}" for k, v in res['metrics'].items()])
                
                last_history = res['train']['y'].iloc[-1]
                if abs(last_history) < 1e-4:
                    last_value_text = "near zero (neutral)"
                else:
                    last_value_text = f"{last_history:.4f}"

                forecast_vals = res['forecast']['yhat']

                avg_mag = forecast_vals.abs().mean()

                if avg_mag < 0.001:
                    forecast_behavior = "low-amplitude, mean-reverting"
                elif avg_mag < 0.01:
                    forecast_behavior = "moderate fluctuations"
                else:
                    forecast_behavior = "high volatility movement"

                                
                summary = f"""
                DATA CONTEXT: {data_context}
                FORECAST MODE: {res['mode']}
                PERFORMANCE METRICS: {metrics_text}
                LAST KNOWN VALUE: {last_value_text}
                FORECAST CHARACTER: {forecast_behavior}
                """
                
                if llm_engine:
                    st.info(cached_llm_call(summary, data_context))
                else:
                    st.error("LLM Engine not available.")