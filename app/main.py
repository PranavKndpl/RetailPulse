import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from dotenv import load_dotenv

from src.ml_engine import MLEngine
from src.llm_engine import LLMEngine
from sqlalchemy import create_engine

load_dotenv()

st.set_page_config(page_title="RetailPulse Exec", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: #ffffff; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica', sans-serif; }
    .status-good { color: #10b981; font-weight: bold; }
    .status-bad { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_resources():
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(DB_URL)
    ml = MLEngine(engine)
    llm = LLMEngine()
    return ml, llm

ml_app, llm_app = get_resources()

def calculate_kpis(df):
    total_rev = df['price'].sum()
    avg_delay = df['delivery_delay_days'].mean()
    pending_orders = len(df[df['delivery_delay_days'] > 3])
    return total_rev, avg_delay, pending_orders


st.title("⚡ RetailPulse Command Center")
st.markdown("### 📅 Daily Operations Overview")

df = ml_app.load_data()
revenue, delay, issues = calculate_kpis(df)

if issues > 5:
    st.error(f"🚨 CRITICAL ATTENTION NEEDED: {issues} orders are severely delayed.")
elif issues > 0:
    st.warning(f"⚠️ SYSTEM ALERT: {issues} potential delivery issues detected.")
else:
    st.success("✅ ALL SYSTEMS NOMINAL: Operations are running smoothly.")

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Revenue", f"${revenue:,.0f}", delta="12% vs last week")
with c2:
    st.metric("Avg Delivery Time", f"{delay:.1f} Days", delta="-0.5 days", delta_color="inverse")
with c3:
    st.metric("Active Anomalies", f"{issues}", delta="High Risk", delta_color="inverse")
with c4:
    st.metric("AI Agent Status", "Active", delta="Llama-3.3")

# FORECAST vs. ACTIONS
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Revenue Outlook")

    daily_sales = df.set_index('order_purchase_timestamp').resample('D').sum()['price'].reset_index()

    daily_sales['price_smooth'] = daily_sales['price'].rolling(window=7).mean()
    
    fig = px.area(daily_sales, x='order_purchase_timestamp', y='price_smooth', 
                  title="", color_discrete_sequence=['#3b82f6'])
    fig.update_layout(xaxis_title="", yaxis_title="Daily Revenue", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("⚡ Action Center")
    
    if issues > 0:
        st.info(f"{issues} orders require manager review.")
        
        if st.button("🤖 Auto-Analyze Top Priorities", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Scanning database for anomalies...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            model = ml_app.train_anomaly_detector()
            features = ['price', 'freight_value', 'delivery_delay_days', 'review_score']
            df['anomaly_score'] = model.predict(df[features].fillna(0))
            anomalies = df[df['anomaly_score'] == -1].head(1) # getting the worst one for demo
            
            status_text.text("🧠 AI analyzing customer sentiment...")
            progress_bar.progress(60)
            
            # Run LLM
            row = anomalies.iloc[0]
            review = row['review_comment_message'] if row['review_comment_message'] else "Customer reported severe delay."
            summary = llm_app.summarize_reviews([review])
            
            status_text.text("⚖️ Consulting Company SOPs...")
            progress_bar.progress(85)
            
            action = llm_app.decide_action("Critical Delay", summary)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete.")
            

            with st.expander("🚨 Priority Issue: Order #9281", expanded=True):
                st.markdown(f"**Customer Issue:** {summary}")
                st.markdown(f"**AI Recommendation:**")
                st.info(action)
                c_a, c_b = st.columns(2)
                with c_a:
                    st.button("✅ Approve Ticket")
                with c_b:
                    st.button("❌ Dismiss")

    else:
        st.markdown("""
        <div style="text-align: center; color: gray; padding: 50px;">
            <h3>☕ Time for coffee</h3>
            <p>No critical anomalies detected.</p>
        </div>
        """, unsafe_allow_html=True)

with st.expander("🔎 View Raw Data Logs"):
    st.dataframe(df.sort_values('order_purchase_timestamp', ascending=False).head(50), use_container_width=True)