import streamlit as st
import pandas as pd
import altair as alt
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics import get_analytics_data, init_db

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
        color: white;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Live Analytics Dashboard")
st.markdown("Monitor performance, token usage, and user queries in real time.")

# Ensure DB is initialized
init_db()

# Load Data
@st.cache_data(ttl=5) # Refresh every 5 seconds
def load_data():
    data = get_analytics_data()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

if df.empty:
    st.info("No analytics data available yet. Start asking questions in the chat!")
    st.stop()

# --- Key Metrics ---
total_queries = len(df)
avg_latency = df['total_time_sec'].mean()
total_tokens = df['input_tokens'].sum() + df['output_tokens'].sum()
rag_queries = len(df[df['source_type'] == 'rag'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_queries}</div><div class="metric-label">Total Queries</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_latency:.2f}s</div><div class="metric-label">Avg Response Time</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_tokens:,}</div><div class="metric-label">Total Tokens Used</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rag_queries}</div><div class="metric-label">RAG Served</div></div>', unsafe_allow_html=True)

st.divider()

# --- Charts Row 1 ---
colA, colB = st.columns(2)

with colA:
    st.subheader("Performance Latency (sec)")
    # Line chart of total time over time
    chart_data = df[['timestamp', 'retrieval_time_sec', 'llm_time_sec']].set_index('timestamp')
    st.line_chart(chart_data)

with colB:
    st.subheader("Token Usage Over Time")
    token_data = df[['timestamp', 'input_tokens', 'output_tokens']].set_index('timestamp')
    st.bar_chart(token_data)

st.divider()

# --- Charts Row 2 ---
colC, colD = st.columns(2)

with colC:
    st.subheader("Source Usage (RAG vs Web)")
    source_counts = df['source_type'].value_counts().reset_index()
    source_counts.columns = ['Source Type', 'Count']
    
    # Altair Pie Chart
    pie_chart = alt.Chart(source_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Source Type", type="nominal", scale=alt.Scale(scheme='set2')),
        tooltip=['Source Type', 'Count']
    ).properties(height=300)
    st.altair_chart(pie_chart, use_container_width=True)

with colD:
    st.subheader("Recent Queries Logs")
    # Display recent queries in a scrollable dataframe
    display_df = df[['timestamp', 'user_query', 'source_type', 'total_time_sec']].copy()
    display_df['total_time_sec'] = display_df['total_time_sec'].round(2)
    display_df = display_df.rename(columns={
        'timestamp': 'Time',
        'user_query': 'Query',
        'source_type': 'Source',
        'total_time_sec': 'Latency (s)'
    })
    st.dataframe(display_df, use_container_width=True, height=300)
