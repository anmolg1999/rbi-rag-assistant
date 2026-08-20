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

# ──────────────────────────────────────────────
# Premium CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 14, 28) 0%, rgb(25, 25, 50) 90%);
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-title {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 237, 234, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    
    .section-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Global Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Complete historical record of all queries across all user sessions. Data is permanent and will not reset on logout.</div>', unsafe_allow_html=True)

if df.empty:
    st.info("No analytics data available yet. Start asking questions in the chat!")
    st.stop()

# --- Key Metrics ---
total_queries = len(df)
avg_latency = df['total_time_sec'].mean()
total_tokens = df['input_tokens'].sum() + df['output_tokens'].sum()
rag_queries = len(df[df['source_type'] == 'rag'])
general_queries = len(df[df['source_type'] == 'general'])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_queries}</div><div class="metric-label">Total Queries</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_latency:.2f}s</div><div class="metric-label">Avg Latency</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_tokens:,}</div><div class="metric-label">Tokens Used</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rag_queries}</div><div class="metric-label">RAG Hits</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{general_queries}</div><div class="metric-label">General/JAIIB</div></div>', unsafe_allow_html=True)

# --- Charts Row 1 ---
st.markdown('### 📈 Performance & Usage Patterns')
st.markdown('<div class="section-container">', unsafe_allow_html=True)
colA, colB = st.columns(2)

with colA:
    st.markdown("#### System Latency (Seconds)")
    chart_data = df[['timestamp', 'retrieval_time_sec', 'llm_time_sec']].set_index('timestamp')
    st.area_chart(chart_data, color=["#fbc2eb", "#a6c1ee"])

with colB:
    st.markdown("#### Intent Routing Distribution")
    # Rename sources for better display
    source_mapping = {
        'rag': 'RBI Documents (RAG)',
        'general': 'General Banking & JAIIB',
        'live_rate': 'Live RBI Rates',
        'meta': 'Chatbot Features',
        'out_of_scope': 'Out of Scope'
    }
    
    source_counts = df['source_type'].map(source_mapping).value_counts().reset_index()
    source_counts.columns = ['Intent Category', 'Count']
    
    # Premium Altair Donut Chart
    pie_chart = alt.Chart(source_counts).mark_arc(innerRadius=70, cornerRadius=5).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Intent Category", type="nominal", scale=alt.Scale(scheme='pastel1')),
        tooltip=['Intent Category', 'Count']
    ).properties(height=350, background='transparent')
    
    st.altair_chart(pie_chart, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Full History Explorer ---
st.markdown('### 🗄️ Permanent Global Conversation History')
st.markdown("Browse through every question ever asked across all user sessions. This data is securely saved to the database.")

st.markdown('<div class="section-container">', unsafe_allow_html=True)

# Search functionality
search_term = st.text_input("🔍 Search Historical Queries...", "")

display_df = df[['timestamp', 'user_query', 'answer', 'source_type', 'total_time_sec', 'input_tokens', 'output_tokens']].copy()

if search_term:
    display_df = display_df[display_df['user_query'].str.contains(search_term, case=False, na=False) | 
                            display_df['answer'].str.contains(search_term, case=False, na=False)]

display_df['source_type'] = display_df['source_type'].map(source_mapping)
display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
display_df['total_time_sec'] = display_df['total_time_sec'].round(2).astype(str) + "s"

# Format columns nicely
display_df = display_df.rename(columns={
    'timestamp': 'Date & Time',
    'user_query': 'User Question',
    'answer': 'Bot Answer',
    'source_type': 'Routed Intent',
    'total_time_sec': 'Response Time',
    'input_tokens': 'In Tokens',
    'output_tokens': 'Out Tokens'
})

st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Bot Answer": st.column_config.TextColumn("Bot Answer", width="large"),
        "User Question": st.column_config.TextColumn("User Question", width="medium"),
    },
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)
