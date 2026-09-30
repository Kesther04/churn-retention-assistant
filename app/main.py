import sys
from pathlib import Path

# Add project root directory to python path for modular imports
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.models.predict import predict_churn_risk
from src.agent.agent import get_retention_agent

load_dotenv()

# ==========================================
# 1. Page Configuration & Custom CSS (Purple Theme + Hero Styling)
# ==========================================

st.set_page_config(
    page_title="Churn Retention Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling mirroring reference UI aesthetics with Purple accent
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,600;0,800;1,600;1,800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Status Pill Badge */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        background-color: #F3E8FF;
        color: #7C3AED;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid #E9D5FF;
        margin-bottom: 1rem;
    }
    
    /* Hero Typography */
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 2.75rem;
        font-weight: 800;
        font-style: italic;
        color: #7C3AED;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 1.25rem;
    }
    
    .hero-description {
        font-size: 1rem;
        color: #475569;
        max-width: 720px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Containers */
    .purple-card {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%);
        color: white;
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Metric Typography */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 0.5rem;
    }
    
    /* Playbook Source Tag */
    .source-tag {
        display: inline-block;
        background-color: #F3E8FF;
        color: #6B21A8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin-top: 8px;
        border: 1px solid #E9D5FF;
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background: #7C3AED !important;
        color: white !important;
        border-radius: 9999px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        background: #6D28D9 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.54) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    try:
        st.session_state.agent = get_retention_agent()
    except Exception as e:
        st.session_state.agent = None

# ==========================================
# 2. Hero Section
# ==========================================

st.markdown('<div class="pill-badge">⚡ AI RETENTION CONTROL</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Predict customer risk.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Retention Assistant brings insights together.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-description">An AI-powered retention platform designed for telecommunications teams. Analyze real-time subscriber profiles, predict churn probabilities, and execute playbook-backed strategies seamlessly.</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# 3. Main Dashboard Layout (Sidebar Form + Main Content)
# ==========================================

with st.sidebar:
    st.markdown("### 📋 Customer Details")
    st.caption("Configure subscriber attributes to calculate risk score.")
    
    with st.form("customer_profile_form"):
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=2)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=89.85)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=179.70)
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        
        submit_btn = st.form_submit_button("Predict Churn Risk", use_container_width=True)

# Main Grid Layout: Risk Analytics & Agent Chat
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📊 Risk Diagnosis")
    
    if submit_btn:
        profile_dict = {
            "tenure": tenure,
            "Contract": contract,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security
        }
        with st.spinner("Evaluating machine learning models..."):
            try:
                # 1. Churn Probability Calculation
                prediction_result = predict_churn_risk(profile_dict)
                
                # Extract the float from the returned dict
                churn_prob = prediction_result.get("churn_probability", 0.0)
                risk_pct = round(churn_prob * 100, 1)
                
                # Display Card
                st.markdown(f"""
                <div class="purple-card">
                    <div style="text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; opacity: 0.9;">Calculated Risk Level</div>
                    <div class="metric-value">{risk_pct}%</div>
                    <div style="margin-top: 0.75rem; font-size: 0.9rem; opacity: 0.9;">
                        {"🚨 High Risk Customer - Immediate Retention Action Advised" if risk_pct >= 50 else "✅ Low Risk Customer - Standard Engagement"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. Risk Factors Analysis (Native Streamlit Bar Chart)
                st.markdown("#### Top Risk Contributing Factors")
                
                risk_factors = []
                if contract == "Month-to-month":
                    risk_factors.append({"Factor": "Month-to-month Contract", "Impact Score": 0.35})
                if payment_method == "Electronic check":
                    risk_factors.append({"Factor": "Electronic Check Payment", "Impact Score": 0.25})
                if tenure <= 6:
                    risk_factors.append({"Factor": "Short Tenure (≤ 6 mos)", "Impact Score": 0.22})
                if monthly_charges > 70:
                    risk_factors.append({"Factor": "High Monthly Charge", "Impact Score": 0.18})
                if tech_support == "No":
                    risk_factors.append({"Factor": "No Tech Support Service", "Impact Score": 0.12})

                if not risk_factors:
                    risk_factors = [{"Factor": "Standard Baseline Profile", "Impact Score": 0.05}]
                    
                df_factors = pd.DataFrame(risk_factors).set_index("Factor")
                
                # Render using native Streamlit chart with purple color
                st.bar_chart(df_factors["Impact Score"], color="#7C3AED", horizontal=True)
                
            except Exception as e:
                st.error(f"Error computing prediction: {str(e)}")
    else:
        st.info("👈 Fill in customer details in the sidebar and click **Predict Churn Risk** to generate analytics.")

with col2:
    st.markdown("### 💬 Retention Agent Chat")
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.markdown(f'<div class="source-tag">📚 Source: {message["sources"]}</div>', unsafe_allow_html=True)

    # Chat Input Box
    if user_query := st.chat_input("Ask about retention strategies, talk tracks, or discount offers..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing playbooks & generating response..."):
                try:
                    if st.session_state.agent:
                        response_text = st.session_state.agent.invoke(user_query)
                    else:
                        response_text = "Agent unavailable. Please verify API configuration."
                    
                    # Extract source if present in response string
                    source_info = "Retention Playbook v2.4" if "playbook" in response_text.lower() else None
                    
                    st.markdown(response_text)
                    if source_info:
                        st.markdown(f'<div class="source-tag">📚 Source: {source_info}</div>', unsafe_allow_html=True)
                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": source_info
                    })
                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")