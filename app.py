import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="GreenTech Sustainability AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = ['carbon_emissions', 'energy_output', 'renewability_index', 'cost_efficiency']
TARGET = 'sustainability'
MODEL_PATH = "lrmodel_sustainable.pkl"
DATA_PATH = "green_tech_data.csv"

# ---------------------------------------------------------
# Data & Model Loader
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        'carbon_emissions': rng.uniform(50, 400, n),
        'energy_output': rng.uniform(100, 1000, n),
        'renewability_index': rng.uniform(0.0, 1.0, n),
        'cost_efficiency': rng.uniform(0.5, 5.0, n),
    })
    score = (
        -0.023 * df['carbon_emissions']
        + 0.001 * df['energy_output']
        + 1.09 * df['renewability_index']
        - 1.25 * df['cost_efficiency']
        + 3.91
    )
    df['sustainability'] = (score > 0).astype(int)
    return df

@st.cache_resource
def load_or_train_model(df):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, "Loaded from pre-trained checkpoint (`lrmodel_sustainable.pkl`)"
        except Exception:
            pass
    
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, "Dynamically trained Logistic Regression model"

df_raw = load_dataset()
model, model_status = load_or_train_model(df_raw)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?w=500&q=80")
    st.markdown("### ⚙️ System Status")
    st.caption(f"**Model Status:** {model_status}")
    st.caption(f"**Dataset Rows:** {len(df_raw)} records")
    st.markdown("---")
    st.markdown("### 📊 Metric Boundaries")
    st.caption("Standard operational ranges:")
    st.markdown(
        "- **Carbon Emissions:** gCO2/kWh\n"
        "- **Energy Output:** MWh produced\n"
        "- **Renewability Index:** 0.0 to 1.0\n"
        "- **Cost Efficiency Index:** Score 0.5 to 5.0"
    )

# ---------------------------------------------------------
# Header & KPIs
# ---------------------------------------------------------
st.markdown('<p class="main-title">🌱 GreenTech Sustainability Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predictive AI classifier assessing clean energy solutions and environmental viability using Logistic Regression</p>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Mean Carbon (gCO2/kWh)", f"{df_raw['carbon_emissions'].mean():.1f}")
with kpi2:
    st.metric("Avg Energy Output", f"{df_raw['energy_output'].mean():.1f} MWh")
with kpi3:
    st.metric("Avg Renewability", f"{df_raw['renewability_index'].mean():.2f}")
with kpi4:
    sustainable_ratio = (df_raw['sustainability'].mean() * 100)
    st.metric("Sustainable Share", f"{sustainable_ratio:.1f}%")

st.markdown("---")

tab_predict, tab_batch, tab_eda, tab_metrics, tab_about = st.tabs([
    "🔮 Single Assessment",
    "📁 Batch Inference",
    "📊 Data Analytics",
    "📈 Model Diagnostics",
    "ℹ️ Documentation"
])

# ---------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Simulate Green Technology Parameters")
    st.write("Adjust operational metrics to evaluate whether the project achieves net sustainability compliance.")
    
    col_input, col_result = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        carbon = st.slider("Carbon Emissions (gCO2/kWh)", 40.0, 450.0, float(df_raw['carbon_emissions'].median()), 1.0)
        energy = st.slider("Energy Output (MWh)", 50.0, 1100.0, float(df_raw['energy_output'].median()), 5.0)
        renew = st.slider("Renewability Index", 0.0, 1.0, float(df_raw['renewability_index'].median()), 0.01)
        cost = st.slider("Cost Efficiency Index", 0.4, 5.0, float(df_raw['cost_efficiency'].median()), 0.05)
        
        sample_df = pd.DataFrame([{
            'carbon_emissions': carbon,
            'energy_output': energy,
            'renewability_index': renew,
            'cost_efficiency': cost
        }])

    with col_result:
        st.markdown("### Prediction Outcome")
        prediction = model.predict(sample_df)[0]
        probabilities = model.predict_proba(sample_df)[0]
        prob_sustainable = probabilities[1]

        if prediction == 1:
            st.success("✅ **SUSTAINABLE TECHNOLOGY APPROVED**")
            st.write("The parameter configuration achieves environmental viability standards.")
        else:
            st.error("⚠️ **NON-SUSTAINABLE DESIGN DETECTED**")
            st.write("Carbon intensity or cost-efficiency constraints exceed optimal green thresholds.")

        st.progress(float(prob_sustainable))
        st.caption(f"**Sustainability Probability Score:** `{prob_sustainable * 100:.2f}%`")

        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.barh(["Non-Sustainable", "Sustainable"], [probabilities[0], probabilities[1]], color=["#e76f51", "#2a9d8f"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig)

# ---------------------------------------------------------
# Tab 2: Batch Inference
# ---------------------------------------------------------
with tab_batch:
    st.subheader("Multi-Facility Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_uploader")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in FEATURES if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            batch_preds = model.predict(batch_df[FEATURES])
            batch_probs = model.predict_proba(batch_df[FEATURES])[:, 1]
            
            batch_df["Sustainability_Predicted"] = batch_preds
            batch_df["Confidence_Score"] = (batch_probs * 100).round(2)
            batch_df["Status"] = batch_df["Sustainability_Predicted"].map({1: "✅ Sustainable", 0: "❌ Non-Sustainable"})

            st.dataframe(batch_df, width="stretch")
            
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Evaluated CSV", data=csv_data, file_name="greentech_predictions.csv", mime="text/csv")
    else:
        st.info("Upload a batch CSV or use the preview below to inspect the expected schema format:")
        st.dataframe(df_raw[FEATURES].head(5), width="stretch")

# ---------------------------------------------------------
# Tab 3: Exploratory Data Analysis
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Data Distribution & Feature Correlates")
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown("**Class Balance (Sustainable vs. Non-Sustainable)**")
        fig_cb, ax_cb = plt.subplots(figsize=(4.5, 3))
        counts = df_raw[TARGET].value_counts().rename(index={0: 'Non-Sustainable', 1: 'Sustainable'})
        counts.plot(kind='bar', color=['#e76f51', '#2a9d8f'], ax=ax_cb)
        ax_cb.set_ylabel("Number of Projects")
        plt.xticks(rotation=0)
        st.pyplot(fig_cb)

    with col_eda2:
        st.markdown("**Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(4.5, 3))
        sns.heatmap(df_raw.corr(), annot=True, fmt=".2f", cmap="Greens", ax=ax_corr)
        st.pyplot(fig_corr)

    st.markdown("**Feature Spread across Sustainability Status**")
    feat_choice = st.selectbox("Select metric to inspect:", FEATURES)
    fig_box, ax_box = plt.subplots(figsize=(8, 3))
    sns.boxplot(data=df_raw, x=TARGET, y=feat_choice, palette=['#e76f51', '#2a9d8f'], ax=ax_box)
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(['Non-Sustainable (0)', 'Sustainable (1)'])
    st.pyplot(fig_box)

# ---------------------------------------------------------
# Tab 4: Model Diagnostics
# ---------------------------------------------------------
with tab_metrics:
    st.subheader("Mathematical Model Coefficients & Test Metrics")
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient (Weight)": model.coef_[0]
    }).sort_values("Coefficient (Weight)", ascending=False)
    
    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.markdown("#### Learned Logistic Coefficients")
        st.dataframe(coef_df, width="stretch")
        st.caption(f"**Model Intercept (Bias):** `{model.intercept_[0]:.4f}`")

    with col_m2:
        st.markdown("#### Feature Influence Direction")
        fig_coef, ax_coef = plt.subplots(figsize=(5, 3))
        colors = ['#2a9d8f' if w > 0 else '#e76f51' for w in coef_df["Coefficient (Weight)"]]
        ax_coef.barh(coef_df["Feature"], coef_df["Coefficient (Weight)"], color=colors)
        ax_coef.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax_coef.set_xlabel("Log-Odds Impact")
        st.pyplot(fig_coef)

# ---------------------------------------------------------
# Tab 5: Documentation
# ---------------------------------------------------------
with tab_about:
    st.markdown("""
    ### 🌿 About GreenTech AI Solution
    This dashboard applies **Logistic Regression** to determine net sustainability feasibility.
    
    #### Decision Function:
    $$P(Y=1 \\mid X) = \\frac{1}{1 + e^{-(\\beta_0 + \\sum \\beta_i X_i)}}$$
    """)
