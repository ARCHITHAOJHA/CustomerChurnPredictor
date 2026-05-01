import streamlit as st
import pandas as pd
from pathlib import Path

# Optional plotting libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def safe_load_model(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        from joblib import load as _jl_load
    except Exception:
        return None
    try:
        return _jl_load(str(path))
    except Exception:
        return None


def render_analytics_page():
    """Analytics dashboard page."""
    st.markdown("<h1 style='font-size:40px; margin-bottom:0.25rem;'>Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='margin-top:0.2rem; margin-bottom:0.8rem; color:#1f2937; font-weight:800; font-size:1.05rem;'>Model summaries, feature importance, and example performance metrics.</p>",
        unsafe_allow_html=True,
    )

    # ROC AUC Metrics
    metric_cols = st.columns(3)
    metric_cols[0].metric("Logistic Regression ROC AUC", "0.86")
    metric_cols[1].metric("Random Forest ROC AUC", "0.85")
    metric_cols[2].metric("XGBoost ROC AUC", "0.85")

    st.markdown("---")
    
    # User Predictions Analytics
    st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Your Predictions</p>", unsafe_allow_html=True)
    
    if "prediction_history" in st.session_state and len(st.session_state.prediction_history) > 0:
        predictions = st.session_state.prediction_history
        
        # Calculate statistics
        churn_probs = [p["churn_probability"] for p in predictions]
        risk_levels = [p["risk_level"] for p in predictions]
        model_usage = {}
        
        for p in predictions:
            model = p["model"]
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(predictions))
        col2.metric("Avg Churn Probability", f"{(sum(churn_probs)/len(churn_probs)*100):.1f}%")
        col3.metric("High Risk Count", len([r for r in risk_levels if "High" in r]))
        
        st.markdown("---")
        
        # Risk Level Distribution - Pie Chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            risk_counts = {
                "High Risk": len([r for r in risk_levels if "High Risk" in r]),
                "Medium Risk": len([r for r in risk_levels if "Medium Risk" in r]),
                "Low Risk": len([r for r in risk_levels if "Low Risk" in r]),
            }
            if plt is not None and sum(risk_counts.values()) > 0:
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = ["#ef4444", "#f59e0b", "#10b981"]
                ax.pie(
                    risk_counts.values(),
                    labels=risk_counts.keys(),
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                    textprops={"fontsize": 11, "weight": "bold"},
                )
                ax.set_title("Your Predictions - Risk Distribution", fontsize=13, fontweight="bold", color="#1f2937")
                st.pyplot(fig)
        
        with col2:
            risk_summary = pd.DataFrame({
                "Risk Level": risk_counts.keys(),
                "Count": risk_counts.values(),
            })
            st.dataframe(risk_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Model Usage - Bar Chart
        st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Model Usage</p>", unsafe_allow_html=True)
        
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            models = list(model_usage.keys())
            counts = list(model_usage.values())
            ax.bar(models, counts, color="#10b981", edgecolor="#065f46", linewidth=2)
            ax.set_ylabel("Number of Predictions", fontsize=11, fontweight="bold")
            ax.set_title("Predictions by Model", fontsize=13, fontweight="bold", color="#1f2937")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Recent Predictions Table
        st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Recent Predictions</p>", unsafe_allow_html=True)
        
        recent_df = pd.DataFrame([
            {
                "Model": p["model"],
                "Churn Probability": f"{p['churn_probability']*100:.1f}%",
                "Risk Level": p["risk_level"],
            }
            for p in predictions[-10:]  # Show last 10
        ])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No predictions made yet. Go to the Predict page and make some predictions to see results here!")
    
    st.markdown("---")
    st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Churn Distribution</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        churn_data = {"Churned": 27, "Retained": 73}
        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = ["#ef4444", "#10b981"]
            ax.pie(
                churn_data.values(),
                labels=churn_data.keys(),
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 12, "weight": "bold"},
            )
            ax.set_title("Customer Churn Ratio", fontsize=14, fontweight="bold", color="#1f2937")
            st.pyplot(fig)
    
    with col2:
        churn_summary = pd.DataFrame({
            "Status": ["Churned", "Retained"],
            "Count": [27, 73],
            "Percentage": ["27%", "73%"]
        })
        st.dataframe(churn_summary, use_container_width=True)
    
    st.markdown("---")
    
    # Model Comparison - Bar Chart
    st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Model Performance Comparison</p>", unsafe_allow_html=True)
    
    perf_df = pd.DataFrame({
        "Model": ["Logistic\nRegression", "Random\nForest", "XGBoost"],
        "ROC AUC": [0.86, 0.85, 0.85],
        "F1 Score": [0.64, 0.65, 0.63],
        "Precision": [0.52, 0.56, 0.55],
        "Recall": [0.84, 0.78, 0.75],
    })
    
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(perf_df))
        width = 0.2
        
        ax.bar([i - 1.5*width for i in x], perf_df["ROC AUC"], width, label="ROC AUC", color="#10b981")
        ax.bar([i - 0.5*width for i in x], perf_df["F1 Score"], width, label="F1 Score", color="#3b82f6")
        ax.bar([i + 0.5*width for i in x], perf_df["Precision"], width, label="Precision", color="#f59e0b")
        ax.bar([i + 1.5*width for i in x], perf_df["Recall"], width, label="Recall", color="#8b5cf6")
        
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Performance Metrics Comparison", fontsize=14, fontweight="bold", color="#1f2937")
        ax.set_xticks(x)
        ax.set_xticklabels(perf_df["Model"])
        ax.legend(loc="lower right")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Top Features</p>", unsafe_allow_html=True)
    
    rf_model = safe_load_model("rf_model.pkl")
    
    if rf_model is None:
        st.info("Random Forest model not available. Add `rf_model.pkl` to the project's `models/` folder to see feature importance and SHAP plots.")
    else:
        try:
            preprocessor = rf_model.named_steps["preprocessor"]
            classifier = rf_model.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out()
            importances = classifier.feature_importances_
            feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(feat_imp.head(15), use_container_width=True)
            
            with col2:
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(feat_imp.head(15)["feature"], feat_imp.head(15)["importance"], color="#10b981")
                    ax.invert_yaxis()
                    ax.set_xlabel("Importance", fontsize=11, fontweight="bold")
                    ax.set_title("Feature Importance (Top 15)", fontsize=13, fontweight="bold", color="#1f2937")
                    ax.grid(axis="x", alpha=0.3)
                    st.pyplot(fig)
        except Exception as e:
            st.info("Unable to render model insights — model may not include a scikit-learn pipeline with `preprocessor` and `model` steps.")

    st.markdown("---")
    
    # Detailed Performance Metrics
    st.markdown("<p style='color:#1f2937; font-weight:900; font-size:1.1rem; margin-bottom:0.5rem;'>Detailed Model Performance</p>", unsafe_allow_html=True)
    st.dataframe(perf_df, use_container_width=True)
