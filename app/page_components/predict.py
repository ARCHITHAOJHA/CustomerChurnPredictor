import streamlit as st
import pandas as pd
from pathlib import Path
import random

# Optional plotting / explainability libs — import lazily and handle missing packages
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import shap
except Exception:
    shap = None


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


def render_overlay_select(field_name, options, *, key, help=None):
    """Render a visible label overlay and a native selectbox with collapsed label.

    The overlay is a simple HTML div placed before the widget and styled via CSS
    so it visually appears inside the green select box while the dropdown still
    shows only option values when opened.
    """
    value = st.selectbox("", options, key=key, label_visibility="collapsed", help=help)
    st.markdown(f"<div class='field-overlay'>{field_name}</div>", unsafe_allow_html=True)
    return value


def create_compatible_demo_models():
    """Create demo models that accept the same columns as the prediction form."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from joblib import dump as _jl_dump
    except Exception as e:
        return False, f"Required ML packages are not available: {e}"

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        n = 700
        rows = []
        for _ in range(n):
            tenure = random.randint(0, 72)
            monthly = round(random.uniform(18.0, 125.0), 2)
            total = round(monthly * max(tenure, 1) * random.uniform(0.7, 1.2), 2)

            row = {
                "gender": random.choice(["Male", "Female"]),
                "SeniorCitizen": random.choice([0, 1]),
                "Partner": random.choice(["Yes", "No"]),
                "Dependents": random.choice(["Yes", "No"]),
                "tenure": tenure,
                "PhoneService": random.choice(["Yes", "No"]),
                "MultipleLines": random.choice(["Yes", "No"]),
                "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
                "OnlineSecurity": random.choice(["Yes", "No"]),
                "OnlineBackup": random.choice(["Yes", "No"]),
                "DeviceProtection": random.choice(["Yes", "No"]),
                "TechSupport": random.choice(["Yes", "No"]),
                "StreamingTV": random.choice(["Yes", "No"]),
                "StreamingMovies": random.choice(["Yes", "No"]),
                "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": random.choice(["Yes", "No"]),
                "PaymentMethod": random.choice([
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]),
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
            rows.append(row)

        X = pd.DataFrame(rows)

        # Synthetic target with realistic churn tendencies plus small randomness.
        y = []
        for _, r in X.iterrows():
            risk = 0.05
            if r["Contract"] == "Month-to-month":
                risk += 0.22
            if r["tenure"] < 12:
                risk += 0.18
            if r["InternetService"] == "Fiber optic":
                risk += 0.08
            if r["TechSupport"] == "No":
                risk += 0.08
            if r["OnlineSecurity"] == "No":
                risk += 0.07
            if r["MonthlyCharges"] > 85:
                risk += 0.07
            if r["Partner"] == "Yes":
                risk -= 0.05
            if r["Dependents"] == "Yes":
                risk -= 0.04
            if r["Contract"] == "Two year":
                risk -= 0.10

            risk = max(0.02, min(0.95, risk))
            y.append(1 if random.random() < risk else 0)

        cat_cols = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        logistic_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1200))]
        )
        logistic_pipeline.fit(X, y)
        _jl_dump(logistic_pipeline, MODELS_DIR / "logistic_model.pkl")

        rf_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=220, random_state=42))]
        )
        rf_pipeline.fit(X, y)
        _jl_dump(rf_pipeline, MODELS_DIR / "rf_model.pkl")

        try:
            from xgboost import XGBClassifier

            xgb_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
                ]
            )
            xgb_pipeline.fit(X, y)
            _jl_dump(xgb_pipeline, MODELS_DIR / "xgb_model.pkl")
        except Exception:
            # XGBoost is optional.
            pass

        return True, "Demo models created"
    except Exception as e:
        return False, f"Failed to create demo models: {e}"


def render_prediction_page():
    """Churn prediction page."""
    st.markdown("<h1 style='font-size:40px; margin-bottom:0.25rem;'>Predict Customer Churn</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; font-weight:700; color:#1f2937; margin-top:0.5rem;'>Enter customer details below and choose a model to generate a churn probability.</p>", unsafe_allow_html=True)

    model_choice = render_overlay_select("Select a model", ["Logistic Regression", "Random Forest", "XGBoost"], key="model_choice")
    st.markdown("---")

    with st.form("input_form"):
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="Number of months customer has stayed with provider")

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("**CUSTOMER DEMOGRAPHICS**")
            gender = render_overlay_select("Gender", ["Male", "Female"], key="gender", help="Customer gender")
            senior = render_overlay_select("Senior Citizen", [0, 1], key="senior_citizen", help="1 if customer is a senior citizen")
            partner = render_overlay_select("Partner", ["Yes", "No"], key="partner", help="Does the customer have a partner?")
            dependents = render_overlay_select("Dependents", ["Yes", "No"], key="dependents", help="Does the customer have dependents?")
            monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 60.0, help="Customer's monthly bill amount")
            total = st.number_input("Total Charges", 0.0, 100000.0, 1200.0, help="Total charges to date for the customer")

        with right_col:
            st.subheader("**SERVICES SUBSCRIBED**")
            phoneservice = render_overlay_select("Phone Service", ["Yes", "No"], key="phone_service", help="Does the customer have phone service?")
            multiplelines = render_overlay_select("Multiple Lines", ["Yes", "No"], key="multiple_lines", help="Multiple phone lines?")
            internet = render_overlay_select("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service", help="Type of internet service")
            onlinesecurity = render_overlay_select("Online Security", ["Yes", "No"], key="online_security", help="Online security subscription")
            onlinebackup = render_overlay_select("Online Backup", ["Yes", "No"], key="online_backup", help="Online backup subscription")
            deviceprotection = render_overlay_select("Device Protection", ["Yes", "No"], key="device_protection", help="Device protection subscription")
            streamingtv = render_overlay_select("Streaming TV", ["Yes", "No"], key="streaming_tv", help="Streaming TV subscription")
            streamingmovies = render_overlay_select("Streaming Movies", ["Yes", "No"], key="streaming_movies", help="Streaming movies subscription")
            contract = render_overlay_select("Contract", ["Month-to-month", "One year", "Two year"], key="contract", help="Type of service contract")
            techsupport = render_overlay_select("Tech Support", ["Yes", "No"], key="tech_support", help="Tech support subscription")
            paperless = render_overlay_select("Paperless Billing", ["Yes", "No"], key="paperless_billing", help="Is paperless billing enabled?")
            payment = render_overlay_select(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
                help="Customer's payment method",
            )

        _, btn_c2, _ = st.columns([1, 1, 1])
        with btn_c2:
            submit = st.form_submit_button("Predict Churn")

    if submit:
        logistic_model = safe_load_model("logistic_model.pkl")
        rf_model = safe_load_model("rf_model.pkl")
        xgb_model = safe_load_model("xgb_model.pkl")
        
        models_available = any([logistic_model, rf_model, xgb_model])
        
        if not models_available:
            with st.spinner("Models missing. Creating demo models..."):
                ok, msg = create_compatible_demo_models()
            if not ok:
                st.error(msg)
                return

            logistic_model = safe_load_model("logistic_model.pkl")
            rf_model = safe_load_model("rf_model.pkl")
            xgb_model = safe_load_model("xgb_model.pkl")
            models_available = any([logistic_model, rf_model, xgb_model])

            if not models_available:
                st.error("Prediction still unavailable after demo-model setup. Please verify dependencies in requirements.txt.")
                return
            st.success("Demo models created automatically. Running prediction...")

        data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phoneservice],
            "MultipleLines": [multiplelines],
            "InternetService": [internet],
            "OnlineSecurity": [onlinesecurity],
            "OnlineBackup": [onlinebackup],
            "DeviceProtection": [deviceprotection],
            "TechSupport": [techsupport],
            "StreamingTV": [streamingtv],
            "StreamingMovies": [streamingmovies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
        })

        if model_choice == "Random Forest":
            model = rf_model
        elif model_choice == "XGBoost":
            model = xgb_model
        else:
            model = logistic_model

        if model is None:
            st.error("Selected model is not available. Choose another model or add the model file.")
        else:
            try:
                prob = model.predict_proba(data)[0][1]
            except Exception:
                try:
                    preprocessor = getattr(rf_model, "named_steps", {}).get("preprocessor") if rf_model else None
                    if preprocessor is not None:
                        X = preprocessor.transform(data)
                        prob = model.predict_proba(X)[0][1]
                    else:
                        st.error("Model requires preprocessed input but preprocessor not found.")
                        prob = None
                except Exception:
                    st.error("Prediction failed — model input mismatch.")
                    prob = None

            if prob is not None:
                colA, colB = st.columns([2, 1])
                with colA:
                    st.metric("Churn Probability", f"{prob*100:.1f}%")
                    if prob > 0.6:
                        st.error("High Risk Customer")
                    elif prob > 0.3:
                        st.warning("Medium Risk Customer")
                    else:
                        st.success("Low Risk Customer")

                with colB:
                    st.subheader("Input Summary")
                    st.table(data.T)
                
                # Store prediction result in session state for analytics page
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                
                prediction_record = {
                    "model": model_choice,
                    "churn_probability": prob,
                    "risk_level": "High Risk" if prob > 0.6 else ("Medium Risk" if prob > 0.3 else "Low Risk"),
                    "input_data": data.to_dict(orient="records")[0],
                }
                st.session_state.prediction_history.append(prediction_record)
                
                st.success("Prediction saved to Analytics!")

                if model_choice == "Random Forest" and rf_model is not None:
                    if shap is None or plt is None:
                        st.info("SHAP or matplotlib not available — install `shap` and `matplotlib` to enable explanations.")
                    else:
                        try:
                            explainer = shap.Explainer(rf_model.named_steps["model"])
                            X_trans = rf_model.named_steps["preprocessor"].transform(data)
                            shap_values = explainer(X_trans)
                            fig = plt.figure()
                            shap.plots.waterfall(shap_values[0, :, 1], show=False)
                            st.pyplot(fig)
                        except Exception:
                            st.info("SHAP explanation not available for this input.")
