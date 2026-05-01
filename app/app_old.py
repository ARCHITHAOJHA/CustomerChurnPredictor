import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import matplotlib.pyplot as plt
import shap

import json
import hashlib
import urllib.parse
from typing import Dict

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# Minimal custom styling
st.markdown(
    """
    <style>
    .stForm { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .stSelectbox, .stNumberInput, .stSlider, .stTextInput { font-size: 1rem; padding: 8px 10px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 6px; height: 2.6em; font-size: 1rem; }
    .stMarkdown p { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    .card { padding: 12px; border-radius: 8px; background: rgba(255,255,255,0.03); color: inherit; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .hero-wrap { background: linear-gradient(135deg,#3944ab 0%, #5b63d6 55%, #5561f2 100%); padding: 38px 20px; border-radius: 18px; }
    .hero-card { max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 24px; padding: 28px 32px; display: flex; gap: 24px; align-items: center; box-shadow: 0 14px 36px rgba(16,24,40,0.18); }
    .hero-left { flex: 1; }
    .hero-right { flex: 1; display:flex; justify-content:center; }
    .hero-title { font-size: 40px; line-height:1.0; margin:0; color:#0f172a; font-weight:800; letter-spacing:-0.03em; }
    .hero-sub { margin-top:10px; color:#6b7280; font-size:15px; line-height:1.6; max-width: 470px; }
    .btn-hero { margin-top:16px; }
    .btn-hero a { background: #375ef7; color: #fff; padding: 10px 18px; border-radius: 999px; text-decoration:none; font-weight:600; }
    .site-nav { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .hero-logo { font-weight:800; color:#375ef7; font-size:20px; margin-right:auto; }
    .nav-badge, .nav-pill, .nav-current { display:inline-flex; align-items:center; border-radius:999px; padding:0.55rem 0.9rem; font-size:0.92rem; background: #f3f6ff; color:#5f6b7d; border:1px solid #e5eaf7; }
    .nav-current { background: rgba(55,94,247,0.12); color:#375ef7; font-weight:700; border-color: rgba(55,94,247,0.16); }
    .nav-pills { display:flex; gap:10px; flex-wrap:wrap; margin-top: 8px; }
    .nav-action-bar { display:flex; gap:10px; flex-wrap:wrap; margin-top: 12px; }
    .nav-action { display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:0.58rem 1rem; font-size:0.94rem; border:1px solid #dde3f4; background:#fff; color:#334155; box-shadow:0 4px 10px rgba(15,23,42,0.04); }
    .nav-action.primary { background: linear-gradient(135deg,#375ef7 0%, #5365f5 100%); color:#fff; border-color: transparent; }
    .nav-action.secondary { background:#f8fafc; }
    .nav-action:hover, .btn-hero a:hover { filter: brightness(0.98); }
    @media (max-width:900px){ .hero-card{flex-direction:column;text-align:center} .hero-right{order:-1} .hero-logo{margin-right:0} }
    /* Fade animation for page transitions */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
    .stApp > div[data-testid="stAppViewContainer"] { animation: fadeIn 360ms ease; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

USERS_FILE = BASE_DIR / "users.json"


def load_users() -> Dict[str, str]:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: Dict[str, str]):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def avatar_data_uri(name: str) -> str:
    initials = (name[:2] if name else "U").upper()
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 36 36'><rect rx='18' width='36' height='36' fill='%23375ef7'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='14' font-family='Arial' fill='white'>{initials}</text></svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


# Safe loader
def safe_load_model(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        return load(str(path))
    except Exception:
        return None


logistic_model = safe_load_model("logistic_model.pkl")
rf_model = safe_load_model("rf_model.pkl")
xgb_model = safe_load_model("xgb_model.pkl")

models_available = any([logistic_model, rf_model, xgb_model])

if "page" not in st.session_state:
    st.session_state.page = "main" if st.session_state.get("user") else "login"


def set_page(page_name):
    st.session_state.page = page_name


def render_nav():
    nav_cols = st.columns([1.35, 1.0, 1.0, 1.0, 1.15, 1.15])
    user = st.session_state.get("user")
    with nav_cols[0]:
        st.markdown('<div class="hero-logo" style="margin-top:0.1rem;">YOUR LOGO</div>', unsafe_allow_html=True)
    if user:
        with nav_cols[1]:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "main"
        with nav_cols[2]:
            if st.button("🔮 Predict", use_container_width=True):
                st.session_state.page = "predict"
        with nav_cols[3]:
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.page = "analytics"
        # avatar + dropdown menu
        with nav_cols[4]:
            avatar = avatar_data_uri(user)
            st.markdown(f'<img src="{avatar}" style="width:36px;height:36px;border-radius:50%;vertical-align:middle;margin-right:8px;"> <strong>{user}</strong>', unsafe_allow_html=True)
        with nav_cols[5]:
            choice = st.selectbox("", ["Account","Profile","Predict","Analytics","Logout"], key="user_menu")
            if choice != "Account":
                if choice == "Logout":
                    st.session_state.pop("user", None)
                    st.session_state.page = "home"
                elif choice == "Predict":
                    st.session_state.page = "predict"
                elif choice == "Analytics":
                    st.session_state.page = "analytics"
                elif choice == "Profile":
                    st.session_state.page = "main"
                # reset selection
                st.session_state.user_menu = "Account"
    else:
        with nav_cols[1]:
            if st.button("🔐 Login", use_container_width=True):
                st.session_state.page = "login"
        with nav_cols[2]:
            if st.button("✍️ Sign up", use_container_width=True):
                st.session_state.page = "signup"
        with nav_cols[3]:
            if st.button("🔮 Predict", use_container_width=True):
                st.session_state.page = "login"
        with nav_cols[4]:
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.page = "login"
        with nav_cols[5]:
            st.markdown('<div class="nav-current">Welcome</div>', unsafe_allow_html=True)


def render_home_page():
    hero_html = """
<div class="hero-wrap">
  <div class="hero-card">
    <div class="hero-left">
      <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
        <div class="hero-logo">YOUR LOGO</div>
                <div class="site-nav">
                    <span class="nav-pill">Home</span>
                    <span class="nav-pill">Predict</span>
                    <span class="nav-pill">Analytics</span>
        </div>
      </div>
      <h1 class="hero-title">BECOME A<br/><span style="color:#375ef7">DATA SCIENTIST</span></h1>
      <p class="hero-sub">Use customer data to predict churn, inspect dashboards, and move from insight to action.</p>
            <div class="nav-action-bar">
                <div class="nav-action primary">⬢ Get Started</div>
                <div class="nav-action secondary">✨ Explore Analytics</div>
            </div>
    </div>
    <div class="hero-right">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 220" width="320" height="220" role="img" aria-label="illustration">
                <defs>
                    <linearGradient id="panelGrad" x1="0" x2="1" y1="0" y2="1">
                        <stop offset="0%" stop-color="#eef2ff"/>
                        <stop offset="100%" stop-color="#dbe4ff"/>
                    </linearGradient>
                    <linearGradient id="screenGrad" x1="0" x2="1" y1="0" y2="1">
                        <stop offset="0%" stop-color="#3957f1"/>
                        <stop offset="100%" stop-color="#7b61ff"/>
                    </linearGradient>
                </defs>
                <rect rx="26" x="34" y="18" width="248" height="184" fill="url(#panelGrad)" />
                <rect x="84" y="52" width="90" height="118" rx="16" fill="#1f4cff" opacity="0.08" />
                <rect x="138" y="30" width="48" height="130" rx="10" fill="#ffffff" opacity="0.35" />
                <rect x="145" y="28" width="34" height="114" rx="8" fill="#334155" opacity="0.28" />
                <rect x="173" y="84" width="70" height="72" rx="12" fill="url(#screenGrad)" />
                <circle cx="208" cy="120" r="23" fill="#7dd3fc" opacity="0.85" />
                <path d="M208 120 L208 97 A23 23 0 0 1 229 111 Z" fill="#eef2ff" opacity="0.9" />
                <rect x="70" y="140" width="50" height="10" rx="5" fill="#38bdf8" opacity="0.9" />
                <rect x="70" y="156" width="78" height="8" rx="4" fill="#94a3b8" opacity="0.55" />
                <rect x="90" y="102" width="8" height="44" rx="4" fill="#375ef7" />
                <rect x="104" y="88" width="8" height="58" rx="4" fill="#64748b" />
                <rect x="118" y="116" width="8" height="30" rx="4" fill="#a78bfa" />
                <circle cx="78" cy="52" r="8" fill="#22c55e" opacity="0.9" />
                <circle cx="230" cy="52" r="10" fill="none" stroke="#38bdf8" stroke-width="4" opacity="0.9" />
      </svg>
    </div>
  </div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)
    home_cols = st.columns(4)
    with home_cols[0]:
                st.button("🔮 Go to Prediction", use_container_width=True, key="home_predict", on_click=set_page, args=("predict",))
    with home_cols[1]:
        st.button("📊 View Analytics", use_container_width=True, key="home_analytics", on_click=set_page, args=("analytics",))
    with home_cols[2]:
        st.button("🔐 Login", use_container_width=True, key="home_login")
    with home_cols[3]:
        st.button("✍️ Sign up", use_container_width=True, key="home_signup")


def render_login_page():
    st.markdown("<h2>Login</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        users = load_users()
        if username in users and users[username] == hash_password(password):
            st.session_state.user = username
            st.success(f"Welcome back, {username}!")
            st.session_state.page = "main"
        else:
            st.error("Invalid credentials. If you don't have an account please sign up.")


def render_signup_page():
    st.markdown("<h2>Sign up</h2>", unsafe_allow_html=True)
    with st.form("signup_form"):
        username = st.text_input("Choose username")
        password = st.text_input("Choose password", type="password")
        submitted = st.form_submit_button("Create account")
    if submitted:
        if not username:
            st.error("Please choose a username.")
        else:
            users = load_users()
            if username in users:
                st.error("Username already exists. Choose another.")
            else:
                users[username] = hash_password(password)
                save_users(users)
                st.session_state.user = username
                st.success(f"Account created. Welcome, {username}!")
                st.session_state.page = "main"


def render_main_page():
    st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;'><h1>Welcome, {st.session_state.get('user','')}</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔮 Predict", use_container_width=True):
            st.session_state.page = "predict"
    with c2:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
    with c3:
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.pop("user", None)
            st.session_state.page = "home"


def render_prediction_page():
    if not st.session_state.get("user"):
        st.warning("Please log in to access prediction features.")
        if st.button("Go to login"):
            st.session_state.page = "login"
        return

    st.markdown("<h1 style='font-size:40px; margin-bottom:0.25rem;'>Predict Customer Churn!</h1>", unsafe_allow_html=True)
    st.caption("Enter the customer details below and choose a model to generate a churn probability.")
    model_choice = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "XGBoost"])
    st.markdown("---")

    with st.form("input_form"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="Number of months customer has stayed with provider")

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer gender")
            senior = st.selectbox("Senior Citizen", [0, 1], help="1 if customer is a senior citizen")
            partner = st.selectbox("Partner", ["Yes", "No"], help="Does the customer have a partner?")
            dependents = st.selectbox("Dependents", ["Yes", "No"], help="Does the customer have dependents?")
            monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 60.0, help="Customer's monthly bill amount")
            total = st.number_input("Total Charges", 0.0, 100000.0, 1200.0, help="Total charges to date for the customer")

        with right_col:
            st.subheader("Services Subscribed")
            phoneservice = st.selectbox("Phone Service", ["Yes", "No"], help="Does the customer have phone service?")
            multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], help="Multiple phone lines?")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service")
            onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help="Online security subscription")
            onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help="Online backup subscription")
            deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help="Device protection subscription")
            streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help="Streaming TV subscription")
            streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help="Streaming movies subscription")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Type of service contract")
            techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help="Tech support subscription")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], help="Is paperless billing enabled?")
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Customer's payment method",
            )

        btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
        with btn_c2:
            submit = st.form_submit_button("Predict Churn")

    if submit:
        if not models_available:
            st.error("Prediction not available — model files are missing in `models/`.")
        else:
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

                    if model_choice == "Random Forest" and rf_model is not None:
                        try:
                            explainer = shap.Explainer(rf_model.named_steps["model"])
                            X_trans = rf_model.named_steps["preprocessor"].transform(data)
                            shap_values = explainer(X_trans)
                            fig = plt.figure()
                            shap.plots.waterfall(shap_values[0, :, 1], show=False)
                            st.pyplot(fig)
                        except Exception:
                            st.info("SHAP explanation not available for this input.")


def render_analytics_page():
    if not st.session_state.get("user"):
        st.warning("Please log in to access analytics.")
        if st.button("Go to login"):
            st.session_state.page = "login"
        return

    st.markdown("<h1 style='font-size:40px; margin-bottom:0.25rem;'>Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Model summaries, feature importance, and example performance metrics.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Logistic Regression ROC AUC", "0.86")
    metric_cols[1].metric("Random Forest ROC AUC", "0.85")
    metric_cols[2].metric("XGBoost ROC AUC", "0.85")

    st.markdown("---")
    if rf_model is None:
        st.info("Random Forest model not available. Add `rf_model.pkl` to the project's `models/` folder to see feature importance and SHAP plots.")
    else:
        try:
            preprocessor = rf_model.named_steps["preprocessor"]
            classifier = rf_model.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out()
            importances = classifier.feature_importances_
            feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

            st.markdown("**Top features**")
            st.table(feat_imp.head(15))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feat_imp.head(15)["feature"], feat_imp.head(15)["importance"], color="#4c78a8")
            ax.invert_yaxis()
            st.pyplot(fig)
        except Exception:
            st.info("Unable to render model insights — model may not include a scikit-learn pipeline with `preprocessor` and `model` steps.")

    perf_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC AUC": [0.86, 0.85, 0.85],
        "F1 Score": [0.64, 0.65, 0.63],
        "Precision": [0.52, 0.56, 0.55],
        "Recall": [0.84, 0.78, 0.75],
    })
    st.subheader("Model Performance (example)")
    st.dataframe(perf_df)

# Page router
render_nav()

page = st.session_state.page
if page == "predict":
    render_prediction_page()
elif page == "analytics":
    render_analytics_page()
elif page == "login":
    render_login_page()
elif page == "signup":
    render_signup_page()
elif page == "main":
    render_main_page()
else:
    render_home_page()
