import streamlit as st
from pathlib import Path
import json
import hashlib
import urllib.parse

# Import page modules
from page_components.home import render_home_page
from page_components.main import render_main_page
from page_components.profile import render_profile_page
from page_components.edit_profile import render_edit_profile_page
from page_components.predict import render_prediction_page
from page_components.analytics import render_analytics_page

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
    /* Page background and typography */
    .stApp, .css-ffhzg2 { background: #ffffff !important; color: #0f172a; }
    body { background: #ffffff; }
    .stForm { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        .stSelectbox, .stNumberInput, .stSlider, .stTextInput { font-size: 1rem; padding: 8px 10px; }
        .stButton>button, .stFormSubmitButton>button { border-radius: 8px; height: 2.6em; font-size: 1rem; background: #10b981 !important; color: #fff !important; border: 1px solid #10b981 !important; box-shadow: 0 8px 24px rgba(16,185,129,0.12); font-weight:700; }
        .stButton>button:hover, .stFormSubmitButton>button:hover { filter: brightness(0.95); transform: translateY(-1px); }
        .stTextInput [data-baseweb="input"] {
            background: #dcfce7 !important;
            border: 2px solid #059669 !important;
            border-radius: 12px !important;
            min-height: 56px !important;
            align-items: center !important;
            overflow: hidden !important;
        }
        .stTextInput [data-baseweb="input"] > div,
        .stTextInput [data-baseweb="input"] > div > div {
            background: #dcfce7 !important;
        }
        .stTextInput [data-baseweb="input"] input {
            background: #dcfce7 !important;
            color: #14532d !important;
            height: 56px !important;
            line-height: 56px !important;
            padding: 0 18px !important;
            border: none !important;
            box-shadow: none !important;
            -webkit-text-fill-color: #14532d !important;
            caret-color: #14532d !important;
        }
        .stTextInput [data-baseweb="input"] button {
            background: #dcfce7 !important;
            color: #14532d !important;
            border: none !important;
            height: 56px !important;
            padding-right: 14px !important;
        }
        .stTextInput input::placeholder { color: #14532d !important; opacity: 0.78; }
        .stTextInput label, .stTextInput [data-testid="stWidgetLabel"] p { color: #14532d !important; font-weight: 700; }
        .auth-title { margin: 0 0 12px 0; font-size: 2rem; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; color: #064e3b; }
    .stMarkdown p { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
    .stApp > div[data-testid="stAppViewContainer"] { animation: fadeIn 360ms ease; }

    /* Sidebar styling: green buttons with visible text */
    .sidebar-nav { display: flex; flex-direction: column; gap: 12px; }
    .sidebar-item { display: flex; align-items: center; padding: 10px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s; background: #d1fae5; color: #065f46; }
    .sidebar-item:hover { background: #bbf7d0; }
    .sidebar-item.active { background: #86efac; color: #065f46; font-weight: 700; }
    .stSidebar { background: white !important; }
    .stSidebar .stButton { width: 100%; margin: 6px 0 !important; }
    .stSidebar .stButton>button { background: #10b981 !important; color: #ffffff !important; border: 2px solid #059669 !important; font-weight: 700 !important; border-radius: 8px !important; width: 100% !important; height: 50px !important; }
    .stSidebar [data-testid="baseButton-secondary"] { background: #10b981 !important; color: #ffffff !important; }
    .stSidebar .stButton>button svg { color: #ffffff !important; fill: #ffffff !important; }
    .stSidebar .stButton>button span, .stSidebar .stButton>button p { color: #ffffff !important; font-weight: 700 !important; }
    .stSidebar .stButton>button:hover { background: #059669 !important; color: #ffffff !important; }
    /* Make form controls and icons on content pages use green accents */
    /* Select boxes */
    .stSelectbox [data-baseweb="select"] { border-radius: 8px !important; }
    .stSelectbox [data-baseweb="select"] > div, .stSelectbox [data-baseweb="select"] > div > div { background: #10b981 !important; color: #ffffff !important; }
    .stSelectbox [data-baseweb="select"] button, .stSelectbox [data-baseweb="select"] svg { color: #ffffff !important; fill: #ffffff !important; }
    .stSelectbox .css-1d391kg { background: #10b981 !important; color: #fff !important; }
    .stSelectbox [data-testid="stWidgetLabel"] { display: none !important; }
    .stSelectbox [data-baseweb="select"] { position: relative !important; }
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] > div > div,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Make dropdowns click-select only (no typing in the internal input) */
    .stSelectbox [data-baseweb="select"] input {
        pointer-events: none !important;
        user-select: none !important;
        caret-color: transparent !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    /* Closed select: hide selected option text so only overlay field name is visible */
    .stSelectbox [data-baseweb="select"] > div > div:first-child,
    .stSelectbox [data-baseweb="select"] > div > div:first-child * {
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: none !important;
    }
    /* Keep dropdown indicator icon visible */
    .stSelectbox [data-baseweb="select"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    .stSelectbox [data-baseweb="select"]::before {
        position: absolute;
        left: 18px;
        top: 50%;
        transform: translateY(-50%);
        color: #ffffff;
        font-weight: 700;
        pointer-events: none;
        z-index: 1;
        white-space: nowrap;
    }
    .stSelectbox [data-baseweb="popover"] {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    .stSelectbox [data-baseweb="popover"] * {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    /* Show a tick mark on the currently selected dropdown option */
    .stSelectbox [data-baseweb="popover"] [role="option"] {
        position: relative !important;
        padding-right: 34px !important;
    }
    .stSelectbox [data-baseweb="popover"] [role="option"][aria-selected="true"]::after,
    .stSelectbox [role="listbox"] [role="option"][aria-selected="true"]::after {
        content: "\2713";
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: #ffffff !important;
        font-weight: 900;
        font-size: 16px;
        line-height: 1;
    }
    /* Visible overlay label for each field — inserted after the select and pulled into it */
    .field-overlay {
        position: relative;
        margin: -58px 0 34px 18px;
        z-index: 50;
        color: #ffffff;
        font-weight: 800;
        pointer-events: none;
        font-size: 16px;
        line-height: 1;
    }

    /* Number input steppers: make + / - icons green */
    .stNumberInput [data-baseweb="input"] button,
    .stNumberInput button,
    .stNumberInput svg,
    .stNumberInput [data-baseweb="input"] svg {
        color: #10b981 !important;
        fill: #10b981 !important;
    }
    /* rc-input-number handlers used by some Streamlit versions */
    .rc-input-number-handler-up, .rc-input-number-handler-down,
    .rc-input-number-handler-up::before, .rc-input-number-handler-down::before {
        color: #10b981 !important;
        fill: #10b981 !important;
        background: transparent !important;
    }
    /* Broader SVG/stroke targets for different Streamlit versions */
    .stNumberInput svg, .stNumberInput svg *,
    .stNumberInput button svg, .stNumberInput button svg *,
    .stNumberInput .css-1t5f0fr svg, .stNumberInput .css-1t5f0fr svg *,
    .rc-input-number .anticon, .rc-input-number .anticon svg, .rc-input-number svg, .rc-input-number svg * {
        color: #10b981 !important;
        fill: #10b981 !important;
        stroke: #10b981 !important;
    }
    /* Ensure the +/- pseudo elements also pick up color */
    .stNumberInput button::before, .stNumberInput button::after,
    .rc-input-number-handler-up::after, .rc-input-number-handler-down::after {
        color: #10b981 !important;
        background: transparent !important;
    }

    /* Text inputs and number inputs - green border/background */
    .stTextInput [data-baseweb="input"] { border: 2px solid #10b981 !important; background: rgba(16,185,129,0.06) !important; }

    /* Make field names clearly visible */
    .stNumberInput label,
    .stNumberInput [data-testid="stWidgetLabel"] p,
    .stSlider label,
    .stSlider [data-testid="stWidgetLabel"] p,
    .stTextInput label,
    .stTextInput [data-testid="stWidgetLabel"] p {
        color: #10b981 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    /* Improve readability for prediction results and summary */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: #10b981 !important;
        font-weight: 900 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] p,
    [data-testid="stMetricValue"] span {
        color: #000000 !important;
        font-weight: 900 !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stTable"] table,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        color: #0f172a !important;
        opacity: 1 !important;
        background: rgba(255, 255, 255, 0.78) !important;
    }

    /* Slider accents (rc-slider classes used by Streamlit) */
    .stSlider .rc-slider-rail { background: #e6f6ef !important; }
    .stSlider .rc-slider-track { background: #10b981 !important; }
    .stSlider .rc-slider-handle { border-color: #10b981 !important; background: #10b981 !important; box-shadow: 0 0 0 8px rgba(16,185,129,0.12) !important; }

    /* Metric and info boxes - use green for positive accents */
    .stMetric value, .stMetricDelta { color: #065f46 !important; }

    /* Ensure any inline SVG icons use green where appropriate */
    .stApp svg { fill: currentColor; }
    .sidebar-avatar { text-align: center; padding: 8px 10px; border-bottom: 1px solid rgba(0,0,0,0.06); }
    .sidebar-username { font-weight: 700; margin-top: 10px; font-size: 14px; color: #065f46; }
    /* Hide expanders and their content in sidebar but keep clickable buttons functional */
    .stSidebar .streamlit-expanderContent { display: none !important; }
    .stSidebar .stExpander { display: none !important; height: 0 !important; width: 0 !important; }
    .stSidebar .stExpander button { position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden; }

    .page-content { padding: 20px; }
    .top-header { padding: 12px 20px; border-bottom: 1px solid rgba(0,0,0,0.06); display: flex; align-items: center; justify-content: space-between; background: transparent; }
    .logo-text { font-weight: 900; color: #0f172a; font-size: 22px; }
    .hero-title { font-family: 'Lobster', cursive; }
    .hero-sub { font-family: 'Playfair Display', serif; font-size:28px; color:#1f2937; margin:12px auto 22px; max-width:900px; text-align:center; font-weight:700; line-height:1.35; display:block; }

    /* Hero buttons */
    .hero-cta { display:inline-block; padding: 12px 28px; border-radius: 12px; font-weight:700; }
    .hero-cta.primary { background: #10b981; color: white; box-shadow: 0 8px 24px rgba(16,185,129,0.16); }
    .hero-cta.secondary { background: white; color: #10b981; border: 2px solid #10b981; }

        /* Decorative background icons (bar chart + pie chart) - tiled and faint, clearer shapes */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            width: 100%;
            height: 100%;
            /* clearer bar chart (three distinct bars) and pie slices */
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'><rect x='6' y='56' width='18' height='58' rx='5' fill='%231089ff' /><rect x='44' y='30' width='18' height='84' rx='5' fill='%2310b981' /><rect x='82' y='42' width='18' height='72' rx='5' fill='%23f59e0b' /></svg>"), url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'><circle cx='60' cy='60' r='52' fill='none' stroke='%23e6f0ff' stroke-width='0' /><path d='M60 60 L110 60 A50 50 0 0 1 60 110 Z' fill='%2310b981' /><path d='M60 60 L60 10 A50 50 0 0 1 110 60 Z' fill='%231089ff' /><path d='M60 60 L12 60 A48 48 0 0 1 60 12 Z' fill='%23f59e0b' opacity='0.95' /></svg>");
            background-repeat: repeat, repeat;
            background-position: 0 0, 40% 50%;
            background-size: 320px 320px, 280px 280px;
            /* keep them faint but readable */
            opacity: 0.08;
            filter: blur(0.6px) saturate(1.02);
            pointer-events: none;
            z-index: 0;
        }

        /* Ensure app content sits above the decorative background */
        .stApp > div[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
USERS_FILE = BASE_DIR / "users.json"


def load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def avatar_data_uri(name: str) -> str:
    initials = (name[:2] if name else "U").upper()
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 72 72'><rect rx='36' width='72' height='72' fill='%2310b981'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='28' font-family='Arial' fill='white' font-weight='bold'>{initials}</text></svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "main" if st.session_state.get("user") else "home"


def render_login_page():
    """Login page."""
    _, col_c, _ = st.columns([1, 2, 1])
    
    with col_c:
        st.markdown("<div style='text-align:center; margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("<h2 class='auth-title'>LOGIN</h2>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 Login", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                users = load_users()
                if username in users and users[username] == hash_password(password):
                    st.session_state.user = username
                    st.session_state.page = "main"
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again or sign up.")
        
        st.markdown("<div style='margin-top: -0.5rem; margin-bottom: 0.5rem;'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("✍️ Sign Up", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_signup_page():
    """Sign up page."""
    _, col_c, _ = st.columns([1, 2, 1])
    
    with col_c:
        st.markdown("<div style='text-align:center; margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("<h2 class='auth-title'>SIGN UP</h2>", unsafe_allow_html=True)
        
        with st.form("signup_form"):
            username = st.text_input("Choose Username", placeholder="Pick a unique username")
            password = st.text_input("Choose Password", type="password", placeholder="Create a strong password")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submitted = st.form_submit_button("✍️ Create Account", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
            elif password != password_confirm:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if username in users:
                    st.error("Username already exists. Choose another.")
                else:
                    users[username] = hash_password(password)
                    save_users(users)
                    st.session_state.user = username
                    st.session_state.page = "main"
                    st.success(f"Account created! Welcome, {username}!")
                    st.rerun()
        
        st.markdown("<div style='margin-top: -0.5rem; margin-bottom: 0.5rem;'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("🔐 Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar():
    """Render left sidebar with navigation icons."""
    user = st.session_state.get("user")
    
    if user:
        # Avatar section
        avatar = avatar_data_uri(user)
        st.markdown(
            f"""
            <div class="sidebar-avatar">
                <img src="{avatar}" style="width:140px;height:140px;border-radius:50%; display:block; margin:0 auto;">
                <div class="sidebar-username">{user}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Navigation items - native Streamlit buttons (styled globally)
        nav_items = [
            ("👤 Your Profile", "profile"),
            ("🔮 Predict", "predict"),
            ("📊 Analytics", "analytics"),
        ]

        for label, page in nav_items:
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.page = page
                st.rerun()

        # Logout (placed directly after nav items to remove extra gap)
        if st.button("🔓 Logout", use_container_width=True, key="nav_logout"):
            st.session_state.pop("user", None)
            st.session_state.page = "home"
            st.rerun()


def render_top_header():
    """Render top header with logo."""
    # remove left-corner logo per request; keep right-side user info only
    _, _, col3 = st.columns([1, 2, 1])
    with col3:
        user = st.session_state.get("user")
        if user:
            st.markdown(f"<div style='text-align: right; font-size: 14px;'>Logged in as: <strong>{user}</strong></div>", unsafe_allow_html=True)


# Main app logic
if st.session_state.page == "home":
    render_top_header()
    st.markdown("---")
    render_home_page()

elif st.session_state.page == "login":
    render_login_page()

elif st.session_state.page == "signup":
    render_signup_page()

elif st.session_state.get("user"):
    # Authenticated pages with sidebar layout
    render_top_header()
    st.markdown("---")
    
    # Two-column layout: sidebar + content
    sidebar_col, content_col = st.columns([0.2, 0.8])
    
    with sidebar_col:
        render_sidebar()
    
    with content_col:
        page = st.session_state.page
        
        if page == "main":
            render_main_page()
        elif page == "profile":
            render_profile_page()
        elif page == "edit_profile":
            render_edit_profile_page()
        elif page == "predict":
            render_prediction_page()
        elif page == "analytics":
            render_analytics_page()
        else:
            render_main_page()

else:
    # Not authenticated
    st.warning("Please log in to continue.")
    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()
