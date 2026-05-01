import streamlit as st


def render_home_page():
    """Home page for unauthenticated users."""
    st.markdown(
        """
                <div style="display:flex;align-items:center;gap:18px;justify-content:center;padding:8px 20px;white-space:nowrap;">
                    <div style="font-size:56px;flex:0 0 auto;">📊</div>
                    <h1 class="hero-title" style="font-size:56px; margin:0; font-weight:900; color:#0f172a; text-transform:uppercase; letter-spacing:-1px; text-shadow: 0 2px 0 rgba(0,0,0,0.03); flex:0 1 auto;">CUSTOMER CHURN PREDICTOR</h1>
                </div>
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:6px 20px 20px 20px;">
                <div style="width:100%;display:flex;justify-content:center;"><p class="hero-sub">Predict customer churn, analyze trends, and make data-driven decisions.</p></div>
                <!-- inline hero CTAs removed to use native Streamlit buttons below -->
            </div>
        """,
        unsafe_allow_html=True,
    )
    
    # single row login/signup buttons (no duplicate large green bars)
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        if st.button("🔐 Login", use_container_width=True, key="home-login"):
            st.session_state.page = "login"
            st.rerun()
        if st.button("✍️ Sign Up", use_container_width=True, key="home-signup"):
            st.session_state.page = "signup"
            st.rerun()
