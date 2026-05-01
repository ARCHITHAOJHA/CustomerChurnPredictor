import streamlit as st
import urllib.parse


def avatar_data_uri(name: str) -> str:
    initials = (name[:2] if name else "U").upper()
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 36 36'><rect rx='18' width='36' height='36' fill='%2310b981'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='14' font-family='Arial' fill='white'>{initials}</text></svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def render_main_page():
    """Main dashboard page for authenticated users."""
    user = st.session_state.get("user", "")
    avatar = avatar_data_uri(user)
    
    st.markdown(f"<h1>Welcome back, {user}!</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(
            f'<div style="text-align: center;"><img src="{avatar}" style="width:100px;height:100px;border-radius:50%;"></div>',
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='text-align: center; font-weight: bold;'>{user}</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Quick Actions**")
        col_predict, col_analytics = st.columns(2)
        with col_predict:
            if st.button("🔮 Go to Predict", use_container_width=True):
                st.session_state.page = "predict"
                st.rerun()
        with col_analytics:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.page = "analytics"
                st.rerun()
    
    st.markdown("---")
