import streamlit as st
import urllib.parse


def avatar_data_uri(name: str) -> str:
    initials = (name[:2] if name else "U").upper()
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 36 36'><rect rx='18' width='36' height='36' fill='%23375ef7'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='14' font-family='Arial' fill='white'>{initials}</text></svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def render_profile_page():
    """User profile page."""
    user = st.session_state.get("user", "")
    
    st.markdown("<h1>Your Profile</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        avatar = avatar_data_uri(user)
        st.markdown(
            f'<div style="text-align: center;"><img src="{avatar}" style="width:120px;height:120px;border-radius:50%;"></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"**Username:** `{user}`")
        st.markdown("**Account Status:** Active")
        st.markdown("**Member Since:** 2024")
        st.markdown("---")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.page = "edit_profile"
            st.rerun()
