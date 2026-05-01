import streamlit as st
import json
from pathlib import Path
import hashlib


BASE_DIR = Path(__file__).parent.parent
USERS_FILE = BASE_DIR / "users.json"


def load_users():
    """Load users from JSON file, creating if necessary."""
    if not USERS_FILE.exists():
        # Create empty users file if it doesn't exist
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        save_users({})
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    """Save users to JSON file."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(pw: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def render_edit_profile_page():
    """Edit profile page for changing username and password."""
    user = st.session_state.get("user", "")
    
    st.markdown("<h1>Edit Profile</h1>", unsafe_allow_html=True)
    st.markdown("Update your account information below.")
    st.markdown("---")
    
    # Create tabs for different edit options
    tab1, tab2 = st.tabs(["Change Username", "Change Password"])
    
    with tab1:
        st.markdown("### Change Username")
        new_username = st.text_input(
            "New Username",
            value=user,
            placeholder="Enter new username",
            help="Choose a unique username"
        )
        
        if st.button("Update Username", use_container_width=True, key="update_username_btn"):
            if not new_username or new_username.strip() == "":
                st.error("Username cannot be empty.")
            elif new_username == user:
                st.info("New username is the same as current username.")
            else:
                users = load_users()
                
                if new_username in users and new_username != user:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    try:
                        # Remove old username if it exists, add new one
                        if user in users:
                            users[new_username] = users.pop(user)
                        else:
                            # If user doesn't exist in database, create new entry
                            users[new_username] = ""
                        
                        save_users(users)
                        
                        # Update session state
                        st.session_state.user = new_username
                        
                        st.success(f"✓ Username updated to '{new_username}'")
                        st.info("Refresh the page to see changes everywhere.")
                    except Exception as e:
                        st.error(f"Error updating username: {str(e)}")
    
    with tab2:
        st.markdown("### Change Password")
        
        users = load_users()
        user_exists = user in users
        
        # Only ask for current password if user exists in database
        if user_exists:
            current_password = st.text_input(
                "Current Password",
                type="password",
                placeholder="Enter your current password for verification"
            )
        else:
            current_password = None
        
        # New password
        new_password = st.text_input(
            "New Password",
            type="password",
            placeholder="Enter new password (min 6 characters)"
        )
        
        # Confirm new password
        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            placeholder="Re-enter new password"
        )
        
        if st.button("Update Password", use_container_width=True, key="update_password_btn"):
            users = load_users()
            
            if not new_password or len(new_password) < 6:
                st.error("New password must be at least 6 characters long.")
            elif new_password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                try:
                    # Verify current password only if user exists in database
                    if user_exists and current_password and hash_password(current_password) != users.get(user):
                        st.error("Current password is incorrect.")
                    else:
                        # Update password
                        users[user] = hash_password(new_password)
                        save_users(users)
                        st.success("✓ Password updated successfully!")
                except Exception as e:
                    st.error(f"Error updating password: {str(e)}")
    
    st.markdown("---")
    if st.button("← Back to Profile", use_container_width=True):
        st.session_state.page = "profile"
        st.rerun()
