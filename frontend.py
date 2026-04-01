import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MediAssist - AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        border-radius: 8px;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    div[data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Session state initialization
for key in ["logged_in", "patient_id", "name", "messages", "processing"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["logged_in", "processing"] else [] if key == "messages" else None

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

# Sidebar
with st.sidebar:
    st.title("🩺 MediAssist")
    st.markdown("---")
    
    if check_api():
        st.success("✅ Secure Connection Established")
    else:
        st.error("❌ System Offline - Check Backend")
    
    st.markdown("---")
    
    if st.session_state.logged_in:
        st.markdown(f"### 👤 {st.session_state.name}")
        st.markdown("### ⚡ Quick Insights")
        
        actions = {
            "📋 Health Summary": "Please provide a comprehensive summary of my health records.",
            "💊 Current Medications": "What medications am I currently taking, and are there any interactions?",
            "🍎 Diet & Wellness Plan": "Based on my records, suggest a proactive diet and wellness plan.",
            "🩺 Recent Lab Results": "Explain my recent lab results in simple terms."
        }
        
        for label, prompt in actions.items():
            if st.button(label, use_container_width=True):
                if not st.session_state.processing:
                    add_message("user", prompt)
                    # CRITICAL BUG FIX: Tell the app to start processing the request
                    st.session_state.processing = True 
                    st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Secure Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        st.markdown("### 📝 Demo Access")
        st.info("Use the details from your database to log in. E.g., \n**Name:** Arjun Mehta\n**DOB:** 1982-11-30")

# Main content
if not st.session_state.logged_in:
    st.title("🔐 MediAssist Secure Portal")
    st.markdown("Access your medical records with intelligent, proactive AI insights.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            name = st.text_input("Full Name", placeholder="Enter your full name")
            dob = st.text_input("Date of Birth (YYYY-MM-DD)", type="password", placeholder="YYYY-MM-DD")
            
            if st.form_submit_button("Access Records", use_container_width=True):
                if name and dob:
                    with st.spinner("Authenticating securely..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/login",
                                json={"name": name.strip(), "password": dob.strip()},
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.logged_in = True
                                st.session_state.patient_id = data["patient_id"]
                                st.session_state.name = data["name"]
                                
                                welcome_msg = f"Hello {data['name']}, I am MediAssist. I have securely loaded your medical profile. How can I help you understand your health today?"
                                add_message("assistant", welcome_msg)
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Please check your name and date of birth.")
                        except Exception as e:
                            st.error(f"Unable to connect to the server. Please make sure the backend is running. Error: {e}")
                else:
                    st.error("Please enter both name and date of birth.")

else:
    st.title("🩺 MediAssist")
    
    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(msg["timestamp"])
    
    # Text input box at the bottom
    if prompt := st.chat_input("Ask about your health, medications, or records...", disabled=st.session_state.processing):
        add_message("user", prompt)
        st.session_state.processing = True
        st.rerun()
    
    # Processing the AI response
    if st.session_state.processing and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your medical records..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": last_msg,
                            "patient_id": st.session_state.patient_id
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.markdown(answer)
                        add_message("assistant", answer)
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"I encountered an error analyzing your records: {error_detail}")
                except Exception as e:
                    st.error(f"Communication error with the server. Please check your backend connection.")
        
        # Reset processing state so the user can ask the next question
        st.session_state.processing = False
        st.rerun()