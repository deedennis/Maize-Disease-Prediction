import streamlit as st
import hashlib
import os
import datetime
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from supabase import create_client, Client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MaizeGuard AI", page_icon="🌽", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-title { font-family: 'Playfair Display', serif; font-size: 2.6rem; color: #1a3a1a; letter-spacing: -1px; }
.subtitle { color: #5a7a5a; font-size: 1rem; font-weight: 300; }
.card { background: #f7faf7; border: 1px solid #d4e6d4; border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.disease-card { background: linear-gradient(135deg, #f0f7f0, #e8f5e8); border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; font-size: 0.85rem; }
.disease-card h4 { color: #1b5e20; margin: 0 0 0.3rem 0; font-family: 'Playfair Display', serif; font-size: 0.95rem; }
.disease-card p { color: #4a6741; margin: 0; line-height: 1.4; }
.prediction-box { background: linear-gradient(135deg, #1b5e20, #2e7d32); color: white; border-radius: 14px; padding: 1.8rem; text-align: center; margin: 1rem 0; }
.prediction-box h2 { font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0; }
.prediction-box p { margin: 0.5rem 0 0 0; opacity: 0.85; }
.confidence-bar-bg { background: rgba(255,255,255,0.2); border-radius: 20px; height: 10px; margin: 0.8rem 0; overflow: hidden; }
.log-row { background: #fff; border: 1px solid #e0ece0; border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; color: #2d4a2d; }
.stButton > button { background: #2e7d32 !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; padding: 0.55rem 1.4rem !important; transition: background 0.2s !important; }
.stButton > button:hover { background: #1b5e20 !important; }
section[data-testid="stSidebar"] { background: #f1f8f1; }
</style>
""", unsafe_allow_html=True)

# ── Supabase connection ───────────────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase()

# ── DB helpers ────────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user(username, hashed_pw):
    res = supabase.table("users").select("*").eq("username", username).eq("password", hashed_pw).execute()
    if res.data:
        user = res.data[0]
        if user.get("is_active") in (True, 1, "true", "True"):
            return user
    return None

def get_user_by_username(username):
    res = supabase.table("users").select("*").eq("username", username).execute()
    return res.data[0] if res.data else None

def create_user(username, email, hashed_pw):
    supabase.table("users").insert({"username": username, "email": email, "password": hashed_pw, "role": "user", "is_active": True}).execute()

def log_action(user_id, username, action, details=""):
    supabase.table("logs").insert({"user_id": user_id, "username": username, "action": action, "details": details}).execute()

def save_prediction(user_id, username, filename, pred_class, confidence, all_probs):
    supabase.table("predictions").insert({"user_id": user_id, "username": username, "filename": filename, "predicted_class": pred_class, "confidence": confidence, "all_probs": json.dumps(all_probs)}).execute()

# ── CNN Model ─────────────────────────────────────────────────────────────────
class MyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7,7))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*7*7,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

CLASS_NAMES = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']
DISEASE_INFO = {
    'Blight':        {'color':'#b71c1c','emoji':'🔴','severity':'High',  'description':'Northern/Southern corn leaf blight causes large tan lesions. Spreads in warm, humid conditions. Use resistant varieties and fungicides.'},
    'Common_Rust':   {'color':'#e65100','emoji':'🟠','severity':'Medium','description':'Caused by Puccinia sorghi. Brick-red pustules on both leaf surfaces. Favours cool, moist weather. Apply fungicides early.'},
    'Gray_Leaf_Spot':{'color':'#616161','emoji':'⚫','severity':'High',  'description':'Fungal disease causing rectangular gray/tan lesions. Thrives in high humidity. Rotate crops and use resistant hybrids.'},
    'Healthy':       {'color':'#2e7d32','emoji':'🟢','severity':'None',  'description':'No disease detected. Your maize plant appears healthy. Continue good agronomic practices.'},
}

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = MyCNN(4).to(device)
    path = "corn_disease_checkpoint.pth"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)
        model.eval()
        return model, device, True
    return model, device, False

def predict_image(img, model, device):
    t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    tensor = t(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy().tolist()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs[idx]*100, {CLASS_NAMES[i]: round(probs[i]*100,2) for i in range(4)}

# ── Session state ─────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.page = 'login'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Playfair Display,serif;color:#1b5e20;'>🌽 MaizeGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a5a;font-size:0.85rem;'>Disease Classification System</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🔬 Disease Reference")
    for name, info in DISEASE_INFO.items():
        st.markdown(f'<div class="disease-card"><h4>{info["emoji"]} {name.replace("_"," ")}</h4><p>{info["description"][:100]}…</p><p style="margin-top:0.3rem;font-size:0.78rem;color:#888;">Severity: <strong style="color:{info["color"]}">{info["severity"]}</strong></p></div>', unsafe_allow_html=True)
    st.divider()
    if st.session_state.logged_in:
        st.markdown(f"👤 **{st.session_state.user['username']}**")
        st.markdown(f"<span style='font-size:0.8rem;color:#888;'>Role: {st.session_state.user['role']}</span>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            log_action(st.session_state.user['id'], st.session_state.user['username'], "LOGOUT")
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()

# ── Pages ─────────────────────────────────────────────────────────────────────
def page_login():
    c1,c2,c3 = st.columns([1,1.4,1])
    with c2:
        st.markdown("<div class='main-title' style='text-align:center;'>🌽 MaizeGuard AI</div>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align:center;'>Maize Disease Classification Platform</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Sign In")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        ca,cb = st.columns(2)
        with ca:
            if st.button("Login", use_container_width=True):
                user = get_user(username, hash_password(password))
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state.page = 'dashboard'
                    log_action(user['id'], user['username'], "LOGIN")
                    st.rerun()
                else:
                    st.error("Invalid credentials or account disabled.")
        with cb:
            if st.button("Register", use_container_width=True):
                st.session_state.page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def page_register():
    c1,c2,c3 = st.columns([1,1.4,1])
    with c2:
        st.markdown("<div class='main-title' style='text-align:center;'>🌽 Create Account</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### New User Registration")
        nu = st.text_input("Username")
        ne = st.text_input("Email")
        np1 = st.text_input("Password", type="password")
        np2 = st.text_input("Confirm Password", type="password")
        ca,cb = st.columns(2)
        with ca:
            if st.button("Create Account", use_container_width=True):
                if not nu or not ne or not np1:
                    st.warning("All fields are required.")
                elif np1 != np2:
                    st.error("Passwords do not match.")
                elif len(np1) < 6:
                    st.error("Password must be at least 6 characters.")
                elif get_user_by_username(nu):
                    st.error("Username already exists.")
                else:
                    try:
                        create_user(nu, ne, hash_password(np1))
                        new_user = get_user_by_username(nu)
                        log_action(new_user['id'], nu, "REGISTER")
                        st.success("✅ Account created! Please login.")
                        st.session_state.page = 'login'; st.rerun()
                    except Exception as ex:
                        st.error(f"Registration failed: {ex}")
        with cb:
            if st.button("Back to Login", use_container_width=True):
                st.session_state.page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def page_dashboard():
    model, device, model_loaded = load_model()
    st.markdown("<div class='main-title'>🌽 Maize Disease Classifier</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a maize leaf image to detect disease</p>", unsafe_allow_html=True)
    st.divider()
    if not model_loaded:
        st.warning("⚠️ Model file **corn_disease_checkpoint.pth** not found. Running in demo mode.")
    uploaded = st.file_uploader("Upload Maize Leaf Image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        col1,col2 = st.columns([1,1.2])
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Analysing leaf..."):
                if model_loaded:
                    pred_class, confidence, all_probs = predict_image(img, model, device)
                else:
                    import random
                    pred_class = random.choice(CLASS_NAMES)
                    confidence = round(random.uniform(70,98),2)
                    rem = 100-confidence
                    others = [c for c in CLASS_NAMES if c != pred_class]
                    splits = sorted([random.uniform(0,rem) for _ in range(2)])
                    all_probs = {pred_class:confidence, others[0]:round(splits[0],2), others[1]:round(splits[1]-splits[0],2), others[2]:round(rem-splits[1],2)}
            info = DISEASE_INFO[pred_class]
            st.markdown(f'<div class="prediction-box" style="background:linear-gradient(135deg,{info["color"]},{info["color"]}cc);"><div style="font-size:2.5rem;">{info["emoji"]}</div><h2>{pred_class.replace("_"," ")}</h2><p>Confidence: {confidence:.1f}%</p><div class="confidence-bar-bg"><div style="background:white;height:10px;border-radius:20px;width:{confidence}%;"></div></div><p style="font-size:0.85rem;">Severity: {info["severity"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f"**ℹ️ About this disease:** {info['description']}")
            st.markdown("**📊 All Class Probabilities**")
            for cls, prob in sorted(all_probs.items(), key=lambda x:-x[1]):
                st.progress(int(prob), text=f"{cls.replace('_',' ')}: {prob:.1f}%")
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_prediction(st.session_state.user['id'], st.session_state.user['username'], uploaded.name, pred_class, confidence, all_probs)
            log_action(st.session_state.user['id'], st.session_state.user['username'], "PREDICTION", f"{uploaded.name} → {pred_class} ({confidence:.1f}%)")
            recs = ("• Apply appropriate fungicide immediately.\n• Remove and destroy infected leaves.\n• Improve field drainage and air circulation.\n• Monitor neighbouring plants for spread." if pred_class!='Healthy' else "• Continue current agronomic practices.\n• Maintain regular scouting (weekly).\n• Ensure balanced fertilization.\n• Monitor for pest activity.")
            report = f"""MAIZEGUARD AI — DISEASE CLASSIFICATION REPORT\n==============================================\nDate/Time   : {ts}\nUser        : {st.session_state.user['username']}\nFile        : {uploaded.name}\n\nRESULT\n------\nPredicted Disease : {pred_class.replace('_',' ')}\nConfidence        : {confidence:.2f}%\nSeverity Level    : {info['severity']}\n\nDescription:\n{info['description']}\n\nALL CLASS PROBABILITIES\n-----------------------\n""" + "\n".join([f"  {k.replace('_',' '):<20}: {v:.2f}%" for k,v in sorted(all_probs.items(), key=lambda x:-x[1])]) + f"\n\nRECOMMENDATIONS\n---------------\n{recs}\n\n--------------------------------------------------\nGenerated by MaizeGuard AI | {ts}\n"""
            st.markdown("---")
            st.download_button("📥 Download Report (.txt)", data=report, file_name=f"maizeguard_{uploaded.name.split('.')[0]}_{ts.replace(' ','_').replace(':','-')}.txt", mime="text/plain", use_container_width=True)

def page_my_predictions():
    st.markdown("<div class='main-title'>📋 My Predictions</div>", unsafe_allow_html=True)
    st.divider()
    rows = supabase.table("predictions").select("*").eq("user_id", st.session_state.user['id']).order("timestamp", desc=True).execute().data
    if not rows:
        st.info("No predictions yet. Upload a maize leaf image to get started!"); return
    st.markdown(f"**Total predictions: {len(rows)}**")
    for row in rows:
        info = DISEASE_INFO.get(row['predicted_class'], DISEASE_INFO['Healthy'])
        st.markdown(f'<div class="log-row">{info["emoji"]} &nbsp;<strong>{row["predicted_class"].replace("_"," ")}</strong> &nbsp;·&nbsp; Confidence: <strong>{row["confidence"]:.1f}%</strong> &nbsp;·&nbsp; File: {row["filename"]} &nbsp;·&nbsp; <span style="color:#888;">{str(row["timestamp"])[:19]}</span></div>', unsafe_allow_html=True)

def page_admin():
    if st.session_state.user['role'] != 'admin':
        st.error("Access denied."); return

    from collections import Counter

    st.markdown("<div class='main-title'>🛠️ Admin Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>System overview, user management and audit logs</p>", unsafe_allow_html=True)
    st.divider()

    # ── Fetch data ──
    all_users = supabase.table("users").select("id,username,email,role,created_at,is_active").order("created_at", desc=True).execute().data
    all_preds = supabase.table("predictions").select("*").order("timestamp", desc=True).execute().data
    all_logs  = supabase.table("logs").select("*").order("timestamp", desc=True).limit(200).execute().data

    counts = Counter([p['predicted_class'] for p in all_preds]) if all_preds else {}
    total_users = len([u for u in all_users if u['role'] != 'admin'])
    active_users = len([u for u in all_users if u['is_active'] and u['role'] != 'admin'])

    # ── Summary metrics ──
    st.markdown("### 📈 Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("👥 Total Users",       total_users)
    m2.metric("✅ Active Users",       active_users)
    m3.metric("📊 Total Predictions", len(all_preds))
    m4.metric("🔴 Blight",            counts.get('Blight', 0))
    m5.metric("🟠 Common Rust",       counts.get('Common_Rust', 0))
    m6.metric("⚫ Gray Leaf Spot",    counts.get('Gray_Leaf_Spot', 0))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-class breakdown bar ──
    if all_preds:
        st.markdown("### 🌽 Predictions by Disease Class")
        total = len(all_preds)
        bar_cols = st.columns(4)
        for i, cls in enumerate(CLASS_NAMES):
            info = DISEASE_INFO[cls]
            cnt = counts.get(cls, 0)
            pct = round(cnt / total * 100, 1) if total else 0
            bar_cols[i].markdown(f"""
            <div style="background:{info['color']}22;border-left:4px solid {info['color']};border-radius:8px;padding:0.8rem 1rem;">
                <div style="font-size:1.5rem;">{info['emoji']}</div>
                <div style="font-weight:600;color:{info['color']};">{cls.replace('_',' ')}</div>
                <div style="font-size:1.4rem;font-weight:700;color:#1a3a1a;">{cnt}</div>
                <div style="font-size:0.8rem;color:#888;">{pct}% of total</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📋 Prediction History", "📜 Audit Logs"])

    with tab1:
        st.markdown(f"**{total_users} registered user(s)**")
        st.markdown("<br>", unsafe_allow_html=True)
        for u in all_users:
            if u['username'] == 'admin': continue
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            with c1: st.markdown(f"**{u['username']}** <span style='font-size:0.8rem;color:#888;'>({u['email']})</span>", unsafe_allow_html=True)
            with c2: st.markdown(f"Role: `{u['role']}` &nbsp;|&nbsp; Joined: {str(u['created_at'])[:10]}", unsafe_allow_html=True)
            with c3: st.markdown("✅ Active" if u['is_active'] else "❌ Disabled")
            with c4:
                if st.button("Disable" if u['is_active'] else "Enable", key=f"tog_{u['id']}"):
                    new_s = not u['is_active']
                    supabase.table("users").update({"is_active": new_s}).eq("id", u['id']).execute()
                    log_action(st.session_state.user['id'], st.session_state.user['username'],
                               "USER_STATUS_CHANGE", f"{u['username']} → {'enabled' if new_s else 'disabled'}")
                    st.rerun()

    with tab2:
        st.markdown(f"**{len(all_preds)} total prediction(s)**")
        if not all_preds:
            st.info("No predictions yet.")
        for p in all_preds:
            info = DISEASE_INFO.get(p['predicted_class'], DISEASE_INFO['Healthy'])
            st.markdown(
                f'<div class="log-row">{info["emoji"]} <strong>{p["predicted_class"].replace("_"," ")}</strong>'
                f' &nbsp;·&nbsp; Confidence: <strong>{p["confidence"]:.1f}%</strong>'
                f' &nbsp;·&nbsp; File: {p["filename"]}'
                f' &nbsp;·&nbsp; User: <strong>{p["username"]}</strong>'
                f' &nbsp;·&nbsp; <span style="color:#888;">{str(p["timestamp"])[:19]}</span></div>',
                unsafe_allow_html=True)

    with tab3:
        st.markdown("**Recent audit logs (last 200):**")
        if not all_logs:
            st.info("No logs yet.")
        for log in all_logs:
            details_html = f' &nbsp;·&nbsp; {log["details"]}' if log.get("details") else ""
            st.markdown(
                f'<div class="log-row">🕐 <span style="color:#888;">{str(log["timestamp"])[:19]}</span>'
                f' &nbsp;·&nbsp; <strong>{log["username"]}</strong>'
                f' &nbsp;·&nbsp; <code>{log["action"]}</code>{details_html}</div>',
                unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.page == 'register': page_register()
    else: page_login()
else:
    is_admin = st.session_state.user['role'] == 'admin'

    # Redirect admin away from user-only pages
    if is_admin and st.session_state.page in ('dashboard', 'my_predictions'):
        st.session_state.page = 'admin'

    with st.sidebar:
        st.divider()
        st.markdown("### Navigation")
        if not is_admin:
            if st.button("🔬 Classify Image", use_container_width=True):
                st.session_state.page = 'dashboard'; st.rerun()
            if st.button("📋 My Predictions", use_container_width=True):
                st.session_state.page = 'my_predictions'; st.rerun()
        else:
            if st.button("🛠️ Admin Dashboard", use_container_width=True):
                st.session_state.page = 'admin'; st.rerun()

    if st.session_state.page == 'dashboard' and not is_admin:
        page_dashboard()
    elif st.session_state.page == 'my_predictions' and not is_admin:
        page_my_predictions()
    elif st.session_state.page == 'admin' and is_admin:
        page_admin()
    elif is_admin:
        page_admin()
    else:
        page_dashboard()
