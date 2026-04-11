import streamlit as st
import sqlite3
import hashlib
import os
import datetime
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaizeGuard AI",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #1a3a1a;
    letter-spacing: -1px;
}

.subtitle {
    color: #5a7a5a;
    font-size: 1rem;
    font-weight: 300;
}

.card {
    background: #f7faf7;
    border: 1px solid #d4e6d4;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.disease-card {
    background: linear-gradient(135deg, #f0f7f0, #e8f5e8);
    border-left: 4px solid #2e7d32;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
}

.disease-card h4 {
    color: #1b5e20;
    margin: 0 0 0.3rem 0;
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
}

.disease-card p {
    color: #4a6741;
    margin: 0;
    line-height: 1.4;
}

.prediction-box {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
}

.prediction-box h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    margin: 0;
}

.prediction-box p {
    margin: 0.5rem 0 0 0;
    opacity: 0.85;
}

.confidence-bar-bg {
    background: rgba(255,255,255,0.2);
    border-radius: 20px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
}

.log-row {
    background: #fff;
    border: 1px solid #e0ece0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #2d4a2d;
}

.stButton > button {
    background: #2e7d32 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: #1b5e20 !important;
}

.sidebar-disease {
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
    border: 1px solid #c8e6c9;
}

section[data-testid="stSidebar"] {
    background: #f1f8f1;
}
</style>
""", unsafe_allow_html=True)

# ── Database setup ────────────────────────────────────────────────────────────
DB_PATH = "maizeguard.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            all_probs TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT,
            details TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create default admin
    admin_pw = hash_password("admin123")
    c.execute("""
        INSERT OR IGNORE INTO users (username, email, password, role)
        VALUES (?, ?, ?, ?)
    """, ("admin", "admin@maizeguard.ai", admin_pw, "admin"))
    conn.commit()
    conn.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def log_action(user_id, username, action, details=""):
    conn = get_db()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO logs (user_id, username, action, details, timestamp) VALUES (?,?,?,?,?)",
        (user_id, username, action, details, ts)
    )
    conn.commit()
    conn.close()

init_db()

# ── CNN Model ─────────────────────────────────────────────────────────────────
class MyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

DISEASE_INFO = {
    'Blight': {
        'color': '#b71c1c',
        'emoji': '🔴',
        'description': 'Northern/Southern corn leaf blight causes large tan lesions on leaves. Spreads in warm, humid conditions. Use resistant varieties and fungicides.',
        'severity': 'High'
    },
    'Common_Rust': {
        'color': '#e65100',
        'emoji': '🟠',
        'description': 'Caused by Puccinia sorghi. Produces brick-red pustules on both leaf surfaces. Favours cool, moist weather. Apply fungicides early.',
        'severity': 'Medium'
    },
    'Gray_Leaf_Spot': {
        'color': '#616161',
        'emoji': '⚫',
        'description': 'Fungal disease causing rectangular gray/tan lesions. Thrives in high humidity. Rotate crops and use resistant hybrids.',
        'severity': 'High'
    },
    'Healthy': {
        'color': '#2e7d32',
        'emoji': '🟢',
        'description': 'No disease detected. Your maize plant appears healthy. Continue good agronomic practices to maintain plant health.',
        'severity': 'None'
    }
}

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = MyCNN(4).to(device)
    model_path = "corn_disease_checkpoint.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both checkpoint dict and plain state_dict formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, device, True
    return model, device, False

def predict_image(img: Image.Image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    probs_list = probs.cpu().numpy().tolist()
    idx = int(np.argmax(probs_list))
    return CLASS_NAMES[idx], probs_list[idx] * 100, {CLASS_NAMES[i]: round(probs_list[i]*100, 2) for i in range(4)}

# ── Session helpers ───────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.page = 'login'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Playfair Display,serif;color:#1b5e20;'>🌽 MaizeGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a5a;font-size:0.85rem;'>Disease Classification System</p>", unsafe_allow_html=True)
    st.divider()

    # Disease reference cards
    st.markdown("### 🔬 Disease Reference")
    sidebar_images = {
        'Blight': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Corn_leaf_blight.jpg/320px-Corn_leaf_blight.jpg',
        'Common_Rust': None,
        'Gray_Leaf_Spot': None,
        'Healthy': None,
    }
    for name, info in DISEASE_INFO.items():
        st.markdown(f"""
        <div class="disease-card">
            <h4>{info['emoji']} {name.replace('_',' ')}</h4>
            <p>{info['description'][:100]}…</p>
            <p style="margin-top:0.3rem;font-size:0.78rem;color:#888;">Severity: <strong style="color:{info['color']}">{info['severity']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

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

# ── Auth pages ────────────────────────────────────────────────────────────────
def page_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("<div class='main-title' style='text-align:center;'>🌽 MaizeGuard AI</div>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align:center;'>Maize Disease Classification Platform</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Sign In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Login", use_container_width=True):
                    conn = get_db()
                    user = conn.execute(
                        "SELECT * FROM users WHERE username=? AND password=? AND is_active=1",
                        (username, hash_password(password))
                    ).fetchone()
                    conn.close()
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = dict(user)
                        st.session_state.page = 'dashboard'
                        log_action(user['id'], user['username'], "LOGIN")
                        st.rerun()
                    else:
                        st.error("Invalid credentials or account disabled.")
            with col_b:
                if st.button("Register", use_container_width=True):
                    st.session_state.page = 'register'
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def page_register():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("<div class='main-title' style='text-align:center;'>🌽 Create Account</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### New User Registration")
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_pw = st.text_input("Password", type="password")
        new_pw2 = st.text_input("Confirm Password", type="password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create Account", use_container_width=True):
                if not new_username or not new_email or not new_pw:
                    st.warning("All fields are required.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        conn = get_db()
                        conn.execute(
                            "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                            (new_username, new_email, hash_password(new_pw))
                        )
                        conn.commit()
                        user = conn.execute("SELECT * FROM users WHERE username=?", (new_username,)).fetchone()
                        conn.close()
                        log_action(user['id'], new_username, "REGISTER")
                        st.success("Account created! Please login.")
                        st.session_state.page = 'login'
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Username or email already exists.")
        with col_b:
            if st.button("Back to Login", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Predict page ──────────────────────────────────────────────────────────────
def page_dashboard():
    model, device, model_loaded = load_model()

    st.markdown("<div class='main-title'>🌽 Maize Disease Classifier</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a maize leaf image to detect disease</p>", unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.warning("⚠️ Model file **corn_disease_model.pth** not found. Place it in the app directory. Running in demo mode with random predictions.")

    uploaded = st.file_uploader("Upload Maize Leaf Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Analysing leaf..."):
                if model_loaded:
                    pred_class, confidence, all_probs = predict_image(img, model, device)
                else:
                    # Demo mode
                    import random
                    pred_class = random.choice(CLASS_NAMES)
                    confidence = round(random.uniform(70, 98), 2)
                    rem = 100 - confidence
                    others = [c for c in CLASS_NAMES if c != pred_class]
                    all_probs = {pred_class: confidence}
                    splits = sorted([random.uniform(0, rem) for _ in range(2)])
                    all_probs[others[0]] = round(splits[0], 2)
                    all_probs[others[1]] = round(splits[1] - splits[0], 2)
                    all_probs[others[2]] = round(rem - splits[1], 2)

            info = DISEASE_INFO[pred_class]
            st.markdown(f"""
            <div class="prediction-box" style="background:linear-gradient(135deg,{info['color']},{info['color']}cc);">
                <div style="font-size:2.5rem;">{info['emoji']}</div>
                <h2>{pred_class.replace('_',' ')}</h2>
                <p>Confidence: {confidence:.1f}%</p>
                <div class="confidence-bar-bg">
                    <div style="background:white;height:10px;border-radius:20px;width:{confidence}%;"></div>
                </div>
                <p style="font-size:0.85rem;">Severity: {info['severity']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**ℹ️ About this disease:** {info['description']}")

            st.markdown("**📊 All Class Probabilities**")
            for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                st.progress(int(prob), text=f"{cls.replace('_',' ')}: {prob:.1f}%")

            # Save prediction
            conn = get_db()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO predictions (user_id, username, filename, predicted_class, confidence, all_probs, timestamp) VALUES (?,?,?,?,?,?,?)",
                (st.session_state.user['id'], st.session_state.user['username'],
                 uploaded.name, pred_class, confidence, json.dumps(all_probs), ts)
            )
            conn.commit()
            conn.close()
            log_action(st.session_state.user['id'], st.session_state.user['username'],
                       "PREDICTION", f"{uploaded.name} → {pred_class} ({confidence:.1f}%)")

            # ── Download Report ──
            st.markdown("---")
            report = f"""
MAIZEGUARD AI — DISEASE CLASSIFICATION REPORT
==============================================
Date/Time   : {ts}
User        : {st.session_state.user['username']}
File        : {uploaded.name}

RESULT
------
Predicted Disease : {pred_class.replace('_',' ')}
Confidence        : {confidence:.2f}%
Severity Level    : {info['severity']}

Description:
{info['description']}

ALL CLASS PROBABILITIES
------------------------
""" + "\n".join([f"  {k.replace('_',' '):<20}: {v:.2f}%" for k,v in sorted(all_probs.items(), key=lambda x:-x[1])]) + """

RECOMMENDATIONS
---------------
""" + (
    "• Apply appropriate fungicide immediately.\n• Remove and destroy infected leaves.\n• Improve field drainage and air circulation.\n• Monitor neighbouring plants for spread."
    if pred_class != 'Healthy'
    else "• Continue current agronomic practices.\n• Maintain regular scouting (weekly).\n• Ensure balanced fertilization.\n• Monitor for pest activity."
) + f"""

--------------------------------------------------
Generated by MaizeGuard AI | {ts}
"""
            st.download_button(
                "📥 Download Report (.txt)",
                data=report,
                file_name=f"maizeguard_report_{uploaded.name.split('.')[0]}_{ts.replace(' ','_').replace(':','-')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ── My Predictions page ───────────────────────────────────────────────────────
def page_my_predictions():
    st.markdown("<div class='main-title'>📋 My Predictions</div>", unsafe_allow_html=True)
    st.divider()
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
        (st.session_state.user['id'],)
    ).fetchall()
    conn.close()

    if not rows:
        st.info("No predictions yet. Upload a maize leaf image to get started!")
        return

    st.markdown(f"**Total predictions: {len(rows)}**")
    for row in rows:
        info = DISEASE_INFO.get(row['predicted_class'], DISEASE_INFO['Healthy'])
        st.markdown(f"""
        <div class="log-row">
            {info['emoji']} &nbsp;<strong>{row['predicted_class'].replace('_',' ')}</strong>
            &nbsp;·&nbsp; Confidence: <strong>{row['confidence']:.1f}%</strong>
            &nbsp;·&nbsp; File: {row['filename']}
            &nbsp;·&nbsp; <span style="color:#888;">{row['timestamp']}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Admin page ────────────────────────────────────────────────────────────────
def page_admin():
    if st.session_state.user['role'] != 'admin':
        st.error("Access denied.")
        return

    st.markdown("<div class='main-title'>🛠️ Admin Dashboard</div>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📊 All Predictions", "📜 System Logs"])

    conn = get_db()

    with tab1:
        users = conn.execute("SELECT id, username, email, role, created_at, is_active FROM users ORDER BY created_at DESC").fetchall()
        st.markdown(f"**Total users: {len(users)}**")
        for u in users:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.markdown(f"**{u['username']}** ({u['email']})")
            with col2:
                st.markdown(f"Role: `{u['role']}` | Joined: {u['created_at'][:10]}")
            with col3:
                status = "✅ Active" if u['is_active'] else "❌ Disabled"
                st.markdown(status)
            with col4:
                if u['username'] != 'admin':
                    new_status = 0 if u['is_active'] else 1
                    label = "Disable" if u['is_active'] else "Enable"
                    if st.button(label, key=f"tog_{u['id']}"):
                        conn.execute("UPDATE users SET is_active=? WHERE id=?", (new_status, u['id']))
                        conn.commit()
                        log_action(st.session_state.user['id'], st.session_state.user['username'],
                                   "USER_STATUS_CHANGE", f"{u['username']} → {'disabled' if new_status==0 else 'enabled'}")
                        st.rerun()

    with tab2:
        preds = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC").fetchall()
        st.markdown(f"**Total predictions: {len(preds)}**")

        # Stats
        if preds:
            from collections import Counter
            counts = Counter([p['predicted_class'] for p in preds])
            cols = st.columns(4)
            for i, cls in enumerate(CLASS_NAMES):
                info = DISEASE_INFO[cls]
                cols[i].metric(f"{info['emoji']} {cls.replace('_',' ')}", counts.get(cls, 0))

        st.markdown("---")
        for p in preds:
            info = DISEASE_INFO.get(p['predicted_class'], DISEASE_INFO['Healthy'])
            st.markdown(f"""
            <div class="log-row">
                {info['emoji']} <strong>{p['predicted_class'].replace('_',' ')}</strong>
                · Confidence: <strong>{p['confidence']:.1f}%</strong>
                · File: {p['filename']}
                · User: <strong>{p['username']}</strong>
                · <span style="color:#888;">{p['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        logs = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 200").fetchall()
        st.markdown(f"**Recent logs (last 200):**")
        for log in logs:
            st.markdown(f"""
            <div class="log-row">
                🕐 <span style="color:#888;">{log['timestamp']}</span>
                &nbsp;·&nbsp; <strong>{log['username']}</strong>
                &nbsp;·&nbsp; <code>{log['action']}</code>
                {f"&nbsp;·&nbsp; {log['details']}" if log['details'] else ""}
            </div>
            """, unsafe_allow_html=True)

    conn.close()

# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.page == 'register':
        page_register()
    else:
        page_login()
else:
    user = st.session_state.user
    with st.sidebar:
        st.divider()
        st.markdown("### Navigation")
        if st.button("🔬 Classify Image", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
        if st.button("📋 My Predictions", use_container_width=True):
            st.session_state.page = 'my_predictions'
            st.rerun()
        if user['role'] == 'admin':
            if st.button("🛠️ Admin Panel", use_container_width=True):
                st.session_state.page = 'admin'
                st.rerun()

    if st.session_state.page == 'dashboard':
        page_dashboard()
    elif st.session_state.page == 'my_predictions':
        page_my_predictions()
    elif st.session_state.page == 'admin':
        page_admin()
    else:
        page_dashboard()