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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Jost:wght@300;400;500;600&display=swap');

:root {
    --green-dark: #0d2b12;
    --green-mid: #1a5c24;
    --green-accent: #3a9e4f;
    --green-light: #c8eecf;
    --gold: #c9a84c;
    --cream: #f9f6ef;
    --text-dark: #0d2b12;
    --text-muted: #5a7a5a;
    --card-bg: rgba(255,255,255,0.85);
    --border: #d4e6d4;
}

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    color: var(--text-dark);
}

/* ── HERO LANDING ── */
.hero-section {
    background: linear-gradient(160deg, #0d2b12 0%, #1a5c24 45%, #2d7a3a 70%, #0d2b12 100%);
    border-radius: 20px;
    padding: 4rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero-section::after {
    content: '';
    position: absolute;
    bottom: -20%;
    left: -5%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(58,158,79,0.2) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(201,168,76,0.15);
    border: 1px solid rgba(201,168,76,0.4);
    color: #c9a84c;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
    width: fit-content;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.08;
    letter-spacing: -1.5px;
    margin: 0 0 0.8rem 0;
}

.hero-title span {
    color: #c9a84c;
}

.hero-subtitle {
    color: rgba(255,255,255,0.72);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 2rem;
}

.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}

.hero-stat {
    text-align: left;
}

.hero-stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #c9a84c;
    line-height: 1;
}

.hero-stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.hero-illustration {
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 9rem;
    opacity: 0.12;
    user-select: none;
    pointer-events: none;
    line-height: 1;
}

.hero-corn-large {
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0.06;
    font-size: 14rem;
    pointer-events: none;
}

/* ── FEATURE PILLS ── */
.feature-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}

.feature-pill {
    background: white;
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    color: var(--green-mid);
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── DISEASE CARDS ── */
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
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem;
}

.disease-card p {
    color: #4a6741;
    margin: 0;
    line-height: 1.4;
}

/* ── AUTH CARDS ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

/* ── PREDICTION RESULT ── */
.prediction-box {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1rem 0;
}

.prediction-box h2 {
    font-family: 'Cormorant Garamond', serif;
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

/* ── RECOMMENDATION CARDS ── */
.rec-section {
    background: #ffffff;
    border: 1px solid #d4e6d4;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}

.rec-section h4 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    color: var(--green-dark);
    margin: 0 0 0.8rem 0;
    border-bottom: 2px solid var(--green-light);
    padding-bottom: 0.4rem;
}

.rec-step {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f7f0;
    font-size: 0.88rem;
    color: #2d4a2d;
    line-height: 1.5;
}

.rec-step:last-child { border-bottom: none; }

.rec-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 0.05rem;
}

.fungicide-tag {
    display: inline-block;
    background: #e8f5e8;
    border: 1px solid #a5d6a7;
    color: #1b5e20;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    margin: 0.2rem 0.2rem 0 0;
}

.fungicide-tag.warning {
    background: #fff3e0;
    border-color: #ffcc80;
    color: #e65100;
}

/* ── LOG ROWS ── */
.log-row {
    background: #fff;
    border: 1px solid #e0ece0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #2d4a2d;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #1a5c24 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s !important;
}

.stButton > button:hover { background: #0d2b12 !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background: #f1f8f1; }

/* ── SECTION TITLES ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    color: var(--green-dark);
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}

.section-sub {
    color: var(--text-muted);
    font-size: 0.92rem;
    font-weight: 300;
    margin-bottom: 1.5rem;
}
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
    'Blight': {
        'color': '#b71c1c',
        'emoji': '🔴',
        'severity': 'High',
        'description': 'Northern/Southern corn leaf blight causes large tan to grayish-brown lesions on leaves. Spreads rapidly in warm, humid conditions (18–27°C). Caused by Exserohilum turcicum (NCLB) or Cochliobolus heterostrophus (SCLB).',
        'recommendations': [
            ('🌱', 'Plant resistant or tolerant hybrids as the first line of defence against blight.'),
            ('✂️', 'Remove and burn or bury infected plant debris after harvest to reduce overwintering inoculum.'),
            ('💧', 'Improve field drainage and avoid overhead irrigation to reduce leaf wetness periods.'),
            ('🔄', 'Rotate with non-host crops such as soybean or wheat for at least one season.'),
            ('👁️', 'Scout fields weekly, especially after tasselling when plants are most vulnerable.'),
            ('🌬️', 'Ensure adequate plant spacing to promote airflow and reduce humidity within the canopy.'),
        ],
        'fungicides': [
            {'name': 'Azoxystrobin (Quadris)', 'type': 'Strobilurin', 'rate': '0.75–1.5 L/ha', 'timing': 'At first lesion appearance or VT/R1 stage'},
            {'name': 'Propiconazole (Tilt)', 'type': 'Triazole', 'rate': '0.5–1.0 L/ha', 'timing': 'Preventive; before disease onset'},
            {'name': 'Mancozeb (Dithane M-45)', 'type': 'Contact', 'rate': '2.0–2.5 kg/ha', 'timing': 'Early season, repeat every 7–10 days'},
            {'name': 'Picoxystrobin + Cyproconazole (Aproach Prima)', 'type': 'SDHI + Triazole', 'rate': '0.75 L/ha', 'timing': 'Single application at VT stage'},
        ],
        'ipm_note': 'Combine cultural practices with fungicides. Apply at VT (tasselling) or R1 (silking) for best economic returns.',
    },
    'Common_Rust': {
        'color': '#e65100',
        'emoji': '🟠',
        'severity': 'Medium',
        'description': 'Caused by Puccinia sorghi. Characterised by brick-red to cinnamon-brown pustules (uredia) on both leaf surfaces. Favours cool (16–23°C), moist conditions with heavy dew. Can reduce yield by up to 40% if unchecked.',
        'recommendations': [
            ('🌱', 'Grow rust-resistant hybrid varieties — consult your seed supplier for locally adapted options.'),
            ('🗓️', 'Plant early to allow crops to mature before peak rust season (cool, wet months).'),
            ('✂️', 'Remove heavily infected leaves and do not carry infected plant material between fields.'),
            ('🔄', 'Crop rotation reduces but does not eliminate rust, as spores can travel long distances on wind.'),
            ('👁️', 'Monitor both leaf surfaces for the characteristic brick-red pustules from seedling stage onward.'),
            ('📊', 'Apply fungicides when >5% of plants show rust pustules before tasselling for economic benefit.'),
        ],
        'fungicides': [
            {'name': 'Tebuconazole (Folicur)', 'type': 'Triazole', 'rate': '0.75–1.0 L/ha', 'timing': 'At first detection of pustules'},
            {'name': 'Trifloxystrobin + Tebuconazole (Nativo)', 'type': 'Strobilurin + Triazole', 'rate': '0.75 L/ha', 'timing': 'Preventive; VT to R1 stage'},
            {'name': 'Azoxystrobin (Quadris)', 'type': 'Strobilurin', 'rate': '1.0–1.5 L/ha', 'timing': 'Before disease spread; repeat in 14 days if needed'},
            {'name': 'Pyraclostrobin (Headline)', 'type': 'Strobilurin', 'rate': '0.8–1.0 L/ha', 'timing': 'At V8 to VT growth stage'},
        ],
        'ipm_note': 'Rust spores spread on wind — early detection is critical. Rotate fungicide chemical classes to prevent resistance.',
    },
    'Gray_Leaf_Spot': {
        'color': '#424242',
        'emoji': '⚫',
        'severity': 'High',
        'description': 'Caused by Cercospora zeae-maydis. Produces rectangular, tan-to-gray lesions running parallel to leaf veins. Thrives in high humidity (>90% RH) and temperatures of 25–30°C. One of the most yield-limiting foliar diseases globally.',
        'recommendations': [
            ('🌱', 'Use highly resistant hybrid varieties — this is the single most cost-effective control measure.'),
            ('🔄', 'Rotate to non-corn crops for at least one year; the fungus overwinters on corn residue.'),
            ('🌾', 'Incorporate or bury crop residues post-harvest to reduce inoculum levels in the next season.'),
            ('💧', 'Avoid excessive nitrogen applications; lush growth increases disease susceptibility.'),
            ('🌬️', 'Reduce plant population density in high-risk areas to improve canopy airflow.'),
            ('👁️', 'Begin scouting at V6 stage; prioritise fields with history of GLS or minimum tillage.'),
            ('📊', 'Fungicide application is most economic when >50% of plants show lesions on the 3rd leaf below ear.'),
        ],
        'fungicides': [
            {'name': 'Azoxystrobin + Propiconazole (Quilt Xcel)', 'type': 'Strobilurin + Triazole', 'rate': '1.0–1.5 L/ha', 'timing': 'VT (tasselling) stage — optimal window'},
            {'name': 'Fluxapyroxad + Pyraclostrobin (Priaxor)', 'type': 'SDHI + Strobilurin', 'rate': '0.5–0.7 L/ha', 'timing': 'Single application at VT to R1'},
            {'name': 'Propiconazole (Tilt 250 EC)', 'type': 'Triazole', 'rate': '0.5 L/ha', 'timing': 'Early — at first lesion observation'},
            {'name': 'Mancozeb (Dithane M-45)', 'type': 'Protectant Contact', 'rate': '2.0 kg/ha', 'timing': 'Preventive; repeat every 10–14 days in wet conditions'},
        ],
        'ipm_note': 'GLS thrives under conservation tillage. Consider conventional tillage in severely affected fields. Apply strobilurin-based fungicides before disease reaches the ear leaf.',
    },
    'Healthy': {
        'color': '#2e7d32',
        'emoji': '🟢',
        'severity': 'None',
        'description': 'No disease detected. Your maize plant appears healthy with no visible foliar pathogens.',
        'recommendations': [
            ('✅', 'Continue current agronomic practices — your crop management is on track.'),
            ('👁️', 'Maintain weekly field scouting to catch any early disease or pest pressure promptly.'),
            ('💧', 'Ensure consistent irrigation or rely on well-distributed rainfall; water stress can predispose plants to disease.'),
            ('🌿', 'Apply balanced NPK fertilisation based on soil test results to maintain plant vigour.'),
            ('🐛', 'Scout for fall armyworm (Spodoptera frugiperda) and stalk borers alongside disease monitoring.'),
            ('📝', 'Record field history (disease, yields, inputs) to inform future hybrid and fungicide decisions.'),
        ],
        'fungicides': [],
        'ipm_note': 'Preventive fungicide applications on healthy crops are generally not economically justified. Focus on integrated crop management.',
    },
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
    st.session_state.page = 'landing'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Cormorant Garamond,serif;color:#1b5e20;font-size:1.5rem;'>🌽 MaizeGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a5a;font-size:0.83rem;'>AI-Powered Disease Classification</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🔬 Disease Reference")
    for name, info in DISEASE_INFO.items():
        st.markdown(
            f'<div class="disease-card">'
            f'<h4>{info["emoji"]} {name.replace("_"," ")}</h4>'
            f'<p>{info["description"][:100]}…</p>'
            f'<p style="margin-top:0.3rem;font-size:0.78rem;color:#888;">Severity: <strong style="color:{info["color"]}">{info["severity"]}</strong></p>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.divider()
    if st.session_state.logged_in:
        st.markdown(f"👤 **{st.session_state.user['username']}**")
        st.markdown(f"<span style='font-size:0.8rem;color:#888;'>Role: {st.session_state.user['role']}</span>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            log_action(st.session_state.user['id'], st.session_state.user['username'], "LOGOUT")
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'landing'
            st.rerun()

# ── LANDING PAGE ──────────────────────────────────────────────────────────────
def page_landing():
    # ── Hero ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-corn-large">🌽</div>
        <div class="hero-badge">🔬 AI-Powered Precision Agriculture</div>
        <div class="hero-title">Protect Your Maize.<br><span>Detect Disease Early.</span></div>
        <div class="hero-subtitle">
            Upload a photo of any maize leaf and MaizeGuard AI instantly identifies blight, rust,
            gray leaf spot, or a healthy plant — with tailored treatment recommendations.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-num">4</div>
                <div class="hero-stat-label">Disease Classes</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">CNN</div>
                <div class="hero-stat-label">Deep Learning Model</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">&lt;3s</div>
                <div class="hero-stat-label">Analysis Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature pills ──
    st.markdown("""
    <div class="feature-row">
        <div class="feature-pill">🧠 Deep Learning CNN</div>
        <div class="feature-pill">📋 Detailed Treatment Plans</div>
        <div class="feature-pill">💊 Fungicide Recommendations</div>
        <div class="feature-pill">📊 Confidence Scoring</div>
        <div class="feature-pill">📥 Downloadable Reports</div>
        <div class="feature-pill">🔒 Secure & Private</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disease overview cards ──
    st.markdown("<div class='section-title'>Detectable Diseases</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>MaizeGuard AI is trained to identify four classes across diverse maize leaf conditions.</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    disease_list = [
        ('Blight', '🔴', '#b71c1c', 'High', 'Large tan/grayish lesions. Caused by Exserohilum turcicum. Rapid spread in warm, humid weather.'),
        ('Common Rust', '🟠', '#e65100', 'Medium', 'Brick-red pustules on both leaf surfaces. Wind-dispersed spores from Puccinia sorghi.'),
        ('Gray Leaf Spot', '⚫', '#424242', 'High', 'Rectangular gray-tan lesions along leaf veins. Thrives at high humidity. Major yield threat.'),
        ('Healthy', '🟢', '#2e7d32', 'None', 'No disease detected. Plant shows normal, vigorous growth with no foliar pathogens.'),
    ]

    for i, (name, emoji, color, severity, desc) in enumerate(disease_list):
        with cols[i]:
            st.markdown(f"""
            <div style="background:white;border:1px solid {color}33;border-top:4px solid {color};border-radius:12px;padding:1.2rem;height:100%;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                <div style="font-size:2.2rem;margin-bottom:0.5rem;">{emoji}</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:700;color:#0d2b12;margin-bottom:0.4rem;">{name}</div>
                <div style="font-size:0.78rem;font-weight:600;color:{color};background:{color}15;display:inline-block;padding:0.15rem 0.6rem;border-radius:4px;margin-bottom:0.7rem;">Severity: {severity}</div>
                <div style="font-size:0.83rem;color:#5a7a5a;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works ──
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Three simple steps from photo to actionable treatment plan.</div>", unsafe_allow_html=True)

    hc1, hc2, hc3 = st.columns(3)
    steps = [
        ('01', '📸', 'Upload a Photo', 'Take a clear photo of the maize leaf and upload it directly in the app. JPG or PNG formats are supported.'),
        ('02', '🤖', 'AI Analysis', 'Our CNN model processes the image in seconds, analysing patterns, colours, and textures to classify the condition.'),
        ('03', '📋', 'Get Recommendations', 'Receive a full diagnosis with confidence scores, treatment steps, and specific fungicide recommendations to act fast.'),
    ]
    for col, (num, icon, title, desc) in zip([hc1, hc2, hc3], steps):
        with col:
            st.markdown(f"""
            <div style="background:white;border:1px solid #d4e6d4;border-radius:12px;padding:1.4rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;color:#c9a84c;font-weight:700;opacity:0.5;line-height:1;">{num}</div>
                <div style="font-size:1.8rem;margin:0.4rem 0;">{icon}</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:700;color:#0d2b12;margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.84rem;color:#5a7a5a;line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── CTA ──
    cta1, cta2, cta3 = st.columns([1, 1.2, 1])
    with cta2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d2b12,#1a5c24);border-radius:16px;padding:2.5rem 2rem;text-align:center;box-shadow:0 8px 30px rgba(13,43,18,0.25);">
            <div style="font-size:3rem;margin-bottom:0.5rem;">🌽</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.8rem;color:white;font-weight:700;margin-bottom:0.6rem;">Ready to protect your crop?</div>
            <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-bottom:1.5rem;">Sign in or create a free account to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔑 Sign In", use_container_width=True):
                st.session_state.page = 'login'; st.rerun()
        with b2:
            if st.button("✨ Create Account", use_container_width=True):
                st.session_state.page = 'register'; st.rerun()

# ── AUTH PAGES ────────────────────────────────────────────────────────────────
def page_login():
    c1,c2,c3 = st.columns([1,1.4,1])
    with c2:
        st.markdown("<div style='text-align:center;font-family:Cormorant Garamond,serif;font-size:2.4rem;color:#0d2b12;font-weight:700;'>🌽 MaizeGuard AI</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#5a7a5a;font-size:0.95rem;margin-bottom:1.5rem;'>Maize Disease Classification Platform</p>", unsafe_allow_html=True)
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
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.page = 'landing'; st.rerun()

def page_register():
    c1,c2,c3 = st.columns([1,1.4,1])
    with c2:
        st.markdown("<div style='text-align:center;font-family:Cormorant Garamond,serif;font-size:2.2rem;color:#0d2b12;font-weight:700;'>🌽 Create Account</div>", unsafe_allow_html=True)
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

# ── RECOMMENDATION WIDGET ─────────────────────────────────────────────────────
def render_recommendations(pred_class: str, info: dict):
    """Render rich recommendations and fungicide table for a prediction."""

    if pred_class == 'Healthy':
        st.markdown("""
        <div class="rec-section">
            <h4>✅ Crop Management Recommendations</h4>
        """, unsafe_allow_html=True)
        for icon, text in info['recommendations']:
            st.markdown(f'<div class="rec-step"><span class="rec-icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.8rem;color:#5a7a5a;margin-top:0.8rem;font-style:italic;">💡 {info["ipm_note"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Field management steps
    st.markdown("""
    <div class="rec-section">
        <h4>🌾 Field Management Steps</h4>
    """, unsafe_allow_html=True)
    for icon, text in info['recommendations']:
        st.markdown(f'<div class="rec-step"><span class="rec-icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.8rem;color:#5a7a5a;margin-top:0.8rem;font-style:italic;">💡 IPM Note: {info["ipm_note"]}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fungicide table
    if info.get('fungicides'):
        st.markdown("""
        <div class="rec-section" style="margin-top:0.8rem;">
            <h4>💊 Recommended Fungicides</h4>
        """, unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:0.84rem;">
            <thead>
                <tr style="background:#e8f5e8;color:#0d2b12;">
                    <th style="padding:0.55rem 0.8rem;text-align:left;border-radius:6px 0 0 0;">Product (Active Ingredient)</th>
                    <th style="padding:0.55rem 0.8rem;text-align:left;">Class</th>
                    <th style="padding:0.55rem 0.8rem;text-align:left;">Application Rate</th>
                    <th style="padding:0.55rem 0.8rem;text-align:left;border-radius:0 6px 0 0;">Timing</th>
                </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)
        for i, f in enumerate(info['fungicides']):
            bg = "#ffffff" if i % 2 == 0 else "#f7faf7"
            st.markdown(f"""
            <tr style="background:{bg};border-bottom:1px solid #e8f5e8;">
                <td style="padding:0.5rem 0.8rem;font-weight:600;color:#1b5e20;">{f['name']}</td>
                <td style="padding:0.5rem 0.8rem;color:#5a7a5a;">{f['type']}</td>
                <td style="padding:0.5rem 0.8rem;font-family:monospace;color:#0d2b12;">{f['rate']}</td>
                <td style="padding:0.5rem 0.8rem;color:#5a7a5a;">{f['timing']}</td>
            </tr>
            """, unsafe_allow_html=True)
        st.markdown("""
            </tbody>
        </table>
        <p style="font-size:0.78rem;color:#888;margin-top:0.7rem;">
            ⚠️ Always read and follow label instructions. Rotate between chemical classes to prevent fungicide resistance.
            Observe pre-harvest intervals (PHI) specific to your country's regulations.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── DASHBOARD PAGE ────────────────────────────────────────────────────────────
def page_dashboard():
    model, device, model_loaded = load_model()

    st.markdown("<div class='section-title'>🌽 Maize Disease Classifier</div>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Upload a maize leaf image to detect disease and receive a full treatment plan</p>", unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.warning("⚠️ Model file **corn_disease_checkpoint.pth** not found. Running in demo mode.")

    uploaded = st.file_uploader("Upload Maize Leaf Image", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Analysing leaf…"):
                if model_loaded:
                    pred_class, confidence, all_probs = predict_image(img, model, device)
                else:
                    import random
                    pred_class = random.choice(CLASS_NAMES)
                    confidence = round(random.uniform(70, 98), 2)
                    rem = 100 - confidence
                    others = [c for c in CLASS_NAMES if c != pred_class]
                    splits = sorted([random.uniform(0, rem) for _ in range(2)])
                    all_probs = {
                        pred_class: confidence,
                        others[0]: round(splits[0], 2),
                        others[1]: round(splits[1] - splits[0], 2),
                        others[2]: round(rem - splits[1], 2),
                    }

            info = DISEASE_INFO[pred_class]

            # Result box
            st.markdown(
                f'<div class="prediction-box" style="background:linear-gradient(135deg,{info["color"]},{info["color"]}cc);">'
                f'<div style="font-size:2.5rem;">{info["emoji"]}</div>'
                f'<h2>{pred_class.replace("_"," ")}</h2>'
                f'<p>Confidence: {confidence:.1f}%</p>'
                f'<div class="confidence-bar-bg"><div style="background:white;height:10px;border-radius:20px;width:{confidence}%;"></div></div>'
                f'<p style="font-size:0.85rem;">Severity: {info["severity"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"**ℹ️ About this condition:** {info['description']}")

            st.markdown("**📊 All Class Probabilities**")
            for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                st.progress(int(prob), text=f"{cls.replace('_',' ')}: {prob:.1f}%")

        # ── Full-width recommendations ──
        st.markdown("<br>", unsafe_allow_html=True)
        render_recommendations(pred_class, info)

        # ── Save & report ──
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_prediction(
            st.session_state.user['id'], st.session_state.user['username'],
            uploaded.name, pred_class, confidence, all_probs
        )
        log_action(
            st.session_state.user['id'], st.session_state.user['username'],
            "PREDICTION", f"{uploaded.name} → {pred_class} ({confidence:.1f}%)"
        )

        # Build text report
        fung_block = ""
        if info.get('fungicides'):
            fung_block = "\nFUNGICIDE RECOMMENDATIONS\n--------------------------\n"
            for f in info['fungicides']:
                fung_block += f"  {f['name']} ({f['type']})\n    Rate: {f['rate']}\n    Timing: {f['timing']}\n\n"

        rec_block = "\nFIELD MANAGEMENT STEPS\n-----------------------\n"
        for icon, step in info['recommendations']:
            rec_block += f"  • {step}\n"

        report = (
            f"MAIZEGUARD AI — DISEASE CLASSIFICATION REPORT\n"
            f"==============================================\n"
            f"Date/Time   : {ts}\n"
            f"User        : {st.session_state.user['username']}\n"
            f"File        : {uploaded.name}\n\n"
            f"RESULT\n------\n"
            f"Predicted Disease : {pred_class.replace('_',' ')}\n"
            f"Confidence        : {confidence:.2f}%\n"
            f"Severity Level    : {info['severity']}\n\n"
            f"Description:\n{info['description']}\n\n"
            f"ALL CLASS PROBABILITIES\n-----------------------\n"
            + "\n".join([f"  {k.replace('_',' '):<20}: {v:.2f}%" for k,v in sorted(all_probs.items(), key=lambda x:-x[1])])
            + rec_block
            + fung_block
            + f"\nIPM NOTE\n--------\n  {info['ipm_note']}\n\n"
            f"--------------------------------------------------\n"
            f"Generated by MaizeGuard AI | {ts}\n"
        )

        st.markdown("---")
        st.download_button(
            "📥 Download Full Report (.txt)",
            data=report,
            file_name=f"maizeguard_{uploaded.name.split('.')[0]}_{ts.replace(' ','_').replace(':','-')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ── MY PREDICTIONS PAGE ───────────────────────────────────────────────────────
def page_my_predictions():
    st.markdown("<div class='section-title'>📋 My Predictions</div>", unsafe_allow_html=True)
    st.divider()
    rows = supabase.table("predictions").select("*").eq("user_id", st.session_state.user['id']).order("timestamp", desc=True).execute().data
    if not rows:
        st.info("No predictions yet. Upload a maize leaf image to get started!")
        return
    st.markdown(f"**Total predictions: {len(rows)}**")
    for row in rows:
        info = DISEASE_INFO.get(row['predicted_class'], DISEASE_INFO['Healthy'])
        st.markdown(
            f'<div class="log-row">{info["emoji"]} &nbsp;<strong>{row["predicted_class"].replace("_"," ")}</strong>'
            f' &nbsp;·&nbsp; Confidence: <strong>{row["confidence"]:.1f}%</strong>'
            f' &nbsp;·&nbsp; File: {row["filename"]}'
            f' &nbsp;·&nbsp; <span style="color:#888;">{str(row["timestamp"])[:19]}</span></div>',
            unsafe_allow_html=True
        )

# ── ADMIN PAGE ────────────────────────────────────────────────────────────────
def page_admin():
    if st.session_state.user['role'] != 'admin':
        st.error("Access denied."); return

    from collections import Counter

    st.markdown("<div class='section-title'>🛠️ Admin Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>System overview, user management and audit logs</p>", unsafe_allow_html=True)
    st.divider()

    all_users = supabase.table("users").select("id,username,email,role,created_at,is_active").order("created_at", desc=True).execute().data
    all_preds = supabase.table("predictions").select("*").order("timestamp", desc=True).execute().data
    all_logs  = supabase.table("logs").select("*").order("timestamp", desc=True).limit(200).execute().data

    counts = Counter([p['predicted_class'] for p in all_preds]) if all_preds else {}
    total_users  = len([u for u in all_users if u['role'] != 'admin'])
    active_users = len([u for u in all_users if u['is_active'] and u['role'] != 'admin'])

    st.markdown("### 📈 Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("👥 Total Users",       total_users)
    m2.metric("✅ Active Users",       active_users)
    m3.metric("📊 Total Predictions", len(all_preds))
    m4.metric("🔴 Blight",            counts.get('Blight', 0))
    m5.metric("🟠 Common Rust",       counts.get('Common_Rust', 0))
    m6.metric("⚫ Gray Leaf Spot",    counts.get('Gray_Leaf_Spot', 0))

    st.markdown("<br>", unsafe_allow_html=True)

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

# ── ROUTER ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.page == 'register':
        page_register()
    elif st.session_state.page == 'login':
        page_login()
    else:
        page_landing()
else:
    is_admin = st.session_state.user['role'] == 'admin'

    if is_admin and st.session_state.page in ('dashboard', 'my_predictions', 'landing'):
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
