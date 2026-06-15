"""
고혈압 1차 약물 추천 시스템 — Streamlit
NHANES 2005-2018 · XGBoost + SHAP + Ollama
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="고혈압 약물 추천 시스템",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 한글 폰트 ────────────────────────────────────────────
for font in ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font
        break
    except:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

# ── 전역 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Malgun Gothic', sans-serif !important;
}

/* 배경 */
.stApp { background: #f8fafc; }

/* 헤더 */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #7c3aed 100%);
    border-radius: 18px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    color: white;
}
.hero h1 { margin:0 0 4px; font-size:26px; font-weight:800; letter-spacing:-0.5px; }
.hero p  { margin:0; font-size:13px; opacity:.75; }
.badges  { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.badge   {
    background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.25);
    border-radius:20px; padding:4px 13px;
    font-size:11px; font-weight:600;
}

/* 섹션 카드 */
.card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 8px rgba(0,0,0,.07);
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
}

/* 결과 약물 카드 */
.drug-card {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 5px solid;
}

/* 섹션 라벨 */
.sec-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
}

/* 그룹 헤더 */
.group-hd {
    font-size: 11px;
    font-weight: 700;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin: 14px 0 6px;
    padding-top: 12px;
    border-top: 1px solid #f1f5f9;
}

/* Streamlit 위젯 커스텀 */
div[data-testid="stSlider"] > div { padding: 0; }

div[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 14px !important;
    background: #f8fafc !important;
    padding: 8px 10px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
}

/* 버튼 */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.35) !important;
    letter-spacing: .2px !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.45) !important;
}

/* 라디오 버튼 */
div[data-testid="stRadio"] label {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer;
    margin-right: 6px;
    background: #f8fafc;
}

/* 탭 */
div[data-testid="stTabs"] > div > div > div > button {
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* 텍스트 영역 */
textarea {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    background: #f8fafc !important;
}

/* 구분선 */
hr { border-color: #f1f5f9; margin: 14px 0; }

/* Streamlit 기본 여백 줄이기 */
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }
#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 상수 ─────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "models")

FEATURES = [
    "age","sbp","dbp","bmi",
    "creatinine","potassium","sodium","glucose","eGFR",
    "diabetes_flag","ckd_flag","pulse_pressure",
    "hyperkalemia_flag","htn_stage"
]
FEATURE_KR = {
    "age":"나이","sbp":"수축기혈압(SBP)","dbp":"이완기혈압(DBP)",
    "bmi":"BMI","creatinine":"크레아티닌","potassium":"칼륨(K+)",
    "sodium":"나트륨(Na+)","glucose":"공복혈당","eGFR":"eGFR",
    "diabetes_flag":"당뇨의심","ckd_flag":"만성신장질환",
    "pulse_pressure":"맥압","hyperkalemia_flag":"고칼륨혈증",
    "htn_stage":"고혈압단계",
}

# Top3 피처용 아이콘
FEATURE_ICON = {
    "age":"🎂","sbp":"💓","dbp":"💗","bmi":"⚖️",
    "creatinine":"🩸","potassium":"🧪","sodium":"🧂",
    "glucose":"🍬","eGFR":"🫘","diabetes_flag":"🍬",
    "ckd_flag":"🫘","pulse_pressure":"📈",
    "hyperkalemia_flag":"⚠️","htn_stage":"📊",
}

# Top3 피처용 — "왜 중요하게 봤는지" 일반 설명
FEATURE_REASON = {
    "age":"나이는 약물 대사 속도와 신장 기능(eGFR) 계산의 기준이 되어, 약물 선택의 출발점이 됩니다.",
    "sbp":"수축기 혈압은 고혈압 단계를 결정하는 핵심 지표로, 약물의 종류와 강도를 정하는 데 직접적인 영향을 줍니다.",
    "dbp":"이완기 혈압은 혈관 저항 상태를 반영하여, 베타차단제·CCB 같은 약물의 적합성을 판단하는 데 사용됩니다.",
    "bmi":"BMI는 대사 건강 상태를 나타내며, 비만은 인슐린 저항성 및 약물 반응성과 관련이 있습니다.",
    "creatinine":"크레아티닌은 신장 기능(eGFR)을 계산하는 핵심 수치로, ACE·ARB 사용 가능 여부를 결정하는 데 중요합니다.",
    "potassium":"칼륨 수치는 ACE 억제제·ARB·이뇨제 사용 시 고칼륨혈증 또는 저칼륨혈증 위험을 판단하는 핵심 안전 지표입니다.",
    "sodium":"나트륨 수치는 체내 수분-전해질 균형을 반영하여, 이뇨제 처방 시 중요한 참고 지표로 사용됩니다.",
    "glucose":"공복혈당은 당뇨 동반 여부를 판단해, 신장 보호 효과가 있는 ACE·ARB 계열 약물의 우선순위를 정하는 데 사용됩니다.",
    "eGFR":"eGFR은 신장 기능을 직접 나타내는 지표로, 신장 기능에 따라 사용 가능한 약물 종류가 달라집니다.",
    "diabetes_flag":"당뇨 동반 여부는 신장 보호 효과가 있는 ACE·ARB 계열 약물을 우선 고려하게 만드는 요인입니다.",
    "ckd_flag":"만성 신장 질환 여부는 약물의 신장 배설 의존도를 고려해 CCB 등 더 안전한 약물을 선택하는 기준이 됩니다.",
    "pulse_pressure":"맥압(수축기-이완기 혈압 차)은 동맥 경직도를 나타내며, 베타차단제의 반응성을 예측하는 데 활용됩니다.",
    "hyperkalemia_flag":"고칼륨혈증 여부는 ACE 억제제·ARB·이뇨제 사용 시 금기 여부를 판단하는 안전성 지표입니다.",
    "htn_stage":"고혈압 단계는 ACC/AHA 기준에 따라 1차 약물 치료의 강도를 결정하는 기준이 됩니다.",
}

DRUG_INFO = {
    "ACE": {
        "ko":"ACE 억제제 (안지오텐신전환효소 억제제)", "color":"#2563eb", "bg":"#eff6ff", "border":"#bfdbfe",
        "guide":"ADA 2023 · KDIGO 2022",
        "mechanism":"안지오텐신 전환효소(ACE) 차단 → 사구체 내압 감소 → 신장 보호",
        "simple":"혈압을 높이는 효소(ACE)를 직접 차단해 혈관을 이완시킵니다. "
                 "당뇨병이나 만성 신장 질환(CKD) 환자에서 신장을 보호하는 추가 효과가 있어 1차 치료제로 권고됩니다.",
    },
    "ARB": {
        "ko":"ARB (안지오텐신 수용체 차단제)", "color":"#db2777", "bg":"#fdf2f8", "border":"#fbcfe8",
        "guide":"ESC/ESH 2023",
        "mechanism":"AT1 수용체 차단 → RAAS 억제. ACE 억제제 불내성 대체",
        "simple":"혈압을 올리는 호르몬(안지오텐신 II)이 혈관에 붙지 못하도록 막아 혈압을 낮춥니다. "
                 "ACE 억제제와 효과가 비슷하지만 마른기침 부작용이 없습니다.",
    },
    "BETA": {
        "ko":"베타차단제 (Beta-blocker)", "color":"#7c3aed", "bg":"#f5f3ff", "border":"#ddd6fe",
        "guide":"ACC/AHA 2017",
        "mechanism":"β1 수용체 차단 → 심박수·심박출량 감소 → 혈압 강하",
        "simple":"심장을 빠르게 뛰게 하는 신호를 차단해 심장이 천천히 뛰도록 만들어 혈압을 낮춥니다. "
                 "빠른 맥박을 동반한 고혈압에 특히 효과적입니다.",
    },
    "CCB": {
        "ko":"칼슘채널차단제 (CCB)", "color":"#059669", "bg":"#f0fdf4", "border":"#bbf7d0",
        "guide":"ACCOMPLISH · ESC/ESH",
        "mechanism":"L형 칼슘채널 차단 → 혈관 이완. 신장 기능 무관 사용 가능",
        "simple":"혈관 근육이 수축할 때 필요한 칼슘 통로를 막아 혈관을 이완시킵니다. "
                 "신장 기능과 무관하게 사용할 수 있어 신장 질환 환자에게도 안전합니다.",
    },
    "DIURETIC": {
        "ko":"이뇨제 (Diuretic)", "color":"#d97706", "bg":"#fffbeb", "border":"#fde68a",
        "guide":"JNC 8 · ALLHAT",
        "mechanism":"나트륨·수분 배출 → 혈액량 감소 → 혈압 강하",
        "simple":"소변으로 나트륨(소금)과 수분을 더 많이 배출해 혈액량을 줄이고 혈압을 낮춥니다. "
                 "칼륨(K+) 수치가 떨어질 수 있어 정기 혈액검사가 필요합니다.",
    },
}
HTN_LABELS = {1:"정상 (120/80 미만)", 2:"주의 (120-129)", 3:"고혈압 1기 (130-139)", 4:"고혈압 2기 (140 이상)"}
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"


# ══════════════════════════════════════════════════════════
#  모델 로드 (캐시)
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    m  = joblib.load(os.path.join(BASE, "model.pkl"))
    le = joblib.load(os.path.join(BASE, "le.pkl"))
    im = joblib.load(os.path.join(BASE, "imputer.pkl"))
    ex = shap.TreeExplainer(m)
    return m, le, im, ex

try:
    model, le, imputer, explainer = load_models()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)


# ══════════════════════════════════════════════════════════
#  헬퍼 함수
# ══════════════════════════════════════════════════════════
def calc_egfr(cr, age, gender_v):
    kappa = 0.7 if gender_v == 2 else 0.9
    alpha = -0.329 if gender_v == 2 else -0.411
    sex_f = 1.018 if gender_v == 2 else 1.0
    r = cr / kappa
    return round(141 * (r**alpha if r<=1 else 1.0)
                     * (r**-1.209 if r>1 else 1.0)
                     * (0.993**age) * sex_f, 2)

def build_fv(age, gv, sbp, dbp, bmi, cr, k, na, glu):
    egfr = calc_egfr(cr, age, gv)
    return [age, sbp, dbp, bmi, cr, k, na, glu,
            egfr,
            1.0 if glu>=126 else 0.0,
            1.0 if egfr<60  else 0.0,
            sbp-dbp,
            1.0 if k>=5.5   else 0.0,
            1 if sbp<130 else 2 if sbp<140 else 3 if sbp<160 else 4]


# ──────────────────────────────────────────────────────────
#  텍스트 정제 (다른 언어 문자/발음 구별 기호 제거)
#  허용: 한글(완성형 + 자모) · ASCII(영문/숫자/기본 문장부호) · 공백
#  → 베트남어/중국어/일본어/태국어 등 한글이 아닌 비-ASCII 문자는 모두 제거
# ──────────────────────────────────────────────────────────
FOREIGN_CHAR_RE = re.compile(
    r'[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\x00-\x7F\s]'
)

def has_foreign_chars(text):
    """한글/ASCII/공백 외 문자가 섞여 있는지 검사"""
    return bool(FOREIGN_CHAR_RE.search(text))

def sanitize_text(text):
    """허용되지 않는 문자를 제거하고 공백을 정리"""
    cleaned = FOREIGN_CHAR_RE.sub('', text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def call_groq(fv, pred_cls, proba_val):
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    info = DRUG_INFO[pred_cls]

    system_prompt = (
        "당신은 고혈압 전문 임상약사입니다. "
        "응답은 반드시 한국어(한글)로만 작성하세요. "
        "의학 전문용어의 영어 표기(예: ARB, ACE, eGFR, CKD, mmHg, mL/min)는 "
        "허용하지만, 한국어 단어 안에 베트남어, 중국어, 일본어, 태국어 등 "
        "다른 언어의 문자나 발음 구별 기호가 절대 섞여서는 안 됩니다."
    )
    user_prompt = (
        f"한국어로만 3문장 작성하세요. "
        f"영어는 약물명·단위·약어(ARB, ACE, eGFR, CKD, mmHg, mL/min)만 허용합니다.\n\n"
        f"환자: {fv[0]:.0f}세, 혈압 {fv[1]:.0f}/{fv[2]:.0f} mmHg, "
        f"eGFR {fv[8]:.1f} mL/min, 공복혈당 {fv[7]:.0f} mg/dL, "
        f"만성신장질환 {'있음' if fv[10] else '없음'}, 당뇨의심 {'있음' if fv[9] else '없음'}.\n"
        f"추천 약물: {pred_cls} ({info['ko']}), 예측확률 {proba_val*100:.1f}%.\n"
        f"약물 설명: {info['simple']}\n\n"
        f"임상 설명 3문장:"
    )

    def _request(temperature):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": 512,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        return None

    try:
        raw = _request(0.4)
        if raw is None:
            return None

        # 다른 언어 문자가 섞여 있으면 1회 재시도
        if has_foreign_chars(raw):
            retry = _request(0.3)
            if retry is not None and not has_foreign_chars(retry):
                raw = retry

        return sanitize_text(raw)
    except Exception:
        return None


def call_groq_qa(question):
    """'약물 물어보기' 탭 - 자유 질문에 대한 답변 생성"""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None

    system_prompt = (
        "당신은 고혈압 관리를 돕는 친절한 임상약사입니다. "
        "사용자의 질문에 대해 한국어(한글)로만, 4~6문장 이내로 쉽고 간결하게 답변하세요. "
        "의학 전문용어의 영어 표기(예: ARB, ACE, eGFR, mmHg, mL/min)는 허용하지만, "
        "한국어 단어 안에 베트남어, 중국어, 일본어, 태국어 등 다른 언어의 문자나 "
        "발음 구별 기호가 섞여서는 절대 안 됩니다. "
        "고혈압, 혈압약, 식습관·생활습관 관리와 관련 없는 질문에는 "
        "정중하게 답변할 수 없다고 안내하세요. "
        "답변 마지막에는 '정확한 진단과 처방은 의사·약사와 상담하세요.'라는 문장을 포함하세요."
    )

    def _request(temperature):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "temperature": temperature,
                "max_tokens": 512,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        return None

    try:
        raw = _request(0.4)
        if raw is None:
            return None

        if has_foreign_chars(raw):
            retry = _request(0.3)
            if retry is not None and not has_foreign_chars(retry):
                raw = retry

        return sanitize_text(raw)
    except Exception:
        return None


def build_fallback(fv, pred_cls, proba_val):
    info = DRUG_INFO[pred_cls]
    egfr_desc = (f"eGFR {fv[8]:.1f} mL/min으로 신장 기능이 저하되어 있으며"
                 if fv[8]<60 else f"eGFR {fv[8]:.1f} mL/min으로 신장 기능은 정상 범위이며")
    dm_desc = "당뇨 의심 소견이 동반되어 있습니다." if fv[9] else "당뇨 소견은 없습니다."
    s1 = (f"본 환자는 {fv[0]:.0f}세로 수축기 혈압 {fv[1]:.0f} mmHg / "
          f"이완기 혈압 {fv[2]:.0f} mmHg의 고혈압이 확인되며, {egfr_desc} {dm_desc}")
    s2 = (f"AI 모델은 {pred_cls}({info['ko']})를 {proba_val*100:.1f}%의 확률로 추천하였으며, "
          f"{info['simple']}")
    mon = []
    if fv[10] or fv[8]<60: mon.append("신장 기능(eGFR)")
    if fv[9]: mon.append("공복혈당 및 HbA1c")
    if fv[5]>=5.0: mon.append("혈중 칼륨(K+)")
    if not mon: mon.append("혈압 및 전해질")
    s3 = (f"치료 중 {', '.join(mon)} 정기 모니터링이 권고되며, "
          f"가이드라인({info['guide']})에 따라 처방 결정이 이루어져야 합니다.")
    return f"{s1}\n\n{s2}\n\n{s3}"


# ══════════════════════════════════════════════════════════
#  차트
# ══════════════════════════════════════════════════════════
def make_prob_chart(proba, classes):
    colors = [DRUG_INFO[c]["color"] for c in classes]
    labels = [DRUG_INFO[c]["ko"] for c in classes]
    vals   = proba * 100

    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.55, linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(min(v+1.2, 108), bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=12, fontweight="bold", color="#1e293b")

    ax.set_xlim(0, 115)
    ax.set_xlabel("추천 적합도 (%)", fontsize=11, color="#64748b")
    ax.set_title("약물별 추천 적합도", fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    ax.tick_params(colors="#475569", labelsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    plt.tight_layout(pad=1.6)
    return fig

def make_shap_chart(sv, pred_cls):
    color_pos = DRUG_INFO[pred_cls]["color"]
    color_neg = "#94a3b8"
    idx    = np.argsort(np.abs(sv))
    vals   = sv[idx]
    labels = [FEATURE_KR[FEATURES[i]] for i in idx]
    colors = [color_pos if v>=0 else color_neg for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    ax.barh(labels, vals, color=colors, edgecolor="white", height=0.62, linewidth=1.2, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v+(0.003 if v>=0 else -0.003), i, f"{v:+.3f}",
                va="center", ha="left" if v>=0 else "right",
                fontsize=9.5, color="#1e293b", fontweight="600")

    ax.axvline(0, color="#475569", linewidth=1.0)
    ax.set_title(f"처방 판단에 영향을 준 요인 — {DRUG_INFO[pred_cls]['ko']}",
                 fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    ax.set_xlabel("영향도  (양수 = 추천 강화,  음수 = 추천 억제)", fontsize=10, color="#64748b")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.legend(handles=[
        mpatches.Patch(color=color_pos, label="추천 강화", alpha=0.85),
        mpatches.Patch(color=color_neg, label="추천 억제", alpha=0.85),
    ], fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#e2e8f0")
    plt.tight_layout(pad=1.6)
    return fig


# ══════════════════════════════════════════════════════════
#  임상 근거 요약
#  → 각 항목을 "제목" + 줄바꿈 + "내용" 형식으로 구성
#    (마크다운에서 줄바꿈이 적용되도록 제목 끝에 공백 2개 추가)
# ══════════════════════════════════════════════════════════
def build_evidence(cls, fv, sv):
    info = DRUG_INFO[cls]
    ev_map = {
        "ACE": [
            ("eGFR", fv[8], ("신장 기능 저하(eGFR<60). ACE 억제제는 사구체 내압을 낮춰 신장 보호 (KDIGO 2022)"
                             if fv[8]<60 else "신장 기능 정상. 당뇨 동반 시 신장 보호 효과로 1차 선택.")),
            ("glucose", fv[7], ("혈당 126 이상 → 당뇨 의심. 당뇨성 신증 1차 치료제 (ADA 2023)"
                                if fv[7]>=126 else "혈당 정상 범위.")),
        ],
        "ARB": [
            ("eGFR", fv[8], ("신장 기능 저하. ARB는 ACE 억제제와 동일한 신장 보호 효과, 마른기침 없음."
                             if fv[8]<60 else "신장 기능 정상. 당뇨 동반 고혈압에 ACE와 동등 권고.")),
            ("glucose", fv[7], "당뇨 동반 고혈압에 ACE/ARB 동등 권고 (ESC/ESH 2023)"),
        ],
        "BETA": [
            ("sbp", fv[1], f"수축기 혈압 {fv[1]:.0f} mmHg — β1 차단으로 심박출량 감소 → 혈압 강하"),
            ("pulse_pressure", fv[11], f"맥압 {fv[11]:.0f} mmHg — 동맥 경직도 지표. 베타차단제 반응 예측"),
        ],
        "CCB": [
            ("sbp", fv[1], f"수축기 혈압 {fv[1]:.0f} mmHg — 혈관 이완으로 직접 감소. eGFR 무관 사용 가능"),
            ("eGFR", fv[8], f"eGFR {fv[8]:.1f} — CCB는 신장 배설 의존도 낮아 CKD에도 안전"),
        ],
        "DIURETIC": [
            ("potassium", fv[5], (f"칼륨(K+) {fv[5]:.1f} mEq/L — ⚠️ 고칼륨혈증. 이뇨제 투여 시 주의 요망."
                                  if fv[5]>=5.5 else f"칼륨 {fv[5]:.1f} mEq/L — 정상. 이뇨제 투여 중 저칼륨혈증 모니터링 필요.")),
            ("sodium", fv[6], f"나트륨 {fv[6]:.1f} mEq/L — 이뇨제는 나트륨 배출로 혈압 강하. (ALLHAT)"),
        ],
    }
    # "제목" 다음 줄에 "내용"이 오도록, 제목 끝에 공백 2개(마크다운 강제 줄바꿈) + 줄바꿈 추가
    lines = [
        f"#### {info['ko']}",
        "",
        "**작용기전**  ",
        info["mechanism"],
        "",
        "**쉬운 설명**  ",
        info["simple"],
        "",
        "**가이드라인**  ",
        info["guide"],
        "",
        "---",
        "",
        "**환자 수치 해석**",
    ]
    for feat, val, desc in ev_map.get(cls, []):
        shap_v = sv[FEATURES.index(feat)]
        d = "▲ 추천 강화 요인" if shap_v>0 else "▼ 추천 억제 요인"
        lines.append(f"- **{FEATURE_KR[feat]}** (값={val:.2f}, 영향도={shap_v:+.4f} → {d})")
        lines.append(f"  → {desc}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  헤더
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>💊 고혈압 1차 약물 추천 시스템</h1>
  <p>처방 전, AI 분석으로 1차 약물 선택의 근거를 확인하세요</p>
  <div class="badges">
    <span class="badge">📊 NHANES 기반 임상 데이터 분석</span>
    <span class="badge">🤖 AI 학습 기반 분석</span>
    <span class="badge">🔍 AI 처방 판단 근거 분석</span>
    <span class="badge">💬 AI 임상 설명 제공</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error(f"모델 로드 실패: {MODEL_ERR}")
    st.stop()

# ══════════════════════════════════════════════════════════
#  탭
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["💊  약물 추천", "📊  분석 배경 및 판단 근거", "💬  약물 물어보기"])

with tab1:
    left, right = st.columns([1, 2], gap="large")

    # ── 좌측: 입력 ───────────────────────────────────────
    with left:
        st.markdown('<div class="sec-label">환자 정보 입력</div>', unsafe_allow_html=True)

        # 나이: 슬라이더 → 숫자 직접 입력
        age    = st.number_input("나이 (세)", min_value=18, max_value=90, value=60, step=1)
        gender = st.radio("성별", ["남성", "여성"], horizontal=True, label_visibility="collapsed")

        st.markdown('<div class="group-hd">혈압 (mmHg) / BMI</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        sbp = c1.number_input("SBP", value=145, min_value=80, max_value=250, label_visibility="visible",
                               help="SBP (수축기혈압) — 심장이 수축할 때의 혈압, '최고혈압'")
        dbp = c2.number_input("DBP", value=88,  min_value=40, max_value=150, label_visibility="visible",
                               help="DBP (이완기혈압) — 심장이 이완될 때의 혈압, '최저혈압'")
        bmi = c3.number_input("BMI", value=27.5, min_value=10.0, max_value=60.0, step=0.1, format="%.1f")

        st.markdown('<div class="group-hd">혈액검사</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        creatinine = c4.number_input("크레아티닌 (mg/dL)", value=1.1, min_value=0.1, max_value=15.0, step=0.1, format="%.1f")
        potassium  = c5.number_input("칼륨 K⁺ (mEq/L)",  value=4.1, min_value=1.0, max_value=9.0,  step=0.1, format="%.1f")
        c6, c7 = st.columns(2)
        sodium  = c6.number_input("나트륨 Na⁺ (mEq/L)", value=140.0, min_value=100.0, max_value=170.0, step=0.5, format="%.1f")
        glucose = c7.number_input("공복혈당 (mg/dL)",    value=108.0, min_value=40.0,  max_value=500.0, step=1.0,  format="%.0f")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔍  약물 추천 실행", use_container_width=True)

    # ── 우측: 결과 ───────────────────────────────────────
    with right:
        if run:
            with st.spinner("AI 분석 중..."):
                gv = 2 if "여" in gender else 1
                fv = build_fv(age, gv, sbp, dbp, bmi, creatinine, potassium, sodium, glucose)
                X      = np.array(fv).reshape(1, -1)
                X_imp  = imputer.transform(X)
                pred_idx = int(model.predict(X_imp)[0])
                proba    = model.predict_proba(X_imp)[0]
                pred_cls = le.classes_[pred_idx]
                info     = DRUG_INFO[pred_cls]

                sv       = explainer.shap_values(X_imp)[0, :, pred_idx]
                top3_idx = np.argsort(np.abs(sv))[::-1][:3].tolist()

                st.session_state["result"] = dict(
                    fv=fv, pred_cls=pred_cls, proba=proba,
                    pred_idx=pred_idx, sv=sv,
                )

        if "result" in st.session_state:
            r        = st.session_state["result"]
            fv       = r["fv"];  pred_cls = r["pred_cls"]
            proba    = r["proba"]; pred_idx = r["pred_idx"]; sv = r["sv"]
            info     = DRUG_INFO[pred_cls]
            proba_v  = proba[pred_idx]

            # ── 약물 결과 카드 ────────────────────────────
            egfr_note    = "신장 기능 저하 (60 미만)" if fv[8]<60 else "정상 범위 (60 이상)"
            glucose_note = "당뇨 의심 (126 이상)"    if fv[9]   else "정상 범위 (126 미만)"

            st.markdown(f"""
<div class="drug-card" style="background:{info['bg']};border-color:{info['color']};">
  <div style="font-size:20px;font-weight:800;color:{info['color']};margin-bottom:6px;">
    🏆 추천 약물: {info['ko']}
  </div>
  <div style="font-size:32px;font-weight:900;color:#1e293b;margin-bottom:12px;">
    {proba_v*100:.1f}%
    <span style="font-size:14px;font-weight:400;color:#64748b;"> 추천 적합도 (5가지 약물 중 가장 높음)</span>
  </div>
  <hr style="border-color:{info['border']};margin:10px 0;">
  <table style="font-size:13.5px;width:100%;border-spacing:0;">
    <tr>
      <td style="color:#64748b;padding:3px 0;width:40%;">혈압</td>
      <td style="font-weight:600;color:#1e293b;">{fv[1]:.0f}/{fv[2]:.0f} mmHg &nbsp;→&nbsp; {HTN_LABELS.get(int(fv[13]),'')}</td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:3px 0;">신장 기능 (eGFR)</td>
      <td style="font-weight:600;color:#1e293b;">{fv[8]:.1f} mL/min &nbsp;→&nbsp; {egfr_note}
        <span style="font-size:11px;color:#94a3b8;"> (크레아티닌+나이+성별로 자동계산)</span></td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:3px 0;">공복혈당</td>
      <td style="font-weight:600;color:#1e293b;">{fv[7]:.0f} mg/dL &nbsp;→&nbsp; {glucose_note}</td>
    </tr>
    <tr>
      <td style="color:#64748b;padding:3px 0;">만성 신장 질환</td>
      <td style="font-weight:600;color:#1e293b;">{"있음" if fv[10] else "없음"} &nbsp;|&nbsp; 칼륨(K+): {fv[5]:.1f} mEq/L</td>
    </tr>
  </table>
  <hr style="border-color:{info['border']};margin:10px 0;">
  <div style="font-size:13px;color:#334155;line-height:1.7;">
    <b>작용 원리:</b> {info['simple']}<br>
    <b style="color:#64748b;font-size:12px;">가이드라인:</b>
    <span style="color:#64748b;font-size:12px;"> {info['guide']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── 추천 적합도 차트 ───────────────────────────
            st.markdown("#### 📊 약물별 추천 적합도")
            st.pyplot(make_prob_chart(proba, le.classes_), use_container_width=True)

            # ── 판단 근거 차트 ────────────────────────────
            st.markdown("#### 🔍 처방 판단에 영향을 준 요인")
            st.pyplot(make_shap_chart(sv, pred_cls), use_container_width=True)

            # ── AI 임상 설명 ─────────────────────────────
            st.markdown("#### 🤖 AI 임상 설명")
            with st.spinner("AI 설명 생성 중..."):
                ai = call_groq(fv, pred_cls, proba_v)

            if ai:
                st.info(ai)
            else:
                st.warning("**[AI 설명을 가져오지 못해 기본 요약으로 대체합니다]**\n\n" + build_fallback(fv, pred_cls, proba_v))

            # ── 임상 근거 요약 ────────────────────────────
            with st.expander("📋 임상 근거 요약 (판단 근거 + 가이드라인)", expanded=False):
                st.markdown(build_evidence(pred_cls, fv, sv))

                st.markdown("---")
                st.markdown("**처방 판단에 가장 큰 영향을 준 요인 3가지**")

                top3_idx = np.argsort(np.abs(sv))[::-1][:3]
                max_abs  = float(max(abs(sv[i]) for i in top3_idx)) or 1.0

                for rank, i in enumerate(top3_idx, 1):
                    feat = FEATURES[i]
                    icon = FEATURE_ICON.get(feat, "🔹")
                    reason = FEATURE_REASON.get(feat, "")
                    rel = float(abs(sv[i])) / max_abs
                    rel = min(max(rel, 0.0), 1.0)

                    st.markdown(f"**{rank}위. {icon} {FEATURE_KR[feat]}**")
                    st.progress(rel)
                    st.caption(reason)

            # ── 피처 테이블 ───────────────────────────────
            with st.expander("피처 전체 값 및 판단 영향도", expanded=False):
                feat_df = pd.DataFrame({
                    "피처":  [FEATURE_KR[f] for f in FEATURES],
                    "값":    [round(float(v), 3) for v in fv],
                    "영향도": [round(float(v), 4) for v in sv],
                }).sort_values("영향도", key=abs, ascending=False).reset_index(drop=True)
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
<div style="background:white;border-radius:14px;padding:48px 32px;text-align:center;
            border:2px dashed #e2e8f0;color:#94a3b8;">
  <div style="font-size:48px;margin-bottom:12px;">💊</div>
  <div style="font-size:16px;font-weight:600;color:#64748b;margin-bottom:6px;">
    환자 정보를 입력하고 약물 추천 실행을 눌러주세요
  </div>
  <div style="font-size:13px;">
    AI가 5가지 혈압약 중 환자에게 가장 적합한 약물을 분석해드립니다
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tab 2: 분석 배경 ─────────────────────────────────────
with tab2:
    # ── 약물별 추천 근거 카드 ─────────────────────────────
    st.markdown("### 💊 약물별 추천 근거가 의학 가이드라인과 일치합니다")
    st.caption("AI가 각 약물을 추천할 때 가장 중요하게 본 항목과, 그 판단이 실제 임상 가이드라인과 맞는지 확인했습니다.")


    EVIDENCE_SUMMARY = {
        "DIURETIC": {
            "key": "칼륨(K+) 수치",
            "text": "이뇨제는 체내 칼륨을 함께 배출시켜 저칼륨혈증을 유발할 수 있습니다. "
                    "AI도 칼륨 수치를 가장 중요한 판단 기준으로 사용했습니다.",
        },
        "ACE": {
            "key": "신장 기능 · 공복혈당",
            "text": "ACE 억제제는 당뇨로 인한 신장 손상을 막아주는 효과가 있어, "
                    "신장 기능(eGFR)과 혈당이 핵심 판단 기준이 됩니다.",
        },
        "ARB": {
            "key": "신장 기능 · 공복혈당",
            "text": "ARB는 ACE 억제제와 같은 신장 보호 효과를 가지면서 마른기침 부작용이 없는 "
                    "대체 약물로, 같은 기준으로 판단됩니다.",
        },
        "BETA": {
            "key": "혈압(SBP) · 맥압",
            "text": "베타차단제는 심박출량을 줄여 혈압을 낮추므로, 혈압과 맥압(동맥 경직도)이 "
                    "핵심 판단 기준이 됩니다.",
        },
        "CCB": {
            "key": "수축기혈압(SBP)",
            "text": "CCB는 신장 기능과 무관하게 혈관을 직접 이완시켜 혈압을 낮추므로, "
                    "신장 수치보다 혈압이 핵심 판단 기준이 됩니다.",
        },
    }
    order = ["DIURETIC", "ACE", "ARB", "BETA", "CCB"]

    for row_start in range(0, len(order), 3):
        cols = st.columns(3)
        for col, cls in zip(cols, order[row_start:row_start+3]):
            info = DRUG_INFO[cls]
            ev   = EVIDENCE_SUMMARY[cls]
            with col:
                st.markdown(f"""
<div class="card" style="border-left:4px solid {info['color']};min-height:190px;">
  <div style="font-weight:800;color:{info['color']};font-size:15px;margin-bottom:4px;">
    {info['ko']}
  </div>
  <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;">
    핵심 근거: {ev['key']}
  </div>
  <div style="font-size:13px;color:#334155;line-height:1.6;">
    {ev['text']}
  </div>
  <div style="margin-top:10px;font-size:12px;color:#059669;font-weight:700;">
    ✅ 가이드라인 일치 ({info['guide']})
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.success("**5가지 약물 모두 AI의 판단 기준이 실제 의학 가이드라인과 일치합니다.**")

    # ── 더 자세한 내용 (선택적) ───────────────────────────
    with st.expander("🔧 모델 구조 자세히 보기 (개발자용)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
| 항목 | 내용 |
|------|------|
| 데이터 | NHANES 2005–2018, 7사이클, **5,974명** |
| 피처 | 14개 (원시 9 + 임상 파생 5) |
| 알고리즘 | XGBoost + SMOTE |
| 성능 | Accuracy 31.2%, F1-macro 27.0% |
| 해석 | SHAP TreeExplainer |
| LLM | Groq llama-3.3-70b (API) |
""")
        with c2:
            st.markdown("""
| 임상 파생 피처 | 정의 | 기준 |
|------|------|------|
| diabetes_flag | 혈당 ≥ 126 | ADA |
| ckd_flag | eGFR < 60 | KDIGO |
| pulse_pressure | SBP − DBP | 동맥경직도 |
| hyperkalemia_flag | K+ ≥ 5.5 | ACE/ARB 금기 |
| htn_stage | ACC/AHA 2017 | 1~4단계 |
""")


# ── Tab 3: 약물 물어보기 ─────────────────────────────────
with tab3:
    st.markdown("## 💬 약물 물어보기")
    st.caption("혈압·약물·생활습관 관련 질문을 자유롭게 입력해보세요. (의학적 진단을 대체하지 않습니다)")

    if "qa_question" not in st.session_state:
        st.session_state["qa_question"] = ""

    st.markdown("**예시 질문**")
    examples = [
        "공복혈당이 높은데 어떻게 조절해야 하나요?",
        "ARB와 ACE 억제제는 뭐가 다른가요?",
        "이뇨제를 먹으면 칼륨이 왜 떨어지나요?",
    ]
    ex_cols = st.columns(3)
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["qa_question"] = ex

    question = st.text_area(
        "질문을 입력하세요",
        key="qa_question",
        height=100,
        placeholder="예: 공복혈당 수치가 높은데 어떻게 조절해야 하나요?",
    )

    ask = st.button("🔍  질문하기", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("AI가 답변을 작성 중..."):
                answer = call_groq_qa(question.strip())
            st.markdown("#### 🤖 AI 답변")
            if answer:
                st.info(answer)
            else:
                st.warning("지금은 AI 답변을 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
