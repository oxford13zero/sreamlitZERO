# app.py — Bullying Dashboard
# Menu -> (1) Patrones descriptivos + LLM
#      -> (2) Comportamiento futuro y patrones + LLM
# Back to menu

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from supabase import create_client

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Bullying Dashboard", layout="wide")

# -------------------------------------------------
# Simple router (menu / descriptivo / futuro)
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "menu"  # menu | descriptivo | futuro

def go(page_name: str):
    st.session_state.page = page_name
    st.rerun()

# -------------------------------------------------
# Supabase config
# -------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# Scoring scale
# -------------------------------------------------
SCALE = {
    "Nunca": 0,
    "Sólo una vez": 1,
    "Varias veces al mes": 2,
    "Casi cada semana": 3,
    "Casi cada día": 4,
    "Sí": 2,
    "A veces": 1,
    "No": 0,
}

# -------------------------------------------------
# Supabase pagination helpers
# -------------------------------------------------
@st.cache_data(ttl=300)
def _fetch_all(table: str, batch: int = 1000) -> pd.DataFrame:
    rows = []
    start = 0
    while True:
        data = (
            supabase.table(table)
            .select("*")
            .range(start, start + batch - 1)
            .execute()
            .data
        )
        if not data:
            break
        rows.extend(data)
        if len(data) < batch:
            break
        start += batch
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def load_tables():
    return (
        _fetch_all("survey_responses"),
        _fetch_all("question_answers"),
        _fetch_all("answer_selected_options"),
        _fetch_all("question_options"),
        _fetch_all("questions"),
        _fetch_all("schools"),
    )

# -------------------------------------------------
# Data preparation
# -------------------------------------------------
def build_long_df(responses, answers, aso, qopts, questions, schools):
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    df = (
        answers
        .merge(responses, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id", suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt"))
    )

    # Avoid MergeError: rename schools.id to school_pk before merging
    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(
            columns={"id": "school_pk", "name": "school_name"}
        )
        df = df.merge(
            schools_small,
            left_on="school_id",
            right_on="school_pk",
            how="left",
        )
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    df["question_text"] = df["question_text"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["victimization"] = d["question_text"].str.contains(
        r"has sido agredido|te han molestado|te han ignorado", case=False, na=False
    ) * d["score"]

    d["cyberbullying"] = d["question_text"].str.contains(
        r"internet|mensajería|mensajes|videos|fotos", case=False, na=False
    ) * d["score"]

    d["trust_friends"] = d["question_text"].str.contains("amigos", case=False, na=False) * d["score"]
    d["trust_adults"] = d["question_text"].str.contains("adultos", case=False, na=False) * d["score"]
    d["trust_parents"] = d["question_text"].str.contains("padres", case=False, na=False) * d["score"]

    d["safety"] = d["question_text"].str.contains(
        r"me siento seguro|me gusta venir|buen lugar", case=False, na=False
    ) * d["score"]

    student = d.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        victimization=("victimization", "sum"),
        cyberbullying=("cyberbullying", "sum"),
        trust_friends=("trust_friends", "sum"),
        trust_adults=("trust_adults", "sum"),
        trust_parents=("trust_parents", "sum"),
        safety=("safety", "sum"),
    ).reset_index()

    p80 = float(student["victimization"].quantile(0.80)) if len(student) else 0.0
    student["risk_level"] = np.where(student["victimization"] >= p80, "ALTO", "MEDIO/BAJO")

    student["knows_friends"] = student["trust_friends"] > 0
    student["knows_adults"] = student["trust_adults"] > 0
    student["knows_parents"] = student["trust_parents"] > 0

    return student

# -------------------------------------------------
# LLM (Groq now, HF later)
# -------------------------------------------------
def generate_llm_report(summary: dict, mode: str) -> str:
    provider = st.secrets.get("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return _groq_report(summary, mode=mode)
    elif provider == "hf":
        return "Hugging Face not enabled yet."
    return "LLM provider not configured."

def _groq_report(summary: dict, mode: str = "descriptivo") -> str:
    api_key = st.secrets.get("GROQ_API_KEY")
    # usa el modelo que ya te funcionó
    model = st.secrets.get("LLM_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return "Groq API key missing."

    if mode == "futuro":
        prompt = f"""
Eres un experto en clima escolar, prevención del bullying y análisis de riesgo futuro.

Redacta un informe claro, en lenguaje NO técnico, dirigido a directivos
y equipos escolares, basándote ÚNICAMENTE en los datos agregados a continuación.

Incluye:
- Breve resumen del estado actual
- Proyección razonada de la situación futura (mejora, estabilidad o empeoramiento)
- Posibles grupos más vulnerables si se observan patrones
- 3 acciones preventivas concretas para los próximos 30–60 días

IMPORTANTE:
- Escribe el informe completamente en ESPAÑOL
- No utilices lenguaje técnico ni estadístico complejo
- No inventes información que no esté en los datos

DATOS AGREGADOS:
{summary}
"""

    else:
        prompt = f"""

        Eres un experto en clima escolar y prevención del bullying.

        Redacta un informe claro, en lenguaje NO técnico, dirigido a directivos
        y equipos escolares, basándote ÚNICAMENTE en los datos agregados a continuación.

        Incluye:
        - Nivel general de riesgo de victimización
        - Principales patrones de comportamiento que se puedan inferir
        - Quiénes parecen estar informados de lo que ocurre (amigos, adultos, padres)
        - 3 recomendaciones concretas y prácticas para la escuela

        IMPORTANTE:
        - Escribe el informe completamente en ESPAÑOL
        - Usa un tono profesional, empático y orientado a la acción
        - No inventes información que no esté en los datos

DATOS AGREGADOS:
{summary}
"""


    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You generate concise school safety reports."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text}")

    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------
# Shared helpers for UI
# -------------------------------------------------
def render_school_selector(student_df: pd.DataFrame) -> pd.DataFrame:
    school_list = sorted(student_df["school_name"].dropna().unique().tolist())
    school = st.sidebar.selectbox("School", ["(All)"] + school_list)
    return student_df if school == "(All)" else student_df[student_df["school_name"] == school]

def render_top_metrics(view: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", int(len(view)))
    c2.metric("High risk", int((view["risk_level"] == "ALTO").sum()))
    c3.metric("Avg victimization", float(round(view["victimization"].mean(), 2)))
    c4.metric("Avg cyberbullying", float(round(view["cyberbullying"].mean(), 2)))

def summary_descriptivo(view: pd.DataFrame) -> dict:
    return {
        "mode": "descriptivo",
        "students": int(len(view)),
        "high_risk_pct": float((view["risk_level"] == "ALTO").mean()),
        "avg_victimization": float(view["victimization"].mean()),
        "avg_cyberbullying": float(view["cyberbullying"].mean()),
        "knows_friends_pct": float(view["knows_friends"].mean()),
        "knows_adults_pct": float(view["knows_adults"].mean()),
        "knows_parents_pct": float(view["knows_parents"].mean()),
    }

def summary_futuro(view: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    # forecast: columns risk_level_pred (ALTO/MEDIO/BAJO or ALTO/MEDIO/BAJO simplified)
    return {
        "mode": "futuro",
        "students": int(len(view)),
        "current_high_risk_pct": float((view["risk_level"] == "ALTO").mean()),
        "current_avg_victimization": float(view["victimization"].mean()),
        "current_avg_safety": float(view["safety"].mean()),
        "forecast_high_risk_pct": float((forecast["risk_level_pred"] == "ALTO").mean()),
        "forecast_note": "Heuristic forecast based on current victimization & safety signals (baseline).",
    }

# -------------------------------------------------
# Simple future forecast (baseline heuristic)
# -------------------------------------------------
def make_future_forecast(view: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline forecast (no ML yet): projects next-term victimization by adding
    small noise and a safety-based drift.
    """
    if view.empty:
        return pd.DataFrame(columns=["survey_response_id", "victimization_pred", "risk_level_pred"])

    v = view.copy()

    # drift: if safety is low, increase expected victimization slightly
    safety_mean = float(v["safety"].mean()) if len(v) else 0.0
    drift = 0.10 if safety_mean < 1.0 else (0.03 if safety_mean < 2.0 else 0.0)

    noise = np.random.normal(loc=0.0, scale=0.5, size=len(v))
    v["victimization_pred"] = np.clip(v["victimization"].values * (1.0 + drift) + noise, 0, None)

    # keep same threshold logic using current P80 as baseline
    p80 = float(view["victimization"].quantile(0.80)) if len(view) else 0.0
    v["risk_level_pred"] = np.where(v["victimization_pred"] >= p80, "ALTO", "MEDIO/BAJO")

    return v[["survey_response_id", "victimization_pred", "risk_level_pred"]]

# -------------------------------------------------
# Load data ONCE (shared across pages)
# -------------------------------------------------
responses, answers, aso, qopts, questions, schools = load_tables()

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.title("Dashboard Bullying")
    st.warning("Not enough data yet.")
    with st.expander("Debug counts"):
        st.write({
            "survey_responses": len(responses),
            "question_answers": len(answers),
            "answer_selected_options": len(aso),
            "question_options": len(qopts),
            "questions": len(questions),
            "schools": len(schools),
        })
    st.stop()

student = compute_student_metrics(df_long)

# -------------------------------------------------
# Pages
# -------------------------------------------------
def page_menu():
    st.title("TECH4ZERO — Dashboard")
    st.write("Elige qué quieres ver:")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Patrones descriptivos", use_container_width=True, type="primary"):
            go("descriptivo")
    with c2:
        if st.button("Comportamiento futuro y patrones", use_container_width=True):
            go("futuro")

    with st.expander("Debug counts"):
        st.write({
            "survey_responses": len(responses),
            "question_answers": len(answers),
            "answer_selected_options": len(aso),
            "question_options": len(qopts),
            "questions": len(questions),
            "schools": len(schools),
        })

def page_descriptivo():
    st.title("Patrones descriptivos")

    view = render_school_selector(student)
    render_top_metrics(view)

    st.divider()

    st.subheader("Risk distribution")
    st.bar_chart(view["risk_level"].value_counts())

    st.subheader("Who knows (intention to tell)")
    st.bar_chart(pd.DataFrame({
        "Friends": [view["knows_friends"].mean()],
        "Adults": [view["knows_adults"].mean()],
        "Parents": [view["knows_parents"].mean()],
    }).T)

    st.divider()
    st.subheader("Explicación en lenguaje humano (IA)")

    if st.button("Generar explicación IA", type="primary"):
        with st.spinner("Generando reporte…"):
            summary = summary_descriptivo(view)
            report = generate_llm_report(summary, mode="descriptivo")
            st.markdown(report)

    st.divider()
    if st.button("Volver menú principal"):
        go("menu")

def page_futuro():
    st.title("Comportamiento futuro y patrones")

    view = render_school_selector(student)
    render_top_metrics(view)

    st.divider()

    forecast = make_future_forecast(view)

    st.subheader("Predicción (baseline): distribución de riesgo próximo período")
    st.bar_chart(forecast["risk_level_pred"].value_counts())

    st.subheader("Predicción (baseline): victimización proyectada")
    st.line_chart(forecast["victimization_pred"].reset_index(drop=True))

    st.divider()
    st.subheader("Explicación futura en lenguaje humano (IA)")

    if st.button("Generar explicación IA (futuro)", type="primary"):
        with st.spinner("Generando reporte…"):
            summary = summary_futuro(view, forecast)
            report = generate_llm_report(summary, mode="futuro")
            st.markdown(report)

    st.divider()
    if st.button("Volver menú principal"):
        go("menu")

# -------------------------------------------------
# Router
# -------------------------------------------------
if st.session_state.page == "menu":
    page_menu()
elif st.session_state.page == "descriptivo":
    page_descriptivo()
elif st.session_state.page == "futuro":
    page_futuro()
else:
    st.session_state.page = "menu"
    st.rerun()


