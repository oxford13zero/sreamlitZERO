# app.py — Bullying Dashboard (Scientific Indicators + LLM Report)

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from supabase import create_client

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Bullying Dashboard",
    layout="wide"
)

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
# Scoring scale (frequency-based)
# -------------------------------------------------
SCALE = {
    "Nunca": 0,
    "No": 0,
    "Sólo una vez": 1,
    "A veces": 1,
    "Varias veces al mes": 2,
    "Sí": 2,
    "Casi cada semana": 3,
    "Casi cada día": 4,
}

HIGH_FREQ_THRESHOLD = 2     # mensual o más
VERY_HIGH_FREQ = 3          # semanal o diaria

# -------------------------------------------------
# PROMPT LLM (ESPAÑOL, ÉTICO, EDUCATIVO)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar, prevención del bullying y análisis de datos educativos.

Tu tarea es interpretar los siguientes indicadores agregados de una escuela.
NO minimices riesgos ni afirmes que "todo está bien".
Incluso valores bajos pueden representar riesgo latente.

Habla en un lenguaje:
- Claro
- Respetuoso
- Ético
- Comprensible para directivos y docentes (no técnico)

Debes entregar:

1. Una lectura general del clima de convivencia escolar.
2. Señales de alerta temprana, aunque sean incipientes.
3. Interpretación del silencio institucional (cuando estudiantes no recurren a adultos).
4. Diferenciación entre riesgo bajo, medio y alto, explicando qué significa cada uno.
5. EXACTAMENTE 3 recomendaciones prácticas y accionables para la escuela.

Datos agregados de la escuela:
{summary}
"""

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
        .merge(responses, left_on="survey_response_id", right_on="id",
               suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id")
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner")
    )

    schools_small = schools[["id", "name"]].rename(
        columns={"id": "school_pk", "name": "school_name"}
    )
    df = df.merge(
        schools_small,
        left_on="school_id",
        right_on="school_pk",
        how="left",
    )

    df["question_text"] = df["question_text"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

# -------------------------------------------------
# Student-level scientific indicators
# -------------------------------------------------
def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    is_victim = d["question_text"].str.contains(
        r"agredido|molestado|ignorado", case=False, na=False
    )
    is_cyber = d["question_text"].str.contains(
        r"internet|mensajería|mensajes|videos|fotos", case=False, na=False
    )
    trust_adults = d["question_text"].str.contains(
        r"adultos|profesores|docentes", case=False, na=False
    )

    d["victim_score"] = np.where(is_victim, d["score"], 0)
    d["cyber_score"] = np.where(is_cyber, d["score"], 0)
    d["trust_adult_score"] = np.where(trust_adults, d["score"], 0)

    student = d.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        victim_max=("victim_score", "max"),
        cyber_max=("cyber_score", "max"),
        trust_adult_max=("trust_adult_score", "max"),
    ).reset_index()

    student["victim_freq"] = student["victim_max"] >= HIGH_FREQ_THRESHOLD
    student["cyber_freq"] = student["cyber_max"] >= HIGH_FREQ_THRESHOLD
    student["high_persistence"] = student["victim_max"] >= VERY_HIGH_FREQ
    student["silence_flag"] = student["victim_freq"] & (student["trust_adult_max"] == 0)

    student["risk_group"] = np.select(
        [student["high_persistence"], student["victim_freq"]],
        ["ALTO", "MEDIO"],
        default="BAJO"
    )

    return student

# -------------------------------------------------
# Aggregated summary
# -------------------------------------------------
def build_school_summary(view: pd.DataFrame) -> dict:
    return {
        "total_estudiantes": int(len(view)),
        "prevalencia_victimizacion_pct": round(100 * view["victim_freq"].mean(), 1),
        "prevalencia_cyberbullying_pct": round(100 * view["cyber_freq"].mean(), 1),
        "alta_persistencia_pct": round(100 * view["high_persistence"].mean(), 1),
        "silencio_institucional_pct": round(100 * view["silence_flag"].mean(), 1),
        "distribucion_riesgo": view["risk_group"].value_counts().to_dict(),
    }

# -------------------------------------------------
# LLM
# -------------------------------------------------
def generate_llm_report(summary: dict) -> str:
    api_key = st.secrets.get("GROQ_API_KEY")
    model = st.secrets.get("LLM_MODEL", "llama3-8b-8192")

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Especialista en convivencia escolar."},
                {"role": "user", "content": LLM_PROMPT_ES.format(summary=summary)},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )

    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Dashboard de Convivencia Escolar — Análisis Científico")

responses, answers, aso, qopts, questions, schools = load_tables()
df_long = build_long_df(responses, answers, aso, qopts, questions, schools)

if df_long.empty:
    st.warning("Aún no hay datos suficientes.")
    st.stop()

student = compute_student_metrics(df_long)

school = st.sidebar.selectbox(
    "Escuela",
    ["(Todas)"] + sorted(student["school_name"].unique())
)

view = student if school == "(Todas)" else student[student["school_name"] == school]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", len(view))
c2.metric("Victimización ≥ mensual", f"{view['victim_freq'].mean()*100:.1f}%")
c3.metric("Alta persistencia", f"{view['high_persistence'].mean()*100:.1f}%")
c4.metric("Silencio institucional", f"{view['silence_flag'].mean()*100:.1f}%")

st.divider()
st.bar_chart(view["risk_group"].value_counts())

st.divider()
if st.button("Generar informe interpretativo"):
    report = generate_llm_report(build_school_summary(view))
    st.markdown(report)
