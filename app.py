# app.py — Bullying Dashboard (Scientific Indicators + LLM Report + Segmentation)

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

# Thresholds (scientific interpretation)
HIGH_FREQ_THRESHOLD = 2     # mensual o más
VERY_HIGH_FREQ = 3          # semanal o diaria

# -------------------------------------------------
# PROMPT LLM (ES)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar, prevención del bullying y análisis ético de datos educativos.

Interpreta SOLO los datos agregados entregados. No inventes información.
No afirmes “todo está bien” aunque los indicadores sean bajos: valores bajos pueden ocultar riesgo latente.

Escribe en español, tono profesional, educativo, cuidadoso y respetuoso.

Incluye:
1) Lectura general de convivencia (qué significa en términos escolares).
2) Señales de alerta temprana (aunque sean incipientes).
3) Interpretación del “silencio institucional” (victimización + no recurrir a adultos).
4) Diferencias relevantes por género / edad / curso si existen.
5) EXACTAMENTE 3 recomendaciones prácticas y accionables.

DATOS AGREGADOS:
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
        .merge(questions, left_on="question_id", right_on="id",
               suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id",
               how="inner", suffixes=("", "_opt"))
    )

    # Merge schools safely
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
    df["option_code"] = df.get("option_code", "").fillna("").astype(str)

    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)
    return df

# -------------------------------------------------
# Demographics extraction (edad/curso/género)
# -------------------------------------------------
def extract_demographics(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "curso", "genero"])

    d = df_long.copy()

    # Identify demographic questions by text (robust enough for your survey wording)
    is_age_q = d["question_text"].str.contains(r"\bedad\b", case=False, na=False)
    is_grade_q = d["question_text"].str.contains(r"grado|curso", case=False, na=False)
    is_gender_q = d["question_text"].str.contains(r"género|genero", case=False, na=False)

    age = (
        d[is_age_q]
        .groupby("survey_response_id")["option_code"]
        .first()
        .rename("edad")
        .reset_index()
    )
    grade = (
        d[is_grade_q]
        .groupby("survey_response_id")["option_code"]
        .first()
        .rename("curso")
        .reset_index()
    )
    gender = (
        d[is_gender_q]
        .groupby("survey_response_id")["option_code"]
        .first()
        .rename("genero")
        .reset_index()
    )

    demo = age.merge(grade, on="survey_response_id", how="outer").merge(gender, on="survey_response_id", how="outer")

    # Normalize types
    demo["edad"] = pd.to_numeric(demo["edad"], errors="coerce")
    demo["curso"] = pd.to_numeric(demo["curso"], errors="coerce")
    demo["genero"] = demo["genero"].fillna("N/A").astype(str)

    # Map codes to readable labels (your stored codes)
    gender_map = {"M": "Masculino", "F": "Femenino", "O": "Otro", "N": "No responde", "N/A": "N/A"}
    demo["genero_label"] = demo["genero"].map(gender_map).fillna(demo["genero"])

    return demo

# -------------------------------------------------
# Student-level scientific indicators
# -------------------------------------------------
def compute_student_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    d = df_long.copy()

    # Identify question types by semantics
    is_victim = d["question_text"].str.contains(
        r"agredido|molestado|ignorado", case=False, na=False
    )

    is_cyber = d["question_text"].str.contains(
        r"internet|mensajería|mensajes|videos|fotos", case=False, na=False
    )

    trust_adults = d["question_text"].str.contains(
        r"adultos|profesores|docentes", case=False, na=False
    )

    # Scores per dimension
    d["victim_score"] = np.where(is_victim, d["score"], 0)
    d["cyber_score"] = np.where(is_cyber, d["score"], 0)
    d["trust_adult_score"] = np.where(trust_adults, d["score"], 0)

    student = d.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        victim_max=("victim_score", "max"),
        cyber_max=("cyber_score", "max"),
        trust_adult_max=("trust_adult_score", "max"),
    ).reset_index()

    # Scientific indicators
    student["victim_freq"] = student["victim_max"] >= HIGH_FREQ_THRESHOLD
    student["cyber_freq"] = student["cyber_max"] >= HIGH_FREQ_THRESHOLD
    student["high_persistence"] = student["victim_max"] >= VERY_HIGH_FREQ

    # Silence: victimization + no adult trust
    student["silence_flag"] = student["victim_freq"] & (student["trust_adult_max"] == 0)

    # Absolute risk classification
    student["risk_group"] = np.select(
        [student["high_persistence"], student["victim_freq"]],
        ["ALTO", "MEDIO"],
        default="BAJO"
    )

    # Attach demographics
    demo = extract_demographics(df_long)
    if not demo.empty:
        student = student.merge(demo, on="survey_response_id", how="left")
    else:
        student["edad"] = np.nan
        student["curso"] = np.nan
        student["genero_label"] = "N/A"

    return student

# -------------------------------------------------
# Aggregated summary for LLM / PDF
# -------------------------------------------------
def _group_rates(view: pd.DataFrame, group_col: str) -> dict:
    if view.empty or group_col not in view.columns:
        return {}
    tmp = view.copy()
    tmp = tmp.dropna(subset=[group_col])
    if tmp.empty:
        return {}

    out = {}
    for g, sub in tmp.groupby(group_col):
        out[str(g)] = {
            "n": int(len(sub)),
            "victimizacion_pct": round(100 * sub["victim_freq"].mean(), 1),
            "cyber_pct": round(100 * sub["cyber_freq"].mean(), 1),
            "alta_persistencia_pct": round(100 * sub["high_persistence"].mean(), 1),
            "silencio_pct": round(100 * sub["silence_flag"].mean(), 1),
        }
    return out

def build_school_summary(view: pd.DataFrame) -> dict:
    n = len(view)
    if n == 0:
        return {}

    summary = {
        "total_estudiantes": int(n),
        "prevalencia_victimizacion_pct": round(100 * view["victim_freq"].mean(), 1),
        "prevalencia_cyberbullying_pct": round(100 * view["cyber_freq"].mean(), 1),
        "alta_persistencia_pct": round(100 * view["high_persistence"].mean(), 1),
        "silencio_institucional_pct": round(100 * view["silence_flag"].mean(), 1),
        "distribucion_riesgo": view["risk_group"].value_counts().to_dict(),
        "segmentos": {
            "por_genero": _group_rates(view, "genero_label"),
            "por_edad": _group_rates(view.dropna(subset=["edad"]).assign(edad=view["edad"].astype(int)), "edad"),
            "por_curso": _group_rates(view.dropna(subset=["curso"]).assign(curso=view["curso"].astype(int)), "curso"),
        }
    }
    return summary

# -------------------------------------------------
# LLM (Groq)
# -------------------------------------------------
def generate_llm_report(summary: dict) -> str:
    provider = st.secrets.get("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return _groq_report(summary)
    return "Proveedor LLM no configurado."

def _groq_report(summary: dict) -> str:
    api_key = st.secrets.get("GROQ_API_KEY")
    model = st.secrets.get("LLM_MODEL", "llama3-8b-8192")

    if not api_key:
        return "Falta la API Key de Groq."

    prompt = LLM_PROMPT_ES.format(summary=summary)

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Eres un especialista en convivencia escolar y prevención del bullying."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text}")

    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Dashboard de Convivencia Escolar — Análisis Científico")

responses, answers, aso, qopts, questions, schools = load_tables()

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.warning("Aún no hay datos suficientes para análisis.")
    st.stop()

student = compute_student_metrics(df_long)

# Sidebar filters
school_list = sorted(student["school_name"].dropna().unique().tolist())
school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)
view = student if school == "(Todas)" else student[student["school_name"] == school]

st.sidebar.divider()
st.sidebar.subheader("Segmentación")
dim = st.sidebar.selectbox("Ver por", ["(Sin segmentación)", "Género", "Edad", "Curso"])

# Optional sub-filters (helpful for exploration)
if "genero_label" in view.columns:
    gender_filter = st.sidebar.multiselect("Filtrar género", sorted(view["genero_label"].dropna().unique().tolist()))
    if gender_filter:
        view = view[view["genero_label"].isin(gender_filter)]

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", len(view))
c2.metric("Victimización ≥ mensual", f"{view['victim_freq'].mean()*100:.1f}%")
c3.metric("Alta persistencia", f"{view['high_persistence'].mean()*100:.1f}%")
c4.metric("Silencio institucional", f"{view['silence_flag'].mean()*100:.1f}%")

st.divider()

# Risk distribution
st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view["risk_group"].value_counts())

# Segmented charts
def _seg_chart(df: pd.DataFrame, group_col: str, label: str):
    if df.empty or group_col not in df.columns:
        st.info(f"No hay datos suficientes para segmentar por {label}.")
        return

    tmp = df.dropna(subset=[group_col]).copy()
    if tmp.empty:
        st.info(f"No hay datos suficientes para segmentar por {label}.")
        return

    agg = (
        tmp.groupby(group_col)
        .agg(
            n=("survey_response_id", "count"),
            victimizacion_pct=("victim_freq", "mean"),
            cyber_pct=("cyber_freq", "mean"),
            alta_persistencia_pct=("high_persistence", "mean"),
            silencio_pct=("silence_flag", "mean"),
        )
        .reset_index()
    )
    for col in ["victimizacion_pct", "cyber_pct", "alta_persistencia_pct", "silencio_pct"]:
        agg[col] = (agg[col] * 100).round(1)

    st.subheader(f"Indicadores por {label}")
    st.dataframe(agg, use_container_width=True)

    chart = agg.set_index(group_col)[["victimizacion_pct", "silencio_pct", "alta_persistencia_pct"]]
    st.bar_chart(chart)

if dim == "Género":
    _seg_chart(view, "genero_label", "género")
elif dim == "Edad":
    view2 = view.copy()
    view2["edad"] = pd.to_numeric(view2.get("edad", np.nan), errors="coerce")
    view2["edad"] = view2["edad"].dropna().astype(int)
    _seg_chart(view2.dropna(subset=["edad"]), "edad", "edad")
elif dim == "Curso":
    view2 = view.copy()
    view2["curso"] = pd.to_numeric(view2.get("curso", np.nan), errors="coerce")
    view2["curso"] = view2["curso"].dropna().astype(int)
    _seg_chart(view2.dropna(subset=["curso"]), "curso", "curso")

st.divider()

# LLM report
st.subheader("Informe interpretativo (IA)")

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)
