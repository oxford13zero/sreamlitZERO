# app.py — Bullying Dashboard (Scientific Indicators + Demographics from selected options + 40 charts)

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
# Constants: demographic question UUIDs
# -------------------------------------------------
QID_EDAD = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO = "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"
QID_TIEMPO = "7c5d8e66-1d8d-4f0c-8a4f-8a6b6b5c4c11"

DEMO_QIDS = {QID_EDAD, QID_GRADO, QID_GENERO, QID_TIEMPO}

GENDER_ORDER = ["M", "F", "O", "N"]
GENDER_LABELS = {"M": "Hombres", "F": "Mujeres", "O": "Otro", "N": "No responde"}

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
HIGH_FREQ_THRESHOLD = 2
VERY_HIGH_FREQ = 3

# -------------------------------------------------
# LLM prompt (Spanish, ethical/educational)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un/a especialista en convivencia escolar, prevención del acoso y análisis de encuestas.
Tu objetivo es ayudar a un equipo directivo escolar a interpretar resultados SIN etiquetar ni identificar estudiantes.
Usa lenguaje respetuoso, educativo, y orientado a acciones de protección.

INSTRUCCIONES:
- No inventes datos: usa SOLO el resumen agregado entregado.
- No digas que “todo está perfecto” si hay señales de riesgo; expresa incertidumbre con cuidado.
- Evita culpabilizar a estudiantes. Enfócate en condiciones del entorno escolar y apoyo.
- Incluye recomendaciones prácticas y medidas preventivas.

ENTREGA EN ESPAÑOL con este formato:

1) Lectura general (2–4 líneas)
2) Señales relevantes (bullets)
3) Interpretación cuidadosa (2–5 líneas)
4) Recomendaciones concretas (3–6 bullets)
5) Siguientes pasos de medición (2–4 bullets)

RESUMEN AGREGADO:
{summary}
""".strip()

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
def build_long_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    """
    Long format with one row per selected option.
    Brings: survey_response_id, school_id, question_id, question_text, option_code, option_text, created_at
    """
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    df = (
        answers
        .merge(responses, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id", suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt"))
    )

    # Schools join (safe)
    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        df = df.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    df["question_text"] = df["question_text"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["option_code"] = df["option_code"].astype(str)

    # Use answer created_at if present
    if "created_at_ans" in df.columns:
        df["created_at_ans"] = pd.to_datetime(df["created_at_ans"], errors="coerce", utc=True)
    else:
        df["created_at_ans"] = pd.NaT

    # scoring
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)
    return df

# -------------------------------------------------
# Demographics extraction (FROM SELECTED OPTIONS!)
# -------------------------------------------------
def extract_demographics_from_options(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per survey_response_id with columns: grado(int), genero(str), edad(int), tiempo(int), school_name
    Reads ONLY from df_long option_code/option_text for the demographic question_ids.
    """
    if df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    d = df_long.copy()

    # filter only demographic questions
    d = d[d["question_id"].astype(str).isin(DEMO_QIDS)].copy()
    if d.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    # keep latest per (response, question) if duplicates
    d["qid"] = d["question_id"].astype(str)
    d = d.sort_values(["survey_response_id", "qid", "created_at_ans"], ascending=[True, True, True])
    d_last = d.groupby(["survey_response_id", "qid"], as_index=False).tail(1)

    # helper: code -> int safely
    def to_int_safe(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            if s == "":
                return np.nan
            return int(float(s))
        except Exception:
            return np.nan

    # pivot to columns by question_id
    pivot = d_last.pivot_table(
        index=["survey_response_id", "school_name"],
        columns="qid",
        values="option_code",
        aggfunc="first"
    ).reset_index()

    # map columns
    pivot["edad"] = pivot.get(QID_EDAD, np.nan).apply(to_int_safe)
    pivot["grado"] = pivot.get(QID_GRADO, np.nan).apply(to_int_safe)

    # genero already code M/F/O/N
    g = pivot.get(QID_GENERO, np.nan)
    if isinstance(g, pd.Series):
        pivot["genero"] = g.astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    else:
        pivot["genero"] = np.nan

    pivot["tiempo"] = pivot.get(QID_TIEMPO, np.nan).apply(to_int_safe)

    # normalize genero
    pivot.loc[~pivot["genero"].isin(GENDER_ORDER), "genero"] = np.nan

    out = pivot[["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"]].copy()
    return out

# -------------------------------------------------
# Student-level scientific indicators (uses df_long scores on bullying items)
# -------------------------------------------------
def compute_student_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    d = df_long.copy()

    # Identify victimization items by text
    is_victim = d["question_text"].str.contains(r"agredid|molestad|ignorado|acoso|bullying", case=False, na=False)
    is_cyber = d["question_text"].str.contains(r"internet|mensajer|mensaje|video|foto|redes", case=False, na=False)
    trust_adults = d["question_text"].str.contains(r"adult|profesor|docente|inspect|direct", case=False, na=False)

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
# Aggregated summary for LLM
# -------------------------------------------------
def build_school_summary(view_student: pd.DataFrame, view_demo: pd.DataFrame) -> dict:
    n = len(view_student)
    if n == 0:
        return {}

    # demographic distribution if available
    grade_dist = {}
    gender_dist = {}
    if not view_demo.empty:
        if "grado" in view_demo.columns:
            grade_dist = view_demo["grado"].dropna().astype(int).value_counts().sort_index().to_dict()
        if "genero" in view_demo.columns:
            gender_dist = view_demo["genero"].dropna().value_counts().to_dict()

    return {
        "total_estudiantes": int(n),
        "prevalencia_victimizacion_pct": round(100 * view_student["victim_freq"].mean(), 1),
        "prevalencia_cyberbullying_pct": round(100 * view_student["cyber_freq"].mean(), 1),
        "alta_persistencia_pct": round(100 * view_student["high_persistence"].mean(), 1),
        "silencio_institucional_pct": round(100 * view_student["silence_flag"].mean(), 1),
        "distribucion_riesgo": view_student["risk_group"].value_counts().to_dict(),
        "distribucion_grado": grade_dist,
        "distribucion_genero": gender_dist,
    }

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
    model = st.secrets.get("LLM_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return "Falta la API Key de Groq."

    prompt = LLM_PROMPT_ES.format(summary=summary)

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Eres un especialista en convivencia escolar y prevención del bullying."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=45,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text}")

    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------
# Charts: 40 graphs (one per question) — Grade x Gender counts
# -------------------------------------------------
def render_40_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, title_prefix: str = ""):
    """
    For each non-demographic question, show chart:
      X = grade
      Y = counts by gender (M/F/O/N)
    Count = number of unique students (responses) who answered that question.
    """
    st.subheader("Gráficos por cada pregunta (40) — Grado × Género")

    if demo_df.empty:
        st.warning("No hay demografía (edad/grado/género). Sin eso no se pueden construir los gráficos.")
        return

    # Only keep responses with grade and gender
    demo_ok = demo_df.dropna(subset=["grado", "genero"]).copy()
    if demo_ok.empty:
        st.warning("Demografía incompleta: faltan grado o género en todas las respuestas.")
        return

    demo_ok["grado"] = demo_ok["grado"].astype(int)
    demo_ok["genero"] = demo_ok["genero"].astype(str)

    # Pick the first 40 non-demographic questions from df_long (stable order)
    q_meta = (
        df_long[~df_long["question_id"].astype(str).isin(DEMO_QIDS)]
        .loc[:, ["question_id", "question_text"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if q_meta.empty:
        st.warning("No se encontraron preguntas no-demográficas en los datos.")
        return

    q_meta_40 = q_meta.head(40)

    # For each question, find which responses answered it (as selected options)
    # df_long has one row per selected option; we want unique (survey_response_id, question_id)
    answered = (
        df_long[~df_long["question_id"].astype(str).isin(DEMO_QIDS)]
        .loc[:, ["survey_response_id", "question_id"]]
        .drop_duplicates()
    )

    # Join answered with demographics to get grade+gender for each answered item
    answered_demo = answered.merge(
        demo_ok[["survey_response_id", "grado", "genero"]],
        on="survey_response_id",
        how="inner"
    )

    # Normalize gender buckets
    answered_demo["genero"] = answered_demo["genero"].where(answered_demo["genero"].isin(GENDER_ORDER), np.nan)
    answered_demo = answered_demo.dropna(subset=["genero"])

    for _, row in q_meta_40.iterrows():
        qid = str(row["question_id"])
        qtext = (row["question_text"] or "").strip()
        if not qtext:
            qtext = f"Pregunta {qid}"

        # Filter answers for this question
        sub = answered_demo[answered_demo["question_id"].astype(str) == qid].copy()
        if sub.empty:
            st.caption(f"**{qtext}**")
            st.info("Sin respuestas para esta pregunta.")
            st.divider()
            continue

        # Build pivot: index=grade, columns=gender, values=count of responses
        pivot = (
            sub.groupby(["grado", "genero"])["survey_response_id"]
            .nunique()
            .reset_index(name="n")
            .pivot(index="grado", columns="genero", values="n")
            .fillna(0)
            .astype(int)
        )

        # Ensure all gender columns exist
        for g in GENDER_ORDER:
            if g not in pivot.columns:
                pivot[g] = 0
        pivot = pivot[GENDER_ORDER]

        # nicer column names for display
        pivot_disp = pivot.rename(columns=GENDER_LABELS).sort_index()

        st.caption(f"**{qtext}**")
        st.bar_chart(pivot_disp)
        st.divider()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Dashboard de Convivencia Escolar — Análisis Científico")

responses, answers, aso, qopts, questions, schools = load_tables()

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.warning("Aún no hay datos suficientes para análisis (faltan tablas o respuestas).")
    st.stop()

student = compute_student_metrics(df_long)

# Demographics now from selected options
demo_df = extract_demographics_from_options(df_long)

# Sidebar: school filter
school_list = sorted(student["school_name"].dropna().unique().tolist())
school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view_student = student if school == "(Todas)" else student[student["school_name"] == school]
view_demo = demo_df if school == "(Todas)" else demo_df[demo_df["school_name"] == school]

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", int(len(view_student)))
c2.metric("Victimización ≥ mensual", f"{view_student['victim_freq'].mean()*100:.1f}%")
c3.metric("Alta persistencia", f"{view_student['high_persistence'].mean()*100:.1f}%")
c4.metric("Silencio institucional", f"{view_student['silence_flag'].mean()*100:.1f}%")

st.divider()

st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view_student["risk_group"].value_counts())

st.divider()

# 40 charts: Grade x Gender per question
render_40_question_charts(df_long if school == "(Todas)" else df_long[df_long["school_name"] == school], view_demo)

st.divider()

st.subheader("Informe interpretativo (IA)")
if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view_student, view_demo)
        report = generate_llm_report(summary)
        st.markdown(report)

with st.expander("Debug (recomendado ahora)"):
    st.write("Rows df_long:", int(len(df_long)))
    st.write("Rows demo_df:", int(len(demo_df)))
    st.write("Ejemplo demo_df (head):")
    st.dataframe(demo_df.head(20))
    st.write("Ejemplos de preguntas (question_text):")
    st.dataframe(
        df_long.loc[:, ["question_id", "question_text"]].drop_duplicates().head(10),
        use_container_width=True
    )
