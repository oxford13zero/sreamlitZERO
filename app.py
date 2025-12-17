# app.py — Bullying Dashboard (Scientific Indicators + Demographics + Per-question Charts + LLM Report)

import os
import re
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
# Constants: Demographic Question IDs (your UUIDs)
# -------------------------------------------------
QID_EDAD = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO = "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"

DEMOGRAPHIC_QIDS = {QID_EDAD, QID_GRADO, QID_GENERO}

# -------------------------------------------------
# Scoring scale (frequency-based)
# -------------------------------------------------
SCALE = {
    "Nunca": 0,
    "No": 0,
    "Sólo una vez": 1,
    "Solo una vez": 1,
    "A veces": 1,
    "Varias veces al mes": 2,
    "Sí": 2,
    "Si": 2,
    "Casi cada semana": 3,
    "Casi cada día": 4,
    "Casi cada dia": 4,
}

# Thresholds (scientific interpretation)
HIGH_FREQ_THRESHOLD = 2  # mensual o más
VERY_HIGH_FREQ = 3       # semanal o diaria

# -------------------------------------------------
# Prompt (Spanish, educational & ethical)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un/a especialista en convivencia escolar y prevención del bullying.
Tu objetivo es ayudar a equipos directivos y orientadores escolares.

INSTRUCCIONES IMPORTANTES:
- Usa un lenguaje educativo, ético y no alarmista.
- No diagnostiques ni identifiques a estudiantes.
- No inventes datos: usa SOLO el resumen entregado.
- Si un indicador es bajo, NO concluyas “no hay problemas”; explica límites y recomendaciones preventivas.
- Entrega acciones concretas a nivel de escuela (no a nivel individual).

TAREA:
Escribe un informe claro y no técnico basado ÚNICAMENTE en los datos agregados.

Incluye:
1) Interpretación breve de los indicadores principales.
2) Patrones o señales a vigilar (p. ej., persistencia, cyberbullying, posibles barreras de reporte).
3) Implicancias prácticas para la convivencia escolar.
4) 5 recomendaciones concretas, realistas y priorizadas (1 = alta prioridad).
5) “Próximos pasos” sugeridos para mejorar la calidad del diagnóstico (p. ej., aumentar muestra, repetir medición).

DATOS AGREGADOS:
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
# Data preparation (long table)
# -------------------------------------------------
def build_long_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    # Ensure expected columns exist
    for df_, col in [
        (answers, "id"),
        (responses, "id"),
        (questions, "id"),
        (qopts, "id"),
        (aso, "question_answer_id"),
    ]:
        if col not in df_.columns:
            return pd.DataFrame()

    df = (
        answers
        .merge(
            responses,
            left_on="survey_response_id",
            right_on="id",
            suffixes=("_ans", "_resp"),
            how="left",
        )
        .merge(
            questions,
            left_on="question_id",
            right_on="id",
            suffixes=("", "_q"),
            how="left",
        )
        .merge(
            aso,
            left_on="id_ans",
            right_on="question_answer_id",
            how="inner",
        )
        .merge(
            qopts,
            left_on="option_id",
            right_on="id",
            how="inner",
            suffixes=("", "_opt"),
        )
    )

    # Merge schools safely
    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        df = df.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    # Clean strings
    df["question_text"] = df.get("question_text", "").fillna("").astype(str)
    df["option_text"] = df.get("option_text", "").fillna("").astype(str)
    df["option_code"] = df.get("option_code", "").fillna("").astype(str)

    # Score for frequency questions
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

# -------------------------------------------------
# Demographics extraction (by question_id, NOT by text)
# -------------------------------------------------
def _safe_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def extract_demographics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Returns:
      columns: survey_response_id, edad, grado, genero
    Values come from option_code/option_text for the demographic question_ids.
    """
    if df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"])

    base_cols = ["survey_response_id", "question_id", "option_text", "option_code"]
    if any(c not in df_long.columns for c in base_cols):
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"])

    # Age
    age = (
        df_long[df_long["question_id"] == QID_EDAD][["survey_response_id", "option_text"]]
        .drop_duplicates("survey_response_id")
        .copy()
    )
    age["edad"] = age["option_text"].apply(_safe_int)
    age = age[["survey_response_id", "edad"]]

    # Grade
    grade = (
        df_long[df_long["question_id"] == QID_GRADO][["survey_response_id", "option_text"]]
        .drop_duplicates("survey_response_id")
        .copy()
    )
    grade["grado"] = grade["option_text"].apply(_safe_int)
    grade = grade[["survey_response_id", "grado"]]

    # Gender (VECTORIZE: no apply to avoid pandas edge cases)
    gender = (
        df_long[df_long["question_id"] == QID_GENERO][["survey_response_id", "option_text", "option_code"]]
        .drop_duplicates("survey_response_id")
        .copy()
    )

    code = gender["option_code"].astype(str).str.strip().str.upper()
    genero = code.where(code.isin(["M", "F", "O", "N"]), np.nan)

    txt = gender["option_text"].astype(str).str.strip().str.lower()
    fallback = np.select(
        [
            txt.str.startswith("masc"),
            txt.str.startswith("fem"),
            txt.str.contains(r"\botro\b", regex=True),
            txt.str.contains("no") & txt.str.contains("respond"),
        ],
        ["M", "F", "O", "N"],
        default=np.nan,
    )

    genero = genero.fillna(pd.Series(fallback, index=gender.index))
    gender["genero"] = genero

    gender = gender[["survey_response_id", "genero"]]

    demo = age.merge(grade, on="survey_response_id", how="outer").merge(gender, on="survey_response_id", how="outer")
    return demo

# -------------------------------------------------
# Student-level scientific indicators (school-wide)
# -------------------------------------------------
def compute_student_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    d = df_long.copy()

    # Identify question types by semantics (kept from your version)
    is_victim = d["question_text"].str.contains(r"agredido|molestado|ignorado", case=False, na=False)
    is_cyber = d["question_text"].str.contains(r"internet|mensajería|mensajes|videos|fotos", case=False, na=False)
    trust_adults = d["question_text"].str.contains(r"adultos|profesores|docentes", case=False, na=False)

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
        default="BAJO",
    )

    return student

# -------------------------------------------------
# Per-question charts: Grade (X) x Count (Y) x Gender (series)
# Count = number of students with score >= mensual for that question
# -------------------------------------------------
def render_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, school_name_filter: str | None = None):
    st.subheader("Resultados por pregunta (Grado × Género)")

    if df_long.empty:
        st.info("No hay datos para graficar.")
        return

    if demo_df.empty:
        st.warning("No se encontraron respuestas demográficas (edad/grado/género).")
        return

    dfl = df_long.copy()
    if school_name_filter and school_name_filter != "(Todas)":
        dfl = dfl[dfl["school_name"] == school_name_filter]

    questions_df = (
        dfl[~dfl["question_id"].isin(DEMOGRAPHIC_QIDS)][["question_id", "question_text"]]
        .drop_duplicates()
        .copy()
    )

    if questions_df.empty:
        st.info("No hay preguntas (no-demográficas) para graficar.")
        return

    dfl = dfl.merge(demo_df, on="survey_response_id", how="left")

    # Need grade + gender
    dfl = dfl.dropna(subset=["grado", "genero"])
    if dfl.empty:
        st.warning("No se encontraron respuestas demográficas (edad/grado/género).")
        return

    question_list = questions_df.sort_values("question_text")["question_text"].tolist()
    q_text_sel = st.selectbox("Selecciona una pregunta para ver el gráfico", question_list, index=0)
    qid_sel = questions_df.loc[questions_df["question_text"] == q_text_sel, "question_id"].iloc[0]

    per_student = (
        dfl[dfl["question_id"] == qid_sel]
        .groupby(["survey_response_id", "grado", "genero"], as_index=False)
        .agg(score_max=("score", "max"))
    )

    if per_student.empty:
        st.info("No hay respuestas para esta pregunta.")
        return

    per_student["flag_mensual_o_mas"] = per_student["score_max"] >= HIGH_FREQ_THRESHOLD

    agg = (
        per_student[per_student["flag_mensual_o_mas"]]
        .groupby(["grado", "genero"])
        .size()
        .reset_index(name="conteo")
    )

    gender_order = ["M", "F", "O", "N"]
    grade_order = sorted(per_student["grado"].dropna().unique().tolist())

    pivot = (
        agg.pivot_table(index="grado", columns="genero", values="conteo", aggfunc="sum", fill_value=0)
        .reindex(columns=gender_order, fill_value=0)
        .reindex(index=grade_order, fill_value=0)
    )

    pivot = pivot.rename(columns={
        "M": "Hombres (M)",
        "F": "Mujeres (F)",
        "O": "Otro (O)",
        "N": "No responde (N)",
    })

    st.markdown(f"**Pregunta:** {q_text_sel}")
    st.caption(f"Conteo de estudiantes con respuesta de frecuencia **≥ mensual** (score ≥ {HIGH_FREQ_THRESHOLD}).")
    st.bar_chart(pivot)

    base = (
        per_student.groupby(["grado", "genero"])
        .size()
        .reset_index(name="respondieron")
        .pivot_table(index="grado", columns="genero", values="respondieron", aggfunc="sum", fill_value=0)
        .reindex(columns=gender_order, fill_value=0)
        .reindex(index=grade_order, fill_value=0)
    )

    with st.expander("Ver cuántos estudiantes respondieron (base) por Grado × Género"):
        st.dataframe(base)

# -------------------------------------------------
# Aggregated summary for LLM
# -------------------------------------------------
def build_school_summary(view: pd.DataFrame) -> dict:
    n = len(view)
    if n == 0:
        return {}
    return {
        "total_estudiantes": int(n),
        "prevalencia_victimizacion_pct": round(100 * view["victim_freq"].mean(), 1),
        "prevalencia_cyberbullying_pct": round(100 * view["cyber_freq"].mean(), 1),
        "alta_persistencia_pct": round(100 * view["high_persistence"].mean(), 1),
        "silencio_institucional_pct": round(100 * view["silence_flag"].mean(), 1),
        "distribucion_riesgo": view["risk_group"].value_counts().to_dict(),
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
    model = st.secrets.get("LLM_MODEL", "llama3-8b-8192")

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

demo_df = extract_demographics(df_long)
student = compute_student_metrics(df_long)

school_list = sorted(student["school_name"].dropna().unique().tolist())
school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view = student if school == "(Todas)" else student[student["school_name"] == school]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", len(view))
c2.metric("Victimización ≥ mensual", f"{view['victim_freq'].mean()*100:.1f}%")
c3.metric("Alta persistencia", f"{view['high_persistence'].mean()*100:.1f}%")
c4.metric("Silencio institucional", f"{view['silence_flag'].mean()*100:.1f}%")

st.divider()

st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view["risk_group"].value_counts())

st.divider()

render_question_charts(df_long, demo_df, school_name_filter=school)

st.divider()

st.subheader("Informe interpretativo (IA)")

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)
