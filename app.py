# app.py — Bullying Dashboard (Scientific Indicators + LLM Report + Charts by Grade/Gender)

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
import altair as alt
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
# DEMO questions (UUIDs you provided)
# -------------------------------------------------
QID_EDAD  = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO= "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"

# -------------------------------------------------
# Prompt (Spanish, ethical, educational)
# -------------------------------------------------
LLM_PROMPT_ES_V1 = """
Eres un/a especialista en convivencia escolar, prevención del bullying y protección de niños, niñas y adolescentes.
Tu respuesta debe ser en español, con un tono educativo, respetuoso, ético, no estigmatizante y orientado a la mejora.
No inventes datos ni diagnósticos. No identifiques personas ni sugieras medidas punitivas. Habla de tendencias agregadas.

Tarea:
Redacta un informe ejecutivo para el equipo directivo y convivencia escolar, basado ÚNICAMENTE en los datos agregados.

Incluye:
1) Lectura general del clima y la convivencia (qué indican los números, sin alarmismo).
2) Señales de alerta (si las hay) y su interpretación prudente.
3) Posibles patrones (por ejemplo, concentración por cursos o diferencias por grupos) SOLO si los datos lo permiten.
4) Recomendaciones prácticas (mínimo 5), priorizadas, con foco preventivo, apoyo socioemocional, reportabilidad segura y cultura de cuidado.
5) Sugerencias de seguimiento (qué medir en la próxima encuesta y por qué).

DATOS AGREGADOS (JSON):
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
def build_long_df(responses, answers, aso, qopts, questions, schools):
    """
    Long format with ONLY selected options (answer_selected_options).
    Each row ~ one (survey_response_id, question_id, option_id).
    """
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
    df["option_code"] = df.get("option_code", "").fillna("")

    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    # Normalize ids as strings
    df["survey_response_id"] = df["survey_response_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)

    return df

# -------------------------------------------------
# Student-level scientific indicators
# -------------------------------------------------
def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

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
    student["silence_flag"] = (
        student["victim_freq"] & (student["trust_adult_max"] == 0)
    )

    # Absolute risk classification
    student["risk_group"] = np.select(
        [
            student["high_persistence"],
            student["victim_freq"],
        ],
        ["ALTO", "MEDIO"],
        default="BAJO"
    )

    return student

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
# Demografía (grado/género/edad) por survey_response_id
# -------------------------------------------------
def _to_int_or_nan(x):
    try:
        return int(str(x).strip())
    except Exception:
        return np.nan

@st.cache_data(ttl=300)
def extract_demographics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per survey_response_id:
      - edad (int)
      - grado (int)
      - genero_code in {M,F,O,N}
    Uses option_code from question_options.
    """
    if df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero_code"])

    demo = df_long[df_long["question_id"].isin([QID_EDAD, QID_GRADO, QID_GENERO])].copy()
    if demo.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero_code"])

    # For each response_id & question_id take the first option_code (should be 1 selected)
    demo_small = (
        demo.sort_values(["survey_response_id", "question_id"])
        .groupby(["survey_response_id", "question_id"], as_index=False)
        .agg(option_code=("option_code", "first"))
    )

    # pivot
    piv = demo_small.pivot(index="survey_response_id", columns="question_id", values="option_code").reset_index()
    piv["edad"] = piv.get(QID_EDAD).apply(_to_int_or_nan) if QID_EDAD in piv.columns else np.nan
    piv["grado"] = piv.get(QID_GRADO).apply(_to_int_or_nan) if QID_GRADO in piv.columns else np.nan
    piv["genero_code"] = piv.get(QID_GENERO) if QID_GENERO in piv.columns else None

    # keep only expected codes
    piv["genero_code"] = piv["genero_code"].astype(str).str.strip().str.upper()
    piv.loc[~piv["genero_code"].isin(["M", "F", "O", "N"]), "genero_code"] = "N"

    return piv[["survey_response_id", "edad", "grado", "genero_code"]]

def render_question_charts(df_long: pd.DataFrame, questions_df: pd.DataFrame, school_name_filter: str):
    """
    For each non-demographic question:
      - x: grado
      - color: genero
      - y: #students who answered with score >= HIGH_FREQ_THRESHOLD (monthly+)
    """
    st.subheader("Resultados por pregunta (Grado × Género)")

    if df_long.empty:
        st.info("No hay datos para graficar.")
        return

    # Filter by school (match your selector)
    if school_name_filter and school_name_filter != "(Todas)":
        df = df_long[df_long["school_name"] == school_name_filter].copy()
    else:
        df = df_long.copy()

    if df.empty:
        st.info("No hay respuestas para la escuela seleccionada.")
        return

    demo = extract_demographics(df)
    if demo.empty:
        st.warning("No se encontraron respuestas demográficas (edad/grado/género).")
        return

    # Merge demographics into df
    df = df.merge(demo, on="survey_response_id", how="left")

    # Only consider rows where we have grade + gender
    df = df[~df["grado"].isna()].copy()
    if df.empty:
        st.warning("No hay datos suficientes con grado/género para graficar.")
        return

    # Exclude demographic questions from chart iteration
    exclude_qids = {QID_EDAD, QID_GRADO, QID_GENERO}

    # We want stable question ordering
    q_list = (
        questions_df[~questions_df["id"].astype(str).isin(exclude_qids)][["id", "question_text"]]
        .dropna()
        .drop_duplicates()
        .sort_values("question_text")
        .values
        .tolist()
    )

    # Gender labels + colors
    gender_map = {
        "M": "Masculino",
        "F": "Femenino",
        "O": "Otro",
        "N": "No responde",
    }
    gender_order = ["Masculino", "Femenino", "Otro", "No responde"]
    color_scale = alt.Scale(
        domain=gender_order,
        range=["#1f77b4", "#f2c300", "#2ca02c", "#8a2be2"],  # azul, amarillo, verde, púrpura
    )

    # Threshold control (optional but useful)
    umbral = st.sidebar.selectbox(
        "Umbral para contar casos por pregunta",
        options=[
            ("Mensual o más (≥ 2)", 2),
            ("Semanal o diario (≥ 3)", 3),
        ],
        index=0,
        format_func=lambda x: x[0],
    )
    threshold = int(umbral[1])

    # Iterate questions with expanders
    for qid, qtext in q_list:
        qid = str(qid)

        sub = df[df["question_id"] == qid].copy()
        if sub.empty:
            continue

        # For each student (survey_response_id) & grade & gender, compute max score for that question
        sub_max = (
            sub.groupby(["survey_response_id", "grado", "genero_code"], as_index=False)
            .agg(max_score=("score", "max"))
        )
        sub_max["genero"] = sub_max["genero_code"].map(gender_map).fillna("No responde")

        # Count as "case" if max_score >= threshold
        cases = sub_max[sub_max["max_score"] >= threshold].copy()
        if cases.empty:
            # still show empty chart? show a note
            with st.expander(qtext, expanded=False):
                st.info(f"Sin casos con umbral {umbral[0]} para esta pregunta.")
            continue

        agg = (
            cases.groupby(["grado", "genero"], as_index=False)
            .size()
            .rename(columns={"size": "estudiantes"})
        )

        # ensure all genders appear per grade (for consistent legend)
        # (optional) but helps: create cartesian frame
        grados = sorted([int(x) for x in agg["grado"].dropna().unique().tolist()])
        base = pd.DataFrame(
            [(g, ge) for g in grados for ge in gender_order],
            columns=["grado", "genero"]
        )
        agg = base.merge(agg, on=["grado", "genero"], how="left")
        agg["estudiantes"] = agg["estudiantes"].fillna(0).astype(int)

        with st.expander(qtext, expanded=False):
            st.caption(f"Conteo de estudiantes con respuesta **{umbral[0]}** para esta pregunta.")
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("grado:O", title="Curso / Grado"),
                    y=alt.Y("estudiantes:Q", title="Número de estudiantes"),
                    color=alt.Color("genero:N", scale=color_scale, sort=gender_order, title="Género"),
                    xOffset=alt.XOffset("genero:N", sort=gender_order),
                    tooltip=["grado", "genero", "estudiantes"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------
# LLM (Groq now, HF later)
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

    prompt = LLM_PROMPT_ES_V1.format(summary=summary)

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un especialista en convivencia escolar y prevención del bullying."
                },
                {
                    "role": "user",
                    "content": prompt
                }
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

school_list = sorted(student["school_name"].dropna().unique().tolist())
school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view = student if school == "(Todas)" else student[student["school_name"] == school]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", len(view))
c2.metric("Victimización ≥ mensual", f"{view['victim_freq'].mean()*100:.1f}%")
c3.metric("Alta persistencia", f"{view['high_persistence'].mean()*100:.1f}%")
c4.metric("Silencio demuestra baja confianza", f"{view['silence_flag'].mean()*100:.1f}%")

st.divider()

st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view["risk_group"].value_counts())

st.divider()

# >>> NUEVA SECCIÓN: gráficos por pregunta (grado x género)
render_question_charts(df_long, questions, school)

st.divider()

st.subheader("Informe interpretativo (IA)")

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)
