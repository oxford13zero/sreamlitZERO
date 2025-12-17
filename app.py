# app.py — Bullying Dashboard (Scientific Indicators + Demographics-by-Options + 40 Charts + LLM ES)

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
# Constants (Question IDs for demographics)
# -------------------------------------------------
QID_EDAD = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO = "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"
QID_TIEMPO = "7c5d8e66-1d8d-4f0c-8a4f-8a6b6b5c4c11"

DEMO_QIDS = {QID_EDAD, QID_GRADO, QID_GENERO, QID_TIEMPO}
GEN_ORDER = ["M", "F", "O", "N"]

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
# LLM prompt (Spanish, educational & ethical)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un/a especialista en convivencia escolar y prevención del bullying.

Tu tarea: redactar un informe claro, educativo y ético para directivos/as y equipos escolares,
basado ÚNICAMENTE en los indicadores agregados provistos (no inventes datos).

Requisitos:
- No identifiques ni infieras identidades individuales.
- Evita lenguaje acusatorio; usa enfoque preventivo, de apoyo y mejora continua.
- Señala límites del análisis (es un diagnóstico descriptivo con encuesta, no prueba causal).
- Incluye acciones concretas, priorizadas y realistas para una escuela.

Estructura sugerida:
1) Resumen ejecutivo (3–5 bullets)
2) Hallazgos principales (qué está alto/medio/bajo y qué significa)
3) Señales de alerta (si aplica) y posibles interpretaciones prudentes
4) Recomendaciones (mínimo 6), separadas por:
   - Acciones inmediatas (0–30 días)
   - Acciones de mediano plazo (1–3 meses)
   - Acciones sostenibles (3–12 meses)
5) Indicadores a monitorear (qué medir en la próxima encuesta)

INDICADORES AGREGADOS:
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
# Data preparation: long df (options-based)
# -------------------------------------------------
def build_long_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    df = (
        answers
        .merge(responses, left_on="survey_response_id", right_on="id",
               suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id",
               suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt"))
    )

    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        df = df.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    df["question_text"] = df["question_text"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["option_code"] = df["option_code"].fillna("").astype(str)

    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

# -------------------------------------------------
# Student-level scientific indicators
# -------------------------------------------------
def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

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
        default="BAJO"
    )

    return student

# -------------------------------------------------
# Demographics extraction FROM OPTIONS (answer_selected_options)
#   FIXED: always returns same-length series
# -------------------------------------------------
def _to_int_safe(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        return int(float(s))
    except Exception:
        return np.nan

def _norm_gender_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().upper()
    if s in {"M", "F", "O", "N"}:
        return s
    if s.startswith("MASC"):
        return "M"
    if s.startswith("FEM"):
        return "F"
    return np.nan

def extract_demographics_from_options(df_long: pd.DataFrame, school_filter: str | None = None) -> pd.DataFrame:
    """
    One row per survey_response_id:
      survey_response_id, school_name, edad, grado, genero, tiempo
    Extracted from option_code for demographic question IDs.
    """
    cols = ["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"]
    if df_long.empty:
        return pd.DataFrame(columns=cols)

    d = df_long.copy()
    if school_filter and school_filter != "(Todas)":
        d = d[d["school_name"] == school_filter]

    demo = d[d["question_id"].isin(DEMO_QIDS)].copy()
    if demo.empty:
        return pd.DataFrame(columns=cols)

    demo = demo[["survey_response_id", "school_name", "question_id", "option_code"]].copy()

    # 1 row per (response, question)
    demo = (
        demo.sort_values(["survey_response_id", "question_id"])
            .groupby(["survey_response_id", "question_id"], as_index=False)
            .agg({"school_name": "first", "option_code": "first"})
    )

    pivot = demo.pivot(index="survey_response_id", columns="question_id", values="option_code")
    # School name mapping (aligned to pivot.index)
    school_map = demo.groupby("survey_response_id")["school_name"].first().reindex(pivot.index)

    # helper to ALWAYS return a Series of len(pivot.index)
    def col_series(qid: str) -> pd.Series:
        if qid in pivot.columns:
            return pivot[qid].reindex(pivot.index)
        return pd.Series([np.nan] * len(pivot.index), index=pivot.index)

    edad_s = col_series(QID_EDAD).apply(_to_int_safe)
    grado_s = col_series(QID_GRADO).apply(_to_int_safe)
    genero_s = col_series(QID_GENERO).apply(_norm_gender_code)
    tiempo_s = col_series(QID_TIEMPO).apply(_to_int_safe)

    out = pd.DataFrame({
        "survey_response_id": pivot.index,
        "school_name": school_map.values,
        "edad": edad_s.values,
        "grado": grado_s.values,
        "genero": genero_s.values,
        "tiempo": tiempo_s.values,
    })

    # sanity ranges
    out.loc[(out["edad"] < 0) | (out["edad"] > 120), "edad"] = np.nan
    out.loc[(out["grado"] < 0) | (out["grado"] > 20), "grado"] = np.nan
    out.loc[(out["tiempo"] < 0) | (out["tiempo"] > 50), "tiempo"] = np.nan

    return out

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
                {"role": "system", "content": "Eres un especialista en convivencia escolar y prevención del bullying. Respondes en español."},
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
# Charts: 40 question charts (Grade × Gender)
# -------------------------------------------------
def render_40_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, school_filter: str):
    st.subheader("Gráficos por cada pregunta (40) — Grado × Género")

    if df_long.empty:
        st.warning("No hay datos (df_long vacío).")
        return

    if demo_df.empty:
        st.warning("No hay demografía (edad/grado/género/tiempo) en answer_selected_options. Sin eso no se pueden construir los gráficos.")
        return

    d = df_long.copy()
    if school_filter and school_filter != "(Todas)":
        d = d[d["school_name"] == school_filter]
        demo = demo_df[demo_df["school_name"] == school_filter].copy()
    else:
        demo = demo_df.copy()

    demo = demo.dropna(subset=["grado", "genero"]).copy()
    if demo.empty:
        st.warning("Demografía existe, pero no hay filas con (grado y género) válidos.")
        return

    demo["grado"] = demo["grado"].astype(int)
    demo["genero"] = demo["genero"].astype(str)

    q_df = d[~d["question_id"].isin(DEMO_QIDS)].dropna(subset=["question_id"]).copy()

    questions_present = (
        q_df[["question_id", "question_text"]]
        .drop_duplicates()
        .sort_values("question_text")
        .head(40)
    )

    if questions_present.empty:
        st.warning("No hay preguntas no-demográficas para graficar.")
        return

    m = q_df.merge(
        demo[["survey_response_id", "grado", "genero"]],
        on="survey_response_id",
        how="inner"
    )

    if m.empty:
        st.warning("No hay intersección entre respuestas y demografía (por survey_response_id).")
        return

    m["genero"] = pd.Categorical(m["genero"], categories=GEN_ORDER, ordered=True)

    for _, row in questions_present.iterrows():
        qid = row["question_id"]
        qtext = row["question_text"]

        sub = m[m["question_id"] == qid].copy()
        if sub.empty:
            continue

        counts = (
            sub.groupby(["grado", "genero"])["survey_response_id"]
               .nunique()
               .unstack(fill_value=0)
        )

        for g in GEN_ORDER:
            if g not in counts.columns:
                counts[g] = 0
        counts = counts[GEN_ORDER].sort_index()

        st.markdown(f"### {qtext}")
        st.bar_chart(counts, height=320)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Dashboard de Convivencia Escolar — Análisis Científico")

responses, answers, aso, qopts, questions, schools = load_tables()

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.warning("Aún no hay datos suficientes para análisis (faltan respuestas/opciones).")
    st.stop()

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

demo_df = extract_demographics_from_options(df_long, school_filter=school)
render_40_question_charts(df_long, demo_df, school_filter=school)

st.divider()

st.subheader("Informe interpretativo (IA)")
if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)

# -------------------------------------------------
# Debug
# -------------------------------------------------
with st.expander("Debug (recomendado ahora)", expanded=False):
    st.write("Rows df_long:", int(len(df_long)))
    st.write("Rows demo_df:", int(len(demo_df)))

    if not demo_df.empty:
        st.write("demo_df (head):")
        st.dataframe(demo_df.head(10), use_container_width=True)

    dcounts = (
        df_long[df_long["question_id"].isin(DEMO_QIDS)]
        .groupby(["question_id", "question_text"])["survey_response_id"]
        .nunique()
        .reset_index()
        .rename(columns={"survey_response_id": "n_responses"})
        .sort_values("question_text")
    )
    st.write("Respuestas demográficas detectadas en df_long (por question_id):")
    st.dataframe(dcounts, use_container_width=True)

    sample = (
        df_long[df_long["question_id"].isin(DEMO_QIDS)]
        [["survey_response_id", "question_text", "option_code", "option_text"]]
        .drop_duplicates()
        .head(25)
    )
    st.write("Muestra de filas (demografía) desde options:")
    st.dataframe(sample, use_container_width=True)
