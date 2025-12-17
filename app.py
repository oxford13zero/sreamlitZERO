# app.py — Bullying Dashboard (Scientific Indicators + Demographics + 40 Charts + LLM Report)

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
# IDs de preguntas demográficas (según tu BD)
# -------------------------------------------------
QID_EDAD = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO = "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"
QID_TIEMPO = "7c5d8e66-1d8d-4f0c-8a4f-8a6b6b5c4c11"

# -------------------------------------------------
# Scoring scale (frequency-based) - para indicadores
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
# LLM Prompt (ES)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar y prevención del bullying.

Tu tarea: redactar un informe claro, educativo, ético y no alarmista para un equipo directivo escolar,
basado ÚNICAMENTE en los datos agregados siguientes.

Reglas:
- NO identifiques ni infieras identidades de estudiantes.
- Evita juicios morales; usa lenguaje cuidadoso, centrado en prevención y apoyo.
- Señala limitaciones: muestra que son datos de una encuesta y no un “diagnóstico”.

Entrega en español, con estructura:

1) Resumen ejecutivo (3-5 bullets)
2) Hallazgos principales (con porcentajes)
3) Señales de alerta a monitorear (si aplica)
4) Recomendaciones prácticas (5 acciones concretas, realistas para escuela)
5) Próximos pasos de medición (qué mejorar en encuesta/datos)

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
# Helpers: normalización robusta de columnas de options
# -------------------------------------------------
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _norm_gender_from_value(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s:
        return None

    # Si ya viene en código
    if s.upper() in {"M", "F", "O", "N"}:
        return s.upper()

    # Si viene como texto
    sl = s.lower()
    if "masc" in sl:
        return "M"
    if "fem" in sl:
        return "F"
    if "otro" in sl:
        return "O"
    if "no quiero" in sl or "prefiero" in sl or "no responder" in sl:
        return "N"

    return None

def _to_int_safe(v) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    # Algunos option_code podrían venir como "07" o "7"
    try:
        return int(float(s))
    except Exception:
        return None

# -------------------------------------------------
# Data preparation: long df (answers + selections + options + questions + schools)
# -------------------------------------------------
def build_long_df(responses, answers, aso, qopts, questions, schools):
    """
    Devuelve DF a nivel de opción seleccionada:
    - 1 fila por (question_answer_id, option_id)
    Incluye:
    - question_text
    - opt_code (robusto)
    - opt_text (robusto)
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

    # Merge schools safely
    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        df = df.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    # Robusto: option code/text puede llamarse distinto según tu backend
    code_col = _pick_col(df, ["option_code", "code", "value"])
    text_col = _pick_col(df, ["option_text", "text", "label"])

    df["question_text"] = df.get("question_text", "").fillna("")
    df["opt_code"] = df[code_col].astype(str) if code_col else ""
    df["opt_text"] = df[text_col].astype(str) if text_col else ""

    # Limpieza básica
    df["opt_code"] = df["opt_code"].replace({"nan": "", "None": ""}).fillna("")
    df["opt_text"] = df["opt_text"].replace({"nan": "", "None": ""}).fillna("")

    # Score por texto (para indicadores clásicos)
    df["score"] = df["opt_text"].map(SCALE).fillna(0).astype(float)

    return df

# -------------------------------------------------
# Student-level scientific indicators (basado en question_text + score)
# -------------------------------------------------
def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    is_victim = d["question_text"].str.contains(r"agredido|molestado|ignorado", case=False, na=False)
    is_cyber  = d["question_text"].str.contains(r"internet|mensajería|mensajes|videos|fotos", case=False, na=False)
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
# Demografía: desde answer_selected_options (df_long)
# -------------------------------------------------
def extract_demographics_from_options(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Construye demo_df (1 fila por survey_response_id) con:
    edad (int), grado (int), genero (M/F/O/N), tiempo (int)
    leyendo preferentemente:
      - opt_code (si es numérico / M F O N)
      - si no, opt_text
    """
    if df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    demo_rows = df_long[df_long["question_id"].isin([QID_EDAD, QID_GRADO, QID_GENERO, QID_TIEMPO])].copy()
    if demo_rows.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    # value_raw: preferir opt_code si viene, si no opt_text
    demo_rows["value_raw"] = demo_rows["opt_code"]
    demo_rows.loc[demo_rows["value_raw"].astype(str).str.strip().eq(""), "value_raw"] = demo_rows["opt_text"]

    # Pivot: 1 col por question_id
    pivot = demo_rows.pivot_table(
        index=["survey_response_id", "school_name"],
        columns="question_id",
        values="value_raw",
        aggfunc="first"
    )

    # Acceso seguro a columnas que podrían no existir
    def col_series(qid: str) -> pd.Series:
        if qid in pivot.columns:
            return pivot[qid]
        return pd.Series([None] * len(pivot), index=pivot.index)

    edad_s = col_series(QID_EDAD).apply(_to_int_safe)
    grado_s = col_series(QID_GRADO).apply(_to_int_safe)
    tiempo_s = col_series(QID_TIEMPO).apply(_to_int_safe)

    genero_s = col_series(QID_GENERO).apply(_norm_gender_from_value)

    out = pd.DataFrame({
        "survey_response_id": [idx[0] for idx in pivot.index],
        "school_name": [idx[1] for idx in pivot.index],
        "edad": edad_s.values,
        "grado": grado_s.values,
        "genero": genero_s.values,
        "tiempo": tiempo_s.values,
    })

    # Limpieza final: convertir a int cuando corresponda
    for c in ["edad", "grado", "tiempo"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out

# -------------------------------------------------
# Charts: 40 gráficos por pregunta (Grado x Género)
# -------------------------------------------------
def render_40_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, max_questions: int = 40):
    st.subheader(f"Gráficos por cada pregunta ({max_questions}) — Grado × Género")

    if demo_df.empty:
        st.warning("No hay demografía (edad/grado/género). Sin eso no se pueden construir los gráficos.")
        return

    # Necesitamos al menos grado y genero válidos
    demo_ok = demo_df.dropna(subset=["grado", "genero"]).copy()
    if demo_ok.empty:
        st.warning("Demografía existe, pero no hay filas con (grado y género) válidos.")
        return

    # Base: respuestas NO demográficas
    base = df_long[~df_long["question_id"].isin([QID_EDAD, QID_GRADO, QID_GENERO, QID_TIEMPO])].copy()
    if base.empty:
        st.warning("No hay respuestas no-demográficas para graficar.")
        return

    # Unir demografía por survey_response_id
    base = base.merge(
        demo_ok[["survey_response_id", "grado", "genero"]],
        on="survey_response_id",
        how="inner"
    )

    # Quedarnos con preguntas únicas (hasta 40)
    q_list = (
        base[["question_id", "question_text"]]
        .drop_duplicates()
        .sort_values("question_text")
        .head(max_questions)
        .to_dict("records")
    )

    # Para cada pregunta: contar # de respuestas por grado y género
    for qi, qrow in enumerate(q_list, start=1):
        qid = qrow["question_id"]
        qtext = (qrow["question_text"] or "").strip()
        if not qtext:
            qtext = f"Pregunta {qi}"

        sub = base[base["question_id"] == qid].copy()
        if sub.empty:
            continue

        # Aquí contamos “personas” que respondieron esa pregunta (por grado/género)
        # Si hay preguntas multi-selección, habrá más de una fila por estudiante -> deduplicar
        sub_unique = sub.drop_duplicates(subset=["survey_response_id", "grado", "genero"])

        counts = (
            sub_unique
            .groupby(["grado", "genero"])
            .size()
            .reset_index(name="n")
        )

        # pivot para barras agrupadas
        pivot = counts.pivot(index="grado", columns="genero", values="n").fillna(0).astype(int)

        # asegurar columnas en orden M/F/O/N (si existen)
        for col in ["M", "F", "O", "N"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["M", "F", "O", "N"]]

        # asegurar grados 1..13 en el eje X
        all_grades = pd.Index(range(1, 14), name="grado")
        pivot = pivot.reindex(all_grades, fill_value=0)

        st.markdown(f"### {qtext}")
        st.caption("Eje X: grado | Eje Y: número de estudiantes (que respondieron la pregunta), desagregado por género.")
        st.bar_chart(pivot)

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

# ---- Demografía + 40 gráficos
demo_df = extract_demographics_from_options(df_long)

# Filtrar demo_df si el usuario seleccionó una escuela específica
if school != "(Todas)":
    demo_df_view = demo_df[demo_df["school_name"] == school].copy()
else:
    demo_df_view = demo_df.copy()

render_40_question_charts(df_long if school == "(Todas)" else df_long[df_long["school_name"] == school], demo_df_view, max_questions=40)

st.divider()

st.subheader("Informe interpretativo (IA)")
if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)

with st.expander("Debug (recomendado ahora)"):
    st.write("Rows df_long:", len(df_long))
    st.write("Rows demo_df:", len(demo_df))
    st.write("demo_df (head):")
    st.dataframe(demo_df.head(20), use_container_width=True)
    st.write("Ejemplos de preguntas (question_text):")
    st.dataframe(df_long[["question_id", "question_text"]].drop_duplicates().head(20), use_container_width=True)
