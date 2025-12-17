# app.py — Bullying Dashboard (Scientific Indicators + 40 Charts per Question + LLM Report)

import os
import re
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import requests
import altair as alt
from supabase import create_client


# =================================================
# Page config
# =================================================
st.set_page_config(page_title="Bullying Dashboard", layout="wide")


# =================================================
# Supabase config
# =================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =================================================
# Prompt versioning (traceable)
# =================================================
PROMPT_VERSION = "ES_v3_scientific_demographics_2025-12-16"

LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar, prevención del bullying y bienestar estudiantil.
Tu objetivo es ayudar a equipos directivos y docentes a comprender datos agregados de una encuesta.

REGLAS ÉTICAS Y DE CALIDAD (OBLIGATORIAS):
- No identifiques ni infieras identidades individuales.
- No culpes a estudiantes. Evita lenguaje estigmatizante.
- No diagnostiques condiciones clínicas.
- Si un indicador sugiere riesgo, recomienda rutas de apoyo y protocolos escolares.
- Sé claro, educativo, y accionable. Usa español neutro.

TAREA:
Con base SOLO en los datos agregados provistos abajo, redacta un informe breve para administradores escolares.

ESTRUCTURA:
1) Resumen ejecutivo (3-5 bullets)
2) Lectura de indicadores (victimización frecuente, alta persistencia, cyber, silencio institucional)
3) Patrones observables (si existen) por grado y género (si los datos lo permiten)
4) Recomendaciones prácticas (mínimo 6), separadas en:
   - Prevención universal
   - Intervención focalizada
   - Canales de reporte y protección
5) Nota metodológica (2-4 líneas): que esto es agregado y depende de la calidad de respuestas.

DATOS AGREGADOS:
{summary_json}

Incluye al final:
- "Versión del prompt: {prompt_version}"
- "Fecha (UTC): {utc_now}"
""".strip()


# =================================================
# Scoring scale
# =================================================
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

HIGH_FREQ_THRESHOLD = 2  # mensual o más
VERY_HIGH_FREQ = 3       # semanal o diaria


# =================================================
# Demographic question UUIDs
# =================================================
QID_EDAD = "1b5f4f28-8f41-4ed7-9bfa-07d927b2d1b4"
QID_GRADO = "6b5b3cdd-5e6d-4c02-a6c4-63d7b3c52e30"
QID_GENERO = "c0a89b93-2b39-4e4c-9f10-8f58dbf8d0c7"


# =================================================
# Supabase pagination helpers
# =================================================
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


# =================================================
# Data preparation
# =================================================
def build_long_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    """
    Long format with:
    survey_response_id, school_id, school_name, question_id, question_text, option_id, option_text, option_code, score
    """
    needed = [responses, answers, aso, qopts, questions]
    if any(x is None or x.empty for x in needed):
        return pd.DataFrame()

    df = (
        answers
        .merge(responses, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id", suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt"))
    )

    # Merge schools safely
    if schools is not None and not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        df = df.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        df["school_name"] = df["school_name"].fillna(df["school_id"].astype(str))
    else:
        df["school_name"] = df["school_id"].astype(str)

    df["question_text"] = df.get("question_text", "").fillna("").astype(str)
    df["option_text"] = df.get("option_text", "").fillna("").astype(str)
    if "option_code" not in df.columns:
        df["option_code"] = ""

    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)
    return df


# =================================================
# Demographics extraction
# =================================================
def _parse_int_from_text(x: str):
    if x is None:
        return None
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_demographics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: survey_response_id, edad, grado, genero (M/F/O/N)
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"])

    needed = {"survey_response_id", "question_id", "option_text", "option_code"}
    if not needed.issubset(set(df_long.columns)):
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"])

    demo = df_long[df_long["question_id"].astype(str).isin([QID_EDAD, QID_GRADO, QID_GENERO])].copy()
    if demo.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"])

    demo["question_id"] = demo["question_id"].astype(str)
    demo["option_text"] = demo["option_text"].astype(str)
    demo["option_code"] = demo["option_code"].astype(str)

    # edad
    edad = demo[demo["question_id"] == QID_EDAD][["survey_response_id", "option_text", "option_code"]].copy()
    if not edad.empty:
        v1 = edad["option_text"].map(_parse_int_from_text)
        v2 = edad["option_code"].map(_parse_int_from_text)
        edad["edad"] = v1.fillna(v2)
        edad = edad.dropna(subset=["edad"])
        edad["edad"] = edad["edad"].astype(int)
        edad = edad[(edad["edad"] >= 6) & (edad["edad"] <= 20)]
        edad = edad.groupby("survey_response_id", as_index=False)["edad"].first()
    else:
        edad = pd.DataFrame(columns=["survey_response_id", "edad"])

    # grado
    grado = demo[demo["question_id"] == QID_GRADO][["survey_response_id", "option_text", "option_code"]].copy()
    if not grado.empty:
        v1 = grado["option_text"].map(_parse_int_from_text)
        v2 = grado["option_code"].map(_parse_int_from_text)
        grado["grado"] = v1.fillna(v2)
        grado = grado.dropna(subset=["grado"])
        grado["grado"] = grado["grado"].astype(int)
        grado = grado[(grado["grado"] >= 1) & (grado["grado"] <= 13)]
        grado = grado.groupby("survey_response_id", as_index=False)["grado"].first()
    else:
        grado = pd.DataFrame(columns=["survey_response_id", "grado"])

    # genero
    gender = demo[demo["question_id"] == QID_GENERO][["survey_response_id", "option_text", "option_code"]].copy()
    if not gender.empty:
        code = gender["option_code"].str.strip().str.upper()
        code = code.where(code.isin(["M", "F", "O", "N"]), other=pd.NA)

        txt = gender["option_text"].str.strip().str.lower()
        c1 = txt.str.startswith("masc", na=False).to_numpy(dtype=bool)
        c2 = txt.str.startswith("fem", na=False).to_numpy(dtype=bool)
        c3 = txt.str.contains(r"\botro\b", regex=True, na=False).to_numpy(dtype=bool)
        c4 = (txt.str.contains("no", na=False) & txt.str.contains("respond", na=False)).to_numpy(dtype=bool)

        fallback = np.select([c1, c2, c3, c4], ["M", "F", "O", "N"], default=np.nan)
        fallback_series = pd.Series(fallback, index=gender.index)

        gender["genero"] = code.fillna(fallback_series)
        gender = gender.dropna(subset=["genero"])
        gender["genero"] = gender["genero"].astype(str).str.upper()
        gender = gender[gender["genero"].isin(["M", "F", "O", "N"])]
        gender = gender.groupby("survey_response_id", as_index=False)["genero"].first()
    else:
        gender = pd.DataFrame(columns=["survey_response_id", "genero"])

    out = pd.DataFrame({"survey_response_id": demo["survey_response_id"].unique()})
    out = out.merge(edad, on="survey_response_id", how="left")
    out = out.merge(grado, on="survey_response_id", how="left")
    out = out.merge(gender, on="survey_response_id", how="left")
    return out


# =================================================
# Student indicators
# =================================================
def compute_student_metrics(df_long: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    d = df_long.copy()

    is_victim = d["question_text"].str.contains(r"agredido|molestado|ignorado", case=False, na=False)
    is_cyber = d["question_text"].str.contains(r"internet|mensajería|mensajes|videos|fotos", case=False, na=False)
    is_trust_adults = d["question_text"].str.contains(r"adultos|profesores|docentes", case=False, na=False)

    d["victim_score"] = np.where(is_victim, d["score"], 0)
    d["cyber_score"] = np.where(is_cyber, d["score"], 0)
    d["trust_adult_score"] = np.where(is_trust_adults, d["score"], 0)

    student = d.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        school_id=("school_id", "first"),
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

    if demo_df is not None and not demo_df.empty:
        student = student.merge(demo_df, on="survey_response_id", how="left")

    return student


# =================================================
# Summary for LLM
# =================================================
def build_school_summary(view_students: pd.DataFrame) -> dict:
    n = int(len(view_students))
    if n == 0:
        return {}

    def pct(x):
        return round(100 * float(x.mean()), 1) if len(x) else 0.0

    summary = {
        "total_estudiantes": n,
        "prevalencia_victimizacion_pct": pct(view_students["victim_freq"]),
        "prevalencia_cyberbullying_pct": pct(view_students["cyber_freq"]),
        "alta_persistencia_pct": pct(view_students["high_persistence"]),
        "silencio_institucional_pct": pct(view_students["silence_flag"]),
        "distribucion_riesgo": view_students["risk_group"].value_counts().to_dict(),
    }

    if "grado" in view_students.columns:
        g = view_students["grado"].dropna()
        if len(g):
            summary["distribucion_grado"] = g.astype(int).value_counts().sort_index().to_dict()

    if "genero" in view_students.columns:
        ge = view_students["genero"].dropna()
        if len(ge):
            summary["distribucion_genero"] = ge.value_counts().to_dict()

    if "edad" in view_students.columns:
        ed = view_students["edad"].dropna()
        if len(ed):
            summary["edad_min"] = int(ed.min())
            summary["edad_max"] = int(ed.max())
            summary["edad_promedio"] = round(float(ed.mean()), 2)

    return summary


# =================================================
# LLM (Groq)
# =================================================
def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


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

    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    prompt = LLM_PROMPT_ES.format(
        summary_json=summary_json,
        prompt_version=PROMPT_VERSION,
        utc_now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    _ = _prompt_fingerprint(prompt)

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Generas informes en español, claros, educativos y éticos."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text}")

    return r.json()["choices"][0]["message"]["content"]


# =================================================
# 40 charts: per question (Grade x Gender) with fixed colors
# =================================================
GENDER_DOMAIN = ["M", "F", "O", "N"]
GENDER_LABELS = {"M": "Hombres", "F": "Mujeres", "O": "Otro", "N": "No responde"}
GENDER_COLORS = {
    "M": "#1f77b4",  # azul
    "F": "#f2c400",  # amarillo
    "O": "#2ca02c",  # verde
    "N": "#9467bd",  # purpura
}


def _make_question_chart(counts: pd.DataFrame, question_title: str):
    """
    counts columns: grado(int), genero(str), n_estudiantes(int)
    grouped bars by gender, fixed colors
    """
    # show gender labels in legend
    counts = counts.copy()
    counts["Genero"] = counts["genero"].map(GENDER_LABELS).fillna(counts["genero"])
    counts["GeneroCode"] = counts["genero"]

    color_scale = alt.Scale(
        domain=[GENDER_LABELS[g] for g in GENDER_DOMAIN],
        range=[GENDER_COLORS[g] for g in GENDER_DOMAIN],
    )

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("grado:O", title="Grado"),
            xOffset=alt.XOffset("Genero:N"),
            y=alt.Y("n_estudiantes:Q", title="Número de estudiantes"),
            color=alt.Color("Genero:N", scale=color_scale, legend=alt.Legend(title="Género")),
            tooltip=[
                alt.Tooltip("grado:O", title="Grado"),
                alt.Tooltip("Genero:N", title="Género"),
                alt.Tooltip("n_estudiantes:Q", title="Estudiantes"),
            ],
        )
        .properties(
            title=question_title,
            height=280,
        )
    )
    return chart


def render_40_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, school_sel: str):
    st.subheader("Gráficos por cada pregunta (40) — Grado × Género")

    if df_long is None or df_long.empty:
        st.info("No hay datos para graficar.")
        return

    if demo_df is None or demo_df.empty:
        st.warning("No hay demografía (edad/grado/género). Sin eso no se pueden construir los gráficos.")
        return

    # Filter by school if selected
    d = df_long.copy()
    if school_sel != "(Todas)":
        d = d[d["school_name"] == school_sel]

    # Exclude demographic questions themselves
    d = d[~d["question_id"].astype(str).isin([QID_EDAD, QID_GRADO, QID_GENERO])]

    if d.empty:
        st.info("No hay respuestas no-demográficas para graficar en este filtro.")
        return

    # Unique questions (question_id, question_text)
    q_table = (
        d[["question_id", "question_text"]]
        .dropna()
        .drop_duplicates()
        .assign(question_text=lambda x: x["question_text"].astype(str).str.strip())
    )
    q_table = q_table[q_table["question_text"] != ""]
    if q_table.empty:
        st.info("No se encontraron textos de preguntas para mostrar.")
        return

    # Keep only first 40 (stable order)
    q_table = q_table.sort_values("question_text").head(40).reset_index(drop=True)

    # Pre-merge demographics once (fast)
    d = d.merge(demo_df, on="survey_response_id", how="left")

    # Clean demographics needed for plots
    d["grado"] = pd.to_numeric(d["grado"], errors="coerce")
    d["genero"] = d["genero"].astype(str).str.upper()
    d = d.dropna(subset=["grado", "genero"])
    d["grado"] = d["grado"].astype(int)

    # Keep only valid gender codes
    d = d[d["genero"].isin(GENDER_DOMAIN)]

    if d.empty:
        st.warning("Hay respuestas, pero no hay suficientes registros con (grado + género) para graficar.")
        return

    st.caption("Cada gráfico cuenta estudiantes *distintos* que respondieron esa pregunta (por grado y género).")

    for i, row in q_table.iterrows():
        qid = str(row["question_id"])
        qtext = str(row["question_text"]).strip()

        dd = d[d["question_id"].astype(str) == qid].copy()
        if dd.empty:
            continue

        # Count distinct students per (grado, genero)
        counts = (
            dd.groupby(["grado", "genero"])["survey_response_id"]
            .nunique()
            .reset_index(name="n_estudiantes")
        )

        # ensure grades 1..13 exist (optional) & all genders exist
        # (not mandatory, but helps charts look consistent)
        full = []
        for gr in range(1, 14):
            for g in GENDER_DOMAIN:
                full.append({"grado": gr, "genero": g})
        full = pd.DataFrame(full)
        counts = full.merge(counts, on=["grado", "genero"], how="left").fillna({"n_estudiantes": 0})
        counts["n_estudiantes"] = counts["n_estudiantes"].astype(int)

        # Put chart inside expander to avoid heavy UI load
        with st.expander(f"{i+1}. {qtext}", expanded=(i == 0)):
            chart = _make_question_chart(counts, qtext)
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Ver tabla (grado × género)", expanded=False):
                pivot = counts.pivot(index="grado", columns="genero", values="n_estudiantes").fillna(0).astype(int)
                pivot = pivot.reindex(range(1, 14))
                pivot = pivot.rename(columns=GENDER_LABELS)
                st.dataframe(pivot, use_container_width=True)


# =================================================
# UI
# =================================================
st.title("Dashboard de Convivencia Escolar — Análisis Científico")

responses, answers, aso, qopts, questions, schools = load_tables()

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.warning("Aún no hay datos suficientes para análisis.")
    st.stop()

demo_df = extract_demographics(df_long)
student = compute_student_metrics(df_long, demo_df)

school_list = sorted(student["school_name"].dropna().unique().tolist())
school_sel = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view_students = student if school_sel == "(Todas)" else student[student["school_name"] == school_sel]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", int(len(view_students)))
c2.metric("Victimización ≥ mensual", f"{(view_students['victim_freq'].mean()*100):.1f}%")
c3.metric("Alta persistencia", f"{(view_students['high_persistence'].mean()*100):.1f}%")
c4.metric("Silencio institucional", f"{(view_students['silence_flag'].mean()*100):.1f}%")

st.divider()

st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view_students["risk_group"].value_counts())

st.divider()

# 40 question charts
render_40_question_charts(df_long, demo_df, school_sel)

st.divider()

st.subheader("Informe interpretativo (IA)")

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view_students)
        report = generate_llm_report(summary)
        st.markdown(report)

with st.expander("Debug (opcional)"):
    st.write("Rows df_long:", len(df_long))
    st.write("Rows demo_df:", len(demo_df))
    st.write("Rows student:", len(student))
    if not demo_df.empty:
        st.write("Demografía (muestra):")
        st.dataframe(demo_df.head(10), use_container_width=True)
