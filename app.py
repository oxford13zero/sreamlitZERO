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
# Demographics extraction (ROBUST: detect by question_text)
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


def _detect_demographic_qids(df_long: pd.DataFrame) -> dict:
    """
    Detect demographic question_ids by question_text (robust across recreated surveys).
    Returns dict: {"edad": qid or None, "grado": qid or None, "genero": qid or None}
    """
    if df_long is None or df_long.empty:
        return {"edad": None, "grado": None, "genero": None}

    qt = df_long[["question_id", "question_text"]].drop_duplicates().copy()
    qt["qtext"] = qt["question_text"].astype(str).str.strip().str.lower()

    def pick(patterns):
        mask = False
        for p in patterns:
            mask = mask | qt["qtext"].str.contains(p, regex=True, na=False)
        hit = qt[mask]
        if hit.empty:
            return None
        # si hay varias, tomamos la primera estable
        return str(hit.iloc[0]["question_id"])

    edad_qid = pick([r"\bedad\b", r"cu[aá]l es tu edad"])
    grado_qid = pick([r"\bgrado\b", r"en qu[eé] grado est[aá]s", r"curso"])
    genero_qid = pick([r"\bg[eé]nero\b", r"cu[aá]l es tu g[eé]nero", r"sexo"])

    return {"edad": edad_qid, "grado": grado_qid, "genero": genero_qid}


def extract_demographics(df_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns (demo_df, detected_qids)
    demo_df columns: survey_response_id, edad, grado, genero (M/F/O/N)
    """
    detected = _detect_demographic_qids(df_long)

    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"]), detected

    needed = {"survey_response_id", "question_id", "option_text", "option_code", "question_text"}
    if not needed.issubset(set(df_long.columns)):
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"]), detected

    qid_edad = detected["edad"]
    qid_grado = detected["grado"]
    qid_genero = detected["genero"]

    qids = [x for x in [qid_edad, qid_grado, qid_genero] if x]
    if not qids:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"]), detected

    demo = df_long[df_long["question_id"].astype(str).isin(qids)].copy()
    if demo.empty:
        return pd.DataFrame(columns=["survey_response_id", "edad", "grado", "genero"]), detected

    demo["question_id"] = demo["question_id"].astype(str)
    demo["option_text"] = demo["option_text"].astype(str)
    demo["option_code"] = demo["option_code"].astype(str)

    # edad
    if qid_edad:
        edad = demo[demo["question_id"] == qid_edad][["survey_response_id", "option_text", "option_code"]].copy()
    else:
        edad = pd.DataFrame()
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
    if qid_grado:
        grado = demo[demo["question_id"] == qid_grado][["survey_response_id", "option_text", "option_code"]].copy()
    else:
        grado = pd.DataFrame()
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
    if qid_genero:
        gender = demo[demo["question_id"] == qid_genero][["survey_response_id", "option_text", "option_code"]].copy()
    else:
        gender = pd.DataFrame()
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
    return out, detected


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
    "N": "#9467bd",  # púrpura
}


def _make_question_chart(counts: pd.DataFrame, question_title: str):
    counts = counts.copy()
    counts["Genero"] = counts["genero"].map(GENDER_LABELS).fillna(counts["genero"])

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
        .properties(title=question_title, height=280)
    )
    return chart


def render_40_question_charts(df_long: pd.DataFrame, demo_df: pd.DataFrame, detected_qids: dict, school_sel: str):
    st.subheader("Gráficos por cada pregunta (40) — Grado × Género")

    if df_long is None or df_long.empty:
        st.info("No hay datos para graficar.")
        return

    if demo_df is None or demo_df.empty:
        st.warning("No hay demografía (edad/grado/género). Sin eso no se pueden construir los gráficos.")
        st.info(f"Detectados por texto: edad={detected_qids.get('edad')}, grado={detected_qids.get('grado')}, genero={detected_qids.get('genero')}")
        return

    # Filter by school
    d = df_long.copy()
    if school_sel != "(Todas)":
        d = d[d["school_name"] == school_sel]

    # Exclude demographic questions by detected IDs
    demo_qids = [detected_qids.get("edad"), detected_qids.get("grado"), detected_qids.get("genero")]
    demo_qids = [x for x in demo_qids if x]
    if demo_qids:
        d = d[~d["question_id"].astype(str).isin(demo_qids)]

    if d.empty:
        st.info("No hay respuestas no-demográficas para graficar en este filtro.")
        return

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

    q_table = q_table.sort_values("question_text").head(40).reset_index(drop=True)

    # Merge demographics once
    d = d.merge(demo_df, on="survey_response_id", how="left")

    d["grado"] = pd.to_numeric(d["grado"], errors="coerce")
    d["genero"] = d["genero"].astype(str).str.upper()
    d = d.dropna(subset=["grado", "genero"])
    d["grado"] = d["grado"].astype(int)
    d = d[d["genero"].isin(GENDER_DOMAIN)]

    if d.empty:
        st.warning("Hay respuestas, pero no hay suficientes registros con (grado + género) para graficar.")
        return

    st.caption("Cada gráfico cuenta estudiantes distintos que respondieron esa pregunta (por grado y género).")

    for i, row in q_table.iterrows():
        qid = str(row["question_id"])
        qtext = str(row["question_text"]).strip()

        dd = d[d["question_id"].astype(str) == qid].copy()
        if dd.empty:
            continue

        counts = (
            dd.groupby(["grado", "genero"])["survey_response_id"]
            .nunique()
            .reset_index(name="n_estudiantes")
        )

        # Full grid 1..13 × genders
        full = pd.DataFrame(
            [{"grado": gr, "genero": g} for gr in range(1, 14) for g in GENDER_DOMAIN]
        )
        counts = full.merge(counts, on=["grado", "genero"], how="left").fillna({"n_estudiantes": 0})
        counts["n_estudiantes"] = counts["n_estudiantes"].astype(int)

        with st.expander(f"{i+1}. {qtext}", expanded=(i == 0)):
            st.altair_chart(_make_question_chart(counts, qtext), use_container_width=True)

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

demo_df, detected_qids = extract_demographics(df_long)
student = compute_student_metrics(df_long, demo_df)

school_list = sorted(student["school_name"].dropna().unique().tolist())
school_sel = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view_students = student if school_sel == "(Todas)" else student[student["school_name"] == school_sel]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes", int(len(view_students)))
c2.metric("Victimización ≥ mensual", f"{(view_students['victim_freq'].mean()*100):.1f}%")
c3.metric("Alta persistencia", f"{(view_students['high_persistence'].mean()*100):.1f}%")
c4.metric("Silencio institucional", f"{(view_students['silence_flag'].mean()*100):.1f}%")

st.divider()

st.subheader("Distribución de niveles de riesgo")
st.bar_chart(view_students["risk_group"].value_counts())

st.divider()

render_40_question_charts(df_long, demo_df, detected_qids, school_sel)

st.divider()

st.subheader("Informe interpretativo (IA)")

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view_students)
        report = generate_llm_report(summary)
        st.markdown(report)

with st.expander("Debug (recomendado ahora)", expanded=True):
    st.write("Rows df_long:", len(df_long))
    st.write("Demografía detectada por texto:", detected_qids)
    st.write("Rows demo_df:", len(demo_df))
    if not df_long.empty:
        st.write("Ejemplos de preguntas (question_text):")
        st.dataframe(
            df_long[["question_id", "question_text"]].drop_duplicates().head(30),
            use_container_width=True
        )
    if not demo_df.empty:
        st.write("Demografía (muestra):")
        st.dataframe(demo_df.head(20), use_container_width=True)
