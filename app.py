# app.py — Bullying Dashboard (Scientific Indicators + Demographics + 40 Charts + LLM Report)
# FIXED: Uses submitted only, avoids false negatives from missing answers, uses prevalence + thresholds, silence flags,
# robust denominators, and keeps 40 charts (Grado × Género) from selected options.
#
# DASHBOARD UPDATE (visible-first):
# - KPIs show % + counts (with correct denominators)
# - Adds “semaforo” (5/10/20) based on victimización frecuente
# - Adds explicit caution about subreporte + “silencio institucional”
# - Adds subgroup blocks (género/grado × victim/cyber) with correct denominators
#
# UPDATE (requested):
# - If URL includes school_id + analysis_dt:
#   - Load ONLY survey_responses for that school and exact analysis_requested_dt == analysis_dt
#   - Load ONLY related question_answers / answer_selected_options / question_options / questions / schools
#   - Chunk `.in_()` requests to avoid PostgREST 400 Bad Request due to too-long querystrings
#
# FIX (this round):
# - Fix IndentationError in _groq_report (prompt + requests.post properly indented)
# - Ensure LLM prompt receives required variables (n_encuestas_submitted_total, n_estudiantes_con_datos_relevantes)

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
QID_EDAD = "zero_general_edad"
QID_GRADO = "zero_general_curso"
QID_GENERO = "zero_general_genero"
QID_TIEMPO = "zero_general_tiempo"
DEMOGRAPHIC_QIDS = {QID_EDAD, QID_GRADO, QID_GENERO, QID_TIEMPO}

# -------------------------------------------------
# Scoring scale (frequency-based) - para indicadores
# -------------------------------------------------
SCALE = {
    # Frecuencia (0-3) — ZERO-R (últimas 8 semanas)
    "Nunca": 0,
    "0": 0,
    "1 o 2 veces": 1,
    "1": 1,
    "3 a 5 veces": 2,
    "2": 2,
    "Más de 5 veces": 3,
    "Mas de 5 veces": 3,
    "3": 3,

    # Escala A-D (0-3) — items de liderazgo/ambiente/etc.
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,

    # Compatibilidad con encuestas previas (si quedan datos históricos)
    "No": 0,
    "Sí": 2,
    "Si": 2,
    "Solo una vez": 1,
    "Sólo una vez": 1,
    "A veces": 1,
    "Varias veces al mes": 2,
    "Casi cada semana": 3,
    "Casi cada día": 4,
    "Casi cada dia": 4,
}

HIGH_FREQ_THRESHOLD = 2  # mensual o más
VERY_HIGH_FREQ = 3       # semanal o diaria

# -------------------------------------------------
# Marco científico: umbrales escuela (prevalencia absoluta)
# -------------------------------------------------
def threshold_flag(pct: float) -> str:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "SIN_DATOS"
    if pct >= 20:
        return "CRISIS"
    if pct >= 10:
        return "INTERVENCION"
    if pct >= 5:
        return "ATENCION"
    return "MONITOREO"


def risk_level_message(pct: float) -> tuple[str, str]:
    """
    pct in 0..100 (prevalencia absoluta)
    """
    lvl = threshold_flag(pct)
    if lvl == "CRISIS":
        return lvl, "≥20% reporta victimización frecuente (≥ mensual). Requiere acción inmediata a nivel escuela."
    if lvl == "INTERVENCION":
        return lvl, "≥10% reporta victimización frecuente (≥ mensual). Requiere intervención estructurada."
    if lvl == "ATENCION":
        return lvl, "≥5% reporta victimización frecuente (≥ mensual). Requiere monitoreo y acciones preventivas."
    if lvl == "MONITOREO":
        return lvl, "<5% reporta victimización frecuente (≥ mensual). Mantener monitoreo preventivo."
    return lvl, "Sin datos suficientes para clasificar el nivel."

# -------------------------------------------------
# LLM Prompt (ES)
# -------------------------------------------------
LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar y prevención del bullying con 30 años de experiencia, tanto en Chile
como en EEUU. Ademas, eres un experto en analisis estadistico y experto en tomar encuestas estudiantiles para analizar
ambiente escolar.

Tu tarea: redactar un informe claro, educativo, ético y no alarmista para un equipo directivo escolar,
basado ÚNICAMENTE en los datos agregados siguientes.

Tamaño de muestra:
- Encuestas submitted totales: {n_encuestas_submitted_total}
- Estudiantes con datos válidos para indicadores sensibles: {n_estudiantes_con_datos_relevantes}

Reglas:
Reglas:
- Cuando describas la encuesta, usa SIEMPRE el número de encuestas submitted totales.
- Cuando describas un indicador específico, usa su n_with_data correspondiente.
- Nunca infieras el tamaño muestral a partir de n_with_data.
- NO identifiques ni infieras identidades de estudiantes.
- Evita juicios morales; usa lenguaje cuidadoso, centrado en prevención y apoyo.
- Señala limitaciones: muestra que son datos de una encuesta y no un “diagnóstico”.
- Si hay valores faltantes (missing), explícitalo y evita conclusiones optimistas por falta de datos.
- Senala clase con mas altos indices de bullying
- En que grupo femenino, masculino, no responde u otro existe mas violencia escolar
- Detecta donde se concentran los grupos de estudiantes con mas agresores, y con mas victimas
- Detecta el o los lugares donde se producen mas agresiones y listalos desde los mas inseguros hasta los mas seguros.
- Indica en quien confian mas los estudiantes al momento de reportar bullying, los padres? profesores? otros estudiantes? etc


Entrega en español, con estructura:

1) Resumen ejecutivo (3-5 bullets)
2) Hallazgos principales (con porcentajes y denominadores)
3) Señales de alerta a monitorear (si aplica)
4) Recomendaciones prácticas (5 acciones concretas, realistas para escuela)
5) Próximos pasos de medición (qué mejorar en encuesta/datos)

DATOS AGREGADOS:
{summary}
""".strip()

# -------------------------------------------------
# Helpers: normalización robusta de columnas
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

    if s.upper() in {"M", "F", "O", "N"}:
        return s.upper()

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
    try:
        return int(float(s))
    except Exception:
        return None


def _safe_bool_rate(series: pd.Series) -> dict:
    """
    Returns prevalence with correct denominator.
    Treats NA as missing (excluded from denominator).
    """
    if series is None or len(series) == 0:
        return {"pct": np.nan, "n_with_data": 0, "n_missing": 0, "missing_pct": np.nan}
    s = series.copy()
    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    n_with = int(n_total - n_missing)
    if n_with == 0:
        return {
            "pct": np.nan,
            "n_with_data": 0,
            "n_missing": n_missing,
            "missing_pct": round(100 * n_missing / n_total, 1) if n_total else np.nan
        }
    pct = float((s.dropna().astype(bool).mean()) * 100.0)
    return {
        "pct": round(pct, 1),
        "n_with_data": n_with,
        "n_missing": n_missing,
        "missing_pct": round(100 * n_missing / n_total, 1) if n_total else np.nan,
    }


def _safe_bool_count(series: pd.Series) -> dict:
    """
    Returns counts consistent with _safe_bool_rate:
      - n_true among non-missing
      - n_with_data
      - n_missing
    """
    if series is None or len(series) == 0:
        return {"n_true": 0, "n_with_data": 0, "n_missing": 0}
    s = series.copy()
    n_missing = int(s.isna().sum())
    s2 = s.dropna().astype(bool)
    return {
        "n_true": int(s2.sum()),
        "n_with_data": int(len(s2)),
        "n_missing": int(n_missing)
    }

# -------------------------------------------------
# IMPORTANT: Use submitted only
# -------------------------------------------------
def filter_submitted_responses(responses: pd.DataFrame) -> pd.DataFrame:
    if responses.empty:
        return responses
    if "status" not in responses.columns:
        return responses
    return responses[responses["status"].astype(str).str.lower().eq("submitted")].copy()

# -------------------------------------------------
# Supabase fetch helpers (chunked in_ to avoid 400 Bad Request)
# -------------------------------------------------
def _fetch_in_chunks(table: str, col: str, ids: list, select: str = "*", chunk_size: int = 200) -> list[dict]:
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        rows = (
            supabase.table(table)
            .select(select)
            .in_(col, chunk)
            .execute()
            .data
        )
        if rows:
            out.extend(rows)
    return out

# -------------------------------------------------
# Load tables (filtered when school_id + analysis_dt are present)
# -------------------------------------------------
@st.cache_data(ttl=300)
def load_tables_filtered(school_id: str, analysis_dt: str):
    # 1) survey_responses for this school and exact analysis_requested_dt
    responses_rows = (
        supabase.table("survey_responses")
        .select("*")
        .eq("school_id", school_id)
        .eq("analysis_requested_dt", analysis_dt)
        .execute()
        .data
    )
    responses = pd.DataFrame(responses_rows or [])
    if responses.empty:
        # return empty frames in same order
        return (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )

    # 2) question_answers for those survey_response_id
    resp_ids = responses["id"].dropna().tolist() if "id" in responses.columns else []
    answers_rows = _fetch_in_chunks("question_answers", "survey_response_id", resp_ids, select="*")
    answers = pd.DataFrame(answers_rows or [])
    if answers.empty:
        return (
            responses, pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )

    # 3) answer_selected_options for those question_answer_id (CHUNKED)
    ans_ids = answers["id"].dropna().tolist() if "id" in answers.columns else []
    aso_rows = _fetch_in_chunks("answer_selected_options", "question_answer_id", ans_ids, select="*")
    aso = pd.DataFrame(aso_rows or [])

    # 4) question_options only for option_ids present
    option_ids = aso["option_id"].dropna().tolist() if (not aso.empty and "option_id" in aso.columns) else []
    qopts_rows = _fetch_in_chunks("question_options", "id", option_ids, select="*")
    qopts = pd.DataFrame(qopts_rows or [])

    # 5) questions only for question_ids present in answers
    qids = answers["question_id"].dropna().tolist() if "question_id" in answers.columns else []
    questions_rows = _fetch_in_chunks("questions", "id", qids, select="*")
    questions = pd.DataFrame(questions_rows or [])

    # 6) schools only this school
    schools_rows = (
        supabase.table("schools")
        .select("*")
        .eq("id", school_id)
        .execute()
        .data
    )
    schools = pd.DataFrame(schools_rows or [])

    return responses, answers, aso, qopts, questions, schools


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
def load_tables_unfiltered():
    return (
        _fetch_all("survey_responses"),
        _fetch_all("question_answers"),
        _fetch_all("answer_selected_options"),
        _fetch_all("question_options"),
        _fetch_all("questions"),
        _fetch_all("schools"),
    )

# -------------------------------------------------
# Build answer-level DF (NO inner join with ASO)
# 1 row per question_answer (can have no selected option)
# -------------------------------------------------
def build_answer_level_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    if responses.empty or answers.empty or questions.empty:
        return pd.DataFrame()

    responses_sub = filter_submitted_responses(responses)
    if responses_sub.empty:
        return pd.DataFrame()

    base = (
        answers
        .merge(responses_sub, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id", suffixes=("", "_q"))
    )

    ans_id_col = "id_ans" if "id_ans" in base.columns else _pick_col(base, ["id_x", "id", "id_answer"])
    if ans_id_col is None:
        return pd.DataFrame()

    base = base.rename(columns={ans_id_col: "question_answer_id"})

    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        schools_small = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        base = base.merge(schools_small, left_on="school_id", right_on="school_pk", how="left")
        base["school_name"] = base["school_name"].fillna(base["school_id"].astype(str))
    else:
        base["school_name"] = base["school_id"].astype(str)

    if aso is not None and not aso.empty and {"question_answer_id", "option_id"}.issubset(aso.columns):
        base = base.merge(
            aso[["question_answer_id", "option_id"]],
            on="question_answer_id",
            how="left"
        )
    else:
        base["option_id"] = None

    if qopts is not None and not qopts.empty and "id" in qopts.columns:
        qopts_small = qopts.copy().rename(columns={"id": "option_pk"})
        base = base.merge(
            qopts_small,
            left_on="option_id",
            right_on="option_pk",
            how="left",
            suffixes=("", "_opt")
        )
    else:
        base["option_pk"] = None

    code_col = _pick_col(base, ["option_code", "code", "value"])
    text_col = _pick_col(base, ["option_text", "text", "label"])

    base["question_text"] = base.get("question_text", "").fillna("")
    # external_id (texto) para mapear preguntas del JSON ZERO-R
    base["question_external_id"] = base.get("external_id")

    base["opt_code"] = base[code_col].astype(str) if code_col else ""
    base["opt_text"] = base[text_col].astype(str) if text_col else ""

    base["opt_code"] = base["opt_code"].replace({"nan": "", "None": ""}).fillna("")
    base["opt_text"] = base["opt_text"].replace({"nan": "", "None": ""}).fillna("")

    # Score: prioriza opt_code (0..3 o A..D) y luego fallback por texto
    def _score_from_code_or_text(code: str, text: str):
        c = (code or "").strip()
        t = (text or "").strip()
        if c in {"0", "1", "2", "3"}:
            return float(int(c))
        if c in {"A", "B", "C", "D"}:
            return float({"A": 0, "B": 1, "C": 2, "D": 3}[c])
        return SCALE.get(t)

    base["score"] = base.apply(lambda r: _score_from_code_or_text(r.get("opt_code"), r.get("opt_text")), axis=1)
    base.loc[base["opt_text"].astype(str).str.strip().eq("") & base["opt_code"].astype(str).str.strip().eq(""), "score"] = np.nan
    base["score"] = pd.to_numeric(base["score"], errors="coerce")

    if "survey_response_id" not in base.columns:
        return pd.DataFrame()

    return base

# -------------------------------------------------
# Build selected-options DF for charts
# -------------------------------------------------
def build_selected_df(answer_level_df: pd.DataFrame) -> pd.DataFrame:
    """Return answer rows that have a selected option (single_choice).

    Important: For analysis we standardize question_id to the *external_id*
    (e.g., "zero_general_edad") when available, so that constructs and
    demographics use stable identifiers across survey versions.
    """
    if answer_level_df.empty:
        return pd.DataFrame()

    df = answer_level_df[answer_level_df["option_id"].notna()].copy()

    # Prefer stable external ids for analysis keys.
    if "question_external_id" in df.columns:
        df["question_id_uuid"] = df["question_id"]
        df["question_id"] = df["question_external_id"].fillna(df["question_id"])

    return df
def compute_student_metrics(answer_level_df: pd.DataFrame) -> pd.DataFrame:
    if answer_level_df.empty:
        return pd.DataFrame()

    d = answer_level_df.copy()

    # Preferimos external_id (texto) si existe (ZERO-R)
    qcol = "question_external_id" if "question_external_id" in d.columns else "question_id"

    victim_qids = [q for q in d[qcol].dropna().astype(str).unique().tolist() if re.search(VICTIM_REGEX, q)]

    if victim_qids:
        is_victim = d[qcol].astype(str).isin(victim_qids)
    else:
        is_victim = d["question_text"].str.contains(VICTIM_REGEX, case=False, na=False)

    cyber_qids = [q for q in d[qcol].dropna().astype(str).unique().tolist() if re.search(CYBER_REGEX, q)]

    if cyber_qids:
        is_cyber = d[qcol].astype(str).isin(cyber_qids)
    else:
        is_cyber = d["question_text"].str.contains(CYBER_REGEX, case=False, na=False)

    trust_qids = [q for q in d[qcol].dropna().astype(str).unique().tolist() if re.search(TRUST_REGEX, q)]

    if trust_qids:
        is_trust = d[qcol].astype(str).isin(trust_qids)
    else:
        is_trust = d["question_text"].str.contains(TRUST_REGEX, case=False, na=False)

    d_v = d[is_victim][["survey_response_id", "school_name", "score"]].copy()
    d_c = d[is_cyber][["survey_response_id", "score"]].copy()
    d_t = d[is_trust][["survey_response_id", "score"]].copy()

    victim_max = d_v.groupby("survey_response_id")["score"].max(min_count=1)
    cyber_max  = d_c.groupby("survey_response_id")["score"].max(min_count=1)
    trust_max  = d_t.groupby("survey_response_id")["score"].max(min_count=1)

    victim_answered = d_v.groupby("survey_response_id")["score"].apply(lambda s: s.notna().any())
    cyber_answered  = d_c.groupby("survey_response_id")["score"].apply(lambda s: s.notna().any())
    trust_answered  = d_t.groupby("survey_response_id")["score"].apply(lambda s: s.notna().any())

    base = (
        d.groupby("survey_response_id")
        .agg(
            school_name=("school_name", "first"),
            school_id=("school_id", "first"),
        )
        .reset_index()
        .set_index("survey_response_id")
    )

    base["victim_max"] = victim_max
    base["cyber_max"] = cyber_max
    base["trust_adult_max"] = trust_max
    base["victim_answered"] = victim_answered
    base["cyber_answered"] = cyber_answered
    base["trust_answered"] = trust_answered
    base = base.reset_index()

    base["victim_freq"] = np.where(base["victim_answered"], base["victim_max"] >= HIGH_FREQ_THRESHOLD, np.nan)
    base["cyber_freq"]  = np.where(base["cyber_answered"],  base["cyber_max"]  >= HIGH_FREQ_THRESHOLD, np.nan)

    base["victim_persist"] = np.where(base["victim_answered"], base["victim_max"] >= VERY_HIGH_FREQ, np.nan)
    base["cyber_persist"]  = np.where(base["cyber_answered"],  base["cyber_max"]  >= VERY_HIGH_FREQ, np.nan)

    victim_persist_bool = base["victim_persist"].fillna(False).astype(bool)
    cyber_persist_bool  = base["cyber_persist"].fillna(False).astype(bool)

    answered_any = (
        base["victim_answered"].fillna(False).astype(bool)
        | base["cyber_answered"].fillna(False).astype(bool)
    )

    base["any_persist"] = np.where(
        answered_any,
        (victim_persist_bool | cyber_persist_bool),
        np.nan
    )

    base["silence_flag_strict"] = np.where(
        (base["victim_freq"] == True) & (base["trust_answered"] == True) & (base["trust_adult_max"] == 0),
        True,
        np.where((base["victim_freq"] == True) & (base["trust_answered"] == True), False, np.nan)
    )

    base["silence_flag_missing"] = np.where(
        (base["victim_freq"] == True) & (base["trust_answered"] == False),
        True,
        np.where((base["victim_freq"] == True), False, np.nan)
    )

    answered_any2 = (
        base["victim_answered"].fillna(False).astype(bool)
        | base["cyber_answered"].fillna(False).astype(bool)
    )
    high = (base["victim_persist"].fillna(False).astype(bool) | base["cyber_persist"].fillna(False).astype(bool))
    med  = (base["victim_freq"].fillna(False).astype(bool)    | base["cyber_freq"].fillna(False).astype(bool))

    base["risk_group"] = np.select(
        [answered_any2 & high, answered_any2 & (~high) & med, answered_any2 & (~high) & (~med)],
        ["ALTO", "MEDIO", "BAJO"],
        default="SIN_DATOS"
    )

    return base

# -------------------------------------------------
# Demografía: desde opciones seleccionadas (selected_df)
# -------------------------------------------------
def extract_demographics_from_options(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    demo_rows = selected_df[selected_df["question_id"].isin(list(DEMOGRAPHIC_QIDS))].copy()
    if demo_rows.empty:
        return pd.DataFrame(columns=["survey_response_id", "school_name", "edad", "grado", "genero", "tiempo"])

    demo_rows["value_raw"] = demo_rows["opt_code"]
    demo_rows.loc[demo_rows["value_raw"].astype(str).str.strip().eq(""), "value_raw"] = demo_rows["opt_text"]

    pivot = demo_rows.pivot_table(
        index=["survey_response_id", "school_name"],
        columns="question_id",
        values="value_raw",
        aggfunc="first"
    )

    def col_series(qid: str) -> pd.Series:
        if qid in pivot.columns:
            return pivot[qid]
        return pd.Series([None] * len(pivot), index=pivot.index)

    edad_s   = col_series(QID_EDAD).apply(_to_int_safe)
    grado_s  = col_series(QID_GRADO).apply(_to_int_safe)
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

    for c in ["edad", "grado", "tiempo"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out

# -------------------------------------------------
# Subgroup prevalence helper (no ML, denominators correct)
# -------------------------------------------------
def subgroup_prevalence(
    student_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    indicator_col: str,
    group_col: str
) -> pd.DataFrame:
    if student_df.empty or demo_df.empty:
        return pd.DataFrame()

    merged = student_df.merge(demo_df[["survey_response_id", group_col]], on="survey_response_id", how="left")
    if group_col not in merged.columns:
        return pd.DataFrame()

    merged = merged[merged[group_col].notna()].copy()
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for g, sub in merged.groupby(group_col):
        s = sub[indicator_col]
        stats = _safe_bool_rate(s)
        cnts = _safe_bool_count(s)
        rows.append({
            group_col: g,
            "n_students": int(len(sub)),
            "n_with_data": int(stats["n_with_data"]),
            "n_true": int(cnts["n_true"]),
            "missing_pct": stats["missing_pct"],
            "pct_true": stats["pct"],
        })

    return pd.DataFrame(rows).sort_values(["pct_true", "n_students"], ascending=[False, False])

# -------------------------------------------------
# Charts: 40 gráficos por pregunta (Grado x Género)
# -------------------------------------------------
def render_40_question_charts(selected_df: pd.DataFrame, demo_df: pd.DataFrame, max_questions: int = 40):
    st.subheader(f"Gráficos por cada pregunta ({max_questions}) — Grado × Género")

    if demo_df.empty:
        st.warning("No hay demografía (edad/grado/género). Sin eso no se pueden construir los gráficos.")
        return

    demo_ok = demo_df.dropna(subset=["grado", "genero"]).copy()
    if demo_ok.empty:
        st.warning("Demografía existe, pero no hay filas con (grado y género) válidos.")
        return

    base = selected_df[~selected_df["question_id"].isin(list(DEMOGRAPHIC_QIDS))].copy()
    if base.empty:
        st.warning("No hay respuestas no-demográficas para graficar (opciones seleccionadas).")
        return

    base = base.merge(
        demo_ok[["survey_response_id", "grado", "genero"]],
        on="survey_response_id",
        how="inner"
    )

    q_list = (
        base[["question_id", "question_text"]]
        .drop_duplicates()
        .sort_values("question_text")
        .head(max_questions)
        .to_dict("records")
    )

    for qi, qrow in enumerate(q_list, start=1):
        qid = qrow["question_id"]
        qtext = (qrow["question_text"] or "").strip() or f"Pregunta {qi}"

        sub = base[base["question_id"] == qid].copy()
        if sub.empty:
            continue

        sub_unique = sub.drop_duplicates(subset=["survey_response_id", "grado", "genero"])

        counts = (
            sub_unique
            .groupby(["grado", "genero"])
            .size()
            .reset_index(name="n")
        )

        pivot = counts.pivot(index="grado", columns="genero", values="n").fillna(0).astype(int)

        for col in ["M", "F", "O", "N"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["M", "F", "O", "N"]]

        all_grades = pd.Index(range(6, 13), name="grado")
        pivot = pivot.reindex(all_grades, fill_value=0)

        st.markdown(f"### {qtext}")
        st.caption("Eje X: grado | Eje Y: número de estudiantes (que respondieron la pregunta), desagregado por género.")
        st.bar_chart(pivot)

# -------------------------------------------------
# Aggregated summary for LLM
# -------------------------------------------------
def build_school_summary(student_view: pd.DataFrame, demo_view: pd.DataFrame) -> dict:
    n = int(len(student_view))
    if n == 0:
        return {}

    victim_stats = _safe_bool_rate(student_view["victim_freq"])
    cyber_stats = _safe_bool_rate(student_view["cyber_freq"])
    victim_persist_stats = _safe_bool_rate(student_view["victim_persist"])
    cyber_persist_stats = _safe_bool_rate(student_view["cyber_persist"])
    any_persist_stats = _safe_bool_rate(student_view["any_persist"])
    silence_strict_stats = _safe_bool_rate(student_view["silence_flag_strict"])
    silence_missing_stats = _safe_bool_rate(student_view["silence_flag_missing"])

    risk_dist = student_view["risk_group"].value_counts(dropna=False).to_dict()

    summary = {
        "n_estudiantes_submitted": n,
        "n_encuestas_submitted_total": int(len(responses)) if "responses" in globals() else int(n),
        "n_estudiantes_con_datos_relevantes": int(len(student_view)),
        "victimizacion_mensual": {**victim_stats, "threshold_flag": threshold_flag(victim_stats["pct"])},
        "cyberbullying_mensual": {**cyber_stats, "threshold_flag": threshold_flag(cyber_stats["pct"])},
        "persistencia_victimizacion_semanal": {**victim_persist_stats, "threshold_flag": threshold_flag(victim_persist_stats["pct"])},
        "persistencia_cyber_semanal": {**cyber_persist_stats, "threshold_flag": threshold_flag(cyber_persist_stats["pct"])},
        "persistencia_cualquier_semanal": {**any_persist_stats, "threshold_flag": threshold_flag(any_persist_stats["pct"])},
        "silencio_institucional_strict": {**silence_strict_stats, "threshold_flag": threshold_flag(silence_strict_stats["pct"])},
        "silencio_institucional_missing_trust": {
            **silence_missing_stats,
            "note": "Bandera de medición: victimización frecuente con confianza en adultos sin respuesta (posible subreporte o falta de seguridad).",
        },
        "distribucion_riesgo_estudiante": risk_dist,
    }

    if demo_view is not None and not demo_view.empty:
        if "genero" in demo_view.columns:
            by_gender_victim = subgroup_prevalence(student_view, demo_view, "victim_freq", "genero")
            if not by_gender_victim.empty:
                summary["subgrupos_genero_victim_freq"] = by_gender_victim.head(10).to_dict("records")

        if "edad" in demo_view.columns:
            demo_age = demo_view.copy()
            if demo_age["edad"].notna().sum() > 0:
                demo_age["edad_bucket"] = demo_age["edad"].astype("Int64").astype(str)
                by_age_cyber = subgroup_prevalence(student_view, demo_age, "cyber_freq", "edad_bucket")
                if not by_age_cyber.empty:
                    summary["subgrupos_edad_cyber_freq"] = by_age_cyber.head(10).to_dict("records")

        if "grado" in demo_view.columns:
            demo_grade = demo_view.copy()
            demo_grade["grado_bucket"] = demo_grade["grado"]
            by_grade_victim = subgroup_prevalence(student_view, demo_grade, "victim_freq", "grado_bucket")
            if not by_grade_victim.empty:
                summary["subgrupos_grado_victim_freq"] = by_grade_victim.head(13).to_dict("records")

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

    # IMPORTANT: Prompt must include required formatting variables
    prompt = LLM_PROMPT_ES.format(
        summary=summary,
        n_encuestas_submitted_total=summary.get(
            "n_encuestas_submitted_total",
            summary.get("n_estudiantes_submitted", 0),
        ),
        n_estudiantes_con_datos_relevantes=summary.get(
            "n_estudiantes_con_datos_relevantes",
            summary.get("n_estudiantes_submitted", 0),
        ),
    )

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
# UI  (UPDATED: visible-first dashboard)
# -------------------------------------------------
st.title("Dashboard de Convivencia Escolar — Indicadores Científicos")

# Read query params (school_id, analysis_dt)
qp = st.query_params
qp_school_id = (qp.get("school_id") or "").strip()
qp_analysis_dt = (qp.get("analysis_dt") or "").strip()

if qp_school_id and qp_analysis_dt:
    responses, answers, aso, qopts, questions, schools = load_tables_filtered(qp_school_id, qp_analysis_dt)
else:
    responses, answers, aso, qopts, questions, schools = load_tables_unfiltered()

answer_level = build_answer_level_df(responses, answers, aso, qopts, questions, schools)
if answer_level.empty:
    st.warning("Aún no hay datos suficientes para análisis (o no hay encuestas con status=submitted).")
    st.stop()

selected_df = build_selected_df(answer_level)
student = compute_student_metrics(answer_level)

if student.empty:
    st.warning("No se pudieron construir métricas de estudiantes.")
    st.stop()

# If filtered by URL, lock the view to that school (no '(Todas)')
if qp_school_id and qp_analysis_dt:
    school_list = sorted(student["school_name"].dropna().unique().tolist())
    school = school_list[0] if school_list else "(Todas)"
    st.sidebar.info(f"Filtrado por school_id={qp_school_id} y analysis_dt={qp_analysis_dt}")
else:
    school_list = sorted(student["school_name"].dropna().unique().tolist())
    school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view = student if school == "(Todas)" else student[student["school_name"] == school]

demo_df = extract_demographics_from_options(selected_df)
demo_view = demo_df if school == "(Todas)" else demo_df[demo_df["school_name"] == school].copy()

# -----------------------------
# A) KPIs (prevalencia + conteo)
# -----------------------------
st.subheader("Panel principal (prevalencia + conteos)")

vict = _safe_bool_rate(view["victim_freq"]); vict_c = _safe_bool_count(view["victim_freq"])
pers = _safe_bool_rate(view["any_persist"]);  pers_c = _safe_bool_count(view["any_persist"])
cybr = _safe_bool_rate(view["cyber_freq"]);  cybr_c = _safe_bool_count(view["cyber_freq"])
sil  = _safe_bool_rate(view["silence_flag_strict"]); sil_c = _safe_bool_count(view["silence_flag_strict"])

def _fmt_pct(p):
    return f"{p:.1f}%" if p is not None and not (isinstance(p, float) and np.isnan(p)) else "NA"

def _fmt_count(n_true, n_with):
    return f"{n_true}/{n_with}" if n_with and n_with > 0 else "0/0"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes (submitted)", f"{len(view)}")
c2.metric("Victimización frecuente (≥ mensual)", _fmt_pct(vict["pct"]), _fmt_count(vict_c["n_true"], vict_c["n_with_data"]))
c3.metric("Persistencia alta (≥ semanal)", _fmt_pct(pers["pct"]), _fmt_count(pers_c["n_true"], pers_c["n_with_data"]))
c4.metric("Silencio institucional (strict)", _fmt_pct(sil["pct"]), _fmt_count(sil_c["n_true"], sil_c["n_with_data"]))

st.caption(
    "Nota metodológica: los valores 'NA' o 'missing' indican falta de respuesta (no se convierten a 0). "
    "Cero reportes no prueba ausencia; puede existir subreporte."
)

# -----------------------------
# B) Semáforo (5/10/20) por victimización frecuente
# -----------------------------
st.divider()
lvl, msg = risk_level_message(vict["pct"])
if lvl == "CRISIS":
    st.error(f"Semáforo (victimización ≥ mensual): {lvl} — {msg}")
elif lvl == "INTERVENCION":
    st.warning(f"Semáforo (victimización ≥ mensual): {lvl} — {msg}")
elif lvl == "ATENCION":
    st.info(f"Semáforo (victimización ≥ mensual): {lvl} — {msg}")
else:
    st.success(f"Semáforo (victimización ≥ mensual): {lvl} — {msg}")

# -----------------------------
# C) Distribución riesgo estudiante
# -----------------------------
st.divider()
st.subheader("Distribución de riesgo (por estudiante)")
risk_counts = view["risk_group"].value_counts().reindex(["ALTO", "MEDIO", "BAJO", "SIN_DATOS"], fill_value=0)
st.bar_chart(risk_counts)

# -----------------------------
# D) Detalles: denominadores + missing + banderas
# -----------------------------
with st.expander("Detalles (denominadores, missing, banderas 5/10/20)", expanded=False):
    cols = st.columns(2)

    def metric_block(title: str, stats: dict):
        pct = stats["pct"]
        st.markdown(f"**{title}**")
        st.write({
            "pct": pct,
            "n_with_data": stats["n_with_data"],
            "n_missing": stats["n_missing"],
            "missing_pct": stats["missing_pct"],
            "threshold_flag": threshold_flag(pct),
        })

    with cols[0]:
        metric_block("Victimización frecuente (≥ mensual)", vict)
        metric_block("Cyberbullying frecuente (≥ mensual)", cybr)
        metric_block("Persistencia (victimización ≥ semanal)", _safe_bool_rate(view["victim_persist"]))
        metric_block("Persistencia (cyber ≥ semanal)", _safe_bool_rate(view["cyber_persist"]))

    with cols[1]:
        metric_block("Persistencia (cualquiera ≥ semanal)", pers)
        metric_block("Silencio institucional (strict)", sil)
        st.markdown("**Silencio institucional (missing confianza)**")
        st.write(_safe_bool_rate(view["silence_flag_missing"]))

# -----------------------------
# E) Subgrupos mínimos (sin ML)
# -----------------------------
st.divider()
st.subheader("Subgrupos mínimos (sin ML): señales concentradas")

if demo_view.empty:
    st.info("No hay demografía disponible (edad/grado/género) para esta selección.")
else:
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Género × Victimización (≥ mensual)")
        if "genero" in demo_view.columns:
            t = subgroup_prevalence(view, demo_view, "victim_freq", "genero")
            st.dataframe(t if not t.empty else pd.DataFrame(), use_container_width=True)

        st.markdown("### Género × Cyberbullying (≥ mensual)")
        if "genero" in demo_view.columns:
            t = subgroup_prevalence(view, demo_view, "cyber_freq", "genero")
            st.dataframe(t if not t.empty else pd.DataFrame(), use_container_width=True)

    with colB:
        st.markdown("### Grado × Victimización (≥ mensual)")
        if "grado" in demo_view.columns:
            dg = demo_view.copy()
            dg["grado_bucket"] = dg["grado"]
            t = subgroup_prevalence(view, dg, "victim_freq", "grado_bucket")
            st.dataframe(t if not t.empty else pd.DataFrame(), use_container_width=True)

        st.markdown("### Grado × Cyberbullying (≥ mensual)")
        if "grado" in demo_view.columns:
            dg = demo_view.copy()
            dg["grado_bucket"] = dg["grado"]
            t = subgroup_prevalence(view, dg, "cyber_freq", "grado_bucket")
            st.dataframe(t if not t.empty else pd.DataFrame(), use_container_width=True)

# -----------------------------
# F) 40 charts (use selected_df for charts; filter by school)
# -----------------------------
st.divider()
selected_view = selected_df if school == "(Todas)" else selected_df[selected_df["school_name"] == school].copy()
render_40_question_charts(selected_view, demo_view, max_questions=40)

# -----------------------------
# G) Informe IA
# -----------------------------
st.divider()
st.subheader("Informe interpretativo (IA)")
if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe..."):
        summary = build_school_summary(view, demo_view)
        report = generate_llm_report(summary)
        st.markdown(report)

# -----------------------------
# Debug
# -----------------------------
with st.expander("Debug (recomendado ahora)"):
    st.write("Query params:", dict(st.query_params))
    st.write("Rows responses:", len(responses))
    st.write("Rows answers:", len(answers))
    st.write("Rows aso:", len(aso))
    st.write("Rows qopts:", len(qopts))
    st.write("Rows questions:", len(questions))
    st.write("Rows schools:", len(schools))
    st.write("Rows answer_level:", len(answer_level))
    st.write("Rows selected_df:", len(selected_df))
    st.write("Rows student:", len(student))
    st.write("Rows demo_df:", len(demo_df))
    st.write("Ejemplos question_text (head):")
    st.dataframe(answer_level[["question_id", "question_text"]].drop_duplicates().head(30), use_container_width=True)
    st.write("Constructs config (victim/cyber/trust) — recomendado completar con question_id:")

