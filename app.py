# app.py — Bullying Dashboard v2.1
# ═══════════════════════════════════════════════════════════════════════════════
# ALIGNED TO: TECH4ZERO-MX v1.0 / survey_002_v3.json
#
# STATISTICAL IMPROVEMENTS OVER v1:
#   1. Sum scores (not max) for construct aggregation — reduces false positives
#   2. Cronbach's alpha per construct — reliability check before reporting
#   3. Wilson 95% CI on all prevalence estimates
#   4. Chi-square + Cramér's V for subgroup comparisons (gender, grade)
#   5. Bonferroni correction for multiple comparisons
#   6. Item-level descriptives (mean, SD, % high-freq per item)
#   7. Missing data pattern analysis
#
# LLM:
#   - Claude claude-sonnet-4-5-20250929 via Anthropic API
#   - Structured JSON input (not raw dict)
#   - Numerical fidelity rules + CI reporting in prompt
#   - max_tokens 2048
#
# SURVEY ALIGNMENT (v2.1):
#   - CONSTRUCTS dict updated to exact external_ids from survey_002_v3
#   - Scale updated to 0-4 (frequency_0_4 + likert_0_4) — no more 0-3
#   - cyberbullying split into cybervictimizacion + cyberagresion
#   - Global screener items (zero_victim_general, zero_agresor_general)
#     excluded from Cronbach alpha
#   - Ecology section treated as conditional (missing ≠ zero)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from scipy import stats as scipy_stats
from supabase import create_client

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Convivencia Escolar", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Supabase
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Demographic question external_ids (survey_002_v3)
# ─────────────────────────────────────────────────────────────────────────────
QID_CURSO       = "zero_general_curso"
QID_EDAD        = "zero_general_edad"
QID_GENERO      = "zero_general_genero"
QID_LENGUA      = "zero_general_lengua"
QID_ORIENTACION = "zero_general_orientacion"
QID_TIEMPO      = "zero_general_tiempo"
QID_TIPO_ESC    = "zero_general_tipo_escuela"

DEMOGRAPHIC_QIDS = {
    QID_CURSO, QID_EDAD, QID_GENERO, QID_LENGUA,
    QID_ORIENTACION, QID_TIEMPO, QID_TIPO_ESC,
}

# ─────────────────────────────────────────────────────────────────────────────
# Scoring — unified 0-4 scale (TECH4ZERO-MX v1.0)
# ─────────────────────────────────────────────────────────────────────────────
SCALE = {
    # Numeric codes (primary)
    "0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0,
    # Frequency text (Sección D/E — OBVQ-R)
    "Nunca en los últimos 2 meses":   0.0,
    "Ha pasado 1 o 2 veces":          1.0,
    "2 o 3 veces al mes":             2.0,
    "Más o menos 1 vez por semana":   3.0,
    "Varias veces por semana":        4.0,
    # Likert text (Secciones B, C, F, G, H, I)
    "Nunca":       0.0,
    "Casi nunca":  1.0,
    "A veces":     2.0,
    "A menudo":    3.0,
    "Siempre":     4.0,
    # Legacy 0-3 backward compat
    "1 o 2 veces":    1.0,
    "3 a 5 veces":    2.0,
    "Más de 5 veces": 3.0,
    "Mas de 5 veces": 3.0,
}

HIGH_FREQ_THRESHOLD = 2   # monthly-or-more (score >= 2)
VERY_HIGH_FREQ      = 3   # weekly-or-more  (score >= 3)
MIN_ITEMS_FOR_ALPHA = 3
ALPHA_THRESHOLD     = 0.60
ALPHA_STAT          = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCT DEFINITIONS — exact external_ids from survey_002_v3.json
# Global screener items excluded from alpha calculation.
# ─────────────────────────────────────────────────────────────────────────────
CONSTRUCTS = {
    "victimizacion": frozenset({
        "zero_victima_agresion_fisica",
        "zero_victima_amenazas",
        "zero_victima_insultos",
        "zero_victima_rumores",
        "zero_victima_exclusion",
        "zero_victima_robo_danos",
        "zero_victima_coercion",
        "zero_victima_discriminacion",
        # zero_victim_general excluded (global screener)
    }),
    "perpetracion": frozenset({
        "zero_agresor_agresion_fisica",
        "zero_agresor_amenazas",
        "zero_agresor_insultos",
        "zero_agresor_rumores",
        "zero_agresor_exclusion",
        "zero_agresor_robo_danos",
        "zero_agresor_coercion",
        "zero_agresor_discriminacion",
        # zero_agresor_general excluded (global screener)
    }),
    "cybervictimizacion": frozenset({
        "zero_cyber_victima_mensajes",
        "zero_cyber_victima_fotos",
        "zero_cyber_victima_visto",
    }),
    "cyberagresion": frozenset({
        "zero_cyber_agresor_mensajes",
        "zero_cyber_agresor_exclusion",
        "zero_cyber_agresor_visto",
    }),
    "clima_docente": frozenset({
        "zero_clima_normas",
        "zero_clima_interviene",
        "zero_clima_bienestar",
        "zero_clima_seguimiento",
    }),
    "normas_grupo": frozenset({
        "zero_normas_reaccion",
        "zero_normas_popularidad",
        "zero_normas_defensa",
        "zero_normas_intervencion_activa",
    }),
    "ecologia_espacios": frozenset({
        "zero_ecologia_aula",
        "zero_ecologia_patio",
        "zero_ecologia_pasillos_banos",
        "zero_ecologia_entrada_salida",
        "zero_ecologia_transporte",
        "zero_ecologia_casa_companeros",
        "zero_ecologia_cafeteria",
        "zero_ecologia_online",
    }),
    "apoyo_institucional": frozenset({
        "zero_apoyo_reporte_efectivo",
        "zero_apoyo_resolucion",
        "zero_apoyo_orientador",
        "zero_apoyo_conoce_recurso",
    }),
    "impacto": frozenset({
        "zero_impacto_rendimiento",
        "zero_impacto_emocional",
        "zero_impacto_evitar_escuela",
        "zero_impacto_buscar_ayuda",
    }),
}

GLOBAL_SCREENERS = {"zero_victim_general", "zero_agresor_general"}

# ─────────────────────────────────────────────────────────────────────────────
# Prevalence thresholds
# ─────────────────────────────────────────────────────────────────────────────
def threshold_flag(pct: float) -> str:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "SIN_DATOS"
    if pct >= 20: return "CRISIS"
    if pct >= 10: return "INTERVENCION"
    if pct >= 5:  return "ATENCION"
    return "MONITOREO"


def risk_level_message(pct: float) -> tuple[str, str]:
    lvl = threshold_flag(pct)
    messages = {
        "CRISIS":       "≥20% reporta victimización frecuente (≥ mensual). Requiere acción inmediata.",
        "INTERVENCION": "≥10% reporta victimización frecuente (≥ mensual). Requiere intervención estructurada.",
        "ATENCION":     "≥5% reporta victimización frecuente (≥ mensual). Requiere monitoreo y acciones preventivas.",
        "MONITOREO":    "<5% reporta victimización frecuente (≥ mensual). Mantener monitoreo preventivo.",
        "SIN_DATOS":    "Sin datos suficientes para clasificar el nivel.",
    }
    return lvl, messages[lvl]


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL TOOLKIT
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    if n_total == 0:
        return (np.nan, np.nan)
    z      = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat  = n_success / n_total
    denom  = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
    return (
        round(max(0.0, center - spread) * 100, 1),
        round(min(1.0, center + spread) * 100, 1),
    )


def cronbach_alpha(item_matrix: pd.DataFrame) -> float:
    df = item_matrix.dropna()
    if df.shape[0] < 2 or df.shape[1] < MIN_ITEMS_FOR_ALPHA:
        return np.nan
    k         = df.shape[1]
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return round(float((k / (k - 1)) * (1 - item_vars.sum() / total_var)), 3)


def chi2_cramer(group_series: pd.Series, binary_series: pd.Series) -> dict:
    df = pd.DataFrame({"group": group_series, "outcome": binary_series}).dropna()
    if df.empty or df["group"].nunique() < 2:
        return {"chi2": np.nan, "p_value": np.nan, "cramers_v": np.nan, "n": 0, "significant": False}
    ct = pd.crosstab(df["group"], df["outcome"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "cramers_v": np.nan, "n": int(len(df)), "significant": False}
    chi2_val, p_val, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
    n         = int(ct.values.sum())
    min_dim   = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2_val / (n * max(min_dim, 1)))) if n > 0 else np.nan
    return {
        "chi2":       round(float(chi2_val), 3),
        "p_value":    round(float(p_val), 4),
        "cramers_v":  round(cramers_v, 3),
        "n":          n,
        "significant": bool(p_val < ALPHA_STAT),
    }


def bonferroni_threshold(n_tests: int) -> float:
    return ALPHA_STAT / max(n_tests, 1)


def item_descriptives(answer_level_df: pd.DataFrame, construct_ids: frozenset) -> pd.DataFrame:
    qcol  = "question_external_id" if "question_external_id" in answer_level_df.columns else "question_id"
    mask  = answer_level_df[qcol].fillna("").astype(str).isin(construct_ids)
    items = answer_level_df[mask][["question_id", "question_text", "score"]].copy()
    if items.empty:
        return pd.DataFrame()
    rows = []
    for (qid, qtext), grp in items.groupby(["question_id", "question_text"]):
        s     = grp["score"].dropna()
        n_ans = int(len(s))
        rows.append({
            "question_text": (str(qtext) or "")[:65],
            "mean":          round(float(s.mean()), 2) if n_ans > 0 else np.nan,
            "sd":            round(float(s.std(ddof=1)), 2) if n_ans > 1 else np.nan,
            "pct_high_freq": round(float((s >= HIGH_FREQ_THRESHOLD).mean() * 100), 1) if n_ans > 0 else np.nan,
            "n_answered":    n_ans,
            "n_missing":     int(grp["score"].isna().sum()),
        })
    return pd.DataFrame(rows).sort_values("pct_high_freq", ascending=False)


def missing_pattern_summary(student_df: pd.DataFrame) -> dict:
    cols    = [c for c in student_df.columns if c.endswith("_freq") or c.endswith("_sum")]
    n_total = int(len(student_df))
    out     = {}
    for col in cols:
        n_miss = int(student_df[col].isna().sum())
        out[col] = {
            "n_missing":   n_miss,
            "n_total":     n_total,
            "missing_pct": round(100 * n_miss / n_total, 1) if n_total else np.nan,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_gender(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s  = str(v).strip()
    sl = s.lower()
    if s.upper() in {"M", "F", "O", "N"}: return s.upper()
    if "masc" in sl or "hombre" in sl:    return "M"
    if "fem"  in sl or "mujer"  in sl:    return "F"
    if "binar" in sl or "otro"  in sl:    return "O"
    if "prefer" in sl or "no resp" in sl: return "N"
    return None


def _to_int_safe(v) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def _safe_bool_rate(series: pd.Series) -> dict:
    empty = {"pct": np.nan, "n_with_data": 0, "n_missing": 0,
             "missing_pct": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    if series is None or len(series) == 0:
        return empty
    s         = series.copy()
    n_total   = int(len(s))
    n_missing = int(s.isna().sum())
    n_with    = n_total - n_missing
    if n_with == 0:
        return {**empty, "n_missing": n_missing,
                "missing_pct": round(100 * n_missing / n_total, 1) if n_total else np.nan}
    s_bool  = s.dropna().astype(bool)
    n_true  = int(s_bool.sum())
    pct     = float(s_bool.mean() * 100.0)
    ci_lo, ci_hi = wilson_ci(n_true, n_with)
    return {
        "pct":         round(pct, 1),
        "n_with_data": n_with,
        "n_missing":   n_missing,
        "missing_pct": round(100 * n_missing / n_total, 1) if n_total else np.nan,
        "ci_lower":    ci_lo,
        "ci_upper":    ci_hi,
    }


def _safe_bool_count(series: pd.Series) -> dict:
    if series is None or len(series) == 0:
        return {"n_true": 0, "n_with_data": 0, "n_missing": 0}
    s         = series.copy()
    n_missing = int(s.isna().sum())
    s2        = s.dropna().astype(bool)
    return {"n_true": int(s2.sum()), "n_with_data": int(len(s2)), "n_missing": n_missing}


def _fmt_pct_ci(stats: dict) -> str:
    p = stats.get("pct")
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    lo, hi = stats.get("ci_lower", np.nan), stats.get("ci_upper", np.nan)
    return f"{p:.1f}% (IC95%: {lo}–{hi})"


# ─────────────────────────────────────────────────────────────────────────────
# Filter: submitted only
# ─────────────────────────────────────────────────────────────────────────────
def filter_submitted(responses: pd.DataFrame) -> pd.DataFrame:
    if responses.empty or "status" not in responses.columns:
        return responses
    return responses[responses["status"].astype(str).str.lower().eq("submitted")].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Supabase fetch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_in_chunks(table: str, col: str, ids: list,
                     select: str = "*", chunk_size: int = 200) -> list[dict]:
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), chunk_size):
        rows = (
            supabase.table(table).select(select)
            .in_(col, ids[i:i + chunk_size])
            .execute().data
        )
        if rows:
            out.extend(rows)
    return out


@st.cache_data(ttl=300)
def load_tables_filtered(school_id: str, analysis_dt: str):
    responses_rows = (
        supabase.table("survey_responses").select("*")
        .eq("school_id", school_id)
        .eq("analysis_requested_dt", analysis_dt)
        .execute().data
    )
    responses = pd.DataFrame(responses_rows or [])
    if responses.empty:
        return (pd.DataFrame(),) * 6

    resp_ids  = responses["id"].dropna().tolist()
    answers   = pd.DataFrame(_fetch_in_chunks("question_answers", "survey_response_id", resp_ids))
    if answers.empty:
        return (responses,) + (pd.DataFrame(),) * 5

    ans_ids   = answers["id"].dropna().tolist()
    aso       = pd.DataFrame(_fetch_in_chunks("answer_selected_options", "question_answer_id", ans_ids))
    opt_ids   = aso["option_id"].dropna().tolist() if (not aso.empty and "option_id" in aso.columns) else []
    qopts     = pd.DataFrame(_fetch_in_chunks("question_options", "id", opt_ids))
    qids      = answers["question_id"].dropna().tolist()
    questions = pd.DataFrame(_fetch_in_chunks("questions", "id", qids))
    schools   = pd.DataFrame(
        supabase.table("schools").select("*").eq("id", school_id).execute().data or []
    )
    return responses, answers, aso, qopts, questions, schools


@st.cache_data(ttl=300)
def _fetch_all(table: str, batch: int = 1000) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        data = (
            supabase.table(table).select("*")
            .range(start, start + batch - 1).execute().data
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


# ─────────────────────────────────────────────────────────────────────────────
# Build answer-level DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def build_answer_level_df(responses, answers, aso, qopts, questions, schools) -> pd.DataFrame:
    if responses.empty or answers.empty or questions.empty:
        return pd.DataFrame()

    responses_sub = filter_submitted(responses)
    if responses_sub.empty:
        return pd.DataFrame()

    base = (
        answers
        .merge(responses_sub, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions,     left_on="question_id",        right_on="id", suffixes=("", "_q"))
    )

    ans_id_col = "id_ans" if "id_ans" in base.columns else _pick_col(base, ["id_x", "id", "id_answer"])
    if ans_id_col is None:
        return pd.DataFrame()
    base = base.rename(columns={ans_id_col: "question_answer_id"})

    if not schools.empty and {"id", "name"}.issubset(schools.columns):
        s_sm = schools[["id", "name"]].rename(columns={"id": "school_pk", "name": "school_name"})
        base = base.merge(s_sm, left_on="school_id", right_on="school_pk", how="left")
        base["school_name"] = base["school_name"].fillna(base["school_id"].astype(str))
    else:
        base["school_name"] = base["school_id"].astype(str)

    if aso is not None and not aso.empty and {"question_answer_id", "option_id"}.issubset(aso.columns):
        base = base.merge(aso[["question_answer_id", "option_id"]],
                          on="question_answer_id", how="left")
    else:
        base["option_id"] = None

    if qopts is not None and not qopts.empty and "id" in qopts.columns:
        qo = qopts.rename(columns={"id": "option_pk"})
        base = base.merge(qo, left_on="option_id", right_on="option_pk",
                          how="left", suffixes=("", "_opt"))

    code_col = _pick_col(base, ["option_code", "code", "value"])
    text_col = _pick_col(base, ["option_text", "text", "label"])

    base["question_text"]        = base.get("question_text", "").fillna("")
    base["question_external_id"] = base.get("external_id")

    base["opt_code"] = (base[code_col].astype(str) if code_col else "").replace({"nan": "", "None": ""}).fillna("")
    base["opt_text"] = (base[text_col].astype(str) if text_col else "").replace({"nan": "", "None": ""}).fillna("")

    def _score(code: str, text: str):
        c = (code or "").strip()
        t = (text or "").strip()
        if c in {"0", "1", "2", "3", "4"}:
            return float(int(c))
        return SCALE.get(t)

    base["score"] = base.apply(lambda r: _score(r.get("opt_code", ""), r.get("opt_text", "")), axis=1)
    base.loc[
        base["opt_text"].str.strip().eq("") & base["opt_code"].str.strip().eq(""),
        "score"
    ] = np.nan
    base["score"] = pd.to_numeric(base["score"], errors="coerce")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Build selected options DF (for charts)
# ─────────────────────────────────────────────────────────────────────────────
def build_selected_df(answer_level_df: pd.DataFrame) -> pd.DataFrame:
    if answer_level_df.empty:
        return pd.DataFrame()
    df = answer_level_df[answer_level_df["option_id"].notna()].copy()
    if "question_external_id" in df.columns:
        df["question_id_uuid"] = df["question_id"]
        df["question_id"]      = df["question_external_id"].fillna(df["question_id"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Compute student metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_student_metrics(answer_level_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if answer_level_df.empty:
        return pd.DataFrame(), {}

    d    = answer_level_df.copy()
    qcol = "question_external_id" if "question_external_id" in d.columns else "question_id"

    construct_sums = {}
    reliability    = {}

    for name, id_set in CONSTRUCTS.items():
        mask  = d[qcol].fillna("").astype(str).isin(id_set)
        items = d[mask][["survey_response_id", "question_id", "score"]].copy()

        if items.empty:
            construct_sums[name] = pd.Series(dtype=float)
            reliability[name]    = {"alpha": np.nan, "n_items": 0, "reliable": False}
            continue

        pivot    = items.pivot_table(index="survey_response_id", columns="question_id",
                                     values="score", aggfunc="mean")
        n_items  = pivot.shape[1]
        alpha    = cronbach_alpha(pivot)
        reliable = (not np.isnan(alpha)) and (alpha >= ALPHA_THRESHOLD)

        answered = pivot.notna().sum(axis=1)
        row_sum  = pivot.sum(axis=1, min_count=1)
        row_sum[answered < 1] = np.nan

        construct_sums[name] = row_sum
        reliability[name]    = {"alpha": alpha, "n_items": n_items, "reliable": reliable}

    for screener_id in GLOBAL_SCREENERS:
        mask = d[qcol].fillna("").astype(str).eq(screener_id)
        if mask.any():
            construct_sums[screener_id] = d[mask].groupby("survey_response_id")["score"].mean()

    base = (
        d.groupby("survey_response_id")
        .agg(school_name=("school_name", "first"), school_id=("school_id", "first"))
        .reset_index().set_index("survey_response_id")
    )

    for name, series in construct_sums.items():
        base[f"{name}_sum"] = series

    base = base.reset_index()

    for construct in ["victimizacion", "cybervictimizacion"]:
        sum_col = f"{construct}_sum"
        id_set  = CONSTRUCTS.get(construct, frozenset())
        mask    = d[qcol].fillna("").astype(str).isin(id_set)
        items   = d[mask][["survey_response_id", "score"]].copy()
        n_ans   = items.groupby("survey_response_id")["score"].apply(lambda s: s.notna().sum())
        base    = base.merge(n_ans.rename(f"_{construct}_n_ans"), on="survey_response_id", how="left")

        n_col    = f"_{construct}_n_ans"
        mean_col = f"{construct}_mean"
        base[mean_col] = np.where(
            base[n_col].fillna(0) > 0,
            base[sum_col] / base[n_col],
            np.nan
        )
        answered = base[n_col].fillna(0) > 0
        base[f"{construct}_freq"]    = np.where(answered, base[mean_col] >= HIGH_FREQ_THRESHOLD, np.nan)
        base[f"{construct}_persist"] = np.where(answered, base[mean_col] >= VERY_HIGH_FREQ,      np.nan)

    base["victim_freq"]     = base.get("victimizacion_freq")
    base["victim_persist"]  = base.get("victimizacion_persist")
    base["cyber_freq"]      = base.get("cybervictimizacion_freq")
    base["cyber_persist"]   = base.get("cybervictimizacion_persist")
    base["trust_adult_sum"] = base.get("apoyo_institucional_sum")

    has_trust = base["trust_adult_sum"].notna()
    base["silence_flag_strict"] = np.where(
        (base["victim_freq"] == True) & has_trust & (base["trust_adult_sum"] == 0),
        True,
        np.where((base["victim_freq"] == True) & has_trust, False, np.nan)
    )
    base["silence_flag_missing"] = np.where(
        (base["victim_freq"] == True) & ~has_trust, True,
        np.where(base["victim_freq"] == True, False, np.nan)
    )

    answered_any = base["victim_freq"].notna() | base["cyber_freq"].notna()
    high = (base["victim_persist"].fillna(False).astype(bool) |
            base["cyber_persist"].fillna(False).astype(bool))
    med  = (base["victim_freq"].fillna(False).astype(bool) |
            base["cyber_freq"].fillna(False).astype(bool))

    base["any_persist"] = np.where(answered_any, high, np.nan)
    base["risk_group"]  = np.select(
        [answered_any & high,
         answered_any & (~high) & med,
         answered_any & (~high) & (~med)],
        ["ALTO", "MEDIO", "BAJO"],
        default="SIN_DATOS"
    )

    return base, reliability


# ─────────────────────────────────────────────────────────────────────────────
# Demographics
# ─────────────────────────────────────────────────────────────────────────────
def extract_demographics(selected_df: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["survey_response_id", "school_name",
                                   "curso", "edad", "genero", "tipo_escuela"])
    if selected_df.empty:
        return empty

    key_col   = "question_external_id" if "question_external_id" in selected_df.columns else "question_id"
    demo_rows = selected_df[selected_df[key_col].isin(DEMOGRAPHIC_QIDS)].copy()
    if demo_rows.empty:
        return empty

    demo_rows["value_raw"] = demo_rows["opt_code"]
    demo_rows.loc[demo_rows["value_raw"].str.strip().eq(""), "value_raw"] = demo_rows["opt_text"]

    pivot = demo_rows.pivot_table(
        index=["survey_response_id", "school_name"],
        columns="question_id", values="value_raw", aggfunc="first"
    )

    def col_s(qid):
        return pivot[qid] if qid in pivot.columns else pd.Series([None] * len(pivot), index=pivot.index)

    out = pd.DataFrame({
        "survey_response_id": [i[0] for i in pivot.index],
        "school_name":        [i[1] for i in pivot.index],
        "curso":              col_s(QID_CURSO).apply(_to_int_safe).values,
        "edad":               col_s(QID_EDAD).apply(_to_int_safe).values,
        "genero":             col_s(QID_GENERO).apply(_norm_gender).values,
        "tipo_escuela":       col_s(QID_TIPO_ESC).values,
    })
    for c in ["curso", "edad"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Subgroup prevalence
# ─────────────────────────────────────────────────────────────────────────────
def subgroup_prevalence(
    student_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    indicator_col: str,
    group_col: str,
    n_total_tests: int = 1,
) -> pd.DataFrame:
    if student_df.empty or demo_df.empty:
        return pd.DataFrame()

    merged = (
        student_df
        .merge(demo_df[["survey_response_id", group_col]], on="survey_response_id", how="left")
        .pipe(lambda df: df[df[group_col].notna()])
    )
    if merged.empty:
        return pd.DataFrame()

    chi2_res   = chi2_cramer(merged[group_col], merged[indicator_col])
    alpha_bonf = bonferroni_threshold(n_total_tests)

    rows = []
    for g, sub in merged.groupby(group_col):
        s     = sub[indicator_col]
        stats = _safe_bool_rate(s)
        cnts  = _safe_bool_count(s)
        rows.append({
            group_col:     g,
            "n_students":  int(len(sub)),
            "n_with_data": stats["n_with_data"],
            "n_true":      cnts["n_true"],
            "missing_pct": stats["missing_pct"],
            "pct_true":    stats["pct"],
            "ci_lower":    stats["ci_lower"],
            "ci_upper":    stats["ci_upper"],
        })

    df_out = pd.DataFrame(rows).sort_values(["pct_true", "n_students"], ascending=[False, False])
    df_out["global_chi2_p"]          = chi2_res["p_value"]
    df_out["global_cramers_v"]       = chi2_res["cramers_v"]
    df_out["group_diff_significant"] = bool(
        not np.isnan(chi2_res.get("p_value", np.nan))
        and chi2_res["p_value"] < alpha_bonf
    )
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Question charts (Curso × Género)
# ─────────────────────────────────────────────────────────────────────────────
def render_question_charts(selected_df: pd.DataFrame, demo_df: pd.DataFrame, max_questions: int = 40):
    st.subheader(f"Gráficos por pregunta (hasta {max_questions}) — Curso × Género")

    if demo_df.empty:
        st.warning("Sin demografía disponible.")
        return

    demo_ok = demo_df.dropna(subset=["curso", "genero"]).copy()
    if demo_ok.empty:
        st.warning("Sin filas con curso y género válidos.")
        return

    base = selected_df[~selected_df["question_id"].isin(DEMOGRAPHIC_QIDS)].copy()
    if base.empty:
        st.warning("Sin respuestas no-demográficas para graficar.")
        return

    base = base.merge(demo_ok[["survey_response_id", "curso", "genero"]],
                      on="survey_response_id", how="inner")

    q_list = (
        base[["question_id", "question_text"]]
        .drop_duplicates().sort_values("question_text")
        .head(max_questions).to_dict("records")
    )

    for qi, qrow in enumerate(q_list, start=1):
        qid   = qrow["question_id"]
        qtext = (qrow["question_text"] or "").strip() or f"Pregunta {qi}"
        sub   = base[base["question_id"] == qid].drop_duplicates(
                    subset=["survey_response_id", "curso", "genero"])
        if sub.empty:
            continue

        counts = sub.groupby(["curso", "genero"]).size().reset_index(name="n")
        pivot  = counts.pivot(index="curso", columns="genero", values="n").fillna(0).astype(int)
        for col in ["M", "F", "O", "N"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["M", "F", "O", "N"]]

        st.markdown(f"### {qi}. {qtext}")
        st.caption("Eje X: curso | Eje Y: nº estudiantes | M=Hombre F=Mujer O=No binario N=No responde")
        st.bar_chart(pivot)


# ─────────────────────────────────────────────────────────────────────────────
# Build summary for LLM
# ─────────────────────────────────────────────────────────────────────────────
def build_school_summary(
    student_view: pd.DataFrame,
    demo_view: pd.DataFrame,
    answer_view: pd.DataFrame,
    reliability: dict,
    school_name: str = "",
    n_submitted_total: int = 0,
) -> dict:
    n = int(len(student_view))
    if n == 0:
        return {}

    vict  = _safe_bool_rate(student_view["victim_freq"])
    cybr  = _safe_bool_rate(student_view["cyber_freq"])
    vpers = _safe_bool_rate(student_view["victim_persist"])
    cpers = _safe_bool_rate(student_view["cyber_persist"])
    aper  = _safe_bool_rate(student_view["any_persist"])
    sil_s = _safe_bool_rate(student_view["silence_flag_strict"])
    sil_m = _safe_bool_rate(student_view["silence_flag_missing"])

    item_desc = {}
    for name, id_set in CONSTRUCTS.items():
        desc = item_descriptives(answer_view, id_set)
        if not desc.empty:
            item_desc[name] = desc.head(10).to_dict("records")

    qcol     = "question_external_id" if "question_external_id" in answer_view.columns else "question_id"
    eco_ids  = CONSTRUCTS["ecologia_espacios"]
    eco_mask = answer_view[qcol].fillna("").astype(str).isin(eco_ids)
    eco_data = answer_view[eco_mask][["question_text", "score"]].copy()
    hotspots = []
    if not eco_data.empty:
        for qtext, grp in eco_data.groupby("question_text"):
            s = grp["score"].dropna()
            if len(s) > 0:
                hotspots.append({
                    "lugar":      str(qtext)[:60],
                    "mean_score": round(float(s.mean()), 2),
                    "pct_high":   round(float((s >= HIGH_FREQ_THRESHOLD).mean() * 100), 1),
                    "n":          int(len(s)),
                })
        hotspots = sorted(hotspots, key=lambda x: x["mean_score"], reverse=True)

    subgroups = {}
    n_tests   = 4
    if demo_view is not None and not demo_view.empty:
        if "genero" in demo_view.columns:
            for col, label in [("victim_freq", "victimizacion"), ("cyber_freq", "cybervictimizacion")]:
                t = subgroup_prevalence(student_view, demo_view, col, "genero", n_tests)
                if not t.empty:
                    subgroups[f"genero_{label}"] = t.to_dict("records")
        if "curso" in demo_view.columns:
            dg = demo_view.copy()
            dg["curso_bucket"] = dg["curso"]
            for col, label in [("victim_freq", "victimizacion"), ("cyber_freq", "cybervictimizacion")]:
                t = subgroup_prevalence(student_view, dg, col, "curso_bucket", n_tests)
                if not t.empty:
                    subgroups[f"curso_{label}"] = t.to_dict("records")

    return {
        "escuela":                            school_name or "No especificada",
        "n_encuestas_submitted_total":        n_submitted_total or n,
        "n_estudiantes_con_datos_relevantes": n,
        "fiabilidad_escalas":                 reliability,
        "nota_fiabilidad": (
            "Alpha < 0.60: baja consistencia. reliable=false: interpretar con cautela. "
            "Items globales excluidos del alpha."
        ),
        "indicadores_prevalencia": {
            "victimizacion_frecuente_mensual":   {**vict,  "threshold": threshold_flag(vict["pct"])},
            "cybervictimizacion_frecuente":      {**cybr,  "threshold": threshold_flag(cybr["pct"])},
            "victimizacion_persistente_semanal": {**vpers, "threshold": threshold_flag(vpers["pct"])},
            "cyber_persistente_semanal":         {**cpers, "threshold": threshold_flag(cpers["pct"])},
            "cualquier_persistencia_semanal":    {**aper,  "threshold": threshold_flag(aper["pct"])},
            "silencio_institucional_confirmado": {**sil_s, "threshold": threshold_flag(sil_s["pct"])},
            "silencio_institucional_posible":    {
                **sil_m,
                "nota": "Victimización frecuente con apoyo institucional sin respuesta."
            },
        },
        "distribucion_riesgo":    student_view["risk_group"].value_counts(dropna=False).to_dict(),
        "hotspots_ecologia":      hotspots,
        "subgrupos":              subgroups,
        "descriptivos_por_item":  item_desc,
        "patron_datos_faltantes": missing_pattern_summary(student_view),
        "nota_metodologica": (
            "Escala 0-4 TECH4ZERO-MX v1.0. Puntajes = SUMA de ítems. "
            "IC95% Wilson. Bonferroni aplicado. "
            "Ecología: condicional, missing ≠ 0."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM: Claude claude-sonnet-4-5-20250929
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROMPT_ES = """
Eres un especialista en convivencia escolar y prevención del bullying con 30 años de experiencia,
tanto en Chile y México como en Estados Unidos. También eres experto en estadística aplicada a
encuestas escolares y en comunicación de datos a equipos directivos no técnicos.

Tu tarea: redactar un informe claro, educativo, ético y NO alarmista para un equipo directivo
escolar, basado ÚNICAMENTE en los datos del JSON adjunto.

══════════════════════════════════════════════
REGLAS CRÍTICAS — NO NEGOCIABLES
══════════════════════════════════════════════

1. FIDELIDAD NUMÉRICA
   - Usa ÚNICAMENTE los números presentes en el JSON.
   - NO calcules, NO estimes, NO interpolos ningún valor adicional.
   - Si un dato no está en el JSON, escribe exactamente: "dato no disponible".

2. DENOMINADORES CORRECTOS
   - Encuesta general: usa n_encuestas_submitted_total.
   - Indicador específico: usa su n_with_data correspondiente.
   - Nunca inferirás el total muestral a partir de n_with_data.

3. INTERVALOS DE CONFIANZA
   - Reporta siempre IC 95% junto a cada porcentaje.
   - Formato: "12.3% (IC95%: 8.1–17.4)".

4. FIABILIDAD DE ESCALAS
   - Si alpha < 0.60 o reliable = false: señala que ese constructo debe interpretarse con cautela.

5. SIGNIFICANCIA ESTADÍSTICA
   - Solo describe diferencias como "significativas" si group_diff_significant = true.
   - Si false: "diferencia no significativa estadísticamente (p > α Bonferroni)".

6. PRIVACIDAD
   - NO identifiques ni infieras identidades de estudiantes.

7. TONO
   - Lenguaje cuidadoso, centrado en prevención y apoyo.
   - Esto es una encuesta, no un diagnóstico clínico.

══════════════════════════════════════════════
ESTRUCTURA (5 secciones)
══════════════════════════════════════════════

1) RESUMEN EJECUTIVO — 3-5 bullets + semáforo
2) HALLAZGOS PRINCIPALES — prevalencias con IC95%, fiabilidad, subgrupos,
   curso con mayor victimización, género con mayor violencia, hotspots ecología,
   nivel de confianza/apoyo institucional
3) SEÑALES DE ALERTA — silencio institucional, alpha bajo, datos faltantes >20%
4) RECOMENDACIONES — exactamente 5 acciones concretas
5) PRÓXIMOS PASOS DE MEDICIÓN — mejoras encuesta, cuándo repetir

══════════════════════════════════════════════
DATOS (JSON):
══════════════════════════════════════════════
{summary_json}
""".strip()


def generate_llm_report(summary: dict, school_name: str = "", grade_range: str = "") -> str:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    if not api_key:
        return "❌ Falta ANTHROPIC_API_KEY en secrets."

    prompt     = LLM_PROMPT_ES.format(
        summary_json=json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    system_msg = (
        "Eres un especialista en convivencia escolar y prevención del bullying. "
        f"Escuela: {school_name or 'No especificada'}. "
        f"Grados: {grade_range or 'No especificado'}. "
        "Instrumento: TECH4ZERO-MX v1.0 (OBVQ-R, ECIP-Q, ZERO-R). "
        "Responde siempre en español formal."
    )

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":       "claude-sonnet-4-5-20250929",
            "max_tokens":  2048,
            "system":      system_msg,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=60,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Anthropic API HTTP {r.status_code}: {r.text}")

    content = r.json().get("content", [])
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("Dashboard de Convivencia Escolar — TECH4ZERO-MX v1.0")

qp             = st.query_params
qp_school_id   = (qp.get("school_id")   or "").strip()
qp_analysis_dt = (qp.get("analysis_dt") or "").strip()

if qp_school_id and qp_analysis_dt:
    responses, answers, aso, qopts, questions, schools = load_tables_filtered(qp_school_id, qp_analysis_dt)
else:
    responses, answers, aso, qopts, questions, schools = load_tables_unfiltered()

answer_level = build_answer_level_df(responses, answers, aso, qopts, questions, schools)
if answer_level.empty:
    st.warning("Sin datos suficientes (o sin encuestas con status=submitted).")
    st.stop()

selected_df          = build_selected_df(answer_level)
student, reliability = compute_student_metrics(answer_level)

if student.empty:
    st.warning("No se pudieron construir métricas de estudiantes.")
    st.stop()

if qp_school_id and qp_analysis_dt:
    school_list = sorted(student["school_name"].dropna().unique().tolist())
    school = school_list[0] if school_list else "(Todas)"
    st.sidebar.info(f"school_id={qp_school_id} | analysis_dt={qp_analysis_dt}")
else:
    school_list = sorted(student["school_name"].dropna().unique().tolist())
    school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

view        = student      if school == "(Todas)" else student[student["school_name"] == school]
demo_df     = extract_demographics(selected_df)
demo_view   = demo_df      if school == "(Todas)" else demo_df[demo_df["school_name"] == school].copy()
answer_view = answer_level if school == "(Todas)" else answer_level[answer_level["school_name"] == school].copy()

# ── A) Reliability ────────────────────────────────────────────────────────────
st.subheader("Fiabilidad de escalas (Alpha de Cronbach)")
rel_rows = []
for name, info in reliability.items():
    alpha_val = info.get("alpha", np.nan)
    rel_rows.append({
        "Constructo":     name,
        "N ítems":        info["n_items"],
        "Alpha":          f"{alpha_val:.3f}" if not np.isnan(alpha_val) else "N/A",
        "Fiable (≥0.60)": "✅" if info.get("reliable") else "⚠️ No",
    })
st.dataframe(pd.DataFrame(rel_rows), use_container_width=True)
st.caption("Alpha < 0.60 = baja consistencia interna. Items globales (zero_victim_general, zero_agresor_general) excluidos del cálculo.")

# ── B) KPIs ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Panel principal (prevalencia + IC 95%)")

vict  = _safe_bool_rate(view["victim_freq"]);  vict_c = _safe_bool_count(view["victim_freq"])
pers  = _safe_bool_rate(view["any_persist"]);  pers_c = _safe_bool_count(view["any_persist"])
cybr  = _safe_bool_rate(view["cyber_freq"]);   cybr_c = _safe_bool_count(view["cyber_freq"])
sil   = _safe_bool_rate(view["silence_flag_strict"]); sil_c = _safe_bool_count(view["silence_flag_strict"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes (submitted)", f"{len(view)}")
c2.metric("Victimización frecuente (≥ mensual)",  _fmt_pct_ci(vict), f"{vict_c['n_true']}/{vict_c['n_with_data']}")
c3.metric("Persistencia alta (≥ semanal)",         _fmt_pct_ci(pers), f"{pers_c['n_true']}/{pers_c['n_with_data']}")
c4.metric("Silencio institucional (confirmado)",   _fmt_pct_ci(sil),  f"{sil_c['n_true']}/{sil_c['n_with_data']}")

st.caption(
    "IC 95% Wilson. Escala 0–4. Puntajes = SUMA de ítems (no máximo). "
    "Datos faltantes excluidos del denominador. Cero reportes ≠ ausencia de bullying."
)

# ── C) Semáforo ───────────────────────────────────────────────────────────────
st.divider()
lvl, msg   = risk_level_message(vict["pct"])
display_fn = {"CRISIS": st.error, "INTERVENCION": st.warning, "ATENCION": st.info}.get(lvl, st.success)
display_fn(f"🚦 Semáforo: **{lvl}** — {msg}")

# ── D) Risk distribution ──────────────────────────────────────────────────────
st.divider()
st.subheader("Distribución de riesgo por estudiante")
risk_counts = view["risk_group"].value_counts().reindex(["ALTO", "MEDIO", "BAJO", "SIN_DATOS"], fill_value=0)
st.bar_chart(risk_counts)

# ── E) Statistical details ────────────────────────────────────────────────────
with st.expander("📊 Detalles estadísticos (denominadores, IC, chi², Bonferroni, ítems)", expanded=False):
    tabs = st.tabs(["Prevalencias", "Ítems por constructo", "Datos faltantes"])

    with tabs[0]:
        prev_rows = []
        for label, col in [
            ("Victimización ≥ mensual",      "victim_freq"),
            ("Cybervictimización ≥ mensual", "cyber_freq"),
            ("Victimización ≥ semanal",      "victim_persist"),
            ("Cyber ≥ semanal",              "cyber_persist"),
            ("Cualquier persistencia",       "any_persist"),
            ("Silencio institucional",       "silence_flag_strict"),
        ]:
            s = _safe_bool_rate(view[col]) if col in view.columns else {}
            prev_rows.append({
                "Indicador":   label,
                "Prevalencia": f"{s.get('pct','NA')}%",
                "IC95% inf":   s.get("ci_lower", "NA"),
                "IC95% sup":   s.get("ci_upper", "NA"),
                "n con datos": s.get("n_with_data", 0),
                "n faltantes": s.get("n_missing", 0),
                "% faltantes": f"{s.get('missing_pct','NA')}%",
                "Semáforo":    threshold_flag(s.get("pct")),
            })
        st.dataframe(pd.DataFrame(prev_rows), use_container_width=True)

    with tabs[1]:
        for name, id_set in CONSTRUCTS.items():
            info      = reliability.get(name, {})
            alpha_val = info.get("alpha", np.nan)
            alpha_str = f"{alpha_val:.3f}" if not np.isnan(alpha_val) else "N/A"
            badge     = "✅" if info.get("reliable") else "⚠️"
            st.markdown(f"**{name}** — Alpha: {alpha_str} {badge}")
            desc = item_descriptives(answer_view, id_set)
            if not desc.empty:
                st.dataframe(desc, use_container_width=True)
            else:
                st.caption("Sin ítems detectados.")

    with tabs[2]:
        mp = missing_pattern_summary(view)
        if mp:
            st.dataframe(pd.DataFrame(mp).T, use_container_width=True)
            st.caption("missing_pct > 20% sugiere posible sesgo sistemático.")
        else:
            st.caption("Sin análisis disponible.")

# ── F) Subgroups ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Subgrupos — Género y Curso (chi² + Cramér's V + Bonferroni)")

if demo_view.empty:
    st.info("Sin demografía disponible para esta selección.")
else:
    n_tests = 4
    colA, colB = st.columns(2)

    with colA:
        for label, col in [
            ("Victimización ≥ mensual",      "victim_freq"),
            ("Cybervictimización ≥ mensual", "cyber_freq"),
        ]:
            st.markdown(f"### Género × {label}")
            t = subgroup_prevalence(view, demo_view, col, "genero", n_tests)
            if not t.empty:
                sig   = t["group_diff_significant"].iloc[0]
                cV    = t["global_cramers_v"].iloc[0]
                p_val = t["global_chi2_p"].iloc[0]
                badge = "✅ Significativa" if sig else "— No significativa"
                st.caption(f"p={p_val:.4f} | V={cV:.3f} | {badge} (α Bonferroni={bonferroni_threshold(n_tests):.4f})")
                st.dataframe(
                    t.drop(columns=["global_chi2_p", "global_cramers_v", "group_diff_significant"], errors="ignore"),
                    use_container_width=True
                )

    with colB:
        for label, col in [
            ("Victimización ≥ mensual",      "victim_freq"),
            ("Cybervictimización ≥ mensual", "cyber_freq"),
        ]:
            st.markdown(f"### Curso × {label}")
            dg = demo_view.copy()
            dg["curso_bucket"] = dg["curso"]
            t = subgroup_prevalence(view, dg, col, "curso_bucket", n_tests)
            if not t.empty:
                sig   = t["group_diff_significant"].iloc[0]
                cV    = t["global_cramers_v"].iloc[0]
                p_val = t["global_chi2_p"].iloc[0]
                badge = "✅ Significativa" if sig else "— No significativa"
                st.caption(f"p={p_val:.4f} | V={cV:.3f} | {badge} (α Bonferroni={bonferroni_threshold(n_tests):.4f})")
                st.dataframe(
                    t.drop(columns=["global_chi2_p", "global_cramers_v", "group_diff_significant"], errors="ignore"),
                    use_container_width=True
                )

# ── G) Ecology hotspots ───────────────────────────────────────────────────────
st.divider()
st.subheader("🗺️ Ecología del Bullying — Lugares más inseguros")

eco_ids  = CONSTRUCTS["ecologia_espacios"]
qcol_aw  = "question_external_id" if "question_external_id" in answer_view.columns else "question_id"
eco_mask = answer_view[qcol_aw].fillna("").astype(str).isin(eco_ids)
eco_data = answer_view[eco_mask][["question_text", "score"]].copy()

if not eco_data.empty:
    hotspot_df = (
        eco_data.groupby("question_text")["score"]
        .agg(mean_score="mean", n="count")
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )
    hotspot_df["mean_score"]    = hotspot_df["mean_score"].round(2)
    hotspot_df["question_text"] = hotspot_df["question_text"].str[:60]
    st.dataframe(
        hotspot_df.rename(columns={
            "question_text": "Lugar",
            "mean_score":    "Score medio (0-4)",
            "n":             "N respuestas"
        }),
        use_container_width=True
    )
    st.caption("Solo estudiantes que reportaron haber sufrido bullying. Mayor score = mayor riesgo en ese espacio.")
else:
    st.info("Sin datos de ecología disponibles.")

# ── H) Question charts ────────────────────────────────────────────────────────
st.divider()
selected_view = selected_df if school == "(Todas)" else selected_df[selected_df["school_name"] == school].copy()
render_question_charts(selected_view, demo_view, max_questions=40)

# ── I) LLM Report ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📝 Informe interpretativo (Claude Sonnet 4.5)")

grade_range_str = ""
if not demo_view.empty and "curso" in demo_view.columns:
    valid = demo_view["curso"].dropna()
    if len(valid):
        grade_range_str = f"{int(valid.min())}° a {int(valid.max())}° secundaria/preparatoria"

n_submitted_total = int(filter_submitted(responses).shape[0]) if not responses.empty else int(len(view))

if st.button("Generar informe en lenguaje humano"):
    with st.spinner("Generando informe con Claude Sonnet 4.5..."):
        try:
            summary = build_school_summary(
                view, demo_view, answer_view, reliability,
                school_name=school,
                n_submitted_total=n_submitted_total,
            )
            report = generate_llm_report(summary, school_name=school, grade_range=grade_range_str)
            st.markdown(report)
        except Exception as e:
            st.error(f"Error al generar informe: {e}")

# ── Debug ─────────────────────────────────────────────────────────────────────
with st.expander("🔧 Debug"):
    st.write("Query params:",        dict(st.query_params))
    st.write("Rows responses:",      len(responses))
    st.write("Rows answers:",        len(answers))
    st.write("Rows aso:",            len(aso))
    st.write("Rows qopts:",          len(qopts))
    st.write("Rows questions:",      len(questions))
    st.write("Rows schools:",        len(schools))
    st.write("Rows answer_level:",   len(answer_level))
    st.write("Rows selected_df:",    len(selected_df))
    st.write("Rows student:",        len(student))
    st.write("Rows demo_df:",        len(demo_df))
    st.write("Reliability:",         reliability)
    st.write("Constructs loaded:",   list(CONSTRUCTS.keys()))
    st.write("Question IDs sample:")
    st.dataframe(
        answer_level[["question_id", "question_external_id", "question_text"]]
        .drop_duplicates().head(30),
        use_container_width=True
    )
