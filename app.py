# app.py — Bullying Dashboard (Phase 1: Current State + LLM Report)

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
# Scoring scale
# -------------------------------------------------
SCALE = {
    "Nunca": 0,
    "Sólo una vez": 1,
    "Varias veces al mes": 2,
    "Casi cada semana": 3,
    "Casi cada día": 4,
    "Sí": 2,
    "A veces": 1,
    "No": 0,
}

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
        .merge(responses, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp"))
        .merge(questions, left_on="question_id", right_on="id", suffixes=("", "_q"))
        .merge(aso, left_on="id_ans", right_on="question_answer_id", how="inner")
        .merge(qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt"))
    )

    # FIX: avoid MergeError by renaming schools.id to school_pk
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
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

def compute_student_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["victimization"] = d["question_text"].str.contains(
        r"has sido agredido|te han molestado|te han ignorado", case=False, na=False
    ) * d["score"]

    d["cyberbullying"] = d["question_text"].str.contains(
        r"internet|mensajería|mensajes|videos|fotos", case=False, na=False
    ) * d["score"]

    d["trust_friends"] = d["question_text"].str.contains("amigos", case=False, na=False) * d["score"]
    d["trust_adults"] = d["question_text"].str.contains("adultos", case=False, na=False) * d["score"]
    d["trust_parents"] = d["question_text"].str.contains("padres", case=False, na=False) * d["score"]

    d["safety"] = d["question_text"].str.contains(
        r"me siento seguro|me gusta venir|buen lugar", case=False, na=False
    ) * d["score"]

    student = d.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        victimization=("victimization", "sum"),
        cyberbullying=("cyberbullying", "sum"),
        trust_friends=("trust_friends", "sum"),
        trust_adults=("trust_adults", "sum"),
        trust_parents=("trust_parents", "sum"),
        safety=("safety", "sum"),
    ).reset_index()

    p80 = float(student["victimization"].quantile(0.80)) if len(student) else 0.0
    student["risk_level"] = np.where(student["victimization"] >= p80, "ALTO", "MEDIO/BAJO")

    student["knows_friends"] = student["trust_friends"] > 0
    student["knows_adults"] = student["trust_adults"] > 0
    student["knows_parents"] = student["trust_parents"] > 0

    return student

# -------------------------------------------------
# LLM (Groq now, Hugging Face later)
# -------------------------------------------------
def build_school_summary(view: pd.DataFrame) -> dict:
    return {
        "students": int(len(view)),
        "high_risk_pct": float((view["risk_level"] == "ALTO").mean()),
        "avg_victimization": float(view["victimization"].mean()),
        "avg_cyberbullying": float(view["cyberbullying"].mean()),
        "knows_friends_pct": float(view["knows_friends"].mean()),
        "knows_adults_pct": float(view["knows_adults"].mean()),
        "knows_parents_pct": float(view["knows_parents"].mean()),
    }

def generate_llm_report(summary: dict) -> str:
    provider = st.secrets.get("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return _groq_report(summary)
    elif provider == "hf":
        return "Hugging Face not enabled yet."
    return "LLM provider not configured."

def _groq_report(summary: dict) -> str:
    api_key = st.secrets.get("GROQ_API_KEY")
    model = st.secrets.get("LLM_MODEL", "llama3-8b-8192")

    if not api_key:
        return "Groq API key missing."

    prompt = f"""
You are an expert in school climate and bullying prevention.

Write a clear, non-technical report for school administrators based ONLY
on the aggregated data below.

Include:
- Overall level of victimization risk
- Key patterns you can infer
- Who seems to be aware (friends, adults, parents)
- 3 concrete, practical recommendations

AGGREGATED DATA:
{summary}
"""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You generate concise school safety reports."},
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
st.title("Dashboard Bullying — Phase 1 (Current state)")

responses, answers, aso, qopts, questions, schools = load_tables()

# Debug counts (keep for now)
st.write("Counts", {
    "survey_responses": len(responses),
    "question_answers": len(answers),
    "answer_selected_options": len(aso),
    "question_options": len(qopts),
    "questions": len(questions),
    "schools": len(schools),
})

df_long = build_long_df(responses, answers, aso, qopts, questions, schools)
if df_long.empty:
    st.warning("Not enough data yet.")
    st.stop()

student = compute_student_metrics(df_long)

school_list = sorted(student["school_name"].dropna().unique().tolist())
school = st.sidebar.selectbox("School", ["(All)"] + school_list)

view = student if school == "(All)" else student[student["school_name"] == school]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", int(len(view)))
c2.metric("High risk", int((view["risk_level"] == "ALTO").sum()))
c3.metric("Avg victimization", float(round(view["victimization"].mean(), 2)))
c4.metric("Avg cyberbullying", float(round(view["cyberbullying"].mean(), 2)))

st.divider()

st.subheader("Risk distribution")
st.bar_chart(view["risk_level"].value_counts())

st.subheader("Who knows (intention to tell)")
st.bar_chart(pd.DataFrame({
    "Friends": [view["knows_friends"].mean()],
    "Adults": [view["knows_adults"].mean()],
    "Parents": [view["knows_parents"].mean()],
}).T)

st.divider()
st.subheader("AI-generated School Report")

if st.button("Generate AI Report"):
    with st.spinner("Generating report…"):
        summary = build_school_summary(view)
        report = generate_llm_report(summary)
        st.markdown(report)


