# app.py — Dashboard Streamlit (Fase 1: Estado actual)
# Ejecuta: streamlit run app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# -----------------------------
# Configuración
# -----------------------------
st.set_page_config(page_title="Bullying Dashboard", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Faltan credenciales SUPABASE_URL / SUPABASE_KEY.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=300)
def load_tables():
    responses = supabase.table("survey_responses").select("*").execute().data
    answers = supabase.table("question_answers").select("*").execute().data
    aso = supabase.table("answer_selected_options").select("*").execute().data
    qopts = supabase.table("question_options").select("*").execute().data
    questions = supabase.table("questions").select("*").execute().data
    schools = supabase.table("schools").select("*").execute().data
    return (
        pd.DataFrame(responses),
        pd.DataFrame(answers),
        pd.DataFrame(aso),
        pd.DataFrame(qopts),
        pd.DataFrame(questions),
        pd.DataFrame(schools),
    )

def build_long_df(responses, answers, aso, qopts, questions, schools):
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    df = (
        answers
        .merge(responses, left_on="survey_response_id", right_on="id", suffixes=("_ans","_resp"))
        .merge(questions, left_on="question_id", right_on="id")
        .merge(aso, left_on="id_ans", right_on="question_answer_id")
        .merge(qopts, left_on="option_id", right_on="id", suffixes=("", "_opt"))
    )

    if not schools.empty and "school_id" in df.columns:
        df = df.merge(schools[["id", "name"]], left_on="school_id", right_on="id", how="left")
        df.rename(columns={"name": "school_name"}, inplace=True)
    else:
        df["school_name"] = "N/A"

    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)
    return df

def compute_student_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    df["victimization"] = df["question_text"].str.contains(
        "has sido agredido|te han molestado|te han ignorado", case=False, na=False
    ) * df["score"]

    df["cyberbullying"] = df["question_text"].str.contains(
        "internet|mensajería|mensajes de texto|videos|fotos", case=False, na=False
    ) * df["score"]

    df["trust_friends"] = df["question_text"].str.contains("amigos", case=False, na=False) * df["score"]
    df["trust_adults"]  = df["question_text"].str.contains("adultos", case=False, na=False) * df["score"]
    df["trust_parents"] = df["question_text"].str.contains("padres", case=False, na=False) * df["score"]

    df["safety"] = df["question_text"].str.contains(
        "me siento seguro|me gusta venir|buen lugar", case=False, na=False
    ) * df["score"]

    # Demografía (si existe como opciones)
    def pick_demo(contains):
        sub = df[df["question_text"].str.contains(contains, case=False, na=False)]
        return sub.groupby("survey_response_id")["option_text"].agg(
            lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""
        )

    gender = pick_demo("género")
    grade  = pick_demo("grado")
    age    = pick_demo("edad")

    student = df.groupby("survey_response_id").agg(
        school_name=("school_name", "first"),
        victimization=("victimization", "sum"),
        cyberbullying=("cyberbullying", "sum"),
        trust_friends=("trust_friends", "sum"),
        trust_adults=("trust_adults", "sum"),
        trust_parents=("trust_parents", "sum"),
        safety=("safety", "sum"),
    ).reset_index()

    student["gender"] = student["survey_response_id"].map(gender).fillna("Sin dato")
    student["grade"]  = student["survey_response_id"].map(grade).fillna("Sin dato")
    student["age"]    = student["survey_response_id"].map(age).fillna("")

    p80 = student["victimization"].quantile(0.80) if len(student) else 0
    student["risk_level"] = np.where(student["victimization"] >= p80, "ALTO", "MEDIO/BAJO")

    student["knows_friends"] = student["trust_friends"] > 0
    student["knows_adults"]  = student["trust_adults"] > 0
    student["knows_parents"] = student["trust_parents"] > 0

    return student

# -----------------------------
# UI
# -----------------------------
st.title("Dashboard Bullying — Fase 1 (Estado a
