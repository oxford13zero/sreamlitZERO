# app.py — Dashboard Streamlit (Fase 1: Estado actual)
# Ejecuta local: streamlit run app.py

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
# Data loading (con paginación)
# -----------------------------
@st.cache_data(ttl=300)
def _fetch_all(table_name: str, batch: int = 1000) -> pd.DataFrame:
    all_rows = []
    start = 0
    while True:
        data = (
            supabase.table(table_name)
            .select("*")
            .range(start, start + batch - 1)
            .execute()
            .data
        )
        if not data:
            break
        all_rows.extend(data)
        if len(data) < batch:
            break
        start += batch
    return pd.DataFrame(all_rows)

@st.cache_data(ttl=300)
def load_tables():
    responses = _fetch_all("survey_responses")
    answers = _fetch_all("question_answers")
    aso = _fetch_all("answer_selected_options")
    qopts = _fetch_all("question_options")
    questions = _fetch_all("questions")
    schools = _fetch_all("schools")
    return responses, answers, aso, qopts, questions, schools

# -----------------------------
# Transformaciones
# -----------------------------
def build_long_df(responses, answers, aso, qopts, questions, schools):
    # Si falta algo, no seguimos
    if responses.empty or answers.empty or aso.empty or qopts.empty or questions.empty:
        return pd.DataFrame()

    # Normalizar columnas mínimas esperadas
    for df, cols in [
        (responses, ["id", "school_id"]),
        (answers, ["id", "survey_response_id", "question_id"]),
        (aso, ["question_answer_id", "option_id"]),
        (qopts, ["id", "option_text"]),
        (questions, ["id", "question_text"]),
    ]:
        for c in cols:
            if c not in df.columns:
                raise KeyError(f"Falta columna '{c}' en tabla correspondiente.")

    # Merge seguro (paso a paso)
    df = answers.merge(
        responses, left_on="survey_response_id", right_on="id", suffixes=("_ans", "_resp")
    )
    df = df.merge(
        questions, left_on="question_id", right_on="id", suffixes=("", "_q")
    )
    df = df.merge(
        aso, left_on="id_ans", right_on="question_answer_id", how="inner"
    )
    df = df.merge(
        qopts, left_on="option_id", right_on="id", how="inner", suffixes=("", "_opt")
    )

    # School name
    if not schools.empty and "id" in schools.columns and "name" in schools.columns:
        df = df.merge(
            schools[["id", "name"]],
            left_on="school_id",
            right_on="id",
            how="left",
            suffixes=("", "_school"),
        )
        df.rename(columns={"name": "school_name"}, inplace=True)
    else:
        df["school_name"] = df["school_id"].astype(str)

    df["question_text"] = df["question_text"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["score"] = df["option_text"].map(SCALE).fillna(0).astype(float)

    return df

def compute_student_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    victim_kw = r"has sido agredido|te han molestado|te han ignorado"
    cyber_kw = r"internet|mensajería|mensajes de texto|videos|fotos"
    safety_kw = r"me siento seguro|me gusta venir|buen lugar"

    df["victimization"] = df["question_text"].str.contains(victim_kw, case=False, na=False) * df["score"]
    df["cyberbullying"] = df["question_text"].str.contains(cyber_kw, case=False, na=False) * df["score"]
    df["trust_friends"] = df["question_text"].str.contains("amigos", case=False, na=False) * df["score"]
    df["trust_adults"]  = df["question_text"].str.contains("adultos", case=False, na=False) * df["score"]
    df["trust_parents"] = df["question_text"].str.contains("padres", case=False, na=False) * df["score"]
    df["safety"] = df["question_text"].str.contains(safety_kw, case=False, na=False) * df["score"]

    # Demografía (si existe como opciones)
    def pick_demo(contains: str) -> pd.Series:
        sub = df[df["question_text"].str.contains(contains, case=False, na=False)]
        if sub.empty:
            return pd.Series(dtype=str)
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

    # Riesgo alto por percentil
    p80 = float(student["victimization"].quantile(0.80)) if len(student) else 0.0
    student["risk_level"] = np.where(student["victimization"] >= p80, "ALTO", "MEDIO/BAJO")

    # “Quién sabe” (intención de contar)
    student["knows_friends"] = student["trust_friends"] > 0
    student["knows_adults"]  = student["trust_adults"] > 0
    student["knows_parents"] = student["trust_parents"] > 0

    return student

# -----------------------------
# UI
# -----------------------------
st.title("Dashboard Bullying — Fase 1 (Estado actual)")

with st.sidebar:
    st.header("Opciones")
    refresh = st.button("Recargar datos")
    risk_pct = st.slider("Umbral riesgo alto (percentil)", 50, 95, 80, 5)

if refresh:
    st.cache_data.clear()

try:
    responses, answers, aso, qopts, questions, schools = load_tables()

    # Debug visible (para evitar “pantalla en blanco”)
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
        st.warning("No hay datos suficientes en Supabase para construir el dashboard todavía.")
        st.stop()

    student = compute_student_metrics(df_long)

    # Recalcular riesgo con percentil elegido
    thr = float(student["victimization"].quantile(risk_pct / 100.0)) if len(student) else 0.0
    student["risk_level"] = np.where(student["victimization"] >= thr, "ALTO", "MEDIO/BAJO")

    # Filtro por escuela
    school_list = sorted(student["school_name"].fillna("N/A").unique().tolist())
    selected_school = st.sidebar.selectbox("Escuela", ["(Todas)"] + school_list)

    view = student.copy()
    if selected_school != "(Todas)":
        view = view[view["school_name"] == selected_school]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiantes", f"{len(view):,}")
    c2.metric("Riesgo ALTO", f"{(view['risk_level']=='ALTO').sum():,}")
    c3.metric("Victimización promedio", f"{view['victimization'].mean():.2f}")
    c4.metric("Ciberbullying promedio", f"{view['cyberbullying'].mean():.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Distribución de riesgo")
        risk_counts = view["risk_level"].value_counts()
        st.bar_chart(risk_counts)

        st.subheader("Top 20 — mayor victimización")
        st.dataframe(
            view.sort_values("victimization", ascending=False).head(20),
            use_container_width=True
        )

    with right:
        st.subheader("Quién sabe lo que pasa (intención de contar)")
        knows = pd.DataFrame({
            "Amigos": [view["knows_friends"].mean()],
            "Adultos colegio": [view["knows_adults"].mean()],
            "Padres": [view["knows_parents"].mean()],
        }).T
        st.bar_chart(knows)

        st.subheader("Riesgo ALTO por género / grado")
        a, b = st.columns(2)

        with a:
            g = view.copy()
            g["gender"] = g["gender"].replace("", "Sin dato")
            g_tab = (g.assign(high=(g["risk_level"] == "ALTO").astype(int))
                      .groupby("gender")["high"].mean()
                      .sort_values(ascending=False)
                      .reset_index()
                      .rename(columns={"high": "proporcion_riesgo_alto"}))
            st.dataframe(g_tab, use_container_width=True)

        with b:
            gr = view.copy()
            gr["grade"] = gr["grade"].replace("", "Sin dato")
            gr_tab = (gr.assign(high=(gr["risk_level"] == "ALTO").astype(int))
                       .groupby("grade")["high"].mean()
                       .sort_values(ascending=False)
                       .reset_index()
                       .rename(columns={"high": "proporcion_riesgo_alto"}))
            st.dataframe(gr_tab, use_container_width=True)

    st.divider()

    st.subheader("Exportar")
    st.download_button(
        "Descargar CSV (estado actual)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="bullying_estado_actual.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error("Ocurrió un error ejecutando el dashboard.")
    st.exception(e)
