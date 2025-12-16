import streamlit as st

# =========================
# NAVEGACIÓN (MENÚ)
# =========================
st.set_page_config(page_title="TECH4ZERO Dashboard", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "menu"   # "menu" | "descriptivo" | "futuro"

def go(page_name: str):
    st.session_state.page = page_name
    st.rerun()

# =========================
# PLACEHOLDERS: conecta con TU app real
# Reemplaza estas funciones con las tuyas (o llama a tus funciones existentes).
# =========================

def get_school_view():
    """
    Aquí debes devolver el dataset ya cargado para la escuela seleccionada,
    por ejemplo el df_long o la estructura que ya usas en tu app.
    """
    return None

def build_summary_descriptivo(view):
    """Construye el summary dict/text que envías a la LLM para el reporte descriptivo."""
    return {"mode": "descriptivo"}

def build_summary_futuro(view):
    """Construye el summary dict/text que envías a la LLM para el reporte predictivo."""
    return {"mode": "futuro"}

def generate_llm_report(summary: dict) -> str:
    """Tu función existente (Groq/Llama) que retorna texto."""
    return f"Reporte IA (demo). Modo: {summary.get('mode')}"

def render_descriptivo_dashboard(view):
    """Tu dashboard descriptivo actual (gráficos, tablas, etc.)."""
    st.subheader("Dashboard: Patrones descriptivos")
    st.info("Aquí van tus gráficos descriptivos (distribuciones, lugares, género, edad, etc.).")

def render_futuro_dashboard(view):
    """Tu dashboard predictivo/futuro."""
    st.subheader("Dashboard: Comportamiento futuro y patrones")
    st.info("Aquí van tus gráficos de predicción (tendencia, riesgo futuro, etc.).")

# =========================
# PÁGINAS
# =========================

def page_menu():
    st.title("TECH4ZERO — Dashboard")
    st.write("Elige qué quieres analizar:")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Patrones descriptivos", use_container_width=True):
            go("descriptivo")
    with c2:
        if st.button("Comportamiento futuro y patrones", use_container_width=True):
            go("futuro")

def page_descriptivo():
    st.title("Patrones descriptivos")

    view = get_school_view()

    # 1) Dashboard
    render_descriptivo_dashboard(view)

    # 2) Explicación IA
    st.divider()
    st.subheader("Explicación en lenguaje humano (IA)")
    if st.button("Generar explicación IA", type="primary"):
        with st.spinner("Generando explicación…"):
            summary = build_summary_descriptivo(view)
            report = generate_llm_report(summary)
            st.markdown(report)

    st.divider()
    if st.button("Volver menú principal"):
        go("menu")

def page_futuro():
    st.title("Comportamiento futuro y patrones")

    view = get_school_view()

    # 1) Dashboard
    render_futuro_dashboard(view)

    # 2) Explicación IA
    st.divider()
    st.subheader("Explicación en lenguaje humano (IA)")
    if st.button("Generar explicación IA", type="primary"):
        with st.spinner("Generando explicación…"):
            summary = build_summary_futuro(view)
            report = generate_llm_report(summary)
            st.markdown(report)

    st.divider()
    if st.button("Volver menú principal"):
        go("menu")

# =========================
# ROUTER
# =========================
if st.session_state.page == "menu":
    page_menu()
elif st.session_state.page == "descriptivo":
    page_descriptivo()
elif st.session_state.page == "futuro":
    page_futuro()
else:
    st.session_state.page = "menu"
    st.rerun()
