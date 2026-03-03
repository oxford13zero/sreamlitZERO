# app.py
"""
TECH4ZERO-MX v3.0 — Streamlit Dashboard
========================================
Statistical analysis dashboard for school climate and bullying survey.

Version: 3.0 (Modular Architecture)
Survey: SURVEY_003
Platform: Streamlit Cloud / Vercel
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Import custom modules
from construct_definitions import (
    parse_external_id,
    get_construct_metadata,
    get_all_constructs,
    get_construct_items,
    validate_construct_coverage,
    is_global_screener,
    CONSTRUCT_METADATA,
)
from stats_engine import (
    analyze_reliability,
    calculate_prevalence,
    compare_subgroups,
    compare_subgroups_continuous,
    bonferroni_threshold,
    item_descriptives,
    construct_correlation_matrix,
    missing_pattern_summary,
    run_cfa,
    classify_bully_victim_typology,
    analyze_bullying_overlap,
    calculate_risk_index,
    assess_sample_representativeness,
)
from visualization import (
    plot_reliability_comparison,
    plot_prevalence_by_construct,
    plot_subgroup_comparison,
    plot_correlation_heatmap,
    plot_item_severity_ranking,
    plot_ecology_hotspots,
)

# Supabase import
from supabase import create_client


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TECH4ZERO-MX Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Supabase configuration (WORKING PATTERN from debug)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Survey configuration
SURVEY_CODE = "SURVEY_004"


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_survey_data(school_id=None, analysis_dt=None):
    """
    Load all survey data from Supabase.
    
    Args:
        school_id: Filter by specific school
        analysis_dt: Filter by specific analysis date
    """
    try:
        # Get survey ID
        survey_result = supabase.table('surveys').select('id').eq('code', SURVEY_CODE).execute()
        
        if not survey_result.data:
            st.warning(f"Survey {SURVEY_CODE} not found")
            return None, None, None
        
        survey_id = survey_result.data[0]['id']
        
        # Get all active survey IDs dynamically
        active_surveys = supabase.table('surveys').select('id').eq('is_active', True).execute()

        if not active_surveys.data:
            st.error("❌ No active surveys found in database")
            return None, None, None

        survey_ids = [str(s['id']) for s in active_surveys.data]
        print(f"📋 Survey IDs: {survey_ids}")
        print(f"📋 Type: {type(survey_ids)}")
        print(f"📋 First ID type: {type(survey_ids[0]) if survey_ids else 'empty'}")

        # Build query step by step
        query = supabase.table('survey_responses').select(
            'id, survey_id, school_id, student_external_id, status, analysis_requested_dt'
        ).eq('status', 'submitted')

        # Filter by school_id if provided
        if school_id:
            query = query.eq('school_id', school_id)

        # Filter by analysis_requested_dt if provided
        if analysis_dt:
            query = query.eq('analysis_requested_dt', analysis_dt)

        # Filter by active survey IDs (apply .in_() AFTER other filters)
            query = query.in_('survey_id', survey_ids)

        responses = query.execute()
        responses_df = pd.DataFrame(responses.data)

        if responses_df.empty:
            st.warning(f"⚠️ No submitted responses found for this analysis")
            return None, None, None

        response_ids = responses_df['id'].tolist()

        # 3. Load answers (chunked)
        answers_data = []
        chunk_size = 100
        for i in range(0, len(response_ids), chunk_size):
            chunk = response_ids[i:i + chunk_size]
            result = supabase.table('question_answers').select(
                'id, survey_response_id, question_id'
            ).in_('survey_response_id', chunk).execute()
            answers_data.extend(result.data)

        answers_df = pd.DataFrame(answers_data)

        if answers_df.empty:
            return responses_df, None, None
        
 



        
        
        # 4. Load questions
        question_ids = answers_df['question_id'].unique().tolist()
        questions_data = []
        for i in range(0, len(question_ids), chunk_size):
            chunk = question_ids[i:i + chunk_size]
            result = supabase.table('questions').select(
                'id, external_id, question_text'
            ).in_('id', chunk).execute()
            questions_data.extend(result.data)
        
        questions_df = pd.DataFrame(questions_data)
        
        # 5. Load selected options
        answer_ids = answers_df['id'].tolist()
        selected_data = []
        for i in range(0, len(answer_ids), chunk_size):
            chunk = answer_ids[i:i + chunk_size]
            result = supabase.table('answer_selected_options').select(
                'question_answer_id, option_id'
            ).in_('question_answer_id', chunk).execute()
            selected_data.extend(result.data)
        
        selected_df = pd.DataFrame(selected_data)
        
        # 6. Load options
        if not selected_df.empty:
            option_ids = selected_df['option_id'].unique().tolist()
            options_data = []
            for i in range(0, len(option_ids), chunk_size):
                chunk = option_ids[i:i + chunk_size]
                result = supabase.table('question_options').select(
                    'id, option_code, option_text'
                ).in_('id', chunk).execute()
                options_data.extend(result.data)
            
            options_df = pd.DataFrame(options_data)
        else:
            options_df = pd.DataFrame()
        
        # 7. Build answer-level dataframe (CLEAN MERGES - NO DUPLICATE COLUMNS)
        # Step 1: Rename to avoid conflicts
        answers_clean = answers_df.rename(columns={'id': 'answer_id'})
        questions_clean = questions_df.rename(columns={'id': 'q_id'})
        
        merged = answers_clean.merge(
            questions_clean,
            left_on='question_id',
            right_on='q_id',
            how='left'
        )
        
        # Use external_id as question_id
        merged['question_id'] = merged['external_id']
        merged = merged.drop(columns=['q_id', 'external_id'], errors='ignore')
        
        # Step 2: Add selected options
        if not selected_df.empty:
            selected_clean = selected_df.rename(columns={'option_id': 'opt_id'})
            
            merged = merged.merge(
                selected_clean,
                left_on='answer_id',
                right_on='question_answer_id',
                how='left'
            )
            
            # Step 3: Add option details
            if not options_df.empty:
                options_clean = options_df.rename(columns={'id': 'option_pk'})
                
                merged = merged.merge(
                    options_clean,
                    left_on='opt_id',
                    right_on='option_pk',
                    how='left'
                )
        
        # Convert option_code to score
        if 'option_code' in merged.columns:
            merged['score'] = pd.to_numeric(merged['option_code'], errors='coerce')
        else:
            merged['score'] = None
        
        # 8. Create student-level dataframe (AVOID DUPLICATE MERGES)
        students_df = responses_df[['id', 'school_id']].copy()
        
        # Extract demographics
        demo_map = {
            'survey_004_zero_general_grado':  'grado',
            'survey_004_zero_general_genero': 'genero',
            'survey_004_zero_general_edad':   'edad',
            'survey_004_zero_general_lengua': 'lengua_indigena',
        }
        
        for ext_id, col_name in demo_map.items():
            # Drop duplicates BEFORE merge
            demo_data = merged[merged['question_id'] == ext_id][
                ['survey_response_id', 'option_text']
            ].drop_duplicates('survey_response_id')
            
            # Rename to avoid conflicts
            demo_data = demo_data.rename(columns={
                'survey_response_id': 'resp_id',
                'option_text': col_name
            })
            
            students_df = students_df.merge(
                demo_data,
                left_on='id',
                right_on='resp_id',
                how='left'
            )
            
            # Clean up temporary column
            if 'resp_id' in students_df.columns:
                students_df = students_df.drop(columns=['resp_id'])
        
        return responses_df, merged, students_df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


# ══════════════════════════════════════════════════════════════
# CONSTRUCT SCORE CALCULATION
# ══════════════════════════════════════════════════════════════

def calculate_construct_scores(answers_df, students_df):
    """
    Calculate sum scores for each construct.
    
    Args:
        answers_df: Answer-level DataFrame
        students_df: Student-level DataFrame
    
    Returns:
        Updated students_df with construct scores
    """
    if answers_df is None or answers_df.empty:
        return students_df
    
    all_constructs = [c for c in get_all_constructs() if c != 'demographic']
    all_external_ids = answers_df['question_id'].unique().tolist()
    
    for construct in all_constructs:
        # Get items for this construct
        construct_items = get_construct_items(construct, all_external_ids)
        construct_items = [item for item in construct_items if not is_global_screener(item)]
        
        if not construct_items:
            continue
        
        # Calculate sum score
        construct_answers = answers_df[
            answers_df['question_id'].isin(construct_items)
        ][['survey_response_id', 'score']]
        
        sum_scores = construct_answers.groupby('survey_response_id')['score'].sum().reset_index()
        sum_scores.columns = ['id', f'{construct}_sum']
        
        students_df = students_df.merge(sum_scores, on='id', how='left')
        
        # Calculate frequency indicator (mean score >= 2)
        mean_scores = construct_answers.groupby('survey_response_id')['score'].mean().reset_index()
        mean_scores[f'{construct}_freq'] = mean_scores['score'] >= 2
        mean_scores = mean_scores[['survey_response_id', f'{construct}_freq']].rename(
            columns={'survey_response_id': 'id'}
        )
        
        students_df = students_df.merge(mean_scores, on='id', how='left')
    
    return students_df


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════
def main():
    """Main Streamlit application"""
    
    st.title("🏫 TECH4ZERO-MX v1.0")
    st.markdown("**Encuesta de Clima Escolar, Bullying y Cyberbullying**")
    st.markdown("---")
    
    #  Get URL parameters
    params = st.query_params
    url_school_id = params.get('school_id')
    url_analysis_dt = params.get('analysis_dt')
    
    # Convert school_id to integer if provided
    if url_school_id:
        try:
            url_school_id = int(url_school_id)
        except:
            url_school_id = None
    
    # Show analysis info
    if url_analysis_dt:
        st.info(f"📊 Mostrando análisis específico del: {url_analysis_dt}")
    
    #  Load data with BOTH filters
    with st.spinner("Cargando datos..."):
        responses_df, answers_df, students_df = load_survey_data(
            school_id=url_school_id,
            analysis_dt=url_analysis_dt
        )
    
    if responses_df is None or answers_df is None or students_df is None:
        st.error("❌ No se pudieron cargar los datos para este análisis.")
        return
    
    # Show metrics for this specific analysis
    st.success(f"✅ Datos cargados: {len(students_df)} estudiantes en este análisis")

    # ── Representativeness warning banner ────────────────────
    if st.session_state.get('rep_status') == "NO_REPRESENTATIVA":
        st.error(
            f"🔴 **MUESTRA NO REPRESENTATIVA** — "
            f"{st.session_state.get('rep_warning', '')} "
            f"Los resultados no deben generalizarse al plantel completo."
        )
    elif st.session_state.get('rep_status') == "INSUFICIENTE":
        st.warning(
            f"🟠 **MUESTRA INSUFICIENTE** — "
            f"{st.session_state.get('rep_warning', '')} "
            f"Interprete los resultados con cautela."
        )
    
    # Calculate construct scores
    students_df = calculate_construct_scores(answers_df, students_df)
    
    # Get all external IDs
    all_external_ids = answers_df['question_id'].unique().tolist()
    
    # Validate coverage
    coverage = validate_construct_coverage(all_external_ids)
    
    # Build sidebar (data is already loaded)
    with st.sidebar:
        st.header("⚙️ Configuración")

        school_id      = None
        school_name    = "Escuela Secundaria Federal"
        encargado      = "No disponible"
        n_primaria     = 0
        n_secundaria   = 0
        n_preparatoria = 0

        if 'school_id' in students_df.columns:
            school_id = students_df['school_id'].iloc[0]

            # Load school name + enrollment counts in one query
            try:
                school_data = supabase.table('schools').select(
                    'name, students_primaria, students_secundaria, students_preparatoria'
                ).eq('id', school_id).execute()

                if school_data.data:
                    row = school_data.data[0]
                    school_name    = row.get('name', 'Escuela sin nombre')
                    n_primaria     = int(row.get('students_primaria',     0) or 0)
                    n_secundaria   = int(row.get('students_secundaria',   0) or 0)
                    n_preparatoria = int(row.get('students_preparatoria', 0) or 0)
            except:
                pass

            # Load encargado
            try:
                enc_data = supabase.table('encargado_escolar').select(
                    'first_name, pat_last_name, mat_last_name'
                ).eq('school_id', school_id).execute()

                if enc_data.data:
                    enc = enc_data.data[0]
                    encargado = f"{enc.get('first_name','')} {enc.get('pat_last_name','')} {enc.get('mat_last_name','')}".strip()
                else:
                    encargado = "No asignado"
            except:
                pass

        st.text_input("🏫 Escuela",    value=school_name, disabled=True)
        st.text_input("👤 Encargado",  value=encargado,   disabled=True)

        # ── Representativeness semáforo ───────────────────────
        st.markdown("---")
        st.markdown("**📐 Representatividad Estadística**")

        rep = assess_sample_representativeness(
            n_respondents        = len(students_df),
            students_primaria    = n_primaria,
            students_secundaria  = n_secundaria,
            students_preparatoria= n_preparatoria,
        )

        # Main semáforo badge
        st.markdown(
            f"<div style='font-size:1.1rem; font-weight:700; padding:6px 0'>"
            f"{rep.semaforo}</div>",
            unsafe_allow_html=True
        )

        # Compact metrics row
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Respondieron",
                rep.n_respondents,
                delta=f"mín. {rep.n_min_required}" if rep.n_min_required else None,
                delta_color="off",
            )
        with m2:
            if rep.n_enrolled:
                st.metric(
                    "Matriculados",
                    rep.n_enrolled,
                    delta=f"{rep.coverage_pct:.1f}% cobertura",
                    delta_color="off",
                )
            else:
                st.metric("Matriculados", "—")

        # Margin of error + caveat
        if not np.isnan(rep.margin_of_error):
            st.markdown(
                f"**Margen de error:** ±{rep.margin_of_error}%  "
                f"(IC 95%)"
            )

        # Enrollment breakdown (only if at least one level has data)
        if rep.n_enrolled:
            with st.expander("Ver desglose por nivel"):
                if n_primaria:
                    st.markdown(f"· Primaria: **{n_primaria}**")
                if n_secundaria:
                    st.markdown(f"· Secundaria: **{n_secundaria}**")
                if n_preparatoria:
                    st.markdown(f"· Preparatoria: **{n_preparatoria}**")
                st.markdown(f"· **Total: {rep.n_enrolled}**")

        # Caveat box — colour-coded by status
        caveat_colors = {
            "REPRESENTATIVA":    ("#d4edda", "#155724"),
            "ACEPTABLE":         ("#fff3cd", "#856404"),
            "INSUFICIENTE":      ("#ffe0b2", "#7f3b00"),
            "NO_REPRESENTATIVA": ("#f8d7da", "#721c24"),
            "SIN_DATOS":         ("#e9ecef", "#495057"),
        }
        bg, fg = caveat_colors.get(rep.status, ("#e9ecef", "#495057"))
        st.markdown(
            f"<div style='background:{bg}; color:{fg}; border-radius:6px; "
            f"padding:8px 10px; font-size:0.82rem; margin-top:6px;'>"
            f"{rep.caveat}</div>",
            unsafe_allow_html=True
        )

        # Hard warning banner in main area if not representative
        if rep.status in ("INSUFICIENTE", "NO_REPRESENTATIVA"):
            st.session_state['rep_warning'] = rep.caveat
            st.session_state['rep_status']  = rep.status
        else:
            st.session_state.pop('rep_warning', None)
            st.session_state.pop('rep_status',  None)

        # ── Analysis info ─────────────────────────────────────
        if url_analysis_dt:
            st.markdown("---")
            st.markdown("**📊 Análisis:**")
            st.code(url_analysis_dt)

        st.markdown("---")
        st.markdown("**📋 Encuesta:** SURVEY_004")
        st.markdown("**💾 BD:** Supabase")

        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()
    
    # ════════════════════════════════════════════════════════════
    # SECTION 1: OVERVIEW (KPIs)
    # ════════════════════════════════════════════════════════════
    
    st.header("📊 1. Panorama General")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Estudiantes", len(students_df))
    
    with col2:
        if 'victimizacion_freq' in students_df.columns:
            prev = calculate_prevalence(students_df['victimizacion_freq'])
            st.metric(
                "Victimización Frecuente",
                f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "",
                delta_color="inverse"
            )
    
    with col3:
        if 'cybervictimizacion_freq' in students_df.columns:
            prev = calculate_prevalence(students_df['cybervictimizacion_freq'])
            st.metric(
                "Cybervictimización",
                f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "",
                delta_color="inverse"
            )
    
    with col4:
        if 'victimizacion_freq' in students_df.columns:
            prev = calculate_prevalence(students_df['victimizacion_freq'])
            
            threshold_emoji = {
                'CRISIS': '🔴',
                'INTERVENCION': '🟠',
                'ATENCION': '🟡',
                'MONITOREO': '🟢',
                'SIN_DATOS': '⚪',
            }
            
            st.metric(
                "Semáforo Victimización",
                f"{threshold_emoji.get(prev.threshold_category, '⚪')} {prev.threshold_category}"
            )

    with col5:
        risk_idx = calculate_risk_index(students_df)
        st.metric(
            "Índice de Riesgo Escolar",
            f"{risk_idx.ssri:.1f} / 100" if not np.isnan(risk_idx.ssri) else "N/A",
            delta=risk_idx.threshold_color,
            delta_color="off",
            help=(
                "Índice compuesto 0–100. Combina factores de riesgo "
                "(victimización, agresión, ciberbullying) ponderados por "
                "factores protectores (autoridad docente, normas grupales, "
                "respuesta institucional). Verde <20 · Amarillo 20-40 · "
                "Naranja 40-60 · Rojo ≥60."
            )
        )

    # Risk Index detail expander
    with st.expander("📐 Detalle del Índice de Riesgo Escolar"):
        col_r, col_p = st.columns(2)
        with col_r:
            st.metric("Componente de Riesgo", f"{risk_idx.risk_component:.1f} / 100")
        with col_p:
            st.metric("Componente Protector", f"{risk_idx.protective_component:.1f} / 100")

        score_rows = []
        for construct, score in risk_idx.construct_scores.items():
            score_rows.append({
                'Constructo': construct,
                'Puntuación Normalizada (0–1)': f"{score:.3f}" if not np.isnan(score) else "—",
            })
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 2: RELIABILITY
    # ════════════════════════════════════════════════════════════
    
    st.header("🔬 2. Calidad y Fiabilidad de Datos")
    
    # Calculate reliability for all constructs
    reliability_results = []
    substantive_constructs = [c for c in get_all_constructs() if c != 'demographic']
    
    for construct in substantive_constructs:
        items = get_construct_items(construct, all_external_ids)
        result = analyze_reliability(answers_df, construct, list(items))
        reliability_results.append(result)
    
    # Reliability table
    rel_data = []
    for r in reliability_results:
        metadata = get_construct_metadata(r.construct)
        rel_data.append({
            'Constructo': metadata.display_name if metadata else r.construct,
            'N Ítems': r.n_items,
            'Cronbach α': f"{r.cronbach_alpha:.3f}" if not np.isnan(r.cronbach_alpha) else "—",
            'McDonald ω': f"{r.mcdonald_omega:.3f}" if not np.isnan(r.mcdonald_omega) else "—",
            'Rango Publicado': f"{r.published_alpha_range[0]:.2f}-{r.published_alpha_range[1]:.2f}" if r.published_alpha_range[0] > 0 else "—",
            'Estado': '✅ Fiable' if r.alpha_meets_threshold else '⚠️ Bajo',
        })
    
    st.dataframe(pd.DataFrame(rel_data), use_container_width=True)
    
    # Reliability comparison chart
    if reliability_results:
        fig = plot_reliability_comparison(reliability_results)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 3: PREVALENCE BY CONSTRUCT
    # ════════════════════════════════════════════════════════════
    
    st.header("📈 3. Prevalencia por Constructo")
    
    # Calculate prevalence for all constructs
    prevalence_data = {}
    for construct in substantive_constructs:
        if f'{construct}_freq' in students_df.columns:
            prev = calculate_prevalence(students_df[f'{construct}_freq'])
            prevalence_data[construct] = prev
    
    # Prevalence chart
    if prevalence_data:
        fig = plot_prevalence_by_construct(prevalence_data)
        st.pyplot(fig)
    
    # Prevalence table
    prev_table = []
    for construct, prev in prevalence_data.items():
        metadata = get_construct_metadata(construct)
        prev_table.append({
            'Constructo': metadata.display_name if metadata else construct,
            'Prevalencia': f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
            'IC 95%': f"{prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "N/A",
            'N Afectados': f"{prev.n_true}/{prev.n_with_data}",
            'Categoría': prev.threshold_category,
        })
    
    st.dataframe(pd.DataFrame(prev_table), use_container_width=True)
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 4: SUBGROUP ANALYSIS
    # ════════════════════════════════════════════════════════════
    
    st.header("👥 4. Análisis por Subgrupos")
    
    # Demographic variable selector
    demo_vars = {
        'genero':         'Género',
        'grado':          'Grado Escolar',
        'edad':           'Edad',
        'lengua_indigena':'Lengua Indígena',
        'orientacion':    'Orientación Sexual',
        'tipo_escuela':   'Tipo de Escuela',
    }
    
    available_vars = {k: v for k, v in demo_vars.items() if k in students_df.columns}
    
    if available_vars:
        tab_bin, tab_cont = st.tabs([
            "📊 Comparación de Prevalencia (Chi-cuadrado)",
            "📈 Comparación de Puntajes Continuos (Mann-Whitney / Kruskal-Wallis)"
        ])

        # ── Tab 1: Binary / Chi-square (original) ──────────────
        with tab_bin:
            selected_demo_bin = st.selectbox(
                "Variable demográfica:",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x],
                key="demo_bin"
            )

            outcome_options_bin = {
                f'{c}_freq': get_construct_metadata(c).display_name 
                for c in substantive_constructs 
                if f'{c}_freq' in students_df.columns and get_construct_metadata(c)
            }

            if outcome_options_bin:
                selected_outcome_bin = st.selectbox(
                    "Indicador de prevalencia:",
                    options=list(outcome_options_bin.keys()),
                    format_func=lambda x: outcome_options_bin[x],
                    key="outcome_bin"
                )

                n_comparisons = len(available_vars) * len(outcome_options_bin)
                comparison = compare_subgroups(
                    students_df,
                    outcome_col=selected_outcome_bin,
                    grouping_col=selected_demo_bin,
                    n_total_tests=n_comparisons
                )

                if comparison:
                    fig = plot_subgroup_comparison(comparison)
                    st.pyplot(fig)

                    st.dataframe(comparison.group_stats, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("χ² (Chi-cuadrado)", f"{comparison.chi2:.2f}")
                    with col2:
                        st.metric("p-value", f"{comparison.p_value:.4f}")
                    with col3:
                        st.metric("V de Cramér", f"{comparison.cramers_v:.2f}")

                    if comparison.is_significant:
                        st.success(f"✅ Diferencia estadísticamente significativa (p < {comparison.bonferroni_alpha:.4f})")
                    else:
                        st.info(f"ℹ️ Diferencia no significativa (p ≥ {comparison.bonferroni_alpha:.4f})")
                else:
                    st.warning("No hay suficientes datos para este análisis")

        # ── Tab 2: Continuous / non-parametric ─────────────────
        with tab_cont:
            st.caption(
                "Usa Mann-Whitney U (2 grupos) o Kruskal-Wallis H + post-hoc de Dunn (≥3 grupos). "
                "Apropiado para puntajes suma de escala ordinal 0–4."
            )

            selected_demo_cont = st.selectbox(
                "Variable demográfica:",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x],
                key="demo_cont"
            )

            outcome_options_cont = {
                f'{c}_sum': get_construct_metadata(c).display_name
                for c in substantive_constructs
                if f'{c}_sum' in students_df.columns and get_construct_metadata(c)
            }

            if outcome_options_cont:
                selected_outcome_cont = st.selectbox(
                    "Puntaje de constructo:",
                    options=list(outcome_options_cont.keys()),
                    format_func=lambda x: outcome_options_cont[x],
                    key="outcome_cont"
                )

                n_comp_cont = len(available_vars) * len(outcome_options_cont)
                cont_result = compare_subgroups_continuous(
                    students_df,
                    outcome_col=selected_outcome_cont,
                    grouping_col=selected_demo_cont,
                    n_total_tests=n_comp_cont,
                )

                if cont_result:
                    # Group stats table
                    gs_rows = []
                    for gs in cont_result.group_stats:
                        gs_rows.append({
                            selected_demo_cont: gs.group,
                            'N':           gs.n,
                            'Mediana':     gs.median,
                            'Media':       gs.mean,
                            'DE':          gs.sd,
                            'Q1':          gs.q1,
                            'Q3':          gs.q3,
                            '% ≥ frec.2':  f"{gs.pct_high:.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(gs_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Statistical results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(cont_result.test_name, f"{cont_result.statistic:.3f}")
                    with col2:
                        st.metric("p-value", f"{cont_result.p_value:.4f}")
                    with col3:
                        st.metric(
                            f"Tamaño del efecto ({cont_result.effect_label})",
                            f"{cont_result.effect_size:.3f}"
                        )
                    with col4:
                        st.metric("Magnitud", cont_result.effect_interp)

                    if cont_result.is_significant:
                        st.success(
                            f"✅ Diferencia significativa (p < {cont_result.bonferroni_alpha:.4f}) — "
                            f"efecto {cont_result.effect_interp}"
                        )
                    else:
                        st.info(f"ℹ️ Diferencia no significativa (p ≥ {cont_result.bonferroni_alpha:.4f})")

                    # Dunn post-hoc (3+ groups only)
                    if cont_result.dunn_matrix is not None:
                        with st.expander("🔬 Post-hoc de Dunn (p-valores ajustados por Bonferroni)"):
                            st.caption("Valores < 0.05 indican diferencia significativa entre el par de grupos.")
                            st.dataframe(
                                cont_result.dunn_matrix.style.background_gradient(
                                    cmap='RdYlGn_r', vmin=0, vmax=0.05
                                ),
                                use_container_width=True
                            )
                else:
                    st.warning("No hay suficientes datos (mínimo 5 estudiantes por grupo)")
    else:
        st.warning("No hay variables demográficas disponibles")
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 5: ITEM-LEVEL ANALYSIS
    # ════════════════════════════════════════════════════════════
    
    st.header("🔍 5. Análisis a Nivel de Ítems")
    
    construct_selector = st.selectbox(
        "Seleccionar constructo:",
        options=substantive_constructs,
        format_func=lambda x: get_construct_metadata(x).display_name if get_construct_metadata(x) else x
    )
    
    if construct_selector:
        items = get_construct_items(construct_selector, all_external_ids)
        item_data = item_descriptives(answers_df, list(items))
        
        if not item_data.empty:
            # Item severity chart
            fig = plot_item_severity_ranking(item_data, construct_selector)
            st.pyplot(fig)
            
            # Item table
            st.dataframe(item_data, use_container_width=True)
        else:
            st.warning("No hay datos de ítems disponibles")
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 6: CONSTRUCT CORRELATIONS
    # ════════════════════════════════════════════════════════════
    
    st.header("🔗 6. Correlaciones entre Constructos (Spearman)")
    
    corr_result = construct_correlation_matrix(students_df, substantive_constructs)
    
    if isinstance(corr_result, tuple):
        corr_matrix, p_matrix = corr_result
    else:
        corr_matrix, p_matrix = corr_result, pd.DataFrame()
    
    if not corr_matrix.empty:
        fig = plot_correlation_heatmap(corr_matrix)
        st.pyplot(fig)
        
        col_rho, col_pval = st.columns(2)
        with col_rho:
            with st.expander("Ver matriz ρ de Spearman"):
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
        with col_pval:
            if not p_matrix.empty:
                with st.expander("Ver p-valores (significancia)"):
                    st.caption("Verde = p < 0.05 (significativo). Rojo = p ≥ 0.05.")
                    st.dataframe(
                        p_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=0.1)
                    )
    else:
        st.warning("No hay suficientes datos para calcular correlaciones")
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 7: ECOLOGY HOTSPOTS
    # ════════════════════════════════════════════════════════════
    
    st.header("🗺️ 7. Ecología del Bullying: Espacios de Riesgo")
    
    st.info("📌 Esta sección analiza ÚNICAMENTE a estudiantes que reportaron victimización.")
    
    # Filter to victims only
    if 'victimizacion_freq' in students_df.columns:
        victims_df = students_df[students_df['victimizacion_freq'] == True]
        
        if len(victims_df) >= 5:
            # Ecology questions
            ecology_items = get_construct_items('ecologia_espacios', all_external_ids)
            
            # Calculate hotspot scores
            hotspot_data = []
            for item in ecology_items:
                item_answers = answers_df[
                    (answers_df['question_id'] == item) &
                    (answers_df['survey_response_id'].isin(victims_df['id']))
                ]
                
                if not item_answers.empty:
                    scores = item_answers['score'].dropna()
                    if len(scores) > 0:
                        # Get question text
                        q_text = item_answers.iloc[0].get('question_text', item)
                        
                        hotspot_data.append({
                            'lugar': str(q_text)[:50],
                            'mean_score': float(scores.mean()),
                            'pct_high': float((scores >= 2).mean() * 100),
                            'n': int(len(scores)),
                        })
            
            if hotspot_data:
                hotspot_data = sorted(hotspot_data, key=lambda x: x['mean_score'], reverse=True)
                
                # Hotspot chart
                fig = plot_ecology_hotspots(hotspot_data)
                st.pyplot(fig)
                
                # Hotspot table
                st.dataframe(pd.DataFrame(hotspot_data), use_container_width=True)
            else:
                st.warning("No hay datos de ecología disponibles")
        else:
            st.warning(f"Insuficientes víctimas para análisis (n={len(victims_df)}, mínimo=5)")
    else:
        st.warning("No hay datos de victimización disponibles")
    
    st.markdown("---")
    
    # ════════════════════════════════════════════════════════════
    # SECTION 8: BULLY-VICTIM TYPOLOGY (Olweus)
    # ════════════════════════════════════════════════════════════
    
    st.header("🎯 8. Tipología Agresor-Víctima (Modelo Olweus)")

    st.info(
        "Clasifica a cada estudiante en uno de cuatro perfiles mutuamente excluyentes "
        "según sus puntajes de victimización y agresión. "
        "Los **Agresores-Víctimas** presentan el mayor riesgo psicosocial."
    )

    if 'victimizacion_sum' in students_df.columns and 'perpetracion_sum' in students_df.columns:
        students_df, typology_result = classify_bully_victim_typology(students_df)

        if typology_result.n_classified > 0:
            # KPI row
            t_cols = st.columns(4)
            profile_order = ['Agresor-Víctima', 'Víctima', 'Agresor', 'No Involucrado']
            profile_icons = {'Agresor-Víctima': '🔴', 'Víctima': '🟠', 'Agresor': '🟡', 'No Involucrado': '🟢'}

            for i, label in enumerate(profile_order):
                pct  = typology_result.percentages.get(label, np.nan)
                ci   = typology_result.ci.get(label, (np.nan, np.nan))
                cnt  = typology_result.counts.get(label, 0)
                with t_cols[i]:
                    st.metric(
                        f"{profile_icons[label]} {label}",
                        f"{pct:.1f}%" if not np.isnan(pct) else "N/A",
                        delta=f"n={cnt} | IC95%: {ci[0]:.1f}-{ci[1]:.1f}" if not np.isnan(pct) else "",
                        delta_color="off"
                    )

            # Summary table
            with st.expander("Ver tabla de clasificación completa"):
                type_rows = []
                for label in profile_order:
                    ci = typology_result.ci.get(label, (np.nan, np.nan))
                    type_rows.append({
                        'Perfil':    label,
                        'N':         typology_result.counts.get(label, 0),
                        '%':         f"{typology_result.percentages.get(label, np.nan):.1f}%",
                        'IC 95%':    f"{ci[0]:.1f}–{ci[1]:.1f}%" if not np.isnan(ci[0]) else "—",
                    })
                st.dataframe(
                    pd.DataFrame(type_rows),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.caption(
                f"Umbral: puntaje medio ≥ 1.0 en escala 0–4. "
                f"N clasificados: {typology_result.n_classified} / {typology_result.n_total}"
            )
        else:
            st.warning("Sin datos suficientes para clasificar perfiles.")
    else:
        st.warning("Se requieren puntajes de victimización y perpetración para este análisis.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SECTION 9: CYBER vs. TRADITIONAL BULLYING OVERLAP
    # ════════════════════════════════════════════════════════════

    st.header("🔀 9. Solapamiento: Bullying Tradicional vs. Cyberbullying")

    st.info(
        "Analiza cuántos estudiantes sufren ambos tipos de victimización simultáneamente. "
        "Un solapamiento alto (φ ≥ 0.30) indica que los programas de intervención "
        "deben abordar ambos contextos de forma conjunta."
    )

    if 'victimizacion_freq' in students_df.columns and 'cybervictimizacion_freq' in students_df.columns:
        overlap = analyze_bullying_overlap(students_df)

        if overlap:
            ov_cols = st.columns(4)
            with ov_cols[0]:
                st.metric("Víctimas Tradicionales", f"{overlap.n_trad_victim}",
                          delta=f"{overlap.n_trad_victim / overlap.n_total * 100:.1f}% del total")
            with ov_cols[1]:
                st.metric("Cibervíctimas", f"{overlap.n_cyber_victim}",
                          delta=f"{overlap.n_cyber_victim / overlap.n_total * 100:.1f}% del total")
            with ov_cols[2]:
                st.metric("Afectados en Ambos", f"{overlap.n_both}",
                          delta=f"{overlap.overlap_pct:.1f}% de víctimas trad.")
            with ov_cols[3]:
                st.metric("Coeficiente φ (phi)", f"{overlap.phi:.3f}",
                          delta=f"χ²={overlap.chi2:.2f}  p={overlap.p_value:.4f}",
                          delta_color="off")

            st.markdown(f"**Índice de Jaccard:** `{overlap.jaccard:.3f}`")
            st.markdown(f"**Interpretación:** {overlap.interpretation}")

            # 2×2 breakdown
            with st.expander("Ver tabla de contingencia"):
                ct_df = pd.DataFrame({
                    '': ['Trad. Víctima ✅', 'Trad. No Víctima ❌'],
                    'Cyber Víctima ✅': [overlap.n_both,
                                        overlap.n_cyber_victim - overlap.n_both],
                    'Cyber No Víctima ❌': [overlap.n_trad_victim - overlap.n_both,
                                           overlap.n_neither],
                })
                st.dataframe(ct_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Sin datos suficientes para el análisis de solapamiento.")
    else:
        st.warning("Se requieren indicadores de victimización y cybervictimización.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SECTION 10: LLM REPORT (previously Section 8)
    # ════════════════════════════════════════════════════════════
    
    st.header("📝 10. Informe para Docentes")
    st.info("🔧 La generación de informes con IA está temporalmente deshabilitada. Próximamente disponible.")
    
    # Footer
    st.markdown("---")
    st.caption("TECH4ZERO-MX v3.0 | Powered by Streamlit + Supabase | SURVEY_004")


# ══════════════════════════════════════════════════════════════
# RUN APP
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
