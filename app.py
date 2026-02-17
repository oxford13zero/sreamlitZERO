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
    bonferroni_threshold,
    item_descriptives,
    construct_correlation_matrix,
    missing_pattern_summary,
    run_cfa,
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
SURVEY_CODE = "SURVEY_003"


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_survey_data():
    """Load all survey data from Supabase."""
    try:
        # 1. Get survey ID
        survey_result = supabase.table('surveys').select('id').eq('code', SURVEY_CODE).execute()
        
        if not survey_result.data:
            st.warning(f"⚠️ Survey {SURVEY_CODE} not found")
            return None, None, None
        
        survey_id = survey_result.data[0]['id']
        
        # 2. Load responses (submitted only)
        responses = supabase.table('survey_responses').select(
            'id, survey_id, school_id, student_external_id, status'
        ).eq('survey_id', survey_id).eq('status', 'submitted').execute()
        
        responses_df = pd.DataFrame(responses.data)
        
        if responses_df.empty:
            st.warning(f"⚠️ No submitted responses for {SURVEY_CODE}")
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
            'survey_003_zero_general_curso': 'curso',
            'survey_003_zero_general_genero': 'genero',
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
    
    # Header
    st.title("🏫 TECH4ZERO-MX v1.0")
    st.markdown("**Encuesta de Clima Escolar, Bullying y Cyberbullying**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        # Get school info from database
        #if students_df is not None and not students_df.empty and 'school_id' in students_df.columns:
        #    school_id = students_df['school_id'].iloc[0]
        #else:
        #    school_name = "Escuela Secundaria Federal"
        #    encargado = "No disponible"   


        
        school_name = st.text_input(
            "Nombre de la Escuela",
            value="Escuela Secundaria Federal",
            help="Aparecerá en reportes (cuando se habilite)"
        )








        
        st.markdown("---")
        st.markdown("**📊 Encuesta:** SURVEY_003")
        st.markdown("**💾 Base de Datos:** Supabase")
        st.markdown("**📈 Versión:** 3.0")
        
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Cargando datos..."):
        responses_df, answers_df, students_df = load_survey_data()
    
    if responses_df is None or answers_df is None or students_df is None:
        st.error("❌ No se pudieron cargar los datos. Verifica la configuración.")
        return
    
    # Calculate construct scores
    students_df = calculate_construct_scores(answers_df, students_df)
    
    # Get all external IDs
    all_external_ids = answers_df['question_id'].unique().tolist()
    
    # Validate coverage
    coverage = validate_construct_coverage(all_external_ids)
    
    # ════════════════════════════════════════════════════════════
    # SECTION 1: OVERVIEW (KPIs)
    # ════════════════════════════════════════════════════════════
    
    st.header("📊 1. Panorama General")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
                "Semáforo",
                f"{threshold_emoji.get(prev.threshold_category, '⚪')} {prev.threshold_category}"
            )
    
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
        'genero': 'Género',
        'curso': 'Curso/Grado',
        'lengua_indigena': 'Lengua Indígena',
        'orientacion': 'Orientación Sexual',
        'tipo_escuela': 'Tipo de Escuela',
    }
    
    available_vars = {k: v for k, v in demo_vars.items() if k in students_df.columns}
    
    if available_vars:
        selected_demo = st.selectbox(
            "Seleccionar variable demográfica:",
            options=list(available_vars.keys()),
            format_func=lambda x: available_vars[x]
        )
        
        # Outcome selector
        outcome_options = {
            f'{c}_freq': get_construct_metadata(c).display_name 
            for c in substantive_constructs 
            if f'{c}_freq' in students_df.columns and get_construct_metadata(c)
        }
        
        if outcome_options:
            selected_outcome = st.selectbox(
                "Seleccionar indicador:",
                options=list(outcome_options.keys()),
                format_func=lambda x: outcome_options[x]
            )
            
            # Calculate comparison
            n_comparisons = len(available_vars) * len(outcome_options)
            
            comparison = compare_subgroups(
                students_df,
                outcome_col=selected_outcome,
                grouping_col=selected_demo,
                n_total_tests=n_comparisons
            )
            
            if comparison:
                # Comparison chart
                fig = plot_subgroup_comparison(comparison)
                st.pyplot(fig)
                
                # Detailed table
                st.dataframe(comparison.group_stats, use_container_width=True)
                
                # Statistical summary
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
    
    st.header("🔗 6. Correlaciones entre Constructos")
    
    corr_matrix = construct_correlation_matrix(students_df, substantive_constructs)
    
    if not corr_matrix.empty:
        fig = plot_correlation_heatmap(corr_matrix)
        st.pyplot(fig)
        
        with st.expander("Ver matriz de correlaciones"):
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
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
    # SECTION 8: LLM REPORT (DISABLED FOR NOW)
    # ════════════════════════════════════════════════════════════
    
    st.header("📝 8. Informe para Docentes")
    st.info("🔧 La generación de informes con IA está temporalmente deshabilitada. Próximamente disponible.")
    
    # Footer
    st.markdown("---")
    st.caption("TECH4ZERO-MX v3.0 | Powered by Streamlit + Supabase")


# ══════════════════════════════════════════════════════════════
# RUN APP
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()


