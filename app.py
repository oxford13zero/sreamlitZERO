# app.py
"""
TECH4ZERO-MX v1.0 — Streamlit Dashboard
========================================
Statistical analysis dashboard for school climate and bullying survey.

Version: 3.0 (Modular Architecture)
Survey: SURVEY_003
LLM: Llama (active) / Claude Sonnet 4.5 (commented, ready to activate)
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
    plot_prevalence_interactive,
)

# LLM import (Llama active, Claude commented)
# ACTIVE: Llama via Groq
try:
    from groq import Groq
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    st.warning("Groq library not installed. LLM reports disabled. Install with: pip install groq")

# COMMENTED: Claude (activate when you create Anthropic account)
# from llm_reporting import generate_teacher_report
# CLAUDE_AVAILABLE = True


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TECH4ZERO-MX Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
from supabase import create_client

# Supabase config
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SURVEY_CODE = "SURVEY_003"


# ══════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"❌ Error connecting to Supabase: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_survey_data():
    """
    Load all survey data from Supabase.
    
    Returns:
        Tuple of (responses_df, answers_df, students_df)
    """
    supabase = get_supabase_client()
    if not supabase:
        return None, None, None
    
    try:
        # Load survey responses
        responses = supabase.table('survey_responses').select(
            'id, survey_id, school_id, student_external_id, status, started_at, submitted_at'
        ).eq('survey_id', 
            supabase.table('surveys').select('id').eq('code', SURVEY_CODE).execute().data[0]['id']
        ).execute()
        
        responses_df = pd.DataFrame(responses.data)
        
        if responses_df.empty:
            st.warning("⚠️ No hay respuestas para SURVEY_003")
            return None, None, None
        
        # Load question answers with question details
        response_ids = responses_df['id'].tolist()
        
        answers = supabase.table('question_answers').select(
            '''
            id,
            survey_response_id,
            question_id,
            questions!inner(external_id, question_text, question_type),
            answer_selected_options!left(
                option_id,
                question_options!inner(option_code, option_text)
            )
            '''
        ).in_('survey_response_id', response_ids).execute()
        
        # Flatten nested structure
        answers_data = []
        for ans in answers.data:
            # Extract selected option (for single_choice questions)
            selected_option = None
            option_code = None
            
            if ans.get('answer_selected_options'):
                opt_data = ans['answer_selected_options'][0]  # First selected option
                if opt_data and 'question_options' in opt_data:
                    selected_option = opt_data['question_options']['option_text']
                    option_code = opt_data['question_options']['option_code']
            
            # Convert option_code to numeric score (0-4 scale)
            score = None
            if option_code is not None:
                try:
                    score = int(option_code) if option_code.isdigit() else None
                except:
                    score = None
            
            answers_data.append({
                'answer_id': ans['id'],
                'survey_response_id': ans['survey_response_id'],
                'question_id': ans['questions']['external_id'],
                'question_text': ans['questions']['question_text'],
                'question_type': ans['questions']['question_type'],
                'selected_option': selected_option,
                'option_code': option_code,
                'score': score,
            })
        
        answers_df = pd.DataFrame(answers_data)
        
        # Create student-level dataframe
        students_df = responses_df.copy()
        
        # Add demographic variables from answers
        demo_questions = {
            'survey_003_zero_general_curso': 'curso',
            'survey_003_zero_general_edad': 'edad',
            'survey_003_zero_general_genero': 'genero',
            'survey_003_zero_general_lengua': 'lengua_indigena',
            'survey_003_zero_general_orientacion': 'orientacion',
            'survey_003_zero_general_tiempo': 'tiempo_escuela',
            'survey_003_zero_general_tipo_escuela': 'tipo_escuela',
        }
        
        for ext_id, var_name in demo_questions.items():
            demo_answers = answers_df[answers_df['question_id'] == ext_id][
                ['survey_response_id', 'selected_option']
            ].rename(columns={'selected_option': var_name})
            
            students_df = students_df.merge(demo_answers, 
                                           left_on='id', 
                                           right_on='survey_response_id', 
                                           how='left')
            students_df = students_df.drop(columns=['survey_response_id'], errors='ignore')
        
        return responses_df, answers_df, students_df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
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
    all_constructs = [c for c in get_all_constructs() if c != 'demographic']
    all_external_ids = answers_df['question_id'].unique().tolist()
    
    for construct in all_constructs:
        # Get items for this construct
        construct_items = get_construct_items(construct, all_external_ids)
        construct_items = [item for item in construct_items if not is_global_screener(item)]
        
        # Calculate sum score
        construct_answers = answers_df[answers_df['question_id'].isin(construct_items)][
            ['survey_response_id', 'score']
        ]
        
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


# Helper import
from construct_definitions import is_global_screener


# ══════════════════════════════════════════════════════════════
# LLM REPORT GENERATION (LLAMA ACTIVE)
# ══════════════════════════════════════════════════════════════

def generate_llm_report_llama(summary_data: dict, school_name: str = "") -> str:
    """
    Generate teacher report using Llama via Groq.
    
    Args:
        summary_data: Dictionary with statistical results
        school_name: School name
    
    Returns:
        Markdown-formatted report
    """
    if not LLAMA_AVAILABLE or not GROQ_API_KEY:
        return "❌ Error: Groq API key not configured. Set GROQ_API_KEY in Secrets."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Build prompt (simplified version of Claude prompt)
        summary_json = json.dumps(summary_data, indent=2, ensure_ascii=False, default=str)
        
        prompt = f"""
Eres un psicólogo educativo experto en prevención del bullying.

Genera un informe técnico para docentes basado en estos datos de la encuesta TECH4ZERO-MX:

ESCUELA: {school_name or 'No especificada'}
DATOS:
{summary_json}

ESTRUCTURA DEL INFORME:
1. Resumen Ejecutivo (3-5 puntos clave)
2. Prevalencia General (con intervalos de confianza)
3. Análisis por Subgrupos
4. Lugares de Riesgo (ecología)
5. Recomendaciones Concretas

REGLAS:
- Usa SOLO los números del JSON
- Siempre reporta intervalos de confianza: "X% (IC95%: Y-Z)"
- Lenguaje formal profesional (ustedes)
- Recomendaciones específicas y accionables
"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un experto en clima escolar y prevención del bullying."},
                {"role": "user", "content": prompt}
            ],
            model=LLAMA_MODEL,
            temperature=0.3,
            max_tokens=4000,
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"❌ Error generating report with Llama: {str(e)}"


# COMMENTED: Claude report generation (activate when ready)
# def generate_llm_report_claude(summary_data: dict, school_name: str = "") -> str:
#     """
#     Generate teacher report using Claude Sonnet 4.5.
#     
#     Args:
#         summary_data: Dictionary with statistical results
#         school_name: School name
#     
#     Returns:
#         Markdown-formatted report
#     """
#     return generate_teacher_report(
#         summary_data=summary_data,
#         school_name=school_name,
#         api_key=ANTHROPIC_API_KEY
#     )


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
        
        school_name = st.text_input(
            "Nombre de la Escuela",
            value="Escuela Secundaria Federal",
            help="Aparecerá en el informe generado"
        )
        
        st.markdown("---")
        st.markdown("**📊 Encuesta:** SURVEY_003")
        st.markdown("**🤖 LLM Activo:** Llama 3.3 70B")
        st.markdown("**💾 Base de Datos:** Supabase")
        
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Cargando datos..."):
        responses_df, answers_df, students_df = load_survey_data()
    
    if responses_df is None or answers_df is None or students_df is None:
        st.error("❌ No se pudieron cargar los datos. Verifica la configuración de Supabase.")
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
                f"{prev.pct:.1f}%",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}",
                delta_color="inverse"
            )
    
    with col3:
        if 'cybervictimizacion_freq' in students_df.columns:
            prev = calculate_prevalence(students_df['cybervictimizacion_freq'])
            st.metric(
                "Cybervictimización",
                f"{prev.pct:.1f}%",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}",
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
            'Estado': 'Fiable' if r.alpha_meets_threshold else 'Bajo',
        })
    
    st.dataframe(pd.DataFrame(rel_data), use_container_width=True)
    
    # Reliability comparison chart
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
    fig = plot_prevalence_by_construct(prevalence_data)
    st.pyplot(fig)
    
    # Prevalence table
    prev_table = []
    for construct, prev in prevalence_data.items():
        metadata = get_construct_metadata(construct)
        prev_table.append({
            'Constructo': metadata.display_name if metadata else construct,
            'Prevalencia': f"{prev.pct:.1f}%",
            'IC 95%': f"{prev.ci_lower:.1f}-{prev.ci_upper:.1f}",
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
        outcome_options = {f'{c}_freq': get_construct_metadata(c).display_name 
                          for c in substantive_constructs 
                          if f'{c}_freq' in students_df.columns}
        
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
                st.success(f" Diferencia estadísticamente significativa (p < {comparison.bonferroni_alpha:.4f})")
            else:
                st.info(f" Diferencia no significativa (p ≥ {comparison.bonferroni_alpha:.4f})")
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
                        hotspot_data.append({
                            'lugar': item_answers.iloc[0]['question_text'][:50],
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
    # SECTION 8: GENERATE TEACHER REPORT
    # ════════════════════════════════════════════════════════════
    
    st.header("📝 8. Generar Informe para Docentes")
    
    st.markdown("""
    Este informe generado por IA proporciona:
    - Análisis estadístico detallado en lenguaje accesible
    - Interpretación de resultados con contexto educativo
    - Recomendaciones concretas y accionables
    - Referencias a investigación científica
    """)
    
    if st.button("🤖 Generar Informe con IA", type="primary"):
        with st.spinner("Generando informe... (esto puede tomar 30-60 segundos)"):
            
            # Prepare summary data
            summary_data = {
                'escuela': school_name,
                'n_estudiantes': len(students_df),
                'fecha_aplicacion': datetime.now().strftime("%Y-%m-%d"),
                'grados_incluidos': students_df['curso'].unique().tolist() if 'curso' in students_df.columns else [],
                
                # Reliability
                'fiabilidad_escalas': {
                    r.construct: {
                        'cronbach_alpha': float(r.cronbach_alpha) if not np.isnan(r.cronbach_alpha) else None,
                        'mcdonald_omega': float(r.mcdonald_omega) if not np.isnan(r.mcdonald_omega) else None,
                        'n_items': r.n_items,
                        'rango_publicado': list(r.published_alpha_range),
                        'fiable': r.alpha_meets_threshold,
                    }
                    for r in reliability_results
                },
                
                # Prevalence
                'prevalencias': {
                    construct: {
                        'pct': float(prev.pct) if not np.isnan(prev.pct) else None,
                        'ci_lower': float(prev.ci_lower) if not np.isnan(prev.ci_lower) else None,
                        'ci_upper': float(prev.ci_upper) if not np.isnan(prev.ci_upper) else None,
                        'n_with_data': prev.n_with_data,
                        'n_true': prev.n_true,
                        'threshold': prev.threshold_category,
                    }
                    for construct, prev in prevalence_data.items()
                },
                
                # Subgroups (simplified - just victimization by gender as example)
                'subgrupos': {},
                
                # Ecology
                'ecologia_hotspots': hotspot_data if 'hotspot_data' in locals() else [],
                
                # Missing data
                'datos_faltantes': missing_pattern_summary(students_df),
            }
            
            # Generate report with LLAMA (active)
            report = generate_llm_report_llama(summary_data, school_name)
            
            # COMMENTED: Generate with Claude (activate when ready)
            # report = generate_llm_report_claude(summary_data, school_name)
            
            # Display report
            st.markdown("---")
            st.markdown("## 📄 Informe Generado")
            st.markdown(report)
            
            # Download button
            st.download_button(
                label="📥 Descargar Informe (Markdown)",
                data=report,
                file_name=f"informe_{school_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )


# ══════════════════════════════════════════════════════════════
# RUN APP
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()








