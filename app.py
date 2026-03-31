"""
TECH4ZERO-MX v3.2 — Streamlit Dashboard
========================================
Statistical analysis dashboard for school climate and bullying survey.

Version: 3.2 (Interactive Plotly Charts)
Survey: SURVEY_004
Platform: Streamlit Cloud / Vercel

Changelog v3.2:
- Replaced all st.pyplot() calls with st.plotly_chart() (interactive)
- New Plotly chart functions (self-contained, no changes to visualization.py):
    plot_prevalence_plotly        → Sec. 3  horizontal bars + CI + semáforo colors
    plot_reliability_plotly       → Sec. 2  grouped bar α vs ω + threshold line
    plot_subgroup_plotly          → Sec. 4  grouped bars by demographic group
    plot_correlation_plotly       → Sec. 6  annotated heatmap (Spearman ρ)
    plot_item_severity_plotly     → Sec. 5  ranked horizontal bars per item
    plot_ecology_hotspots_plotly  → Sec. 7  dual-axis: mean score + % high
    plot_typology_donut_plotly    → Sec. 8  donut chart Olweus profiles

Changelog v3.1:
- Added interactive sidebar filters: género, tipo_escuela, edad, lengua_indígena
- All sections now operate on filtered_df / filtered_answers_df
- Dynamic subsample banner when filters are active
- Filter state preserved in st.session_state
- Safe fallback when demographic columns are absent
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import anthropic

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
    construct_correlation_pvalues,
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

# Supabase configuration
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Missing SUPABASE_URL / SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

    Returns:
        Tuple of (responses_df, answers_df, students_df)
    """
    try:
        survey_result = supabase.table('surveys').select('id').eq('code', SURVEY_CODE).execute()

        if not survey_result.data:
            st.warning(f"Survey {SURVEY_CODE} not found")
            return None, None, None

        survey_id = survey_result.data[0]['id']

        active_surveys = supabase.table('surveys').select('id').eq('is_active', True).execute()

        if not active_surveys.data:
            st.error("❌ No active surveys found in database")
            return None, None, None

        survey_ids = [str(s['id']) for s in active_surveys.data]

        query = supabase.table('survey_responses').select(
            'id, survey_id, school_id, student_external_id, status, analysis_requested_dt'
        ).eq('status', 'submitted')

        if school_id:
            query = query.eq('school_id', school_id)

        if analysis_dt:
            query = query.eq('analysis_requested_dt', analysis_dt)

        query = query.in_('survey_id', survey_ids)

        responses = query.execute()
        responses_df = pd.DataFrame(responses.data)

        if responses_df.empty:
            st.warning("⚠️ No submitted responses found for this analysis")
            return None, None, None

        response_ids = responses_df['id'].tolist()

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

        question_ids = answers_df['question_id'].unique().tolist()
        questions_data = []
        for i in range(0, len(question_ids), chunk_size):
            chunk = question_ids[i:i + chunk_size]
            result = supabase.table('questions').select(
                'id, external_id, question_text'
            ).in_('id', chunk).execute()
            questions_data.extend(result.data)

        questions_df = pd.DataFrame(questions_data)

        answer_ids = answers_df['id'].tolist()
        selected_data = []
        for i in range(0, len(answer_ids), chunk_size):
            chunk = answer_ids[i:i + chunk_size]
            result = supabase.table('answer_selected_options').select(
                'question_answer_id, option_id'
            ).in_('question_answer_id', chunk).execute()
            selected_data.extend(result.data)

        selected_df = pd.DataFrame(selected_data)

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

        answers_clean = answers_df.rename(columns={'id': 'answer_id'})
        questions_clean = questions_df.rename(columns={'id': 'q_id'})

        merged = answers_clean.merge(
            questions_clean,
            left_on='question_id',
            right_on='q_id',
            how='left'
        )

        merged['question_id'] = merged['external_id']
        merged = merged.drop(columns=['q_id', 'external_id'], errors='ignore')

        if not selected_df.empty:
            selected_clean = selected_df.rename(columns={'option_id': 'opt_id'})

            merged = merged.merge(
                selected_clean,
                left_on='answer_id',
                right_on='question_answer_id',
                how='left'
            )

            if not options_df.empty:
                options_clean = options_df.rename(columns={'id': 'option_pk'})

                merged = merged.merge(
                    options_clean,
                    left_on='opt_id',
                    right_on='option_pk',
                    how='left'
                )

        if 'option_code' in merged.columns:
            merged['score'] = pd.to_numeric(merged['option_code'], errors='coerce')
        else:
            merged['score'] = None

        students_df = responses_df[['id', 'school_id']].copy()

        demo_map = {
            'zero_general_genero_v2':       'genero',
            'zero_general_edad_v2':         'edad',
            'zero_general_lengua_v2':       'lengua_indigena',
            'zero_general_tiempo_v2':       'tiempo',
            'zero_general_tipo_escuela_v2': 'tipo_escuela',
        }

        for ext_id, col_name in demo_map.items():
            demo_data = merged[merged['question_id'] == ext_id][
                ['survey_response_id', 'option_text']
            ].drop_duplicates('survey_response_id')

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
        Updated students_df with construct scores appended
    """
    if answers_df is None or answers_df.empty:
        return students_df

    all_constructs = [c for c in get_all_constructs() if c != 'demographic']
    all_external_ids = answers_df['question_id'].unique().tolist()

    for construct in all_constructs:
        construct_items = get_construct_items(construct, all_external_ids)
        construct_items = [item for item in construct_items if not is_global_screener(item)]

        if not construct_items:
            continue

        construct_answers = answers_df[
            answers_df['question_id'].isin(construct_items)
        ][['survey_response_id', 'score']]

        sum_scores = construct_answers.groupby('survey_response_id')['score'].sum().reset_index()
        sum_scores.columns = ['id', f'{construct}_sum']
        students_df = students_df.merge(sum_scores, on='id', how='left')

        mean_scores = construct_answers.groupby('survey_response_id')['score'].mean().reset_index()
        mean_scores[f'{construct}_freq'] = mean_scores['score'] >= 2
        mean_scores = mean_scores[['survey_response_id', f'{construct}_freq']].rename(
            columns={'survey_response_id': 'id'}
        )
        students_df = students_df.merge(mean_scores, on='id', how='left')

    return students_df


# ══════════════════════════════════════════════════════════════
# SIDEBAR FILTERS  (v3.1 — NEW)
# ══════════════════════════════════════════════════════════════

def build_sidebar_filters(students_df: pd.DataFrame) -> dict:
    """
    Render interactive demographic filters in the sidebar.

    Filters are shown only when the corresponding column exists in students_df
    and has at least 2 non-null distinct values.

    Returns:
        dict with keys: genero, tipo_escuela, edades, lengua_indigena
        Values are the selected option(s), or None / [] when filter is inactive.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 Filtros de Análisis**")

    selections = {
        "genero": None,
        "tipo_escuela": None,
        "edades": [],
        "lengua_indigena": None,
    }

    # ── Género ────────────────────────────────────────────────
    if (
        'genero' in students_df.columns
        and students_df['genero'].dropna().nunique() >= 2
    ):
        opciones_genero = ['Todos'] + sorted(
            students_df['genero'].dropna().unique().tolist()
        )
        sel = st.sidebar.selectbox(
            "Género",
            opciones_genero,
            key="filter_genero",
        )
        selections["genero"] = None if sel == 'Todos' else sel

    # ── Tipo / Nivel escolar ──────────────────────────────────
    if (
        'tipo_escuela' in students_df.columns
        and students_df['tipo_escuela'].dropna().nunique() >= 2
    ):
        opciones_nivel = ['Todos'] + sorted(
            students_df['tipo_escuela'].dropna().unique().tolist()
        )
        sel = st.sidebar.selectbox(
            "Nivel escolar",
            opciones_nivel,
            key="filter_nivel",
        )
        selections["tipo_escuela"] = None if sel == 'Todos' else sel

    # ── Edad ─────────────────────────────────────────────────
    if (
        'edad' in students_df.columns
        and students_df['edad'].dropna().nunique() >= 2
    ):
        todas_edades = sorted(students_df['edad'].dropna().unique().tolist())
        sel_edades = st.sidebar.multiselect(
            "Edad",
            options=todas_edades,
            default=todas_edades,
            key="filter_edad",
        )
        # Treat "all selected" same as no filter (avoids empty-list edge case)
        selections["edades"] = (
            [] if set(sel_edades) == set(todas_edades) else sel_edades
        )

    # ── Lengua indígena ───────────────────────────────────────
    if (
        'lengua_indigena' in students_df.columns
        and students_df['lengua_indigena'].dropna().nunique() >= 2
    ):
        opciones_lengua = ['Todos'] + sorted(
            students_df['lengua_indigena'].dropna().unique().tolist()
        )
        sel = st.sidebar.selectbox(
            "Lengua indígena",
            opciones_lengua,
            key="filter_lengua",
        )
        selections["lengua_indigena"] = None if sel == 'Todos' else sel

    return selections


def apply_sidebar_filters(
    students_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    selections: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the filter selections to students_df and answers_df.

    Returns:
        (filtered_students_df, filtered_answers_df)
    """
    mask = pd.Series(True, index=students_df.index)

    if selections.get("genero"):
        mask &= students_df['genero'] == selections["genero"]

    if selections.get("tipo_escuela"):
        mask &= students_df['tipo_escuela'] == selections["tipo_escuela"]

    if selections.get("edades"):          # non-empty list = active filter
        mask &= students_df['edad'].isin(selections["edades"])

    if selections.get("lengua_indigena"):
        mask &= students_df['lengua_indigena'] == selections["lengua_indigena"]

    filtered_students = students_df[mask].copy()

    # Keep only answers that belong to the filtered students
    filtered_response_ids = filtered_students['id'].tolist()
    filtered_answers = answers_df[
        answers_df['survey_response_id'].isin(filtered_response_ids)
    ].copy() if answers_df is not None else answers_df

    return filtered_students, filtered_answers


def show_filter_banner(n_filtered: int, n_total: int, selections: dict) -> None:
    """
    Display a compact sidebar banner when a subsample is active.
    Shows nothing when no filter is applied.
    """
    active = [k for k, v in selections.items() if v]
    if not active:
        return

    active_labels = {
        "genero": "Género",
        "tipo_escuela": "Nivel",
        "edades": "Edad",
        "lengua_indigena": "Lengua",
    }
    tags = ", ".join(active_labels[k] for k in active)

    pct = n_filtered / n_total * 100 if n_total else 0
    st.sidebar.warning(
        f"⚠️ **Submuestra activa** ({tags})\n\n"
        f"**{n_filtered}** de {n_total} estudiantes ({pct:.0f}%)"
    )

    # Also show a non-blocking notice in the main area (once, at the top)
    if not st.session_state.get("_filter_banner_shown"):
        st.info(
            f"🔍 Mostrando submuestra filtrada — **{n_filtered} estudiantes** "
            f"({pct:.0f}% del total). Filtros activos: *{tags}*."
        )
        st.session_state["_filter_banner_shown"] = True


# ══════════════════════════════════════════════════════════════
# PLOTLY CHART FUNCTIONS  (v3.2 — NEW)
# Each function is self-contained. They receive plain Python /
# pandas objects so they can be unit-tested independently.
# ══════════════════════════════════════════════════════════════

# Shared palette — semáforo colors used across charts
_SEMAFORO = {
    'CRISIS':       '#d32f2f',
    'INTERVENCION': '#f57c00',
    'ATENCION':     '#fbc02d',
    'MONITOREO':    '#388e3c',
    'SIN_DATOS':    '#9e9e9e',
}

_PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='sans-serif', size=13),
    margin=dict(l=10, r=10, t=50, b=10),
)


def plot_prevalence_plotly(prevalence_data: dict, get_construct_metadata_fn) -> go.Figure:
    """
    Section 3 — Horizontal bar chart with 95% CI error bars.
    Bars are color-coded by semáforo category.
    """
    rows = []
    for construct, prev in prevalence_data.items():
        meta = get_construct_metadata_fn(construct)
        label = meta.display_name if meta else construct
        pct   = prev.pct if not np.isnan(prev.pct) else 0.0
        rows.append({
            'label':    label,
            'pct':      round(pct, 1),
            'ci_minus': round(pct - prev.ci_lower, 1) if not np.isnan(prev.ci_lower) else 0,
            'ci_plus':  round(prev.ci_upper - pct, 1) if not np.isnan(prev.ci_upper) else 0,
            'color':    _SEMAFORO.get(prev.threshold_category, _SEMAFORO['SIN_DATOS']),
            'cat':      prev.threshold_category,
            'n':        f"{prev.n_true}/{prev.n_with_data}",
        })

    # Sort by prevalence descending
    rows = sorted(rows, key=lambda x: x['pct'], reverse=True)

    fig = go.Figure(go.Bar(
        x=[r['pct'] for r in rows],
        y=[r['label'] for r in rows],
        orientation='h',
        marker_color=[r['color'] for r in rows],
        error_x=dict(
            type='data',
            symmetric=False,
            array=[r['ci_plus'] for r in rows],
            arrayminus=[r['ci_minus'] for r in rows],
            color='#555',
            thickness=1.5,
            width=5,
        ),
        customdata=[[r['cat'], r['n']] for r in rows],
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Prevalencia: <b>%{x:.1f}%</b><br>'
            'Categoría: %{customdata[0]}<br>'
            'N afectados: %{customdata[1]}'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Prevalencia por Constructo (con IC 95%)',
        xaxis=dict(title='% de estudiantes afectados', range=[0, 100]),
        yaxis=dict(autorange='reversed'),
        height=max(350, len(rows) * 42),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_reliability_plotly(reliability_results: list) -> go.Figure:
    """
    Section 2 — Grouped bar: Cronbach α (blue) vs McDonald ω (teal).
    Horizontal threshold line at α = 0.70.
    """
    from construct_definitions import get_construct_metadata as _meta

    labels, alphas, omegas, statuses = [], [], [], []
    for r in reliability_results:
        meta = _meta(r.construct)
        labels.append(meta.display_name if meta else r.construct)
        alphas.append(round(r.cronbach_alpha, 3) if not np.isnan(r.cronbach_alpha) else None)
        omegas.append(round(r.mcdonald_omega, 3) if not np.isnan(r.mcdonald_omega) else None)
        statuses.append('✅' if r.alpha_meets_threshold else '⚠️')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Cronbach α',
        x=labels,
        y=alphas,
        marker_color='#1565c0',
        text=[f'{a:.3f}' if a is not None else '—' for a in alphas],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>α = %{y:.3f}<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        name='McDonald ω',
        x=labels,
        y=omegas,
        marker_color='#00838f',
        text=[f'{o:.3f}' if o is not None else '—' for o in omegas],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>ω = %{y:.3f}<extra></extra>',
    ))

    # Threshold line at 0.70
    fig.add_hline(
        y=0.70,
        line_dash='dash',
        line_color='#d32f2f',
        annotation_text='Umbral mínimo α = 0.70',
        annotation_position='top right',
        annotation_font_color='#d32f2f',
    )

    fig.update_layout(
        title='Confiabilidad por Constructo',
        barmode='group',
        yaxis=dict(title='Coeficiente', range=[0, 1.15]),
        xaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_subgroup_plotly(comparison) -> go.Figure:
    """
    Section 4 — Grouped horizontal bars, one bar per demographic group.
    comparison.group_stats is a DataFrame with columns: group, pct (or mean), n.
    Adapts to both binary (prevalence %) and continuous (mean score) comparisons.
    """
    df = comparison.group_stats.copy() if hasattr(comparison, 'group_stats') else pd.DataFrame()

    if df.empty:
        return go.Figure()

    # Detect column to plot: 'pct' for binary, 'mean' or 'median' for continuous
    value_col  = 'pct'   if 'pct'    in df.columns else \
                 'mean'  if 'mean'   in df.columns else \
                 'median' if 'median' in df.columns else df.columns[1]
    group_col  = df.columns[0]
    x_title    = '% afectados' if value_col == 'pct' else 'Puntaje promedio'

    fig = go.Figure(go.Bar(
        x=df[value_col],
        y=df[group_col].astype(str),
        orientation='h',
        marker_color='#1565c0',
        text=[f'{v:.1f}' for v in df[value_col]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' + x_title + ': %{x:.1f}<extra></extra>',
    ))

    fig.update_layout(
        title='Comparación por Subgrupo',
        xaxis=dict(title=x_title),
        yaxis=dict(autorange='reversed'),
        height=max(300, len(df) * 45),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_correlation_plotly(corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame = None) -> go.Figure:
    """
    Section 6 — Annotated heatmap of Spearman ρ.
    Cells with p ≥ 0.05 (non-significant) get a lighter annotation so
    the viewer's eye is drawn to significant correlations.
    """
    if corr_matrix.empty:
        return go.Figure()

    # Shorten labels for readability on screen
    short = [c.replace('_sum', '').replace('_', ' ').title() for c in corr_matrix.columns]

    z     = corr_matrix.values.tolist()
    annot = []
    for i, row in enumerate(corr_matrix.values):
        annot_row = []
        for j, val in enumerate(row):
            if i == j:
                annot_row.append('')
            else:
                sig = ''
                if p_matrix is not None and not p_matrix.empty:
                    pval = p_matrix.iloc[i, j]
                    sig  = '*' if pval < 0.05 else ''
                annot_row.append(f'{val:.2f}{sig}')
        annot.append(annot_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=short,
        y=short,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=annot,
        texttemplate='%{text}',
        textfont=dict(size=11),
        colorbar=dict(title='ρ', thickness=14),
        hovertemplate='%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title='Correlaciones entre Constructos (Spearman ρ) — * p < 0.05',
        height=520,
        xaxis=dict(tickangle=-35),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_item_severity_plotly(item_data: pd.DataFrame, construct: str) -> go.Figure:
    """
    Section 5 — Horizontal bars ranked by mean score per item.
    Color encodes severity: green → yellow → red.
    """
    if item_data.empty:
        return go.Figure()

    from construct_definitions import get_construct_metadata as _meta
    meta  = _meta(construct)
    title = meta.display_name if meta else construct

    df = item_data.copy().sort_values('mean', ascending=True)

    # Map mean score (0–4) to semáforo color
    def _score_color(m):
        if m >= 2.5: return _SEMAFORO['CRISIS']
        if m >= 1.5: return _SEMAFORO['INTERVENCION']
        if m >= 0.8: return _SEMAFORO['ATENCION']
        return _SEMAFORO['MONITOREO']

    colors = [_score_color(m) for m in df['mean']]

    # Truncate long item labels
    labels = [str(lbl)[:60] + '…' if len(str(lbl)) > 60 else str(lbl)
              for lbl in df.index]

    hover_cols = [c for c in ['mean', 'sd', 'pct_high', 'n'] if c in df.columns]

    fig = go.Figure(go.Bar(
        x=df['mean'],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{m:.2f}' for m in df['mean']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Media: %{x:.2f}<extra></extra>'
        ),
    ))

    fig.update_layout(
        title=f'Severidad de Ítems — {title}',
        xaxis=dict(title='Puntuación media (0–4)', range=[0, 4.5]),
        yaxis=dict(autorange='reversed'),
        height=max(350, len(df) * 40),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_ecology_hotspots_plotly(hotspot_data: list) -> go.Figure:
    """
    Section 7 — Dual-axis: bars = mean score (left), line = % high (right).
    Only victims are included (caller's responsibility).
    """
    if not hotspot_data:
        return go.Figure()

    df = pd.DataFrame(hotspot_data).sort_values('mean_score', ascending=True)

    fig = go.Figure()

    # Bars — mean score
    fig.add_trace(go.Bar(
        name='Puntuación media',
        x=df['mean_score'],
        y=df['lugar'],
        orientation='h',
        marker_color='#7b1fa2',
        opacity=0.75,
        hovertemplate='<b>%{y}</b><br>Media: %{x:.2f}<extra></extra>',
    ))

    # Scatter — % high, on secondary x-axis
    fig.add_trace(go.Scatter(
        name='% Frecuencia alta (≥2)',
        x=df['pct_high'],
        y=df['lugar'],
        mode='markers+text',
        marker=dict(color='#d32f2f', size=10, symbol='diamond'),
        text=[f'{v:.0f}%' for v in df['pct_high']],
        textposition='middle right',
        xaxis='x2',
        hovertemplate='<b>%{y}</b><br>% Alto: %{x:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        title='Espacios de Riesgo (solo víctimas)',
        xaxis=dict(title='Puntuación media (0–4)', range=[0, 4.5], side='bottom'),
        xaxis2=dict(title='% Frecuencia alta', overlaying='x', side='top',
                    range=[0, 110], showgrid=False),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1),
        height=max(350, len(df) * 44),
        **_PLOTLY_LAYOUT,
    )
    return fig


def plot_typology_donut_plotly(typology_result) -> go.Figure:
    """
    Section 8 — Donut chart for Olweus bully-victim profiles.
    Color-coded: red → orange → yellow → green.
    """
    profile_order  = ['Agresor-Víctima', 'Víctima', 'Agresor', 'No Involucrado']
    profile_colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']

    values = [typology_result.counts.get(p, 0) for p in profile_order]
    pcts   = [typology_result.percentages.get(p, 0.0) for p in profile_order]
    cis    = [typology_result.ci.get(p, (0, 0)) for p in profile_order]

    fig = go.Figure(go.Pie(
        labels=profile_order,
        values=values,
        hole=0.48,
        marker=dict(colors=profile_colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'N: %{value}<br>'
            '%{percent}<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Distribución de Perfiles Olweus',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        **_PLOTLY_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════

def main():
    """Main Streamlit application."""

    # Reset per-run session flags
    st.session_state["_filter_banner_shown"] = False

    st.title("🏫 TECH4ZERO-MX v3.2")
    st.markdown("**Encuesta de Clima Escolar, Bullying y Cyberbullying**")
    st.markdown("---")

    # ── URL parameters ────────────────────────────────────────
    params = st.query_params
    url_school_id  = params.get('school_id')
    url_analysis_dt = params.get('analysis_dt')

    if url_school_id:
        try:
            url_school_id = int(url_school_id)
        except Exception:
            url_school_id = None

    if url_analysis_dt:
        st.info(f"📊 Mostrando análisis específico del: {url_analysis_dt}")

    # ── Load raw data ─────────────────────────────────────────
    with st.spinner("Cargando datos..."):
        responses_df, answers_df, students_df = load_survey_data(
            school_id=url_school_id,
            analysis_dt=url_analysis_dt,
        )

    if responses_df is None or answers_df is None or students_df is None:
        st.error("❌ No se pudieron cargar los datos para este análisis.")
        return

    # ── Representativeness warning banner ─────────────────────
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

    # ── Calculate construct scores on FULL data ───────────────
    # Important: scores are calculated on the full dataset BEFORE filtering.
    # This ensures reliability metrics and item descriptives use complete data.
    students_df = calculate_construct_scores(answers_df, students_df)

    all_external_ids = answers_df['question_id'].unique().tolist()
    coverage = validate_construct_coverage(all_external_ids)
    substantive_constructs = [c for c in get_all_constructs() if c != 'demographic']

    # ════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("⚙️ Configuración")

        school_id      = None
        school_name    = "Escuela Secundaria Federal"
        school_country = "MX"   # default; overridden from Supabase
        encargado      = "No disponible"
        n_primaria     = 0
        n_secundaria   = 0
        n_preparatoria = 0

        if 'school_id' in students_df.columns:
            school_id = students_df['school_id'].iloc[0]

            try:
                school_data = supabase.table('schools').select(
                    'name, country, students_primaria, students_secundaria, students_preparatoria'
                ).eq('id', school_id).execute()

                if school_data.data:
                    row = school_data.data[0]
                    school_name    = row.get('name', 'Escuela sin nombre')
                    school_country = row.get('country', 'MX') or 'MX'
                    n_primaria     = int(row.get('students_primaria',     0) or 0)
                    n_secundaria   = int(row.get('students_secundaria',   0) or 0)
                    n_preparatoria = int(row.get('students_preparatoria', 0) or 0)
            except Exception:
                pass

            try:
                enc_data = supabase.table('encargado_escolar').select(
                    'first_name, pat_last_name, mat_last_name'
                ).eq('school_id', school_id).execute()

                if enc_data.data:
                    enc = enc_data.data[0]
                    encargado = (
                        f"{enc.get('first_name','')} "
                        f"{enc.get('pat_last_name','')} "
                        f"{enc.get('mat_last_name','')}".strip()
                    )
                else:
                    encargado = "No asignado"
            except Exception:
                pass

        st.text_input("🏫 Escuela",   value=school_name, disabled=True)
        st.text_input("👤 Encargado", value=encargado,   disabled=True)

        # ── Representativeness semáforo ───────────────────────
        st.markdown("---")
        st.markdown("**📐 Representatividad Estadística**")

        rep = assess_sample_representativeness(
            n_respondents         = len(students_df),
            students_primaria     = n_primaria,
            students_secundaria   = n_secundaria,
            students_preparatoria = n_preparatoria,
        )

        st.markdown(
            f"<div style='font-size:1.1rem; font-weight:700; padding:6px 0'>"
            f"{rep.semaforo}</div>",
            unsafe_allow_html=True,
        )

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

        if not np.isnan(rep.margin_of_error):
            st.markdown(f"**Margen de error:** ±{rep.margin_of_error}%  (IC 95%)")

        if rep.n_enrolled:
            with st.expander("Ver desglose por nivel"):
                if n_primaria:
                    st.markdown(f"· Primaria: **{n_primaria}**")
                if n_secundaria:
                    st.markdown(f"· Secundaria: **{n_secundaria}**")
                if n_preparatoria:
                    st.markdown(f"· Preparatoria: **{n_preparatoria}**")
                st.markdown(f"· **Total: {rep.n_enrolled}**")

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
            unsafe_allow_html=True,
        )

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

        # ── DEMOGRAPHIC FILTERS (v3.1) ────────────────────────
        # Rendered here; applied below after sidebar context closes.
        filter_selections = build_sidebar_filters(students_df)

    # ── Apply filters (outside sidebar context) ───────────────
    filtered_df, filtered_answers_df = apply_sidebar_filters(
        students_df, answers_df, filter_selections
    )

    # Guard: if filters produce an empty dataset, warn and revert to full data
    if filtered_df.empty:
        st.warning(
            "⚠️ Los filtros seleccionados no producen resultados. "
            "Mostrando todos los estudiantes."
        )
        filtered_df      = students_df.copy()
        filtered_answers_df = answers_df.copy()

    # Banner (sidebar + main area) when subsample is active
    show_filter_banner(len(filtered_df), len(students_df), filter_selections)

    # NOTE: Do NOT call calculate_construct_scores() again here.
    # students_df already has all _sum / _freq columns computed above.
    # apply_sidebar_filters() row-filters students_df into filtered_df,
    # so each student's scores are already correct. A second call would
    # produce duplicate columns (_freq_x / _freq_y) that break all
    # prevalence lookups silently.

    all_external_ids_filtered = (
        filtered_answers_df['question_id'].unique().tolist()
        if filtered_answers_df is not None
        else all_external_ids
    )

    st.success(f"✅ Datos cargados: {len(filtered_df)} estudiantes en este análisis")

    # ════════════════════════════════════════════════════════════
    # SECTION 1: OVERVIEW (KPIs)
    # ════════════════════════════════════════════════════════════

    st.header("📊 1. Panorama General")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Estudiantes", len(filtered_df))

    with col2:
        if 'victimizacion_freq' in filtered_df.columns:
            prev = calculate_prevalence(filtered_df['victimizacion_freq'])
            st.metric(
                "Victimización Frecuente",
                f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "",
                delta_color="inverse",
            )

    with col3:
        if 'cybervictimizacion_freq' in filtered_df.columns:
            prev = calculate_prevalence(filtered_df['cybervictimizacion_freq'])
            st.metric(
                "Cybervictimización",
                f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
                delta=f"IC95%: {prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "",
                delta_color="inverse",
            )

    with col4:
        if 'victimizacion_freq' in filtered_df.columns:
            prev = calculate_prevalence(filtered_df['victimizacion_freq'])
            threshold_emoji = {
                'CRISIS':       '🔴',
                'INTERVENCION': '🟠',
                'ATENCION':     '🟡',
                'MONITOREO':    '🟢',
                'SIN_DATOS':    '⚪',
            }
            st.metric(
                "Semáforo Victimización",
                f"{threshold_emoji.get(prev.threshold_category, '⚪')} {prev.threshold_category}",
            )

    with col5:
        risk_idx = calculate_risk_index(filtered_df)
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
            ),
        )

    with st.expander("📐 Detalle del Índice de Riesgo Escolar"):
        col_r, col_p = st.columns(2)
        with col_r:
            st.metric("Componente de Riesgo",    f"{risk_idx.risk_component:.1f} / 100")
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
    # Note: reliability uses FULL data (answers_df) intentionally —
    # psychometric properties should not be recalculated on subsamples.
    # ════════════════════════════════════════════════════════════

    st.header("🔬 2. Calidad y Fiabilidad de Datos")
    st.caption(
        "ℹ️ La fiabilidad se calcula sobre la muestra completa "
        "(no se ve afectada por los filtros activos)."
    )

    reliability_results = []
    for construct in substantive_constructs:
        items = get_construct_items(construct, all_external_ids)
        result = analyze_reliability(answers_df, construct, list(items))
        reliability_results.append(result)

    rel_data = []
    for r in reliability_results:
        metadata = get_construct_metadata(r.construct)
        rel_data.append({
            'Constructo':      metadata.display_name if metadata else r.construct,
            'N Ítems':         r.n_items,
            'Cronbach α':      f"{r.cronbach_alpha:.3f}" if not np.isnan(r.cronbach_alpha) else "—",
            'McDonald ω':      f"{r.mcdonald_omega:.3f}" if not np.isnan(r.mcdonald_omega) else "—",
            'Rango Publicado': f"{r.published_alpha_range[0]:.2f}-{r.published_alpha_range[1]:.2f}" if r.published_alpha_range[0] > 0 else "—",
            'Estado':          '✅ Fiable' if r.alpha_meets_threshold else '⚠️ Bajo',
        })

    st.dataframe(pd.DataFrame(rel_data), use_container_width=True)

    if reliability_results:
        fig = plot_reliability_plotly(reliability_results)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SECTION 3: PREVALENCE BY CONSTRUCT
    # ════════════════════════════════════════════════════════════

    st.header("📈 3. Prevalencia por Constructo")

    prevalence_data = {}
    for construct in substantive_constructs:
        if f'{construct}_freq' in filtered_df.columns:
            prev = calculate_prevalence(filtered_df[f'{construct}_freq'])
            prevalence_data[construct] = prev

    if prevalence_data:
        fig = plot_prevalence_plotly(prevalence_data, get_construct_metadata)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos de prevalencia disponibles.")
        with st.expander("🔧 Diagnóstico (para desarrolladores)"):
            freq_cols = [c for c in filtered_df.columns if c.endswith('_freq')]
            sum_cols  = [c for c in filtered_df.columns if c.endswith('_sum')]
            st.write(f"**Columnas _freq encontradas:** `{freq_cols}`")
            st.write(f"**Columnas _sum encontradas:** `{sum_cols}`")
            st.write(f"**Constructos esperados:** `{substantive_constructs}`")
            st.write(f"**Todas las columnas de filtered_df:** `{filtered_df.columns.tolist()}`")

    prev_table = []
    for construct, prev in prevalence_data.items():
        metadata = get_construct_metadata(construct)
        prev_table.append({
            'Constructo':  metadata.display_name if metadata else construct,
            'Prevalencia': f"{prev.pct:.1f}%" if not np.isnan(prev.pct) else "N/A",
            'IC 95%':      f"{prev.ci_lower:.1f}-{prev.ci_upper:.1f}" if not np.isnan(prev.ci_lower) else "N/A",
            'N Afectados': f"{prev.n_true}/{prev.n_with_data}",
            'Categoría':   prev.threshold_category,
        })

    st.dataframe(pd.DataFrame(prev_table), use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SECTION 4: SUBGROUP ANALYSIS
    # ════════════════════════════════════════════════════════════

    st.header("👥 4. Análisis por Subgrupos")

    demo_vars = {
        'genero':          'Género',
        'grado':           'Grado Escolar',
        'edad':            'Edad',
        'lengua_indigena': 'Lengua Indígena',
        'orientacion':     'Orientación Sexual',
        'tipo_escuela':    'Tipo de Escuela',
    }

    available_vars = {k: v for k, v in demo_vars.items() if k in filtered_df.columns}

    if available_vars:
        tab_bin, tab_cont = st.tabs([
            "📊 Comparación de Prevalencia (Chi-cuadrado)",
            "📈 Comparación de Puntajes Continuos (Mann-Whitney / Kruskal-Wallis)",
        ])

        # ── Tab 1: Binary / Chi-square ────────────────────────
        with tab_bin:
            selected_demo_bin = st.selectbox(
                "Variable demográfica:",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x],
                key="demo_bin",
            )

            outcome_options_bin = {
                f'{c}_freq': get_construct_metadata(c).display_name
                for c in substantive_constructs
                if f'{c}_freq' in filtered_df.columns and get_construct_metadata(c)
            }

            if outcome_options_bin:
                selected_outcome_bin = st.selectbox(
                    "Indicador de prevalencia:",
                    options=list(outcome_options_bin.keys()),
                    format_func=lambda x: outcome_options_bin[x],
                    key="outcome_bin",
                )

                n_comparisons = len(available_vars) * len(outcome_options_bin)
                comparison = compare_subgroups(
                    filtered_df,
                    outcome_col=selected_outcome_bin,
                    grouping_col=selected_demo_bin,
                    n_total_tests=n_comparisons,
                )

                if comparison:
                    fig = plot_subgroup_plotly(comparison)
                    st.plotly_chart(fig, use_container_width=True)
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

        # ── Tab 2: Continuous / non-parametric ───────────────
        with tab_cont:
            st.caption(
                "Usa Mann-Whitney U (2 grupos) o Kruskal-Wallis H + post-hoc de Dunn (≥3 grupos). "
                "Apropiado para puntajes suma de escala ordinal 0–4."
            )

            selected_demo_cont = st.selectbox(
                "Variable demográfica:",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x],
                key="demo_cont",
            )

            outcome_options_cont = {
                f'{c}_sum': get_construct_metadata(c).display_name
                for c in substantive_constructs
                if f'{c}_sum' in filtered_df.columns and get_construct_metadata(c)
            }

            if outcome_options_cont:
                selected_outcome_cont = st.selectbox(
                    "Puntaje de constructo:",
                    options=list(outcome_options_cont.keys()),
                    format_func=lambda x: outcome_options_cont[x],
                    key="outcome_cont",
                )

                n_comp_cont = len(available_vars) * len(outcome_options_cont)
                cont_result = compare_subgroups_continuous(
                    filtered_df,
                    outcome_col=selected_outcome_cont,
                    grouping_col=selected_demo_cont,
                    n_total_tests=n_comp_cont,
                )

                if cont_result:
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
                    st.dataframe(pd.DataFrame(gs_rows), use_container_width=True, hide_index=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(cont_result.test_name, f"{cont_result.statistic:.3f}")
                    with col2:
                        st.metric("p-value", f"{cont_result.p_value:.4f}")
                    with col3:
                        st.metric(
                            f"Tamaño del efecto ({cont_result.effect_label})",
                            f"{cont_result.effect_size:.3f}",
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

                    if cont_result.dunn_matrix is not None:
                        with st.expander("🔬 Post-hoc de Dunn (p-valores ajustados por Bonferroni)"):
                            st.caption("Valores < 0.05 indican diferencia significativa entre el par de grupos.")
                            st.dataframe(
                                cont_result.dunn_matrix.style.background_gradient(
                                    cmap='RdYlGn_r', vmin=0, vmax=0.05
                                ),
                                use_container_width=True,
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
        format_func=lambda x: get_construct_metadata(x).display_name if get_construct_metadata(x) else x,
    )

    if construct_selector:
        # Use full external IDs list — items are defined by the instrument,
        # not by which questions happen to appear in the filtered subsample.
        items = get_construct_items(construct_selector, all_external_ids)
        item_data = item_descriptives(filtered_answers_df, list(items))

        if not item_data.empty:
            fig = plot_item_severity_plotly(item_data, construct_selector)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(item_data, use_container_width=True)
        else:
            st.warning("No hay datos de ítems disponibles")
            with st.expander("🔧 Diagnóstico (para desarrolladores)"):
                st.write(f"Ítems buscados: `{items}`")
                st.write(f"question_ids en filtered_answers_df: `{filtered_answers_df['question_id'].unique().tolist()[:20]}`")
                st.write(f"Columnas en filtered_answers_df: `{filtered_answers_df.columns.tolist()}`")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # SECTION 6: CONSTRUCT CORRELATIONS
    # ════════════════════════════════════════════════════════════

    st.header("🔗 6. Correlaciones entre Constructos (Spearman)")

    corr_matrix = construct_correlation_matrix(filtered_df, substantive_constructs)
    p_matrix    = construct_correlation_pvalues(filtered_df, substantive_constructs)

    if not corr_matrix.empty:
        fig = plot_correlation_plotly(corr_matrix, p_matrix)
        st.plotly_chart(fig, use_container_width=True)

        col_rho, col_pval = st.columns(2)
        with col_rho:
            with st.expander("Ver matriz ρ de Spearman (tabla)"):
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))
        with col_pval:
            if not p_matrix.empty:
                with st.expander("Ver p-valores (tabla)"):
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

    if 'victimizacion_freq' in filtered_df.columns:
        victims_df = filtered_df[filtered_df['victimizacion_freq'] == True]

        if len(victims_df) >= 5:
            ecology_items = get_construct_items('ecologia_espacios', all_external_ids_filtered)

            hotspot_data = []
            for item in ecology_items:
                item_answers = filtered_answers_df[
                    (filtered_answers_df['question_id'] == item) &
                    (filtered_answers_df['survey_response_id'].isin(victims_df['id']))
                ]

                if not item_answers.empty:
                    scores = item_answers['score'].dropna()
                    if len(scores) > 0:
                        q_text = item_answers.iloc[0].get('question_text', item)
                        hotspot_data.append({
                            'lugar':      str(q_text)[:50],
                            'mean_score': float(scores.mean()),
                            'pct_high':   float((scores >= 2).mean() * 100),
                            'n':          int(len(scores)),
                        })

            if hotspot_data:
                hotspot_data = sorted(hotspot_data, key=lambda x: x['mean_score'], reverse=True)
                fig = plot_ecology_hotspots_plotly(hotspot_data)
                st.plotly_chart(fig, use_container_width=True)
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

    if 'victimizacion_sum' in filtered_df.columns and 'perpetracion_sum' in filtered_df.columns:
        filtered_df, typology_result = classify_bully_victim_typology(filtered_df)

        if typology_result.n_classified > 0:
            t_cols = st.columns(4)
            profile_order  = ['Agresor-Víctima', 'Víctima', 'Agresor', 'No Involucrado']
            profile_icons  = {
                'Agresor-Víctima': '🔴',
                'Víctima':         '🟠',
                'Agresor':         '🟡',
                'No Involucrado':  '🟢',
            }

            for i, label in enumerate(profile_order):
                pct = typology_result.percentages.get(label, np.nan)
                ci  = typology_result.ci.get(label, (np.nan, np.nan))
                cnt = typology_result.counts.get(label, 0)
                with t_cols[i]:
                    st.metric(
                        f"{profile_icons[label]} {label}",
                        f"{pct:.1f}%" if not np.isnan(pct) else "N/A",
                        delta=f"n={cnt} | IC95%: {ci[0]:.1f}-{ci[1]:.1f}" if not np.isnan(pct) else "",
                        delta_color="off",
                    )

            # Donut chart (v3.2)
            fig_donut = plot_typology_donut_plotly(typology_result)
            st.plotly_chart(fig_donut, use_container_width=True)

            with st.expander("Ver tabla de clasificación completa"):
                type_rows = []
                for label in profile_order:
                    ci = typology_result.ci.get(label, (np.nan, np.nan))
                    type_rows.append({
                        'Perfil': label,
                        'N':      typology_result.counts.get(label, 0),
                        '%':      f"{typology_result.percentages.get(label, np.nan):.1f}%",
                        'IC 95%': f"{ci[0]:.1f}–{ci[1]:.1f}%" if not np.isnan(ci[0]) else "—",
                    })
                st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)

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

    if 'victimizacion_freq' in filtered_df.columns and 'cybervictimizacion_freq' in filtered_df.columns:
        overlap = analyze_bullying_overlap(filtered_df)

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

            with st.expander("Ver tabla de contingencia"):
                ct_df = pd.DataFrame({
                    '': ['Trad. Víctima ✅', 'Trad. No Víctima ❌'],
                    'Cyber Víctima ✅':    [overlap.n_both,
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
    # SECTION 10: FULL DIRECTOR REPORT (5 chapters + PDF)
    # ════════════════════════════════════════════════════════════

    st.header("📝 10. Informe Completo para el Director")
    st.markdown(
        "Genera un informe ejecutivo de 5 capítulos basado en los datos de la encuesta "
        "y en los manuales del Programa ZERO. El documento puede descargarse como PDF."
    )

    # ── Import report modules ─────────────────────────────────
    from manual_loader import load_category, load_action_plan, get_manual_status
    from report_generator import generate_full_report, markdown_to_pdf, CHAPTERS

    # ── Manual status panel ───────────────────────────────────
    manual_status = get_manual_status()
    total_manuals = sum(v for k, v in manual_status.items() if k != 'plan_de_accion')
    plan_loaded   = manual_status.get('plan_de_accion', 0) == 1

    with st.expander("📚 Estado de los manuales cargados"):
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        labels = {
            'fenomeno':      '🔍 Fenómeno',
            'enfoque':       '🔗 Enfoque',
            'intervencion':  '🛡️ Intervención',
            'prevencion':    '🌱 Prevención',
            'plan_de_accion':'📋 Plan de Acción',
        }
        for col, (key, label) in zip(
            [col_m1, col_m2, col_m3, col_m4, col_m5],
            labels.items()
        ):
            count = manual_status.get(key, 0)
            icon  = "✅" if count > 0 else "⚠️"
            col.metric(label, f"{icon} {count} archivo{'s' if count != 1 else ''}")

        if total_manuals == 0 and not plan_loaded:
            st.warning(
                "⚠️ No se encontraron manuales en /manuales/. "
                "El informe usará solo los datos de la encuesta."
            )

    # ── Country / language context ────────────────────────────
    COUNTRY_CONTEXT = {
        "MX": {
            "idioma":        "español mexicano",
            "pais":          "México",
            "marco":         "Nueva Escuela Mexicana (NEM)",
            "ley":           "Ley General de Educación y protocolos SEP contra el acoso escolar",
            "director_title":"Director(a)",
            "escuela_term":  "plantel",
            "bullying_term": "acoso escolar",
            "saludo":        "Estimado(a) Director(a):",
        },
        "CL": {
            "idioma":        "español chileno",
            "pais":          "Chile",
            "marco":         "Política de Convivencia Educativa del MINEDUC",
            "ley":           "Ley de Violencia Escolar (Ley 20.536) y protocolos MINEDUC",
            "director_title":"Director(a) / Jefe(a) de UTP",
            "escuela_term":  "establecimiento educacional",
            "bullying_term": "acoso escolar",
            "saludo":        "Estimado(a) Director(a):",
        },
        "US": {
            "idioma":        "English",
            "pais":          "United States",
            "marco":         "School Safety and Anti-Bullying Policy",
            "ley":           "applicable federal and state anti-bullying regulations",
            "director_title":"Principal",
            "escuela_term":  "school",
            "bullying_term": "bullying",
            "saludo":        "Dear Principal,",
        },
    }
    country_ctx = COUNTRY_CONTEXT.get(school_country.upper(), COUNTRY_CONTEXT["MX"])

    def _build_report_context(
        school_name, filtered_df, prevalence_data,
        substantive_constructs, get_construct_metadata_fn,
        country_ctx: dict,
    ) -> dict:
        n = len(filtered_df)

        # Prevalence — include n_true, n_total and pct for richer context
        prev_summary = {}
        for construct, prev in prevalence_data.items():
            meta = get_construct_metadata_fn(construct)
            label = meta.display_name if meta else construct
            prev_summary[label] = {
                "pct":       round(prev.pct, 1) if not np.isnan(prev.pct) else None,
                "n_afectados": int(prev.n_true),
                "n_total":   int(prev.n_with_data),
                "categoria": prev.threshold_category,
                "ci_lower":  round(prev.ci_lower, 1) if not np.isnan(prev.ci_lower) else None,
                "ci_upper":  round(prev.ci_upper, 1) if not np.isnan(prev.ci_upper) else None,
            }

        risk_constructs = {k: v for k, v in prev_summary.items() if v["pct"] is not None}
        top3 = sorted(risk_constructs.items(), key=lambda x: x[1]["pct"], reverse=True)[:3]

        # Demographics — include absolute counts too
        demo_summary = {}
        for col, label in [('genero', 'Género'), ('edad', 'Edad'), ('tipo_escuela', 'Nivel escolar')]:
            if col in filtered_df.columns:
                vc_abs  = filtered_df[col].value_counts()
                vc_pct  = filtered_df[col].value_counts(normalize=True).round(3)
                demo_summary[label] = {
                    str(k): {"n": int(vc_abs[k]), "pct": f"{vc_pct[k]*100:.1f}%"}
                    for k in vc_abs.index
                }

        # Typology — include absolute counts
        typology_summary = {}
        if 'bully_victim_type' in filtered_df.columns:
            vc_abs = filtered_df['bully_victim_type'].value_counts()
            vc_pct = filtered_df['bully_victim_type'].value_counts(normalize=True).round(3)
            typology_summary = {
                str(k): {"n": int(vc_abs[k]), "pct": f"{vc_pct[k]*100:.1f}%"}
                for k in vc_abs.index
            }

        # Cyber overlap
        cyber_overlap = None
        if ('victimizacion_freq' in filtered_df.columns and
                'cybervictimizacion_freq' in filtered_df.columns):
            n_trad  = int(filtered_df['victimizacion_freq'].sum())
            n_cyber = int(filtered_df['cybervictimizacion_freq'].sum())
            n_both  = int((filtered_df['victimizacion_freq'] & filtered_df['cybervictimizacion_freq']).sum())
            pct_trad  = round(n_trad / n * 100, 1) if n > 0 else 0
            pct_cyber = round(n_cyber / n * 100, 1) if n > 0 else 0
            pct_both  = round(n_both / n_trad * 100, 1) if n_trad > 0 else 0
            cyber_overlap = {
                "victimas_tradicionales": n_trad,  "pct_tradicionales": pct_trad,
                "cibervictimas":          n_cyber,  "pct_cyber":          pct_cyber,
                "ambos":                  n_both,   "pct_ambos_de_trad":  pct_both,
            }

        # Risk index
        risk_idx = calculate_risk_index(filtered_df)
        risk_summary = {
            "indice":              round(risk_idx.ssri, 1) if not np.isnan(risk_idx.ssri) else None,
            "semaforo":            risk_idx.threshold_color,
            "componente_riesgo":   round(risk_idx.risk_component, 1) if not np.isnan(risk_idx.risk_component) else None,
            "componente_protector":round(risk_idx.protective_component, 1) if not np.isnan(risk_idx.protective_component) else None,
        }

        return {
            "escuela":        school_name,
            "n_estudiantes":  n,
            "prevalencias":   prev_summary,
            "top3_riesgo":    [{"area": k, "pct": v["pct"], "n": v["n_afectados"],
                                "n_total": v["n_total"], "categoria": v["categoria"]}
                               for k, v in top3],
            "demograficos":   demo_summary,
            "tipologia":      typology_summary,
            "cyber_overlap":  cyber_overlap,
            "indice_riesgo":  risk_summary,
            "fecha":          datetime.now().strftime("%d de %B de %Y"),
            "idioma":         country_ctx["idioma"],
            "pais":           country_ctx["pais"],
            "marco":          country_ctx["marco"],
            "ley":            country_ctx["ley"],
            "director_title": country_ctx["director_title"],
            "escuela_term":   country_ctx["escuela_term"],
            "bullying_term":  country_ctx["bullying_term"],
            "saludo":         country_ctx["saludo"],
        }

    # ── Build report context ──────────────────────────────────
    ctx_report = _build_report_context(
        school_name, filtered_df, prevalence_data,
        substantive_constructs, get_construct_metadata,
        country_ctx=country_ctx,
    )
    # Add school_country for report_generator
    ctx_report['school_country'] = school_country

    # ── REPORT MODE ───────────────────────────────────────────
    # DEV: uses Haiku, ~200 words per chapter, minimal cost.
    # Uncomment PROD block and comment DEV block when going to production.

    # DEV ─────────────────────────────────────────────────────
    report_model            = "claude-haiku-4-5"   # ~$0.00
    max_tokens_per_chapter  = 300
    btn_label               = "🧪 Generar Informe (modo prueba)"
    mode_caption            = "Modo prueba — Haiku · ~$0.00 · ~200 palabras por capítulo"

    # PROD ────────────────────────────────────────────────────
    # report_model            = "claude-opus-4-5"   # ~$0.05-0.10 total
    # max_tokens_per_chapter  = 1000
    # btn_label               = "📄 Generar Informe Completo"
    # mode_caption            = "Modo producción — Opus · ~$0.05-0.10 · ~600 palabras por capítulo"
    # ─────────────────────────────────────────────────────────

    st.caption(mode_caption)
    generate_report = st.button(btn_label, type="primary")

    if generate_report:

        # Load manual texts (cached after first load)
        with st.spinner("📖 Cargando manuales..."):
            manual_texts = {
                "fenomeno":     load_category("fenomeno"),
                "enfoque":      load_category("enfoque"),
                "intervencion": load_category("intervencion"),
                "prevencion":   load_category("prevencion"),
                "plan_de_accion": load_action_plan(),
            }

        # Progress tracking
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        chapters_done = []

        def on_chapter_done(chapter_num, chapter_title, text):
            chapters_done.append(text)
            progress_bar.progress(chapter_num / len(CHAPTERS))
            status_text.markdown(
                f"✅ Capítulo {chapter_num} de {len(CHAPTERS)} completado: "
                f"*{chapter_title}*"
            )

        # Generate all 5 chapters
        try:
            client = anthropic.Anthropic(
                api_key=st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
            )
            status_text.markdown(f"⏳ Generando Capítulo 1 de {len(CHAPTERS)}...")

            chapters_list, full_document = generate_full_report(
                ctx=ctx_report,
                manual_texts=manual_texts,
                client=client,
                model=report_model,
                max_tokens_per_chapter=max_tokens_per_chapter,
                progress_callback=on_chapter_done,
            )

            progress_bar.progress(1.0)
            status_text.markdown("✅ Informe completo generado")

            # Preview
            st.markdown("---")
            with st.expander("👁️ Vista previa del informe", expanded=True):
                st.markdown(full_document)
            st.markdown("---")

            # PDF download
            with st.spinner("📄 Generando PDF..."):
                pdf_bytes = markdown_to_pdf(full_document, school_name)

            file_date = datetime.now().strftime("%Y%m%d")
            safe_name = school_name.replace(" ", "_").replace("/", "-")

            st.download_button(
                label="📥 Descargar Informe PDF",
                data=pdf_bytes,
                file_name=f"informe_TECH4ZERO_{safe_name}_{file_date}.pdf",
                mime="application/pdf",
                type="primary",
            )

        except anthropic.AuthenticationError:
            st.error(
                "❌ API key de Anthropic no configurada o inválida. "
                "Agrega ANTHROPIC_API_KEY en los secrets de Streamlit Cloud."
            )
        except Exception as e:
            st.error(f"❌ Error al generar el informe: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.caption("TECH4ZERO-MX v3.2 | Powered by Streamlit + Supabase | SURVEY_004")


# ══════════════════════════════════════════════════════════════
# RUN APP
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
