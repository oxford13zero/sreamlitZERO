# visualization.py
"""
TECH4ZERO-MX v1.0 — Visualization Module
=========================================
Generates charts and graphs for Streamlit dashboard.

All functions return matplotlib or plotly figures that can be displayed
in Streamlit using st.pyplot() or st.plotly_chart().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings

# Optional: Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive charts disabled. Install with: pip install plotly")

from stats_engine import ReliabilityResult, PrevalenceResult, SubgroupComparison
from construct_definitions import CONSTRUCT_METADATA


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Color scheme (professional, accessible)
COLORS = {
    'primary': '#1f77b4',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'success': '#2ca02c',
    'info': '#17becf',
    'neutral': '#7f7f7f',
}

RISK_COLORS = {
    'CRISIS': '#d62728',
    'INTERVENCION': '#ff7f0e',
    'ATENCION': '#ffd700',
    'MONITOREO': '#2ca02c',
    'SIN_DATOS': '#cccccc',
}

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# ══════════════════════════════════════════════════════════════
# RELIABILITY VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_reliability_comparison(
    reliability_results: List[ReliabilityResult],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot Cronbach's α vs McDonald's ω with published benchmark ranges.
    
    Args:
        reliability_results: List of ReliabilityResult objects
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure
    """
    # Filter out demographic (no reliability expected)
    results = [r for r in reliability_results if r.construct != 'demographic']
    
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No reliability data available', 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Prepare data
    constructs = [r.construct for r in results]
    alphas = [r.cronbach_alpha if not np.isnan(r.cronbach_alpha) else 0 for r in results]
    omegas = [r.mcdonald_omega if not np.isnan(r.mcdonald_omega) else 0 for r in results]
    
    # Published ranges
    pub_mins = [r.published_alpha_range[0] for r in results]
    pub_maxs = [r.published_alpha_range[1] for r in results]
    
    x = np.arange(len(constructs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, alphas, width, label='Cronbach α (actual)', 
                   color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, omegas, width, label='McDonald ω (actual)', 
                   color=COLORS['info'], alpha=0.8)
    
    # Plot benchmark ranges as error bars (only for constructs with published ranges)
    for i, r in enumerate(results):
        if r.published_alpha_range[0] > 0:  # Has published benchmark
            ax.plot([i, i], r.published_alpha_range, 
                   color='gray', linewidth=2, marker='_', markersize=10,
                   label='Rango publicado' if i == 0 else '')
    
    # Threshold line
    ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1, 
               label='Umbral mínimo (α≥0.60)', alpha=0.7)
    ax.axhline(y=0.70, color='orange', linestyle='--', linewidth=1,
               label='Umbral ω (ω≥0.70)', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Constructo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coeficiente de Fiabilidad', fontsize=12, fontweight='bold')
    ax.set_title('Fiabilidad de Escalas: α vs ω (con rangos publicados)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in constructs], 
                       rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# PREVALENCE VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_prevalence_by_construct(
    prevalence_data: Dict[str, PrevalenceResult],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Bar chart of prevalence by construct with confidence intervals.
    
    Args:
        prevalence_data: Dict mapping construct → PrevalenceResult
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if not prevalence_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No prevalence data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    # Prepare data
    constructs = []
    prevalences = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    for construct, result in prevalence_data.items():
        if construct == 'demographic':
            continue
        
        constructs.append(construct.replace('_', ' ').title())
        prevalences.append(result.pct if not np.isnan(result.pct) else 0)
        ci_lowers.append(result.ci_lower if not np.isnan(result.ci_lower) else 0)
        ci_uppers.append(result.ci_upper if not np.isnan(result.ci_upper) else 0)
        colors.append(RISK_COLORS.get(result.threshold_category, COLORS['neutral']))
    
    # Calculate error bars
    errors = [[prevalences[i] - ci_lowers[i], ci_uppers[i] - prevalences[i]] 
              for i in range(len(prevalences))]
    errors = np.array(errors).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(constructs))
    bars = ax.barh(y_pos, prevalences, xerr=errors, color=colors, 
                   alpha=0.8, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Threshold lines
    ax.axvline(x=5, color=RISK_COLORS['ATENCION'], linestyle='--', 
               linewidth=1.5, label='Atención (5%)', alpha=0.7)
    ax.axvline(x=10, color=RISK_COLORS['INTERVENCION'], linestyle='--',
               linewidth=1.5, label='Intervención (10%)', alpha=0.7)
    ax.axvline(x=20, color=RISK_COLORS['CRISIS'], linestyle='--',
               linewidth=1.5, label='Crisis (20%)', alpha=0.7)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constructs)
    ax.set_xlabel('Prevalencia (%) con IC 95%', fontsize=12, fontweight='bold')
    ax.set_title('Prevalencia de Factores de Riesgo por Constructo',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(prevalences + [25]) * 1.1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SUBGROUP VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_subgroup_comparison(
    comparison: SubgroupComparison,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot subgroup comparison with statistical significance indicators.
    
    Args:
        comparison: SubgroupComparison object
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    df = comparison.group_stats.copy()
    
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No subgroup data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    groups = df[comparison.grouping_var].astype(str)
    prevalences = df['pct']
    ci_lowers = df['ci_lower']
    ci_uppers = df['ci_upper']
    
    # Error bars
    errors = [[prevalences.iloc[i] - ci_lowers.iloc[i], 
               ci_uppers.iloc[i] - prevalences.iloc[i]] 
              for i in range(len(prevalences))]
    errors = np.array(errors).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, prevalences, yerr=errors, 
                  color=COLORS['primary'], alpha=0.7,
                  error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 5})
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel('Prevalencia (%) con IC 95%', fontsize=12, fontweight='bold')
    ax.set_xlabel(comparison.grouping_var.replace('_', ' ').title(), 
                  fontsize=12, fontweight='bold')
    
    # Statistical significance in title
    sig_text = ""
    if comparison.is_significant:
        sig_text = f" (χ²={comparison.chi2:.2f}, p={comparison.p_value:.4f}, V={comparison.cramers_v:.2f}) ✓ SIGNIFICATIVO"
    else:
        sig_text = f" (χ²={comparison.chi2:.2f}, p={comparison.p_value:.4f}, V={comparison.cramers_v:.2f}) — No significativo"
    
    title = f"{comparison.outcome_var.replace('_', ' ').title()} por {comparison.grouping_var.replace('_', ' ').title()}"
    ax.set_title(title + sig_text, fontsize=12, fontweight='bold', pad=20)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# CORRELATION VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Correlation heatmap for construct scores.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if corr_matrix.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No correlation data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlación de Pearson'},
        ax=ax
    )
    
    ax.set_title('Matriz de Correlaciones entre Constructos',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Clean up labels
    labels = [label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# ITEM-LEVEL VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_item_severity_ranking(
    item_data: pd.DataFrame,
    construct_name: str,
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Horizontal bar chart of items ranked by % high frequency.
    
    Args:
        item_data: DataFrame from item_descriptives()
        construct_name: Name of construct for title
        top_n: Number of top items to show
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if item_data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No item data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    # Take top N items
    df = item_data.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['pct_high_freq'], color=COLORS['danger'], alpha=0.7)
    
    # Color code by severity
    for i, (idx, row) in enumerate(df.iterrows()):
        pct = row['pct_high_freq']
        if pct >= 20:
            bars[i].set_color(RISK_COLORS['CRISIS'])
        elif pct >= 10:
            bars[i].set_color(RISK_COLORS['INTERVENCION'])
        elif pct >= 5:
            bars[i].set_color(RISK_COLORS['ATENCION'])
        else:
            bars[i].set_color(RISK_COLORS['MONITOREO'])
    
    # Format labels
    labels = [text[:60] + '...' if len(text) > 60 else text 
              for text in df['question_text']]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('% Estudiantes con Frecuencia Alta (≥mensual)', 
                  fontsize=11, fontweight='bold')
    ax.set_title(f'Ítems Más Severos: {construct_name.replace("_", " ").title()}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['pct_high_freq'] + 0.5, i, f"{row['pct_high_freq']:.1f}%",
                va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# ECOLOGY VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_ecology_hotspots(
    hotspot_data: List[Dict],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Bar chart of ecology hotspots ranked by mean score.
    
    Args:
        hotspot_data: List of dicts with keys: lugar, mean_score, pct_high, n
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if not hotspot_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No ecology data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    df = pd.DataFrame(hotspot_data).head(8)  # Top 8 locations
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['mean_score'], color=COLORS['danger'], alpha=0.7)
    
    # Color by severity
    for i, row in df.iterrows():
        score = row['mean_score']
        if score >= 3:
            bars[i].set_color(RISK_COLORS['CRISIS'])
        elif score >= 2:
            bars[i].set_color(RISK_COLORS['INTERVENCION'])
        elif score >= 1:
            bars[i].set_color(RISK_COLORS['ATENCION'])
        else:
            bars[i].set_color(RISK_COLORS['MONITOREO'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['lugar'], fontsize=10)
    ax.set_xlabel('Puntuación Media (0-4)', fontsize=11, fontweight='bold')
    ax.set_title('Lugares Más Inseguros (Reportados por Víctimas)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 4)
    ax.grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, row in df.iterrows():
        ax.text(row['mean_score'] + 0.1, i, 
                f"{row['mean_score']:.2f} ({row['pct_high']:.1f}%, n={row['n']})",
                va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY CHARTS (Optional)
# ══════════════════════════════════════════════════════════════

def plot_prevalence_interactive(prevalence_data: Dict[str, PrevalenceResult]) -> Optional[go.Figure]:
    """
    Interactive prevalence chart with Plotly (if available).
    
    Returns:
        Plotly figure or None if plotly not installed
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    constructs = []
    prevalences = []
    ci_lowers = []
    ci_uppers = []
    categories = []
    
    for construct, result in prevalence_data.items():
        if construct == 'demographic':
            continue
        constructs.append(construct.replace('_', ' ').title())
        prevalences.append(result.pct if not np.isnan(result.pct) else 0)
        ci_lowers.append(result.ci_lower if not np.isnan(result.ci_lower) else 0)
        ci_uppers.append(result.ci_upper if not np.isnan(result.ci_upper) else 0)
        categories.append(result.threshold_category)
    
    df = pd.DataFrame({
        'Constructo': constructs,
        'Prevalencia': prevalences,
        'CI_Lower': ci_lowers,
        'CI_Upper': ci_uppers,
        'Categoría': categories,
    })
    
    fig = px.bar(
        df,
        x='Prevalencia',
        y='Constructo',
        orientation='h',
        color='Categoría',
        color_discrete_map=RISK_COLORS,
        error_x=[df['CI_Upper'] - df['Prevalencia']],
        error_x_minus=[df['Prevalencia'] - df['CI_Lower']],
        title='Prevalencia por Constructo (Interactivo)',
        labels={'Prevalencia': 'Prevalencia (%) con IC 95%'},
    )
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig


if __name__ == '__main__':
    # Self-test
    print("visualization.py loaded successfully")
    print(f"Plotly available: {PLOTLY_AVAILABLE}")
    
    # Test color scheme
    print("\nColor scheme:")
    for name, color in COLORS.items():
        print(f"  {name}: {color}")
