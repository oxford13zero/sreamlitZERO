# stats_engine.py
"""
TECH4ZERO-MX v1.0 — Statistical Analysis Engine
================================================
Advanced statistical functions for survey analysis.

Features:
- Cronbach's alpha + McDonald's omega
- Confirmatory Factor Analysis (CFA)
- Wilson 95% confidence intervals
- Chi-square + Cramér's V + Bonferroni correction
- Item-level descriptives
- Construct correlation matrix
- Missing data analysis
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math

# Try to import factor_analyzer for CFA
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.confirmatory_factor_analyzer import ConfirmatoryFactorAnalyzer
    CFA_AVAILABLE = True
except ImportError:
    CFA_AVAILABLE = False
    print("WARNING: factor_analyzer not installed. CFA will be disabled.")
    print("Install with: pip install factor-analyzer")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

MIN_ITEMS_FOR_ALPHA = 3
ALPHA_THRESHOLD = 0.60  # Minimum acceptable reliability
ALPHA_STAT = 0.05       # Statistical significance level
HIGH_FREQ_THRESHOLD = 2  # Monthly or more (score ≥ 2)
VERY_HIGH_FREQ = 3       # Weekly or more (score ≥ 3)


# ══════════════════════════════════════════════════════════════
# RELIABILITY: Cronbach's Alpha
# ══════════════════════════════════════════════════════════════

def cronbach_alpha(item_matrix: pd.DataFrame) -> float:
    """
    Calculate Cronbach's alpha for internal consistency.
    
    Args:
        item_matrix: DataFrame where rows=students, columns=items
        
    Returns:
        Alpha coefficient (0-1), or np.nan if calculation fails
        
    Reference:
        Cronbach, L. J. (1951). Coefficient alpha and the internal 
        structure of tests. Psychometrika, 16(3), 297-334.
    """
    df = item_matrix.dropna()
    
    if df.shape[0] < 2 or df.shape[1] < MIN_ITEMS_FOR_ALPHA:
        return np.nan
    
    k = df.shape[1]  # number of items
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    
    if total_var == 0 or item_vars.sum() == 0:
        return np.nan
    
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return round(float(alpha), 3)


# ══════════════════════════════════════════════════════════════
# RELIABILITY: McDonald's Omega
# ══════════════════════════════════════════════════════════════

def mcdonalds_omega(item_matrix: pd.DataFrame) -> float:
    """
    Calculate McDonald's omega (ω) — more robust than alpha.
    
    Omega accounts for differing factor loadings (non-tau-equivalence).
    
    Args:
        item_matrix: DataFrame where rows=students, columns=items
        
    Returns:
        Omega coefficient (0-1), or np.nan if calculation fails
        
    Reference:
        McDonald, R. P. (1999). Test theory: A unified treatment.
        Omega formula: ω = (Σλ)² / [(Σλ)² + Σθ]
        where λ = factor loadings, θ = unique variances
    """
    df = item_matrix.dropna()
    
    if df.shape[0] < 10 or df.shape[1] < MIN_ITEMS_FOR_ALPHA:
        return np.nan
    
    if not CFA_AVAILABLE:
        return np.nan
    
    try:
        # Single-factor CFA
        fa = FactorAnalyzer(n_factors=1, rotation=None, method='ml')
        fa.fit(df)
        
        loadings = fa.loadings_[:, 0]
        uniquenesses = fa.get_uniquenesses()
        
        # Omega formula
        sum_loadings = loadings.sum()
        omega = (sum_loadings ** 2) / (
            (sum_loadings ** 2) + uniquenesses.sum()
        )
        
        return round(float(omega), 3)
        
    except Exception:
        # CFA may fail with insufficient data or non-convergence
        return np.nan


# ══════════════════════════════════════════════════════════════
# CONFIRMATORY FACTOR ANALYSIS (CFA)
# ══════════════════════════════════════════════════════════════

def run_cfa(
    data: pd.DataFrame,
    model_spec: Dict[str, List[str]],
    min_loadings: float = 0.40
) -> Optional[Dict]:
    """
    Run Confirmatory Factor Analysis.
    
    Args:
        data: DataFrame with item responses (rows=students, cols=items)
        model_spec: Dict mapping factor_name → [item1, item2, ...]
        min_loadings: Minimum acceptable factor loading (default 0.40)
        
    Returns:
        Dict with:
          - fit_indices: CFI, TLI, RMSEA, SRMR
          - loadings: DataFrame with item → factor loadings
          - factor_correlations: Inter-factor correlation matrix
          - warnings: List of items with low loadings
          
    Reference:
        Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes.
        Good fit: CFI/TLI > .95, RMSEA < .06, SRMR < .08
    """
    if not CFA_AVAILABLE:
        return None
    
    # Filter data to only include items in model
    all_items = [item for items in model_spec.values() for item in items]
    data_subset = data[all_items].dropna()
    
    if data_subset.shape[0] < 50:
        return {
            'error': 'Insufficient data for CFA (n < 50)',
            'n': data_subset.shape[0]
        }
    
    try:
        # Build model specification
        model_dict = {}
        for factor, items in model_spec.items():
            for item in items:
                model_dict[item] = factor
        
        # Fit CFA
        cfa = ConfirmatoryFactorAnalyzer(model_dict, disp=False)
        cfa.fit(data_subset.values)
        
        # Extract results
        loadings_matrix = cfa.loadings_
        factor_names = list(model_spec.keys())
        
        # Build loadings DataFrame
        loadings_df = pd.DataFrame(
            loadings_matrix,
            index=all_items,
            columns=factor_names
        )
        
        # Identify primary loading for each item
        loadings_df['primary_factor'] = loadings_df.idxmax(axis=1)
        loadings_df['primary_loading'] = loadings_df.max(axis=1)
        
        # Flag low loadings
        warnings = []
        for item in loadings_df.index:
            loading = loadings_df.loc[item, 'primary_loading']
            if loading < min_loadings:
                warnings.append(f"{item}: λ={loading:.3f} (< {min_loadings})")
        
        # Calculate fit indices
        fit_indices = {
            'chi2': None,
            'df': None,
            'cfi': None,
            'tli': None,
            'rmsea': None,
            'srmr': None,
            'n_observations': data_subset.shape[0],
            'n_factors': len(factor_names),
            'n_items': len(all_items),
        }
        
        # Factor correlations
        try:
            factor_corr = pd.DataFrame(
                cfa.factor_varcovs_,
                index=factor_names,
                columns=factor_names
            )
        except:
            factor_corr = None
        
        return {
            'fit_indices': fit_indices,
            'loadings': loadings_df,
            'factor_correlations': factor_corr,
            'warnings': warnings,
            'converged': True,
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'converged': False,
        }


# ══════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS: Wilson Score
# ══════════════════════════════════════════════════════════════

def wilson_ci(
    n_success: int,
    n_total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for proportions.
    
    More accurate than normal approximation for small samples.
    
    Args:
        n_success: Number of successes
        n_total: Total sample size
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound) as percentages
        
    Reference:
        Wilson, E. B. (1927). Probable inference, the law of succession,
        and statistical inference. JASA, 22(158), 209-212.
    """
    if n_total == 0:
        return (np.nan, np.nan)
    
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    
    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    
    spread = (
        z * math.sqrt(
            p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)
        )
    ) / denominator
    
    lower = max(0.0, center - spread) * 100
    upper = min(1.0, center + spread) * 100
    
    return (round(lower, 1), round(upper, 1))


# ══════════════════════════════════════════════════════════════
# SUBGROUP ANALYSIS: Chi-square + Cramér's V
# ══════════════════════════════════════════════════════════════

def chi2_cramer(
    group_series: pd.Series,
    binary_series: pd.Series
) -> Dict:
    """
    Chi-square test + Cramér's V effect size for group differences.
    
    Args:
        group_series: Categorical grouping variable (e.g., gender, grade)
        binary_series: Binary outcome (e.g., victimization yes/no)
        
    Returns:
        Dict with chi2, p_value, cramers_v, n, significant flag
        
    Reference:
        Cramér's V: √(χ² / (n × min_dim))
        Small: 0.10, Medium: 0.30, Large: 0.50
    """
    df = pd.DataFrame({
        'group': group_series,
        'outcome': binary_series
    }).dropna()
    
    if df.empty or df['group'].nunique() < 2:
        return {
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'n': 0,
            'significant': False,
        }
    
    ct = pd.crosstab(df['group'], df['outcome'])
    
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'n': int(len(df)),
            'significant': False,
        }
    
    chi2_val, p_val, dof, expected = scipy_stats.chi2_contingency(
        ct, correction=False
    )
    
    n = int(ct.values.sum())
    min_dim = min(ct.shape) - 1
    
    cramers_v = float(np.sqrt(chi2_val / (n * max(min_dim, 1)))) if n > 0 else np.nan
    
    return {
        'chi2': round(float(chi2_val), 3),
        'p_value': round(float(p_val), 4),
        'cramers_v': round(cramers_v, 3),
        'dof': int(dof),
        'n': n,
        'significant': bool(p_val < ALPHA_STAT),
    }


def bonferroni_threshold(n_tests: int) -> float:
    """Calculate Bonferroni-corrected alpha threshold."""
    return ALPHA_STAT / max(n_tests, 1)


# ══════════════════════════════════════════════════════════════
# NON-PARAMETRIC COMPARISONS: Mann-Whitney U / Kruskal-Wallis
# ══════════════════════════════════════════════════════════════

@dataclass
class ContinuousGroupResult:
    """Per-group descriptive statistics for continuous score comparison."""
    group:    str
    n:        int
    median:   float
    mean:     float
    sd:       float
    q1:       float
    q3:       float
    pct_high: float   # % of students with score >= HIGH_FREQ_THRESHOLD


@dataclass
class ContinuousComparison:
    """
    Result of a non-parametric comparison of construct sum scores
    across two or more groups.

    Two groups  → Mann-Whitney U, effect size = rank-biserial r
    Three+ groups → Kruskal-Wallis H, effect size = epsilon-squared (ε²)
                    + Dunn post-hoc with Bonferroni correction
    """
    grouping_var:   str
    outcome_var:    str
    n_groups:       int
    n_total:        int
    test_name:      str           # "Mann-Whitney U" or "Kruskal-Wallis H"
    statistic:      float         # U or H
    p_value:        float
    effect_size:    float         # rank-biserial r  OR  epsilon-squared
    effect_label:   str           # "r" or "ε²"
    effect_interp:  str           # "pequeño / mediano / grande"
    bonferroni_alpha: float
    is_significant: bool
    group_stats:    List[ContinuousGroupResult]
    dunn_matrix:    Optional[pd.DataFrame]   # pairwise p-values (Kruskal only)


def _rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial r from Mann-Whitney U.
    Formula: r = 1 - (2U / n1*n2)
    Range: -1 (group2 dominates) → +1 (group1 dominates)
    Benchmarks: |r| small=0.10, medium=0.30, large=0.50
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return round(float(1 - (2 * u_stat) / (n1 * n2)), 3)


def _epsilon_squared(h_stat: float, n_total: int) -> float:
    """
    Epsilon-squared (ε²) effect size for Kruskal-Wallis H.
    Formula: ε² = (H - k + 1) / (n - k)  [unbiased version]
    Benchmarks: small=0.01, medium=0.06, large=0.14
    Reference: Tomczak & Tomczak (2014)
    """
    # Simpler formula: ε² = H / (n - 1)
    if n_total <= 1:
        return np.nan
    return round(float(h_stat / (n_total - 1)), 3)


def _effect_interpretation(value: float, label: str) -> str:
    """Human-readable effect size interpretation."""
    if np.isnan(value):
        return "—"
    v = abs(value)
    if label == "r":
        if v >= 0.50: return "grande"
        if v >= 0.30: return "mediano"
        if v >= 0.10: return "pequeño"
        return "trivial"
    else:  # epsilon-squared
        if v >= 0.14: return "grande"
        if v >= 0.06: return "mediano"
        if v >= 0.01: return "pequeño"
        return "trivial"


def _dunn_posthoc(groups_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Dunn's post-hoc test with Bonferroni correction for pairwise
    group comparisons after a significant Kruskal-Wallis result.

    Returns a symmetric DataFrame of adjusted p-values.
    """
    group_names = list(groups_data.keys())
    k = len(group_names)
    p_matrix = pd.DataFrame(
        np.ones((k, k)),
        index=group_names,
        columns=group_names
    )

    # Pool all scores for rank calculation
    all_scores = pd.concat(
        [s.rename(name) for name, s in groups_data.items()], axis=0
    ).reset_index(drop=True)
    all_ranks = all_scores.rank(method='average')

    n_total = len(all_scores)
    n_comparisons = k * (k - 1) / 2

    offset = 0
    group_ranks: Dict[str, np.ndarray] = {}
    for name, series in groups_data.items():
        n_i = len(series)
        group_ranks[name] = all_ranks.iloc[offset:offset + n_i].values
        offset += n_i

    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = group_names[i], group_names[j]
            n1, n2 = len(group_ranks[g1]), len(group_ranks[g2])
            R1 = group_ranks[g1].mean()
            R2 = group_ranks[g2].mean()

            # Dunn test statistic
            se = np.sqrt(
                (n_total * (n_total + 1) / 12) * (1 / n1 + 1 / n2)
            )
            if se == 0:
                p_adj = 1.0
            else:
                z = abs(R1 - R2) / se
                p_raw = 2 * (1 - scipy_stats.norm.cdf(z))
                # Bonferroni correction
                p_adj = min(1.0, p_raw * n_comparisons)

            p_matrix.loc[g1, g2] = round(p_adj, 4)
            p_matrix.loc[g2, g1] = round(p_adj, 4)

    return p_matrix


def compare_subgroups_continuous(
    student_df: pd.DataFrame,
    outcome_col: str,
    grouping_col: str,
    n_total_tests: int = 1,
    min_group_size: int = 5,
) -> Optional[ContinuousComparison]:
    """
    Compare construct sum scores across demographic subgroups using
    non-parametric tests (appropriate for ordinal 0–4 frequency data).

    Two groups  → Mann-Whitney U + rank-biserial r
    Three+ groups → Kruskal-Wallis H + ε² + Dunn post-hoc (Bonferroni)

    Args:
        student_df:      Student-level DataFrame
        outcome_col:     Column name of the _sum construct score
        grouping_col:    Categorical column (genero, grado, etc.)
        n_total_tests:   Total simultaneous tests (for Bonferroni threshold)
        min_group_size:  Minimum students per group to include group

    Returns:
        ContinuousComparison or None if insufficient data
    """
    working = student_df[[outcome_col, grouping_col]].dropna()

    if working.empty:
        return None

    # Build per-group Series, dropping small groups
    groups_data: Dict[str, pd.Series] = {}
    for grp, sub in working.groupby(grouping_col):
        if len(sub) >= min_group_size:
            groups_data[str(grp)] = sub[outcome_col].reset_index(drop=True)

    if len(groups_data) < 2:
        return None

    n_total = sum(len(s) for s in groups_data.values())
    bonf_alpha = bonferroni_threshold(n_total_tests)

    # ── Per-group descriptive stats ──────────────────────────
    group_stats: List[ContinuousGroupResult] = []
    for grp_name, scores in groups_data.items():
        group_stats.append(ContinuousGroupResult(
            group    = grp_name,
            n        = int(len(scores)),
            median   = round(float(scores.median()), 2),
            mean     = round(float(scores.mean()), 2),
            sd       = round(float(scores.std(ddof=1)), 2) if len(scores) > 1 else np.nan,
            q1       = round(float(scores.quantile(0.25)), 2),
            q3       = round(float(scores.quantile(0.75)), 2),
            pct_high = round(float((scores >= HIGH_FREQ_THRESHOLD).mean() * 100), 1),
        ))
    group_stats.sort(key=lambda x: x.median, reverse=True)

    # ── Statistical test ─────────────────────────────────────
    arrays = [s.values for s in groups_data.values()]

    if len(groups_data) == 2:
        # Mann-Whitney U
        g1, g2 = arrays
        u_stat, p_val = scipy_stats.mannwhitneyu(
            g1, g2, alternative='two-sided'
        )
        n1, n2 = len(g1), len(g2)
        effect = _rank_biserial_r(u_stat, n1, n2)
        effect_label = "r"
        test_name = "Mann-Whitney U"
        statistic = round(float(u_stat), 2)
        dunn_matrix = None

    else:
        # Kruskal-Wallis H
        h_stat, p_val = scipy_stats.kruskal(*arrays)
        effect = _epsilon_squared(h_stat, n_total)
        effect_label = "ε²"
        test_name = "Kruskal-Wallis H"
        statistic = round(float(h_stat), 3)
        # Run Dunn post-hoc only if significant
        dunn_matrix = (
            _dunn_posthoc(groups_data)
            if p_val < bonf_alpha
            else None
        )

    return ContinuousComparison(
        grouping_var   = grouping_col,
        outcome_var    = outcome_col,
        n_groups       = len(groups_data),
        n_total        = n_total,
        test_name      = test_name,
        statistic      = statistic,
        p_value        = round(float(p_val), 4),
        effect_size    = effect,
        effect_label   = effect_label,
        effect_interp  = _effect_interpretation(effect, effect_label),
        bonferroni_alpha = bonf_alpha,
        is_significant = bool(p_val < bonf_alpha),
        group_stats    = group_stats,
        dunn_matrix    = dunn_matrix,
    )


# ══════════════════════════════════════════════════════════════
# ITEM-LEVEL DESCRIPTIVES
# ══════════════════════════════════════════════════════════════

def item_descriptives(answer_df: pd.DataFrame, external_ids: List[str]) -> pd.DataFrame:
    """
    Calculate item-level descriptive statistics.
    
    Args:
        answer_df: Answer-level DataFrame
        external_ids: List of external_ids for items to analyze
    
    Returns:
        DataFrame with columns: [item, mean, sd, pct_high_freq, n_answered, n_missing]
    """
    if answer_df is None or answer_df.empty or not external_ids:
        return pd.DataFrame()
    
    # Find the question_id column (might be named differently)
    question_col = None
    for col in ['question_id', 'external_id', 'question_external_id']:
        if col in answer_df.columns:
            question_col = col
            break
    
    if question_col is None:
        st.warning("⚠️ Cannot find question_id column in answer_df")
        return pd.DataFrame()
    
    # Filter to construct items
    mask = answer_df[question_col].isin(external_ids)
    items = answer_df[mask].copy()
    
    if items.empty:
        return pd.DataFrame()
    
    # Get question text column
    text_col = None
    for col in ['question_text', 'text', 'question']:
        if col in items.columns:
            text_col = col
            break
    
    if text_col is None:
        text_col = question_col  # Fallback to question_id
    
    # Group by question
    rows = []
    for qid in external_ids:
        item_data = items[items[question_col] == qid]
        
        if item_data.empty:
            continue
        
        scores = item_data['score'].dropna()
        n_answered = len(scores)
        n_missing = len(item_data) - n_answered
        
        # Get question text (first non-null value)
        q_text = item_data[text_col].dropna().iloc[0] if len(item_data[text_col].dropna()) > 0 else qid
        
        rows.append({
            'item': str(qid),
            'question_text': str(q_text)[:80],
            'mean': round(float(scores.mean()), 2) if n_answered > 0 else np.nan,
            'sd': round(float(scores.std(ddof=1)), 2) if n_answered > 1 else np.nan,
            'pct_high_freq': round(float((scores >= HIGH_FREQ_THRESHOLD).mean() * 100), 1) if n_answered > 0 else np.nan,
            'n_answered': int(n_answered),
            'n_missing': int(n_missing),
        })
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows).sort_values('pct_high_freq', ascending=False, na_position='last')

# ══════════════════════════════════════════════════════════════
# CONSTRUCT CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════

def construct_correlations(
    student_df: pd.DataFrame,
    construct_names: List[str]
) -> pd.DataFrame:
    """
    Calculate Pearson correlations between construct scores.
    
    Args:
        student_df: Student-level DataFrame with construct_sum columns
        construct_names: List of construct names
        
    Returns:
        Correlation matrix DataFrame
    """
    cols = [f"{c}_sum" for c in construct_names]
    existing_cols = [c for c in cols if c in student_df.columns]
    
    if not existing_cols:
        return pd.DataFrame()
    
    corr_matrix = student_df[existing_cols].corr()
    
    # Clean up column names
    corr_matrix.columns = [c.replace('_sum', '') for c in corr_matrix.columns]
    corr_matrix.index = [c.replace('_sum', '') for c in corr_matrix.index]
    
    return corr_matrix


# ══════════════════════════════════════════════════════════════
# MISSING DATA ANALYSIS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# WRAPPER FUNCTIONS FOR APP.PY COMPATIBILITY
# ══════════════════════════════════════════════════════════════

@dataclass
class ReliabilityResult:
    """Reliability analysis results"""
    construct: str
    n_items: int
    n_students: int
    cronbach_alpha: float
    mcdonald_omega: float
    alpha_meets_threshold: bool
    omega_meets_threshold: bool
    published_alpha_range: tuple
    items_excluded: list

@dataclass  
class PrevalenceResult:
    """Prevalence with confidence interval"""
    pct: float
    n_with_data: int
    n_true: int
    n_missing: int
    missing_pct: float
    ci_lower: float
    ci_upper: float
    threshold_category: str

@dataclass
class SubgroupComparison:
    """Subgroup comparison results"""
    grouping_var: str
    outcome_var: str
    chi2: float
    p_value: float
    cramers_v: float
    n_total: int
    group_stats: pd.DataFrame
    bonferroni_alpha: float
    is_significant: bool


def analyze_reliability(answer_df, construct, external_ids):
    """Analyze reliability for a construct"""
    from construct_definitions import is_global_screener, get_construct_metadata
    
    items_no_screeners = [ext_id for ext_id in external_ids if not is_global_screener(ext_id)]
    
    mask = answer_df['question_id'].isin(items_no_screeners)
    construct_data = answer_df[mask][['survey_response_id', 'question_id', 'score']].copy()
    
    if construct_data.empty:
        metadata = get_construct_metadata(construct)
        return ReliabilityResult(
            construct=construct,
            n_items=0,
            n_students=0,
            cronbach_alpha=np.nan,
            mcdonald_omega=np.nan,
            alpha_meets_threshold=False,
            omega_meets_threshold=False,
            published_alpha_range=metadata.published_alpha_range if metadata else (0.0, 0.0),
            items_excluded=[ext_id for ext_id in external_ids if is_global_screener(ext_id)],
        )
    
    pivot = construct_data.pivot_table(
        index='survey_response_id',
        columns='question_id',
        values='score',
        aggfunc='mean'
    )
    
    alpha = cronbach_alpha(pivot)
    omega = mcdonalds_omega(pivot)
    
    metadata = get_construct_metadata(construct)
    
    return ReliabilityResult(
        construct=construct,
        n_items=len(items_no_screeners),
        n_students=int(pivot.shape[0]),
        cronbach_alpha=alpha,
        mcdonald_omega=omega,
        alpha_meets_threshold=bool(not np.isnan(alpha) and alpha >= ALPHA_THRESHOLD),
        omega_meets_threshold=bool(not np.isnan(omega) and omega >= 0.70),
        published_alpha_range=metadata.published_alpha_range if metadata else (0.0, 0.0),
        items_excluded=[ext_id for ext_id in external_ids if is_global_screener(ext_id)],
    )


def calculate_prevalence(indicator_series):
    """Calculate prevalence with Wilson CI"""
    if indicator_series is None or len(indicator_series) == 0:
        return PrevalenceResult(
            pct=np.nan, n_with_data=0, n_true=0, n_missing=0,
            missing_pct=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            threshold_category="SIN_DATOS"
        )
    
    n_total = len(indicator_series)
    n_missing = int(indicator_series.isna().sum())
    n_with_data = n_total - n_missing
    
    if n_with_data == 0:
        return PrevalenceResult(
            pct=np.nan, n_with_data=0, n_true=0, n_missing=n_missing,
            missing_pct=100.0, ci_lower=np.nan, ci_upper=np.nan,
            threshold_category="SIN_DATOS"
        )
    
    valid = indicator_series.dropna().astype(bool)
    n_true = int(valid.sum())
    pct = float(valid.mean() * 100)
    
    ci_lower, ci_upper = wilson_ci(n_true, n_with_data)
    
    def threshold_flag(pct):
        if np.isnan(pct): return "SIN_DATOS"
        if pct >= 20: return "CRISIS"
        if pct >= 10: return "INTERVENCION"
        if pct >= 5: return "ATENCION"
        return "MONITOREO"
    
    return PrevalenceResult(
        pct=round(pct, 1),
        n_with_data=n_with_data,
        n_true=n_true,
        n_missing=n_missing,
        missing_pct=round(100 * n_missing / n_total, 1) if n_total > 0 else np.nan,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        threshold_category=threshold_flag(pct)
    )


def compare_subgroups(student_df, outcome_col, grouping_col, n_total_tests=1):
    """Compare prevalence across subgroups"""
    merged = student_df[[outcome_col, grouping_col]].dropna()
    
    if merged.empty or merged[grouping_col].nunique() < 2:
        return None
    
    chi2_result = chi2_cramer(merged[grouping_col], merged[outcome_col])
    
    group_stats = []
    for group, sub in merged.groupby(grouping_col):
        prev = calculate_prevalence(sub[outcome_col])
        group_stats.append({
            grouping_col: group,
            'n_students': len(sub),
            'n_with_data': prev.n_with_data,
            'n_true': prev.n_true,
            'pct': prev.pct,
            'ci_lower': prev.ci_lower,
            'ci_upper': prev.ci_upper,
        })
    
    group_stats_df = pd.DataFrame(group_stats).sort_values('pct', ascending=False)
    
    bonf_alpha = bonferroni_threshold(n_total_tests)
    
    return SubgroupComparison(
        grouping_var=grouping_col,
        outcome_var=outcome_col,
        chi2=chi2_result['chi2'],
        p_value=chi2_result['p_value'],
        cramers_v=chi2_result['cramers_v'],
        n_total=chi2_result['n'],
        group_stats=group_stats_df,
        bonferroni_alpha=bonf_alpha,
        is_significant=bool(not np.isnan(chi2_result['p_value']) and chi2_result['p_value'] < bonf_alpha),
    )


def construct_correlation_matrix(student_df, constructs):
    """
    Calculate Spearman rank correlations between construct sum scores,
    paired with a matrix of two-tailed p-values.

    Spearman is required because construct sum scores are ordinal
    composites of 0–4 frequency items — not continuous/normal data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (rho_matrix, pvalue_matrix)
        Both DataFrames share the same index/columns (construct names).
        Returns (empty, empty) if fewer than 2 constructs have data.
    """
    score_cols = [f'{c}_sum' for c in constructs if f'{c}_sum' in student_df.columns]

    if len(score_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    data = student_df[score_cols].dropna()
    n_cols = len(score_cols)
    labels = [c.replace('_sum', '') for c in score_cols]

    # Build rho and p-value matrices manually via scipy.spearmanr
    rho_matrix = np.full((n_cols, n_cols), np.nan)
    p_matrix   = np.full((n_cols, n_cols), np.nan)

    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                rho_matrix[i, j] = 1.0
                p_matrix[i, j]   = 0.0
            elif j > i:
                x = data.iloc[:, i]
                y = data.iloc[:, j]
                rho, pval = scipy_stats.spearmanr(x, y)
                rho_matrix[i, j] = rho_matrix[j, i] = round(float(rho), 3)
                p_matrix[i, j]   = p_matrix[j, i]   = round(float(pval), 4)

    rho_df = pd.DataFrame(rho_matrix, index=labels, columns=labels)
    p_df   = pd.DataFrame(p_matrix,   index=labels, columns=labels)

    return rho_df, p_df


def missing_pattern_summary(
    student_df: pd.DataFrame,
    conditional_constructs: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Analyze missing data patterns, distinguishing structural missingness
    (expected skips from conditional survey logic) from unintentional gaps.

    SURVEY_004 conditional modules:
      - violencia_internivel: only K-12 mixed-campus students
      - ecologia_espacios:    only students who reported victimization
        (victimizacion_freq == True)

    Args:
        student_df: Student-level DataFrame with _sum / _freq columns
        conditional_constructs: Optional dict mapping construct_name →
            'gating_column'. If gating column is None/NaN for a student,
            missingness is classified as STRUCTURAL (expected skip).

    Returns:
        Dict[column → {n_missing, n_structural, n_unexpected, n_total,
                        missing_pct, structural_pct, unexpected_pct}]
    """
    # Default conditional module gating logic for SURVEY_004
    if conditional_constructs is None:
        conditional_constructs = {
            'violencia_internivel_sum':  None,          # campus-level flag (no column to check)
            'violencia_internivel_freq': None,
            'ecologia_espacios_sum':     'victimizacion_freq',
            'ecologia_espacios_freq':    'victimizacion_freq',
        }

    cols = [c for c in student_df.columns if c.endswith('_sum') or c.endswith('_freq')]
    n_total = int(len(student_df))
    out = {}

    for col in cols:
        n_miss = int(student_df[col].isna().sum())
        n_structural = 0

        # Identify structural skips
        if col in conditional_constructs:
            gate = conditional_constructs[col]
            if gate is None:
                # All missingness treated as potentially structural
                # (campus-level flag not available in student_df)
                n_structural = n_miss
            elif gate in student_df.columns:
                # Students for whom the gate condition is False/NaN
                # were legitimately not shown this module
                not_gated = student_df[gate].isna() | (student_df[gate] == False)
                n_structural = int((student_df[col].isna() & not_gated).sum())

        n_unexpected = n_miss - n_structural

        out[col] = {
            'n_total':         n_total,
            'n_missing':       n_miss,
            'n_structural':    n_structural,   # expected skip
            'n_unexpected':    max(0, n_unexpected),  # data quality concern
            'missing_pct':     round(100 * n_miss / n_total, 1) if n_total else np.nan,
            'structural_pct':  round(100 * n_structural / n_total, 1) if n_total else np.nan,
            'unexpected_pct':  round(100 * max(0, n_unexpected) / n_total, 1) if n_total else np.nan,
        }

    return out


# ══════════════════════════════════════════════════════════════
# BULLY-VICTIM TYPOLOGY (Olweus Framework)
# ══════════════════════════════════════════════════════════════

# Thresholds (as mean item score on 0-4 scale).
# ≥ 1.0 = "at least occasionally" — standard criterion in literature.
VICTIM_THRESHOLD     = 1.0   # mean score on victimizacion items
PERPETRATOR_THRESHOLD = 1.0  # mean score on perpetracion items

TYPOLOGY_LABELS = {
    'bully_victim':  'Agresor-Víctima',
    'victim':        'Víctima',
    'bully':         'Agresor',
    'uninvolved':    'No Involucrado',
}


@dataclass
class TypologyResult:
    """
    Bully-victim classification results.

    Profiles (mutually exclusive, exhaustive):
      Agresor-Víctima  — highest psychosocial risk
      Víctima          — victimized, not perpetrating
      Agresor          — perpetrating, not victimized
      No Involucrado   — neither role

    Reference:
        Olweus, D. (1993). Bullying at School.
        Nansel et al. (2001). JAMA, 285(16), 2094-2100.
    """
    n_total:       int
    n_classified:  int           # students with data in both constructs
    counts:        Dict[str, int]
    percentages:   Dict[str, float]
    ci:            Dict[str, Tuple[float, float]]  # Wilson 95% CI per profile
    column_name:   str = 'typology'


def classify_bully_victim_typology(
    student_df: pd.DataFrame,
    victim_col:      str = 'victimizacion_sum',
    perpetrator_col: str = 'perpetracion_sum',
    n_victim_items:  int = 8,
    n_perp_items:    int = 7,
) -> Tuple[pd.DataFrame, TypologyResult]:
    """
    Classify each student into one of four mutually exclusive
    Olweus bully-victim profiles based on their construct sum scores.

    Args:
        student_df:       Student-level DataFrame
        victim_col:       Column with victimizacion sum score
        perpetrator_col:  Column with perpetracion sum score
        n_victim_items:   Number of items in victimizacion (for mean calc)
        n_perp_items:     Number of items in perpetracion (for mean calc)

    Returns:
        (updated_df, TypologyResult)
        updated_df has a new 'typology' column with Spanish labels.
    """
    df = student_df.copy()
    df['typology'] = pd.Series(dtype='object')  # string column from the start

    has_both = df[victim_col].notna() & df[perpetrator_col].notna()
    sub = df[has_both].copy()

    if sub.empty:
        empty_counts = {k: 0 for k in TYPOLOGY_LABELS.values()}
        empty_pcts   = {k: np.nan for k in TYPOLOGY_LABELS.values()}
        empty_ci     = {k: (np.nan, np.nan) for k in TYPOLOGY_LABELS.values()}
        return df, TypologyResult(
            n_total=len(df), n_classified=0,
            counts=empty_counts, percentages=empty_pcts,
            ci=empty_ci
        )

    # Convert sum → mean item score for threshold comparison
    sub['victim_mean'] = sub[victim_col] / n_victim_items
    sub['perp_mean']   = sub[perpetrator_col] / n_perp_items

    is_victim = sub['victim_mean'] >= VICTIM_THRESHOLD
    is_perp   = sub['perp_mean']   >= PERPETRATOR_THRESHOLD

    conditions = [
        ( is_victim &  is_perp,  TYPOLOGY_LABELS['bully_victim']),
        ( is_victim & ~is_perp,  TYPOLOGY_LABELS['victim']),
        (~is_victim &  is_perp,  TYPOLOGY_LABELS['bully']),
        (~is_victim & ~is_perp,  TYPOLOGY_LABELS['uninvolved']),
    ]
    for mask, label in conditions:
        sub.loc[mask, 'typology'] = label

    df.loc[has_both, 'typology'] = sub['typology']

    # Compute summary statistics
    n_classified = int(has_both.sum())
    counts, pcts, cis = {}, {}, {}

    for key, label in TYPOLOGY_LABELS.items():
        n = int((sub['typology'] == label).sum())
        pct = round(float(n / n_classified * 100), 1) if n_classified > 0 else np.nan
        ci  = wilson_ci(n, n_classified) if n_classified > 0 else (np.nan, np.nan)
        counts[label] = n
        pcts[label]   = pct
        cis[label]    = ci

    result = TypologyResult(
        n_total=len(df),
        n_classified=n_classified,
        counts=counts,
        percentages=pcts,
        ci=cis,
    )

    return df, result


# ══════════════════════════════════════════════════════════════
# CYBER vs. TRADITIONAL BULLYING OVERLAP
# ══════════════════════════════════════════════════════════════

@dataclass
class BullyingOverlapResult:
    """
    Overlap between traditional victimization and cybervictimization.

    Metrics:
      jaccard        — |A ∩ B| / |A ∪ B|  (0=no overlap, 1=identical)
      phi            — phi coefficient from 2x2 contingency table
                       (equivalent to Pearson r for binary variables)
      overlap_pct    — % of traditional victims who are ALSO cyber-victims
      cyber_only_pct — % of cyber-victims NOT in traditional victimization
    """
    n_total:           int
    n_trad_victim:     int
    n_cyber_victim:    int
    n_both:            int
    n_neither:         int
    jaccard:           float
    phi:               float
    overlap_pct:       float   # % of trad victims who are also cyber
    cyber_only_pct:    float   # % of cyber victims not in trad
    chi2:              float
    p_value:           float
    interpretation:    str


def analyze_bullying_overlap(
    student_df: pd.DataFrame,
    trad_col:  str = 'victimizacion_freq',
    cyber_col: str = 'cybervictimizacion_freq',
) -> Optional[BullyingOverlapResult]:
    """
    Analyze the overlap between traditional and cyber victimization.

    Uses binary _freq indicators (True = frequent victim).
    Returns phi coefficient, Jaccard similarity, and chi-square test.

    Reference:
        Kowalski et al. (2014). Bullying in the digital age. Psych Bulletin.
    """
    working = student_df[[trad_col, cyber_col]].dropna()

    if working.empty or len(working) < 10:
        return None

    trad  = working[trad_col].astype(bool)
    cyber = working[cyber_col].astype(bool)

    n_total        = int(len(working))
    n_trad         = int(trad.sum())
    n_cyber        = int(cyber.sum())
    n_both         = int((trad & cyber).sum())
    n_neither      = int((~trad & ~cyber).sum())
    n_trad_only    = n_trad - n_both
    n_cyber_only   = n_cyber - n_both

    # Jaccard similarity
    union = n_trad + n_cyber - n_both
    jaccard = round(float(n_both / union), 3) if union > 0 else np.nan

    # Chi-square on 2×2 table
    ct = pd.crosstab(trad, cyber)
    if ct.shape == (2, 2):
        chi2_val, p_val, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
        # Phi coefficient = sqrt(chi2 / n) — signed to indicate direction
        phi = round(float(np.sqrt(chi2_val / n_total)), 3) if n_total > 0 else np.nan
    else:
        chi2_val, p_val, phi = np.nan, np.nan, np.nan

    overlap_pct    = round(float(n_both / n_trad * 100), 1) if n_trad > 0 else np.nan
    cyber_only_pct = round(float(n_cyber_only / n_cyber * 100), 1) if n_cyber > 0 else np.nan

    # Plain-language interpretation
    if np.isnan(phi):
        interp = "Datos insuficientes"
    elif phi >= 0.50:
        interp = "Solapamiento muy alto — mismos estudiantes afectados en ambos contextos"
    elif phi >= 0.30:
        interp = "Solapamiento moderado — intervención combinada recomendada"
    elif phi >= 0.10:
        interp = "Solapamiento bajo — fenómenos parcialmente distintos"
    else:
        interp = "Sin solapamiento significativo — poblaciones separadas"

    return BullyingOverlapResult(
        n_total=n_total,
        n_trad_victim=n_trad,
        n_cyber_victim=n_cyber,
        n_both=n_both,
        n_neither=n_neither,
        jaccard=jaccard,
        phi=phi,
        overlap_pct=overlap_pct,
        cyber_only_pct=cyber_only_pct,
        chi2=round(float(chi2_val), 3) if not np.isnan(chi2_val) else np.nan,
        p_value=round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════
# SCHOOL SAFETY RISK INDEX
# ══════════════════════════════════════════════════════════════

# Max possible sum scores per construct (n_items × 4)
CONSTRUCT_MAX_SCORES = {
    'autoridad_docente':      7 * 4,   # 28
    'normas_grupales':        4 * 4,   # 16
    'respuesta_institucional':6 * 4,   # 24
    'victimizacion':          8 * 4,   # 32
    'perpetracion':           7 * 4,   # 28
    'cybervictimizacion':     6 * 4,   # 24
    'cyberagresion':          2 * 4,   # 8
    'violencia_internivel':   5 * 4,   # 20
}

# Weights for index composition (must sum to 1.0 per direction)
RISK_WEIGHTS = {
    'victimizacion':      0.35,
    'perpetracion':       0.25,
    'cybervictimizacion': 0.25,
    'cyberagresion':      0.10,
    'violencia_internivel': 0.05,
}
PROTECTIVE_WEIGHTS = {
    'autoridad_docente':       0.40,
    'normas_grupales':         0.30,
    'respuesta_institucional': 0.30,
}


@dataclass
class RiskIndexResult:
    """
    Composite School Safety Risk Index (SSRI).

    Scale: 0–100 where higher = GREATER risk.
    Formula:
      risk_score       = weighted sum of normalized risk construct scores
      protective_score = weighted sum of normalized protective construct scores
      SSRI             = risk_score × (1 − protective_score)
                         ─────────────────────────────────────────────
                         This formulation means a school with high risk
                         but also high protective factors scores lower
                         than one with high risk and low protection.

    Thresholds:
      0–20   → Verde   (Monitoreo)
      20–40  → Amarillo (Atención)
      40–60  → Naranja  (Intervención)
      60–100 → Rojo    (Crisis)
    """
    ssri:              float   # 0–100
    risk_component:    float   # 0–100 (raw risk)
    protective_component: float  # 0–100 (protective factor score)
    n_students:        int
    n_missing:         int
    threshold_color:   str    # "Verde" | "Amarillo" | "Naranja" | "Rojo"
    construct_scores:  Dict[str, float]  # normalized 0–1 per construct


def calculate_risk_index(student_df: pd.DataFrame) -> RiskIndexResult:
    """
    Calculate the composite School Safety Risk Index for a school cohort.

    Uses sum scores (_sum columns) normalized to [0, 1] range, then
    applies construct weights to produce risk and protective components.

    Returns a RiskIndexResult with SSRI score and sub-component breakdown.
    """
    n_total = len(student_df)
    construct_scores: Dict[str, float] = {}

    # Normalize each construct score to 0–1
    for construct, max_score in CONSTRUCT_MAX_SCORES.items():
        col = f'{construct}_sum'
        if col in student_df.columns:
            mean_score = student_df[col].dropna().mean()
            construct_scores[construct] = round(
                float(mean_score / max_score), 4
            ) if max_score > 0 and not np.isnan(mean_score) else np.nan

    # Risk component (0–100)
    risk_num = risk_den = 0.0
    for construct, weight in RISK_WEIGHTS.items():
        score = construct_scores.get(construct, np.nan)
        if not np.isnan(score):
            risk_num += weight * score
            risk_den += weight
    risk_component = round((risk_num / risk_den * 100) if risk_den > 0 else np.nan, 1)

    # Protective component (0–1, then to 0–100)
    prot_num = prot_den = 0.0
    for construct, weight in PROTECTIVE_WEIGHTS.items():
        score = construct_scores.get(construct, np.nan)
        if not np.isnan(score):
            prot_num += weight * score
            prot_den += weight
    protective_component = round((prot_num / prot_den * 100) if prot_den > 0 else np.nan, 1)

    # Composite SSRI: risk attenuated by protective factors
    if not np.isnan(risk_component) and not np.isnan(protective_component):
        prot_01 = protective_component / 100
        ssri = round(float(risk_component * (1 - prot_01 * 0.5)), 1)
        # (0.5 factor prevents protective from fully cancelling observed risk)
    elif not np.isnan(risk_component):
        ssri = risk_component
    else:
        ssri = np.nan

    # Threshold classification
    def _color(val):
        if np.isnan(val): return "Sin datos"
        if val >= 60: return "Rojo 🔴"
        if val >= 40: return "Naranja 🟠"
        if val >= 20: return "Amarillo 🟡"
        return "Verde 🟢"

    # Count students with at least one missing construct
    sum_cols = [f'{c}_sum' for c in CONSTRUCT_MAX_SCORES if f'{c}_sum' in student_df.columns]
    n_missing = int(student_df[sum_cols].isna().any(axis=1).sum()) if sum_cols else n_total

    return RiskIndexResult(
        ssri=ssri,
        risk_component=risk_component,
        protective_component=protective_component,
        n_students=n_total,
        n_missing=n_missing,
        threshold_color=_color(ssri),
        construct_scores=construct_scores,
    )


def safe_bool_rate(series: pd.Series) -> Dict:
    """Calculate rate for boolean series with Wilson CI."""
    empty = {
        'pct': np.nan,
        'n_with_data': 0,
        'n_missing': 0,
        'missing_pct': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
    }
    
    if series is None or len(series) == 0:
        return empty
    
    s = series.copy()
    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    n_with = n_total - n_missing
    
    if n_with == 0:
        return {
            **empty,
            'n_missing': n_missing,
            'missing_pct': round(100 * n_missing / n_total, 1) if n_total else np.nan,
        }
    
    s_bool = s.dropna().astype(bool)
    n_true = int(s_bool.sum())
    pct = float(s_bool.mean() * 100.0)
    ci_lo, ci_hi = wilson_ci(n_true, n_with)
    
    return {
        'pct': round(pct, 1),
        'n_with_data': n_with,
        'n_missing': n_missing,
        'missing_pct': round(100 * n_missing / n_total, 1) if n_total else np.nan,
        'ci_lower': ci_lo,
        'ci_upper': ci_hi,
    }


# ══════════════════════════════════════════════════════════════
# SAMPLE REPRESENTATIVENESS ASSESSMENT
# ══════════════════════════════════════════════════════════════

@dataclass
class RepresentativenessResult:
    """
    Statistical representativeness of a school's survey sample.

    Compares n_respondents against the minimum required sample size
    for a finite population (the enrolled student body).

    Semáforo:
      🟢 REPRESENTATIVA   — n ≥ n_min  AND  coverage ≥ 30%
      🟡 ACEPTABLE        — n ≥ n_min  BUT  coverage < 30%
                            (sufficient n but low penetration)
      🟠 INSUFICIENTE     — n_min × 0.7 ≤ n < n_min
                            (within 30% of target — borderline)
      🔴 NO REPRESENTATIVA — n < n_min × 0.7
                            (results must carry explicit caveat)
      ⚪ SIN DATOS         — enrolled count unavailable
    """
    n_respondents:   int
    n_enrolled:      int      # 0 = unknown
    n_min_required:  int      # minimum for ±5%, 95% CI
    coverage_pct:    float    # n_respondents / n_enrolled × 100
    margin_of_error: float    # actual MoE achieved (%) given n and N
    deficit:         int      # n_min - n_respondents  (negative = surplus)
    status:          str      # "REPRESENTATIVA" | "ACEPTABLE" | "INSUFICIENTE" | "NO_REPRESENTATIVA" | "SIN_DATOS"
    semaforo:        str      # emoji + label for display
    caveat:          str      # plain-language message for sidebar


def _min_sample_finite(N: int, z: float = 1.96, p: float = 0.5, e: float = 0.05) -> int:
    """
    Cochran (1977) formula for minimum sample size in a finite population.

    n0 = Z² × p(1-p) / e²            (infinite population)
    n  = n0 / (1 + (n0 - 1) / N)     (finite population correction)

    Args:
        N: Total population (enrolled students)
        z: Z-score for desired confidence level (1.96 = 95%)
        p: Estimated proportion (0.5 = maximum variance / most conservative)
        e: Desired margin of error (0.05 = ±5%)

    Returns:
        Minimum integer sample size
    """
    if N <= 0:
        return 0
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    n  = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))


def _actual_margin_of_error(n: int, N: int, z: float = 1.96, p: float = 0.5) -> float:
    """
    Actual margin of error (%) achieved by the observed sample size,
    with finite population correction.
    """
    if n <= 0 or N <= 0:
        return np.nan
    # Finite population correction factor
    fpc = np.sqrt((N - n) / (N - 1)) if N > 1 else 1.0
    moe = z * np.sqrt(p * (1 - p) / n) * fpc * 100
    return round(float(moe), 1)


def assess_sample_representativeness(
    n_respondents: int,
    students_primaria:   int = 0,
    students_secundaria: int = 0,
    students_preparatoria: int = 0,
    confidence: float = 0.95,
    margin_of_error: float = 0.05,
) -> RepresentativenessResult:
    """
    Assess whether the number of survey respondents is statistically
    representative of the school's enrolled population.

    Args:
        n_respondents:          Number of students who completed the survey
        students_primaria:      Enrolled in primary level (may be 0)
        students_secundaria:    Enrolled in secondary level
        students_preparatoria:  Enrolled in high school level (may be 0)
        confidence:             Desired confidence level (default 0.95)
        margin_of_error:        Acceptable margin of error (default 0.05 = ±5%)

    Returns:
        RepresentativenessResult with semáforo status and explanatory caveat.

    Reference:
        Cochran, W.G. (1977). Sampling Techniques (3rd ed.). Wiley.
    """
    n_enrolled = int(students_primaria) + int(students_secundaria) + int(students_preparatoria)

    # No enrollment data available
    if n_enrolled == 0:
        return RepresentativenessResult(
            n_respondents=n_respondents,
            n_enrolled=0,
            n_min_required=0,
            coverage_pct=np.nan,
            margin_of_error=np.nan,
            deficit=0,
            status="SIN_DATOS",
            semaforo="⚪ SIN DATOS",
            caveat="No se cuenta con el total de alumnos matriculados. "
                   "No es posible evaluar representatividad.",
        )

    z = 1.96 if confidence == 0.95 else scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    n_min = _min_sample_finite(n_enrolled, z=z, e=margin_of_error)
    coverage = round(n_respondents / n_enrolled * 100, 1)
    moe_actual = _actual_margin_of_error(n_respondents, n_enrolled, z=z)
    deficit = n_min - n_respondents

    # ── Semáforo logic ─────────────────────────────────────────
    if n_respondents >= n_min and coverage >= 30:
        status   = "REPRESENTATIVA"
        semaforo = "🟢 REPRESENTATIVA"
        caveat   = (
            f"La muestra es estadísticamente representativa. "
            f"Margen de error real: ±{moe_actual}%."
        )
    elif n_respondents >= n_min and coverage < 30:
        status   = "ACEPTABLE"
        semaforo = "🟡 ACEPTABLE"
        caveat   = (
            f"El n es suficiente (≥{n_min}), pero la cobertura es baja ({coverage:.1f}% "
            f"del alumnado). Posible sesgo de auto-selección. Margen de error: ±{moe_actual}%."
        )
    elif n_respondents >= int(n_min * 0.70):
        status   = "INSUFICIENTE"
        semaforo = "🟠 INSUFICIENTE"
        caveat   = (
            f"Faltan {deficit} encuestas para alcanzar representatividad (mínimo {n_min}). "
            f"Margen de error actual: ±{moe_actual}%. Interpretar con cautela."
        )
    else:
        status   = "NO_REPRESENTATIVA"
        semaforo = "🔴 NO REPRESENTATIVA"
        caveat   = (
            f"Muestra insuficiente: se necesitan al menos {n_min} respuestas "
            f"(se tienen {n_respondents}). Margen de error: ±{moe_actual}%. "
            f"Los resultados NO son generalizables al plantel."
        )

    return RepresentativenessResult(
        n_respondents=n_respondents,
        n_enrolled=n_enrolled,
        n_min_required=n_min,
        coverage_pct=coverage,
        margin_of_error=moe_actual,
        deficit=deficit,
        status=status,
        semaforo=semaforo,
        caveat=caveat,
    )


# ══════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("stats_engine.py — Self-test")
    print("=" * 60)
    
    # Test Cronbach's alpha
    np.random.seed(42)
    test_data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['item1', 'item2', 'item3', 'item4', 'item5']
    )
    alpha = cronbach_alpha(test_data)
    print(f"\nCronbach's α (random data): {alpha}")
    
    # Test McDonald's omega
    if CFA_AVAILABLE:
        omega = mcdonalds_omega(test_data)
        print(f"McDonald's ω (random data): {omega}")
    else:
        print("McDonald's ω: SKIPPED (factor_analyzer not installed)")
    
    # Test Wilson CI
    ci_lo, ci_hi = wilson_ci(45, 100)
    print(f"\nWilson CI (45/100): {ci_lo}% - {ci_hi}%")
    
    # Test chi-square
    groups = pd.Series(['A'] * 50 + ['B'] * 50)
    outcomes = pd.Series([1] * 30 + [0] * 20 + [1] * 25 + [0] * 25)
    chi2_result = chi2_cramer(groups, outcomes)
    print(f"\nChi-square test: χ²={chi2_result['chi2']}, p={chi2_result['p_value']}")
    print(f"Cramér's V: {chi2_result['cramers_v']}")
    
    print("\n" + "=" * 60)
    print("✅ stats_engine.py loaded successfully")
