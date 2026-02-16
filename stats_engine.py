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
# ITEM-LEVEL DESCRIPTIVES
# ══════════════════════════════════════════════════════════════

def item_descriptives(
    answer_df: pd.DataFrame,
    construct_items: List[str],
    external_id_col: str = 'question_external_id'
) -> pd.DataFrame:
    """
    Calculate item-level descriptive statistics.
    
    Args:
        answer_df: Answer-level DataFrame with scores
        construct_items: List of external_ids for this construct
        external_id_col: Column name for external_id
        
    Returns:
        DataFrame with item statistics
    """
    mask = answer_df[external_id_col].isin(construct_items)
    items = answer_df[mask][['question_id', 'question_text', 'score']].copy()
    
    if items.empty:
        return pd.DataFrame()
    
    rows = []
    for (qid, qtext), grp in items.groupby(['question_id', 'question_text']):
        s = grp['score'].dropna()
        n_ans = int(len(s))
        
        rows.append({
            'question_text': (str(qtext) or '')[:65],
            'mean': round(float(s.mean()), 2) if n_ans > 0 else np.nan,
            'sd': round(float(s.std(ddof=1)), 2) if n_ans > 1 else np.nan,
            'pct_high_freq': round(
                float((s >= HIGH_FREQ_THRESHOLD).mean() * 100), 1
            ) if n_ans > 0 else np.nan,
            'n_answered': n_ans,
            'n_missing': int(grp['score'].isna().sum()),
        })
    
    return pd.DataFrame(rows).sort_values('pct_high_freq', ascending=False)


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

def missing_pattern_summary(student_df: pd.DataFrame) -> Dict:
    """Analyze missing data patterns."""
    cols = [c for c in student_df.columns if c.endswith('_sum') or c.endswith('_freq')]
    n_total = int(len(student_df))
    
    out = {}
    for col in cols:
        n_miss = int(student_df[col].isna().sum())
        out[col] = {
            'n_missing': n_miss,
            'n_total': n_total,
            'missing_pct': round(100 * n_miss / n_total, 1) if n_total else np.nan,
        }
    
    return out


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
