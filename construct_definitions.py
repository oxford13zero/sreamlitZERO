# construct_definitions.py
"""
TECH4ZERO-MX v1.1 — Construct Definitions & Parsing
====================================================
Handles external_id parsing and construct mapping for SURVEY_003 / SURVEY_004.

Supported external_id formats:
  Format A (SURVEY_003): survey_003_zero_SECTION_SUBSECTION
    e.g. survey_003_zero_cyber_victima_mensajes

  Format B (SURVEY_004): zero_SECTION_SUBSECTION_v2
    e.g. zero_cyber_victima_mensajes_v2
         zero_general_genero_v2
         zero_victima_rumores_v2

Changelog v1.1:
  - parse_external_id() now handles both formats via _normalize_external_id()
  - _v2 (and any _vN) version suffixes are stripped before parsing
  - All downstream functions (get_construct_items, is_global_screener, etc.)
    benefit automatically — no changes needed in app.py or stats_engine.py
"""

from typing import Dict, Tuple, Optional, FrozenSet
from dataclasses import dataclass
import re

# ══════════════════════════════════════════════════════════════
# PARSING CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Format A: survey_NNN_zero_<remainder>
_PATTERN_A = re.compile(r'^survey_\d+_zero_(.+)$')

# Format B: zero_<remainder>  (SURVEY_004 style)
_PATTERN_B = re.compile(r'^zero_(.+)$')

# Version suffix: _v1, _v2, _v3 … at the END of the string
_VERSION_SUFFIX = re.compile(r'_v\d+$')

# Section → Construct mapping (aligned with TECH4ZERO-MX v1.0 PDF)
SECTION_TO_CONSTRUCT = {
    'general':  'demographic',
    'clima':    'clima_docente',
    'normas':   'normas_grupo',
    'victima':  'victimizacion',
    'victim':   'victimizacion',   # zero_victim_general edge case
    'agresor':  'perpetracion',
    'cyber':    'cyberbullying',   # parent; resolved further by subsection
    'ecologia': 'ecologia_espacios',
    'apoyo':    'apoyo_institucional',
    'impacto':  'impacto',
}

# Cyber subsections → specific constructs
CYBER_SUBSECTION_MAP = {
    'victima': 'cybervictimizacion',
    'agresor': 'cyberagresion',
}


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class ParsedExternalID:
    """Structured representation of a parsed external_id."""
    original: str
    section: str
    subsection: Optional[str]
    sub_subsection: Optional[str]
    construct: str
    item_label: str  # human-readable label (last component after stripping version)


@dataclass
class ConstructMetadata:
    """Metadata for each construct per TECH4ZERO-MX v1.0."""
    name: str
    display_name: str
    pdf_section: str
    n_items_expected: int
    published_alpha_range: Tuple[float, float]
    scale_type: str        # 'likert_0_4' | 'frequency_0_4' | 'categorical'
    score_direction: str   # 'higher_means_more_risk' | 'higher_means_more_protective'
    base_instrument: str
    is_conditional: bool = False  # True for ecologia_espacios


# ══════════════════════════════════════════════════════════════
# CONSTRUCT METADATA (from PDF)
# ══════════════════════════════════════════════════════════════

CONSTRUCT_METADATA: Dict[str, ConstructMetadata] = {
    'demographic': ConstructMetadata(
        name='demographic',
        display_name='Información General',
        pdf_section='A',
        n_items_expected=7,
        published_alpha_range=(0.0, 0.0),
        scale_type='categorical',
        score_direction='none',
        base_instrument='ECOBVQ-R, ZERO-R, INEGI',
    ),
    'clima_docente': ConstructMetadata(
        name='clima_docente',
        display_name='Clima de Aula y Liderazgo del Tutor',
        pdf_section='B',
        n_items_expected=4,
        published_alpha_range=(0.80, 0.85),
        scale_type='likert_0_4',
        score_direction='higher_means_more_protective',
        base_instrument='Programme Zero (Roland, 2000)',
    ),
    'normas_grupo': ConstructMetadata(
        name='normas_grupo',
        display_name='Normas y Eficacia Grupal',
        pdf_section='C',
        n_items_expected=4,
        published_alpha_range=(0.75, 0.82),
        scale_type='likert_0_4',
        score_direction='higher_means_more_protective',
        base_instrument='Bandura (1997), Roland & Galloway (2004)',
    ),
    'victimizacion': ConstructMetadata(
        name='victimizacion',
        display_name='Victimización',
        pdf_section='D',
        n_items_expected=9,
        published_alpha_range=(0.90, 0.95),
        scale_type='frequency_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='OBVQ-R (Gaete et al., 2021)',
    ),
    'perpetracion': ConstructMetadata(
        name='perpetracion',
        display_name='Agresión',
        pdf_section='E',
        n_items_expected=9,
        published_alpha_range=(0.88, 0.93),
        scale_type='frequency_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='OBVQ-R',
    ),
    'cybervictimizacion': ConstructMetadata(
        name='cybervictimizacion',
        display_name='Cybervictimización',
        pdf_section='F',
        n_items_expected=3,
        published_alpha_range=(0.85, 0.90),
        scale_type='likert_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='ECIP-Q (Ortega-Ruiz et al., 2016)',
    ),
    'cyberagresion': ConstructMetadata(
        name='cyberagresion',
        display_name='Cyberagresión',
        pdf_section='F',
        n_items_expected=3,
        published_alpha_range=(0.82, 0.88),
        scale_type='likert_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='ECIP-Q',
    ),
    'ecologia_espacios': ConstructMetadata(
        name='ecologia_espacios',
        display_name='Ecología del Bullying',
        pdf_section='G',
        n_items_expected=8,
        published_alpha_range=(0.0, 0.0),
        scale_type='likert_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='Bronfenbrenner (1979), Astor et al. (2004)',
        is_conditional=True,
    ),
    'apoyo_institucional': ConstructMetadata(
        name='apoyo_institucional',
        display_name='Respuesta Institucional y Recursos de Apoyo',
        pdf_section='H',
        n_items_expected=4,
        published_alpha_range=(0.78, 0.84),
        scale_type='likert_0_4',
        score_direction='higher_means_more_protective',
        base_instrument='Programme Zero, Cohen & Wills (1985)',
    ),
    'impacto': ConstructMetadata(
        name='impacto',
        display_name='Impacto y Consecuencias',
        pdf_section='I',
        n_items_expected=4,
        published_alpha_range=(0.85, 0.91),
        scale_type='likert_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='UAEMex (2012), Lucio López & Gómez Triana (2020)',
    ),
}


# ══════════════════════════════════════════════════════════════
# GLOBAL SCREENER ITEMS (excluded from Cronbach's alpha)
# ══════════════════════════════════════════════════════════════

GLOBAL_SCREENERS = frozenset({
    'victim_general',   # Q24: global victimization screener
    'agresor_general',  # Q33: global perpetration screener
})


# ══════════════════════════════════════════════════════════════
# DEMOGRAPHIC QUESTION IDs
# ══════════════════════════════════════════════════════════════

DEMOGRAPHIC_QUESTIONS = frozenset({
    'general_curso',
    'general_edad',
    'general_genero',
    'general_lengua',
    'general_orientacion',
    'general_tiempo',
    'general_tipo_escuela',
})


# ══════════════════════════════════════════════════════════════
# NORMALIZATION  (v1.1 — NEW)
# ══════════════════════════════════════════════════════════════

def _normalize_external_id(external_id: str) -> Optional[str]:
    """
    Extract the bare section_subsection_... remainder from any supported format.

    Supported inputs:
        survey_003_zero_victima_rumores       → 'victima_rumores'
        survey_004_zero_cyber_victima_v2      → 'cyber_victima'   (version stripped)
        zero_victima_rumores_v2               → 'victima_rumores' (version stripped)
        zero_general_genero_v2               → 'general_genero'

    Returns:
        Normalized remainder string, or None if format is unrecognized.
    """
    if not external_id:
        return None

    # Try Format A first: survey_NNN_zero_<remainder>
    m = _PATTERN_A.match(external_id)
    if m:
        remainder = m.group(1)
        return _VERSION_SUFFIX.sub('', remainder)

    # Try Format B: zero_<remainder>
    m = _PATTERN_B.match(external_id)
    if m:
        remainder = m.group(1)
        return _VERSION_SUFFIX.sub('', remainder)

    return None


# ══════════════════════════════════════════════════════════════
# PARSING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def parse_external_id(external_id: str) -> Optional[ParsedExternalID]:
    """
    Parse an external_id into hierarchical components.

    Supports both SURVEY_003 (survey_NNN_zero_...) and
    SURVEY_004 (zero_..._v2) formats transparently.

    Examples:
        'survey_003_zero_general_curso'     → section: general, construct: demographic
        'zero_general_genero_v2'            → section: general, construct: demographic
        'survey_003_zero_cyber_victima_msg' → section: cyber,   construct: cybervictimizacion
        'zero_cyber_victima_mensajes_v2'    → section: cyber,   construct: cybervictimizacion
        'zero_victima_rumores_v2'           → section: victima, construct: victimizacion

    Returns:
        ParsedExternalID or None if format is unrecognized.
    """
    remainder = _normalize_external_id(external_id)
    if remainder is None:
        return None

    parts = remainder.split('_')
    if not parts:
        return None

    section       = parts[0]
    subsection    = parts[1] if len(parts) > 1 else None
    sub_subsection = parts[2] if len(parts) > 2 else None

    construct  = _resolve_construct(section, subsection)
    item_label = parts[-1]  # most specific component (version already stripped)

    return ParsedExternalID(
        original=external_id,
        section=section,
        subsection=subsection,
        sub_subsection=sub_subsection,
        construct=construct,
        item_label=item_label,
    )


def _resolve_construct(section: str, subsection: Optional[str]) -> str:
    """
    Resolve hierarchical section → construct mapping.

    Special cases:
        cyber + victima  → cybervictimizacion
        cyber + agresor  → cyberagresion
    """
    if section == 'cyber' and subsection:
        return CYBER_SUBSECTION_MAP.get(subsection, 'cyberbullying')

    return SECTION_TO_CONSTRUCT.get(section, 'unknown')


def get_construct_items(construct: str, all_external_ids: list) -> FrozenSet[str]:
    """
    Get all external_ids belonging to a specific construct.

    Args:
        construct: e.g., 'victimizacion', 'cybervictimizacion'
        all_external_ids: list of all question external_ids in the dataset

    Returns:
        Frozen set of matching external_ids.
    """
    items = set()

    for ext_id in all_external_ids:
        parsed = parse_external_id(ext_id)
        if parsed and parsed.construct == construct:
            items.add(ext_id)

    return frozenset(items)


def is_global_screener(external_id: str) -> bool:
    """
    Check if question is a global screener (excluded from alpha calculation).

    Screeners are matched by their section_subsection compound key:
        victim_general  → matches zero_victim_general_v2
        agresor_general → matches zero_agresor_general_v2
    """
    parsed = parse_external_id(external_id)
    if not parsed:
        return False
    # Check compound key: section alone, or section_subsection
    compound = parsed.section
    if parsed.subsection:
        compound = f"{parsed.section}_{parsed.subsection}"
    return compound in GLOBAL_SCREENERS


def is_demographic(external_id: str) -> bool:
    """Check if question is demographic."""
    parsed = parse_external_id(external_id)
    if not parsed:
        return False
    return parsed.construct == 'demographic'


def get_construct_metadata(construct: str) -> Optional[ConstructMetadata]:
    """Get metadata for a construct."""
    return CONSTRUCT_METADATA.get(construct)


def get_all_constructs() -> list:
    """Get list of all construct names."""
    return list(CONSTRUCT_METADATA.keys())


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_construct_coverage(external_ids: list) -> Dict[str, int]:
    """
    Validate that all constructs have the expected number of items.

    Returns:
        Dict mapping construct → actual item count found in external_ids.
    """
    counts = {}
    for construct in get_all_constructs():
        items = get_construct_items(construct, external_ids)
        counts[construct] = len(items)
    return counts


# ══════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    test_cases = [
        # Format A — SURVEY_003
        ('survey_003_zero_general_curso',          'demographic'),
        ('survey_003_zero_clima_normas',           'clima_docente'),
        ('survey_003_zero_cyber_victima_mensajes', 'cybervictimizacion'),
        ('survey_003_zero_victima_rumores',        'victimizacion'),
        ('survey_003_zero_victim_general',         'victimizacion'),
        # Format B — SURVEY_004 (zero_ prefix + _v2 suffix)
        ('zero_general_genero_v2',                 'demographic'),
        ('zero_general_edad_v2',                   'demographic'),
        ('zero_victima_rumores_v2',                'victimizacion'),
        ('zero_agresor_golpes_v2',                 'perpetracion'),
        ('zero_cyber_victima_mensajes_v2',         'cybervictimizacion'),
        ('zero_cyber_agresor_fotos_v2',            'cyberagresion'),
        ('zero_clima_liderazgo_v2',                'clima_docente'),
        ('zero_ecologia_pasillo_v2',               'ecologia_espacios'),
        ('zero_apoyo_director_v2',                 'apoyo_institucional'),
        ('zero_impacto_tristeza_v2',               'impacto'),
    ]

    print("=" * 60)
    print("construct_definitions v1.1 — Self-test")
    print("=" * 60)

    passed = 0
    failed = 0
    for ext_id, expected_construct in test_cases:
        parsed = parse_external_id(ext_id)
        actual = parsed.construct if parsed else 'PARSE_FAILED'
        status = '✅' if actual == expected_construct else '❌'
        if actual == expected_construct:
            passed += 1
        else:
            failed += 1
        print(f"{status} {ext_id}")
        if actual != expected_construct:
            print(f"   expected: {expected_construct}")
            print(f"   got:      {actual}")

    print(f"\nResult: {passed} passed, {failed} failed")

    print("\n--- Screener check ---")
    print(f"zero_victim_general_v2 is_screener: {is_global_screener('zero_victim_general_v2')}")
    print(f"zero_victima_rumores_v2 is_screener: {is_global_screener('zero_victima_rumores_v2')}")
