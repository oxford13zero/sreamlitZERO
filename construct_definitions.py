# construct_definitions.py
"""
TECH4ZERO-MX v1.0 — Construct Definitions & Parsing
====================================================
Handles external_id parsing and construct mapping for SURVEY_003.

External ID format: survey_XXX_zero_SECTION_SUBSECTION_SUBSUBSECTION
Example: survey_003_zero_cyber_victima_mensajes
  → section: cyber
  → subsection: victima  
  → sub-subsection: mensajes
  → construct: cybervictimizacion
"""

from typing import Dict, Tuple, Optional, FrozenSet
from dataclasses import dataclass
import re

# ══════════════════════════════════════════════════════════════
# PARSING CONFIGURATION  
# ══════════════════════════════════════════════════════════════

# Pattern A: Extract components from external_id
EXTERNAL_ID_PATTERN = re.compile(r'^survey_\d+_zero_(.+)$')

# Section → Construct mapping (aligned with TECH4ZERO-MX v1.0 PDF)
SECTION_TO_CONSTRUCT = {
    'general': 'demographic',
    'clima': 'clima_docente',
    'normas': 'normas_grupo',
    'victima': 'victimizacion',
    'victim': 'victimizacion',  # handles zero_victim_general edge case
    'agresor': 'perpetracion',
    'cyber': 'cyberbullying',   # parent category
    'ecologia': 'ecologia_espacios',
    'apoyo': 'apoyo_institucional',
    'impacto': 'impacto',
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
    """Structured representation of a parsed external_id"""
    original: str
    section: str
    subsection: Optional[str]
    sub_subsection: Optional[str]
    construct: str
    item_label: str  # human-readable label (last component)


@dataclass
class ConstructMetadata:
    """Metadata for each construct per TECH4ZERO-MX v1.0"""
    name: str
    display_name: str
    pdf_section: str
    n_items_expected: int
    published_alpha_range: Tuple[float, float]
    scale_type: str  # 'likert_0_4' or 'frequency_0_4' or 'categorical'
    score_direction: str  # 'higher_means_more_risk' or 'higher_means_more_protective'
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
        published_alpha_range=(0.0, 0.0),  # N/A for demographics
        scale_type='categorical',
        score_direction='none',
        base_instrument='ECOBVQ-R, ZERO-R, INEGI',
        is_conditional=False,
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
        is_conditional=False,
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
        is_conditional=False,
    ),
    'victimizacion': ConstructMetadata(
        name='victimizacion',
        display_name='Victimización',
        pdf_section='D',
        n_items_expected=9,  # includes zero_victim_general
        published_alpha_range=(0.90, 0.95),
        scale_type='frequency_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='OBVQ-R (Gaete et al., 2021)',
        is_conditional=False,
    ),
    'perpetracion': ConstructMetadata(
        name='perpetracion',
        display_name='Agresión',
        pdf_section='E',
        n_items_expected=9,  # includes zero_agresor_general
        published_alpha_range=(0.88, 0.93),
        scale_type='frequency_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='OBVQ-R',
        is_conditional=False,
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
        is_conditional=False,
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
        is_conditional=False,
    ),
    'ecologia_espacios': ConstructMetadata(
        name='ecologia_espacios',
        display_name='Ecología del Bullying',
        pdf_section='G',
        n_items_expected=8,
        published_alpha_range=(0.0, 0.0),  # Not reported in PDF
        scale_type='likert_0_4',
        score_direction='higher_means_more_risk',
        base_instrument='Bronfenbrenner (1979), Astor et al. (2004)',
        is_conditional=True,  # Only for students who reported victimization
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
        is_conditional=False,
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
        is_conditional=False,
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
# PARSING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def parse_external_id(external_id: str) -> Optional[ParsedExternalID]:
    """
    Parse SURVEY_003 external_id into hierarchical components.
    
    Examples:
        survey_003_zero_general_curso 
          → section: general, subsection: curso, construct: demographic
        
        survey_003_zero_cyber_victima_mensajes
          → section: cyber, subsection: victima, sub_subsection: mensajes
          → construct: cybervictimizacion
    
    Returns:
        ParsedExternalID or None if parsing fails
    """
    if not external_id:
        return None
    
    # Extract everything after "survey_XXX_zero_"
    match = EXTERNAL_ID_PATTERN.match(external_id)
    if not match:
        return None
    
    remainder = match.group(1)  # e.g., "cyber_victima_mensajes"
    parts = remainder.split('_')
    
    if not parts:
        return None
    
    section = parts[0]
    subsection = parts[1] if len(parts) > 1 else None
    sub_subsection = parts[2] if len(parts) > 2 else None
    
    # Determine construct
    construct = _resolve_construct(section, subsection)
    
    # Item label is the last component (most specific)
    item_label = parts[-1] if parts else section
    
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
    
    Special case: cyber_victima → cybervictimizacion
                  cyber_agresor → cyberagresion
    """
    if section == 'cyber' and subsection:
        return CYBER_SUBSECTION_MAP.get(subsection, 'cyberbullying')
    
    return SECTION_TO_CONSTRUCT.get(section, 'unknown')


def get_construct_items(construct: str, all_external_ids: list[str]) -> FrozenSet[str]:
    """
    Get all external_ids belonging to a specific construct.
    
    Args:
        construct: e.g., 'victimizacion', 'cybervictimizacion'
        all_external_ids: list of all question external_ids
    
    Returns:
        Frozen set of external_ids for this construct
    """
    items = set()
    
    for ext_id in all_external_ids:
        parsed = parse_external_id(ext_id)
        if parsed and parsed.construct == construct:
            items.add(ext_id)
    
    return frozenset(items)


def is_global_screener(external_id: str) -> bool:
    """Check if question is a global screener (excluded from alpha)."""
    parsed = parse_external_id(external_id)
    if not parsed:
        return False
    
    return parsed.item_label in GLOBAL_SCREENERS


def is_demographic(external_id: str) -> bool:
    """Check if question is demographic."""
    parsed = parse_external_id(external_id)
    if not parsed:
        return False
    
    return parsed.construct == 'demographic'


def get_construct_metadata(construct: str) -> Optional[ConstructMetadata]:
    """Get metadata for a construct."""
    return CONSTRUCT_METADATA.get(construct)


def get_all_constructs() -> list[str]:
    """Get list of all construct names."""
    return list(CONSTRUCT_METADATA.keys())


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_construct_coverage(external_ids: list[str]) -> Dict[str, int]:
    """
    Validate that all constructs have expected number of items.
    
    Returns:
        Dict mapping construct → actual item count
    """
    counts = {}
    
    for construct in get_all_constructs():
        items = get_construct_items(construct, external_ids)
        counts[construct] = len(items)
    
    return counts


if __name__ == '__main__':
    # Self-test
    test_cases = [
        'survey_003_zero_general_curso',
        'survey_003_zero_clima_normas',
        'survey_003_zero_cyber_victima_mensajes',
        'survey_003_zero_victima_rumores',
        'survey_003_zero_victim_general',
    ]
    
    print("Parsing Test:")
    for test_id in test_cases:
        parsed = parse_external_id(test_id)
        if parsed:
            print(f"\n{test_id}")
            print(f"  section: {parsed.section}")
            print(f"  subsection: {parsed.subsection}")
            print(f"  sub_subsection: {parsed.sub_subsection}")
            print(f"  construct: {parsed.construct}")
            print(f"  item_label: {parsed.item_label}")
            print(f"  is_screener: {is_global_screener(test_id)}")
