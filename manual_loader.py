# manual_loader.py
"""
TECH4ZERO — Manual Loader v2.0
================================
Extracts and caches text from the ZERO programme manuals stored in /Manuales/.

Folder structure expected in the repository root:
    Manuales/
    ├── enfoque/          ← 4 PDFs — Manual_Enfoque_Integrado_1-4
    ├── fenomeno/         ← 4 DOCXs — Manual_Bullying_1-4
    ├── intervencion/     ← 4 PDFs — Manual_Intervención_1-4
    ├── prevencion/       ← 4 PDFs — Manual_Prevencion_1-4
    ├── plan_de_accion/   ← 1 DOCX — guia_plan_accion_final.docx
    └── Ejemplos/         ← 11 real school action plans (commented out — pending index)

Supports: .pdf, .docx, .doc, .txt, .md
"""

import os
import re
import streamlit as st

# ── Keywords that identify action-relevant sections ───────────────────────────
ACTION_KEYWORDS = [
    # Spanish
    'intervención', 'intervencion', 'acción', 'accion', 'protocolo',
    'recomendación', 'recomendacion', 'estrategia', 'implementar',
    'aplicar', 'paso', 'procedimiento', 'actividad', 'plan',
    'prevención', 'prevencion', 'resolución', 'resolucion',
    'descubrir', 'detectar', 'supervisión', 'supervision',
    'seguimiento', 'respuesta', 'comunidad', 'familia', 'padres',
    'equipo', 'pilar', 'objetivo', 'acoso', 'bullying', 'víctima',
    'agresor', 'testigo', 'docente', 'apoderado',
    # English
    'intervention', 'action', 'protocol', 'recommendation',
    'strategy', 'implement', 'procedure', 'step', 'prevention',
    'detection', 'follow-up', 'response', 'victim', 'aggressor',
]

MAX_CHARS_PER_CATEGORY = 25_000   # ~6,000 tokens
MAX_CHARS_PLAN         = 40_000   # full plan de acción guide
MAX_CHARS_EXAMPLES     = 15_000   # per example (3 examples max)


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as e:
        return f"[Error reading PDF {os.path.basename(path)}: {e}]"


def _extract_docx(path: str) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[Error reading DOCX {os.path.basename(path)}: {e}]"


def _extract_text_file(path: str) -> str:
    """Read plain text or markdown files."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file {os.path.basename(path)}: {e}]"


def _extract_file(path: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return _extract_pdf(path)
    elif ext in ('.docx', '.doc'):
        return _extract_docx(path)
    elif ext in ('.txt', '.md'):
        return _extract_text_file(path)
    else:
        return ""


# ── Section filtering ─────────────────────────────────────────────────────────

def _filter_relevant_sections(text: str, max_chars: int) -> str:
    """
    Extract paragraphs containing action keywords.
    Falls back to full text (truncated) if no keywords are found.
    """
    if not text:
        return ""

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    relevant = []
    total = 0

    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in ACTION_KEYWORDS):
            relevant.append(para)
            total += len(para)
            if total >= max_chars:
                break

    if not relevant:
        return text[:max_chars]

    return "\n\n".join(relevant)[:max_chars]


# ── Base directory detection ──────────────────────────────────────────────────

def _find_base_dir() -> str:
    """
    Locate the manuals root folder regardless of case (Manuales vs manuales)
    and regardless of the current working directory.
    """
    candidates = [
        "Manuales",
        "manuales",
        os.path.join(os.path.dirname(__file__), "Manuales"),
        os.path.join(os.path.dirname(__file__), "manuales"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return "Manuales"  # fallback — will return empty strings gracefully


def _find_subfolder(base_dir: str, name: str) -> str | None:
    """
    Find a subfolder case-insensitively.
    Returns the full path if found, None otherwise.
    """
    try:
        entries = os.listdir(base_dir)
    except OSError:
        return None
    for entry in entries:
        if entry.lower() == name.lower():
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                return full
    return None


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_category(category: str, base_dir: str = None) -> str:
    """
    Load and filter text from all files in a category subfolder.

    Args:
        category: 'fenomeno' | 'enfoque' | 'intervencion' | 'prevencion'
        base_dir: root folder (auto-detected if None)

    Returns:
        Filtered text string, max ~25,000 chars.
        Returns empty string if folder not found.
    """
    if base_dir is None:
        base_dir = _find_base_dir()

    folder = _find_subfolder(base_dir, category)
    if not folder:
        return ""

    all_text_parts = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        raw = _extract_file(fpath)
        if raw:
            all_text_parts.append(f"### [{fname}]\n{raw}")

    combined = "\n\n".join(all_text_parts)
    return _filter_relevant_sections(combined, MAX_CHARS_PER_CATEGORY)


@st.cache_data(ttl=3600)
def load_action_plan(base_dir: str = None) -> str:
    """
    Load the Plan de Acción guide from Manuales/plan_de_accion/.

    Returns:
        Text content up to MAX_CHARS_PLAN chars.
        Returns empty string if nothing found.
    """
    if base_dir is None:
        base_dir = _find_base_dir()

    # Option 1: subfolder plan_de_accion
    folder = _find_subfolder(base_dir, "plan_de_accion")
    if folder:
        parts = []
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                raw = _extract_file(fpath)
                if raw:
                    parts.append(f"### [{fname}]\n{raw}")
        combined = "\n\n".join(parts)
        return combined[:MAX_CHARS_PLAN]

    # Option 2: single .md file (legacy)
    for candidate in ["plan_de_accion_zero.md", "plan_de_accion.md"]:
        md_path = os.path.join(base_dir, candidate)
        if os.path.isfile(md_path):
            return _extract_text_file(md_path)[:MAX_CHARS_PLAN]

    return ""


# ── Ejemplos loader (commented out — pending index creation) ─────────────────
#
# @st.cache_data(ttl=3600)
# def load_examples(base_dir: str = None, max_examples: int = 3) -> str:
#     """
#     Load a sample of real school action plan examples from Manuales/Ejemplos/.
#     Limited to max_examples to control context window size.
#
#     Args:
#         base_dir:     root folder (auto-detected if None)
#         max_examples: maximum number of examples to load (default 3)
#
#     Returns:
#         Concatenated example text, up to MAX_CHARS_EXAMPLES * max_examples chars.
#         Returns empty string if folder not found.
#     """
#     if base_dir is None:
#         base_dir = _find_base_dir()
#
#     folder = _find_subfolder(base_dir, "Ejemplos")
#     if not folder:
#         return ""
#
#     files = sorted([
#         f for f in os.listdir(folder)
#         if os.path.isfile(os.path.join(folder, f))
#         and os.path.splitext(f)[1].lower() in ('.pdf', '.docx', '.doc', '.txt', '.md')
#     ])
#
#     # Once an index is created, select best examples based on school profile.
#     # For now, load the first max_examples files alphabetically.
#     selected = files[:max_examples]
#
#     parts = []
#     for fname in selected:
#         fpath = os.path.join(folder, fname)
#         raw = _extract_file(fpath)
#         if raw:
#             filtered = _filter_relevant_sections(raw, MAX_CHARS_EXAMPLES)
#             parts.append(f"=== EJEMPLO: {fname} ===\n{filtered}")
#
#     return "\n\n".join(parts)
#
# ─────────────────────────────────────────────────────────────────────────────


def get_manual_status(base_dir: str = None) -> dict:
    """
    Returns a status dict showing how many files are loaded per category.
    Used by the UI to show which manuals are available.
    """
    if base_dir is None:
        base_dir = _find_base_dir()

    status = {}

    for cat in ['fenomeno', 'enfoque', 'intervencion', 'prevencion']:
        folder = _find_subfolder(base_dir, cat)
        if folder:
            files = [
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and os.path.splitext(f)[1].lower() in ('.pdf', '.docx', '.doc', '.txt', '.md')
            ]
            status[cat] = len(files)
        else:
            status[cat] = 0

    # plan_de_accion
    plan_folder = _find_subfolder(base_dir, "plan_de_accion")
    if plan_folder:
        files = [
            f for f in os.listdir(plan_folder)
            if os.path.isfile(os.path.join(plan_folder, f))
        ]
        status['plan_de_accion'] = len(files)
    else:
        status['plan_de_accion'] = 0

    # Ejemplos (counted but not loaded yet)
    ejemplos_folder = _find_subfolder(base_dir, "Ejemplos")
    if ejemplos_folder:
        files = [
            f for f in os.listdir(ejemplos_folder)
            if os.path.isfile(os.path.join(ejemplos_folder, f))
            and os.path.splitext(f)[1].lower() in ('.pdf', '.docx', '.doc', '.txt', '.md')
        ]
        status['ejemplos'] = len(files)
    else:
        status['ejemplos'] = 0

    return status
