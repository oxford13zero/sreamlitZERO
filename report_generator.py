# report_generator.py
"""
TECH4ZERO-MX — Report Generator
=================================
Generates a 5-chapter director report using Claude.
Each chapter makes a separate API call with targeted manual context.

Chapters:
    1. Diagnóstico — what is happening in this school
    2. Contexto — the bullying phenomenon explained simply
    3. Cómo intervenir — intervention protocols from manuals
    4. Comunidad educativa — engaging teachers, parents, students
    5. Plan de Acción — concrete school action plan

Usage:
    from report_generator import generate_full_report
    chapters, full_text = generate_full_report(ctx, manual_texts, client, model)
"""

import json
from datetime import datetime

# ── Chapter definitions ───────────────────────────────────────────────────────

CHAPTERS = [
    {
        "num":       1,
        "title":     "Diagnóstico: La Situación Actual de Su Escuela",
        "manual_keys": ["fenomeno"],
        "spinner":   "📊 Analizando los datos de su escuela...",
    },
    {
        "num":       2,
        "title":     "Contexto: Entendiendo el Bullying en Su Comunidad",
        "manual_keys": ["fenomeno", "enfoque"],
        "spinner":   "📚 Contextualizando los hallazgos con evidencia del Programa ZERO...",
    },
    {
        "num":       3,
        "title":     "Cómo Intervenir ante un Caso de Bullying",
        "manual_keys": ["intervencion"],
        "spinner":   "🛡️ Generando protocolo de intervención...",
    },
    {
        "num":       4,
        "title":     "Cómo Involucrar a Toda la Comunidad Educativa",
        "manual_keys": ["prevencion", "enfoque"],
        "spinner":   "👥 Elaborando estrategias de participación comunitaria...",
    },
    {
        "num":       5,
        "title":     "Plan de Acción para Su Establecimiento",
        "manual_keys": ["plan_de_accion"],
        "spinner":   "📋 Construyendo el Plan de Acción personalizado...",
    },
]


# ── Country/language context (mirrors app.py COUNTRY_CONTEXT) ────────────────

COUNTRY_CONTEXT = {
    "MX": {
        "idioma":        "español mexicano",
        "pais":          "México",
        "marco":         "Nueva Escuela Mexicana (NEM)",
        "ley":           "Ley General de Educación y protocolos SEP",
        "director_title":"Director(a)",
        "escuela_term":  "plantel",
        "bullying_term": "acoso escolar",
    },
    "CL": {
        "idioma":        "español chileno",
        "pais":          "Chile",
        "marco":         "Política de Convivencia Educativa del MINEDUC",
        "ley":           "Ley de Violencia Escolar (Ley 20.536)",
        "director_title":"Director(a) / Jefe(a) de UTP",
        "escuela_term":  "establecimiento educacional",
        "bullying_term": "acoso escolar",
    },
    "US": {
        "idioma":        "English",
        "pais":          "United States",
        "marco":         "School Safety and Anti-Bullying Policy",
        "ley":           "applicable federal and state anti-bullying regulations",
        "director_title":"Principal",
        "escuela_term":  "school",
        "bullying_term": "bullying",
    },
}

FORBIDDEN_TERMS_ES = (
    "prevalencia, percentil, IC 95%, Cronbach, p-valor, chi-cuadrado, "
    "correlación, constructo, variable, n=, estadísticamente significativo, coeficiente"
)
FORBIDDEN_TERMS_EN = (
    "prevalence, percentile, 95% CI, Cronbach, p-value, chi-square, "
    "correlation, construct, variable, n=, statistically significant, coefficient"
)

CAT_MAP = {
    "CRISIS":       "situación de emergencia que requiere acción inmediata",
    "INTERVENCION": "situación preocupante que requiere atención urgente",
    "ATENCION":     "situación que merece seguimiento y atención",
    "MONITOREO":    "situación bajo control",
    "SIN_DATOS":    None,
}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _base_instructions(ctx: dict, chapter_num: int, chapter_title: str) -> str:
    """Common instructions injected into every chapter prompt."""
    cc = COUNTRY_CONTEXT.get(ctx.get("school_country", "MX"), COUNTRY_CONTEXT["MX"])
    forbidden = FORBIDDEN_TERMS_EN if cc["idioma"] == "English" else FORBIDDEN_TERMS_ES

    return f"""You are a school climate and bullying prevention specialist for the ZERO Programme (Roland, University of Stavanger).
Write CHAPTER {chapter_num}: {chapter_title}

MANDATORY WRITING RULES:
1. Write entirely in {cc['idioma']}.
2. Target: 400-600 words for this chapter.
3. FORBIDDEN technical terms: {forbidden}
4. Use proportional language instead of exact percentages:
   "casi la mitad", "1 de cada 3", "la mayoría", "un grupo importante"
5. Tone: professional, warm, action-oriented. Not alarmist.
6. Reference framework: {cc['marco']} and {cc['ley']}.
7. Use "{cc['bullying_term']}" — avoid anglicisms.
8. If any data value is null, silently omit that subsection.
9. Address the {cc['director_title']} directly ("su escuela", "sus estudiantes").
10. Start the chapter directly with the heading: ## Capítulo {chapter_num}: {chapter_title}
    Do not add preamble or meta-commentary.
"""


def _data_block(ctx: dict) -> str:
    """Serialize the survey context into a readable block."""
    prev_lines = []
    for area, data in ctx.get("prevalencias", {}).items():
        cat = CAT_MAP.get(data.get("categoria", ""), None)
        pct = data.get("pct")
        if cat is None or pct is None:
            continue
        prev_lines.append(f"  - {area}: {pct}% de estudiantes afectados — {cat}")

    top3 = "\n".join(
        f"  - {x['area']} ({x['pct']}%)"
        for x in ctx.get("top3_riesgo", [])
        if x.get("pct") is not None
    )

    demo_lines = []
    for label, dist in ctx.get("demograficos", {}).items():
        vals = ", ".join(f"{k}: {v}" for k, v in dist.items())
        demo_lines.append(f"  - {label}: {vals}")

    tipo_lines = "\n".join(
        f"  - {k}: {v}" for k, v in ctx.get("tipologia", {}).items()
    )

    cyber = ctx.get("cyber_overlap")
    cyber_block = (
        f"  - Bullying tradicional: {cyber['victimas_tradicionales']} estudiantes\n"
        f"  - Cyberbullying: {cyber['cibervictimas']} estudiantes\n"
        f"  - Afectados en ambos: {cyber['ambos']} estudiantes"
    ) if cyber else "  - Sin datos de cyberbullying"

    risk = ctx.get("indice_riesgo", {})
    risk_val = risk.get("indice")
    if risk_val is None:
        risk_label = "no disponible"
    elif risk_val >= 60:
        risk_label = "alto"
    elif risk_val >= 40:
        risk_label = "moderado-alto"
    elif risk_val >= 20:
        risk_label = "moderado"
    else:
        risk_label = "bajo"

    return f"""SURVEY DATA FOR {ctx.get('escuela', 'the school').upper()}
Date: {ctx.get('fecha', datetime.now().strftime('%d/%m/%Y'))}
Students surveyed: {ctx.get('n_estudiantes', 'N/A')}
Overall risk level: {risk_label}

Findings by area:
{chr(10).join(prev_lines) or '  - No data available'}

Top 3 areas of concern:
{top3 or '  - No data available'}

Student profile distribution (Olweus typology):
{tipo_lines or '  - No data available'}

Traditional vs. cyberbullying:
{cyber_block}

Student demographics:
{chr(10).join(demo_lines) or '  - No demographic data'}
"""


def _build_chapter_prompt(
    chapter: dict,
    ctx: dict,
    manual_texts: dict,
) -> str:
    """Build the full prompt for one chapter."""

    instructions = _base_instructions(ctx, chapter["num"], chapter["title"])
    data = _data_block(ctx)

    # Gather manual context for this chapter
    manual_blocks = []
    for key in chapter["manual_keys"]:
        text = manual_texts.get(key, "")
        if text:
            label = {
                "fenomeno":     "MANUALS — The Bullying Phenomenon",
                "enfoque":      "MANUALS — Integrated Approach",
                "intervencion": "MANUALS — Intervention Protocols",
                "prevencion":   "MANUALS — Prevention Strategies",
                "plan_de_accion": "ACTION PLAN — ZERO Programme",
            }.get(key, key.upper())
            manual_blocks.append(f"=== {label} ===\n{text}")

    manual_section = (
        "\n\nSOURCE MATERIAL FROM ZERO PROGRAMME MANUALS\n"
        "Base your recommendations EXCLUSIVELY on this material.\n"
        "Do not invent protocols or strategies not found here.\n\n"
        + "\n\n".join(manual_blocks)
    ) if manual_blocks else (
        "\n\nNOTE: Manual content not available. "
        "Base recommendations on established ZERO Programme principles."
    )

    return f"{instructions}\n\n{data}{manual_section}"


# ── Main generator ────────────────────────────────────────────────────────────

def generate_full_report(
    ctx: dict,
    manual_texts: dict,
    client,
    model: str = "claude-haiku-4-5",   # swap to claude-opus-4-5 in production
    max_tokens_per_chapter: int = 1000,
    progress_callback=None,
) -> tuple[list[str], str]:
    """
    Generate all 5 chapters sequentially.

    Args:
        ctx:           Survey context dict (from _build_report_context in app.py)
        manual_texts:  Dict with keys fenomeno/enfoque/intervencion/prevencion/plan_de_accion
        client:        Initialized anthropic.Anthropic client
        model:         Claude model string
        max_tokens_per_chapter: Token limit per chapter (1000 ≈ 600-700 words)
        progress_callback: Optional callable(chapter_num, chapter_title, text)

    Returns:
        (chapters_list, full_document_string)
    """
    chapters_text = []

    for i, chapter in enumerate(CHAPTERS):
        prompt = _build_chapter_prompt(chapter, ctx, manual_texts)

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens_per_chapter,
            messages=[{"role": "user", "content": prompt}],
        )
        chapter_text = message.content[0].text
        chapters_text.append(chapter_text)

        if progress_callback:
            progress_callback(i + 1, chapter["title"], chapter_text)

    # ── Assemble full document ────────────────────────────────────────────────
    cc = COUNTRY_CONTEXT.get(ctx.get("school_country", "MX"), COUNTRY_CONTEXT["MX"])

    header = f"""# Informe TECH4ZERO-MX
## {ctx.get('escuela', 'Establecimiento Educacional')}
**{cc['director_title']}:** {ctx.get('director_title', '')}
**Fecha:** {ctx.get('fecha', datetime.now().strftime('%d/%m/%Y'))}
**Estudiantes analizados:** {ctx.get('n_estudiantes', 'N/A')}
**Marco de referencia:** {cc['marco']}

---

*Este informe ha sido generado con base en los resultados de la Encuesta de Clima Escolar
TECH4ZERO-MX y el conocimiento sistematizado del Programa ZERO (Erling Roland,
Universidad de Stavanger). Las recomendaciones están fundamentadas en evidencia
científica y en la experiencia de implementación en escuelas de Noruega, Chile y México.*

---
"""

    footer = """
---

## Equipo TECH4ZERO-MX

*"Este remedio está garantizado que sí funciona. Lo que resta ahora es saber tomárselo."*
— Profesor Erling Roland, Universidad de Stavanger

---
*Documento generado por TECH4ZERO-MX · Programa ZERO · Universidad de Stavanger*
"""

    full_document = header + "\n\n".join(chapters_text) + footer
    return chapters_text, full_document


# ── PDF generator ─────────────────────────────────────────────────────────────

def markdown_to_pdf(markdown_text: str, school_name: str) -> bytes:
    """
    Convert the markdown report to a styled PDF using reportlab.

    Returns:
        PDF as bytes (ready for st.download_button)
    """
    import io
    import re as _re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title=f"Informe TECH4ZERO — {school_name}",
        author="TECH4ZERO-MX",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_h1 = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontSize=18, textColor=colors.HexColor('#1a237e'),
        spaceAfter=12, spaceBefore=0,
    )
    style_h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#283593'),
        spaceAfter=8, spaceBefore=14,
    )
    style_h3 = ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontSize=12, textColor=colors.HexColor('#3949ab'),
        spaceAfter=6, spaceBefore=10,
    )
    style_body = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10.5, leading=16,
        spaceAfter=8, alignment=TA_JUSTIFY,
    )
    style_bullet = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontSize=10.5, leading=15,
        leftIndent=18, spaceAfter=4,
    )
    style_italic = ParagraphStyle(
        'Italic', parent=styles['Normal'],
        fontSize=10, leading=14,
        textColor=colors.HexColor('#546e7a'),
        spaceAfter=6,
    )
    style_meta = ParagraphStyle(
        'Meta', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#78909c'),
        alignment=TA_CENTER, spaceAfter=4,
    )

    story = []

    def escape_xml(text: str) -> str:
        """Escape XML special chars for ReportLab Paragraphs."""
        return (
            text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
        )

    def add_paragraph(text: str, style):
        clean = escape_xml(text.strip())
        if clean:
            story.append(Paragraph(clean, style))

    lines = markdown_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith('# '):
            add_paragraph(line[2:], style_h1)
        elif line.startswith('## '):
            # Chapter headings get a page break (except the first)
            if story:
                story.append(PageBreak())
            add_paragraph(line[3:], style_h2)
        elif line.startswith('### '):
            add_paragraph(line[4:], style_h3)
        elif line.startswith('**') and line.endswith('**') and len(line) > 4:
            # Bold-only lines treated as sub-headings
            inner = line[2:-2]
            add_paragraph(f"<b>{escape_xml(inner)}</b>", style_body)
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = '• ' + line[2:].strip()
            add_paragraph(bullet_text, style_bullet)
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            inner = line[1:-1]
            add_paragraph(f"<i>{escape_xml(inner)}</i>", style_italic)
        elif line.startswith('---'):
            story.append(HRFlowable(
                width='100%', thickness=0.5,
                color=colors.HexColor('#b0bec5'),
                spaceAfter=8, spaceBefore=8,
            ))
        elif line.strip() == '':
            story.append(Spacer(1, 4))
        else:
            # Handle inline bold (**text**) within normal paragraphs
            processed = _re.sub(
                r'\*\*(.+?)\*\*',
                lambda m: f'<b>{escape_xml(m.group(1))}</b>',
                escape_xml(line)
            )
            # Handle inline italic (*text*)
            processed = _re.sub(
                r'\*(.+?)\*',
                lambda m: f'<i>{m.group(1)}</i>',
                processed
            )
            if processed.strip():
                story.append(Paragraph(processed, style_body))

        i += 1

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
