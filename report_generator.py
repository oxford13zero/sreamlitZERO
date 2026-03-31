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
    "CRISIS":       "requiere acción prioritaria",
    "INTERVENCION": "requiere atención",
    "ATENCION":     "merece seguimiento",
    "MONITOREO":    "bajo control",
    "SIN_DATOS":    None,
}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _get_cc(ctx: dict) -> dict:
    return COUNTRY_CONTEXT.get(ctx.get("school_country", "MX"), COUNTRY_CONTEXT["MX"])


def _forbidden(ctx: dict) -> str:
    cc = _get_cc(ctx)
    return FORBIDDEN_TERMS_EN if cc["idioma"] == "English" else FORBIDDEN_TERMS_ES


def _data_block(ctx: dict) -> str:
    """Compact survey data block — used by all chapter prompts."""
    prev_lines = []
    for area, data in ctx.get("prevalencias", {}).items():
        cat = CAT_MAP.get(data.get("categoria", ""), None)
        pct = data.get("pct")
        if cat is None or pct is None:
            continue
        prev_lines.append(f"  - {area}: {pct}% — {cat}")

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
        f"  - Tradicional: {cyber['victimas_tradicionales']} | Cyber: {cyber['cibervictimas']} | Ambos: {cyber['ambos']}"
    ) if cyber else "  - Sin datos"

    risk = ctx.get("indice_riesgo", {})
    risk_val = risk.get("indice")
    if risk_val is None:      risk_label = "no disponible"
    elif risk_val >= 60:      risk_label = "alto"
    elif risk_val >= 40:      risk_label = "moderado-alto"
    elif risk_val >= 20:      risk_label = "moderado"
    else:                     risk_label = "bajo"

    return (
        f"ESCUELA: {ctx.get('escuela', 'N/A')} | FECHA: {ctx.get('fecha', '')} | "
        f"ESTUDIANTES: {ctx.get('n_estudiantes', 'N/A')} | RIESGO GLOBAL: {risk_label}\n"
        f"\nÁREAS (todas):\n{chr(10).join(prev_lines) or '  - Sin datos'}\n"
        f"\nTOP 3:\n{top3 or '  - Sin datos'}\n"
        f"\nTIPOLOGÍA OLWEUS:\n{tipo_lines or '  - Sin datos'}\n"
        f"\nCYBER vs TRADICIONAL:\n{cyber_block}\n"
        f"\nDEMOGRAFÍA:\n{chr(10).join(demo_lines) or '  - Sin datos'}"
    )


def _manual_block(chapter: dict, manual_texts: dict) -> str:
    """Gather and label manual text for a chapter."""
    labels = {
        "fenomeno":       "MANUALES — Fenómeno del acoso escolar",
        "enfoque":        "MANUALES — Enfoque integrado",
        "intervencion":   "MANUALES — Protocolos de intervención",
        "prevencion":     "MANUALES — Estrategias de prevención",
        "plan_de_accion": "PLAN DE ACCIÓN — Programa ZERO",
    }
    blocks = []
    for key in chapter["manual_keys"]:
        text = manual_texts.get(key, "")
        if text:
            blocks.append(f"=== {labels.get(key, key.upper())} ===\n{text}")
    if blocks:
        return "\n\nFUENTE — MANUALES PROGRAMA ZERO\nBasa el contenido EXCLUSIVAMENTE en este material:\n\n" + "\n\n".join(blocks)
    return "\n\nNOTA: Manuales no disponibles. Usa principios generales del Programa ZERO."


def _build_chapter_prompt(chapter: dict, ctx: dict, manual_texts: dict) -> str:
    """Build a tight, chapter-specific prompt. Each chapter has explicit output rules."""
    cc       = _get_cc(ctx)
    forbidden = _forbidden(ctx)
    data     = _data_block(ctx)
    manuals  = _manual_block(chapter, manual_texts)
    num      = chapter["num"]
    title    = chapter["title"]

    # ── Common header (injected in all chapters) ──────────────
    header = (
        f"Eres especialista en convivencia escolar del Programa ZERO (Roland, Stavanger).\n"
        f"Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.\n"
        f"Términos prohibidos: {forbidden}.\n"
        f"Marco normativo: {cc['marco']}.\n"
        f"REGLA GLOBAL: sin redundancias entre capítulos. Cada capítulo tiene un rol único.\n"
        f"Comienza directamente con el encabezado markdown del capítulo.\n"
    )

    # ── Chapter-specific instructions ─────────────────────────
    if num == 1:
        instructions = (
            f"## Capítulo 1: {title}\n\n"
            f"INSTRUCCIONES ESTRICTAS:\n"
            f"- Máximo 250 palabras en total.\n"
            f"- Párrafo 1 (3 oraciones): panorama general — nivel de riesgo global en lenguaje simple.\n"
            f"- Lista de 3 a 5 ítems: hallazgos clave de la encuesta, uno por área, con el dato y su significado práctico.\n"
            f"- Párrafo final (2 oraciones): perfil de los estudiantes más afectados según tipología y demografía.\n"
            f"- NO incluyas recomendaciones ni llamados a la acción — eso va en capítulos posteriores.\n"
            f"- Usa frases proporcionales ('1 de cada 3', 'la mayoría') en vez de porcentajes exactos.\n"
        )

    elif num == 2:
        instructions = (
            f"## Capítulo 2: {title}\n\n"
            f"INSTRUCCIONES ESTRICTAS:\n"
            f"- Máximo 300 palabras en total.\n"
            f"- Párrafo 1 (3 oraciones): qué es el acoso escolar según el Programa ZERO — definición breve y clara.\n"
            f"- Párrafo 2 (3 oraciones): CONECTA la definición con los datos reales de ESTA escuela. "
            f"  Menciona los tipos de acoso encontrados (tradicional, cyber, etc.) y en qué se manifiestan aquí.\n"
            f"- Párrafo 3 (3 oraciones): explica por qué las cifras de esta escuela son importantes "
            f"  en el contexto del Programa ZERO — qué nos dice sobre la dinámica escolar.\n"
            f"- NO repitas el nivel de riesgo global ya mencionado en el Capítulo 1.\n"
            f"- NO incluyas recomendaciones.\n"
        )

    elif num == 3:
        instructions = (
            f"## Capítulo 3: {title}\n\n"
            f"INSTRUCCIONES ESTRICTAS:\n"
            f"- Máximo 350 palabras en total.\n"
            f"- Basándote en los manuales, describe el protocolo de intervención en 3 pasos concretos:\n"
            f"  Paso 1: cómo detectar y confirmar un caso.\n"
            f"  Paso 2: qué hace el adulto responsable con la víctima y el agresor por separado.\n"
            f"  Paso 3: seguimiento post-incidente con la clase y los padres.\n"
            f"- Cada paso: máximo 2 oraciones, lenguaje de acción (verbos en infinitivo).\n"
            f"- Un párrafo final (2 oraciones) sobre cuándo involucrar a externos (familia, autoridades).\n"
            f"- NO uses la frase 'acción inmediata' — describe la acción específica en su lugar.\n"
            f"- NO repitas datos de la encuesta ya presentados en los capítulos anteriores.\n"
        )

    elif num == 4:
        instructions = (
            f"## Capítulo 4: {title}\n\n"
            f"INSTRUCCIONES ESTRICTAS:\n"
            f"- Máximo 350 palabras en total.\n"
            f"- 3 secciones breves, una por actor:\n"
            f"  **Docentes**: 2 acciones concretas basadas en los manuales.\n"
            f"  **Padres y apoderados**: 2 acciones concretas basadas en los manuales.\n"
            f"  **Estudiantes**: 2 acciones concretas basadas en los manuales.\n"
            f"- Cada acción: 1 oración, verbo en infinitivo, específica y realizable.\n"
            f"- NO repitas el protocolo de intervención del Capítulo 3.\n"
            f"- NO uses 'acción inmediata' — describe la acción específica.\n"
        )

    elif num == 5:
        instructions = (
            f"## Capítulo 5: {title}\n\n"
            f"INSTRUCCIONES ESTRICTAS:\n"
            f"- Máximo 400 palabras en total.\n"
            f"- Basándote en el Plan de Acción ZERO y los datos de esta escuela específica:\n"
            f"  **Objetivo del año**: 1 objetivo cuantificable y realista para este establecimiento.\n"
            f"  **Próximas 2 semanas**: 2 acciones concretas e inmediatas.\n"
            f"  **Próximo mes**: 2 acciones de mediano plazo.\n"
            f"  **Durante el ciclo escolar**: 2 iniciativas permanentes.\n"
            f"- Cada acción: 1 oración, verbo en infinitivo, específica para esta escuela.\n"
            f"- El objetivo debe incluir un número (ej: 'reducir en un 20%' o 'implementar 3 actividades').\n"
            f"- NO repitas recomendaciones ya mencionadas en capítulos anteriores.\n"
            f"- Cierra con 1 oración motivacional firmada: *Equipo TECH4ZERO-MX*.\n"
        )

    else:
        instructions = f"## Capítulo {num}: {title}\n\nEscribe el contenido de este capítulo en máximo 300 palabras.\n"

    return f"{header}\n{instructions}\nDATOS DE LA ENCUESTA:\n{data}{manuals}"


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
