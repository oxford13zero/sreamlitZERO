# report_generator.py
"""
TECH4ZERO — Report Generator v2.0
===================================
Generates TWO documents from one button click:

  Document 1 — Informe de Diagnóstico (3 sections, ~10-15 pages)
    S1: ¿Qué encontramos en tu escuela?       (Executive findings)
    S2: Cómo leemos estos resultados          (Methodology in plain language)
    S3: El clima de tu escuela                (Full analysis with graphs)

  Document 2 — Plan de Acción (4 sections, ~8-12 pages)
    S4: Plan de Acción                        (4 Pillars, dynamic, semi-completed)
    S5: Errores comunes a evitar
    S6: Cómo saber si estamos avanzando       (Monitoring + re-survey)
    S7: Recursos y referencias

Each section is a separate API call so progress can be shown to the user.
The existing markdown_to_pdf() function is reused for both documents.

Country context (MX/CL/US) is applied throughout — no hardcoded country.
"""

import json
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# COUNTRY CONTEXT
# ══════════════════════════════════════════════════════════════

COUNTRY_CONTEXT = {
    "MX": {
        "idioma":         "español mexicano",
        "pais":           "México",
        "marco":          "Nueva Escuela Mexicana (NEM)",
        "ley":            "Ley General de Educación y protocolos SEP contra el acoso escolar",
        "ley_cita":       (
            "Artículo 73 de la Ley General de Educación (LGE), que obliga a las escuelas "
            "a establecer mecanismos para prevenir, atender y erradicar la violencia escolar; "
            "y los Lineamientos para la Prevención, Detección y Actuación en Casos de Abuso "
            "Sexual Infantil, Acoso Escolar y Maltrato de la SEP (2019)."
        ),
        "director_title": "Director(a)",
        "escuela_term":   "plantel",
        "bullying_term":  "acoso escolar",
        "saludo":         "Estimado(a) Director(a):",
        "encargado_term": "Encargado(a) de Convivencia Escolar",
        "nivel_term":     "nivel educativo",
    },
    "CL": {
        "idioma":         "español chileno",
        "pais":           "Chile",
        "marco":          "Política de Convivencia Educativa del MINEDUC",
        "ley":            "Ley de Violencia Escolar (Ley 20.536) y protocolos MINEDUC",
        "ley_cita":       (
            "Ley N° 20.536 sobre Violencia Escolar, que obliga a todo establecimiento "
            "educacional a contar con un Encargado de Convivencia Escolar, un Reglamento "
            "Interno que regule las conductas de acoso, y un protocolo de actuación ante "
            "situaciones de violencia (Art. 16 A, B y C). El incumplimiento puede derivar "
            "en sanciones administrativas por parte de la Superintendencia de Educación."
        ),
        "director_title": "Director(a)",
        "escuela_term":   "establecimiento educacional",
        "bullying_term":  "acoso escolar",
        "saludo":         "Estimado(a) Director(a):",
        "encargado_term": "Encargado(a) de Convivencia Escolar",
        "nivel_term":     "nivel educativo",
    },
    "US": {
        "idioma":         "English",
        "pais":           "United States",
        "marco":          "School Safety and Anti-Bullying Policy",
        "ley":            "applicable federal and state anti-bullying regulations",
        "ley_cita":       (
            "Title IV of the Elementary and Secondary Education Act (Every Student Succeeds Act, "
            "ESSA, 2015), which requires schools to provide a safe and supportive learning "
            "environment. Most states have enacted specific anti-bullying laws requiring schools "
            "to maintain a written anti-bullying policy, investigate reports promptly, and "
            "notify parents of incidents. Schools are encouraged to consult their state "
            "education agency for specific compliance requirements."
        ),
        "director_title": "Principal",
        "escuela_term":   "school",
        "bullying_term":  "bullying",
        "saludo":         "Dear Principal,",
        "encargado_term": "School Counselor / Dean of Students",
        "nivel_term":     "grade level",
    },
}

# ══════════════════════════════════════════════════════════════
# SECTION DEFINITIONS
# ══════════════════════════════════════════════════════════════

# Document 1 — Diagnostic Report
DIAGNOSTIC_SECTIONS = [
    {
        "num":         1,
        "doc":         1,
        "title":       "¿Qué encontramos en tu escuela?",
        "manual_keys": [],
        "spinner":     "📊 Escribiendo los hallazgos principales...",
    },
    {
        "num":         2,
        "doc":         1,
        "title":       "Cómo leemos estos resultados",
        "manual_keys": [],
        "spinner":     "📐 Explicando la metodología en lenguaje simple...",
    },
    {
        "num":         3,
        "doc":         1,
        "title":       "El clima de tu escuela: análisis completo",
        "manual_keys": [],
        "spinner":     "🔬 Desarrollando el análisis completo...",
    },
]

# Document 2 — Action Plan
ACTION_SECTIONS = [
    {
        "num":         4,
        "doc":         2,
        "title":       "Plan de Acción",
        "manual_keys": ["plan_de_accion", "intervencion", "prevencion", "enfoque"],
        "spinner":     "📋 Construyendo el Plan de Acción personalizado...",
    },
    {
        "num":         5,
        "doc":         2,
        "title":       "Errores comunes a evitar",
        "manual_keys": ["intervencion", "prevencion"],
        "spinner":     "⚠️ Identificando errores comunes...",
    },
    {
        "num":         6,
        "doc":         2,
        "title":       "Cómo saber si estamos avanzando",
        "manual_keys": ["plan_de_accion"],
        "spinner":     "📈 Elaborando estrategia de seguimiento...",
    },
    {
        "num":         7,
        "doc":         2,
        "title":       "Recursos y referencias",
        "manual_keys": [],
        "spinner":     "📚 Compilando recursos y referencias...",
    },
]

ALL_SECTIONS = DIAGNOSTIC_SECTIONS + ACTION_SECTIONS

# Semáforo label map
CAT_LABEL = {
    "CRISIS":       "CRISIS — requiere acción prioritaria",
    "INTERVENCION": "INTERVENCIÓN — requiere atención urgente",
    "ATENCION":     "ATENCIÓN — merece seguimiento",
    "MONITOREO":    "MONITOREO — bajo control",
    "SIN_DATOS":    None,
}

CAT_PLAIN = {
    "CRISIS":       "crisis",
    "INTERVENCION": "intervención urgente",
    "ATENCION":     "atención",
    "MONITOREO":    "monitoreo",
    "SIN_DATOS":    "sin datos",
}


# ══════════════════════════════════════════════════════════════
# DATA BLOCK BUILDER
# ══════════════════════════════════════════════════════════════

def _get_cc(ctx: dict) -> dict:
    return COUNTRY_CONTEXT.get(
        ctx.get("school_country", ctx.get("pais", "MX"))[:2].upper(),
        COUNTRY_CONTEXT["MX"]
    )


def _data_block(ctx: dict) -> str:
    """
    Build a rich, plain-language data block from the survey context dict.
    Includes absolute counts, percentages, CIs, subgroups, and ecology.
    """
    n = ctx.get("n_estudiantes", 0) or 1

    # Prevalence
    prev_lines = []
    for area, data in ctx.get("prevalencias", {}).items():
        cat   = CAT_PLAIN.get(data.get("categoria", ""), None)
        pct   = data.get("pct")
        n_af  = data.get("n_afectados", data.get("n"))
        n_to  = data.get("n_total", n)
        ci_l  = data.get("ci_lower")
        ci_u  = data.get("ci_upper")
        if cat is None or pct is None:
            continue
        ci_str = f" (rango confiable: {ci_l}%-{ci_u}%)" if ci_l and ci_u else ""
        prev_lines.append(
            f"  - {area}: {n_af} de {n_to} estudiantes ({pct}%){ci_str} — nivel {cat}"
        )

    # Top 3
    top3_lines = []
    for x in ctx.get("top3_riesgo", []):
        if x.get("pct") is None:
            continue
        top3_lines.append(
            f"  - {x['area']}: {x.get('n','?')} de {x.get('n_total', n)} estudiantes ({x['pct']}%)"
        )

    # Subgroups by grade and gender
    sub = ctx.get("subgrupos_reporte", {})
    sub_lines = []
    for key, rows in sub.items():
        if not rows:
            continue
        label = {
            "agresion_por_grado":       "Agresores por grado",
            "victimizacion_por_grado":  "Víctimas por grado",
            "agresion_por_genero":      "Agresores por género",
            "victimizacion_por_genero": "Víctimas por género",
        }.get(key, key)
        top = rows[:3]
        parts = [f"{r['grupo']} ({r['pct']}%, n={r['n']})" for r in top]
        sub_lines.append(f"  - {label}: {', '.join(parts)}")

    # Ecology hotspots
    eco_lines = []
    for e in ctx.get("ecologia_reporte", [])[:5]:
        eco_lines.append(
            f"  - {e['lugar']}: puntuación media {e['puntuacion_media']} "
            f"({e['pct_alta_frecuencia']}% frecuencia alta)"
        )

    # Typology
    tipo_lines = []
    for k, v in ctx.get("tipologia", {}).items():
        if isinstance(v, dict):
            tipo_lines.append(f"  - {k}: {v.get('n','?')} estudiantes ({v.get('pct','?')})")

    # Demographics
    demo_lines = []
    for label, dist in ctx.get("demograficos", {}).items():
        parts = []
        for cat_val, val in dist.items():
            if isinstance(val, dict):
                parts.append(f"{cat_val}: {val.get('n','?')} ({val.get('pct','?')})")
            else:
                parts.append(f"{cat_val}: {val}")
        demo_lines.append(f"  - {label}: {', '.join(parts)}")

    # Cyber overlap
    cyber = ctx.get("cyber_overlap")
    if cyber:
        cyber_block = (
            f"  - Bullying tradicional: {cyber['victimas_tradicionales']} estudiantes "
            f"({cyber.get('pct_tradicionales','?')}%)\n"
            f"  - Cyberbullying: {cyber['cibervictimas']} estudiantes "
            f"({cyber.get('pct_cyber','?')}%)\n"
            f"  - Afectados en AMBOS: {cyber['ambos']} estudiantes "
            f"({cyber.get('pct_ambos_de_trad','?')}% de las víctimas tradicionales)"
        )
    else:
        cyber_block = "  - Sin datos de cyberbullying"

    # Risk index
    risk     = ctx.get("indice_riesgo", {})
    risk_val = risk.get("indice")
    risk_r   = risk.get("componente_riesgo")
    risk_p   = risk.get("componente_protector")
    if risk_val is None:   risk_label = "no disponible"
    elif risk_val >= 60:   risk_label = "alto"
    elif risk_val >= 40:   risk_label = "moderado-alto"
    elif risk_val >= 20:   risk_label = "moderado"
    else:                  risk_label = "bajo"
    risk_str = f"{risk_val}/100 — nivel {risk_label}" if risk_val else "no disponible"
    if risk_r and risk_p:
        risk_str += f" (factores de riesgo: {risk_r}/100 | factores protectores: {risk_p}/100)"

    return (
        f"═══ DATOS DE LA ENCUESTA ═══\n"
        f"Escuela: {ctx.get('escuela', 'N/A')}\n"
        f"País: {ctx.get('pais', 'N/A')}\n"
        f"Fecha: {ctx.get('fecha', '')}\n"
        f"Total estudiantes encuestados: {ctx.get('n_estudiantes', 'N/A')}\n"
        f"Índice de riesgo escolar: {risk_str}\n"
        f"\nRESULTADOS POR ÁREA:\n"
        f"{chr(10).join(prev_lines) or '  - Sin datos'}\n"
        f"\nÁREAS MÁS CRÍTICAS (top 3):\n"
        f"{chr(10).join(top3_lines) or '  - Sin datos'}\n"
        f"\nAGRESORES Y VÍCTIMAS POR GRADO Y GÉNERO:\n"
        f"{chr(10).join(sub_lines) or '  - Sin datos de subgrupos'}\n"
        f"\nLUGARES DEL ESTABLECIMIENTO CON MÁS AGRESIONES:\n"
        f"{chr(10).join(eco_lines) or '  - Sin datos de ecología'}\n"
        f"\nTIPOLOGÍA DE ESTUDIANTES (modelo Olweus):\n"
        f"{chr(10).join(tipo_lines) or '  - Sin datos'}\n"
        f"\nBULLYING TRADICIONAL vs CYBERBULLYING:\n"
        f"{cyber_block}\n"
        f"\nCOMPOSICIÓN DE LA MUESTRA:\n"
        f"{chr(10).join(demo_lines) or '  - Sin datos'}"
    )


def _manual_block(section: dict, manual_texts: dict) -> str:
    """Gather manual text for a section."""
    labels = {
        "fenomeno":       "MANUALES — Fenómeno del acoso escolar",
        "enfoque":        "MANUALES — Enfoque integrado",
        "intervencion":   "MANUALES — Protocolos de intervención",
        "prevencion":     "MANUALES — Estrategias de prevención",
        "plan_de_accion": "GUÍA — Plan de Acción Programa ZERO",
    }
    blocks = []
    for key in section.get("manual_keys", []):
        text = manual_texts.get(key, "")
        if text:
            blocks.append(f"=== {labels.get(key, key.upper())} ===\n{text}")
    if blocks:
        return (
            "\n\nFUENTE — MANUALES PROGRAMA ZERO\n"
            "Basa el contenido EXCLUSIVAMENTE en este material:\n\n"
            + "\n\n".join(blocks)
        )
    return ""


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDERS — DOCUMENT 1 (DIAGNOSTIC)
# ══════════════════════════════════════════════════════════════

def _prompt_s1(ctx: dict, data: str, cc: dict) -> str:
    """
    Section 1 — Executive findings in plain language.
    No statistics jargon. Direct. Goes straight to findings.
    """
    # Build semáforo summary
    prev = ctx.get("prevalencias", {})
    semaforo_lines = []
    for area, d in prev.items():
        cat = CAT_LABEL.get(d.get("categoria", ""), None)
        pct = d.get("pct")
        n_af = d.get("n_afectados", d.get("n", "?"))
        n_to = d.get("n_total", ctx.get("n_estudiantes", "?"))
        if cat and pct is not None:
            semaforo_lines.append(f"  - {area}: {n_af} de {n_to} estudiantes ({pct}%) — {cat}")

    top3 = ctx.get("top3_riesgo", [])
    top3_str = ", ".join(
        f"{x['area']} ({x['pct']}%)"
        for x in top3 if x.get("pct") is not None
    ) or "sin datos"

    risk_val = ctx.get("indice_riesgo", {}).get("indice")
    risk_label = (
        "alto" if risk_val and risk_val >= 60 else
        "moderado-alto" if risk_val and risk_val >= 40 else
        "moderado" if risk_val and risk_val >= 20 else
        "bajo" if risk_val else "no disponible"
    )

    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO (Roland, Stavanger).
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.
País: {cc['pais']}. Marco normativo: {cc['marco']}.

REGLA ABSOLUTA: NUNCA uses términos estadísticos. PROHIBIDO: prevalencia, percentil, IC 95%, \
Cronbach, p-valor, chi-cuadrado, correlación, constructo, variable, estadísticamente significativo, \
coeficiente. En su lugar usa lenguaje cotidiano: "de cada 10 estudiantes", "la mayoría", "casi la mitad".

## Sección 1: ¿Qué encontramos en tu escuela?

INSTRUCCIONES:
- Máximo 400 palabras.
- Tono: directo, empático, profesional. Va directo a los hallazgos sin introducción larga.
- Párrafo 1 (3-4 oraciones): El hallazgo más importante de la encuesta. \
  Menciona el índice de riesgo ({risk_label}, {risk_val}/100 si disponible) en lenguaje simple. \
  Ejemplo de lenguaje correcto: "Aproximadamente 1 de cada 4 estudiantes reporta haber \
  sido agredido con frecuencia. Esto sitúa a {ctx.get('escuela','la escuela')} \
  en una situación que requiere atención."
- Lista "Lo que encontramos": para CADA área con datos, escribe 1 línea con \
  cuántos estudiantes están afectados en lenguaje natural y qué significa. \
  Usa los datos exactos: {chr(10).join(semaforo_lines) or '(ver datos adjuntos)'}
- Párrafo 2 (3 oraciones): Las 3 áreas más críticas son: {top3_str}. \
  Explica brevemente por qué estas áreas son prioritarias para esta escuela.
- Párrafo 3 (2-3 oraciones): Perfil de estudiantes más afectados. \
  Usa los datos de subgrupos por grado y género de los datos adjuntos. \
  Menciona grados y géneros específicos explícitamente.
- NO incluyas recomendaciones — eso va en el Documento 2.
- Cierra con 1 oración que enmarque los hallazgos como punto de partida, no como veredicto.

DATOS:
{data}
"""


def _prompt_s2(ctx: dict, data: str, cc: dict) -> str:
    """
    Section 2 — Methodology explained in plain language.
    What was measured, how many, what the numbers mean.
    """
    n = ctx.get("n_estudiantes", "N/A")

    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.

REGLA ABSOLUTA: Explica los conceptos estadísticos en lenguaje cotidiano. \
Si mencionas un término técnico, SIEMPRE explícalo inmediatamente en una frase simple.

## Sección 2: Cómo leemos estos resultados

INSTRUCCIONES:
- Máximo 400 palabras.
- Subsección 2.1 — La encuesta (2 párrafos):
  • Qué mide la encuesta TECH4ZERO: clima escolar, victimización, agresión, cyberbullying, \
    relaciones, y los espacios físicos donde ocurren las agresiones.
  • Cuántos estudiantes participaron: {n} estudiantes. \
    Explica en lenguaje simple qué significa este número para la confianza en los resultados. \
    Usa analogía si ayuda: "Es como si hubiéramos preguntado a cada estudiante de manera anónima..."
- Subsección 2.2 — Cómo interpretar los números (3-4 párrafos cortos):
  • Qué significa el semáforo (CRISIS/INTERVENCIÓN/ATENCIÓN/MONITOREO): \
    explícalo con una analogía médica o de temperatura — algo que el director reconozca.
  • Qué significa "frecuente" en este instrumento: \
    agresión que ocurre al menos una vez al mes durante más de 3 meses.
  • Qué es el índice de riesgo escolar: un número de 0 a 100 que combina \
    los factores de riesgo y los factores protectores de la escuela. \
    Explica la diferencia entre ambos en 1 oración cada uno.
  • Qué son los perfiles Olweus (agresor, víctima, agresor-víctima, no involucrado): \
    explica cada uno en 1 oración simple, sin jerga.
- Subsección 2.3 — Lo que esta encuesta NO mide (1 párrafo):
  • No identifica estudiantes individuales (es anónima).
  • No reemplaza la observación directa del docente.
  • Los números son una fotografía de un momento — el contexto escolar puede cambiar.
- NO incluyas datos específicos de esta escuela — eso va en Sección 3.

DATOS (para referencia de contexto solamente):
{data}
"""


def _prompt_s3(ctx: dict, data: str, cc: dict) -> str:
    """
    Section 3 — Full analysis in plain language.
    All findings explained with references to graphs.
    Explicit grade and gender naming.
    """
    # Build subgroup narrative hints
    sub = ctx.get("subgrupos_reporte", {})
    agr_grado = sub.get("agresion_por_grado", [])
    vic_grado  = sub.get("victimizacion_por_grado", [])
    agr_gen    = sub.get("agresion_por_genero", [])
    vic_gen    = sub.get("victimizacion_por_genero", [])

    agr_grado_str = ", ".join(
        f"{r['grupo']} ({r['pct']}%)" for r in agr_grado[:3]
    ) or "sin datos por grado"
    vic_grado_str = ", ".join(
        f"{r['grupo']} ({r['pct']}%)" for r in vic_grado[:3]
    ) or "sin datos por grado"
    agr_gen_str = ", ".join(
        f"{r['grupo']} ({r['pct']}%)" for r in agr_gen
    ) or "sin datos por género"
    vic_gen_str = ", ".join(
        f"{r['grupo']} ({r['pct']}%)" for r in vic_gen
    ) or "sin datos por género"

    eco = ctx.get("ecologia_reporte", [])
    eco_str = ", ".join(
        f"{e['lugar']} (puntuación {e['puntuacion_media']})" for e in eco[:5]
    ) or "sin datos de espacios"

    cyber = ctx.get("cyber_overlap")
    cyber_str = (
        f"{cyber['victimas_tradicionales']} estudiantes con bullying tradicional, "
        f"{cyber['cibervictimas']} con cyberbullying, "
        f"{cyber['ambos']} afectados en ambos simultáneamente"
        if cyber else "sin datos de cyberbullying"
    )

    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.
País: {cc['pais']}.

REGLA ABSOLUTA: Lenguaje cotidiano siempre. Menciona EXPLÍCITAMENTE grados y géneros \
cuando los datos lo permitan. No generalices cuando tienes datos específicos.

## Sección 3: El clima de tu escuela — análisis completo

INSTRUCCIONES (máximo 700 palabras en total):

**3.1 Panorama general de convivencia**
- 1 párrafo: nivel de riesgo global de la escuela con el semáforo correspondiente. \
  Explica qué significa ese nivel para esta escuela específica, no en abstracto.
- Menciona las áreas en CRISIS o INTERVENCIÓN primero; luego las que están bajo control.

**3.2 ¿Quiénes son más afectados? — Por grado y género**
- Nota introductoria: "El gráfico adjunto muestra el porcentaje de agresores y víctimas \
  por grado y género. A continuación interpretamos lo que nos dicen estos datos:"
- Agresores: Los grados con mayor nivel de agresión son {agr_grado_str}. \
  Por género: {agr_gen_str}. Explica qué significa esto para la escuela \
  (ej: qué debe hacer un docente de ese grado con esa información).
- Víctimas: Los grados con mayor victimización son {vic_grado_str}. \
  Por género: {vic_gen_str}. Explica la implicación práctica.
- Si hay un grado o género que aparece tanto en agresores como en víctimas, \
  señálalo explícitamente — es una señal de alerta importante.

**3.3 ¿Dónde ocurren las agresiones? — Espacios de riesgo**
- Nota: "El gráfico adjunto muestra los espacios físicos del establecimiento \
  donde se reportan más agresiones."
- Los espacios más críticos son: {eco_str}.
- Explica qué significa supervisar activamente esos espacios en la práctica. \
  Qué puede hacer un docente o asistente hoy mismo en esos lugares.

**3.4 Perfiles de estudiantes — agresor, víctima, agresor-víctima**
- Explica los 4 perfiles Olweus en 1 oración cada uno.
- Presenta los conteos de esta escuela para cada perfil (usa datos de tipología).
- Destaca especialmente el perfil agresor-víctima si hay datos: \
  "Estos estudiantes son los que más necesitan apoyo porque están en ambos roles."

**3.5 Bullying tradicional y cyberbullying**
- Datos: {cyber_str}.
- Explica qué significa el solapamiento en términos prácticos: \
  "Cuando un estudiante es víctima en ambos espacios, el daño se multiplica \
  porque no tiene un espacio seguro ni en la escuela ni en casa."

**3.6 Factores protectores — lo que está funcionando**
- Identifica 2-3 áreas donde la escuela muestra fortalezas \
  (áreas con MONITOREO o puntajes protectores altos).
- Enmarcar esto como recursos que el plan de acción puede aprovechar.

CIERRE DE DOCUMENTO 1 (1 párrafo):
- Frase de transición hacia el Documento 2: \
  "Con este diagnóstico como base, el Plan de Acción que acompaña este informe \
  propone acciones concretas adaptadas a la realidad de {ctx.get('escuela','esta escuela')}."

DATOS:
{data}
"""


# ══════════════════════════════════════════════════════════════
# PROMPT BUILDERS — DOCUMENT 2 (ACTION PLAN)
# ══════════════════════════════════════════════════════════════

def _prompt_s4(ctx: dict, data: str, cc: dict, manuals: str) -> str:
    """
    Section 4 — Action Plan, 4 Pillars, semi-completed.
    Claude fills the WHAT and WHY. School fills the WHO and WHEN.
    """
    n = ctx.get("n_estudiantes", "?")
    top3 = ctx.get("top3_riesgo", [])
    top3_str = ", ".join(
        f"{x['area']} ({x['pct']}%)" for x in top3 if x.get("pct") is not None
    ) or "las áreas identificadas en el diagnóstico"

    risk_val = ctx.get("indice_riesgo", {}).get("indice")
    if risk_val and risk_val >= 60:
        obj_guidance = (
            "El nivel de riesgo es alto. El Programa ZERO recomienda un objetivo conservador "
            "para el primer año: reducir entre 10% y 15% el número de estudiantes afectados "
            "en las áreas más críticas. Un objetivo mayor genera frustración y abandono del plan."
        )
        urgency = "CRISIS — las primeras acciones deben iniciarse esta semana"
    elif risk_val and risk_val >= 40:
        obj_guidance = (
            "El nivel de riesgo es moderado-alto. El Programa ZERO recomienda apuntar a una "
            "reducción del 15% en las áreas más críticas, combinada con 2-3 iniciativas "
            "de prevención nuevas durante el año escolar."
        )
        urgency = "INTERVENCIÓN URGENTE — las primeras acciones deben iniciarse este mes"
    elif risk_val and risk_val >= 20:
        obj_guidance = (
            "El nivel de riesgo es moderado. El Programa ZERO recomienda mantener las áreas "
            "bajo control y reducir un 10% en las áreas que aún requieren atención, "
            "con foco en fortalecer los factores protectores."
        )
        urgency = "ATENCIÓN — las primeras acciones deben iniciarse este trimestre"
    else:
        obj_guidance = (
            "El nivel de riesgo es bajo o no disponible. Formula un objetivo de mantenimiento "
            "y mejora incremental: implementar al menos 3 actividades preventivas nuevas "
            "y realizar la encuesta de seguimiento al final del año."
        )
        urgency = "MONITOREO — mantener y fortalecer"

    eco = ctx.get("ecologia_reporte", [])
    eco_top = [e["lugar"] for e in eco[:3]] if eco else ["patios", "pasillos", "baños"]

    sub = ctx.get("subgrupos_reporte", {})
    agr_grado = sub.get("agresion_por_grado", [])
    vic_grado  = sub.get("victimizacion_por_grado", [])
    priority_grades = list({
        r["grupo"] for r in (agr_grado[:2] + vic_grado[:2])
    })
    priority_grades_str = ", ".join(priority_grades) if priority_grades else "los grados más afectados"

    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO (Roland, Stavanger), \
con experiencia en {cc['pais']}.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.
Marco normativo: {cc['marco']}.
Referencia legal obligatoria: {cc['ley_cita']}

PROPÓSITO DE ESTE DOCUMENTO:
Crear un Plan de Acción semi-completado. Claude genera el QUÉ hacer y el POR QUÉ \
basado en los datos reales de esta escuela. La escuela completa el QUIÉN y el CUÁNDO.

FORMATO OBLIGATORIO para cada acción:
**[Nombre de la acción]**
Qué hacer: [descripción específica, 2-3 oraciones, basada en manuales ZERO]
Por qué es prioritaria para {ctx.get('escuela','esta escuela')}: \
[justificación con datos reales de la encuesta]
Responsable: _______________________
Fecha de inicio: ___________________
Indicador de logro: [pre-completado por Claude con métrica concreta]

## Sección 4: Plan de Acción — {ctx.get('escuela','la escuela')}

**Nota inicial (1 párrafo breve):**
Urgencia: {urgency}. Menciona las áreas críticas: {top3_str}.
Referencia legal: menciona {cc['ley']} indicando la obligación del establecimiento.

---

### Objetivo del plan para este año escolar

Formula 1 objetivo cuantificable y realista.
Guía: {obj_guidance}
El objetivo debe mencionar QUÉ, en CUÁNTO, y PARA CUÁNDO.
Ejemplo: "Reducir en un 12% el número de estudiantes que reportan victimización frecuente, \
pasando de X a Y estudiantes afectados, medido en la encuesta de seguimiento de [mes]."
NUNCA pongas un porcentaje mayor al 20% para el primer año.

---

### PILAR 1 — DESCUBRIR: Conocer la realidad del establecimiento

**Acción 1.1 — Formación del Equipo Zero Bullying**
Qué hacer: [Basado en manuales: composición del equipo, frecuencia de reunión, roles]
Por qué: [Justificación con datos de esta escuela]
Responsable: _______________________
Fecha de inicio: ___________________
Indicador de logro: Equipo conformado y primera reunión realizada antes de ___

**Acción 1.2 — Socialización de los resultados con la comunidad**
Qué hacer: [Cómo compartir estos resultados con docentes, estudiantes y padres]
Por qué: [Los datos muestran que {n} estudiantes fueron encuestados — la comunidad debe conocer los resultados]
Responsable: _______________________
Fecha de socialización: ___________________
Indicador de logro: 100% del cuerpo docente informado y acta de reunión firmada

---

### PILAR 2 — RESOLVER: Qué hace el establecimiento ante un caso

**Acción 2.1 — Protocolo de actuación ante un caso de {cc['bullying_term']}**
Qué hacer: [Basado en manuales: pasos concretos del protocolo, quién hace qué]
Por qué prioritaria: [Las áreas en crisis son {top3_str} — el establecimiento debe estar preparado]
Responsable: _______________________
Fecha de implementación: ___________________
Indicador de logro: Protocolo escrito, aprobado y conocido por 100% del personal

**Acción 2.2 — Trabajo post-incidente con víctima, agresor y testigos**
Qué hacer: [Basado en manuales: acompañamiento diferenciado para cada perfil]
Por qué: [El perfil agresor-víctima requiere atención especial según los datos]
Responsable: _______________________
Plazo de seguimiento por caso: ___________________
Indicador de logro: Registro de seguimiento completado para cada caso reportado

---

### PILAR 3 — PREVENIR: Construir el ambiente donde el bullying no prospera

**Acción 3.1 — Zonas de Seguridad en espacios de riesgo**
Qué hacer: Implementar supervisión adulta activa y visible en los espacios más críticos \
identificados por la encuesta: {', '.join(eco_top)}. \
[Basado en manuales: sistema de chaquetas/petos identificables del Programa ZERO, \
horarios de supervisión, criterios de intervención]
Por qué prioritaria: La encuesta identifica estos espacios como los de mayor frecuencia \
de agresiones. Sin supervisión visible, las agresiones continuarán en los mismos lugares.
Responsable zona 1 ({eco_top[0] if eco_top else 'por definir'}): _______________________
Responsable zona 2 ({eco_top[1] if len(eco_top) > 1 else 'por definir'}): _______________________
Horario de supervisión: ___________________
Indicador de logro: Supervisión activa operando en los 3 espacios prioritarios antes de ___

**Acción 3.2 — Fortalecimiento de relaciones en grados prioritarios**
Qué hacer: [Basado en manuales: actividades de cohesión, normas co-construidas, \
gestión del aula, dirigidas específicamente a {priority_grades_str}]
Por qué: Estos son los grados con mayor concentración de agresores y víctimas \
según los datos de la encuesta.
Docente responsable por grado:
  - {priority_grades[0] if priority_grades else 'Grado 1'}: _______________________
  - {priority_grades[1] if len(priority_grades) > 1 else 'Grado 2'}: _______________________
Frecuencia de actividades: ___________________
Indicador de logro: Al menos 1 actividad de cohesión por mes en cada grado prioritario

**Acción 3.3 — Involucramiento de familias**
Qué hacer: [Basado en manuales: qué decirles a los padres, cómo involucrarlos, \
mensajes consistentes entre escuela y hogar]
Por qué: Las familias son un factor protector clave. Sin su conocimiento y participación, \
las intervenciones escolares tienen la mitad del impacto.
Responsable: _______________________
Instancia de comunicación (reunión, circular, taller): ___________________
Indicador de logro: Al menos 1 instancia de información a familias por semestre

**Acción 3.4 — Involucramiento de estudiantes**
Qué hacer: [Basado en manuales: rol de los estudiantes como defensores activos, \
cómo formar defensores en los grados con más testigos]
Por qué: Los testigos son el grupo más numeroso. Cuando saben cómo actuar, \
el bullying se reduce significativamente.
Responsable: _______________________
Grados donde se implementa: ___________________
Indicador de logro: Al menos 5 estudiantes formados como defensores activos por grado

---

### PILAR 4 — SOSTENER: Cómo el plan sigue vivo

**Acción 4.1 — Reuniones periódicas del Equipo Zero**
Qué hacer: El Equipo Zero se reúne mensualmente durante la implementación del plan. \
Agenda fija: revisión de incidentes, avance de actividades, ajustes necesarios.
Responsable de convocar: _______________________
Día fijo de reunión: ___________________
Indicador de logro: Acta de reunión mensual archivada

**Acción 4.2 — Encuesta de seguimiento al final del año**
Qué hacer: Aplicar la encuesta TECH4ZERO nuevamente al final del año escolar \
para medir el avance respecto a este diagnóstico.
Por qué es obligatoria: Es la única forma de saber si el plan funcionó. \
Sin medición, no hay evidencia de mejora — ni para la escuela ni para las familias.
Responsable: _______________________
Fecha propuesta de aplicación: ___________________
Indicador de logro: Encuesta aplicada al 80% o más de los estudiantes encuestados este año

{manuals}

DATOS:
{data}
"""


def _prompt_s5(ctx: dict, data: str, cc: dict, manuals: str) -> str:
    """Section 5 — Common mistakes to avoid."""
    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.

## Sección 5: Errores comunes a evitar

INSTRUCCIONES (máximo 500 palabras):
- Presenta 6 errores comunes en formato consistente:
  **Error [N]: [Nombre del error]**
  Qué suele pasar: [descripción del error en 1-2 oraciones]
  Por qué no funciona: [explicación basada en los manuales, 1-2 oraciones]
  En cambio: [qué hacer en su lugar, 1-2 oraciones, específico]

- Los 6 errores OBLIGATORIOS (en este orden):
  1. Reunir al agresor y a la víctima para que "hablen" sin preparación previa
  2. Creer que hablar una sola vez del tema en clases es suficiente
  3. Responsabilizar únicamente al agresor sin atender al entorno
  4. Ignorar el rol de los testigos (el grupo más numeroso)
  5. No informar a las familias o informarlas demasiado tarde
  6. Implementar el plan solo durante el primer mes y luego abandonarlo

- Para cada error, si los datos de esta escuela son relevantes, menciónalo. \
  Ejemplo: "En {ctx.get('escuela','esta escuela')}, los datos muestran X — \
  lo que hace que este error sea especialmente crítico."
- Tono: constructivo, nunca acusatorio. El objetivo es orientar, no culpar.

{manuals}

DATOS (para contextualizar con la realidad de esta escuela):
{data}
"""


def _prompt_s6(ctx: dict, data: str, cc: dict, manuals: str) -> str:
    """Section 6 — Monitoring and re-survey recommendation."""
    n = ctx.get("n_estudiantes", "?")
    escuela = ctx.get("escuela", "la escuela")

    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {escuela}.

## Sección 6: Cómo saber si estamos avanzando

INSTRUCCIONES (máximo 400 palabras):

**6.1 Indicadores mensuales sin encuesta (1 párrafo + lista)**
- Introduce la idea de que el equipo puede hacer seguimiento mensual sin necesidad \
  de aplicar la encuesta completa.
- Lista de 5 indicadores observables simples que cualquier docente puede registrar:
  • Número de incidentes reportados al orientador/encargado de convivencia
  • Observación directa en los espacios de riesgo identificados por la encuesta \
    ({', '.join(e['lugar'] for e in ctx.get('ecologia_reporte',[])[:3]) or 'ver diagnóstico'})
  • Participación de estudiantes en actividades de cohesión
  • Número de familias que asistieron a instancias de comunicación
  • Registro de casos donde se activó el protocolo

**6.2 La encuesta de seguimiento — por qué es indispensable (2 párrafos)**
- Párrafo 1: Explica por qué los indicadores mensuales no son suficientes por sí solos. \
  La encuesta es la única forma de medir cambios reales en la percepción de los estudiantes, \
  que es lo que más importa.
- Párrafo 2 (CONVINCENTE — este es el argumento de venta de la re-encuesta): \
  "{escuela} ya invirtió en conocer su realidad. Aplicar la encuesta nuevamente al final \
  del año cierra el ciclo: permite saber qué funcionó, qué no, y cómo mejorar el plan \
  para el año siguiente. Sin esta medición, la inversión inicial queda incompleta. \
  Con {n} estudiantes encuestados este año, la comparación año a año tendrá un valor \
  científico real para la comunidad escolar."
- Menciona que el Programa ZERO documenta reducciones de 40-50% en escuelas que aplican \
  la encuesta de seguimiento y ajustan el plan.

**6.3 Propuesta de calendario de seguimiento**
- Tabla simple con 5 momentos del año escolar:
  | Momento | Actividad | Responsable sugerido |
  |---------|-----------|---------------------|
  | Inicio de año | Activación del Equipo Zero | Director(a) |
  | Mensual | Revisión de indicadores mensuales | Encargado(a) de convivencia |
  | Mitad de año | Revisión del plan y ajustes | Equipo Zero |
  | Fin de año | Aplicación de encuesta TECH4ZERO | Encargado(a) + TECH4ZERO |
  | Fin de año | Comparación con diagnóstico inicial y planificación siguiente año | Equipo Zero |

{manuals}

DATOS:
{data}
"""


def _prompt_s7(ctx: dict, cc: dict) -> str:
    """Section 7 — Resources and references."""
    return f"""Eres un especialista senior en convivencia escolar del Programa ZERO.
Idioma: {cc['idioma']}. Destinatario: {cc['director_title']} de {ctx.get('escuela','la escuela')}.

## Sección 7: Recursos y referencias

INSTRUCCIONES (máximo 300 palabras):

**7.1 Manuales del Programa ZERO utilizados en este plan**
Lista los 4 grupos de manuales utilizados:
- Manuales de Fenómeno del Acoso Escolar
- Manuales de Enfoque Integrado
- Manuales de Intervención
- Manuales de Prevención
- Guía de Plan de Acción ZERO

**7.2 Marco legal de referencia**
Cita textualmente y con detalle:
{cc['ley_cita']}

**7.3 Referencias científicas**
Lista estas referencias exactas:
- Roland, E. (2000). Bullying in school: Three national innovations in Norwegian schools \
  in 15 years. *Aggressive Behavior*, 26(1), 135-143.
- Roland, E., & Galloway, D. (2002). Classroom influences on bullying. \
  *Educational Research*, 44(3), 299-312.
- Gaete, J., et al. (2021). Validation of OBVQ-R in Chile. \
  *Frontiers in Psychology*, 12, 578661.
- Olweus, D. (1993). *Bullying at school: What we know and what we can do*. \
  Blackwell Publishers.

**7.4 Contacto y soporte TECH4ZERO**
- Plataforma: TECH4ZERO (sistema de análisis de clima escolar)
- Para consultas sobre este informe o la encuesta de seguimiento, \
  contactar al equipo TECH4ZERO a través del encargado escolar designado.

**Nota final (1 párrafo):**
Este plan fue elaborado con base en los datos reales de {ctx.get('escuela','la escuela')} \
y los manuales del Programa ZERO. Las recomendaciones están fundamentadas en evidencia \
científica y en la experiencia de implementación en escuelas de Noruega, Chile, México \
y Estados Unidos. El plan debe revisarse y actualizarse cada año escolar.
"""


# ══════════════════════════════════════════════════════════════
# SECTION GENERATOR
# ══════════════════════════════════════════════════════════════

def _generate_section(
    section: dict,
    ctx: dict,
    manual_texts: dict,
    client,
    model: str,
    max_tokens: int,
) -> str:
    """Generate a single section via Claude API."""
    cc      = _get_cc(ctx)
    data    = _data_block(ctx)
    manuals = _manual_block(section, manual_texts)
    num     = section["num"]

    if num == 1:
        prompt = _prompt_s1(ctx, data, cc)
    elif num == 2:
        prompt = _prompt_s2(ctx, data, cc)
    elif num == 3:
        prompt = _prompt_s3(ctx, data, cc)
    elif num == 4:
        prompt = _prompt_s4(ctx, data, cc, manuals)
    elif num == 5:
        prompt = _prompt_s5(ctx, data, cc, manuals)
    elif num == 6:
        prompt = _prompt_s6(ctx, data, cc, manuals)
    elif num == 7:
        prompt = _prompt_s7(ctx, cc)
    else:
        prompt = f"Escribe la sección {num}: {section['title']} en máximo 300 palabras."

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ══════════════════════════════════════════════════════════════
# DOCUMENT ASSEMBLERS
# ══════════════════════════════════════════════════════════════

def _build_doc1_header(ctx: dict) -> str:
    cc = _get_cc(ctx)
    return f"""# Informe de Diagnóstico — TECH4ZERO
## {ctx.get('escuela', 'Establecimiento Educacional')}

**{cc['director_title']}:** {ctx.get('escuela', '')}
**Fecha:** {ctx.get('fecha', datetime.now().strftime('%d/%m/%Y'))}
**Estudiantes analizados:** {ctx.get('n_estudiantes', 'N/A')}
**País:** {cc['pais']} — {cc['marco']}

---

*Este informe presenta los resultados de la Encuesta de Clima Escolar TECH4ZERO, \
basada en el Programa ZERO de la Universidad de Stavanger (Noruega). \
Los hallazgos descritos corresponden exclusivamente a los datos recopilados \
en este establecimiento y son estrictamente confidenciales.*

---

"""


def _build_doc2_header(ctx: dict) -> str:
    cc = _get_cc(ctx)
    return f"""# Plan de Acción — TECH4ZERO
## {ctx.get('escuela', 'Establecimiento Educacional')}

**{cc['director_title']}:** {ctx.get('escuela', '')}
**Fecha de elaboración:** {ctx.get('fecha', datetime.now().strftime('%d/%m/%Y'))}
**Basado en:** Diagnóstico TECH4ZERO — {ctx.get('n_estudiantes', 'N/A')} estudiantes encuestados
**Marco:** {cc['marco']}

---

*Este Plan de Acción está elaborado con base en los datos reales de este \
establecimiento. Las acciones marcadas con _______  requieren ser completadas \
por el equipo directivo con nombres, fechas y responsables específicos. \
El plan debe ser revisado y actualizado al inicio de cada año escolar.*

---

"""


def _build_footer(doc_num: int) -> str:
    if doc_num == 1:
        return """
---

*El Plan de Acción elaborado a partir de este diagnóstico se encuentra en el documento adjunto.*

---
*Documento generado por TECH4ZERO · Programa ZERO · Universidad de Stavanger*
"""
    else:
        return """
---

*"Este remedio está garantizado que sí funciona. Lo que resta ahora es saber tomárselo."*
— Profesor Erling Roland, Universidad de Stavanger

---
*Plan elaborado por TECH4ZERO · Programa ZERO · Universidad de Stavanger*
"""


# ══════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════════

def generate_diagnostic_report(
    ctx: dict,
    manual_texts: dict,
    client,
    model: str = "claude-haiku-4-5",
    max_tokens_per_section: int = 1200,
    progress_callback=None,
) -> tuple[list[str], str]:
    """
    Generate Document 1 — Informe de Diagnóstico (3 sections).

    Args:
        ctx:                     Survey context dict from app.py
        manual_texts:            Dict with manual text by category key
        client:                  Initialized anthropic.Anthropic client
        model:                   Claude model string
        max_tokens_per_section:  Token limit per section
        progress_callback:       Optional callable(section_num, title, text)

    Returns:
        (sections_list, full_document_string)
    """
    sections_text = []
    total = len(DIAGNOSTIC_SECTIONS)

    for i, section in enumerate(DIAGNOSTIC_SECTIONS):
        text = _generate_section(section, ctx, manual_texts, client, model, max_tokens_per_section)
        sections_text.append(text)
        if progress_callback:
            progress_callback(i + 1, total, section["title"], text)

    header   = _build_doc1_header(ctx)
    footer   = _build_footer(1)
    full_doc = header + "\n\n---\n\n".join(sections_text) + footer
    return sections_text, full_doc


def generate_action_plan(
    ctx: dict,
    manual_texts: dict,
    client,
    model: str = "claude-haiku-4-5",
    max_tokens_per_section: int = 1500,
    progress_callback=None,
) -> tuple[list[str], str]:
    """
    Generate Document 2 — Plan de Acción (4 sections).

    Args:
        ctx:                     Survey context dict from app.py
        manual_texts:            Dict with manual text by category key
        client:                  Initialized anthropic.Anthropic client
        model:                   Claude model string
        max_tokens_per_section:  Token limit per section (higher for action plan)
        progress_callback:       Optional callable(section_num, total, title, text)

    Returns:
        (sections_list, full_document_string)
    """
    sections_text = []
    total = len(ACTION_SECTIONS)

    for i, section in enumerate(ACTION_SECTIONS):
        text = _generate_section(section, ctx, manual_texts, client, model, max_tokens_per_section)
        sections_text.append(text)
        if progress_callback:
            progress_callback(i + 1, total, section["title"], text)

    header   = _build_doc2_header(ctx)
    footer   = _build_footer(2)
    full_doc = header + "\n\n---\n\n".join(sections_text) + footer
    return sections_text, full_doc


# ══════════════════════════════════════════════════════════════
# PDF GENERATOR (unchanged from v1)
# ══════════════════════════════════════════════════════════════

def markdown_to_pdf(markdown_text: str, school_name: str, doc_title: str = "Informe") -> bytes:
    """
    Convert markdown report to a styled PDF using reportlab.

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
        title=f"{doc_title} TECH4ZERO — {school_name}",
        author="TECH4ZERO",
    )

    styles = getSampleStyleSheet()

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
    style_fill = ParagraphStyle(
        'Fill', parent=styles['Normal'],
        fontSize=10.5, leading=15,
        textColor=colors.HexColor('#37474f'),
        leftIndent=18, spaceAfter=6,
        borderPad=4,
    )

    story = []

    def escape_xml(text: str) -> str:
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
            if story:
                story.append(PageBreak())
            add_paragraph(line[3:], style_h2)
        elif line.startswith('### '):
            add_paragraph(line[4:], style_h3)
        elif line.startswith('**') and line.endswith('**') and len(line) > 4:
            inner = line[2:-2]
            add_paragraph(f"<b>{escape_xml(inner)}</b>", style_body)
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = '• ' + line[2:].strip()
            add_paragraph(bullet_text, style_bullet)
        # Detect fill-in lines (lines with ___ blanks)
        elif '___' in line:
            processed = escape_xml(line)
            story.append(Paragraph(processed, style_fill))
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
            processed = _re.sub(
                r'\*\*(.+?)\*\*',
                lambda m: f'<b>{escape_xml(m.group(1))}</b>',
                escape_xml(line)
            )
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


# Keep backward-compatible alias so app.py import doesn't break
CHAPTERS = DIAGNOSTIC_SECTIONS + ACTION_SECTIONS


def generate_full_report(
    ctx: dict,
    manual_texts: dict,
    client,
    model: str = "claude-haiku-4-5",
    max_tokens_per_chapter: int = 1000,
    progress_callback=None,
) -> tuple[list[str], str]:
    """
    Backward-compatible wrapper — generates BOTH documents and
    returns them concatenated as a single string.
    Use generate_diagnostic_report() and generate_action_plan() directly
    for the two-document workflow.
    """
    total_sections = len(ALL_SECTIONS)
    all_texts = []
    counter = [0]

    def _cb(sec_num, total, title, text):
        counter[0] += 1
        if progress_callback:
            progress_callback(counter[0], title, text)

    d1_texts, d1_doc = generate_diagnostic_report(
        ctx, manual_texts, client, model, max_tokens_per_chapter, _cb
    )
    d2_texts, d2_doc = generate_action_plan(
        ctx, manual_texts, client, model, max_tokens_per_chapter, _cb
    )

    all_texts = d1_texts + d2_texts
    full_doc  = d1_doc + "\n\n" + d2_doc
    return all_texts, full_doc


if __name__ == '__main__':
    print("report_generator.py v2.0 loaded successfully")
    print(f"Diagnostic sections: {len(DIAGNOSTIC_SECTIONS)}")
    print(f"Action plan sections: {len(ACTION_SECTIONS)}")
    print(f"Total sections: {len(ALL_SECTIONS)}")
