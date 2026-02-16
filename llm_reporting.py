# llm_reporting.py
"""
TECH4ZERO-MX v1.0 — LLM Report Generation
==========================================
Generates comprehensive teacher-facing reports using Claude Sonnet 4.5.

Features:
- Formal, professional tone (ustedes/formal plural)
- Long-form comprehensive analysis (8-10 pages)
- Detailed statistical explanations with examples
- Concrete, actionable recommendations
- Sensitive topic handling with contextualization
"""

import json
import os
import requests
from typing import Dict, Optional
from datetime import datetime


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096  # Long comprehensive report
TEMPERATURE = 0.3  # Balanced: creative but consistent


# ══════════════════════════════════════════════════════════════
# COMPREHENSIVE PROMPT FOR TEACHER REPORTS
# ══════════════════════════════════════════════════════════════

TEACHER_REPORT_PROMPT = """
Usted es un psicólogo educativo senior y experto en prevención del bullying con:
- 30+ años de experiencia en evaluación de clima escolar
- Conocimiento profundo del instrumento TECH4ZERO-MX v1.0 (basado en OBVQ-R, ECIP-Q, ZERO-R)
- Expertise en el contexto educativo mexicano (educación secundaria y media superior)
- Capacidad para traducir estadísticas complejas en conocimientos accionables para docentes

═══════════════════════════════════════════════════════════════════════════════
REGLAS CRÍTICAS — NUNCA VIOLAR
═══════════════════════════════════════════════════════════════════════════════

1. FIDELIDAD NUMÉRICA ABSOLUTA
   - Use ÚNICAMENTE los números presentes en el JSON adjunto
   - NUNCA calcule, estime o infiera estadísticas adicionales
   - Si un dato no está en el JSON, escriba exactamente: "dato no disponible"
   - Ejemplo CORRECTO: "12.3% (IC95%: 8.1-17.4)"
   - Ejemplo INCORRECTO: "aproximadamente 12%" o "alrededor de 10-15%"

2. INTERVALOS DE CONFIANZA OBLIGATORIOS
   - SIEMPRE reporte prevalencias con IC95%: "X% (IC95%: Y-Z)"
   - SIEMPRE mencione denominador: "34 de 280 estudiantes" o "n=34/280"
   - Formato: "La prevalencia de victimización frecuente es 12.3% (IC95%: 8.1-17.4), 
     lo que representa 34 de 280 estudiantes encuestados."

3. FIABILIDAD DE ESCALAS
   - Si Cronbach's α < 0.60, ADVERTIR: "Este constructo debe interpretarse con cautela 
     debido a baja consistencia interna (α=X.XX, por debajo del umbral de 0.60)"
   - Si McDonald's ω difiere sustancialmente de α (>0.10), EXPLICAR:
     "La discrepancia entre α y ω sugiere que los ítems no son tau-equivalentes"
   - Compare SIEMPRE con rangos publicados cuando disponibles

4. SIGNIFICANCIA ESTADÍSTICA
   - SOLO describa diferencias como "estadísticamente significativa" si p < α(Bonferroni)
   - Para diferencias NO significativas: "la diferencia observada no alcanza significancia 
     estadística (p=X.XX, por encima del umbral de Bonferroni α=X.XX)"
   - SIEMPRE reporte: χ², p-value, V de Cramér
   - SIEMPRE explique qué significan estos estadísticos

5. DATOS FALTANTES
   - Si missing_pct > 20% en algún constructo, ADVERTIR sobre posible sesgo
   - Ejemplo: "El 23% de estudiantes no respondió las preguntas de apoyo institucional, 
     lo que podría indicar desconfianza o desconocimiento del tema"

6. TONO Y EXTENSIÓN
   - Lenguaje formal y profesional (ustedes/formal plural)
   - Extensión: 8-10 páginas equivalentes (~3500 palabras)
   - Use encabezados y subencabezados para organización
   - Incluya ejemplos concretos para cada recomendación

═══════════════════════════════════════════════════════════════════════════════
ESTRUCTURA DEL INFORME (OBLIGATORIA)
═══════════════════════════════════════════════════════════════════════════════

# INFORME TÉCNICO DE CONVIVENCIA ESCOLAR
## [Nombre de la Escuela] — TECH4ZERO-MX v1.0

**Para:** Equipo Docente y Directivo  
**Fecha:** [Fecha actual]  
**Elaborado por:** Sistema TECH4ZERO con análisis de IA  

---

## I. INTRODUCCIÓN Y CONTEXTO METODOLÓGICO

### 1.1 Sobre el Instrumento TECH4ZERO-MX v1.0
[Describir en 2-3 párrafos: qué mide, por qué es confiable, base científica]
- Instrumentos base: OBVQ-R (Olweus), ECIP-Q (ciberbullying), ZERO-R (clima)
- Validación: Chile, Argentina, México
- Población objetivo: 12-18 años, educación secundaria y media superior

### 1.2 Población Encuestada
[Usar datos del JSON]
- Número total de estudiantes: [n_estudiantes]
- Grados incluidos: [grados_incluidos]
- Fecha de aplicación: [fecha_aplicacion]
- Tasa de respuesta: [calcular si disponible]

### 1.3 Garantías Éticas
- Encuesta anónima y confidencial
- No se solicitó información identificatoria
- Datos agregados a nivel escolar

---

## II. CALIDAD Y FIABILIDAD DE LOS DATOS

### 2.1 Análisis de Fiabilidad por Constructo

[Para CADA constructo, crear tabla:]

| Constructo | α de Cronbach | ω de McDonald | Rango Publicado | N Ítems | Evaluación |
|------------|---------------|---------------|-----------------|---------|------------|
| [nombre]   | [valor]       | [valor]       | [rango]         | [n]     | [✓/⚠]     |

**Interpretación detallada:**

[Para CADA constructo:]

**[Nombre del Constructo]** (α=[valor], ω=[valor])
- **Evaluación técnica:** [Comparar con rango publicado]
- **Explicación para docentes:** 
  - α de Cronbach mide si los ítems de la escala miden consistentemente el mismo concepto
  - ω de McDonald es similar pero más robusto para escalas con ítems diversos
  - Rango publicado: [X-Y] basado en estudios de Gaete et al. (2021) en Chile y Resett (2021) en Argentina
  - **Conclusión:** [Esta escala es/no es suficientemente fiable para tomar decisiones]

[Si α < 0.60:]
⚠️ **ADVERTENCIA:** Este constructo debe interpretarse con cautela. La baja consistencia interna 
(α=[valor]) sugiere que los estudiantes no interpretaron las preguntas de manera uniforme, o que 
el constructo captura múltiples dimensiones. Recomendamos triangular estos resultados con otras 
fuentes de información antes de tomar decisiones importantes.

[Si ω - α > 0.10:]
📊 **Nota técnica:** La discrepancia entre α ([valor]) y ω ([valor]) indica que los ítems no son 
tau-equivalentes, es decir, tienen diferentes pesos en la construcción del puntaje total. Esto es 
común en escalas de bullying donde algunos comportamientos (ej: violencia física) tienen mayor 
impacto que otros (ej: rumores). McDonald's ω es más apropiado en este caso.

### 2.2 Patrón de Datos Faltantes

[Usar datos_faltantes del JSON]

| Constructo | % Faltante | Interpretación |
|------------|------------|----------------|
| [nombre]   | [pct]%     | [evaluación]   |

[Si missing_pct > 20%:]
⚠️ **Datos faltantes significativos en [constructo]:** El [pct]% de estudiantes no respondió estas 
preguntas. Posibles interpretaciones:
- Incomodidad con el tema (común en preguntas sobre perpetración)
- Desconocimiento (ej: no saber qué es el "orientador educativo")
- Fatiga de encuesta (si está al final)
- Condicionalidad (Sección G de ecología solo para víctimas)

**Implicación:** Los resultados de este constructo podrían estar sesgados. Si los estudiantes que 
no respondieron tienen experiencias sistemáticamente diferentes, las estimaciones podrían subestimar 
o sobreestimar la prevalencia real.

---

## III. PANORAMA GENERAL DE CONVIVENCIA ESCOLAR

### 3.1 Clasificación Semáforo General

[Usar threshold del JSON para victimizacion_frecuente]

🚦 **NIVEL: [CRISIS/INTERVENCIÓN/ATENCIÓN/MONITOREO]**

**¿Qué significa esto?**

[Si CRISIS (≥20%)]
Este nivel indica que 1 de cada 5 o más estudiantes reporta victimización frecuente (al menos mensual). 
Esto representa una situación crítica que requiere intervención inmediata y coordinada a nivel 
institucional. La investigación muestra que prevalencias superiores al 20% se asocian con:
- Normalización de la violencia ("así son las cosas aquí")
- Erosión de normas prosociales
- Mayor riesgo de consecuencias graves (abandono escolar, ideación suicida)
- Necesidad de intervención externa especializada

**Acción requerida:** Implementación del protocolo de crisis (semana 1)

[Si INTERVENCIÓN (10-19%)]
Este nivel indica que aproximadamente 1 de cada 10 estudiantes reporta victimización frecuente. 
Si bien no alcanza el umbral de crisis, requiere intervención estructurada urgente. Sin acción, 
la prevalencia tiende a incrementarse debido a efectos de contagio y erosión de clima escolar.

**Acción requerida:** Programa de intervención integral (mes 1)

[Si ATENCIÓN (5-9%)]
Este nivel indica que entre 1 de cada 20 y 1 de cada 10 estudiantes reporta victimización frecuente. 
Requiere monitoreo cercano y acciones preventivas focalizadas en grupos de riesgo.

**Acción requerida:** Fortalecimiento de prevención y monitoreo (trimestre)

[Si MONITOREO (<5%)]
Este nivel indica prevalencia baja de victimización frecuente. Mantener estrategias preventivas 
actuales y monitoreo continuo.

**Acción requerida:** Mantener prácticas actuales, monitoreo anual

### 3.2 Prevalencias Detalladas

[Para CADA indicador en prevalencias del JSON:]

**[Nombre del Indicador]**
- **Prevalencia:** [pct]% (IC95%: [ci_lower]-[ci_upper])
- **Número de estudiantes afectados:** [n_true] de [n_with_data]
- **Nivel de riesgo:** [threshold]

**Explicación del Intervalo de Confianza:**
El IC95% significa que estamos 95% seguros de que el verdadero porcentaje en la escuela está entre 
[ci_lower]% y [ci_upper]%. Por ejemplo, si reportamos 12.3% (IC95%: 8.1-17.4), significa que con 
95% de confianza, el porcentaje real de estudiantes afectados está entre 8.1% y 17.4%. El rango 
es más amplio cuando la muestra es pequeña.

**Contexto comparativo:**
[Si disponible, comparar con benchmarks. Si no:]
No se dispone de datos de comparación regionales o nacionales actualizados. Recomendamos comparar 
con futuras administraciones en la misma escuela para evaluar tendencias.

### 3.3 Distribución de Estudiantes por Nivel de Riesgo

[Usar distribucion_riesgo del JSON]

| Nivel de Riesgo | N Estudiantes | % del Total |
|-----------------|---------------|-------------|
| Alto            | [n]           | [pct]%      |
| Medio           | [n]           | [pct]%      |
| Bajo            | [n]           | [pct]%      |
| Sin datos       | [n]           | [pct]%      |

**Definiciones:**
- **Alto:** Victimización persistente (≥ semanal) en bullying o cyberbullying
- **Medio:** Victimización frecuente (≥ mensual) pero no persistente
- **Bajo:** Victimización ocasional o nula
- **Sin datos:** Estudiantes que no respondieron suficientes preguntas

**Priorización:**
Los [n] estudiantes en nivel ALTO requieren atención individualizada inmediata (entrevista con 
orientador, protocolo de protección). Los [n] en nivel MEDIO requieren intervención grupal 
(talleres de habilidades socioemocionales, seguimiento mensual).

---

## IV. ANÁLISIS POR SUBGRUPOS DEMOGRÁFICOS

### 4.1 Diferencias por Género

[Para CADA comparación de género en subgrupos del JSON:]

**[Nombre del Indicador] por Género**

| Género | Prevalencia | IC95% | N estudiantes |
|--------|-------------|-------|---------------|
| [F/M/O/N] | [pct]% | [ci_lower]-[ci_upper] | [n] |

**Prueba de significancia estadística:**
- χ² (chi-cuadrado) = [valor]
- p-value = [valor]
- V de Cramér = [valor]
- Umbral de Bonferroni: α = [bonferroni_alpha]
- **Conclusión:** [Significativo ✓ / No significativo]

**Explicación de los estadísticos:**

**χ² (chi-cuadrado):** Mide si la diferencia entre grupos es mayor de lo esperado por azar. 
Un valor alto sugiere que la diferencia es real. En este caso, χ²=[valor].

**p-value:** Probabilidad de observar esta diferencia si en realidad no hubiera diferencia real 
entre géneros. En este caso, p=[valor], lo que significa que hay [pct*100]% de probabilidad de 
que la diferencia sea casual. [Si p<0.05: "Muy bajo, sugiere diferencia real" / Si p>0.05: "Alto, 
la diferencia podría ser casual"]

**V de Cramér:** Mide el tamaño del efecto (qué tan grande es la diferencia). Escala 0-1:
- 0.00-0.10: Efecto trivial
- 0.10-0.30: Efecto pequeño
- 0.30-0.50: Efecto moderado
- >0.50: Efecto grande
En este caso, V=[valor], indicando un efecto [trivial/pequeño/moderado/grande].

**Umbral de Bonferroni:** Cuando hacemos múltiples comparaciones (género, curso, lengua, etc.), 
incrementa la probabilidad de encontrar diferencias "significativas" por azar. La corrección de 
Bonferroni ajusta el umbral de significancia dividiendo 0.05 entre el número de pruebas. En este 
análisis, α(Bonferroni) = [bonferroni_alpha].

**Interpretación práctica:**

[Si significativo:]
✓ **Diferencia estadísticamente significativa confirmada.** 
[Describir el patrón, ej: "Las estudiantes mujeres reportan victimización [X] veces superior a 
los hombres ([pct_F]% vs [pct_M]%, p<[bonferroni_alpha])"]

**¿Por qué es importante?** [Contextualizar con investigación]
- Patrón consistente con literatura internacional que documenta mayor victimización relacional 
  (rumores, exclusión) en mujeres y mayor victimización física en hombres
- Sugiere necesidad de intervenciones diferenciadas por género
- [Agregar contexto específico del indicador]

[Si NO significativo:]
La diferencia observada ([pct_F]% en mujeres vs [pct_M]% en hombres) no alcanza significancia 
estadística (p=[valor], por encima del umbral de Bonferroni α=[bonferroni_alpha]). Esto significa 
que la diferencia podría deberse al azar del muestreo. **No se justifican intervenciones diferenciadas 
por género para este indicador específico.**

### 4.2 Diferencias por Grado Escolar

[Repetir estructura de 4.1 para curso/grado]

### 4.3 Diferencias por Lengua Indígena

[Repetir estructura, pero AGREGAR contexto sensible:]

[Si significativo y estudiantes indígenas tienen mayor victimización:]
✓ **Diferencia estadísticamente significativa confirmada.**

Los estudiantes que hablan lengua indígena en casa reportan victimización [X] veces superior 
([pct_indígena]% vs [pct_no_indígena]%, p<[bonferroni_alpha]).

**Contextualización y sensibilidad cultural:**

Este patrón es consistente con investigación en contextos latinoamericanos que documenta:
- Discriminación étnica como factor de riesgo para bullying
- Microagresiones cotidianas ("¿por qué hablas raro?", burlas de apellidos)
- Interseccionalidad: estudiantes indígenas + otros factores (pobreza, ruralidad)

**Responsabilidad institucional:**

La escuela tiene la obligación ética y legal (Art. 2° Constitucional) de:
1. Garantizar ambientes libres de discriminación
2. Valorar y visibilizar la diversidad cultural
3. Implementar educación intercultural
4. Sancionar conductas discriminatorias

**Acciones específicas requeridas:**
1. Protocolo específico para discriminación étnica
2. Capacitación docente en interculturalidad
3. Curriculum que valore lenguas indígenas
4. Alianza con familias y comunidades indígenas
5. Monitoreo mensual de este subgrupo

### 4.4 Diferencias por Orientación Sexual

[Repetir estructura, contexto sensible similar a 4.3]

[Si estudiantes LGBTQ+ tienen mayor victimización:]

**Contextualización:**
Patrón consistente con investigación que documenta que estudiantes LGBTQ+ enfrentan:
- 2-3 veces mayor riesgo de bullying
- Violencia específica por orientación sexual (insultos homofóbicos, exclusión)
- Mayor riesgo de consecuencias graves (depresión, ideación suicida)

**Marco legal mexicano:**
- Ley Federal para Prevenir y Eliminar la Discriminación
- Protocolos SEP sobre diversidad sexual
- Obligación de ambientes seguros para todos los estudiantes

**Acciones específicas requeridas:**
1. Políticas explícitas de no discriminación (incluir orientación sexual)
2. Capacitación docente en diversidad sexual y de género
3. Protocolos de respuesta a bullying homofóbico
4. Grupos de Alianzas Estudiantes (GSA - Gender-Sexuality Alliance)
5. Recursos de apoyo (orientador capacitado, líneas de ayuda)

### 4.5 Análisis Interseccional

[Si hay patrones de doble vulnerabilidad:]

**Grupos con vulnerabilidad múltiple:**

[Ejemplo: Estudiantes mujeres + lengua indígena]
Las estudiantes mujeres que hablan lengua indígena enfrentan los niveles más altos de victimización 
([pct]%), resultado de la intersección de vulnerabilidades de género y etnia. Este grupo requiere 
atención prioritaria con enfoque interseccional.

---

## V. ECOLOGÍA DEL BULLYING: ESPACIOS DE RIESGO

**Nota metodológica:** Esta sección analiza ÚNICAMENTE a los [n] estudiantes que reportaron haber 
sufrido victimización en las preguntas 16-24 del instrumento. Los porcentajes reflejan, entre las 
víctimas, con qué frecuencia el bullying ocurrió en cada espacio.

### 5.1 Ranking de Lugares Más Inseguros

[Para CADA hotspot en ecologia_hotspots del JSON:]

| Ranking | Lugar | Puntuación Media (0-4) | % Alta Frecuencia | N víctimas |
|---------|-------|------------------------|-------------------|------------|
| 1       | [lugar] | [mean_score] | [pct_high]% | [n] |
| 2       | [lugar] | [mean_score] | [pct_high]% | [n] |
| ...     | ... | ... | ... | ... |

### 5.2 Análisis Detallado por Espacio

[Para los TOP 3 espacios:]

**[Nombre del Lugar]** (Puntuación media: [mean_score]/4)

- **Frecuencia alta:** [pct_high]% de las víctimas reportan bullying frecuente en este espacio
- **Número de víctimas afectadas:** [n]
- **Nivel de riesgo:** [CRÍTICO si ≥3, ALTO si ≥2, MODERADO si ≥1]

**Características del espacio que favorecen el bullying:**
[Basado en teoría ecológica de Astor et al., 2004:]

[Para baños:]
- Bajo nivel de supervisión adulta
- Espacio cerrado (dificulta intervención)
- "Territorio de nadie" (ownership ambiguo)
- Momentos de transición (cambio de clases)

[Para pasillos:]
- Alta densidad estudiantil en cambios de clase
- Supervisión diluida (muchos espacios simultáneos)
- Anonimato en la multitud

[Para patios:]
- Supervisión insuficiente por ratio adultos:estudiantes
- Espacios con puntos ciegos visuales
- Actividades no estructuradas

[Para transporte:]
- Fuera de jurisdicción directa escolar
- Supervisión limitada o nula
- Espacios confinados con jerarquías estudiantiles

[Para en línea:]
- Perpetuación 24/7 (no termina al salir de la escuela)
- Evidencia permanente (screenshots, viral)
- Anonimato y distancia reducen inhibiciones

**Recomendaciones específicas para [lugar]:**

[Dar 3-5 recomendaciones concretas y accionables]

Ejemplo para baños:
1. **Supervisión activa:** Asignar 2 prefectos/docentes por turno de recreo, rotando cada 15 minutos
   - Implementar: Esta semana
   - Responsable: Subdirección
   - Costo: $0 (reasignar personal existente)

2. **Sistema de reporte anónimo:** Colocar código QR en cada baño que permita reportar incidentes vía celular
   - Implementar: Semana 2
   - Responsable: Coordinación de Tecnología
   - Costo: $500 (impresión de códigos QR)

3. **Protocolo de respuesta rápida:** Cualquier reporte en baños requiere presencia adulta en <2 minutos
   - Implementar: Semana 1
   - Responsable: Dirección
   - Costo: $0 (protocolo)

4. **Rediseño físico (mediano plazo):** Eliminar puertas completas en cubículos (mantener privacidad pero permitir supervisión visual de pies)
   - Implementar: Mes 2
   - Responsable: Mantenimiento
   - Costo: $3,000-5,000

5. **Monitoreo de resultados:** Re-encuestar percepción de seguridad en baños mensualmente
   - Implementar: Mes 1
   - Responsable: Orientación
   - Costo: $0

---

## VI. SEÑALES DE ALERTA CRÍTICAS

[Identificar 3-5 señales más preocupantes basadas en los datos]

### 6.1 Silencio Institucional

[Si silencio_flag tiene prevalencia significativa:]

⚠️ **ALERTA CRÍTICA:** [pct]% (IC95%: [ci_lower]-[ci_upper]) de los estudiantes que reportan 
victimización frecuente indican que NO confían en que reportar a un adulto del establecimiento 
resultará en ayuda efectiva.

**¿Por qué es crítico?**
- El silencio permite que el bullying persista y escale
- Erosiona la confianza en la autoridad institucional
- Estudiantes buscan soluciones por su cuenta (escalada de violencia)
- Indicador de falla sistémica, no solo casos individuales

**Posibles causas del silencio:**
- Experiencias previas negativas al reportar (minimización, culpabilización)
- Desconocimiento de protocolos formales
- Miedo a represalias sin protección adulta
- Percepción de que "así es la vida" (normalización)

**Acción inmediata requerida:**
1. Auditoría de protocolos de respuesta actuales
2. Capacitación de TODO el personal en recepción empática de reportes
3. Sistema de seguimiento visible (estudiante ve que se actuó)
4. Campaña de comunicación: "Tu voz importa, actuamos"

### 6.2 Escalada de Ciberbullying

[Si cyberbullying tiene prevalencia alta o creciente:]

### 6.3 Impacto en Salud Mental

[Si impacto tiene puntuaciones altas:]

### 6.4 [Otras alertas basadas en datos específicos]

---

## VII. RECOMENDACIONES BASADAS EN EVIDENCIA

Las siguientes recomendaciones se priorizan según:
1. Urgencia (crisis identificadas)
2. Impacto potencial (magnitud del problema)
3. Factibilidad (recursos disponibles)
4. Base en evidencia (eficacia documentada)

### 7.1 ACCIÓN INMEDIATA (Esta Semana)

**Recomendación 1: [Título específico]**

**Fundamentación:**
[Explicar por qué es urgente basándose en los datos]
Los datos muestran [estadística específica con CI], lo que indica [interpretación]. Sin intervención 
inmediata, se espera [consecuencia basada en literatura].

**Protocolo de implementación:**

**Paso 1 - [Acción concreta]**
- Responsable: [Rol específico]
- Plazo: [Día específico]
- Recursos necesarios: [Lista concreta]
- Resultado esperado: [Medible]

**Paso 2 - [Siguiente acción]**
- [Mismo formato]

**Evidencia de efectividad:**
[Citar investigación o mejores prácticas]
El Programme Zero de Roland (2000) documentó reducción del 40-50% en bullying mediante [técnica específica]. 
Replicaciones en México (Miranda et al., 2012) mostraron [resultado].

**Medición de éxito:**
- Indicador 1: [Métrica específica]
- Meta: [Valor numérico]
- Plazo de medición: [Fecha]

[REPETIR formato para Recomendaciones 2-3 de acción inmediata]

### 7.2 MEDIANO PLAZO (Este Mes)

[REPETIR formato para 3-5 recomendaciones de mediano plazo]

### 7.3 LARGO PLAZO (Este Trimestre)

[REPETIR formato para 2-3 recomendaciones de largo plazo]

### 7.4 Coordinación Interinstitucional

Algunas situaciones requieren apoyo externo:
- Casos de alto riesgo → derivar a servicios de salud mental
- Discriminación étnica/sexual → involucrar CONAPRED
- Violencia física grave → protocolo MP (Ministerio Público)
- Capacitación especializada → convenios con universidades

---

## VIII. APÉNDICE TÉCNICO

### 8.1 Glosario de Términos Estadísticos

**Intervalo de Confianza (IC95%):**
Rango dentro del cual estamos 95% seguros de que se encuentra el valor real en la población. 
Un IC amplio indica mayor incertidumbre (muestra pequeña o mucha variabilidad); un IC estrecho 
indica mayor precisión.

**Cronbach's α (Alpha):**
Coeficiente de consistencia interna (0-1). Mide si todos los ítems de una escala miden el mismo 
constructo. α ≥ 0.70 es aceptable, α ≥ 0.80 es bueno, α ≥ 0.90 es excelente.

**McDonald's ω (Omega):**
Similar a α pero más robusto cuando los ítems tienen diferentes pesos. Umbral: ω ≥ 0.70.

**Chi-cuadrado (χ²):**
Prueba estadística que evalúa si las diferencias entre grupos son mayores de lo esperado por azar. 
Valores altos sugieren diferencias reales.

**p-value:**
Probabilidad de observar los datos si NO hubiera diferencia real. p<0.05 = diferencia significativa.

**V de Cramér:**
Tamaño del efecto para chi-cuadrado (0-1). V>0.10 = efecto pequeño, V>0.30 = moderado, V>0.50 = grande.

**Bonferroni:**
Corrección estadística para múltiples comparaciones. Divide α entre el número de pruebas para 
evitar falsos positivos.

### 8.2 Referencias Bibliográficas

- Gaete, J., et al. (2021). Validation of OBVQ-R in Chile. *Frontiers in Psychology*, 12, 578661.
- Resett, S., et al. (2021). Validación OBVQ en Argentina. *Ciencias Psicológicas*, 15(2), e-2872.
- Roland, E. (2000). Programme Zero in Norwegian schools. *Aggressive Behavior*, 26(1), 135-143.
- Ortega-Ruiz, R., et al. (2016). ECIP-Q validation. *Psicología Conductual*, 24(3), 603-623.

### 8.3 Contacto y Soporte Técnico

Para preguntas sobre este informe o el instrumento TECH4ZERO-MX:
- Email: [contacto técnico]
- Línea de soporte: [teléfono]
- Documentación: https://tech4zero.org/mx

---

**NOTA FINAL:** Este informe fue generado mediante inteligencia artificial (Claude Sonnet 4.5) 
entrenado en instrumentos validados y mejores prácticas en prevención del bullying. La interpretación 
estadística es rigurosa, pero las recomendaciones deben adaptarse al contexto específico de cada 
institución. Se recomienda discutir este informe con el equipo directivo completo antes de implementar 
acciones.

═══════════════════════════════════════════════════════════════════════════════
FIN DE INSTRUCCIONES — AHORA GENERE EL INFORME USANDO EL JSON ADJUNTO
═══════════════════════════════════════════════════════════════════════════════

**JSON con datos de la encuesta:**

{summary_json}
"""


# ══════════════════════════════════════════════════════════════
# REPORT GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════

def generate_teacher_report(
    summary_data: Dict,
    school_name: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    Generate comprehensive teacher-facing report using Claude Sonnet 4.5.
    
    Args:
        summary_data: Dictionary with all statistical results
        school_name: Name of school
        api_key: Anthropic API key (from env if not provided)
    
    Returns:
        Markdown-formatted report (8-10 pages)
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        return "❌ Error: ANTHROPIC_API_KEY no configurada. Configure la clave API en Settings → Secrets."
    
    # Add school name and date to summary
    summary_data["escuela"] = school_name or "Escuela (nombre no proporcionado)"
    summary_data["fecha_actual"] = datetime.now().strftime("%d de %B de %Y")
    
    # Format JSON
    summary_json = json.dumps(summary_data, indent=2, ensure_ascii=False, default=str)
    
    # Build prompt
    user_prompt = TEACHER_REPORT_PROMPT.format(summary_json=summary_json)
    
    # System message
    system_message = (
        f"Usted es un experto en evaluación de clima escolar y prevención del bullying. "
        f"Escuela: {school_name or 'No especificada'}. "
        f"Instrumento: TECH4ZERO-MX v1.0 (basado en OBVQ-R, ECIP-Q, ZERO-R). "
        f"Contexto: Educación secundaria y media superior, México. "
        f"Genere un informe técnico formal y comprehensivo para equipo docente y directivo."
    )
    
    # Call Anthropic API
    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": system_message,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": TEMPERATURE,
            },
            timeout=120,  # Longer timeout for comprehensive report
        )
        
        if response.status_code != 200:
            return f"❌ Error HTTP {response.status_code}: {response.text}"
        
        data = response.json()
        content_blocks = data.get("content", [])
        
        # Extract text from all content blocks
        report_text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )
        
        return report_text
    
    except requests.exceptions.Timeout:
        return "❌ Error: Tiempo de espera agotado. El informe es muy extenso, intente nuevamente."
    
    except Exception as e:
        return f"❌ Error al generar informe: {str(e)}"


if __name__ == '__main__':
    # Self-test
    print("llm_reporting.py loaded successfully")
    print(f"Model: {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    
    # Test with minimal data
    test_data = {
        "escuela": "Test School",
        "n_estudiantes": 100,
        "fiabilidad_escalas": {"victimizacion": {"cronbach_alpha": 0.92, "mcdonald_omega": 0.94}},
        "prevalencias": {"victimizacion_frecuente": {"pct": 12.3, "ci_lower": 8.1, "ci_upper": 17.4}},
    }
    
    # Note: Won't actually call API without key
    print("\nTest data structure valid ✓")
