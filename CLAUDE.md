# SalarioPA — Contexto completo del proyecto
> Leer este archivo antes de cualquier acción. Contiene todo el historial de decisiones.

---

## ¿Qué es SalarioPA?

Estimador de salarios para el mercado laboral de Panamá (USD/mes bruto).
Proyecto ML end-to-end: datos reales → modelo entrenado → API → app web.

**Estado actual: Fase 1 en curso** — reemplazar dataset sintético con datos reales
de Computrabajo Panamá y Michael Page (Estudio de Remuneración 2024).

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Datos / ML | Python · pandas · scikit-learn · joblib |
| Modelo | GradientBoostingRegressor |
| Backend | FastAPI · uvicorn |
| Frontend | React + Vite (HTML prototipo ya funciona) |
| Deploy | Render (API) + Vercel (frontend) |

---

## Estructura de archivos del proyecto

```
salariopa/
├── CLAUDE.md                        ← este archivo
├── data/
│   ├── salarios_referencia_panama.csv  ← datos reales (54 cargos, Fase 1)
│   └── salarios_panama.csv             ← dataset sintético anterior (referencia)
├── model/
│   ├── generar_dataset.py           ← genera dataset + entrena modelo
│   ├── modelo_salario_pa.pkl        ← modelo entrenado (output)
│   └── modelo_metadata.json         ← metadata del modelo (output)
├── backend/
│   ├── main.py                      ← API FastAPI
│   └── requirements.txt
└── frontend/
    └── SalarioPA.html               ← prototipo HTML funcional (modo demo)
```

---

## Features del modelo (variables de entrada)

```python
# Categóricas ordinales (tienen orden lógico)
nivel_estudio:  ["Técnico / Universitario incompleto", "Licenciatura",
                 "Postgrado / Especialización", "Maestría", "Doctorado"]
nivel_ingles:   ["Ninguno", "Básico", "Intermedio", "Avanzado", "Nativo/Bilingüe"]

# Categóricas nominales
area:        ["Tecnología / IT", "Finanzas / Banca", "Ingeniería", "Salud",
              "Recursos Humanos", "Marketing / Ventas", "Administración",
              "Logística / Operaciones", "Legal", "Educación"]
sector:      ["Banca / Financiero", "Zona Libre de Colón", "Multinacional",
              "Tecnología / Startup", "Salud / Farmacéutica", "Gobierno / Público",
              "Retail / Comercio", "Construcción", "ONG / Sin fines de lucro",
              "Educación / Academia"]
tipo_empresa: ["Multinacional", "Empresa local grande", "PYME", "Startup",
               "Gobierno", "Freelance / Independiente"]
modalidad:   ["Presencial", "Híbrido", "Remoto"]
provincia:   ["Panamá", "Colón", "Chiriquí", "Herrera", "Los Santos",
              "Veraguas", "Coclé", "Panamá Oeste"]

# Numéricas
años_experiencia: int (0–35)
tiene_python:     0 o 1
tiene_powerbi:    0 o 1
tiene_sql:        0 o 1
tiene_sap:        0 o 1
tiene_excel_av:   0 o 1

# Target (lo que predice el modelo)
salario_usd: float (mensual bruto en USD)
```

---

## Piso salarial legal — Decreto N°13 Mitradel (vigente 2026)

Usar como restricción: `salario = max(piso_legal[area], salario_calculado)`

```python
PISO_MITRADEL = {
    "Tecnología / IT":         789,   # 3.29/h × 240h — Información y comunicaciones
    "Finanzas / Banca":        754,   # 3.14/h × 240h — Actividades financieras
    "Ingeniería":              842,   # 3.51/h × 240h — Construcción
    "Salud":                   739,   # 3.08/h × 240h — Salud
    "Administración":          725,   # 3.02/h × 240h — Comercio gran empresa
    "Recursos Humanos":        725,   # igual a Administración
    "Educación":               725,
    "Logística / Operaciones": 754,   # Transporte
    "Marketing / Ventas":      725,
    "Legal":                   725,
    "default":                 637,   # Promedio nacional Mitradel
}
```

---

## Datos reales disponibles — `salarios_referencia_panama.csv`

54 cargos con: cargo, area, seniority, sal_min, sal_max, sal_promedio, fuente, es_imputado

**Fuentes:**
- **Computrabajo Panamá** — medianas reales reportadas (0 = dato real)
- **Michael Page** — Estudio de Remuneración 2024, Panamá (0 = dato real)
- Filas con `es_imputado = 1` son estimaciones razonables, no datos directos

**Distribución por área:**
- Finanzas / Banca: 20 cargos
- Tecnología / IT: 21 cargos
- Recursos Humanos: 13 cargos

**Multiplicadores de seniority que surgen de los datos:**
- junior:    ~0.55× del promedio del área
- mid:       ~1.00× (base)
- senior:    ~1.80×
- gerencial: ~3.50×
- ejecutivo: ~7.00×

---

## Multiplicadores confirmados por datos reales

```python
# Estos multiplicadores vienen de los datos reales, no inventados
SECTOR_MUL = {
    "Banca / Financiero":      1.25,
    "Zona Libre de Colón":     1.10,
    "Multinacional":           1.20,
    "Tecnología / Startup":    1.15,
    "Salud / Farmacéutica":    1.05,
    "Gobierno / Público":      0.85,
    "Retail / Comercio":       0.90,
    "Construcción":            1.00,
    "ONG / Sin fines de lucro":0.80,
    "Educación / Academia":    0.82,
}
ESTUDIO_MUL = {
    "Técnico / Universitario incompleto": 0.82,
    "Licenciatura":                        1.00,
    "Postgrado / Especialización":         1.18,
    "Maestría":                            1.30,
    "Doctorado":                           1.45,
}
INGLES_MUL = {
    # Dato real: contador bilingüe CPA ($1,511) vs contador general ($1,123) = +34%
    "Ninguno":         0.88,
    "Básico":          0.95,
    "Intermedio":      1.05,
    "Avanzado":        1.18,
    "Nativo/Bilingüe": 1.28,
}
EMPRESA_MUL = {
    "Multinacional":        1.18,
    "Empresa local grande": 1.00,
    "PYME":                 0.88,
    "Startup":              0.95,
    "Gobierno":             0.87,
    "Freelance":            1.05,
}
MODALIDAD_MUL = {"Presencial": 1.00, "Híbrido": 1.04, "Remoto": 1.08}
PROVINCIA_MUL = {
    "Panamá":       1.00, "Colón":      0.92, "Chiriquí":   0.88,
    "Herrera":      0.82, "Los Santos": 0.80, "Veraguas":   0.81,
    "Coclé":        0.83, "Panamá Oeste":0.90,
}
HERRAMIENTA_BONUS = {
    "tiene_python":   0.10,
    "tiene_powerbi":  0.07,
    "tiene_sql":      0.08,
    "tiene_sap":      0.06,
    "tiene_excel_av": 0.04,
}
```

---

## Lo que está construido y funciona ✅

### Backend — `backend/main.py`
FastAPI con 3 endpoints:
- `GET /opciones` → devuelve todas las opciones del formulario
- `POST /predict` → recibe perfil, retorna estimación + rango + tips
- `GET /health` → estado del modelo

Respuesta de `/predict`:
```json
{
  "salario_estimado": 4920,
  "rango_min": 4330,
  "rango_max": 5511,
  "moneda": "USD",
  "periodicidad": "mensual bruto",
  "tips": ["💡 Mejorar inglés..."],
  "modelo_version": "1.0"
}
```

### Frontend — `frontend/SalarioPA.html`
Prototipo HTML completo con:
- Ticker tape animado con promedios por sector
- Formulario completo (todos los campos del modelo)
- Panel resultado con animación ticker numérico
- Barra mín → estimado → máx animada
- Tips personalizados según perfil
- Modo demo: corre la lógica localmente sin necesitar el backend

### Prototipo React — `frontend/SalarioPA.jsx`
Versión React lista para integrar en proyecto Vite.
Cambiar `API_URL` (línea 6) con la URL de Render al desplegar.

---

## Deploy (cuando esté listo)

| Servicio | Plataforma | Costo |
|---|---|---|
| Backend FastAPI | Render (Free) | $0/mes |
| Frontend React | Vercel (Hobby) | $0/mes |
| Dominio (opcional) | Namecheap | ~$12/año |

**Render — Start Command:**
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
(No usar --port 8000 fijo — Render asigna el puerto dinámicamente)

**CORS en producción:** actualizar `allow_origins` en `main.py` con el dominio de Vercel.

---

## Tarea actual — Fase 1

**Objetivo:** actualizar `generar_dataset.py` para usar `salarios_referencia_panama.csv`
como base real en lugar de rangos inventados.

**Pasos:**
1. Leer `data/salarios_referencia_panama.csv`
2. Para cada cargo real, generar N registros sintéticos alrededor de ese rango
   (distribución normal con media=sal_promedio, std=15% del rango)
3. Aplicar los multiplicadores (sector, estudio, inglés, empresa, modalidad, provincia, herramientas)
4. Aplicar piso Mitradel por área
5. Entrenar el modelo GradientBoostingRegressor con el dataset resultante
6. Guardar `modelo_salario_pa.pkl` y `modelo_metadata.json`
7. Evaluar MAE y R² — esperar mejora respecto a R²=0.61 del dataset sintético

**Criterio de éxito:** R² > 0.70 y MAE < $350

---

## Próximas fases (no hacer ahora)

- **Fase 2:** encuesta propia en LinkedIn para recolectar datos reales de usuarios
- **Fase 3:** percentil de mercado en la respuesta del `/predict`
- **Fase 4:** comparador por provincia (gráfico)
- **Fase 5:** autenticación + historial de estimaciones (Supabase)

---

## Notas importantes

- Todos los salarios en USD/mes (= B/. por paridad 1:1 en Panamá)
- El modelo actual tiene R²=0.61 (dataset 100% sintético) — la Fase 1 debe mejorarlo
- Los registros con `es_imputado=1` son estimaciones razonables, no datos reales
- Las áreas fuera de Finanzas/IT/RRHH (Ingeniería, Salud, Legal, etc.)
  siguen usando rangos sintéticos por falta de datos — prioridad baja por ahora
- El frontend HTML funciona sin backend (modo demo con lógica JavaScript)
- Juan trabaja con <10h/semana en este proyecto — priorizar tareas de alto impacto
