"""
SalarioPA — Backend FastAPI
Endpoints: GET /opciones, POST /predict, GET /health
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SalarioPA API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # actualizar con dominio Vercel en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "modelo_salario_pa.pkl"
META_PATH = BASE / "modelo_metadata.json"

model = None
metadata = {}

PISO_MITRADEL = {
    "Tecnología / IT":         789,
    "Finanzas / Banca":        754,
    "Ingeniería":              842,
    "Salud":                   739,
    "Administración":          725,
    "Recursos Humanos":        725,
    "Educación":               725,
    "Logística / Operaciones": 754,
    "Marketing / Ventas":      725,
    "Legal":                   725,
    "default":                 637,
}


@app.on_event("startup")
def load_model():
    global model, metadata
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)


class PerfilRequest(BaseModel):
    area: str
    sector: str
    tipo_empresa: str
    modalidad: str
    provincia: str
    nivel_estudio: str
    nivel_ingles: str
    años_experiencia: int
    tiene_python: int = 0
    tiene_powerbi: int = 0
    tiene_sql: int = 0
    tiene_sap: int = 0
    tiene_excel_av: int = 0


@app.get("/opciones")
def opciones():
    return {
        "areas":          metadata.get("areas", []),
        "sectores":       metadata.get("sectores", []),
        "tipos_empresa":  metadata.get("tipos_empresa", []),
        "modalidades":    metadata.get("modalidades", []),
        "provincias":     metadata.get("provincias", []),
        "niveles_estudio": metadata.get("niveles_estudio", []),
        "niveles_ingles": metadata.get("niveles_ingles", []),
    }


@app.post("/predict")
def predict(perfil: PerfilRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    row = {
        "area":             perfil.area,
        "sector":           perfil.sector,
        "tipo_empresa":     perfil.tipo_empresa,
        "modalidad":        perfil.modalidad,
        "provincia":        perfil.provincia,
        "nivel_estudio":    perfil.nivel_estudio,
        "nivel_ingles":     perfil.nivel_ingles,
        "años_experiencia": perfil.años_experiencia,
        "tiene_python":     perfil.tiene_python,
        "tiene_powerbi":    perfil.tiene_powerbi,
        "tiene_sql":        perfil.tiene_sql,
        "tiene_sap":        perfil.tiene_sap,
        "tiene_excel_av":   perfil.tiene_excel_av,
    }
    X = pd.DataFrame([row])
    pred = float(model.predict(X)[0])

    piso = PISO_MITRADEL.get(perfil.area, PISO_MITRADEL["default"])
    salario = max(pred, piso)
    rango_min = max(round(salario * 0.88), piso)
    rango_max = round(salario * 1.12)

    tips = _generar_tips(perfil, salario)

    return {
        "salario_estimado": round(salario),
        "rango_min": rango_min,
        "rango_max": rango_max,
        "moneda": "USD",
        "periodicidad": "mensual bruto",
        "tips": tips,
        "modelo_version": metadata.get("version", "1.1"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_cargado": model is not None,
        "r2": metadata.get("r2"),
        "mae": metadata.get("mae"),
        "version": metadata.get("version"),
    }


def _generar_tips(perfil: PerfilRequest, salario: float) -> list[str]:
    tips = []
    if perfil.nivel_ingles in ("Ninguno", "Básico"):
        tips.append("💡 Mejorar inglés a nivel Avanzado puede incrementar tu salario hasta un 18%")
    if perfil.nivel_estudio == "Técnico / Universitario incompleto":
        tips.append("🎓 Completar una licenciatura puede aumentar tu salario ~22%")
    elif perfil.nivel_estudio == "Licenciatura":
        tips.append("🎓 Un postgrado o maestría puede incrementar tu compensación entre 18% y 30%")
    if perfil.area == "Tecnología / IT":
        if not perfil.tiene_python:
            tips.append("🐍 Python es una habilidad de alta demanda en IT — considera aprenderlo")
        if not perfil.tiene_sql:
            tips.append("🗄️ SQL es requerido en la mayoría de roles de datos y desarrollo backend")
    if perfil.area == "Finanzas / Banca":
        if not perfil.tiene_excel_av:
            tips.append("📊 Excel avanzado (tablas dinámicas, Power Query) es clave en finanzas")
        if not perfil.tiene_sap:
            tips.append("💼 Dominar SAP puede diferenciarte en roles de finanzas y contabilidad")
    if perfil.tipo_empresa == "PYME":
        tips.append("🏢 Las multinacionales y empresas grandes pagan hasta 30% más que las PYMEs en Panamá")
    if perfil.provincia != "Panamá":
        tips.append("🏙️ Trabajar en Ciudad de Panamá o en modalidad remota mejora los ingresos")
    if perfil.modalidad == "Presencial":
        tips.append("🏠 La modalidad híbrida/remota puede representar un aumento efectivo del 4-8%")
    if not tips:
        tips.append("✅ Tu perfil está bien posicionado para el mercado panameño")
    return tips[:3]
