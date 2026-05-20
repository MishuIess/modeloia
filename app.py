# =====================================================
# PREDICCIÓN DE APROBACIÓN
# =====================================================

import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

# =====================================================
# CONFIGURAR MLFLOW
# =====================================================

mlflow.set_tracking_uri("http://127.0.0.1:9090")

# =====================================================
# CARGAR MODELO
# =====================================================

modelo = mlflow.sklearn.load_model(
    # "models:/Modelo_estudiantes/1"
    # "models:/Modelo_estudiantes/2"
    # "models:/Modelo_estudiantes/5"
    # "models:/Modelo_estudiantes/7"
    # "models:/Modelo_estudiantes/8"
    "models:/Modelo_estudiantes/10"
)

# =====================================================
# INTERFAZ
# =====================================================

st.title("Predicción de Aprobación")

st.write(
    "Modelo de Machine Learning para predecir si un estudiante aprueba o reprueba"
)

# =====================================================
# ENTRADAS DEL CSV
# =====================================================

carrera = st.selectbox(
    "Carrera",
    [
        "Arquitectura",
        "Computacion",
        "Derecho",
        "Economia",
        "Industrial",
        "Medicina"
        
    ]
)

modalidad = st.selectbox(
    "Modalidad",
    [
        "Presencial",
        "Virtual",
        "Hibrida"
    ]
)

beca = st.selectbox(
    "Beca",
    [
        "Si",
        "No"
    ]
)

edad = st.slider(
    "Edad",
    18,
    30,
    25
)

promedio = st.slider(
    "Promedio",
    0.0,
    10.0,
    6.5
)

asistencias = st.slider(
    "Asistencias",
    0,
    100,
    75
)

# =====================================================
# PREDICCIÓN
# =====================================================

if st.button("Predecir"):

    datos = pd.DataFrame([{

        "carrera": carrera,
        "modalidad": modalidad,
        "beca": beca,
        "edad": edad,
        "promedio": promedio,
        "asistencias": asistencias

    }])

    # Predicción
    prediccion = modelo.predict(datos)[0]

    # Probabilidad
    probabilidad = modelo.predict_proba(datos)[0]

    st.subheader("Resultado")

    # Resultado final
    if prediccion == 1 or prediccion == "Aprueba":

        st.success(
            "El estudiante APRUEBA"
        )

        st.write(
            f"Probabilidad de aprobar: {probabilidad[1]:.2%}"
        )

    else:

        st.error(
            "El estudiante REPRUEBA"
        )

        st.write(
            f"Probabilidad de reprobar: {probabilidad[0]:.2%}"
        )

    # Mostrar datos
    st.write("Datos ingresados")

    st.dataframe(datos)
