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
    "models:/Modelo_Salud_Mental/1"
)

# =====================================================
# INTERFAZ
# =====================================================

st.title("Predicción de Salud Mental Adolescente")

st.write(
    "Sistema de Machine Learning usando MLflow y Streamlit"
)

# =====================================================
# ENTRADAS
# =====================================================

edad = st.slider(
    "Edad",
    13,
    19,
    16
)

genero = st.selectbox(
    "Género",
    [
        "Male",
        "Female"
    ]
)

horas_sueno = st.slider(
    "Horas de sueño",
    1,
    12,
    7
)

horas_redes = st.slider(
    "Horas en redes sociales",
    0,
    15,
    5
)

actividad_fisica = st.selectbox(
    "Actividad física",
    [
        "Low",
        "Medium",
        "High"
    ]
)

estres_academico = st.slider(
    "Nivel de estrés académico",
    1,
    10,
    5
)

# =====================================================
# PREDICCIÓN
# =====================================================

if st.button("Predecir"):

    datos = pd.DataFrame([{

        "Age": edad,
        "Gender": genero,
        "Sleep_Hours": horas_sueno,
        "Social_Media_Hours": horas_redes,
        "Physical_Activity": actividad_fisica,
        "Academic_Stress": estres_academico

    }])

    prediccion = modelo.predict(datos)[0]

    probabilidad = modelo.predict_proba(datos)[0][1]

    st.subheader("Resultado")

    if prediccion == 1:

        st.error(
            "El adolescente presenta riesgo de afectación en salud mental"
        )

    else:

        st.success(
            "El adolescente presenta bajo riesgo"
        )

    st.write(
        f"Probabilidad de riesgo: {probabilidad:.2%}"
    )

    st.write("Datos ingresados:")

    st.dataframe(datos)
