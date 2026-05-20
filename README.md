# modeloia
# Nombre Mishell Mena
# Proyecto MLflow Práctica 2 El proyecto predice si un estudiante aprueba reprueba.

Para el proyecto se utilizo:

- MLflow
- Scikit-learn
- Streamlit

Se crea un dataset mediante Python y se ubica en data/estudiantes.csv

# Para la ejecución se tiene tres Terminales o cmd

# El primero ejecuta MLFLOW con el comando:
mlflow server --backend-store-uri sqlite:///mlflow.db  --host 0.0.0.0 --port 9090    

# http://127.0.0.1:9090/#/experiments/4/runs?startTime=ALL
# http://127.0.0.1:9090/#/experiments/4/runs?startTime=ALL
# http://127.0.0.1:9090/#/models/Modelo_estudiantes

# El segundo ejecuta el notebook con el comando:
jupyter lab 

# se abre el documento practica2_mlflow_pipeline_students.ipynb y se lo corre o ejecuta línea por línea.

# El tercero es la ejecución del streamlit que es el archivo app.pyy se ejecuta con el comando:
streamlit run app_streamlit.py

# Se abre con http://localhost:8501/
# y perfecto se tiene la predición