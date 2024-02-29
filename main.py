# uvicorn main:app --reload
# http://127.0.0.1:8000/docs Documentacion

import pandas as pd
from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

df = pd.read_csv("tabla_ml.csv")
df["Borough"] = df["Borough"].str.lower()

app = FastAPI()

X = df[["PULocationID", "dia", "mes", "hora"]]
Y = df["demand"]

# Inicializar y entrenar el modelo de árbol de decisión para clasificación
decision_tree_classifier = DecisionTreeClassifier(max_depth=18, min_samples_split=6, min_samples_leaf=4, random_state=123)
decision_tree_classifier.fit(X, Y)

dias_a_numeros = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

fecha_actual = datetime.now()

franja_hora = fecha_actual.replace(minute=0, second=0, microsecond=0)
franja_hora = franja_hora.hour
dia = fecha_actual.strftime('%A')
dia = dias_a_numeros[dia]
mes = fecha_actual.month

@app.get("/") 
def read_root():
    return {"Bienvenido" : "Bienvenido/s al proyecto grupal final del bootcamp SoyHenry"}

@app.get("/predicciondemanda/{distrito}")
def prediccion_demanda(distrito : str):
    distrito = int(distrito)
    if distrito in df["PULocationID"].values:
        datos_prediccion = {
        "PULocationID": distrito,  # Ejemplo de valores de PULocationID
        "dia": dia,  # Ejemplo de valores de día
        "mes": mes,  # Ejemplo de valores de mes
        "hora": franja_hora  # Ejemplo de valores de hora
        }
        preds = decision_tree_classifier.predict([list(datos_prediccion.values())])[0]
        preds = int(preds)
        if preds == 0:
            return("Demanda muy baja")
        elif preds == 1:
            return("Demanda baja")
        elif preds == 2:
            return("Demanda normal")
        elif preds == 3:
            return("Demanda alta")
        else:
            return("Demanda muy alta")
    else:
        return ("Tu distrito: {} no existe".format(distrito)) 