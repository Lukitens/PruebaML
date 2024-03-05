# uvicorn main:app --reload
# http://127.0.0.1:8000/docs Documentacion
# https://pruebaml-zth8.onrender.com/docs#/ en render


#Importamos las librerias
import pandas as pd
from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

#Leemos el archivo a usar
df = pd.read_parquet("modelo_demanda.parquet")

#Iniciamos FastApi en una variable
app = FastAPI()

#Seleccionamos las columnas a usar
X = df[["PULocationID", "PUDay", "PUMonth", "PUHour"]]
Y = df["Demand"]

# Inicializar y entrenar el modelo de árbol de decisión para clasificación
decision_tree_classifier = DecisionTreeClassifier(max_depth=18, min_samples_split=6, min_samples_leaf=4, random_state=123)
decision_tree_classifier.fit(X, Y)

#Creamos un diccionario con los dias y su numero para despues reemplazarlo
dias_a_numeros = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

#Hacemos una variable con la fecha y hora del presente
fecha_actual = datetime.now()

#Dejamos solo la franja horaria
franja_hora = fecha_actual.replace(minute=0, second=0, microsecond=0)
franja_hora = franja_hora.hour
#Pasamos el día del mes a dia de la semana, ejemplo si hoy sería 21/3 pasaría el dia a jueves
# y luego lo pasa a numero 4 usando el diccionario creado anteriormente
dia = fecha_actual.strftime('%A')
dia = dias_a_numeros[dia]
#Extrae el mes actual
mes = fecha_actual.month

#Creamos una funcion para la bienvenida
@app.get("/") 
def bienvenida():
    return {"Bienvenido" : "Bienvenido/s al proyecto grupal final del bootcamp SoyHenry"}

#Creamos la funcion para predecir la demanda actual
@app.get("/predicciondemanda/{distrito}")
def prediccion_demanda(distrito : str):
    """
    Se introduce el ID del distrito para ver la demanda actual
    """
    distrito = int(distrito)
    #Si el id ingresado sigue con el codigo
    if distrito in df["PULocationID"].values:
        #Usamos los datos de la fecha actual
        datos_prediccion = {
        "PULocationID": distrito,  # Ejemplo de valores de PULocationID
        "dia": dia,  # Ejemplo de valores de día
        "mes": mes,  # Ejemplo de valores de mes
        "hora": franja_hora  # Ejemplo de valores de hora
        }
        #Predecimos con los datos de la fecha actual
        preds = decision_tree_classifier.predict([list(datos_prediccion.values())])[0]
        preds = int(preds)
        #Pasamos la categorización de la demanda a palabra para que el usuario entienda
        # 0 = demanda muy baja, 1 = demanda baja, 2 = demanda normal 3 = alta, 4 = muy alta
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
    #Si el id ingresado no existe se le avisa al usuario
    else:
        return ("Tu distrito: {} no existe".format(distrito)) 