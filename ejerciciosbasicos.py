#ejercicio 01

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Generamos datos de ejemplo (X: entrada, Y: salida esperada)
X = np.array([[0], [1], [2], [3], [4]], dtype=float)
Y = np.array([[0], [2], [4], [6], [8]], dtype=float)  # Relación Y = 2X

# 2. Creamos un modelo de red neuronal con una capa densa
modelo = keras.Sequential([
    keras.Input(shape=(1,)),  # Definir explícitamente la forma de entrada
    keras.layers.Dense(units=1)  # 1 neurona en la capa de salida
])

# 3. Compilamos el modelo con optimizador y función de pérdida
modelo.compile(optimizer="sgd", loss="mean_squared_error")

# 4. Entrenamos el modelo
print("Entrenando...")
modelo.fit(X, Y, epochs=500, verbose=0)  # Entrena durante 500 épocas

# 5. Probamos con una nueva entrada
entrada = np.array([[5]])  # Convertimos a np.array para evitar errores
prediccion = modelo.predict(entrada)
print(f"Predicción para X=5: {prediccion[0][0]:.2f}")  # Redondeamos a 2 decimales


#ejercicio 02
              #para realizar este ejercicio se importo Numpy 
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Probamos con valores
valores = np.array([-2, 0, 2])
print("Resultados de la función sigmoide:", sigmoide(valores))


#ejercicio 03
             #para realizar este ejercicio se importo Numpy
# Definimos la función AND con un perceptrón
def perceptron_AND(x1, x2):
    w1, w2, bias = 1, 1, -1.5  # Pesos y sesgo
    suma = x1*w1 + x2*w2 + bias
    return 1 if suma > 0 else 0  # Función de activación escalón

# Probamos todas las combinaciones de AND
entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in entradas:
    print(f"AND({x1}, {x2}) = {perceptron_AND(x1, x2)}")