#ejercicio 01

import tensorflow as tf

a = tf.constant(5)
b = tf.constant(3)
suma = tf.add(a, b)

print("La suma de 5 + 3 es:", suma.numpy())  # Convierte el tensor a un número

#ejercicio 02
             #ejercicio relizado con TensorFlow 
# Crear un tensor de 3x3 con números aleatorios
tensor = tf.random.uniform([3, 3])

print("Tensor creado:\n", tensor.numpy())  # Convertir a numpy para imprimir
print("Forma del tensor:", tensor.shape)


#ejercicio 03
             #jercicio realizado con TensorFlow
# Crear dos matrices de 2x2
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

# Multiplicación de matrices
resultado = tf.matmul(A, B)

print("Matriz A:\n", A.numpy())
print("Matriz B:\n", B.numpy())
print("Resultado de A x B:\n", resultado.numpy())

#ejercicio 04 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generamos valores de X
X = np.linspace(-10, 10, 100)  # 100 valores entre -10 y 10

# Definimos la función Y = X^2 en TensorFlow
X_tf = tf.constant(X, dtype=tf.float32)
Y_tf = tf.pow(X_tf, 2)  # Elevar al cuadrado

# Convertimos a numpy para graficar
Y = Y_tf.numpy()

# Graficamos la función Y = X^2
plt.plot(X, Y, label="Y = X^2", color="blue")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gráfica de Y = X^2 con TensorFlow")
plt.legend()
plt.grid()
plt.show()