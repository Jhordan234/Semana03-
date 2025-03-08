import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar el dataset MNIST (imágenes de números 0-9)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalizar los datos (escala de 0 a 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 3. Crear el modelo de red neuronal
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Convertir imagen 28x28 en un vector
    keras.layers.Dense(128, activation='relu'),  # Capa oculta con 128 neuronas
    keras.layers.Dense(10, activation='softmax') # Capa de salida con 10 clases (0-9)
])

# 4. Compilar el modelo
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# 5. Entrenar el modelo
print("Entrenando...")
modelo.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 6. Evaluar en datos de prueba
test_loss, test_acc = modelo.evaluate(X_test, y_test, verbose=2)
print("\nPrecisión en datos de prueba:", test_acc)

# 7. Mostrar una imagen del dataset y su predicción
indice = np.random.randint(0, len(X_test))
imagen = X_test[indice]

plt.imshow(imagen, cmap="gray")
plt.title(f"Predicción del modelo: {np.argmax(modelo.predict(np.expand_dims(imagen, axis=0)))}")
plt.show()