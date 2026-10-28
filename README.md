# Laboratorio: Red Neuronal Multi-Capa (MLP) - Piedra, Papel o Tijera

---

## 1. Analisis del Codigo Fuente: Ubicacion de Configuraciones

En esta seccion se detalla la ubicacion exacta y las lineas de codigo correspondientes a cada elemento solicitado en el taller:

### 1.1. La creacion de la red neuronal
* **Archivo:** `main.py`
* **Descripcion:** Se crea la instancia global de la red neuronal especificando la estructura de capas ocultas y el hiperparametro de tasa de aprendizaje (`learning_rate`).
* **Codigo:**
 
  from neural import NeuralNetwork

# Se instancia la red neuronal pasando la estructura de capas ocultas y la tasa de aprendizaje
  nn = NeuralNetwork(hidden=(12, 8), learning_rate=0.01)

---
1.2. La sumatoria de las (Entradas * Pesos) (X * W)
**Archivo:** `neural.py`
 **Metodo:** forward(self, X)
**Descripcion:** Realiza el producto punto entre la matriz de entradas/activaciones anteriores y la matriz de pesos, sumando el vector de sesgo (bias).
 **Codigo:**
# Multiplicacion matricial de la entrada por los pesos mas el sesgo (bias)
z = np.dot(activation, w) + b

---
1.3. La funcion de activacion ReLU
**Archivo:** `neural.py`
**Descripcion:**Funcion auxiliar que rectifica los valores negativos a cero ($f(x) = \max(0, x)$).
 **Codigo:**
f relu(x):
    return np.maximum(0, x)

---
1.4. La funcion de activacion Softmax
**Archivo:** `neural.py`
**Descripcion:** Aplicada en la capa de salida para convertir los valores de corte (logits) en una distribucion de probabilidad normalizada sobre las 3 clases posibles.
 **Codigo:**
f softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

---
1.5. Donde se aplica Backpropagation
**Archivo:** `neural.py`
 **Metodo:** backward(self, X, y) / train_step(self, X, y)
**Descripcion:** Calcula la derivada de la funcion de perdida respecto a los pesos y sesgos, propagando el error hacia atras para actualizar las matrices.
 **Codigo:**
# Calculo del error en la salida
error = output - y_one_hot

# Propagacion del error hacia atras para actualizar weights y biases
# self.weights y self.biases se ajustan en funcion de los gradientes

---
1.6. Donde se definen las entradas
**Archivo:** `neural.py`
**Descripcion:** Se define el tamano de entrada en la lista sizes. Corresponde a 15 valores (las ultimas 5 jugadas codificadas en One-Hot Encoding: 5 x 3 = 15).
 **Codigo:**
# Cinco jugadas recientes x tres valores one-hot = 15 entradas
sizes = [15, *hidden, 3]

---
1.7. Donde se definen las capas ocultas
**Archivo:** `main.py` y `neural.py`
 **Parametro:** hidden=(12, 8)
**Descripcion:** Se especifica a traves de una tupla que por defecto crea dos capas ocultas: la primera con 12 neuronas y la segunda con 8 neuronas.
 **Codigo:**
# neural.py
def __init__(self, hidden=(12, 8), learning_rate=0.01):
# main.py
nn = NeuralNetwork(hidden=(12, 8), learning_rate=0.01)

---
1.8. Donde se define la capa de salida
**Archivo:** `neural.py`
**Descripcion:** El ultimo elemento de la lista sizes define las 3 neuronas de salida, una por cada clase posible (Piedra, Papel o Tijera).
 **Codigo:**
sizes = [15, *hidden, 3]

---
1.9. Donde se definen las Epochs
**Archivo:** `main.py`
**Descripcion:** Se definen en el bucle iterativo que ejecuta los pasos de entrenamiento por cada lote o ciclo.
 **Codigo:**
# Bucle que ejecuta las iteraciones de entrenamiento por epoch
for epoch in range(epochs):
    nn.train_step(X_train, y_train)


 ## 2. Experimentos de Vibe Coding

### 2.1. ¿Qué pasa si creo otras dos capas ocultas de 12 neuronas `(12, 12, 12, 8)` en `hidden`?

* **En el código (`main.py`):**
  ```python
nn = NeuralNetwork(hidden=(12, 12, 12, 8), learning_rate=0.01)

---
## Explicación de qué pasa:

Aprende patrones más complejos: La red se vuelve más profunda, así que tiene más 
capacidad para memorizar o detectar secuencias raras en las jugadas.

Es más lenta: Al agregar dos capas más, la computadora tiene que hacer muchas más 
multiplicaciones de matrices cada vez que calcula una jugada y cuando actualiza los pesos.

Riesgo de sobreajuste (Overfitting): Como el juego solo tiene 15 entradas, meter tantas capas es 
"demasiada red" para un problema tan sencillo. Puede terminar memorizándose las 
últimas jugadas del jugador en vez de aprender a jugar bien.

## 2.2. ¿Qué pasa si aumento las epochs a 100?

* **En el código (`main.py`):**

Python
epochs = 100
for epoch in range(epochs):
    nn.train_step(X_train, y_train)
Explicación de qué pasa:

Entrena más rápido al principio: Al repasar 100 veces el historial en cada turno, la red ajusta sus pesos mucho
 más rápido y aprende casi de inmediato a ganarle a la jugada que estés repitiendo.

Se vuelve terca / Le cuesta adaptarse: Si juegas muchas veces Piedra y la red entrenó 100 épocas para ganarle a
la Piedra, cuando de repente cambies a Tijera le va a costar trabajo "desaprender" lo anterior y adaptarse a tu nuevo truco.


