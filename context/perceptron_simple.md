# Contexto teórico: Perceptrón Simple

## Idea principal

El perceptrón simple busca **replicar el modelo de una neurona biológica** de manera simplificada.

### Componentes del modelo

- **Entradas:** $x_i$ — valores que recibe la neurona.
- **Pesos sinápticos:** $w_j$ — ponderan la importancia de cada entrada.
- **Estado de excitación:** $h$ — resultado de la suma ponderada de entradas por sus pesos.
- **Función de activación:** $\theta$ — se evalúa en $(h - u)$, siendo $u$ el umbral (*threshold*).
- **Salida real:** $O$ — resultado final de la neurona.

> **Nota sobre el umbral/bias (b) y el peso $w_0$:**
> En la práctica, el umbral $u$ se puede absorber incluyendo una entrada **extra** $x_0 = 1$ con peso $w_0 = -u$. Esto equivale a usar un **bias** $b = w_0$, y permite que el hiperplano de separación **no esté forzado a pasar por el origen**.

## ¿Para qué nos sirve?

El perceptrón simple escalón resuelve **problemas de clasificación entre dos grupos**.

Separa los datos trazando una **recta**, un **plano** o un **hiperplano** de decisión. Si un dato cae de un lado del hiperplano pertenece a un grupo; si cae del otro lado, pertenece al otro.

---

## Regla de aprendizaje

Cada vez que la neurona recibe un estímulo, los pesos se actualizan de forma iterativa según la regla de **Rosenblatt**:

$$
\Delta w = \eta \cdot (t^{\mu} - O^{\mu}) \cdot x^{\mu}
$$

Donde:

- **$t^{\mu}$**: salida **esperada** para el dato $\mu$.
- **$O^{\mu}$**: salida **obtenida** por el perceptrón para el dato $\mu$.
- **$\mu$**: índice del dato de entrenamiento (ejemplo: el décimo dato tiene $\mu = 10$, o $9$ si empieza en $0$).
- **$\eta$**: tasa de aprendizaje. Generalmente entre $0$ y $0.1$.
  - Una tasa muy alta puede modificar los parámetros de forma demasiado brusca.
  - Una tasa muy baja puede hacer que el modelo nunca converja.

Luego: $w^{nuevo} = w^{anterior} + \Delta w$.

---

## Algoritmo de entrenamiento del perceptrón simple escalón

En la práctica, se incluye el bias como un peso adicional $w_0$ con entrada $x_0 = 1$:

```
Initialize weights w to small random values
Initialize weight w_0 to a small random value (or 0)
Set learning rate eta

for a fixed number of epochs:
  for each training example mu in the dataset:
   1. Calculate the weighted sum:
     h^mu = w_0 * 1 + w_1 * x^mu_1 + w_2 * x^mu_2 + ... + w_n * x^mu_n
   2. Compute activation given by theta (step function):
     O^mu = theta(h^mu)
     
     theta(h) = 1  if h > threshold (or h > 0 if threshold absorbed in w_0)
           else 0

   3. Update the weights:
      For each weight w_i (i = 0..n):
         w_i = w_i + eta * (t^mu - O^mu) * x^mu_i
      (Recordar que x^mu_0 = 1 para actualizar w_0)

  4. After iterating over ALL examples, calculate global perceptron error
     (e.g., MSE or number of misclassifications)
  5. Check for convergence:
     convergence = true if error < epsilon (or, ideally, error == 0 for all data)
     if convergence: break

  If convergence was NOT reached, continue to the next epoch.
End
```

---

## Funciones de error

Algunas opciones para evaluar el desempeño del perceptrón al final de cada época:

1. **MSE (Mean Squared Error):**
   $$
   MSE = \frac{1}{p} \sum_{\mu=1}^{p} (t^\mu - O^\mu)^2
   $$
   Donde $p$ es la **cantidad de patrones** (datos) en el conjunto de entrenamiento.

2. **Diferencia absoluta** entre lo esperado y lo obtenido.

3. **Suma de errores absolutos** igual a 0 (todos los datos bien clasificados).

4. **Accuracy** del 100% (para problemas de clasificación).

---

## Fases del perceptrón

Existen dos fases bien definidas:

1. **Aprendizaje:** Se ajustan los pesos iterativamente usando los datos de entrenamiento, buscando minimizar el error o la función de costo sobre el conjunto de datos.
2. **Generalización:** Se evalúa el modelo con datos no vistos (datos de test) para medir su capacidad de desempeñarse correctamente. Depende directamente de la calidad del aprendizaje (*Garbage In, Garbage Out*).
