# Contexto teórico: Perceptrón Multicapa (MLP)

## Motivación: Limitaciones del Perceptrón Simple

El perceptrón simple (escalón, lineal o no lineal) está limitado a problemas que puedan resolverse con una sola neurona. Por ejemplo, no puede resolver el problema del **XOR** combinando los datos con una única recta o función.

La solución consiste en **combinar múltiples perceptrones** en una red de neuronas organizadas en capas.

---

## Teorema de Aproximación Universal

**1989 - Cybenko, G.** (expansión en 1990/1991, Hornik, K.)

> Con un perceptrón multicapa, con **una capa oculta**, con una cantidad indefinida de neuronas en esa capa oculta, puedo representar **cualquier función continua**.

Este teorema dice que, en teoría, un MLP con una sola capa oculta suficientemente grande puede aproximar cualquier función continua. En la práctica, la arquitectura (cantidad de capas y neuronas) se define de forma **empírica**.

---

## Arquitectura del Perceptrón Multicapa

### Estructura general

- **Capa 0:** Datos de entrada.
- **Capas intermedias (ocultas):** Desde la capa 1 hasta la capa M-1.
- **Capa M:** Capa de salida.

Puede haber **más de una neurona en la capa de salida**, dependiendo del problema. Por ejemplo, para clasificación en muchos grupos se usa una neurona de salida por clase.

### Notación

| Símbolo | Descripción |
|---------|-------------|
| $i$ | Índice de la neurona de la capa de salida (o siguiente) |
| $j$ | Índice de la neurona de la capa intermedia |
| $k$ | Índice de la neurona de entrada o de la capa anterior |
| $m$ | Índice de la capa intermedia |
| $p$ | Cantidad de datos |
| $\mu$ | Dato en particular |

### Feed-Forward Pass (Propagación hacia adelante)

Para calcular la salida de la red, la información se propaga de izquierda a derecha (de la entrada a la salida):

1. Cada neurona de la capa siguiente calcula su estado de excitación sumando las entradas ponderadas por los pesos.
2. Se aplica la función de activación.
3. El resultado se envía como entrada a la siguiente capa.

```
Entrada (x_k) → [h_j = Σ w_jk · x_k] → θ(h_j) → Salida (V_j)
```

> **Tip:** expresar todo de manera matricial utilizando alguna librería como **numpy**.

---

## Entrenamiento: Retropropagación (Backpropagation)

El entrenamiento del perceptrón multicapa requiere:
1. Una **función de costo** (error).
2. Un **mecanismo de optimización** (por ahora, gradiente descendente).
3. La **regla de la cadena** para calcular el efecto de los pesos de las capas ocultas.

### Problema central

Si queremos calcular el error en una **capa intermedia**, no podemos hacerlo directamente porque no conocemos la salida esperada de esas neuronas intermedias. Tenemos que ver **cómo se afectan los pesos posteriores** y, a su vez, cómo estos dependen de las capas anteriores.

### Algoritmo de backpropagation (Rumelhart, Hinton, Williams - 1986)

Backpropagation provee un mecanismo para calcular, de forma **eficiente**, cuánto contribuye cada neurona al error total del perceptrón multicapa. Es lo que en Python hace **`.backward()`** en librerías como PyTorch o TensorFlow.

### Concepto de Delta (δ)

Se define un **delta** $\delta_j$ asociado a cada neurona $j$. El delta representa **"la porción de la regla de la cadena que se repite si quiero reutilizar ese camino"**.

#### Delta en la capa de salida ($j = M$)

En la capa de salida, podemos calcular el error directamente comparando con la salida esperada:

$$
\delta_i = (t_i - O_i) \cdot \theta'(h_i)
$$

Donde:
- $t_i$: salida esperada de la neurona de salida $i$.
- $O_i$: salida obtenida por la neurona.
- $\theta'(h_i)$: derivada de la función de activación evaluada en $h_i$.

Luego se actualizan los pesos que **conectan la última capa oculta con la salida**.

#### Delta en capas ocultas ($j = M-1, \dots, 1$)

Para una neurona en una capa oculta, el delta se calcula **propagando hacia atrás** los deltas de la capa siguiente:

$$
\delta_j = \theta'(h_j) \cdot \sum_{i} (W_{ij} \cdot \delta_i)
$$

Donde:
- $\delta_i$: delta de la neurona $i$ de la capa siguiente (más cercana a la salida).
- $W_{ij}$: peso que conecta la neurona $j$ de la capa actual con la neurona $i$ de la capa siguiente.
- El sumatorio se extiende sobre todas las neuronas $i$ de la capa siguiente.

### Algoritmo en versión Batch (matricial)

El algoritmo de backpropagation en su versión **batch** procesa todos los datos antes de actualizar los pesos. Se expresa de manera **matricial** para eficiencia:

```
Initialize all weights W^(m) to small random values (o 0 para bias si existe)
Set learning rate η

for each epoch:
  1. Initialize accumulator for weight changes: ΔW = 0

  2. For each training example μ = 1 to p:
     a. FEED-FORWARD PASS:
        - Compute hidden layer activations: h^(1) = W^(1) · x^μ
        - V^(1) = θ(h^(1))
        - Continue for each layer m
        - Final output: O^μ = V^(M) = θ^(M)(h^(M))

     b. BACKWARD PASS:
        - Calculate error at output layer: δ^(M) = (t^μ - O^μ) ⊙ θ'(h^(M))
        - For each hidden layer m from M-1 down to 1:
          δ^(m) = θ'(h^(m)) ⊙ [ (W^(m+1))^T · δ^(m+1) ]

     c. Compute weight changes for this example:
        - ΔW^(m) += η · δ^(m) · (V^(m-1))^T
        - (for m=1, V^(0) = x^μ)

  3. Update all weights:
     W^(m) = W^(m) + ΔW^(m) / p     (dividir por p si se prefiere promedio)

  4. Calculate global error
  5. Check for convergence
End
```

**Notación:** `⊙` denota producto elemento a elemento (Hadamard).

**Tip:** En la práctica, se suele implementar utilizando operaciones matriciales con **numpy** para mejorar el rendimiento.

---

### Actualización de pesos (vista por neurona)

Una vez calculados los deltas para todas las neuronas, se actualizan los pesos neurona a neurona:

**Pesos de capa oculta a capa de salida (W mayúsculas):**
$$
\Delta W_{ij} = \eta \cdot \delta_i \cdot V_j
$$

**Pesos de capa de entrada a capa oculta (w minúsculas):**
$$
\Delta w_{jk} = \eta \cdot \delta_j \cdot x_k
$$

Donde $V_j$ es la salida de la neurona $j$ de la capa oculta.

En versión batch, estos cambios se **acumulan** sobre todos los ejemplos y se promedian antes de aplicar.

---

## Estrategias de entrenamiento

Existen tres modos principales de actualizar los pesos, dependiendo de cuándo se aplica la corrección:

| Modo | Descripción | Nombre formal |
|------|-------------|---------------|
| **Incremental / Online** | Se calcula $\Delta w$ para **un** elemento del conjunto y se actualiza el peso inmediatamente. | Gradiente Descendente Estocástico (SGD) |
| **Mini Lote / Mini Batch** | Se calcula $\Delta w$ para un **subconjunto** de elementos y se actualiza. | Gradiente Descendente Estocástico (SGD) |
| **Lote / Batch** | Se calcula $\Delta w$ para **todos** los elementos del conjunto y luego se actualiza. | Gradiente Descendente |

Evaluar cada estrategia según:
- Costo de cada época.
- Requerimientos de memoria.
- Cómo manejan el ruido (ejemplo: outliers).
- Velocidad de convergencia.

---

## Función de costo

Se puede usar **Mean Square Error (MSE)** como función de costo:

$$
E = \frac{1}{2} \sum_{\mu=1}^{p} \sum_{i} (t_i^{\mu} - O_i^{\mu})^2
$$

Donde la suma en $i$ recorre todas las neuronas de salida.

---

## Inicialización de pesos

- **NO inicializar los pesos en cero**: lleva al problema de **simetría**. Si todos los pesos son iguales, todas las neuronas de una misma capa aprenderán exactamente lo mismo.
- **Recomendado:** inicializar los pesos con valores aleatorios de una **distribución uniforme/gaussiana**, en valores pequeños.

---

## Bias en el perceptrón multicapa

Se puede incluir un bias (umbral) en cada capa agregando una entrada constante $x_0 = 1$ con peso $w_0$ de forma análoga al perceptrón simple.

- Esto permite flexibilizar la forma de la salida de cada neurona (desplazamiento).
- **No es estrictamente necesario en todas las capas**: se puede probar incluirlo solo en la entrada, en todas las capas, o en ninguna. El bias permite más parámetros libres para aprender patrones más complejos.
- Se puede incorporar en los cálculos de manera **matricial**.

---

## Resumen

| Perceptrón Simple | Perceptrón Multicapa |
|---|---|
| Predecir: suma ponderada + función de activación | Múltiples capas de sumas ponderadas + activaciones |
| Función de costo: MSE o error absoluto | MSE sobre todas las neuronas de salida |
| Aprendizaje: regla directa (Rosenblatt) o gradiente | Backpropagation + Gradiente Descendente |
| Un solo paso de actualización | Dos pasos: forward para predecir, backward para corregir |
| Arquitectura fija (1 neurona) | Arquitectura definida manualmente (capas, neuronas) |
