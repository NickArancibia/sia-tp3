# Contexto teórico: Métricas, Generalización y Sobreajuste

## Idea principal

Hasta ahora vimos cómo **entrenar** un perceptrón. Falta responder tres preguntas que sostienen toda la práctica:

1. ¿Cómo dividimos los datos para que el modelo realmente generalice?
2. ¿Cómo medimos cuantitativamente la calidad de un clasificador o regresor?
3. ¿Es verdad que si $E(w) \equiv 0$ sobre el conjunto de entrenamiento, el clasificador es perfecto?

> **Spoiler de la pregunta 3:** *no.* Error 0 sobre el conjunto de entrenamiento puede significar que el modelo **memorizó** los datos sin aprender el patrón subyacente. Eso es **overfitting** (sobreajuste).

---

## Train / Validation / Test

Un único split train/test no alcanza cuando hay hiperparámetros que ajustar. La división recomendada es en **tres** conjuntos:

| Conjunto | Para qué | Cuándo se mira |
|---|---|---|
| **Train** | Ajustar los pesos $w$ (lo que minimiza la loss) | En cada paso del optimizador |
| **Validation** | Ajustar **hiperparámetros** ($\eta$, arquitectura, $\beta$, regularización…) y decidir cuándo parar | Después de cada época (o cada $k$ épocas) |
| **Test** | Estimar la performance final del modelo elegido | **Una sola vez**, al final del proceso |

> **Regla de oro:** si tocás el modelo en función de lo que viste en test, ya no tenés un test. Pasa a ser otro val, y perdés la estimación honesta de generalización.

### Proporciones típicas

Sin razones para cambiarlo: **70 / 15 / 15** o **80 / 10 / 10**.

### Estratificación

Si las clases están **desbalanceadas** (típico en detección de fraude — Ej1), el split debe ser **estratificado**: la proporción de cada clase en train, val y test debe coincidir con la del dataset completo. Si no se estratifica, una clase rara puede caer entera en val y desaparecer del train.

```
def stratified_split(X, y, val_frac, rng):
    train_idx, val_idx = [], []
    for c in unique(y):
        idx = where(y == c)
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_frac))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])
    return train_idx, val_idx
```

---

## K-Fold Cross Validation

Es la respuesta a *"¿cómo sé si la partición train/test es apropiada?"*.

**Idea:** dividir el dataset en $k$ partes ("folds") aproximadamente iguales. Para $j = 1..k$:

1. Tomar el fold $j$ como validation.
2. Entrenar con los $k-1$ folds restantes.
3. Calcular las métricas sobre el fold $j$.

Al final se reporta el **promedio** (y el desvío) de las métricas sobre los $k$ folds.

**Cuándo usarlo:**

- Datasets chicos donde un único split puede ser tan ruidoso que las conclusiones cambien según la seed.
- Comparación de hiperparámetros: si dos configuraciones difieren en 0.5 % con un único split, puede ser ruido. Con k-fold se ve si la diferencia es consistente.

**Cuándo NO usarlo:**

- Datasets grandes donde un único split val ya es estadísticamente confiable.
- Modelos caros de entrenar: hacer $k$ entrenamientos completos puede ser inviable. En grids grandes, conviene dejar k-fold solo para el modelo finalista.

**Valores típicos de $k$:** 5 o 10. Stratified k-fold cuando hay desbalance.

---

## Matriz de confusión

### Caso binario

Convención: filas = clase real, columnas = clase predicha.

|  | Pred. Positivo | Pred. Negativo |
|---|---|---|
| **Real Positivo** | TP | FN |
| **Real Negativo** | FP | TN |

- **TP** (true positive): clasificó positivo y era positivo.
- **TN** (true negative): clasificó negativo y era negativo.
- **FP** (false positive, *error tipo I*): clasificó positivo pero era negativo.
- **FN** (false negative, *error tipo II*): clasificó negativo pero era positivo.

> **Convención del positivo en Ej1.** En detección de fraude, "positivo" = "es fraude". Entonces FN = fraude que dejamos pasar, FP = transacción legítima marcada como fraude. Los costos son muy distintos según el negocio.

### Caso multiclase

Para $C$ clases, la matriz es $C \times C$ con $M_{ij}$ = cantidad de muestras de clase real $i$ predichas como $j$. La **diagonal** son los aciertos. Fuera de la diagonal vive la información más interesante: **qué clases se confunden con qué**. En Ej2/Ej3 esto importa: si el modelo confunde 3↔8, 4↔9, 1↔7, eso sugiere por dónde mejorar.

---

## Métricas

### Métricas binarias (todas derivadas de TP/FN/FP/TN)

| Métrica | Fórmula | Lectura |
|---|---|---|
| **Accuracy** | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | Fracción total de aciertos. |
| **Precision** | $\dfrac{TP}{TP + FP}$ | De las que predije positivo, ¿cuántas lo eran? |
| **Recall (TPR)** | $\dfrac{TP}{TP + FN}$ | De las realmente positivas, ¿cuántas detecté? |
| **F1-score** | $\dfrac{2 \cdot P \cdot R}{P + R}$ | Media armónica de precision y recall. |
| **FPR** | $\dfrac{FP}{FP + TN}$ | Fracción de negativos que clasifico mal. |
| **Specificity (TNR)** | $\dfrac{TN}{TN + FP}$ | $= 1 - \mathrm{FPR}$. |

### ⚠️ Por qué accuracy es engañosa con clases desbalanceadas

> **Esto es crítico para Ej1.** Datos de fraude son siempre desbalanceados (típicamente ≪ 1 % de transacciones son fraude). Un modelo trivial que predice "siempre legítimo" alcanza ~99 % de accuracy y es **inútil**.

Por eso en Ej1 hay que mirar **al menos**:

- **Precision y recall** sobre la clase positiva (fraude).
- **F1** como combinación de las dos.
- **AUC-PR** (ver más abajo) como métrica resumen.

Accuracy puede acompañar pero **no debe ser la métrica principal** cuando hay desbalance.

### Cuándo importa precision vs recall

| Situación | Métrica que prioriza | Ejemplo |
|---|---|---|
| Falsos positivos son caros | **Precision** alta | Marcar mails legítimos como spam. |
| Falsos negativos son caros | **Recall** alto | Detección de cáncer (o fraude). |
| Ambos importan parecido | **F1** | Default razonable. |

### Métricas multiclase — promediado

Para $C$ clases (Ej2/Ej3 con $C = 10$), accuracy es directa. Para precision/recall/F1 hay que decidir cómo agregar:

| Promedio | Cómo | Cuándo usarlo |
|---|---|---|
| **Macro** | Promedio simple por clase | Tratar todas las clases por igual, aunque haya desbalance. |
| **Weighted** | Promedio ponderado por soporte (cantidad de muestras de cada clase) | Refleja la métrica global respetando frecuencias. |
| **Micro** | Calcula TP/FP/FN globales y aplica la fórmula | En clasificación multiclase con un solo predictor, equivale a accuracy. |

Para Ej2/Ej3 (dígitos probablemente balanceados), macro y weighted dan parecido y accuracy es informativa. Si hay desbalance, **macro F1** es la métrica honesta.

### Métricas de regresión

Para problemas donde la salida es continua (perceptrón lineal, perceptrón no lineal modelando una probabilidad continua):

| Métrica | Fórmula | Comentario |
|---|---|---|
| **MSE** | $\tfrac{1}{p}\sum_\mu (\zeta^\mu - O^\mu)^2$ | Mismo que la loss. Penaliza errores grandes. |
| **RMSE** | $\sqrt{\mathrm{MSE}}$ | Misma unidad que la salida. |
| **MAE** | $\tfrac{1}{p}\sum_\mu \|\zeta^\mu - O^\mu\|$ | Más robusta a outliers. |
| **$R^2$** | $1 - \tfrac{\sum (\zeta - O)^2}{\sum (\zeta - \bar\zeta)^2}$ | Fracción de varianza explicada. |

---

## Selección de umbral (Ej1)

El perceptrón no lineal con activación logística devuelve $O \in (0, 1)$ — una probabilidad. Para clasificar hay que elegir un **umbral** $\tau$: predigo "positivo" si $O \geq \tau$.

### Curva ROC

Variando $\tau$ de 0 a 1 se obtienen distintos puntos $(\mathrm{FPR}(\tau), \mathrm{TPR}(\tau))$. Plotearlos da la curva ROC:

- Eje x: FPR ($= 1 -$ specificity).
- Eje y: TPR ($=$ recall).
- Modelo ideal: pasa por $(0, 1)$.
- Modelo random: diagonal $\mathrm{TPR} = \mathrm{FPR}$.

**AUC-ROC** = área bajo la curva $\in [0, 1]$:

- 0.5 = clasificador aleatorio.
- 1.0 = clasificador perfecto.
- Interpretación útil: AUC = probabilidad de que el modelo asigne mayor score a un positivo random que a un negativo random. **Es independiente del umbral**.

### Curva Precision-Recall (PR)

> ⚠️ **Con clases muy desbalanceadas (fraude), la ROC engaña.** Como hay muchísimos negativos, el FPR se mantiene bajo aunque el modelo tenga muchos FPs en términos absolutos. La PR curve (precision en función de recall, variando $\tau$) es **más informativa** en estos casos. **AUC-PR** es la métrica análoga a AUC-ROC pero sobre la PR curve.

### Estrategias para elegir el umbral

| Criterio | Cómo se elige | Cuándo usar |
|---|---|---|
| **F1-óptimo** | $\tau^* = \arg\max_\tau F_1(\tau)$ | Default razonable cuando precision y recall importan parecido. |
| **Youden's J** | $\tau^* = \arg\max_\tau (\mathrm{TPR} - \mathrm{FPR})$ | Punto más alejado de la diagonal en la ROC. |
| **Recall mínimo** | menor $\tau$ tal que $\mathrm{recall} \geq r_{\min}$ | "Quiero detectar al menos el 90 % de los fraudes". |
| **Precision mínima** | menor $\tau$ tal que $\mathrm{precision} \geq p_{\min}$ | "Cuando marque fraude, que lo sea el 95 % de las veces". |
| **Costo del negocio** | $\tau^* = \arg\min_\tau (c_{FN}\cdot FN + c_{FP}\cdot FP)$ | El correcto, si hay costos explícitos. |

> **Recomendación operativa para Ej1.** Como el enunciado no da costos explícitos, conviene presentar al cliente **2-3 umbrales con distintos perfiles** (F1-óptimo, recall alto, precision alta) y dejar que decida según su modelo de negocio.

---

## Overfitting y Underfitting

### Diagnóstico

```
                  ¿Error alto en train?
                  /                  \
                SI                    NO
                |                      |
          UNDERFITTING       ¿Error alto en test?
                              /              \
                            SI                NO
                            |                  |
                       OVERFITTING       BUEN MODELO
```

| Diagnóstico | Síntomas |
|---|---|
| **Underfitting** | Loss alta en train **y** test. Modelo sin capacidad suficiente, o no se entrenó bastante, o $\eta$ está mal. |
| **Overfitting** | Loss baja en train, **alta** en test. Gap creciente entre las dos curvas. |
| **Buen modelo** | Loss baja en ambos, las curvas convergen a un valor cercano. |

### Curvas de aprendizaje

El gráfico de las **dos curvas (train + val por época)** es el diagnóstico más importante de un experimento. Sin él no se puede saber qué pasó. Casos canónicos:

- **Buen modelo:** train y val suben juntas hacia un valor alto y se estabilizan.
- **Overfitting:** train sigue subiendo, val se estanca o **baja** después de un máximo. El gap crece con las épocas.

> Conviene plotear esta curva en **cada experimento** de Ej2/Ej3.

### Causas del overfitting

1. **Conjunto de datos no balanceado:** el modelo aprende la clase mayoritaria y "memoriza" la minoritaria.
2. **Pocos registros:** capacidad del modelo > información en los datos. La red aprende el ruido.
3. **Datos con mucho ruido:** la red no puede distinguir señal de ruido y ajusta ambos.
4. **Modelo demasiado grande** para el problema (muchos parámetros vs muestras).
5. **Entrenar demasiadas épocas** sin early stopping.
6. **Features con leakage:** información del target metida en las entradas. Conviene auditar las columnas para Ej1.

---

## Cómo combatir overfitting

### Más datos

Lo más efectivo casi siempre. Es exactamente lo que ofrece Ej3 al introducir `more_data_digits.csv`. La pregunta (c) del Ej3 pide reconocer este factor explícitamente.

### Reducir la capacidad del modelo

Bajar la cantidad de neuronas y/o capas. Si Ej2 quedó con `[n0, 128, 64, 10]` y overfittea, probar `[n0, 32, 10]`.

### Regularización L2 (weight decay)

Se agrega al loss un término que penaliza pesos grandes:

$$
E_{\text{reg}}(w) = E(w) + \frac{\lambda}{2} \sum_i w_i^2
$$

El gradiente queda $\nabla E_{\text{reg}} = \nabla E + \lambda w$. En la implementación, el update lleva un término extra proporcional a $-\lambda w$ (con la convención de signo que se elija; ver advertencia en `perceptron_multicapa.md` sobre signos).

- $\lambda = 0$ $\to$ sin regularización.
- $\lambda$ típica: $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$.
- $\lambda$ muy grande $\to$ underfitting (los pesos se aplastan a cero).
- **Convención:** no regularizar los biases.

### Regularización L1

$$
E_{\text{reg}}(w) = E(w) + \lambda \sum_i |w_i|
$$

Promueve **sparsity** (muchos pesos exactamente en 0). Útil cuando se busca selección de features. Para el TP, L2 es la opción natural.

### Early stopping

**Idea:** parar el entrenamiento en el momento donde la val_loss es mejor, no donde la train_loss es mejor.

```
best_val_loss = inf
best_weights = None
patience = 10                 # epocas sin mejora antes de parar
strikes = 0

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = evaluate(val_set)

    if val_loss < best_val_loss - tol:
        best_val_loss = val_loss
        best_weights = copy(W)
        strikes = 0
    else:
        strikes = strikes + 1
        if strikes >= patience:
            break

# Al final, usar best_weights, NO los ultimos
```

### Dropout

Durante training, "apaga" una fracción $p$ de neuronas aleatorias en cada paso (las pone en 0). Durante inference, todas activas (escaladas por $1-p$). Reduce co-adaptación.

> No está en la PPT y agrega complejidad. Solo recurrir si las técnicas anteriores no alcanzan.

### Resumen — palancas anti-overfitting

| Palanca | Costo de implementación | Impacto típico | Cuándo |
|---|---|---|---|
| Early stopping | Bajo | Medio-alto | **Siempre.** Default. |
| Más datos | Externo (no depende del código) | Alto | Ej3 lo da gratis. |
| L2 weight decay | Bajo | Medio | Cuando el gap train/val es grande. |
| Reducir arquitectura | Bajo | Medio | Si el modelo es claramente sobredimensionado. |
| Dropout | Medio | Medio | Solo si lo anterior no alcanza. |

---

## Procedimiento experimental

Para cada experimento del TP:

1. Entrenar con el conjunto de entrenamiento, calculando $w$.
2. Predecir sobre el conjunto de validación (y eventualmente test) con esos $w$.
3. Calcular las métricas en **ambos** conjuntos (train y val/test).
4. **Repetir** para distintos valores de épocas: 1, 10, 20, ..., 300 (o intervalos según convenga).

Esto da las curvas de aprendizaje por época que permiten diagnosticar over/underfitting. En Ej2/Ej3 conviene loguear cada 1 o cada 5 épocas para tener buena resolución.

---

## Normalización de datos

### Por qué normalizar

Si una feature está en escala $[0, 3500]$ y otra en $[0, 1]$, los pesos asociados a la primera dominan y la red satura (la sigmoidea entra en su zona plana, con derivada cercana a cero). Después de estandarizar, todas las features tienen escala comparable y el optimizador se comporta de manera mucho más estable.

### Las tres formas

**Min-max scaling** — al rango $[a, b]$:

$$
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}} (b - a) + a
$$

Caso típico $[0, 1]$. Útil cuando se conoce el rango y se quiere preservarlo. **Sensible a outliers**.

**Estandarización (Z-score)** — media 0, desvío 1:

$$
\tilde{X}_i = \frac{X_i - \bar{X}_i}{s_i}
$$

Default razonable, especialmente con tanh y logística.

**Unit length scaling** — dividir cada vector por su norma 2:

$$
X' = \frac{X}{\|X\|}
$$

Lleva todo el dataset a la esfera unitaria. Más usado en NLP o cuando importa la dirección y no la magnitud.

### ⚠️ Regla crítica: ajustar con train, aplicar a val/test

El scaler (los $X_{\min}, X_{\max}$, $\bar{X}, s$) se calcula **solo con el conjunto de entrenamiento**, y los mismos parámetros se aplican a val y test. Si se calcula con todo el dataset, hay **data leakage**: información del test se filtra al training.

```
# CORRECTO
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_n = (X_train - mu) / sigma
X_val_n   = (X_val   - mu) / sigma     # mismos mu, sigma
X_test_n  = (X_test  - mu) / sigma     # mismos mu, sigma

# MAL — leakage
mu, sigma = X.mean(axis=0), X.std(axis=0)   # incluye val/test!
```

---

## Conexión con los ejercicios del TP

### Ej1 — fraude (`transactions.csv`)

| Pregunta del enunciado | Aporte de este capítulo |
|---|---|
| *"¿qué métricas seleccionaron y por qué?"* | Datos desbalanceados: NO accuracy sola. Reportar precision, recall, F1, AUC-ROC, **AUC-PR**. |
| *"¿qué estrategia para manipular el dataset?"* | Split estratificado train/val. K-fold CV si el dataset es chico. Ajustar scaler solo con train. |
| *"¿cómo se elige el mejor conjunto de entrenamiento?"* | Por validación: el mejor modelo es el que minimiza la métrica elegida sobre val (no sobre train, eso sería overfitting al training). |
| *"recomendación de umbral"* | Calcular PR curve y presentar 2-3 umbrales con perfiles distintos (F1-óptimo, recall alto, precision alta). |

### Ej2 — dígitos baseline (`digits.csv` + `digits_test.csv`)

| Pregunta | Aporte |
|---|---|
| *"¿cómo evalúo el desempeño?"* | Multiclase: accuracy global + matriz de confusión 10×10 + macro F1 si hay desbalance. |
| Variantes a barrer | Para cada combinación de hiperparámetros, plotear curvas train/val. Identificar overfitting/underfitting. |
| Selección del mejor modelo | Por accuracy (o macro F1) sobre val, **no** sobre `digits_test.csv`. |

### Ej3 — dígitos hasta ≥98 % (`more_data_digits.csv`)

| Pregunta del enunciado | Aporte |
|---|---|
| *"¿cuál es el mejor resultado?"* | Reportar accuracy sobre `digits_test.csv` **una sola vez** al final, y la matriz de confusión. |
| *"¿qué técnicas utilizaron?"* | Combinar palancas anti-overfitting (early stopping + L2 + arquitectura adecuada) con palancas de optimización (Adam, Xavier, schedules). Reportar el delta de cada una por separado. |
| *"¿qué factores externos influyeron?"* | **El dataset creció** con `more_data_digits.csv`. Esa es la respuesta principal. |

> **Disciplina experimental para Ej3:** cambiar **una cosa a la vez** y medir el delta. Si se cambian 5 cosas al mismo tiempo y el accuracy sube de 92 % a 98 %, no se sabe cuál de las 5 fue la responsable y el informe se queda sin contenido.
