# Contexto teórico: Optimización No Lineal Sin Restricciones

## Idea principal

Entrenar un perceptrón (simple o multicapa) es, en el fondo, un problema de **optimización no lineal sin restricciones**: queremos encontrar los pesos $w$ que minimizan una función de error $E(w)$.

$$
\min_{w \in \mathbb{R}^n} E(w)
$$

Donde:

- $w = (w_1, \dots, w_n)$ son los **pesos sinápticos** (variables libres del problema).
- $E(w)$ es la **función de error / costo**. Para aprendizaje supervisado sobre $p$ patrones $(\xi^\mu, \zeta^\mu)$ y un modelo $g(\xi, w)$, una formulación clásica es MSE:

$$
E(w) = \frac{1}{2} \sum_{\mu=1}^{p} \left( \zeta^\mu - g(\xi^\mu, w) \right)^2
$$

> Otras funciones de error (cross-entropy, log-loss, etc.) pueden ser más apropiadas según el problema. La elección de $E$ define la geometría del problema y, por lo tanto, qué métodos funcionan mejor.

---

## Condiciones de optimalidad

Asumiendo $E$ diferenciable con primera y segunda derivadas continuas:

- **Necesaria de 1er orden:** si $w^*$ es mínimo local, entonces $\nabla E(w^*) = 0$.
- **Necesaria de 2do orden:** además, el Hessiano $H_E(w^*)$ es semidefinido positivo.
- **Suficiente de 1er orden:** si $\nabla E(w^*) = 0$ y $H_E(w^*)$ es definido positivo, $w^*$ es mínimo local estricto.

> **Matriz definida positiva:** $x^T A x > 0$ para todo $x \neq 0$ (equivalentemente, todos sus autovalores son positivos). Semidefinida positiva: $\geq 0$.

### Convexidad — por qué importa

- $E$ **convexa** $\iff$ Hessiano semidefinido positivo en todo el dominio. **Cualquier mínimo local es global.**
- $E$ **cóncava** $\iff$ Hessiano semidefinido negativo.

Esto solo está garantizado en el **perceptrón lineal con MSE** (problema cuadrático convexo). En perceptrones no lineales y multicapa el problema es **no convexo**: hay múltiples mínimos locales, mesetas, puntos silla, y solo podemos aspirar a buenos mínimos locales. Esta es una de las razones de fondo por las que existen tantos métodos de optimización: en no convexo, la trayectoria importa.

---

## Procedimiento iterativo general

Todos los métodos que se ven en la materia comparten la misma estructura:

$$
w_{k+1} = w_k + \alpha_k \, d_k
$$

Con:

- $w_k$: punto actual (vector de pesos en la iteración $k$).
- $d_k$: **dirección de descenso**. Debe cumplir $d_k^T \nabla E(w_k) < 0$ para que la función decrezca localmente.
- $\alpha_k$: **tasa de aprendizaje** o longitud de paso ($\eta$ en notación de perceptrones).

> **Recordatorio clave:** el gradiente $\nabla E$ apunta en la dirección de **máximo crecimiento**. Cualquier dirección con producto interno negativo con el gradiente sirve como dirección de descenso.

Lo que diferencia un método de otro es **cómo elige $d_k$ y $\alpha_k$**.

---

## Tasa de aprendizaje

Trade-off central:

- $\alpha$ muy grande $\to$ el método puede no converger (oscila o diverge).
- $\alpha$ muy chico $\to$ converge muy lento.

Estrategias habituales:

| Estrategia | Descripción |
|---|---|
| **Fija** | Lo más simple. Buen punto de partida. |
| **Búsqueda unidimensional (line search)** | Minimiza $g(\alpha) = E(w_k + \alpha d_k)$ en cada paso. Más caro pero más robusto. |
| **Schedule (decay)** | $\alpha$ decae con el tiempo (p. ej., $\alpha_t = \alpha_0 / (1 + \gamma t)$, decay exponencial). |
| **Adaptativa por coordenada** | Cada peso tiene su propio paso efectivo. Lo hacen ADAGrad y Adam. |

---

## Métodos clásicos (1er y 2do orden)

### Gradiente descendente (GD)

$$
d_k = -\nabla E(w_k)
$$

- Simple, no requiere segundas derivadas.
- Convergencia lenta. Avanza en zig-zag en valles alargados.

### GD con Momentum

$$
w_{k+1} = w_k - \alpha_k \nabla E(w_k) - \beta \, \alpha_{k-1} \nabla E(w_{k-1})
$$

- $\beta \in (0, 1)$ es la **inercia** del sistema. Suaviza el zig-zag de GD.
- A mayor $\beta$, más difícil cambiar de dirección.
- Valor típico: $\beta = 0.9$.
- **Recomendado** como primera mejora sobre GD: costo computacional casi idéntico, ganancia notable.

### Método de Newton

$$
d_k = -H^{-1}(w_k) \nabla E(w_k)
$$

- Convergencia cuadrática cerca del óptimo.
- Requiere calcular **e invertir** el Hessiano: $O(n^3)$ por paso.
- **Inviable para redes neuronales** medianas o grandes.

### Quasi-Newton (BFGS / L-BFGS)

Aproximan $H^{-1}$ con una matriz $B_k$ definida positiva que se actualiza iterativamente. **L-BFGS** ("limited memory") es la variante que sí escala. En redes neuronales modernas se usa poco.

### Direcciones conjugadas

Un conjunto de direcciones $H$-conjugadas ($d_i^T H d_j = 0$ para $i \neq j$, con $H$ simétrica definida positiva) garantiza convergencia en $n$ pasos para una función cuadrática.

- **Gradientes conjugados** (Hestenes-Stiefel, 1952): necesita gradiente y Hessiano.
- **Powell** (1964): genera direcciones conjugadas **sin necesidad de derivadas**. Útil cuando el gradiente es caro o ruidoso.

> ⚠️ Para el TP, los métodos de Newton, Quasi-Newton y direcciones conjugadas son **contexto teórico**. No se implementan salvo que el equipo decida explorarlos como variante extra. Lo que sí se implementa es GD, GD con Momentum, y los métodos estocásticos de la siguiente sección.

---

## Métodos estocásticos

### ¿Por qué estocásticos?

La función de error de aprendizaje supervisado es una **suma sobre patrones**:

$$
E(w) = \frac{1}{p} \sum_{\mu=1}^{p} E^\mu(\zeta^\mu, \xi^\mu, w)
$$

Es un *estimador muestral* del error poblacional. Calcular el gradiente exacto requiere recorrer **todo el dataset** por paso. En problemas grandes (Ej2/Ej3 del TP, con miles de imágenes de dígitos) esto es muy costoso y motiva aproximarlo.

### SGD (Stochastic Gradient Descent)

Aproxima el gradiente con un **subconjunto aleatorio** $\mathcal{B}$ (minibatch) de tamaño $k \ll p$:

$$
w_{t+1} = w_t - \eta_t \, \frac{1}{|\mathcal{B}|} \sum_{\mu \in \mathcal{B}} \nabla_w E^\mu(\xi^\mu, w_t)
$$

Casos extremos:

- $|\mathcal{B}| = 1$: SGD puro (online). Es el modo en el que originalmente entrenaba **ADALINE** vía la regla de Widrow-Hoff / LMS.
- $|\mathcal{B}| = p$: equivale a GD batch (full-batch).
- $1 < |\mathcal{B}| < p$: **minibatch**. Lo más usado en la práctica.

> ⚠️ SGD **converge** pero **no necesariamente desciende monótonamente**. La curva de loss tendrá ruido, esto es esperado, no es un bug.

### ADAGrad (Duchi et al., 2011)

Adapta $\eta$ **por coordenada** según la acumulación histórica de gradientes:

$$
w^{t+1}_i = w^t_i - \frac{\eta_t}{\sqrt{G^t_{ii} + \epsilon}} \, \nabla_{w_i} E
$$

Con $G^t = \sum_{\tau=1}^{t} g_\tau g_\tau^T$ (en la práctica solo se mantiene la diagonal).

- ✅ Cada peso tiene su propio "paso efectivo". Útil con features de escalas muy diferentes o gradientes dispersos.
- ❌ El denominador crece monótonamente, así que el paso efectivo tiende a cero (problema de *stalling* en entrenamientos largos).

### Adam (Kingma & Ba, 2015) — *Adaptive Moment Estimation*

Combina momentum (1er momento del gradiente) con escalado por varianza (2do momento):

```
inicialización:  m_0 = 0,  v_0 = 0,  t = 0

mientras no converja:
    t   <- t + 1
    g_t <- gradiente de E en w_{t-1}
    m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t      # 1er momento (media móvil del gradiente)
    v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t^2    # 2do momento (media móvil del cuadrado)
    
    # Corrección de bias (importante en las primeras iteraciones)
    m_hat <- m_t / (1 - beta1^t)
    v_hat <- v_t / (1 - beta2^t)
    
    w_t <- w_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
```

Hiperparámetros sugeridos por los autores:

- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$
- $\alpha$ típica: $10^{-3}$

> ✅ **Adam es hoy el default razonable** para redes neuronales. Para Ej2/Ej3 es muy probable que sea el optimizador con el que más fácil se llegue al accuracy ≥ 98 % solicitado en Ej3.

> 📌 **Sobre la corrección de bias:** en las primeras iteraciones $m_t$ y $v_t$ están sesgados hacia cero (porque arrancan en cero). La corrección $\hat{m}_t = m_t / (1 - \beta_1^t)$ y $\hat{v}_t = v_t / (1 - \beta_2^t)$ compensa ese sesgo. Conviene incluirla en la implementación.

---

## Resumen comparativo

| Método | Información que necesita | Costo por paso | Cuándo usarlo |
|---|---|---|---|
| GD | Gradiente | Bajo (una pasada por dataset) | Baseline. Problemas chicos. |
| GD + Momentum | Gradiente | Bajo | Mejora barata sobre GD. |
| Newton | Gradiente + Hessiano | $O(n^3)$ | Inviable para redes. |
| Quasi-Newton (L-BFGS) | Gradiente | Medio | Problemas medianos no estocásticos. |
| SGD / Minibatch | Gradiente sobre subconjunto | Muy bajo | Datasets grandes. |
| AdaGrad | Gradiente | Bajo | Gradientes dispersos. |
| Adam | Gradiente | Bajo | **Default moderno** para redes neuronales. |

---

## Criterios de parada

Combinables entre sí:

- **Número máximo de épocas** (siempre, para evitar bucles infinitos).
- **Norma del gradiente:** $\|\nabla E\| < \epsilon$.
- **Cambio relativo en $E$** menor que un umbral durante $N$ épocas consecutivas.
- **Early stopping** sobre conjunto de validación (ver `metricas_sobreajuste.md`).

---

## Conexión con los ejercicios del TP

| Punto del TP | Aporte de esta capa |
|---|---|
| **Validación (escalón / lineal / no lineal / XOR)** | GD o SGD básico alcanzan. No hace falta Adam para problemas de juguete. |
| **Ej1 — fraude** | Comparar lineal vs no lineal con el **mismo** optimizador (GD o SGD). La elección del optimizador no debe ser lo que distinga underfitting de saturación. |
| **Ej2 — dígitos** | El enunciado pide explícitamente *"variantes de tasa de aprendizaje"* y *"variantes de mecanismos de optimización"*. Acá se barren GD vs SGD vs Momentum vs Adam. |
| **Ej3 — dígitos hasta 98 %** | Si Ej2 quedó con SGD/Momentum, pasar a **Adam** suele ser la palanca más barata para ganar puntos de accuracy. Combinar con `more_data_digits.csv` y, si hace falta, *learning rate schedules*. |
