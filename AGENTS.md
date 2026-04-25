# SIA TP3 - Perceptrones

## Contexto

Proyecto para la materia **Sistemas de Inteligencia Artificial (SIA)** del ITBA.

El objetivo es implementar, entrenar y evaluar **perceptrones** (simples y multicapa) resolviendo 3 ejercicios prácticos. Cada ejercicio implica:

- Cuenta con su propio dataset para entrenamiento y testing.
- Se realizan experimentos variando hiperparámetros.
- Los hiperparámetros se definen en un `config.yaml` propio del ejercicio.
- Se analizan comportamientos, convergencia, errores, decision boundaries, etc.

## Tecnología

- **Lenguaje:** Python 3.10+
- **Manejo de dependencias:** `requirements.txt` en raíz
- **Librerías:**
  - `numpy` — operaciones matriciales (requisito explícito del enunciado).
  - `pandas` — carga y exploración de los CSVs (`transactions.csv`, `digits.csv`, `digits_test.csv`, `more_data_digits.csv`).
  - `matplotlib` — gráficos (curvas de aprendizaje, ROC/PR, matrices de confusión).
- **Configuración por ejercicio:** `config.yaml` dentro de cada carpeta `ejN/`
- **Sin frameworks de deep learning** (PyTorch, TF): el TP exige implementar perceptrones y backpropagation desde cero.

## Estructura del repositorio

```
sia-tp3/
├── AGENTS.md           # Este archivo con contexto para agentes
├── README.md           # Documentación pública del proyecto
├── requirements.txt    # Dependencias Python globales
│
├── context/            # Material teórico de apoyo (notas sobre las clases)
│   ├── perceptron_simple.md
│   ├── perceptron_lineal_no_lineal.md
│   ├── perceptron_multicapa.md
│   ├── optimizacion.md
│   └── metricas_sobreajuste.md
│
├── shared/             # Código reutilizable entre ejercicios
│   ├── __init__.py
│   ├── perceptron.py       # Implementación base de perceptrón (escalón / lineal / no lineal)
│   ├── mlp.py              # Multi-Layer Perceptron (forward + backprop)
│   ├── activations.py      # Funciones de activación y derivadas (escalón, identidad, tanh, logística)
│   ├── initializers.py     # Inicialización de pesos (random pequeño, Xavier, He)
│   ├── optimizers.py       # GD, SGD, Momentum, Adam (interfaz común tipo `step(params, grads)`)
│   ├── losses.py           # MSE (default); cross-entropy opcional para Ej2/3
│   ├── metrics.py          # Accuracy, precision, recall, F1, AUC-ROC, AUC-PR, matriz de confusión
│   ├── preprocessing.py    # Scalers (z-score, min-max), splits estratificados, one-hot
│   ├── regularization.py   # L2 / weight decay, early stopping
│   ├── utils.py            # Helpers generales (loaders CSV, logging, plots)
│   └── config_loader.py    # Parser genérico de config.yaml
│
├── ej1/                # Ejercicio 1
│   ├── README.md
│   ├── config.yaml
│   ├── main.py
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   └── results/
│       └── (plots, logs, etc.)
│
├── ej2/                # Ejercicio 2
│   ├── README.md
│   ├── config.yaml
│   ├── main.py
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   └── results/
│
└── ej3/                # Ejercicio 3
    ├── README.md
    ├── config.yaml
    ├── main.py
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    └── results/
```

## Convenciones

- Cada ejercicio es **self-contained**: puede correrse de forma independiente ejecutando `python main.py` dentro de su carpeta.
- Las dependencias compartidas se importan como `from shared.perceptron import ...`.
- Los `config.yaml` deben seguir un esquema común mínimo (ver `shared/config_loader.py`) para facilitar la carga y permitir extensión por ejercicio.
- Resultados (gráficos, logs, CSVs de métricas) se guardan en `ejN/results/` y se ignoran en `.gitignore` si son muy pesados.

## Archivos de configuración (`config.yaml`)

Esquema sugerido por ejercicio. No todos los campos aplican a todos los ejercicios; los que no apliquen se omiten o se setean a `null`.

```yaml
model:
  type: "simple_step" | "simple_linear" | "simple_nonlinear" | "mlp"
  architecture: [n_in, h1, h2, ..., n_out]   # solo para mlp
  activation: "step" | "identity" | "tanh" | "logistic" | "sigmoid" | "relu" 
  beta: 1.0                       # pendiente sigmoidea (no lineal / mlp con tanh o logística)
  initializer: "random_normal" | "xavier" | "he"
  init_scale: 0.1                 # σ para random_normal

training:
  optimizer: "gd" | "sgd" | "momentum" | "adam"
  learning_rate: 0.01
  batch_size: 32                  # 0 = full-batch, 1 = online
  shuffle: true
  epochs: 1000
  loss: "mse"                     # opcional: "cross_entropy" para Ej2/3
  momentum: 0.9                   # solo si optimizer == momentum
  adam_betas: [0.9, 0.999]        # solo si optimizer == adam
  adam_eps: 1.0e-8                # solo si optimizer == adam
  weight_decay: 0.0               # λ de regularización L2
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_loss"
  stop_criteria:
    epsilon: 0.01                 # corte por error (loss < epsilon)
    grad_norm: null               # corte por norma del gradiente

data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  preprocess:
    feature_scaler: "z-score" | "min-max" | "none"
    label_scaler: "none" | "min-max"
  split:
    val_frac: 0.2
    stratify: true                # estratificar por la etiqueta (Ej1)
    seed: 42

experiment:
  name: "ej1_baseline"
  seed: 42
  save_plots: true
  log_every: 1                    # cada cuántas épocas loguear loss/métricas
  log_level: "INFO"
  save_model: true                # guardar pesos + estado del optimizador + scaler
```

---

## Consigna Completa del TP

### Ejercicios de Validación (NO se presentan)

Estos ejercicios sirven **únicamente para validar** que las implementaciones de los algoritmos funcionan correctamente. Los conjuntos de datos son pequeños y simples, lo que permite verificar fácilmente si hay errores.

**Nota:** Lo que dice "validación" en el enunciado no se presenta; es solo para que, una vez implementados los algoritmos, verifiquemos que funcionan bien.

#### Perceptrón Simple Escalón
- Función lógica **AND** con entradas: `x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}}`
- Salida esperada: `y = {-1, -1, -1, 1}`

#### Perceptrón Simple Lineal
- Tomar un conjunto de muestras (aprox. 50) de una función lineal (por ejemplo, `y = x`) y tratar de que el perceptrón ajuste el conjunto de datos.

#### Perceptrón Simple No Lineal
- Tomar un conjunto de muestras (aprox. 50) de una función no lineal (por ejemplo, `y = tanh(x)`) y tratar de que el perceptrón ajuste el conjunto de datos.

#### Perceptrón Multicapa
- Función lógica **XOR** con entradas: `x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}}`
- Salida esperada: `y = {1, 1, -1, -1}`
- Se recomienda realizar los cálculos a mano para las arquitecturas `[2, 2, 1]` y `[2, 3, 2, 1]`.
- Comparar cómo se comporta el multicapa con el perceptrón simple escalón para resolver este problema.

---

### Ejercicio 1: Prevención de Fraude (Knowledge Distillation con Perceptrón Simple)

**Contexto:** La empresa CompanyX usa un modelo llamado **BigModel** para estimar la probabilidad de fraude en transacciones online. Este modelo es muy caro de usar para inferencia. Se nos pide desarrollar un **TinyModel** (perceptrón simple) que replique el comportamiento de BigModel pero sea mucho más barato de utilizar y almacenar.

**Objetivo:** Estimar la probabilidad de que una transacción online sea fraudulenta, siendo `0` equivalente a 0% y `1` a 100%. El dataset provisto es `transactions.csv`.

**Parte 1 — Análisis de Aprendizaje:**
- Comparar el **perceptrón simple lineal** vs. el **perceptrón simple no lineal**.
- Responder:
  - a) ¿Se observa underfitting?
  - b) ¿Se observa saturación de las capacidades?
  - c) ¿Cuál seleccionarían para generalización, de acuerdo al potencial de aprendizaje de cada uno?
- **Nota:** El estudio se realiza utilizando **todas las muestras** del conjunto de datos (se pueden alimentar todos los datos y observar cómo queda la función de pérdida).

**Parte 2 — Generalización:**
- Una vez seleccionado uno de los perceptrones, realizar un estudio de generalización.
- Responder:
  - a) ¿Qué métricas de evaluación seleccionaron y por qué?
  - b) ¿Qué estrategia utilizaron para manipular el conjunto de datos durante la generalización? ¿Cómo creen que se elige el mejor conjunto de entrenamiento?
  - c) ¿Cuál es el mejor modelo que obtuvieron para presentar al cliente? CompanyX solicita, además de tener un modelo más pequeño, una **recomendación del umbral de detección de fraude**.

**Importante:**
- Es de suma importancia **explorar el conjunto de datos antes de trabajar con él**: revisar la documentación de cada columna, rangos de valores, composición del dataset, limpieza de datos.
- Prestar atención a si hay datos cuyos valores necesitamos restringir; puede dar errores si no.

**Opcionales (Ejercicio 1):**
- Utilizar otra función de activación en el perceptrón simple no lineal (ej. ReLU) y ver su efecto en las conclusiones.
- Feature engineering: ¿qué otros features se pueden construir? ¿cuáles descartar?
- Investigar el concepto de **calibración**: ¿por qué sería apropiado realizar un análisis y ajuste de calibración en este caso?

---

### Ejercicio 2: Clasificación de Dígitos Escritos a Mano (Perceptrón Multicapa)

**Contexto:** CompanyX quiere desarrollar un modelo de detección automática de dígitos escritos a mano para acelerar procesos de distribución.

**Objetivo:** Clasificar dígitos entre el **0 y el 9** usando un **perceptrón multicapa**.

**Datasets:**
- `digits.csv`: para la etapa de aprendizaje y ajuste de hiperparámetros.
- `digits_test.csv`: para estudiar la generalización. Tratarlo como si estuviera en producción (datos que "no deberíamos haber visto").
- **Nota:** Explorar los datasets primero; se puede apoyar en la documentación que viene en el archivo comprimido y que explica qué representa cada columna.

**Preguntas a responder:**
- (a) ¿Cómo evalúo el desempeño de mi sistema?
- (b) ¿Qué variantes realizo para encontrar la solución?

**Como mínimo deberán analizar:**
- Variantes de **tasa de aprendizaje**
- Variantes de **arquitectura** (cantidad de capas, nodos por capa) — **probar sí o sí**
- Variantes de **mecanismos de optimización**

**Recomendaciones:**
- Usar **10 neuronas en la capa de salida**, una para cada dígito, para entrenar e identificar mejor en qué dígitos hay problemas.
- Para la arquitectura y mecanismos de optimización, tratar de hacerlo lo más **genérico posible** (son hiperparámetros; debería ser lo más genérico posible para no complicarse en futuros TPs).
- Dividir el dataset en **3 partes**: `TRAIN` (entrenar), `VAL` (ajustar hiperparámetros), `TEST` (producción, datos no vistos).

---

### Ejercicio 3: Mejora de Accuracy con Nuevos Datos (Digits)

**Contexto:** Los resultados de la primera iteración con CompanyX no fueron satisfactorios. La empresa solicita alcanzar una **accuracy mayor o igual al 98%**.

**Dataset:** `more_data_digits.csv` (dataset más grande).

**Preguntas a responder:**
- (a) ¿Cuál es el mejor resultado que pudieron obtener con este nuevo conjunto de datos?
- (b) ¿Qué técnicas utilizaron para mejorar el rendimiento con respecto al caso anterior?
- (c) Además de sus propias técnicas, ¿existen otros factores que influyeron en el cambio de rendimiento entre este ejercicio y el anterior?

**Nota:** Si no se llega al 98% no pasa nada; la idea es tratar de llegar o superarlo.
- El dataset es gigante; se puede **bajar la resolución** si hay problemas de performance.

**Opcionales (Ejercicios 2 y 3):**
- **Robustez al ruido:** ¿Es posible afirmar que el modelo es robusto? Probar con la mejor solución sobre `digits_test.csv` agregando ruido gaussiano a las muestras.
- **Interpretabilidad:** ¿Cómo distingue la red entre los distintos dígitos? Explorar métodos de atribución.

---

## Hiperparámetros a experimentar

No es necesario probar todas las combinaciones; sí incluir las marcadas como obligatorias del enunciado. Valores sugeridos a partir del material teórico:

| Hiperparámetro | Valores sugeridos | Aplica a | Obligatorio |
|---|---|---|---|
| Cantidad de épocas | 100, 500, 1000, 5000 | todos | sí |
| Tasa de aprendizaje (η) | 1e-1, 1e-2, 1e-3, 1e-4 | todos | sí |
| Arquitectura (capas, neuronas) | varía por ejercicio (ver §"Decisiones técnicas") | mlp | **sí** |
| Inicialización de pesos | random N(0, σ²) con σ ∈ {0.01, 0.1}; Xavier; He | mlp / no lineal | sí |
| Función de activación | escalón, identidad, tanh, logística | según ejercicio | sí |
| Beta (β, pendiente sigmoidea) | 0.5, 1, 2 | sigmoideas | recomendado |
| Función de pérdida | MSE; cross-entropy (opcional Ej2/3) | todos | MSE sí |
| Modo de entrenamiento | online (1), minibatch (32, 64), batch | todos | sí |
| Optimizador | GD, SGD, Momentum, Adam | todos | **GD y Adam sí**, Momentum recomendado |
| Regularización L2 (λ) | 0, 1e-4, 1e-3 | mlp (sobre todo Ej3) | si overfittea |

> **Adam (referencia):** β₁=0.9, β₂=0.999, ε=1e-8, α=1e-3 como punto de partida (ver §"Decisiones técnicas" para detalles, incluida la corrección de bias).

---

## Recomendaciones del enunciado

- **Operaciones matriciales** para mejorar la performance.
- **Reportar progreso** cuando se ponen a correr los modelos (log por época).
- **Configuración extensible** (`config.yaml`) que puedan almacenar para revisar parámetros.
- **Guardar y levantar modelos:** permitir seguir entrenando un modelo sin arrancar de cero. Hacerlo en conjunto con la configuración extensible.
- **División de responsabilidades:** almacenar información relacionada a cada experimento (loss por época, hiperparámetros, velocidad de ejecución, etc.) y realizar el análisis (gráficos, tablas, etc.) por separado. Similar al enfoque de simulación de sistemas.
- **NO arrancar ejercicios opcionales** antes de tener implementados los puntos necesarios del trabajo práctico.

## Decisiones de diseño pendientes (para discutir)

1. ¿Usar NumPy puro o incluir dependencias tipo PyTorch solo para cálculos matriciales?
2. ¿Formato de datasets: CSV con headers o numpy arrays serializados?
3. ¿Cómo estructurar la salida de experimentos (carpetas por timestamp vs. por nombre)?
4. ¿Qué métricas se computan por defecto (accuracy, MSE, RMSE, matriz de confusión)?
