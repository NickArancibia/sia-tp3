# SIA TP3 - Perceptrones

Trabajo practico de Sistemas de Inteligencia Artificial (ITBA) sobre perceptron simple y multicapa.

El repositorio esta organizado en tres ejercicios:

- `ej1`: distillation de un modelo de fraude con perceptron simple.
- `ej2`: clasificacion de digitos escritos a mano con MLP.
- `ej3`: mejora del clasificador de digitos con mas datos.

## Requisitos

- `python3`
- `pip`
- dependencias de `requirements.txt`

Instalacion minima:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura

- `shared/`: implementaciones reutilizables de activaciones, perceptrones, MLP, metricas, optimizadores y preprocesamiento.
- `ej1/`: ejercicio de fraude.
- `ej2/`: ejercicio de digitos con MLP.
- `ej3/`: mejora del clasificador con mas datos.
- `presentacion/`: slides en Beamer.
- `context/`: apuntes de soporte teorico.

## Como correr

### EJ1

Corre Parte 1 y Parte 2:

```bash
python3 ej1/main.py
```

Si queres correrlas por separado:

```bash
python3 ej1/main_part1.py
python3 ej1/main_part2.py
```

### EJ2

```bash
python3 ej2/main.py
```

### EJ3

```bash
python3 ej3/main.py
```

### Batch de experimentos

```bash
bash run_all.sh
```

`run_all.sh` orquesta los experimentos de `ej2` y `ej3` usados para generar graficos de resultados.

## EJ1 - Resumen del flujo

### Parte 1

La Parte 1 compara tres variantes sobre todo el dataset:

- `identity`
- `logistic`
- `ReLU`

Se generan:

- una curva de loss conjunta
- un grafico `internal_function` por activacion
- un grafico `target_vs_prediction` por activacion

### Parte 2

La Parte 2 estudia generalizacion, seleccion de estrategia de datos y recomendacion de umbral.

La estrategia seleccionada hoy es `S3` y el umbral final del modelo seleccionado es `0.75`.

## EJ1 - Como se dividen los datos

En `ej1`, `VAL` no sale del `TEST`.

El flujo correcto es:

1. Primero se separa `TEST = 15%` del dataset total.
2. El `85%` restante queda como bloque `TRAIN+VAL`.
3. Dentro de ese `85%` se arma `VAL` segun la estrategia elegida.

Esto evita contaminar el conjunto de test con decisiones de entrenamiento, early stopping o seleccion de umbral.

### Estrategia S1 - Random-Split

- `TEST = 15%` del total
- `VAL = 15%` del bloque `TRAIN+VAL`
- `TRAIN = 85% - VAL`

En porcentaje sobre el dataset completo:

- `TRAIN = 72.25%`
- `VAL = 12.75%`
- `TEST = 15%`

### Estrategia S3 - 5-Fold Cross-Validation

Primero se deja `TEST = 15%` afuera. Luego el `85%` restante se divide en `5` folds.

En cada fold:

- `VAL = 1/5` del `85%`
- `TRAIN = 4/5` del `85%`
- `TEST` sigue siendo siempre el mismo `15%` separado

En porcentaje sobre el dataset completo, para cada fold:

- `TRAIN = 68%`
- `VAL = 17%`
- `TEST = 15%`

Con `7500` muestras totales:

- `TEST = 1125`
- `TRAIN+VAL = 6375`
- por fold en `S3`: `TRAIN = 5100`, `VAL = 1275`

Al final del proceso, el modelo seleccionado se reentrena con todo el bloque `TRAIN+VAL` y se evalua una sola vez sobre `TEST`.

## EJ2 - Clasificacion de digitos

`ej2` resuelve clasificacion multiclase de digitos con un MLP implementado desde cero.

### Datos

- `ej2/data/digits.csv`: dataset base para entrenamiento y validacion.
- `ej2/data/digits_test.csv`: test separado, tratado como conjunto no visto.

El flujo por default es:

1. tomar `digits.csv`
2. separar `TRAIN` y `VAL` con `val_frac = 0.2`
3. entrenar sobre `TRAIN`
4. usar `VAL` para early stopping y seleccion de hiperparametros
5. reportar resultado final sobre `digits_test.csv`

### Modelo y configuracion

- salida one-hot de `10` clases
- arquitectura configurable desde `ej2/config.yaml`
- activaciones, inicializacion, optimizador, batch size y learning rate configurables
- early stopping sobre validacion

### Experimentos de `part2`

En `ej2/part2/` hay scripts separados para los barridos principales:

- `learning_rate/`: barrido de tasa de aprendizaje
- `architecture/`: comparacion de arquitecturas
- `optimizer/`: comparacion de optimizadores
- `batch_lr/`: heatmap `batch size x learning rate`
- `optimizer_lr/`: comparacion `optimizer x learning rate`
- `activation/`: comparacion de activaciones ocultas
- `two_layer_heatmap/` y `two_layer_heatmap_relu/`: analisis mas fino de arquitecturas de dos capas
- `selected_model/` y `selected_model_512_1layer/`: evaluacion del modelo final en multiples seeds
- `noise_robustness/`: experimento opcional de robustez al ruido

### Comandos utiles

Baseline:

```bash
python3 ej2/main.py
```

Ejemplo de experimento puntual:

```bash
python3 ej2/part2/learning_rate/run.py
python3 ej2/part2/learning_rate/plot.py
```

## EJ3 - Mejora con mas datos

`ej3` reutiliza el problema de digitos, pero cambia el foco: medir cuanto mejora el modelo cuando se entrena con mas datos y empujar la accuracy hacia el objetivo del enunciado.

### Datos

- `ej3/data/merged_augmented.csv`: dataset de entrenamiento usado por el entry point principal
- `ej2/data/digits_test.csv`: test final compartido con `ej2`

Esto permite comparar cambios de rendimiento manteniendo fijo el conjunto de test.

El flujo por default es:

1. tomar el dataset de entrenamiento configurado en `ej3/config.yaml`
2. separar `TRAIN` y `VAL` con `val_frac = 0.2`
3. entrenar el modelo con regularizacion y early stopping
4. evaluar sobre el mismo `digits_test.csv` de `ej2`

### Modelo y configuracion

- MLP configurable desde `ej3/config.yaml`
- configuracion actual orientada a `ReLU + Adam`
- soporte para `weight_decay`
- soporte para `online_augmentation`

### Experimentos de `part2`

En `ej3/part2/` hay scripts para comparar estrategias distintas:

- `data_comparison/`: compara `digits.csv`, `more_digits.csv` y `merged_digits.csv` sobre el mismo test
- `selected_model/`: corrida del modelo elegido para buscar el mejor resultado final
- `warmstart_digits_to_more/`: experimento opcional de pretrain en `digits` y finetune en `more_digits`

### Comandos utiles

Baseline:

```bash
python3 ej3/main.py
```

Comparacion entre datasets:

```bash
python3 ej3/part2/data_comparison/run.py
python3 ej3/part2/data_comparison/plot.py
```

## Resultados

Los artefactos generados se guardan en `ejN/results/`.

Ejemplos:

- `ej1/results/part1/`
- `ej1/results/part2/selected_model/`
- `ej2/results/`
- `ej3/results/`
