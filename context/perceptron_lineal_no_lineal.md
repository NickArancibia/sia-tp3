# Contexto teórico: Perceptrón Lineal y No Lineal

## Motivación

Hasta ahora vimos el perceptrón simple escalón, que resuelve problemas de **clasificación binaria** separando datos con una recta o hiperplano. Ahora vemos dos extensiones:

1. **Perceptrón Lineal:** cambia la función de activación a la **identidad**, convirtiendo el problema en uno de **regresión lineal** (aproximar una recta a los datos).
2. **Perceptrón No Lineal:** cambia la función de activación por una función continua no lineal (sigmoidea, tangente hiperbólica, logística) para resolver problemas que **no son linealmente separables**.

---

## Perceptrón Lineal (ADALINE - Widrow-Hoff, 1960)

### Cambio principal

Se reemplaza la función de activación escalón $\theta$ por la **función identidad**:

$$
\theta(h) = h
$$

### Consecuencias

- La salida del perceptrón **ya no está confinada a ser binaria**: ahora toma valores en los **reales**.
- En vez de buscar una recta que clasifique dos grupos, buscamos un **hiperplano que ajuste lo mejor posible al conjunto de datos** (minimizar la distancia de los puntos a la recta).

### Función de error (costo)

Para medir el error usamos la **diferencia al cuadrado** (o media del cuadrado):

$$
E = \frac{1}{2} \sum_{\mu=1}^{p} (t^{\mu} - O^{\mu})^2
$$

La salida del perceptrón depende de los pesos sinápticos, por lo que el error es función de los pesos.

### Actualización de pesos (Gradient Descent)

Para minimizar el error, vamos **en contra del gradiente** (hacia donde decrece la función de error). Derivando la función de costo respecto a cada peso:

$$
\Delta w = \eta \cdot (t^{\mu} - O^{\mu}) \cdot \theta'(h) \cdot x^{\mu}
$$

Como $\theta(h) = h$, entonces $\theta'(h) = 1$. Por lo tanto, la regla de actualización se simplifica a:

$$
\Delta w = \eta \cdot (t^{\mu} - O^{\mu}) \cdot x^{\mu}
$$

> **Nota:** Este $\Delta w$ está hecho **para esta función de error específica**. Si cambia la función de error, hay que volver a derivar.

### Algoritmo

El algoritmo es **idéntico** al del perceptrón simple escalón, solo cambia la función de activación (de escalón a identidad) y, por consiguiente, la interpretación del problema:

```
Initialize weights w to small random values
Set learning rate eta

for a fixed number of epochs:
  for each training example mu in the dataset:
   1. Calculate the weighted sum:
     h^mu = w_0 + w_1 * x^mu_1 + ... + w_n * x^mu_n
   2. Compute activation (identity):
     O^mu = h^mu
   3. Update the weights:
      For each weight w_i (i = 0..n):
         w_i = w_i + eta * (t^mu - O^mu) * x^mu_i

  4. After iterating over ALL examples, calculate global error (MSE)
  5. Check for convergence
End
```

> **Tip de implementación:** Construirse un conjunto de datos que sean puntos pertenecientes a una recta y ajustarlos.

> El tipo de entrenamiento corresponde al formato **ONLINE**: se actualiza el peso con cada ejemplo a medida que se presenta.

---

## Perceptrón No Lineal

### Cambio principal

Se reemplaza la función de activación por una función continua **no lineal**:

- **Sigmoidea / Logística**
- **Tangente hiperbólica (tanh)**

### ¿Para qué sirve?

Resolvemos problemas donde **no es posible trazar una recta** para separar o aproximar los datos. La función escalón sirve para clasificación; la identidad sirve para regresión lineal; ahora necesitamos **adaptar el modelo para problemas no lineales**.

### Función de error

> **Lo bueno es que la función de error es la misma**, así que podemos usar la misma fórmula de error (diferencia al cuadrado) que en el perceptrón lineal.

### Actualización de pesos

La fórmula general sigue siendo la misma (Gradient Descent):

$$
\Delta w = \eta \cdot (t^{\mu} - O^{\mu}) \cdot \theta'(h) \cdot x^{\mu}
$$

Pero ahora **$\theta'(h) \neq 1$**, porque la función de activación ya no es la identidad. La derivada de la función de activación aparece explícitamente en la fórmula de actualización.

### Funciones de activación disponibles

| Función | Forma | Notas |
|---------|-------|-------|
| **Sigmoidea / Logística** | Curva en forma de S | Similar a escalón pero suave |
| **Tangente hiperbólica (tanh)** | Curva en forma de S centrada en 0 | Salida entre -1 y 1 |

### Parámetro $beta$

Tanto la función sigmoidea como la tangente hiperbólica cuentan con un **parámetro $\beta$** que determina la forma de la función:

- Si $\beta$ es chico, la función se **parece más a una función lineal**.
- Si $\beta$ es grande, la función se **parece más a una función escalón**.
- **Para elegir una función de activación hay que testearlas**.

---

## Resumen visual

| Tipo | Función de activación $\theta(h)$ | Derivada $\theta'(h)$ | Uso |
|------|-----------------------------------|----------------------|-----|
| Escalón | $1$ si $h > u$, $0$ si no | No definida en $u$ (discontinua) | Clasificación binaria |
| Lineal (Identidad) | $h$ | $1$ | Regresión lineal |
| No lineal (Sigmoide/tanh) | Curva S | Función derivable | Clasificación/Regresión no lineal |

---

## Algoritmo general (para cualquier función de activación derivable)

```
Initialize weights w to small random values
Set learning rate eta

for a fixed number of epochs:
  for each training example mu in the dataset:
   1. Calculate the weighted sum:
     h^mu = w_0 + w_1 * x^mu_1 + ... + w_n * x^mu_n
   2. Compute activation:
     O^mu = theta(h^mu)
   3. Compute derivative of activation:
     theta_prime = theta'(h^mu)
   4. Update the weights:
      For each weight w_i (i = 0..n):
         w_i = w_i + eta * (t^mu - O^mu) * theta_prime * x^mu_i

  5. After iterating over ALL examples, calculate global error (MSE)
  6. Check for convergence
End
```

Cuando $\theta$ es la identidad, $\theta' = 1$ y recuperamos el perceptrón lineal.
Cuando $\theta$ es escalón, no se puede usar este algoritmo directamente (se usa la regla de Rosenblatt).
