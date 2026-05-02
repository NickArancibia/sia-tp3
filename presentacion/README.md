# Presentación TP3 — Perceptrón Simple y Multicapa

## Compilar

```bash
cd presentacion
pdflatex main.tex
pdflatex main.tex   # segunda vez para resolver referencias
```

Requiere LaTeX con el tema `metropolis` instalado:
```bash
sudo apt install texlive-latex-extra texlive-fonts-extra
```

## Estructura

- `main.tex` — slides Beamer en español.
- Las imágenes referenciadas viven en `../ej2/results/part2/...` y
  `../ej3/results/part2/...`. Hay que correr antes los scripts
  `ej{2,3}/part2/*/run.py` y `plot.py`.

## Antes de compilar

Asegurate de tener todos los gráficos generados:

```bash
bash ../run_all.sh
```

Esto corre todos los experimentos de EJ2 y EJ3 y genera los PNGs.
Tarda 30-50 min en una laptop moderna.

## Slides incluidas

1. **Marco teórico**: arquitectura MLP, forward, backward, MSE, capacidad.
2. **EJ2** — Setup + hallazgo crítico (clase 8 ausente del train).
3. **EJ2** — Sweeps de lr, arquitectura, optimizador.
4. **EJ2** — Modelo seleccionado (5 seeds, confusion matrix, per-class).
5. **EJ2** — Robustez al ruido gaussiano (opcional).
6. **EJ3** — Comparación digits vs more_digits (mismo test).
7. **EJ3** — Modelo seleccionado para 98% accuracy.
8. **Conclusiones** + referencias al material.
