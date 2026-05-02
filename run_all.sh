#!/bin/bash
# Orquesta todos los experimentos del TP3 en orden.
# Uso: bash run_all.sh
#
# Tiempo total estimado: 30-50 minutos (depende del hardware).

set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "EJ2 - Sweeps de hiperparámetros"
echo "================================================================"

echo "[EJ2-LR]  Variantes de learning rate..."
python3 -u ej2/part2/learning_rate/run.py
python3 -u ej2/part2/learning_rate/plot.py

echo "[EJ2-ARCH] Variantes de arquitectura..."
python3 -u ej2/part2/architecture/run.py
python3 -u ej2/part2/architecture/plot.py

echo "[EJ2-OPT] Variantes de optimizador..."
python3 -u ej2/part2/optimizer/run.py
python3 -u ej2/part2/optimizer/plot.py

echo "[EJ2-BATCH-LR] Sweep 2D batch_size x learning rate..."
python3 -u ej2/part2/batch_lr/run.py
python3 -u ej2/part2/batch_lr/plot.py

echo "[EJ2-OPT-LR] Sweep 2D optimizer x learning rate..."
python3 -u ej2/part2/optimizer_lr/run.py
python3 -u ej2/part2/optimizer_lr/plot.py

echo "[EJ2-ACT] Variantes de activación intermedia..."
python3 -u ej2/part2/activation/run.py
python3 -u ej2/part2/activation/plot.py

echo "[EJ2-SEL] Modelo seleccionado..."
python3 -u ej2/part2/selected_model/run.py
python3 -u ej2/part2/selected_model/plot.py

echo "[EJ2-NOISE] Robustez al ruido..."
python3 -u ej2/part2/noise_robustness/run.py
python3 -u ej2/part2/noise_robustness/plot.py

echo "================================================================"
echo "EJ3 - Comparación con más datos + alcanzar 98%"
echo "================================================================"

echo "[EJ3-CMP] Comparación digits vs more_digits..."
python3 -u ej3/part2/data_comparison/run.py
python3 -u ej3/part2/data_comparison/plot.py

echo "[EJ3-SEL] Modelo seleccionado para 98%..."
python3 -u ej3/part2/selected_model/run.py
python3 -u ej3/part2/selected_model/plot.py

echo ""
echo "================================================================"
echo "TODO LISTO. Para compilar la presentación:"
echo "  cd presentacion && pdflatex main.tex"
echo "================================================================"
