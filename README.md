# 🚀 Score Matching

Generación condicional de imágenes en MNIST mediante un modelo de difusión
(score matching), comparando classifier-free guidance (CFG) con `w = 1`
(red condicional pura) vs. `w != 1` (red con dropout de etiqueta).

## 🎖️ Estructura del repositorio

```
score_matching/
├── download_data.py    # descarga MNIST (torchvision) -> data/
├── model.py            # Modelo: UNet condicional (backbone de difusion)
├── model_clf.py        # Clasificador: CNN auxiliar (juez automatico)
├── train.py            # entrena una red de difusion (--label-dropout)
├── train_clf.py        # entrena el clasificador auxiliar
├── sample.py           # genera imagenes integrando la SDE reversa
├── evaluate.py         # compara modelo_cond vs modelo_cfg (fidelidad/diversidad)
├── pipeline.mmd        # fuente Mermaid del diagrama de flujo entre archivos
├── requirements.txt
├── data/               # MNIST descargado (no versionado, solo el script)
├── checkpoints/        # modelo_cond.pt, modelo_cfg.pt, clasificador.pt
├── figures/
│   ├── loss/           # curvas de perdida (.png) de cada red de difusion
│   ├── samples/        # grillas .png generadas por sample.py
│   ├── evaluate/       # tablas/graficos generados por evaluate.py
│   └── pipeline.png    # diagrama del flujo entre archivos (ver mas abajo)
├── informe/            # informe.pdf (maximo 3 planas) + fuente si aplica
└── notebooks/
    ├── train_colab.ipynb   # orquestador: clona el repo y entrena en GPU (ver seccion de abajo)
    └── ...                 # notebooks de exploracion inicial (opcional, no evaluado)
```

### 🧩 Grafo entre archivos

`figures/pipeline.png` documenta como se invocan los archivos entre si (`imports`)
y como fluyen los artefactos en disco (`data/` -> checkpoints -> figuras). Se
genera con [Mermaid](https://mermaid.js.org/syntax/flowchart.html) a partir de
un archivo `.mmd` con el diagrama:

```bash
npm install -g @mermaid-js/mermaid-cli   # una sola vez

mmdc -i pipeline.mmd -o figures/pipeline.png -b white -s 2
```

Para editar el diagrama, modifica `pipeline.mmd` y vuelve a correr `mmdc`. El
diagrama tambien puede visualizarse sin instalar nada pegando el contenido de
`pipeline.mmd` en [mermaid.live](https://mermaid.live).

## ☕ Entorno

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducir resultados

> TODO: completar con los comandos exactos una vez implementado el código,
> de modo que reproduzcan lo mostrado en el video.

```bash
python download_data.py

python train.py --label-dropout 0.0 --out checkpoints        # -> modelo_cond.pt
python train.py --label-dropout 0.1 --out checkpoints        # -> modelo_cfg.pt
python train_clf.py --out checkpoints                        # -> clasificador.pt

python sample.py --checkpoint checkpoints/modelo_cfg.pt --w <w_elegido>

python evaluate.py --modelo-cond checkpoints/modelo_cond.pt \
    --modelo-cfg checkpoints/modelo_cfg.pt \
    --clasificador checkpoints/clasificador.pt \
    --w <w_elegido>
```

## 🎮 Entrenamiento en GPU (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anmedinas/score_matching/blob/main/notebooks/train_colab.ipynb)

`notebooks/train_colab.ipynb` es el notebook que usamos para entrenar los
checkpoints entregados en GPU. **No reimplementa nada**: clona este mismo
repo desde GitHub y ejecuta `download_data.py`, `train.py`, `train_clf.py`,
`sample.py` y `evaluate.py` tal cual están en el código (los mismos comandos
de la sección "Reproducir resultados" de arriba), así el checkpoint
resultante queda trazado a un commit concreto, no a una copia pegada en una
celda.

Para usarlo: abrirlo con el botón de arriba (o subirlo a Colab manualmente),
`Entorno de ejecución -> Cambiar tipo de entorno de ejecución -> GPU`, y
ejecutar todas las celdas. Los hiperparámetros (épocas, batch size,
learning rate, `label_dropout`, `w` de CFG, pasos de muestreo) son
editables en una celda de formulario antes de entrenar — no hay valores
altos fijados de antemano, se ajustan según el tiempo de GPU disponible. Al
final, empaqueta `checkpoints/` y `figures/` en un `.zip` descargable.

## 🍺 Delibery

| Item | Ubicación |
|---|---|
| Informe (.pdf, máx. 3 planas) | `informe/informe.pdf` |
| Código + requirements + README | raíz del repo |
| Checkpoints (`modelo_cond.pt`, `modelo_cfg.pt`, `clasificador.pt`) | `checkpoints/` |
| Curvas de pérdida (2) | `figures/loss/` |
| Script de comparación | `evaluate.py` (salida en `figures/evaluate/`) |
| Video (≤ 15 min) | enlace: TODO |
