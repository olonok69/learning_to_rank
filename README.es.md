# Demo de Learning-to-Rank

Una demo ejecutable y autocontenida de *learning-to-rank* (LTR) que cubre cuatro
estrategias de modelado sobre un mismo conjunto de datos sintético agrupado por consulta:

1. **Línea base puntual (*pointwise*)** — `XGBClassifier` calibrado, ordenado por probabilidad predicha
2. **Por pares / por lista (basado en árboles)** — `XGBRanker` (`rank:ndcg`)
3. **Por pares / por lista (basado en árboles)** — `LGBMRanker` (`lambdarank`)
4. **Por lista (neural)** — [allRank](https://github.com/allegro/allRank), un framework LTR en PyTorch

Los cuatro enfoques se entrenan y evalúan contra el mismo dataset generado, de modo que
sus resultados de `NDCG@5` / `NDCG@10` (y, en la línea base, `MAP@k` / `MRR@k`) son
directamente comparables.

> Documentación en inglés: [README.md](README.md)

## Por qué existe este repositorio

Los problemas de *learning-to-rank* (búsqueda, emparejamiento en marketplaces,
recomendaciones) tratan de ordenar candidatos *dentro de una consulta/slate*, no de
predecir una única puntuación global. Este repo es una referencia compacta y funcional
para:

- cómo se representa el agrupamiento por consulta (`qid` / `group`) en la API de cada librería
- cómo las etiquetas de relevancia graduada alimentan objetivos listwise como NDCG/LambdaMART
- cómo se compara una línea base puntual simple frente a objetivos de ranking dedicados
- cómo hacer funcionar allRank en local, incluido en Windows

Para un recorrido completo de teoría a código, consulta
[docs/THEORY_TO_PRACTICE_GUIDE.es.md](docs/THEORY_TO_PRACTICE_GUIDE.es.md).
Para una hoja de comandos condensada, consulta
[docs/DEMO_RUNBOOK.es.md](docs/DEMO_RUNBOOK.es.md).

## Estructura del proyecto

```text
.
├─ pyproject.toml
├─ data/
│  └─ demo/
│     ├─ dataset.npz            # formato para rankers de árboles + línea base
│     └─ allrank/                # formato LibSVM (train.txt, vali.txt)
├─ docs/
│  ├─ DEMO_RUNBOOK.md            # secuencia de comandos + resolución de problemas (EN)
│  ├─ DEMO_RUNBOOK.es.md         # secuencia de comandos + resolución de problemas (ES)
│  ├─ THEORY_TO_PRACTICE_GUIDE.md# teoría LTR mapeada a este código (EN)
│  └─ THEORY_TO_PRACTICE_GUIDE.es.md
├─ outputs/
│  └─ allrank/                   # config generada + resultados/logs de entrenamiento
├─ scripts/
│  ├─ generate_dataset.py        # construye el dataset sintético (NPZ + LibSVM)
│  ├─ run_pointwise_baseline.py  # línea base clasificador calibrado + NDCG/MAP/MRR
│  ├─ run_tree_rankers.py        # XGBRanker + LGBMRanker
│  └─ run_allrank.py             # construye la config de allRank y lanza el entrenamiento
└─ src/ltr_demo/
   ├─ data_utils.py              # generación de datos sintéticos + exportación NPZ/LibSVM
   └─ metrics.py                 # NDCG@k, MAP@k, MRR@k agrupados
```

## 1) Configuración del entorno (uv)

Este repositorio apunta a Python 3.10 para una compatibilidad fluida con allRank.

```powershell
cd D:\repos\learning_to_rank

# Crear y activar un entorno virtual
uv venv .venv --python 3.10
.\.venv\Scripts\Activate.ps1

# Instalar dependencias base (XGBoost, LightGBM, scikit-learn, numpy, scipy)
uv sync
```

allRank tiene dependencias extra y más pesadas (PyTorch), así que se instala por separado:

```powershell
uv pip install --python .venv\Scripts\python.exe torch torchvision
uv pip install --python .venv\Scripts\python.exe git+https://github.com/allegro/allRank.git --no-deps
uv pip install --python .venv\Scripts\python.exe attrs flatten-dict tensorboardX gcsfs google-auth pandas
```

## 2) Generar el dataset de la demo

```powershell
uv run --python .venv\Scripts\python.exe python scripts/generate_dataset.py
```

Esto construye un dataset sintético de ranking — “slates” de consulta de tamaño fijo con
etiquetas de relevancia graduada (`0`–`4`) — y lo escribe en dos formatos:

- `data/demo/dataset.npz` — usado por los scripts de rankers de árboles y de línea base puntual
- `data/demo/allrank/train.txt` y `data/demo/allrank/vali.txt` — formato LibSVM con `qid`, usado por allRank

Consulta `generate_synthetic_dataset` en [src/ltr_demo/data_utils.py](src/ltr_demo/data_utils.py)
para la lógica de generación (señal de features, términos no lineales, cuantización a
etiquetas graduadas).

## 3) Ejecutar la línea base puntual

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_pointwise_baseline.py
```

Entrena un `XGBClassifier` calibrado (`binary:logistic` + `CalibratedClassifierCV`) sobre
una versión binarizada de las etiquetas graduadas, ordena los ítems por la probabilidad
calibrada de la clase positiva y reporta:

- `NDCG@5`, `NDCG@10` (relevancia graduada)
- `MAP@5`, `MAP@10` (relevancia binaria)
- `MRR@5`, `MRR@10` (relevancia binaria)

Es la línea base más simple posible: un buen punto de referencia antes de comparar
contra los objetivos de ranking dedicados de abajo.

## 4) Ejecutar los rankers basados en árboles

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_tree_rankers.py
```

Entrena y evalúa, en paralelo, sobre el mismo split de entrenamiento/validación:

- `XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@5", "ndcg@10"])`
- `LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7, 15])`

Ambos usan los vectores `group`/`qid` para definir los límites de consulta, y ambos se
puntúan con la misma implementación de NDCG agrupado para una comparación justa.

## 5) Ejecutar la demo neural de allRank

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_allrank.py
```

Este script:

1. Construye en memoria una config JSON de allRank (red FC pequeña, `lambdaLoss` con
   `lambdaRank_scheme`, métricas NDCG@5/@10) y la escribe en `outputs/allrank/allrank_config.json`
2. Lanza `python -m allrank.main` como subproceso usando los datos LibSVM
3. En Windows, instala *shims* ligeros de `cp`/`rm` (ver `prepare_windows_shims` en
   [scripts/run_allrank.py](scripts/run_allrank.py)) para que las llamadas internas de
   allRank funcionen sin parchear el paquete
4. Escribe los resultados (checkpoints del modelo, logs de TensorBoard, predicciones) en
   `outputs/allrank/results/demo_allrank/`

## Métricas de evaluación

Todas las métricas agrupadas viven en [src/ltr_demo/metrics.py](src/ltr_demo/metrics.py) y
se calculan **por consulta**, y luego se promedian — nunca se mezclan entre límites de consulta:

| Métrica | Función | Tipo de relevancia | Usada por |
|---|---|---|---|
| `NDCG@k` | `grouped_ndcg_at_k` / `summarize_ndcg` | graduada (0–4) | los cuatro enfoques |
| `MAP@k` | `grouped_map_at_k` | binaria | línea base puntual |
| `MRR@k` | `grouped_mrr_at_k` | binaria | línea base puntual |

## Inspeccionar el entrenamiento de allRank con TensorBoard

Las curvas de entrenamiento (pérdida, learning rate, NDCG@5/@10 de train/val) se
registran como archivos de eventos de TensorBoard en `outputs/allrank/tb_evals/`:

```powershell
uv run --python .venv\Scripts\python.exe tensorboard --logdir outputs/allrank/tb_evals
```

## Notas y limitaciones conocidas

- Las etiquetas son relevancia graduada sintética (`0`–`4`); este repo sirve para comparar
  *enfoques de modelado*, no para representar un dataset de producto real.
- allRank tiene restricciones de dependencias transitivas más antiguas y estrechas, y
  puede ser sensible al entorno en Windows. Si falla la instalación, mantén primero
  funcionando las demos de rankers de árboles y de línea base puntual, y luego reintenta
  allRank en un entorno limpio/aislado (ver la sección de resolución de problemas en
  [docs/DEMO_RUNBOOK.es.md](docs/DEMO_RUNBOOK.es.md)).
- NDCG/MAP/MRR offline son necesarios pero no suficientes: validan la calidad de ranking
  en datos reservados, no el impacto real de usuario/negocio. La experimentación online
  es el siguiente paso natural al integrar cualquiera de estos enfoques en un producto
  en vivo.
