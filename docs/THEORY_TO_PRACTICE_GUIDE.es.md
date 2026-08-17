# Learning-to-Rank: de la teoría a la práctica en este repositorio

Esta guía es una inmersión técnica para ingenieros que quieran entender:

1. Por qué importa cada concepto de ranking.
2. Cómo se representa cada concepto en este código.
3. Cómo ejecutar y extender la implementación de forma segura.

Alcance: este repositorio demuestra tres stacks de learning-to-rank sobre un mismo
dataset sintético compartido:

- XGBoost con XGBRanker
- LightGBM con LGBMRanker (lambdarank)
- allRank (framework LTR neural en PyTorch)

> Versión en inglés: [THEORY_TO_PRACTICE_GUIDE.md](THEORY_TO_PRACTICE_GUIDE.md)

---

## 0) Primeros conceptos de LTR: qué es learning-to-rank y por qué es distinto

Learning-to-rank (LTR) es el problema de ordenar ítems candidatos para cada consulta de
modo que los más relevantes aparezcan arriba.

Ejemplos:

- Ranking de búsqueda: ordenar documentos para una consulta de usuario.
- Ranking de marketplace: ordenar proveedores para una solicitud de servicio.
- Ranking de recomendación: ordenar productos/contenido candidatos para una sesión.

Por qué LTR no es clasificación/regresión estándar:

- El objetivo es el orden relativo dentro de cada grupo de consulta, no la calidad
  absoluta de la puntuación en todo el dataset.
- La posición importa. Acertar el top 3 suele ser mucho más valioso que acertar las
  posiciones 30 a 40.
- Los datos de entrenamiento tienen contexto de consulta. El mismo valor de feature de
  un ítem puede importar de forma distinta bajo consultas distintas.

### 0.1 Principales familias de algoritmos LTR

#### Puntual (*pointwise*)

Idea:

- Predecir la puntuación de relevancia de cada ítem de forma independiente.

Cómo funciona:

- Entrenar un modelo estándar de regresión/clasificación.
- En inferencia, ordenar candidatos por la puntuación predicha.

Pros:

- Línea base simple y rápida.
- Reutiliza el tooling estándar de ML.

Contras:

- La pérdida no optimiza de forma explícita el orden de pares/listas.
- Puede rendir peor en métricas de ranking.

Cuándo es relevante:

- Línea base sólida, fase de arranque del proyecto, etiquetas de ranking limitadas.

#### Por pares (*pairwise*)

Idea:

- Aprender qué ítem debe quedar por encima de otro para la misma consulta.

Cómo funciona:

- Construir pares de ítems por consulta.
- Optimizar una pérdida de preferencia de pares (por ejemplo, pérdida logística de
  pares al estilo RankNet).

Pros:

- Modela directamente las preferencias de orden.

Contras:

- La generación de pares puede ser costosa.
- Sigue siendo una aproximación indirecta a la calidad de la lista completa.

Cuándo es relevante:

- Buen punto intermedio cuando los métodos listwise son demasiado pesados.

#### Por lista (*listwise*)

Idea:

- Optimizar una pérdida sobre la lista completa de la consulta, a menudo alineada con
  métricas de ranking como NDCG.

Cómo funciona:

- Calcular gradientes que reflejen cómo los cambios de puntuación impactan la calidad
  de la parte alta de la lista.
- Los métodos LambdaMART/lambdarank son elecciones prácticas habituales.

Pros:

- Suele dar la mejor calidad de ranking offline para ranking tabular en producción.

Contras:

- El comportamiento del objetivo y su ajuste son más complejos.

Cuándo es relevante:

- Elección por defecto una vez establecida la línea base y cuando importa la calidad
  de ranking.

### 0.2 Cómo funciona el entrenamiento en la práctica

Para cada consulta q con ítems d_1..d_n:

1. Construir puntuaciones del modelo s_i = f(x_i).
2. Calcular la pérdida de ranking según el objetivo puntual/por pares/por lista.
3. Actualizar los parámetros del modelo para mejorar el orden a nivel de consulta.
4. Evaluar con métricas de ranking a nivel de consulta (no métricas globales por fila).

En este repo:

- El agrupamiento por consulta es explícito mediante vectores qid y group.
- Los pipelines de árboles y neurales comparten la misma señal de relevancia generada.
- La evaluación es NDCG agrupado por consulta.

### 0.3 Métricas de evaluación: qué miden y cuándo son relevantes

#### NDCG@k (Normalized Discounted Cumulative Gain)

Qué mide:

- Calidad de relevancia graduada con descuento por posición hasta el rango k.

Por qué importa:

- Captura la importancia de la parte alta de la lista y las etiquetas graduadas.
- Métrica estándar en la mayoría de sistemas LTR.

Cuándo es relevante:

- Métrica offline principal cuando las etiquetas son graduadas (como en este repo).

#### MAP (Mean Average Precision)

Qué mide:

- Precisión a lo largo de las posiciones relevantes, normalmente para relevancia binaria.

Cuándo es relevante:

- Tareas de recuperación con etiquetas de relevancia binaria.

#### MRR (Mean Reciprocal Rank)

Qué mide:

- Inverso del rango del primer resultado relevante.

Cuándo es relevante:

- El valor para el usuario está dominado por el primer acierto.

#### Precision@k / Recall@k

Qué miden:

- Fracción de ítems relevantes en el top k / cobertura de ítems relevantes en el top k.

Cuándo son relevantes:

- Interpretabilidad para stakeholders y diagnóstico.

### 0.4 Relevancia offline vs online

Las métricas offline son necesarias pero no suficientes.

- Offline te dice si la señal de ranking mejoró en datos históricos o de validación.
- Los experimentos online te dicen el impacto de negocio bajo tráfico real.

Métricas online típicas de sistemas de ranking:

- CTR o tasa de engagement.
- Tasa de conversión/reserva/cierre.
- Ingresos por sesión/consulta.
- Guardarraíles como cancelación/satisfacción.

Lo más relevante para este repo ahora:

- NDCG offline es el objetivo clave porque este repo es un stack de demo offline.
- El siguiente paso práctico es cablear experimentos para métricas online al integrarlo
  en un producto.

---

## 1) Fundamentos de ranking y dónde aparecen en el código

### 1.1 Grupos de consulta (forma central de los datos)

LTR no es una tarea de predicción fila a fila. Los datos se agrupan por consulta. Cada
consulta tiene una lista de ítems candidatos que hay que ordenar.

Mapeo en el repositorio:

- Los ids de consulta por fila se crean en src/ltr_demo/data_utils.py.
- Los vectores de tamaño de grupo que esperan los rankers de árboles también se crean ahí.
- allRank recibe un agrupamiento equivalente mediante filas LibSVM con valores qid.

Consecuencia práctica:

- Si los límites de consulta están mal, las métricas de ranking dejan de tener sentido
  porque el modelo se puntúa mezclando slates distintos.

### 1.2 Etiquetas de relevancia y ganancia

Esta demo usa etiquetas graduadas (enteros de 0 a 4). La relevancia graduada hace que
NDCG tenga sentido y habilita objetivos basados en ganancia.

Mapeo en el repositorio:

- Las etiquetas se generan y recortan en src/ltr_demo/data_utils.py.
- Las ganancias de LightGBM se configuran con label_gain en scripts/run_tree_rankers.py.

Consecuencia práctica:

- El diseño de etiquetas codifica la intención de negocio. Cambiar la semántica de las
  etiquetas cambia lo que significa un “buen ranking”.

### 1.3 NDCG como métrica offline

NDCG descuenta las posiciones más bajas y se calcula por consulta, y luego se promedia.

Mapeo en el repositorio:

- grouped_ndcg_at_k en src/ltr_demo/metrics.py recorta cada bloque de consulta y calcula
  ndcg_score por grupo.
- summarize_ndcg devuelve ndcg@5 y ndcg@10 usados en scripts/run_tree_rankers.py.

Consecuencia práctica:

- Las métricas de correlación globales a nivel de fila pueden verse bien mientras la
  calidad de ranking en la parte alta de la lista es pobre.

---

## 2) Pipeline de datos: de la señal sintética a formatos listos para el modelo

### 2.1 Diseño de la señal sintética

La generación de datos combina:

- Señal lineal ponderada de las features más fuertes.
- Términos no lineales con sin/cos.
- Cuantización a etiquetas graduadas.

Mapeo en el repositorio:

- _build_split en src/ltr_demo/data_utils.py.

Por qué importa:

- Crea un problema de ranking no trivial para que cada modelo tenga estructura
  significativa que aprender.

### 2.2 Estrategia de exportación dual

El mismo dataset en memoria se exporta en dos formatos:

- NPZ para XGBoost y LightGBM.
- LibSVM train.txt y vali.txt para allRank.

Mapeo en el repositorio:

- save_npz y save_libsvm en src/ltr_demo/data_utils.py.
- Script de entrada: scripts/generate_dataset.py.

Por qué importa:

- Una sola fuente de datos permite comparar familias de modelos de forma justa.

---

## 3) Rankers de árboles: XGBRanker y LGBMRanker

### 3.1 Configuración de XGBoost

Parámetros clave en scripts/run_tree_rankers.py:

- objective="rank:ndcg"
- eval_metric=["ndcg@5", "ndcg@10"]
- group pasado a fit

Interpretación:

- La optimización está orientada directamente al ranking, no a regresión/clasificación
  simple.

### 3.2 Configuración de LightGBM

Parámetros clave en scripts/run_tree_rankers.py:

- objective="lambdarank"
- metric="ndcg"
- label_gain=[0, 1, 3, 7, 15]
- group pasado a fit

Interpretación:

- Las actualizaciones al estilo LambdaMART se centran en el orden de pares/listas y en
  la mejora de ranking sensible a la ganancia.

### 3.3 Evaluación offline

Ambos modelos se evalúan con exactamente las mismas métricas agrupadas.

Mapeo en el repositorio:

- summarize_ndcg en src/ltr_demo/metrics.py.
- salida impresa lado a lado en scripts/run_tree_rankers.py.

Esto es esencial para una comparación justa.

---

## 3b) Línea base puntual y métricas (referencia TaskRabbit)

El documento de TaskRabbit señala una línea base puntual como punto de partida
importante:

- entrenar un modelo probabilístico fila a fila (por ejemplo, probabilidad de reserva)
- ordenar por la probabilidad predicha
- evaluar la calidad de ranking con métricas agrupadas por consulta

Implementación en el repositorio:

- Script: `scripts/run_pointwise_baseline.py`
- Modelo: `XGBClassifier` + `CalibratedClassifierCV`
- Puntuación de ranking: probabilidad calibrada de la clase positiva

Elección de métricas en este script:

- `NDCG@5`, `NDCG@10` sobre etiquetas de relevancia graduada
- `MAP@5`, `MAP@10` sobre etiquetas de relevancia binaria
- `MRR@5`, `MRR@10` sobre etiquetas de relevancia binaria

Funciones auxiliares de métricas:

- `grouped_map_at_k` en `src/ltr_demo/metrics.py`
- `grouped_mrr_at_k` en `src/ltr_demo/metrics.py`

Por qué es relevante esta línea base:

- Rápida de entrenar y fácil de explicar.
- Produce probabilidades interpretables para stakeholders de negocio.
- Ofrece un punto de referencia antes de pasar a pérdidas de ranking pairwise/listwise.

---

## 4) allRank: configuración y ejecución de LTR neural

### 4.1 Síntesis de la config de allRank

Este repositorio no depende de archivos JSON estáticos commiteados a mano. Construye la
config de forma dinámica en tiempo de ejecución.

Mapeo en el repositorio:

- build_config en scripts/run_allrank.py escribe outputs/allrank/allrank_config.json.

Elecciones configuradas:

- Red FC con capas ocultas pequeñas para una ejecución local rápida.
- lambdaLoss con lambdaRank_scheme.
- métricas ndcg_5 y ndcg_10.
- calendario corto de epochs para velocidad de demo.

### 4.2 Capa de compatibilidad con Windows

Los internos de allRank llaman a los comandos Unix cp y rm. En Windows faltan por
defecto.

Mapeo en el repositorio:

- prepare_windows_shims en scripts/run_allrank.py crea cp.cmd y rm.cmd.
- El script antepone el directorio de shims al PATH del proceso hijo.

Por qué importa:

- Conserva el comportamiento de allRank aguas arriba sin parchear los archivos del
  paquete instalado.

### 4.3 Salida de la ejecución de allRank

Los artefactos se escriben en outputs/allrank/results/<run-id> e incluyen el estado del
modelo y los logs de entrenamiento.

Mapeo en el repositorio:

- Lanzamiento del comando e impresión de la ruta de resultados en scripts/run_allrank.py.

---

## 5) Flujo de ejecución de extremo a extremo

### Paso 1: entorno

Usa docs/DEMO_RUNBOOK.es.md para los comandos exactos de uv y la instalación de
dependencias.

### Paso 2: generación de datos

Ejecuta scripts/generate_dataset.py para materializar las representaciones NPZ y LibSVM.

### Paso 3: rankers de árboles

Ejecuta scripts/run_tree_rankers.py e inspecciona ndcg@5 / ndcg@10.

### Paso 4: allRank

Ejecuta scripts/run_allrank.py e inspecciona los logs de allRank + la carpeta de salida.
