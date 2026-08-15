# Learning-to-Rank: Theory to Practice in This Repository

This guide is a technical deep dive for engineers who want to understand:

1. Why each ranking concept matters.
2. How each concept is represented in this codebase.
3. How to run and extend the implementation safely.

Scope: this repository demonstrates three learning-to-rank stacks over one shared synthetic dataset:

- XGBoost with XGBRanker
- LightGBM with LGBMRanker (lambdarank)
- allRank (neural LTR framework in PyTorch)

---

## 0) LTR theory primer: what learning-to-rank is and why it is different

Learning-to-rank (LTR) is the problem of ordering candidate items for each query so the most relevant items appear at the top.

Examples:

- Search ranking: sort documents for a user query.
- Marketplace ranking: sort providers for a service request.
- Recommender ranking: sort candidate products/content for a session.

Why LTR is not standard classification/regression:

- The objective is relative order within each query group, not absolute score quality across the whole dataset.
- Position matters. Getting the top 3 right is usually much more valuable than getting positions 30 to 40 right.
- Training data has query context. The same item feature value can matter differently under different queries.

### 0.1 Main LTR algorithm families

#### Pointwise

Idea:

- Predict relevance score per item independently.

How it works:

- Train a standard regression/classification model.
- At inference time, sort candidates by predicted score.

Pros:

- Simple and fast baseline.
- Reuses standard ML tooling.

Cons:

- Loss does not explicitly optimize pair/list order.
- Can underperform on ranking metrics.

When relevant:

- Strong baseline, cold start project phase, limited ranking labels.

#### Pairwise

Idea:

- Learn which item should be above another item for the same query.

How it works:

- Build item pairs per query.
- Optimize pair preference loss (for example, RankNet-style logistic pair loss).

Pros:

- Directly models ordering preferences.

Cons:

- Pair generation can be expensive.
- Still an indirect approximation to full list quality.

When relevant:

- Good middle ground when listwise methods are too heavy.

#### Listwise

Idea:

- Optimize a loss over the full query list, often aligned with ranking metrics like NDCG.

How it works:

- Compute gradients that reflect how score changes impact top-of-list quality.
- LambdaMART/lambdarank methods are common practical choices.

Pros:

- Usually strongest offline ranking quality for production tabular ranking.

Cons:

- More complex objective behavior and tuning.

When relevant:

- Default choice once baseline is established and ranking quality matters.

### 0.2 How training works in practice

For each query q with items d_1..d_n:

1. Build model scores s_i = f(x_i).
2. Compute ranking loss based on pointwise/pairwise/listwise objective.
3. Update model parameters to improve query-level ordering.
4. Evaluate by query-level ranking metrics (not global row metrics).

In this repo:

- Query grouping is explicit via qid and group vectors.
- Tree and neural pipelines share the same generated relevance signal.
- Evaluation is query-grouped NDCG.

### 0.3 Evaluation metrics: what they measure and when they are relevant

#### NDCG@k (Normalized Discounted Cumulative Gain)

What it measures:

- Graded relevance quality with position discount up to rank k.

Why it matters:

- Captures top-of-list importance and graded labels.
- Standard metric for most LTR systems.

When relevant:

- Primary offline metric when labels are graded (as in this repo).

#### MAP (Mean Average Precision)

What it measures:

- Precision across relevant positions, usually for binary relevance.

When relevant:

- Retrieval tasks with binary relevance labels.

#### MRR (Mean Reciprocal Rank)

What it measures:

- Inverse rank of first relevant result.

When relevant:

- User value is dominated by first successful hit.

#### Precision@k / Recall@k

What they measure:

- Fraction of relevant items in top k / coverage of relevant items in top k.

When relevant:

- Interpretability for stakeholders and diagnostics.

### 0.4 Offline vs online relevance

Offline metrics are necessary but not sufficient.

- Offline tells you if ranking signal improved on historical or validation data.
- Online experiments tell you business impact under live traffic.

Typical online metrics for ranking systems:

- CTR or engagement rate.
- Conversion/booking/close rate.
- Revenue per session/query.
- Guardrails such as cancellation/satisfaction.

Most relevant for this repo now:

- Offline NDCG is the key target because this repo is an offline demo stack.
- The next practical step is experiment wiring for online metrics when integrated into a product.

---

## 1) Ranking fundamentals and where they appear in code

### 1.1 Query groups (core data shape)

LTR is not a plain row-wise prediction task. Data is grouped by query. Each query has a list of candidate items that must be sorted.

Repository mapping:

- Query ids per row are created in src/ltr_demo/data_utils.py.
- Group-size vectors expected by tree rankers are also created there.
- allRank gets equivalent grouping via LibSVM rows with qid values.

Practical consequence:

- If query boundaries are wrong, ranking metrics become meaningless because the model is scored across mixed slates.

### 1.2 Relevance labels and gain

This demo uses graded labels (integers from 0 to 4). Graded relevance makes NDCG meaningful and enables gain-based objectives.

Repository mapping:

- Labels are generated and clipped in src/ltr_demo/data_utils.py.
- LightGBM gains are configured with label_gain in scripts/run_tree_rankers.py.

Practical consequence:

- Label design encodes business intent. Changing label semantics changes what “good ranking” means.

### 1.3 NDCG as an offline metric

NDCG discounts lower positions and is computed per query, then averaged.

Repository mapping:

- grouped_ndcg_at_k in src/ltr_demo/metrics.py slices each query block and computes ndcg_score per group.
- summarize_ndcg returns ndcg@5 and ndcg@10 used in scripts/run_tree_rankers.py.

Practical consequence:

- Global row-level correlation metrics can look good while top-of-list ranking quality is poor.

---

## 2) Data pipeline: from synthetic signal to model-ready formats

### 2.1 Synthetic signal design

Data generation combines:

- Weighted linear signal from strongest features.
- Non-linear terms with sin/cos.
- Quantization to graded labels.

Repository mapping:

- _build_split in src/ltr_demo/data_utils.py.

Why this matters:

- It creates a non-trivial ranking problem so each model has meaningful structure to learn.

### 2.2 Dual export strategy

The same in-memory dataset is exported in two formats:

- NPZ for XGBoost and LightGBM.
- LibSVM train.txt and vali.txt for allRank.

Repository mapping:

- save_npz and save_libsvm in src/ltr_demo/data_utils.py.
- Entry point script: scripts/generate_dataset.py.

Why this matters:

- One data source lets you compare model families fairly.

---

## 3) Tree rankers: XGBRanker and LGBMRanker

### 3.1 XGBoost configuration

Key parameters in scripts/run_tree_rankers.py:

- objective="rank:ndcg"
- eval_metric=["ndcg@5", "ndcg@10"]
- group passed to fit

Interpretation:

- Optimization is directly ranking-oriented rather than plain regression/classification.

### 3.2 LightGBM configuration

Key parameters in scripts/run_tree_rankers.py:

- objective="lambdarank"
- metric="ndcg"
- label_gain=[0, 1, 3, 7, 15]
- group passed to fit

Interpretation:

- LambdaMART-style updates focus on pair/list ordering and gain-sensitive ranking improvement.

### 3.3 Offline evaluation

Both models are evaluated with the exact same grouped metrics.

Repository mapping:

- summarize_ndcg in src/ltr_demo/metrics.py.
- printed side-by-side output in scripts/run_tree_rankers.py.

This is essential for an apples-to-apples comparison.

---

## 3b) Pointwise baseline and metrics (from TaskRabbit reference)

The TaskRabbit document calls out a pointwise baseline as an important starting point:

- train a row-wise probabilistic model (for example booking probability)
- rank by predicted probability
- evaluate ranking quality with query-grouped metrics

Repository implementation:

- Script: `scripts/run_pointwise_baseline.py`
- Model: `XGBClassifier` + `CalibratedClassifierCV`
- Ranking score: calibrated positive-class probability

Metric choices in this script:

- `NDCG@5`, `NDCG@10` on graded relevance labels
- `MAP@5`, `MAP@10` on binary relevance labels
- `MRR@5`, `MRR@10` on binary relevance labels

Helper metric functions:

- `grouped_map_at_k` in `src/ltr_demo/metrics.py`
- `grouped_mrr_at_k` in `src/ltr_demo/metrics.py`

Why this baseline is relevant:

- Fast to train and easy to explain.
- Produces interpretable probabilities for business stakeholders.
- Provides a reference point before moving to pairwise/listwise ranking losses.

---

## 4) allRank: neural LTR configuration and execution

### 4.1 allRank config synthesis

This repository does not rely on static JSON files committed by hand. It builds config dynamically at runtime.

Repository mapping:

- build_config in scripts/run_allrank.py writes outputs/allrank/allrank_config.json.

Configured choices:

- FC network with small hidden layers for fast local execution.
- lambdaLoss with lambdaRank_scheme.
- ndcg_5 and ndcg_10 metrics.
- short epoch schedule for demo speed.

### 4.2 Windows compatibility layer

allRank internals call Unix commands cp and rm. On Windows, they are missing by default.

Repository mapping:

- prepare_windows_shims in scripts/run_allrank.py creates cp.cmd and rm.cmd.
- Script prepends shim directory to PATH for the child process.

Why this matters:

- This preserves upstream allRank behavior without patching installed package files.

### 4.3 allRank run output

Artifacts are written under outputs/allrank/results/<run-id> and include model state and training logs.

Repository mapping:

- Command launch and result path printout in scripts/run_allrank.py.

---

## 5) End-to-end execution flow

### Step 1: environment

Use docs/DEMO_RUNBOOK.md for exact uv commands and dependency setup.

### Step 2: data generation

Run scripts/generate_dataset.py to materialize both NPZ and LibSVM representations.

### Step 3: tree rankers

Run scripts/run_tree_rankers.py and inspect ndcg@5 / ndcg@10.

### Step 4: allRank

Run scripts/run_allrank.py and inspect allRank logs + output folder.

---

## 6) How to extend this repository

### 6.1 Data and labels

Where to change:

- src/ltr_demo/data_utils.py

Typical extensions:

- Variable slate sizes per query.
- Additional non-linear interactions.
- Alternative label generation to mimic business outcomes.

### 6.2 Metrics

Where to change:

- src/ltr_demo/metrics.py

Typical extensions:

- Add MRR@k or MAP.
- Break metrics down by query segments.

### 6.3 Model hyperparameters

Where to change:

- scripts/run_tree_rankers.py
- scripts/run_allrank.py (build_config)

Typical extensions:

- Grid/Optuna hyperparameter sweeps.
- Multiple random seeds with confidence intervals.

### 6.4 Experiment tracking

Current state:

- Console-first outputs and allRank artifact folders.

Suggested next step:

- Add MLflow logging wrappers around both tree and allRank runs.

---

## 7) Common failure modes and reasoning checks

1. Misaligned group vector and row order.
Outcome: invalid ranking loss/metric behavior.

2. Label gains inconsistent with label scale.
Outcome: distorted optimization priorities.

3. allRank dependency/environment mismatch.
Outcome: import/runtime failures. Follow docs/DEMO_RUNBOOK.md exactly.

4. Using global metrics that ignore query boundaries.
Outcome: deceptively optimistic quality estimates.

---

## 8) File map for quick navigation

- Data generation logic: src/ltr_demo/data_utils.py
- Ranking metrics: src/ltr_demo/metrics.py
- Dataset entrypoint: scripts/generate_dataset.py
- Tree rankers entrypoint: scripts/run_tree_rankers.py
- allRank entrypoint and Windows shims: scripts/run_allrank.py
- Runbook commands: docs/DEMO_RUNBOOK.md
- High-level onboarding: README.md

---

## 9) Technical summary

This repository operationalizes a full mini-LTR lifecycle:

- Query-grouped data construction with graded labels.
- Consistent offline ranking evaluation with grouped NDCG.
- Two strong tree baselines (XGBoost and LightGBM).
- Neural LTR example via allRank, including platform compatibility handling.

The code now includes detailed inline comments and docstrings so every implementation step can be traced back to ranking theory.
