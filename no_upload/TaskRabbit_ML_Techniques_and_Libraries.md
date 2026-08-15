# TaskRabbit ML Engineer — Techniques & Libraries Reference

**Purpose:** A study companion for the ranking / optimisation / MLOps interview. Organised by the four problems in the JD. For each: the concepts, the concrete techniques, the Python libraries, and short code you can speak to. End sections give a build sequence, a library cheat-sheet, and ready soundbites.

The real problem behind the JD: *rank taskers for each client search to optimise revenue, close rate, and satisfaction, while keeping the marketplace healthy by uplifting new/inexperienced taskers — and prove every change with experiments.* That is **learning-to-rank + multi-objective optimisation + exploration/fairness + marketplace A/B testing + MLOps**.

---

## 1. Learning-to-Rank (LTR)

### Concept
A client search is a **query**. The candidate taskers shown are the **query group**. You learn to order each group so the items most likely to produce a good outcome (booking + completion + high rating) sit at the top. You optimise a **rank-aware** loss, not plain classification, because position matters.

Three families:
- **Pointwise** — predict a score per item independently (e.g. P(booking)). Simplest; ignores within-list ordering. A calibrated XGBoost classifier ranked by predicted probability is a legitimate pointwise baseline.
- **Pairwise** — learn that item A should rank above item B (RankNet, LambdaRank). Optimises relative order.
- **Listwise** — optimise the whole list against a rank metric directly (LambdaMART, ListNet). Usually best for NDCG.

### Metrics
- **NDCG@k** — Normalised Discounted Cumulative Gain. Position-discounted relevance, normalised to [0,1]. The default ranking metric.
- **MAP** — Mean Average Precision, for binary relevance.
- **MRR** — Mean Reciprocal Rank, when you care mostly about the first good result.
- Marketplace-specific online metrics: **close/booking rate**, completion rate, satisfaction, plus new-tasker exposure share.

### Libraries
- **XGBoost** — `XGBRanker`, objectives `rank:ndcg`, `rank:pairwise`, `rank:map`. They named xgboost in the JD; lead with this.
- **LightGBM** — `LGBMRanker` with `objective="lambdarank"`. Often the strongest/fastest LambdaMART implementation; great talking point as the natural upgrade from XGBoost.
- **CatBoost** — `YetiRank` / `PairLogit` objectives; strong with categorical features (tasker category, city) with no manual encoding.
- **scikit-learn** — not LTR-native, but the home for the pointwise baseline, calibration (`CalibratedClassifierCV`), and `ndcg_score` / `dcg_score` in `sklearn.metrics`.
- **TensorFlow Ranking** / **allRank** (PyTorch) — neural LTR if they ever go deep; mention as "where I'd go if linear/tree LTR plateaus."

### Code you can speak to
```python
import xgboost as xgb
import numpy as np

# group = number of candidate taskers per client search, in row order
# e.g. groups [5, 3, 8] => first 5 rows are search 1, next 3 are search 2, ...
ranker = xgb.XGBRanker(
    objective="rank:ndcg",
    eval_metric=["ndcg@5", "ndcg@10"],
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
)
ranker.fit(X_train, y_relevance, group=group_train)   # y_relevance: graded label per row

# LightGBM LambdaMART equivalent — usually the stronger listwise option
from lightgbm import LGBMRanker
lgb_ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=500,
    learning_rate=0.05,
    label_gain=[0, 1, 3, 7],   # graded relevance gains
)
lgb_ranker.fit(X_train, y_relevance, group=group_train)
```

### The relevance label — the design decision that matters most
You construct the graded label from outcomes, e.g.
`0 = shown not clicked`, `1 = clicked`, `2 = booked`, `3 = completed`, `4 = completed + high rating`.
Say in interview: *"The label encodes the business objective; if revenue matters I weight high-value completed tasks higher in the gain vector."* This shows you understand that the objective lives in the label, not just the loss.

---

## 2. Multi-Objective Optimisation

You must trade revenue vs close rate vs satisfaction vs new-tasker uplift. Three approaches — know all three and the order to try them.

### A. Scalarisation (start here)
Combine objective scores into one ranking score with weights:
```
score = w_rev * value_pred + w_close * p_book + w_sat * sat_pred + w_new * exploration_bonus
```
Pros: simple, tunable, A/B-testable per weight. Cons: weights are implicit business decisions; fixed trade-off.

### B. Constrained optimisation
Maximise the primary objective subject to floors/ceilings, e.g. *maximise revenue s.t. new-tasker exposure ≥ X% and predicted satisfaction ≥ threshold.* This is how "uplift new taskers" becomes a **hard guardrail** rather than a soft term. Tools: linear/ILP solvers for re-ranking the top-N (`PuLP`, `OR-Tools`), or post-hoc re-rank with quotas.

### C. Multi-task / multi-head models
Predict close-rate, satisfaction, and value with separate heads (shared representation), then combine at serving time. More flexible, more infra. Libraries: any NN framework, or simply train three gradient-boosted models and blend.

### Pareto framing (to sound senior)
There is no single optimum across competing objectives — there's a **Pareto frontier**. You pick an operating point on it via business-agreed weights, and A/B tests tell you where on the frontier the business actually wants to sit. Tools for explicit Pareto search: **Optuna** (multi-objective study), **pymoo** (NSGA-II).

```python
import optuna

def objective(trial):
    w_rev   = trial.suggest_float("w_rev", 0, 1)
    w_close = trial.suggest_float("w_close", 0, 1)
    w_sat   = trial.suggest_float("w_sat", 0, 1)
    # ... build blended ranking, evaluate offline ...
    return revenue_proxy, satisfaction_proxy   # two objectives -> Pareto front

study = optuna.create_study(directions=["maximize", "maximize"])
study.optimize(objective, n_trials=100)
# study.best_trials gives the Pareto-optimal weightings
```

**Interview line:** *"I'd start with a weighted blend because it's interpretable and testable, treat new-tasker exposure as a hard guardrail not a soft weight, and use multi-objective tuning to map the trade-off frontier before committing."*

---

## 3. Cold-Start & "Uplifting Inexperienced Taskers"

This is the highest-signal part of the JD. It's an **exploration vs exploitation** plus **fairness-of-exposure** problem. A pure exploit ranker buries new taskers forever (no history → low rank → no tasks → no history), starving marketplace supply.

### Cold-start features (no history required)
Score new taskers on: skill/category match, proximity, availability, profile completeness, price competitiveness, onboarding signals. Reserve history-based features (completion rate, ratings) for tenured taskers and let the model learn from what's present.

### Exploration techniques
- **Epsilon-greedy** — with prob ε, inject/boost a new tasker into the visible slots. Trivial to implement and explain.
- **Upper Confidence Bound (UCB)** — rank by `estimate + uncertainty`; high-uncertainty (new) taskers get a bonus that shrinks as data arrives.
- **Thompson sampling** — sample from each tasker's posterior outcome distribution; naturally explores uncertain (new) taskers. The elegant Bayesian answer.
- **Contextual bandits** — exploration conditioned on context (task type, location, time). The principled framing for "which tasker to surface given this query."

### Fairness-of-exposure
Beyond exploration, you can guarantee new/underrepresented taskers a **minimum exposure share** — amortised fairness, or constrained re-ranking. Relevant because TaskRabbit has documented academic scrutiny over ranking bias against underrepresented taskers.
- **FA*IR** algorithm — fair top-k re-ranking with a protected-group exposure constraint.
- **Inverse-propensity weighting (IPS)** — correct the training data for **position bias**: items shown higher get more clicks regardless of quality, so naive logs over-reward whatever the old ranker already favoured (the feedback-loop trap). IPS de-biases this.

### Libraries
- **contextualbandits** (Python) — epsilon-greedy, UCB, Thompson, LinUCB out of the box.
- **Vowpal Wabbit** — production-grade contextual bandits / online learning at scale.
- **River** — online/incremental learning, useful for taskers whose stats update continuously.
- **Fairlearn** — fairness metrics and mitigation (exposure/selection-rate parity).
- **fairsearch-fair** — implementation of the FA*IR fair-ranking algorithm.

```python
# Thompson-sampling flavour: blend exploration bonus into the ranking score
import numpy as np

def exploration_bonus(n_shown, n_booked, prior_a=1.0, prior_b=1.0):
    # Beta posterior on booking prob; sample => uncertain (new) taskers can spike high
    return np.random.beta(prior_a + n_booked, prior_b + (n_shown - n_booked))

final_score = base_relevance + lambda_explore * exploration_bonus(n_shown, n_booked)
```

**Interview line:** *"I treat new-tasker uplift as exploration plus a fairness constraint. New taskers score on history-free features with an uncertainty bonus — Thompson-style — and I'd enforce a minimum exposure budget. Crucially I'd correct training logs for position bias with inverse-propensity weighting, otherwise the ranker just reinforces whoever it already promoted."*

---

## 4. A/B Testing & Experimentation (Marketplace-Aware)

The basics matter, but the thing that marks you as senior is knowing why naive A/B testing **breaks in a two-sided marketplace**.

### The interference problem (lead with this)
Treating some clients/taskers changes the shared tasker supply that the control group also competes for. Treatment and control are not independent → **SUTVA violated** → biased results. Solutions:
- **Switchback testing** — alternate the whole market between treatment/control over time windows. Standard for marketplaces (Uber/Lyft/DoorDash use it).
- **Cluster / geo randomisation** — randomise by city/market so interference stays within a unit. (TaskRabbit literally rolled out a system change market-by-market historically, so this framing fits their operating model.)

### Core design
- **Hypothesis + primary metric** — usually close/booking rate or revenue per search.
- **Guardrail metrics** — satisfaction, cancellation rate, new-tasker exposure. A revenue win that tanks satisfaction is a ship blocker.
- **Randomisation unit** — client, session, or market (see interference above).
- **Power / MDE / duration** — compute sample size for the minimum detectable effect *before* launch.
- **Decision rule agreed up front** — avoid p-hacking / peeking.

### Variance reduction & rigour
- **CUPED** — use pre-experiment covariates to cut variance, shrinking required sample/time. Strong thing to name.
- **Sequential testing / always-valid p-values** — peek safely without inflating false positives.
- **Novelty & primacy effects** — early behaviour change isn't the steady state; run long enough.

### Libraries
- **statsmodels** / **scipy.stats** — t-tests, proportions z-test, power analysis (`statsmodels.stats.power`).
- **GeoLift** (Meta, R; Python ports exist) — geo experiment design & measurement.
- **CausalImpact** (Google) — quasi-experiment when a clean A/B isn't possible.
- **PyMC** — Bayesian A/B testing for probabilistic "P(B > A)" statements.

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

# 1) Pre-launch: sample size for a 2pp lift on a 20% baseline close rate
effect = 0.02 / ((0.20 * 0.80) ** 0.5)          # standardised effect size
n_per_arm = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8)

# 2) Post-launch: did treatment beat control on bookings?
counts = [bookings_treat, bookings_ctrl]
nobs   = [shown_treat,    shown_ctrl]
z, p = proportions_ztest(counts, nobs)
```

**Interview line:** *"In a marketplace I'm wary of naive A/B because treatment changes the supply control also draws from. I'd default to switchback or geo-cluster randomisation, define primary plus guardrail metrics with a pre-registered decision rule, and use CUPED to cut variance so we can read results faster."*

---

## 5. Feature Engineering for Ranking

What goes into `X`. Group features by entity:

- **Tasker features:** completion rate, avg rating, response/acceptance rate, recency of activity, tenure, price, no-show/cancellation rate, category expertise, profile completeness.
- **Client features:** past booking behaviour, price sensitivity, repeat vs new, preferred categories.
- **Query–tasker interaction (the ranking signal):** skill-to-task match, distance/proximity, availability overlap with requested time, price vs task budget, prior client–tasker history.
- **Context:** time of day, day of week, local supply/demand, seasonality, market.
- **Recency / decay:** exponentially weighted recent performance beats lifetime averages.
- **Embeddings (advanced):** learned tasker/category embeddings for semantic skill match; collaborative-filtering signals (clients like you booked taskers like these).

Tools: **Spark / PySpark** for feature computation at scale (your strength), **Feast** or Databricks Feature Store for a serving feature store (avoids training-serving skew), **category_encoders** for high-cardinality categoricals.

**Interview line:** *"The ranking signal lives in the query–tasker interaction features — skill match, proximity, availability, price fit — more than in either entity alone. I'd compute these in Spark and serve them through a feature store to kill training-serving skew."*

---

## 6. MLOps for Ranking Systems

This is where your Databricks/MLflow/AI-Mart background converts directly. "Enhance our in-house ML framework + MLOps" = your home turf.

### The pillars
- **Experiment tracking & registry:** MLflow tracking + Model Registry, staged promotion (Staging → Production), reproducible runs. You've done this.
- **Feature store:** Feast / Databricks FS for consistent offline-training and online-serving features.
- **Deployment:** batch precompute of rankings vs real-time scoring service; canary / shadow deployment before full rollout.
- **Monitoring — ranking-specific:**
  - **Feature drift** (input distributions shift) — Evidently, `whylogs`.
  - **Prediction/quality drift** — NDCG / close rate trending down.
  - **Feedback-loop monitoring** — is the ranker collapsing diversity / starving new taskers over time?
- **Retraining:** scheduled + trigger-on-drift; Databricks Asset Bundles / Airflow for orchestration (you use both).
- **Position-bias correction in the training loop** — IPS weighting so logged data doesn't just echo the old ranker (ties back to §3).

### Libraries / platforms
- **MLflow** — tracking, registry, model packaging.
- **Databricks Asset Bundles** — dev/prod deployment (your AI Mart pattern maps 1:1).
- **Feast** — feature store.
- **Evidently** / **whylogs** — drift & data-quality monitoring.
- **Airflow** / Databricks Workflows — orchestration & retraining.
- **BentoML** / **Seldon** / Databricks Model Serving — real-time serving.

**Interview line:** *"I'd wire the ranker into MLflow for tracking and a registry-gated promotion flow, serve features from a store to avoid skew, and monitor not just accuracy but ranking quality and marketplace-health drift — including whether the model is quietly starving new supply. On Databricks I'd deploy with Asset Bundles across dev/prod, which is exactly how I run my current production data platform."*

---

## 7. A Pragmatic Build Sequence (if asked "how would you start")

1. **Baseline:** pointwise calibrated XGBoost classifier on P(booking), rank by score. Ship, measure, establish the A/B harness.
2. **Upgrade to listwise LTR:** LightGBM LambdaMART / `XGBRanker rank:ndcg` with query groups; richer graded relevance label.
3. **Add multi-objective:** scalarised score (revenue + close + satisfaction) with new-tasker exposure as a hard guardrail.
4. **Add exploration:** Thompson/UCB bonus for cold-start taskers; IPS correction on training logs.
5. **Harden experimentation:** switchback / geo-cluster A/B, CUPED, guardrail dashboards.
6. **Productionise:** feature store, MLflow registry, drift + marketplace-health monitoring, scheduled retrain.

Stating this sequence answers "where would you start and how would you grow it" in one breath — and signals you ship simple first and earn complexity with experiments.

---

## 8. Library Cheat-Sheet

| Problem | Go-to libraries |
|---|---|
| Learning-to-rank | XGBoost (`XGBRanker`), LightGBM (`LGBMRanker`/lambdarank), CatBoost (YetiRank), TF-Ranking / allRank |
| Pointwise baseline & metrics | scikit-learn (`CalibratedClassifierCV`, `ndcg_score`) |
| Multi-objective / tuning | Optuna (multi-objective), pymoo (NSGA-II), OR-Tools / PuLP (constrained re-rank) |
| Exploration / bandits | contextualbandits, Vowpal Wabbit, River |
| Fairness / exposure | Fairlearn, fairsearch-fair (FA*IR) |
| A/B testing | statsmodels, scipy.stats, PyMC, GeoLift, CausalImpact |
| Features at scale | PySpark, Feast / Databricks Feature Store, category_encoders |
| MLOps | MLflow, Databricks Asset Bundles, Airflow, Evidently / whylogs, BentoML / Seldon |

---

## 9. Soundbites (memorise 4–5)

- *"A client search is a query group; I optimise NDCG with the relevance label encoding the business objective — booking, completion, high rating, weighted by value."*
- *"Start with a weighted multi-objective score, but make new-tasker exposure a hard guardrail, not a soft weight."*
- *"Uplifting new taskers is exploration plus fairness: history-free features, a Thompson-style uncertainty bonus, and inverse-propensity weighting so the ranker doesn't just reinforce who it already promoted."*
- *"In a marketplace I distrust naive A/B because treatment changes the supply control competes for — I'd use switchback or geo-cluster randomisation, with CUPED to read results faster."*
- *"I'd monitor not just accuracy but ranking-quality and marketplace-health drift, deployed on Databricks with Asset Bundles and an MLflow registry — the same production pattern I run today."*

---

## 10. Two-line honesty script (keep ready)

*"I haven't shipped a LambdaMART ranker in production specifically, but I've deployed the model families it's built on — gradient-boosted propensity and next-best-action ranking on Databricks — and I understand learning-to-rank, multi-objective trade-offs, and marketplace experimentation. On a contract I ramp on a bespoke stack fast; I do it on every engagement."*
