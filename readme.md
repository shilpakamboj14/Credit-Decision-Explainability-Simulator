# Explainable Credit Decisioning Simulator

A credit approval model where the real deliverable isn't the prediction — it's the **explanation**. Every approval or denial comes with a plain-English, auditor-readable reason, modeled on what U.S. lenders are legally required to provide under the Equal Credit Opportunity Act (ECOA).

🔗 **[Live demo]()** &nbsp;|&nbsp; 🧠 Built with Python, scikit-learn, SHAP, Streamlit

---

## Why this project exists

Credit models don't just need to be accurate — they need to be **explainable**. In the U.S., ECOA's "adverse action notice" requirement means a lender can't just reject an application; they must tell the applicant *specifically why*. That's not a hypothetical AI-ethics concern — it's existing law, and it's exactly the kind of requirement banks are now trying to satisfy with modern ML models instead of simple scorecards.

This project builds a small, honest version of that pipeline: train a real model on real (public) credit data, then wrap it in an explainability layer that turns statistical output into a sentence a compliance officer — or a rejected applicant — could actually read and understand.

---

## What it does

1. Takes an applicant's financial details (credit utilization, age, late payment history, debt ratio, income)
2. Predicts whether they're likely to default
3. If declined, uses **SHAP (SHapley Additive exPlanations)** to identify the top 3 factors driving that decision
4. Converts those factors into a plain-English sentence — no SHAP values, no jargon, no raw coefficients
5. Approved applications get no justification text, mirroring real regulatory practice: lenders must explain declines, not approvals

**Example output:**
> *"This application was declined primarily due to: a history of late payments (30-59 days past due), applicant age, high debt-to-income ratio."*

---

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) — ~150,000 historical loan applicants with a known outcome: did they experience serious delinquency within 2 years?

Features used (kept deliberately simple and interpretable — no heavily engineered black-box features):

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilization ratio |
| `age` | Applicant age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30–59 day late payments |
| `DebtRatio` | Debt relative to income |
| `MonthlyIncome` | Monthly income |
| `NumberOfTimes90DaysLate` | Count of 90+ day late payments |

`MonthlyIncome` had ~20% missing values, imputed with the median rather than dropped, to avoid losing a fifth of the dataset.

---

## Modeling approach

**Model:** Logistic Regression — chosen deliberately over a more "accurate" black-box model (e.g. gradient boosting), because its behavior is directly explainable and this project's entire premise is explainability over raw performance.

**The class imbalance problem:** Only ~7% of applicants in the dataset actually default. A naive model achieves 93% accuracy just by predicting "no default" for everyone — while missing 98%+ of actual defaulters. That's a model that looks good on a dashboard and is useless (dangerous, even) in production.

**Fix:** Applied `class_weight` tuning to the logistic regression, testing several ratios and comparing precision, recall, and F1-score for the minority (defaulter) class rather than trusting overall accuracy:

| `class_weight` (defaulter) | Recall (defaulters) | Recall (non-defaulters) | F1 (defaulters) |
|---|---|---|---|
| None (default) | 1% | 100% | 0.03 |
| 3 | 3% | 100% | — |
| `"balanced"` (~13–14) | 63% | 78% | — |
| 7 | 49% | 90% | 0.34 |
| **9 (final)** | **46%** | **92%** | **0.36** |
| 11 | 53% | 88% | 0.33 |

Weight `9` was selected as the best balance — it gave the strongest F1-score for the defaulter class while keeping the non-defaulter recall reasonably high, avoiding the false-alarm-heavy behavior seen at higher weights.

This is a real, ongoing trade-off in credit risk modeling: missing a defaulter costs the lender directly, while over-rejecting safe applicants costs them in lost business. There's no "correct" answer — only a deliberate choice, which is itself something an explainability layer should be able to surface.

---

## Explainability layer (the core of this project)

For every **declined** application:

1. **SHAP** computes how much each feature pushed the model's internal score toward "high risk," relative to a baseline drawn from the training population
2. The top 3 contributing features are selected — mirroring the "specific reasons" requirement in real adverse action notices, not a dump of every input
3. Each feature is mapped to a fixed, pre-written plain-English phrase (e.g. `NumberOfTime30-59DaysPastDueNotWorse` → *"a history of late payments (30-59 days past due)"*) — deliberately not auto-generated from the column name, to keep output consistent and audit-safe
4. The phrases are assembled into a single, fixed-template sentence

Approved applications return a simple approval message with no justification — reflecting the actual legal asymmetry in ECOA: reasons are required for denials, not approvals.

---

## Architecture

```
credit_project/
├── data/
│   ├── cs-training.csv          # raw Kaggle dataset
│   ├── cleaned_data.csv         # cleaned, feature-selected data
│   ├── model.pkl                # trained logistic regression model
│   └── scaler.pkl               # fitted StandardScaler
├── prepare_data.py              # load + clean raw data
├── train_model.py               # train, evaluate, save model
├── reason_utils.py              # shared SHAP + explanation logic
├── run_decision.py              # CLI: score a single new applicant
└── dashboard.py                 # Streamlit interactive demo
```

`reason_utils.py` centralizes the explanation logic so both the CLI script and the dashboard call the same function (`get_decision_text`) rather than duplicating it — one place to fix or extend the explainability behavior.

---

## Interactive dashboard

Built with Streamlit. Enter an applicant's details, click **Check Application**, and get:
- An approve/decline decision
- If declined, the top 3 plain-English reasons
- An expandable "How this works" section explaining the model, the class-weighting decision, and the explainability approach — for anyone reviewing the tool, not just the applicant

```bash
streamlit run dashboard.py
```

---

## Key takeaways / what this project demonstrates

- **Accuracy alone is a misleading metric for imbalanced classification** — a 93%-accurate model can catch essentially zero real defaulters
- **Precision/recall/F1 trade-offs are business decisions, not just math** — tuning `class_weight` is really a decision about which error costs more
- **Explainability isn't a bolt-on visualization** — SHAP output only means something once it's translated into language a non-technical stakeholder (auditor, applicant, regulator) can act on
- **Regulatory requirements can shape technical design directly** — the "top 3 reasons, denials only" behavior isn't arbitrary, it's a direct reflection of ECOA's adverse action notice rules

---

## Possible extensions

- Add a black-box model (e.g. gradient boosting) alongside logistic regression to explicitly compare accuracy vs. explainability
- Add SHAP force-plot visualizations to the dashboard alongside the text explanation
- Track applicant `cust_id` end-to-end for auditability of individual decisions
- Deploy the dashboard publicly (Streamlit Community Cloud) for a live demo link

---

## Tech stack

`Python` · `pandas` · `scikit-learn` (LogisticRegression, StandardScaler, train_test_split) · `SHAP` · `Streamlit` · `joblib`