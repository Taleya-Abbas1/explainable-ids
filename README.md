# Explainable Network Intrusion Detection

A machine learning pipeline that detects malicious network traffic **and explains
why** each connection was flagged, using SHAP (SHapley Additive exPlanations).

## Problem

Modern intrusion detection systems increasingly rely on machine learning to flag
suspicious traffic — but most of these models are black boxes. They output a
decision (malicious/benign) with no explanation of *why*. As a SOC analyst, this
gap is a daily frustration: an alert fires, and you still have to manually dig
through logs to understand what actually triggered it. This slows down triage,
reduces analyst trust in the tool, and makes it harder to justify escalations or
filter out false positives.

This project builds a small end-to-end system that detects attacks **and**
surfaces which specific traffic features drove each decision — directly
addressing that transparency gap.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved version of the
classic KDD Cup 99 intrusion detection dataset. 125,973 labeled training
connections and 22,544 test connections, each described by 41 features
(protocol type, connection duration, byte counts, failed login attempts, error
rates, etc.), labeled as either `normal` or one of 20+ specific attack types
(e.g. `neptune`, `smurf`, `satan`, `guess_passwd`).

For this project, attack labels were collapsed into a binary target:
`normal` vs. `attack`.

## Approach

1. **Preprocessing** — encoded categorical features (protocol type, service,
   flag) into numeric form, scaled all numeric features with `StandardScaler`
   (fit on train, applied to test to avoid data leakage), and converted labels
   to a binary target.
2. **Model** — trained a `RandomForestClassifier` (100 trees) on the training
   set.
3. **Evaluation** — measured accuracy, precision, recall, and F1-score on the
   held-out test set.
4. **Explainability** — applied SHAP's `TreeExplainer` to generate:
   - A **global explanation**: which features matter most across all
     predictions.
   - A **local explanation**: for individual flagged connections, exactly which
     features pushed that specific prediction toward "attack," and by how much.
5. **Interactive demo** — built a Streamlit app where a user can select any test
   connection and see the model's prediction alongside its SHAP explanation, in
   both chart and plain-text form.

## Results

| Metric    | Normal | Attack |
|   ---     |  ---   |   ---  |
| Precision |  0.66  |  0.97  |
| Recall    |  0.97  |  0.62  |
| F1-score  |  0.78  |  0.75  |

Overall accuracy: **77%**

**Confusion matrix:**
```
                Predicted Normal   Predicted Attack
Actual Normal        9433               278
Actual Attack         4892              7941
```

**What this shows:** the model is highly precise when it flags something as an
attack (97% of "attack" predictions are correct) but misses a meaningful share
of real attacks (62% recall). This is expected and worth discussing: NSL-KDD's
test set intentionally includes attack types the model never saw during
training, so this gap reflects a realistic and known challenge — models trained
on known attack patterns can struggle to generalize to novel ones. In a SOC
context, this maps to a real tradeoff: a model tuned for high precision reduces
false-positive alert fatigue, but at the cost of missing some genuinely
malicious traffic.

## Explainability findings

The global SHAP summary plot (`figures/summary_plot.png`) shows which features
matter most across the whole model. The local explanation
(`figures/local_explanation_row0.png`) breaks down a single flagged connection,
showing exactly which feature values pushed it toward "attack."

## Demo

An interactive Streamlit app (`app.py`) lets you pick any test connection and
see:
- The model's prediction vs. the actual label
- A SHAP force plot for that specific connection
- A plain-text breakdown of the top contributing features and their impact

Run it locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reflection

*As someone early in my cybersecurity path, I've seen — through coursework,
personal projects, and following SOC workflows — how much of an analyst's
time goes into manually investigating why an alert fired, rather than acting
on it. This project let me experience that gap directly: training a model
that performs well is one thing, but understanding *why* it made a specific
decision is a separate and, in practice, more valuable skill. Building the
SHAP explainability layer changed how I think about ML in security — a
model's accuracy score alone doesn't tell an analyst anything actionable,
but a feature-level explanation does. This is exactly the kind of
transparency I'd want to keep exploring in future SOC-related work.*

## Tools

`pandas`, `scikit-learn`, `shap`, `matplotlib`, `streamlit`, `joblib`

## Project structure
```
explainable-ids/
├── data/                    # NSL-KDD files (not committed — see Dataset section)
├── figures/                 # Saved SHAP plots
├── day2_preprocess.py       # Data cleaning, encoding, scaling
├── day3_train.py            # Model training, evaluation, SHAP
├── app.py                   # Streamlit demo
├── model.pkl                # Saved trained model
└── requirements.txt
```
