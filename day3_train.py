import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")["target"]
X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_parquet("data/y_test.parquet")["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained.")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))
print(confusion_matrix(y_test, y_pred))

import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global explanation — which features matter most overall
shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
plt.tight_layout()
plt.savefig("summary_plot.png")
plt.close()

print("Saved summary_plot.png")

# Find one connection the model flagged as attack
attack_idx = 0
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        attack_idx = i
        break

print("Row index:", attack_idx)
print("Actual label:", y_test.iloc[attack_idx])
print("Predicted label:", y_pred[attack_idx])

shap.force_plot(
    explainer.expected_value[1],
    shap_values[attack_idx, :, 1],
    X_test.iloc[attack_idx],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("local_explanation_row0.png")
plt.close()

print("Saved local_explanation_row0.png")

import joblib
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
