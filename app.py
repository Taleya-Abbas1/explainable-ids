import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Explainable Network Intrusion Detection")
st.caption("Model trained on the NSL-KDD dataset")

model = joblib.load("model.pkl")
X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_parquet("data/y_test.parquet")["target"]

explainer = shap.TreeExplainer(model)

row_idx = st.slider("Pick a test connection", 0, len(X_test) - 1, 0)

row = X_test.iloc[[row_idx]]
prediction = model.predict(row)[0]
actual = y_test.iloc[row_idx]

st.write("Prediction:", "Attack" if prediction == 1 else "Normal")
st.write("Actual label:", "Attack" if actual == 1 else "Normal")

st.subheader("Why did the model decide this?")
shap_values = explainer.shap_values(row)

# Show only top contributing features for a cleaner plot
row_shap = pd.Series(shap_values[0, :, 1], index=row.columns)
top_features = row_shap.abs().sort_values(ascending=False).head(8).index

shap.force_plot(
    explainer.expected_value[1],
    row_shap[top_features].values,
    row[top_features].iloc[0],
    matplotlib=True,
    show=False
)
fig = plt.gcf()
st.pyplot(fig)
plt.close(fig)
st.write("**Top contributing features for this prediction:**")
for feat in top_features:
    val = row[feat].iloc[0]
    impact = row_shap[feat]
    direction = "pushed toward Attack" if impact > 0 else "pushed toward Normal"
    st.write(f"- `{feat}` = {val:.3f} → {direction} (impact: {impact:.3f})")
