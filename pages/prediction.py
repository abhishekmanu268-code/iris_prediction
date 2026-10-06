from pathlib import Path
import pickle

import streamlit as st

st.set_page_config(page_title="Prediction", page_icon="🔮")

MODEL_PATH = Path(__file__).resolve().parent.parent / "iris_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)

model = load_model()

st.title("Iris Prediction")
st.caption("Use flower measurements to predict the species.")

with st.form("iris_prediction_form"):
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
    submitted = st.form_submit_button("Predict Species")

if submitted:
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    class_names = ["Setosa", "Versicolor", "Virginica"]
    predicted_species = class_names[prediction]

    st.success(f"Predicted species: {predicted_species}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        prob_df = {}
        for name, prob in zip(class_names, probabilities):
            prob_df[name] = round(float(prob) * 100, 2)
        st.write("Prediction confidence:")
        st.bar_chart(prob_df)

    st.write("Input values:")
    st.json({
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    })
