import streamlit as st
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Home", page_icon="🌼", layout="wide")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = [iris.target_names[i] for i in iris.target]

st.title("Iris Flower Dashboard")
st.caption("A simple three-page app for exploring the famous Iris dataset and predicting flower species.")

col1, col2, col3 = st.columns(3)
col1.metric("Samples", len(df))
col2.metric("Features", len(df.columns) - 1)
col3.metric("Species", len(iris.target_names))

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Species distribution")
count_by_species = df["species"].value_counts().rename_axis("Species").reset_index(name="Count")
st.bar_chart(count_by_species.set_index("Species"))

st.markdown("### About the Iris dataset")
st.write(
    "The Iris dataset contains measurements of iris flowers and is commonly used in machine learning "
    "examples. It includes three species: Setosa, Versicolor, and Virginica."
)

st.info("Use the pages in the sidebar to learn more about the species and make predictions.")