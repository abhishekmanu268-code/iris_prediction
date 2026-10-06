import streamlit as st
from sklearn.datasets import load_iris

st.set_page_config(page_title="Species", page_icon="🌿")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = [iris.target_names[i] for i in iris.target]

st.title("Iris Species")
st.caption("Three species of iris flowers are included in the classic dataset.")

species_summary = (
    df.groupby("species")[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    .mean()
    .round(2)
)

st.dataframe(species_summary, use_container_width=True)

st.subheader("Species overview")

species_info = {
    "setosa": {
        "description": "Setosa has short, broad petals and is usually the easiest to identify visually.",
        "traits": ["Small petals", "Narrow sepal width range", "Often lower petal length"]
    },
    "versicolor": {
        "description": "Versicolor is a mid-sized iris species with moderately long petals and balanced dimensions.",
        "traits": ["Medium petals", "Balanced silhouette", "Moderate sepal dimensions"]
    },
    "virginica": {
        "description": "Virginica is the largest of the three species, with the longest and widest petals.",
        "traits": ["Longest petals", "Widest petal width", "Largest overall size"]
    }
}

for species_name, details in species_info.items():
    with st.container():
        st.markdown(f"### {species_name.title()}")
        st.write(details["description"])
        st.write("Key traits:")
        for trait in details["traits"]:
            st.write(f"- {trait}")
        st.write("---")
