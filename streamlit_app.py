from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans


st.set_page_config(page_title="Segmentacion de Clientes K-Means", layout="wide")
st.title("Despliegue: Segmentacion de Clientes con K-Means")

MODEL_PATH = Path("kmeans_mall_customers.joblib")
DATA_PATH = Path("Mall_Customers.csv")

if not MODEL_PATH.exists():
    st.error("No se encontro kmeans_mall_customers.joblib. Ejecuta primero la exportacion en el notebook.")
    st.stop()

if not DATA_PATH.exists():
    st.error("No se encontro Mall_Customers.csv en la carpeta del proyecto.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
cluster_names = bundle.get("cluster_names", {})
strategy_map = bundle.get("strategy_map", {})
k_opt = int(bundle.get("k", model.n_clusters))

df = pd.read_csv(DATA_PATH)
X = df[features].copy()
X_scaled = scaler.transform(X)
labels = model.predict(X_scaled)

st.subheader("Metodo del codo")
k_values = bundle.get("k_values_elbow")
inertias = bundle.get("inertias_elbow")

if k_values is None or inertias is None:
    k_values = list(range(1, 11))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(k_values, inertias, marker="o", color="royalblue")
ax_elbow.axvline(k_opt, color="crimson", linestyle="--", label=f"K seleccionado = {k_opt}")
ax_elbow.set_title("Metodo del codo")
ax_elbow.set_xlabel("Numero de clusters (K)")
ax_elbow.set_ylabel("Inercia (WCSS)")
ax_elbow.legend()
st.pyplot(fig_elbow)

st.subheader("Clusters generados")
fig_clusters, ax_clusters = plt.subplots(figsize=(8, 5))
scatter = ax_clusters.scatter(
    X[features[0]],
    X[features[1]],
    c=labels,
    cmap="tab10",
    alpha=0.75,
)
centroids_original = scaler.inverse_transform(model.cluster_centers_)
ax_clusters.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    marker="X",
    s=220,
    c="black",
    label="Centroides",
)
ax_clusters.set_title(f"Clusters de clientes (K={k_opt})")
ax_clusters.set_xlabel(features[0])
ax_clusters.set_ylabel(features[1])
ax_clusters.legend()
st.pyplot(fig_clusters)

st.subheader("Clasificacion de un nuevo cliente")
col1, col2 = st.columns(2)

with col1:
    value_1 = st.slider(
        features[0],
        min_value=float(X[features[0]].min()),
        max_value=float(X[features[0]].max()),
        value=float(X[features[0]].median()),
        step=1.0,
    )
with col2:
    value_2 = st.slider(
        features[1],
        min_value=float(X[features[1]].min()),
        max_value=float(X[features[1]].max()),
        value=float(X[features[1]].median()),
        step=1.0,
    )

new_point = np.array([[value_1, value_2]])
new_point_scaled = scaler.transform(new_point)
pred_cluster = int(model.predict(new_point_scaled)[0])

segment_name = cluster_names.get(pred_cluster, f"Cluster {pred_cluster}")
strategy = strategy_map.get(segment_name, "Aplicar estrategia de retencion y crecimiento personalizada.")

st.markdown(f"**Cluster asignado:** {pred_cluster}")
st.markdown(f"**Segmento:** {segment_name}")
st.markdown(f"**Estrategia recomendada:** {strategy}")

fig_new, ax_new = plt.subplots(figsize=(8, 5))
ax_new.scatter(X[features[0]], X[features[1]], c=labels, cmap="tab10", alpha=0.35)
ax_new.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    marker="X",
    s=220,
    c="black",
    label="Centroides",
)
ax_new.scatter(new_point[0, 0], new_point[0, 1], marker="D", s=180, c="red", edgecolor="black", label="Nuevo cliente")
ax_new.set_title("Nuevo cliente sobre clusters")
ax_new.set_xlabel(features[0])
ax_new.set_ylabel(features[1])
ax_new.legend()
st.pyplot(fig_new)
