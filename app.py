# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:44:28 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load saved KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Set title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Description
st.subheader("📊 Cluster Visualization (PCA Projection)")
st.markdown("This demo uses the **Iris dataset** and displays clustering results with PCA for visualization.")

# Load Iris dataset
iris = load_iris()
X = iris.data

# PCA transformation to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Predict using loaded model
labels = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='tab10')

# Plot centroids (transformed to PCA space)
centers_original = loaded_model.cluster_centers_
centers_pca = pca.transform(centers_original)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, alpha=0.75, label='Centroids')

ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show in Streamlit
st.pyplot(fig)
