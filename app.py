# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:44:28 2025

@author: LAB
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# App title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Sidebar - Number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)

# PCA for 2D visualization
pca = PCA(2)
X_pca = pca.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50)

# Create proper legend
colors = cm.get_cmap('tab10', k)
legend_labels = [f"Cluster {i}" for i in range(k)]
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors(i), markersize=10)
           for i in range(k)]
ax.legend(handles, legend_labels)

# Labels and title
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Display plot
st.pyplot(fig)

