import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

def perform_kmeans(df, features, n_clusters=3, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].dropna())
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)
    pca_df["聚类类别"] = cluster_labels.astype(str)
    return pca_df, model.inertia_

def plot_kmeans_pca(pca_df):
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="聚类类别",
        title="KMeans聚类结果（PCA降维可视化）",
        labels={"聚类类别": "类别"}
    )
    return fig
