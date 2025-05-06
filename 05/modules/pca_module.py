import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def compute_pca(df, features, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].dropna())
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_df, pca.explained_variance_ratio_

def plot_pca_2d(pca_df, color=None):
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color=color,
        title="PCA降维2D可视化",
        labels={"PC1": "主成分1", "PC2": "主成分2"}
    )
    return fig
