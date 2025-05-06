from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

def evaluate_regression(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

def evaluate_classification(y_true, y_pred):
    return {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred),
        "召回率": recall_score(y_true, y_pred),
        "F1分数": f1_score(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["负类", "正类"]
    z_text = [[str(cell) for cell in row] for row in cm]
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels, annotation_text=z_text,
        colorscale="Blues", showscale=True
    )
    fig.update_layout(title="混淆矩阵")
    return fig
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC曲线"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="基线", line=dict(dash="dash")))
    fig.update_layout(
        title=f"ROC曲线（AUC = {auc_score:.2f}）",
        xaxis_title="假阳性率 (FPR)",
        yaxis_title="真阳性率 (TPR)"
    )
    return fig, auc_score
