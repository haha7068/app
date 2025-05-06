import plotly.express as px
import pandas as pd
def plot_histogram(df, column):
    fig = px.histogram(df, x=column, title=f"{column} 的直方图")
    return fig
import plotly.express as px

def plot_fitted_vs_actual(y_true, y_pred):
    df_plot = pd.DataFrame({"实际值": y_true, "预测值": y_pred})
    fig = px.scatter(df_plot, x="实际值", y="预测值",
                     title="拟合图：实际值 vs 预测值",
                     trendline="ols")
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    df_plot = pd.DataFrame({"预测值": y_pred, "残差": residuals})
    fig = px.scatter(df_plot, x="预测值", y="残差",
                     title="残差图：残差 vs 预测值")
    return fig

def plot_logistic_prediction(X, y_true, y_pred_prob):
    df_plot = pd.DataFrame({
        "X": X.iloc[:, 0],  # 仅取第一列做展示（适用于单变量）
        "真实标签": y_true,
        "预测概率": y_pred_prob
    })
    fig = px.scatter(df_plot, x="X", y="预测概率", color="真实标签",
                     title="逻辑回归预测概率分布图",
                     labels={"X": "自变量", "预测概率": "患病概率"})
    return fig
import plotly.graph_objects as go

def plot_radar_chart(df, group_col, value_cols):
    """生成雷达图，支持多个组对比"""
    categories = value_cols
    fig = go.Figure()

    for group in df[group_col].unique():
        data = df[df[group_col] == group][value_cols].mean()
        fig.add_trace(go.Scatterpolar(
            r=data.values,
            theta=categories,
            fill="toself",
            name=str(group)
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="多变量雷达图",
        showlegend=True
    )
    return fig

def export_plotly_figure_to_png(fig, filename="chart.png"):
    fig.write_image(filename, format="png", engine="kaleido")
    with open(filename, "rb") as f:
        return f.read()

def export_dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")
