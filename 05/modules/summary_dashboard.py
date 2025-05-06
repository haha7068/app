import plotly.graph_objects as go
import pandas as pd

def create_sample_summary_card(title, interpretation, fig=None):
    return {
        "title": title,
        "text": interpretation,
        "figure": fig
    }

def generate_interpretation_text(df, field, group_col):
    group_avg = df.groupby(group_col)[field].mean().sort_values(ascending=False)
    top_group = group_avg.index[0]
    bottom_group = group_avg.index[-1]
    diff = group_avg.iloc[0] - group_avg.iloc[-1]
    return f"{field} 在 {top_group} 中最高，在 {bottom_group} 中最低，差值为 {diff:.2f}。"
