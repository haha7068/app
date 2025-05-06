from scipy.stats import ttest_ind

def perform_t_test(df, numeric_col, group_col):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("t检验需要恰好两个分组")
    a = df[df[group_col] == groups[0]][numeric_col]
    b = df[df[group_col] == groups[1]][numeric_col]
    stat, p = ttest_ind(a, b, nan_policy='omit')
    return stat, p
from scipy.stats import mannwhitneyu, kruskal, pearsonr, spearmanr
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def perform_mannwhitneyu(df, numeric_col, group_col):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Mann–Whitney U检验需要恰好两个分组")
    a = df[df[group_col] == groups[0]][numeric_col]
    b = df[df[group_col] == groups[1]][numeric_col]
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return stat, p

def perform_kruskal(df, numeric_col, group_col):
    groups = df[group_col].dropna().unique()
    data = [df[df[group_col] == g][numeric_col] for g in groups]
    stat, p = kruskal(*data)
    return stat, p

def compute_correlations(df, method="pearson"):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr(method=method)
    return corr

def plot_correlation_heatmap(corr_matrix):
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="变量相关性热力图",
        labels=dict(x="变量", y="变量", color="相关系数")
    )
    return fig
import statsmodels.api as sm
import pandas as pd

def linear_regression(df, x_col, y_col):
    X = df[[x_col]].dropna()
    y = df[y_col].loc[X.index]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def logistic_regression(df, x_col, y_col):
    X = df[[x_col]].dropna()
    y = df[y_col].loc[X.index]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    return model
def multivariable_linear_regression(df, x_cols, y_col):
    X = df[x_cols].dropna()
    y = df[y_col].loc[X.index]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def multivariable_logistic_regression(df, x_cols, y_col):
    X = df[x_cols].dropna()
    y = df[y_col].loc[X.index]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    return model
from scipy.stats import shapiro
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm

def test_normality(df, column):
    """Shapiro-Wilk检验"""
    stat, p = shapiro(df[column].dropna())
    return stat, p

def plot_qq(df, column):
    """绘制QQ图"""
    fig = go.Figure()
    data = df[column].dropna()
    qq = sm.ProbPlot(data)
    theoretical, sample = qq.theoretical_quantiles, qq.sample_quantiles
    fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers", name="QQ点"))
    fig.add_trace(go.Line(x=theoretical, y=theoretical, name="参考线"))
    fig.update_layout(title=f"{column} 的QQ图", xaxis_title="理论分位数", yaxis_title="样本分位数")
    return fig
