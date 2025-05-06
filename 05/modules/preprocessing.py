def fill_missing(df, strategy="mean"):
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "drop":
        return df.dropna()
    else:
        raise ValueError("暂仅支持 'mean' 或 'drop' 策略")
import pandas as pd
import numpy as np
import plotly.express as px

def detect_outliers_iqr(df, columns, threshold=1.5):
    """使用IQR方法检测异常值，返回异常值布尔掩码"""
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    return outlier_mask

def remove_outliers(df, columns, threshold=1.5):
    """删除包含异常值的行"""
    mask = detect_outliers_iqr(df, columns, threshold)
    return df[~mask.any(axis=1)]

def visualize_missing_values(df):
    """生成缺失值比例的柱状图"""
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        fig = px.bar(x=missing.index, y=missing.values,
                     labels={'x': '字段', 'y': '缺失比例'},
                     title="各字段缺失值比例")
        return fig
    else:
        return None

def get_numeric_columns(df):
    """返回所有数值型字段列表"""
    return df.select_dtypes(include='number').columns.tolist()
def detect_outliers_summary(df, columns, threshold=1.5):
    """返回异常值所在的索引集合和列名"""
    outlier_rows = set()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)].index
        outlier_rows.update(outliers)
    return list(outlier_rows)

def plot_outlier_boxplot(df, columns):
    import plotly.express as px
    melted = df[columns].melt(var_name="变量", value_name="值")
    fig = px.box(melted, x="变量", y="值", points="all", title="异常值箱线图")
    return fig
