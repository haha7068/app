import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def filter_feature_selection(df, x_cols, y_col, k=5):
    X = df[x_cols].dropna()
    y = df[y_col].loc[X.index]
    selector = SelectKBest(score_func=f_classif, k=min(k, len(x_cols)))
    selector.fit(X, y)
    scores = selector.scores_
    result = pd.DataFrame({
        "字段": x_cols,
        "F检验得分": scores
    }).sort_values(by="F检验得分", ascending=False)
    return result

def wrapper_feature_importance(df, x_cols, y_col):
    X = df[x_cols].dropna()
    y = df[y_col].loc[X.index]
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    result = pd.DataFrame({
        "字段": x_cols,
        "重要性得分": importances
    }).sort_values(by="重要性得分", ascending=False)
    return result
