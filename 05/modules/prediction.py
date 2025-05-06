import pandas as pd

def predict_with_model(model, new_data, x_cols):
    if not set(x_cols).issubset(set(new_data.columns)):
        raise ValueError("上传数据中缺少模型所需的字段")

    X_new = new_data[x_cols].dropna()
    y_pred = model.predict(X_new)
    new_data = new_data.copy()
    new_data["预测结果"] = y_pred
    return new_data
