import pandas as pd

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("仅支持 CSV 和 Excel 文件格式")
    return df
