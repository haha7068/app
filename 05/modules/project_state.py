import json
import pandas as pd

def save_project_state(filename, session_state):
    export_data = {}

    # 仅保存可序列化的对象
    for key in session_state.keys():
        value = session_state[key]
        if isinstance(value, pd.DataFrame):
            export_data[key] = {
                "_type": "dataframe",
                "data": value.to_csv(index=False)
            }
        elif isinstance(value, (str, int, float, list, dict, bool, type(None))):
            export_data[key] = {
                "_type": "value",
                "data": value
            }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

def load_project_state(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = json.load(f)

    restored = {}
    for key, item in content.items():
        if item["_type"] == "dataframe":
            from io import StringIO
            restored[key] = pd.read_csv(StringIO(item["data"]))
        else:
            restored[key] = item["data"]

    return restored
