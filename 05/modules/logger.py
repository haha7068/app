import datetime

def log_action(action_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "log_history" not in globals():
        globals()["log_history"] = []
    globals()["log_history"].append(f"[{timestamp}] {action_text}")

def get_log():
    return globals().get("log_history", [])
def clear_log():
    globals()["log_history"] = []

def export_log():
    log = get_log()
    return "\n".join(log)
