import pandas as pd
import os
from datetime import datetime
from config_paths import BASE_DIR, DATA_DIR, REPORTS_DIR

def save_session(data, participant_id):
    df = pd.DataFrame(data)
    df["participant_id"] = participant_id
    df["session_date"] = datetime.now()

    os.makedirs(DATA_DIR, exist_ok=True)

    path = os.path.join(DATA_DIR, f"auditory_test_{participant_id}.csv")

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(path, index=False)
    return path