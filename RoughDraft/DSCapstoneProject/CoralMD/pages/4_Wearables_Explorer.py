from pathlib import Path

import pandas as pd
import streamlit as st

from utils.visualizations import wearable_summary, wearable_insights

# --- Path to wearable data ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "wearables"


@st.cache_data(show_spinner="Loading wearable datasets...")
def load_wearable_datasets():
    """Load all CSVs in data/wearables/ into a dict name -> DataFrame."""
    files = sorted(DATA_PATH.glob("*.csv"))
    datasets: dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        # normalize column names
        df.columns = df.columns.str.lower()
        datasets[f.stem] = df
    return datasets


# ========================= Streamlit UI =========================

st.set_page_config(page_title="CoralMD — Wearables Explorer", layout="wide")
st.title("⌚ Wearables Explorer")

try:
    datasets = load_wearable_datasets()
    if not datasets:
        raise FileNotFoundError
except FileNotFoundError:
    st.error(
        "No wearable CSV files found.\n\n"
        "Place one or more files (e.g., aw_fb_data.csv, data_for_weka_aw.csv) "
        "in `data/wearables/` inside the CoralMD project."
    )
    st.stop()

dataset_name = st.selectbox("Choose dataset", list(datasets.keys()))
df = datasets[dataset_name]

st.caption(f"`{dataset_name}.csv` — {len(df)} rows")

# pick which row to interpret as a patient
if len(df) > 1:
    row_index = st.number_input(
        "Select row to view as a single patient",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )
else:
    row_index = 0

patient = df.iloc[row_index]

st.subheader("Raw wearable row")
st.dataframe(patient.to_frame().T, use_container_width=True)

st.subheader("Clinical-style summary")
summary_df = wearable_summary(patient)
st.table(summary_df)

st.subheader("Lifestyle insights (prototype, not medical advice)")
for line in wearable_insights(patient):
    st.markdown(line)

st.caption("Columns expected (case-insensitive, best effort): "
           "`age, gender, height, weight, hear_rate, resting_heart, steps, "
           "calories, distance, activity`.")
