# 4_Wearables_Explorer.py

from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.visualizations import wearable_summary, wearable_insights
from utils.db_connect import get_conn, list_patients

# --- Paths for sample wearable data ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "wearables"


# ---------------- DB helpers ----------------

def get_patient_uploads(patient_id: int) -> pd.DataFrame:
    """Return uploads df for a given patient_id."""
    with get_conn() as conn:
        uploads = pd.read_sql(
            """
            SELECT kind, filename, uploaded_at
            FROM uploads
            WHERE patient_id = ?
            ORDER BY uploaded_at DESC
            """,
            conn,
            params=(patient_id,),
        )
    return uploads


def load_latest_csv(uploads: pd.DataFrame, kind: str) -> Optional[pd.DataFrame]:
    if uploads.empty:
        return None
    sub = uploads[uploads["kind"] == kind]
    if sub.empty:
        return None
    latest = sub.sort_values("uploaded_at", ascending=False).iloc[0]
    path = Path(latest["filename"])
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------- Sample loader ----------------

@st.cache_data(show_spinner="Loading sample wearable datasets...")
def load_wearable_datasets() -> Dict[str, pd.DataFrame]:
    """Load all CSVs in data/wearables/ into a dict name -> DataFrame."""
    files = sorted(DATA_PATH.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No wearable CSV files found under {DATA_PATH}")
    datasets: Dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        datasets[f.stem] = df
    return datasets


def physiologic_flags(df: pd.DataFrame) -> List[str]:
    """Heuristic lifestyle/physiology flags aligned with risk model thresholds."""
    flags: List[str] = []

    def mean(col_candidates: List[str]) -> Optional[float]:
        for c in col_candidates:
            if c in df.columns:
                s = df[c].dropna()
                if not s.empty:
                    return float(s.astype(float).mean())
        return None

    steps = mean(["steps", "step_count"])
    sleep = mean(["sleep_hours", "sleep", "sleep_duration"])
    rhr = mean(["resting_hr", "resting_heart", "resting_heart_rate"])
    hrv = mean(["hrv"])

    if steps is not None:
        if steps < 4000:
            flags.append("Very low activity (<4k steps/day on average)")
        elif steps < 7000:
            flags.append("Below activity target (~7k steps/day)")
        elif steps >= 10000:
            flags.append("High daily activity (≥10k steps/day)")

    if sleep is not None:
        if sleep < 6:
            flags.append("Short sleep (avg <6 hours)")
        elif sleep < 7:
            flags.append("Slightly short sleep (avg <7 hours)")

    if rhr is not None:
        if rhr > 90:
            flags.append("High resting HR (>90 bpm)")
        elif rhr > 80:
            flags.append("Elevated resting HR (>80 bpm)")

    if hrv is not None:
        if hrv < 25:
            flags.append("Low HRV (stress / low recovery)")
        elif hrv > 40:
            flags.append("High HRV")

    return flags


# ========================= Streamlit UI =========================

st.set_page_config(page_title="CoralMD — Wearables Explorer", layout="wide")
st.title("⌚ Wearables / Physiology Explorer")
st.caption("Explore daily activity, sleep, heart rate, and HRV over time for a patient or sample file.")

# -------- Choose data source --------
source = st.radio(
    "Choose data source",
    ["By patient id (from uploads)", "Sample file from data/wearables"],
    index=0,
)

df: Optional[pd.DataFrame] = None
data_label = ""

if source.startswith("By patient"):
    rows = list_patients()
    if not rows:
        st.error("No patients in the registry yet.")
        st.stop()

    df_patients = pd.DataFrame(rows, columns=["id", "name", "last_risk"])
    option = st.selectbox(
        "Choose patient",
        [f"{row['name']} (id={row['id']})" for _, row in df_patients.iterrows()],
    )
    patient_id = int(option.split("id=")[-1].rstrip(")"))

    uploads = get_patient_uploads(patient_id)
    df = load_latest_csv(uploads, "wearables")

    if df is None or df.empty:
        st.error("No usable wearables CSV found for this patient yet.")
        st.stop()

    df.columns = df.columns.str.lower()
    data_label = f"Wearables for patient {patient_id}"
else:
    try:
        datasets = load_wearable_datasets()
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\n"
            "Create a folder `data/wearables/` and place one or more CSVs there "
            "(columns like date, steps, sleep_hours, resting_hr, hrv)."
        )
        st.stop()

    dataset_name = st.selectbox("Choose wearable sample dataset", list(datasets.keys()))
    df = datasets[dataset_name].copy()
    data_label = f"`{dataset_name}.csv`"

st.caption(f"{data_label} — {len(df)} days")

# Ensure date is parsed & sorted
date_col = None
for candidate in ["date", "day", "timestamp"]:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
else:
    st.warning("No `date` column found — trends will be limited.")
    date_col = None

st.divider()

# ========================= Summary metrics =========================

st.subheader("Summary metrics")

def mean(col_candidates: List[str]) -> Optional[float]:
    for c in col_candidates:
        if c in df.columns:
            s = df[c].dropna()
            if not s.empty:
                return float(s.astype(float).mean())
    return None

steps = mean(["steps", "step_count"])
sleep = mean(["sleep_hours", "sleep", "sleep_duration"])
rhr = mean(["resting_hr", "resting_heart", "resting_heart_rate"])
hrv = mean(["hrv"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg steps/day", f"{steps:.0f}" if steps is not None else "N/A")
with col2:
    st.metric("Avg sleep (h)", f"{sleep:.1f}" if sleep is not None else "N/A")
with col3:
    st.metric("Avg resting HR (bpm)", f"{rhr:.0f}" if rhr is not None else "N/A")
with col4:
    st.metric("Avg HRV (ms)", f"{hrv:.0f}" if hrv is not None else "N/A")

flags = physiologic_flags(df)
if flags:
    st.markdown("**Physiology / lifestyle flags (prototype):**")
    for f in flags:
        st.markdown(f"- {f}")
else:
    st.caption("No major flags detected under current thresholds.")

st.divider()

# ========================= Time-series plots =========================

st.subheader("📈 Time-series views")

if date_col:
    if "steps" in df.columns:
        fig_steps = px.line(df, x=date_col, y="steps", title="Steps per day")
        st.plotly_chart(fig_steps, use_container_width=True)

    for colname, label in [
        ("sleep_hours", "Sleep duration (hours)"),
        ("resting_hr", "Resting heart rate (bpm)"),
        ("hrv", "HRV (ms)"),
    ]:
        if colname in df.columns:
            fig = px.line(df, x=date_col, y=colname, title=label)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Add a `date` column to see trends over time.")

st.divider()

# ========================= Single-day / last-day snapshot =========================

st.subheader("🧍 Snapshot: last day in file")

if not df.empty:
    last_row = df.iloc[-1]
    st.caption("Interpreting the last row as the most recent day for this individual.")

    st.markdown("**Raw values**")
    st.dataframe(last_row.to_frame().T, use_container_width=True)

    st.markdown("**Clinical-style summary**")
    summary_df = wearable_summary(last_row)
    st.table(summary_df)

    st.markdown("**Lifestyle insights (prototype, not medical advice)**")
    for line in wearable_insights(last_row):
        st.markdown(line)
else:
    st.write("No rows in wearable dataset.")
