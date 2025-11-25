import json
import io

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.db_connect import add_patient, add_upload, add_prediction
from utils.ml_models import compute_mock_risk
from utils.visualizations import risk_card_html, simple_trend, wearable_summary, wearable_insights

st.set_page_config(page_title="CoralMD — Patient", layout="wide")
st.title("🧍 Patient Home")

# --- create / select patient record ---
name = st.text_input("Your name", value="Alex Doe")

if st.button("Create / use my record"):
    pid = add_patient(name)
    st.session_state["patient_id"] = pid
    st.success(f"Patient record ready (id={pid}).")

pid = st.session_state.get("patient_id")
if not pid:
    st.info("Enter your name and click the button to begin.")
    st.stop()

st.divider()
st.subheader("Upload your data")

col1, col2, col3 = st.columns(3)
genomics_file = col1.file_uploader("Genomics (CSV summary)", type=["csv"])
wear_file     = col2.file_uploader("Wearables (CSV)", type=["csv"])
ehr_file      = col3.file_uploader("EHR / labs (CSV)", type=["csv"])

def read_csv_safe(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        return pd.read_csv(io.BytesIO(upload.read()))

dfs = []
df_g = df_w = df_e = None  # keep references for later

if genomics_file:
    df_g = read_csv_safe(genomics_file)
    if df_g is not None:
        dfs.append(df_g)
        add_upload(pid, "genomics", genomics_file.name)

if wear_file:
    df_w = read_csv_safe(wear_file)
    if df_w is not None:
        dfs.append(df_w)
        add_upload(pid, "wearables", wear_file.name)

if ehr_file:
    df_e = read_csv_safe(ehr_file)
    if df_e is not None:
        dfs.append(df_e)
        add_upload(pid, "ehr", ehr_file.name)

if not dfs:
    st.warning("Upload at least one CSV file to see a preliminary risk estimate.")
    st.stop()

# merge all columns (very forgiving for prototype)
df = pd.concat(dfs, axis=1)
df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate col names

st.subheader("Preview of your data")
st.dataframe(df.head(), use_container_width=True)

# --- compute mock risk ---
st.subheader("Preliminary risk estimate (prototype)")
risk, details = compute_mock_risk(df)
components.html(risk_card_html(risk), height=150)
add_prediction(pid, "mock_multimodal", float(risk), json.dumps(details))


# --- detailed wearables interpretation (if provided) ---
st.subheader("Wearables interpretation (optional)")

if df_w is not None:
    df_w_local = df_w.copy()
    df_w_local.columns = df_w_local.columns.str.lower()

    st.caption(f"Wearables file `{wear_file.name}` — {len(df_w_local)} rows detected.")

    max_idx = len(df_w_local) - 1
    row_index = 0
    if max_idx > 0:
        row_index = st.number_input(
            "Select row to view as **you**",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
        )

    patient_row = df_w_local.iloc[row_index]

    st.markdown("### Raw wearable row")
    st.dataframe(patient_row.to_frame().T, use_container_width=True)

    st.markdown("### Clinical-style summary")
    summary_df = wearable_summary(patient_row)
    st.table(summary_df)

    st.markdown("### Lifestyle insights (prototype, not medical advice)")
    for line in wearable_insights(patient_row):
        st.markdown(line)
else:
    st.caption("Upload a wearables CSV to see a detailed lifestyle summary.")


# --- simple trend plot if we can guess a time + value column ---
time_col = None
for candidate in ["date", "time", "timestamp", "day"]:
    if candidate in df.columns:
        time_col = candidate
        break

value_col = None
for candidate in ["glucose", "hr_resting", "steps", "cholesterol"]:
    if candidate in df.columns:
        value_col = candidate
        break

if time_col and value_col:
    fig = simple_trend(df, time_col, value_col, f"{value_col} over time")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Add columns like 'date' and 'glucose' in your CSVs to see trends.")
