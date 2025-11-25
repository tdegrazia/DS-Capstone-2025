import pandas as pd
import streamlit as st

from utils.db_connect import list_patients, get_conn

st.set_page_config(page_title="CoralMD — Practitioner", layout="wide")
st.title("🩺 Practitioner Home")

st.subheader("Patient registry")

rows = list_patients()
df = pd.DataFrame(rows, columns=["id", "name", "last_risk"])
if df.empty:
    st.info("No patients yet — ask a patient to create a record and upload data.")
    st.stop()

st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("Open a specific patient")

patient_ids = df["id"].tolist()
selected_id = st.selectbox("Select patient ID", patient_ids)

if st.button("Open patient"):
    with get_conn() as conn:
        uploads = pd.read_sql(
            "SELECT kind, filename, uploaded_at FROM uploads WHERE patient_id=? ORDER BY uploaded_at DESC",
            conn,
            params=(selected_id,),
        )
        preds = pd.read_sql(
            "SELECT model_name, risk_score, created_at FROM predictions WHERE patient_id=? ORDER BY created_at DESC",
            conn,
            params=(selected_id,),
        )

    st.write(f"### Patient {selected_id} — uploads")
    if uploads.empty:
        st.write("No uploads yet.")
    else:
        st.dataframe(uploads, use_container_width=True)

    st.write("### Model outputs")
    if preds.empty:
        st.write("No predictions yet.")
    else:
        st.dataframe(preds, use_container_width=True)
