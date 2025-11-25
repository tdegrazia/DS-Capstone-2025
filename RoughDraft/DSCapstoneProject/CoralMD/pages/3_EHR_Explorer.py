import os
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Path to MIMIC demo data (relative to project root) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mimic_hosp"


@st.cache_data(show_spinner="Loading MIMIC demo tables...")
def load_data():
    patients = pd.read_csv(DATA_PATH / "patients.csv.gz")
    admissions = pd.read_csv(DATA_PATH / "admissions.csv.gz")
    diagnoses = pd.read_csv(DATA_PATH / "diagnoses_icd.csv.gz")
    diag_dict = pd.read_csv(DATA_PATH / "d_icd_diagnoses.csv.gz")
    labs = pd.read_csv(DATA_PATH / "labevents.csv.gz")
    lab_items = pd.read_csv(DATA_PATH / "d_labitems.csv.gz")
    return patients, admissions, diagnoses, diag_dict, labs, lab_items


def ehr_summary(
    subject_id: int,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    diag_dict: pd.DataFrame,
    labs: pd.DataFrame,
    lab_items: pd.DataFrame,
):
    if subject_id not in patients["subject_id"].values:
        return None

    p = patients[patients["subject_id"] == subject_id].iloc[0]
    a = admissions[admissions["subject_id"] == subject_id]
    diag = diagnoses[diagnoses["subject_id"] == subject_id].merge(
        diag_dict[["icd_code", "long_title"]], on="icd_code", how="left"
    )
    lab = labs[labs["subject_id"] == subject_id].merge(
        lab_items[["itemid", "label"]], on="itemid", how="left"
    )

    demo = {
        "Subject ID": int(subject_id),
        "Gender": p.get("gender", "N/A"),
        "Anchor Age": p.get("anchor_age", "N/A"),
        "Number of admissions": int(a.shape[0]),
    }

    diag_titles = []
    if not diag.empty:
        diag_titles = list(diag["long_title"].dropna().unique()[:15])

    lab_summary = None
    if not lab.empty and "valuenum" in lab.columns:
        lab_summary = (
            lab.groupby("label")["valuenum"]
            .mean()
            .reset_index()
            .sort_values(by="valuenum", ascending=False)
            .head(10)
        )

    note = None
    if diag_titles:
        note = (
            f"Based on records such as '{diag_titles[0]}', "
            "this patient’s history may warrant closer monitoring."
        )

    return demo, diag_titles, lab_summary, note


# ========================= Streamlit UI =========================

st.set_page_config(page_title="CoralMD — EHR Explorer", layout="wide")
st.title("🏥 EHR Explorer (MIMIC-IV demo)")

try:
    patients, admissions, diagnoses, diag_dict, labs, lab_items = load_data()
except FileNotFoundError:
    st.error(
        "MIMIC demo files not found.\n\n"
        "Expected them under `data/mimic_hosp/` inside the CoralMD project."
    )
    st.stop()

st.caption("Data: MIMIC-IV Clinical Database Demo (PhysioNet)")

all_ids = patients["subject_id"].unique()
default_id = int(all_ids[0])

col_left, col_right = st.columns([1, 3])

with col_left:
    subject_id = st.number_input(
        "Enter patient (subject_id)",
        min_value=int(all_ids.min()),
        max_value=int(all_ids.max()),
        value=default_id,
        step=1,
    )
    if st.button("Lookup patient"):
        st.session_state["ehr_subject_id"] = int(subject_id)

subject = st.session_state.get("ehr_subject_id", default_id)

summary = ehr_summary(
    subject,
    patients,
    admissions,
    diagnoses,
    diag_dict,
    labs,
    lab_items,
)

if summary is None:
    st.warning(f"No patient found with ID {subject}.")
    st.stop()

demo, diag_titles, lab_summary, note = summary

with col_right:
    st.subheader(f"Patient {subject} — demographics")
    st.write(
        {
            "Gender": demo["Gender"],
            "Anchor age": demo["Anchor Age"],
            "Number of admissions": demo["Number of admissions"],
        }
    )

st.divider()

st.subheader("🧾 Diagnosis history")
if not diag_titles:
    st.write("No diagnosis data available.")
else:
    for d in diag_titles:
        st.markdown(f"- {d}")

st.subheader("🧪 Lab snapshot (mean values)")
if lab_summary is None or lab_summary.empty:
    st.write("No lab data available.")
else:
    st.dataframe(lab_summary, use_container_width=True)

if note:
    st.subheader("💡 Clinical note (prototype, not medical advice)")
    st.write(note)

st.caption("CoralMD prototype view built on MIMIC-IV demo features.")
