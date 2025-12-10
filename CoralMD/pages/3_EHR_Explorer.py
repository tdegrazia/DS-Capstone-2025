# 3_EHR_Explorer.py

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.db_connect import get_conn, list_patients

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ehr"


# ---------------- DB helpers (same logic as practitioner home) ----------------

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
    """
    Given a patient's uploads df, load the most recent CSV for the given kind
    (e.g. 'ehr', 'genomics', 'wearables').
    """
    if uploads.empty:
        return None

    sub = uploads[uploads["kind"] == kind]
    if sub.empty:
        return None

    latest = sub.sort_values("uploaded_at", ascending=False).iloc[0]
    path = Path(latest["filename"])

    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


# ---------------- Local sample loader (fallback) ----------------

@st.cache_data(show_spinner="Loading sample EHR datasets...")
def load_ehr_datasets() -> Dict[str, pd.DataFrame]:
    """Load all CSVs in data/ehr/ into a dict name -> DataFrame."""
    files = sorted(DATA_PATH.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No EHR CSV files found under {DATA_PATH}")
    datasets: Dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        datasets[f.stem] = df
    return datasets


def ehr_lab_flags(df: pd.DataFrame) -> List[str]:
    """Compute simple cardio/metabolic lab flags (same thresholds as elsewhere)."""
    flags: List[str] = []
    if not {"lab_name", "lab_value"}.issubset(df.columns):
        return flags

    labs = df[["lab_name", "lab_value"]].dropna().copy()
    labs["lab_name"] = labs["lab_name"].astype(str).str.lower()

    def mean_lab(keyword: str) -> Optional[float]:
        sel = labs[labs["lab_name"].str.contains(keyword, na=False)]
        if sel.empty:
            return None
        return float(sel["lab_value"].astype(float).mean())

    ldl = mean_lab("ldl")
    if ldl is not None:
        if ldl >= 160:
            flags.append("Very high LDL cholesterol (≥160 mg/dL)")
        elif ldl >= 130:
            flags.append("High LDL cholesterol (≥130 mg/dL)")

    a1c = mean_lab("hba1c") or mean_lab("a1c")
    if a1c is not None:
        if a1c >= 6.5:
            flags.append("Diabetes-range HbA1c (≥6.5%)")
        elif 5.7 <= a1c < 6.5:
            flags.append("Prediabetes HbA1c (5.7–6.4%)")

    creat = mean_lab("creatinine")
    if creat is not None and creat >= 1.3:
        flags.append("Elevated creatinine (≥1.3 mg/dL) – kidney strain")

    return flags


# ========================= UI =========================

st.set_page_config(page_title="CoralMD — EHR Explorer", layout="wide")
st.title("🏥 EHR Explorer")
st.caption(
    "Deep dive into clinical encounters, diagnoses, and lab patterns for a specific patient "
    "or a sample dataset."
)

# -------- Choose data source: patient vs sample file --------
source = st.radio(
    "Choose data source",
    ["By patient id (from uploads)", "Sample file from data/ehr"],
    index=0,
)

df: Optional[pd.DataFrame] = None
data_label = ""

if source.startswith("By patient"):
    # load patient list
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
    df = load_latest_csv(uploads, "ehr")

    if df is None or df.empty:
        st.error("No usable EHR CSV found for this patient yet.")
        st.stop()

    df.columns = df.columns.str.lower()
    data_label = f"EHR for patient {patient_id}"
else:
    # sample files
    try:
        ehr_datasets = load_ehr_datasets()
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\n"
            "Create a folder `data/ehr/` and place one or more EHR CSVs in it "
            "(columns like age, gender, admission_id, diagnosis_name, lab_name, lab_value, ...)."
        )
        st.stop()

    dataset_name = st.selectbox("Choose EHR sample dataset", list(ehr_datasets.keys()))
    df = ehr_datasets[dataset_name].copy()
    data_label = f"`{dataset_name}.csv`"

st.caption(f"{data_label} — {len(df)} rows")
st.divider()

# ========================= Demographics & admissions =========================

col_demo, col_adm = st.columns([1, 2])

with col_demo:
    st.subheader("Demographics")

    age = df["age"].iloc[0] if "age" in df.columns else "N/A"
    gender = df["gender"].iloc[0] if "gender" in df.columns else "N/A"

    st.metric("Age (anchor)", value=age)
    st.metric("Gender", value=gender)

with col_adm:
    st.subheader("Admissions & clinical rows")

    adm_col = None
    for candidate in ["admission_id", "admission", "visit_id", "encounter_id"]:
        if candidate in df.columns:
            adm_col = candidate
            break

    if adm_col:
        n_adm = int(df[adm_col].nunique())
        st.metric("Number of admissions", value=n_adm)
    else:
        st.write("No explicit admission ID column; treating rows as clinical events.")
        n_adm = "—"

    st.write("Preview of raw EHR rows:")
    st.dataframe(df.head(), use_container_width=True)

st.divider()

# ========================= Diagnosis overview =========================

st.subheader("🧾 Diagnosis profile")

diag_col = None
for candidate in ["diagnosis_name", "diagnosis", "diag"]:
    if candidate in df.columns:
        diag_col = candidate
        break

if diag_col:
    diag_series = df[diag_col].dropna().astype(str)
    if diag_series.empty:
        st.write("No diagnosis values found.")
    else:
        counts = diag_series.value_counts().reset_index()
        counts.columns = ["diagnosis", "count"]

        st.write(f"{counts.shape[0]} unique diagnoses.")
        st.dataframe(counts.head(15), use_container_width=True)

        top_n = st.slider("Show top N diagnoses", min_value=3, max_value=20, value=10)
        fig_diag = px.bar(
            counts.head(top_n),
            x="diagnosis",
            y="count",
            title="Most frequent diagnoses",
        )
        fig_diag.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_diag, use_container_width=True)
else:
    st.write("No diagnosis column detected (expected something like `diagnosis_name`).")

st.divider()

# ========================= Lab overview =========================

st.subheader("🧪 Lab snapshot")

if {"lab_name", "lab_value"}.issubset(df.columns):
    labs = df[["lab_name", "lab_value"]].dropna().copy()
    labs["lab_name"] = labs["lab_name"].astype(str)

    lab_summary = (
        labs.groupby("lab_name")["lab_value"]
        .mean()
        .reset_index()
        .sort_values("lab_value", ascending=False)
    )

    st.write("Average lab values:")
    st.dataframe(lab_summary.head(20), use_container_width=True)

    # Focused bar chart for cardiometabolic labs
    cardiomet_keywords = [
        "ldl",
        "hdl",
        "total cholesterol",
        "triglyceride",
        "hba1c",
        "a1c",
        "glucose",
        "creatinine",
    ]
    mask = lab_summary["lab_name"].str.lower().str.contains("|".join(cardiomet_keywords), na=False)
    cardio_labs = lab_summary[mask]

    if not cardio_labs.empty:
        fig_labs = px.bar(
            cardio_labs,
            x="lab_name",
            y="lab_value",
            title="Key cardiometabolic lab means",
        )
        fig_labs.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_labs, use_container_width=True)

    # Risk flags using same thresholds as the risk model
    flags = ehr_lab_flags(df)
    if flags:
        st.markdown("**Highlighted lab risk flags (prototype):**")
        for f in flags:
            st.markdown(f"- {f}")
    else:
        st.caption("No cardiometabolic lab flags detected with current thresholds.")
else:
    st.write("No lab columns detected (expected `lab_name` and `lab_value`).")

st.divider()

# ========================= Lab trends over time (optional) =========================

st.subheader("📈 Lab trends over time (if dates available)")

date_col = None
for candidate in ["lab_date", "date", "timestamp"]:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col and "lab_name" in df.columns and "lab_value" in df.columns:
    df_time = df[["lab_name", "lab_value", date_col]].dropna().copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
    df_time = df_time.dropna(subset=[date_col])
    if df_time.empty:
        st.write("Lab date column found but no valid timestamps.")
    else:
        lab_choices = sorted(df_time["lab_name"].astype(str).unique())
        selected_lab = st.selectbox("Choose a lab to trend", lab_choices)

        sub = df_time[df_time["lab_name"].astype(str) == selected_lab].copy()
        sub = sub.sort_values(date_col)

        fig_time = px.line(
            sub,
            x=date_col,
            y="lab_value",
            title=f"{selected_lab} over time",
        )
        st.plotly_chart(fig_time, use_container_width=True)
else:
    st.caption("Add a `lab_date` column to see lab trends over time.")
