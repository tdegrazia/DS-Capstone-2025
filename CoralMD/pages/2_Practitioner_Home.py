# 2_Practitioner_Home.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json  # NEW

import numpy as np
import pandas as pd
import streamlit as st

from utils.db_connect import list_patients, get_conn

# ----------------------------
# Helper: safe page switching
# ----------------------------

def go_to_page(page_script: str):
    """
    Try to switch to another Streamlit page by script name.
    If st.switch_page is unavailable or fails, show a gentle message instead.
    """
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(page_script)
        except Exception:
            st.warning(
                f"Could not auto-switch to `{page_script}`. "
                "Please open it from the sidebar."
            )
    else:
        st.info(
            "This Streamlit version does not support programmatic page switching. "
            "Use the sidebar to open the detailed explorer pages."
        )

# ----------------------------
# Helpers: load patient uploads
# ----------------------------

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


def get_patient_predictions(patient_id: int) -> pd.DataFrame:
    """Return predictions df for a given patient_id."""
    with get_conn() as conn:
        preds = pd.read_sql(
            """
            SELECT model_name, risk_score, details_json, created_at
            FROM predictions
            WHERE patient_id = ?
            ORDER BY created_at DESC
            """,
            conn,
            params=(patient_id,),
        )
    return preds


def load_latest_csv(uploads: pd.DataFrame, kind: str) -> Optional[pd.DataFrame]:
    """
    Given a patient's uploads df, load the most recent CSV for the given kind
    (e.g. 'ehr', 'genomics', 'wearables').
    Assumes uploads.filename contains an absolute or project-relative path.
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
        # If reading fails for any reason, fail gracefully.
        return None


# ----------------------------
# EHR summary helpers
# ----------------------------

def summarize_ehr(df_ehr: Optional[pd.DataFrame]) -> Dict:
    """
    Compute a compact, practitioner-facing EHR summary.
    Falls back to N/A when columns are missing.
    """
    summary = {
        "has_data": False,
        "n_rows": 0,
        "age": "N/A",
        "gender": "N/A",
        "n_admissions": "N/A",
        "n_diagnoses": "N/A",
        "key_diagnoses": [],
        "flags": [],
    }

    if df_ehr is None or df_ehr.empty:
        return summary

    df = df_ehr.copy()
    df.columns = [c.lower() for c in df.columns]
    summary["has_data"] = True
    summary["n_rows"] = len(df)

    # Age & gender
    if "age" in df.columns:
        summary["age"] = df["age"].iloc[0]
    elif "anchor_age" in df.columns:
        summary["age"] = df["anchor_age"].iloc[0]

    if "gender" in df.columns:
        summary["gender"] = df["gender"].iloc[0]

    # Admissions
    adm_col = None
    for candidate in ["admission_id", "admission", "visit_id", "encounter_id"]:
        if candidate in df.columns:
            adm_col = candidate
            break
    if adm_col:
        summary["n_admissions"] = int(df[adm_col].nunique())
    else:
        # fall back to treating rows as "clinical rows"
        summary["n_admissions"] = "—"

    # Diagnoses
    diag_col = None
    for candidate in ["diagnosis", "diagnosis_name", "icd_code", "icd10", "diag"]:
        if candidate in df.columns:
            diag_col = candidate
            break

    if diag_col:
        diag_series = df[diag_col].dropna().astype(str)
        summary["n_diagnoses"] = int(diag_series.nunique())
        # pick up to 3 most frequent diagnoses
        key_diag = (
            diag_series.value_counts()
            .head(3)
            .index.tolist()
        )
        summary["key_diagnoses"] = key_diag

    # Very simple "risk flags" based on lab-style columns if present
    # Expect something like: lab_name, lab_value
    if "lab_name" in df.columns and "lab_value" in df.columns:
        lab_df = df[["lab_name", "lab_value"]].dropna()
        # try to aggregate
        lab_mean = (
            lab_df.groupby("lab_name")["lab_value"]
            .mean()
            .reset_index()
        )

        def find_lab(keyword: str) -> Optional[float]:
            sel = lab_mean[lab_mean["lab_name"].str.contains(keyword, case=False, na=False)]
            if sel.empty:
                return None
            return float(sel["lab_value"].iloc[0])

        flags: List[str] = []

        ldl = find_lab("ldl")
        if ldl is not None:
            if ldl >= 160:
                flags.append("Very high LDL cholesterol")
            elif ldl >= 130:
                flags.append("High LDL cholesterol")

        a1c = find_lab("a1c")
        if a1c is None:
            a1c = find_lab("hba1c")
        if a1c is not None:
            if a1c >= 6.5:
                flags.append("Diabetes-range HbA1c")
            elif 5.7 <= a1c < 6.5:
                flags.append("Prediabetes HbA1c")

        creat = find_lab("creatinine")
        if creat is not None and creat >= 1.3:
            flags.append("Elevated creatinine (renal risk)")

        summary["flags"] = flags

    return summary


# ----------------------------
# Genomics summary helpers
# ----------------------------

KEY_GENES = ["APOE", "LDLR", "PCSK9", "TCF7L2", "FTO", "MC4R"]


def summarize_genomics(df_gen: Optional[pd.DataFrame]) -> Dict:
    summary = {
        "has_data": False,
        "n_variants": 0,
        "n_high_impact": 0,
        "n_cardiomet": 0,
        "key_genes": [],
        "flags": [],
    }

    if df_gen is None or df_gen.empty:
        return summary

    df = df_gen.copy()
    df.columns = [c.lower() for c in df.columns]
    summary["has_data"] = True
    summary["n_variants"] = len(df)

    # Columns
    gene_col = "gene" if "gene" in df.columns else None
    cond_col = None
    for candidate in ["condition", "trait", "phenotype"]:
        if candidate in df.columns:
            cond_col = candidate
            break
    path_col = None
    for candidate in ["pathogenicity_score", "score", "impact_score"]:
        if candidate in df.columns:
            path_col = candidate
            break

    # High-impact definition
    high_mask = pd.Series(False, index=df.index)
    if path_col:
        high_mask |= df[path_col].astype(float) >= 0.8
    if "effect" in df.columns:
        high_mask |= df["effect"].astype(str).str.lower().isin(
            ["pathogenic", "likely_pathogenic", "likely pathogenic"]
        )
    summary["n_high_impact"] = int(high_mask.sum())

    # Cardiometabolic subset
    cardiomet_keywords = [
        "coronary",
        "cad",
        "myocardial",
        "hypercholesterolemia",
        "lipid",
        "cholesterol",
        "diabetes",
        "obesity",
        "hypertension",
    ]
    cardiomet_mask = pd.Series(False, index=df.index)
    if cond_col:
        s = df[cond_col].astype(str).str.lower()
        for kw in cardiomet_keywords:
            cardiomet_mask |= s.str.contains(kw, na=False)

    if gene_col:
        cardiomet_mask |= df[gene_col].astype(str).str.upper().isin(
            ["LDLR", "PCSK9", "APOB", "TCF7L2", "FTO", "MC4R"]
        )

    summary["n_cardiomet"] = int((cardiomet_mask & high_mask).sum())

    # Key genes present
    if gene_col:
        genes_present = (
            df[gene_col]
            .astype(str)
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )
        key_genes = [g for g in genes_present if g in KEY_GENES][:3]
        summary["key_genes"] = key_genes

    # Risk flags
    flags: List[str] = []
    if cond_col:
        cond = df[cond_col].astype(str).str.lower()

        if ((cond.str.contains("diabetes", na=False)) & high_mask).any():
            flags.append("↑ T2D predisposition")

        if (
            cond.str.contains("coronary", na=False)
            | cond.str.contains("cad", na=False)
            | cond.str.contains("myocardial", na=False)
        & high_mask
        ).any():
            flags.append("↑ CAD / lipid risk")

        if cond.str.contains("obesity", na=False).any():
            flags.append("↑ obesity risk")

        if cond.str.contains("alzheimer", na=False).any() and (
            df.get(gene_col, pd.Series([])).astype(str).str.upper() == "APOE"
        ).any():
            flags.append("APOE-linked Alzheimer’s risk")

    summary["flags"] = flags
    return summary


# ----------------------------
# Physiology (wearables) summary
# ----------------------------

def summarize_physiology(df_wear: Optional[pd.DataFrame]) -> Dict:
    summary = {
        "has_data": False,
        "n_days": 0,
        "steps_avg": "N/A",
        "sleep_avg": "N/A",
        "rhr_avg": "N/A",
        "hrv_avg": "N/A",
        "flags": [],
    }

    if df_wear is None or df_wear.empty:
        return summary

    df = df_wear.copy()
    df.columns = [c.lower() for c in df.columns]
    summary["has_data"] = True

    # Date handling
    date_col = None
    for candidate in ["date", "day", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        # last 30 days window if date present
        if df[date_col].notna().any():
            latest_date = df[date_col].max()
            window_start = latest_date - pd.Timedelta(days=30)
            df = df[df[date_col] >= window_start]

    summary["n_days"] = len(df)

    # Numeric metrics
    def safe_mean(col_candidates: List[str]) -> Optional[float]:
        for c in col_candidates:
            if c in df.columns:
                return float(df[c].dropna().astype(float).mean())
        return None

    steps_avg = safe_mean(["steps", "step_count"])
    sleep_avg = safe_mean(["sleep_hours", "sleep", "sleep_duration"])
    rhr_avg = safe_mean(["resting_hr", "resting_heart", "resting_heart_rate"])
    hrv_avg = safe_mean(["hrv"])

    summary["steps_avg"] = steps_avg if steps_avg is not None else "N/A"
    summary["sleep_avg"] = sleep_avg if sleep_avg is not None else "N/A"
    summary["rhr_avg"] = rhr_avg if rhr_avg is not None else "N/A"
    summary["hrv_avg"] = hrv_avg if hrv_avg is not None else "N/A"

    flags: List[str] = []

    # Activity flags
    if steps_avg is not None:
        if steps_avg < 4000:
            flags.append("Very low activity (<4k steps/day)")
        elif steps_avg < 7000:
            flags.append("Below activity target")
        elif steps_avg >= 10000:
            flags.append("High daily activity (≥10k steps)")

        # proportion of very low days if we have steps column
        if "steps" in df.columns:
            low_days = (df["steps"] < 5000).sum()
            if len(df) > 0 and (low_days / len(df)) > 0.5:
                flags.append("Most days <5k steps")

    # Resting HR flags
    if rhr_avg is not None:
        if rhr_avg > 90:
            flags.append("High resting HR (>90 bpm)")
        elif rhr_avg > 80:
            flags.append("Elevated resting HR (>80 bpm)")

    # Sleep flags
    if sleep_avg is not None:
        if sleep_avg < 6:
            flags.append("Short sleep (avg <6 h)")
        elif sleep_avg < 7:
            flags.append("Slightly short sleep (<7 h)")

    # HRV flags (very heuristic)
    if hrv_avg is not None:
        if hrv_avg < 25:
            flags.append("Low HRV (stress / low recovery)")
        elif hrv_avg > 40:
            flags.append("High HRV")

    summary["flags"] = flags
    return summary


# ----------------------------
# Storyline summary
# ----------------------------

def summarize_storyline(
    risk_score: Optional[float],
    ehr: Dict,
    gen: Dict,
    phys: Dict,
) -> Dict:
    """
    Very simple synthesis across modalities for the storyline tab.
    """
    summary = {
        "risk_score": risk_score,
        "bullets": [],
    }

    bullets: List[str] = []

    # Genomics-based bullet
    if gen.get("has_data") and gen.get("flags"):
        bullets.append(
            f"Inherited: {', '.join(gen['flags'][:2])}."
        )

    # EHR-based bullet
    if ehr.get("has_data"):
        dx = ehr.get("key_diagnoses", [])
        if dx:
            bullets.append(
                "Clinical: history includes " + ", ".join(dx[:3]) + "."
            )
        if ehr.get("flags"):
            bullets.append(
                "Lab highlights: " + ", ".join(ehr["flags"][:2]) + "."
            )

    # Physiology-based bullet
    if phys.get("has_data"):
        phys_flags = phys.get("flags", [])
        if phys_flags:
            bullets.append(
                "Everyday physiology: " + ", ".join(phys_flags[:2]) + "."
            )

    summary["bullets"] = bullets
    return summary


# ================================================================
#                    STREAMLIT UI
# ================================================================

st.set_page_config(page_title="CoralMD — Practitioner", layout="wide")

st.title("🩺 Practitioner Home")
st.caption("Browse patients, inspect multimodal data, and follow their health storyline.")

# ----------------------------
# Patient registry
# ----------------------------

rows = list_patients()
df_patients = pd.DataFrame(rows, columns=["id", "name", "last_risk"])

if df_patients.empty:
    st.info("No patients yet — ask a patient to create a record and upload data.")
    st.stop()

st.subheader("Patient registry")

search = st.text_input("Search for patient (name or id)")
filtered = df_patients.copy()

if search:
    search_lower = search.lower()
    mask = (
        df_patients["name"].astype(str).str.lower().str.contains(search_lower)
        | df_patients["id"].astype(str).str.contains(search_lower)
    )
    filtered = df_patients[mask]

st.dataframe(filtered, use_container_width=True)

st.divider()

# ----------------------------
# Open a specific patient
# ----------------------------

st.subheader("Open patient")

patient_options = [
    f"{row['name']} (id={row['id']})" for _, row in df_patients.iterrows()
]
selected_label = st.selectbox("Choose a patient", patient_options)
selected_id = int(selected_label.split("id=")[-1].rstrip(")"))

if not selected_id:
    st.warning("Select a valid patient.")
    st.stop()

if st.button("Open patient"):
    st.session_state["active_patient_id"] = selected_id

patient_id = st.session_state.get("active_patient_id", selected_id)

# Load uploads + predictions for the active patient
uploads = get_patient_uploads(patient_id)
preds = get_patient_predictions(patient_id)

# Load latest CSVs for each modality
df_ehr = load_latest_csv(uploads, "ehr")
df_gen = load_latest_csv(uploads, "genomics")
df_wear = load_latest_csv(uploads, "wearables")

ehr_summary = summarize_ehr(df_ehr)
gen_summary = summarize_genomics(df_gen)
phys_summary = summarize_physiology(df_wear)

# ---- latest risk + per-domain scores (from details_json) ----
latest_risk: Optional[float] = None
per_domain: Dict[str, float] = {}

if not preds.empty:
    latest_row = preds.iloc[0]
    if pd.notna(latest_row["risk_score"]):
        latest_risk = float(latest_row["risk_score"])
    try:
        details = json.loads(latest_row.get("details_json") or "{}")
        per_domain = details.get("per_domain", {}) or {}
    except json.JSONDecodeError:
        per_domain = {}

storyline = summarize_storyline(latest_risk, ehr_summary, gen_summary, phys_summary)

# ----------------------------
# Quick look cards
# ----------------------------

st.subheader("Quick look")

col_ehr, col_gen, col_phys, col_story = st.columns(4)

with col_ehr:
    st.markdown("**🧾 EHR**")
    n_ehr_uploads = int((uploads["kind"] == "ehr").sum())
    st.caption(f"Uploads: {n_ehr_uploads}")
    if ehr_summary["has_data"]:
        st.write(f"{ehr_summary['n_rows']} rows of clinical data.")
    else:
        st.write("No EHR data loaded.")
    st.caption("Lab panels, diagnoses, and clinical events snapshot.")
    if st.button("🩺 Open full EHR view", key="open_ehr_btn"):
        st.session_state["deep_dive_patient_id"] = patient_id
        st.session_state["deep_dive_kind"] = "ehr"
        go_to_page("3_EHR_Explorer.py")  # adjust name/path if needed

with col_gen:
    st.markdown("**🧬 Genomics**")
    n_gen_uploads = int((uploads["kind"] == "genomics").sum())
    st.caption(f"Uploads: {n_gen_uploads}")
    if gen_summary["has_data"]:
        st.write(f"{gen_summary['n_variants']} variants loaded.")
    else:
        st.write("No genomics data loaded.")
    st.caption("Variant summaries and inherited risk (prototype).")
    if st.button("🔍 Open full genomics view", key="open_gen_btn"):
        st.session_state["deep_dive_patient_id"] = patient_id
        st.session_state["deep_dive_kind"] = "genomics"
        go_to_page("5_Genomics_Explorer.py")  # adjust if you name it differently

with col_phys:
    st.markdown("**📈 Physiology**")
    n_phys_uploads = int((uploads["kind"] == "wearables").sum())
    st.caption(f"Uploads: {n_phys_uploads}")
    if phys_summary["has_data"]:
        st.write(f"{phys_summary['n_days']} days of wearable data.")
    else:
        st.write("No wearable data loaded.")
    st.caption("Heart rate, activity, sleep, and trend views.")
    if st.button("📊 Open full physiology view", key="open_phys_btn"):
        st.session_state["deep_dive_patient_id"] = patient_id
        st.session_state["deep_dive_kind"] = "wearables"
        go_to_page("4_Wearables_Explorer.py")

with col_story:
    st.markdown("**📚 Storyline**")
    if latest_risk is not None:
        st.caption("Latest risk score")
        st.write(f"{latest_risk:.2f}")
    else:
        st.caption("Latest risk score")
        st.write("—")
    st.caption(
        "Summarizes inherited risk, clinical history, and everyday physiology "
        "into one narrative timeline (prototype)."
    )

st.divider()

# ----------------------------
# Patient summaries section
# ----------------------------

st.subheader("Patient summaries ↪")

tab_ehr, tab_gen, tab_phys, tab_story = st.tabs(
    ["EHR summary", "Genomics summary", "Physiology summary", "Storyline notes"]
)

with tab_ehr:
    st.markdown(f"**EHR overview for patient {patient_id}**")

    if not ehr_summary["has_data"]:
        st.write("No EHR data loaded.")
    else:
        st.markdown(
            f"""
- **Gender:** {ehr_summary['gender']}
- **Age (anchor):** {ehr_summary['age']}
- **Number of admissions:** {ehr_summary['n_admissions']}
- **Number of diagnoses:** {ehr_summary['n_diagnoses']}
"""
        )
        if ehr_summary["key_diagnoses"]:
            st.markdown(
                "**Key diagnoses:** " + ", ".join(ehr_summary["key_diagnoses"])
            )
        if ehr_summary["flags"]:
            st.markdown(
                "**Highlighted lab flags:** " + ", ".join(ehr_summary["flags"])
            )

with tab_gen:
    st.markdown(f"**Genomic overview for patient {patient_id}**")

    if not gen_summary["has_data"]:
        st.write("No genomics data loaded.")
    else:
        st.markdown(
            f"""
- **Variants in file:** {gen_summary['n_variants']}
- **High-impact variants:** {gen_summary['n_high_impact']}
- **Cardiometabolic variants:** {gen_summary['n_cardiomet']}
"""
        )
        if gen_summary["key_genes"]:
            st.markdown(
                "**Genes represented:** " + ", ".join(gen_summary["key_genes"])
            )
        if gen_summary["flags"]:
            st.markdown("**Genomic risk highlights:**")
            for f in gen_summary["flags"]:
                st.markdown(f"- {f}")

with tab_phys:
    st.markdown(f"**Physiology overview for patient {patient_id}**")

    if not phys_summary["has_data"]:
        st.write("No wearable data loaded.")
    else:
        steps = phys_summary["steps_avg"]
        sleep = phys_summary["sleep_avg"]
        rhr = phys_summary["rhr_avg"]
        hrv = phys_summary["hrv_avg"]

        st.markdown(
            f"""
- **Avg steps/day:** {steps if steps != 'N/A' else 'N/A'}
- **Avg sleep duration:** {sleep if sleep != 'N/A' else 'N/A'} h
- **Avg resting HR:** {rhr if rhr != 'N/A' else 'N/A'} bpm
- **Avg HRV:** {hrv if hrv != 'N/A' else 'N/A'} ms
"""
        )
        if phys_summary["flags"]:
            st.markdown("**Physiology / lifestyle flags:**")
            for f in phys_summary["flags"]:
                st.markdown(f"- {f}")

with tab_story:
    st.markdown(f"**Storyline notes for patient {patient_id}**")

    if storyline["risk_score"] is not None:
        st.markdown(
            f"- **Overall prototype risk score:** {storyline['risk_score']:.2f}"
        )
    else:
        st.markdown("- **Overall prototype risk score:** N/A")

    # --- Risk ribbon (per-domain scores) ---
    if per_domain:
        st.markdown("##### Chronic disease risk ribbon")
        cols = st.columns(4)

        domain_labels = [
            ("cardio_cerebro", "❤️ Cardio / cerebro"),
            ("metabolic", "🧬 Metabolic"),
            ("neurodegenerative", "🧠 Neurodegenerative"),
            ("cancer", "🧪 Cancer"),
        ]

        for col, (key, label) in zip(cols, domain_labels):
            score = per_domain.get(key)
            with col:
                if score is None:
                    st.write(label)
                    st.write("—")
                else:
                    s = float(score)
                    if s < 0.3:
                        level = "low"
                    elif s < 0.6:
                        level = "moderate"
                    else:
                        level = "high"
                    st.write(label)
                    st.write(f"**{s:.2f}** ({level})")
                    st.progress(float(np.clip(s, 0, 1)))
                    st.caption("Toy score, not for clinical use.")
    else:
        st.caption("Run a prediction from the patient view to see per-domain scores.")

    # --- Existing storyline bullets ---
    if storyline["bullets"]:
        st.markdown("##### Narrative bullets")
        for b in storyline["bullets"]:
            st.markdown(f"- {b}")
    else:
        st.write(
            "Storyline synthesis will appear here once EHR, genomics, and physiology data are available."
        )

st.divider()

# ----------------------------
# Raw uploads + model outputs
# ----------------------------

st.subheader("Files and model outputs")

st.markdown("**Uploads**")
if uploads.empty:
    st.write("No uploads for this patient yet.")
else:
    st.dataframe(uploads, use_container_width=True)

st.markdown("**Predictions**")
if preds.empty:
    st.write("No predictions recorded yet.")
else:
    st.dataframe(preds, use_container_width=True)
