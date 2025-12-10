# 6_Storyline_View.py
#
# Prototype “trajectory engine” for CoralMD.
# - Reads latest genomics / EHR / wearable uploads for a patient
# - Builds a simple lifetime metabolic / cardiometabolic risk trajectory
# - Breaks down the chronic-disease ribbon by data stream
# - Adds uncertainty annotations and a small "what-if" sandbox

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.db_connect import get_conn, list_patients


# ----------------------------
# DB helpers (same patterns as Practitioner home)
# ----------------------------

def get_patient_uploads(patient_id: int) -> pd.DataFrame:
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
    if uploads.empty:
        return None
    sub = uploads[uploads["kind"] == kind]
    if sub.empty:
        return None
    latest = sub.sort_values("uploaded_at", ascending=False).iloc[0]
    path = Path(latest["filename"])
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return None


# ----------------------------
# Feature extraction helpers
# (re-using same heuristics as summaries / risk model)
# ----------------------------

KEY_GENES_METAB = ["TCF7L2", "FTO", "MC4R", "LDLR", "PCSK9", "APOB"]
KEY_GENES_NEURO = ["APOE"]


def extract_genomic_factors(df_gen: Optional[pd.DataFrame]) -> List[Dict]:
    """
    Turn genomics table into 'factors' that can move domain risks.
    Each factor: {id, stream, domain, label, delta, modifiable, confidence}
    """
    factors: List[Dict] = []
    if df_gen is None or df_gen.empty:
        return factors

    df = df_gen.copy()
    df.columns = df.columns.str.lower()

    gene_col = "gene" if "gene" in df.columns else None
    cond_col = None
    for c in ["condition", "trait", "phenotype"]:
        if c in df.columns:
            cond_col = c
            break

    genes = df[gene_col].astype(str).str.upper() if gene_col else pd.Series([], dtype=str)
    cond = df[cond_col].astype(str).str.lower() if cond_col else pd.Series([], dtype=str)

    # Helper: add factor
    def add_factor(fid, domain, label, delta, modifiable=False, confidence="high"):
        factors.append(
            {
                "id": fid,
                "stream": "Genomics",
                "domain": domain,
                "label": label,
                "delta": float(delta),
                "modifiable": bool(modifiable),
                "confidence": confidence,
            }
        )

    # T2D predisposition
    if (genes.isin(["TCF7L2"]).any()) or cond.str.contains("diabetes", na=False).any():
        add_factor(
            "gen_t2d",
            "metabolic",
            "Inherited T2D susceptibility (e.g. TCF7L2 / diabetes GWAS hits)",
            0.20,
            modifiable=False,
            confidence="high",
        )

    # Obesity risk (FTO / MC4R / obesity traits)
    if (genes.isin(["FTO", "MC4R"]).any()) or cond.str.contains("obesity", na=False).any():
        add_factor(
            "gen_obesity",
            "metabolic",
            "Inherited obesity / BMI risk (FTO / MC4R / obesity traits)",
            0.12,
            modifiable=False,
            confidence="high",
        )

    # Lipid / CAD risk
    if genes.isin(["LDLR", "PCSK9", "APOB"]).any() or cond.str.contains("cholesterol", na=False).any():
        add_factor(
            "gen_lipids",
            "cardio_cerebro",
            "Lipid / LDL / coronary disease variants (LDLR / PCSK9 / APOB)",
            0.10,
            modifiable=False,
            confidence="high",
        )

    # APOE / Alzheimer
    if genes.isin(["APOE"]).any() or cond.str.contains("alzheimer", na=False).any():
        add_factor(
            "gen_apoE",
            "neurodegenerative",
            "APOE / Alzheimer’s susceptibility variants",
            0.15,
            modifiable=False,
            confidence="high",
        )

    return factors


def ehr_lab_flags(df: pd.DataFrame) -> List[str]:
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


def extract_ehr_factors(df_ehr: Optional[pd.DataFrame]) -> Tuple[List[Dict], List[str]]:
    factors: List[Dict] = []
    all_flags: List[str] = []
    if df_ehr is None or df_ehr.empty:
        return factors, all_flags

    df = df_ehr.copy()
    df.columns = df.columns.str.lower()

    diag_col = None
    for c in ["diagnosis_name", "diagnosis", "diag"]:
        if c in df.columns:
            diag_col = c
            break

    diagnoses = (
        df[diag_col].dropna().astype(str).str.lower() if diag_col else pd.Series([], dtype=str)
    )

    # Helper
    def add_factor(fid, domain, label, delta, modifiable, confidence="medium"):
        factors.append(
            {
                "id": fid,
                "stream": "EHR / Labs",
                "domain": domain,
                "label": label,
                "delta": float(delta),
                "modifiable": bool(modifiable),
                "confidence": confidence,
            }
        )

    # Diagnosis-based factors
    if diagnoses.str.contains("obesity", na=False).any():
        add_factor(
            "ehr_obesity_dx",
            "metabolic",
            "Clinical diagnosis of obesity",
            0.10,
            modifiable=True,
        )

    if diagnoses.str.contains("hypertension", na=False).any():
        add_factor(
            "ehr_htn",
            "cardio_cerebro",
            "Hypertension / elevated blood pressure diagnosis",
            0.08,
            modifiable=True,
        )

    # Lab flags
    lab_flags = ehr_lab_flags(df)
    all_flags.extend(lab_flags)

    for f in lab_flags:
        if "LDL" in f:
            add_factor(
                "ehr_ldl",
                "cardio_cerebro",
                f,
                0.08,
                modifiable=True,
            )
        elif "HbA1c" in f:
            add_factor(
                "ehr_a1c",
                "metabolic",
                f,
                0.10,
                modifiable=True,
            )
        elif "creatinine" in f:
            add_factor(
                "ehr_creat",
                "cardio_cerebro",
                f,
                0.06,
                modifiable= True,
                confidence="low",
            )

    return factors, all_flags


def extract_wearable_factors(df_wear: Optional[pd.DataFrame]) -> Tuple[List[Dict], List[str], int]:
    factors: List[Dict] = []
    flags: List[str] = []
    n_days = 0
    if df_wear is None or df_wear.empty:
        return factors, flags, n_days

    df = df_wear.copy()
    df.columns = df.columns.str.lower()

    date_col = None
    for c in ["date", "day", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    n_days = len(df)

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

    def add_factor(fid, domain, label, delta, confidence="medium"):
        factors.append(
            {
                "id": fid,
                "stream": "Wearables",
                "domain": domain,
                "label": label,
                "delta": float(delta),
                "modifiable": True,
                "confidence": confidence,
            }
        )
        flags.append(label)

    # Activity
    if steps is not None:
        if steps < 4000:
            add_factor(
                "wear_sedentary",
                "metabolic",
                f"Very low activity (~{steps:.0f} steps/day)",
                0.08,
            )
        elif steps < 7000:
            add_factor(
                "wear_below_steps",
                "metabolic",
                f"Below activity target (~{steps:.0f} steps/day)",
                0.05,
            )
        elif steps >= 10000:
            factors.append(
                {
                    "id": "wear_high_steps",
                    "stream": "Wearables",
                    "domain": "metabolic",
                    "label": f"High daily activity (~{steps:.0f} steps/day)",
                    "delta": -0.04,
                    "modifiable": True,
                    "confidence": "medium",
                }
            )

    # Sleep
    if sleep is not None:
        if sleep < 6:
            add_factor(
                "wear_short_sleep",
                "metabolic",
                f"Short sleep (avg {sleep:.1f} h/night)",
                0.06,
            )
        elif sleep < 7:
            add_factor(
                "wear_slight_sleep",
                "metabolic",
                f"Slightly short sleep (avg {sleep:.1f} h/night)",
                0.03,
            )

    # Resting HR
    if rhr is not None and rhr > 80:
        add_factor(
            "wear_rhr",
            "cardio_cerebro",
            f"Elevated resting HR (~{rhr:.0f} bpm)",
            0.05,
        )

    # HRV
    if hrv is not None and hrv < 25:
        add_factor(
            "wear_low_hrv",
            "cardio_cerebro",
            f"Low HRV (~{hrv:.0f} ms)",
            0.04,
            confidence="low",
        )

    return factors, flags, n_days


# ----------------------------
# Storyline risk engine
# ----------------------------

def build_trajectory(
    genomic_factors: List[Dict],
    ehr_factors: List[Dict],
    wear_factors: List[Dict],
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Collapse factors into a sequence of events with start/end risk.
    For now we track a single 'metabolic' storyline risk in [0,1].
    """

    events: List[Dict] = []

    # Start with low baseline for a generic individual
    risk = 0.10

    # Event 1: birth / genomics
    gen_delta = sum(f["delta"] for f in genomic_factors if f["domain"] == "metabolic")
    risk_after_gen = float(np.clip(risk + gen_delta, 0.0, 1.0))
    events.append(
        {
            "stage": 0,
            "label": "Birth / inherited baseline",
            "kind": "genomics",
            "risk_before": risk,
            "risk_after": risk_after_gen,
            "delta": risk_after_gen - risk,
            "drivers": [f for f in genomic_factors if f["domain"] == "metabolic"],
        }
    )
    risk = risk_after_gen

    # Event 2: labs / diagnoses
    ehr_delta = sum(f["delta"] for f in ehr_factors if f["domain"] == "metabolic")
    risk_after_ehr = float(np.clip(risk + ehr_delta, 0.0, 1.0))
    events.append(
        {
            "stage": 1,
            "label": "Clinical snapshot (diagnoses / labs)",
            "kind": "ehr",
            "risk_before": risk,
            "risk_after": risk_after_ehr,
            "delta": risk_after_ehr - risk,
            "drivers": [f for f in ehr_factors if f["domain"] == "metabolic"],
        }
    )
    risk = risk_after_ehr

    # Event 3: habits / wearables
    wear_delta = sum(f["delta"] for f in wear_factors if f["domain"] == "metabolic")
    risk_after_wear = float(np.clip(risk + wear_delta, 0.0, 1.0))
    events.append(
        {
            "stage": 2,
            "label": "Everyday behavior (wearables)",
            "kind": "wearables",
            "risk_before": risk,
            "risk_after": risk_after_wear,
            "delta": risk_after_wear - risk,
            "drivers": [f for f in wear_factors if f["domain"] == "metabolic"],
        }
    )

    traj_df = pd.DataFrame(
        {
            "stage": [e["stage"] for e in events],
            "label": [e["label"] for e in events],
            "risk": [e["risk_after"] for e in events],
        }
    )

    return traj_df, events


def aggregate_ribbon_contributions(
    genomic_factors: List[Dict],
    ehr_factors: List[Dict],
    wear_factors: List[Dict],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for f in genomic_factors + ehr_factors + wear_factors:
        rows.append(
            {
                "domain": f["domain"],
                "stream": f["stream"],
                "label": f["label"],
                "delta": f["delta"],
                "modifiable": f["modifiable"],
                "confidence": f["confidence"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["domain", "stream", "label", "delta", "modifiable", "confidence"])
    df = pd.DataFrame(rows)
    return df


def compute_uncertainty_badge(df_gen, df_ehr, df_wear, n_wear_days: int) -> str:
    # Very simple heuristic: more streams + more days => higher confidence
    streams = sum(
        [
            1 if df_gen is not None and not df_gen.empty else 0,
            1 if df_ehr is not None and not df_ehr.empty else 0,
            1 if df_wear is not None and not df_wear.empty else 0,
        ]
    )

    if streams == 0:
        return "Very low – almost no data"

    if streams == 1:
        return "Low – based on a single data stream"

    # 2–3 streams
    if n_wear_days >= 30:
        return "Higher – multiple data streams, ≥30 days of wearables"
    if n_wear_days >= 7:
        return "Moderate – multiple streams, ~1–4 weeks of wearables"
    return "Moderate – multiple streams, but limited longitudinal data"


# ----------------------------
# What-if sandbox
# ----------------------------

def apply_what_if(
    factors: pd.DataFrame,
    *,
    more_steps: bool,
    improved_lipids: bool,
    improved_weight: bool,
) -> Dict[str, float]:
    """
    Take the ribbon contributions table and 'turn down' some modifiable deltas.
    Returns new domain scores (sum of deltas, clipped to [0,1]) assuming baseline 0.10.
    """
    df = factors.copy()

    # Start by potentially neutralising some factors
    def zero_factor(mask):
        df.loc[mask, "delta"] = 0.0

    if more_steps:
        zero_factor(df["id"].fillna("").str.contains("wear_sedentary|wear_below_steps"))

    if improved_lipids:
        zero_factor(df["id"].fillna("").str.contains("ehr_ldl"))

    if improved_weight:
        zero_factor(df["id"].fillna("").str.contains("ehr_obesity_dx|gen_obesity"))

    # Re-aggregate by domain
    # Need ids column – reconstruct from labels if missing
    if "id" not in df.columns:
        df["id"] = ""

    base = 0.10
    out: Dict[str, float] = {}
    for domain in ["cardio_cerebro", "metabolic", "neurodegenerative", "cancer"]:
        dom_delta = float(df[df["domain"] == domain]["delta"].sum()) if "domain" in df.columns else 0.0
        out[domain] = float(np.clip(base + dom_delta, 0.0, 1.0))
    return out


# ================================================================
#                    STREAMLIT UI
# ================================================================

st.set_page_config(page_title="CoralMD — Storyline View", layout="wide")

st.title("📚 Storyline View")
st.caption(
    "Prototype timeline that weaves together genomics, EHR, and wearables "
    "to explain changes in metabolic / cardiometabolic risk over a lifetime."
)

# ----- choose patient -----

rows = list_patients()
df_patients = pd.DataFrame(rows, columns=["id", "name", "last_risk"])

if df_patients.empty:
    st.info("No patients in the registry yet. Create one from the patient view first.")
    st.stop()

patient_labels = [f"{row['name']} (id={row['id']})" for _, row in df_patients.iterrows()]
selected_label = st.selectbox("Patient for storyline view", patient_labels)
selected_id = int(selected_label.split("id=")[-1].rstrip(")"))

if st.button("Load storyline"):
    st.session_state["story_patient_id"] = selected_id

pid = st.session_state.get("story_patient_id", selected_id)

st.markdown(f"### Patient id = {pid}")

# ----- pull data for that patient -----

uploads = get_patient_uploads(pid)
preds = get_patient_predictions(pid)

df_ehr = load_latest_csv(uploads, "ehr")
df_gen = load_latest_csv(uploads, "genomics")
df_wear = load_latest_csv(uploads, "wearables")

# Extract factors
gen_factors = extract_genomic_factors(df_gen)
ehr_factors, ehr_flags = extract_ehr_factors(df_ehr)
wear_factors, wear_flags, n_wear_days = extract_wearable_factors(df_wear)

traj_df, events = build_trajectory(gen_factors, ehr_factors, wear_factors)
ribbon_contrib = aggregate_ribbon_contributions(gen_factors, ehr_factors, wear_factors)

uncertainty_text = compute_uncertainty_badge(df_gen, df_ehr, df_wear, n_wear_days)

# ----- top-level risk trajectory -----

st.subheader("Risk trajectory (toy metabolic risk 0–1)")

if traj_df.empty:
    st.write("Not enough data to build a storyline yet — upload at least one modality.")
else:
    st.caption("Estimated metabolic risk over key life events (prototype)")
    fig_traj = px.line(
        traj_df,
        x="stage",
        y="risk",
        markers=True,
        text="label",
        labels={"stage": "Storyline stage", "risk": "Toy metabolic risk (0–1)"},
    )
    fig_traj.update_traces(textposition="top center")
    st.plotly_chart(fig_traj, use_container_width=True)

st.caption(
    "This is a non-clinical prototype that uses simple rules to nudge risk up or down "
    "based on genetic variants, diagnoses, labs, and daily behavior."
)
st.markdown(f"**Uncertainty badge:** {uncertainty_text}")

st.divider()

# ----------------------------
# Deep ribbon breakdown
# ----------------------------

st.subheader("Deep breakdown of the chronic-disease ribbon")

if ribbon_contrib.empty:
    st.write("No interpretable factors found for this patient yet.")
else:
    # Per-domain × stream stacked bar
    agg = (
        ribbon_contrib.groupby(["domain", "stream"], as_index=False)["delta"]
        .sum()
        .copy()
    )
    agg["delta"] = agg["delta"].astype(float)

    fig_ribbon = px.bar(
        agg,
        x="domain",
        y="delta",
        color="stream",
        barmode="relative",
        title="How each data stream pushes each disease domain up or down (toy deltas)",
    )
    st.plotly_chart(fig_ribbon, use_container_width=True)

    st.markdown("**Factor-level table**")
    # Add ids for sandbox if missing
    if "id" not in ribbon_contrib.columns:
        ribbon_contrib["id"] = ""
    st.dataframe(
        ribbon_contrib[["domain", "stream", "label", "delta", "modifiable", "confidence"]],
        use_container_width=True,
    )

st.divider()

# ----------------------------
# Narrative explanation
# ----------------------------

st.subheader("Narrative explanation")

if traj_df.empty:
    st.write("Storyline not available yet.")
else:
    for ev in events:
        drivers = ev["drivers"]
        if not drivers and ev["kind"] == "genomics":
            summary = "No clearly cardiometabolic variants detected – baseline risk only."
        elif not drivers and ev["kind"] == "ehr":
            summary = "No major metabolic diagnoses / labs identified."
        elif not drivers and ev["kind"] == "wearables":
            summary = "Daily behavior is roughly neutral with respect to metabolic risk."
        else:
            # Build natural language from driver labels
            driver_text = "; ".join(d["label"] for d in drivers)
            if ev["kind"] == "genomics":
                summary = (
                    f"Inherited signal from genomics: {driver_text}. "
                    "These variants set an elevated baseline metabolic risk that cannot be "
                    "changed, but can be counterbalanced by lifestyle."
                )
            elif ev["kind"] == "ehr":
                summary = (
                    f"Clinical history / labs: {driver_text}. "
                    "These findings show how risk has begun to manifest in the clinical record."
                )
            else:
                summary = (
                    f"Everyday physiology from wearables: {driver_text}. "
                    "These are near-term levers a patient could modify."
                )

        st.markdown(f"**{ev['label']}** — {summary}")
        st.markdown(
            f"_Risk ↑ changed from **{ev['risk_before']:.2f}** to **{ev['risk_after']:.2f}** "
            f"(Δ {ev['delta']:+.2f})._"
        )
        st.write("")

st.divider()

# ----------------------------
# What-if sandbox
# ----------------------------

st.subheader("What-if sandbox (toy intervention simulation)")

if ribbon_contrib.empty:
    st.write("Need at least one factor to run the sandbox.")
else:
    st.caption("Toggle hypothetical changes and see how domain scores might shift (toy logic).")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        more_steps = st.checkbox("Increase steps to ≥7k/day", value=False)
    with col_b:
        improved_lipids = st.checkbox("Improve LDL cholesterol to <130 mg/dL", value=False)
    with col_c:
        improved_weight = st.checkbox("Resolve clinical obesity diagnosis", value=False)

    # baseline (current factors)
    baseline_scores = apply_what_if(
        ribbon_contrib.assign(id=ribbon_contrib.get("id", "")),
        more_steps=False,
        improved_lipids=False,
        improved_weight=False,
    )
    # scenario
    scenario_scores = apply_what_if(
        ribbon_contrib.assign(id=ribbon_contrib.get("id", "")),
        more_steps=more_steps,
        improved_lipids=improved_lipids,
        improved_weight=improved_weight,
    )

    comp_rows = []
    for domain in ["cardio_cerebro", "metabolic", "neurodegenerative", "cancer"]:
        comp_rows.append(
            {
                "domain": domain,
                "current": baseline_scores[domain],
                "what_if": scenario_scores[domain],
                "delta": scenario_scores[domain] - baseline_scores[domain],
            }
        )
    comp_df = pd.DataFrame(comp_rows)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Domain scores (current vs. what-if)**")
        st.dataframe(comp_df, use_container_width=True)

    with col2:
        fig_comp = px.bar(
            comp_df,
            x="domain",
            y="delta",
            title="Change in domain risk under what-if scenario (toy)",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.caption(
        "These numbers are illustrative only. They are not calibrated, "
        "and they ignore many important clinical factors."
    )

st.divider()

# ----------------------------
# Underlying data (debug)
# ----------------------------

st.subheader("Underlying data (for debugging / exploration)")

with st.expander("Genomics table"):
    if df_gen is None or df_gen.empty:
        st.write("No genomics file loaded for this patient.")
    else:
        st.dataframe(df_gen.head(50), use_container_width=True)

with st.expander("EHR / labs table"):
    if df_ehr is None or df_ehr.empty:
        st.write("No EHR file loaded for this patient.")
    else:
        st.dataframe(df_ehr.head(50), use_container_width=True)

with st.expander("Wearables table"):
    if df_wear is None or df_wear.empty:
        st.write("No wearable file loaded for this patient.")
    else:
        st.dataframe(df_wear.head(50), use_container_width=True)
