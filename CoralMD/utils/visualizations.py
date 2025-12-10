import plotly.express as px
import pandas as pd


# ========= EXISTING COMPONENTS =========

def risk_card_html(risk: float) -> str:
    """Simple neutral risk card (no red/green)."""
    return f"""
    <div style="padding:16px;border-radius:14px;
                background:rgba(255,255,255,0.07);
                border:1px solid rgba(255,255,255,0.15);">
      <h3 style="margin:0 0 8px 0;font-size:18px;">Estimated Risk (0 – 1)</h3>
      <div style="font-size:40px;font-weight:700;">{risk:.2f}</div>
      <p style="font-size:12px;opacity:0.8;margin-top:6px;">
        Prototype estimate; interpret alongside clinical context and uncertainty.
      </p>
    </div>
    """


def simple_trend(df: pd.DataFrame, xcol: str, ycol: str, title: str):
    if xcol not in df.columns or ycol not in df.columns:
        return None
    fig = px.line(df, x=xcol, y=ycol, title=title, template="plotly_white")
    return fig


def wearable_summary(patient: pd.Series) -> pd.DataFrame:
    """Return a Metric/Value table for one wearable patient row."""
    summary_data = {
        "Age": patient.get("age", "N/A"),
        "Gender": patient.get("gender", "N/A"),
        "Height (cm)": patient.get("height", "N/A"),
        "Weight (kg)": patient.get("weight", "N/A"),
        "Avg Heart Rate (bpm)": patient.get("hear_rate", "N/A"),
        "Resting Heart Rate": patient.get("resting_heart", "N/A"),
        "Steps per Day": patient.get("steps", "N/A"),
        "Calories Burned": patient.get("calories", "N/A"),
        "Distance (m)": patient.get("distance", "N/A"),
        "Activity Level": patient.get("activity", "N/A"),
    }
    return pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])


def wearable_insights(patient: pd.Series) -> list[str]:
    """Return a list of bullet-point lifestyle insights for one row."""
    insights: list[str] = []

    if patient.get("resting_heart", 0) > 80:
        insights.append("• Elevated resting heart rate — may indicate stress or low cardiovascular fitness.")
    if patient.get("steps", 0) < 5000:
        insights.append("• Low daily activity detected; may benefit from increased movement.")
    if patient.get("steps", 0) > 10000:
        insights.append("• Excellent daily activity — strong daily activity level.")
    if patient.get("hear_rate", 0) > 110:
        insights.append("• High average heart rate; monitor for overtraining or stress.")

    if not insights:
        insights.append("• Heart rate and activity within healthy ranges.")

    return insights


# ========= NEW: BRIEF PREVIEW HELPERS =========

def ehr_preview_from_df(df: pd.DataFrame) -> dict:
    """
    Very short, interpretable EHR snapshot for Quick Look.

    Works best with columns like:
      hadm_id, admit_time, discharge_time, diagnosis, icd_code,
      and optional lab columns (e.g., glucose, ldl, hdl, creatinine).
    Degrades gracefully if some columns are missing.
    """
    if df is None or df.empty:
        return {
            "headline": "No EHR data loaded.",
            "details": [],
        }

    preview = {}

    # Admissions / rows
    if "hadm_id" in df.columns:
        n_adm = df["hadm_id"].nunique()
    else:
        n_adm = len(df)
    preview["n_admissions"] = int(n_adm)

    # Date range (best effort)
    date_col = None
    for cand in ["admit_time", "admittime", "date", "charttime"]:
        if cand in df.columns:
            date_col = cand
            break
    date_txt = None
    if date_col is not None:
        try:
            dt = pd.to_datetime(df[date_col])
            start = dt.min().date()
            end = dt.max().date()
            if start == end:
                date_txt = f"on {start}"
            else:
                date_txt = f"from {start} to {end}"
        except Exception:
            date_txt = None

    # Top diagnoses
    top_diag_txt = None
    if "diagnosis" in df.columns:
        vc = df["diagnosis"].dropna().astype(str).value_counts()
        if len(vc) > 0:
            top = vc.index[0]
            top_diag_txt = f"Most common diagnosis: **{top}**."

    # Simple lab mention
    lab_bits = []
    for col, label in [
        ("glucose", "glucose"),
        ("ldl", "LDL cholesterol"),
        ("hdl", "HDL cholesterol"),
        ("creatinine", "creatinine"),
    ]:
        if col in df.columns:
            try:
                val = float(df[col].astype("float").mean())
                lab_bits.append(f"mean {label} ~ {val:.1f}")
            except Exception:
                continue

    lab_txt = None
    if lab_bits:
        lab_txt = " | ".join(lab_bits)

    headline_parts = [f"{n_adm} recorded admission(s)"]
    if date_txt:
        headline_parts.append(date_txt)
    preview["headline"] = ", ".join(headline_parts) + "."

    details: list[str] = []
    if top_diag_txt:
        details.append(top_diag_txt)
    if lab_txt:
        details.append(lab_txt)

    preview["details"] = details
    return preview


def genomics_preview_from_df(df: pd.DataFrame) -> dict:
    """
    Brief genomics snapshot.

    Works best with columns like:
      gene, variant, impact, consequence, disease, pathogenicity.
    """
    if df is None or df.empty:
        return {
            "headline": "No genomics data loaded.",
            "details": [],
        }

    n_var = len(df)
    headline = f"{n_var} variant(s) in uploaded panel."

    high_impact = None
    if "impact" in df.columns:
        hi = df[df["impact"].astype(str).str.contains("high", case=False)]
        high_impact = len(hi) if not hi.empty else 0

    pathogenic = None
    if "pathogenicity" in df.columns:
        pat = df[df["pathogenicity"].astype(str).str.contains("pathogenic", case=False)]
        pathogenic = len(pat) if not pat.empty else 0

    diseases_txt = None
    for col in ["disease", "condition", "trait"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique()
            if len(vals) > 0:
                diseases_txt = ", ".join(vals[:4])
            break

    details: list[str] = []
    if high_impact is not None:
        details.append(f"{high_impact} variant(s) flagged as high impact.")
    if pathogenic is not None:
        details.append(f"{pathogenic} variant(s) labelled pathogenic/likely pathogenic.")
    if diseases_txt:
        details.append(f"Associated disease tags include: {diseases_txt}.")

    return {"headline": headline, "details": details}


def physiology_preview_from_df(df: pd.DataFrame) -> dict:
    """
    Brief physiology (wearables) snapshot.

    Works best with columns like:
      date, steps, resting_hr or resting_heart, sleep_hours, hrv.
    """
    if df is None or df.empty:
        return {
            "headline": "No wearable data loaded.",
            "details": [],
        }

    # Use recent window if we have dates
    df_use = df.copy()
    date_col = None
    for cand in ["date", "day", "timestamp"]:
        if cand in df_use.columns:
            date_col = cand
            break

    if date_col:
        try:
            df_use[date_col] = pd.to_datetime(df_use[date_col])
            df_use = df_use.sort_values(date_col).tail(30)
        except Exception:
            pass
    else:
        df_use = df_use.tail(30)

    # Steps
    steps_col = None
    for cand in ["steps", "step_count"]:
        if cand in df_use.columns:
            steps_col = cand
            break

    steps_txt = None
    if steps_col:
        steps_mean = float(df_use[steps_col].astype("float").mean())
        steps_txt = f"Avg steps (recent) ≈ **{steps_mean:,.0f}** / day."

    # Resting HR
    rhr_col = None
    for cand in ["resting_hr", "resting_heart", "hr_resting"]:
        if cand in df_use.columns:
            rhr_col = cand
            break

    rhr_txt = None
    if rhr_col:
        rhr_mean = float(df_use[rhr_col].astype("float").mean())
        rhr_txt = f"Mean resting HR ≈ **{rhr_mean:.0f} bpm**."

    # Sleep
    sleep_col = None
    for cand in ["sleep_hours", "sleep", "sleep_duration"]:
        if cand in df_use.columns:
            sleep_col = cand
            break

    sleep_txt = None
    if sleep_col:
        sleep_mean = float(df_use[sleep_col].astype("float").mean())
        sleep_txt = f"Avg sleep ≈ **{sleep_mean:.1f} h/night**."

    # HRV (if present)
    hrv_col = None
    for cand in ["hrv", "rMSSD", "sdnn"]:
        if cand in df_use.columns:
            hrv_col = cand
            break

    hrv_txt = None
    if hrv_col:
        hrv_mean = float(df_use[hrv_col].astype("float").mean())
        hrv_txt = f"Avg HRV ≈ **{hrv_mean:.0f} ms** (higher is generally better)."

    headline = "Recent everyday physiology summary."

    details = [t for t in [steps_txt, rhr_txt, sleep_txt, hrv_txt] if t]

    return {"headline": headline, "details": details}
