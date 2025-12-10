# utils/ml_models.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _normalize_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _score_cardio_from_ehr(df: pd.DataFrame) -> float:
    score = 0.0
    if df is None or df.empty:
        return score

    df = _normalize_cols(df)

    # diagnosis-based cardio flags
    diag_col = None
    for c in ["diagnosis_name", "diagnosis", "diag"]:
        if c in df.columns:
            diag_col = c
            break

    if diag_col:
        diag = df[diag_col].astype(str).str.lower()
        if diag.str.contains("hypertension", na=False).any():
            score += 2.0
        if diag.str.contains("elevated blood pressure", na=False).any():
            score += 1.0
        if (
            diag.str.contains("myocardial", na=False)
            | diag.str.contains("infarc", na=False)
            | diag.str.contains("coronary", na=False)
            | diag.str.contains("cad", na=False)
        ).any():
            score += 4.0

    # lipids / creatinine
    if {"lab_name", "lab_value"}.issubset(df.columns):
        labs = (
            df[["lab_name", "lab_value"]]
            .dropna()
            .copy()
        )
        labs["lab_name"] = labs["lab_name"].astype(str).str.lower()

        def mean_lab(keyword: str) -> Optional[float]:
            sel = labs[labs["lab_name"].str.contains(keyword, na=False)]
            if sel.empty:
                return None
            return float(sel["lab_value"].astype(float).mean())

        ldl = mean_lab("ldl")
        if ldl is not None:
            if ldl >= 160:
                score += 3.0
            elif ldl >= 130:
                score += 2.0

        creat = mean_lab("creatinine")
        if creat is not None and creat >= 1.3:
            score += 1.0  # kidney strain, indirect cardio signal

    return score


def _score_metabolic_from_ehr(df: pd.DataFrame) -> float:
    score = 0.0
    if df is None or df.empty:
        return score

    df = _normalize_cols(df)

    # diagnoses
    diag_col = None
    for c in ["diagnosis_name", "diagnosis", "diag"]:
        if c in df.columns:
            diag_col = c
            break

    if diag_col:
        diag = df[diag_col].astype(str).str.lower()
        if diag.str.contains("overweight", na=False).any():
            score += 1.0
        if diag.str.contains("obesity", na=False).any():
            score += 2.0
        if diag.str.contains("prediabetes", na=False).any():
            score += 2.0
        if diag.str.contains("type 2 diabetes", na=False).any():
            score += 4.0

    # HbA1c etc.
    if {"lab_name", "lab_value"}.issubset(df.columns):
        labs = df[["lab_name", "lab_value"]].dropna().copy()
        labs["lab_name"] = labs["lab_name"].astype(str).str.lower()

        def mean_lab(keyword: str) -> Optional[float]:
            sel = labs[labs["lab_name"].str.contains(keyword, na=False)]
            if sel.empty:
                return None
            return float(sel["lab_value"].astype(float).mean())

        a1c = mean_lab("hba1c") or mean_lab("a1c")
        if a1c is not None:
            if a1c >= 6.5:
                score += 4.0
            elif 5.7 <= a1c < 6.5:
                score += 2.0

    return score


def _score_from_genomics(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    scores = {
        "cardio_cerebro": 0.0,
        "metabolic": 0.0,
        "neurodegenerative": 0.0,
        "cancer": 0.0,
    }
    if df is None or df.empty:
        return scores

    df = _normalize_cols(df)

    gene_col = "gene" if "gene" in df.columns else None
    cond_col = None
    for c in ["condition", "trait", "phenotype"]:
        if c in df.columns:
            cond_col = c
            break

    path_col = None
    for c in ["pathogenicity_score", "score", "impact_score"]:
        if c in df.columns:
            path_col = c
            break

    # high-impact mask
    high = pd.Series(False, index=df.index)
    if path_col:
        high |= df[path_col].astype(float) >= 0.8
    if "effect" in df.columns:
        high |= df["effect"].astype(str).str.lower().isin(
            ["pathogenic", "likely_pathogenic", "likely pathogenic"]
        )

    if cond_col:
        cond = df[cond_col].astype(str).str.lower()

        # metabolic
        if ((cond.str.contains("diabetes", na=False)) & high).any():
            scores["metabolic"] += 2.5
        if ((cond.str.contains("obesity", na=False)) & high).any():
            scores["metabolic"] += 2.5

        # cardio / cerebro
        if (
            (cond.str.contains("coronary", na=False))
            | (cond.str.contains("cad", na=False))
            | (cond.str.contains("myocardial", na=False))
        & high
        ).any():
            scores["cardio_cerebro"] += 3.0

        # neurodegenerative
        if cond.str.contains("alzheimer", na=False).any():
            scores["neurodegenerative"] += 3.0

        # cancer (very approximate)
        if (
            cond.str.contains("cancer", na=False)
            | cond.str.contains("carcinoma", na=False)
        & high
        ).any():
            scores["cancer"] += 3.0

    # gene-level nudges
    if gene_col:
        genes = df[gene_col].astype(str).str.upper()
        if (genes == "APOE").any():
            scores["neurodegenerative"] += 1.5
        if genes.isin(["LDLR", "PCSK9", "APOB"]).any():
            scores["cardio_cerebro"] += 1.5
        if genes.isin(["TCF7L2", "FTO", "MC4R"]).any():
            scores["metabolic"] += 2.0

    return scores


def _score_from_wearables(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    scores = {
        "cardio_cerebro": 0.0,
        "metabolic": 0.0,
        "neurodegenerative": 0.0,
        "cancer": 0.0,
    }
    if df is None or df.empty:
        return scores

    df = _normalize_cols(df)

    # restrict to last 30 days if we have dates
    date_col = None
    for c in ["date", "day", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        if df[date_col].notna().any():
            latest = df[date_col].max()
            df = df[df[date_col] >= latest - pd.Timedelta(days=30)]

    def mean(col_candidates):
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

    # metabolic + cardio from activity / sleep
    if steps is not None:
        if steps < 4000:
            scores["metabolic"] += 2.0
            scores["cardio_cerebro"] += 1.5
        elif steps < 7000:
            scores["metabolic"] += 1.0
    if sleep is not None and sleep < 6:
        scores["metabolic"] += 1.0
        scores["neurodegenerative"] += 1.0

    # cardio from resting HR
    if rhr is not None:
        if rhr > 90:
            scores["cardio_cerebro"] += 3.0
        elif rhr > 80:
            scores["cardio_cerebro"] += 2.0

    # recovery / stress angle via HRV (very rough)
    if hrv is not None and hrv < 25:
        scores["cardio_cerebro"] += 1.0
        scores["metabolic"] += 0.5

    return scores


def compute_multimodal_risk(
    df_ehr: Optional[pd.DataFrame] = None,
    df_gen: Optional[pd.DataFrame] = None,
    df_wear: Optional[pd.DataFrame] = None,
) -> (float, Dict[str, Any]):
    """
    Toy multimodal risk model.

    Returns
    -------
    overall_risk : float in [0, 1]
    details : dict with keys:
        - 'per_domain': {
            'cardio_cerebro': float,
            'metabolic': float,
            'neurodegenerative': float,
            'cancer': float,
          }
    """
    # domain scores before normalization
    domain_scores = {
        "cardio_cerebro": 0.0,
        "metabolic": 0.0,
        "neurodegenerative": 0.0,
        "cancer": 0.0,
    }

    # ---- EHR contributions ----
    if df_ehr is not None and not df_ehr.empty:
        domain_scores["cardio_cerebro"] += _score_cardio_from_ehr(df_ehr)
        domain_scores["metabolic"] += _score_metabolic_from_ehr(df_ehr)

    # ---- Genomic contributions ----
    gen_scores = _score_from_genomics(df_gen)
    for k in domain_scores:
        domain_scores[k] += gen_scores.get(k, 0.0)

    # ---- Wearable contributions ----
    wear_scores = _score_from_wearables(df_wear)
    for k in domain_scores:
        domain_scores[k] += wear_scores.get(k, 0.0)

    # Max scores used for normalization (hand-tuned for this toy model)
    max_per_domain = {
        "cardio_cerebro": 12.0,
        "metabolic": 10.0,
        "neurodegenerative": 8.0,
        "cancer": 8.0,
    }

    per_domain_norm: Dict[str, float] = {}
    for k, raw in domain_scores.items():
        max_val = max_per_domain[k]
        per_domain_norm[k] = float(np.clip(raw / max_val, 0.0, 1.0))

    # overall = simple mean of non-null domains
    vals = list(per_domain_norm.values())
    overall = float(np.mean(vals)) if vals else 0.0

    details: Dict[str, Any] = {
        "per_domain": per_domain_norm,
        "version": "0.1-toy",
        "raw_scores": domain_scores,
    }
    return overall, details


# Optional: keep old name as a wrapper if some code still imports it
def compute_mock_risk(df: pd.DataFrame):
    """
    Backwards-compatible wrapper: treat a combined df as if it has all
    modalities in one table. This just passes df as EHR and wearables.
    """
    df = _normalize_cols(df)
    # crude split: we don't really know which columns belong where,
    # but for older prototypes this keeps things from crashing.
    overall, details = compute_multimodal_risk(df_ehr=df, df_gen=None, df_wear=df)
    return overall, details
