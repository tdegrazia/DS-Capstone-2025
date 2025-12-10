# 5_Genomics_Explorer.py

from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.db_connect import get_conn, list_patients

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "genomics"

KEY_GENES = ["APOE", "LDLR", "PCSK9", "APOB", "TCF7L2", "FTO", "MC4R"]


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

@st.cache_data(show_spinner="Loading genomic datasets...")
def load_genomic_datasets() -> Dict[str, pd.DataFrame]:
    files = sorted(DATA_PATH.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No genomic CSV files found under {DATA_PATH}")
    datasets: Dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        datasets[f.stem] = df
    return datasets


def high_impact_mask(df: pd.DataFrame) -> pd.Series:
    """Same rough definition as the risk model: score >= 0.8 or effect is pathogenic."""
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(False, index=df.index)

    if "pathogenicity_score" in df.columns:
        mask |= df["pathogenicity_score"].astype(float) >= 0.8

    if "effect" in df.columns:
        mask |= df["effect"].astype(str).str.lower().isin(
            ["pathogenic", "likely_pathogenic", "likely pathogenic"]
        )

    return mask


def cardiometabolic_mask(df: pd.DataFrame) -> pd.Series:
    """Variants linked to lipids, T2D, obesity, etc."""
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(False, index=df.index)

    cond_col = None
    for c in ["condition", "trait", "phenotype"]:
        if c in df.columns:
            cond_col = c
            break

    if cond_col:
        cond = df[cond_col].astype(str).str.lower()
        keywords = [
            "coronary",
            "cad",
            "myocardial",
            "hypercholesterolemia",
            "lipid",
            "cholesterol",
            "diabetes",
            "type 2",
            "obesity",
            "hypertension",
        ]
        for kw in keywords:
            mask |= cond.str.contains(kw, na=False)

    if "gene" in df.columns:
        genes = df["gene"].astype(str).str.upper()
        mask |= genes.isin(["LDLR", "PCSK9", "APOB", "TCF7L2", "FTO", "MC4R"])

    return mask


# ========================= UI =========================

st.set_page_config(page_title="CoralMD — Genomics Explorer", layout="wide")
st.title("🧬 Genomics Explorer")
st.caption("Variant-level view of inherited risk (toy, not for clinical use).")

# -------- Choose data source --------
source = st.radio(
    "Choose data source",
    ["By patient id (from uploads)", "Sample file from data/genomics"],
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
    df = load_latest_csv(uploads, "genomics")

    if df is None or df.empty:
        st.error("No usable genomics CSV found for this patient yet.")
        st.stop()

    df.columns = df.columns.str.lower()
    data_label = f"Genomics for patient {patient_id}"
else:
    try:
        datasets = load_genomic_datasets()
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\n"
            "Create a folder `data/genomics/` and place one or more CSVs there "
            "(columns like gene, variant_id, condition, effect, pathogenicity_score)."
        )
        st.stop()

    dataset_name = st.selectbox("Choose genomic sample dataset", list(datasets.keys()))
    df = datasets[dataset_name].copy()
    data_label = f"`{dataset_name}.csv`"

if df.empty:
    st.write("Dataset is empty.")
    st.stop()

st.caption(f"{data_label} — {len(df)} variants")
st.divider()

# ========================= Basic counts =========================

st.subheader("Variant overview")

hi_mask = high_impact_mask(df)
cardio_mask = cardiometabolic_mask(df)
cardio_hi_mask = hi_mask & cardio_mask

n_total = len(df)
n_hi = int(hi_mask.sum())
n_cardio_hi = int(cardio_hi_mask.sum())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total variants", n_total)
with col2:
    st.metric("High-impact variants", n_hi)
with col3:
    st.metric("Cardiometabolic high-impact variants", n_cardio_hi)

st.divider()

# ========================= Pathogenicity distribution =========================

st.subheader("Pathogenicity score distribution")

if "pathogenicity_score" in df.columns:
    fig_hist = px.histogram(
        df,
        x="pathogenicity_score",
        nbins=20,
        title="Distribution of pathogenicity scores",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.write("No `pathogenicity_score` column found.")

st.divider()

# ========================= Conditions & genes =========================

st.subheader("Conditions / traits represented")

cond_col = None
for c in ["condition", "trait", "phenotype"]:
    if c in df.columns:
        cond_col = c
        break

if cond_col:
    cond_counts = (
        df[cond_col].dropna().astype(str).value_counts().reset_index()
    )
    cond_counts.columns = ["condition", "count"]
    st.dataframe(cond_counts.head(15), use_container_width=True)

    top_n = st.slider("Show top N conditions", min_value=3, max_value=25, value=10)
    fig_cond = px.bar(
        cond_counts.head(top_n),
        x="condition",
        y="count",
        title="Most frequent conditions / traits",
    )
    fig_cond.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cond, use_container_width=True)
else:
    st.write("No condition/trait column found.")

st.subheader("Key risk genes present")

if "gene" in df.columns:
    genes = df["gene"].astype(str).str.upper()
    gene_counts = genes.value_counts().reset_index()
    gene_counts.columns = ["gene", "count"]

    st.dataframe(gene_counts.head(15), use_container_width=True)

    mask_key = gene_counts["gene"].isin(KEY_GENES)
    key_gene_counts = gene_counts[mask_key]

    if not key_gene_counts.empty:
        fig_gene = px.bar(
            key_gene_counts,
            x="gene",
            y="count",
            title="KEY_GENES represented",
        )
        st.plotly_chart(fig_gene, use_container_width=True)
    else:
        st.caption("No canonical cardiometabolic risk genes found in this file.")
else:
    st.write("No `gene` column found.")

st.divider()

# ========================= High-impact cardiometabolic variants =========================

st.subheader("High-impact cardiometabolic variants")

if "gene" in df.columns:
    cols_to_show: List[str] = []
    for c in ["gene", "variant_id", cond_col, "effect", "pathogenicity_score"]:
        if c is not None and c in df.columns and c not in cols_to_show:
            cols_to_show.append(c)

    subset = df[cardio_hi_mask][cols_to_show].copy()
    if subset.empty:
        st.write("No high-impact cardiometabolic variants detected with current rules.")
    else:
        st.dataframe(subset, use_container_width=True)
else:
    st.write("Cannot filter cardiometabolic variants — no `gene` column found.")
