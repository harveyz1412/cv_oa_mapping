"""
Client → Cotiviti Data Field Mapper
Streamlit web app — no API key required.
"""

import io
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_COL = "Client Data Fields"
TARGET_COL = "Cotiviti Data Fields"
BUNDLED_REF = os.path.join(os.path.dirname(__file__), "Data_Mod_Mapping_Reference.xlsx")

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
PREFIXES = ["udf_ch_", "udf_", "fld_", "col_", "ch_"]


def strip_prefixes(text: str) -> str:
    t = text.lower()
    for p in PREFIXES:
        if t.startswith(p):
            return t[len(p):]
    return t


def normalize(text: str) -> str:
    text = strip_prefixes(str(text))
    text = re.sub(r"[_\-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compress(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_prefixes(str(text)))


def lcs_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def token_overlap(a: str, b: str) -> float:
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# ---------------------------------------------------------------------------
# Scoring / mapping
# ---------------------------------------------------------------------------

def score_pair(input_field: str, ref_field: str) -> float:
    c_input = compress(input_field)
    c_ref = compress(ref_field)
    lcs = lcs_ratio(c_input, c_ref)
    tok = token_overlap(input_field, ref_field)
    return 0.7 * lcs + 0.3 * tok


def map_fields(client_fields: list, ref_df: pd.DataFrame) -> pd.DataFrame:
    cotiviti_fields = ref_df[TARGET_COL].astype(str).tolist()
    raw_client_fields = ref_df[RAW_COL].astype(str).tolist()

    rows = []
    for field in client_fields:
        scores = [score_pair(field, ref) for ref in raw_client_fields]
        best_idx = int(np.argmax(scores))
        score = scores[best_idx]
        cotiviti = cotiviti_fields[best_idx]
        matched_ref = raw_client_fields[best_idx]

        if score >= 0.6:
            confidence, status = "High", "Matched"
        elif score >= 0.35:
            confidence, status = "Medium", "Low Confidence"
        else:
            confidence, status = "Low", "Unmatched"

        rows.append({
            "Client Field": field,
            "Suggested Cotiviti Field": cotiviti,
            "Score": round(score, 2),
            "Confidence": confidence,
            "Status": status,
            "Best Reference Match": f"Best match: {matched_ref}",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Excel export helpers
# ---------------------------------------------------------------------------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Two-sheet Excel: Confirmed_Matched + Needs_Review."""
    buf = io.BytesIO()
    confirmed = df[df["Status"].isin(["Matched", "Confirmed", "Edited"])]
    review = df[~df["Status"].isin(["Matched", "Confirmed", "Edited"])]
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        confirmed.to_excel(writer, sheet_name="Confirmed_Matched", index=False)
        review.to_excel(writer, sheet_name="Needs_Review", index=False)
    return buf.getvalue()


def ref_to_excel_bytes(ref_df: pd.DataFrame) -> bytes:
    """Updated reference file for download."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ref_df.to_excel(writer, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Load reference
# ---------------------------------------------------------------------------

def load_reference(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name=0)
    elif os.path.exists(BUNDLED_REF):
        df = pd.read_excel(BUNDLED_REF, sheet_name=0)
    else:
        st.error(
            f"No reference file found. Please upload `{os.path.basename(BUNDLED_REF)}` "
            "using the sidebar uploader."
        )
        st.stop()

    if RAW_COL not in df.columns or TARGET_COL not in df.columns:
        st.error(
            f"Reference file must contain columns **'{RAW_COL}'** and **'{TARGET_COL}'**."
        )
        st.stop()

    df = df[df[RAW_COL].astype(str).str.lower() != "src col name"]
    df = df.dropna(subset=[RAW_COL, TARGET_COL]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Client → Cotiviti Field Mapper",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Client → Cotiviti Data Field Mapper")
st.caption("Local AI fuzzy matching — no API key required.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_ref = st.file_uploader(
        "Upload reference file (optional)",
        type=["xlsx"],
        help=f"Expected columns: '{RAW_COL}' and '{TARGET_COL}'.",
    )

    threshold_high = st.slider("High-confidence threshold", 0.0, 1.0, 0.60, 0.01)
    threshold_med = st.slider("Medium-confidence threshold", 0.0, 1.0, 0.35, 0.01)

    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "- Paste client field names (one per line).\n"
        "- Click **Run Mapping**.\n"
        "- Edit any *Suggested Cotiviti Field* inline.\n"
        "- Download results as Excel.\n"
        "- Optionally download the updated reference."
    )

# ── Main area ────────────────────────────────────────────────────────────────
ref_df = load_reference(uploaded_ref)
st.success(f"Reference loaded: **{len(ref_df)}** rows")

client_text = st.text_area(
    "Paste client field names (one per line):",
    height=180,
    placeholder="e.g.\nCLM_ID\nMEMBER_DOB\nPROVIDER_NPI",
)

run_col, _, _ = st.columns([1, 3, 3])
run_clicked = run_col.button("▶  Run Mapping", type="primary", use_container_width=True)

if run_clicked:
    client_fields = [f.strip() for f in client_text.splitlines() if f.strip()]
    if not client_fields:
        st.warning("Please paste at least one client field name.")
    else:
        with st.spinner("Running semantic matching…"):
            result_df = map_fields(client_fields, ref_df)
        st.session_state["result_df"] = result_df

# ── Results ───────────────────────────────────────────────────────────────────
if "result_df" in st.session_state:
    df = st.session_state["result_df"]

    high = (df["Confidence"] == "High").sum()
    med = (df["Confidence"] == "Medium").sum()
    low = (df["Confidence"] == "Low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("✅ High confidence", high)
    c3.metric("⚠️ Medium confidence", med)
    c4.metric("❌ Low confidence", low)

    st.markdown("### Results  _(edit **Suggested Cotiviti Field** inline)_")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Confidence": st.column_config.SelectboxColumn(
                options=["High", "Medium", "Low"], required=True
            ),
            "Status": st.column_config.SelectboxColumn(
                options=["Matched", "Confirmed", "Low Confidence", "Unmatched", "Edited"],
                required=True,
            ),
        },
        disabled=["Client Field", "Score", "Confidence", "Status", "Best Reference Match"],
        key="result_editor",
    )

    st.markdown("---")
    dl1, dl2, dl3 = st.columns(3)

    dl1.download_button(
        label="⬇️ Download Results (.xlsx)",
        data=df_to_excel_bytes(edited_df),
        file_name="output_client_to_cotiviti_mapping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # Build updated reference with confirmed mappings appended
    existing_lower = set(ref_df[RAW_COL].astype(str).str.lower())
    new_rows = [
        {RAW_COL: r["Client Field"], TARGET_COL: r["Suggested Cotiviti Field"]}
        for _, r in edited_df.iterrows()
        if r["Client Field"].lower() not in existing_lower
    ]
    updated_ref = pd.concat(
        [ref_df, pd.DataFrame(new_rows)], ignore_index=True
    ) if new_rows else ref_df

    dl2.download_button(
        label="⬇️ Download Updated Reference (.xlsx)",
        data=ref_to_excel_bytes(updated_ref),
        file_name="Data_Mod_Mapping_Reference_updated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help=f"{len(new_rows)} new row(s) will be appended.",
    )

    if dl3.button("🗑️ Clear results", use_container_width=True):
        del st.session_state["result_df"]
        st.rerun()
