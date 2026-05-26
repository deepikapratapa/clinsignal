import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

PROC = Path("data/processed")
RAW = Path("data/raw/sdtm")

def load_narratives(path: Path) -> dict:
    narratives = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.search(r"Subject (\d{2}-\d{3}-\d{4})", block)
        if match:
            subj_id = match.group(1)
            narratives[subj_id] = block
    print(f"Parsed {len(narratives)} clinical narratives")
    return narratives

def construct_narrative(row) -> str:
    parts = []
    if pd.notna(row.get("AETERM")):
        parts.append(f"The subject experienced {str(row['AETERM']).lower()}")
    if pd.notna(row.get("AEBODSYS")):
        parts.append(f"affecting the {str(row['AEBODSYS']).lower()}")
    if pd.notna(row.get("AESEV")):
        parts.append(f"Severity: {str(row['AESEV']).lower()}")
    if pd.notna(row.get("AESER")) and str(row["AESER"]).upper() == "Y":
        parts.append("The event was serious")
    if pd.notna(row.get("AEREL")):
        parts.append(f"Relationship to study drug: {str(row['AEREL']).lower()}")
    if pd.notna(row.get("AEOUT")):
        parts.append(f"Outcome: {str(row['AEOUT']).lower()}")
    if pd.notna(row.get("ARM")):
        parts.append(f"Treatment arm: {str(row['ARM']).lower()}")
    return ". ".join(parts) + "." if parts else ""

def get_narrative(row, narratives):
    usubjid = str(row["USUBJID"])
    match = re.search(r"(\d{3}-\d{4})", usubjid)
    if match:
        short_id = match.group(1)
        for key in narratives:
            if short_id in key:
                return narratives[key]
    return construct_narrative(row)

def main():
    print("=== SDTM AE Domain Parser ===\n")
    ae = pd.read_csv(PROC / "AE.csv")
    dm = pd.read_csv(PROC / "DM.csv")
    cm = pd.read_csv(PROC / "CM.csv")
    print(f"AE: {len(ae)} records")
    print(f"DM: {len(dm)} subjects")
    print(f"CM: {len(cm)} records\n")

    dm_cols = ["USUBJID", "AGE", "SEX", "RACE", "ARMCD", "ARM", "DTHFL"]
    dm_sub = dm[[c for c in dm_cols if c in dm.columns]].drop_duplicates("USUBJID")
    ae_dm = ae.merge(dm_sub, on="USUBJID", how="left")

    conmeds = cm.groupby("USUBJID")["CMTRT"].apply(
        lambda x: "; ".join(x.dropna().astype(str).unique())
    ).reset_index()
    conmeds.columns = ["USUBJID", "CONMEDS"]
    ae_final = ae_dm.merge(conmeds, on="USUBJID", how="left")

    narratives = load_narratives(RAW / "narratives.txt")

    print("Mapping narratives to AE records...")
    ae_final["NARRATIVE"] = ae_final.apply(
        lambda row: get_narrative(row, narratives), axis=1
    )

    real_narr = ae_final["NARRATIVE"].apply(
        lambda x: any(s in str(x) for s in ["began receiving", "diagnosed", "experienced"])
    ).sum()
    print(f"Real clinical narratives mapped: {real_narr}")
    print(f"Constructed pseudo-narratives: {len(ae_final) - real_narr}")

    ae_final["IS_SERIOUS"] = ae_final["AESER"].str.upper() == "Y"
    sev_map = {"MILD": 1, "MODERATE": 2, "SEVERE": 3}
    ae_final["SEV_GRADE"] = ae_final["AESEV"].str.upper().map(sev_map)

    out_path = PROC / "ae_analysis_ready.csv"
    ae_final.to_csv(out_path, index=False)

    print(f"\n=== Dataset Summary ===")
    print(f"Total AE records: {len(ae_final)}")
    print(f"Unique subjects: {ae_final['USUBJID'].nunique()}")
    print(f"Serious AEs: {ae_final['IS_SERIOUS'].sum()}")
    print(f"Treatment arms:\n{ae_final['ARM'].value_counts()}")
    print(f"\nTop body systems:\n{ae_final['AEBODSYS'].value_counts().head(8)}")
    print(f"\nSample narrative:")
    sample = ae_final[ae_final["IS_SERIOUS"]]["NARRATIVE"].iloc[0]
    print(sample[:400])
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
