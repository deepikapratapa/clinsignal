import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw/sdtm")
OUT = Path("data/processed")
OUT.mkdir(exist_ok=True)

domains = {
    "ae": RAW / "ae.xpt",
    "dm": RAW / "dm.xpt",
    "cm": RAW / "cm.xpt",
    "suppae": RAW / "suppae.xpt",
    "adae": RAW / "adae.xpt",
}

summary = {}

for name, path in domains.items():
    print(f"Converting {name.upper()}...")
    df = pd.read_sas(path, encoding="latin-1")
    df.columns = df.columns.str.upper()
    out_path = OUT / f"{name.upper()}.csv"
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} rows, {len(df.columns)} cols -> {out_path}")
    print(f"  Columns: {df.columns.tolist()}")
    print()
    summary[name] = {"rows": len(df), "cols": df.columns.tolist()}

with open(OUT / "domain_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("All domains converted.")
