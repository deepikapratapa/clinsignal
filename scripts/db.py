import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "clinsignal")
DB_USER = os.getenv("DB_USER", "clinsignal")
DB_PASSWORD = os.getenv("DB_PASSWORD", "clinsignal_dev_2024")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_engine():
    return create_engine(DATABASE_URL)

def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"Connected: {version[:60]}")
            return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def write_dataframe(df, table, data_source="cdisc_pilot"):
    engine = get_engine()
    df = df.copy()
    df["data_source"] = data_source
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"Wrote {len(df)} rows to {table}")

def read_table(table):
    return pandas.read_sql(f"SELECT * FROM {table}", get_engine())

def get_stats():
    engine = get_engine()
    with engine.connect() as conn:
        stats = {}
        for t in ["ae_records","signal_candidates","signal_assessments","faers_reports"]:
            stats[t] = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).fetchone()[0]
        return stats

if __name__ == "__main__":
    test_connection()
    print(get_stats())
