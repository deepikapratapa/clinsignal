-- ClinSignal Database Schema

CREATE TABLE IF NOT EXISTS ae_records (
    id SERIAL PRIMARY KEY,
    usubjid VARCHAR(50),
    aeterm TEXT,
    aedecod TEXT,
    aebodsys TEXT,
    aesev VARCHAR(20),
    aeser VARCHAR(5),
    aerel VARCHAR(50),
    aeout TEXT,
    arm VARCHAR(100),
    age FLOAT,
    sex VARCHAR(10),
    race VARCHAR(100),
    conmeds TEXT,
    narrative TEXT,
    is_serious BOOLEAN,
    sev_grade INTEGER,
    topic_id INTEGER,
    data_source VARCHAR(50),
    loaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signal_candidates (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER,
    n_records INTEGER,
    n_subjects INTEGER,
    serious_count INTEGER,
    top_soc TEXT,
    top_terms JSONB,
    arm_distribution JSONB,
    severity_distribution JSONB,
    signal_score FLOAT,
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signal_assessments (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER,
    top_soc TEXT,
    signal_label TEXT,
    signal_type VARCHAR(50),
    biological_plausibility VARCHAR(20),
    dose_response VARCHAR(20),
    clinical_significance VARCHAR(20),
    assessment TEXT,
    recommended_action VARCHAR(100),
    n_records INTEGER,
    n_subjects INTEGER,
    serious_count INTEGER,
    signal_score FLOAT,
    data_source VARCHAR(50),
    assessed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS faers_reports (
    id SERIAL PRIMARY KEY,
    primaryid VARCHAR(50),
    caseid VARCHAR(50),
    drug_name TEXT,
    reaction TEXT,
    outcome TEXT,
    reporter_country VARCHAR(10),
    report_quarter VARCHAR(10),
    narrative TEXT,
    loaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE,
    data_source VARCHAR(50),
    status VARCHAR(20),
    records_processed INTEGER,
    signals_detected INTEGER,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_ae_usubjid ON ae_records(usubjid);
CREATE INDEX IF NOT EXISTS idx_ae_aebodsys ON ae_records(aebodsys);
CREATE INDEX IF NOT EXISTS idx_ae_topic ON ae_records(topic_id);
CREATE INDEX IF NOT EXISTS idx_signals_score ON signal_candidates(signal_score DESC);
CREATE INDEX IF NOT EXISTS idx_assessments_type ON signal_assessments(signal_type);
CREATE INDEX IF NOT EXISTS idx_faers_drug ON faers_reports(drug_name);
