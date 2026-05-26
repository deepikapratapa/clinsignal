#!/bin/bash
# ClinSignal — Full Pipeline Runner
# Runs all pipeline steps in order

set -e

echo "=================================================="
echo "ClinSignal Pipeline"
echo "=================================================="

PYTHON="/opt/anaconda3/envs/clinsignal/bin/python3"

echo ""
echo "Step 1/4 — Converting XPT to CSV..."
$PYTHON scripts/ingestion/00_convert_xpt.py

echo ""
echo "Step 2/4 — Parsing SDTM AE domain..."
$PYTHON scripts/ingestion/02_parse_sdtm_ae.py

echo ""
echo "Step 3/4 — Running NLP pipeline..."
$PYTHON scripts/nlp/03_nlp_pipeline.py

echo ""
echo "Step 4/4 — Running RAG assessment..."
$PYTHON scripts/rag/04_rag_pipeline.py

echo ""
echo "=================================================="
echo "Pipeline complete. Launch app with:"
echo "  streamlit run app/app.py"
echo "=================================================="
