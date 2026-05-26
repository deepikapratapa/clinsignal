# ClinSignal

RAG-powered adverse event signal detection from unstructured SDTM clinical trial narratives.

## What it does

ClinSignal detects latent pharmacovigilance signals from adverse event narratives in SDTM-formatted clinical trial data. Unlike traditional signal detection that relies solely on coded MedDRA terms, ClinSignal extracts meaning from unstructured narrative text using NLP and grounds signal assessments in biomedical ontology knowledge via RAG.

## Pipeline

Step 1 - SDTM Ingestion: pandas, SAS XPT parsing. Output: clean AE dataset with narratives.
Step 2 - NLP Extraction: scispaCy NER and sentence-transformers. Output: entity lists and embeddings.
Step 3 - Signal Clustering: BERTopic (UMAP and HDBSCAN). Output: 51 signal clusters.
Step 4 - RAG Grounding: ChromaDB and MedDRA/DrugBank knowledge. Output: context-enriched signals.
Step 5 - LLM Assessment: Ollama/Mistral local inference. Output: structured signal assessments.

## Dataset

CDISC Pilot SDTM Dataset (public) — Xanomeline vs Placebo, Alzheimer's Disease.
1,191 adverse event records, 225 subjects, 3 treatment arms.
Domains: AE, DM, CM, SUPPAE, ADAE.

## Results

- 51 signal clusters discovered via BERTopic
- 15 top signals assessed via RAG-grounded Mistral LLM
- 4 known signals confirmed (nervous system, GI, application site, cardiac)
- 11 potential new signals flagged for investigation
- Dose-dependent neurological signal: 42 high dose vs 3 placebo records

## Setup

1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Pull model: ollama pull mistral
3. Create env: conda env create -f envs/clinsignal_env.yml
4. Activate: conda activate clinsignal
5. Install scispaCy model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
6. Run pipeline: bash run_pipeline.sh
7. Launch app: streamlit run app/app.py

## Stack

Python, scispaCy, BERTopic, sentence-transformers, ChromaDB, LangChain, Ollama/Mistral, Streamlit, pandas, plotly

## Author

Deepika Sarala Pratapa
MS Applied Data Science, University of Florida
GitHub: https://github.com/deepikapratapa
