# ClinSignal

RAG-powered adverse event signal detection from unstructured SDTM clinical trial narratives.

## What it does
Extracts latent safety signals from free-text adverse event narratives in SDTM-formatted clinical trial data, grounds detected signals in MedDRA and DrugBank ontologies via RAG, and benchmarks recovery against established FAERS pharmacovigilance signals.

## Pipeline
1. SDTM ingestion (AE, DM, CM domains)
2. NLP signal extraction (scispaCy NER + ClinicalBERT embeddings + BERTopic)
3. RAG grounding (MedDRA + DrugBank vector store)
4. Signal validation (FAERS benchmark)
5. Streamlit deployment

## Data sources
- CDISC pilot SDTM datasets (public)
- FDA FAERS (public)
- MedDRA subset / WHO-DD subset (public training versions)
- DrugBank vocabulary (open data)

## Stack
Python, scispaCy, BERTopic, LangChain, ChromaDB, ClinicalBERT, Streamlit, HuggingFace

## Results
*To be updated after analysis*
