"""
RAG Grounding Layer
- Builds vector store from MedDRA SOC/PT descriptions and drug knowledge
- For each signal candidate, retrieves relevant context
- Uses Ollama/Mistral to generate structured signal assessments
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import requests
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

RESULTS = Path("results/signals")
VALIDATION = Path("results/validation")
VALIDATION.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# ── Knowledge Base ──────────────────────────────────────────────────────────
# MedDRA System Organ Classes with clinical descriptions
MEDDRA_SOC_KNOWLEDGE = [
    {
        "id": "soc_nervous",
        "text": "Nervous System Disorders (MedDRA SOC): Includes conditions affecting the central and peripheral nervous system. Key adverse events include dizziness, headache, syncope, tremor, seizures, cognitive impairment, somnolence, and balance disorders. Cholinergic drugs commonly cause nervous system effects including dizziness and syncope due to vasovagal mechanisms. Dose-dependent neurological effects are a primary safety concern for muscarinic agonists.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_cardiac",
        "text": "Cardiac Disorders (MedDRA SOC): Includes conditions affecting heart structure and function. Key adverse events include bradycardia, tachycardia, atrial fibrillation, myocardial infarction, and cardiac arrest. Muscarinic agonists like xanomeline can cause bradycardia through M2 receptor activation on cardiac tissue. Sinus bradycardia is a known pharmacological effect of cholinergic agents.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_skin",
        "text": "Skin and Subcutaneous Tissue Disorders (MedDRA SOC): Includes dermatological conditions. Key adverse events include rash, pruritus, erythema, urticaria, and dermatitis. Transdermal drug delivery systems commonly cause local skin reactions at the application site due to both pharmacological effects and physical irritation from the patch adhesive. Pruritus and erythema are among the most common adverse events for transdermal patches.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_gi",
        "text": "Gastrointestinal Disorders (MedDRA SOC): Includes conditions affecting the digestive system. Key adverse events include nausea, vomiting, diarrhea, abdominal pain, and constipation. Muscarinic agonists stimulate M3 receptors in the GI tract, increasing motility and secretions. Nausea and vomiting are among the most common dose-limiting adverse effects of cholinergic drugs.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_general",
        "text": "General Disorders and Administration Site Conditions (MedDRA SOC): Includes systemic conditions and local reactions at drug administration sites. Application site reactions are particularly common with transdermal formulations. These include erythema, pruritus, irritation, warmth, and discharge at the patch site. These reactions may be caused by the drug itself, excipients, or mechanical effects of the adhesive.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_psychiatric",
        "text": "Psychiatric Disorders (MedDRA SOC): Includes mental health conditions. Key adverse events include agitation, hallucinations, confusional state, anxiety, and depression. In Alzheimer's disease trials, distinguishing drug-related psychiatric effects from disease progression is challenging. Cholinergic agents have been associated with vivid dreams and agitation.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_respiratory",
        "text": "Respiratory, Thoracic and Mediastinal Disorders (MedDRA SOC): Includes conditions affecting the respiratory system. Key adverse events include cough, dyspnea, bronchospasm, and rhinorrhea. Muscarinic agonists can increase bronchial secretions and cause bronchoconstriction through M3 receptor activation in the airways.",
        "category": "MedDRA_SOC"
    },
    {
        "id": "soc_infections",
        "text": "Infections and Infestations (MedDRA SOC): Includes infectious diseases. Nasopharyngitis, urinary tract infections, and upper respiratory infections are commonly reported in clinical trials, often as background noise rather than drug-related signals. Distinguishing drug-related immunosuppression from background infection rates requires comparison with placebo.",
        "category": "MedDRA_SOC"
    }
]

# Xanomeline-specific knowledge
DRUG_KNOWLEDGE = [
    {
        "id": "xanomeline_moa",
        "text": "Xanomeline is a selective M1/M4 muscarinic acetylcholine receptor agonist investigated for Alzheimer's disease. It was administered as a transdermal patch in the CDISC pilot study. Mechanism: activates muscarinic receptors in the CNS to improve cognitive function. Known safety profile includes cholinergic adverse effects: nausea, vomiting, diarrhea, sweating, salivation, bradycardia, and application site reactions from the transdermal formulation.",
        "category": "Drug_MOA"
    },
    {
        "id": "xanomeline_safety",
        "text": "Xanomeline clinical safety signals: The most commonly reported adverse events in xanomeline trials are application site reactions (erythema, pruritus, irritation) due to transdermal delivery, gastrointestinal effects (nausea, vomiting, diarrhea) from peripheral M3 receptor activation, cardiovascular effects (bradycardia, syncope) from M2 receptor activation, and neurological effects (dizziness, headache) at higher doses. The dose-dependent nature of these effects is a key pharmacological characteristic.",
        "category": "Drug_Safety"
    },
    {
        "id": "xanomeline_dose",
        "text": "Xanomeline dose-response relationship: Higher doses of xanomeline are associated with increased frequency and severity of cholinergic adverse effects. In the CDISC pilot study, xanomeline was tested at low and high doses versus placebo. Dose-dependent increases in nervous system disorders, application site reactions, and gastrointestinal effects are expected pharmacological findings consistent with muscarinic receptor activation.",
        "category": "Drug_Dose"
    },
    {
        "id": "alzheimer_context",
        "text": "Alzheimer's disease clinical trial context: Patients in AD trials are typically elderly with multiple comorbidities including cardiovascular disease, diabetes, and prior neurological events. Background rates of cardiovascular events, falls, and cognitive decline are high. Distinguishing drug-related adverse events from disease progression or comorbidity-related events requires careful causality assessment. Syncope and cardiac events in elderly AD patients may reflect underlying disease rather than drug effect.",
        "category": "Disease_Context"
    }
]

# Pharmacovigilance signal criteria
SIGNAL_CRITERIA = [
    {
        "id": "signal_definition",
        "text": "Pharmacovigilance signal definition (WHO): A signal is information that arises from one or multiple sources, including observations and experiments, which suggests a new potentially causal association, or a new aspect of a known association, between an intervention and an event or set of related events, either adverse or beneficial, that is judged to be of sufficient likelihood to justify verificatory action. Key signal criteria include: biological plausibility, dose-response relationship, temporal association, consistency across cases, and strength of association.",
        "category": "PV_Methodology"
    },
    {
        "id": "disproportionality",
        "text": "Disproportionality analysis in pharmacovigilance: Signal detection commonly uses reporting odds ratio (ROR) and proportional reporting ratio (PRR) to identify drug-event combinations reported more frequently than expected. A PRR >= 2 with chi-square >= 4 and at least 3 cases is a common threshold for signal detection. Treatment-to-placebo ratio of adverse events is a primary comparator in randomized clinical trials.",
        "category": "PV_Methodology"
    }
]

def build_knowledge_base(model: SentenceTransformer) -> chromadb.Collection:
    """Build ChromaDB vector store from clinical knowledge."""
    print("Building RAG knowledge base...")
    
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Delete existing collection if present
    try:
        client.delete_collection("clinsignal_kb")
    except:
        pass
    
    collection = client.create_collection(
        name="clinsignal_kb",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Combine all knowledge
    all_docs = MEDDRA_SOC_KNOWLEDGE + DRUG_KNOWLEDGE + SIGNAL_CRITERIA
    
    texts = [doc["text"] for doc in all_docs]
    ids = [doc["id"] for doc in all_docs]
    metadatas = [{"category": doc["category"]} for doc in all_docs]
    
    embeddings = model.encode(texts).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Knowledge base built: {len(all_docs)} documents indexed")
    return collection

def retrieve_context(signal: dict, collection: chromadb.Collection, 
                     model: SentenceTransformer, n_results: int = 4) -> str:
    """Retrieve relevant knowledge for a signal candidate."""
    
    query = f"{signal['top_soc']} adverse event signal clinical trial {signal.get('top_terms_list', '')}"
    
    query_embedding = model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    context = "\n\n".join(results["documents"][0])
    return context

def assess_signal_with_llm(signal: dict, context: str) -> dict:
    """Use Ollama/Mistral to generate structured signal assessment."""
    
    top_terms = json.loads(signal.get("top_terms", "{}"))
    arms = json.loads(signal.get("arm_distribution", "{}"))
    
    terms_list = list(top_terms.keys())[:5]
    arm_str = ", ".join([f"{k}: {v}" for k,v in arms.items()])
    
    prompt = f"""You are a pharmacovigilance expert. Respond with ONLY a JSON object, no explanation, no markdown, no backticks.

Signal data: SOC={signal['top_soc']}, terms={terms_list}, records={signal['n_records']}, subjects={signal['n_subjects']}, serious={signal['serious_count']}, arms=[{arm_str}]

Context: {context[:500]}

Respond with exactly this JSON structure:
{{"signal_label": "5 word label", "signal_type": "Known_Signal or Potential_New_Signal or Background_Noise", "biological_plausibility": "High or Medium or Low", "dose_response": "Yes or No or Unclear", "clinical_significance": "High or Medium or Low", "assessment": "2 sentence interpretation", "recommended_action": "Monitor or Investigate_Further or Label_Update or No_Action"}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            raw = response.json()["response"].strip()
            # Try multiple JSON extraction strategies
            # Strategy 1: direct parse
            try:
                return json.loads(raw)
            except:
                pass
            # Strategy 2: find outermost braces
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except:
                    pass
            # Strategy 3: clean common issues and retry
            cleaned = raw.replace("\n", " ").replace("\'", "'")
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except:
                    pass
            # Strategy 4: return partial parse with raw text
            return {
                "signal_label": "Parse error - manual review needed",
                "signal_type": "Unknown",
                "biological_plausibility": "Unknown",
                "dose_response": "Unknown",
                "clinical_significance": "Unknown",
                "assessment": raw[:300] if raw else "No response",
                "recommended_action": "Investigate_Further",
                "parse_error": True
            }
        
        return {"error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 50)
    print("ClinSignal RAG Grounding Layer")
    print("=" * 50)
    
    # Load signal candidates
    signals = pd.read_csv(RESULTS / "signal_candidates.csv")
    print(f"Loaded {len(signals)} signal candidates")
    
    # Load sentence transformer
    print("Loading sentence transformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Build knowledge base
    collection = build_knowledge_base(model)
    
    # Check Ollama is running
    print("\nChecking Ollama connection...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running. Available models: {models}")
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        print("Start Ollama with: ollama serve")
        return
    
    # Process top 15 signal candidates
    top_signals = signals.head(15)
    assessments = []
    
    print(f"\nAssessing top {len(top_signals)} signal candidates...")
    print("-" * 50)
    
    for i, (_, signal) in enumerate(top_signals.iterrows()):
        print(f"\n[{i+1}/{len(top_signals)}] Topic {signal['topic_id']} — {signal['top_soc']}")
        
        # Add terms list for query
        terms = json.loads(signal.get("top_terms", "{}"))
        signal_dict = signal.to_dict()
        signal_dict["top_terms_list"] = " ".join(list(terms.keys())[:3])
        
        # Retrieve context
        context = retrieve_context(signal_dict, collection, model)
        
        # Get LLM assessment
        assessment = assess_signal_with_llm(signal_dict, context)
        
        if "error" not in assessment:
            print(f"  Label: {assessment.get('signal_label', 'N/A')}")
            print(f"  Type: {assessment.get('signal_type', 'N/A')}")
            print(f"  Clinical significance: {assessment.get('clinical_significance', 'N/A')}")
            print(f"  Action: {assessment.get('recommended_action', 'N/A')}")
        else:
            print(f"  Error: {assessment['error']}")
        
        assessment["topic_id"] = int(signal["topic_id"])
        assessment["top_soc"] = signal["top_soc"]
        assessment["n_records"] = int(signal["n_records"])
        assessment["n_subjects"] = int(signal["n_subjects"])
        assessment["serious_count"] = int(signal["serious_count"])
        assessment["signal_score"] = float(signal["signal_score"])
        
        assessments.append(assessment)
    
    # Save assessments
    assessments_df = pd.DataFrame(assessments)
    assessments_df.to_csv(VALIDATION / "signal_assessments.csv", index=False)
    
    print("\n" + "=" * 50)
    print("RAG Assessment Complete")
    print(f"Assessed {len(assessments)} signals")
    print(f"Saved to {VALIDATION}/signal_assessments.csv")
    
    # Summary
    if "signal_type" in assessments_df.columns:
        print(f"\nSignal type distribution:")
        print(assessments_df["signal_type"].value_counts())
    if "recommended_action" in assessments_df.columns:
        print(f"\nRecommended actions:")
        print(assessments_df["recommended_action"].value_counts())

if __name__ == "__main__":
    main()
