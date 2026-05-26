"""
NLP Signal Extraction Pipeline
- scispaCy NER on AE narratives
- Sentence embeddings via sentence-transformers
- BERTopic clustering to find signal candidates
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = Path("data/processed")
RESULTS = Path("results/signals")
RESULTS.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC / "ae_analysis_ready.csv")
    narratives = df["NARRATIVE"].fillna("").astype(str)
    print(f"Loaded {len(df)} AE records")
    print(f"Unique narratives: {narratives.nunique()}")
    return df

def run_ner(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """Extract named entities from narratives using scispaCy."""
    print("\nRunning scispaCy NER...")
    
    entities_list = []
    unique_narratives = df["NARRATIVE"].drop_duplicates().reset_index(drop=True)
    narr_to_entities = {}
    
    for i, text in enumerate(unique_narratives):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(unique_narratives)}...")
        doc = nlp(str(text))
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if len(ent.text) > 2
        ]
        narr_to_entities[text] = entities
    
    df["ENTITIES"] = df["NARRATIVE"].map(narr_to_entities)
    df["ENTITY_TEXTS"] = df["ENTITIES"].apply(
        lambda x: [e["text"] for e in x] if isinstance(x, list) else []
    )
    df["N_ENTITIES"] = df["ENTITY_TEXTS"].apply(len)
    
    # Flatten all entities for frequency analysis
    all_entities = []
    for ents in df["ENTITIES"]:
        if isinstance(ents, list):
            all_entities.extend(ents)
    
    entity_df = pd.DataFrame(all_entities)
    print(f"\nTotal entities extracted: {len(entity_df)}")
    if len(entity_df) > 0:
        print(f"Entity types: {entity_df['label'].value_counts().to_dict()}")
        print(f"\nTop 15 entities:")
        print(entity_df["text"].value_counts().head(15))
    
    return df, entity_df

def run_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    """Generate sentence embeddings for all unique narratives."""
    print("\nGenerating sentence embeddings...")
    
    unique_narr = df["NARRATIVE"].unique().tolist()
    print(f"Embedding {len(unique_narr)} unique narratives...")
    
    embeddings_map = {}
    batch_size = 32
    
    for i in range(0, len(unique_narr), batch_size):
        batch = unique_narr[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        for text, emb in zip(batch, batch_embeddings):
            embeddings_map[text] = emb
        if i % 100 == 0:
            print(f"  {i}/{len(unique_narr)} embedded...")
    
    embeddings = np.array([embeddings_map[n] for n in df["NARRATIVE"]])
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    # Save embeddings
    np.save(RESULTS / "narrative_embeddings.npy", embeddings)
    print(f"Embeddings saved to {RESULTS}/narrative_embeddings.npy")
    
    return embeddings

def run_bertopic(df: pd.DataFrame, embeddings: np.ndarray) -> tuple:
    """Cluster narratives into signal topics using BERTopic."""
    print("\nRunning BERTopic clustering...")
    
    narratives = df["NARRATIVE"].tolist()
    
    # Configure BERTopic
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_features=5000
    )
    
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        nr_topics="auto",
        min_topic_size=5,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(narratives, embeddings)
    
    df["TOPIC"] = topics
    df["TOPIC_PROB"] = probs
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"\nTopics discovered: {len(topic_info) - 1}")  # -1 for outlier topic
    print("\nTop topics:")
    print(topic_info.head(10).to_string())
    
    # Save topic info
    topic_info.to_csv(RESULTS / "topic_info.csv", index=False)
    
    # Get top words per topic
    topics_dict = {}
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        topics_dict[int(topic_id)] = {
            "words": [w for w, _ in words[:10]],
            "scores": [float(s) for _, s in words[:10]],
            "count": int(topic_info[topic_info["Topic"]==topic_id]["Count"].values[0])
        }
    
    with open(RESULTS / "topics.json", "w") as f:
        json.dump(topics_dict, f, indent=2)
    
    return df, topic_model, topic_info

def generate_signal_candidates(df: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signal candidates from topic clusters.
    A signal candidate is a topic cluster with clinical significance markers.
    """
    print("\nGenerating signal candidates...")
    
    signals = []
    
    for topic_id in df["TOPIC"].unique():
        if topic_id == -1:
            continue
        
        topic_df = df[df["TOPIC"] == topic_id]
        
        # Signal features
        n_records = len(topic_df)
        n_subjects = topic_df["USUBJID"].nunique()
        serious_count = topic_df["IS_SERIOUS"].sum() if "IS_SERIOUS" in topic_df.columns else 0
        
        # Body system distribution
        if "AEBODSYS" in topic_df.columns:
            top_soc = topic_df["AEBODSYS"].value_counts().index[0] if len(topic_df) > 0 else "Unknown"
        else:
            top_soc = "Unknown"
        
        # Treatment arm distribution
        arm_counts = {}
        if "ARM" in topic_df.columns:
            arm_counts = topic_df["ARM"].value_counts().to_dict()
        
        # Severity distribution
        sev_counts = {}
        if "AESEV" in topic_df.columns:
            sev_counts = topic_df["AESEV"].value_counts().to_dict()
        
        # Top terms in this cluster
        if "AEDECOD" in topic_df.columns:
            top_terms = topic_df["AEDECOD"].value_counts().head(5).to_dict()
        else:
            top_terms = {}
        
        # Signal score — higher if: more subjects, more serious, 
        # concentrated in treatment vs placebo
        placebo_n = arm_counts.get("Placebo", 0)
        treatment_n = n_records - placebo_n
        
        signal_score = (
            min(n_subjects / 10, 1.0) * 0.3 +
            min(serious_count / max(n_records, 1), 1.0) * 0.4 +
            (treatment_n / max(n_records, 1)) * 0.3
        )
        
        signals.append({
            "topic_id": int(topic_id),
            "n_records": int(n_records),
            "n_subjects": int(n_subjects),
            "serious_count": int(serious_count),
            "top_soc": top_soc,
            "top_terms": json.dumps(top_terms),
            "arm_distribution": json.dumps(arm_counts),
            "severity_distribution": json.dumps(sev_counts),
            "signal_score": round(float(signal_score), 4)
        })
    
    signals_df = pd.DataFrame(signals).sort_values("signal_score", ascending=False)
    signals_df.to_csv(RESULTS / "signal_candidates.csv", index=False)
    
    print(f"Signal candidates generated: {len(signals_df)}")
    print(f"\nTop 10 signal candidates by score:")
    print(signals_df.head(10)[["topic_id","n_records","n_subjects",
                                "serious_count","top_soc","signal_score"]].to_string())
    
    return signals_df

def main():
    print("=" * 50)
    print("ClinSignal NLP Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Load models
    print("\nLoading scispaCy model...")
    nlp = spacy.load("en_core_sci_lg")
    nlp.max_length = 2000000
    
    print("Loading sentence transformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Run NER
    df, entity_df = run_ner(df, nlp)
    entity_df.to_csv(RESULTS / "entities.csv", index=False)
    
    # Run embeddings
    embeddings = run_embeddings(df, model)
    
    # Run BERTopic
    df, topic_model, topic_info = run_bertopic(df, embeddings)
    
    # Save enriched AE dataset
    df_save = df.drop(columns=["ENTITIES", "ENTITY_TEXTS"], errors="ignore")
    df_save.to_csv(RESULTS / "ae_with_signals.csv", index=False)
    
    # Generate signal candidates
    signals_df = generate_signal_candidates(df, topic_info)
    
    print("\n" + "=" * 50)
    print("NLP Pipeline Complete")
    print(f"Outputs in {RESULTS}/")
    print("  - narrative_embeddings.npy")
    print("  - entities.csv")
    print("  - topic_info.csv")
    print("  - topics.json")
    print("  - signal_candidates.csv")
    print("  - ae_with_signals.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()
