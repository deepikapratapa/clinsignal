import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(
    page_title="ClinSignal",
    page_icon="🔬",
    layout="wide"
)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_ae_data():
    return pd.read_csv("results/signals/ae_with_signals.csv")

@st.cache_data
def load_signals():
    return pd.read_csv("results/signals/signal_candidates.csv")

@st.cache_data
def load_assessments():
    df = pd.read_csv("results/validation/signal_assessments.csv")
    df["recommended_action"] = df["recommended_action"].apply(
        lambda x: x.split(".")[0].strip() if len(str(x)) > 30 else str(x).strip()
    )
    return df

@st.cache_data
def load_topics():
    with open("results/signals/topics.json") as f:
        return json.load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/microscope.png", width=60)
st.sidebar.title("ClinSignal")
st.sidebar.caption("RAG-powered adverse event signal detection from SDTM clinical trial narratives")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "AE Explorer", "Signal Candidates", "RAG Assessments"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset**")
st.sidebar.caption("CDISC Pilot Study — Xanomeline vs Placebo — Alzheimer's Disease")
st.sidebar.caption("1,191 AE records · 225 subjects · 3 treatment arms")

# ── Page 1: Overview ─────────────────────────────────────────────────────────
if page == "Overview":
    st.title("🔬 ClinSignal")
    st.subheader("RAG-powered adverse event signal detection from SDTM clinical trial narratives")
    st.divider()

    ae = load_ae_data()
    signals = load_signals()
    assessments = load_assessments()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AE Records", f"{len(ae):,}")
    col2.metric("Unique Subjects", f"{ae['USUBJID'].nunique():,}")
    col3.metric("Signal Clusters", f"{len(signals)}")
    col4.metric("Assessed Signals", f"{len(assessments)}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Adverse Events by Body System")
        soc_counts = ae["AEBODSYS"].value_counts().reset_index()
        soc_counts.columns = ["Body System", "Count"]
        fig = px.bar(
            soc_counts.head(10),
            x="Count", y="Body System",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig.update_layout(showlegend=False, height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("AE Distribution by Treatment Arm")
        if "ARM" in ae.columns:
            arm_soc = ae.groupby(["ARM", "AEBODSYS"]).size().reset_index(name="Count")
            top_socs = ae["AEBODSYS"].value_counts().head(6).index.tolist()
            arm_soc_top = arm_soc[arm_soc["AEBODSYS"].isin(top_socs)]
            fig2 = px.bar(
                arm_soc_top,
                x="AEBODSYS", y="Count", color="ARM",
                barmode="group",
                template="plotly_white",
                labels={"AEBODSYS": "Body System", "ARM": "Treatment Arm"}
            )
            fig2.update_layout(height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pipeline Architecture")
    st.markdown("""
    | Step | Component | Technology |
    |------|-----------|------------|
    | 1 | SDTM Ingestion | pandas, SAS XPT parsing |
    | 2 | NLP Extraction | scispaCy NER + sentence-transformers |
    | 3 | Signal Clustering | BERTopic (UMAP + HDBSCAN) |
    | 4 | RAG Grounding | ChromaDB + MedDRA/DrugBank knowledge |
    | 5 | LLM Assessment | Ollama/Mistral local inference |
    """)

# ── Page 2: AE Explorer ───────────────────────────────────────────────────────
elif page == "AE Explorer":
    st.title("Adverse Event Explorer")
    st.caption("Explore individual AE records from the CDISC pilot SDTM dataset")
    st.divider()

    ae = load_ae_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        soc_filter = st.multiselect(
            "Body System",
            options=sorted(ae["AEBODSYS"].dropna().unique()),
            default=[]
        )
    with col2:
        arm_filter = st.multiselect(
            "Treatment Arm",
            options=sorted(ae["ARM"].dropna().unique()) if "ARM" in ae.columns else [],
            default=[]
        )
    with col3:
        serious_filter = st.selectbox(
            "Serious AEs only",
            options=["All", "Serious only", "Non-serious only"]
        )

    filtered = ae.copy()
    if soc_filter:
        filtered = filtered[filtered["AEBODSYS"].isin(soc_filter)]
    if arm_filter:
        filtered = filtered[filtered["ARM"].isin(arm_filter)]
    if serious_filter == "Serious only":
        filtered = filtered[filtered["IS_SERIOUS"] == True]
    elif serious_filter == "Non-serious only":
        filtered = filtered[filtered["IS_SERIOUS"] == False]

    st.markdown(f"**{len(filtered):,} records** matching filters")

    display_cols = ["USUBJID", "AETERM", "AEDECOD", "AEBODSYS", "AESEV",
                    "AESER", "AEREL", "AEOUT", "ARM"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=400)

    st.divider()
    st.subheader("Narrative Viewer")
    if len(filtered) > 0:
        selected_idx = st.selectbox(
            "Select record",
            options=range(min(50, len(filtered))),
            format_func=lambda i: f"{filtered.iloc[i]['USUBJID']} — {filtered.iloc[i]['AEDECOD']}"
        )
        row = filtered.iloc[selected_idx]
        st.info(row.get("NARRATIVE", "No narrative available"))

# ── Page 3: Signal Candidates ─────────────────────────────────────────────────
elif page == "Signal Candidates":
    st.title("Signal Candidates")
    st.caption("BERTopic clusters ranked by pharmacovigilance signal score")
    st.divider()

    signals = load_signals()
    topics = load_topics()

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            signals,
            x="n_subjects",
            y="signal_score",
            size="n_records",
            color="top_soc",
            hover_data=["topic_id", "serious_count", "n_records"],
            title="Signal Score vs Number of Subjects",
            template="plotly_white",
            labels={"n_subjects": "Unique Subjects", "signal_score": "Signal Score", "top_soc": "Body System"}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score Components")
        st.markdown("""
        Signal score = weighted sum of:
        - **Subject coverage** (0.3) — breadth across patients
        - **Serious AE rate** (0.4) — clinical severity
        - **Treatment enrichment** (0.3) — treatment vs placebo ratio
        """)
        st.divider()
        st.metric("Total clusters", len(signals))
        st.metric("Clusters with serious AEs", int((signals["serious_count"] > 0).sum()))
        st.metric("Top signal score", f"{signals['signal_score'].max():.3f}")

    st.divider()
    st.subheader("All Signal Candidates")

    display_signals = signals[[
        "topic_id", "top_soc", "n_records", "n_subjects",
        "serious_count", "signal_score"
    ]].sort_values("signal_score", ascending=False)

    st.dataframe(display_signals.reset_index(drop=True), use_container_width=True, height=400)

    st.divider()
    st.subheader("Topic Word Explorer")
    topic_ids = [str(k) for k in topics.keys()]
    selected_topic = st.selectbox("Select topic", topic_ids,
                                   format_func=lambda x: f"Topic {x}")
    if selected_topic in topics:
        topic_data = topics[selected_topic]
        words = topic_data["words"]
        scores = topic_data["scores"]
        fig_words = px.bar(
            x=scores[:10], y=words[:10],
            orientation="h",
            title=f"Topic {selected_topic} — Top Terms",
            template="plotly_white",
            labels={"x": "Relevance Score", "y": "Term"}
        )
        fig_words.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
        st.plotly_chart(fig_words, use_container_width=True)

# ── Page 4: RAG Assessments ───────────────────────────────────────────────────
elif page == "RAG Assessments":
    st.title("RAG Signal Assessments")
    st.caption("Mistral LLM assessments grounded in MedDRA and drug knowledge via ChromaDB RAG")
    st.divider()

    assessments = load_assessments()

    col1, col2, col3 = st.columns(3)
    known = (assessments["signal_type"] == "Known_Signal").sum()
    potential = (assessments["signal_type"] == "Potential_New_Signal").sum()
    noise = (assessments["signal_type"] == "Background_Noise").sum()
    col1.metric("Known Signals", int(known))
    col2.metric("Potential New Signals", int(potential))
    col3.metric("Background Noise", int(noise))

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        fig_type = px.pie(
            assessments,
            names="signal_type",
            title="Signal Type Distribution",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with col_right:
        fig_action = px.pie(
            assessments,
            names="recommended_action",
            title="Recommended Actions",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_action, use_container_width=True)

    st.divider()
    st.subheader("Signal Assessment Cards")

    type_filter = st.selectbox(
        "Filter by signal type",
        ["All"] + assessments["signal_type"].unique().tolist()
    )

    filtered_a = assessments if type_filter == "All" else assessments[assessments["signal_type"] == type_filter]

    for _, row in filtered_a.iterrows():
        color = {
            "Known_Signal": "🟡",
            "Potential_New_Signal": "🔴",
            "Background_Noise": "🟢"
        }.get(row.get("signal_type", ""), "⚪")

        with st.expander(f"{color} {row.get('signal_label', 'Unknown')} — Score: {row['signal_score']:.3f}"):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Type:** {row.get('signal_type', 'N/A')}")
            col2.markdown(f"**Significance:** {row.get('clinical_significance', 'N/A')}")
            col3.markdown(f"**Action:** {row.get('recommended_action', 'N/A')}")

            st.markdown(f"**Body System:** {row['top_soc']}")
            st.markdown(f"**Records:** {row['n_records']} | **Subjects:** {row['n_subjects']} | **Serious:** {row['serious_count']}")
            st.markdown(f"**Biological plausibility:** {row.get('biological_plausibility', 'N/A')} | **Dose-response:** {row.get('dose_response', 'N/A')}")
            st.info(row.get("assessment", "No assessment available"))
