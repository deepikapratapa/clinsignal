import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="ClinSignal", page_icon="\u2695", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --navy:#0a0e1a; --navy-mid:#111827; --navy-light:#1a2235;
    --teal:#00d4aa; --teal-dim:#00d4aa22; --amber:#f59e0b;
    --coral:#f87171; --slate:#8892a4; --slate-light:#b0bac9;
    --white:#f0f4ff; --border:#1e2d47; --card-bg:#111827;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--navy)!important;color:var(--white)!important;}
section[data-testid="stSidebar"]{background:var(--navy-mid)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--slate-light)!important;}
.main .block-container{padding:2rem 2.5rem!important;max-width:1400px!important;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;}
[data-testid="metric-container"]{background:var(--navy-light)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1.2rem 1.4rem!important;}
[data-testid="metric-container"] label{color:var(--slate)!important;font-size:11px!important;letter-spacing:.08em!important;text-transform:uppercase!important;font-family:'DM Mono',monospace!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--teal)!important;font-family:'DM Mono',monospace!important;font-size:2rem!important;font-weight:500!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important;}
.stSelectbox>div>div{background:var(--navy-light)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stMultiSelect>div>div{background:var(--navy-light)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.streamlit-expanderHeader{background:var(--navy-light)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--white)!important;}
.streamlit-expanderContent{background:var(--navy-mid)!important;border:1px solid var(--border)!important;border-top:none!important;border-radius:0 0 8px 8px!important;}
.stAlert{background:var(--teal-dim)!important;border:1px solid var(--teal)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="DM Sans", color="#8892a4", size=12),
    margin=dict(l=0, r=0, t=30, b=0)
)
TEAL_SCALE = [[0,"#0a0e1a"],[0.5,"#00b899"],[1,"#00d4aa"]]
SIGNAL_COLORS = ["#00d4aa","#f59e0b","#f87171","#818cf8","#34d399","#60a5fa","#fb923c","#a78bfa","#4ade80","#f472b6"]

@st.cache_data
def load_ae(): return pd.read_csv("results/signals/ae_with_signals.csv")
@st.cache_data
def load_signals(): return pd.read_csv("results/signals/signal_candidates.csv")
@st.cache_data
def load_assessments():
    df = pd.read_csv("results/validation/signal_assessments.csv")
    df["recommended_action"] = df["recommended_action"].apply(lambda x: str(x).split(".")[0].strip()[:40])
    return df
@st.cache_data
def load_topics():
    with open("results/signals/topics.json") as f: return json.load(f)

with st.sidebar:
    st.markdown("""<div style='padding:.5rem 0 1.5rem'><div style='font-family:Space Grotesk;font-size:22px;font-weight:700;color:#00d4aa;letter-spacing:-.5px'>&#9877; ClinSignal</div><div style='font-size:11px;color:#8892a4;margin-top:4px;letter-spacing:.04em;line-height:1.5'>Pharmacovigilance signal detection<br>from SDTM clinical narratives</div></div>""", unsafe_allow_html=True)
    page = st.radio("", ["Overview","AE Explorer","Signal Candidates","RAG Assessments"], label_visibility="collapsed")
    st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:10px;color:#4a5568;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;font-family:DM Mono'>Dataset</div><div style='font-size:12px;color:#8892a4;line-height:1.7'>CDISC Pilot Study<br>Xanomeline vs Placebo<br>Alzheimer Disease<br><span style='color:#00d4aa'>1191 AE records</span> · 225 subjects</div>""", unsafe_allow_html=True)
    st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:10px;color:#4a5568;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;font-family:DM Mono'>Stack</div><div style='font-size:11px;color:#8892a4;line-height:2'>scispaCy · BERTopic · ChromaDB<br>Mistral · PostgreSQL · Docker<br>AWS EC2 · ECR</div>""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown("""<div style='margin-bottom:2rem'><div style='font-family:Space Grotesk;font-size:36px;font-weight:700;color:#f0f4ff;letter-spacing:-1px;line-height:1.1'>Adverse Event<br><span style='color:#00d4aa'>Signal Detection</span></div><div style='font-size:14px;color:#8892a4;margin-top:10px;max-width:560px;line-height:1.7'>RAG-powered pharmacovigilance pipeline extracting latent safety signals from unstructured SDTM clinical trial narratives using NLP clustering and LLM-grounded assessment.</div></div>""", unsafe_allow_html=True)
    ae = load_ae(); signals = load_signals(); assessments = load_assessments()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AE Records", f"{len(ae):,}")
    c2.metric("Unique Subjects", f"{ae['USUBJID'].nunique():,}")
    c3.metric("Signal Clusters", f"{len(signals)}")
    c4.metric("Assessed Signals", f"{len(assessments)}")
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div style='font-family:Space Grotesk;font-size:16px;font-weight:600;color:#f0f4ff;margin-bottom:1rem'>Adverse Events by Body System</div>", unsafe_allow_html=True)
        soc = ae["AEBODSYS"].value_counts().reset_index(); soc.columns = ["System","Count"]
        fig = go.Figure(go.Bar(x=soc["Count"].head(10), y=soc["System"].head(10), orientation="h",
            marker=dict(color=soc["Count"].head(10), colorscale=TEAL_SCALE, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>%{x} events<extra></extra>"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, yaxis=dict(categoryorder="total ascending", gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("<div style='font-family:Space Grotesk;font-size:16px;font-weight:600;color:#f0f4ff;margin-bottom:1rem'>Treatment Arm Distribution</div>", unsafe_allow_html=True)
        if "ARM" in ae.columns:
            top_socs = ae["AEBODSYS"].value_counts().head(5).index.tolist()
            arm_soc = ae[ae["AEBODSYS"].isin(top_socs)].groupby(["ARM","AEBODSYS"]).size().reset_index(name="Count")
            fig2 = px.bar(arm_soc, x="AEBODSYS", y="Count", color="ARM", barmode="group",
                color_discrete_sequence=["#475569","#00d4aa","#f59e0b"], template="none")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=dict(tickangle=-25, tickfont=dict(size=10), gridcolor="#1e2d47", linecolor="#1e2d47"))
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Grotesk;font-size:16px;font-weight:600;color:#f0f4ff;margin-bottom:1rem'>Pipeline Architecture</div>", unsafe_allow_html=True)
    steps = [("01","#00d4aa","SDTM Ingestion","pandas · SAS XPT · AE DM CM"),("02","#00b899","NLP Extraction","scispaCy · sentence-transformers"),("03","#f59e0b","Signal Clustering","BERTopic · UMAP · HDBSCAN"),("04","#818cf8","RAG Grounding","ChromaDB · MedDRA · DrugBank"),("05","#f87171","LLM Assessment","Ollama · Mistral · local inference")]
    cols = st.columns(5)
    for col,(num,color,title,sub) in zip(cols,steps):
        col.markdown(f"""<div style='background:#111827;border:1px solid #1e2d47;border-top:2px solid {color};border-radius:10px;padding:14px 16px'><div style='font-family:DM Mono;font-size:10px;color:{color};letter-spacing:.08em;margin-bottom:6px'>{num}</div><div style='font-size:13px;font-weight:600;color:#f0f4ff;margin-bottom:4px'>{title}</div><div style='font-size:11px;color:#8892a4;line-height:1.5'>{sub}</div></div>""", unsafe_allow_html=True)

elif page == "AE Explorer":
    st.markdown("""<div style='margin-bottom:1.5rem'><div style='font-family:Space Grotesk;font-size:28px;font-weight:700;color:#f0f4ff;letter-spacing:-.5px'>Adverse Event Explorer</div><div style='font-size:13px;color:#8892a4;margin-top:6px'>Browse and filter 1,191 AE records from the CDISC pilot SDTM dataset</div></div>""", unsafe_allow_html=True)
    ae = load_ae()
    c1,c2,c3 = st.columns(3)
    with c1: soc_filter = st.multiselect("Body System", sorted(ae["AEBODSYS"].dropna().unique()))
    with c2: arm_filter = st.multiselect("Treatment Arm", sorted(ae["ARM"].dropna().unique()) if "ARM" in ae.columns else [])
    with c3: ser_filter = st.selectbox("Severity", ["All","Serious only","Non-serious only"])
    filtered = ae.copy()
    if soc_filter: filtered = filtered[filtered["AEBODSYS"].isin(soc_filter)]
    if arm_filter: filtered = filtered[filtered["ARM"].isin(arm_filter)]
    if ser_filter == "Serious only": filtered = filtered[filtered["IS_SERIOUS"]==True]
    elif ser_filter == "Non-serious only": filtered = filtered[filtered["IS_SERIOUS"]==False]
    st.markdown(f"""<div style='font-family:DM Mono;font-size:12px;color:#8892a4;margin:1rem 0 .5rem;padding:8px 14px;background:#111827;border-radius:6px;border-left:3px solid #00d4aa;display:inline-block'>{len(filtered):,} records matched</div>""", unsafe_allow_html=True)
    display_cols = [c for c in ["USUBJID","AETERM","AEDECOD","AEBODSYS","AESEV","AESER","AEREL","AEOUT","ARM"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=380)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Grotesk;font-size:16px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Narrative Viewer</div>", unsafe_allow_html=True)
    if len(filtered) > 0:
        sel = st.selectbox("Select record", range(min(50,len(filtered))), format_func=lambda i: f"{filtered.iloc[i]['USUBJID']}  —  {filtered.iloc[i]['AEDECOD']}")
        row = filtered.iloc[sel]
        st.markdown(f"""<div style='background:#0d1a2e;border:1px solid #1e2d47;border-left:3px solid #00d4aa;border-radius:8px;padding:16px 20px;font-size:14px;color:#b0bac9;line-height:1.8'>{row.get("NARRATIVE","No narrative available")}</div>""", unsafe_allow_html=True)

elif page == "Signal Candidates":
    st.markdown("""<div style='margin-bottom:1.5rem'><div style='font-family:Space Grotesk;font-size:28px;font-weight:700;color:#f0f4ff;letter-spacing:-.5px'>Signal Candidates</div><div style='font-size:13px;color:#8892a4;margin-top:6px'>51 BERTopic clusters ranked by composite pharmacovigilance signal score</div></div>""", unsafe_allow_html=True)
    signals = load_signals(); topics = load_topics()
    m1,m2,m3 = st.columns(3)
    m1.metric("Total Clusters", len(signals))
    m2.metric("Clusters with Serious AEs", int((signals["serious_count"]>0).sum()))
    m3.metric("Top Signal Score", f"{signals['signal_score'].max():.3f}")
    st.markdown("<br>", unsafe_allow_html=True)
    col_l,col_r = st.columns([3,2], gap="large")
    with col_l:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Signal Score vs Subject Coverage</div>", unsafe_allow_html=True)
        fig = px.scatter(signals, x="n_subjects", y="signal_score", size="n_records", color="top_soc",
            hover_data=["topic_id","serious_count","n_records"], color_discrete_sequence=SIGNAL_COLORS,
            size_max=40, template="none", labels={"n_subjects":"Unique Subjects","signal_score":"Signal Score","top_soc":"Body System"})
        fig.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(font=dict(size=10)))
        fig.update_traces(marker=dict(line=dict(width=0.5, color="#0a0e1a")))
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Score Methodology</div>", unsafe_allow_html=True)
        st.markdown("""<div style='background:#111827;border:1px solid #1e2d47;border-radius:10px;padding:16px'>
            <div style='margin-bottom:12px'><div style='display:flex;justify-content:space-between;margin-bottom:4px'><span style='font-size:12px;color:#b0bac9'>Subject coverage</span><span style='font-family:DM Mono;font-size:11px;color:#00d4aa'>30%</span></div><div style='background:#1e2d47;border-radius:4px;height:4px'><div style='background:#00d4aa;width:30%;height:4px;border-radius:4px'></div></div></div>
            <div style='margin-bottom:12px'><div style='display:flex;justify-content:space-between;margin-bottom:4px'><span style='font-size:12px;color:#b0bac9'>Serious AE rate</span><span style='font-family:DM Mono;font-size:11px;color:#f59e0b'>40%</span></div><div style='background:#1e2d47;border-radius:4px;height:4px'><div style='background:#f59e0b;width:40%;height:4px;border-radius:4px'></div></div></div>
            <div><div style='display:flex;justify-content:space-between;margin-bottom:4px'><span style='font-size:12px;color:#b0bac9'>Treatment enrichment</span><span style='font-family:DM Mono;font-size:11px;color:#818cf8'>30%</span></div><div style='background:#1e2d47;border-radius:4px;height:4px'><div style='background:#818cf8;width:30%;height:4px;border-radius:4px'></div></div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_tbl,col_words = st.columns([3,2], gap="large")
    with col_tbl:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>All Signal Candidates</div>", unsafe_allow_html=True)
        disp = signals[["topic_id","top_soc","n_records","n_subjects","serious_count","signal_score"]].sort_values("signal_score",ascending=False)
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=350)
    with col_words:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Topic Word Explorer</div>", unsafe_allow_html=True)
        sel_topic = st.selectbox("Topic", list(topics.keys()), format_func=lambda x: f"Topic {x}")
        if sel_topic in topics:
            td = topics[sel_topic]
            fig_w = go.Figure(go.Bar(x=td["scores"][:10], y=td["words"][:10], orientation="h",
                marker=dict(color=td["scores"][:10], colorscale=TEAL_SCALE, line=dict(width=0))))
            fig_w.update_layout(**PLOTLY_LAYOUT, height=300, yaxis=dict(categoryorder="total ascending", gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=11)))
            st.plotly_chart(fig_w, use_container_width=True)

elif page == "RAG Assessments":
    st.markdown("""<div style='margin-bottom:1.5rem'><div style='font-family:Space Grotesk;font-size:28px;font-weight:700;color:#f0f4ff;letter-spacing:-.5px'>RAG Signal Assessments</div><div style='font-size:13px;color:#8892a4;margin-top:6px'>Mistral LLM assessments grounded in MedDRA ontology and drug knowledge via ChromaDB</div></div>""", unsafe_allow_html=True)
    assessments = load_assessments()
    known = int((assessments["signal_type"]=="Known_Signal").sum())
    potential = int((assessments["signal_type"]=="Potential_New_Signal").sum())
    noise = int((assessments["signal_type"]=="Background_Noise").sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("Known Signals", known); c2.metric("Potential New Signals", potential); c3.metric("Background Noise", noise)
    st.markdown("<br>", unsafe_allow_html=True)
    col_l,col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Signal Classification</div>", unsafe_allow_html=True)
        fig_t = px.pie(assessments, names="signal_type", hole=0.55,
            color_discrete_map={"Known_Signal":"#f59e0b","Potential_New_Signal":"#00d4aa","Background_Noise":"#475569"})
        fig_t.update_layout(**PLOTLY_LAYOUT, height=280,
            legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1e2d47", borderwidth=1))
        fig_t.update_traces(textfont_size=12, marker=dict(
            colors=["#f59e0b","#00d4aa","#475569"],
            line=dict(color="#0a0e1a", width=2)))
        st.plotly_chart(fig_t, use_container_width=True)
    with col_r:
        st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Recommended Actions</div>", unsafe_allow_html=True)
        ac = assessments["recommended_action"].value_counts().reset_index(); ac.columns = ["Action","Count"]
        fig_a = px.bar(ac, x="Count", y="Action", orientation="h", color="Count", color_continuous_scale=TEAL_SCALE, template="none")
        fig_a.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False, coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending", gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=11)))
        fig_a.update_traces(marker_line_width=0)
        st.plotly_chart(fig_a, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Grotesk;font-size:15px;font-weight:600;color:#f0f4ff;margin-bottom:.75rem'>Signal Assessment Cards</div>", unsafe_allow_html=True)
    type_filter = st.selectbox("Filter", ["All"]+assessments["signal_type"].dropna().unique().tolist())
    filtered_a = assessments if type_filter=="All" else assessments[assessments["signal_type"]==type_filter]
    TYPE_COLORS = {"Known_Signal":"#f59e0b","Potential_New_Signal":"#00d4aa","Background_Noise":"#475569"}
    TYPE_BG = {"Known_Signal":"#f59e0b15","Potential_New_Signal":"#00d4aa15","Background_Noise":"#47556915"}
    for _,row in filtered_a.iterrows():
        sig_type = row.get("signal_type","Unknown")
        color = TYPE_COLORS.get(sig_type,"#475569"); bg = TYPE_BG.get(sig_type,"#47556915")
        with st.expander(f"{row.get('signal_label','Unknown')}  —  Score: {row['signal_score']:.3f}"):
            st.markdown(f"""<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px'>
                <div style='background:{bg};border:1px solid {color}33;border-radius:8px;padding:10px 14px'><div style='font-size:10px;color:{color};letter-spacing:.06em;text-transform:uppercase;font-family:DM Mono;margin-bottom:4px'>Type</div><div style='font-size:13px;color:#f0f4ff;font-weight:500'>{sig_type.replace("_"," ")}</div></div>
                <div style='background:#111827;border:1px solid #1e2d47;border-radius:8px;padding:10px 14px'><div style='font-size:10px;color:#8892a4;letter-spacing:.06em;text-transform:uppercase;font-family:DM Mono;margin-bottom:4px'>Significance</div><div style='font-size:13px;color:#f0f4ff;font-weight:500'>{row.get("clinical_significance","N/A")}</div></div>
                <div style='background:#111827;border:1px solid #1e2d47;border-radius:8px;padding:10px 14px'><div style='font-size:10px;color:#8892a4;letter-spacing:.06em;text-transform:uppercase;font-family:DM Mono;margin-bottom:4px'>Action</div><div style='font-size:13px;color:#f0f4ff;font-weight:500'>{row.get("recommended_action","N/A")}</div></div>
            </div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px'>
                <div style='font-size:11px;color:#8892a4'><span style='color:#4a5568;font-family:DM Mono'>SOC</span><br>{str(row["top_soc"])[:30]}</div>
                <div style='font-size:11px;color:#8892a4'><span style='color:#4a5568;font-family:DM Mono'>Records</span><br><span style='font-family:DM Mono;color:#00d4aa'>{row["n_records"]}</span></div>
                <div style='font-size:11px;color:#8892a4'><span style='color:#4a5568;font-family:DM Mono'>Subjects</span><br><span style='font-family:DM Mono;color:#00d4aa'>{row["n_subjects"]}</span></div>
                <div style='font-size:11px;color:#8892a4'><span style='color:#4a5568;font-family:DM Mono'>Serious</span><br><span style='font-family:DM Mono;color:{"#f87171" if row["serious_count"]>0 else "#475569"}'>{row["serious_count"]}</span></div>
            </div>
            <div style='background:#0d1a2e;border-left:3px solid {color};border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;color:#b0bac9;line-height:1.7'>{row.get("assessment","No assessment available")}</div>""", unsafe_allow_html=True)
