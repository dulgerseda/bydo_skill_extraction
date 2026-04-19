import json
import streamlit as st
from pathlib import Path
from pyvis.network import Network
import tempfile
import os
from collections import Counter
import pandas as pd
import graphviz
import plotly.express as px

st.set_page_config(page_title="BYDO · Skill Extraction Pipeline", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0f0f0f; color: #e0e0e0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #ffffff; }
.metric-box { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px 20px; text-align: center; }
.metric-label { font-size: 11px; color: #666; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 500; font-family: 'IBM Plex Mono', monospace; color: #fff; }
.section-header { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #555; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid #222; padding-bottom: 8px; margin-bottom: 16px; }
.pill { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.pill-K   { background: #1a2a1a; color: #4caf50; border: 1px solid #2d4a2d; }
.pill-OS  { background: #1a1a2a; color: #5c9fff; border: 1px solid #2d2d4a; }
.pill-SF  { background: #2a1a1a; color: #ff7043; border: 1px solid #4a2d2d; }
.pill-IPS { background: #2a2218; color: #ffb300; border: 1px solid #4a3a1a; }
.decision-accept   { color: #4caf50; }
.decision-review   { color: #ffb300; }
.decision-emerging { color: #ff5252; }
.view-label { font-family: IBM Plex Mono; font-size: 11px; color: #888; font-style: italic; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

OUTPUT_PATH = Path("results/output.json")

@st.cache_data
def load_output():
    if not OUTPUT_PATH.exists():
        return None
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_output()

st.markdown("## BYDO · Skill Extraction Pipeline")
st.markdown("<p style='color:#555;font-family:IBM Plex Mono;font-size:12px;'>Upload sources · Extract skills · Build a knowledge graph</p>", unsafe_allow_html=True)

if data is None:
    st.error("results/output.json not found. Run reranking_pipeline.py first.")
    st.stop()

def get_skills(job_data):
    if isinstance(job_data, list):
        return job_data
    return job_data.get("skills", [])

def get_domain(job_id, job_data):
    if isinstance(job_data, dict):
        return job_data.get("domain", job_id)
    return job_id

all_skills     = [s for jd in data.values() for s in get_skills(jd)]
total_docs     = len(data)
total_skills   = len(all_skills)
total_emerging = sum(1 for s in all_skills if s["gap"])
total_accept   = sum(1 for s in all_skills if s["decision"] == "accept")

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip([c1,c2,c3,c4],
    ["Documents","Skills","ESCO match","Emerging"],
    [total_docs, total_skills, total_accept, total_emerging]):
    col.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_graph, tab_registry, tab_log = st.tabs(["Graph", "Registry", "Extraction log"])

# ══ GRAPH TAB ══════════════════════════════════════════════════
with tab_graph:

    selected_job    = st.selectbox("Job description", list(data.keys()))
    job_data        = data[selected_job]
    selected_skills = get_skills(job_data)
    job_domain      = get_domain(selected_job, job_data)
    doc_label       = selected_job.replace("_", " ")

    # ── Main graph ──
    net = Network(height="540px", width="100%", bgcolor="#111111", font_color="#ffffff")
    net.force_atlas_2based(gravity=-60, central_gravity=0.02, spring_length=120, spring_strength=0.06, damping=0.9)

    net.add_node("doc", label=doc_label,
        color={"background": "#1D9E75", "border": "#085041"},
        size=20, shape="circle",
        font={"color": "#ffffff", "size": 11, "bold": True})

    added = {"doc"}
    type_colors = {
        "Tool":       "#E8814A",
        "Methodology":"#7F77DD",
        "Knowledge":  "#378ADD",
        "Soft Skill": "#D85A30",
    }

    for skill in selected_skills:
        noun       = skill["noun"]
        cat        = skill["category"]
        decision   = skill["decision"]
        conf       = round(skill.get("confidence", 0), 2)
        esco       = skill.get("esco_match")
        domain     = skill.get("domain", "Unknown")
        skill_type = skill.get("skill_type", "Knowledge")

        if domain not in added:
            net.add_node(domain, label=domain,
                color={"background": "#111111", "border": "#7EC8E3"},
                size=14, shape="circle",
                font={"color": "#7EC8E3", "size": 9})
            net.add_edge("doc", domain, color="#2a2a2a", width=1, dashes=True)
            added.add(domain)

        type_key = f"type_{skill_type}"
        if type_key not in added:
            tc = type_colors.get(skill_type, "#888780")
            net.add_node(type_key, label=skill_type,
                color={"background": "#111111", "border": tc},
                size=13, shape="circle",
                font={"color": tc, "size": 9})
            net.add_edge(domain, type_key, color="#2a2a2a", width=0.8, dashes=True)
            added.add(type_key)

        if noun not in added:
            net.add_node(noun, label=noun,
                color={"background": "#111111", "border": "#7F77DD"},
                size=10, shape="circle",
                title=f"Category: {cat}\nDecision: {decision}\nConf: {conf}\nESCO: {esco or 'no match'}",
                font={"color": "#cccccc", "size": 9})
            net.add_edge(type_key, noun, color="#333333", width=0.6, dashes=True)
            added.add(noun)

        if cat == "IPS" and skill.get("tool") and skill.get("action"):
            tool   = skill["tool"]
            action = skill["action"]
            for n in [tool, action]:
                if n not in added:
                    net.add_node(n, label=n,
                        color={"background": "#111111", "border": "#7F77DD"},
                        size=10, shape="circle",
                        font={"color": "#cccccc", "size": 9})
                    net.add_edge(type_key, n, color="#333333", width=0.6, dashes=True)
                    added.add(n)
            net.add_edge(tool, action,
                label="used_for", color="#BA7517", width=1.5, dashes=True,
                font={"color": "#BA7517", "size": 8})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
    os.unlink(tmp.name)

    st.components.v1.html(html, height=560, scrolling=False)

    st.markdown("""
    <div style='display:flex;gap:20px;align-items:center;margin:8px 0 16px;flex-wrap:wrap;font-size:12px;color:#666;'>
      <span style='display:flex;align-items:center;gap:5px;'><span style='width:12px;height:12px;border-radius:50%;border:2px solid #7F77DD;background:#111;display:inline-block'></span>Skill</span>
      <span style='display:flex;align-items:center;gap:5px;'><span style='width:12px;height:12px;border-radius:50%;background:#1D9E75;display:inline-block'></span>Document</span>
      <span style='display:flex;align-items:center;gap:5px;'><span style='width:12px;height:12px;border-radius:50%;border:2px solid #7EC8E3;background:#111;display:inline-block'></span>Domain</span>
      <span style='display:flex;align-items:center;gap:5px;'><span style='width:12px;height:12px;border-radius:50%;border:2px solid #E8814A;background:#111;display:inline-block'></span>Skill type</span>
      <span style='display:flex;align-items:center;gap:5px;'><span style='border-bottom:2px dashed #BA7517;width:24px;display:inline-block'></span>IPS edge</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 EXPANDERS ──
    st.markdown("<br>", unsafe_allow_html=True)

    domains = {}
    for s in selected_skills:
        d = s.get("domain", "Unknown")
        domains.setdefault(d, []).append(s["noun"])

    main_domain = max(domains, key=lambda d: len(domains[d]))

    with st.expander(f"What should I teach in {main_domain}? · View 1 · Domain"):
        st.markdown("<p class='view-label'>Most requested skills across all job ads</p>", unsafe_allow_html=True)

        ds_skill_counts = Counter()
        for jd in data.values():
            for s in get_skills(jd):
                if s.get("domain") == main_domain:
                    ds_skill_counts[s["noun"]] += 1

        top_skills = ds_skill_counts.most_common(5)

        g = graphviz.Digraph()
        g.attr(rankdir="TB", bgcolor="transparent")
        g.node(main_domain,
            style="filled,rounded", fillcolor="#ddd6fe",
            color="#7c3aed", fontcolor="#3b0764",
            fontname="IBM Plex Mono", shape="box")

        for skill, count in top_skills:
            g.node(skill,
                label=f"{skill}\n({count} ads)",
                style="filled,rounded", fillcolor="#bbf7d0",
                color="#15803d", fontcolor="#14532d",
                fontname="IBM Plex Mono", shape="box")
            g.edge(main_domain, skill, color="#a78bfa")

        st.graphviz_chart(g, use_container_width=True)

    with st.expander("What should I learn at most?  · View 2 · Type"):
        st.markdown("<p class='view-label'>Which tools does the market demand most? — job-ad frequency</p>", unsafe_allow_html=True)

        tool_counts = Counter()
        for jd in data.values():
            for s in get_skills(jd):
                if s.get("skill_type") == "Tool":
                    tool_counts[s["noun"]] += 1

        top_tools = tool_counts.most_common(5)
        df_tools = pd.DataFrame(top_tools, columns=["Skill", "Job ads"])
        df_tools = df_tools.sort_values("Job ads", ascending=True)

        fig = px.bar(df_tools, x="Job ads", y="Skill", orientation="h",
        color="Job ads", color_continuous_scale=["#f4c4b0", "#D85A30"])
        
        fig.update_traces(
            marker_line_width=0,
            marker_cornerradius=6,
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cccccc",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=220,
            yaxis_title=None,
            xaxis_title=None,
        )

        st.plotly_chart(fig, use_container_width=True)


    with st.expander("Where does a skill appear?  · View 3 · Concept"):
        st.markdown("<p class='view-label'>Select a skill to see its profile</p>", unsafe_allow_html=True)
        concept = st.selectbox("Skill", [s["noun"] for s in selected_skills], key="concept_sel")
        skill_obj = next((s for s in selected_skills if s["noun"] == concept), None)
        if skill_obj:
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Category",   skill_obj["category"])
            cb.metric("Skill type", skill_obj.get("skill_type","—"))
            cc.metric("Domain",     skill_obj.get("domain","—"))
            cd.metric("Decision",   skill_obj["decision"])
            source = skill_obj.get("source","")
            co = [s["noun"] for s in selected_skills if s.get("source")==source and s["noun"]!=concept]
            if co:
                st.markdown("<span style='color:#555;font-size:12px'>Co-occurs with:</span>  " + "  ".join([f"`{n}`" for n in co]), unsafe_allow_html=True)
            esco = skill_obj.get("esco_match")
            conf = round(skill_obj.get("confidence",0), 2)
            if not skill_obj["gap"] and esco:
                st.markdown(f"**ESCO:** `{esco}` — confidence: `{conf}`")
            else:
                st.markdown(f"**ESCO:** no match → **emerging skill** (conf: `{conf}`)")

    # ── SKILL TABLE ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Skill table — click column headers to sort</div>", unsafe_allow_html=True)

    df_table = pd.DataFrame([
        {
            "Skill":      s["noun"],
            "Category":   s["category"],
            "Skill type": s.get("skill_type", "—"),
            "Domain":     s.get("domain", "—"),
            "Decision":   s["decision"],
            "Confidence": round(s.get("confidence", 0), 2),
            "ESCO match": s.get("esco_match") or "—",
        }
        for s in selected_skills
    ])

    st.dataframe(
        df_table,
        use_container_width=True,
        height=420,
        column_config={
            "Skill":      st.column_config.TextColumn("Skill", width="large"),
            "Category":   st.column_config.TextColumn("Category", width="small"),
            "Skill type": st.column_config.TextColumn("Skill type", width="medium"),
            "Domain":     st.column_config.TextColumn("Domain", width="medium"),
            "Decision":   st.column_config.TextColumn("Decision", width="small"),
            "Confidence": st.column_config.NumberColumn("Confidence", width="small", format="%.2f"),
            "ESCO match": st.column_config.TextColumn("ESCO match", width="large"),
        }
    )

# ══ REGISTRY TAB ═══════════════════════════════════════════════
with tab_registry:
    job_filter = st.selectbox("Job description", ["All"] + list(data.keys()), key="reg_job")
    col1, col2 = st.columns(2)
    cat_filter = col1.multiselect("Category", ["K","OS","SF","IPS"], default=["K","OS","SF","IPS"])
    dec_filter = col2.multiselect("Decision", ["accept","review","emerging"], default=["accept","review","emerging"])

    if job_filter == "All":
        filtered = all_skills
    else:
        filtered = get_skills(data[job_filter])
    filtered = [s for s in filtered if s["category"] in cat_filter and s["decision"] in dec_filter]

    st.markdown(f"<div class='section-header'>{len(filtered)} skills</div>", unsafe_allow_html=True)

    for skill in filtered:
        cat      = skill["category"]
        decision = skill["decision"]
        conf     = skill.get("confidence", 0)
        esco     = skill.get("esco_match") or "—"
        with st.expander(f"{skill['noun']}"):
            ca, cb, cc = st.columns(3)
            ca.markdown(f'<span class="pill pill-{cat}">{cat}</span>', unsafe_allow_html=True)
            cb.markdown(f'<span class="decision-{decision}">{decision}</span>', unsafe_allow_html=True)
            cc.markdown(f'`conf: {round(conf,2)}`')
            st.markdown(f"**Verb:** {skill.get('verb','—')}")
            st.markdown(f"**ESCO form:** {skill.get('esco_form','—')}")
            st.markdown(f"**ESCO match:** {esco}")
            st.markdown(f"**Domain:** {skill.get('domain','—')}  |  **Skill type:** {skill.get('skill_type','—')}")
            if skill.get("tool"):
                st.markdown(f"**Tool:** {skill['tool']}  →  **Action:** {skill['action']}")
            st.caption(skill.get("source",""))

# ══ EXTRACTION LOG TAB ══════════════════════════════════════════
with tab_log:
    cat_clr = {"K":"#4caf50","OS":"#5c9fff","SF":"#ff7043","IPS":"#ffb300"}
    dec_clr = {"accept":"#4caf50","review":"#ffb300","emerging":"#ff5252"}

    for job_id, job_data in data.items():
        skills     = get_skills(job_data)
        job_domain = get_domain(job_id, job_data)
        st.markdown(f"<div class='section-header'>{job_id} · {job_domain}</div>", unsafe_allow_html=True)
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Total",    len(skills))
        cb.metric("Accept",   sum(1 for s in skills if s["decision"]=="accept"))
        cc.metric("Review",   sum(1 for s in skills if s["decision"]=="review"))
        cd.metric("Emerging", sum(1 for s in skills if s["decision"]=="emerging"))

        for skill in skills:
            cat = skill["category"]
            dec = skill["decision"]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid #1a1a1a;">'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;color:{cat_clr.get(cat,"#aaa")};width:32px">{cat}</span>'
                f'<span style="font-size:13px;flex:1;color:#ddd">{skill["noun"]}</span>'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;color:{dec_clr.get(dec,"#aaa")}">{dec}</span>'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#444">{round(skill.get("confidence",0),2)}</span>'
                f'</div>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)