import io
import time
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable
)

from utils.preprocess import clean_text
from utils.helper import category_map

# ================= LOAD =================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AutoDomain AI", layout="wide")

st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: 0.3s;
}
.card:hover { transform: scale(1.02); }
.card h3 { color: #60a5fa; }
.card p  { color: #cbd5f5; }
.center   { text-align: center; }
.big-title { font-size: 42px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= NAVIGATION =================
menu = st.sidebar.radio(
    "🧭 Navigation",
    ["🏠 Home", "📄 Predict", "🔍 Compare", "⏳ Time Machine", "📊 Dashboard", "📜 History"]
)

# ================= HELPERS =================

def is_valid_input(text):
    return len(text.split()) >= 8 and len(text.strip()) >= 30


def explain_prediction(text, vectorizer, model, top_n=10):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_
    pred_class = model.predict(vec)[0]
    class_index = list(model.classes_).index(pred_class)
    weights = coefs[class_index]
    indices = vec.toarray()[0].nonzero()[0]
    word_scores = [(feature_names[i], weights[i]) for i in indices]
    return sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:top_n]


def generate_pdf_report(user_input, results, keywords):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#0C447C"), spaceAfter=6
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#185FA5"),
        spaceBefore=14, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=colors.HexColor("#2C2C2A")
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#5F5E5A")
    )

    story = []

    # Header
    story.append(Paragraph("AutoDomain AI", title_style))
    story.append(Paragraph("Research Paper Classification Report", styles["Heading3"]))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#B5D4F4"), spaceAfter=10))

    # Input preview
    story.append(Paragraph("Input text", section_style))
    preview = user_input[:400].replace("\n", " ")
    if len(user_input) > 400:
        preview += "..."
    story.append(Paragraph(preview, body_style))
    story.append(Spacer(1, 8))

    # Predictions table
    story.append(Paragraph("Predicted domains", section_style))
    table_data = [["Rank", "Domain", "Confidence"]]
    medals = ["1st", "2nd", "3rd"]
    for i, (name, score) in enumerate(results):
        table_data.append([medals[i] if i < 3 else str(i+1), name, f"{score:.1f}%"])

    pred_table = Table(table_data, colWidths=[2*cm, 10*cm, 4*cm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#E6F1FB")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.HexColor("#0C447C")),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
        ("FONTSIZE",      (0, 1), (-1, -1), 10),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#2C2C2A")),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",         (2, 0), (2, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#D3D1C7")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 12))

    # Confidence bar chart
    story.append(Paragraph("Confidence chart", section_style))
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    domains = [r[0] for r in results]
    scores  = [r[1] for r in results]
    bar_colors = ["#378ADD", "#85B7EB", "#B5D4F4"][:len(results)]
    bars = ax.barh(domains, scores, color=bar_colors, height=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    chart_buf = io.BytesIO()
    plt.savefig(chart_buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    chart_buf.seek(0)
    story.append(Image(chart_buf, width=15*cm, height=5.5*cm))
    story.append(Spacer(1, 12))

    # Keywords table
    story.append(Paragraph("Top influential keywords", section_style))
    story.append(Paragraph(
        "These words pushed the model toward the predicted domain.", small_style
    ))
    story.append(Spacer(1, 6))
    kw_data = [["Keyword", "Weight"]]
    for word, weight in keywords[:10]:
        kw_data.append([word, f"{weight:.4f}"])
    kw_table = Table(kw_data, colWidths=[8*cm, 4*cm])
    kw_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#EAF3DE")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.HexColor("#27500A")),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#2C2C2A")),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#D3D1C7")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(kw_table)
    story.append(Spacer(1, 16))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#D3D1C7")))
    story.append(Spacer(1, 4))
    from datetime import datetime
    story.append(Paragraph(
        f"Generated by AutoDomain AI  •  {datetime.now().strftime('%d %B %Y, %H:%M')}",
        small_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# =================================================================
# 🏠 HOME
# =================================================================
if menu == "🏠 Home":

    st.markdown('<div class="center big-title">🤖 AutoDomain AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="center">Smart Research Paper Domain Classifier</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="card"><h3>📌 About</h3>
        <p>Classifies research papers into domains using content analysis of title and abstract.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card"><h3>🧭 How to Use</h3>
        <p>Go to Predict → Enter text or upload PDF → Click Predict → View results.</p>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class="card"><h3>📊 Features</h3>
        <p>Top predictions, charts, explanation, comparison, time machine, PDF export & history.</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="card"><h3>⚠️ Guidelines</h3>
        <p>Use meaningful input (8+ words). Better results with full abstract.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Example Input")
    st.code("""Deep learning based image classification using convolutional neural networks
for real-time object detection.""")
    st.success("🚀 Ready to classify your research paper!")


# =================================================================
# 📄 PREDICT
# =================================================================
elif menu == "📄 Predict":

    st.header("📄 Predict Domain")

    option     = st.radio("Choose Input Method", ["✍️ Text", "📄 Upload PDF"])
    user_input = ""

    if option == "✍️ Text":
        title    = st.text_input("Enter Title")
        abstract = st.text_area("Enter Abstract")
        user_input = (title + " " + abstract).strip()

    elif option == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            user_input = text[:2000]

    if st.button("🎯 Predict"):

        if not user_input.strip():
            st.warning("⚠️ Enter input")
            st.stop()

        if not is_valid_input(user_input):
            st.error("❌ Input too short — need at least 8 words and 30 characters.")
            st.stop()

        with st.spinner("🔍 Analyzing..."):
            time.sleep(1)

        vec   = vectorizer.transform([clean_text(user_input)])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🎯 Predictions")
        results = []
        for i in top3_idx:
            name  = category_map.get(classes[i], classes[i])
            score = probs[i] * 100
            results.append((name, score))
            st.write(f"👉 {name} — {score:.2f}%")
            st.progress(int(score))

        # Save to history
        st.session_state.history.append({"input": user_input, "result": results})

        # Pie chart
        df_pie = pd.DataFrame({"Domain": [r[0] for r in results],
                               "Score":  [r[1] for r in results]})
        st.plotly_chart(px.pie(df_pie, names="Domain", values="Score"))

        # Explanation
        st.subheader("🧠 Explanation")
        keywords = explain_prediction(user_input, vectorizer, model)
        for word, score in keywords:
            st.write(f"🔹 {word} ({score:.3f})")

        # PDF download
        st.markdown("---")
        st.subheader("📥 Download Report")
        pdf_buffer = generate_pdf_report(user_input, results, keywords)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name="autodomain_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =================================================================
# 🔍 COMPARE
# =================================================================
elif menu == "🔍 Compare":

    st.header("🔍 Paper Comparison")
    st.write("Compare two research papers and see their domain overlap.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Paper A")
        title_a    = st.text_input("Title A", key="title_a")
        abstract_a = st.text_area("Abstract A", height=200, key="abstract_a")
    with col2:
        st.subheader("📄 Paper B")
        title_b    = st.text_input("Title B", key="title_b")
        abstract_b = st.text_area("Abstract B", height=200, key="abstract_b")

    if st.button("⚡ Compare Papers"):

        text_a = (title_a + " " + abstract_a).strip()
        text_b = (title_b + " " + abstract_b).strip()

        if not text_a or not text_b:
            st.warning("⚠️ Please fill in both papers.")
            st.stop()

        if not is_valid_input(text_a) or not is_valid_input(text_b):
            st.error("❌ Each paper needs at least 8 words.")
            st.stop()

        vec_a = vectorizer.transform([clean_text(text_a)])
        vec_b = vectorizer.transform([clean_text(text_b)])

        probs_a = model.predict_proba(vec_a)[0]
        probs_b = model.predict_proba(vec_b)[0]
        classes = model.classes_

        top5_a = {category_map.get(classes[i], classes[i]): round(probs_a[i] * 100, 1)
                  for i in probs_a.argsort()[-5:][::-1]}
        top5_b = {category_map.get(classes[i], classes[i]): round(probs_b[i] * 100, 1)
                  for i in probs_b.argsort()[-5:][::-1]}

        shared = set(top5_a.keys()) & set(top5_b.keys())
        only_a = set(top5_a.keys()) - set(top5_b.keys())
        only_b = set(top5_b.keys()) - set(top5_a.keys())

        similarity = cosine_similarity(vec_a, vec_b)[0][0]
        sim_pct    = round(similarity * 100, 1)

        st.markdown("---")
        st.subheader("📊 Comparison Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("🔗 Similarity Score", f"{sim_pct}%")
        c2.metric("🤝 Shared Domains",   len(shared))
        c3.metric("🔀 Unique Domains",   len(only_a) + len(only_b))
        st.progress(int(sim_pct))

        st.subheader("🤝 Shared Domains")
        if shared:
            for domain in shared:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.write(f"✅ {domain} — {top5_a[domain]}%")
                    st.progress(int(top5_a[domain]))
                with dc2:
                    st.write(f"✅ {domain} — {top5_b[domain]}%")
                    st.progress(int(top5_b[domain]))
        else:
            st.info("No shared top domains found.")

        st.subheader("🔀 Unique to Each Paper")
        u1, u2 = st.columns(2)
        with u1:
            st.markdown("**Paper A only**")
            for d in only_a:
                st.write(f"🔹 {d} — {top5_a[d]}%")
        with u2:
            st.markdown("**Paper B only**")
            for d in only_b:
                st.write(f"🔹 {d} — {top5_b[d]}%")

        all_domains = list(set(top5_a.keys()) | set(top5_b.keys()))
        scores_a    = [top5_a.get(d, 0) for d in all_domains]
        scores_b    = [top5_b.get(d, 0) for d in all_domains]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Paper A", x=all_domains, y=scores_a, marker_color="#60a5fa"))
        fig.add_trace(go.Bar(name="Paper B", x=all_domains, y=scores_b, marker_color="#f472b6"))
        fig.update_layout(
            barmode="group", title="Domain Score Comparison",
            xaxis_title="Domain", yaxis_title="Confidence (%)",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧾 Verdict")
        if sim_pct >= 70:
            st.success(f"🟢 Very similar papers ({sim_pct}%) — likely in the same research area.")
        elif sim_pct >= 40:
            st.warning(f"🟡 Moderately related ({sim_pct}%) — some overlap in methods or topics.")
        else:
            st.error(f"🔴 Very different papers ({sim_pct}%) — distinct research domains.")


# =================================================================
# ⏳ TIME MACHINE
# =================================================================
elif menu == "⏳ Time Machine":

    st.title("⏳ Domain Trend Time Machine")
    st.write("See how CS research domains grew or shrank from 2010 to 2024 using arXiv metadata.")

    @st.cache_data
    def load_trend_data():
        df = pd.read_json(
            "data/arxiv-metadata-oai-snapshot.json",
            lines=True, nrows=200000
        )
        df = df[["categories", "update_date"]].dropna()
        df["year"] = pd.to_datetime(df["update_date"], errors="coerce").dt.year
        df = df[df["year"].between(2010, 2024)]
        df["category"] = df["categories"].apply(lambda x: x.split()[0])
        df = df[df["category"].str.startswith("cs.")]
        return df

    with st.spinner("📦 Loading arXiv data..."):
        df = load_trend_data()

    col1, col2 = st.columns([2, 1])
    with col1:
        top_n = st.slider("Number of domains to show", min_value=3, max_value=15, value=8)
    with col2:
        chart_type = st.selectbox("Chart type",
                                  ["Animated Bar Race", "Line Chart", "Area Chart", "Heatmap"])

    top_domains = df["category"].value_counts().head(top_n).index.tolist()
    df_filtered = df[df["category"].isin(top_domains)]
    yearly = (df_filtered.groupby(["year", "category"])
              .size().reset_index(name="count"))
    yearly["domain_name"] = yearly["category"].apply(lambda x: category_map.get(x, x))

    COLORS = [
        "#378ADD", "#1D9E75", "#D4537E", "#BA7517",
        "#534AB7", "#D85A30", "#639922", "#E24B4A",
        "#0F6E56", "#993556", "#854F0B", "#185FA5",
        "#3B6D11", "#A32D2D", "#3C3489"
    ]
    domain_colors = {d: COLORS[i % len(COLORS)] for i, d in enumerate(top_domains)}

    # ── Animated Bar Race ───────────────────────────────────────
    if chart_type == "Animated Bar Race":

        st.subheader("🏁 Bar Chart Race — CS Domain Growth")
        years  = sorted(yearly["year"].unique())
        frames = []

        for year in years:
            yr_data = (yearly[yearly["year"] == year]
                       .sort_values("count", ascending=True).tail(top_n))
            frames.append(go.Frame(
                data=[go.Bar(
                    x=yr_data["count"], y=yr_data["domain_name"],
                    orientation="h",
                    marker_color=[domain_colors.get(
                        yearly[yearly["domain_name"] == d]["category"].values[0],
                        "#378ADD") for d in yr_data["domain_name"]],
                    text=yr_data["count"], textposition="outside"
                )],
                name=str(year),
                layout=go.Layout(title_text=f"CS arXiv Papers — {year}")
            ))

        first = (yearly[yearly["year"] == years[0]]
                 .sort_values("count", ascending=True).tail(top_n))

        fig = go.Figure(
            data=[go.Bar(
                x=first["count"], y=first["domain_name"], orientation="h",
                marker_color=[domain_colors.get(
                    yearly[yearly["domain_name"] == d]["category"].values[0],
                    "#378ADD") for d in first["domain_name"]],
                text=first["count"], textposition="outside"
            )],
            layout=go.Layout(
                title=f"CS arXiv Papers — {years[0]}",
                xaxis=dict(title="Paper count", range=[0, yearly["count"].max() * 1.2]),
                yaxis=dict(title="Domain"),
                updatemenus=[dict(
                    type="buttons", showactive=False, y=1.15, x=0.5, xanchor="center",
                    buttons=[
                        dict(label="▶  Play", method="animate",
                             args=[None, dict(frame=dict(duration=600, redraw=True),
                                             fromcurrent=True, mode="immediate")]),
                        dict(label="⏸  Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode="immediate")])
                    ]
                )],
                sliders=[dict(
                    currentvalue=dict(prefix="Year: ", visible=True, xanchor="center"),
                    steps=[dict(method="animate", label=str(y),
                               args=[[str(y)], dict(mode="immediate",
                                                     frame=dict(duration=300, redraw=True))])
                           for y in years]
                )],
                height=500
            ),
            frames=frames
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Line Chart ──────────────────────────────────────────────
    elif chart_type == "Line Chart":

        st.subheader("📈 Domain Growth Lines (2010–2024)")
        fig = go.Figure()
        for cat in top_domains:
            cat_data = yearly[yearly["category"] == cat].sort_values("year")
            name = category_map.get(cat, cat)
            fig.add_trace(go.Scatter(
                x=cat_data["year"], y=cat_data["count"],
                mode="lines+markers", name=name,
                line=dict(color=domain_colors[cat], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>"
            ))
        fig.update_layout(xaxis_title="Year", yaxis_title="Number of papers",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.4),
                          hovermode="x unified", height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Growth Rate (2010 → 2024)")
        growth_rows = []
        for cat in top_domains:
            cat_data = yearly[yearly["category"] == cat].sort_values("year")
            if len(cat_data) >= 2:
                start  = cat_data.iloc[0]["count"]
                end    = cat_data.iloc[-1]["count"]
                growth = ((end - start) / max(start, 1)) * 100
                growth_rows.append({
                    "Domain": category_map.get(cat, cat), "Code": cat,
                    "2010 papers": int(start), "2024 papers": int(end),
                    "Growth %": f"+{growth:.0f}%" if growth >= 0 else f"{growth:.0f}%"
                })
        st.dataframe(pd.DataFrame(growth_rows).sort_values("2024 papers", ascending=False),
                     use_container_width=True, hide_index=True)

    # ── Area Chart ──────────────────────────────────────────────
    elif chart_type == "Area Chart":

        st.subheader("🌊 Stacked Area — Domain Share Over Time")
        pivot = yearly.pivot_table(
            index="year", columns="domain_name", values="count", fill_value=0
        ).reset_index()

        fig = go.Figure()
        for cat in top_domains:
            name = category_map.get(cat, cat)
            if name in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot["year"], y=pivot[name], name=name,
                    stackgroup="one", mode="none",
                    fillcolor=domain_colors.get(cat, "#378ADD") + "CC",
                    hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>"
                ))
        fig.update_layout(xaxis_title="Year", yaxis_title="Total papers",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.5),
                          height=480)
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ─────────────────────────────────────────────────
    elif chart_type == "Heatmap":

        st.subheader("🔥 Domain × Year Heatmap")
        pivot = yearly.pivot_table(
            index="domain_name", columns="year", values="count", fill_value=0
        )
        fig = px.imshow(pivot, color_continuous_scale="Blues",
                        labels=dict(x="Year", y="Domain", color="Papers"),
                        aspect="auto", title="Paper volume per domain per year")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🚀 Biggest single-year jumps")
        jumps = []
        for cat in top_domains:
            cat_data = (yearly[yearly["category"] == cat]
                        .sort_values("year")
                        .assign(delta=lambda d: d["count"].diff()))
            max_jump = cat_data.loc[cat_data["delta"].idxmax()]
            jumps.append({
                "Domain": category_map.get(cat, cat),
                "Boom year": int(max_jump["year"]),
                "New papers added": int(max_jump["delta"])
            })
        st.dataframe(pd.DataFrame(jumps).sort_values("New papers added", ascending=False),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🧠 Quick Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total papers loaded",   f"{len(df):,}")
    c2.metric("Most dominant domain",  category_map.get(df['category'].value_counts().idxmax(),
                                                         df['category'].value_counts().idxmax()))
    c3.metric("Most active year",      str(df["year"].value_counts().idxmax()))


# =================================================================
# 📊 DASHBOARD
# =================================================================
elif menu == "📊 Dashboard":

    st.title("📊 Dashboard")

    try:
        df = pd.read_json("data/arxiv-metadata-oai-snapshot.json", lines=True, nrows=5000)
        df["category"] = df["categories"].apply(lambda x: x.split()[0])
        df = df[df["category"].str.startswith("cs")]

        counts = df["category"].value_counts().head(8).reset_index()
        counts.columns = ["Category", "Count"]
        counts["Domain"] = counts["Category"].apply(lambda x: category_map.get(x, x))

        st.plotly_chart(px.bar(counts, x="Domain", y="Count",
                               title="Top CS Domains in arXiv Dataset"))
    except Exception:
        st.warning("⚠️ Dataset not found at data/arxiv-metadata-oai-snapshot.json")


# =================================================================
# 📜 HISTORY
# =================================================================
elif menu == "📜 History":

    st.title("📜 Prediction History")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"### 📄 Prediction {i + 1}")
            st.write(item["input"][:150] + "...")
            for name, score in item["result"]:
                st.write(f"👉 {name} — {score:.2f}%")
            st.markdown("---")

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No history yet — make a prediction first!")
