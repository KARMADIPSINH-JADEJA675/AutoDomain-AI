import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfReader
import time
import json
from utils.preprocess import clean_text
from utils.helper import category_map

# ================= LOAD =================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= PAGE =================
st.set_page_config(page_title="AutoDomain AI", layout="wide")

# 🎨 UI PRO CSS
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
.card:hover {
    transform: scale(1.02);
}
.card h3 { color: #60a5fa; }
.card p { color: #cbd5f5; }
.center { text-align: center; }
.big-title { font-size: 42px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= NAVIGATION =================
menu = st.sidebar.radio("🧭 Navigation", ["🏠 Home", "📄 Predict", "🔍 Compare", "⏳ Time Machine", "📊 Dashboard", "📜 History"])
# ================= VALIDATION =================
def is_valid_input(text):
    return len(text.split()) >= 8 and len(text.strip()) >= 30

# ================= EXPLAIN =================
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

# ================= HOME =================
if menu == "🏠 Home":

    st.markdown('<div class="center big-title">🤖 AutoDomain AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="center">Smart Research Paper Domain Classifier</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>📌 About</h3>
        <p>Classifies research papers into domains using content analysis of title and abstract.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🧭 How to Use</h3>
        <p>Go to Predict → Enter text or upload PDF → Click Predict → View results.</p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>📊 Features</h3>
        <p>Top predictions, charts, explanation, history tracking, dashboard analytics.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
        <h3>⚠️ Guidelines</h3>
        <p>Use meaningful input (8+ words). Better results with full abstract.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("💡 Example Input")
    st.code("""
Deep learning based image classification using convolutional neural networks 
for real-time object detection.
""")

    st.success("🚀 Ready to classify your research paper!")

# ================= PREDICT =================
elif menu == "📄 Predict":

    st.header("📄 Predict Domain")

    option = st.radio("Choose Input Method", ["✍️ Text", "📄 Upload PDF"])
    user_input = ""

    if option == "✍️ Text":
        title = st.text_input("Enter Title")
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
            st.error("❌ Invalid input")
            st.stop()

        with st.spinner("🔍 Analyzing..."):
            time.sleep(1)

        vec = vectorizer.transform([clean_text(user_input)])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🎯 Predictions")

        results = []

        for i in top3_idx:
            name = category_map.get(classes[i], classes[i])
            score = probs[i] * 100
            results.append((name, score))

            st.write(f"👉 {name} — {score:.2f}%")
            st.progress(int(score))

        # Save history
        st.session_state.history.append({
            "input": user_input,
            "result": results
        })

        # PIE CHART
        df = pd.DataFrame({
            "Domain": [r[0] for r in results],
            "Score": [r[1] for r in results]
        })
        st.plotly_chart(px.pie(df, names="Domain", values="Score"))

        # EXPLANATION
        st.subheader("🧠 Explanation")
        keywords = explain_prediction(user_input, vectorizer, model)
        for word, score in keywords:
            st.write(f"🔹 {word} ({score:.3f})")

# ================= DASHBOARD =================
elif menu == "📊 Dashboard":

    st.title("📊 Dashboard")

    try:
        df = pd.read_json("data/arxiv-metadata-oai-snapshot.json", lines=True, nrows=5000)
        df['category'] = df['categories'].apply(lambda x: x.split()[0])
        df = df[df['category'].str.startswith('cs')]

        counts = df['category'].value_counts().head(8).reset_index()
        counts.columns = ['Category', 'Count']
        counts['Domain'] = counts['Category'].apply(lambda x: category_map.get(x, x))

        st.plotly_chart(px.bar(counts, x='Domain', y='Count'))

    except:
        st.warning("Dataset not found")

# ================= HISTORY =================
elif menu == "📜 History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"### 📄 Input {i+1}")
            st.write(item["input"][:150] + "...")

            for name, score in item["result"]:
                st.write(f"👉 {name} — {score:.2f}%")

            st.markdown("---")

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No history available")
        # ================= COMPARE =================
elif menu == "🔍 Compare":

    st.header("🔍 Paper Comparison")
    st.write("Compare two research papers and see their domain overlap.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Paper A")
        title_a = st.text_input("Title A", key="title_a")
        abstract_a = st.text_area("Abstract A", height=200, key="abstract_a")

    with col2:
        st.subheader("📄 Paper B")
        title_b = st.text_input("Title B", key="title_b")
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

        # --- Get top-5 predictions for each paper ---
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

        # --- Similarity Score ---
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(vec_a, vec_b)[0][0]
        sim_pct = round(similarity * 100, 1)

        st.markdown("---")
        st.subheader("📊 Comparison Results")

        # Similarity metric
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        sim_col1.metric("🔗 Similarity Score", f"{sim_pct}%")
        sim_col2.metric("🤝 Shared Domains", len(shared))
        sim_col3.metric("🔀 Unique Domains", len(only_a) + len(only_b))

        st.progress(int(sim_pct))

        # Shared domains
        st.subheader("🤝 Shared Domains")
        if shared:
            for domain in shared:
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"✅ {domain} — {top5_a[domain]}%")
                    st.progress(int(top5_a[domain]))
                with c2:
                    st.write(f"✅ {domain} — {top5_b[domain]}%")
                    st.progress(int(top5_b[domain]))
        else:
            st.info("No shared top domains found.")

        # Side-by-side unique domains
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

        # Radar/Bar chart overlay
        import plotly.graph_objects as go
        import pandas as pd

        all_domains = list(set(top5_a.keys()) | set(top5_b.keys()))
        scores_a = [top5_a.get(d, 0) for d in all_domains]
        scores_b = [top5_b.get(d, 0) for d in all_domains]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Paper A", x=all_domains, y=scores_a, marker_color="#60a5fa"))
        fig.add_trace(go.Bar(name="Paper B", x=all_domains, y=scores_b, marker_color="#f472b6"))
        fig.update_layout(
            barmode='group',
            title="Domain Score Comparison",
            xaxis_title="Domain",
            yaxis_title="Confidence (%)",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Verdict
        st.subheader("🧾 Verdict")
        if sim_pct >= 70:
            st.success(f"🟢 Very similar papers ({sim_pct}%) — likely in the same research area.")
        elif sim_pct >= 40:
            st.warning(f"🟡 Moderately related papers ({sim_pct}%) — some overlap in methods or topics.")
        else:
            st.error(f"🔴 Very different papers ({sim_pct}%) — distinct research domains.")
            # ================= TIME MACHINE =================
elif menu == "⏳ Time Machine":

    st.title("⏳ Domain Trend Time Machine")
    st.write("See how CS research domains grew or shrank from 2010 to 2024 using arXiv metadata.")

    # ---- Load & cache data ----
    @st.cache_data
    def load_trend_data():
        df = pd.read_json(
            "data/arxiv-metadata-oai-snapshot.json",
            lines=True,
            nrows=200000
        )
        df = df[['categories', 'update_date']].dropna()

        # Parse year
        df['year'] = pd.to_datetime(df['update_date'], errors='coerce').dt.year
        df = df[df['year'].between(2010, 2024)]

        # Primary category only, CS only
        df['category'] = df['categories'].apply(lambda x: x.split()[0])
        df = df[df['category'].str.startswith('cs.')]

        return df

    with st.spinner("📦 Loading arXiv data..."):
        df = load_trend_data()

    # ---- Controls ----
    col1, col2 = st.columns([2, 1])

    with col1:
        top_n = st.slider("Number of domains to show", min_value=3, max_value=15, value=8)

    with col2:
        chart_type = st.selectbox("Chart type", ["Animated Bar Race", "Line Chart", "Area Chart", "Heatmap"])

    # ---- Compute top domains ----
    top_domains = (
        df['category']
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    df_filtered = df[df['category'].isin(top_domains)]

    yearly = (
        df_filtered
        .groupby(['year', 'category'])
        .size()
        .reset_index(name='count')
    )

    yearly['domain_name'] = yearly['category'].apply(
        lambda x: category_map.get(x, x)
    )

    # ---- Color palette ----
    COLORS = [
        "#378ADD", "#1D9E75", "#D4537E", "#BA7517",
        "#534AB7", "#D85A30", "#639922", "#E24B4A",
        "#0F6E56", "#993556", "#854F0B", "#185FA5",
        "#3B6D11", "#A32D2D", "#3C3489"
    ]
    domain_colors = {d: COLORS[i % len(COLORS)] for i, d in enumerate(top_domains)}

    import plotly.graph_objects as go
    import plotly.express as px

    # ================================================================
    # CHART 1: Animated Bar Race
    # ================================================================
    if chart_type == "Animated Bar Race":

        st.subheader("🏁 Bar Chart Race — CS Domain Growth")

        years = sorted(yearly['year'].unique())
        frames = []

        for year in years:
            year_data = yearly[yearly['year'] == year].sort_values('count', ascending=True).tail(top_n)
            frames.append(go.Frame(
                data=[go.Bar(
                    x=year_data['count'],
                    y=year_data['domain_name'],
                    orientation='h',
                    marker_color=[domain_colors.get(
                        yearly[yearly['domain_name'] == d]['category'].values[0], "#378ADD"
                    ) for d in year_data['domain_name']],
                    text=year_data['count'],
                    textposition='outside'
                )],
                name=str(year),
                layout=go.Layout(title_text=f"CS arXiv Papers — {year}")
            ))

        first_year = yearly[yearly['year'] == years[0]].sort_values('count', ascending=True).tail(top_n)

        fig = go.Figure(
            data=[go.Bar(
                x=first_year['count'],
                y=first_year['domain_name'],
                orientation='h',
                marker_color=[domain_colors.get(
                    yearly[yearly['domain_name'] == d]['category'].values[0], "#378ADD"
                ) for d in first_year['domain_name']],
                text=first_year['count'],
                textposition='outside'
            )],
            layout=go.Layout(
                title=f"CS arXiv Papers — {years[0]}",
                xaxis=dict(title="Paper count", range=[0, yearly['count'].max() * 1.2]),
                yaxis=dict(title="Domain"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.15, x=0.5, xanchor="center",
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

    # ================================================================
    # CHART 2: Line Chart
    # ================================================================
    elif chart_type == "Line Chart":

        st.subheader("📈 Domain Growth Lines (2010–2024)")

        fig = go.Figure()

        for cat in top_domains:
            cat_data = yearly[yearly['category'] == cat].sort_values('year')
            name = category_map.get(cat, cat)
            fig.add_trace(go.Scatter(
                x=cat_data['year'],
                y=cat_data['count'],
                mode='lines+markers',
                name=name,
                line=dict(color=domain_colors[cat], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>"
            ))

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of papers",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            hovermode="x unified",
            height=480
        )

        st.plotly_chart(fig, use_container_width=True)

        # Growth rate table
        st.subheader("📊 Growth Rate (2010 → 2024)")
        growth_rows = []
        for cat in top_domains:
            cat_data = yearly[yearly['category'] == cat].sort_values('year')
            if len(cat_data) >= 2:
                start = cat_data.iloc[0]['count']
                end = cat_data.iloc[-1]['count']
                growth = ((end - start) / max(start, 1)) * 100
                growth_rows.append({
                    "Domain": category_map.get(cat, cat),
                    "Code": cat,
                    "2010 papers": int(start),
                    "2024 papers": int(end),
                    "Growth %": f"+{growth:.0f}%" if growth >= 0 else f"{growth:.0f}%"
                })

        growth_df = pd.DataFrame(growth_rows).sort_values("2024 papers", ascending=False)
        st.dataframe(growth_df, use_container_width=True, hide_index=True)

    # ================================================================
    # CHART 3: Area Chart
    # ================================================================
    elif chart_type == "Area Chart":

        st.subheader("🌊 Stacked Area — Domain Share Over Time")

        pivot = yearly.pivot_table(
            index='year', columns='domain_name', values='count', fill_value=0
        ).reset_index()

        fig = go.Figure()

        domain_names = [category_map.get(c, c) for c in top_domains]

        for name in domain_names:
            if name in pivot.columns:
                cat = next((c for c in top_domains if category_map.get(c, c) == name), name)
                fig.add_trace(go.Scatter(
                    x=pivot['year'],
                    y=pivot[name],
                    name=name,
                    stackgroup='one',
                    mode='none',
                    fillcolor=domain_colors.get(cat, "#378ADD") + "CC",
                    hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>"
                ))

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Total papers",
            legend=dict(orientation="h", yanchor="bottom", y=-0.5),
            height=480
        )

        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # CHART 4: Heatmap
    # ================================================================
    elif chart_type == "Heatmap":

        st.subheader("🔥 Domain × Year Heatmap")

        pivot = yearly.pivot_table(
            index='domain_name', columns='year', values='count', fill_value=0
        )

        fig = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            labels=dict(x="Year", y="Domain", color="Papers"),
            aspect="auto",
            title="Paper volume per domain per year"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Highlight boom years
        st.subheader("🚀 Biggest single-year jumps")
        jumps = []
        for cat in top_domains:
            cat_data = yearly[yearly['category'] == cat].sort_values('year')
            cat_data = cat_data.assign(delta=cat_data['count'].diff())
            max_jump = cat_data.loc[cat_data['delta'].idxmax()]
            jumps.append({
                "Domain": category_map.get(cat, cat),
                "Boom year": int(max_jump['year']),
                "New papers added": int(max_jump['delta'])
            })

        jumps_df = pd.DataFrame(jumps).sort_values("New papers added", ascending=False)
        st.dataframe(jumps_df, use_container_width=True, hide_index=True)

    # ---- Footer insight ----
    st.markdown("---")
    st.subheader("🧠 Quick Insights")

    total_papers = len(df)
    top_domain = df['category'].value_counts().idxmax()
    top_year = df['year'].value_counts().idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total papers loaded", f"{total_papers:,}")
    c2.metric("Most dominant domain", category_map.get(top_domain, top_domain))
    c3.metric("Most active year", str(top_year))
