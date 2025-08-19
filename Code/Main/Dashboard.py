import streamlit as st
import os
import pickle
import pandas as pd
import plotly.express as px
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, '..', 'Cleaned Datasets')
out_dir   = os.path.join(base_dir, '..', 'Output Files')
img_dir   = os.path.join(base_dir, '..', 'images') 

# Page setup
st.set_page_config(
    page_title="Outbreak Severity Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        /* Background with blur overlay */
        .stApp {{
            background: url("data:image/webp;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Apply blur using a pseudo-element overlay */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: inherit;
            filter: blur(8px);   /* adjust blur strength */
            z-index: -1;
        }}

        /* Make all text white */
        .stApp, .stMarkdown, .stSidebar, .stText, .stHeader, .css-1d391kg, .css-1v3fvcr {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call it with your .webp file
add_bg_from_local(os.path.join(img_dir, "3d-illustration-coronavirus-covid19-virus-260nw-1702140763.webp"))

# Sidebar Navigation
st.sidebar.title(" Dashboard Navigation")
section = st.sidebar.radio("", [
    " Project Overview",
    " Model Comparison",
    " Geospatial Analysis",
    " Temporal Trends",
    " Feature Exploration",
    " Predictions (May 2025–May 2026)"
])

# Section Routing
if section == " Project Overview":
    st.title("Outbreak Severity Insights & Predictions")
    st.markdown("""
    This interactive dashboard presents insights derived from **machine learning** and **deep learning** models designed to predict the **severity of infectious disease outbreaks**.

    ---
    **Key Goals:**
    - Predict outbreak severity using ML, DL, and Hybrid models
    - Visualize temporal, spatial, and feature-level patterns
    - Provide explainable and exportable insights to aid decision-making

    **Technologies Used:** Streamlit, Python, Scikit-learn, TensorFlow, XGBoost, Folium, Plotly
    """)

    # Try loading dataset & model metrics
    df, metrics_df = None, None
    try:
        df = pd.read_csv(os.path.join(clean_dir, "preprocessed_dataset.csv"))
    except Exception as e:
        st.error(f"⚠️ Dataset load failed: {e}")

    try:
        with open(os.path.join(out_dir, "model_metrics.pkl"), "rb") as f:
            model_results = pickle.load(f)
        metrics_df = pd.DataFrame(model_results).T
    except Exception:
        model_results = None
        metrics_df = None

    # KPIs row
    col1, col2, col3, col4 = st.columns(4)

    if df is not None:
        total_outbreaks = len(df)
        unique_institutions = df["Institution Name"].nunique() if "Institution Name" in df.columns else None
        year_min = int(df["Year"].min()) if "Year" in df.columns else None
        year_max = int(df["Year"].max()) if "Year" in df.columns else None

        col1.metric("Total Outbreaks", f"{total_outbreaks:,}")
        col2.metric("Institutions", f"{unique_institutions:,}" if unique_institutions else "—")
        col3.metric("Date Range", f"{year_min}–{year_max}" if (year_min and year_max) else "—")

        if metrics_df is not None:
            acc_col = next((c for c in metrics_df.columns if c.lower() == "accuracy"), None)
            if acc_col:
                best_model_name = metrics_df[acc_col].astype(float).idxmax()
                best_acc = float(metrics_df.loc[best_model_name, acc_col])
                col4.metric("Top Model", f"{best_model_name} ({best_acc:.3f})")
            else:
                col4.metric("Models in Comparison", f"{metrics_df.shape[0]}")
        else:
            col4.metric("Models in Comparison", "—")
    else:
        col1.metric("Total Outbreaks", "—")
        col2.metric("Institutions", "—")
        col3.metric("Date Range", "—")
        col4.metric("Models in Comparison", "—")

    st.markdown("""
    ---
    **Navigation Tips:**  
                
    •  Model Comparison → Review and compare model performance  
    •  Temporal Trends → See outbreak patterns across years, months, and seasons  
    •  Feature Exploration → Explore drivers of outbreak severity  
    """)

elif section == " Model Comparison":
    st.title(" Model Comparison")
    st.markdown("Compare performance of machine learning, deep learning, and hybrid models.")

    try:
        # Load all model metrics
        with open(os.path.join(out_dir, "model_metrics.pkl"), "rb") as f:
            model_results = pickle.load(f)

        # Convert to DataFrame for table display
        df_metrics = pd.DataFrame(model_results).T  # Transpose so models are rows
        df_metrics_rounded = df_metrics.round(4)

        st.subheader(" Overall Model Comparison Table")
        st.dataframe(df_metrics_rounded, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 View Detailed Metrics for a Single Model")
        model_names = list(model_results.keys())
        selected_model = st.selectbox("**Select a model to view metrics:**", model_names)

        st.markdown(f"###  Metrics for {selected_model}")
        selected_metrics = model_results[selected_model]
        for metric_name, value in selected_metrics.items():
            if isinstance(value, float):
                st.metric(label=metric_name, value=f"{value:.4f}")
            else:
                st.metric(label=metric_name, value=value)

    except FileNotFoundError:
        st.error("⚠️ 'model_metrics.pkl' not found. Please run `main.py` to generate it first.")
    except Exception as e:
        st.error(f"❌ Failed to load or process model metrics: {str(e)}")

elif section == " Geospatial Analysis":
    st.title(" Geospatial-style Outbreak Overview")
    st.markdown("Explore outbreak counts by institution and setting (filtered by year and severity).")

    try:
        df = pd.read_csv(os.path.join(clean_dir, "preprocessed_dataset.csv"))

        def map_severity(days):
            if days <= 7:
                return "Mild"
            elif days <= 21:
                return "Moderate"
            else:
                return "Severe"

        df["Severity"] = df["Outbreak Duration (days)"].apply(map_severity)

        years = sorted(df["Year"].dropna().unique())
        selected_year = st.selectbox("**Select Year:**", years)
        severity_options = df["Severity"].unique().tolist()
        selected_severities = st.multiselect("**Select Severity Level(s):**", severity_options, default=severity_options)

        filtered_df = df[(df["Year"] == selected_year) & (df["Severity"].isin(selected_severities))]

        st.success(f"Showing {len(filtered_df)} outbreaks for {selected_year} with selected severity levels.")

        institution_counts = filtered_df["Institution Name"].value_counts().reset_index()
        institution_counts.columns = ["Institution Name", "Count"]

        fig1 = px.bar(
            institution_counts,
            x="Institution Name",
            y="Count",
            title="Outbreaks per Institution"
        )
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

        setting_counts = filtered_df["Outbreak Setting"].value_counts().reset_index()
        setting_counts.columns = ["Outbreak Setting", "Count"]

        fig2 = px.bar(
            setting_counts,
            x="Outbreak Setting",
            y="Count",
            title="Outbreaks by Setting"
        )
        st.plotly_chart(fig2, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ File 'preprocessed_dataset.csv' not found in 'Cleaned Datasets/'. Please check the path.")
    except Exception as e:
        st.error(f"❌ Failed to load or process dataset: {str(e)}")

elif section == " Temporal Trends":
    st.title(" Temporal Trends")
    st.markdown("Visualize outbreak patterns over time and across seasons.")

    try:
        # Load dataset
        df = pd.read_csv(os.path.join(clean_dir, "preprocessed_dataset.csv"))

        # Map Severity (kept for consistency)
        def map_severity(days):
            if days <= 7:
                return "Mild"
            elif days <= 21:
                return "Moderate"
            else:
                return "Severe"

        df["Severity"] = df["Outbreak Duration (days)"].apply(map_severity)

        # Ensure date columns are in datetime format
        df["Date Outbreak Began"] = pd.to_datetime(df["Date Outbreak Began"], errors="coerce")
        df = df.dropna(subset=["Date Outbreak Began"])

        # ----------------- Yearly Trends (no dropdown by design) -----------------
        st.subheader(" Outbreaks per Year")
        yearly_counts = df["Year"].value_counts().sort_index().reset_index()
        yearly_counts.columns = ["Year", "Outbreak Count"]
        fig_yearly = px.bar(yearly_counts, x="Year", y="Outbreak Count", title="Outbreaks per Year")
        st.plotly_chart(fig_yearly, use_container_width=True)

        # Precompute helpers
        all_years = sorted(df["Year"].dropna().unique().tolist())
        years_with_all = ["All"] + all_years

        # ----------------- Seasonal Trends -----------------
        st.subheader(" Seasonal Trends")
        selected_season_year = st.selectbox("Select Year for Seasonal Trends:", years_with_all, key="season_year")

        if selected_season_year == "All":
            df_season = df.copy()
        else:
            df_season = df[df["Year"] == selected_season_year].copy()

        df_season["MonthNum"] = df_season["Date Outbreak Began"].dt.month
        df_season["Season"] = df_season["MonthNum"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })

        seasonal_counts = (
            df_season["Season"].value_counts()
            .reindex(["Winter", "Spring", "Summer", "Fall"])
            .fillna(0)
            .reset_index()
        )
        seasonal_counts.columns = ["Season", "Outbreak Count"]
        season_title_suffix = "All Years" if selected_season_year == "All" else str(selected_season_year)
        fig_seasonal = px.bar(
            seasonal_counts,
            x="Season",
            y="Outbreak Count",
            title=f"Seasonal Outbreaks — {season_title_suffix}"
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        # ----------------- Monthly Trends -----------------
        st.subheader(" Monthly Trends")
        selected_month_year = st.selectbox("Select Year for Monthly Trends:", years_with_all, key="month_year")

        if selected_month_year == "All":
            df_month = df.copy()
        else:
            df_month = df[df["Year"] == selected_month_year].copy()

        df_month["MonthNum"] = df_month["Date Outbreak Began"].dt.month
        df_month["Month"] = df_month["MonthNum"].apply(lambda m: calendar.month_name[m])

        monthly_counts = (
            df_month["Month"].value_counts()
            .reindex(list(calendar.month_name)[1:])  # Jan..Dec in order
            .fillna(0)
            .reset_index()
        )
        monthly_counts.columns = ["Month", "Outbreak Count"]

        month_title_suffix = "All Years" if selected_month_year == "All" else str(selected_month_year)
        fig_monthly = px.bar(
            monthly_counts,
            x="Month",
            y="Outbreak Count",
            title=f"Monthly Outbreaks — {month_title_suffix}"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ----------------- Weekday Distributions -----------------
        st.subheader(" Weekday Distribution")
        selected_weekday_year = st.selectbox("Select Year for Weekday Distribution:", years_with_all, key="weekday_year")

        if selected_weekday_year == "All":
            df_weekday = df.copy()
        else:
            df_weekday = df[df["Year"] == selected_weekday_year].copy()

        df_weekday["Weekday"] = df_weekday["Date Outbreak Began"].dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_counts = (
            df_weekday["Weekday"].value_counts()
            .reindex(weekday_order)
            .fillna(0)
            .reset_index()
        )
        weekday_counts.columns = ["Weekday", "Outbreak Count"]

        wd_title_suffix = "All Years" if selected_weekday_year == "All" else str(selected_weekday_year)
        fig_weekday = px.bar(
            weekday_counts,
            x="Weekday",
            y="Outbreak Count",
            title=f"Weekday Outbreaks — {wd_title_suffix}"
        )
        st.plotly_chart(fig_weekday, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ 'preprocessed_dataset.csv' not found in 'Cleaned Datasets/'. Please check the path.")
    except Exception as e:
        st.error(f"❌ Failed to load or process dataset: {str(e)}")

elif section == " Feature Exploration":
    st.title(" Feature Exploration")
    st.markdown("Dive into feature-level insights to understand outbreak drivers. Use the year filter and download any chart as a PNG.")

    # Helper to turn a Matplotlib figure into PNG bytes for st.download_button
    def fig_to_png_bytes(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        buf.seek(0)
        return buf.getvalue()

    try:
        df = pd.read_csv(os.path.join(clean_dir, "preprocessed_dataset.csv"))

        # Severity from duration (consistent with other tabs)
        def map_severity(days):
            if days <= 7:
                return "Mild"
            elif days <= 21:
                return "Moderate"
            else:
                return "Severe"

        df["Severity"] = df["Outbreak Duration (days)"].apply(map_severity)

        # ---- Filters (make it interactive) ----
        years = ["All"] + sorted(df["Year"].dropna().unique().tolist())
        sel_year = st.selectbox("Filter by Year", years, index=0)

        filtered = df.copy()
        if sel_year != "All":
            filtered = filtered[filtered["Year"] == sel_year]

        # Optional: top-N control for long categorical lists
        top_n_agents = st.slider("Top-N Causative Agents to show", min_value=5, max_value=30, value=10, step=1)

        # ---- 1) Boxplot: Duration vs Severity ----
        st.subheader("📦 Boxplot: Duration vs. Severity")
        fig1, ax1 = plt.subplots()
        sns.boxplot(x="Severity", y="Outbreak Duration (days)", data=filtered, ax=ax1)
        ax1.set_title(f"Outbreak Duration by Severity{' — ' + str(sel_year) if sel_year != 'All' else ''}")
        st.pyplot(fig1, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG: Duration vs Severity",
            data=fig_to_png_bytes(fig1),
            file_name=f"boxplot_duration_vs_severity_{sel_year if sel_year!='All' else 'all'}.png",
            mime="image/png"
        )

        # ---- 2) Countplot: Outbreak Setting ----
        st.subheader(" Countplot: Outbreak Setting")
        setting_order = filtered["Outbreak Setting"].value_counts().index
        fig2, ax2 = plt.subplots()
        sns.countplot(data=filtered, y="Outbreak Setting", order=setting_order, ax=ax2)
        ax2.set_title(f"Outbreaks by Setting{' — ' + str(sel_year) if sel_year != 'All' else ''}")
        st.pyplot(fig2, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG: Outbreaks by Setting",
            data=fig_to_png_bytes(fig2),
            file_name=f"countplot_outbreak_setting_{sel_year if sel_year!='All' else 'all'}.png",
            mime="image/png"
        )

        # ---- 3) Countplot: Type of Outbreak ----
        st.subheader(" Countplot: Type of Outbreak")
        type_order = filtered["Type of Outbreak"].value_counts().index
        fig3, ax3 = plt.subplots()
        sns.countplot(data=filtered, y="Type of Outbreak", order=type_order, ax=ax3)
        ax3.set_title(f"Outbreaks by Type{' — ' + str(sel_year) if sel_year != 'All' else ''}")
        st.pyplot(fig3, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG: Outbreaks by Type",
            data=fig_to_png_bytes(fig3),
            file_name=f"countplot_outbreak_type_{sel_year if sel_year!='All' else 'all'}.png",
            mime="image/png"
        )

        # ---- 4) Countplot: Causative Agent (Top-N) ----
        st.subheader(" Countplot: Causative Agent (Top-N)")
        top_agents = filtered["Causative Agent-1"].value_counts().head(top_n_agents).index
        fig4, ax4 = plt.subplots()
        sns.countplot(data=filtered[filtered["Causative Agent-1"].isin(top_agents)],
                      y="Causative Agent-1",
                      order=top_agents,
                      ax=ax4)
        ax4.set_title(f"Top {top_n_agents} Causative Agents{' — ' + str(sel_year) if sel_year != 'All' else ''}")
        st.pyplot(fig4, use_container_width=True)
        st.download_button(
            f"⬇️ Download PNG: Top {top_n_agents} Causative Agents",
            data=fig_to_png_bytes(fig4),
            file_name=f"countplot_top{top_n_agents}_agents_{sel_year if sel_year!='All' else 'all'}.png",
            mime="image/png"
        )

        # ---- 5) Correlation Heatmap (Numeric) ----
        st.subheader(" Correlation Heatmap (Numeric Features)")
        numeric_df = filtered[["Outbreak Duration (days)", "Year"]].copy()
        corr = numeric_df.corr(numeric_only=True)
        fig5, ax5 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
        ax5.set_title(f"Correlation Heatmap{' — ' + str(sel_year) if sel_year != 'All' else ''}")
        st.pyplot(fig5, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG: Correlation Heatmap",
            data=fig_to_png_bytes(fig5),
            file_name=f"heatmap_correlation_{sel_year if sel_year!='All' else 'all'}.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"❌ Failed to load feature plots: {str(e)}")

elif section == " Predictions (May 2025–May 2026)":
    st.title(" Predictions (May 2025–May 2026)")

    files = {
        "XGBoost predictions": os.path.join(out_dir, "predictions_xgb_may2025_may2026.csv"),
        "Ensemble (XGB + CNN) predictions": os.path.join(out_dir, "predictions_ensemble_may2025_may2026.csv"),
    }
    available = {name: path for name, path in files.items() if os.path.exists(path)}
    missing = [name for name, path in files.items() if not os.path.exists(path)]

    if missing:
        st.warning("Missing in **Output Files**:\n- " + "\n- ".join(missing))
    if not available:
        st.stop()

    st.markdown("**Select a predictions file to view:**")
    choice = st.selectbox("", list(available.keys()), label_visibility="collapsed", key="pred_file")
    path = available[choice]

    try:
        df_pred = pd.read_csv(path)

        # (optional but useful) normalize the two filter columns if present
        for col in ["Outbreak Setting", "Type of Outbreak"]:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].astype(str).str.strip()

        st.success(f"Loaded: `{os.path.basename(path)}`  •  {df_pred.shape[0]:,} rows × {df_pred.shape[1]:,} columns")

        # left-align the two filters
        st.markdown("""
            <style>div[data-testid="stHorizontalBlock"]{gap:0rem!important;}</style>
        """, unsafe_allow_html=True)
       #c1, c2, _ = st.columns([3, 3, 5])
        c1, _, c2, _ = st.columns([3,0.1,3,5])

        # Outbreak Setting
        if "Outbreak Setting" in df_pred.columns and df_pred["Outbreak Setting"].notna().any():
            settings = ["All"] + sorted(df_pred["Outbreak Setting"].dropna().unique().tolist())
            sel_set = c1.selectbox("**Outbreak Setting**", settings, index=0, key="f_set")
        else:
            sel_set = "All"

        # Type of Outbreak
        if "Type of Outbreak" in df_pred.columns and df_pred["Type of Outbreak"].notna().any():
            types = ["All"] + sorted(df_pred["Type of Outbreak"].dropna().unique().tolist())
            sel_type = c2.selectbox("**Type of Outbreak**", types, index=0, key="f_type")
        else:
            sel_type = "All"

        # apply filters
        mask = pd.Series(True, index=df_pred.index)
        if sel_set != "All" and "Outbreak Setting" in df_pred.columns:
            mask &= (df_pred["Outbreak Setting"] == sel_set)
        if sel_type != "All" and "Type of Outbreak" in df_pred.columns:
            mask &= (df_pred["Type of Outbreak"] == sel_type)

        df_view = df_pred[mask].copy()

        # small badge
        st.caption(f"Showing **{df_view.shape[0]:,}** of **{df_pred.shape[0]:,}** rows")

        st.dataframe(df_view, use_container_width=True)

        # downloads
        st.download_button(
            "⬇️ Download full CSV",
            data=open(path, "rb").read(),
            file_name=os.path.basename(path),
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Failed to load predictions: {e}")


