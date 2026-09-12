import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# MACHINE LEARNING-BASED BUYER SEGMENTATION
# STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = Path("outputs/data/final_buyer_segmentation.csv")

# These names are based on the actual cluster outputs:
# Cluster 0 -> highest investment and highest number of properties
# Cluster 1 -> highest average property price and area
# Cluster 2 -> largest segment with lower investment/property values
SEGMENT_NAMES = {
    0: "High-Value Portfolio Buyers",
    1: "Premium Property Buyers",
    2: "Standard Home Buyers"
}


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    possible_paths = [
        Path("outputs/data/final_buyer_segmentation.csv"),
        Path("../outputs/data/final_buyer_segmentation.csv"),
        Path("final_buyer_segmentation.csv")
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            return df

    return None


df = load_data()

if df is None:
    st.error(
        "Dataset not found. Run the final export cell in the Jupyter notebook "
        "first so that 'final_buyer_segmentation.csv' is created inside "
        "'outputs/data/'."
    )
    st.stop()


# ------------------------------------------------------------
# PREPROCESS FOR DASHBOARD
# ------------------------------------------------------------

if "segment_name" not in df.columns:
    df["segment_name"] = df["cluster"].map(SEGMENT_NAMES)
else:
    # Replace the old automatically generated names with clearer names
    df["segment_name"] = df["cluster"].map(SEGMENT_NAMES).fillna(df["segment_name"])

if "loan_applied" in df.columns:
    df["loan_applied"] = df["loan_applied"].astype(str).str.title()

if "acquisition_purpose" in df.columns:
    df["acquisition_purpose"] = df["acquisition_purpose"].astype(str).str.title()

if "client_type" in df.columns:
    df["client_type"] = df["client_type"].astype(str).str.title()


# ------------------------------------------------------------
# BASIC LANDING INTERFACE
# ------------------------------------------------------------

if "dashboard_started" not in st.session_state:
    st.session_state.dashboard_started = False

if not st.session_state.dashboard_started:
    st.markdown(
        """
        ## 👋 Welcome to the Buyer Intelligence Dashboard

        This dashboard provides machine-learning-based buyer segmentation
        and investment profiling for real estate market intelligence.

        ### What you can explore
        - 📊 Buyer segmentation and cluster distribution
        - 💰 Investment behavior and portfolio patterns
        - 🏡 Acquisition and financing behavior
        - 🌍 Geographic buyer distribution
        - 👥 Buyer demographics
        - 💡 Segment-specific business recommendations
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buyers", f"{len(df):,}")
    col2.metric("Buyer Segments", f"{df['cluster'].nunique():,}")
    col3.metric(
        "Total Investment",
        f"${df['total_investment'].sum():,.0f}"
    )

    st.markdown("### Ready to explore?")
    st.write(
        "Click the button below to open the interactive analytics dashboard."
    )

    if st.button("🚀 Explore Buyer Dashboard", type="primary"):
        st.session_state.dashboard_started = True
        st.rerun()

    st.stop()


# ------------------------------------------------------------
# OPTIONAL SIDEBAR FILTERS
# ------------------------------------------------------------
# Filters are hidden initially so the dashboard opens with a clean
# basic interface. Users can expand the section when they want to
# perform a focused analysis. Empty selections mean "All".

filtered_df = df.copy()

with st.sidebar.expander("🔎 Buyer Filters", expanded=False):
    if st.button("← Back to Overview"):
        st.session_state.dashboard_started = False
        st.rerun()

    st.caption("Filters are optional. Leave them empty to view all buyers.")

    # Country filter
    if "country" in df.columns:
        countries = sorted(df["country"].dropna().unique())
        selected_countries = st.multiselect(
            "Country",
            options=countries,
            default=[],
            key="country_filter"
        )
        if selected_countries:
            filtered_df = filtered_df[
                filtered_df["country"].isin(selected_countries)
            ]

    # Region filter
    if "region" in df.columns:
        region_source = filtered_df if selected_countries else df
        regions = sorted(region_source["region"].dropna().unique())
        selected_regions = st.multiselect(
            "Region",
            options=regions,
            default=[],
            key="region_filter"
        )
        if selected_regions:
            filtered_df = filtered_df[
                filtered_df["region"].isin(selected_regions)
            ]

    # Acquisition purpose filter
    if "acquisition_purpose" in df.columns:
        purposes = sorted(df["acquisition_purpose"].dropna().unique())
        selected_purposes = st.multiselect(
            "Acquisition Purpose",
            options=purposes,
            default=[],
            key="purpose_filter"
        )
        if selected_purposes:
            filtered_df = filtered_df[
                filtered_df["acquisition_purpose"].isin(selected_purposes)
            ]

    # Client type filter
    if "client_type" in df.columns:
        client_types = sorted(df["client_type"].dropna().unique())
        selected_client_types = st.multiselect(
            "Client Type",
            options=client_types,
            default=[],
            key="client_type_filter"
        )
        if selected_client_types:
            filtered_df = filtered_df[
                filtered_df["client_type"].isin(selected_client_types)
            ]

    if st.button("↩ Reset Filters", use_container_width=True):
        for key in [
            "country_filter",
            "region_filter",
            "purpose_filter",
            "client_type_filter"
        ]:
            st.session_state[key] = []
        st.rerun()


# ------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------

st.title("🏠 Real Estate Buyer Segmentation Dashboard")
st.markdown(
    "### Machine Learning-Based Buyer Segmentation and Investment Profiling"
)

st.write(
    "This dashboard analyzes buyer demographics, property investments, "
    "geographic distribution, financing behavior, and machine-learning "
    "buyer segments."
)

if filtered_df.empty:
    st.warning("No data is available for the selected filters.")
    st.stop()


# ------------------------------------------------------------
# KEY METRICS
# ------------------------------------------------------------

st.subheader("📊 Buyer Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Buyers",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Total Investment",
    f"${filtered_df['total_investment'].sum():,.0f}"
)

col3.metric(
    "Average Investment",
    f"${filtered_df['total_investment'].mean():,.0f}"
)

col4.metric(
    "Average Properties per Buyer",
    f"{filtered_df['total_properties'].mean():.2f}"
)


# ------------------------------------------------------------
# CLUSTER DISTRIBUTION
# ------------------------------------------------------------

st.divider()
st.subheader("🧩 Buyer Segment Distribution")

segment_counts = (
    filtered_df
    .groupby("segment_name")
    .size()
    .reset_index(name="buyer_count")
    .sort_values("buyer_count", ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        segment_counts,
        x="segment_name",
        y="buyer_count",
        text="buyer_count",
        title="Number of Buyers in Each Segment"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Number of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        segment_counts,
        names="segment_name",
        values="buyer_count",
        title="Buyer Segment Share"
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# INVESTOR / INVESTMENT BEHAVIOR
# ------------------------------------------------------------

st.divider()
st.subheader("💰 Investment Behavior by Buyer Segment")

segment_investment = (
    filtered_df
    .groupby("segment_name")
    .agg(
        buyer_count=("client_id", "count"),
        average_investment=("total_investment", "mean"),
        total_investment=("total_investment", "sum"),
        average_properties=("total_properties", "mean"),
        average_property_price=("average_property_price", "mean")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        segment_investment,
        x="segment_name",
        y="average_investment",
        text_auto=".2s",
        title="Average Investment by Segment"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Average Investment"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        segment_investment,
        x="segment_name",
        y="average_properties",
        text_auto=".2f",
        title="Average Number of Properties"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Average Properties"
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    segment_investment.style.format({
        "average_investment": "${:,.2f}",
        "total_investment": "${:,.2f}",
        "average_properties": "{:.2f}",
        "average_property_price": "${:,.2f}"
    }),
    use_container_width=True
)


# ------------------------------------------------------------
# ACQUISITION AND FINANCING BEHAVIOR
# ------------------------------------------------------------

st.divider()
st.subheader("🏡 Acquisition and Financing Behavior")

col1, col2 = st.columns(2)

with col1:
    purpose_segment = (
        filtered_df
        .groupby(["segment_name", "acquisition_purpose"])
        .size()
        .reset_index(name="buyer_count")
    )

    fig = px.bar(
        purpose_segment,
        x="segment_name",
        y="buyer_count",
        color="acquisition_purpose",
        barmode="group",
        title="Acquisition Purpose by Segment"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Number of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    loan_segment = (
        filtered_df
        .groupby(["segment_name", "loan_applied"])
        .size()
        .reset_index(name="buyer_count")
    )

    fig = px.bar(
        loan_segment,
        x="segment_name",
        y="buyer_count",
        color="loan_applied",
        barmode="group",
        title="Loan Application Behavior by Segment"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Number of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# GEOGRAPHIC BUYER ANALYSIS
# ------------------------------------------------------------

st.divider()
st.subheader("🌍 Geographic Buyer Analysis")

country_summary = (
    filtered_df
    .groupby("country")
    .agg(
        buyer_count=("client_id", "count"),
        total_investment=("total_investment", "sum")
    )
    .reset_index()
)

country_summary = country_summary.sort_values(
    "buyer_count",
    ascending=False
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        country_summary,
        x="country",
        y="buyer_count",
        title="Buyer Distribution by Country"
    )
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter_geo(
        country_summary,
        locations="country",
        locationmode="country names",
        size="buyer_count",
        hover_name="country",
        hover_data={
            "buyer_count": True,
            "total_investment": ":,.0f"
        },
        title="Geographic Distribution of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Buyer Distribution by Region and Segment")

region_segment = (
    filtered_df
    .groupby(["region", "segment_name"])
    .size()
    .reset_index(name="buyer_count")
)

top_regions = (
    filtered_df["region"]
    .value_counts()
    .head(15)
    .index
)

region_segment = region_segment[
    region_segment["region"].isin(top_regions)
]

fig = px.bar(
    region_segment,
    x="region",
    y="buyer_count",
    color="segment_name",
    barmode="stack",
    title="Top Regions by Buyer Segment"
)

fig.update_layout(
    xaxis_title="Region",
    yaxis_title="Number of Buyers"
)

st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# CUSTOMER DEMOGRAPHICS
# ------------------------------------------------------------

st.divider()
st.subheader("👥 Buyer Demographics")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        filtered_df,
        x="age",
        color="segment_name",
        nbins=20,
        barmode="overlay",
        title="Age Distribution by Segment"
    )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    satisfaction_segment = (
        filtered_df
        .groupby("segment_name")["satisfaction_score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        satisfaction_segment,
        x="segment_name",
        y="satisfaction_score",
        title="Average Satisfaction Score by Segment"
    )
    fig.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Average Satisfaction Score"
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# SEGMENT INSIGHTS PANEL
# ------------------------------------------------------------

st.divider()
st.subheader("🔍 Segment Insights Panel")

selected_segment = st.selectbox(
    "Select a Buyer Segment",
    sorted(filtered_df["segment_name"].unique())
)

segment_df = filtered_df[
    filtered_df["segment_name"] == selected_segment
]

st.markdown(f"### {selected_segment}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Buyers", f"{len(segment_df):,}")
col2.metric(
    "Average Investment",
    f"${segment_df['total_investment'].mean():,.0f}"
)
col3.metric(
    "Average Properties",
    f"{segment_df['total_properties'].mean():.2f}"
)
col4.metric(
    "Average Age",
    f"{segment_df['age'].mean():.1f}"
)

segment_stats = segment_df[
    [
        "age",
        "satisfaction_score",
        "total_properties",
        "total_investment",
        "average_property_price",
        "average_property_area"
    ]
].describe().T

st.dataframe(segment_stats, use_container_width=True)


# ------------------------------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------------------------------

st.divider()
st.subheader("💡 Business Recommendations")

insights = {
    "High-Value Portfolio Buyers": [
        "Prioritize premium account management and personalized property recommendations.",
        "Promote portfolio expansion opportunities because this segment owns the highest number of properties.",
        "Target high-value investment packages and exclusive property launches."
    ],
    "Premium Property Buyers": [
        "Promote larger and higher-priced properties.",
        "Use premium property recommendations and personalized location-based campaigns.",
        "Focus on maintaining satisfaction through high-quality customer support."
    ],
    "Standard Home Buyers": [
        "Focus on practical home recommendations and affordability.",
        "Use broad digital marketing because this is the largest buyer segment.",
        "Provide clear financing and property-comparison tools."
    ]
}

for recommendation in insights.get(selected_segment, []):
    st.write(f"• {recommendation}")


# ------------------------------------------------------------
# RAW DATA
# ------------------------------------------------------------

st.divider()
st.subheader("📄 Filtered Buyer Data")

show_columns = [
    column for column in [
        "client_id",
        "segment_name",
        "country",
        "region",
        "age",
        "client_type",
        "acquisition_purpose",
        "loan_applied",
        "satisfaction_score",
        "total_properties",
        "total_investment",
        "average_property_price",
        "average_property_area"
    ]
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[show_columns],
    use_container_width=True,
    height=400
)


# ------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Buyer Data",
    data=csv,
    file_name="filtered_buyer_segmentation.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()
st.caption(
    "Unified Mentor Internship Project | "
    "Machine Learning-Based Buyer Segmentation and Investment Profiling"
)
