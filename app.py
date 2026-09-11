import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEGMENT_NAMES = {
    0: "High-Value Portfolio Buyers",
    1: "Premium Property Buyers",
    2: "Standard Home Buyers"
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_paths = [
        Path("outputs/data/final_buyer_segmentation.csv"),
        Path("../outputs/data/final_buyer_segmentation.csv"),
        Path("final_buyer_segmentation.csv")
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return None


df = load_data()


if df is None:

    st.error(
        "Dataset not found. Run the final export cell in the Jupyter notebook "
        "first so that 'final_buyer_segmentation.csv' is created inside "
        "'outputs/data/'."
    )

    st.stop()


# ============================================================
# DATA PREPROCESSING FOR DASHBOARD
# ============================================================

if "segment_name" not in df.columns:

    df["segment_name"] = df["cluster"].map(SEGMENT_NAMES)

else:

    df["segment_name"] = (
        df["cluster"]
        .map(SEGMENT_NAMES)
        .fillna(df["segment_name"])
    )


if "loan_applied" in df.columns:

    df["loan_applied"] = (
        df["loan_applied"]
        .astype(str)
        .str.title()
    )


if "acquisition_purpose" in df.columns:

    df["acquisition_purpose"] = (
        df["acquisition_purpose"]
        .astype(str)
        .str.title()
    )


if "client_type" in df.columns:

    df["client_type"] = (
        df["client_type"]
        .astype(str)
        .str.title()
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Buyer Filters")

filtered_df = df.copy()


# ------------------------------------------------------------
# COUNTRY FILTER
# ------------------------------------------------------------

if "country" in filtered_df.columns:

    countries = sorted(
        filtered_df["country"]
        .dropna()
        .unique()
    )

    selected_countries = st.sidebar.multiselect(
        "Country",
        options=countries,
        default=countries
    )

    filtered_df = filtered_df[
        filtered_df["country"].isin(selected_countries)
    ]


# ------------------------------------------------------------
# REGION FILTER
# ------------------------------------------------------------

if "region" in filtered_df.columns:

    regions = sorted(
        filtered_df["region"]
        .dropna()
        .unique()
    )

    selected_regions = st.sidebar.multiselect(
        "Region",
        options=regions,
        default=regions
    )

    filtered_df = filtered_df[
        filtered_df["region"].isin(selected_regions)
    ]


# ------------------------------------------------------------
# ACQUISITION PURPOSE FILTER
# ------------------------------------------------------------

if "acquisition_purpose" in filtered_df.columns:

    purposes = sorted(
        filtered_df["acquisition_purpose"]
        .dropna()
        .unique()
    )

    selected_purposes = st.sidebar.multiselect(
        "Acquisition Purpose",
        options=purposes,
        default=purposes
    )

    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"].isin(selected_purposes)
    ]


# ------------------------------------------------------------
# CLIENT TYPE FILTER
# ------------------------------------------------------------

if "client_type" in filtered_df.columns:

    client_types = sorted(
        filtered_df["client_type"]
        .dropna()
        .unique()
    )

    selected_client_types = st.sidebar.multiselect(
        "Client Type",
        options=client_types,
        default=client_types
    )

    filtered_df = filtered_df[
        filtered_df["client_type"].isin(selected_client_types)
    ]


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏠 Real Estate Buyer Segmentation Dashboard")

st.markdown(
    "### Machine Learning-Based Buyer Segmentation and Investment Profiling"
)

st.write(
    "This dashboard analyzes buyer demographics, property investments, "
    "geographic distribution, financing behavior, and machine-learning "
    "buyer segments."
)


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No data is available for the selected filters. "
        "Please change your filter selections."
    )

    st.stop()


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

    "📊 Overview",
    "💰 Investment Analysis",
    "🏡 Acquisition & Financing",
    "🌍 Geographic Analysis",
    "👥 Demographics",
    "🔍 Segment Insights",
    "📄 Data Explorer"

])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

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


    st.divider()


    # --------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🧩 Buyer Segment Distribution")

    segment_counts = (
        filtered_df
        .groupby("segment_name")
        .size()
        .reset_index(name="buyer_count")
        .sort_values(
            "buyer_count",
            ascending=False
        )
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

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.pie(

            segment_counts,

            names="segment_name",

            values="buyer_count",

            title="Buyer Segment Share"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # --------------------------------------------------------
    # QUICK SEGMENT SUMMARY
    # --------------------------------------------------------

    st.subheader("📌 Segment Summary")

    overview_summary = (
        filtered_df
        .groupby("segment_name")
        .agg(

            Buyers=("client_id", "count"),

            Average_Investment=(
                "total_investment",
                "mean"
            ),

            Average_Properties=(
                "total_properties",
                "mean"
            ),

            Average_Satisfaction=(
                "satisfaction_score",
                "mean"
            )

        )
        .reset_index()
    )


    st.dataframe(

        overview_summary.style.format({

            "Average_Investment": "${:,.2f}",

            "Average_Properties": "{:.2f}",

            "Average_Satisfaction": "{:.2f}"

        }),

        use_container_width=True

    )


# ============================================================
# TAB 2 — INVESTMENT ANALYSIS
# ============================================================

with tab2:

    st.subheader("💰 Investment Behavior by Buyer Segment")


    segment_investment = (

        filtered_df

        .groupby("segment_name")

        .agg(

            buyer_count=("client_id", "count"),

            average_investment=(
                "total_investment",
                "mean"
            ),

            total_investment=(
                "total_investment",
                "sum"
            ),

            average_properties=(
                "total_properties",
                "mean"
            ),

            average_property_price=(
                "average_property_price",
                "mean"
            )

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )


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

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        fig = px.bar(

            segment_investment,

            x="segment_name",

            y="average_property_price",

            text_auto=".2s",

            title="Average Property Price by Segment"

        )

        fig.update_layout(

            xaxis_title="Buyer Segment",

            yaxis_title="Average Property Price"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.bar(

            segment_investment,

            x="segment_name",

            y="total_investment",

            text_auto=".2s",

            title="Total Investment by Segment"

        )

        fig.update_layout(

            xaxis_title="Buyer Segment",

            yaxis_title="Total Investment"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    st.subheader("📋 Investment Statistics")


    st.dataframe(

        segment_investment.style.format({

            "average_investment": "${:,.2f}",

            "total_investment": "${:,.2f}",

            "average_properties": "{:.2f}",

            "average_property_price": "${:,.2f}"

        }),

        use_container_width=True

    )


# ============================================================
# TAB 3 — ACQUISITION & FINANCING
# ============================================================

with tab3:

    st.subheader("🏡 Acquisition and Financing Behavior")


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ACQUISITION PURPOSE
    # --------------------------------------------------------

    with col1:

        purpose_segment = (

            filtered_df

            .groupby([
                "segment_name",
                "acquisition_purpose"
            ])

            .size()

            .reset_index(
                name="buyer_count"
            )

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


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # LOAN APPLICATION
    # --------------------------------------------------------

    with col2:

        loan_segment = (

            filtered_df

            .groupby([
                "segment_name",
                "loan_applied"
            ])

            .size()

            .reset_index(
                name="buyer_count"
            )

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


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # --------------------------------------------------------
    # OVERALL LOAN DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        purpose_overall = (

            filtered_df[
                "acquisition_purpose"
            ]

            .value_counts()

            .reset_index()

        )


        purpose_overall.columns = [

            "acquisition_purpose",

            "buyer_count"

        ]


        fig = px.pie(

            purpose_overall,

            names="acquisition_purpose",

            values="buyer_count",

            title="Overall Acquisition Purpose Distribution"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        loan_overall = (

            filtered_df[
                "loan_applied"
            ]

            .value_counts()

            .reset_index()

        )


        loan_overall.columns = [

            "loan_applied",

            "buyer_count"

        ]


        fig = px.pie(

            loan_overall,

            names="loan_applied",

            values="buyer_count",

            title="Overall Loan Application Distribution"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 4 — GEOGRAPHIC ANALYSIS
# ============================================================

with tab4:

    st.subheader("🌍 Geographic Buyer Analysis")


    country_summary = (

        filtered_df

        .groupby("country")

        .agg(

            buyer_count=(
                "client_id",
                "count"
            ),

            total_investment=(
                "total_investment",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "buyer_count",
            ascending=False
        )

    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # COUNTRY DISTRIBUTION
    # --------------------------------------------------------

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


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

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


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # --------------------------------------------------------
    # REGION ANALYSIS
    # --------------------------------------------------------

    st.subheader("Buyer Distribution by Region and Segment")


    region_segment = (

        filtered_df

        .groupby([
            "region",
            "segment_name"
        ])

        .size()

        .reset_index(
            name="buyer_count"
        )

    )


    top_regions = (

        filtered_df["region"]

        .value_counts()

        .head(15)

        .index

    )


    region_segment = region_segment[

        region_segment["region"]
        .isin(top_regions)

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


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 5 — DEMOGRAPHICS
# ============================================================

with tab5:

    st.subheader("👥 Buyer Demographics")


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

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


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SATISFACTION
    # --------------------------------------------------------

    with col2:

        satisfaction_segment = (

            filtered_df

            .groupby("segment_name")[
                "satisfaction_score"
            ]

            .mean()

            .reset_index()

        )


        fig = px.bar(

            satisfaction_segment,

            x="segment_name",

            y="satisfaction_score",

            text_auto=".2f",

            title="Average Satisfaction Score by Segment"

        )


        fig.update_layout(

            xaxis_title="Buyer Segment",

            yaxis_title="Average Satisfaction Score"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CLIENT TYPE
    # --------------------------------------------------------

    if "client_type" in filtered_df.columns:

        st.divider()

        st.subheader("Client Type Distribution")


        client_type_data = (

            filtered_df

            .groupby([
                "segment_name",
                "client_type"
            ])

            .size()

            .reset_index(
                name="buyer_count"
            )

        )


        fig = px.bar(

            client_type_data,

            x="segment_name",

            y="buyer_count",

            color="client_type",

            barmode="group",

            title="Client Type by Buyer Segment"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# TAB 6 — SEGMENT INSIGHTS
# ============================================================

with tab6:

    st.subheader("🔍 Segment Insights Panel")


    available_segments = sorted(
        filtered_df["segment_name"].unique()
    )


    selected_segment = st.selectbox(

        "Select a Buyer Segment",

        available_segments

    )


    segment_df = filtered_df[

        filtered_df["segment_name"]
        == selected_segment

    ]


    st.markdown(
        f"## {selected_segment}"
    )


    # --------------------------------------------------------
    # SEGMENT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(

        "Buyers",

        f"{len(segment_df):,}"

    )


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


    st.divider()


    # --------------------------------------------------------
    # SEGMENT STATISTICS
    # --------------------------------------------------------

    st.subheader("📊 Detailed Segment Statistics")


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


    st.dataframe(

        segment_stats,

        use_container_width=True

    )


    st.divider()


    # --------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # --------------------------------------------------------

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


    for recommendation in insights.get(
        selected_segment,
        []
    ):

        st.write(
            f"• {recommendation}"
        )


# ============================================================
# TAB 7 — DATA EXPLORER
# ============================================================

with tab7:

    st.subheader("📄 Filtered Buyer Data")


    show_columns = [

        column

        for column in [

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

        height=500

    )


    st.divider()


    # --------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # --------------------------------------------------------

    csv = (

        filtered_df
        .to_csv(index=False)
        .encode("utf-8")

    )


    st.download_button(

        label="⬇ Download Filtered Buyer Data",

        data=csv,

        file_name="filtered_buyer_segmentation.csv",

        mime="text/csv"

    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Unified Mentor Internship Project | "
    "Machine Learning-Based Buyer Segmentation and Investment Profiling"

)