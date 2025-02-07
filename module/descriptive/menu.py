import streamlit as st
from . import descriptive_statistics, descriptives_time_series, flexplot, raincloud_plots

def descriptive_analysis(df):
    st.title("Mô tả")

    # Menu phân tích mô tả ở sidebar
    analysis_option = st.sidebar.selectbox(
        "Chọn loại phân tích mô tả",
        ["Descriptives Statistics", "Descriptives Time Series", "Flexplot", "Raincloud Plots"]
    )

    if analysis_option == "Descriptives Statistics":
        # Tạo một instance của class DescriptiveStatistics
        descriptive_stats = descriptive_statistics.DescriptiveStatistics(df)
        descriptive_stats.descriptive_statistics_analysis(df)  # Gọi method từ instance
    elif analysis_option == "Descriptives Time Series":
        descriptives_time_series.time_series_analysis(df)
    elif analysis_option == "Flexplot":
        flexplot.flexplot_analysis(df)
    elif analysis_option == "Raincloud Plots":
        raincloud_plots.raincloud_plots_analysis(df)