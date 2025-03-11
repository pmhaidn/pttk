import streamlit as st
from . import classical_meta_analysis
from . import bayesian_meta_analysis
from . import forest_plot
from . import funnel_plot
from . import selection_models
from . import effect_size

def meta_analysis(df):
    st.title("Meta Analysis")
    
    # Chọn phương pháp phân tích
    analysis_type = st.sidebar.selectbox(
        "Chọn phương pháp phân tích",
        ["Classical Meta Analysis", 
         "Bayesian Meta Analysis",
         "Effect Size",
         "Forest Plot",
         "Funnel Plot",
         "Selection Models"]
    )
    
    # Thực hiện phân tích tương ứng
    if analysis_type == "Classical Meta Analysis":
        classical_meta_analysis.run_analysis(df)
    elif analysis_type == "Bayesian Meta Analysis":
        bayesian_meta_analysis.run_analysis(df)
    elif analysis_type == "Effect Size":
        effect_size.run_analysis(df)
    elif analysis_type == "Forest Plot":
        forest_plot.run_analysis(df)
    elif analysis_type == "Funnel Plot":
        funnel_plot.run_analysis(df)
    else:  # Selection Models
        selection_models.run_analysis(df) 