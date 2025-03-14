import streamlit as st
from .. import model_specification
from .. import model_estimation
from .. import model_fit
from .. import modification_indices
from .. import path_diagram
from .. import factor_analysis

def sem_analysis(df):
    st.title("Structural Equation Modeling")
    
    # Chọn phương pháp phân tích
    analysis_type = st.sidebar.selectbox(
        "Chọn phương pháp phân tích",
        ["Model Specification", 
         "Model Estimation",
         "Model Fit Indices",
         "Modification Indices",
         "Path Diagram",
         "Factor Analysis"]
    )
    
    # Thực hiện phân tích tương ứng
    if analysis_type == "Model Specification":
        model_specification.run_analysis(df)
    elif analysis_type == "Model Estimation":
        model_estimation.run_analysis(df)
    elif analysis_type == "Model Fit Indices":
        model_fit.run_analysis(df)
    elif analysis_type == "Modification Indices":
        modification_indices.run_analysis(df)
    elif analysis_type == "Path Diagram":
        path_diagram.run_analysis(df)
    else:  # Factor Analysis
        factor_analysis.run_analysis(df) 