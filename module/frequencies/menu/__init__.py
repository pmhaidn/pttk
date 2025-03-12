import streamlit as st
import pandas as pd

def frequencies_analysis(df):
    st.header("Frequencies Analysis")
    
    with st.sidebar:
        analysis_type = st.selectbox(
            "Chọn loại phân tích",
            ["Descriptives", "Binomial Test", "Multinomial Test", "Contingency Tables"]
        )
        
        if analysis_type == "Descriptives":
            from ..descriptives import run_analysis
            run_analysis(df)
            
        elif analysis_type == "Binomial Test":
            from ..binomial_test import run_analysis
            run_analysis(df)
            
        elif analysis_type == "Multinomial Test":
            from ..multinomial_test import run_analysis
            run_analysis(df)
            
        elif analysis_type == "Contingency Tables":
            from ..contingency_tables import run_analysis
            run_analysis(df) 