import streamlit as st
import pandas as pd

def ttest_analysis(df):
    st.header("Kiểm định T")
    
    # Phần cấu hình trong sidebar
    with st.sidebar:
        st.subheader("Cấu hình phân tích")
        test_type = st.selectbox(
            "Chọn loại kiểm định T",
            ["One Sample T-Test", 
             "Independent Samples T-Test",
             "Paired Samples T-Test"],
            format_func=lambda x: {
                "One Sample T-Test": "Kiểm định T một mẫu",
                "Independent Samples T-Test": "Kiểm định T hai mẫu độc lập",
                "Paired Samples T-Test": "Kiểm định T hai mẫu ghép cặp"
            }[x]
        )
    
    # Phần kết quả phân tích trong màn hình chính
    if test_type == "One Sample T-Test":
        from ..one_sample import run_analysis
        run_analysis(df)
        
    elif test_type == "Independent Samples T-Test":
        from ..independent_samples import run_analysis
        run_analysis(df)
        
    elif test_type == "Paired Samples T-Test":
        from ..paired_samples import run_analysis
        run_analysis(df) 