import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def run_binomial_test(data, variable, test_value=0.5, alternative='two-sided'):
    """Thực hiện kiểm định nhị thức"""
    # Đếm tần số các giá trị
    value_counts = data[variable].value_counts()
    
    if len(value_counts) != 2:
        raise ValueError("Biến phải có đúng 2 giá trị khác nhau cho kiểm định nhị thức")
    
    # Lấy số quan sát của giá trị đầu tiên
    n_success = value_counts.iloc[0]
    n_total = len(data[variable].dropna())
    
    # Thực hiện kiểm định
    result = stats.binomtest(n_success, n_total, test_value, alternative=alternative)
    
    return {
        'n_success': n_success,
        'n_total': n_total,
        'observed_prop': n_success / n_total,
        'test_value': test_value,
        'p_value': result.pvalue,
        'alternative': alternative
    }

def run_analysis(df):
    st.subheader("Binomial Test")
    
    # Chọn biến phân tích
    binary_vars = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if len(df[col].unique()) == 2:
            binary_vars.append(col)
    
    if not binary_vars:
        st.warning("Không tìm thấy biến nhị phân trong dữ liệu.")
        return
    
    selected_var = st.selectbox(
        "Chọn biến phân tích",
        options=binary_vars
    )
    
    if selected_var:
        # Hiển thị giá trị trong biến
        unique_values = df[selected_var].unique()
        st.write("Các giá trị trong biến:", unique_values)
        
        # Cấu hình kiểm định
        with st.sidebar:
            st.subheader("Test Configuration")
            
            test_value = st.slider(
                "Tỉ lệ kiểm định",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            alternative = st.selectbox(
                "Giả thuyết thay thế",
                options=['two-sided', 'greater', 'less'],
                format_func=lambda x: {
                    'two-sided': 'Two-sided',
                    'greater': 'Greater than',
                    'less': 'Less than'
                }[x]
            )
        
        try:
            # Thực hiện kiểm định
            results = run_binomial_test(
                df, 
                selected_var,
                test_value=test_value,
                alternative=alternative
            )
            
            # Hiển thị kết quả
            st.subheader("Test Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Descriptives:")
                st.write(f"- Total observations: {results['n_total']}")
                st.write(f"- Successes: {results['n_success']}")
                st.write(f"- Observed proportion: {results['observed_prop']:.3f}")
            
            with col2:
                st.write("Test Statistics:")
                st.write(f"- Test value: {results['test_value']}")
                st.write(f"- Alternative: {results['alternative']}")
                st.write(f"- P-value: {results['p_value']:.4f}")
            
            # Kết luận
            alpha = 0.05  # Mức ý nghĩa
            conclusion = "Bác bỏ H0" if results['p_value'] < alpha else "Không bác bỏ H0"
            st.write(f"\nKết luận (α = {alpha}):", conclusion)
            
        except Exception as e:
            st.error(f"Lỗi khi thực hiện kiểm định: {str(e)}") 