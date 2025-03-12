import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def run_multinomial_test(data, variable, probabilities=None):
    """Thực hiện kiểm định đa thức"""
    # Đếm tần số các giá trị
    observed = data[variable].value_counts()
    n_categories = len(observed)
    
    # Nếu không có xác suất kỳ vọng, giả định phân phối đều
    if probabilities is None:
        probabilities = np.ones(n_categories) / n_categories
    
    # Thực hiện kiểm định
    chi_stat, p_value = stats.chisquare(
        observed,
        f_exp=probabilities * sum(observed)
    )
    
    return {
        'observed': observed,
        'expected': probabilities * sum(observed),
        'chi_square': chi_stat,
        'p_value': p_value,
        'df': n_categories - 1
    }

def run_analysis(df):
    st.subheader("Multinomial Test")
    
    # Chọn biến phân tích
    categorical_vars = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if len(df[col].unique()) > 2:
            categorical_vars.append(col)
    
    if not categorical_vars:
        st.warning("Không tìm thấy biến phân loại (>2 giá trị) trong dữ liệu.")
        return
    
    selected_var = st.selectbox(
        "Chọn biến phân tích",
        options=categorical_vars
    )
    
    if selected_var:
        # Hiển thị các giá trị và tần số
        value_counts = df[selected_var].value_counts()
        st.write("Phân phối tần số:")
        st.dataframe(pd.DataFrame({
            'Category': value_counts.index,
            'Frequency': value_counts.values,
            'Proportion': value_counts.values / len(df)
        }))
        
        # Cấu hình kiểm định
        with st.sidebar:
            st.subheader("Test Configuration")
            
            use_custom_probs = st.checkbox(
                "Sử dụng xác suất tùy chỉnh",
                value=False
            )
            
            if use_custom_probs:
                st.write("Nhập xác suất kỳ vọng cho mỗi giá trị:")
                probs = {}
                total_prob = 0
                
                for cat in value_counts.index:
                    prob = st.number_input(
                        f"P({cat})",
                        min_value=0.0,
                        max_value=1.0,
                        value=1/len(value_counts),
                        step=0.05
                    )
                    probs[cat] = prob
                    total_prob += prob
                
                if abs(total_prob - 1) > 1e-10:
                    st.error("Tổng xác suất phải bằng 1")
                    return
                
                probabilities = np.array([probs[cat] for cat in value_counts.index])
            else:
                probabilities = None
        
        try:
            # Thực hiện kiểm định
            results = run_multinomial_test(
                df,
                selected_var,
                probabilities
            )
            
            # Hiển thị kết quả
            st.subheader("Test Results")
            
            # Bảng so sánh tần số quan sát và kỳ vọng
            comparison_df = pd.DataFrame({
                'Category': results['observed'].index,
                'Observed': results['observed'].values,
                'Expected': results['expected'],
                'Difference': results['observed'].values - results['expected']
            })
            st.write("So sánh tần số quan sát và kỳ vọng:")
            st.dataframe(comparison_df)
            
            # Thống kê kiểm định
            st.write("\nTest Statistics:")
            st.write(f"- Chi-square: {results['chi_square']:.4f}")
            st.write(f"- Degrees of freedom: {results['df']}")
            st.write(f"- P-value: {results['p_value']:.4f}")
            
            # Kết luận
            alpha = 0.05  # Mức ý nghĩa
            conclusion = "Bác bỏ H0" if results['p_value'] < alpha else "Không bác bỏ H0"
            st.write(f"\nKết luận (α = {alpha}):", conclusion)
            
        except Exception as e:
            st.error(f"Lỗi khi thực hiện kiểm định: {str(e)}") 