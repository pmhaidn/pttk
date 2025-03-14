import streamlit as st
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns

def check_data_suitability(df, variables):
    """Kiểm tra tính phù hợp của dữ liệu cho phân tích nhân tố"""
    results = {}
    warnings = []
    
    # Kiểm tra kích thước mẫu
    n_samples = len(df)
    n_vars = len(variables)
    if n_samples < n_vars * 5:
        warnings.append(f"Kích thước mẫu ({n_samples}) có thể quá nhỏ cho {n_vars} biến")
    
    # Kiểm tra giá trị thiếu
    missing = df[variables].isnull().sum()
    if missing.any():
        warnings.append(f"Có giá trị thiếu trong các biến: {', '.join(missing[missing > 0].index)}")
    
    # Kiểm tra Bartlett's test
    chi_square, p_value = calculate_bartlett_sphericity(df[variables])
    results['bartlett'] = {
        'chi_square': chi_square,
        'p_value': p_value
    }
    if p_value >= 0.05:
        warnings.append("Bartlett's test không có ý nghĩa thống kê (p >= 0.05)")
    
    # Kiểm tra KMO
    kmo_all, kmo_model = calculate_kmo(df[variables])
    results['kmo'] = {
        'overall': kmo_model,
        'individual': kmo_all
    }
    if kmo_model < 0.6:
        warnings.append(f"Chỉ số KMO thấp ({kmo_model:.3f} < 0.6)")
    
    # Kiểm tra tương quan
    corr = df[variables].corr()
    if not (corr.abs() > 0.3).any().any():
        warnings.append("Không tìm thấy tương quan đủ mạnh giữa các biến")
    
    return results, warnings

def run_efa(df, variables, n_factors, rotation='varimax'):
    """Thực hiện phân tích nhân tố khám phá"""
    # Khởi tạo và thực hiện EFA
    fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
    fa.fit(df[variables])
    
    # Tính toán các chỉ số
    loadings = pd.DataFrame(
        fa.loadings_,
        columns=[f'Factor{i+1}' for i in range(n_factors)],
        index=variables
    )
    
    # Tính phương sai giải thích
    variance = pd.DataFrame({
        'SS Loadings': fa.get_factor_variance()[0],
        'Proportion Var': fa.get_factor_variance()[1],
        'Cumulative Var': fa.get_factor_variance()[2]
    }, index=[f'Factor{i+1}' for i in range(n_factors)])
    
    # Tính điểm số nhân tố
    scores = pd.DataFrame(
        fa.transform(df[variables]),
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )
    
    # Tính communalities
    communalities = pd.Series(fa.get_communalities(), index=variables)
    
    return {
        'loadings': loadings,
        'variance': variance,
        'scores': scores,
        'communalities': communalities,
        'model': fa
    }

def plot_scree(eigenvalues):
    """Vẽ đồ thị scree"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(1, len(eigenvalues) + 1)
    ax.plot(x, eigenvalues, 'bo-')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Factor Number')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree Plot')
    
    plt.tight_layout()
    return fig

def plot_loadings_heatmap(loadings):
    """Vẽ heatmap cho ma trận hệ số tải"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(loadings, 
                annot=True, 
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                ax=ax)
    
    ax.set_title('Factor Loadings Heatmap')
    plt.tight_layout()
    return fig

def run_analysis(df):
    st.header("Factor Analysis")
    
    # Phần cấu hình bên trái
    with st.sidebar:
        st.subheader("Analysis Settings")
        
        # Chọn biến
        variables = st.multiselect(
            "Chọn biến phân tích",
            options=df.columns.tolist(),
            default=None
        )
        
        if variables:
            # Kiểm tra tính phù hợp của dữ liệu
            results, warnings = check_data_suitability(df, variables)
            
            # Hiển thị cảnh báo nếu có
            if warnings:
                st.warning("Cảnh báo về dữ liệu:")
                for warning in warnings:
                    st.write(f"- {warning}")
            
            # Hiển thị kết quả kiểm tra
            st.subheader("Data Suitability")
            st.write(f"KMO = {results['kmo']['overall']:.3f}")
            st.write(f"Bartlett's test p-value = {results['bartlett']['p_value']:.3f}")
            
            # Cấu hình phân tích
            n_factors = st.number_input(
                "Số lượng nhân tố",
                min_value=1,
                max_value=len(variables),
                value=min(3, len(variables))
            )
            
            rotation = st.selectbox(
                "Phương pháp xoay",
                ["varimax", "promax", "oblimin", "quartimax"]
            )
            
            # Tùy chọn hiển thị
            show_loadings = st.checkbox("Hiển thị ma trận hệ số tải", value=True)
            show_variance = st.checkbox("Hiển thị phương sai giải thích", value=True)
            show_scree = st.checkbox("Hiển thị đồ thị scree", value=True)
            show_scores = st.checkbox("Hiển thị điểm số nhân tố", value=False)
    
    # Phần hiển thị chính
    if variables:
        try:
            # Thực hiện EFA
            efa_results = run_efa(df, variables, n_factors, rotation)
            
            # Hiển thị kết quả
            if show_loadings:
                st.subheader("Factor Loadings")
                st.dataframe(efa_results['loadings'])
                
                # Vẽ heatmap
                st.subheader("Factor Loadings Heatmap")
                fig = plot_loadings_heatmap(efa_results['loadings'])
                st.pyplot(fig)
            
            if show_variance:
                st.subheader("Explained Variance")
                st.dataframe(efa_results['variance'])
                
                # Hiển thị communalities
                st.subheader("Communalities")
                st.dataframe(pd.DataFrame({
                    'Communality': efa_results['communalities']
                }))
            
            if show_scree:
                st.subheader("Scree Plot")
                eigenvalues = efa_results['model'].get_eigenvalues()[0]
                fig = plot_scree(eigenvalues)
                st.pyplot(fig)
            
            if show_scores:
                st.subheader("Factor Scores")
                st.dataframe(efa_results['scores'])
            
            # Lưu kết quả vào session state
            st.session_state.efa_results = efa_results
            
        except Exception as e:
            st.error(f"Lỗi khi thực hiện phân tích nhân tố: {str(e)}")
    else:
        st.info("Vui lòng chọn các biến để bắt đầu phân tích.") 