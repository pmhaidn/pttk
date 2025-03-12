import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_fit_indices(model):
    """Tính toán các chỉ số fit của mô hình"""
    fit_indices = {}
    
    # Chỉ số Chi-square và p-value
    fit_indices['Chi-square'] = model.chi2
    fit_indices['df'] = model.df
    fit_indices['p-value'] = model.pvalue
    
    # Chỉ số tuyệt đối
    fit_indices['GFI'] = model.gfi
    fit_indices['AGFI'] = model.agfi
    fit_indices['SRMR'] = model.srmr
    
    # Chỉ số tương đối
    fit_indices['CFI'] = model.cfi
    fit_indices['TLI'] = model.tli
    fit_indices['NFI'] = model.nfi
    
    # Chỉ số tiết kiệm
    fit_indices['RMSEA'] = model.rmsea
    fit_indices['AIC'] = model.aic
    fit_indices['BIC'] = model.bic
    
    return fit_indices

def get_fit_thresholds():
    """Trả về ngưỡng đánh giá cho các chỉ số fit"""
    return {
        'Chi-square p-value': '> 0.05',
        'GFI': '≥ 0.95',
        'AGFI': '≥ 0.90',
        'SRMR': '≤ 0.08',
        'CFI': '≥ 0.95',
        'TLI': '≥ 0.95',
        'NFI': '≥ 0.95',
        'RMSEA': '≤ 0.06'
    }

def evaluate_fit(fit_indices):
    """Đánh giá mức độ phù hợp của mô hình"""
    evaluation = {}
    thresholds = get_fit_thresholds()
    
    # Đánh giá từng chỉ số
    evaluation['Chi-square p-value'] = 'Tốt' if fit_indices['p-value'] > 0.05 else 'Chưa tốt'
    evaluation['GFI'] = 'Tốt' if fit_indices['GFI'] >= 0.95 else 'Chưa tốt'
    evaluation['AGFI'] = 'Tốt' if fit_indices['AGFI'] >= 0.90 else 'Chưa tốt'
    evaluation['SRMR'] = 'Tốt' if fit_indices['SRMR'] <= 0.08 else 'Chưa tốt'
    evaluation['CFI'] = 'Tốt' if fit_indices['CFI'] >= 0.95 else 'Chưa tốt'
    evaluation['TLI'] = 'Tốt' if fit_indices['TLI'] >= 0.95 else 'Chưa tốt'
    evaluation['NFI'] = 'Tốt' if fit_indices['NFI'] >= 0.95 else 'Chưa tốt'
    evaluation['RMSEA'] = 'Tốt' if fit_indices['RMSEA'] <= 0.06 else 'Chưa tốt'
    
    return evaluation

def plot_fit_indices(fit_indices, evaluation):
    """Vẽ biểu đồ đánh giá độ phù hợp"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = ['GFI', 'AGFI', 'CFI', 'TLI', 'NFI']
    values = [fit_indices[idx] for idx in indices]
    colors = ['green' if evaluation[idx] == 'Tốt' else 'red' for idx in indices]
    
    bars = ax.bar(indices, values, color=colors)
    
    # Thêm đường ngưỡng 0.95
    ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.5)
    
    ax.set_ylim(0, 1)
    ax.set_title('Model Fit Indices')
    
    # Thêm giá trị lên các cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def run_analysis(df):
    st.header("Model Fit Assessment")
    
    if 'model' not in st.session_state:
        st.error("Vui lòng định nghĩa và ước lượng mô hình trước.")
        return
    
    model = st.session_state.model
    
    # Kiểm tra xem mô hình đã được ước lượng chưa
    try:
        # Thử truy cập một thuộc tính chỉ có sau khi ước lượng
        _ = model.chi2
    except:
        st.error("Vui lòng ước lượng mô hình trước khi đánh giá độ phù hợp.")
        return
    
    try:
        # Tính các chỉ số fit
        fit_indices = calculate_fit_indices(model)
        evaluation = evaluate_fit(fit_indices)
        thresholds = get_fit_thresholds()
        
        # Hiển thị tổng quan
        st.subheader("Tổng quan độ phù hợp của mô hình")
        
        # Tạo bảng kết quả
        results_df = pd.DataFrame({
            'Chỉ số': list(fit_indices.keys()),
            'Giá trị': list(fit_indices.values()),
            'Ngưỡng': [thresholds.get(idx, 'N/A') for idx in fit_indices.keys()],
            'Đánh giá': [evaluation.get(idx, 'N/A') for idx in fit_indices.keys()]
        })
        
        st.dataframe(results_df)
        
        # Vẽ biểu đồ
        st.subheader("Biểu đồ độ phù hợp")
        fig = plot_fit_indices(fit_indices, evaluation)
        st.pyplot(fig)
        
        # Đề xuất cải thiện
        st.subheader("Đề xuất cải thiện")
        poor_fit = [idx for idx, eval in evaluation.items() if eval == 'Chưa tốt']
        if poor_fit:
            st.warning("Các chỉ số sau cần cải thiện:")
            for idx in poor_fit:
                st.write(f"- {idx}: Giá trị hiện tại = {fit_indices.get(idx, 'N/A')}, "
                        f"Ngưỡng mong muốn = {thresholds.get(idx, 'N/A')}")
        else:
            st.success("Mô hình có độ phù hợp tốt!")
        
    except Exception as e:
        st.error(f"Lỗi khi đánh giá độ phù hợp của mô hình: {str(e)}") 