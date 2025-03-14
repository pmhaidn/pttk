import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_modification_indices(model, threshold=3.84):
    """Tính toán chỉ số điều chỉnh cho mô hình"""
    try:
        # Lấy chỉ số điều chỉnh từ mô hình
        mi = model.modification_indices()
        
        # Lọc theo ngưỡng
        mi = mi[mi['MI'] >= threshold]
        
        # Sắp xếp theo giá trị MI giảm dần
        mi = mi.sort_values('MI', ascending=False)
        
        return mi
    except Exception as e:
        st.error(f"Lỗi khi tính toán chỉ số điều chỉnh: {str(e)}")
        return None

def plot_modification_indices(mi, top_n=10):
    """Vẽ biểu đồ cho chỉ số điều chỉnh"""
    if mi is None or len(mi) == 0:
        return None
    
    # Lấy top N chỉ số
    mi_plot = mi.head(top_n)
    
    # Tạo nhãn cho trục x
    labels = [f"{row['lval']} - {row['rval']}" for _, row in mi_plot.iterrows()]
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(mi_plot)), mi_plot['MI'])
    
    # Thêm nhãn và giá trị
    ax.set_xticks(range(len(mi_plot)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Modification Index')
    ax.set_title('Top Modification Indices')
    
    # Thêm giá trị lên các cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def suggest_modifications(mi, threshold=3.84):
    """Đề xuất các điều chỉnh cho mô hình"""
    suggestions = []
    
    if mi is None or len(mi) == 0:
        return suggestions
    
    for _, row in mi.iterrows():
        if row['MI'] >= threshold:
            # Tạo đề xuất dựa trên loại mối quan hệ
            if row['op'] == '->':
                suggestion = (
                    f"Thêm đường dẫn từ {row['lval']} đến {row['rval']}\n"
                    f"MI = {row['MI']:.2f}, EPC = {row['EPC']:.2f}"
                )
            elif row['op'] == '<->':
                suggestion = (
                    f"Thêm hiệp phương sai giữa {row['lval']} và {row['rval']}\n"
                    f"MI = {row['MI']:.2f}, EPC = {row['EPC']:.2f}"
                )
            suggestions.append(suggestion)
    
    return suggestions

def run_analysis(df):
    st.header("Modification Indices")
    
    if 'model' not in st.session_state:
        st.error("Vui lòng định nghĩa mô hình trước.")
        return
    
    model = st.session_state.model
    
    # Kiểm tra xem mô hình đã được ước lượng chưa
    try:
        # Thử truy cập một thuộc tính chỉ có sau khi ước lượng
        _ = model.inspect()
    except:
        st.error("Vui lòng ước lượng mô hình trước khi xem chỉ số điều chỉnh.")
        return
    
    # Cấu hình hiển thị
    with st.sidebar:
        st.subheader("Modification Settings")
        
        # Ngưỡng cho chỉ số điều chỉnh
        threshold = st.number_input(
            "Ngưỡng chỉ số điều chỉnh",
            min_value=0.0,
            value=3.84,
            help="Chỉ hiển thị các chỉ số lớn hơn ngưỡng này (mặc định: 3.84 tương ứng với p < 0.05)"
        )
        
        # Số lượng chỉ số hiển thị trong biểu đồ
        top_n = st.number_input(
            "Số lượng chỉ số hiển thị",
            min_value=1,
            max_value=20,
            value=10
        )
        
        # Tùy chọn hiển thị
        show_plot = st.checkbox("Hiển thị biểu đồ", value=True)
        show_suggestions = st.checkbox("Hiển thị đề xuất", value=True)
    
    try:
        # Tính toán chỉ số điều chỉnh
        mi = calculate_modification_indices(model, threshold=threshold)
        
        if mi is not None and len(mi) > 0:
            # Hiển thị bảng chỉ số
            st.subheader("Modification Indices")
            st.dataframe(mi)
            
            # Hiển thị biểu đồ
            if show_plot:
                st.subheader("Visualization")
                fig = plot_modification_indices(mi, top_n=top_n)
                if fig is not None:
                    st.pyplot(fig)
            
            # Hiển thị đề xuất
            if show_suggestions:
                st.subheader("Suggested Modifications")
                suggestions = suggest_modifications(mi, threshold=threshold)
                for i, suggestion in enumerate(suggestions, 1):
                    st.info(f"{i}. {suggestion}")
        else:
            st.info(f"Không tìm thấy chỉ số điều chỉnh nào lớn hơn {threshold}")
        
    except Exception as e:
        st.error(f"Lỗi khi phân tích chỉ số điều chỉnh: {str(e)}") 