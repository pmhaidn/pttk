import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_contingency_stats(data, var1, var2):
    """Tính toán các thống kê cho bảng chéo"""
    # Tạo bảng chéo
    contingency_table = pd.crosstab(data[var1], data[var2])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    return {
        'table': contingency_table,
        'expected': expected,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramer_v': cramer_v
    }

def plot_contingency(data, var1, var2, plot_type='heatmap'):
    """Vẽ biểu đồ cho bảng chéo"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'heatmap':
        # Tạo bảng chéo với tỉ lệ phần trăm
        cont_table = pd.crosstab(
            data[var1], 
            data[var2], 
            normalize='all'
        ) * 100
        
        # Vẽ heatmap
        sns.heatmap(
            cont_table,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title(f'Contingency Table Heatmap: {var1} vs {var2}')
        
    elif plot_type == 'mosaic':
        # Tạo bảng chéo
        cont_table = pd.crosstab(data[var1], data[var2])
        
        # Vẽ mosaic plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Tính toán vị trí và kích thước cho các ô
        total = cont_table.sum().sum()
        y_labels = cont_table.index
        x_labels = cont_table.columns
        
        y_positions = np.zeros(len(y_labels))
        heights = cont_table.sum(axis=1) / total
        
        for i, (y_label, row) in enumerate(cont_table.iterrows()):
            x_position = 0
            row_sum = row.sum()
            
            for j, (x_label, value) in enumerate(row.items()):
                width = value / row_sum * heights[i]
                rect = plt.Rectangle(
                    (x_position, y_positions[i]),
                    width,
                    heights[i],
                    facecolor=plt.cm.Set3(j / len(x_labels)),
                    edgecolor='black'
                )
                ax.add_patch(rect)
                
                # Thêm nhãn nếu ô đủ lớn
                if width * heights[i] > 0.05:
                    ax.text(
                        x_position + width/2,
                        y_positions[i] + heights[i]/2,
                        f'{value}\n({value/total*100:.1f}%)',
                        ha='center',
                        va='center'
                    )
                
                x_position += width
            
            y_positions[i+1:] += heights[i]
        
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title(f'Mosaic Plot: {var1} vs {var2}')
        
    return plt.gcf()

def run_analysis(df):
    st.subheader("Contingency Tables")
    
    # Chọn biến phân tích
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_vars) < 2:
        st.warning("Cần ít nhất 2 biến phân loại để tạo bảng chéo.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox(
            "Chọn biến hàng",
            options=categorical_vars,
            key='var1'
        )
    
    with col2:
        var2 = st.selectbox(
            "Chọn biến cột",
            options=[col for col in categorical_vars if col != var1],
            key='var2'
        )
    
    if var1 and var2:
        # Cấu hình hiển thị
        with st.sidebar:
            st.subheader("Display Options")
            
            show_percentages = st.checkbox(
                "Hiển thị phần trăm",
                value=True
            )
            
            percentage_type = st.selectbox(
                "Loại phần trăm",
                options=['row', 'column', 'total'],
                format_func=lambda x: {
                    'row': 'Theo hàng',
                    'column': 'Theo cột',
                    'total': 'Tổng'
                }[x]
            ) if show_percentages else None
            
            plot_type = st.selectbox(
                "Loại biểu đồ",
                options=['heatmap', 'mosaic'],
                format_func=lambda x: x.title()
            )
        
        try:
            # Tính toán kết quả
            results = calculate_contingency_stats(df, var1, var2)
            
            # Hiển thị bảng chéo
            st.subheader("Contingency Table")
            
            if show_percentages:
                # Tính toán phần trăm
                if percentage_type == 'row':
                    table = pd.crosstab(
                        df[var1], df[var2],
                        normalize='index'
                    ) * 100
                    st.write("Phần trăm theo hàng:")
                elif percentage_type == 'column':
                    table = pd.crosstab(
                        df[var1], df[var2],
                        normalize='columns'
                    ) * 100
                    st.write("Phần trăm theo cột:")
                else:
                    table = pd.crosstab(
                        df[var1], df[var2],
                        normalize='all'
                    ) * 100
                    st.write("Phần trăm tổng:")
                
                st.dataframe(table.round(1))
            
            # Hiển thị bảng tần số
            st.write("Tần số quan sát:")
            st.dataframe(results['table'])
            
            # Hiển thị kết quả kiểm định
            st.subheader("Chi-square Test Results")
            st.write(f"- Chi-square statistic: {results['chi2']:.4f}")
            st.write(f"- Degrees of freedom: {results['dof']}")
            st.write(f"- P-value: {results['p_value']:.4f}")
            st.write(f"- Cramer's V: {results['cramer_v']:.4f}")
            
            # Kết luận
            alpha = 0.05  # Mức ý nghĩa
            conclusion = "Bác bỏ H0" if results['p_value'] < alpha else "Không bác bỏ H0"
            st.write(f"\nKết luận (α = {alpha}):", conclusion)
            
            # Vẽ biểu đồ
            st.subheader("Visualization")
            fig = plot_contingency(df, var1, var2, plot_type)
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Lỗi khi phân tích: {str(e)}") 