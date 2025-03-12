import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_descriptives(data, variables):
    """Tính toán các thống kê mô tả cho biến phân loại"""
    results = []
    for var in variables:
        freq = data[var].value_counts()
        prop = data[var].value_counts(normalize=True)
        valid_n = data[var].count()
        missing = data[var].isnull().sum()
        
        for cat in freq.index:
            results.append({
                'Variable': var,
                'Category': cat,
                'Frequency': freq[cat],
                'Percent': prop[cat] * 100,
                'Valid Percent': (freq[cat] / valid_n) * 100,
                'Cumulative Percent': prop[cat].cumsum() * 100
            })
    
    return pd.DataFrame(results)

def plot_frequencies(data, variable, plot_type='bar'):
    """Vẽ biểu đồ tần suất"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'bar':
        sns.countplot(data=data, x=variable)
    elif plot_type == 'pie':
        data[variable].value_counts().plot(kind='pie', autopct='%1.1f%%')
    
    plt.title(f'Frequency Distribution of {variable}')
    return plt.gcf()

def run_analysis(df):
    st.subheader("Frequency Descriptives")
    
    # Chọn biến phân tích
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_vars) == 0:
        st.warning("Không tìm thấy biến phân loại trong dữ liệu.")
        return
        
    selected_vars = st.multiselect(
        "Chọn biến phân tích",
        options=categorical_vars
    )
    
    if not selected_vars:
        st.info("Vui lòng chọn ít nhất một biến để phân tích.")
        return
    
    # Cấu hình hiển thị
    with st.sidebar:
        st.subheader("Display Options")
        
        show_frequencies = st.checkbox("Hiển thị bảng tần số", value=True)
        show_percentages = st.checkbox("Hiển thị phần trăm", value=True)
        show_missing = st.checkbox("Hiển thị missing values", value=True)
        
        plot_options = st.multiselect(
            "Chọn loại biểu đồ",
            options=["Bar Chart", "Pie Chart"],
            default=["Bar Chart"]
        )
    
    # Tính toán và hiển thị kết quả
    try:
        results = calculate_descriptives(df, selected_vars)
        
        if show_frequencies:
            st.subheader("Frequency Tables")
            for var in selected_vars:
                st.write(f"\nFrequency table for {var}:")
                var_results = results[results['Variable'] == var]
                
                display_cols = ['Category', 'Frequency']
                if show_percentages:
                    display_cols.extend(['Percent', 'Valid Percent', 'Cumulative Percent'])
                
                st.dataframe(var_results[display_cols])
                
                if show_missing:
                    missing = df[var].isnull().sum()
                    st.write(f"Missing values: {missing} ({(missing/len(df))*100:.1f}%)")
        
        # Vẽ biểu đồ
        if plot_options:
            st.subheader("Visualizations")
            for var in selected_vars:
                if "Bar Chart" in plot_options:
                    st.write(f"\nBar chart for {var}:")
                    fig = plot_frequencies(df, var, 'bar')
                    st.pyplot(fig)
                    plt.close()
                
                if "Pie Chart" in plot_options:
                    st.write(f"\nPie chart for {var}:")
                    fig = plot_frequencies(df, var, 'pie')
                    st.pyplot(fig)
                    plt.close()
    
    except Exception as e:
        st.error(f"Lỗi khi phân tích: {str(e)}") 