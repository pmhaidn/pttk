import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model
import matplotlib.pyplot as plt
import seaborn as sns

def validate_data(df, observed_vars):
    """Kiểm tra tính hợp lệ của dữ liệu"""
    errors = []
    
    # Kiểm tra giá trị thiếu
    missing = df[observed_vars].isnull().sum()
    if missing.any():
        errors.append(f"Có giá trị thiếu trong các biến: {', '.join(missing[missing > 0].index)}")
    
    # Kiểm tra phương sai bằng 0
    zero_var = df[observed_vars].var() == 0
    if zero_var.any():
        errors.append(f"Các biến sau có phương sai bằng 0: {', '.join(zero_var[zero_var].index)}")
    
    # Kiểm tra ma trận tương quan
    corr = df[observed_vars].corr()
    if not np.all(np.linalg.eigvals(corr) > 0):
        errors.append("Ma trận tương quan không xác định dương")
    
    return errors

def run_analysis(df):
    st.header("Model Estimation")
    
    # Phần cấu hình bên trái
    with st.sidebar:
        st.subheader("Estimation Settings")
        
        # Chọn phương pháp ước lượng
        estimator = st.selectbox(
            "Phương pháp ước lượng",
            ["ML", "GLS", "WLS", "DWLS"]
        )
        
        # Cấu hình hiển thị
        show_standardized = st.checkbox("Hiển thị hệ số chuẩn hóa", value=True)
        show_ci = st.checkbox("Hiển thị khoảng tin cậy", value=True)
        ci_level = st.slider("Mức ý nghĩa", 0.8, 0.99, 0.95, 0.01)
        
        # Cấu hình bootstrap
        use_bootstrap = st.checkbox("Sử dụng Bootstrap", value=False)
        if use_bootstrap:
            n_bootstrap = st.number_input(
                "Số lượng bootstrap samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
    
    # Phần hiển thị chính
    if 'model' in st.session_state and 'observed_vars' in st.session_state:
        model = st.session_state.model
        observed_vars = st.session_state.observed_vars
        
        # Kiểm tra tính hợp lệ của dữ liệu
        errors = validate_data(model.df, observed_vars)
        if errors:
            for error in errors:
                st.error(error)
            return
        
        try:
            # Ước lượng mô hình
            results = estimate_model(
                model, 
                estimator=estimator,
                standardized=show_standardized,
                bootstrap=use_bootstrap,
                n_bootstrap=n_bootstrap if use_bootstrap else None,
                ci_level=ci_level if show_ci else None
            )
            
            if results is None:
                st.error("Mô hình không hội tụ. Vui lòng thử phương pháp ước lượng khác hoặc điều chỉnh mô hình.")
                return
            
            # Hiển thị kết quả
            st.subheader("Parameter Estimates")
            st.dataframe(results['parameters'])
            
            if show_standardized:
                st.subheader("Standardized Estimates")
                st.dataframe(results['standardized'])
            
            # Hiển thị biểu đồ
            st.subheader("Parameter Plot")
            fig = plot_parameters(results['parameters'])
            st.pyplot(fig)
            
            # Hiển thị ma trận tương quan dư
            st.subheader("Residual Correlation Matrix")
            fig = plot_residuals(results['residuals'])
            st.pyplot(fig)
            
            # Lưu kết quả vào session state
            st.session_state.model_results = results
            
        except Exception as e:
            st.error(f"Lỗi khi ước lượng mô hình: {str(e)}")
    else:
        st.info("Vui lòng định nghĩa mô hình trước khi ước lượng.")

def estimate_model(model, estimator='ML', standardized=True, 
                  bootstrap=False, n_bootstrap=1000, ci_level=0.95):
    """Ước lượng mô hình SEM và trả về kết quả"""
    try:
        # Fit mô hình với dữ liệu từ thuộc tính df
        converged = model.fit(model.df, estimator=estimator)
        
        if not converged:
            return None
        
        # Lấy kết quả ước lượng
        params = model.inspect()
        
        if standardized:
            std_params = model.inspect(standardized=True)
        
        if bootstrap:
            try:
                # Thực hiện bootstrap
                boot_results = model.bootstrap(model.df, n_bootstrap)
                
                # Tính khoảng tin cậy
                alpha = 1 - ci_level
                ci_lower = np.percentile(boot_results, alpha/2 * 100, axis=0)
                ci_upper = np.percentile(boot_results, (1-alpha/2) * 100, axis=0)
                
                params['CI_lower'] = ci_lower
                params['CI_upper'] = ci_upper
            except Exception as e:
                st.warning(f"Lỗi khi thực hiện bootstrap: {str(e)}")
        
        # Lấy ma trận tương quan dư
        residuals = model.residuals()
        
        return {
            'parameters': params,
            'standardized': std_params if standardized else None,
            'residuals': residuals
        }
    except Exception as e:
        st.error(f"Lỗi trong quá trình ước lượng: {str(e)}")
        return None

def plot_parameters(params):
    """Vẽ biểu đồ ước lượng tham số"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vẽ forest plot cho các tham số
    y_pos = np.arange(len(params))
    
    ax.errorbar(
        params['Estimate'],
        y_pos,
        xerr=params['Std. Err'],
        fmt='o',
        capsize=5
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index)
    ax.set_xlabel('Estimate')
    ax.set_title('Parameter Estimates with Standard Errors')
    
    plt.tight_layout()
    return fig

def plot_residuals(residuals):
    """Vẽ biểu đồ ma trận tương quan dư"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        residuals,
        annot=True,
        cmap='RdBu_r',
        center=0,
        ax=ax
    )
    
    ax.set_title('Residual Correlation Matrix')
    plt.tight_layout()
    return fig 