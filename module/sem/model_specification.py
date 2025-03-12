import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model

def validate_model_inputs(observed_vars, latent_vars, model_type):
    """Kiểm tra tính hợp lệ của các tham số đầu vào"""
    errors = []
    
    # Kiểm tra số lượng biến quan sát
    if len(observed_vars) < 3:
        errors.append("Cần ít nhất 3 biến quan sát để tạo mô hình SEM")
    
    # Kiểm tra tên biến tiềm ẩn trùng nhau
    if len(latent_vars) != len(set(latent_vars)):
        errors.append("Tên các biến tiềm ẩn không được trùng nhau")
    
    # Kiểm tra tên biến tiềm ẩn trùng với biến quan sát
    overlap = set(latent_vars) & set(observed_vars)
    if overlap:
        errors.append(f"Các biến sau vừa là biến tiềm ẩn vừa là biến quan sát: {', '.join(overlap)}")
    
    # Kiểm tra điều kiện cho từng loại mô hình
    if model_type == "Path Analysis":
        if len(observed_vars) < 3:
            errors.append("Path Analysis cần ít nhất 3 biến quan sát")
    elif model_type == "Confirmatory Factor Analysis":
        if len(latent_vars) < 1:
            errors.append("CFA cần ít nhất 1 biến tiềm ẩn")
        if len(observed_vars) < 3:
            errors.append("CFA cần ít nhất 3 biến quan sát")
    elif model_type == "Full SEM":
        if len(latent_vars) < 2:
            errors.append("Full SEM cần ít nhất 2 biến tiềm ẩn")
        if len(observed_vars) < 4:
            errors.append("Full SEM cần ít nhất 4 biến quan sát")
    
    return errors

def run_analysis(df):
    st.header("Model Specification")
    
    # Phần cấu hình bên trái
    with st.sidebar:
        st.subheader("Model Settings")
        
        # Chọn biến quan sát
        observed_vars = st.multiselect(
            "Chọn biến quan sát",
            options=df.columns.tolist(),
            default=None
        )
        
        # Định nghĩa biến tiềm ẩn
        st.subheader("Latent Variables")
        num_latent = st.number_input("Số lượng biến tiềm ẩn", min_value=1, value=1)
        latent_vars = []
        for i in range(num_latent):
            latent_name = st.text_input(f"Tên biến tiềm ẩn {i+1}", value=f"F{i+1}")
            latent_vars.append(latent_name)
            
        # Chọn kiểu mô hình
        model_type = st.selectbox(
            "Kiểu mô hình",
            ["Path Analysis", "Confirmatory Factor Analysis", "Full SEM"]
        )
        
        # Cấu hình ước lượng
        st.subheader("Estimation Settings")
        estimator = st.selectbox(
            "Phương pháp ước lượng",
            ["ML", "GLS", "WLS", "DWLS"]
        )
        
        standardized = st.checkbox("Hiển thị kết quả chuẩn hóa", value=True)
    
    # Phần hiển thị chính
    if observed_vars:
        st.subheader("Model Preview")
        
        # Kiểm tra tính hợp lệ của đầu vào
        errors = validate_model_inputs(observed_vars, latent_vars, model_type)
        if errors:
            for error in errors:
                st.error(error)
            return
        
        # Hiển thị ma trận tương quan
        if st.checkbox("Hiển thị ma trận tương quan"):
            corr_matrix = df[observed_vars].corr()
            st.write("Ma trận tương quan:")
            st.dataframe(corr_matrix)
        
        # Tạo mô hình
        try:
            # Tạo cú pháp mô hình dựa trên kiểu đã chọn
            model_syntax = create_model_syntax(
                observed_vars, 
                latent_vars, 
                model_type
            )
            
            st.subheader("Model Syntax")
            st.code(model_syntax)
            
            # Tạo đối tượng Model và lưu vào session state
            model = Model(model_syntax)
            model.df = df  # Thêm thuộc tính df vào model
            st.session_state.model = model
            
            # Hiển thị thông tin mô hình
            st.subheader("Model Information")
            st.write(f"Số biến quan sát: {len(observed_vars)}")
            st.write(f"Số biến tiềm ẩn: {len(latent_vars)}")
            st.write(f"Bậc tự do: {model.df}")
            
            # Lưu các thông tin khác vào session state
            st.session_state.observed_vars = observed_vars
            st.session_state.latent_vars = latent_vars
            st.session_state.model_type = model_type
            st.session_state.estimator = estimator
            st.session_state.standardized = standardized
            
        except Exception as e:
            st.error(f"Lỗi khi tạo mô hình: {str(e)}")
    else:
        st.info("Vui lòng chọn các biến quan sát để bắt đầu.")

def create_model_syntax(observed_vars, latent_vars, model_type):
    """Tạo cú pháp mô hình dựa trên các tham số đầu vào"""
    syntax = ""
    
    if model_type == "Confirmatory Factor Analysis":
        # Tạo cú pháp cho CFA
        for lv in latent_vars:
            syntax += f"# Measurement model for {lv}\n"
            for ov in observed_vars:
                syntax += f"{ov} ~ {lv}\n"
    
    elif model_type == "Path Analysis":
        # Tạo cú pháp cho Path Analysis
        syntax += "# Path model\n"
        for i, v1 in enumerate(observed_vars[:-1]):
            for v2 in observed_vars[i+1:]:
                syntax += f"{v2} ~ {v1}\n"
    
    else:  # Full SEM
        # Tạo cú pháp cho Full SEM
        syntax += "# Measurement model\n"
        for lv in latent_vars:
            for ov in observed_vars:
                syntax += f"{ov} ~ {lv}\n"
        
        syntax += "\n# Structural model\n"
        for i, lv1 in enumerate(latent_vars[:-1]):
            for lv2 in latent_vars[i+1:]:
                syntax += f"{lv2} ~ {lv1}\n"
    
    return syntax 