import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize

class SelectionModels:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Selection Models")
        
        # Cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình dữ liệu")
            
            # Chọn cột dữ liệu
            self.effect_col = st.selectbox(
                "Chọn cột Effect Size",
                self.data.columns,
                key="selection_effect_col"
            )
            
            self.se_col = st.selectbox(
                "Chọn cột Standard Error",
                self.data.columns,
                key="selection_se_col"
            )
            
            # Chọn loại mô hình
            self.model_type = st.selectbox(
                "Chọn loại mô hình",
                ["Weight Function", "Step Function"],
                key="selection_model_type"
            )
            
            if self.model_type == "Weight Function":
                self.weight_function = st.selectbox(
                    "Chọn hàm trọng số",
                    ["Half Normal", "Negative Exponential", "Logistic"],
                    key="weight_function"
                )
            else:  # Step Function
                self.num_steps = st.number_input(
                    "Số bước",
                    min_value=2,
                    max_value=5,
                    value=3,
                    key="num_steps"
                )
            
            st.write("### Tùy chỉnh biểu đồ")
            
            # Kích thước biểu đồ
            col1, col2 = st.columns(2)
            with col1:
                self.plot_width = st.number_input(
                    "Chiều rộng",
                    min_value=6,
                    max_value=20,
                    value=10,
                    key="selection_width"
                )
            
            with col2:
                self.plot_height = st.number_input(
                    "Chiều cao",
                    min_value=6,
                    max_value=20,
                    value=8,
                    key="selection_height"
                )
        
        try:
            # Thực hiện phân tích
            self._perform_analysis()
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
    
    def _weight_function_half_normal(self, p_values, params):
        """Hàm trọng số Half Normal"""
        return np.exp(-params[0] * p_values**2)
    
    def _weight_function_negative_exp(self, p_values, params):
        """Hàm trọng số Negative Exponential"""
        return np.exp(-params[0] * p_values)
    
    def _weight_function_logistic(self, p_values, params):
        """Hàm trọng số Logistic"""
        return 1 / (1 + np.exp(params[0] * (p_values - params[1])))
    
    def _step_function(self, p_values, cutoffs, weights):
        """Hàm bước với nhiều mức"""
        result = np.ones_like(p_values)
        for i in range(len(cutoffs)):
            if i == 0:
                mask = p_values <= cutoffs[i]
            else:
                mask = (p_values > cutoffs[i-1]) & (p_values <= cutoffs[i])
            result[mask] = weights[i]
        result[p_values > cutoffs[-1]] = weights[-1]
        return result
    
    def _negative_log_likelihood(self, params, p_values, weight_func):
        """Tính negative log-likelihood"""
        weights = weight_func(p_values, params)
        return -np.sum(np.log(weights))
    
    def _perform_analysis(self):
        # Lấy dữ liệu
        effects = self.data[self.effect_col].values
        ses = self.data[self.se_col].values
        
        # Tính p-values hai phía
        z_scores = effects / ses
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        if self.model_type == "Weight Function":
            # Ước lượng tham số cho weight function
            if self.weight_function == "Half Normal":
                initial_params = [1.0]
                weight_func = self._weight_function_half_normal
            elif self.weight_function == "Negative Exponential":
                initial_params = [1.0]
                weight_func = self._weight_function_negative_exp
            else:  # Logistic
                initial_params = [1.0, 0.05]
                weight_func = self._weight_function_logistic
            
            # Tối ưu hóa
            result = minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(p_values, weight_func),
                method='Nelder-Mead'
            )
            
            # Hiển thị kết quả
            st.write("### Kết quả Weight Function Model")
            st.write(f"Log-likelihood: {-result.fun:.4f}")
            for i, param in enumerate(result.x):
                st.write(f"Parameter {i+1}: {param:.4f}")
            
            # Vẽ weight function
            fig, ax = plt.subplots(figsize=(self.plot_width, self.plot_height))
            p_range = np.linspace(0, 1, 100)
            weights = weight_func(p_range, result.x)
            
            plt.plot(p_range, weights)
            plt.scatter(p_values, weight_func(p_values, result.x), alpha=0.6)
            
            plt.xlabel('p-value')
            plt.ylabel('Weight')
            plt.title(f'{self.weight_function} Weight Function')
            
            # Hiển thị plot
            st.pyplot(fig)
            
        else:  # Step Function
            # Tạo cutoffs tự động
            cutoffs = np.linspace(0, 1, self.num_steps+1)[:-1]
            initial_weights = np.ones(self.num_steps)
            
            # Tối ưu hóa
            def objective(weights):
                return -np.sum(np.log(self._step_function(p_values, cutoffs, weights)))
            
            result = minimize(
                objective,
                initial_weights,
                method='Nelder-Mead',
                bounds=[(0, None)] * self.num_steps
            )
            
            # Hiển thị kết quả
            st.write("### Kết quả Step Function Model")
            st.write(f"Log-likelihood: {-result.fun:.4f}")
            
            # Tạo bảng kết quả
            results_df = pd.DataFrame({
                'Khoảng p-value': [f"≤ {cutoffs[0]:.3f}"] + 
                                 [f"{cutoffs[i-1]:.3f} - {cutoffs[i]:.3f}" for i in range(1, len(cutoffs))] +
                                 [f"> {cutoffs[-1]:.3f}"],
                'Trọng số': np.append(result.x, 1.0)
            })
            st.write(results_df)
            
            # Vẽ step function
            fig, ax = plt.subplots(figsize=(self.plot_width, self.plot_height))
            
            # Vẽ các bước
            for i in range(len(cutoffs)):
                if i == 0:
                    plt.hlines(result.x[i], 0, cutoffs[i], color='blue')
                else:
                    plt.hlines(result.x[i], cutoffs[i-1], cutoffs[i], color='blue')
            plt.hlines(1.0, cutoffs[-1], 1, color='blue')
            
            # Vẽ các điểm dữ liệu
            weights = self._step_function(p_values, cutoffs, result.x)
            plt.scatter(p_values, weights, alpha=0.6)
            
            plt.xlabel('p-value')
            plt.ylabel('Weight')
            plt.title('Step Function Model')
            
            # Hiển thị plot
            st.pyplot(fig)
        
        # Thực hiện likelihood ratio test
        st.write("### Likelihood Ratio Test")
        
        # Tính log-likelihood dưới H0 (không có selection bias)
        null_ll = len(p_values) * np.log(1.0)
        
        # Tính test statistic
        lr_stat = 2 * (null_ll - (-result.fun))
        df = len(result.x)
        p_value = 1 - stats.chi2.cdf(lr_stat, df)
        
        st.write(f"Test statistic: {lr_stat:.4f}")
        st.write(f"Degrees of freedom: {df}")
        st.write(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("Có bằng chứng về selection bias (p < 0.05)")
        elif p_value < 0.1:
            st.write("Có dấu hiệu về selection bias (p < 0.1)")
        else:
            st.write("Không có bằng chứng rõ ràng về selection bias")

def run_analysis(df):
    model = SelectionModels(df)
    model.setup_interface() 