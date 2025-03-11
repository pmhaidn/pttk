import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ClassicalMetaAnalysis:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Phân tích Meta Cổ điển")
        
        # Tất cả cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình phân tích")
            
            # Chọn cột dữ liệu
            self.effect_size_col = st.selectbox(
                "Chọn cột chứa Effect Size",
                self.data.columns,
                key="effect_size_col"
            )
            
            self.se_col = st.selectbox(
                "Chọn cột chứa Standard Error",
                self.data.columns,
                key="se_col"
            )
            
            # Chọn model
            self.model = st.selectbox(
                "Chọn mô hình phân tích",
                ["Fixed Effects", "Random Effects"],
                key="model"
            )
            
            # Chọn method cho random effects
            if self.model == "Random Effects":
                self.method = st.selectbox(
                    "Chọn phương pháp ước lượng",
                    ["DL", "REML", "ML", "EB"],
                    key="method"
                )
            
            # Confidence level
            self.conf_level = st.slider(
                "Mức độ tin cậy",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="conf_level"
            )
            
            # Tùy chọn hiển thị
            st.write("### Tùy chọn hiển thị")
            
            self.plot_width = st.slider(
                "Chiều rộng biểu đồ",
                min_value=6,
                max_value=20,
                value=10,
                key="plot_width"
            )
            
            self.plot_height_per_study = st.slider(
                "Chiều cao cho mỗi nghiên cứu",
                min_value=0.2,
                max_value=1.0,
                value=0.4,
                step=0.1,
                key="plot_height"
            )
            
            self.show_study_labels = st.checkbox(
                "Hiển thị tên nghiên cứu",
                value=True,
                key="show_labels"
            )
            
            if self.show_study_labels:
                self.study_label_col = st.selectbox(
                    "Chọn cột chứa tên nghiên cứu",
                    ["Index"] + list(self.data.columns),
                    key="label_col"
                )
        
        # Thực hiện phân tích và hiển thị kết quả
        try:
            # Lấy dữ liệu
            effect_sizes = self.data[self.effect_size_col].values
            standard_errors = self.data[self.se_col].values
            
            # Thực hiện meta-analysis
            if self.model == "Fixed Effects":
                results = self._fixed_effects_meta(effect_sizes, standard_errors)
            else:
                results = self._random_effects_meta(effect_sizes, standard_errors)
            
            # Hiển thị kết quả
            st.write("### Kết quả phân tích")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tổng hợp Effect Size**")
                st.write(f"Effect Size: {results['mean']:.4f}")
                st.write(f"Standard Error: {np.sqrt(results['var']):.4f}")
                st.write(f"Z-value: {results['z_value']:.4f}")
                st.write(f"p-value: {results['p_value']:.4f}")
            
            with col2:
                st.write("**Khoảng tin cậy**")
                st.write(f"Lower CI ({self.conf_level*100}%): {results['ci_lower']:.4f}")
                st.write(f"Upper CI ({self.conf_level*100}%): {results['ci_upper']:.4f}")
            
            if self.model == "Random Effects":
                col3, col4 = st.columns(2)
                with col3:
                    st.write("**Heterogeneity**")
                    st.write(f"Tau²: {results['tau2']:.4f}")
                    st.write(f"I²: {results['i2']*100:.1f}%")
                    st.write(f"H²: {results['h2']:.2f}")
                
                with col4:
                    st.write("**Kiểm định heterogeneity**")
                    st.write(f"Q-statistic: {results['q_stat']:.2f}")
                    st.write(f"df: {results['k']-1}")
                    st.write(f"p-value: {results['q_pval']:.4f}")
            
            # Vẽ Forest Plot
            self._create_forest_plot(results)
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
    
    def _fixed_effects_meta(self, y, se):
        # Tính weights
        w = 1 / (se**2)
        
        # Tính mean effect
        mean = np.sum(w * y) / np.sum(w)
        var = 1 / np.sum(w)
        
        # Tính z-value và p-value
        z_value = mean / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Tính confidence intervals
        z_crit = stats.norm.ppf(1 - (1 - self.conf_level) / 2)
        ci_lower = mean - z_crit * np.sqrt(var)
        ci_upper = mean + z_crit * np.sqrt(var)
        
        # Tính Q statistic
        q_stat = np.sum(w * (y - mean)**2)
        q_pval = 1 - stats.chi2.cdf(q_stat, df=len(y)-1)
        
        return {
            'mean': mean,
            'var': var,
            'z_value': z_value,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weights': w,
            'q_stat': q_stat,
            'q_pval': q_pval,
            'k': len(y)
        }
    
    def _random_effects_meta(self, y, se):
        # Đầu tiên tính fixed effects
        fe_results = self._fixed_effects_meta(y, se)
        
        # Tính tau^2 using chosen method
        if self.method == "DL":
            tau2 = self._calc_tau2_DL(y, se, fe_results)
        elif self.method == "REML":
            tau2 = self._calc_tau2_REML(y, se)
        else:  # Default to DL if method not implemented
            tau2 = self._calc_tau2_DL(y, se, fe_results)
        
        # Tính random effects weights
        w = 1 / (se**2 + tau2)
        
        # Tính mean effect
        mean = np.sum(w * y) / np.sum(w)
        var = 1 / np.sum(w)
        
        # Tính z-value và p-value
        z_value = mean / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Tính confidence intervals
        z_crit = stats.norm.ppf(1 - (1 - self.conf_level) / 2)
        ci_lower = mean - z_crit * np.sqrt(var)
        ci_upper = mean + z_crit * np.sqrt(var)
        
        # Tính heterogeneity measures
        q_stat = fe_results['q_stat']
        q_pval = fe_results['q_pval']
        df = len(y) - 1
        
        # H^2 = Q/(k-1)
        h2 = q_stat / df
        
        # I^2 = (H^2 - 1)/H^2
        i2 = max(0, (h2 - 1) / h2)
        
        return {
            'mean': mean,
            'var': var,
            'z_value': z_value,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weights': w,
            'tau2': tau2,
            'q_stat': q_stat,
            'q_pval': q_pval,
            'h2': h2,
            'i2': i2,
            'k': len(y)
        }
    
    def _calc_tau2_DL(self, y, se, fe_results):
        # DerSimonian-Laird estimator
        w = 1 / (se**2)
        q = fe_results['q_stat']
        df = len(y) - 1
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0, (q - df) / c)
        return tau2
    
    def _calc_tau2_REML(self, y, se, max_iter=100, tol=1e-6):
        # REML estimator using iterative algorithm
        tau2 = 0
        for _ in range(max_iter):
            w = 1 / (se**2 + tau2)
            mean = np.sum(w * y) / np.sum(w)
            q = np.sum(w * (y - mean)**2)
            
            # Update tau2
            tau2_new = max(0, tau2 + (q - (len(y)-1)) / np.sum(w))
            
            if abs(tau2_new - tau2) < tol:
                break
            tau2 = tau2_new
        
        return tau2
    
    def _create_forest_plot(self, results):
        fig, ax = plt.subplots(figsize=(self.plot_width, len(self.data)*self.plot_height_per_study + 2))
        
        y_positions = np.arange(len(self.data))
        
        # Plot individual studies
        plt.scatter(self.data[self.effect_size_col], y_positions, s=50)
        
        # Add confidence intervals
        for i, (es, se) in enumerate(zip(self.data[self.effect_size_col], self.data[self.se_col])):
            ci_lower = es - 1.96*se
            ci_upper = es + 1.96*se
            plt.hlines(i, ci_lower, ci_upper, color='black')
        
        # Add overall effect
        plt.axvline(results['mean'], color='red', linestyle='--', alpha=0.5)
        plt.axvline(0, color='black', linestyle='-', alpha=0.2)
        
        # Add diamond for overall effect
        diamond_height = 0.4
        diamond_y = -1
        
        diamond_coords = np.array([
            [results['mean'], diamond_y],
            [results['ci_upper'], diamond_y + diamond_height/2],
            [results['mean'], diamond_y + diamond_height],
            [results['ci_lower'], diamond_y + diamond_height/2]
        ])
        
        plt.fill(
            diamond_coords[:, 0],
            diamond_coords[:, 1],
            color='red',
            alpha=0.3
        )
        
        # Add study labels
        if self.show_study_labels:
            if self.study_label_col == "Index":
                labels = [f"Study {i+1}" for i in range(len(self.data))]
            else:
                labels = self.data[self.study_label_col]
            plt.yticks(y_positions, labels)
        else:
            plt.yticks(y_positions, range(1, len(self.data) + 1))
        
        plt.xlabel('Effect Size')
        plt.ylabel('Study')
        plt.title('Forest Plot')
        
        # Thêm chú thích
        plt.text(
            results['mean'],
            diamond_y - 0.8,
            f'Overall Effect: {results["mean"]:.3f} [{results["ci_lower"]:.3f}, {results["ci_upper"]:.3f}]',
            horizontalalignment='center',
            fontsize=10
        )
        
        st.pyplot(fig)

def run_analysis(df):
    analyzer = ClassicalMetaAnalysis(df)
    analyzer.setup_interface() 