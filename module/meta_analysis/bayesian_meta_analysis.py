import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

class BayesianMetaAnalysis:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Phân tích Meta Bayesian")
        
        # Cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình phân tích")
            
            # Chọn cột dữ liệu
            self.effect_size_col = st.selectbox(
                "Chọn cột chứa Effect Size",
                self.data.columns,
                key="bayes_effect_size_col"
            )
            
            self.se_col = st.selectbox(
                "Chọn cột chứa Standard Error",
                self.data.columns,
                key="bayes_se_col"
            )
            
            # Prior settings
            st.write("### Cấu hình Prior")
            
            self.mu_prior_mean = st.number_input(
                "Prior mean của μ",
                value=0.0,
                step=0.1,
                key="mu_prior_mean"
            )
            
            self.mu_prior_sd = st.number_input(
                "Prior SD của μ",
                value=1.0,
                min_value=0.1,
                step=0.1,
                key="mu_prior_sd"
            )
            
            self.tau_prior_shape = st.number_input(
                "Prior shape của τ",
                value=1.0,
                min_value=0.1,
                step=0.1,
                key="tau_prior_shape"
            )
            
            self.tau_prior_rate = st.number_input(
                "Prior rate của τ",
                value=1.0,
                min_value=0.1,
                step=0.1,
                key="tau_prior_rate"
            )
            
            # MCMC settings
            st.write("### Cấu hình MCMC")
            
            self.n_draws = st.number_input(
                "Số lượng draws",
                value=2000,
                min_value=500,
                step=500,
                key="n_draws"
            )
            
            self.n_chains = st.number_input(
                "Số lượng chains",
                value=4,
                min_value=2,
                max_value=8,
                step=1,
                key="n_chains"
            )
            
            self.n_tune = st.number_input(
                "Số lượng tune steps",
                value=1000,
                min_value=500,
                step=500,
                key="n_tune"
            )
            
            # Tùy chọn hiển thị
            st.write("### Tùy chọn hiển thị")
            
            self.plot_type = st.selectbox(
                "Chọn loại biểu đồ",
                ["Forest Plot", "Trace Plot", "Posterior Plot"],
                key="plot_type"
            )
            
            self.show_study_labels = st.checkbox(
                "Hiển thị tên nghiên cứu",
                value=True,
                key="bayes_show_labels"
            )
            
            if self.show_study_labels:
                self.study_label_col = st.selectbox(
                    "Chọn cột chứa tên nghiên cứu",
                    ["Index"] + list(self.data.columns),
                    key="bayes_label_col"
                )
        
        try:
            # Lấy dữ liệu
            effect_sizes = self.data[self.effect_size_col].values
            standard_errors = self.data[self.se_col].values
            
            # Thực hiện phân tích Bayesian
            with pm.Model() as model:
                # Priors
                mu = pm.Normal("mu", mu=self.mu_prior_mean, sigma=self.mu_prior_sd)
                tau = pm.Gamma("tau", alpha=self.tau_prior_shape, beta=self.tau_prior_rate)
                
                # Model
                theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(effect_sizes))
                y = pm.Normal("y", mu=theta, sigma=standard_errors, observed=effect_sizes)
                
                # MCMC sampling
                trace = pm.sample(
                    draws=self.n_draws,
                    chains=self.n_chains,
                    tune=self.n_tune,
                    return_inferencedata=True
                )
            
            # Hiển thị kết quả
            st.write("### Kết quả phân tích")
            
            # Summary statistics
            summary = az.summary(trace, var_names=["mu", "tau"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tổng hợp Effect Size (μ)**")
                st.write(f"Mean: {summary.loc['mu', 'mean']:.4f}")
                st.write(f"SD: {summary.loc['mu', 'sd']:.4f}")
                st.write(f"95% HDI: [{summary.loc['mu', 'hdi_3%']:.4f}, {summary.loc['mu', 'hdi_97%']:.4f}]")
            
            with col2:
                st.write("**Heterogeneity (τ)**")
                st.write(f"Mean: {summary.loc['tau', 'mean']:.4f}")
                st.write(f"SD: {summary.loc['tau', 'sd']:.4f}")
                st.write(f"95% HDI: [{summary.loc['tau', 'hdi_3%']:.4f}, {summary.loc['tau', 'hdi_97%']:.4f}]")
            
            # Vẽ biểu đồ theo lựa chọn
            if self.plot_type == "Forest Plot":
                self._create_forest_plot(trace, effect_sizes, standard_errors)
            elif self.plot_type == "Trace Plot":
                self._create_trace_plot(trace)
            else:  # Posterior Plot
                self._create_posterior_plot(trace)
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
    
    def _create_forest_plot(self, trace, effect_sizes, standard_errors):
        fig, ax = plt.subplots(figsize=(10, len(self.data)*0.4 + 2))
        
        y_positions = np.arange(len(self.data))
        
        # Plot individual studies
        plt.scatter(effect_sizes, y_positions, s=50)
        
        # Add confidence intervals
        for i, (es, se) in enumerate(zip(effect_sizes, standard_errors)):
            ci_lower = es - 1.96*se
            ci_upper = es + 1.96*se
            plt.hlines(i, ci_lower, ci_upper, color='black')
        
        # Add posterior mean and HDI
        summary = az.summary(trace, var_names=["mu"])
        mean = summary.loc['mu', 'mean']
        hdi_lower = summary.loc['mu', 'hdi_3%']
        hdi_upper = summary.loc['mu', 'hdi_97%']
        
        plt.axvline(mean, color='red', linestyle='--', alpha=0.5)
        plt.axvline(0, color='black', linestyle='-', alpha=0.2)
        
        # Add diamond for overall effect
        diamond_height = 0.4
        diamond_y = -1
        
        diamond_coords = np.array([
            [mean, diamond_y],
            [hdi_upper, diamond_y + diamond_height/2],
            [mean, diamond_y + diamond_height],
            [hdi_lower, diamond_y + diamond_height/2]
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
        plt.title('Bayesian Forest Plot')
        
        # Thêm chú thích
        plt.text(
            mean,
            diamond_y - 0.8,
            f'Posterior Mean: {mean:.3f} [{hdi_lower:.3f}, {hdi_upper:.3f}]',
            horizontalalignment='center',
            fontsize=10
        )
        
        st.pyplot(fig)
    
    def _create_trace_plot(self, trace):
        fig = plt.figure(figsize=(12, 6))
        az.plot_trace(trace, var_names=["mu", "tau"])
        plt.tight_layout()
        st.pyplot(fig)
    
    def _create_posterior_plot(self, trace):
        fig = plt.figure(figsize=(12, 6))
        az.plot_posterior(trace, var_names=["mu", "tau"])
        plt.tight_layout()
        st.pyplot(fig)

def run_analysis(df):
    analyzer = BayesianMetaAnalysis(df)
    analyzer.setup_interface() 