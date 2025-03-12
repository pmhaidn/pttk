import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.power as smp
from statsmodels.stats.weightstats import ztest
import arviz as az
import pymc as pm

def run_one_sample_ttest(data, variable, test_value=0, alternative='two-sided', 
                        use_student=True, use_wilcoxon=False, use_ztest=False,
                        use_bayesian=False, prior_std=1.0, prior_samples=1000,
                        z_test_sd=1.0):
    """Thực hiện kiểm định một mẫu với nhiều phương pháp"""
    results = {}
    data_clean = data[variable].dropna()
    
    # Thống kê cơ bản
    n = len(data_clean)
    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    se = std / np.sqrt(n)
    
    results['descriptives'] = {
        'n': n,
        'mean': mean,
        'std': std,
        'se': se,
        'median': np.median(data_clean),
        'skewness': stats.skew(data_clean),
        'kurtosis': stats.kurtosis(data_clean)
    }
    
    # Student's t-test
    if use_student:
        t_stat, p_value = stats.ttest_1samp(data_clean, test_value, alternative=alternative)
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        cohens_d = (mean - test_value) / std
        
        results['student'] = {
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'cohens_d': cohens_d,
            'vs_mppr': np.exp(t_stat**2/2) if p_value < 1 else 1
        }
    
    # Wilcoxon signed-rank test
    if use_wilcoxon:
        stat, p_value = stats.wilcoxon(data_clean - test_value, alternative=alternative)
        results['wilcoxon'] = {
            'statistic': stat,
            'p_value': p_value
        }
    
    # Z-test
    if use_ztest:
        z_stat, p_value = ztest(data_clean, value=test_value, alternative=alternative, std_dev=z_test_sd)
        ci = stats.norm.interval(0.95, loc=mean, scale=z_test_sd/np.sqrt(n))
        results['ztest'] = {
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    # Bayesian t-test
    if use_bayesian:
        with pm.Model() as model:
            # Prior cho mean
            mu = pm.Normal('mu', mu=test_value, sigma=prior_std)
            # Prior cho standard deviation
            sigma = pm.HalfNormal('sigma', sigma=1)
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data_clean)
            # Sampling
            trace = pm.sample(prior_samples, return_inferencedata=True)
        
        # Tính Bayes Factor bằng Savage-Dickey density ratio
        prior_samples = trace.prior['mu'].values
        posterior_samples = trace.posterior['mu'].values.flatten()
        
        # Tính density tại test_value
        prior_density = stats.gaussian_kde(prior_samples)(test_value)[0]
        posterior_density = stats.gaussian_kde(posterior_samples)(test_value)[0]
        bf10 = posterior_density / prior_density
        
        # Tính credible interval
        ci = az.hdi(posterior_samples, hdi_prob=0.95)
        
        results['bayesian'] = {
            'bf10': bf10,
            'bf01': 1/bf10,
            'posterior_mean': np.mean(posterior_samples),
            'posterior_std': np.std(posterior_samples),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'trace': trace
        }
    
    return results

def check_normality(data, variable):
    """Kiểm tra phân phối chuẩn"""
    data_clean = data[variable].dropna()
    stat, p_val = stats.shapiro(data_clean)
    return {'statistic': stat, 'p_value': p_val}

def plot_distribution(data, variable, test_value, horizontal=False):
    """Vẽ biểu đồ phân phối với giá trị kiểm định"""
    plt.figure(figsize=(12, 6))
    
    # Raincloud plot
    if horizontal:
        sns.violinplot(y=data[variable], orient='h', inner=None)
        sns.stripplot(y=data[variable], orient='h', size=4, alpha=0.3)
        plt.axvline(test_value, color='red', linestyle='--', label=f'Test value = {test_value}')
    else:
        sns.violinplot(x=data[variable], orient='v', inner=None)
        sns.stripplot(x=data[variable], orient='v', size=4, alpha=0.3)
        plt.axvline(test_value, color='red', linestyle='--', label=f'Test value = {test_value}')
    
    plt.title(f'Distribution of {variable}')
    plt.legend()
    return plt.gcf()

def plot_qq(data, variable):
    """Vẽ Q-Q plot để kiểm tra phân phối chuẩn"""
    plt.figure(figsize=(8, 6))
    stats.probplot(data[variable].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {variable}')
    return plt.gcf()

def plot_bayesian_results(trace, test_value=0):
    """Vẽ các biểu đồ cho phân tích Bayesian"""
    # Prior and Posterior plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Prior vs Posterior
    az.plot_density(
        trace,
        var_names=['mu'],
        hdi_prob=0.95,
        ax=axes[0,0]
    )
    axes[0,0].axvline(test_value, color='red', linestyle='--', label='Test value')
    axes[0,0].set_title('Prior and Posterior Distribution')
    
    # Plot 2: Trace plot
    az.plot_trace(
        trace,
        var_names=['mu'],
        ax=axes[0,1]
    )
    axes[0,1].set_title('Trace Plot')
    
    # Plot 3: Forest plot
    az.plot_forest(
        trace,
        var_names=['mu'],
        hdi_prob=0.95,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Forest Plot')
    
    # Plot 4: ROPE
    az.plot_posterior(
        trace,
        var_names=['mu'],
        hdi_prob=0.95,
        rope=(-0.1, 0.1),
        ax=axes[1,1]
    )
    axes[1,1].set_title('Posterior with ROPE')
    
    plt.tight_layout()
    return plt.gcf()

def run_analysis(df, is_bayesian=False):
    """Chạy phân tích kiểm định một mẫu"""
    st.subheader("One Sample T-Test")
    
    with st.sidebar:
        st.subheader("Analysis Configuration")
        
        # Chọn biến phân tích
        numeric_vars = df.select_dtypes(include=[np.number]).columns
        if len(numeric_vars) == 0:
            st.warning("No numeric variables found in data.")
            return
        
        selected_var = st.selectbox("Select variable", options=numeric_vars)
        
        # Cấu hình kiểm định
        test_value = st.number_input("Test value", value=0.0, step=0.1)
        
        if not is_bayesian:
            st.subheader("Test Type")
            use_student = st.checkbox("Student's t-test", value=True)
            use_wilcoxon = st.checkbox("Wilcoxon signed-rank test")
            use_ztest = st.checkbox("Z-test")
            
            if use_ztest:
                z_test_sd = st.number_input("Z-test standard deviation", value=1.0, min_value=0.1)
            else:
                z_test_sd = 1.0
        else:
            use_student = False
            use_wilcoxon = False
            use_ztest = False
            z_test_sd = 1.0
            
            st.subheader("Bayesian Configuration")
            prior_std = st.number_input("Prior standard deviation", value=1.0, min_value=0.1)
            prior_samples = st.number_input("MCMC samples", value=1000, min_value=100)
            show_bayesian_plots = st.checkbox("Show Bayesian plots", value=True)
            
        alternative = st.selectbox(
            "Alternative hypothesis",
            options=['two-sided', 'greater', 'less'],
            format_func=lambda x: {
                'two-sided': '≠ Test value',
                'greater': '> Test value',
                'less': '< Test value'
            }[x]
        )
        
        st.subheader("Additional Statistics")
        show_descriptives = st.checkbox("Descriptive statistics", value=True)
        show_effect_size = st.checkbox("Effect size", value=True)
        show_ci = st.checkbox("Confidence interval", value=True)
        if show_ci:
            ci_level = st.slider("Confidence level", min_value=0.8, max_value=0.99, value=0.95, step=0.01)
        
        st.subheader("Assumption Checks")
        show_normality = st.checkbox("Normality test", value=True)
        show_qq = st.checkbox("Q-Q plot", value=True)
        
        st.subheader("Plots")
        show_raincloud = st.checkbox("Raincloud plot", value=True)
        if show_raincloud:
            horizontal_display = st.checkbox("Horizontal display", value=False)
    
    if selected_var:
        try:
            # Thực hiện kiểm định
            results = run_one_sample_ttest(
                df, selected_var, test_value=test_value,
                alternative=alternative, use_student=use_student,
                use_wilcoxon=use_wilcoxon, use_ztest=use_ztest,
                use_bayesian=is_bayesian, prior_std=prior_std if is_bayesian else 1.0,
                prior_samples=prior_samples if is_bayesian else 1000,
                z_test_sd=z_test_sd
            )
            
            # Hiển thị kết quả
            if show_descriptives and 'descriptives' in results:
                st.write("### Descriptive Statistics")
                desc = results['descriptives']
                st.write(f"- N: {desc['n']}")
                st.write(f"- Mean: {desc['mean']:.4f}")
                st.write(f"- Standard deviation: {desc['std']:.4f}")
                st.write(f"- Median: {desc['median']:.4f}")
                st.write(f"- Skewness: {desc['skewness']:.4f}")
                st.write(f"- Kurtosis: {desc['kurtosis']:.4f}")
            
            if not is_bayesian:
                # Student's t-test results
                if use_student and 'student' in results:
                    st.write("### Student's t-test Results")
                    student = results['student']
                    st.write(f"- t-statistic: {student['t_stat']:.4f}")
                    st.write(f"- p-value: {student['p_value']:.4f}")
                    if show_effect_size:
                        st.write(f"- Cohen's d: {student['cohens_d']:.4f}")
                    if show_ci:
                        st.write(f"- {ci_level*100}% CI: [{student['ci_lower']:.4f}, {student['ci_upper']:.4f}]")
                    st.write(f"- Vovk-Sellke MPR: {student['vs_mppr']:.4f}")
                
                # Wilcoxon test results
                if use_wilcoxon and 'wilcoxon' in results:
                    st.write("### Wilcoxon Signed-rank Test Results")
                    wilcoxon = results['wilcoxon']
                    st.write(f"- W-statistic: {wilcoxon['statistic']:.4f}")
                    st.write(f"- p-value: {wilcoxon['p_value']:.4f}")
                
                # Z-test results
                if use_ztest and 'ztest' in results:
                    st.write("### Z-test Results")
                    ztest = results['ztest']
                    st.write(f"- Z-statistic: {ztest['z_stat']:.4f}")
                    st.write(f"- p-value: {ztest['p_value']:.4f}")
                    if show_ci:
                        st.write(f"- {ci_level*100}% CI: [{ztest['ci_lower']:.4f}, {ztest['ci_upper']:.4f}]")
            else:
                # Bayesian results
                if 'bayesian' in results:
                    st.write("### Bayesian t-test Results")
                    bayesian = results['bayesian']
                    st.write(f"- Bayes Factor (BF₁₀): {bayesian['bf10']:.4f}")
                    st.write(f"- Bayes Factor (BF₀₁): {bayesian['bf01']:.4f}")
                    st.write(f"- Posterior mean: {bayesian['posterior_mean']:.4f}")
                    st.write(f"- Posterior SD: {bayesian['posterior_std']:.4f}")
                    st.write(f"- 95% Credible interval: [{bayesian['ci_lower']:.4f}, {bayesian['ci_upper']:.4f}]")
                    
                    if show_bayesian_plots:
                        st.write("### Bayesian Plots")
                        fig = plot_bayesian_results(bayesian['trace'], test_value)
                        st.pyplot(fig)
                        plt.close()
            
            # Kiểm tra phân phối chuẩn
            if show_normality:
                st.write("### Normality Test")
                normality = check_normality(df, selected_var)
                st.write("Shapiro-Wilk test:")
                st.write(f"- Statistic: {normality['statistic']:.4f}")
                st.write(f"- p-value: {normality['p_value']:.4f}")
                
                if normality['p_value'] < 0.05:
                    st.warning("Data may not be normally distributed (p < 0.05)")
                else:
                    st.success("Data appears to be normally distributed (p >= 0.05)")
            
            # Q-Q plot
            if show_qq:
                st.write("### Q-Q Plot")
                fig = plot_qq(df, selected_var)
                st.pyplot(fig)
                plt.close()
            
            # Raincloud plot
            if show_raincloud:
                st.write("### Raincloud Plot")
                fig = plot_distribution(df, selected_var, test_value, horizontal=horizontal_display)
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"Error performing test: {str(e)}") 