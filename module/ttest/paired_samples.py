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

def run_paired_ttest(data, var1, var2, test_value=0, alternative='two-sided',
                     use_student=True, use_wilcoxon=False, use_ztest=False,
                     use_bayesian=False, prior_std=1.0, prior_samples=1000,
                     z_test_sd=1.0):
    """Thực hiện kiểm định mẫu ghép cặp với nhiều phương pháp"""
    results = {}
    
    # Loại bỏ missing values
    data_clean = data[[var1, var2]].dropna()
    data1 = data_clean[var1]
    data2 = data_clean[var2]
    diff = data1 - data2
    
    # Thống kê cơ bản
    n = len(data_clean)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    se1, se2 = std1/np.sqrt(n), std2/np.sqrt(n)
    
    results['descriptives'] = {
        'var1': {
            'n': n,
            'mean': mean1,
            'std': std1,
            'se': se1,
            'median': np.median(data1),
            'skewness': stats.skew(data1),
            'kurtosis': stats.kurtosis(data1)
        },
        'var2': {
            'n': n,
            'mean': mean2,
            'std': std2,
            'se': se2,
            'median': np.median(data2),
            'skewness': stats.skew(data2),
            'kurtosis': stats.kurtosis(data2)
        },
        'difference': {
            'mean': np.mean(diff),
            'std': np.std(diff, ddof=1),
            'se': stats.sem(diff),
            'median': np.median(diff),
            'skewness': stats.skew(diff),
            'kurtosis': stats.kurtosis(diff)
        }
    }
    
    # Student's t-test
    if use_student:
        t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
        ci = stats.t.interval(0.95, df=n-1, loc=np.mean(diff), scale=stats.sem(diff))
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
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
        stat, p_value = stats.wilcoxon(diff, alternative=alternative)
        results['wilcoxon'] = {
            'statistic': stat,
            'p_value': p_value
        }
    
    # Z-test
    if use_ztest:
        z_stat, p_value = ztest(data1, data2, value=test_value, alternative=alternative, std_dev=z_test_sd)
        ci = stats.norm.interval(0.95, loc=np.mean(diff), scale=z_test_sd/np.sqrt(n))
        results['ztest'] = {
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    # Bayesian t-test
    if use_bayesian:
        with pm.Model() as model:
            # Prior for effect size
            effect_size = pm.Normal('effect_size', mu=0, sigma=prior_std)
            # Prior for standard deviation
            sigma = pm.HalfNormal('sigma', sigma=1)
            # Convert effect size to mean difference
            diff_sd = np.std(diff, ddof=1)
            mu = pm.Deterministic('mu', effect_size * sigma)
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=diff)
            # Sampling
            trace = pm.sample(prior_samples, return_inferencedata=True)
        
        # Tính Bayes Factor bằng Savage-Dickey density ratio
        effect_samples = trace.posterior['effect_size'].values.flatten()
        prior_samples = np.random.normal(0, prior_std, size=prior_samples)
        
        # Tính density tại test_value (0 for no effect)
        prior_density = stats.gaussian_kde(prior_samples)(0)[0]
        posterior_density = stats.gaussian_kde(effect_samples)(0)[0]
        bf10 = posterior_density / prior_density
        
        # Tính credible interval cho effect size
        ci = az.hdi(effect_samples, hdi_prob=0.95)
        
        results['bayesian'] = {
            'bf10': bf10,
            'bf01': 1/bf10,
            'posterior_mean': np.mean(effect_samples),
            'posterior_std': np.std(effect_samples),
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'trace': trace
        }
    
    return results

def check_normality(data):
    """Kiểm tra phân phối chuẩn của hiệu"""
    stat, p_val = stats.shapiro(data)
    return {'statistic': stat, 'p_value': p_val}

def plot_distribution(data, var1, var2, horizontal=False):
    """Vẽ biểu đồ phân phối cho cả hai biến và hiệu"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Violin plots cho cả hai biến
    if horizontal:
        data_long = pd.melt(data[[var1, var2]])
        sns.violinplot(data=data_long, x='value', y='variable', orient='h', inner=None, ax=axes[0])
        sns.stripplot(data=data_long, x='value', y='variable', orient='h', size=4, alpha=0.3, ax=axes[0])
    else:
        data_long = pd.melt(data[[var1, var2]])
        sns.violinplot(data=data_long, x='variable', y='value', orient='v', inner=None, ax=axes[0])
        sns.stripplot(data=data_long, x='variable', y='value', orient='v', size=4, alpha=0.3, ax=axes[0])
    axes[0].set_title('Distribution of Variables')
    
    # Plot 2: Histogram của hiệu
    diff = data[var1] - data[var2]
    sns.histplot(diff, kde=True, ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title('Distribution of Differences')
    
    plt.tight_layout()
    return plt.gcf()

def plot_qq(data):
    """Vẽ Q-Q plot cho hiệu"""
    plt.figure(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Differences')
    return plt.gcf()

def plot_bayesian_results(trace):
    """Vẽ các biểu đồ cho phân tích Bayesian"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Prior vs Posterior for effect size
    az.plot_density(
        trace,
        var_names=['effect_size'],
        hdi_prob=0.95,
        ax=axes[0,0]
    )
    axes[0,0].axvline(0, color='red', linestyle='--', label='Null value')
    axes[0,0].set_title('Effect Size Distribution')
    
    # Plot 2: Trace plot
    az.plot_trace(
        trace,
        var_names=['effect_size'],
        ax=axes[0,1]
    )
    axes[0,1].set_title('Trace Plot')
    
    # Plot 3: Forest plot
    az.plot_forest(
        trace,
        var_names=['effect_size'],
        hdi_prob=0.95,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Forest Plot')
    
    # Plot 4: ROPE
    az.plot_posterior(
        trace,
        var_names=['effect_size'],
        hdi_prob=0.95,
        rope=(-0.1, 0.1),
        ax=axes[1,1]
    )
    axes[1,1].set_title('Posterior with ROPE')
    
    plt.tight_layout()
    return plt.gcf()

def run_analysis(df, is_bayesian=False):
    """Chạy phân tích kiểm định hai mẫu ghép cặp"""
    st.subheader("Paired Samples T-Test")
    
    with st.sidebar:
        st.subheader("Analysis Configuration")
        
        # Chọn biến phân tích
        numeric_vars = df.select_dtypes(include=[np.number]).columns
        if len(numeric_vars) < 2:
            st.warning("Need at least 2 numeric variables for analysis.")
            return
            
        var1 = st.selectbox("Select first variable", options=numeric_vars)
        var2 = st.selectbox("Select second variable", 
                           options=[x for x in numeric_vars if x != var1])
        
        if not is_bayesian:
            st.subheader("Test Type")
            use_student = st.checkbox("Student's t-test", value=True)
            use_wilcoxon = st.checkbox("Wilcoxon signed-rank test")
            use_ztest = st.checkbox("Z-test")
            
            if use_ztest:
                z_test_sd = st.number_input("Difference SD", value=1.0, min_value=0.1)
            else:
                z_test_sd = 1.0
        else:
            use_student = use_wilcoxon = use_ztest = False
            z_test_sd = 1.0
            
            st.subheader("Bayesian Configuration")
            prior_std = st.number_input("Prior standard deviation", value=1.0, min_value=0.1)
            prior_samples = st.number_input("MCMC samples", value=1000, min_value=100)
            show_bayesian_plots = st.checkbox("Show Bayesian plots", value=True)
            
        alternative = st.selectbox(
            "Alternative hypothesis",
            options=['two-sided', 'greater', 'less'],
            format_func=lambda x: {
                'two-sided': 'Difference ≠ 0',
                'greater': 'Difference > 0',
                'less': 'Difference < 0'
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
    
    if var1 and var2:
        try:
            # Thực hiện kiểm định
            results = run_paired_ttest(
                df, var1, var2,
                alternative=alternative,
                use_student=use_student,
                use_wilcoxon=use_wilcoxon,
                use_ztest=use_ztest,
                use_bayesian=is_bayesian,
                prior_std=prior_std if is_bayesian else 1.0,
                prior_samples=prior_samples if is_bayesian else 1000,
                z_test_sd=z_test_sd
            )
            
            # Hiển thị kết quả
            if show_descriptives and 'descriptives' in results:
                st.write("### Descriptive Statistics")
                desc = results['descriptives']
                for var_name, var in [('Variable 1', var1), ('Variable 2', var2)]:
                    st.write(f"{var_name} ({var}):")
                    var_desc = desc[var]
                    st.write(f"- N: {var_desc['n']}")
                    st.write(f"- Mean: {var_desc['mean']:.4f}")
                    st.write(f"- Standard deviation: {var_desc['std']:.4f}")
                    st.write(f"- Median: {var_desc['median']:.4f}")
                    st.write(f"- Skewness: {var_desc['skewness']:.4f}")
                    st.write(f"- Kurtosis: {var_desc['kurtosis']:.4f}")
                
                st.write("\nDifference (Variable 1 - Variable 2):")
                diff_desc = desc['difference']
                st.write(f"- Mean: {diff_desc['mean']:.4f}")
                st.write(f"- Standard deviation: {diff_desc['std']:.4f}")
                st.write(f"- Standard error: {diff_desc['se']:.4f}")
            
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
                        fig = plot_bayesian_results(bayesian['trace'])
                        st.pyplot(fig)
                        plt.close()
            
            # Kiểm tra phân phối chuẩn của hiệu
            if show_normality:
                st.write("### Normality Test of Differences")
                normality = check_normality(df, var1, var2)
                st.write("Shapiro-Wilk test:")
                st.write(f"- Statistic: {normality['statistic']:.4f}")
                st.write(f"- p-value: {normality['p_value']:.4f}")
                
                if normality['p_value'] < 0.05:
                    st.warning("Differences may not be normally distributed (p < 0.05)")
                else:
                    st.success("Differences appear to be normally distributed (p >= 0.05)")
            
            # Q-Q plot
            if show_qq:
                st.write("### Q-Q Plot of Differences")
                fig = plot_qq(df, var1, var2)
                st.pyplot(fig)
                plt.close()
            
            # Raincloud plot
            if show_raincloud:
                st.write("### Raincloud Plot")
                fig = plot_distribution(df, var1, var2, horizontal=horizontal_display)
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"Error performing test: {str(e)}") 