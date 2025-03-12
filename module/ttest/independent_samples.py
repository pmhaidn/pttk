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

def run_independent_ttest(data, var1, var2, test_value=0, alternative='two-sided',
                         use_student=True, use_mann_whitney=False, use_ztest=False,
                         use_bayesian=False, prior_std=1.0, prior_samples=1000,
                         equal_var=True, z_test_sd=1.0):
    """Thực hiện kiểm định hai mẫu độc lập với nhiều phương pháp"""
    results = {}
    data1 = data[var1].dropna()
    data2 = data[var2].dropna()
    
    # Thống kê cơ bản
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
    
    results['descriptives'] = {
        'group1': {
            'n': n1,
            'mean': mean1,
            'std': std1,
            'se': se1,
            'median': np.median(data1),
            'skewness': stats.skew(data1),
            'kurtosis': stats.kurtosis(data1)
        },
        'group2': {
            'n': n2,
            'mean': mean2,
            'std': std2,
            'se': se2,
            'median': np.median(data2),
            'skewness': stats.skew(data2),
            'kurtosis': stats.kurtosis(data2)
        }
    }
    
    # Student's t-test
    if use_student:
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2)/(n1+n2-2)) if equal_var else np.sqrt(std1**2/n1 + std2**2/n2)
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Confidence intervals
        if equal_var:
            df = n1 + n2 - 2
            ci = stats.t.interval(0.95, df, loc=mean1-mean2, scale=pooled_std*np.sqrt(1/n1 + 1/n2))
        else:
            # Welch's t-test
            df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
            ci = stats.t.interval(0.95, df, loc=mean1-mean2, scale=np.sqrt(std1**2/n1 + std2**2/n2))
        
        results['student'] = {
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'cohens_d': cohens_d,
            'vs_mppr': np.exp(t_stat**2/2) if p_value < 1 else 1
        }
    
    # Mann-Whitney U test
    if use_mann_whitney:
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
        results['mann_whitney'] = {
            'statistic': stat,
            'p_value': p_value
        }
    
    # Z-test
    if use_ztest:
        z_stat, p_value = ztest(data1, data2, value=test_value, alternative=alternative, std_dev=z_test_sd)
        ci = stats.norm.interval(0.95, loc=mean1-mean2, scale=z_test_sd*np.sqrt(1/n1 + 1/n2))
        results['ztest'] = {
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    # Bayesian t-test
    if use_bayesian:
        with pm.Model() as model:
            # Group means
            mu1 = pm.Normal('mu1', mu=0, sigma=prior_std)
            mu2 = pm.Normal('mu2', mu=0, sigma=prior_std)
            # Common standard deviation for both groups
            sigma = pm.HalfNormal('sigma', sigma=1)
            # Effect size (standardized difference)
            effect = pm.Deterministic('effect', (mu1 - mu2) / sigma)
            # Likelihoods
            likelihood1 = pm.Normal('likelihood1', mu=mu1, sigma=sigma, observed=data1)
            likelihood2 = pm.Normal('likelihood2', mu=mu2, sigma=sigma, observed=data2)
            # Sampling
            trace = pm.sample(prior_samples, return_inferencedata=True)
        
        # Tính Bayes Factor bằng Savage-Dickey density ratio
        effect_samples = trace.posterior['effect'].values.flatten()
        prior_samples = np.random.normal(0, prior_std, size=prior_samples)
        
        # Tính density tại test_value (0 for difference)
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

def check_normality(data, variable, group):
    """Kiểm tra phân phối chuẩn cho từng nhóm"""
    groups = data[group].unique()
    results = {}
    
    for g in groups:
        group_data = data[data[group] == g][variable].dropna()
        stat, p_val = stats.shapiro(group_data)
        results[str(g)] = {
            'statistic': stat,
            'p_value': p_val
        }
    
    return results

def check_homogeneity(data, variable, group):
    """Kiểm tra tính đồng nhất phương sai"""
    groups = data[group].unique()
    group1_data = data[data[group] == groups[0]][variable].dropna()
    group2_data = data[data[group] == groups[1]][variable].dropna()
    
    stat, p_val = stats.levene(group1_data, group2_data)
    return {'statistic': stat, 'p_value': p_val}

def plot_distribution(data, variable, group, horizontal=False):
    """Vẽ biểu đồ phân phối cho từng nhóm"""
    plt.figure(figsize=(12, 6))
    
    if horizontal:
        sns.violinplot(data=data, x=variable, y=group, orient='h', inner=None)
        sns.stripplot(data=data, x=variable, y=group, orient='h', size=4, alpha=0.3)
    else:
        sns.violinplot(data=data, x=group, y=variable, orient='v', inner=None)
        sns.stripplot(data=data, x=group, y=variable, orient='v', size=4, alpha=0.3)
    
    plt.title(f'Distribution of {variable} by {group}')
    return plt.gcf()

def plot_qq(data, variable, group):
    """Vẽ Q-Q plot cho từng nhóm"""
    groups = data[group].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, g in enumerate(groups):
        group_data = data[data[group] == g][variable].dropna()
        stats.probplot(group_data, dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot for {g}')
    
    plt.tight_layout()
    return plt.gcf()

def plot_bayesian_results(trace):
    """Vẽ các biểu đồ cho phân tích Bayesian"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Prior vs Posterior for effect size
    az.plot_density(
        trace,
        var_names=['effect'],
        hdi_prob=0.95,
        ax=axes[0,0]
    )
    axes[0,0].axvline(0, color='red', linestyle='--', label='Null value')
    axes[0,0].set_title('Effect Size Distribution')
    
    # Plot 2: Trace plot
    az.plot_trace(
        trace,
        var_names=['effect'],
        ax=axes[0,1]
    )
    axes[0,1].set_title('Trace Plot')
    
    # Plot 3: Forest plot
    az.plot_forest(
        trace,
        var_names=['effect'],
        hdi_prob=0.95,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Forest Plot')
    
    # Plot 4: ROPE
    az.plot_posterior(
        trace,
        var_names=['effect'],
        hdi_prob=0.95,
        rope=(-0.1, 0.1),
        ax=axes[1,1]
    )
    axes[1,1].set_title('Posterior with ROPE')
    
    plt.tight_layout()
    return plt.gcf()

def run_analysis(df, is_bayesian=False):
    """Chạy phân tích kiểm định hai mẫu độc lập"""
    st.subheader("Independent Samples T-Test")
    
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
            use_welch = st.checkbox("Welch's t-test")
            use_mann_whitney = st.checkbox("Mann-Whitney U test")
            use_ztest = st.checkbox("Z-test")
            
            if use_ztest:
                z_test_sd1 = st.number_input("Group 1 SD", value=1.0, min_value=0.1)
                z_test_sd2 = st.number_input("Group 2 SD", value=1.0, min_value=0.1)
            else:
                z_test_sd1 = z_test_sd2 = 1.0
        else:
            use_student = use_welch = use_mann_whitney = use_ztest = False
            z_test_sd1 = z_test_sd2 = 1.0
            
            st.subheader("Bayesian Configuration")
            prior_std = st.number_input("Prior standard deviation", value=1.0, min_value=0.1)
            prior_samples = st.number_input("MCMC samples", value=1000, min_value=100)
            show_bayesian_plots = st.checkbox("Show Bayesian plots", value=True)
            
        alternative = st.selectbox(
            "Alternative hypothesis",
            options=['two-sided', 'greater', 'less'],
            format_func=lambda x: {
                'two-sided': 'Group 1 ≠ Group 2',
                'greater': 'Group 1 > Group 2',
                'less': 'Group 1 < Group 2'
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
        show_homogeneity = st.checkbox("Homogeneity of variances", value=True)
        show_qq = st.checkbox("Q-Q plot", value=True)
        
        st.subheader("Plots")
        show_raincloud = st.checkbox("Raincloud plot", value=True)
        if show_raincloud:
            horizontal_display = st.checkbox("Horizontal display", value=False)
    
    if var1 and var2:
        try:
            # Thực hiện kiểm định
            results = run_independent_ttest(
                df, var1, var2,
                alternative=alternative,
                use_student=use_student,
                use_welch=use_welch,
                use_mann_whitney=use_mann_whitney,
                use_ztest=use_ztest,
                use_bayesian=is_bayesian,
                prior_std=prior_std if is_bayesian else 1.0,
                prior_samples=prior_samples if is_bayesian else 1000,
                z_test_sd=z_test_sd1 if not is_bayesian else 1.0
            )
            
            # Hiển thị kết quả
            if show_descriptives and 'descriptives' in results:
                st.write("### Descriptive Statistics")
                desc = results['descriptives']
                for group in [1, 2]:
                    st.write(f"Group {group}:")
                    group_desc = desc[f'group{group}']
                    st.write(f"- N: {group_desc['n']}")
                    st.write(f"- Mean: {group_desc['mean']:.4f}")
                    st.write(f"- Standard deviation: {group_desc['std']:.4f}")
                    st.write(f"- Median: {group_desc['median']:.4f}")
                    st.write(f"- Skewness: {group_desc['skewness']:.4f}")
                    st.write(f"- Kurtosis: {group_desc['kurtosis']:.4f}")
            
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
                
                # Welch's t-test results
                if use_welch and 'welch' in results:
                    st.write("### Welch's t-test Results")
                    welch = results['welch']
                    st.write(f"- t-statistic: {welch['t_stat']:.4f}")
                    st.write(f"- p-value: {welch['p_value']:.4f}")
                    if show_ci:
                        st.write(f"- {ci_level*100}% CI: [{welch['ci_lower']:.4f}, {welch['ci_upper']:.4f}]")
                
                # Mann-Whitney test results
                if use_mann_whitney and 'mann_whitney' in results:
                    st.write("### Mann-Whitney U Test Results")
                    mw = results['mann_whitney']
                    st.write(f"- U-statistic: {mw['statistic']:.4f}")
                    st.write(f"- p-value: {mw['p_value']:.4f}")
                
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
            
            # Kiểm tra phân phối chuẩn
            if show_normality:
                st.write("### Normality Test")
                for group in [1, 2]:
                    st.write(f"Group {group}:")
                    normality = check_normality(df, var1 if group == 1 else var2, var2)
                    st.write("Shapiro-Wilk test:")
                    st.write(f"- Statistic: {normality['statistic']:.4f}")
                    st.write(f"- p-value: {normality['p_value']:.4f}")
                    
                    if normality['p_value'] < 0.05:
                        st.warning(f"Group {group} may not be normally distributed (p < 0.05)")
                    else:
                        st.success(f"Group {group} appears to be normally distributed (p >= 0.05)")
            
            # Kiểm tra đồng nhất phương sai
            if show_homogeneity:
                st.write("### Homogeneity of Variances")
                homogeneity = check_homogeneity(df, var1, var2)
                st.write("Levene's test:")
                st.write(f"- Statistic: {homogeneity['statistic']:.4f}")
                st.write(f"- p-value: {homogeneity['p_value']:.4f}")
                
                if homogeneity['p_value'] < 0.05:
                    st.warning("Variances may not be equal (p < 0.05)")
                else:
                    st.success("Variances appear to be equal (p >= 0.05)")
            
            # Q-Q plot
            if show_qq:
                st.write("### Q-Q Plot")
                for group in [1, 2]:
                    st.write(f"Group {group}:")
                    fig = plot_qq(df, var1 if group == 1 else var2, var2)
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