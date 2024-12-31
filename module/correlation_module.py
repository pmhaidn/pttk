import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class CorrelationAnalysis:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.plot_template = "plotly_white"
        
    def sidebar_options(self):
        st.sidebar.title("Phân tích tương quan")
        
        # Chọn biến
        variables = st.sidebar.multiselect(
            "Biến phân tích",
            self.numeric_cols,
            default=self.numeric_cols[:2] if len(self.numeric_cols) >= 2 else self.numeric_cols,
            key = "correlation_variables"
        )
        
        partial_out = st.sidebar.multiselect(
            "Biến điều kiện (Partial out)",
            [col for col in self.numeric_cols if col not in variables],
            key="correlation_partial_out"
        )
        
        # Hệ số tương quan mẫu
        st.sidebar.subheader("Hệ số tương quan mẫu")
        pearson = st.sidebar.checkbox("Pearson's r", value=True, key="correlation_pearson")
        spearman = st.sidebar.checkbox("Spearman's rho", key="correlation_spearman")
        kendall = st.sidebar.checkbox("Kendall's tau-b", key="correlation_kendall")
        
        # Tùy chọn bổ sung
        st.sidebar.subheader("Tùy chọn bổ sung")
        display_pairwise = st.sidebar.checkbox("Hiển thị từng cặp", key="correlation_display_pairwise")
        report_significance = st.sidebar.checkbox("Báo cáo mức ý nghĩa", value=True, key="correlation_report_significance")
        flag_significant = st.sidebar.checkbox("Đánh dấu tương quan có ý nghĩa", key="correlation_flag_significant")
        
        confidence_intervals = st.sidebar.checkbox("Khoảng tin cậy", key="correlation_confidence_intervals")
        if confidence_intervals:
            ci_level = st.sidebar.number_input("Độ tin cậy (%)", value=95.0, key="correlation_ci_level")
            n_bootstrap = st.sidebar.number_input("Số lượng bootstrap", value=1000, key="correlation_n_bootstrap")
            
        vovk_sellke = st.sidebar.checkbox("Tỷ số p tối đa Vovk-Sellke", key="correlation_vovk_sellke")
        effect_size = st.sidebar.checkbox("Cỡ mẫu hiệu quả (Fisher's z)", key="correlation_effect_size")
        sample_size = st.sidebar.checkbox("Kích thước mẫu", key="correlation_sample_size")
        covariance = st.sidebar.checkbox("Hiệp phương sai", key="correlation_covariance")
        
        # Giả thuyết thay thế
        st.sidebar.subheader("Giả thuyết thay thế")
        alt_hypothesis = st.sidebar.radio(
            "",
            ["Có tương quan", "Tương quan dương", "Tương quan âm"],
            key="correlation_alt_hypothesis"
        )
        
        # Biểu đồ
        st.sidebar.subheader("Biểu đồ")
        scatter_plots = st.sidebar.checkbox("Biểu đồ phân tán", key="correlation_scatter_plots")
        if scatter_plots:
            show_density = st.sidebar.checkbox("Mật độ cho biến", key="correlation_show_density")
            show_stats = st.sidebar.checkbox("Thống kê", key="correlation_show_stats")
            ci_plots = st.sidebar.checkbox("Khoảng tin cậy", key="correlation_ci_plots")
            if ci_plots:
                ci_plot_level = st.sidebar.number_input("Độ tin cậy cho biểu đồ (%)", value=95.0, key="correlation_ci_plot_level")
            pred_intervals = st.sidebar.checkbox("Khoảng dự đoán", key="correlation_pred_intervals")
            if pred_intervals:
                pred_interval_level = st.sidebar.number_input("Độ tin cậy cho dự đoán (%)", value=95.0, key="correlation_pred_interval_level")
        heatmap = st.sidebar.checkbox("Heatmap", key="correlation_heatmap")

        
        return {
            'variables': variables,
            'partial_out': partial_out,
            'pearson': pearson,
            'spearman': spearman,
            'kendall': kendall,
            'display_pairwise': display_pairwise,
            'report_significance': report_significance,
            'flag_significant': flag_significant,
            'confidence_intervals': confidence_intervals,
            'ci_level': ci_level if confidence_intervals else None,
            'n_bootstrap': n_bootstrap if confidence_intervals else None,
            'vovk_sellke': vovk_sellke,
            'effect_size': effect_size,
            'sample_size': sample_size,
            'covariance': covariance,
            'alt_hypothesis': alt_hypothesis,
            'scatter_plots': scatter_plots,
            'show_density': show_density if scatter_plots else False,
            'show_stats': show_stats if scatter_plots else False,
            'ci_plots': ci_plots if scatter_plots else False,
            'ci_plot_level': ci_plot_level if scatter_plots and ci_plots else None,
            'pred_intervals': pred_intervals if scatter_plots else False,
            'pred_interval_level': pred_interval_level if scatter_plots and pred_intervals else None,
            'heatmap': heatmap
        }

    def calculate_correlation(self, x, y, method='pearson', partial_vars=None):
        if partial_vars is None or len(partial_vars) == 0:
            if method == 'pearson':
                r, p = stats.pearsonr(x, y)
                return r, p
            elif method == 'spearman':
                r, p = stats.spearmanr(x, y)
                return r, p
            elif method == 'kendall':
                r, p = stats.kendalltau(x, y)
                return r, p
        else:
           # Calculate partial correlation using statsmodels
            data = pd.DataFrame({'y': y, 'x': x})
            for i, var in enumerate(partial_vars):
                data[f'z{i}'] = self.df[var]
            
            # Use statsmodels to calculate partial correlation
            formula = 'y ~ x + ' + ' + '.join([f'z{i}' for i in range(len(partial_vars))])
            model = sm.OLS.from_formula(formula, data).fit()
            partial_r = model.params['x']
            partial_p = model.pvalues['x']
            
            return partial_r, partial_p
    
    def create_correlation_table(self, variables, method='pearson', partial_vars=None):
        data = []
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:
                    if partial_vars:
                         r, p = self.calculate_correlation(
                            self.df[var1],
                            self.df[var2],
                            method=method,
                            partial_vars=partial_vars
                            )
                    else:
                        r, p = self.calculate_correlation(
                            self.df[var1],
                            self.df[var2],
                            method=method
                        )
                    data.append({
                        'Variable': f'{var1} - {var2}',
                        "Hệ số tương quan": f"{r:.3f}",
                        'p-value': f"{p:.3f}"
                    })
        
        return pd.DataFrame(data)
    
    def create_partial_correlation_table(self, variables, partial_vars, method='pearson'):
        data = []
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
               if i < j:
                  r, p = self.calculate_correlation(
                       self.df[var1],
                       self.df[var2],
                        method=method,
                        partial_vars=partial_vars
                   )
                  data.append({
                       'Variable': f'{var1} - {var2}',
                       "Hệ số tương quan": f"{r:.3f}",
                       'p-value': f"{p:.3f}"
                  })
        return pd.DataFrame(data)

    def create_heatmap(self, correlation_matrix, pvalues=None, title="Correlation Matrix"):
        fig = go.Figure()
        
        # Add correlation heatmap
        fig.add_trace(go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            text=np.round(correlation_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1
        ))
        
        # Add significance markers if provided
        if pvalues is not None:
            sig_markers = pvalues < 0.05
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    if sig_markers.iloc[i, j]:
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text="*",
                            showarrow=False,
                            font=dict(size=16, color="black")
                        )
        
        fig.update_layout(
            title=title,
            template=self.plot_template,
            width=800,
            height=800
        )
        
        return fig

    def create_scatter_plot(self, x, y, options):
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data points',
            marker=dict(
                size=8,
                opacity=0.6
            )
        ))
        
        # Add regression line
        if options['show_stats']:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = slope * x_range + intercept
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='Đường hồi quy',
                line=dict(color='red', dash='dash')
            ))
        
        # Add confidence intervals
        if options['ci_plots']:
            ci_level = options['ci_plot_level'] / 100
            x_new = np.linspace(x.min(), x.max(), 100)
            y_pred, y_ci_lower, y_ci_upper = self.calculate_confidence_intervals(
                x, y, x_new, confidence_level=ci_level
            )
            
            fig.add_trace(go.Scatter(
                x=x_new,
                y=y_ci_upper,
                mode='lines',
                name=f'{options["ci_plot_level"]}% CI',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_new,
                y=y_ci_lower,
                mode='lines',
                name=f'{options["ci_plot_level"]}% CI',
                line=dict(color='gray', dash='dot'),
                fill='tonexty'
            ))
        
        fig.update_layout(
            title=f'Biểu đồ phân tán: {x.name} vs {y.name}',
            xaxis_title=x.name,
            yaxis_title=y.name,
            template=self.plot_template
        )
        
        return fig

    def calculate_confidence_intervals(self, x, y, x_new, confidence_level=0.95):
        n = len(x)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Predicted values
        y_pred = slope * x_new + intercept
        
        # Standard error of the prediction
        x_mean = np.mean(x)
        x_std = np.std(x, ddof=1)
        
        # Calculate prediction standard error
        mse = np.sum((y - (slope * x + intercept))**2) / (n-2)
        std_err_pred = np.sqrt(mse * (1/n + (x_new - x_mean)**2 / (n-1) / x_std**2))
        
        # Calculate confidence intervals
        t_value = stats.t.ppf((1 + confidence_level) / 2, n-2)
        ci_lower = y_pred - t_value * std_err_pred
        ci_upper = y_pred + t_value * std_err_pred
        
        return y_pred, ci_lower, ci_upper

    def correlation_analysis(self):
        st.title("Phân tích tương quan")
        options = self.sidebar_options()
        
        if len(options['variables']) < 2:
            st.warning("Vui lòng chọn ít nhất 2 biến để phân tích tương quan.")
            return
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("")  # add blank space to col1
        with col2:
           # Display partial correlation if partial out variable is selected
            if options['partial_out']:
                st.subheader("Tương quan riêng phần Pearson")
                st.markdown("---")
                table = self.create_partial_correlation_table(options['variables'], options['partial_out'], method='pearson')
                st.dataframe(table, width = 800)
                st.markdown("---")
                st.write(f"**Note**: Điều kiện trên các biến: {', '.join(options['partial_out'])}.")
            
           # Tính toán ma trận tương quan
            if options['pearson'] and not options['partial_out']:
                st.subheader("Ma trận tương quan Pearson")
                st.markdown("---")
                table = self.create_correlation_table(options['variables'], method='pearson')
                st.dataframe(table, width = 800)
                st.markdown("---")

            if options['spearman']:
                st.subheader("Ma trận tương quan Spearman")
                st.markdown("---")
                table = self.create_correlation_table(options['variables'], method='spearman')
                st.dataframe(table, width = 800)
                st.markdown("---")
                
            if options['kendall']:
                st.subheader("Ma trận tương quan Kendall")
                st.markdown("---")
                table = self.create_correlation_table(options['variables'], method='kendall')
                st.dataframe(table, width = 800)
                st.markdown("---")
            
            # Hiển thị biểu đồ phân tán
            if options['scatter_plots']:
                st.subheader("Biểu đồ phân tán")
                st.markdown("---")
                for i, var1 in enumerate(options['variables']):
                    for j, var2 in enumerate(options['variables']):
                        if i < j:
                            fig = self.create_scatter_plot(
                                self.df[var1],
                                self.df[var2],
                                options
                            )
                            st.plotly_chart(fig)
            
            if options["heatmap"]:
              st.subheader("Heatmap")
              st.markdown("---")
              matrix, pvalues = self.create_correlation_matrix(options['variables'], method='pearson')
              fig = self.create_heatmap(matrix, title="Heatmap")
              st.plotly_chart(fig)