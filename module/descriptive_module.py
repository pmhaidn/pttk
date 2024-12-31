import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

class DescriptiveStatistics:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.plot_template = "plotly_white"

    def calculate_bin_width(self, data, method="sturges"):
        n = len(data)
        
        if n < 2:
            return 1
            
        if method == "sturges":
            # JASP sử dụng logarit tự nhiên thay vì log2
            n_bins = int(np.ceil(1 + np.log(n)))
            
        elif method == "scott":
            # JASP implements Scott's rule directly
            h = 3.49 * np.std(data, ddof=1) / (n ** (1/3))
            data_range = np.max(data) - np.min(data)
            n_bins = max(int(np.ceil(data_range / h)), 1)
            
        elif method == "freedman-diaconis":  # JASP uses "freedman" instead of "freedman-diaconis"
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr / (n ** (1/3))
            if h == 0:  # Xử lý trường hợp IQR = 0
                h = 2 * np.std(data, ddof=1) / (n ** (1/3))
            data_range = np.max(data) - np.min(data)
            n_bins = max(int(np.ceil(data_range / h)), 1)
            
        elif method == "doane":
            # JASP sử dụng công thức Doane có điều chỉnh
            sigma = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
            g1 = stats.skew(data)
            n_bins = int(1 + np.log(n) + np.log(1 + abs(g1) / sigma))
        
        else:
            raise ValueError(f"Phương pháp '{method}' không được hỗ trợ")
        
        return max(n_bins, 1)

    def sidebar_options(self):
        st.sidebar.title("Thống kê mô tả")
        
        # Chọn biến
        selected_cols = st.sidebar.multiselect(
            "Biến",
            self.numeric_cols,
            default=self.numeric_cols[:1] if self.numeric_cols else None
        )

        # Các nhóm tùy chọn trong sidebar
        with st.sidebar.expander("Thống kê", expanded=True):
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.markdown("##### Kích thước mẫu")
                show_valid = st.checkbox("Hợp lệ", value=True)
                show_missing = st.checkbox("Thiếu", value=True)
                
                st.markdown("##### Xu hướng trung tâm")
                show_mode = st.checkbox("Mode")
                show_median = st.checkbox("Trung vị")
                show_mean = st.checkbox("Trung bình", value=True)
                
                st.markdown("##### Phân tán")
                show_std = st.checkbox("Độ lệch chuẩn", value=True)
                show_variance = st.checkbox("Phương sai")
                show_range = st.checkbox("Khoảng")
                show_minimum = st.checkbox("Giá trị nhỏ nhất", value=True)
                show_maximum = st.checkbox("Giá trị lớn nhất", value=True)
            
            with col2:
                st.markdown("##### Phân vị")
                show_quartiles = st.checkbox("Tứ phân vị")
                show_percentiles = st.checkbox("Phân vị cho:")
                if show_percentiles:
                    n_groups = st.number_input("Số nhóm bằng nhau", value=4)
                
                st.markdown("##### Phân phối")
                show_skewness = st.checkbox("Độ lệch")
                show_kurtosis = st.checkbox("Độ nhọn")
                show_shapiro = st.checkbox("Kiểm định Shapiro-Wilk")
                
                st.markdown("##### Suy luận")
                show_ci_var = st.checkbox("KTC cho phương sai")
                if show_ci_var:
                    ci_width = st.number_input("Độ tin cậy (%)", value=95.0)

        # Tùy chọn biểu đồ
        with st.sidebar.expander("Biểu đồ", expanded=True):
            show_dist_plots = st.checkbox("Biểu đồ phân phối", value=True)
            if show_dist_plots:
                show_density = st.checkbox("Hiển thị mật độ", value=True)
                show_rug = st.checkbox("Hiển thị dấu rug")
                bin_width_type = st.selectbox(
                    "Phương pháp tính độ rộng bin",
                    ["sturges", "scott", "freedman-diaconis", "doane", "manual"]
                )
                if bin_width_type == "manual":
                    n_bins = st.number_input("Số lượng bin", value=30, min_value=1)
                else:
                    n_bins = None
                
            show_qq_plots = st.checkbox("Biểu đồ Q-Q")
            show_interval_plot = st.checkbox("Biểu đồ khoảng tin cậy")
            show_pie_chart = st.checkbox("Biểu đồ tròn")
            show_dot_plot = st.checkbox("Biểu đồ chấm")

            return {
            'selected_cols': selected_cols,
            'show_valid': show_valid,
            'show_missing': show_missing,
            'show_mode': show_mode,
            'show_median': show_median,
            'show_mean': show_mean,
            'show_std': show_std,
            'show_variance': show_variance,
            'show_range': show_range,
            'show_minimum': show_minimum,
            'show_maximum': show_maximum,
            'show_quartiles': show_quartiles,
            'show_percentiles': show_percentiles,
            'n_groups': n_groups if show_percentiles else None,
            'show_skewness': show_skewness,
            'show_kurtosis': show_kurtosis,
            'show_shapiro': show_shapiro,
            'show_ci_var': show_ci_var,
            'ci_width': ci_width if show_ci_var else None,
            'show_dist_plots': show_dist_plots,
            'show_density': show_density,
            'show_rug': show_rug,
            'bin_width_type': bin_width_type,
            'n_bins': n_bins,
            'show_qq_plots': show_qq_plots,
            'show_interval_plot': show_interval_plot,
            'show_pie_chart': show_pie_chart,
            'show_dot_plot': show_dot_plot
        }
    def calculate_statistics(self, var, options):
        stats_dict = {}
        data = self.df[var].dropna()
        
        if options['show_valid']:
            stats_dict["Kích thước mẫu - Hợp lệ"] = len(data)
        if options['show_missing']:
            stats_dict["Kích thước mẫu - Thiếu"] = self.df[var].isnull().sum()
        if options['show_mean']:
            stats_dict["Xu hướng trung tâm - Trung bình"] = data.mean()
        if options['show_median']:
            stats_dict["Xu hướng trung tâm - Trung vị"] = data.median()
        if options['show_mode']:
            stats_dict["Xu hướng trung tâm - Mode"] = data.mode().iloc[0]
        if options['show_std']:
            stats_dict["Phân tán - Độ lệch chuẩn"] = data.std()
        if options['show_variance']:
            stats_dict["Phân tán - Phương sai"] = data.var()
        if options['show_minimum']:
            stats_dict["Phân tán - Giá trị nhỏ nhất"] = data.min()
        if options['show_maximum']:
            stats_dict["Phân tán - Giá trị lớn nhất"] = data.max()
        if options['show_range']:
            stats_dict["Phân tán - Khoảng"] = data.max() - data.min()
        if options['show_skewness']:
            stats_dict["Phân phối - Độ lệch"] = stats.skew(data)
        if options['show_kurtosis']:
            stats_dict["Phân phối - Độ nhọn"] = stats.kurtosis(data)
        if options['show_shapiro']:
            _, p_value = stats.shapiro(data)
            stats_dict["Kiểm định - Shapiro-Wilk p-value"] = p_value
        if options['show_ci_var']:
            n = len(data)
            alpha = 1 - options['ci_width']/100
            chi2_lower = stats.chi2.ppf(alpha/2, n-1)
            chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)
            stats_dict[f"Suy luận - {options['ci_width']}% KTC Phương sai - Trên"] = (n-1) * data.var() / chi2_lower
            stats_dict[f"Suy luận - {options['ci_width']}% KTC Phương sai - Dưới"] = (n-1) * data.var() / chi2_upper
            
        return pd.Series(stats_dict)
    def create_interval_plot(self, var):
        data = self.df[var].dropna()
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[mean],
            y=[0],
            mode='markers',
            name='Mean',
            marker=dict(size=10, color='black')
        ))
        fig.add_trace(go.Scatter(
            x=[ci[0], ci[1]],
            y=[0, 0],
            mode='lines',
            name='95% CI',
            line=dict(color='black', width=2)
        ))
        
        fig.update_layout(
            title=f"Khoảng tin cậy 95% cho giá trị trung bình - {var}",
            xaxis_title=var,
            yaxis_visible=False,
            showlegend=True,
            template=self.plot_template
        )
        
        st.plotly_chart(fig)

    def create_pie_chart(self, var):
        data = self.df[var].dropna()
        bins = pd.qcut(data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        value_counts = bins.value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=value_counts.index,
            values=value_counts.values,
            hole=.3
        )])
        
        fig.update_layout(
            title=f"Phân phối tứ phân vị - {var}",
            template=self.plot_template
        )
        
        st.plotly_chart(fig)

    def create_dot_plot(self, var):
        data = self.df[var].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data,
            y=[0] * len(data),
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                opacity=0.6
            ),
            name='Values'
        ))
        
        fig.update_layout(
            title=f"Biểu đồ chấm - {var}",
            xaxis_title=var,
            yaxis_visible=False,
            showlegend=False,
            template=self.plot_template
        )
        
        st.plotly_chart(fig)
    def create_qq_plot(self, var):
        data = self.df[var].dropna()
        fig = px.scatter(
                x=np.sort(data),
                y=stats.norm.ppf(np.linspace(0.01, 0.99, len(data))),
                title=f"Q-Q Plot - {var}",
                template=self.plot_template
            )
            
        x_range = [data.min(), data.max()]
        y_range = stats.norm.ppf([0.01, 0.99])
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig)
    def create_plots(self, var, options):
        data = self.df[var].dropna()
        
        if options['show_dist_plots']:
            if options['bin_width_type'] == "manual":
                n_bins = options['n_bins']
            else:
                n_bins = self.calculate_bin_width(data, options['bin_width_type'])
            
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=n_bins,
                name="Frequency",
                marker_color='rgb(192,192,192)',
                opacity=0.75
            ))
            
            # Density line
            if options['show_density']:
                kde = stats.gaussian_kde(data)
                x_density = np.linspace(min(data), max(data), 200)
                density = kde(x_density)
                fig.add_trace(go.Scatter(
                    x=x_density,
                    y=density * len(data) * (max(data) - min(data)) / n_bins,
                    name="Density",
                    line=dict(color='black', width=1.5)
                ))
            
            # Rug plot
            if options['show_rug']:
                fig.add_trace(go.Scatter(
                    x=data,
                    y=[-0.1] * len(data),
                    mode='markers',
                    marker=dict(symbol='line-ns', size=10),
                    name="Data points"
                ))
            
            fig.update_layout(
                title=f"Biểu đồ phân phối - {var}",
                xaxis_title=var,
                yaxis_title="Tần số",
                template=self.plot_template
            )
            
            st.plotly_chart(fig)

        # Additional plots
        if options['show_qq_plots']:
            self.create_qq_plot(var)
        if options['show_interval_plot']:
            self.create_interval_plot(var)
        if options['show_pie_chart']:
            self.create_pie_chart(var)
        if options['show_dot_plot']:
            self.create_dot_plot(var)

    def descriptive_analysis(self):
        st.title("Phân tích thống kê mô tả")
        options = self.sidebar_options()
        if not options['selected_cols']:
            st.warning("Vui lòng chọn ít nhất một biến để phân tích.")
            return
        
        # Create tabs for each selected variable
        tabs = st.tabs([f"Biến {col}" for col in options['selected_cols']])
        
        # Process each variable in its respective tab
        for tab, col in zip(tabs, options['selected_cols']):
            with tab:
                # Thống kê mô tả
                stats_results = self.calculate_statistics(col, options)
                stats_results = stats_results.to_frame("Thống kê")
                st.write("Thống kê mô tả")
                st.dataframe(stats_results)
                
                # Biểu đồ
                self.create_plots(col, options)