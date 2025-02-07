import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import warnings
from itertools import combinations
from scipy.stats import gaussian_kde


class DescriptiveStatistics:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.all_cols = df.columns.tolist()
        self.plot_template = "plotly_white"
        self.DEFAULT_MAX_DISTINCT_VALUES = 10  # Set the default value here

    def calculate_bin_width(self, data, method="sturges"):
        n = len(data)

        if n < 2:
            return 1

        if method == "sturges":
            n_bins = int(np.ceil(1 + np.log(n)))

        elif method == "scott":
            h = 3.49 * np.std(data, ddof=1) / (n ** (1/3))
            data_range = np.max(data) - np.min(data)
            n_bins = max(int(np.ceil(data_range / h)), 1)

        elif method == "freedman-diaconis":
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr / (n ** (1/3))
            if h == 0:
                h = 2 * np.std(data, ddof=1) / (n ** (1/3))
            data_range = np.max(data) - np.min(data)
            n_bins = max(int(np.ceil(data_range / h)), 1)

        elif method == "doane":
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
            self.all_cols,
            default=self.all_cols[:1] if self.all_cols else None
        )

        split_by_col = st.sidebar.selectbox(
            "Split by",
            [None] + self.categorical_cols,
            index=0,
            format_func=lambda x: "None" if x is None else x,
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
                show_sum = st.checkbox("Tổng")

                st.markdown("##### Phân tán")
                show_std = st.checkbox("Độ lệch chuẩn", value=True)
                show_variance = st.checkbox("Phương sai")
                show_range = st.checkbox("Khoảng")
                show_minimum = st.checkbox("Giá trị nhỏ nhất", value=True)
                show_maximum = st.checkbox("Giá trị lớn nhất", value=True)
                show_coefficient_of_variation = st.checkbox("Hệ số biến thiên")
                show_mad = st.checkbox("MAD")
                show_mad_robust = st.checkbox("MAD (Robust)")
                show_iqr = st.checkbox("IQR")

            with col2:
                st.markdown("##### Phân vị")
                show_quartiles = st.checkbox("Tứ phân vị")
                show_percentiles = st.checkbox("Phân vị cho:")
                if show_percentiles:
                    n_groups = st.number_input("Số nhóm bằng nhau", value=4)

                show_percentiles_values = st.checkbox("Percentiles cụ thể:")
                if show_percentiles_values:
                    percentile_values_str = st.text_input("Nhập giá trị percentiles (cách nhau bởi dấu phẩy)", value="25, 50, 75")
                    try:
                        percentile_values = [float(x.strip()) for x in percentile_values_str.split(",")]
                    except ValueError:
                        st.error("Giá trị Percentile không hợp lệ. Vui lòng nhập số, cách nhau bằng dấu phẩy.")
                        percentile_values = None  # Set to None to avoid errors later
                else:
                    percentile_values = None  # Set to None when not used

                st.markdown("##### Phân phối")
                show_skewness = st.checkbox("Độ lệch")
                show_kurtosis = st.checkbox("Độ nhọn")
                show_shapiro = st.checkbox("Kiểm định Shapiro-Wilk")
                show_se_mean = st.checkbox("S.E. mean")

                st.markdown("##### Suy luận")
                show_ci_mean = st.checkbox("KTC cho trung bình")
                if show_ci_mean:
                    ci_width_mean = st.number_input("Độ tin cậy (%)", value=95.0)
                    ci_method_mean = st.selectbox("Phương pháp KTC trung bình", ["T model", "Normal model", "Bootstrap"])
                    if ci_method_mean == "Bootstrap":
                        n_bootstrap_samples = st.number_input("Số lượng mẫu Bootstrap", value=1000, min_value=1, max_value=50000)
                else:
                    ci_width_mean = None
                    ci_method_mean = None
                    n_bootstrap_samples = None

                show_ci_sd = st.checkbox("KTC cho độ lệch chuẩn")
                if show_ci_sd:
                    ci_width_sd = st.number_input("Độ tin cậy (%) ", min_value=1.0, max_value=99.9, value=95.0, step=0.1, format="%.1f")
                    ci_method_sd = st.selectbox("Phương pháp KTC độ lệch chuẩn", ["Analytical (chi-square)", "Bootstrap"])
                    if ci_method_sd == "Bootstrap":
                        n_bootstrap_samples = st.number_input("Số lượng mẫu Bootstrap ", value=1000, min_value=1, max_value=50000)
                else:
                    ci_width_sd = None
                    ci_method_sd = None
                    n_bootstrap_samples = None

                show_ci_var = st.checkbox("KTC cho phương sai")
                if show_ci_var:
                    ci_width_var = st.number_input("Độ tin cậy (%)  ", min_value=1.0, max_value=99.9, value=95.0, step=0.1, format="%.1f")
                    ci_method_var = st.selectbox("Phương pháp KTC phương sai", ["Analytical (chi-square)", "Bootstrap"])
                    if ci_method_var == "Bootstrap":
                        n_bootstrap_samples = st.number_input("Số lượng mẫu Bootstrap   ", value=1000, min_value=1, max_value=50000)
                else:
                    ci_width_var = None
                    ci_method_var = None
                    n_bootstrap_samples = None

                st.markdown("##### Khác")
                statistics_values_are_group_midpoints = st.checkbox("Giá trị là điểm giữa của nhóm", value=False)

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
            show_box_plot = st.checkbox("Boxplot")
            if show_box_plot:
                box_plot_box_plot = st.checkbox("Hiển thị Boxplot Element", value=True)
                box_plot_violin = st.checkbox("Hiển thị Violin Element")
                box_plot_jitter = st.checkbox("Hiển thị Jitter Element")
                box_plot_colour_palette = st.checkbox("Sử dụng Bảng Màu")
                box_plot_outlier_label = st.checkbox("Hiển thị Label Outlier")

            show_stem_and_leaf = st.checkbox("Stem and leaf plots")
            if show_stem_and_leaf:
                stem_and_leaf_scale = st.number_input("Tỉ lệ Stem and Leaf", min_value=0.0, max_value=200.0, value=1.0)

            show_pareto_plot = st.checkbox("Pareto Plot")
            if show_pareto_plot:
                show_pareto_plot_rule = st.checkbox("Pareto Rule")
                if show_pareto_plot_rule:
                    pareto_plot_rule_ci = st.number_input("Ci cho Pareto Plot Rule", min_value=0.0, max_value=100.0, value=80.0)
                else:
                    pareto_plot_rule_ci = 80.0
            else:
                pareto_plot_rule_ci = 80.0

            show_likert_plot = st.checkbox("Likert Plot")
            if show_likert_plot:
                likert_plot_assume_variables_same_level = st.checkbox("Giả định các biến cùng level")
                likert_plot_adjustable_font_size = st.selectbox("Font size", ["normal", "small", "medium", "large"])
            else:
                likert_plot_assume_variables_same_level = False
                likert_plot_adjustable_font_size = "normal"

            show_frequency_tables = st.checkbox("Frequency tables")
            if show_frequency_tables:
                frequency_tables_maximum_distinct_values = st.number_input(
                    "Số lượng giá trị khác nhau tối đa",
                    min_value=1,
                    value=self.DEFAULT_MAX_DISTINCT_VALUES,  # Use the class attribute
                    max_value=200
                )
            else:
                frequency_tables_maximum_distinct_values = self.DEFAULT_MAX_DISTINCT_VALUES  # Use the class attribute

        with st.sidebar.expander("Ma trận tương quan", expanded=True):
            show_covariance = st.checkbox("Covariance")
            show_correlation = st.checkbox("Correlation")
            if show_covariance or show_correlation:
                association_matrix_use = st.selectbox(
                    "Cách xử lý giá trị thiếu",
                    ["Everything", "Complete observations", "Pairwise complete observations"]
                )
            else:
                association_matrix_use = "Everything"  # Set a default value

        # Additional scatter plot options
        with st.sidebar.expander("Scatter Plots", expanded=False):
            show_scatter_plot = st.checkbox("Scatter Plots")
            if show_scatter_plot:
                scatter_plot_graph_type_above = st.selectbox("Graph above scatter plot", ["Density", "Histogram", "None"])
                scatter_plot_graph_type_right = st.selectbox("Graph right of scatter plot", ["Density", "Histogram", "None"])
                scatter_plot_regression_line = st.checkbox("Add regression line")
                if scatter_plot_regression_line:
                    scatter_plot_regression_line_type = st.selectbox("Regression line type", ["Linear", "Smooth"])
                    scatter_plot_regression_line_ci = st.checkbox("Show confidence interval")
                    if scatter_plot_regression_line_ci:
                        scatter_plot_regression_line_ci_level = st.number_input("Confidence interval width", min_value=1.0, max_value=99.9, value=95.0, step=0.1, format="%.1f")

                scatter_plot_legend = st.checkbox("Show legend", value=True)

        # Additional density plot options
        with st.sidebar.expander("Density Plots", expanded=False):
            show_density_plot = st.checkbox("Density Plots")
            if show_density_plot:
                density_plot_separate = st.selectbox(
                    "Separate frequencies",
                    [None] + self.categorical_cols,
                    index=0,
                    format_func=lambda x: "None" if x is None else x,
                )
                density_plot_type = st.selectbox("Type for scale variables", ["Density", "Histogram"])
                if density_plot_type == "Histogram":
                     custom_histogram_position = st.selectbox("How to combine separate frequencies", ["Stack", "Identity", "Dodge"])
                else:
                    custom_histogram_position = "Stack"  # Provide a default value for custom_histogram_position

                density_plot_categorical_type = st.selectbox("Type for categorical variables", ["Counts", "Proportions", "Conditional proportions"])
                density_plot_transparency = st.number_input("Transparency", min_value=0, max_value=100, value=20)

        #Additional heatmap plot options
        with st.sidebar.expander("Heatmap Plots", expanded=False):
             show_heatmap_plot = st.checkbox("Tile heatmaps for selected variables")
             if show_heatmap_plot:
                heatmap_horizontal_axis = st.selectbox(
                    "Horizontal axis",
                    [None] + self.categorical_cols,
                    index=0,
                    format_func=lambda x: "None" if x is None else x,
                )
                heatmap_vertical_axis = st.selectbox(
                    "Vertical axis",
                    [None] + self.categorical_cols,
                    index=0,
                    format_func=lambda x: "None" if x is None else x,
                )
                heatmap_tile_width_height_ratio = st.number_input("Width to height ratio of tiles", value=1.0)
                heatmap_display_value = st.checkbox("Display value")
                if heatmap_display_value:
                   heatmap_statistic_continuous = st.selectbox("For scale variables", ["Mean", "Median", "Value itself", "Number of observations"])
                   heatmap_statistic_discrete = st.selectbox("For nominal and ordinal variables", ["Mode", "Value itself", "Number of observations"])
                   heatmap_display_value_relative_text_size = st.number_input("Relative text size", value=1.0)
                else:
                    heatmap_statistic_continuous = "Mean"
                    heatmap_statistic_discrete = "Mode"
                    heatmap_display_value_relative_text_size = 1.0

                heatmap_legend = st.checkbox("Display legend")
        return {
            'selected_cols': selected_cols,
            'split_by_col': split_by_col,
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
            'show_percentiles_values':show_percentiles_values,
            'percentile_values': percentile_values,
            'show_skewness': show_skewness,
            'show_kurtosis': show_kurtosis,
            'show_shapiro': show_shapiro,
            'show_sum': show_sum,
            'show_se_mean': show_se_mean,
            'show_coefficient_of_variation': show_coefficient_of_variation,
            'show_mad': show_mad,
            'show_mad_robust': show_mad_robust,
            'show_iqr': show_iqr,
            'show_ci_mean': show_ci_mean,
            'ci_width_mean': ci_width_mean,
            'ci_method_mean': ci_method_mean,
            'show_ci_sd': show_ci_sd,
            'ci_width_sd': ci_width_sd,
            'ci_method_sd': ci_method_sd,
            'show_ci_var': show_ci_var,
            'ci_width_var': ci_width_var,
            'ci_method_var': ci_method_var,
            'n_bootstrap_samples': n_bootstrap_samples,
            'statistics_values_are_group_midpoints': statistics_values_are_group_midpoints,
            'show_dist_plots': show_dist_plots,
            'show_density': show_density,
            'show_rug': show_rug,
            'bin_width_type': bin_width_type,
            'n_bins': n_bins,
            'show_qq_plots': show_qq_plots,
            'show_interval_plot': show_interval_plot,
            'show_pie_chart': show_pie_chart,
            'show_dot_plot': show_dot_plot,
            'show_box_plot': show_box_plot,
            'box_plot_box_plot': box_plot_box_plot if show_box_plot else False,
            'box_plot_violin': box_plot_violin if show_box_plot else False,
            'box_plot_jitter': box_plot_jitter if show_box_plot else False,
            'box_plot_colour_palette': box_plot_colour_palette if show_box_plot else False,
            'box_plot_outlier_label': box_plot_outlier_label if show_box_plot else False,
            'show_stem_and_leaf': show_stem_and_leaf,
            'stem_and_leaf_scale': stem_and_leaf_scale if show_stem_and_leaf else 1.0,
            'show_covariance': show_covariance,
            'show_correlation': show_correlation,
            'association_matrix_use': association_matrix_use if (show_covariance or show_correlation) else "Everything",  # Keep default even when not used
            'show_pareto_plot': show_pareto_plot,
            'show_pareto_plot_rule': show_pareto_plot_rule if show_pareto_plot else False,
            'pareto_plot_rule_ci': pareto_plot_rule_ci if show_pareto_plot and show_pareto_plot_rule else 80.0,
            'show_likert_plot': show_likert_plot,
            'likert_plot_assume_variables_same_level': likert_plot_assume_variables_same_level if show_likert_plot else False,
            'likert_plot_adjustable_font_size': likert_plot_adjustable_font_size if show_likert_plot else "normal",
            'show_frequency_tables': show_frequency_tables,
            'frequency_tables_maximum_distinct_values': frequency_tables_maximum_distinct_values if show_frequency_tables else self.DEFAULT_MAX_DISTINCT_VALUES,
            'show_scatter_plot': show_scatter_plot,
            'scatter_plot_graph_type_above': scatter_plot_graph_type_above if show_scatter_plot else "Density",
            'scatter_plot_graph_type_right': scatter_plot_graph_type_right if show_scatter_plot else "Density",
            'scatter_plot_regression_line': scatter_plot_regression_line if show_scatter_plot else False,
            'scatter_plot_regression_line_type': scatter_plot_regression_line_type if show_scatter_plot and scatter_plot_regression_line else "Linear",
            'scatter_plot_regression_line_ci': scatter_plot_regression_line_ci if show_scatter_plot and scatter_plot_regression_line else False,
            'scatter_plot_regression_line_ci_level': scatter_plot_regression_line_ci_level if show_scatter_plot and scatter_plot_regression_line and scatter_plot_regression_line_ci else 95.0,
            'scatter_plot_legend': scatter_plot_legend if show_scatter_plot else False,
            'show_density_plot': show_density_plot,
            'density_plot_separate': density_plot_separate if show_density_plot else None,
            'density_plot_type': density_plot_type if show_density_plot else "Density",
            'custom_histogram_position': custom_histogram_position if show_density_plot and density_plot_type == "Histogram" else "Stack",
            'density_plot_categorical_type': density_plot_categorical_type if show_density_plot else "Counts",
            'density_plot_transparency': density_plot_transparency if show_density_plot else 20,
            'show_heatmap_plot': show_heatmap_plot,
            'heatmap_horizontal_axis': heatmap_horizontal_axis if show_heatmap_plot else None,
            'heatmap_vertical_axis': heatmap_vertical_axis if show_heatmap_plot else None,
            'heatmap_tile_width_height_ratio': heatmap_tile_width_height_ratio if show_heatmap_plot else 1.0,
            'heatmap_display_value': heatmap_display_value if show_heatmap_plot else False,
            'heatmap_statistic_continuous': heatmap_statistic_continuous if show_heatmap_plot and heatmap_display_value else "Mean",
            'heatmap_statistic_discrete': heatmap_statistic_discrete if show_heatmap_plot and heatmap_display_value else "Mode",
            'heatmap_display_value_relative_text_size': heatmap_display_value_relative_text_size if show_heatmap_plot and heatmap_display_value else 1.0,
            'heatmap_legend': heatmap_legend if show_heatmap_plot else False
        }
    def calculate_statistics(self, var, options):
        stats_dict = {}
        data = self.df[var].dropna()

        if options['statistics_values_are_group_midpoints']:
           warnings.warn("Giá trị là điểm giữa của nhóm đã được bật. Tính toán có thể không chính xác.")

        if options['show_valid']:
            stats_dict["Kích thước mẫu - Hợp lệ"] = len(data)
        if options['show_missing']:
            stats_dict["Kích thước mẫu - Thiếu"] = self.df[var].isnull().sum()
        if options['show_mean']:
            stats_dict["Xu hướng trung tâm - Trung bình"] = data.mean()
        if options['show_median']:
            stats_dict["Xu hướng trung tâm - Trung vị"] = data.median()
        if options['show_mode']:
            try:
                stats_dict["Xu hướng trung tâm - Mode"] = data.mode().iloc[0]
            except IndexError:
                stats_dict["Xu hướng trung tâm - Mode"] = np.nan  # Handle case where no mode exists
        if options['show_sum']:
            stats_dict["Xu hướng trung tâm - Tổng"] = data.sum()
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
        if options['show_coefficient_of_variation']:
            stats_dict["Phân tán - Hệ số biến thiên"] = data.std() / data.mean() if data.mean() != 0 else np.nan
        if options['show_mad']:
            stats_dict["Phân tán - MAD"] = data.mad()
        if options['show_mad_robust']:
            stats_dict["Phân tán - MAD (Robust)"] = stats.median_abs_deviation(data)
        if options['show_iqr']:
            stats_dict["Phân tán - IQR"] = stats.iqr(data)
        if options['show_skewness']:
            stats_dict["Phân phối - Độ lệch"] = stats.skew(data)
        if options['show_kurtosis']:
            stats_dict["Phân phối - Độ nhọn"] = stats.kurtosis(data)
        if options['show_shapiro']:
            try:
                _, p_value = stats.shapiro(data)
                stats_dict["Kiểm định - Shapiro-Wilk p-value"] = p_value
            except ValueError:
                stats_dict["Kiểm định - Shapiro-Wilk p-value"] = "Không đủ dữ liệu"
        if options['show_se_mean']:
            stats_dict["Suy luận - S.E. mean"] = stats.sem(data)
        if options['show_ci_mean']:
            alpha = 1 - options['ci_width_mean']/100
            if options['ci_method_mean'] == "T model":
                mean = data.mean()
                sem = stats.sem(data)
                ci = stats.t.interval(1-alpha, len(data)-1, loc=mean, scale=sem)
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình - Dưới"] = ci[0]
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình - Trên"] = ci[1]
            elif options['ci_method_mean'] == "Normal model":
                mean = data.mean()
                sem = stats.sem(data)
                z = stats.norm.ppf(1-alpha/2)
                ci_lower = mean - z * sem
                ci_upper = mean + z * sem
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình - Dưới"] = ci_lower
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình - Trên"] = ci_upper
            elif options['ci_method_mean'] == "Bootstrap":
                n_bootstrap_samples = options['n_bootstrap_samples']
                bootstrap_means = [np.random.choice(data, size=len(data), replace=True).mean() for _ in range(n_bootstrap_samples)]
                ci_lower = np.percentile(bootstrap_means, alpha/2*100)
                ci_upper = np.percentile(bootstrap_means, (1-alpha/2)*100)
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình (Bootstrap) - Dưới"] = ci_lower
                stats_dict[f"Suy luận - {options['ci_width_mean']}% KTC Trung bình (Bootstrap) - Trên"] = ci_upper
        if options['show_ci_sd']:
            n = len(data)
            alpha = 1 - options['ci_width_sd']/100
            if options['ci_method_sd'] == "Analytical (chi-square)":
                chi2_lower = stats.chi2.ppf(alpha/2, n-1)
                chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)
                stats_dict[f"Suy luận - {options['ci_width_sd']}% KTC Độ lệch chuẩn - Trên"] = np.sqrt((n-1) * data.var() / chi2_lower)
                stats_dict[f"Suy luận - {options['ci_width_sd']}% KTC Độ lệch chuẩn - Dưới"] = np.sqrt((n-1) * data.var() / chi2_upper)
            elif options['ci_method_sd'] == "Bootstrap":
                n_bootstrap_samples = options['n_bootstrap_samples']
                bootstrap_sds = [np.random.choice(data, size=len(data), replace=True).std() for _ in range(n_bootstrap_samples)]
                ci_lower = np.percentile(bootstrap_sds, alpha/2*100)
                ci_upper = np.percentile(bootstrap_sds, (1-alpha/2)*100)
                stats_dict[f"Suy luận - {options['ci_width_sd']}% KTC Độ lệch chuẩn (Bootstrap) - Dưới"] = ci_lower
                stats_dict[f"Suy luận - {options['ci_width_sd']}% KTC Độ lệch chuẩn (Bootstrap) - Trên"] = ci_upper
        if options['show_ci_var']:
            n = len(data)
            alpha = 1 - options['ci_width_var']/100
            if options['ci_method_var'] == "Analytical (chi-square)":
                chi2_lower = stats.chi2.ppf(alpha/2, n-1)
                chi2_upper = stats.chi2.ppf(1-alpha/2, n-1)
                stats_dict[f"Suy luận - {options['ci_width_var']}% KTC Phương sai - Trên"] = (n-1) * data.var() / chi2_lower
                stats_dict[f"Suy luận - {options['ci_width_var']}% KTC Phương sai - Dưới"] = (n-1) * data.var() / chi2_upper
            elif options['ci_method_var'] == "Bootstrap":
                n_bootstrap_samples = options['n_bootstrap_samples']
                bootstrap_vars = [np.random.choice(data, size=len(data), replace=True).var() for _ in range(n_bootstrap_samples)]
                ci_lower = np.percentile(bootstrap_vars, alpha/2*100)
                ci_upper = np.percentile(bootstrap_vars, (1-alpha/2)*100)
                stats_dict[f"Suy luận - {options['ci_width_var']}% KTC Phương sai (Bootstrap) - Dưới"] = ci_lower
                stats_dict[f"Suy luận - {options['ci_width_var']}% KTC Phương sai (Bootstrap) - Trên"] = ci_upper

        if options['show_quartiles']:
            quartiles = data.quantile([0.25, 0.5, 0.75])
            stats_dict["Phân vị - Q1"] = quartiles[0.25]
            stats_dict["Phân vị - Q2 (Trung vị)"] = quartiles[0.5]
            stats_dict["Phân vị - Q3"] = quartiles[0.75]
        if options['show_percentiles']:
             if options['n_groups']:
                quantiles = [i / options['n_groups'] for i in range(1, options['n_groups'])]
                percentile_values = data.quantile(quantiles).to_dict()
                for q, v in zip(quantiles, data.quantile(quantiles)):
                  stats_dict[f"Phân vị - {int(q*100)}%"] = v

        if options['show_percentiles_values']:
            if options['percentile_values']:
                for p in options['percentile_values']:
                    try:
                        stats_dict[f"Phân vị - {p}%"] = data.quantile(p/100)
                    except:
                        print("không có giá trị percentile này")

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

    def create_distribution_plot(self, var, options):
        data = self.df[var].dropna()

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

    def create_plots(self, var, options):
        data = self.df[var].dropna()

        if options['show_dist_plots']:
            self.create_distribution_plot(var, options)

        # Additional plots
        if options['show_qq_plots']:
            self.create_qq_plot(var)
        if options['show_interval_plot']:
            self.create_interval_plot(var)
        if options['show_pie_chart']:
            self.create_pie_chart(var)
        if options['show_dot_plot']:
            self.create_dot_plot(var)
        if options['show_box_plot']:
            self.create_box_plot(var,options)

    def create_box_plot(self, var, options):
        data = self.df[var].dropna()
        fig = go.Figure()

        if options['box_plot_violin']:
            fig.add_trace(go.Violin(y=data, name=var, box_visible=options['box_plot_box_plot'],
                                    meanline_visible=True, points="all" if options['box_plot_jitter'] else False))
        elif options['box_plot_box_plot']:
            fig.add_trace(go.Box(y=data, name=var, points="all" if options['box_plot_jitter'] else False))

        fig.update_layout(title=f"Biểu đồ Boxplot - {var}", template=self.plot_template)

        st.plotly_chart(fig)

    def create_stem_and_leaf_plot(self, var, options):
        data = self.df[var].dropna()

        # Create stem and leaf plot manually
        stem = data // (10 ** (np.floor(np.log10(data.abs().max())) - 1))
        leaf = data % (10 ** (np.floor(np.log10(data.abs().max())) - 1))

        # Create a DataFrame for better visualization
        stem_leaf_df = pd.DataFrame({'stem': stem, 'leaf': leaf})
        stem_leaf_df = stem_leaf_df.sort_values(by=['stem', 'leaf'])

        st.write(f"Biểu đồ Stem and Leaf - {var}")
        st.write("Lưu ý: Biểu đồ stem and leaf này được tạo thủ công và có thể không chính xác như các thư viện chuyên dụng.")

        for stem_val in sorted(stem_leaf_df['stem'].unique()):
            leaves = stem_leaf_df[stem_leaf_df['stem'] == stem_val]['leaf'].astype(str).str[:1].tolist()  # Keep only the first digit of the leaf
            st.write(f"{stem_val} | {' '.join(leaves)}")

    def create_covariance_correlation_matrix(self,selected_cols, options):
        # Tính ma trận hiệp phương sai hoặc tương quan
        if options['show_covariance'] or options['show_correlation']:
            method = options['association_matrix_use']

            if method == "Everything":
                data = self.df[selected_cols]
            elif method == "Complete observations":
                data = self.df[selected_cols].dropna()
            elif method == "Pairwise complete observations":
                data = self.df[selected_cols]
            else:
                st.error(f"Phương pháp '{method}' không được hỗ trợ.")
                return

            if options['show_covariance']:
                try:
                    covariance_matrix = data.cov(method=method)
                    st.write("Ma trận Hiệp phương sai:")
                    st.dataframe(covariance_matrix)
                except Exception as e:
                    st.error(f"Lỗi khi tính ma trận hiệp phương sai: {e}")

            if options['show_correlation']:
                try:
                    correlation_matrix = data.corr(method='pearson')
                    st.write("Ma trận Tương quan:")
                    st.dataframe(correlation_matrix)
                except Exception as e:
                    st.error(f"Lỗi khi tính ma trận tương quan: {e}")

    def create_pareto_plot(self, var, options):
        data = self.df[var].dropna()
        value_counts = data.value_counts().sort_values(ascending=False)
        cumulative_percentage = (value_counts.cumsum() / value_counts.sum()) * 100

        fig = go.Figure()

        # Bar chart for value counts
        # Ensure the labels and values are distinct
        labels = [str(x) for x in value_counts.index] # Convert labels to strings to ensure uniqueness.

        fig.add_trace(go.Bar(
            x=labels, #Use the string converted labels
            y=value_counts.values,
            name="Tần số",
            marker_color='rgb(158,202,225)'
        ))

        # Line chart for cumulative percentage
        fig.add_trace(go.Scatter(
            x=labels,  # Use the same labels as the bar chart
            y=cumulative_percentage.values,
            name="Tích lũy (%)",
            yaxis="y2",
            marker_color='rgb(231,138,195)'
        ))

        # Update layout for dual y-axis
        fig.update_layout(
            title=f"Biểu đồ Pareto - {var}",
            xaxis_title=var,
            yaxis_title="Tần số",
            yaxis2=dict(
                title="Tích lũy (%)",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            template=self.plot_template
        )

        # Add Pareto Rule lines
        if options['show_pareto_plot_rule']:
            rule_percentage = options['pareto_plot_rule_ci']
            fig.add_trace(go.Scatter(
                x=[labels[0], labels[-1]],  # Extend line across entire x-axis  Use labels[0] and labels[-1]
                y=[rule_percentage, rule_percentage],
                mode="lines",
                name=f"Quy tắc Pareto ({rule_percentage}%)",
                line=dict(color='red', dash='dash'),
                yaxis="y2"
            ))
        st.plotly_chart(fig)

    def create_likert_plot(self, var, options):
        data = self.df[var].dropna()
        value_counts = data.value_counts(normalize=True).sort_index() * 100  # Normalize to proportions and convert to percentage

        # Create stacked bar chart
        fig = go.Figure()

        # Add bars for each level
        # Check if levels (index) and percentages (values) are the same
        for level, percentage in value_counts.items():
            level_str = str(level) # Ensure level is a string for labels
            fig.add_trace(go.Bar(
                y=[var],
                x=[percentage],
                name=level_str, # Use level_str
                orientation='h',
                text=[f"{percentage:.1f}%"],
                textposition='inside'
            ))

        # Update layout
        fig.update_layout(
            title=f"Biểu đồ Likert - {var}",
            xaxis_title="Phần trăm",
            yaxis_title="Biến",
            barmode='stack',
            template=self.plot_template
        )

        st.plotly_chart(fig)

    def create_frequency_tables(self, var, options):
        data = self.df[var].dropna()
        max_distinct_values = options['frequency_tables_maximum_distinct_values']

        # Check the number of distinct values to avoid overwhelming the output
        if data.nunique() > max_distinct_values:
            st.warning(f"Biến '{var}' có quá nhiều giá trị khác nhau ({data.nunique()}). Bảng tần số không được hiển thị.")
            return

        st.write(f"Bảng tần số - {var}:")
        # Ensure the index and values are distinct when creating the DataFrame.

        frequency_table = pd.DataFrame({
            'Tần số': data.value_counts(),
            'Phần trăm': data.value_counts(normalize=True) * 100
        })
        st.dataframe(frequency_table)

    def create_scatter_plot(self, var1, var2, options, split_by_col=None):
        data = self.df[[var1, var2]].dropna()

        if split_by_col:
            data = data.join(self.df[split_by_col], how="inner")  # Ensure that there are not na in the split col too
            fig = px.scatter(data, x=var1, y=var2, color=split_by_col, trendline="ols" if options['scatter_plot_regression_line'] else None)
        else:
            fig = px.scatter(data, x=var1, y=var2, trendline="ols" if options['scatter_plot_regression_line'] else None)

        fig.update_layout(title=f"Scatter plot - {var1} vs {var2}", template=self.plot_template)
        st.plotly_chart(fig)

    def create_density_plot(self, var, options, split_by_col=None):
        data = self.df[var].dropna()
        if split_by_col:
            data = data.to_frame(name=var).join(self.df[split_by_col], how="inner")
            fig = px.histogram(data, x=var, color=split_by_col, marginal="rug",
                                histnorm='density' if options['density_plot_type'] == 'Density' else None)  # or "probability density"
            fig.update_layout(barmode = "overlay", title=f"Density plot - {var}")

        else:
            fig = px.histogram(self.df, x=var, histnorm='density' if options['density_plot_type'] == 'Density' else None, marginal="rug")  # or "probability density"
            fig.update_layout(title=f"Density plot - {var}")

        st.plotly_chart(fig)
    def create_heatmap_plot(self, var1, var2, options):
        #TODO
        st.write("Chức năng heatmap đang được phát triển.")

    def descriptive_statistics_analysis(self,df):
        self.df = df
        st.subheader("Phân tích thống kê mô tả")
        options = self.sidebar_options()
        if not options['selected_cols']:
            st.warning("Vui lòng chọn ít nhất một biến để phân tích.")
            return
        if options['split_by_col'] and options['split_by_col'] not in self.categorical_cols:
             st.warning("Bạn phải lựa chọn trường split là trường giá trị rời rạc (Categorical)")
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
                if col in self.numeric_cols:
                   self.create_plots(col, options)
                   if options['show_stem_and_leaf']:
                      self.create_stem_and_leaf_plot(col, options)
                   if options['show_density_plot']:
                       self.create_density_plot(col, options, options['density_plot_separate'])
                elif col in self.categorical_cols:
                   if options['show_pareto_plot']:
                      self.create_pareto_plot(col, options)
                   if options['show_likert_plot']:
                      self.create_likert_plot(col, options)
                if options['show_frequency_tables']:
                   self.create_frequency_tables(col, options)

        # Association matrix
        numeric_selected_cols = [col for col in options['selected_cols'] if col in self.numeric_cols]
        if len(numeric_selected_cols) >= 2 and (options['show_covariance'] or options['show_correlation']):
            self.create_covariance_correlation_matrix(numeric_selected_cols, options)

        # Scatter plots
        if options['show_scatter_plot'] and len(numeric_selected_cols) >= 2:
           for var1, var2 in combinations(numeric_selected_cols, 2):
              self.create_scatter_plot(var1, var2, options, options['split_by_col'])

        # Heatmap plots
        if options['show_heatmap_plot'] and options['heatmap_horizontal_axis'] and options['heatmap_vertical_axis']:
           self.create_heatmap_plot(options['heatmap_horizontal_axis'], options['heatmap_vertical_axis'], options)