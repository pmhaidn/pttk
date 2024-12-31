import numpy as np
from scipy import stats
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import shapiro, levene, mannwhitneyu
import plotly.figure_factory as ff

class HypothesisTesting:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.plot_template = "plotly_white"

    def setup_interface(self):
        """Setup interface and return options"""
        options = {}
        
        # Left sidebar
        with st.sidebar:
            st.header("Kiểm Định")
            options['test_method'] = st.checkbox("Student", value=True)
            options['welch_test'] = st.checkbox("Welch", value=True)
            options['mann_whitney'] = st.checkbox("Mann-Whitney")

            st.header("Giả Thuyết Thay Thế")
            options['alternative'] = st.radio(
                "",
                ["Nhóm 1 ≠ Nhóm 2", "Nhóm 1 > Nhóm 2", "Nhóm 1 < Nhóm 2"],
                index=0
            )

            st.header("Kiểm Tra Giả Định")
            options['normality'] = st.checkbox("Phân phối chuẩn")
            options['equality_of_variances'] = st.checkbox("Phương sai đồng nhất")
            if options['equality_of_variances']:
                options['variance_test'] = st.radio(
                    "",
                    ["Brown-Forsythe", "Levene"],
                    horizontal=True
                )
            options['qq_plot'] = st.checkbox("Biểu đồ Q-Q")

            st.header("Giá Trị Thiếu")
            options['missing_values'] = st.radio(
                "",
                ["Loại bỏ theo từng biến phụ thuộc", "Loại trừ các trường hợp theo danh sách"],
                index=0
            )

        # Main panel
        st.title("Kiểm Định T Độc Lập")
        
        # Dependent Variables (multiselect)
        st.subheader("Biến Phụ Thuộc")
        options['dependent_vars'] = st.multiselect(
            "",
            self.numeric_cols
        )
        
        # Grouping Variable
        st.subheader("Biến Phân Nhóm")
        options['grouping_var'] = st.selectbox("", self.df.columns)

        # Additional Statistics
        st.header("Thống Kê Bổ Sung")
        options['location_parameter'] = st.checkbox("Tham số vị trí")
        
        col1, col2 = st.columns(2)
        with col1:
            options['confidence_level'] = st.number_input(
                "Khoảng tin cậy",
                min_value=1.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                format="%.1f"
            )
            st.write("%")
        
        options['effect_size'] = st.checkbox("Kích thước hiệu ứng")
        if options['effect_size']:
            options['cohens_d'] = st.checkbox("Cohen's d")
            options['glass_delta'] = st.checkbox("Glass' delta")
            options['hedges_g'] = st.checkbox("Hedges' g")
            
            col1, col2 = st.columns(2)
            with col1:
                options['effect_size_ci'] = st.number_input(
                    "Khoảng tin cậy",
                    min_value=1.0,
                    max_value=99.9,
                    value=95.0,
                    step=0.1,
                    format="%.1f",
                    key="effect_size_ci"
                )
                st.write("%")

        options['descriptives'] = st.checkbox("Thống kê mô tả")
        options['vovk_sellke'] = st.checkbox("Tỉ số p tối đa Vovk-Sellke")

        # Plots
        st.header("Biểu Đồ")
        options['descriptive_plots'] = st.checkbox("Biểu đồ mô tả")
        if options['descriptive_plots']:
            options['desc_plot_ci'] = st.number_input(
                "Khoảng tin cậy",
                min_value=1.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                format="%.1f",
                key="desc_plot_ci"
            )
            st.write("%")

        options['raincloud_plots'] = st.checkbox("Biểu đồ Raincloud")
        if options['raincloud_plots']:
            options['horizontal_display'] = st.checkbox("Hiển thị ngang")

        options['bar_plots'] = st.checkbox("Biểu đồ cột")
        if options['bar_plots']:
            options['bar_plot_ci'] = st.number_input(
                "Khoảng tin cậy",
                min_value=1.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                format="%.1f",
                key="bar_plot_ci"
            )
            st.write("%")
            options['standard_error'] = st.checkbox("Sai số chuẩn")
        
        return options

    def run_analysis(self, options):
        """Run analysis based on selected options"""
        if not options['dependent_vars']:
            st.warning("Vui lòng chọn ít nhất một biến phụ thuộc")
            return
            
        results = []
        assumption_results = {}
        
        for dep_var in options['dependent_vars']:
            # Check for missing values
            if options['missing_values'] == "Loại bỏ theo từng biến phụ thuộc":
                valid_data = self.df[[dep_var, options['grouping_var']]].dropna()
            else:
                valid_data = self.df.dropna()
                
            result = self._analyze_variable(dep_var, options, valid_data)
            if result:
                results.append(result)
                
                # Run assumption checks if requested
                if options['normality'] or options['equality_of_variances']:
                    assumption_results[dep_var] = self._check_assumptions(
                        valid_data[dep_var],
                        valid_data[options['grouping_var']],
                        options
                    )
                
        if results:
            self._display_results_table(results)
            
            if assumption_results:
                self._display_assumption_results(assumption_results)
            
            if options.get('descriptives'):
                self._display_descriptives(options)
                
            if options.get('effect_size'):
                self._display_effect_sizes(results, options)
                
            # Display plots
            if options.get('descriptive_plots'):
                self._plot_descriptives(options)
            if options.get('raincloud_plots'):
                self._plot_raincloud(options)
            if options.get('bar_plots'):
                self._plot_bars(options)
            if options.get('qq_plot'):
                self._plot_qq(options)

    def _analyze_variable(self, dep_var, options, data):
        """Analyze a single dependent variable"""
        group_data = data.groupby(options['grouping_var'])[dep_var].apply(list)
        if len(group_data) != 2:
            st.error(f"Biến phân nhóm phải có đúng hai nhóm cho {dep_var}")
            return None
            
        results = {'variable': dep_var}
        
        if options['test_method']:
            student_results = self._run_student_ttest(group_data, options)
            results.update({'Student': student_results})
            
        if options.get('welch_test'):
            welch_results = self._run_welch_ttest(group_data, options)
            results.update({'Welch': welch_results})
            
        if options.get('mann_whitney'):
            mw_results = self._run_mann_whitney(group_data, options)
            results.update({'Mann-Whitney': mw_results})
            
        return results

    def _check_assumptions(self, dep_var, group_var, options):
        """Check normality and variance assumptions"""
        results = {}
        
        # Convert input series to DataFrame if needed
        df = pd.DataFrame({
            'dependent': dep_var,
            'group': group_var
        })
        
        # Get data for each group
        groups = []
        group_names = []
        
        for name, group in df.groupby('group'):
            groups.append(group['dependent'].values)
            group_names.append(name)
        
        if options['normality']:
            # Shapiro-Wilk test for each group
            normality_results = []
            for i, (name, data) in enumerate(zip(group_names, groups)):
                stat, p_val = shapiro(data)
                normality_results.append({
                    'Nhóm': str(name),
                    'Thống kê': stat,
                    'Giá trị p': p_val
                })
            results['normality'] = normality_results
            
        if options['equality_of_variances'] and len(groups) >= 2:
            # Levene's test or Brown-Forsythe
            if options['variance_test'] == "Levene":
                stat, p_val = levene(*groups)
            else:  # Brown-Forsythe
                stat, p_val = stats.levene(*groups, center='median')
            
            results['variance'] = {
                'test': options['variance_test'],
                'statistic': stat,
                'p_value': p_val
            }
            
        return results

    def _run_student_ttest(self, group_data, options):
        """Run Student's t-test"""
        group1, group2 = group_data.values
        alternative = self._get_alternative(options['alternative'])
        t_stat, p_val = stats.ttest_ind(group1, group2, alternative=alternative)
        
        ci = self._calculate_ci(group1, group2, options['confidence_level'])
        
        return {
            't_statistic': t_stat,
            'p_value': p_val,
            'df': len(group1) + len(group2) - 2,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }

    def _run_welch_ttest(self, group_data, options):
        """Run Welch's t-test"""
        group1, group2 = group_data.values
        alternative = self._get_alternative(options['alternative'])
        t_stat, p_val = stats.ttest_ind(group1, group2, alternative=alternative, equal_var=False)
        
        # Calculate Welch-Satterthwaite degrees of freedom
        n1, n2 = len(group1), len(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        ci = self._calculate_ci(group1, group2, options['confidence_level'], equal_var=False)
        
        return {
            't_statistic': t_stat,
            'p_value': p_val,
            'df': df,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }

    def _run_mann_whitney(self, group_data, options):
        """Run Mann-Whitney U test"""
        group1, group2 = group_data.values
        alternative = self._get_alternative(options['alternative'])
        stat, p_val = mannwhitneyu(group1, group2, alternative=alternative)
        
        return {
            'statistic': stat,
            'p_value': p_val,
            'df': None
        }

    def _calculate_ci(self, group1, group2, confidence_level, equal_var=True):
        """Calculate confidence interval for mean difference"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if equal_var:
            # Pooled variance for Student's t-test
            df = n1 + n2 - 2
            s_pooled = np.sqrt(((n1-1)*np.var(group1, ddof=1) + 
                              (n2-1)*np.var(group2, ddof=1)) / df)
            se = s_pooled * np.sqrt(1/n1 + 1/n2)
        else:
            # Welch's t-test
            v1 = np.var(group1, ddof=1)
            v2 = np.var(group2, ddof=1)
            se = np.sqrt(v1/n1 + v2/n2)
            df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        t_crit = stats.t.ppf((1 + confidence_level/100)/2, df)
        mean_diff = mean1 - mean2
        
        return mean_diff - t_crit*se, mean_diff + t_crit*se

    def _calculate_effect_sizes(self, group1, group2):
        """Calculate various effect size measures"""
        effect_sizes = {}
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        n1 = len(group1)
        n2 = len(group2)
        sd1 = np.std(group1, ddof=1)
        sd2 = np.std(group2, ddof=1)
        
        # Cohen's d
        pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        effect_sizes['cohens_d'] = (mean1 - mean2) / pooled_sd
        
        # Glass' delta
        effect_sizes['glass_delta'] = (mean1 - mean2) / sd2
        
        # Hedges' g
        correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        effect_sizes['hedges_g'] = correction * effect_sizes['cohens_d']
        
        return effect_sizes

    def _display_results_table(self, results):
        """Display results in a formatted table"""
        rows = []
        for result in results:
            var_name = result['variable']
            for test_type, stats in result.items():
                if test_type != 'variable':
                    rows.append({
                        'Biến': var_name,
                        'Kiểm định': test_type,
                        'Thống kê': f"{stats.get('t_statistic', stats.get('statistic', 0)):.3f}",
                        'Bậc tự do': f"{stats.get('df', ''):.3f}" if stats.get('df') is not None else '',
                        'p': f"{stats['p_value']:.3f}"
                    })
        
        df = pd.DataFrame(rows)
        st.write("### Kết Quả Kiểm Định")
        st.write(df)

    def _display_assumption_results(self, results):
        """Display assumption check results"""
        st.write("### Kiểm Tra Giả Định")
        
        for var_name, var_results in results.items():
            st.write(f"#### {var_name}")
            
            if 'normality' in var_results:
                st.write("Kiểm định phân phối chuẩn (Shapiro-Wilk):")
                norm_df = pd.DataFrame(var_results['normality'])
                st.write(norm_df)
                
            if 'variance' in var_results:
                var_test = var_results['variance']
                st.write(f"Kiểm định {var_test['test']} cho phương sai đồng nhất:")
                st.write(f"Thống kê: {var_test['statistic']:.3f}")
                st.write(f"Giá trị p: {var_test['p_value']:.3f}")

    def _display_descriptives(self, options):
        """Display descriptive statistics"""
        st.write("### Thống Kê Mô Tả")
        
        for dep_var in options['dependent_vars']:
            stats_df = self.df.groupby(options['grouping_var'])[dep_var].agg([
                'count',
                'mean',
                'std',
                lambda x: x.skew(),
                lambda x: x.kurtosis()
            ]).round(3)
            
            stats_df.columns = ['N', 'Trung bình', 'Độ lệch chuẩn', 'Độ lệch', 'Độ nhọn']
            st.write(f"#### {dep_var}")
            st.write(stats_df)

    def _display_effect_sizes(self, results, options):
        """Display effect size measures"""
        if not any([options.get('cohens_d'), options.get('glass_delta'), options.get('hedges_g')]):
            return
            
        st.write("### Kích Thước Hiệu Ứng")
        
        effect_size_rows = []
        for result in results:
            var_name = result['variable']
            for test_type, stats in result.items():
                if test_type == 'Student':
                    group_data = self.df.groupby(options['grouping_var'])[var_name]
                    groups = [group for _, group in group_data]
                    if len(groups) == 2:
                        effect_sizes = self._calculate_effect_sizes(groups[0], groups[1])
                        
                        row = {'Biến': var_name}
                        if options.get('cohens_d'):
                            row["Cohen's d"] = f"{effect_sizes['cohens_d']:.3f}"
                        if options.get('glass_delta'):
                            row["Glass' delta"] = f"{effect_sizes['glass_delta']:.3f}"
                        if options.get('hedges_g'):
                            row["Hedges' g"] = f"{effect_sizes['hedges_g']:.3f}"
                            
                        effect_size_rows.append(row)
        
        if effect_size_rows:
            df = pd.DataFrame(effect_size_rows)
            st.write(df)

    def _plot_descriptives(self, options):
        """Create descriptive plots"""
        for dep_var in options['dependent_vars']:
            fig = go.Figure()
            
            stats = self.df.groupby(options['grouping_var'])[dep_var].agg([
                'mean',
                'std',
                'count'
            ])
            
            ci_level = options['desc_plot_ci'] / 100
            z_score = stats.norm.ppf((1 + ci_level) / 2)
            
            stats['ci_lower'] = stats['mean'] - z_score * stats['std'] / np.sqrt(stats['count'])
            stats['ci_upper'] = stats['mean'] + z_score * stats['std'] / np.sqrt(stats['count'])
            
            fig.add_trace(go.Box(
                x=self.df[options['grouping_var']],
                y=self.df[dep_var],
                name='Biểu đồ hộp',
                boxpoints='outliers'
            ))
            
            fig.add_trace(go.Scatter(
                x=stats.index,
                y=stats['mean'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=stats['ci_upper'] - stats['mean'],
                    arrayminus=stats['mean'] - stats['ci_lower']
                ),
                mode='markers',
                marker=dict(size=10),
                name=f'Khoảng tin cậy {options["desc_plot_ci"]}%'
            ))
            
            fig.update_layout(
                title=f'Biểu đồ mô tả: {dep_var}',
                xaxis_title=options['grouping_var'],
                yaxis_title=dep_var,
                template=self.plot_template
            )
            
            st.plotly_chart(fig)

    def _plot_raincloud(self, options):
        """Create raincloud plots"""
        for dep_var in options['dependent_vars']:
            fig = go.Figure()
            
            orientation = 'h' if options.get('horizontal_display') else 'v'
            
            for group in self.df[options['grouping_var']].unique():
                group_data = self.df[self.df[options['grouping_var']] == group][dep_var]
                
                if orientation == 'v':
                    fig.add_trace(go.Violin(
                        y=group_data,
                        name=str(group),
                        side='positive',
                        points='all',
                        pointpos=-0.5,
                        jitter=0.3,
                        box_visible=True,
                        meanline_visible=True
                    ))
                else:
                    fig.add_trace(go.Violin(
                        x=group_data,
                        name=str(group),
                        side='positive',
                        points='all',
                        pointpos=-0.5,
                        jitter=0.3,
                        box_visible=True,
                        meanline_visible=True
                    ))
            
            title = f'Biểu đồ Raincloud: {dep_var}'
            if orientation == 'v':
                fig.update_layout(
                    title=title,
                    xaxis_title=options['grouping_var'],
                    yaxis_title=dep_var,
                    violinmode='overlay',
                    template=self.plot_template
                )
            else:
                fig.update_layout(
                    title=title,
                    yaxis_title=options['grouping_var'],
                    xaxis_title=dep_var,
                    violinmode='overlay',
                    template=self.plot_template
                )
            
            st.plotly_chart(fig)

    def _plot_bars(self, options):
        """Create bar plots with error bars"""
        for dep_var in options['dependent_vars']:
            stats = self.df.groupby(options['grouping_var'])[dep_var].agg([
                'mean',
                'std',
                'count'
            ])
            
            if options.get('standard_error'):
                error = stats['std'] / np.sqrt(stats['count'])
            else:
                ci_level = options['bar_plot_ci'] / 100
                z_score = stats.norm.ppf((1 + ci_level) / 2)
                error = z_score * stats['std'] / np.sqrt(stats['count'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=stats.index,
                y=stats['mean'],
                error_y=dict(
                    type='data',
                    array=error,
                    visible=True
                ),
                name=dep_var
            ))
            
            fig.update_layout(
                title=f'Biểu đồ cột: {dep_var}',
                xaxis_title=options['grouping_var'],
                yaxis_title=f'Trung bình {dep_var}',
                template=self.plot_template
            )
            
            st.plotly_chart(fig)

    def _plot_qq(self, options):
        """Create Q-Q plots for residuals"""
        for dep_var in options['dependent_vars']:
            group_means = self.df.groupby(options['grouping_var'])[dep_var].transform('mean')
            residuals = self.df[dep_var] - group_means
            
            fig = go.Figure()
            
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Phần dư'
            ))
            
            min_val = min(theoretical_quantiles)
            max_val = max(theoretical_quantiles)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val * np.std(residuals), max_val * np.std(residuals)],
                mode='lines',
                name='Đường tham chiếu',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title=f'Biểu đồ Q-Q của phần dư: {dep_var}',
                xaxis_title='Phân vị lý thuyết',
                yaxis_title='Phân vị mẫu',
                template=self.plot_template
            )
            
            st.plotly_chart(fig)

    def _get_alternative(self, alternative):
        """Convert UI alternative hypothesis to scipy format"""
        if alternative == "Nhóm 1 ≠ Nhóm 2":
            return "two-sided"
        elif alternative == "Nhóm 1 > Nhóm 2":
            return "greater"
        else:
            return "less"