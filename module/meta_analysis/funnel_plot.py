import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class FunnelPlot:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Funnel Plot")
        
        # Cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình dữ liệu")
            
            # Chọn cột dữ liệu
            self.effect_col = st.selectbox(
                "Chọn cột Effect Size",
                self.data.columns,
                key="funnel_effect_col"
            )
            
            self.se_col = st.selectbox(
                "Chọn cột Standard Error",
                self.data.columns,
                key="funnel_se_col"
            )
            
            # Tùy chọn precision measure
            self.precision_measure = st.selectbox(
                "Chọn đơn vị đo độ chính xác",
                ["Standard Error", "Precision (1/SE)", "Sample Size"],
                key="precision_measure"
            )
            
            # Tùy chọn method
            self.method = st.selectbox(
                "Chọn phương pháp tổng hợp",
                ["Fixed Effects", "Random Effects"],
                key="funnel_method"
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
                    key="funnel_width"
                )
            
            with col2:
                self.plot_height = st.number_input(
                    "Chiều cao",
                    min_value=6,
                    max_value=20,
                    value=8,
                    key="funnel_height"
                )
            
            # Tùy chọn hiển thị
            st.write("### Tùy chọn hiển thị")
            
            self.show_contour = st.checkbox(
                "Hiển thị contour",
                value=True,
                key="show_contour"
            )
            
            if self.show_contour:
                self.contour_levels = st.multiselect(
                    "Chọn mức ý nghĩa thống kê",
                    ["0.01", "0.05", "0.1"],
                    default=["0.01", "0.05", "0.1"],
                    key="contour_levels"
                )
            
            self.show_egger = st.checkbox(
                "Hiển thị Egger's regression line",
                value=True,
                key="show_egger"
            )
            
            self.show_pseudo_ci = st.checkbox(
                "Hiển thị pseudo 95% CI",
                value=True,
                key="show_pseudo_ci"
            )
        
        try:
            # Vẽ funnel plot
            self._create_funnel_plot()
            
            # Thực hiện và hiển thị Egger's test
            if self.show_egger:
                self._perform_eggers_test()
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình vẽ biểu đồ: {str(e)}")
    
    def _calculate_overall_effect(self):
        effects = self.data[self.effect_col].values
        ses = self.data[self.se_col].values
        variances = ses**2
        
        if self.method == "Fixed Effects":
            weights = 1 / variances
        else:
            # Random effects weights
            w = 1 / variances
            mean_effect = np.sum(w * effects) / np.sum(w)
            q = np.sum(w * (effects - mean_effect)**2)
            df = len(effects) - 1
            c = np.sum(w) - np.sum(w**2) / np.sum(w)
            tau2 = max(0, (q - df) / c)
            weights = 1 / (variances + tau2)
        
        overall_effect = np.sum(weights * effects) / np.sum(weights)
        return overall_effect
    
    def _perform_eggers_test(self):
        # Chuẩn bị dữ liệu cho Egger's test
        effects = self.data[self.effect_col].values
        ses = self.data[self.se_col].values
        precision = 1 / ses
        
        # Tính standardized effect size
        standard_effect = effects / ses
        
        # Thực hiện hồi quy
        slope, intercept, r_value, p_value, std_err = stats.linregress(precision, standard_effect)
        
        # Hiển thị kết quả
        st.write("### Kết quả Egger's Test")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Hệ số hồi quy**")
            st.write(f"Intercept: {intercept:.4f}")
            st.write(f"Slope: {slope:.4f}")
            st.write(f"SE of intercept: {std_err:.4f}")
        
        with col2:
            st.write("**Kiểm định**")
            st.write(f"t-value: {intercept/std_err:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            
        # Diễn giải
        st.write("### Diễn giải")
        if p_value < 0.05:
            st.write("Có bằng chứng về publication bias (p < 0.05)")
        elif p_value < 0.1:
            st.write("Có dấu hiệu về publication bias (p < 0.1)")
        else:
            st.write("Không có bằng chứng rõ ràng về publication bias")
    
    def _create_funnel_plot(self):
        # Lấy dữ liệu
        effects = self.data[self.effect_col].values
        ses = self.data[self.se_col].values
        
        # Tính overall effect
        overall_effect = self._calculate_overall_effect()
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(self.plot_width, self.plot_height))
        
        # Chuẩn bị dữ liệu cho trục y
        if self.precision_measure == "Standard Error":
            y = ses
            ylabel = "Standard Error"
            y_lim = (max(ses) * 1.2, 0)  # Đảo ngược trục y
        elif self.precision_measure == "Precision (1/SE)":
            y = 1 / ses
            ylabel = "Precision (1/SE)"
            y_lim = (0, max(1/ses) * 1.2)
        else:  # Sample Size
            y = 1 / (ses**2)
            ylabel = "Sample Size"
            y_lim = (0, max(1/ses**2) * 1.2)
        
        # Vẽ scatter plot
        plt.scatter(effects, y, alpha=0.6)
        
        # Vẽ vertical line tại overall effect
        plt.axvline(overall_effect, color='red', linestyle='--', alpha=0.5)
        
        # Thêm contour nếu được chọn
        if self.show_contour:
            x_range = np.linspace(min(effects) - 0.5, max(effects) + 0.5, 100)
            
            for level in self.contour_levels:
                alpha = float(level)
                z = stats.norm.ppf(1 - alpha/2)
                
                if self.precision_measure == "Standard Error":
                    upper_bound = z * x_range
                    lower_bound = -z * x_range
                    plt.fill_between(x_range, upper_bound, lower_bound, alpha=0.1)
                else:
                    se_range = np.linspace(min(ses), max(ses), 100)
                    for se in se_range:
                        plt.plot(
                            [overall_effect - z*se, overall_effect + z*se],
                            [1/se, 1/se] if self.precision_measure == "Precision (1/SE)" else [1/se**2, 1/se**2],
                            'k-', alpha=0.1
                        )
        
        # Thêm pseudo CI nếu được chọn
        if self.show_pseudo_ci:
            x_range = np.linspace(min(effects) - 0.5, max(effects) + 0.5, 100)
            if self.precision_measure == "Standard Error":
                plt.plot(x_range, 1.96 * x_range, 'k--', alpha=0.3)
                plt.plot(x_range, -1.96 * x_range, 'k--', alpha=0.3)
            else:
                se_range = np.linspace(min(ses), max(ses), 100)
                for se in se_range:
                    plt.plot(
                        [overall_effect - 1.96*se, overall_effect + 1.96*se],
                        [1/se, 1/se] if self.precision_measure == "Precision (1/SE)" else [1/se**2, 1/se**2],
                        'k--', alpha=0.3
                    )
        
        # Thêm Egger's regression line nếu được chọn
        if self.show_egger:
            precision = 1 / ses
            standard_effect = effects / ses
            slope, intercept, _, _, _ = stats.linregress(precision, standard_effect)
            x_reg = np.array([min(precision), max(precision)])
            y_reg = intercept + slope * x_reg
            if self.precision_measure == "Precision (1/SE)":
                plt.plot(y_reg/x_reg, x_reg, 'r-', alpha=0.5)
        
        # Chỉnh sửa trục và tiêu đề
        plt.xlabel('Effect Size')
        plt.ylabel(ylabel)
        plt.ylim(y_lim)
        plt.title('Funnel Plot')
        
        # Thêm chú thích
        if self.show_contour:
            legend_elements = [
                plt.Line2D([0], [0], color='k', alpha=0.1, label=f'p < {level}')
                for level in self.contour_levels
            ]
            plt.legend(handles=legend_elements, title="Significance Levels")
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Hiển thị plot
        st.pyplot(fig)

def run_analysis(df):
    plotter = FunnelPlot(df)
    plotter.setup_interface() 