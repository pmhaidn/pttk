import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class ForestPlot:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Forest Plot")
        
        # Cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình dữ liệu")
            
            # Chọn cột dữ liệu
            self.effect_col = st.selectbox(
                "Chọn cột Effect Size",
                self.data.columns,
                key="forest_effect_col"
            )
            
            self.se_col = st.selectbox(
                "Chọn cột Standard Error",
                self.data.columns,
                key="forest_se_col"
            )
            
            # Tùy chọn hiển thị study
            self.study_label_col = st.selectbox(
                "Chọn cột tên nghiên cứu",
                ["Index"] + list(self.data.columns),
                key="forest_label_col"
            )
            
            # Tùy chọn nhóm
            self.group_col = st.selectbox(
                "Chọn cột phân nhóm (không bắt buộc)",
                ["None"] + list(self.data.columns),
                key="forest_group_col"
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
                    key="forest_width"
                )
            
            with col2:
                self.plot_height_per_study = st.number_input(
                    "Chiều cao/nghiên cứu",
                    min_value=0.2,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    key="forest_height"
                )
            
            # Tùy chọn màu sắc
            self.color_scheme = st.selectbox(
                "Bảng màu",
                ["Default", "Colorblind", "Pastel", "Deep", "Muted"],
                key="forest_color"
            )
            
            # Tùy chọn hiển thị
            st.write("### Tùy chọn hiển thị")
            
            self.show_stats = st.checkbox(
                "Hiển thị thống kê",
                value=True,
                key="forest_stats"
            )
            
            self.show_weights = st.checkbox(
                "Hiển thị trọng số",
                value=True,
                key="forest_weights"
            )
            
            self.show_overall = st.checkbox(
                "Hiển thị tổng hợp",
                value=True,
                key="forest_overall"
            )
            
            if self.show_overall:
                self.overall_method = st.selectbox(
                    "Phương pháp tổng hợp",
                    ["Fixed Effects", "Random Effects"],
                    key="forest_method"
                )
            
            # Tùy chọn confidence interval
            self.conf_level = st.slider(
                "Mức độ tin cậy",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="forest_conf"
            )
        
        try:
            # Vẽ forest plot
            self._create_forest_plot()
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình vẽ biểu đồ: {str(e)}")
    
    def _calculate_weights(self, variances):
        if self.overall_method == "Fixed Effects":
            # Fixed effects weights
            weights = 1 / variances
        else:
            # Random effects weights
            # Tính tau² bằng DerSimonian-Laird
            effects = self.data[self.effect_col].values
            w = 1 / variances
            mean_effect = np.sum(w * effects) / np.sum(w)
            q = np.sum(w * (effects - mean_effect)**2)
            df = len(effects) - 1
            c = np.sum(w) - np.sum(w**2) / np.sum(w)
            tau2 = max(0, (q - df) / c)
            
            # Random effects weights
            weights = 1 / (variances + tau2)
        
        # Chuẩn hóa weights thành %
        return weights / np.sum(weights) * 100
    
    def _calculate_overall_effect(self, effects, variances):
        # Tính weights
        if self.overall_method == "Fixed Effects":
            weights = 1 / variances
        else:
            # Tính tau² và random effects weights
            w = 1 / variances
            mean_effect = np.sum(w * effects) / np.sum(w)
            q = np.sum(w * (effects - mean_effect)**2)
            df = len(effects) - 1
            c = np.sum(w) - np.sum(w**2) / np.sum(w)
            tau2 = max(0, (q - df) / c)
            weights = 1 / (variances + tau2)
        
        # Tính overall effect và SE
        overall_effect = np.sum(weights * effects) / np.sum(weights)
        overall_se = np.sqrt(1 / np.sum(weights))
        
        # Tính CI
        z = stats.norm.ppf(1 - (1 - self.conf_level) / 2)
        ci_lower = overall_effect - z * overall_se
        ci_upper = overall_effect + z * overall_se
        
        return overall_effect, ci_lower, ci_upper
    
    def _create_forest_plot(self):
        # Lấy dữ liệu
        effects = self.data[self.effect_col].values
        ses = self.data[self.se_col].values
        variances = ses**2
        
        # Tính weights nếu cần
        if self.show_weights:
            weights = self._calculate_weights(variances)
        
        # Tính overall effect nếu cần
        if self.show_overall:
            overall_effect, overall_ci_lower, overall_ci_upper = self._calculate_overall_effect(effects, variances)
        
        # Tạo figure với kích thước tùy chỉnh
        fig_height = len(self.data) * self.plot_height_per_study
        if self.show_overall:
            fig_height += 1  # Thêm không gian cho overall effect
        
        fig, ax = plt.subplots(figsize=(self.plot_width, fig_height))
        
        # Set color palette
        if self.color_scheme != "Default":
            sns.set_palette(self.color_scheme.lower())
        
        # Tạo y positions cho các studies
        y_positions = np.arange(len(self.data))
        
        # Vẽ các studies
        for i, (effect, se) in enumerate(zip(effects, ses)):
            # Tính CI
            z = stats.norm.ppf(1 - (1 - self.conf_level) / 2)
            ci_lower = effect - z * se
            ci_upper = effect + z * se
            
            # Vẽ CI line
            plt.hlines(y_positions[i], ci_lower, ci_upper, color='black')
            
            # Vẽ effect size point với kích thước tỷ lệ với weight nếu có
            if self.show_weights:
                size = 50 + weights[i]
            else:
                size = 50
            plt.scatter(effect, y_positions[i], s=size, color='blue')
        
        # Vẽ overall effect nếu được chọn
        if self.show_overall:
            # Vẽ diamond plot
            diamond_height = 0.4
            diamond_y = -1
            
            diamond_coords = np.array([
                [overall_effect, diamond_y],
                [overall_ci_upper, diamond_y + diamond_height/2],
                [overall_effect, diamond_y + diamond_height],
                [overall_ci_lower, diamond_y + diamond_height/2]
            ])
            
            plt.fill(
                diamond_coords[:, 0],
                diamond_coords[:, 1],
                color='red',
                alpha=0.3
            )
            
            # Thêm vertical line cho overall effect
            plt.axvline(overall_effect, color='red', linestyle='--', alpha=0.5)
        
        # Thêm vertical line tại 0
        plt.axvline(0, color='black', linestyle='-', alpha=0.2)
        
        # Thêm labels cho studies
        if self.study_label_col == "Index":
            labels = [f"Study {i+1}" for i in range(len(self.data))]
        else:
            labels = self.data[self.study_label_col]
        
        # Thêm weights vào labels nếu được chọn
        if self.show_weights:
            labels = [f"{label} ({weight:.1f}%)" for label, weight in zip(labels, weights)]
        
        plt.yticks(y_positions, labels)
        
        # Thêm thống kê nếu được chọn
        if self.show_stats:
            stats_text = []
            for i, (effect, se) in enumerate(zip(effects, ses)):
                z = stats.norm.ppf(1 - (1 - self.conf_level) / 2)
                ci_lower = effect - z * se
                ci_upper = effect + z * se
                stats_text.append(f"{effect:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
            
            # Thêm overall effect vào stats nếu có
            if self.show_overall:
                stats_text.append(f"Overall: {overall_effect:.2f} [{overall_ci_lower:.2f}, {overall_ci_upper:.2f}]")
            
            # Thêm text vào bên phải biểu đồ
            for i, text in enumerate(stats_text):
                plt.text(
                    plt.xlim()[1] * 1.1,
                    y_positions[i] if i < len(y_positions) else diamond_y,
                    text,
                    va='center'
                )
        
        # Chỉnh sửa trục và tiêu đề
        plt.xlabel('Effect Size')
        plt.ylabel('Study')
        plt.title('Forest Plot')
        
        # Điều chỉnh layout để hiển thị đầy đủ
        plt.tight_layout()
        
        # Hiển thị plot trong Streamlit
        st.pyplot(fig)

def run_analysis(df):
    plotter = ForestPlot(df)
    plotter.setup_interface() 