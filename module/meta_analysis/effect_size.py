import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

class EffectSize:
    def __init__(self, data):
        self.data = data
        
    def setup_interface(self):
        st.subheader("Effect Size Calculator")
        
        # Cấu hình trong sidebar
        with st.sidebar:
            st.write("### Cấu hình dữ liệu")
            
            # Chọn cột nhóm
            self.group_col = st.selectbox(
                "Chọn cột phân nhóm",
                self.data.columns,
                key="effect_group_col"
            )
            
            # Kiểm tra số lượng nhóm
            unique_groups = self.data[self.group_col].unique()
            if len(unique_groups) != 2:
                st.error(f"Cần đúng 2 nhóm để tính SMD. Hiện tại có {len(unique_groups)} nhóm.")
                return
            
            # Hiển thị tên các nhóm
            st.write("**Các nhóm:**")
            for group in unique_groups:
                st.write(f"- {group}")
            
            # Chọn cột outcome
            self.outcome_col = st.selectbox(
                "Chọn cột outcome",
                [col for col in self.data.columns if col != self.group_col],
                key="effect_outcome_col"
            )
            
            st.write("### Tùy chọn tính toán")
            
            # Chọn phương pháp tính SD gộp
            self.pooled_sd_method = st.selectbox(
                "Phương pháp tính SD gộp",
                ["Pooled", "Control Group"],
                help="Pooled: SD gộp từ cả 2 nhóm\nControl Group: Chỉ dùng SD của nhóm control",
                key="pooled_sd_method"
            )
            
            # Tùy chọn Hedges' correction
            self.use_hedges = st.checkbox(
                "Sử dụng Hedges' correction",
                value=True,
                help="Hiệu chỉnh độ lệch cho cỡ mẫu nhỏ",
                key="use_hedges"
            )
            
            # Chọn mức tin cậy
            self.conf_level = st.slider(
                "Mức độ tin cậy",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="effect_conf"
            )
        
        try:
            # Tính toán effect size
            self._calculate_smd()
            
        except Exception as e:
            st.error(f"Lỗi trong quá trình tính toán: {str(e)}")
    
    def _calculate_smd(self):
        # Kiểm tra missing values
        if self.data[self.outcome_col].isnull().any():
            st.warning("Dữ liệu có missing values. Các giá trị này sẽ bị loại bỏ.")
            self.data = self.data.dropna(subset=[self.outcome_col])
        
        # Tách dữ liệu theo nhóm
        groups = self.data[self.group_col].unique()
        group1_data = self.data[self.data[self.group_col] == groups[0]][self.outcome_col]
        group2_data = self.data[self.data[self.group_col] == groups[1]][self.outcome_col]
        
        # Tính các thống kê cơ bản
        n1 = len(group1_data)
        n2 = len(group2_data)
        mean1 = np.mean(group1_data)
        mean2 = np.mean(group2_data)
        var1 = np.var(group1_data, ddof=1)
        var2 = np.var(group2_data, ddof=1)
        
        # Tính SD gộp
        if self.pooled_sd_method == "Pooled":
            pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        else:  # Control Group
            pooled_sd = np.sqrt(var2)  # Giả sử nhóm 2 là control
        
        # Tính Cohen's d
        d = (mean1 - mean2) / pooled_sd
        
        # Tính SE của d
        if self.pooled_sd_method == "Pooled":
            se_d = np.sqrt((n1 + n2)/(n1 * n2) + d**2/(2*(n1 + n2)))
        else:
            se_d = np.sqrt(((n1 + n2)/(n1 * n2)) * (1 + d**2/4))
        
        # Áp dụng Hedges' correction nếu được chọn
        if self.use_hedges:
            # Tính hệ số hiệu chỉnh J
            df = n1 + n2 - 2
            j = 1 - (3 / (4 * df - 1))
            
            # Hiệu chỉnh d và SE
            d = j * d
            se_d = j * se_d
        
        # Tính confidence interval
        z = stats.norm.ppf(1 - (1 - self.conf_level)/2)
        ci_lower = d - z * se_d
        ci_upper = d + z * se_d
        
        # Hiển thị kết quả
        st.write("### Kết quả tính toán")
        
        # Thống kê mô tả
        st.write("**Thống kê mô tả:**")
        stats_df = pd.DataFrame({
            'Nhóm': [groups[0], groups[1]],
            'n': [n1, n2],
            'Mean': [mean1, mean2],
            'SD': [np.sqrt(var1), np.sqrt(var2)]
        })
        st.write(stats_df)
        
        # Effect size
        st.write("\n**Effect Size:**")
        effect_name = "Hedges' g" if self.use_hedges else "Cohen's d"
        st.write(f"{effect_name}: {d:.4f}")
        st.write(f"Standard Error: {se_d:.4f}")
        st.write(f"{self.conf_level*100:.1f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Diễn giải
        st.write("\n**Diễn giải:**")
        
        # Diễn giải magnitude
        if abs(d) < 0.2:
            magnitude = "nhỏ"
        elif abs(d) < 0.5:
            magnitude = "nhỏ đến trung bình"
        elif abs(d) < 0.8:
            magnitude = "trung bình đến lớn"
        else:
            magnitude = "lớn"
        
        st.write(f"- Độ lớn của hiệu ứng: {magnitude}")
        
        # Diễn giải hướng
        direction = "cao hơn" if d > 0 else "thấp hơn"
        st.write(f"- Nhóm {groups[0]} có giá trị trung bình {direction} nhóm {groups[1]}")
        
        # Diễn giải ý nghĩa thống kê
        if 0 < ci_lower or 0 > ci_upper:
            st.write(f"- Sự khác biệt có ý nghĩa thống kê ở mức {self.conf_level*100:.1f}%")
        else:
            st.write(f"- Sự khác biệt không có ý nghĩa thống kê ở mức {self.conf_level*100:.1f}%")

def run_analysis(df):
    calculator = EffectSize(df)
    calculator.setup_interface() 