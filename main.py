import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import arviz as az
from module import descriptive_module, ttest_module, correlation_module, linear_regression_module

class StatisticalApp:
    def __init__(self):
        st.set_page_config(page_title="Phân tích Thống kê", layout="wide")
        self.setup_sidebar()
        
    def setup_sidebar(self):
        st.sidebar.title("Menu Phân tích")
        self.analysis_type = st.sidebar.selectbox(
            "Chọn phương pháp phân tích",
            ["Thống kê mô tả", "Kiểm định T", "ANOVA", "Hồi quy", "Tần suất", "Phân tích nhân tố"]
        )
        
        # Add Regression Method Options to Sidebar
        if self.analysis_type == "Hồi quy":
          self.regression_method = st.sidebar.selectbox(
              "Chọn phương pháp hồi quy",
              ["Classical", "Bayesian"]
          )
          if self.regression_method == "Classical":
            self.regression_type = st.sidebar.selectbox(
              "Chọn loại phân tích",
              ["Correlation", "Linear Regression"]
            )
        # Add ANOVA Method and Type options to sidebar
        if self.analysis_type == "ANOVA":
          self.anova_method = st.sidebar.selectbox(
              "Chọn phương pháp ANOVA",
              ["Classical", "Bayesian"]
          )
          if self.anova_method == "Classical":
              self.anova_type = st.sidebar.selectbox(
                  "Chọn loại phân tích ANOVA",
                  ["ANOVA", "Repeated Measures ANOVA", "ANCOVA", "MANOVA"]
                  )
          elif self.anova_method == "Bayesian":
            self.bayesian_anova_type = st.sidebar.selectbox(
                  "Chọn loại phân tích ANOVA (Bayesian)",
                  ["ANOVA", "Repeated Measures ANOVA", "ANCOVA"]
                  )


        # Tải dữ liệu
        self.uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu", type=['csv', 'xlsx'])
        
        if self.uploaded_file:
            self.load_data()
            self.show_data_preview()
            self.run_analysis()
    
    def load_data(self):
        try:
            if self.uploaded_file.name.endswith('csv'):
                self.df = pd.read_csv(self.uploaded_file)
            else:
                self.df = pd.read_excel(self.uploaded_file)
            st.sidebar.success("Đã tải dữ liệu thành công!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {str(e)}")
            
    def show_data_preview(self):
        st.header("Xem trước dữ liệu")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Số hàng:", self.df.shape[0])
        with col2:
            st.write("Số cột:", self.df.shape[1])
        
        # Xem mẫu dữ liệu
        st.subheader("Mẫu dữ liệu")
        n_rows = st.slider("Số hàng hiển thị", 5, 50, 10)
        st.write(self.df.head(n_rows))

    def raincloud_plot(self, vars):
        st.subheader("Biểu đồ Raincloud")
        for var in vars:
            # Tạo violin plot
            violin = go.Violin(
                y=self.df[var],
                name="Violin",
                side="negative",
                line_color="blue"
            )
            
            # Tạo box plot
            box = go.Box(
                y=self.df[var],
                name="Box",
                line_color="red"
            )
            
            # Tạo scatter plot (points)
            scatter = go.Scatter(
                y=self.df[var],
                mode="markers",
                name="Points",
                marker=dict(color="black", size=3)
            )
            
            fig = go.Figure(data=[violin, box, scatter])
            fig.update_layout(title=f"Raincloud Plot của {var}")
            st.plotly_chart(fig)

    def time_series_analysis(self, vars):
        st.subheader("Phân tích chuỗi thời gian")
        
        # Chọn biến thời gian
        time_col = st.selectbox("Chọn biến thời gian", self.df.columns)
        
        if time_col:
            try:
                self.df[time_col] = pd.to_datetime(self.df[time_col])
                for var in vars:
                    fig = px.line(
                        self.df, 
                        x=time_col, 
                        y=var,
                        title=f"Chuỗi thời gian của {var}"
                    )
                    
                    # Thêm tùy chọn đường xu hướng
                    if st.checkbox(f"Hiển thị đường xu hướng cho {var}"):
                        fig.add_scatter(
                            x=self.df[time_col],
                            y=self.df[var].rolling(window=7).mean(),
                            name="Đường xu hướng (MA-7)",
                            line=dict(color="red")
                        )
                    
                    st.plotly_chart(fig)
            except:
                st.error("Không thể chuyển đổi cột thời gian")

    def flexplot_analysis(self, vars):
        st.subheader("Flexplot Analysis")
        if len(vars) >= 2:
            x_var = st.selectbox("Chọn biến X", vars)
            y_var = st.selectbox("Chọn biến Y", vars)
            
            # Scatter plot với các tùy chọn
            fig = px.scatter(
                self.df,
                x=x_var,
                y=y_var,
                title=f"Flexplot: {x_var} vs {y_var}"
            )
            
            # Thêm đường hồi quy
            if st.checkbox("Hiển thị đường hồi quy"):
                fig.add_scatter(
                    x=self.df[x_var],
                    y=np.poly1d(np.polyfit(self.df[x_var], self.df[y_var], 1))(self.df[x_var]),
                    name="Đường hồi quy",
                    line=dict(color="red")
                )
            
            st.plotly_chart(fig)

    def ttest_analysis(self):
        if self.df is not None:
            ttest = ttest_module.HypothesisTesting(self.df)
            options = ttest.setup_interface()  # Get all options from the interface
            ttest.run_analysis(options)  

    def anova_analysis(self):
      st.header("Phân tích ANOVA")

      if self.anova_method == "Classical":
          st.write(f"Performing Classical {self.anova_type}")
          #Add logic to perform classical anova
      elif self.anova_method == "Bayesian":
          st.write(f"Performing Bayesian {self.bayesian_anova_type}")
          #Add logic to perform bayesian anova

    def regression_analysis(self):
        st.header("Phân tích hồi quy")
        
        # Method selection is already handled in sidebar.
        if self.regression_method == "Classical":
          if self.regression_type == "Correlation":
            self.correlation_analysis()
          elif self.regression_type == "Linear Regression":
              self.linear_regression_analysis()
        elif self.regression_method == "Bayesian":
          st.write("Bayesian regression is not implement yet")

    def correlation_analysis(self):
      if self.df is not None:
        correlation = correlation_module.CorrelationAnalysis(self.df)
        correlation.correlation_analysis()
      else:
        st.warning("Vui lòng tải lên file dữ liệu để thực hiện phân tích.")

    def linear_regression_analysis(self):
      if self.df is not None:
        regression = linear_regression_module.LinearRegressionAnalysis(self.df)
        regression.regression_analysis()
      else:
        st.warning("Vui lòng tải lên file dữ liệu để thực hiện phân tích.")
    
    def factor_analysis(self):
        st.header("Phân tích nhân tố")
        
        analysis_type = st.selectbox(
            "Chọn loại phân tích",
            ["Principal Component Analysis", 
             "Exploratory Factor Analysis",
             "Confirmatory Factor Analysis"]
        )
        
        if analysis_type == "Principal Component Analysis":
            self.pca_analysis()
        elif analysis_type == "Exploratory Factor Analysis":
            self.efa_analysis()
        elif analysis_type == "Confirmatory Factor Analysis":
            self.cfa_analysis()

    def run_analysis(self):
        if self.analysis_type == "Thống kê mô tả":
            if self.df is not None:  # Kiểm tra xem dữ liệu đã được tải chưa
                descriptive_stats = descriptive_module.DescriptiveStatistics(self.df)
                descriptive_stats.descriptive_analysis()
            else:
                st.warning("Vui lòng tải lên file dữ liệu để thực hiện phân tích.")
        elif self.analysis_type == "Kiểm định T":
            self.ttest_analysis()
        elif self.analysis_type == "ANOVA":
            self.anova_analysis()
        elif self.analysis_type == "Hồi quy":
            self.regression_analysis()
        elif self.analysis_type == "Tần suất":
            self.frequency_analysis()
        elif self.analysis_type == "Phân tích nhân tố":
            self.factor_analysis()

if __name__ == "__main__":
    app = StatisticalApp()