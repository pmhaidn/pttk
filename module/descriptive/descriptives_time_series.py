import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def time_series_analysis(df):
    st.subheader("Phân tích chuỗi thời gian")

    # --- Variable Selection ---
    st.sidebar.subheader("Chọn biến")
    dependent_variable = st.sidebar.selectbox("Biến phụ thuộc", df.columns, index=0)
    time_variable = st.sidebar.selectbox("Biến thời gian (tùy chọn)", ["Không chọn"] + list(df.columns), index=0)


    # --- Data Filtering ---
    st.sidebar.subheader("Lọc dữ liệu")
    filter_data = st.sidebar.checkbox("Lọc theo")

    if filter_data:
        filter_by = st.sidebar.radio("Lọc theo", ["Số hàng", "Chỉ mục thời gian", "Ngày"], index=0)

        if filter_by == "Số hàng":
            row_start = st.sidebar.number_input("Bắt đầu hàng", min_value=1, max_value=len(df), value=1, step=1)
            row_end = st.sidebar.number_input("Kết thúc hàng", min_value=row_start, max_value=len(df), value=len(df), step=1)
            df = df.iloc[row_start-1:row_end]

        elif filter_by == "Chỉ mục thời gian" and time_variable != "Không chọn":
            try:
                df[time_variable] = pd.to_datetime(df[time_variable])
                time_start = st.sidebar.number_input("Bắt đầu chỉ mục thời gian", min_value=0, max_value=len(df) - 1, value=0, step=1)
                time_end = st.sidebar.number_input("Kết thúc chỉ mục thời gian", min_value=time_start, max_value=len(df) -1, value=len(df)-1, step=1)
                df = df.iloc[time_start:time_end+1]
            except Exception as e:
                st.error(f"Lỗi chuyển đổi cột thời gian. Vui lòng chọn đúng định dạng. Lỗi: {e}")
                return  # Exit if time column is invalid

        elif filter_by == "Ngày" and time_variable != "Không chọn":
            try:
                df[time_variable] = pd.to_datetime(df[time_variable])
                date_start = st.sidebar.text_input("Ngày bắt đầu (YYYY-MM-DD HH:MM:SS)", value=df[time_variable].min().strftime("%Y-%m-%d %H:%M:%S"))
                date_end = st.sidebar.text_input("Ngày kết thúc (YYYY-MM-DD HH:MM:SS)", value=df[time_variable].max().strftime("%Y-%m-%d %H:%M:%S"))
                date_start = pd.to_datetime(date_start)
                date_end = pd.to_datetime(date_end)
                df = df[(df[time_variable] >= date_start) & (df[time_variable] <= date_end)]

            except Exception as e:
                st.error(f"Lỗi định dạng ngày tháng. Vui lòng sử dụng định dạng YYYY-MM-DD HH:MM:SS. Lỗi: {e}")
                return # Exit if date format is invalid

    # --- Descriptive Statistics ---
    st.subheader("Thống kê mô tả")
    transpose_descriptives = st.checkbox("Chuyển vị bảng thống kê mô tả")
    descriptives = df.describe()

    if transpose_descriptives:
        descriptives = descriptives.T

    st.dataframe(descriptives)

    # --- Plots ---
    st.subheader("Biểu đồ")

    # --- Time Series Plot ---
    st.subheader("Biểu đồ chuỗi thời gian")
    time_series_plot = st.checkbox("Biểu đồ chuỗi thời gian", value=True)

    if time_series_plot:
        if time_variable == "Không chọn":
            st.warning("Vui lòng chọn biến thời gian để vẽ biểu đồ chuỗi thời gian.")
        else:
            try:
                plot_type = st.radio("Loại biểu đồ", ["Điểm", "Đường", "Cả hai"], index=2)  # Both is default
                distribution_type = st.radio("Phân phối", ["Mật độ", "Histogram", "Không"], index=2) # None is default

                fig = px.line(df, x=time_variable, y=dependent_variable, title=f"Chuỗi thời gian của {dependent_variable}")

                if plot_type == "Điểm":
                    fig.update_traces(mode="markers")
                elif plot_type == "Đường":
                    fig.update_traces(mode="lines")
                else: # Both
                    fig.update_traces(mode="lines+markers")

                if distribution_type == "Mật độ":
                    st.subheader("Phân phối mật độ")
                    fig_dist = px.histogram(df, x=dependent_variable, marginal="rug", histnorm='density', title=f"Phân phối mật độ của {dependent_variable}")
                    st.plotly_chart(fig_dist)

                elif distribution_type == "Histogram":
                    st.subheader("Biểu đồ Histogram")
                    fig_dist = px.histogram(df, x=dependent_variable, title=f"Histogram của {dependent_variable}")
                    st.plotly_chart(fig_dist)

                st.plotly_chart(fig)


            except Exception as e:
                st.error(f"Không thể vẽ biểu đồ chuỗi thời gian. Lỗi: {e}")

    # --- Lag Plot ---
    st.subheader("Biểu đồ Lag")
    lag_plot = st.checkbox("Biểu đồ Lag")
    if lag_plot:
        lag = st.number_input("Lag", min_value=1, value=1, step=1)
        regression_line = st.checkbox("Thêm đường hồi quy", value=True)

        if regression_line:
            regression_type = st.radio("Loại hồi quy", ["Trơn", "Tuyến tính"], index=0)
            confidence_interval = st.checkbox("Khoảng tin cậy", value=True)
            ci_level = st.number_input("Mức độ tin cậy", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

            if time_variable == "Không chọn":
                x = df[dependent_variable].iloc[:-lag].values
                y = df[dependent_variable].iloc[lag:].values
            else:
                 x = df[dependent_variable].shift(lag).dropna()
                 y = df[dependent_variable].iloc[lag:].values

            fig = px.scatter(x=x, y=y,
                            labels={'x': f'{dependent_variable} (t-{lag})', 'y': f'{dependent_variable} (t)'},
                            title=f"Biểu đồ Lag của {dependent_variable} (Lag={lag})")


            if regression_type == "Trơn": # Smooth
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess
                z = lowess(y, x, frac=0.3) # Use lowess for smoothing
                fig.add_trace(px.line(x=z[:,0], y=z[:,1]).data[0])

            else:  #Linear
                import statsmodels.formula.api as sm
                data = pd.DataFrame({'x': x, 'y': y})
                model = sm.ols("y ~ x", data=data).fit()
                x_range = np.linspace(min(x), max(x), 100)
                y_pred = model.predict(exog=dict(x=x_range))

                fig.add_trace(px.line(x=x_range, y=y_pred, color_discrete_sequence=['red']).data[0])


                if confidence_interval:
                    from scipy import stats
                    predict_mean_y = model.get_prediction(exog=dict(x=x_range))
                    predictions = predict_mean_y.summary_frame(alpha=1-ci_level)
                    fig.add_trace(px.line(x=x_range, y=predictions["obs_ci_lower"], line_color='rgba(255,0,0,0.2)',name="CI Lower").data[0])
                    fig.add_trace(px.line(x=x_range, y=predictions["obs_ci_upper"], line_color='rgba(255,0,0,0.2)', fill='tonexty',name="CI Upper").data[0])



        else:
            if time_variable == "Không chọn":
                x = df[dependent_variable].iloc[:-lag].values
                y = df[dependent_variable].iloc[lag:].values
            else:
                x = df[dependent_variable].shift(lag).dropna()
                y = df[dependent_variable].iloc[lag:].values

            fig = px.scatter(x=x, y=y,
                            labels={'x': f'{dependent_variable} (t-{lag})', 'y': f'{dependent_variable} (t)'},
                            title=f"Biểu đồ Lag của {dependent_variable} (Lag={lag})")

        st.plotly_chart(fig)

    # --- ACF Plot ---
    st.subheader("Hàm tự tương quan (ACF)")
    acf_plot = st.checkbox("Hàm tự tương quan")
    if acf_plot:
        max_lag = st.number_input("Lag tối đa", min_value=1, value=10, step=1)
        zero_lag = st.checkbox("Lag 0")
        confidence_interval = st.checkbox("Khoảng tin cậy", value=True)
        if confidence_interval:
             ci_level = st.number_input("Mức độ tin cậy", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
             ci_type = st.radio("Loại khoảng tin cậy", ["Nhiễu trắng", "Trung bình động"], index=0)
        else:
            ci_level=0
            ci_type="Nhiễu trắng"

        import statsmodels.api as sm
        from statsmodels.graphics.tsaplots import plot_acf
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(df[dependent_variable], lags=max_lag, ax=ax, zero=zero_lag, alpha = 1- ci_level if confidence_interval else None)
        st.pyplot(fig)

    # --- PACF Plot ---
    st.subheader("Hàm tự tương quan một phần (PACF)")
    pacf_plot = st.checkbox("Hàm tự tương quan một phần")
    if pacf_plot:
        max_lag = st.number_input("Lag tối đa", min_value=1, value=10, step=1)
        confidence_interval = st.checkbox("Khoảng tin cậy", value=True)

        if confidence_interval:
             ci_level = st.number_input("Mức độ tin cậy", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

        import statsmodels.api as sm
        from statsmodels.graphics.tsaplots import plot_pacf
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(df[dependent_variable], lags=max_lag, ax=ax, alpha = 1-ci_level if confidence_interval else None)
        st.pyplot(fig)


if __name__ == '__main__':
    st.title("Phân tích chuỗi thời gian")

    # --- Sample Data ---
    data = {
        'Thời gian': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Giá trị': np.random.randn(100).cumsum()
    }
    df = pd.DataFrame(data)

    # --- File Upload ---
    st.sidebar.subheader("Tải lên dữ liệu")
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Tải file thành công!")
        except Exception as e:
            st.error(f"Không thể đọc file. Lỗi: {e}")
            df = pd.DataFrame(data) # Fallback to sample data


    time_series_analysis(df.copy()) # Pass a copy to prevent modification of the original dataframe