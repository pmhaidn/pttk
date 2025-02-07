import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def flexplot_analysis(df):
    st.subheader("Flexplot Analysis")
    vars = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(vars) >= 2:
        x_var = st.selectbox("Chọn biến X", vars)
        y_var = st.selectbox("Chọn biến Y", vars)
        
        # Scatter plot với các tùy chọn
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            title=f"Flexplot: {x_var} vs {y_var}"
        )
        
        # Thêm đường hồi quy
        if st.checkbox("Hiển thị đường hồi quy"):
            try:
                #Remove NA values
                df = df.dropna(subset=[x_var, y_var])
                fig.add_scatter(
                    x=df[x_var],
                    y=np.poly1d(np.polyfit(df[x_var], df[y_var], 1))(df[x_var]),
                    name="Đường hồi quy",
                    line=dict(color="red")
                )
            except Exception as e:
              st.error(f"Lỗi khi tính toán đường hồi quy: {e}")
        
        st.plotly_chart(fig)