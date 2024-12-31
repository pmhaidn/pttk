import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math

class LinearRegressionAnalysis:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.plot_template = "plotly_white"

    def sidebar_options(self):
        st.sidebar.title("Hồi quy tuyến tính")

        # Biến phụ thuộc
        dependent_var = st.sidebar.selectbox(
            "Biến phụ thuộc",
            self.numeric_cols,
            key="dependent_var"
        )

        # Phương pháp
        method = st.sidebar.selectbox(
            "Phương pháp",
            ["Enter", "Stepwise", "Backward", "Forward"],
            index=0,
            key="method"
        )

        # Covariates
        st.sidebar.subheader("Biến định lượng")
        covariates = st.sidebar.multiselect(
            "Chọn biến định lượng",
            [col for col in self.numeric_cols if col != dependent_var],
            key="covariates"
        )

        # Factors
        st.sidebar.subheader("Biến định tính")
        factors = st.sidebar.multiselect(
            "Chọn biến định tính",
            self.categorical_cols,
            key="factors"
        )

        # WLS Weights
        st.sidebar.subheader("Trọng số WLS (tùy chọn)")
        wls_weight = st.sidebar.selectbox(
            "Chọn biến trọng số",
            ["None"] + self.numeric_cols,
            index=0,
            key="wls_weight"
        )

        # Model
        st.sidebar.subheader("Mô hình")
        models_container = st.sidebar.container()
        
        
        
        
        
        if 'model_count' not in st.session_state:
            st.session_state.model_count = 2

        models = []
        
        with models_container:
              for model_index in range(getattr(st.session_state, 'model_count', 0)):
                  st.subheader(f"Mô hình {model_index}")
                  
                  model_covariates = st.multiselect(
                       "Chọn biến định lượng",
                       [col for col in self.numeric_cols if col != dependent_var],
                       key=f"model_{model_index}_covariates"
                   )
                  model_factors = st.multiselect(
                       "Chọn biến định tính",
                       self.categorical_cols,
                       key=f"model_{model_index}_factors"
                   )
                  
                  model_vars = []
                  if len(model_covariates) > 0:
                       model_vars.extend(model_covariates)
                  if len(model_factors) > 0:
                       model_vars.extend(model_factors)
                  if len(model_vars) > 0:
                       model_terms = "+".join(model_vars)
                       models.append(model_terms)
            
        with models_container:
            add_model_button = st.button("Thêm Mô Hình", key='add_model')
            if add_model_button:
                 st.session_state.model_count += 1
            
            remove_model_button = st.button("Xóa Mô Hình", key='remove_model')
            if remove_model_button and st.session_state.model_count > 1:
                st.session_state.model_count -=1

        # Include intercept
        include_intercept = st.sidebar.checkbox("Bao gồm hệ số chặn", value=True, key="include_intercept")

        # Statistics options
        st.sidebar.subheader("Thống kê")
        
        # Model Summary
        st.sidebar.subheader("Tóm tắt mô hình")
        model_summary_options = st.sidebar.multiselect(
            "Chọn các thống kê tóm tắt",
            ["R bình phương thay đổi", "F thay đổi", "AIC và BIC", "Durbin-Watson"],
            key="model_summary_options"
        )

        # Coefficients
        st.sidebar.subheader("Hệ số")
        coefficient_options = st.sidebar.multiselect(
            "Chọn các thống kê hệ số",
            ["Ước lượng", "Khoảng tin cậy", "Kiểm định đa cộng tuyến", "Tỷ số p tối đa Vovk-Sellke"],
            key="coefficient_options"
        )

        # Confidence interval settings
        if "Ước lượng" in coefficient_options and "Khoảng tin cậy" in coefficient_options:
            ci_level = st.sidebar.number_input(
                "Độ tin cậy (%)", 
                value=95.0, 
                min_value=0.0, 
                max_value=100.0,
                key="ci_level"
            )
            n_bootstrap = st.sidebar.number_input(
                "Số lượng bootstrap", 
                value=1000, 
                min_value=100,
                key="n_bootstrap"
            )
        else:
            ci_level = None
            n_bootstrap = None

        # Display options
        st.sidebar.subheader("Hiển thị")
        display_options = st.sidebar.multiselect(
            "Chọn các tùy chọn hiển thị",
            ["Kiểm định mô hình", "Thống kê mô tả", "Tương quan và tương quan riêng phần", 
             "Ma trận hiệp phương sai hệ số", "Chẩn đoán đa cộng tuyến"],
            key="display_options"
        )

        # Residuals
        st.sidebar.subheader("Phần dư")
        residual_options = st.sidebar.multiselect(
            "Chọn các tùy chọn phần dư",
            ["Thống kê", "Chẩn đoán casewise"],
            key="residual_options"
        )

        if "Chẩn đoán casewise" in residual_options:
            casewise_options = st.sidebar.multiselect(
                "Chọn các chẩn đoán casewise",
                ["Phần dư chuẩn hóa", "Khoảng cách Cook", "DFBETAS", "DFFITS", 
                 "Tỷ số hiệp phương sai", "Đòn bẩy"],
                key="casewise_options"
            )
        else:
            casewise_options = None

        # Plots
        show_residual_plots = st.sidebar.checkbox("Biểu đồ phần dư", value=False, key="show_residual_plots")
        show_qq_plot = st.sidebar.checkbox("Biểu đồ Q-Q", value=False, key="show_qq_plot")
        show_partial_plots = st.sidebar.checkbox("Biểu đồ hồi quy từng phần", value=False, key="show_partial_plots")

        return {
            'dependent_var': dependent_var,
            'method': method,
            'covariates': covariates,
            'factors': factors,
            'wls_weight': wls_weight,
            'models': models,
            'include_intercept': include_intercept,
            'model_summary_options': model_summary_options,
            'coefficient_options': coefficient_options,
            'ci_level': ci_level,
            'n_bootstrap': n_bootstrap,
            'display_options': display_options,
            'residual_options': residual_options,
            'casewise_options': casewise_options,
            'show_residual_plots': show_residual_plots,
            'show_qq_plot': show_qq_plot,
            'show_partial_plots': show_partial_plots
        }

    def fit_model(self, X, y, method='Enter', weights=None):
        if weights is not None and weights != "None":
            model = sm.WLS(y, X, weights=self.df[weights]).fit()
        else:
            model = sm.OLS(y, X).fit()
        return model

    def create_model_summary(self, models, X_list, y, options):
        results_list = []
        
        if len(models) == 1:
          model = models[0]
          X = X_list[0]
          results = {
                'Model': "M0",
                'R': np.sqrt(model.rsquared),
                'R²': model.rsquared,
                'Adjusted R²': model.rsquared_adj,
                'RMSE': np.sqrt(model.mse_resid),
                'F': model.fvalue,
                'p': model.f_pvalue,
                'df_model': model.df_model,
                'df_resid': model.df_resid
            }
            
          if "AIC và BIC" in options["model_summary_options"]:
                results["AIC"] = model.aic
                results["BIC"] = model.bic
            
          if "Durbin-Watson" in options["model_summary_options"]:
                durbin_watson = sm.stats.durbin_watson(model.resid)
                results["Durbin-Watson"] = durbin_watson
          results_list.append(results)
          
        else:
            for model_index, model in enumerate(models):
                X = X_list[model_index]
                results = {
                    'Model': f"M{model_index}",
                    'R': np.sqrt(model.rsquared),
                    'R²': model.rsquared,
                    'Adjusted R²': model.rsquared_adj,
                    'RMSE': np.sqrt(model.mse_resid),
                   
                    'df_model': model.df_model,
                    'df_resid': model.df_resid
                }
                
                if model_index > 0:
                   prev_model = models[model_index-1]
                   results['R² Change'] = model.rsquared - prev_model.rsquared
                   
                   if "F thay đổi" in options["model_summary_options"]:
                        f_change = ( (model.ssr - prev_model.ssr)/(prev_model.df_resid - model.df_resid) ) / (model.mse_resid)
                        
                        results['F Change'] = f_change
                        results['df1'] = prev_model.df_resid - model.df_resid
                        results['df2'] = model.df_resid
                        results['p'] = stats.f.sf(f_change, prev_model.df_resid - model.df_resid ,model.df_resid)
                        
                
                if "AIC và BIC" in options["model_summary_options"]:
                    results["AIC"] = model.aic
                    results["BIC"] = model.bic
                
                if "Durbin-Watson" in options["model_summary_options"]:
                    durbin_watson = sm.stats.durbin_watson(model.resid)
                    results["Durbin-Watson"] = durbin_watson
                
                if model_index == 0 and "F thay đổi" in options["model_summary_options"]:
                  results['F Change'] = model.fvalue
                  results['df1'] = model.df_model
                  results['df2'] = model.df_resid
                  results['p'] = model.f_pvalue
                  
                results_list.append(results)

        return pd.DataFrame(results_list)
      
    def create_anova_table(self, models):
      anova_list = []
      for model_index, model in enumerate(models):
        anova_table = sm.stats.anova_lm(model, typ=1)
        anova_table.insert(0, 'Model', f'M{model_index}')
        anova_list.append(anova_table)
      
      return pd.concat(anova_list)
      

    def create_coefficient_table(self, models, options):
        coef_list = []
        for model_index, model in enumerate(models):
          coef_df = pd.DataFrame({
            'Model': f"M{model_index}",
              'Hệ số': model.params,
              'Sai số chuẩn': model.bse,
              't': model.tvalues,
              'p': model.pvalues
          })
          
          if "Khoảng tin cậy" in options["coefficient_options"] and options['ci_level'] is not None:
            ci = model.conf_int(alpha=1 - (options['ci_level'] / 100))
            coef_df['Khoảng tin cậy thấp'] = ci[0]
            coef_df['Khoảng tin cậy cao'] = ci[1]
          
          if "Tỷ số p tối đa Vovk-Sellke" in options["coefficient_options"]:
              p_values = model.pvalues
              p_values = p_values[1:]
              
              def vovk_sellke_max_p_ratio(p):
                if p <= 0.5:
                  return - p * np.log(p)
                else:
                  return np.nan
  
              max_p_ratios = [vovk_sellke_max_p_ratio(p) for p in p_values]
              
              
              max_p_ratios = [np.nan if p is np.nan else p for p in max_p_ratios]
              coef_df['Tỷ số p tối đa Vovk-Sellke'] = [np.nan] + max_p_ratios

          coef_list.append(coef_df)
        
        return pd.concat(coef_list)

    def calculate_vif(self, X_list):
      vif_list = []
      for model_index, X in enumerate(X_list):
        vif_data = pd.DataFrame()
        vif_data["Biến"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data["Model"] = f'M{model_index}'
        vif_list.append(vif_data)
      return pd.concat(vif_list)
    

    def create_residual_plots(self, model):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=model.fittedvalues,
            y=model.resid,
            mode='markers',
            name='Phần dư'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Biểu đồ phần dư',
            xaxis_title='Giá trị dự báo',
            yaxis_title='Phần dư',
            template=self.plot_template
        )
        return fig

    def create_qq_plot(self, residuals):
        fig = go.Figure()
        
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q'
        ))
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Đường tham chiếu',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Biểu đồ Q-Q',
            xaxis_title='Phân vị lý thuyết',
            yaxis_title='Phân vị thực nghiệm',
            template=self.plot_template
        )
        return fig
    
    def calculate_descriptive_stats(self, X_list, y):
      desc_list = []
      for model_index, X in enumerate(X_list):
          data = pd.concat([y, X], axis=1)
          descriptive_stats = data.describe()
          descriptive_stats.insert(0, 'Model', f'M{model_index}')
          desc_list.append(descriptive_stats)
      return pd.concat(desc_list)
      
    
    def calculate_correlation_matrix(self, X_list, y):
        corr_list = []
        for model_index, X in enumerate(X_list):
          data = pd.concat([y, X], axis=1)
          correlation_matrix = data.corr()
          correlation_matrix.insert(0, 'Model', f'M{model_index}')
          corr_list.append(correlation_matrix)
        
        return pd.concat(corr_list)
    
    def calculate_partial_correlation_matrix(self, X_list, y):
       partial_corr_list = []
       for model_index, X in enumerate(X_list):
        num_vars = X.shape[1]
        partial_corr_matrix = np.zeros((num_vars+1, num_vars+1))
        
        data = pd.concat([y, X], axis=1)
        
        
        for i in range(num_vars+1):
            for j in range(num_vars+1):
                if i==0 and j==0:
                   partial_corr_matrix[i, j] = 1
                   continue
                elif i==0:
                   var_i = data.iloc[:,0]
                   var_j = data.iloc[:,j]
                elif j==0:
                   var_i = data.iloc[:,i]
                   var_j = data.iloc[:,0]
                else:
                   var_i = data.iloc[:,i]
                   var_j = data.iloc[:,j]
                   
                covariates = list(data.columns)
                covariates.remove(var_i.name)
                covariates.remove(var_j.name)
                
                if len(covariates) > 0:
                  X_cov = data[covariates]
                  X_cov = sm.add_constant(X_cov)
                  model_i = sm.OLS(var_i, X_cov).fit()
                  model_j = sm.OLS(var_j, X_cov).fit()
                  
                  res_i = model_i.resid
                  res_j = model_j.resid
                  
                  partial_corr = np.corrcoef(res_i, res_j)[0, 1]
                  partial_corr_matrix[i, j] = partial_corr
                else:
                    
                    partial_corr = np.corrcoef(var_i, var_j)[0, 1]
                    partial_corr_matrix[i, j] = partial_corr
                    
        partial_corr_df = pd.DataFrame(partial_corr_matrix, index=data.columns, columns=data.columns)
        partial_corr_df.insert(0, 'Model', f'M{model_index}')
        partial_corr_list.append(partial_corr_df)
       return pd.concat(partial_corr_list)
       
    
    def create_coefficient_covariance_matrix(self, models):
        cov_list = []
        for model_index, model in enumerate(models):
          cov_matrix = model.cov_params()
          cov_df = pd.DataFrame(cov_matrix, index=cov_matrix.index, columns=cov_matrix.columns)
          cov_df.insert(0, 'Model', f'M{model_index}')
          cov_list.append(cov_df)
        return pd.concat(cov_list)
    
    def calculate_casewise_diagnostics(self, models, X_list, options):
      results_list = []
      for model_index, model in enumerate(models):
        X = X_list[model_index]
        influence = model.get_influence(observed=False)
        
        results_df = pd.DataFrame()
        results_df["Model"] = f'M{model_index}'
        
        if "Phần dư chuẩn hóa" in options:
            studentized_residuals = influence.resid_studentized
            results_df['Phần dư chuẩn hóa'] = studentized_residuals
        
        if "Khoảng cách Cook" in options:
            cooks_distance = influence.cooks_distance[0]
            results_df['Khoảng cách Cook'] = cooks_distance
        
        if "DFBETAS" in options:
            dfbetas = influence.dfbetas
            results_df = pd.concat([results_df, pd.DataFrame(dfbetas, columns=[f"DFBETAS_{col}" for col in X.columns])], axis=1)
        
        if "DFFITS" in options:
            dffits = influence.dffits[0]
            results_df['DFFITS'] = dffits
        
        if "Tỷ số hiệp phương sai" in options:
            cov_ratio = influence.cov_ratio
            results_df['Tỷ số hiệp phương sai'] = cov_ratio
        
        if "Đòn bẩy" in options:
            leverage = influence.hat_matrix_diag
            results_df['Đòn bẩy'] = leverage
        
        results_list.append(results_df)
      return pd.concat(results_list)
      

    def regression_analysis(self):
        st.title("Kết quả phân tích hồi quy tuyến tính")
        
        options = self.sidebar_options()
        
        if not options['models']:
            st.warning("Vui lòng chọn ít nhất một biến độc lập.")
            return
        
        
        y = self.df[options['dependent_var']]
        X_list = []
        models = []
        for model_formula in options['models']:
            # Prepare data
            X = pd.get_dummies(self.df[model_formula.split('+')], drop_first=True)
            
            if options['include_intercept']:
                X = sm.add_constant(X)
            X_list.append(X)
            # Fit model
            model = self.fit_model(X, y, method=options['method'], 
                                 weights=options['wls_weight'])
            models.append(model)
        
        # Display results based on selected options
        if len(options['model_summary_options']) > 0:
            st.markdown("### Tóm tắt mô hình")
            summary_df = self.create_model_summary(models, X_list, y, options)
            st.dataframe(summary_df)
            
        if "Kiểm định mô hình" in options["display_options"]:
            st.markdown("### Bảng ANOVA")
            anova_df = self.create_anova_table(models)
            st.dataframe(anova_df)
        
        
        if "Ước lượng" in options['coefficient_options']:
            st.markdown("### Bảng hệ số")
            coef_df = self.create_coefficient_table(models, options)
            st.dataframe(coef_df)
        
        if "Kiểm định đa cộng tuyến" in options['coefficient_options'] or "Chẩn đoán đa cộng tuyến" in options["display_options"]:
            st.markdown("### Kiểm định đa cộng tuyến")
            vif_df = self.calculate_vif(X_list)
            st.dataframe(vif_df)
            
        if "Thống kê mô tả" in options['display_options']:
            st.markdown("### Thống kê mô tả")
            descriptive_df = self.calculate_descriptive_stats(X_list, y)
            st.dataframe(descriptive_df)
        
        if "Tương quan và tương quan riêng phần" in options["display_options"]:
            st.markdown("### Ma trận tương quan")
            corr_df = self.calculate_correlation_matrix(X_list, y)
            st.dataframe(corr_df)
            
            st.markdown("### Ma trận tương quan riêng phần")
            partial_corr_df = self.calculate_partial_correlation_matrix(X_list, y)
            st.dataframe(partial_corr_df)
            
        if "Ma trận hiệp phương sai hệ số" in options["display_options"]:
            st.markdown("### Ma trận hiệp phương sai hệ số")
            cov_matrix = self.create_coefficient_covariance_matrix(models)
            st.dataframe(cov_matrix)

        
        if options['show_residual_plots']:
            for model_index, model in enumerate(models):
              st.markdown(f"### Biểu đồ phần dư - Model M{model_index}")
              fig = self.create_residual_plots(model)
              st.plotly_chart(fig)
        
        if options['show_qq_plot']:
            for model_index, model in enumerate(models):
              st.markdown(f"### Biểu đồ Q-Q - Model M{model_index}")
              fig = self.create_qq_plot(model.resid)
              st.plotly_chart(fig)
        
        if "Thống kê" in options['residual_options']:
            for model_index, model in enumerate(models):
              st.markdown(f"### Thống kê phần dư - Model M{model_index}")
              st.dataframe(pd.DataFrame({"Min": [model.resid.min()], "Max": [model.resid.max()], "Mean": [model.resid.mean()], "Median": [np.median(model.resid)],
                                         "Std":[model.resid.std()]}))
        
        if "Chẩn đoán casewise" in options['residual_options'] and options['casewise_options'] is not None:
            st.markdown("### Chẩn đoán casewise")
            casewise_df = self.calculate_casewise_diagnostics(models, X_list, options['casewise_options'])
            st.dataframe(casewise_df)