import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import colorsys
from scipy import stats


class RaincloudPlot:
    def __init__(self):
        self.default_color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "viridis": ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#90d743", "#fde725"]
        }

    def _calculate_statistics(self, data: pd.Series) -> Dict:
        """Calculate basic statistics for the data."""
        return {
            'mean': data.mean(),
            'median': data.median(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'whisker_low': data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25)),
            'whisker_high': data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)),
            'std': data.std(),
            'n': len(data)
        }

    def _calculate_ci(self, data: pd.Series, ci_width: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""

        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(ci_width, len(data) - 1, loc=mean, scale=se)
        return ci[0], ci[1]

    def _create_violin_trace(self, data: pd.Series, name: str, color: str,
                             violin_config: Dict) -> go.Violin:
        """Create a violin plot trace with custom configuration."""
        return go.Violin(
            y=data,
            name=name,
            side=violin_config.get('side', 'negative'),
            line_color=color if violin_config.get('outline') == 'colorPalette' else 'black',
            line_width=violin_config.get('outline_width', 1),
            fillcolor=color,
            opacity=violin_config.get('opacity', 0.5),
            points=False,
            meanline_visible=False,
            showlegend=violin_config.get('show_legend', True),
            bandwidth=violin_config.get('smoothing', 0.5),
            orientation='h' if violin_config.get('horizontal', False) else 'v',
            visible=violin_config.get('show_violin', True)
        )

    def _create_box_trace(self, data: pd.Series, name: str, color: str,
                           box_config: Dict) -> go.Box:
        """Create a box plot trace with custom configuration."""
        return go.Box(
            y=data,
            name=name,
            line_color=color if box_config.get('outline') == 'colorPalette' else 'black',
            line_width=box_config.get('outline_width', 1),
            fillcolor=color,
            opacity=box_config.get('opacity', 0.5),
            boxpoints=False,
            showlegend=False,
            orientation='h' if box_config.get('horizontal', False) else 'v',
            visible=box_config.get('show_box', True),
            boxmean=box_config.get('show_mean', False)
        )

    def _create_points_trace(self, data: pd.Series, x_pos: Union[float, List[float]],
                             name: str, color: str, point_config: Dict) -> go.Scatter:
        """Create a scatter plot trace for individual points."""
        jitter = point_config.get('jitter', 0)
        if jitter > 0:
            y_jitter = np.random.normal(0, jitter, len(data))
            y_data = data + y_jitter
        else:
            y_data = data

        return go.Scatter(
            y=y_data,
            x=x_pos,
            name=name,
            mode='markers',
            marker=dict(
                color=color,
                size=point_config.get('size', 3),
                opacity=point_config.get('opacity', 0.5)
            ),
            showlegend=False
        )

    def _add_mean_and_intervals(self, fig: go.Figure, data: pd.Series, x_pos: float,
                               color: str, mean_config: Dict):
        """Add mean point and intervals to the plot."""
        stats = self._calculate_statistics(data)

        # Add mean point
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[stats['mean']],
            mode='markers',
            marker=dict(
                color=color,
                size=mean_config.get('size', 6),
                symbol='diamond'
            ),
            showlegend=False
        ))

        # Add intervals if requested
        if mean_config.get('show_interval', False):
            if mean_config.get('interval_type') == 'ci':
                lower, upper = self._calculate_ci(data, mean_config.get('ci_width', 0.95))
            else:  # Standard error
                lower = stats['mean'] - stats['std'] / np.sqrt(stats['n'])
                upper = stats['mean'] + stats['std'] / np.sqrt(stats['n'])

            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[lower, upper],
                mode='lines',
                line=dict(
                    color=color,
                    width=mean_config.get('interval_width', 1)
                ),
                showlegend=False
            ))

    def create_raincloud_plot(self,
                             df: pd.DataFrame,
                             dependent_vars: List[str],
                             primary_factor: Optional[str] = None,
                             secondary_factor: Optional[str] = None,
                             covariate: Optional[str] = None,
                             observation_id: Optional[str] = None,
                             config: Dict = None) -> None:

        if config is None:
            config = {
                'violin': {'show_violin': True, 'opacity': 0.5},
                'box': {'show_box': True, 'opacity': 0.5},
                'points': {'show_points': True, 'opacity': 0.5},
                'mean': {'show_mean': False},
                'layout': {'horizontal': False, 'width': 600, 'height': 400}
            }

        for var in dependent_vars:
            fig = go.Figure()

            # Handle grouping and colors
            if primary_factor is not None:
                groups = df[primary_factor].unique()
                if secondary_factor is not None:
                    subgroups = df[secondary_factor].unique()
                    colors = self.default_color_palettes['default'][:len(subgroups)]
                else:
                    subgroups = [None]
                    colors = self.default_color_palettes['default'][:1]
            else:
                groups = [None]
                subgroups = [None]
                colors = self.default_color_palettes['default'][:1]

            # Create traces for each group/subgroup combination
            for i, group in enumerate(groups):
                for j, subgroup in enumerate(subgroups):
                    # Filter data
                    mask = pd.Series(True, index=df.index)  # Khởi tạo mask với True cho tất cả các hàng
                    if group is not None:
                        mask = mask & (df[primary_factor] == group)
                    if subgroup is not None:
                        mask = mask & (df[secondary_factor] == subgroup)

                    st.write(f"Giá trị của mask trước loc cho var {var}, group {group}, subgroup {subgroup}:")
                    st.write(mask)

                    data = df.loc[mask, var]
                    if len(data) == 0:
                        continue

                    # Calculate x position
                    x_base = i + (j - len(subgroups) / 2) * config.get('box', {}).get('width', 0.1)

                    # Add violin
                    if config['violin']['show_violin']:
                        fig.add_trace(self._create_violin_trace(
                            data, f"{group}-{subgroup}" if subgroup else str(group),
                            colors[j % len(colors)], config['violin']
                        ))

                    # Add box
                    if config['box']['show_box']:
                        fig.add_trace(self._create_box_trace(
                            data, f"{group}-{subgroup}" if subgroup else str(group),
                            colors[j % len(colors)], config['box']
                        ))

                    # Add points
                    if config['points']['show_points']:
                        fig.add_trace(self._create_points_trace(
                            data, x_base, f"{group}-{subgroup}" if subgroup else str(group),
                            colors[j % len(colors)], config['points']
                        ))

                    # Add mean and intervals if requested
                    if config['mean'].get('show_mean', False):
                        self._add_mean_and_intervals(fig, data, x_base, colors[j % len(colors)], config['mean'])

                    # Add ID lines if requested
                    if observation_id is not None and primary_factor is not None:
                        self._add_id_lines(fig, df, var, primary_factor, observation_id,
                                            secondary_factor, subgroup, colors[j % len(colors)],
                                            config.get('id_lines', {}))

        # Update layout
        self._update_layout(fig, var, groups, subgroups, config['layout'])

        # Display plot
        st.plotly_chart(fig)

    def _add_id_lines(self, fig: go.Figure, df: pd.DataFrame, var: str,
                      primary_factor: str, observation_id: str,
                      secondary_factor: Optional[str] = None,
                      subgroup: Optional[str] = None,
                      color: str = 'gray',
                      line_config: Dict = None):
        """Add lines connecting observations for the same ID."""
        if line_config is None:
            line_config = {'opacity': 0.25, 'width': 1}

        # Filter data for subgroup if necessary
        if secondary_factor is not None and subgroup is not None:
            df = df[df[secondary_factor] == subgroup]

        # Get unique IDs
        ids = df[observation_id].unique()

        # Add lines for each ID
        for id_val in ids:
            id_data = df[df[observation_id] == id_val]
            if len(id_data) > 1:
                x_values = []
                for val in id_data[primary_factor]:
                    try:
                        x_values.append(list(df[primary_factor].unique()).index(val))
                    except ValueError:
                        st.warning(f"Value '{val}' not found in primary factor. Skipping ID line.")
                        continue
                if len(x_values) < 2:
                    continue

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=id_data[var],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=line_config.get('width', 1),
                        opacity=line_config.get('opacity', 0.25)
                    ),
                    showlegend=False
                ))

    def _update_layout(self, fig: go.Figure, var_name: str, groups: List,
                      subgroups: List, layout_config: Dict):
        """Update the figure layout with custom configuration."""
        title = f"Biểu đồ Raincloud của {var_name}"
        if layout_config.get('horizontal', False):
            fig.update_layout(
                title=title,
                xaxis_title=var_name,
                yaxis_title="Nhóm",
                width=layout_config.get('width', 600),
                height=layout_config.get('height', 400),
                boxmode='group',
                violinmode='group'
            )
        else:
            fig.update_layout(
                title=title,
                xaxis_title="Nhóm",
                yaxis_title=var_name,
                width=layout_config.get('width', 600),
                height=layout_config.get('height', 400),
                boxmode='group',
                violinmode='group'
            )

        # Update axis limits if specified
        if 'axis_limits' in layout_config:
            if layout_config.get('horizontal', False):
                fig.update_xaxes(range=layout_config['axis_limits'])
            else:
                fig.update_yaxes(range=layout_config['axis_limits'])

        # Update caption if specified
        if layout_config.get('show_caption', True):
            fig.add_annotation(
                text="Tạo bởi RaincloudPlot",
                xref="paper",
                yref="paper",
                x=0,
                y=-0.2,
                showarrow=False,
                font=dict(size=10)
            )


def raincloud_plots_analysis(df: pd.DataFrame, **kwargs):
    """
    Main function to create raincloud plots with streamlit interface.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data frame
    **kwargs :
        Additional configuration parameters
    """
    st.sidebar.header("Cấu hình biểu đồ Raincloud")

    # Initialize RaincloudPlot class
    plotter = RaincloudPlot()

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Streamlit interface for plot configuration
    selected_vars = st.sidebar.multiselect(
        "Chọn biến phụ thuộc",
        numeric_cols,
        default=numeric_cols[0] if numeric_cols else None
    )

    primary_factor = st.sidebar.selectbox(
        "Yếu tố chính (tùy chọn)",
        [None] + object_cols
    )

    secondary_factor = st.sidebar.selectbox(
        "Yếu tố phụ (tùy chọn)",
        [None] + object_cols
    )

    covariate = st.sidebar.selectbox(
        "Biến đồng biến (tùy chọn)",
        [None] + object_cols + numeric_cols
    )

    observation_id = st.sidebar.selectbox(
        "ID quan sát (tùy chọn)",
        [None] + object_cols,
        disabled=primary_factor is None
    )

    show_violin = st.sidebar.checkbox("Hiển thị violin", value=True)
    violin_nudge = st.sidebar.number_input("Violin Nudge", value=0.15)
    violin_height = st.sidebar.number_input("Violin Height", value=0.7)
    violin_opacity = st.sidebar.slider("Độ mờ Violin", 0.0, 1.0, 0.5)
    violin_outline = st.sidebar.selectbox("Đường viền Violin", ["Color palette", "black", "none"])
    violin_outline_width = st.sidebar.number_input("Độ rộng đường viền Violin", value=1.0)
    violin_smoothing = st.sidebar.slider("Độ mịn Violin", 0.0, 1.0, 1.0)

    show_box = st.sidebar.checkbox("Hiển thị box", value=True)
    box_nudge = st.sidebar.number_input("Box Nudge", value=0.0)
    box_width = st.sidebar.number_input("Box Width", value=0.2)
    box_padding = st.sidebar.number_input("Box Padding", value=0.1)
    box_opacity = st.sidebar.slider("Độ mờ Box", 0.0, 1.0, 0.5)
    box_outline = st.sidebar.selectbox("Đường viền Box", ["Color palette", "black", "none"])
    box_outline_width = st.sidebar.number_input("Độ rộng đường viền Box", value=1.0)

    show_points = st.sidebar.checkbox("Hiển thị điểm", value=True)
    point_nudge = st.sidebar.number_input("Điểm Nudge", value=0.19)
    point_spread = st.sidebar.number_input("Độ lan tỏa điểm", value=0.065)
    point_size = st.sidebar.number_input("Kích thước điểm", value=2.5)
    point_opacity = st.sidebar.slider("Độ mờ điểm", 0.0, 1.0, 0.5)
    point_jitter = st.sidebar.checkbox("Jitter điểm", value=False)

    show_mean = st.sidebar.checkbox("Hiển thị giá trị trung bình", value=False)
    mean_position = st.sidebar.selectbox("Vị trí giá trị trung bình", ["likeBox", "onAxisTicks"])
    mean_size = st.sidebar.number_input("Kích thước giá trị trung bình", value=6)
    mean_lines = st.sidebar.checkbox("Kết nối giá trị trung bình bằng đường thẳng", value=False)
    mean_lines_opacity = st.sidebar.slider("Độ mờ đường thẳng trung bình", 0.0, 1.0, 0.5)
    mean_lines_width = st.sidebar.number_input("Độ rộng đường thẳng trung bình", value=1.0)

    mean_interval = st.sidebar.checkbox("Hiển thị khoảng tin cậy trung bình", value=False)
    mean_interval_option = st.sidebar.selectbox("Loại khoảng tin cậy", ["ci", "se"], disabled=not show_mean)
    mean_ci_width = st.sidebar.slider("Độ rộng CI", 0.5, 0.99, 0.95, disabled=not show_mean)
    interval_outline_width = st.sidebar.number_input("Độ rộng đường viền khoảng tin cậy", value=1.0)

    custom_axis_limits = st.sidebar.checkbox("Giới hạn trục tùy chỉnh")
    lower_axis_limit = st.sidebar.number_input("Giới hạn dưới trục", value=float(df[selected_vars].min().min()) if selected_vars else 0.0, disabled=not custom_axis_limits)
    upper_axis_limit = st.sidebar.number_input("Giới hạn trên trục", value=float(df[selected_vars].max().max()) if selected_vars else 1000.0, disabled=not custom_axis_limits)

    horizontal = st.sidebar.checkbox("Biểu đồ ngang", value=False)
    width_plot = st.sidebar.number_input("Chiều rộng biểu đồ", value=600)
    height_plot = st.sidebar.number_input("Chiều cao biểu đồ", value=400)
    show_caption = st.sidebar.checkbox("Hiển thị chú thích", value=True)

    color_palette = st.sidebar.selectbox("Bảng màu", ["default", "viridis"])

    custom_sides = st.sidebar.checkbox("Áp dụng hướng tùy chỉnh")
    mean_interval_custom = st.sidebar.checkbox("Áp dụng giới hạn khoảng tin cậy tùy chỉnh", disabled=not show_mean)
    custom_colors = st.sidebar.checkbox("Áp dụng màu tùy chỉnh", disabled=secondary_factor is not None)

    n_clouds = st.sidebar.number_input(
        "Số lượng cloud đang hiển thị?",
        min_value=1,
        value=len(df[primary_factor].unique()) if primary_factor else 1
    )

    if observation_id:
        observation_id_line_opacity = st.sidebar.slider("Độ mờ đường ID quan sát", 0.0, 1.0, 0.25)
        observation_id_line_width = st.sidebar.slider("Độ rộng đường ID quan sát", 0.5, 3.0, 1.0)
    else:
        observation_id_line_opacity = 0.25
        observation_id_line_width = 1.0

    table = st.sidebar.checkbox("Bảng với thống kê", value=False)
    table_box_statistics = st.sidebar.checkbox("Thống kê box", value=True, disabled=not table)

    # Create configuration dictionary
    config = {
        'violin': {
            'show_violin': show_violin,
            'opacity': violin_opacity,
            'outline': violin_outline,
            'outline_width': violin_outline_width,
            'smoothing': violin_smoothing,
            'side': 'negative',
            'nudge': violin_nudge,
            'height': violin_height,
            'horizontal': horizontal
        },
        'box': {
            'show_box': show_box,
            'opacity': box_opacity,
            'outline': box_outline,
            'outline_width': box_outline_width,
            'width': box_width,
            'padding': box_padding,
            'nudge': box_nudge,
            'horizontal': horizontal
        },
        'points': {
            'show_points': show_points,
            'opacity': point_opacity,
            'size': point_size,
            'jitter': point_jitter,
            'nudge': point_nudge,
            'spread': point_spread,
            'horizontal': horizontal
        },
        'mean': {
            'show_mean': show_mean,
            'size': mean_size,
            'show_interval': mean_interval,
            'interval_type': mean_interval_option,
            'ci_width': mean_ci_width,
            'interval_width': interval_outline_width,
            'lines': mean_lines,
            'lines_opacity': mean_lines_opacity,
            'lines_width': mean_lines_width
        },
        'layout': {
            'horizontal': horizontal,
            'width': width_plot,
            'height': height_plot,
            'show_caption': show_caption
        },
        'color_palette': color_palette,
        'custom_sides': custom_sides,
        'mean_interval_custom': mean_interval_custom,
        'custom_colors': custom_colors,
        'n_clouds': n_clouds,
        'id_lines': {
            'opacity': observation_id_line_opacity,
            'width': observation_id_line_width
        },
        'table': {
            'show': table,
            'box_statistics': table_box_statistics
        }
    }

    if custom_axis_limits:
        config['layout']['axis_limits'] = [lower_axis_limit, upper_axis_limit]

    if primary_factor is not None and observation_id is not None:
        config['id_lines'] = {
            'opacity': observation_id_line_opacity,
            'width': observation_id_line_width
        }

    custom_config = {
        'orientations': [],
        'colors': [],
        'intervals': []
    }

    if st.sidebar.checkbox("Bảng cấu hình nâng cao"):
        st.sidebar.write("Bảng Cấu Hình Nâng Cao")
        for i in range(n_clouds):
            col1, col2, col3, col4 = st.sidebar.columns(4)
            with col1:
                orientation = st.selectbox(
                    f"Cloud {i + 1} Orientation",
                    ['L', 'R'],
                    key=f"orientation_{i}"
                )
                custom_config['orientations'].append(orientation)

            with col2:
                color = st.color_picker(
                    f"Cloud {i + 1} Color",
                    key=f"color_{i}"
                )
                custom_config['colors'].append(color)

            if show_mean and mean_interval:
                with col3:
                    lower = st.number_input(
                        f"Cloud {i + 1} Lower Limit",
                        key=f"lower_{i}"
                    )
                with col4:
                    upper = st.number_input(
                        f"Cloud {i + 1} Upper Limit",
                        key=f"upper_{i}"
                    )
                custom_config['intervals'].append((lower, upper))

        config['custom'] = custom_config

    # Generate plots
    if selected_vars:
        plotter.create_raincloud_plot(
            df,
            selected_vars,
            primary_factor,
            secondary_factor,
            covariate,
            observation_id,
            config
        )

        # Display statistics table if requested
        if config.get('table', {}).get('show', False):
            for var in selected_vars:
                st.subheader(f"Thống kê cho {var}")
                stats_df = _create_statistics_table(
                    df, var, primary_factor, secondary_factor,
                    config['table'].get('box_statistics', True)
                )
                st.dataframe(stats_df)

            

    # Add information and instructions to the main area
    st.title("Phân tích biểu đồ Raincloud")
    st.markdown(
        """
        Chào mừng đến với công cụ tạo biểu đồ Raincloud!

        - **Hướng dẫn:** Tải lên dữ liệu của bạn ở định dạng CSV và sử dụng bảng điều khiển bên trái để tùy chỉnh biểu đồ.
        - **Yêu cầu dữ liệu:** Dữ liệu phải ở định dạng dài.
        - **Thông tin thêm:**
            - Các tùy chọn cấu hình nâng cao có thể được tìm thấy trong bảng "Cấu hình nâng cao" ở thanh bên.
            - Thống kê có thể được hiển thị trong một bảng bằng cách chọn hộp kiểm "Bảng với thống kê" ở thanh bên.
        """
    )


def _create_statistics_table(df: pd.DataFrame,
                             var: str,
                             primary_factor: Optional[str] = None,
                             secondary_factor: Optional[str] = None,
                             include_box_stats: bool = True) -> pd.DataFrame:
    """
    Create a statistics table for the given variable and grouping factors.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data frame
    var : str
        Variable name to compute statistics for
    primary_factor : str, optional
        Primary grouping factor
    secondary_factor : str, optional
        Secondary grouping factor
    include_box_stats : bool
        Whether to include box plot statistics

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing computed statistics
    """
    stats_list = []

    if primary_factor is None:
        groups = [('All', 'All')]
    elif secondary_factor is None:
        groups = [(g, 'All') for g in df[primary_factor].unique()]
    else:
        groups = [(p, s) for p in df[primary_factor].unique()
                  for s in df[secondary_factor].unique()]

    for prim, sec in groups:
        # Filter data
        mask = True
        if prim != 'All':
            mask = mask & (df[primary_factor] == prim)
        if sec != 'All':
            mask = mask & (df[secondary_factor] == sec)

        data = df.loc[mask, var]

        # Calculate statistics
        stats = {
            'Primary': prim,
            'Secondary': sec,
            'N': len(data),
            'Mean': data.mean(),
            'SE': data.std() / np.sqrt(len(data)),
            'SD': data.std()
        }

        if include_box_stats:
            stats.update({
                'Median': data.median(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'Lower Whisker': data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25)),
                'Upper Whisker': data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))
            })

        stats_list.append(stats)

    return pd.DataFrame(stats_list)


# Example usage
if __name__ == "__main__":
    st.title("Công cụ tạo biểu đồ Raincloud")

    # File upload
    uploaded_file = st.file_uploader("Tải lên dữ liệu của bạn (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            raincloud_plots_analysis(df)
        except Exception as e:
            st.error(f"Lỗi đọc tệp: {str(e)}")
    else:
        # Example data
        example_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(2, 1.5, 100)
            ]),
            'group': ['A'] * 100 + ['B'] * 100,
            'subgroup': ['X', 'Y'] * 100
        })

        st.write("Sử dụng dữ liệu ví dụ:")
        raincloud_plots_analysis(example_data)