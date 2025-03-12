import streamlit as st
import pandas as pd
import numpy as np
from semopy import Model
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def create_graph(model, standardized=False):
    """Tạo đồ thị từ mô hình SEM"""
    G = nx.DiGraph()
    
    # Lấy thông tin tham số từ mô hình
    params = model.inspect(standardized=standardized)
    
    # Thêm các nút và cạnh vào đồ thị
    for idx, row in params.iterrows():
        source, op, target = idx
        if op == '~':
            # Thêm nút
            G.add_node(source)
            G.add_node(target)
            
            # Thêm cạnh với trọng số
            weight = row['Estimate']
            std_err = row['Std. Err']
            p_value = row['P-value']
            
            # Định dạng nhãn cạnh
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''
                
            edge_label = f'{weight:.2f}{sig}\n(SE={std_err:.2f})'
            
            G.add_edge(target, source, 
                      weight=abs(weight),
                      label=edge_label,
                      color='green' if weight > 0 else 'red')
    
    return G

def plot_path_diagram(G, title='Path Diagram'):
    """Vẽ sơ đồ đường dẫn"""
    plt.figure(figsize=(12, 8))
    
    # Tạo layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Vẽ các nút
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightblue',
                          node_size=2000,
                          alpha=0.7)
    
    # Vẽ nhãn nút
    nx.draw_networkx_labels(G, pos,
                           font_size=10,
                           font_weight='bold')
    
    # Vẽ các cạnh
    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=colors,
                          width=weights,
                          arrowsize=20,
                          alpha=0.7)
    
    # Vẽ nhãn cạnh
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=8)
    
    plt.title(title)
    plt.axis('off')
    return plt.gcf()

def run_analysis(df):
    st.header("Path Diagram")
    
    if 'model' not in st.session_state:
        st.error("Vui lòng định nghĩa mô hình trước.")
        return
    
    model = st.session_state.model
    
    # Kiểm tra xem mô hình đã được ước lượng chưa
    try:
        # Thử truy cập một thuộc tính chỉ có sau khi ước lượng
        _ = model.inspect()
    except:
        st.error("Vui lòng ước lượng mô hình trước khi vẽ sơ đồ đường dẫn.")
        return
    
    # Cấu hình hiển thị
    with st.sidebar:
        st.subheader("Diagram Settings")
        
        # Tùy chọn hiển thị hệ số chuẩn hóa
        show_standardized = st.checkbox("Hiển thị hệ số chuẩn hóa", value=False)
        
        # Tùy chọn hiển thị ý nghĩa thống kê
        show_significance = st.checkbox("Hiển thị ý nghĩa thống kê", value=True)
        
        # Tùy chọn màu sắc
        edge_color_positive = st.color_picker("Màu đường dẫn dương", value='#00FF00')
        edge_color_negative = st.color_picker("Màu đường dẫn âm", value='#FF0000')
    
    try:
        # Tạo đồ thị
        G = create_graph(model, standardized=show_standardized)
        
        # Cập nhật màu sắc theo tùy chọn
        for u, v in G.edges():
            if G[u][v]['weight'] > 0:
                G[u][v]['color'] = edge_color_positive
            else:
                G[u][v]['color'] = edge_color_negative
        
        # Vẽ sơ đồ
        title = "Standardized Path Diagram" if show_standardized else "Path Diagram"
        fig = plot_path_diagram(G, title=title)
        
        # Hiển thị chú thích
        st.pyplot(fig)
        
        if show_significance:
            st.write("Chú thích:")
            st.write("* : p < 0.05")
            st.write("** : p < 0.01")
            st.write("*** : p < 0.001")
        
        # Lưu đồ thị vào session state
        st.session_state.path_diagram = G
        
    except Exception as e:
        st.error(f"Lỗi khi vẽ sơ đồ đường dẫn: {str(e)}")

def export_diagram(format='png'):
    """Xuất sơ đồ đường dẫn ra file"""
    if 'path_diagram' not in st.session_state:
        st.error("Chưa có sơ đồ đường dẫn để xuất.")
        return
    
    try:
        G = st.session_state.path_diagram
        fig = plot_path_diagram(G)
        
        # Lưu file
        filename = f"path_diagram.{format}"
        plt.savefig(filename, format=format, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    except Exception as e:
        st.error(f"Lỗi khi xuất sơ đồ: {str(e)}")
        return None 