import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Interactive Data Plotter", layout="wide")

st.title("📊 Interactive Data Plotting")

st.sidebar.header("📁 Upload Data Files")
user_data_files = st.sidebar.file_uploader(
    "Upload your data files",
    type=['txt', 'csv', 'dat', 'xy'],
    accept_multiple_files=True,
    help="Upload two-column (x, y) files for 2D plots or three-column (x1, x2, y) files for 3D plots"
)

plot_type = st.sidebar.radio(
    "Select Plot Type",
    ["2D Plot", "3D Plot"],
    help="Choose between 2D line/scatter plots or 3D surface/scatter plots"
)

if user_data_files:

    if plot_type == "2D Plot":
        st.markdown("### 📈 2D Interactive Plot")

        colss, colzz, colx = st.columns([1, 1, 1])
        has_header = colss.checkbox("Files contain a header row", value=False)
        skip_header = colzz.checkbox("Skip header row", value=True)
        normalized_intensity = colx.checkbox("Normalize to 100%", value=False)

        col_fox, col_xmin, col_xmax = st.columns([1, 1, 1])
        fix_x_axis = col_fox.checkbox("Fix x-axis range?", value=False)
        if fix_x_axis:
            x_axis_min = col_xmin.number_input("X-axis Minimum", value=0.0)
            x_axis_max = col_xmax.number_input("X-axis Maximum", value=10.0)

        x_axis_metric = "X-data"
        y_axis_metric = "Y-data"

        if has_header:
            try:
                sample_file = user_data_files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(
                    sample_file,
                    sep=r'\s+|,|;',
                    engine='python',
                    header=0
                )
                x_axis_metric = df_sample.columns[0]
                y_axis_metric = df_sample.columns[1]
            except Exception as e:
                st.error(f"Error reading header: {e}")
                x_axis_metric = "X-data"
                y_axis_metric = "Y-data"

        st.sidebar.markdown("### 🎨 Plot Layout")
        customize_layout = st.sidebar.checkbox("Modify graph layout", value=False)

        if customize_layout:
            col_line, col_marker = st.sidebar.columns(2)
            show_lines = col_line.checkbox("Show Lines", value=True)
            show_markers = col_marker.checkbox("Show Markers", value=False)

            col_thick, col_size = st.sidebar.columns(2)
            line_thickness = col_thick.number_input("Line Thickness", min_value=0.1, max_value=15.0, value=1.0,
                                                    step=0.3)
            marker_size = col_size.number_input("Marker Size", min_value=0.5, max_value=50.0, value=3.0, step=1.0)

            col_title_font, col_axis_font, col_tick_font = st.sidebar.columns(3)
            title_font_size = col_title_font.number_input("Title Font Size", min_value=10, max_value=50, value=36,
                                                          step=2)
            axis_label_font_size = col_axis_font.number_input("Axis Label Font Size", min_value=10, max_value=50,
                                                              value=36, step=2)
            tick_font_size = col_tick_font.number_input("Tick Label Font Size", min_value=8, max_value=40, value=24,
                                                        step=2)

            col_leg_font, col_leg_pos = st.sidebar.columns(2)
            legend_font_size = col_leg_font.number_input("Legend Font Size", min_value=8, max_value=40, value=28,
                                                         step=2)
            legend_position = col_leg_pos.selectbox("Legend Position", options=["Top", "Bottom", "Left", "Right"],
                                                    index=0)

            col_width, col_height = st.sidebar.columns(2)
            graph_width = col_width.number_input("Graph Width (pixels)", min_value=400, max_value=2000, value=1000,
                                                 step=50)
            graph_height = col_height.number_input("Graph Height (pixels)", min_value=300, max_value=1500, value=900,
                                                   step=50)

            st.sidebar.markdown("#### Custom Axis Labels")
            col_x_label, col_y_label = st.sidebar.columns(2)
            custom_x_label = col_x_label.text_input("X-axis Label", value=x_axis_metric)
            custom_y_label = col_y_label.text_input("Y-axis Label", value=y_axis_metric)

            st.sidebar.markdown("#### Custom Series Names")
            series_names = {}
            for i, file in enumerate(user_data_files):
                series_names[i] = st.sidebar.text_input(f"Label for {file.name}", value=file.name,
                                                        key=f"series_name_{i}")

            st.sidebar.markdown("#### Custom Series Colors")
            series_colors = {}
            colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
            for i, file in enumerate(user_data_files):
                default_color = colors[i % len(colors)]
                series_colors[i] = st.sidebar.color_picker(f"Color for {file.name}", value=default_color,
                                                           key=f"series_color_{i}")
        else:
            series_colors = {}
            show_lines = True
            show_markers = False
            line_thickness = 1.0
            marker_size = 3.0
            title_font_size = 36
            axis_label_font_size = 36
            tick_font_size = 24
            legend_font_size = 28
            legend_position = "Top"
            graph_width = 1000
            graph_height = 900
            custom_x_label = x_axis_metric
            custom_y_label = y_axis_metric
            series_names = {}

        offset_cols = st.columns(len(user_data_files))
        y_offsets = []
        for i, file in enumerate(user_data_files):
            offset_val = offset_cols[i].number_input(f"Y offset for {file.name}", value=0.0, key=f"y_offset_{i}")
            y_offsets.append(offset_val)

        scale_cols = st.columns(len(user_data_files))
        y_scales = []
        for i, file in enumerate(user_data_files):
            scale_val = scale_cols[i].number_input(f"Scale factor for {file.name}", min_value=0.01, max_value=100.0,
                                                   value=1.0, step=0.1, key=f"y_scale_{i}")
            y_scales.append(scale_val)

        fig_interactive = go.Figure()

        for i, file in enumerate(user_data_files):
            try:
                file.seek(0)
                if has_header:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=0)
                else:
                    if skip_header:
                        file.seek(0)
                        try:
                            file_content = file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = file.read().decode('latin-1')

                        lines = file_content.splitlines()
                        comment_line_indices = [i for i, line in enumerate(lines) if line.strip().startswith('#')]
                        lines_to_skip = [0] + comment_line_indices
                        lines_to_skip = sorted(set(lines_to_skip))
                        file.seek(0)
                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=lines_to_skip)
                    else:
                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None)
                    df.columns = [f"Column {j + 1}" for j in range(len(df.columns))]

                x_data = df.iloc[:, 0].values
                y_data = df.iloc[:, 1].values

                if normalized_intensity and np.max(y_data) > 0:
                    y_data = (y_data / np.max(y_data)) * 100

                if i < len(y_scales):
                    y_data = y_data * y_scales[i]

                y_data = y_data + y_offsets[i]

                if customize_layout and i in series_colors:
                    color = series_colors[i]
                else:
                    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#000000', '#7f7f7f']
                    color = colors[i % len(colors)]

                mode_str = ""
                if show_lines:
                    mode_str += "lines"
                if show_markers:
                    if mode_str:
                        mode_str += "+markers"
                    else:
                        mode_str = "markers"
                if not mode_str:
                    mode_str = "markers"

                trace_name = series_names.get(i, file.name) if customize_layout else file.name

                fig_interactive.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode_str,
                    name=trace_name,
                    line=dict(dash='solid', width=line_thickness, color=color),
                    marker=dict(color=color, size=marker_size),
                    hovertemplate=(
                        f"<span style='color:{color};'><b>{trace_name}</b><br>"
                        "x = %{x:.2f}<br>y = %{y:.2f}</span><extra></extra>"
                    )
                ))
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

        legend_config = {"font": dict(size=legend_font_size)}

        if legend_position == "Top":
            legend_config.update({"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5})
        elif legend_position == "Bottom":
            legend_config.update({"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "center", "x": 0.5})
        elif legend_position == "Left":
            legend_config.update({"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "right", "x": -0.1})
        elif legend_position == "Right":
            legend_config.update({"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.05})

        fig_interactive.update_layout(
            height=graph_height,
            width=graph_width,
            margin=dict(t=80, b=80, l=60, r=30),
            hovermode="closest",
            showlegend=True,
            legend=legend_config,
            xaxis=dict(
                title=dict(text=custom_x_label, font=dict(size=axis_label_font_size, color='black'), standoff=20),
                tickfont=dict(size=tick_font_size, color='black'),
                fixedrange=fix_x_axis
            ),
            yaxis=dict(
                title=dict(text=custom_y_label, font=dict(size=axis_label_font_size, color='black')),
                tickfont=dict(size=tick_font_size, color='black')
            ),
            hoverlabel=dict(font=dict(size=tick_font_size)),
            font=dict(size=18),
            autosize=False
        )

        if fix_x_axis:
            fig_interactive.update_xaxes(range=[x_axis_min, x_axis_max])

        st.plotly_chart(fig_interactive)

        st.markdown("### 💾 Download Processed Data")
        delimiter_label_to_value = {
            "Comma (`,`)": ",",
            "Space (` `)": " ",
            "Tab (`\\t`)": "\t",
            "Semicolon (`;`)": ";"
        }

        delimiter_label = st.selectbox("Choose delimiter for download:", list(delimiter_label_to_value.keys()))
        delimiter_option = delimiter_label_to_value[delimiter_label]

        for i, file in enumerate(user_data_files):
            x_data = fig_interactive.data[i].x
            y_data = fig_interactive.data[i].y

            if fix_x_axis:
                mask = (x_data >= x_axis_min) & (x_data <= x_axis_max)
                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
            else:
                filtered_x = x_data
                filtered_y = y_data

            df_out = pd.DataFrame({
                x_axis_metric: filtered_x,
                y_axis_metric: filtered_y
            })

            buffer = io.StringIO()
            df_out.to_csv(buffer, sep=delimiter_option, index=False)

            base_name = file.name.rsplit(".", 1)[0]
            download_name = f"{base_name}_processed.csv"

            download_info = ""
            if fix_x_axis:
                download_info = f" (filtered to x-range: {x_axis_min}-{x_axis_max})"

            st.download_button(
                label=f"⬇️ Download processed data for {file.name}{download_info}",
                data=buffer.getvalue(),
                file_name=download_name,
                mime="text/plain",
                key=f"download_btn_{i}_{base_name}"
            )

    elif plot_type == "3D Plot":
        st.markdown("### 📊 3D Interactive Plot")

        colss, colzz = st.columns([1, 1])
        has_header_3d = colss.checkbox("Files contain a header row", value=False, key="3d_header")
        skip_header_3d = colzz.checkbox("Skip header row", value=True, key="3d_skip")

        plot_type_3d = st.radio("3D Plot Type", ["Surface", "Scatter", "Wireframe"], horizontal=True)

        x1_axis_metric = "X1-data"
        x2_axis_metric = "X2-data"
        y_axis_metric = "Y-data"

        if has_header_3d:
            try:
                sample_file = user_data_files[0]
                sample_file.seek(0)
                df_sample = pd.read_csv(sample_file, sep=r'\s+|,|;', engine='python', header=0)
                if len(df_sample.columns) >= 3:
                    x1_axis_metric = df_sample.columns[0]
                    x2_axis_metric = df_sample.columns[1]
                    y_axis_metric = df_sample.columns[2]
            except Exception as e:
                st.error(f"Error reading header: {e}")

        st.sidebar.markdown("### 🎨 3D Plot Layout")
        customize_3d = st.sidebar.checkbox("Modify 3D plot layout", value=False)

        if customize_3d:
            marker_size_3d = st.sidebar.number_input("Marker Size", min_value=1, max_value=20, value=4, step=1,
                                                     key="3d_marker")
            axis_font_size_3d = st.sidebar.number_input("Axis Label Font Size", min_value=10, max_value=30, value=18,
                                                        step=2, key="3d_axis_font")
            tick_font_size_3d = st.sidebar.number_input("Tick Label Font Size", min_value=8, max_value=20, value=14,
                                                        step=2, key="3d_tick_font")

            col_width_3d, col_height_3d = st.sidebar.columns(2)
            graph_width_3d = col_width_3d.number_input("Graph Width (pixels)", min_value=400, max_value=2000,
                                                       value=1000, step=50, key="3d_width")
            graph_height_3d = col_height_3d.number_input("Graph Height (pixels)", min_value=300, max_value=1500,
                                                         value=900, step=50, key="3d_height")

            st.sidebar.markdown("#### Custom Axis Labels")
            custom_x1_label = st.sidebar.text_input("X1-axis Label", value=x1_axis_metric, key="3d_x1_label")
            custom_x2_label = st.sidebar.text_input("X2-axis Label", value=x2_axis_metric, key="3d_x2_label")
            custom_y_label_3d = st.sidebar.text_input("Y-axis Label", value=y_axis_metric, key="3d_y_label")

            colorscale = st.sidebar.selectbox(
                "Color Scale",
                ["Viridis", "Plasma", "Inferno", "Jet", "Rainbow", "Hot", "Cool", "Blues", "Reds"],
                key="3d_colorscale"
            )
        else:
            marker_size_3d = 4
            axis_font_size_3d = 18
            tick_font_size_3d = 14
            graph_width_3d = 1000
            graph_height_3d = 900
            custom_x1_label = x1_axis_metric
            custom_x2_label = x2_axis_metric
            custom_y_label_3d = y_axis_metric
            colorscale = "Viridis"

        st.markdown("#### 🎯 Visualization Options")
        col_contour1, col_contour2, col_opacity = st.columns(3)

        with col_contour1:
            show_contours = st.checkbox("Show Contour Lines", value=False, help="Display contour lines on the surface")

        with col_contour2:
            contour_projection = st.selectbox(
                "Contour Projection",
                ["None", "X-Y Plane", "X-Z Plane", "Y-Z Plane"],
                help="Project contour lines onto planes"
            )

        with col_opacity:
            surface_opacity = st.slider("Surface Opacity", min_value=0.1, max_value=1.0, value=1.0, step=0.1,
                                        help="Adjust surface transparency")

        if show_contours:
            col_ncontours, col_color = st.columns(2)
            with col_ncontours:
                num_contours = st.number_input("Number of Contour Lines", min_value=5, max_value=50, value=15, step=5)
            with col_color:
                contour_color = st.color_picker("Contour Line Color", value="#000000")

        st.markdown("#### 🔍 Critical Points Detection")
        col_minima, col_maxima, col_filter = st.columns(3)

        with col_minima:
            show_minima = st.checkbox("Show Local Minima", value=False,
                                      help="Automatically identify and display local minima")
            if show_minima:
                minima_size = st.slider("Minima Marker Size", min_value=5, max_value=30, value=15, step=1,
                                        key="minima_size")
                minima_color = st.color_picker("Minima Color", value="#FF0000", key="minima_color")

        with col_maxima:
            show_maxima = st.checkbox("Show Local Maxima", value=False,
                                      help="Automatically identify and display local maxima")
            if show_maxima:
                maxima_size = st.slider("Maxima Marker Size", min_value=5, max_value=30, value=15, step=1,
                                        key="maxima_size")
                maxima_color = st.color_picker("Maxima Color", value="#0000FF", key="maxima_color")

        with col_filter:
            if show_minima or show_maxima:
                filter_extrema = st.checkbox("Filter insignificant points", value=False,
                                             help="Only show the most extreme values")
                if filter_extrema:
                    percentile_threshold = st.slider("Percentile threshold", min_value=1, max_value=50, value=10,
                                                     step=1, help="Show only top/bottom X% of extrema")

        fig_3d = go.Figure()

        minima_data = {}
        maxima_data = {}

        for i, file in enumerate(user_data_files):
            try:
                file.seek(0)
                if has_header_3d:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=0)
                else:
                    if skip_header_3d:
                        file.seek(0)
                        try:
                            file_content = file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            file_content = file.read().decode('latin-1')

                        lines = file_content.splitlines()
                        comment_line_indices = [idx for idx, line in enumerate(lines) if line.strip().startswith('#')]
                        lines_to_skip = [0] + comment_line_indices
                        lines_to_skip = sorted(set(lines_to_skip))
                        file.seek(0)
                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None, skiprows=lines_to_skip)
                    else:
                        df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None)

                if len(df.columns) < 3:
                    st.error(f"File {file.name} must have at least 3 columns for 3D plotting")
                    continue

                x1_data = df.iloc[:, 0].values
                x2_data = df.iloc[:, 1].values
                y_data = df.iloc[:, 2].values

                if plot_type_3d == "Scatter":
                    fig_3d.add_trace(go.Scatter3d(
                        x=x1_data,
                        y=x2_data,
                        z=y_data,
                        mode='markers',
                        name=file.name,
                        marker=dict(
                            size=marker_size_3d,
                            color=y_data,
                            colorscale=colorscale,
                            showscale=True,
                            colorbar=dict(title=custom_y_label_3d)
                        ),
                        hovertemplate=(
                            f"<b>{file.name}</b><br>"
                            "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                        )
                    ))
                else:
                    unique_x1 = np.unique(x1_data)
                    unique_x2 = np.unique(x2_data)

                    if len(unique_x1) * len(unique_x2) == len(x1_data):
                        X1_grid = x1_data.reshape(len(unique_x2), len(unique_x1))
                        X2_grid = x2_data.reshape(len(unique_x2), len(unique_x1))
                        Y_grid = y_data.reshape(len(unique_x2), len(unique_x1))
                    else:
                        from scipy.interpolate import griddata

                        X1_grid, X2_grid = np.meshgrid(unique_x1, unique_x2)
                        Y_grid = griddata((x1_data, x2_data), y_data, (X1_grid, X2_grid), method='linear')

                    contours_config = {}
                    if show_contours:
                        contours_config = {
                            'z': {
                                'show': True,
                                'usecolormap': False,
                                'color': contour_color,
                                'width': 2,
                                'highlightcolor': contour_color,
                                'project': {'z': False}
                            }
                        }

                    if contour_projection == "X-Y Plane":
                        if 'z' not in contours_config:
                            contours_config['z'] = {}
                        contours_config['z']['show'] = True
                        contours_config['z']['project'] = {'z': True}
                        contours_config['z']['usecolormap'] = True
                    elif contour_projection == "X-Z Plane":
                        contours_config['y'] = {
                            'show': True,
                            'project': {'y': True},
                            'usecolormap': True
                        }
                    elif contour_projection == "Y-Z Plane":
                        contours_config['x'] = {
                            'show': True,
                            'project': {'x': True},
                            'usecolormap': True
                        }

                    if plot_type_3d == "Wireframe":
                        fig_3d.add_trace(go.Surface(
                            x=X1_grid,
                            y=X2_grid,
                            z=Y_grid,
                            name=file.name,
                            colorscale=colorscale,
                            showscale=True,
                            opacity=surface_opacity,
                            colorbar=dict(title=custom_y_label_3d),
                            contours=contours_config if contours_config else {},
                            hidesurface=True,
                            hovertemplate=(
                                f"<b>{file.name}</b><br>"
                                "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                            )
                        ))

                        fig_3d.add_trace(go.Scatter3d(
                            x=X1_grid.flatten(),
                            y=X2_grid.flatten(),
                            z=Y_grid.flatten(),
                            mode='markers',
                            marker=dict(size=1, color=Y_grid.flatten(), colorscale=colorscale, showscale=False),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        if show_minima or show_maxima:
                            from scipy.ndimage import minimum_filter, maximum_filter

                            if show_minima:
                                min_filtered = minimum_filter(Y_grid, size=3, mode='constant', cval=np.inf)
                                minima_mask = (Y_grid == min_filtered) & (Y_grid < np.inf)

                                edge_mask = np.ones_like(Y_grid, dtype=bool)
                                edge_mask[0, :] = False
                                edge_mask[-1, :] = False
                                edge_mask[:, 0] = False
                                edge_mask[:, -1] = False
                                minima_mask = minima_mask & edge_mask

                                minima_indices = np.where(minima_mask)
                                x1_minima = X1_grid[minima_indices]
                                x2_minima = X2_grid[minima_indices]
                                y_minima = Y_grid[minima_indices]

                                if len(x1_minima) > 0:
                                    if filter_extrema:
                                        min_threshold = np.percentile(y_minima, percentile_threshold)
                                        significant_mask = y_minima <= min_threshold
                                        x1_minima = x1_minima[significant_mask]
                                        x2_minima = x2_minima[significant_mask]
                                        y_minima = y_minima[significant_mask]

                                    sort_indices = np.argsort(y_minima)
                                    x1_minima = x1_minima[sort_indices]
                                    x2_minima = x2_minima[sort_indices]
                                    y_minima = y_minima[sort_indices]

                                    minima_data[file.name] = {
                                        'x1': x1_minima,
                                        'x2': x2_minima,
                                        'y': y_minima
                                    }

                                    fig_3d.add_trace(go.Scatter3d(
                                        x=x1_minima,
                                        y=x2_minima,
                                        z=y_minima,
                                        mode='markers+text',
                                        name=f'Local Minima ({file.name})',
                                        marker=dict(
                                            size=minima_size,
                                            color=minima_color,
                                            symbol='diamond',
                                            line=dict(color='white', width=2)
                                        ),
                                        text=[f'{val:.2f}' for val in y_minima],
                                        textposition='top center',
                                        textfont=dict(size=10, color=minima_color),
                                        hovertemplate=(
                                            f"<b>Local Minimum</b><br>"
                                            "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                                        )
                                    ))

                            if show_maxima:
                                max_filtered = maximum_filter(Y_grid, size=3, mode='constant', cval=-np.inf)
                                maxima_mask = (Y_grid == max_filtered) & (Y_grid > -np.inf)

                                edge_mask = np.ones_like(Y_grid, dtype=bool)
                                edge_mask[0, :] = False
                                edge_mask[-1, :] = False
                                edge_mask[:, 0] = False
                                edge_mask[:, -1] = False
                                maxima_mask = maxima_mask & edge_mask

                                maxima_indices = np.where(maxima_mask)
                                x1_maxima = X1_grid[maxima_indices]
                                x2_maxima = X2_grid[maxima_indices]
                                y_maxima = Y_grid[maxima_indices]

                                if len(x1_maxima) > 0:
                                    if filter_extrema:
                                        max_threshold = np.percentile(y_maxima, 100 - percentile_threshold)
                                        significant_mask = y_maxima >= max_threshold
                                        x1_maxima = x1_maxima[significant_mask]
                                        x2_maxima = x2_maxima[significant_mask]
                                        y_maxima = y_maxima[significant_mask]

                                    sort_indices = np.argsort(-y_maxima)
                                    x1_maxima = x1_maxima[sort_indices]
                                    x2_maxima = x2_maxima[sort_indices]
                                    y_maxima = y_maxima[sort_indices]

                                    maxima_data[file.name] = {
                                        'x1': x1_maxima,
                                        'x2': x2_maxima,
                                        'y': y_maxima
                                    }

                                    fig_3d.add_trace(go.Scatter3d(
                                        x=x1_maxima,
                                        y=x2_maxima,
                                        z=y_maxima,
                                        mode='markers+text',
                                        name=f'Local Maxima ({file.name})',
                                        marker=dict(
                                            size=maxima_size,
                                            color=maxima_color,
                                            symbol='diamond',
                                            line=dict(color='white', width=2)
                                        ),
                                        text=[f'{val:.2f}' for val in y_maxima],
                                        textposition='top center',
                                        textfont=dict(size=10, color=maxima_color),
                                        hovertemplate=(
                                            f"<b>Local Maximum</b><br>"
                                            "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                                        )
                                    ))
                    else:
                        fig_3d.add_trace(go.Surface(
                            x=X1_grid,
                            y=X2_grid,
                            z=Y_grid,
                            name=file.name,
                            colorscale=colorscale,
                            showscale=True,
                            opacity=surface_opacity,
                            colorbar=dict(title=custom_y_label_3d),
                            contours=contours_config if contours_config else {},
                            hovertemplate=(
                                f"<b>{file.name}</b><br>"
                                "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                            )
                        ))

                    if show_minima or show_maxima:
                        from scipy.ndimage import minimum_filter, maximum_filter

                        if show_minima:
                            min_filtered = minimum_filter(Y_grid, size=3, mode='constant', cval=np.inf)
                            minima_mask = (Y_grid == min_filtered) & (Y_grid < np.inf)

                            edge_mask = np.ones_like(Y_grid, dtype=bool)
                            edge_mask[0, :] = False
                            edge_mask[-1, :] = False
                            edge_mask[:, 0] = False
                            edge_mask[:, -1] = False
                            minima_mask = minima_mask & edge_mask

                            minima_indices = np.where(minima_mask)
                            x1_minima = X1_grid[minima_indices]
                            x2_minima = X2_grid[minima_indices]
                            y_minima = Y_grid[minima_indices]

                            if len(x1_minima) > 0:
                                if filter_extrema:
                                    min_threshold = np.percentile(y_minima, percentile_threshold)
                                    significant_mask = y_minima <= min_threshold
                                    x1_minima = x1_minima[significant_mask]
                                    x2_minima = x2_minima[significant_mask]
                                    y_minima = y_minima[significant_mask]

                                sort_indices = np.argsort(y_minima)
                                x1_minima = x1_minima[sort_indices]
                                x2_minima = x2_minima[sort_indices]
                                y_minima = y_minima[sort_indices]

                                minima_data[file.name] = {
                                    'x1': x1_minima,
                                    'x2': x2_minima,
                                    'y': y_minima
                                }

                                fig_3d.add_trace(go.Scatter3d(
                                    x=x1_minima,
                                    y=x2_minima,
                                    z=y_minima,
                                    mode='markers+text',
                                    name=f'Local Minima ({file.name})',
                                    marker=dict(
                                        size=minima_size,
                                        color=minima_color,
                                        symbol='diamond',
                                        line=dict(color='white', width=2)
                                    ),
                                    text=[f'{val:.2f}' for val in y_minima],
                                    textposition='top center',
                                    textfont=dict(size=10, color=minima_color),
                                    hovertemplate=(
                                        f"<b>Local Minimum</b><br>"
                                        "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                                    )
                                ))

                        if show_maxima:
                            max_filtered = maximum_filter(Y_grid, size=3, mode='constant', cval=-np.inf)
                            maxima_mask = (Y_grid == max_filtered) & (Y_grid > -np.inf)

                            edge_mask = np.ones_like(Y_grid, dtype=bool)
                            edge_mask[0, :] = False
                            edge_mask[-1, :] = False
                            edge_mask[:, 0] = False
                            edge_mask[:, -1] = False
                            maxima_mask = maxima_mask & edge_mask

                            maxima_indices = np.where(maxima_mask)
                            x1_maxima = X1_grid[maxima_indices]
                            x2_maxima = X2_grid[maxima_indices]
                            y_maxima = Y_grid[maxima_indices]

                            if len(x1_maxima) > 0:
                                if filter_extrema:
                                    max_threshold = np.percentile(y_maxima, 100 - percentile_threshold)
                                    significant_mask = y_maxima >= max_threshold
                                    x1_maxima = x1_maxima[significant_mask]
                                    x2_maxima = x2_maxima[significant_mask]
                                    y_maxima = y_maxima[significant_mask]

                                sort_indices = np.argsort(-y_maxima)
                                x1_maxima = x1_maxima[sort_indices]
                                x2_maxima = x2_maxima[sort_indices]
                                y_maxima = y_maxima[sort_indices]

                                maxima_data[file.name] = {
                                    'x1': x1_maxima,
                                    'x2': x2_maxima,
                                    'y': y_maxima
                                }

                                fig_3d.add_trace(go.Scatter3d(
                                    x=x1_maxima,
                                    y=x2_maxima,
                                    z=y_maxima,
                                    mode='markers+text',
                                    name=f'Local Maxima ({file.name})',
                                    marker=dict(
                                        size=maxima_size,
                                        color=maxima_color,
                                        symbol='diamond',
                                        line=dict(color='white', width=2)
                                    ),
                                    text=[f'{val:.2f}' for val in y_maxima],
                                    textposition='top center',
                                    textfont=dict(size=10, color=maxima_color),
                                    hovertemplate=(
                                        f"<b>Local Maximum</b><br>"
                                        "x1 = %{x:.2f}<br>x2 = %{y:.2f}<br>y = %{z:.2f}<extra></extra>"
                                    )
                                ))

            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

        fig_3d.update_layout(
            width=graph_width_3d,
            height=graph_height_3d,
            scene=dict(
                xaxis=dict(
                    title=dict(text=custom_x1_label, font=dict(size=axis_font_size_3d)),
                    tickfont=dict(size=tick_font_size_3d)
                ),
                yaxis=dict(
                    title=dict(text=custom_x2_label, font=dict(size=axis_font_size_3d)),
                    tickfont=dict(size=tick_font_size_3d)
                ),
                zaxis=dict(
                    title=dict(text=custom_y_label_3d, font=dict(size=axis_font_size_3d)),
                    tickfont=dict(size=tick_font_size_3d)
                )
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        st.plotly_chart(fig_3d)

        if minima_data or maxima_data:
            st.markdown("---")
            st.markdown("### 📋 Detected Critical Points")

            for filename in user_data_files:
                filename_str = filename.name

                if filename_str in minima_data or filename_str in maxima_data:
                    st.markdown(f"#### File: {filename_str}")

                    col_left, col_right = st.columns(2)

                    if filename_str in minima_data:
                        with col_left:
                            st.markdown(f"**🔴 Local Minima ({len(minima_data[filename_str]['y'])} found)**")
                            minima_df = pd.DataFrame({
                                'Index': range(1, len(minima_data[filename_str]['y']) + 1),
                                custom_x1_label: minima_data[filename_str]['x1'],
                                custom_x2_label: minima_data[filename_str]['x2'],
                                custom_y_label_3d: minima_data[filename_str]['y']
                            })
                            st.dataframe(minima_df, use_container_width=True, hide_index=True)

                            csv_minima = minima_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Minima Data",
                                data=csv_minima,
                                file_name=f"{filename_str.rsplit('.', 1)[0]}_minima.csv",
                                mime="text/csv",
                                key=f"download_minima_{filename_str}"
                            )

                    if filename_str in maxima_data:
                        with col_right:
                            st.markdown(f"**🔵 Local Maxima ({len(maxima_data[filename_str]['y'])} found)**")
                            maxima_df = pd.DataFrame({
                                'Index': range(1, len(maxima_data[filename_str]['y']) + 1),
                                custom_x1_label: maxima_data[filename_str]['x1'],
                                custom_x2_label: maxima_data[filename_str]['x2'],
                                custom_y_label_3d: maxima_data[filename_str]['y']
                            })
                            st.dataframe(maxima_df, use_container_width=True, hide_index=True)

                            csv_maxima = maxima_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Maxima Data",
                                data=csv_maxima,
                                file_name=f"{filename_str.rsplit('.', 1)[0]}_maxima.csv",
                                mime="text/csv",
                                key=f"download_maxima_{filename_str}"
                            )

        st.markdown("### 💾 Download 3D Data")
        delimiter_label_to_value = {
            "Comma (`,`)": ",",
            "Space (` `)": " ",
            "Tab (`\\t`)": "\t",
            "Semicolon (`;`)": ";"
        }

        delimiter_label_3d = st.selectbox("Choose delimiter for download:", list(delimiter_label_to_value.keys()),
                                          key="3d_delimiter")
        delimiter_option_3d = delimiter_label_to_value[delimiter_label_3d]

        for i, file in enumerate(user_data_files):
            try:
                file.seek(0)
                if has_header_3d:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=0)
                else:
                    df = pd.read_csv(file, sep=r'\s+|,|;', engine='python', header=None,
                                     skiprows=1 if skip_header_3d else 0)
                    df.columns = [x1_axis_metric, x2_axis_metric, y_axis_metric]

                buffer = io.StringIO()
                df.to_csv(buffer, sep=delimiter_option_3d, index=False)

                base_name = file.name.rsplit(".", 1)[0]
                download_name = f"{base_name}_3d_data.csv"

                st.download_button(
                    label=f"⬇️ Download 3D data for {file.name}",
                    data=buffer.getvalue(),
                    file_name=download_name,
                    mime="text/plain",
                    key=f"download_3d_{i}_{base_name}"
                )
            except Exception as e:
                st.error(f"Error preparing download for {file.name}: {e}")

else:
    st.info("📁 Please upload your data files using the sidebar to begin.")
    st.markdown("""
    ### How to use this app:

    **For 2D Plots:**
    - Upload files with two columns (x, y)
    - Customize appearance, add offsets, and scale data
    - Download processed data

    **For 3D Plots:**
    - Upload files with three columns (x1, x2, y)
    - Choose between surface, scatter, or wireframe plots
    - Add contour lines to identify minima/maxima
    - **Automatically detect and display local minima and maxima**
    - Project contours onto planes for better analysis
    - Adjust surface opacity for better visualization
    """)
