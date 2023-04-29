import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import griddata
import open3d as o3d
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="Fake Points Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a title for the app
st.title("Fake Points Generator")

# Define function to generate a download link for a CSV file
def generate_csv_download_link(df, file_name, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

# Define function to plot scatter points
def plot_scatter(dataframe, title):
    fig = px.scatter_3d(dataframe, x="X", y="Y", z="Z", title=title)
    st.plotly_chart(fig)
# Create point cloud data
    point_data = dataframe[['X', 'Y', 'Z']].values

    # Create Open3D point cloud geometry
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)

    # Set up a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(geom)

    # Capture the visualization as an image
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()

    # Convert the image to a PIL Image and display it in Streamlit
    pil_image = Image.fromarray((np.asarray(image) * 255).astype(np.uint8))
    st.image(pil_image, caption="Point Cloud Visualization")


# Define function to generate fake points helper
def generate_fake_points_helper_Algo1(df, threshold_density):
    cell_size = 1 
    df['x_cell'] = (df['X'] // cell_size).astype(int)
    df['y_cell'] = (df['Y'] // cell_size).astype(int)

    cell_density_df = df.groupby(['x_cell', 'y_cell']).size().reset_index(name='count')
    cell_density_df['density'] = cell_density_df['count'] / (cell_size * cell_size)
    threshold_density = 500  # pts/m2
    void_regions = cell_density_df[cell_density_df['density'] < threshold_density]
    x = np.linspace(min(df['X']), max(df['X']), num=150)
    y = np.linspace(min(df['Y']), max(df['Y']), num=150)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = griddata((df['X'], df['Y']), df['Z'], (grid_x, grid_y), method='cubic')
    interp_df = pd.DataFrame({'X': grid_x.flatten(), 'Y': grid_y.flatten(), 'Z': grid_z.flatten()})
    filled_df = interp_df[~interp_df.isnull().any(axis=1)]
    merged_df = pd.concat([df, filled_df, void_regions])
    merged_df = merged_df.drop_duplicates(subset=['X', 'Y'], keep='last')
    return merged_df

def generate_fake_points():
    st.header("Generate Fake Points")
    uploaded_file = st.file_uploader("Choose a CSV file to generate fake points:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        threshold_density = st.sidebar.slider("Density Threshold Value", 100, 1000, 500, 100)
        updated_df = generate_fake_points_helper_Algo1(df, threshold_density)
        
        st.subheader("Original Points vs Merged Fake Points")
        col1, col2 = st.columns(2)
        with col1:
            plot_scatter(df, "Original Points")
        with col2:
            plot_scatter(updated_df, "Merged Fake Points")

        download_link = generate_csv_download_link(updated_df, "fake_points.csv", "Download updated point cloud data")
        st.markdown(download_link, unsafe_allow_html=True)

def visualize_points():
    st.header("Visualize Points")
    uploaded_file = st.file_uploader("Choose a CSV file to visualize points:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        plot_scatter(df, "Points Visualization")

# Create sidebar
sidebar = st.sidebar
menu = ["Generate Fake Points", "Visualize Points"]
choice = sidebar.selectbox("Select an option", menu)

# Display content based on user choice
if choice == "Generate Fake Points":
    generate_fake_points()

elif choice == "Visualize Points":
    visualize_points()
