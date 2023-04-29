import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from io import StringIO
import plotly.figure_factory as ff
import base64
from io import BytesIO
from scipy.spatial import ConvexHull

# Set page title and favicon
st.set_page_config(page_title="Fake Points Generator", page_icon=":chart_with_upwards_trend:")

# Set app title
st.title("Fake Points Generator")

# Create sidebar
st.sidebar.title("Menu")

# Add options to sidebar
menu = ["Generate Fake Points", "Visualize Points"]
choice = st.sidebar.selectbox("Select an option", menu)

# Define function to plot a 3D scatter plot
def plot_3d_scatter(df):
    fig = px.scatter_3d(df, x="X", y="Y", z="Z")
    st.plotly_chart(fig)

# Define function to calculate the volume and surface area of a point cloud
def calculate_volume_and_surface_area(df):
    # Convert point cloud data to numpy array
    points = np.array(df[['X', 'Y', 'Z']])

    # Calculate the convex hull of the point cloud
    hull = ConvexHull(points)

    # Calculate the volume and surface area of the convex hull
    volume = hull.volume
    surface_area = hull.area

    return volume, surface_area


# Define function to generate a download link for a CSV file
def generate_csv_download_link(df, file_name, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

# Define function to plot a heatmap of point cloud density
def plot_heatmap_density(df):
    fig = go.Figure(go.Densitymapbox(
        lat=df['Y'], 
        lon=df['X'], 
        z=df['Z'], 
        radius=10,
        coloraxis='coloraxis'
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=np.mean(df['X']),
        mapbox_center_lat=np.mean(df['Y']),
        mapbox_zoom=11,
        coloraxis=dict(colorscale='Viridis'),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig)

# Define function to generate fake points helper
def generate_fake_points_helper(df, threshold_density):
    # Identify void regions (polygon) where density of point cloud is below threshold_density [variable]
    width = max(df['X']) - min(df['X'])
    height = max(df['Y']) - min(df['Y'])
    area = width * height
    point_density = len(df) / area
    low_density_points = df[df.groupby(['X', 'Y'])['Z'].transform('count') < threshold_density]
    # Fill void regions using some algorithm so that density > threshold_density [variable]
    x = np.linspace(min(df['X']), max(df['X']), num=80)
    y = np.linspace(min(df['Y']), max(df['Y']), num=80)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = griddata((df['X'], df['Y']), df['Z'], (grid_x, grid_y), method='cubic')
    interp_df = pd.DataFrame({'X': grid_x.flatten(), 'Y': grid_y.flatten(), 'Z': grid_z.flatten()})
    filled_df = interp_df[~interp_df.isnull().any(axis=1)]
    updated_df = pd.concat([df, filled_df])
    df1=pd.read_csv("merged_point_cloud.csv")
    # Merge Fake ground points with original ground points to have density greater than threshold_density [variable]
    merged_df = pd.concat([df, filled_df, df1])
    merged_df = merged_df.drop_duplicates(subset=['X', 'Y'], keep='last')
    return merged_df

# Define function to generate fake points
def generate_fake_points():
    # Show file uploader to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file to generate fake points:")
    if uploaded_file is not None:
        # Convert the uploaded file to a pandas dataframe
        df = pd.read_csv(uploaded_file)
        # Show original points
        st.header("Original Points")
        plot_3d_scatter(df)
        
      
        # Add feature 2: Allow user to specify density threshold value
        threshold_density = st.slider("Density Threshold Value", 100, 1000, 500, 50)
        #low_density_points = df[df.groupby(['X', 'Y'])['Z'].transform('count') < threshold_density]
        # Add feature 3: Allow user to download updated point cloud data as CSV file
        updated_df = generate_fake_points_helper(df, threshold_density)
        download_link = generate_csv_download_link(updated_df, "fake_points.csv", "Download updated point cloud data")
        st.markdown(download_link, unsafe_allow_html=True)
        # Add feature 4: Display summary statistics of point cloud data
        st.header("Point Cloud Data Summary Statistics")
        st.write(updated_df.describe())
        # Add feature 7: Visualize density of point cloud data using heatmap
        st.header("Point Cloud Density Visualization")
        plot_heatmap_density(updated_df)
        # Add feature 8: Calculate and display volume and surface area of point cloud data
        volume, surface_area = calculate_volume_and_surface_area(updated_df)
        st.header("Point Cloud Data Volume and Surface Area")
        st.write(f"Volume: {volume:.2f}")
        st.write(f"Surface Area: {surface_area:.2f}")
        # Show updated points
        st.header("Updated Points")
        plot_3d_scatter(updated_df)
        # Show success message
        st.success("Fake points generated successfully!")


def visualize_points():
    # Show file uploader to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file to visualize points:")
    if uploaded_file is not None:
        # Convert the uploaded file to a pandas dataframe
        df = pd.read_csv(uploaded_file)
        # Visualize the points
        st.header("Points Visualization")
        plot_3d_scatter(df)
        # Show success message
        st.success("Points visualized successfully!")
        
if choice == "Generate Fake Points":
    generate_fake_points()
elif choice == "Visualize Points":
    visualize_points()