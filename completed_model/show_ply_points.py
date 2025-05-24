import open3d as o3d
import numpy as np


def read_ply_file(file_path):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)

    # Get point coordinates
    points = np.asarray(pcd.points)

    # Get the dot color
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = None

    print(f"Point Cloud Information:")
    print(f"Number of points: {len(points)}")
    print(f"First 5 points: {points[:5]}")

    if colors is not None:
        print(f"First 5 colors: {colors[:5]}")

    o3d.visualization.draw_geometries([pcd])


# Example usage
file_path = r'datasets/Points/facescape_dataset/facescape_128/000016_points.ply'  # Replace with your PLY file path
read_ply_file(file_path)