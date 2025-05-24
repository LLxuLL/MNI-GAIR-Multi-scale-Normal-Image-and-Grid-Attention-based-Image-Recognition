import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata


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

    return points, colors


def plot_3d_surface(points, colors=None):
    # Create a mesh
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a denser mesh to improve smoothness
    xi = np.linspace(min(x), max(x), 200)
    yi = np.linspace(min(y), max(y), 200)
    XI, YI = np.meshgrid(xi, yi)

    # interpolation
    ZI = griddata((x, y), z, (XI, YI), method='cubic')  # Use the cubic method to interpolate to improve
                                                                  # smoothness

    # Draw 3D surfaces
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use color mapping
    surf = ax.plot_surface(XI, YI, ZI, cmap=cm.viridis, linewidth=0, antialiased=True)

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Cancels the display of coordinate systems
    ax.set_axis_off()

    plt.show()


# 示例用法
file_path = r'datasets/Points/facescape_dataset/facescape_128/000016_points.ply'  # 替换为你的 PLY 文件路径
points, colors = read_ply_file(file_path)
plot_3d_surface(points)