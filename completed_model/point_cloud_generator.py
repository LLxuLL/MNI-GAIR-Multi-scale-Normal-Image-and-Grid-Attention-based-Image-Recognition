import cv2
import numpy as np
import os
import open3d as o3d

# Customize innovative normal image algorithms
def extract_high_slope_points_with_grayscale_depth(normal_map_path, rgb_image_path, output_folder):
    """
    Extract points with larger slopes from the normal image, mark them, and generate a 3D point cloud file based on the
    grayscale information of the corresponding RGB image.

    Parameters:
        normal_map_path (str): The path to the normal image.
        rgb_image_path (str): The path to the corresponding RGB image.
        output_folder (str): The path to the output folder.
    """
    # Retrieve the normal image
    normal_map = cv2.imread(normal_map_path)
    if normal_map is None:
        raise FileNotFoundError(f"Unable to read file:{normal_map_path}")

    # Read the RGB image and convert it to grayscale
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Unable to read file:{rgb_image_path}")

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Convert the normal map to floating point format
    normal_map = normal_map.astype(np.float32) / 255.0

    # Separate channels
    b, g, r = cv2.split(normal_map)

    # Calculate the gradient for each channel
    sobelx_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    sobely_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sobely_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_r = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
    sobely_r = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient amplitude for each channel
    gradient_b = np.hypot(sobelx_b, sobely_b)
    gradient_g = np.hypot(sobelx_g, sobely_g)
    gradient_r = np.hypot(sobelx_r, sobely_r)

    # The total gradient amplitude is calculated
    total_gradient = gradient_b + gradient_g + gradient_r

    # Locate the point with the largest gradient amplitude
    num_points = 1000
    flat_indices = np.argpartition(total_gradient.flatten(), -num_points)[-num_points:]
    points = np.unravel_index(flat_indices, total_gradient.shape)

    # Select the points with the largest gradient until the desired number is reached
    selected_points = []
    for y, x in zip(points[0], points[1]):
        # Check if the point is too close to the selected point
        is_close = False
        for py, px in selected_points:
            distance = np.sqrt((y - py) ** 2 + (x - px) ** 2)
            if distance < 3:
                is_close = True
                break
        if not is_close:
            selected_points.append((y, x))
        if len(selected_points) >= num_points:
            break

    # If there are not enough points selected, continue to look for other points
    if len(selected_points) < num_points:
        remaining_points = []
        for y, x in zip(points[0], points[1]):
            if (y, x) not in selected_points:
                remaining_points.append((y, x))
        # Add the remaining points directly until you reach the desired number
        selected_points.extend(remaining_points[:num_points - len(selected_points)])

    # Extract grayscale information and generate a 3D point cloud
    selected_points = np.array(selected_points, dtype=np.float32)
    depths = gray_image[selected_points[:, 0].astype(np.int32), selected_points[:, 1].astype(np.int32)]

    # Map the grayscale value to the depth value (here the maximum depth is assumed to be 255 and the minimum depth
    # is 0)
    max_depth = 255.0
    min_depth = 0.0
    depth_range = max_depth - min_depth
    depths = (depths.astype(np.float32) - min_depth) / depth_range  # Normalizes to a range of 0-1

    # Convert a 2D point to a 3D point, using the depth value as the z-coordinate
    points_3d = np.hstack((selected_points, depths.reshape(-1, 1)))

    # Adjust the depth range of the point cloud so that it is distributed in 3D space
    scale_factor = 50.0  # Adjusting this factor can change the depth range of the point cloud
    points_3d[:, 2] = points_3d[:, 2] * scale_factor  # Enlarge the depth value

    colors = normal_map[selected_points[:, 0].astype(np.int32), selected_points[:, 1].astype(np.int32)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(normal_map_path))[0] + "_points.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    return output_path


if __name__ == "__main__":
    # Enter the folder path
    normal_input_folder = "./datasets/Normal/facescape_dataset/facescape_128"  # Normal image folder path
    rgb_input_folder = "./datasets/RGB/facescape_dataset/facescape_128"  # RGB folder path
    output_folder = "./datasets/Points/facescape_dataset/facescape_128"  # Replace with your output folder path

    # Make sure the folder exists
    if not os.path.exists(normal_input_folder):
        print(f"The folder does not exist:{normal_input_folder}")
        exit()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all the files in the folder
    for filename in os.listdir(normal_input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            normal_path = os.path.join(normal_input_folder, filename)
            rgb_path = os.path.join(rgb_input_folder, filename)
            print(f"Working with files:{filename}")

            try:
                # Points with large slopes are extracted and labeled to generate a PLY point cloud file
                ply_path = extract_high_slope_points_with_grayscale_depth(normal_path, rgb_path, output_folder)
                print(f"Processing Complete:{filename}")
                print(f"The PLY file has been saved to:{ply_path}")

            except Exception as e:
                print(f"Error processing file:{filename} - {e}")

    print("All image processing is complete!")