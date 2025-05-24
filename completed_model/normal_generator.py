import cv2
import numpy as np
import os

def generate_normal_map(height_map_path, scale=1.0):
    """
    Convert the input grayscale height map to a normal map and directly replace the original file.

    参数:
        height_map_path (str): The path to the input grayscale height map.
        scale (float): Control the intensity of the normals. The higher the value, the more pronounced the effect in the
                       normal map.
    """

    height_map = cv2.imread(height_map_path, cv2.IMREAD_GRAYSCALE)
    if height_map is None:
        raise FileNotFoundError(f"Unable to read the file：{height_map_path}")

    # Convert the height map to floating point
    height_map = height_map.astype(np.float32) / 255.0

    # Calculate the normals
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Calculate the gradients (in the x and y directions).
    gradient_x = cv2.filter2D(height_map, -1, kernel_x)
    gradient_y = cv2.filter2D(height_map, -1, kernel_y)

    # Compute the normals
    normal_x = gradient_x * scale
    normal_y = gradient_y * scale
    normal_z = np.sqrt(1 - np.clip(normal_x**2 + normal_y**2, 0, 1))

    # Normalize the normals
    normal_x = normal_x / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_y = normal_y / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_z = normal_z / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

    # Convert the normal map to RGB format
    normal_map = np.stack((normal_x + 1, normal_y + 1, normal_z + 1), axis=2) / 2.0
    normal_map = (normal_map * 255).astype(np.uint8)

    # Save and replace the original file
    cv2.imwrite(height_map_path, normal_map)
    print(f"The normal map has been saved to:{height_map_path}")

# 示例用法
if __name__ == "__main__":
    # The path to the input folder
    input_folder = "./datasets/Normal/facescape_dataset/facescape_128"  # 替换为你的输入文件夹路径

    # Ensure the folder exists
    if not os.path.exists(input_folder):
        print(f"The folder does not exist:{input_folder}")
        exit()

    # Iterate through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing file:{input_path}")

            try:
                generate_normal_map(input_path, scale=20.0)
                print(f"The normal map has replaced the original file:{input_path}")

            except Exception as e:
                print(f"Error occurred while processing the file:{input_path} - {e}")

    print("All images have been processed!")