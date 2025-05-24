import cv2
import numpy as np
import os
import open3d as o3d
import torch
from torch import nn
from torch.utils.data import Dataset
import json
import logging

# Configure logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Normal image generation function
def generate_normal_map(height_map_path, scale=1.0):
    logging.debug("Start generating a normal map")
    # Read the heightmap
    height_map = cv2.imread(height_map_path, cv2.IMREAD_GRAYSCALE)
    if height_map is None:
        raise FileNotFoundError(f"Unable to read file:{height_map_path}")
    logging.debug("The heightmap was read successfully.")

    # Converts the heightmap to floating-point
    height_map = height_map.astype(np.float32) / 255.0

    # Calculate the normals
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Calculate the gradient (x and y directions)
    gradient_x = cv2.filter2D(height_map, -1, kernel_x)
    gradient_y = cv2.filter2D(height_map, -1, kernel_y)

    # Calculate the normals
    normal_x = gradient_x * scale
    normal_y = gradient_y * scale
    normal_z = np.sqrt(1 - np.clip(normal_x ** 2 + normal_y ** 2, 0, 1))

    # Normalize the normals
    normal_x = normal_x / np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    normal_y = normal_y / np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    normal_z = normal_z / np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)

    # Convert normal maps to RGB format
    normal_map = np.stack((normal_x + 1, normal_y + 1, normal_z + 1), axis=2) / 2.0
    normal_map = (normal_map * 255).astype(np.uint8)
    logging.debug("The normal graph is generated successfully.")
    return normal_map


# A point cloud generation function that uses grayscale information as depth
def extract_high_slope_points_with_grayscale_depth(normal_map, grayscale_depth):
    logging.debug("Start extracting high slope points")
    # Converts the normal picture to floating-point
    normal_map = normal_map.astype(np.float32) / 255.0
    grayscale_depth = grayscale_depth.astype(np.float32) / 255.0  # Normalizes to a range of 0-1

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

    selected_points = []
    for y, x in zip(points[0], points[1]):
        # Check if the point is too close to the selected point
        is_close = False
        for py, px in selected_points:
            distance = np.sqrt((y - py) ** 2 + (x - px) ** 2)
            if distance < 3:  # The threshold is set to 3 pixels
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

    # Mark the dot on the image
    marked_image = (normal_map * 255).astype(np.uint8)
    for y, x in selected_points:
        cv2.circle(marked_image, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Extract grayscale information and generate a 3D point cloud
    selected_points = np.array(selected_points, dtype=np.float32)
    depths = grayscale_depth[selected_points[:, 0].astype(np.int32), selected_points[:, 1].astype(np.int32)]


    max_depth = 255.0
    min_depth = 0.0
    depth_range = max_depth - min_depth
    depths = (depths.astype(np.float32) - min_depth) / depth_range  # Normalize to a range of 0-1

    # Convert a 2D point to a 3D point, using the depth value as the z-coordinate
    points_3d = np.hstack((selected_points, depths.reshape(-1, 1)))

    # Adjust the depth range of the point cloud so that it is distributed in 3D space
    scale_factor = 50.0  # Adjusting this factor can change the depth range of the point cloud
    points_3d[:, 2] = points_3d[:, 2] * scale_factor  # Enlarge the depth value

    colors = normal_map[selected_points[:, 0].astype(np.int32), selected_points[:, 1].astype(np.int32)]
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    logging.debug("The high slope point was extracted successfully")
    return pcd

# Define the DGCNN model (consistent with the training module)
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def knn(self, x):
        B, N, _ = x.size()
        valid_k = min(self.k, N - 1)
        dist = torch.cdist(x, x)
        _, idx = torch.topk(dist, k=valid_k + 1, dim=2, largest=False)
        return idx[:, :, 1:]

    def forward(self, x):
        B, C, N = x.size()
        idx = self.knn(x.permute(0, 2, 1))
        x_t = x.permute(0, 2, 1).contiguous()
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, self.k)
        neighbor_features = x_t[batch_indices, idx, :]
        central = x_t.unsqueeze(2)
        edge_feature = torch.cat([central.expand(-1, -1, self.k, -1), neighbor_features - central], dim=-1)
        edge_feature = edge_feature.permute(0, 3, 1, 2)
        out = self.conv(edge_feature)
        return torch.max(out, dim=-1)[0]

class DGCNN(nn.Module):
    def __init__(self, num_classes, k=20):
        super().__init__()
        self.k = k
        self.edge_convs = nn.ModuleList([
            EdgeConv(6, 64, k),
            EdgeConv(64, 64, k),
            EdgeConv(64, 128, k),
            EdgeConv(128, 256, k)
        ])
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for conv in self.edge_convs:
            x = conv(x)
        x = torch.max(x, dim=2)[0]
        return self.fc(x)

# Data preprocessing transformations
class NormalizePointCloud:
    def __call__(self, point_cloud):
        logging.debug("Start normalizing point cloud data")
        # Normalize point cloud data
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:]

        # Normalize point coordinates to [-1, 1]
        centroid = np.mean(points, axis=0)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / furthest_distance

        normalized_point_cloud = np.hstack((points, colors))
        logging.debug("The point cloud data was successfully normalized")
        return normalized_point_cloud


# Dynamically get the number of classes and class mappings in the training set
class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.point_cloud_paths = []
        self.class_to_idx = {}  # Category-to-index mapping

        # Go through all PLY files
        for filename in os.listdir(root_dir):
            if filename.endswith(".ply"):
                self.point_cloud_paths.append(os.path.join(root_dir, filename))
                # Use the file name (excluding extensions) as the category
                class_name = os.path.splitext(filename)[0]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud_path = self.point_cloud_paths[idx]
        # Get the category
        class_name = os.path.splitext(os.path.basename(point_cloud_path))[0]
        label = self.class_to_idx[class_name]

        # Read the PLY file
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        point_cloud = np.hstack((points, colors))

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud, label


# Dynamically get the number of classes and class mappings in the training set
train_dataset = PointCloudDataset(
    root_dir=r'.\datasets\Points\facescape_dataset\facescape_128',
    transform=NormalizePointCloud()
)
num_classes = len(train_dataset.class_to_idx)  # Get the number of categories
class_to_idx = train_dataset.class_to_idx  # Get a category-to-index mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}  # Index-to-category mappings
logging.debug(f"Number of Categories:{num_classes}，Category Mapping:{class_to_idx}")

# Initialize the DGCNN model
model = DGCNN(num_classes=num_classes, k=20)

# Transfer the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
logging.debug(f"The model initialization is successful, and the device:{device}")

# Loads the saved model weights
# Make sure that the weights file you use matches the file saved by the training module
# for example, best_dgcnn_model.pth
try:
    # Use the correct weight file saved with the training module
    model_weight_path = r'.\saved_models\point_cloud_facescape_model.pth'
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Weight file not found:{model_weight_path}")
    model.load_state_dict(torch.load(model_weight_path))
    logging.debug("The model weights were loaded successfully")
except Exception as e:
    logging.error(f"Model weights failed to load:{e}")
model.eval()


# Define a function to predict a single image
def predict_single_image(image_path):
    logging.debug(f"Start Predicting Pictures:{image_path}")
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read file:{image_path}")
    logging.debug("The image loads successfully")

    # Converts the image to a grayscale map for depth information
    grayscale_depth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    logging.debug("The image was converted to grayscale successfully")

    # Generate a normal graph
    normal_map = generate_normal_map(image_path, scale=20.0)
    logging.debug("The normal graph is generated successfully")

    # Generate point cloud data from normal plots and grayscale depth
    pcd = extract_high_slope_points_with_grayscale_depth(normal_map, grayscale_depth)
    logging.debug("The point cloud data is generated successfully")

    # Convert point cloud data to tensors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    point_cloud = np.hstack((points, colors))
    logging.debug("The conversion of point cloud data to tensors is successful")

    # Normalize point cloud data
    normalizer = NormalizePointCloud()
    point_cloud = normalizer(point_cloud)
    logging.debug("The point cloud data was successfully normalized")

    # Convert to tensors and add bulk dimensions
    point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
    logging.debug("The conversion of the point cloud data to tensors and the addition of batch dimensions succeeded")

    with torch.no_grad():
        output = model(point_cloud_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        topk_prob, topk_catid = torch.topk(probabilities, num_classes)  # Use the total number of classifications
        logging.debug("The model inference was successful")

    # Get the category name
    results = {}
    for i in range(num_classes):
        class_name = idx_to_class[topk_catid[0][i].item()].split('.')[0]
        class_name_clean = class_name.replace('_points', '')
        results[class_name_clean] = topk_prob[0][i].item()
    logging.debug("The prediction result is generated successfully")
    return results


def predict_and_save_point_cloud():
    test_image = r".\single_test_image\test.jpg"
    results = predict_single_image(test_image)
    logging.debug(f"Prediction Results:{results}")
    return results