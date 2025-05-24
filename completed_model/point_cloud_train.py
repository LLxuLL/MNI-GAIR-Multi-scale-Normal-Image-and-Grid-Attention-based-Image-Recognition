import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Enable the CuDNN automatic optimizer
torch.autograd.set_detect_anomaly(False)  # Turn off anomaly detection to speed things up

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.point_cloud_paths = []
        self.class_to_idx = {}

        for filename in os.listdir(root_dir):
            if filename.endswith(".ply"):
                path = os.path.join(root_dir, filename)
                pcd = o3d.io.read_point_cloud(path)
                if len(pcd.points) < 1:
                    continue
                self.point_cloud_paths.append(path)
                class_name = os.path.splitext(filename)[0]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud_path = self.point_cloud_paths[idx]
        class_name = os.path.splitext(os.path.basename(point_cloud_path))[0]
        label = self.class_to_idx[class_name]

        pcd = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        point_cloud = np.hstack((points, colors))

        if self.transform:
            point_cloud = self.transform(point_cloud)
        return point_cloud, label


class NormalizePointCloud:
    def __call__(self, point_cloud):
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:]

        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1)) + 1e-6
        points /= max_dist
        return np.hstack((points, colors))


def collate_point_clouds(batch, k=20):
    min_points = k + 1
    max_points = max(item[0].shape[0] for item in batch)
    max_points = max(max_points, min_points)

    padded_batch = []
    labels = []
    for item, label in batch:
        num_points = item.shape[0]
        if num_points < min_points:
            item = np.tile(item, (min_points // num_points + 1, 1))[:min_points]

        padded = np.zeros((max_points, 6))
        padded[:item.shape[0]] = item
        padded_batch.append(padded)
        labels.append(label)

    return torch.from_numpy(np.stack(padded_batch)).float(), torch.tensor(labels)


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
        return idx[:, :, 1:]  # Excludes self points

    def forward(self, x):
        B, C, N = x.size()
        idx = self.knn(x.permute(0, 2, 1))  # (B, N, k)

        x_t = x.permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, self.k)
        neighbor_features = x_t[batch_indices, idx, :]  # (B, N, k, C)

        central = x_t.unsqueeze(2)  # (B, N, 1, C)
        edge_feature = torch.cat([central.expand(-1, -1, self.k, -1), neighbor_features - central], dim=-1)

        edge_feature = edge_feature.permute(0, 3, 1, 2)  # (B, 2C, N, k)
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
        x = x.permute(0, 2, 1)  # (B, C, N)
        for conv in self.edge_convs:
            x = conv(x)
        x = torch.max(x, dim=2)[0]
        return self.fc(x)


if __name__ == "__main__":
    K = 20
    train_dataset = PointCloudDataset(
        r'.\datasets\Points\facescape_dataset\facescape_128',
        transform=NormalizePointCloud()
    )
    val_dataset = PointCloudDataset(
        r'.\datasets\Points\facescape_dataset\facescape_128_validation',
        transform=NormalizePointCloud()
    )

    collate_fn = partial(collate_point_clouds, k=K)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(num_classes=len(train_dataset.class_to_idx), k=K).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(100):
        model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as pbar:
            for points, labels in pbar:
                points = points.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(points)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        correct = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(points)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './saved_models/point_cloud_facescape_model.pth')
        print(f"Epoch {epoch + 1} | Val Acc: {val_acc:.6f}")

    print(f"Best Accuracy: {best_acc:.6f}")