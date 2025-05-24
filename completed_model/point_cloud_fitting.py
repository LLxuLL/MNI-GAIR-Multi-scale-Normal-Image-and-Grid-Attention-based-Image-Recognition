import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from pytorch3d.transforms import so3_exp_map

class PointCloudFitter(nn.Module):
    def __init__(self, num_iters=10):
        super().__init__()
        self.num_iters = num_iters
        # Define the learnable parameters
        self.initial_rot = nn.Parameter(torch.zeros(3))  # Initial rotation parameters
        self.initial_trans = nn.Parameter(torch.zeros(3))  # Initial translation parameters

    def forward(self, source_pcd, target_pcd):
        B = source_pcd.shape[0]
        device = source_pcd.device

        # Broadcast parameters to the batch dimension (B, 3)
        rot_params = self.initial_rot.unsqueeze(0).expand(B, 3)  # (B, 3)
        trans_params = self.initial_trans.unsqueeze(0).expand(B, 3)  # (B, 3)

        # Generate the transformation matrix (while maintaining gradients)
        R = so3_exp_map(rot_params)  # (B, 3, 3)
        t = trans_params.unsqueeze(-1)  # (B, 3, 1)
        T = torch.cat([R, t], dim=-1)  # (B, 3, 4)
        ones = torch.zeros(B, 1, 4, device=device)
        ones[..., 3] = 1
        T = torch.cat([T, ones], dim=1)  # (B, 4, 4)

        # Apply the transformation
        transformed_pcd = (
                torch.einsum('bij,bnj->bni', T[:, :3, :3], source_pcd)
                + T[:, :3, 3].unsqueeze(1)
        )
            # Calculate the registration loss (directly backpropagate through the main model to update parameters)
        knn_result = knn_points(transformed_pcd, target_pcd, K=1)
        loss = knn_result.dists.mean()

        # Return the transformed point cloud and loss (the main model optimizes uniformly)
        return transformed_pcd, loss

    def get_final_transform(self, source_pcd):
        """Used to finally obtain the transformation matrix (not involved in training)"""
        B = source_pcd.shape[0]
        rot_params = self.initial_rot.unsqueeze(0).expand(B, 3)
        trans_params = self.initial_trans.unsqueeze(0).expand(B, 3)
        R = so3_exp_map(rot_params)
        t = trans_params.unsqueeze(-1)
        T = torch.cat([R, t], dim=-1)
        ones = torch.zeros(B, 1, 4, device=source_pcd.device)
        ones[..., 3] = 1
        return torch.cat([T, ones], dim=1)
