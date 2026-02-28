import torch
from utils import box_ops
import torch.nn.functional as F

# TODO: Compute 3d to 2d geometry constrain loss
def compute_box3d_corners(center_x, center_y, center_z, width, height, length, ry):
    """
    Compute 8 corner points of a 3D bounding box
    
    Args:
        center_x (Tensor): (N, 1) X coordinate of center
        center_y (Tensor): (N, 1) Y coordinate of center
        center_z (Tensor): (N, 1) Z coordinate of center
        width (Tensor): (N, 1) Width (x-direction)
        height (Tensor): (N, 1) Height (y-direction)
        length (Tensor): (N, 1) Length (z-direction)
        ry (Tensor): (N, 1) Rotation angle around Y-axis
    
    Returns:
        corners (Tensor): (N, 8, 3) Tensor of corner coordinates
    """
    device = center_x.device
    
    # 1. Create template for unrotated corners (relative to center)
    corner_tmpl = torch.tensor([[-0.5, -0.5, -0.5], # rear-top-left
                                [ 0.5, -0.5, -0.5], # rear-top-right
                                [ 0.5,  0.5, -0.5], # rear-bottom-right
                                [-0.5,  0.5, -0.5],  # rear-bottom-left
                                [-0.5, -0.5,  0.5], # front-top-left
                                [ 0.5, -0.5,  0.5], # front-top-right
                                [ 0.5,  0.5,  0.5], # front-bottom-right
                                [-0.5,  0.5,  0.5], # front-bottom-left
                               ], dtype=torch.float32, device=device)  # Shape: (8, 3)
    
    # 2. Scale template by actual dimensions
    dims = torch.cat([length, height, width], dim=1)  # (N, 3)
    corner_tmpl = corner_tmpl[None] * dims[:, None]    # (N, 8, 3)
    
    # 3. Rotation matrix around Y-axis
    cos_ry = torch.cos(ry).squeeze(-1)  # (N,)
    sin_ry = torch.sin(ry).squeeze(-1)  # (N,)
    
    rotation_matrix = torch.zeros((len(cos_ry), 3, 3), device=device)
    rotation_matrix[:, 0, 0] = cos_ry
    rotation_matrix[:, 0, 2] = sin_ry
    rotation_matrix[:, 1, 1] = 1.0
    rotation_matrix[:, 2, 0] = -sin_ry
    rotation_matrix[:, 2, 2] = cos_ry  # Shape: (N, 3, 3)
    
    # 4. Rotate corners
    # Transpose for matrix multiplication: (N, 3, 3) x (N, 3, 8) -> (N, 3, 8)
    corners_rot = torch.matmul(rotation_matrix, corner_tmpl.transpose(1, 2))
    corners_rot = corners_rot.transpose(1, 2)  # Transpose to (N, 8, 3)
    
    # 5. Translate to absolute position
    centers = torch.cat([center_x, center_y, center_z], dim=1)  # (N, 3)
    corners = corners_rot + centers[:, None, :]  # (N, 8, 3)
    
    return corners

def project_3d_to_2d(corners_3d, calib_matrix):
    """
    Args:
        corners_3d (Tensor): 3D corner [N, 8, 3]
        calib_matrix (Tensor): calib [3, 4]
    
    Returns:
        corners_2d (Tensor): 2D proj coord [N, 8, 2]
    """
    N = corners_3d.shape[0]
    ones = torch.ones((N, 8, 1), device=corners_3d.device)
    corners_3d_homo = torch.cat([corners_3d, ones], dim=-1)  # [N, 8, 4]
    
    calib_matrix = calib_matrix.unsqueeze(0)  # [1, 3, 4]
    points_t = corners_3d_homo.transpose(1, 2)  # [N, 4, 8]
    
    # [1,3,4] @ [N,4,8] -> [N,3,8]
    image_points = torch.matmul(calib_matrix, points_t)  # [N, 3, 8]
    image_points = image_points.transpose(1, 2)  # [N, 8, 3]
    z = image_points[..., 2].clone()
    z[z == 0] = 1e-6
    u = image_points[..., 0] / z
    v = image_points[..., 1] / z
    corners_2d = torch.stack([u, v], dim=-1)  # [N, 8, 2]
    in_front = (corners_3d[..., 2] > 0).float()
    
    # if visible
    #valid_u = (u >= 0) & (u <= 1280)  # assume w as 1280
    #valid_v = (v >= 0) & (v <= 384)   # assume h as 384
    #mask = in_front & valid_u & valid_v

    return corners_2d #mask.float()

def boxes3d_proj2d_loss(outputs, targets, idx, indices):
    bs, _, _ = outputs['pred_3d_dim'].shape
    total_loss = torch.tensor(0.0).to(outputs['pred_3d_dim'].device)
    
    # Data conversion
    for i in range(bs):
        out_mask = (idx[0] == i)
        out_idx1 = idx[1][out_mask]
        if out_idx1.numel() == 0: continue
        tgt_idx = indices[i][1]

        img_size = targets[i]['obj_region'].shape
        img_size = tuple(float(dim) for dim in img_size)
        calibs = targets[i]["calibs_perimg"][0]

        out_3dtmpcenter = outputs["pred_boxes"][i, out_idx1, 0: 2].detach()

        out_depth = outputs["pred_depth"][i, out_idx1, 0: 1]

        out_tmpdim = outputs['pred_3d_dim'][i, out_idx1, :]
        out_tmph, out_tmpw, out_tmpl =  out_tmpdim[...,0].unsqueeze(-1), out_tmpdim[...,1].unsqueeze(-1), out_tmpdim[...,2].unsqueeze(-1)

        out_3dtmpcenterx = out_3dtmpcenter[...,0] * img_size[1]
        out_3dtmpcentery = out_3dtmpcenter[...,1] * img_size[0]

        out_3dcenterx = ((out_3dtmpcenterx.unsqueeze(-1) - calibs[0, 2]) * out_depth) / calibs[0, 0] + (calibs[0, 3] / (-calibs[0, 0]))
        out_3dcentery = ((out_3dtmpcentery.unsqueeze(-1) - calibs[1, 2]) * out_depth) / calibs[1, 1] + (calibs[1, 3] / (-calibs[1, 1]))

        ry = targets[i]["ry"][:, :][tgt_idx]

        # Comfirm current batch contains target objects
        if ry.numel() == 0:
            loss_inter = torch.tensor(0.0).to(out_3dcenterx.device)
            loss_boundary = torch.tensor(0.0).to(out_3dcenterx.device)

        else:
            out_corners = compute_box3d_corners(out_3dcenterx, out_3dcentery, out_depth, out_tmpw, out_tmph, out_tmpl, ry)
            out_proj_2d = project_3d_to_2d(out_corners, calibs)
            out_proj_2d = torch.nan_to_num(out_proj_2d, nan=0.0, posinf=0.0, neginf=0.0)
            out_proj_2d = torch.clamp(out_proj_2d, min=0.0)

            u_values = torch.clamp(out_proj_2d[:, :, 0], max=img_size[1])
            v_values = torch.clamp(out_proj_2d[:, :, 1], max=img_size[0])

            u_values_norm, v_values_norm = u_values / img_size[1], v_values / img_size[0]

            min_u_norm, _ = torch.min(u_values_norm, dim=1)
            max_u_norm, _ = torch.max(u_values_norm, dim=1)
            min_v_norm, _ = torch.min(v_values_norm, dim=1)
            max_v_norm, _ = torch.max(v_values_norm, dim=1)

            out_boxes = torch.stack([min_u_norm, min_v_norm, max_u_norm, max_v_norm], dim=1)
            target_boxes = box_ops.box_cxcylrtb_to_xyxy(targets[i]["boxes_3d"][:, :][tgt_idx])
            target_boxes = torch.clamp(target_boxes, min=0).detach()

            u_min_diff = torch.clamp(target_boxes[:, 0].unsqueeze(-1) - u_values_norm, min=0.0, max=1.0)
            u_max_diff = torch.clamp(u_values_norm - target_boxes[:, 2].unsqueeze(-1), min=0.0, max=1.0)
            v_min_diff = torch.clamp(target_boxes[:, 1].unsqueeze(-1) - v_values_norm, min=0.0, max=1.0)
            v_max_diff = torch.clamp(v_values_norm - target_boxes[:, 3].unsqueeze(-1), min=0.0, max=1.0)

            loss_inter = ((u_min_diff + u_max_diff + v_min_diff + v_max_diff).mean(dim=-1).sum())
            loss_boundary = ((1 - torch.diag(box_ops.generalized_box_iou(out_boxes, target_boxes))).sum())
            #loss_boundary = F.l1_loss(out_boxes, target_boxes, reduction='none').sum()

        total_loss += loss_inter
        total_loss += loss_boundary
    return total_loss