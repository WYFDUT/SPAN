import torch
#from OpenPCDet.pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, paired_boxes_iou3d_gpu

def boxes3d_corner(outputs, targets, idx, indices):
    bs, _, _ = outputs['pred_3d_dim'].shape

    out_3dbboxcenterx, tgt_3dbboxcenterx = [], []
    out_3dbboxcentery, tgt_3dbboxcentery = [], []
    out_3dbboxcenterz, tgt_3dbboxcenterz = [], []
    out_3dbboxry, tgt_3dbboxry = [], []
    out_h, out_w, out_l = [], [], []
    tgt_h, tgt_w, tgt_l = [], [], []
    
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
        tgt_3dtmpcenter = targets[i]["boxes_3d"][:, 0: 2][tgt_idx]

        out_depth = outputs["pred_depth"][i, out_idx1, 0: 1]
        tgt_depth = targets[i]["depth"][tgt_idx]

        out_heading_bin, out_heading_res = outputs["pred_angle"][i, out_idx1, 0:12], outputs["pred_angle"][i, out_idx1, 12:24]
        tgt_heading_bin, tgt_heading_res = targets[i]["heading_bin"][tgt_idx], targets[i]["heading_res"][tgt_idx]

        out_tmpdim = outputs['pred_3d_dim'][i, out_idx1, :]
        out_tmph, out_tmpw, out_tmpl = out_tmpdim[...,0].unsqueeze(-1), out_tmpdim[...,1].unsqueeze(-1), out_tmpdim[...,2].unsqueeze(-1)
        tgt_tmpdim = targets[i]['size_3d'][tgt_idx]
        tgt_tmph, tgt_tmpw, tgt_tmpl = tgt_tmpdim[...,0].unsqueeze(-1), tgt_tmpdim[...,1].unsqueeze(-1), tgt_tmpdim[...,2].unsqueeze(-1)

        out_3dtmpcenterx = out_3dtmpcenter[...,0] * img_size[1]
        out_3dtmpcentery = out_3dtmpcenter[...,1] * img_size[0]
        tgt_3dtmpcenterx = tgt_3dtmpcenter[...,0] * img_size[1]
        tgt_3dtmpcentery = tgt_3dtmpcenter[...,1] * img_size[0]

        out_3dcenterx = ((out_3dtmpcenterx.unsqueeze(-1) - calibs[0, 2]) * out_depth) / calibs[0, 0] + (calibs[0, 3] / (-calibs[0, 0]))
        out_3dcentery = ((out_3dtmpcentery.unsqueeze(-1) - calibs[1, 2]) * out_depth) / calibs[1, 1] + (calibs[1, 3] / (-calibs[1, 1]))
        tgt_3dcenterx = ((tgt_3dtmpcenterx.unsqueeze(-1) - calibs[0, 2]) * tgt_depth) / calibs[0, 0] + (calibs[0, 3] / (-calibs[0, 0]))
        tgt_3dcentery = ((tgt_3dtmpcentery.unsqueeze(-1) - calibs[1, 2]) * tgt_depth) / calibs[1, 1] + (calibs[1, 3] / (-calibs[1, 1]))

        angle_per_class = 2 * torch.pi / 12.0

        out_box2d_coordx_norm = ((outputs["pred_boxes"][i, out_idx1, 0] - outputs["pred_boxes"][i, out_idx1, 2]).relu() + 
                                 (outputs["pred_boxes"][i, out_idx1, 0] + outputs["pred_boxes"][i, out_idx1, 3]).relu()).detach() * 0.5
        tgt_box2d_coordx_norm = ((targets[i]["boxes_3d"][:, 0][tgt_idx] - targets[i]["boxes_3d"][:, 2][tgt_idx]).relu() + 
                                 (targets[i]["boxes_3d"][:, 0][tgt_idx] + targets[i]["boxes_3d"][:, 3][tgt_idx]).relu()).detach() * 0.5
        
        out_box2d_coordx = torch.clamp(out_box2d_coordx_norm * img_size[1], min=1.0)
        tgt_box2d_coordx = torch.clamp(tgt_box2d_coordx_norm * img_size[1], min=1.0)

        out_cls = torch.argmax(out_heading_bin, dim=-1).unsqueeze(-1)
        out_res = torch.gather(out_heading_res, dim=-1, index=out_cls).squeeze(-1)
        out_angle_center = out_cls.squeeze(-1) * angle_per_class
        out_angle = out_angle_center + out_res
        out_angle = torch.where(out_angle > torch.pi, out_angle - 2 * torch.pi, out_angle)
        out_ry = out_angle + torch.arctan2(out_box2d_coordx - calibs[0, 2], calibs[0, 0])
        out_ry = torch.where(out_ry > torch.pi, out_ry - 2 * torch.pi, out_ry)
        out_ry = torch.where(out_ry < -torch.pi, out_ry + 2 * torch.pi, out_ry)

        tgt_angle_center = tgt_heading_bin.squeeze(-1) * angle_per_class
        tgt_angle = tgt_angle_center + tgt_heading_res.squeeze(-1)
        tgt_angle = torch.where(tgt_angle > torch.pi, tgt_angle - 2 * torch.pi, tgt_angle)
        tgt_ry = tgt_angle + torch.arctan2(tgt_box2d_coordx - calibs[0, 2], calibs[0, 0])
        tgt_ry = torch.where(tgt_ry > torch.pi, tgt_ry - 2 * torch.pi, tgt_ry)
        tgt_ry = torch.where(tgt_ry < -torch.pi, tgt_ry + 2 * torch.pi, tgt_ry)

        out_3dbboxcenterx.append(out_3dcenterx), tgt_3dbboxcenterx.append(tgt_3dcenterx)
        out_3dbboxcentery.append(out_3dcentery), tgt_3dbboxcentery.append(tgt_3dcentery)
        out_3dbboxcenterz.append(out_depth), tgt_3dbboxcenterz.append(tgt_depth)
        out_3dbboxry.append(out_ry.unsqueeze(-1)), tgt_3dbboxry.append(tgt_ry.unsqueeze(-1))
        out_h.append(out_tmph), out_w.append(out_tmpw), out_l.append(out_tmpl)
        tgt_h.append(tgt_tmph), tgt_w.append(tgt_tmpw), tgt_l.append(tgt_tmpl)
    
    # Comfirm dot not generate empty tgt list
    if tgt_3dbboxcenterx == []: return None, None
    
    out_3dbboxcenterx, tgt_3dbboxcenterx = torch.cat(out_3dbboxcenterx, dim=0), torch.cat(tgt_3dbboxcenterx, dim=0)
    out_3dbboxcentery, tgt_3dbboxcentery = torch.cat(out_3dbboxcentery, dim=0), torch.cat(tgt_3dbboxcentery, dim=0)
    out_3dbboxcenterz, tgt_3dbboxcenterz = torch.cat(out_3dbboxcenterz, dim=0), torch.cat(tgt_3dbboxcenterz, dim=0)
    out_3dbboxry, tgt_3dbboxry = torch.cat(out_3dbboxry, dim=0), torch.cat(tgt_3dbboxry, dim=0)
    out_h, out_w, out_l = torch.cat(out_h, dim=0), torch.cat(out_w, dim=0), torch.cat(out_l, dim=0)
    tgt_h, tgt_w, tgt_l = torch.cat(tgt_h, dim=0), torch.cat(tgt_w, dim=0), torch.cat(tgt_l, dim=0)

    # Calculate corners for target boxes
    out_corners = compute_box3d_corners(out_3dbboxcenterx, out_3dbboxcentery, out_3dbboxcenterz, out_w, out_h, out_l, out_3dbboxry)
    tgt_corners = compute_box3d_corners(tgt_3dbboxcenterx, tgt_3dbboxcentery, tgt_3dbboxcenterz, tgt_w, tgt_h, tgt_l, tgt_3dbboxry)

    return out_corners, tgt_corners


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
