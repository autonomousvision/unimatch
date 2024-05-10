import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def coords_grid(b: int, h: int, w: int, homogeneous: bool = False, device: Optional[torch.device] = None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min: int, h_max: int, w_min: int, w_max: int, len_h: int, len_w: int, device: Optional[torch.device] = None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          indexing = 'ij')
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords: torch.Tensor, h: int, w: int):
    # coords: [B, H, W, 2]
    c = torch.tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img: torch.Tensor, sample_coords: torch.Tensor,
                    mode: str = 'bilinear',
                    padding_mode: str = 'zeros',
                    return_mask: bool = False
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img, None


def flow_warp(feature:torch.Tensor, flow: torch.Tensor,
              mask: bool = False,
              padding_mode: str = 'zeros'
              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow: torch.Tensor, bwd_flow: torch.Tensor,
                                       alpha: float = 0.01,
                                       beta: float = 0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow, _ = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow, _ = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def back_project(depth: torch.Tensor, intrinsics: torch.Tensor):
    # Back project 2D pixel coords to 3D points
    # depth: [B, H, W]
    # intrinsics: [B, 3, 3]
    b, h, w = depth.shape
    grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)  # [B, 3, H, W]

    intrinsics_inv = torch.inverse(intrinsics)  # [B, 3, 3]

    points = intrinsics_inv.bmm(grid.view(b, 3, -1)).view(b, 3, h, w) * depth.unsqueeze(1)  # [B, 3, H, W]

    return points


def camera_transform(points_ref: torch.Tensor,
                     extrinsics_ref: Optional[torch.Tensor] = None,
                     extrinsics_tgt: Optional[torch.Tensor] = None,
                     extrinsics_rel: Optional[torch.Tensor] = None):
    # Transform 3D points from reference camera to target camera
    # points_ref: [B, 3, H, W]
    # extrinsics_ref: [B, 4, 4]
    # extrinsics_tgt: [B, 4, 4]
    # extrinsics_rel: [B, 4, 4], relative pose transform
    b, _, h, w = points_ref.shape

    if extrinsics_rel is None:
        assert extrinsics_tgt is not None
        assert extrinsics_ref is not None
        extrinsics_rel = torch.bmm(extrinsics_tgt, torch.inverse(extrinsics_ref))  # [B, 4, 4]

    points_tgt = torch.bmm(extrinsics_rel[:, :3, :3],
                           points_ref.view(b, 3, -1)) + extrinsics_rel[:, :3, -1:]  # [B, 3, H*W]

    points_tgt = points_tgt.view(b, 3, h, w)  # [B, 3, H, W]

    return points_tgt


def reproject(points_tgt: torch.Tensor, intrinsics: torch.Tensor,
              return_mask: bool = False
              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # reproject to target view
    # points_tgt: [B, 3, H, W]
    # intrinsics: [B, 3, 3]

    b, _, h, w = points_tgt.shape

    proj_points = torch.bmm(intrinsics, points_tgt.view(b, 3, -1)).view(b, 3, h, w)  # [B, 3, H, W]

    X = proj_points[:, 0]
    Y = proj_points[:, 1]
    Z = proj_points[:, 2].clamp(min=1e-3)

    pixel_coords = torch.stack([X / Z, Y / Z], dim=1).view(b, 2, h, w)  # [B, 2, H, W] in image scale

    if return_mask:
        # valid mask in pixel space
        mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] <= (w - 1)) & (
                pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] <= (h - 1))  # [B, H, W]

        return pixel_coords, mask

    return pixel_coords, None


def reproject_coords(depth_ref: torch.Tensor, intrinsics: torch.Tensor,
                     extrinsics_ref: Optional[torch.Tensor] = None,
                     extrinsics_tgt: Optional[torch.Tensor] = None,
                     extrinsics_rel: Optional[torch.Tensor] = None,
                     return_mask: bool = False
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Compute reprojection sample coords
    points_ref = back_project(depth_ref, intrinsics)  # [B, 3, H, W]
    points_tgt = camera_transform(points_ref, extrinsics_ref, extrinsics_tgt, extrinsics_rel=extrinsics_rel)

    if return_mask:
        reproj_coords, mask = reproject(points_tgt, intrinsics,
                                        return_mask=return_mask)  # [B, 2, H, W] in image scale

        return reproj_coords, mask

    reproj_coords, _ = reproject(points_tgt, intrinsics,
                              return_mask=return_mask)  # [B, 2, H, W] in image scale

    return reproj_coords, None


def compute_flow_with_depth_pose(depth_ref: torch.Tensor, intrinsics: torch.Tensor,
                                 extrinsics_ref: Optional[torch.Tensor] = None,
                                 extrinsics_tgt: Optional[torch.Tensor] = None,
                                 extrinsics_rel: Optional[torch.Tensor] = None,
                                 return_mask: bool = False
                                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, h, w = depth_ref.shape
    coords_init = coords_grid(b, h, w, device=depth_ref.device)  # [B, 2, H, W]

    if return_mask:
        reproj_coords, mask = reproject_coords(depth_ref, intrinsics, extrinsics_ref, extrinsics_tgt,
                                               extrinsics_rel=extrinsics_rel,
                                               return_mask=return_mask)  # [B, 2, H, W]
        rigid_flow = reproj_coords - coords_init

        return rigid_flow, mask

    reproj_coords, _ = reproject_coords(depth_ref, intrinsics, extrinsics_ref, extrinsics_tgt,
                                     extrinsics_rel=extrinsics_rel,
                                     return_mask=return_mask)  # [B, 2, H, W]

    rigid_flow = reproj_coords - coords_init

    return rigid_flow, None
