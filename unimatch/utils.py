import torch
import torch.nn as nn
import torch.nn.functional as F
from .position import PositionEmbeddingSine
from typing import Tuple

def generate_window_grid(h_min: int, h_max: int, w_min: int, w_max: int, len_h: int, len_w: int, device: torch.device = None) -> torch.Tensor:
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          indexing = 'ij')
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0: torch.Tensor, img1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1

class split_feature(nn.Module):
    def forward(self, feature: torch.Tensor, num_splits: int = 2, channel_last: bool = False) -> torch.Tensor:
        if channel_last:  # [B, H, W, C]
            b, h, w, c = feature.size()
            assert h % num_splits == 0 and w % num_splits == 0

            b_new = b * num_splits * num_splits
            h_new = h // num_splits
            w_new = w // num_splits

            feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                                ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
        else:  # [B, C, H, W]
            b, c, h, w = feature.size()
            assert h % num_splits == 0 and w % num_splits == 0

            b_new = b * num_splits * num_splits
            h_new = h // num_splits
            w_new = w // num_splits

            feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                                ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

        return feature

class merge_splits(nn.Module):
    def forward(self, splits: torch.Tensor, num_splits: int = 2, channel_last: bool = False) -> torch.Tensor:
        if channel_last:  # [B*K*K, H/K, W/K, C]
            b, h, w, c = splits.size()
            new_b = b // num_splits // num_splits

            splits = splits.view(new_b, num_splits, num_splits, h, w, c)
            merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
        else:  # [B*K*K, C, H/K, W/K]
            b, c, h, w = splits.size()
            new_b = b // num_splits // num_splits

            splits = splits.view(new_b, num_splits, num_splits, c, h, w)
            merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
                new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

        return merge

class generate_shift_window_attn_mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_feature = split_feature()

    def forward(self, input_resolution: Tuple[int, int], window_size_h: int, window_size_w: int, shift_size_h: int, shift_size_w: int, device: torch.device = torch.device('cuda')) -> torch.Tensor:
        # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for SW-MSA
        h, w = input_resolution

        mask1 = torch.ones((h - window_size_h,            w - window_size_w           )).to(device) * 0
        mask2 = torch.ones((h - window_size_h,            window_size_w - shift_size_w)).to(device) * 1
        mask3 = torch.ones((h - window_size_h,            shift_size_w                )).to(device) * 2
        mask4 = torch.ones((window_size_h - shift_size_h, w - window_size_w           )).to(device) * 3
        mask5 = torch.ones((window_size_h - shift_size_h, window_size_w - shift_size_w)).to(device) * 4
        mask6 = torch.ones((window_size_h - shift_size_h, shift_size_w                )).to(device) * 5
        mask7 = torch.ones((shift_size_h,                 w - window_size_w           )).to(device) * 6
        mask8 = torch.ones((shift_size_h,                 window_size_w - shift_size_w)).to(device) * 7
        mask9 = torch.ones((shift_size_h,                 shift_size_w                )).to(device) * 8
        # Concatenate the masks to create the full mask
        upper_mask  = torch.cat([mask1, mask2, mask3], dim=1)
        middle_mask = torch.cat([mask4, mask5, mask6], dim=1)
        lower_mask  = torch.cat([mask7, mask8, mask9], dim=1)
        img_mask = torch.cat([upper_mask, middle_mask, lower_mask], dim=0).unsqueeze(0).unsqueeze(-1) # Add extra dimensions for batch size and channels

        mask_windows = self.split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)

        mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

class feature_add_position(nn.Module):
    def __init__(self, feature_channels: int):
        super().__init__()
        self.split_feature = split_feature()
        self.merge_splits = merge_splits()
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    def forward(self, feature0: torch.Tensor, feature1: torch.Tensor, attn_splits: int, feature_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if attn_splits > 1:  # add position in splited window
            feature0_splits = self.split_feature(feature0, num_splits=attn_splits)
            feature1_splits = self.split_feature(feature1, num_splits=attn_splits)

            position = self.pos_enc(feature0_splits)

            feature0_splits = feature0_splits + position
            feature1_splits = feature1_splits + position

            feature0 = self.merge_splits(feature0_splits, num_splits=attn_splits)
            feature1 = self.merge_splits(feature1_splits, num_splits=attn_splits)
        else:
            position = self.pos_enc(feature0)

            feature0 = feature0 + position
            feature1 = feature1 + position

        return feature0, feature1


def upsample_flow_with_mask(flow: torch.Tensor, up_mask: torch.Tensor, upsample_factor: int,
                            is_depth: bool = False) -> torch.Tensor:
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = 1 if is_depth else upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                              upsample_factor * w)  # [B, 2, K*H, K*W]

    return up_flow

class split_feature_1d(nn.Module):
    def forward(self, feature: torch.Tensor, num_splits: int = 2) -> torch.Tensor:
        # feature: [B, W, C]
        b, w, c = feature.size()
        assert w % num_splits == 0

        b_new = b * num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, w // num_splits, c
                            ).view(b_new, w_new, c)  # [B*K, W/K, C]

        return feature

class merge_splits_1d(nn.Module):
    def forward(self, splits: torch.Tensor, h: int, num_splits: int = 2) -> torch.Tensor:
        b, w, c = splits.size()
        new_b = b // num_splits // h

        splits = splits.view(new_b, h, num_splits, w, c)
        merge = splits.view(
            new_b, h, num_splits * w, c)  # [B, H, W, C]

        return merge

def window_partition_1d(x: torch.Tensor, window_size_w: int) -> torch.Tensor:
    """
    Args:
        x: (B, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, W, C = x.shape
    x = x.view(B, W // window_size_w, window_size_w, C).view(-1, window_size_w, C)
    return x

class generate_shift_window_attn_mask_1d(nn.Module):
    def forward(self, input_w: int, window_size_w: int, shift_size_w: int, device: torch.device = torch.device('cuda')) -> torch.Tensor:
        # calculate attention mask for SW-MSA

        mask1 = torch.ones((0, input_w - window_size_w     )).to(device) * 0
        mask2 = torch.ones((0, window_size_w - shift_size_w)).to(device) * 1
        mask3 = torch.ones((0, shift_size_w                )).to(device) * 2
        # Concatenate the masks to create the full mask
        img_mask = torch.cat([mask1, mask2, mask3], dim=1).unsqueeze(0).unsqueeze(-1)

        mask_windows = window_partition_1d(img_mask, window_size_w)  # nW, window_size, 1
        mask_windows = mask_windows.view(-1, window_size_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size, window_size
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
