import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import split_feature, merge_splits, split_feature_1d, merge_splits_1d

class single_head_full_attention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [B, L, C]
        assert q.dim() == k.dim() == v.dim() == 3

        scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
        attn = torch.softmax(scores, dim=2)  # [B, L, L]
        out = torch.matmul(attn, v)  # [B, L, C]

        return out

class single_head_full_attention_1d(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                h: Optional[int] = None,
                w: Optional[int] = None
                ) -> torch.Tensor:
        # q, k, v: [B, L, C]
        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c ** 0.5

        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale_factor  # [B, H, W, W]

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).view(b, -1, c)  # [B, H*W, C]

        return out

class single_head_split_window_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_feature = split_feature()
        self.merge_splits = merge_splits()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                num_splits: int = 1,
                with_shift: bool = False,
                h: Optional[int] = None,
                w: Optional[int] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # q, k, v: [B, L, C]
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = int(h // num_splits)
        window_size_w = int(w // num_splits)

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c ** 0.5

        shift_size_w = 0
        shift_size_h = 0

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

        q = self.split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
        k = self.split_feature(k, num_splits=num_splits, channel_last=True)
        v = self.split_feature(v, num_splits=num_splits, channel_last=True)

        scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                            ) / scale_factor  # [B*K*K, H/K*W/K, H/K*W/K]

        if with_shift and attn_mask is not None:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

        out = self.merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                        num_splits=num_splits, channel_last=True)  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)

        return out

class single_head_split_window_attention_1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_feature_1d = split_feature_1d()
        self.merge_splits_1d = merge_splits_1d()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                num_splits: int  = 1,
                with_shift: bool = False,
                h: Optional[int] = None,
                w: Optional[int] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # q, k, v: [B, L, C]

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * h

        window_size_w = w // num_splits

        q = q.view(b * h, w, c)  # [B*H, W, C]
        k = k.view(b * h, w, c)
        v = v.view(b * h, w, c)

        scale_factor = c ** 0.5

        shift_size_w = 0

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=-shift_size_w, dims=1)
            k = torch.roll(k, shifts=-shift_size_w, dims=1)
            v = torch.roll(v, shifts=-shift_size_w, dims=1)

        q = self.split_feature_1d(q, num_splits=num_splits)  # [B*H*K, W/K, C]
        k = self.split_feature_1d(k, num_splits=num_splits)
        v = self.split_feature_1d(v, num_splits=num_splits)

        scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                            ) / scale_factor  # [B*H*K, W/K, W/K]

        if with_shift and attn_mask is not None:
            # attn_mask: [K, W/K, W/K]
            scores += attn_mask.repeat(b * h, 1, 1)  # [B*H*K, W/K, W/K]

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*H*K, W/K, C]

        out = self.merge_splits_1d(out, h, num_splits=num_splits)  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=shift_size_w, dims=2)

        out = out.view(b, -1, c)

        return out


class SelfAttnPropagation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels: int):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0: torch.Tensor, flow: torch.Tensor,
                local_window_attn: bool = False,
                local_window_radius: int = 1
                ) -> torch.Tensor:
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0: torch.Tensor, flow: torch.Tensor,
                                  local_window_radius: int = 1
                                  ) -> torch.Tensor:
        assert flow.size(1) == 2 or flow.size(1) == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = int(2 * local_window_radius + 1)

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=int(kernel_size),
                                   padding=int(local_window_radius))  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, int(kernel_size ** 2), h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, int(kernel_size ** 2))  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=int(kernel_size),
                               padding=int(local_window_radius))  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, value_channel, int(kernel_size ** 2), h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, int(kernel_size ** 2), value_channel)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, value_channel
                                                   ).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out
