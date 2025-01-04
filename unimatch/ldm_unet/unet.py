from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
from einops import rearrange

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from .util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .attention import SpatialTransformer

from .cross_attention import UNetCrossAttentionBlock


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1,
    downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

        self.downsample_3ddim = downsample_3ddim

    def forward(self, x, y=None):
        assert x.shape[1] == self.channels
        if self.dims == 3 and not self.downsample_3ddim:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1, 
    downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)

        if downsample_3ddim:
            assert dims == 3
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, y=None):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        postnorm=False,
        channels_per_group=None,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if postnorm:
            self.in_layers = nn.Sequential(
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                normalization(self.out_channels, channels_per_group=channels_per_group),
                nn.SiLU(),
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # self.emb_layers = nn.Sequential(
        #     nn.SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )

        if postnorm:
            self.out_layers = nn.Sequential(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                zero_module(
                    normalization(self.out_channels, channels_per_group=channels_per_group),
                ),
                nn.SiLU(),
        )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2)
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        # while len(emb_out.shape) < len(h.shape):
        #     emb_out = emb_out[..., None]
        # if self.use_scale_shift_norm:
        #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        #     scale, shift = th.chunk(emb_out, 2, dim=1)
        #     h = out_norm(h) * (1 + scale) + shift
        #     h = out_rest(h)
        # else:
        #     h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        postnorm=False,
        channels_per_group=None,
        num_frames=2,
        use_cross_view_self_attn=False,
    ):
        super().__init__()

        # NOTE: current attention layer doesn't have positional encoding (TODO)
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(
                self.num_heads,
                n_frames=num_frames,
                use_cross_view_self_attn=use_cross_view_self_attn,
            )

        if postnorm:
            self.proj_out = conv_nd(1, channels, channels, 1)
            self.norm = zero_module(normalization(channels, channels_per_group=channels_per_group))
        else:
            self.norm = normalization(channels, channels_per_group=channels_per_group)
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        self.postnorm = postnorm

    def forward(self, x):
        # return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        if self.postnorm:
            qkv = self.qkv(x)
            h = self.attention(qkv)
            h = self.proj_out(h)
            h = self.norm(h)
        else:
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)


class CrossAttentionBlock(nn.Module):
    """
    Corss attention conditioning
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        condition_channels,
        num_heads=8,
        proj_channels=512,
        num_views=3,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        channels_per_group=None,
        with_norm=False,
        tanh_gating=False,  # following Flamingo
        ffn_after_cross_attn=False,  # following Flamingo
    ):
        super().__init__()

        self.channels = channels
        self.num_head = num_heads
        self.num_views = num_views
        self.proj_channels = proj_channels
        self.with_norm = with_norm
        self.tanh_gating = tanh_gating
        self.ffn_after_cross_attn = ffn_after_cross_attn

        self.q_proj = nn.Linear(channels, proj_channels)
        self.k_proj = nn.Linear(condition_channels, proj_channels)
        self.v_proj = nn.Linear(condition_channels, proj_channels)

        # TODO: whether need norm layer
        # self.norm = normalization(proj_channels, channels_per_group=channels_per_group)

        if self.tanh_gating:
            self.out_proj = conv_nd(3, proj_channels, channels, 1)
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
        else:
            if self.with_norm:
                self.out_proj = conv_nd(3, proj_channels, channels, 1)
                self.norm = zero_module(normalization(channels, channels_per_group=channels_per_group))
            else:
                self.out_proj = zero_module(conv_nd(3, proj_channels, channels, 1))

        if self.ffn_after_cross_attn:
            self.ffn_gate = nn.Parameter(torch.tensor([0.]))
            self.ffn = nn.Sequential(nn.Conv3d(channels, channels * 4, 1, 1, 0),
            normalization(channels * 4, channels_per_group=channels_per_group),
            nn.GELU(),
            nn.Conv3d(channels * 4, channels, 1, 1, 0)
            )

    def forward(self, x, y=None):
        # return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch
        return self._forward(x, y)

    def _forward(self, x, y=None):
        # x: [B, C, D, H, W], feature
        # y: [B, H*W, D, C]
        print(x.shape, y.shape)
        assert x.dim() == 5 and y.dim() == 4
        # NOTE: the resolutions of feature x and color y are different

        b, c1, d, h, w = x.size()
        lx = h * w
        ly = y.size(1)
        c2 = y.size(-1)

        identity = x

        x = x.permute(0, 2, 3, 4, 1).reshape(b * d, h * w, c1)  # [B*D, H*W, C1]
        y = y.permute(0, 2, 1, 3).reshape(b * d, ly, c2)  # [B*D, H*W, C2]
        
        c = self.proj_channels

        q = self.q_proj(x)  # [B*D, H*W, C]
        k = self.k_proj(y)  # [B*D, H*W, C]
        v = self.v_proj(y)

        if self.num_head > 1:
            assert c % self.num_head == 0
            q = q.view(b * d, lx, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]
            k = k.view(b * d, ly, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]
            v = v.view(b * d, ly, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]

            scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / ((c // self.num_head) ** 0.5)  # [B*D, N, H*W, H*W]
            prob = torch.softmax(scores, dim=-1)
            out = torch.matmul(prob, v.permute(0, 2, 1, 3))  # [B*D, H*W, N, C]
            out = out.view(b * d, lx, -1)  # [B*D, H*W, C]

        else:
            scores = torch.matmul(q, k.permute(0, 2, 1)) / (c ** 0.5)  # [B*D, H*W, H*W]
            prob = torch.softmax(scores, dim=-1)

            out = torch.matmul(prob, v)  # [B*D, H*W, C]

        out = out.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        # out = self.norm(out)

        if self.tanh_gating:
            # print('tanh', self.attn_gate.tanh())
            out = self.attn_gate.tanh() * self.out_proj(out)
        else:
            if self.with_norm:
                out = self.out_proj(out)
                out = self.norm(out)
            else:
                out = self.out_proj(out)

        out = identity + out

        if self.ffn_after_cross_attn:
            out = out + self.ffn_gate.tanh() * self.ffn(out)

        return out


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, n_frames=2, use_cross_view_self_attn=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.use_cross_view_self_attn = use_cross_view_self_attn

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """

        # move the view dim into T for cross views attention
        # (b v) ...
        if self.use_cross_view_self_attn:
            qkv = rearrange(qkv, "(b v) n t -> b n (v t)", v=self.n_frames)

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)

        # move view dim back to batch dim in original '(b v)' order
        if self.use_cross_view_self_attn:
            a = rearrange(a, "b n (v t) -> (b v) n t", v=self.n_frames)

        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        middle_block_attn=False,  # use attn in middle block
        middle_block_no_identity=False,  # some previous models are trained without the identity layer
        postnorm=False,  # default prenorm doesn't converge
        attn_prenorm=False,  # try postnorm for resblock and prenorm for attn
        downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
        zero_final_layer=False,  # init zero final output layer
        channels_per_group=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        cross_attn_condition=False,  
        tanh_gating=False,
        ffn_after_cross_attn=False,
        cross_attn_with_norm=False,
        condition_channels=384,
        condition_num_views=3,
        no_self_attn=False,
        conv_kernel_size=3,
        concat_condition=False,
        concat_conv3x3=False,
        num_frames=2,
        use_cross_view_self_attn=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.middle_block_attn = middle_block_attn
        
        self.middle_block_no_identity = middle_block_no_identity

        time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.cross_attn_condition = cross_attn_condition

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not no_self_attn:  # only cross attn, without self attn
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(ch,
                            condition_channels,
                            dim=256,
                            no_cross_attn=concat_condition,
                            with_norm=cross_attn_with_norm,
                            concat_conv3x3=concat_conv3x3,
                            )
                        )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch,
                            downsample_3ddim=downsample_3ddim,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        if self.middle_block_attn:
            self.middle_block = nn.Sequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                # original has attention block in the middle
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                    postnorm=False if attn_prenorm else postnorm,
                    channels_per_group=channels_per_group,
                    num_frames=num_frames,
                    use_cross_view_self_attn=use_cross_view_self_attn,
                ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            ),
                # cross attention condition
                UNetCrossAttentionBlock(ch,
                condition_channels,
                dim=256,
                no_cross_attn=concat_condition,
                with_norm=cross_attn_with_norm,
                concat_conv3x3=concat_conv3x3,
                ) if cross_attn_condition else nn.Identity(),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
            )
        else:
            if self.middle_block_no_identity:
                self.middle_block = nn.Sequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                )
            else:
                self.middle_block = nn.Sequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                    UNetCrossAttentionBlock(ch,
                    condition_channels,
                    dim=256,
                    no_cross_attn=concat_condition,
                    with_norm=cross_attn_with_norm,
                    concat_conv3x3=concat_conv3x3,
                    ) if cross_attn_condition else nn.Identity(),
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not no_self_attn:  # only cross attn, without self attn
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(ch,
                            condition_channels,
                            dim=256,
                            no_cross_attn=concat_condition,
                            with_norm=cross_attn_with_norm,
                            concat_conv3x3=concat_conv3x3,
                            )
                        )
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,
                        downsample_3ddim=downsample_3ddim,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        if postnorm:
            self.out = nn.Sequential(
                conv_nd(dims, model_channels, out_channels, 3, padding=1),
                normalization(out_channels, channels_per_group=channels_per_group) if not zero_final_layer else zero_module(normalization(out_channels, channels_per_group=channels_per_group)),
                nn.SiLU(),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch, channels_per_group=channels_per_group),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch, channels_per_group=channels_per_group),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # emb = self.time_embed(t_emb)
        emb = None

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # h = module(h, emb, context)
            # if i == 0:  # conv layer

            if self.cross_attn_condition:
                for submodule in module:
                    if 'UNetCrossAttentionBlock' == submodule.__class__.__name__:
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            # else:
            #     print(module)
            #     h = module(h, context)
            hs.append(h)
        # h = self.middle_block(h, emb, context)
        # h = self.middle_block(h)

        for module in self.middle_block:
            if 'UNetCrossAttentionBlock' == module.__class__.__name__:
                h = module(h, context)
            else:
                h = module(h)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            # h = module(h, emb, context)
            if self.cross_attn_condition:
                for submodule in module:
                    if 'UNetCrossAttentionBlock' == submodule.__class__.__name__:
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            # h = module(h, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class StackUNet(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks=1,
                 attention_resolutions=[],
                 channel_mult=[1, 1, 1, 1],
                 num_head_channels=32,
                 dims=3,
                 postnorm=True,
                 attn_prenorm=False,
                 middle_block_attn=False,
                 num_stacks=1,
                 zero_final_layer=False,
                 resblock_updown=False,
                 channels_per_group=None,
                 cross_attn_condition=False,  
                 cross_attn_with_norm=False,
                condition_channels=128,
                tanh_gating=False,
                ffn_after_cross_attn=False,
                condition_num_views=3,
                no_self_attn=False,
                middle_block_no_identity=False,
                conv_kernel_size=3,
                 ):

        super().__init__()

        self.num_stacks = num_stacks

        self.stacks = nn.ModuleList()

        in_channels = in_channels

        for i in range(num_stacks):
            self.stacks.append(UNetModel(image_size=None,
                            in_channels=in_channels,
                            model_channels=model_channels,
                            out_channels=out_channels,
                            num_res_blocks=num_res_blocks,
                            attention_resolutions=attention_resolutions,
                            channel_mult=channel_mult,
                            num_head_channels=num_head_channels,
                            dims=dims,
                            middle_block_attn=middle_block_attn,
                            middle_block_no_identity=middle_block_no_identity,
                            postnorm=postnorm,
                            attn_prenorm=attn_prenorm,
                            zero_final_layer=zero_final_layer and i == 0,
                            resblock_updown=resblock_updown,
                            channels_per_group=channels_per_group,
                            cross_attn_condition=cross_attn_condition,  
                            tanh_gating=tanh_gating,
                            ffn_after_cross_attn=ffn_after_cross_attn,
                            cross_attn_with_norm=cross_attn_with_norm,
                            condition_channels=condition_channels,
                            condition_num_views=condition_num_views,
                            no_self_attn=no_self_attn,
                            conv_kernel_size=conv_kernel_size,
                            )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList()

        for i in range(num_stacks - 1):
            self.convs.append(zero_module(conv_nd(
                dims, out_channels, in_channels, 3, padding=1
            )))


    def forward(self, x, context=None):
        x = self.stacks[0](x, context=context)
        for i in range(self.num_stacks - 1):
            residual = self.convs[i](self.stacks[i + 1](x, context=context))
            x = x + residual

        return x



