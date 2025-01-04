import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import warp_with_pose_depth_candidates
from .utils import feature_add_position
from .dpt_head import DPTHead

from .ldm_unet.unet import UNetModel

from .vit_fpn import ViTFeaturePyramid

from einops import rearrange


class UniMatchDepthSplat(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        unet_cross_view_attn=True,
        depth_interval_downsample=2,
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(UniMatchDepthSplat, self).__init__()

        # for large size input to grid_sample
        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # half depth interval at high resolution
        self.depth_interval_downsample = depth_interval_downsample

        # monocular features
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates
        self.unet_cross_view_attn = unet_cross_view_attn
        # upsampler
        vit_feature_channel_dict = {
            "vits": 384,
            "vitb": 768,
            "vitl": 1024,
        }

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # mono vit feature
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,  # self.unet_per_scale_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=unet_cross_view_attn,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            concat_features=True,
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        # list of [2B, C, H, W], resolution from high to low
        features = self.backbone(concat)

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def forward(
        self,
        img0,
        img1,
        attn_type=None,
        attn_splits_list=None,
        intrinsics=None,
        pose=None,  # relative pose transform
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 10,
        num_depth_candidates=128,
        pred_bidir_depth=False,
        **kwargs,
    ):

        pred_bidir_depth = True

        results_dict = {}
        depth_preds = []

        mono_features = None

        # list of cnn features, resolution from low to high
        feature0_list_cnn, feature1_list_cnn = self.extract_feature(img0, img1)

        # used for dpt head
        feature0_list_cnn_all_scales = feature0_list_cnn
        feature1_list_cnn_all_scales = feature1_list_cnn

        feature0_list_cnn = feature0_list_cnn[: self.num_scales]
        feature1_list_cnn = feature1_list_cnn[: self.num_scales]

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]
        feature0_cnn_pos, feature1_cnn_pos = feature_add_position(
            feature0_list_cnn[0],
            feature1_list_cnn[0],
            attn_splits,
            self.feature_channels,
        )

        # mv transformer
        feature0_mv, feature1_mv = self.transformer(
            feature0_cnn_pos,
            feature1_cnn_pos,
            attn_type=attn_type,
            attn_num_splits=attn_splits,
        )

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            feature0_list_mv = self.mv_pyramid(feature0_mv)
            feature1_list_mv = self.mv_pyramid(feature1_mv)
        else:
            feature0_list_mv = [feature0_mv]
            feature1_list_mv = [feature1_mv]

        if pred_bidir_depth:
            # cnn feature
            feature0_list_cnn, feature1_list_cnn = [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(feature0_list_cnn, feature1_list_cnn)
            ], [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(feature1_list_cnn, feature0_list_cnn)
            ]

            # mv feature
            feature0_list_mv, feature1_list_mv = [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(feature0_list_mv, feature1_list_mv)
            ], [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(feature1_list_mv, feature0_list_mv)
            ]

            # cnn features for dpt head
            feature0_list_cnn_all_scales, feature1_list_cnn_all_scales = [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(
                    feature0_list_cnn_all_scales, feature1_list_cnn_all_scales
                )
            ], [
                torch.cat((x, y), dim=0)
                for (x, y) in zip(
                    feature1_list_cnn_all_scales, feature0_list_cnn_all_scales
                )
            ]

        # mono feature
        ori_h, ori_w = img0.shape[2:]

        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14

        if pred_bidir_depth:
            concat = torch.cat((img0, img1), dim=0)
        else:
            concat = img0
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        # last one
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            feature_list_mono = self.mono_pyramid(mono_features)
        else:
            feature_list_mono = [mono_features]

        depth = None

        for scale_idx in range(self.num_scales):
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            # scale intrinsics
            intrinsics_curr = intrinsics.clone()
            intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / downsample_factor

            if scale_idx > 0:
                # 2x upsample depth
                assert depth is not None
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            # generate depth candidates
            b, _, h, w = feature0_list_cnn[scale_idx].size()
            if scale_idx == 0:
                depth_candidates = torch.linspace(
                    min_depth, max_depth, num_depth_candidates
                ).type_as(feature0_list_cnn[0])

                depth_candidates = depth_candidates.view(
                    1, num_depth_candidates, 1, 1
                ).repeat(
                    b, 1, h, w
                )  # [B, D, H, W]
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (self.depth_interval_downsample**scale_idx)
                )

                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth)
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth)

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(feature0_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )

            # build cost volume
            feature0_mv = feature0_list_mv[scale_idx]
            feature1_mv = feature1_list_mv[scale_idx]

            if pred_bidir_depth:
                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                pose_curr = torch.cat((pose, torch.inverse(pose)), dim=0)
            else:
                pose_curr = pose

            warped_feature1_mv = warp_with_pose_depth_candidates(
                feature1_mv,
                intrinsics_curr,
                pose_curr,
                1.0 / depth_candidates,
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [B, C, D, H, W]

            # cost volume
            c = warped_feature1_mv.shape[1]

            # [B, C, 1, H, W] * [B, C, D, H, W]
            cost_volume = (feature0_mv.unsqueeze(2) * warped_feature1_mv).sum(1) / (
                c**0.5
            )  # [B, D, H, W]

            # regressor
            features_cnn = feature0_list_cnn[scale_idx]

            features_mono = feature_list_mono[scale_idx]

            concat = torch.cat(
                (cost_volume, features_cnn, feature0_mv, features_mono), dim=1
            )

            if self.unet_cross_view_attn:
                # unet input shape: (b v) instead of (v b)
                concat = rearrange(concat, "(v b) ... -> (b v) ...", v=2)

            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)

            if self.unet_cross_view_attn:
                # reshape back to (v b)
                out = rearrange(out, "(b v) ... -> (v b) ...", v=2)

            # depth head
            match_prob = F.softmax(
                self.depth_head[scale_idx](out), dim=1
            )  # [B, D, H, W]
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [B, 1, H, W]

            # upsample to the original resolution for supervison at training time only
            if self.training:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final scale, learned upsampler
            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=feature0_list_cnn_all_scales[::-1],
                    mv_features=(
                        feature0_mv if self.num_scales == 1 else feature0_list_mv[::-1]
                    ),
                    depth=depth,
                )
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth = (depth_bilinear + residual_depth).clamp(
                    min=min_depth, max=max_depth
                )

                depth_preds.append(depth)

        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_preds[i] = 1.0 / depth_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({"flow_preds": depth_preds})

        return results_dict
