#!/usr/bin/env bash


# depthsplat-depth-small
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--inference_dir demo/depth-scannet \
--output_path output/depthsplat-depth-small \
--resume pretrained/depthsplat-depth-small-3d79dd5e.pth \
--depthsplat_depth 

# predict depth for both images
# --pred_bidir_depth



# depthsplat-depth-base
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--inference_dir demo/depth-scannet \
--output_path output/depthsplat-depth-base \
--resume pretrained/depthsplat-depth-base-f57113bd.pth \
--depthsplat_depth \
--vit_type vitb \
--num_scales 2 \
--upsample_factor 4



# depthsplat-depth-large
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--inference_dir demo/depth-scannet \
--output_path output/depthsplat-depth-large \
--resume pretrained/depthsplat-depth-large-50d3d7cf.pth \
--depthsplat_depth \
--vit_type vitl \
--num_scales 2 \
--upsample_factor 4

