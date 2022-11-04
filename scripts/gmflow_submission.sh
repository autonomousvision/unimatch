#!/usr/bin/env bash


# generate prediction results for submission on sintel and kitti online servers


# submission to sintel
CUDA_VISIBLE_DEVICES=0 python main_flow.py \
--submission \
--output_path submission/sintel-gmflow-scale2-regrefine6-sintelft \
--val_dataset sintel \
--resume pretrained/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth \
--inference_size 416 1024 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6


# you can also visualize the predictions before submission
#CUDA_VISIBLE_DEVICES=0 python main_flow.py \
#--submission \
#--output_path submission/sintel-gmflow-scale2-regrefine6-sintelft-vis \
#--val_dataset sintel \
#--resume pretrained/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth \
#--inference_size 416 1024 \
#--save_vis_flow \
#--no_save_flo \
#--padding_factor 32 \
#--upsample_factor 4 \
#--num_scales 2 \
#--attn_splits_list 2 8 \
#--corr_radius_list -1 4 \
#--prop_radius_list -1 1 \
#--reg_refine \
#--num_reg_refine 6


# submission to kitti
CUDA_VISIBLE_DEVICES=0 python main_flow.py \
--submission \
--output_path submission/kitti-gmflow-scale2-regrefine6 \
--val_dataset kitti \
--resume pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth \
--inference_size 352 1216 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6


