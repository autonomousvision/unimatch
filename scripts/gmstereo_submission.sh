#!/usr/bin/env bash

# generate prediction results for submission on kitti, middlebury and eth3d online servers


# submission to kitti
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--submission \
--val_dataset kitti15 \
--inference_size 352 1216 \
--output_path submission/kitti-gmstereo-scale2-regrefine3 \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3


# submission to middlebury
# set --eth_submission_mode to train and test to generate results on both train and test sets
# use --save_vis_disp to visualize disparity
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--submission \
--val_dataset middlebury \
--middlebury_resolution F \
--middlebury_submission_mode test \
--inference_size 1024 1536 \
--output_path submission/middlebury-test-gmstereo-scale2-regrefine3 \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3


# submission to eth3d
# set --eth_submission_mode to train and test to generate results on both train and test sets
# use --save_vis_disp to visualize disparity
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--submission \
--eth_submission_mode test \
--val_dataset eth3d \
--inference_size 512 768 \
--output_path submission/eth3d-test-gmstereo-scale2-regrefine3 \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-eth3dft-46effc13.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3














