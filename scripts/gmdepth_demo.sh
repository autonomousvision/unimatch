#!/usr/bin/env bash


# gmdepth-scale1-regrefine1
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--inference_dir demo/depth-scannet \
--output_path output/gmdepth-scale1-regrefine1-scannet \
--resume pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth \
--reg_refine \
--num_reg_refine 1

# --pred_bidir_depth

