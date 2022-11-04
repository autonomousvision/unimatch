#!/usr/bin/env bash


# gmdepth-scale1
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--eval \
--resume pretrained/gmdepth-scale1-resumeflowthings-demon-a2fe127b.pth \
--val_dataset demon \
--demon_split scenes11


# gmdepth-scale1-regrefine1, this is our final model
CUDA_VISIBLE_DEVICES=0 python main_depth.py \
--eval \
--resume pretrained/gmdepth-scale1-regrefine1-resumeflowthings-demon-7c23f230.pth \
--val_dataset demon \
--demon_split scenes11 \
--reg_refine \
--num_reg_refine 1

