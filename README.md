<p align="center">
  <h1 align="center">Unifying Flow, Stereo and Depth Estimation</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://scholar.google.com/citations?user=9jH5v74AAAAJ">Jing Zhang</a>
    ·
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a>
    ·
    <a href="https://scholar.google.com/citations?user=VxAuxMwAAAAJ">Hamid Rezatofighi</a>
    ·
    <a href="https://www.yf.io/">Fisher Yu</a>
    ·
    <a href="https://scholar.google.com/citations?user=RwlJNLcAAAAJ">Dacheng Tao</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
  </p>
  <h3 align="center">TPAMI 2023</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2211.05783">Paper</a> | <a href="https://haofeixu.github.io/slides/20221228_synced_unimatch.pdf">Slides</a> | <a href="https://haofeixu.github.io/unimatch/">Project Page</a> | <a href="https://colab.research.google.com/drive/1r5m-xVy3Kw60U-m5VB-aQ98oqqg_6cab?usp=sharing">Colab</a> | <a href="https://huggingface.co/spaces/haofeixu/unimatch">Demo</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/unimatch/resources/teaser.png" alt="Logo" width="70%">
  </a>
</p>


<p align="center">
A unified model for three motion and 3D perception tasks.
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/unimatch/resources/sota_compare.png" alt="Logo" width="100%">
  </a>
</p>

<p align="center">
We achieve the <strong>1st</strong> places on Sintel (clean), Middlebury (rms metric) and Argoverse benchmarks.
</p>

This project is developed based on our previous works: 

- [GMFlow: Learning Optical Flow via Global Matching, CVPR 2022, Oral](https://github.com/haofeixu/gmflow)

- [High-Resolution Optical Flow from 1D Attention and Correlation, ICCV 2021, Oral](https://github.com/haofeixu/flow1d)

- [AANet: Adaptive Aggregation Network for Efficient Stereo Matching, CVPR 2020](https://github.com/haofeixu/aanet)



## Installation

Our code is developed based on pytorch 1.9.0, CUDA 10.2 and python 3.8. Higher version pytorch should also work well.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create -f conda_environment.yml
conda activate unimatch
```

Alternatively, we also support installing with pip:

```
bash pip_install.sh
```



## Model Zoo

A large number of pretrained models with different speed-accuracy trade-offs for flow, stereo and depth are available at [MODEL_ZOO.md](MODEL_ZOO.md).

We assume the downloaded weights are located under the `pretrained` directory.

Otherwise, you may need to change the corresponding paths in the scripts.



## Demo

Given an image pair or a video sequence, our code supports generating prediction results of optical flow, disparity and depth.

Please refer to [scripts/gmflow_demo.sh](scripts/gmflow_demo.sh), [scripts/gmstereo_demo.sh](scripts/gmstereo_demo.sh) and [scripts/gmdepth_demo.sh](scripts/gmdepth_demo.sh) for example usages.




https://user-images.githubusercontent.com/19343475/199893756-998cb67e-37d7-4323-ab6e-82fd3cbcd529.mp4



## Datasets

The datasets used to train and evaluate our models for all three tasks are given in [DATASETS.md](DATASETS.md)



## Evaluation

The evaluation scripts used to reproduce the numbers in our paper are given in [scripts/gmflow_evaluate.sh](scripts/gmflow_evaluate.sh), [scripts/gmstereo_evaluate.sh](scripts/gmstereo_evaluate.sh) and [scripts/gmdepth_evaluate.sh](scripts/gmdepth_evaluate.sh).

For submission to KITTI, Sintel, Middlebury and ETH3D online test sets, you can run [scripts/gmflow_submission.sh](scripts/gmflow_submission.sh) and [scripts/gmstereo_submission.sh](scripts/gmstereo_submission.sh) to generate the prediction results. The results can be submitted directly.



## Training

All training scripts for different model variants on different datasets can be found in [scripts/*_train.sh](scripts).

We support using tensorboard to monitor and visualize the training process. You can first start a tensorboard session with

```
tensorboard --logdir checkpoints
```

and then access [http://localhost:6006](http://localhost:6006/) in your browser.



## Citation

```
@article{xu2023unifying,
  title={Unifying Flow, Stereo and Depth Estimation},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Yu, Fisher and Tao, Dacheng and Geiger, Andreas},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```

This work is a substantial extension of our previous conference paper [GMFlow (CVPR 2022, Oral)](https://arxiv.org/abs/2111.13680), please consider citing GMFlow as well if you found this work useful in your research.

```
@inproceedings{xu2022gmflow,
  title={GMFlow: Learning Optical Flow via Global Matching},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8121-8130},
  year={2022}
}
```



## Acknowledgements

This project would not have been possible without relying on some awesome repos: [RAFT](https://github.com/princeton-vl/RAFT), [LoFTR](https://github.com/zju3dv/LoFTR), [DETR](https://github.com/facebookresearch/detr), [Swin](https://github.com/microsoft/Swin-Transformer), [mmdetection](https://github.com/open-mmlab/mmdetection) and [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/projects/TridentNet/tridentnet/trident_conv.py). We thank the original authors for their excellent work.







