# Datasets



## Optical Flow

The datasets used to train and evaluate our GMFlow model are as follows:

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Sintel](http://sintel.is.tue.mpg.de/)
- [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

By default the dataloader [dataloader/flow/datasets.py](dataloader/flow/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/flow/datasets.py](dataloader/flow/datasets.py).



## Stereo Matching

The datasets used to train and evaluate our GMStereo model are as follows:

- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [TartanAir](https://github.com/castacks/tartanair_tools)
- [Falling Things](https://research.nvidia.com/publication/2018-06_Falling-Things)
- [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)
- [CREStereo Dataset](https://github.com/megvii-research/CREStereo/blob/master/dataset_download.sh)
- [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
- [Middlebury](https://vision.middlebury.edu/stereo/data/)
- [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
- [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)

By default the dataloader [dataloader/stereo/datasets.py](dataloader/stereo/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/stereo/datasets.py](dataloader/flow/datasets.py).



## Depth Estimation

The datasets used to train and evaluate our GMDepth model are as follows:

- [DeMoN](https://github.com/lmb-freiburg/demon)
- [ScanNet](http://www.scan-net.org/)

We support downloading and extracting the DeMoN dataset in our code: [dataloader/depth/download_demon_train.sh](dataloader/depth/download_demon_train.sh),  [dataloader/depth/download_demon_test.sh](dataloader/depth/download_demon_test.sh),  [dataloader/depth/prepare_demon_train.sh](dataloader/depth/prepare_demon_train.sh) and  [dataloader/depth/prepare_demon_test.sh](dataloader/depth/prepare_demon_test.sh).

By default the dataloader [dataloader/depth/datasets.py](dataloader/depth/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/depth/datasets.py](dataloader/depth/datasets.py).

