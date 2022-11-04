# Model Zoo

- The models are named as `model-dataset`. 
- Model definition: `scale1` denotes the 1/8 feature resolution model, `scale2` denotes the 1/8 & 1/4 model, `scaleX-regrefineY` denotes the `X`-scale model with additional `Y` local regression refinements.
- The inference time is averaged over 100 runs, measured with batch size 1 on a single NVIDIA A100 GPU.
- All pretrained models can be downloaded together at [pretrained.zip](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained.zip), or they can be downloaded individually as listed below.



## Optical Flow

- The inference time is measured for Sintel resolution: 448x1024

- The `*-mixdata` models are trained on several mixed public datasets, which are recommended for in-the-wild use cases.

  

| Model                             | Params (M) | Time (ms) |                           Download                           |
| --------------------------------- | :--------: | :-------: | :----------------------------------------------------------: |
| GMFlow-scale1-things              |    4.7     |    26     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth) |
| GMFlow-scale1-mixdata             |    4.7     |    26     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth) |
| GMFlow-scale2-things              |    4.7     |    66     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-things-36579974.pth) |
| GMFlow-scale2-sintel              |    4.7     |    66     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-sintel-3ed1cf48.pth) |
| GMFlow-scale2-mixdata             |    4.7     |    66     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth) |
| GMFlow-scale2-regrefine6-things   |    7.4     |    122    | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-things-776ed612.pth) |
| GMFlow-scale2-regrefine6-sintelft |    7.4     |    122    | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth) |
| GMFlow-scale2-regrefine6-kitti    |    7.4     |    122    | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth) |
| GMFlow-scale2-regrefine6-mixdata  |    7.4     |    122    | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth) |



## Stereo Matching

- The inference time is measured for KITTI resolution: 384x1248
- The `*-resumeflowthings-*` denotes that the models are trained with GMFlow model as initialization, where GMFlow is trained on Chairs and Things dataset for optical flow task.
- The `*-mixdata` models are trained on several mixed public datasets, which are recommended for in-the-wild use cases.

| Model                                                  | Params (M) | Time (ms) |  Download  |
| ------------------------------------------------------ | :--------: | :-------: | :--------: |
| GMStereo-scale1-sceneflow                              |    4.7     |    23     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale1-sceneflow-124a438f.pth) |
| GMStereo-scale1-resumeflowthings-sceneflow             |    4.7     |    23     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale1-resumeflowthings-sceneflow-16e38788.pth) |
| GMStereo-scale2-sceneflow                              |    4.7     |    58     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-sceneflow-ab93ba6a.pth) |
| GMStereo-scale2-resumeflowthings-sceneflow             |    4.7     |    58     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth) |
| GMStereo-scale2-regrefine3-sceneflow                   |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-sceneflow-2dd12e97.pth) |
| GMStereo-scale2-regrefine3-resumeflowthings-sceneflow  |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth) |
| GMStereo-scale2-regrefine3-resumeflowthings-kitti      |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth) |
| GMStereo-scale2-regrefine3-resumeflowthings-middlebury |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth) |
| GMStereo-scale2-regrefine3-resumeflowthings-eth3dft    |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-eth3dft-a807cb16.pth) |
| GMStereo-scale2-regrefine3-resumeflowthings-mixdata    |    7.4     |    86     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth) |



## Depth Estimation

- The inference time is measured for ScanNet resolution: 480x640

- The `*-resumeflowthings-*` models are trained with a pretrained GMFlow model as initialization, where GMFlow is trained on Chairs and Things dataset for optical flow task.

  

| Model                                              | Params (M) | Time (ms) |                           Download                           |
| -------------------------------------------------- | :--------: | :-------: | :----------------------------------------------------------: |
| GMDepth-scale1-scannet                             |    4.7     |    17     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-scannet-d3d1efb5.pth) |
| GMDepth-scale1-resumeflowthings-scannet            |    4.7     |    17     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth) |
| GMDepth-scale1-regrefine1-resumeflowthings-scannet |    4.7     |    17     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth) |
| GMDepth-scale1-demon                               |    7.3     |    20     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-demon-bd64786e.pth) |
| GMDepth-scale1-resumeflowthings-demon              |    7.3     |    20     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-demon-a2fe127b.pth) |
| GMDepth-scale1-regrefine1-resumeflowthings-demon   |    7.3     |    20     | [download](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-demon-7c23f230.pth) |



