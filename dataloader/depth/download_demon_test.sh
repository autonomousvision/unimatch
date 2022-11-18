# Source from https://github.com/lmb-freiburg/demon
#!/bin/bash
clear
cat << EOF
================================================================================
The test datasets are provided for research purposes only.
Some of the test datasets build upon other publicly available data.
Make sure to cite the respective original source of the data if you use the 
provided files for your research.
  * sun3d_test.h5 is based on the SUN3D dataset http://sun3d.cs.princeton.edu/
    J. Xiao, A. Owens, and A. Torralba, “SUN3D: A Database of Big Spaces Reconstructed Using SfM and Object Labels,” in 2013 IEEE International Conference on Computer Vision (ICCV), 2013, pp. 1625–1632.
  
  * rgbd_test.h5 is based on the RGBD SLAM benchmark http://vision.in.tum.de/data/datasets/rgbd-dataset (licensed under CC-BY 3.0)
    
    J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 573–580.
  * scenes11_test.h5 uses objects from shapenet https://www.shapenet.org/
    
    A. X. Chang et al., “ShapeNet: An Information-Rich 3D Model Repository,” arXiv:1512.03012 [cs], Dec. 2015.
  * mvs_test.h5 contains scenes from https://colmap.github.io/datasets.html
    
    J. L. Schönberger and J. M. Frahm, “Structure-from-Motion Revisited,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113.
    J. L. Schönberger, E. Zheng, J.-M. Frahm, and M. Pollefeys, “Pixelwise View Selection for Unstructured Multi-View Stereo,” in Computer Vision – ECCV 2016, 2016, pp. 501–518.
  * nyu2_test.h5 is based on http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
  
    N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor Segmentation and Support Inference from RGBD Images,” in Computer Vision – ECCV 2012, 2012, pp. 746–760.
================================================================================
type Y to start the download.
EOF

read -s -n 1 answer
if [ "$answer" != "Y" -a "$answer" != "y" ]; then
	exit 0
fi
echo

datasets=(sun3d rgbd mvs scenes11)

OLD_PWD="$PWD"
DESTINATION=testdata
mkdir $DESTINATION
cd $DESTINATION

for ds in ${datasets[@]}; do
	if [ -e "${ds}_test.h5" ]; then
		echo "${ds}_test.h5 already exists, skipping ${ds}"
	else
		wget "https://lmb.informatik.uni-freiburg.de/data/demon/testdata/${ds}_test.tgz"
		tar -xvf "${ds}_test.tgz"
	fi
done

cd "$OLD_PWD"