#!/bin/bash
clear
cat << EOF

================================================================================


The train datasets are provided for research purposes only.

Some of the test datasets build upon other publicly available data.
Make sure to cite the respective original source of the data if you use the
provided files for your research.

  * sun3d_train.h5 is based on the SUN3D dataset http://sun3d.cs.princeton.edu/

    J. Xiao, A. Owens, and A. Torralba, “SUN3D: A Database of Big Spaces Reconstructed Using SfM and Object Labels,” in 2013 IEEE International Conference on Computer Vision (ICCV), 2013, pp. 1625–1632.




  * rgbd_bugfix_train.h5 is based on the RGBD SLAM benchmark http://vision.in.tum.de/data/datasets/rgbd-dataset (licensed under CC-BY 3.0)

    J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 573–580.



  * scenes11_train.h5 uses objects from shapenet https://www.shapenet.org/

    A. X. Chang et al., “ShapeNet: An Information-Rich 3D Model Repository,” arXiv:1512.03012 [cs], Dec. 2015.



  * mvs_train.h5 contains the Citywall and Achteck-Turm scenes from MVE (Multi-View Environment) http://www.gcc.tu-darmstadt.de/home/proj/mve/

    S. Fuhrmann, F. Langguth, and M. Goesele, “MVE: A Multi-view Reconstruction Environment,” in Proceedings of the Eurographics Workshop on Graphics and Cultural Heritage, Aire-la-Ville, Switzerland, Switzerland, 2014, pp. 11–18.



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
DESTINATION=traindata
mkdir $DESTINATION
cd $DESTINATION

if [ ! -e "README_traindata" ]; then
	wget --no-check-certificate "https://lmb.informatik.uni-freiburg.de/data/demon/traindata/README_traindata"
fi

for ds in ${datasets[@]}; do
	if [ -e "${ds}_train.h5" ]; then
		echo "${ds}_train.h5 already exists, skipping ${ds}"
	else
		wget --no-check-certificate "https://lmb.informatik.uni-freiburg.de/data/demon/traindata/${ds}_train.tgz"
		tar -xvf "${ds}_train.tgz"
	fi
done

cd "$OLD_PWD"