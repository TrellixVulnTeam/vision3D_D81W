Installation:
=============

Nvidia: jetson nano
-------------------

  ~> uname -a
     Linux nano 4.9.140-tegra #1 SMP PREEMPT Fri Jul 12 17:32:47 PDT 2020 aarch64 aarch64 aarch64 GNU/Linux

  ~> sudo apt-get update
  ~> sudo apt-get -y upgrade

  ~/Programs> git clone https://github.com/opencv/opencv_contrib
  ~/Programs/opencv_contrib> git checkout 4.5.4
  ~/Programs> git clone https://github.com/opencv/opencv
  ~/Programs/opencv> git checkout 4.5.4

  ~> sudo apt-get install build-essential cmake git unzip pkg-config zlib1g-dev \
                          libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev \
                          libpng-dev libtiff-dev libglew-dev \
                          libavcodec-dev libavformat-dev libswscale-dev \
                          libgtk2.0-dev libgtk-3-dev libcanberra-gtk* \
                          python-dev python-numpy python-pip \
                          python3-dev python3-numpy python3-pip \
                          libxvidcore-dev libx264-dev libgtk-3-dev \
                          libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev \
                          libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                          gstreamer1.0-tools libgstreamer-plugins-base1.0-dev \
                          libgstreamer-plugins-good1.0-dev \
                          libv4l-dev v4l-utils v4l2ucp qv4l2 \
                          libtesseract-dev libxine2-dev libpostproc-dev \
                          libavresample-dev libvorbis-dev \
                          libfaac-dev libmp3lame-dev libtheora-dev \
                          libopencore-amrnb-dev libopencore-amrwb-dev \
                          libopenblas-dev libatlas-base-dev libblas-dev \
                          liblapack-dev liblapacke-dev libeigen3-dev gfortran \
                          libhdf5-dev libprotobuf-dev protobuf-compiler \
                          libgoogle-glog-dev libgflags-dev \
                          qt5-default
  ~/Programs/opencv/build> cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=~/Programs/opencv_contrib/modules \
                                 -DBUILD_opencv_python3=ON -DCMAKE_INSTALL_PREFIX:PATH=~/Programs/opencv/local \
                                 -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_GSTREAMER=ON \
                                 -DWITH_CUDA=ON -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
                                 -DBUILD_LIST=core,calib3d,viz,videoio,highgui,python3 ..
  ~/Programs/opencv/build> make -j 2
  ~/Programs/opencv/build> make install
  ~/Programs/opencv/build> sudo ldconfig

Friendlyarm: rockchip
---------------------

  ~> uname -a
     Linux NanoPC-T4 4.4.167 #1 SMP Fri Jul 12 17:32:47 CST 2019 aarch64 aarch64 aarch64 GNU/Linux

  ~> sudo apt-get update
  ~> sudo apt-get -y upgrade

  ~/Programs> git clone https://github.com/opencv/opencv_contrib
  ~/Programs/opencv_contrib> git checkout 4.5.5
  ~/Programs> git clone https://github.com/opencv/opencv
  ~/Programs/opencv> git checkout 4.5.5

  ~> sudo apt-get install build-essential cmake git unzip pkg-config zlib1g-dev \
                          libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev \
                          libpng-dev libtiff-dev libglew-dev \
                          libavcodec-dev libavformat-dev libswscale-dev \
                          libgtk2.0-dev libgtk-3-dev libcanberra-gtk* \
                          python-dev python-numpy python-pip \
                          python3-dev python3-numpy python3-pip \
                          libxvidcore-dev libx264-dev libgtk-3-dev \
                          libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev \
                          libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                          gstreamer1.0-tools libgstreamer-plugins-base1.0-dev \
                          libgstreamer-plugins-good1.0-dev \
                          libv4l-dev v4l-utils v4l2ucp qv4l2 \
                          libtesseract-dev libxine2-dev libpostproc-dev \
                          libavresample-dev libvorbis-dev \
                          libfaac-dev libmp3lame-dev libtheora-dev \
                          libopencore-amrnb-dev libopencore-amrwb-dev \
                          libopenblas-dev libatlas-base-dev libblas-dev \
                          liblapack-dev liblapacke-dev libeigen3-dev gfortran \
                          libhdf5-dev libprotobuf-dev protobuf-compiler \
                          libgoogle-glog-dev libgflags-dev \
                          qt5-default
  ~/Programs/opencv/build> cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=~/Programs/opencv_contrib/modules \
                                 -DBUILD_opencv_python3=ON -DCMAKE_INSTALL_PREFIX:PATH=~/Programs/opencv/local \
                                 -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_GSTREAMER=ON \
                                 -DOPENCV_DNN_OPENCL=ON \
                                 -DBUILD_LIST=core,calib3d,viz,videoio,highgui,python3,dnn ..
  ~/Programs/opencv/build> make -j 2
  ~/Programs/opencv/build> make install
  ~/Programs/opencv/build> sudo ldconfig

Environment modules
-------------------

  ~> sudo apt-get install -y environment-modules tcl-dev
  ~> tail ~/.bashrc
     # Modules environment
     export MODULEPATH="~/Modules"
     module() { eval `/usr/lib/modulecmd.tcl bash $*`; }
  ~> more ~/Modules/opencv
     #%Module1.0
     set rootdir ~/Programs/opencv/local
     prepend-path PATH              $rootdir/bin
     prepend-path LD_LIBRARY_PATH   $rootdir/lib
     prepend-path CMAKE_MODULE_PATH $rootdir/lib/cmake/opencv4
     prepend-path PYTHONPATH        $rootdir/lib/python3.6/dist-packages

vision3D dependencies
---------------------

  ~> tail ~/.bashrc
     # Big/Little endian crash (x86 vs ARM)
     export OPENBLAS_CORETYPE=ARMV8

  ~> sudo apt-get install libblas-dev liblapack-dev libhdf5-dev python3-pip python3-tk python3-pil.imagetk
  ~> pip3 install --upgrade pip
  ~> pip3 install numpy Pillow Cython pkgconfig
  ~> H5PY_SETUP_REQUIRES=0 pip3 install -U --no-build-isolation h5py
  ~> sudo apt-get install hdf5-tools
  ~> sudo apt-get install python3-pyqt5

Test CSI camera with gstreamer:
===============================

Friendlyarm: rockchip
---------------------

  ~> sudo apt-get install v4l-utils
  ~> v4l2-ctl --list-devices
     rkisp1-statistics (platform: rkisp1):
       /dev/video2
       /dev/video3
       /dev/video6
       /dev/video7
     rkisp1_mainpath (platform:ff910000.rkisp1):
       /dev/video0
       /dev/video1 <== CSI camera that can be used.
     rkisp1_mainpath (platform:ff920000.rkisp1):
       /dev/video4
       /dev/video5 <== CSI camera that can be used.

  https://wiki.friendlyarm.com/wiki/index.php/How_to_use_MIPI_camera_on_RK3399_boards
  http://wiki.friendlyarm.com/wiki/index.php/NanoPC-T4#Using_Camera_on_Linux_.28MIPI_Camera_OV13850_.26_OV4689.2C_and_webcam_logitect_C920.29

  ~> gst-launch-1.0 rkisp device=/dev/video1 io-mode=1 ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink
  ~> gst-launch-1.0 rkisp device=/dev/video5 io-mode=1 ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink

Nvidia: jetson nano
-------------------

  ~> v4l2-ctl --list-devices
     vi-output, imx477 6-001a (platform:54080000.vi:0):
       /dev/video0 <== CSI camera that can be used.
     USB 2.0 Camera (usb-70090000.xusb-2.3):
       /dev/video1 <== USB camera that can be used.

  ~> gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM)" ! nvvidconv ! nvoverlaysink
  ~> gst-launch-1.0 -v v4l2src device=/dev/video1 ! 'image/jpeg, format=MJPG' ! jpegdec ! xvimagesink

Vision3D:
=========

  ~/vision3D> module load opencv

  ~/vision3D> python3 capture.py --videoIDLeft 1 --videoIDRight 5 --hardware arm-nanopc

  ~/vision3D> python3 calibrate.py --videoID 1 --hardware arm-nanopc --load-frames [--fisheye]
  ~/vision3D> h5ls -flr CSI1.h5
  ~/vision3D> python3 calibrate.py --videoID 5 --hardware arm-nanopc --load-frames [--fisheye]
  ~/vision3D> h5ls -flr CSI5.h5

  ~/vision3D> python3 vision3D.py --videoIDLeft 1 --videoIDRight 5 --hardware arm-nanopc [--fisheye]
