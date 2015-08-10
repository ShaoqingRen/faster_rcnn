# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun at Microsoft Research

### Introduction

**Faster R-CNN** is a framework for object detection with deep CNNs including a Region Proposal Network (RPN) and an Object Detection Network. Both networks are trained for sharing convolutional layers for fast testing. 

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497).

This repo contains a MATLAB re-implementation of Fast R-CNN. Details about Fast R-CNN are in: [rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @article{ren15fasterrcnn,
        Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
        Title = {{Faster R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
        Journal = {arXiv preprint arXiv:1506.01497},
        Year = {2015}
    }

### Requirements: software

1. Caffe build for Faster R-CNN (included in this repository)
    - If you are using Windows, you may download a compiled mex file by running fetch_data/fetch_caffe_library.m
    - If you are using Linux or you want to compile for Windows, please follow the instructions on our Caffe branch.
2.	MATLAB
 
    
### Requirement: hardware

GPU memory

1. Region Proposal Network (RPN)
    - 2GB GPU memory for ZF net
    - 5GB GPU memory for VGG-16 net
2. Ojbect Detection Network (Fast R-CNN)
    - 3GB GPU memory for ZF net
    - 8GB GPU memory for VGG-16 net

### Preparation:
1.	Run fetch_data/fetch_caffe_library.m to download a compiled Caffe mex (for Windows only).
2.	Run fetch_data/fetch_model_ZF.m to download an ImageNet-pre-trained ZF net.
3.	Run fetch_data/fetch_model_VGG16.m to download an ImageNet-pre-trained VGG-16 net.
4.	Run faster_rcnn_build.m
5.	Run startup.m

### Testing Demo:
1.	Run fetch_data/fetch_model_trained.m to download our trained models.
2.	Run experiments/script_faster_rcnn_demo.m to test a single demo image.
    - The first run might be slower due to memory load.
    - The running time on K40 of this code is about 220ms/image, 10% more than we reported in the paper. This is because of unknown issues when we switch from our older version of Caffe to the newer one.
    - The speed on Titan X is about 2x of on K40.


### Downloads
1. Prototxts and related pre-trained network on ImageNet classification task
    - Zeiler & Fergus (ZF) network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17256&authkey=!AF7wGc1kbUTfI7o&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/sw58b2froihzwyf/model_ZF.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1sj3K21B)
    - VGG 16-layers network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17257&authkey=!AO38BiePXqYrz5M&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/z5rrji25uskha73/model_VGG16.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1pJ9opyr)
2. Pre-complied caffe mex
    - Windows based complied with VS2013 and Cuda6.5 [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17255&authkey=!AHOIeRzQKCYXD3U&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/m6sg347tiaqpcwy/caffe_mex.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1nZYOI)
3. Experiment logs [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17290&authkey=!AGhH4z667tHYYEw&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/wu841r7zmebjp6r/faster_rcnn_logs.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1nt48EJB)
4. Final output models [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17292&authkey=!AFCIads9CKr5-4s&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/jswrnkaln47clg2/faster_rcnn_final_model.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1eQwF64y)
