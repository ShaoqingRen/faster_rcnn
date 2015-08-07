# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun at Microsoft Research

### Introduction

**Faster R-CNN** is a framework for object detection with deep ConvNets including windows proposal network (RPN) and detection network. These two networks are trained for sharing convolution layers to speed up testing speed. 

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497).

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @article{girshick15fastrcnn,
        Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
        Title = {Faster R-CNN},
        Journal = {arXiv preprint arXiv:1506.01497},
        Year = {2015}
    }

### Requirements: software

1. Requirements for matlab interface of [Caffe](http://caffe.berkeleyvision.org/installation.html). Please use the modified     [Caffe for Faster R-CNN](https://github.com/ShaoqingRen/caffe/tree/faster-R-CNN).
2. MATLAB 
    
### Requirements: hardware

1. GPU memory 
   - ~3GB GPU memory for Zeiler & Fergus (ZF) network 
   - ~8GB GPU memory for VGG 16-layers network 

### Downloads
1. Prototxts and related pre-trained network on ImageNet classification task
    - Zeiler & Fergus (ZF) network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17256&authkey=!AF7wGc1kbUTfI7o&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/sw58b2froihzwyf/model_ZF.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1sj3K21B)
    - VGG 16-layers network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17257&authkey=!AO38BiePXqYrz5M&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/z5rrji25uskha73/model_VGG16.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1pJ9opyr)
2. Pre-complied caffe mex
    - Windows based complied with VS2013 and Cuda6.5 [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17255&authkey=!AHOIeRzQKCYXD3U&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/m6sg347tiaqpcwy/caffe_mex.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1nZYOI)
