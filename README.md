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

1. Requirements for matlab interface of [Caffe](http://caffe.berkeleyvision.org/installation.html)). Please use the modified     [Caffe for Faster R-CNN](https://github.com/ShaoqingRen/caffe/tree/faster-R-CNN).
2. MATLAB 
    
### Requirements: hardware

1. In training, you'll need ~3GB GPU memory for Zeiler & Fergus (ZF) network and ~8GB GPU memory for VGG 16-layers network 

### Downloads
1. Prototxts and related pre-trained network on ImageNet classification task
    Zeiler & Fergus (ZF) network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17219&authkey=!AKo99U4eBWjKbcY&ithint=file%2crar) [DropBox](https://www.dropbox.com/s/tqvqcwl7suge985/model_ZF.rar?dl=0) [BaiduYun](http://pan.baidu.com/s/1o668ygU)
    VGG 16-layers network [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17221&authkey=!ACNHeBfDAqzt0Uk&ithint=file%2crar) [DropBox](https://www.dropbox.com/s/8q1ugxhy71zqzhf/models_VGG16.rar?dl=0) [BaiduYun](http://pan.baidu.com/s/1hqkzZFm)
2. Pre-complied caffe mex
    Windows complied with VS2013 and Cuda6.5 [OneDrive](https://onedrive.live.com/download?resid=4006CBB8476FF777!17218&authkey=!AOqDbPj7Idd4O4w&ithint=file%2czip) [DropBox](https://www.dropbox.com/s/mqw7b7qqx0dojkb/caffe_library.zip?dl=0) [BaiduYun](http://pan.baidu.com/s/1mgxjcCC)
