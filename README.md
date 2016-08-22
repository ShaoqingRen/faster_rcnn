# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun at Microsoft Research

### Introduction

**Faster** R-CNN is an object detection framework based on deep convolutional networks, which includes a Region Proposal Network (RPN) and an Object Detection Network. Both networks are trained for sharing convolutional layers for fast testing. 

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497).

This repo contains a MATLAB re-implementation of Fast R-CNN. Details about Fast R-CNN are in: [rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

This code has been tested on Windows 7/8 64-bit, Windows Server 2012 R2, and Linux, and on MATLAB 2014a.

Python version is available at [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).

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

### Main Results
                          | training data                          | test data            | mAP   | time/img
------------------------- |:--------------------------------------:|:--------------------:|:-----:|:-----:
Faster RCNN, VGG-16       | VOC 2007 trainval                      | VOC 2007 test        | 69.9% | 198ms
Faster RCNN, VGG-16       | VOC 2007 trainval + 2012 trainval      | VOC 2007 test        | 73.2% | 198ms
Faster RCNN, VGG-16       | VOC 2012 trainval                      | VOC 2012 test        | 67.0% | 198ms
Faster RCNN, VGG-16       | VOC 2007 trainval&test + 2012 trainval | VOC 2012 test        | 70.4% | 198ms

**Note**: The mAP results are subject to random variations. We have run 5 times independently for ZF net, and the mAPs are 59.9 (as in the paper), 60.4, 59.5, 60.1, and 59.5, with a mean of 59.88 and std 0.39.


### Contents
0. [Requirements: software](#requirements-software)
0. [Requirements: hardware](#requirements-hardware)
0. [Preparation for Testing](#preparation-for-testing)
0. [Testing Demo](#testing-demo)
0. [Preparation for Training](#preparation-for-training)
0. [Training](#training)
0. [Resources](#resources)


### Requirements: software

0. `Caffe` build for Faster R-CNN (included in this repository, see `external/caffe`)
    - If you are using Windows, you may download a compiled mex file by running `fetch_data/fetch_caffe_mex_windows_vs2013_cuda65.m`
    - If you are using Linux or you want to compile for Windows, please follow the [instructions](https://github.com/ShaoqingRen/caffe/tree/faster-R-CNN) on our Caffe branch.
0.	MATLAB
 
    
### Requirements: hardware

GPU: Titan, Titan Black, Titan X, K20, K40, K80.

0. Region Proposal Network (RPN)
    - 2GB GPU memory for ZF net
    - 5GB GPU memory for VGG-16 net
0. Object Detection Network (Fast R-CNN)
    - 3GB GPU memory for ZF net
    - 8GB GPU memory for VGG-16 net


### Preparation for Testing:
0.	Run `fetch_data/fetch_caffe_mex_windows_vs2013_cuda65.m` to download a compiled Caffe mex (for Windows only).
0.	Run `faster_rcnn_build.m`
0.	Run `startup.m`


### Testing Demo:
0.	Run `fetch_data/fetch_faster_rcnn_final_model.m` to download our trained models.
0.	Run `experiments/script_faster_rcnn_demo.m` to test a single demo image.
    - You will see the timing information as below. We get the following running time on K40 @ 875 MHz and Intel Xeon CPU E5-2650 v2 @ 2.60GHz for the demo images with VGG-16:
	```Shell
	001763.jpg (500x375): time 0.201s (resize+conv+proposal: 0.150s, nms+regionwise: 0.052s)
	004545.jpg (500x375): time 0.201s (resize+conv+proposal: 0.151s, nms+regionwise: 0.050s)
	000542.jpg (500x375): time 0.192s (resize+conv+proposal: 0.151s, nms+regionwise: 0.041s)
	000456.jpg (500x375): time 0.202s (resize+conv+proposal: 0.152s, nms+regionwise: 0.050s)
	001150.jpg (500x375): time 0.194s (resize+conv+proposal: 0.151s, nms+regionwise: 0.043s)
	mean time: 0.198s
	```
	and with ZF net:
	```Shell
	001763.jpg (500x375): time 0.061s (resize+conv+proposal: 0.032s, nms+regionwise: 0.029s)
	004545.jpg (500x375): time 0.063s (resize+conv+proposal: 0.034s, nms+regionwise: 0.029s)
	000542.jpg (500x375): time 0.052s (resize+conv+proposal: 0.034s, nms+regionwise: 0.018s)
	000456.jpg (500x375): time 0.062s (resize+conv+proposal: 0.034s, nms+regionwise: 0.028s)
	001150.jpg (500x375): time 0.058s (resize+conv+proposal: 0.034s, nms+regionwise: 0.023s)
	mean time: 0.059s
	```
    - The visual results might be different from those in the paper due to numerical variations.	
    - Running time on other GPUs
    
           GPU / mean time        |        VGG-16        |        ZF          
	:------------------------:|:--------------------:|:--------------------:
		  K40             |        198ms         |       59ms       
	     Titan Black          |        174ms         |       56ms       
		Titan X           |        151ms         |       59ms

### Preparation for Training:
0.	Run `fetch_data/fetch_model_ZF.m` to download an ImageNet-pre-trained ZF net.
0.	Run `fetch_data/fetch_model_VGG16.m` to download an ImageNet-pre-trained VGG-16 net.
0.	Download VOC 2007 and 2012 data to ./datasets


### Training:
0. Run `experiments/script_faster_rcnn_VOC2007_ZF.m` to train a model with ZF net. It runs four steps as follows:
    - Train RPN with conv layers tuned; compute RPN results on the train/test sets.
    - Train Fast R-CNN with conv layers tuned using step-1 RPN proposals; evaluate detection mAP.
    - Train RPN with conv layers fixed; compute RPN results on the train/test sets. 
    - Train Fast R-CNN with conv layers fixed using step-3 RPN proposals; evaluate detection mAP.
    - **Note**: the entire training time is ~12 hours on K40.
0. Run `experiments/script_faster_rcnn_VOC2007_VGG16.m` to train a model with VGG net.
    - **Note**: the entire training time is ~2 days on K40.
0. Check other scripts in `./experiments` for more settings.

### Resources

**Note**: This documentation may contain links to third party websites, which are provided for your convenience only. Such third party websites are not under Microsoft’s control. Microsoft does not endorse or make any representation, guarantee or assurance regarding any third party website, content, service or product. Third party websites may be subject to the third party’s terms, conditions, and privacy statements.

0. Experiment logs: [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!110&authkey=!ACpgYZR2MmfklwI&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/wu841r7zmebjp6r/faster_rcnn_logs.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1ntJ3dLv)
0. Regions proposals of our trained RPN:
    - ZF net trained on VOC 07 trainval [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!115&authkey=!AJJMrFJHKLXIg5c&ithint=file%2czip), [BaiduYun](http://pan.baidu.com/s/1pKGBDyz)
    - ZF net trained on VOC 07/12 trainval [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!117&authkey=!AJiy5F6Cum1iosI&ithint=file%2czip), [BaiduYun](http://pan.baidu.com/s/1jGAgkZW)
    - VGG net trained on VOC 07 trainval [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!116&authkey=!AH4Zi_KAaun7MhQ&ithint=file%2czip), [BaiduYun](http://pan.baidu.com/s/1qWHv4JU)
    - VGG net trained on VOC 07/12 trainval [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!118&authkey=!AB_lKk3dbGyr1-I&ithint=file%2czip), [BaiduYun](http://pan.baidu.com/s/1c0fQpqg)
    - **Note**: the proposals are in the format of [left, top, right, bottom, confidence]

If the automatic "fetch_data" fails, you may manually download resouces from:

0. Pre-complied caffe mex:
    - Windows-based mex complied with VS2013 and Cuda6.5: [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!111&authkey=!AFVWFGTbViiX5tg&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/m6sg347tiaqpcwy/caffe_mex.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1i3m0i0H)
0. ImageNet-pretrained networks:
    - Zeiler & Fergus (ZF) net [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!113&authkey=!AIzdm0sD_SmhUQ4&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/sw58b2froihzwyf/model_ZF.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1o6zipPS)
    - VGG-16 net [OneDrive](https://onedrive.live.com/download?resid=36FEC490FBC32F1A!114&authkey=!AE8uV9B07dREbhM&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/z5rrji25uskha73/model_VGG16.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1mgzSnI4)
0. Final RPN+FastRCNN models: [OneDrive](https://onedrive.live.com/download?resid=D7AF52BADBA8A4BC!114&authkey=!AERHoxZ-iAx_j34&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/jswrnkaln47clg2/faster_rcnn_final_model.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1hsFKmeK)


