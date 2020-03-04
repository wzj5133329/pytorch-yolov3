# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/wzj5133329/pytorch-yolov3
    $ cd pytorch-yolov3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh
    

## Train on Custom Dataset

#### Custom model

> 生成符合自有数据集的模型文件

##### 修改类别数目
```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg' #修改网络中的训练类数目
```
##### 修改anchors
[教程](https://github.com/wzj5133329/MobileNet_yolo/tree/master/create_lmdb)

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

修改 config/custom.data文件

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/   (与imgaes文件夹匹配，且label要与classes.names顺序匹配)  
 Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets

生成包含训练与验证集所有图片路径的 trian.txt与valid.txt文件 使用 data/deal/imagename2txt.py 文件
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.


## Demo

To test on one image :(ESC退出)  (需要安装opencv)

```
$ pip install opencv-python

$ python3 demo.py --image_path=./data/samples/dog.jpg
```

To test on folder & save the images :

```
$ python3 detect.py --image_folder data/samples/
```

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)


>参考 https://github.com/eriklindernoren/PyTorch-YOLOv3

