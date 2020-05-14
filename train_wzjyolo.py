from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


#train.py的主要工作流程

#1.解析输入的各种参数，如没有则使用默认参数
#2.打印各种参数
#3.初始化日志
#4.获得train_path、valid_path和class_names的文件路径
#5.创建model，随机初始化权重，也可以加载预训练的参数
#6.加载训练图像
#7.选择优化器
#8.开始epoch轮，反向传播
#9.开始训练batch_i批
#10.每累积gradient_accumulations批，进行一次梯度下降
#11.记录训练日志
#12.每训练完evaluation_interval轮输出一次评估结果
#13.每训练完opt.checkpoint_interval轮，保存一次checkpoints

if __name__ == "__main__":    #其他代码中import 则不执行下面代码
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")   #要训练的轮数
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")   #需要根据显卡计算力设置
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")   #训练中进行参数优化的间隔，大多数是1
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")      #模型以darknet类型的cfg给出
    parser.add_argument("--data_config", type=str, default="config/person.data", help="path to data config file")          #所有类别名称
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")              #是否采用预训练模型
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")     #cpu训练采用的线程数
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")                        #输入尺寸
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")      #保存模型权重的间隔
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")     #每个evaluation_interval轮输出一次评估结果
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)         #2.打印各种参数

    logger = Logger("tensorboard_log")   #3.初始化日志

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints_wzjyolo", exist_ok=True)

    # Get data configuration
    #4.获得train_path、valid_path和class_names的文件路径
    data_config = parse_data_config(opt.data_config)    #parse_data_config自定义的函数在parse_config.py中
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    # Initiate model
    #5.创建model，随机初始化权重，也可以加载预训练的参数
    #model = Darknet(opt.model_def).to(device)
    model = WZJ_YOLO().to(device)
    #print (model)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:           #有两种方法来加载预训练模型
        if opt.pretrained_weights.endswith(".pth"):         #用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    #6.加载训练图像
    #print (train_path)
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(          #数据集的处理
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    #7.选择优化器
    optimizer = torch.optim.Adam(model.parameters())   


    #输出的内容
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

#print (model.yolo_layers)

    #8.开始训练epoch轮，反向传播
    #训练epoch轮，每轮有 batch_i 批，每批有imgs个图，每个图有targets个目标
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        #9.开始训练batch_i批
        for batch_i, (_, imgs, targets) in enumerate(dataloader):   #当前是第epoch轮中的第 batch_i批
            batches_done = len(dataloader) * epoch + batch_i        #当前已经训练的总 batch 数目

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)  #imgs, targets喂给model,得到loss, outputs
            loss.backward()

            #10.每累积gradient_accumulations批，进行一次梯度下降  手动设置的，这里设置为2
            if batches_done % opt.gradient_accumulations:
                print ('更新权值参数')
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            #11.记录训练日志
            # ----------------
            #   Log progress
            # ----------------
            #      当前epoch/总epoch        当前epoch 训练的batch数目/ 当前epoch 总batch数目
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):  #记录YOLO layer评估结果
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        #12.每个evaluation_interval轮输出一次评估结果
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        #13.每训练opt.checkpoint_interval轮，保存一次checkpoints
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints_wzjyolo/yolov3_ckpt_%d.pth" % epoch)
