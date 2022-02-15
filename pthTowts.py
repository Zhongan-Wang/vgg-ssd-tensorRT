import torch
from torch import nn
#load你的模型
import os
import struct

from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
from vision.ssd.vgg_ssd import create_vgg_ssd

def main():
    label_path = "models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names) 
    net = create_vgg_ssd(num_classes, is_test=True)
    net.load('models/ssd_vgg.pth') #loadpth文件

    net = net.to('cuda:0')
    net.eval()
 
    f = open("models/ssd_vgg.wts", 'w') #自己命名wts文件
    f.write("{}\n".format(len(net.state_dict().keys())))  #保存所有keys的数量
    for k,v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))  #保存每一层名称和参数长度
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())  #使用struct把权重封装成字符串
        f.write("\n")

if __name__ == '__main__':
    main()
