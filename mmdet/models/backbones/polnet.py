
import torch.nn as nn
from .resnet import ResNet
from ..registry import BACKBONES
import math


# 函数：make_cnn_model
# 功能：构造后端分类网络
# 参数：
#     model_name: 分类网络类型；目前支持resnet18 resnet34 resnet50 resnet101 alexnet vgg16_bn densenet squeezenet
#     num_classes：类别数
#     input_size：输入网络的图像尺寸
#     input_channel：输入网络的图像通道数
#     feature_extract: False训练整个网络，True只训练最后分类层
#     use_pretrained：
def make_cnn_model(model_name, num_classes, input_size=[224, 244], input_channel=3,
                   feature_extract=False, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None  # 最终构造的后端网络
    input_h = input_size[0]
    input_w = input_size[1]

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs*f(input_size[0])*f(input_size[1]), num_classes)


    elif model_name == "resnet34":
        model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs * f(input_size[0]) * f(input_size[1]), num_classes)

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs * f(input_size[0]) * f(input_size[1]), num_classes)

    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs * f(input_size[0]) * f(input_size[1]), num_classes)

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs * f(input_size[0]) * f(input_size[1]), num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16_bn":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = 512*math.floor(input_h/32)*math.floor(input_w/32)
        model_ft.classifier[0] = nn.Linear(num_ftrs, 4096, bias=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    # elif model_name == "inception":
    #     """ Inception v3
    #     Be careful, expects (299,299) sized images and has auxiliary output
    #     """
    #     model_ft = models.inception_v3(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     # Handle the auxilary net
    #     num_ftrs = model_ft.AuxLogits.fc.in_features
    #     model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    #     # Handle the primary net
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #     input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


@BACKBONES.register_module
class PolNet(nn.Module):
    def __init__(self,
                 fusion_cfg,

                 cnn_name):
        super(PolNet, self).__init__()
        self.fusion = self._make_layers(fusion_cfg, src_channels)
        if self.fusion is not None:
            self.cnn = make_cnn_model(cnn_name, num_classes, src_size, fusion_cfg[-1])
        else:
            self.cnn = make_cnn_model(cnn_name, num_classes, src_size, src_channels)
        self.fusion_out = None

    def forward(self, x):
        if self.fusion is not None:
            fusion_out = self.fusion(x)
            x = self.cnn(fusion_out)
        else:
            x = self.cnn(x)
        return x, fusion_out

    '''
    名称：_make_layers
    功能：构造偏振前端网络
    参数：
        cfg:[out_ch_num_1,out_ch_num_2,...out_ch_num_n]
            偏振前端网络的结构，out_ch_num_n代表每层的合成通道数
        src_channels：输入的偏振方向图通道数
    '''
    def _make_layers(self, cfg, src_channels):
        layers = []
        in_channels = src_channels
        if cfg is not None:
            for o in cfg:
                conv2d = nn.Conv2d(in_channels, o, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                layers += [conv2d, nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = o
            # layers += [nn.InstanceNorm2d(4)]
            return nn.Sequential(*layers)
        else:
            return None


