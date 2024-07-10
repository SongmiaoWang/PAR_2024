import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MyResnet_18(nn.Module):
    def __init__(self):
        super(MyResnet_18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        for param in resnet18.parameters():
            param.requires_grad = False  # False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
        # models.resnet18.avgpool = lambda x: x ---没用
        # models.resnet18.fc = lambda x: x
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)

        outputs.append(x)

        return outputs

class MyInception_v3(nn.Module):
    def __init__(self, transform_input=False, pretrained=False):
        super(MyInception_v3, self).__init__()
        self.transform_input = transform_input
        inception = models.inception_v3(pretrained=pretrained)

        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

    def forward(self, x):
        outputs = []

        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        outputs.append(x)

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        outputs.append(x)

        return outputs


class MyVGG16(nn.Module):
    def __init__(self, pretrained=False):
        super(MyVGG16, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)

        self.features = vgg.features

    def forward(self, x):
        x = self.features(x)
        return [x]


class MyVGG19(nn.Module):
    def __init__(self, pretrained=False):
        super(MyVGG19, self).__init__()

        vgg = models.vgg19(pretrained=pretrained)

        self.features = vgg.features

    def forward(self, x):
        x = self.features(x)
        return [x]
