import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from model.simple import get_pretrained_resnet, get_resnet

from utils.eegutils import freeze_params


# Define the He initialization method
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ResNetC3(nn.Module):
    def __init__(self, num_classes=40, model_name='resnet101', pretrained=True):
        super().__init__()
        self.resnet = get_pretrained_resnet(model_name)
        if pretrained == False:
            self.resnet.apply(init_weights)
        else:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        # self.resnet = nn.Sequential(*list(self.resnet.children()))[:-1]
        # self.resnet.fc = nn.Conv2d(in_features, num_classes, kernel_size=1)
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze()
        
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=40):
        super(ClassifierHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class TesNet(nn.Module):
    def __init__(self, num_classes=40, model_name='resnet18', pretrained=True):
        super().__init__()
        if pretrained == False:
            resnet = get_resnet(model_name)
        else:
            resnet = get_pretrained_resnet(model_name)
        self.fea_e = nn.Sequential(*list(resnet.children()))[:-1]
        self.pro_h = ProjectionHead(input_dim=resnet.fc.in_features)
        self.cls_h = ClassifierHead(input_dim=resnet.fc.in_features, output_dim=num_classes)
    
    def forward(self, x):
        emb = self.fea_e(x).squeeze()
        proj = self.pro_h(emb)
        out = self.cls_h(emb).squeeze()
        return emb, proj, out
    
    # load pretrained model from local file
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print('load pretrained model from {}'.format(model_path))

class IceResNet101(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()  
        self.resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet101.fc = nn.Linear(2048, num_classes)
        # 冻结卷积层参数
        for name, param in self.resnet101.named_parameters():
            if 'conv1' not in name and 'fc' not in name:
                param.requires_grad = False
        
    def forward(self, x):
        return self.resnet101(x)
    
class FusionResNet101(nn.Module):
    def __init__(self, num_classes=40,pretrained=True):
        super().__init__()
        if pretrained:
            self.raw_resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.diff_resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            self.raw_resnet101 = models.resnet101(weights=None)
            self.diff_resnet101 = models.resnet101(weights=None)

        self.raw_resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.raw_resnet101 = nn.Sequential(*list(self.raw_resnet101.children())[:-1])
        self.diff_resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.diff_resnet101 = nn.Sequential(*list(self.diff_resnet101.children())[:-1])

        # 添加一个1x1卷积层，将特征的通道数转换为所需的分类数
        self.conv = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, raw_x,diff_x):
        raw_feature = self.raw_resnet101(raw_x)  # ResNet1 的输出
        diff_feature = self.diff_resnet101(diff_x)  # ResNet2 的输出
        x=raw_feature*self.alpha+diff_feature*(1-self.alpha)
        # x = torch.cat((raw_feature, diff_feature), dim=1)
        # 将融合后的特征输入到1x1卷积层中，进行通道数转换
        x = self.conv(x).squeeze()
        return x



class BaseFusionResNet101(nn.Module):
    def __init__(self, num_classes=40,pretrained=True):
        super().__init__()
        if pretrained:
            self.raw_resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.diff_resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            self.raw_resnet101 = models.resnet101(weights=None)
            self.diff_resnet101 = models.resnet101(weights=None)

        self.raw_resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.diff_resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.raw_resnet101.fc = nn.Linear(2048, num_classes)
        self.diff_resnet101.fc = nn.Linear(2048, num_classes)
        
    def forward(self, raw_x,diff_x):
        raw_y = self.raw_resnet101(raw_x)  # ResNet1 的输出
        diff_y = self.diff_resnet101(diff_x)  # ResNet2 的输出
        y=raw_y+diff_y
        return y




class BFusionResNet101(BaseFusionResNet101):
    def __init__(self, num_classes=40,pretrained=True):
        super().__init__(num_classes=num_classes,pretrained=pretrained)
        self.raw_resnet101 = nn.Sequential(*list(self.raw_resnet101.children())[:-1])
        self.diff_resnet101 = nn.Sequential(*list(self.diff_resnet101.children())[:-1])
        # 添加一个1x1卷积层，将特征的通道数转换为所需的分类数
        self.conv1 = torch.nn.Conv2d(in_channels=2*2048, out_channels=1024, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
 
    def forward(self, raw_x,diff_x):
        raw_feature = self.raw_resnet101(raw_x)  # ResNet1 的输出
        diff_feature = self.diff_resnet101(diff_x)  # ResNet2 的输出
        x = torch.cat((raw_feature, diff_feature), dim=1)
        # 将融合后的特征输入到1x1卷积层中，进行通道数转换
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x).squeeze()
        return x