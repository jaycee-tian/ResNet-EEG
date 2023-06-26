import torch
import torch.nn as nn
import torchvision.models as models

from utils.eegutils import freeze_params

# mobilenet

class SmallNet(nn.Module):
    def __init__(self, type='A',num_classes=40, pretrained=True):
        super(SmallNet, self).__init__()
        if pretrained:
            if type == 'A':
                self.smallnet = models.get_model('mobilenet_v2', weights=models.MobileNet_V2_Weights.DEFAULT)
            elif type == 'B':
                self.smallnet = models.get_model('mobilenet_v3_large', weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            elif type == 'C':
                self.smallnet = models.get_model('mobilenet_v3_small', weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            elif type == 'D':
                self.smallnet = models.get_model('efficientnet_b0', weights=models.EfficientNet_B0_Weights.DEFAULT)
            elif type == 'E':
                self.smallnet = models.get_model('efficientnet_b1', weights=models.EfficientNet_B1_Weights.DEFAULT)
            elif type == 'F':
                self.smallnet = models.get_model('efficientnet_b2', weights=models.EfficientNet_B2_Weights.DEFAULT)
            elif type == 'G':
                self.smallnet = models.get_model('efficientnet_b3', weights=models.EfficientNet_B3_Weights.DEFAULT)
            elif type == 'H':
                self.smallnet = models.get_model('convnext_small', weights=models.ConvNeXt_Small_Weights.DEFAULT)
            # freeze
            # freeze_params(self.smallnet)
        # else:
        #     self.smallnet.
        
        if type == 'B' or type == 'C':
            in_features = self.smallnet.classifier[0].in_features
            self.smallnet.classifier = nn.Linear(in_features, num_classes)
        elif type == 'H':
            in_features = self.smallnet.classifier[2].in_features
            self.smallnet.classifier[2] = nn.Linear(in_features, num_classes)
        else:
            in_features = self.smallnet.classifier[1].in_features



            
    
    def forward(self, x):
        return self.smallnet(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
        