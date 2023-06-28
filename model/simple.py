from torchvision import models
import torch.nn as nn
resnet_name_A = 'resnet18'
resnet_name_B = 'resnet34'
resnet_name_C = 'resnet50'
resnet_name_D = 'resnet101'
resnet_name_E = 'resnet152'


def get_resnet(model_name):
    if model_name == resnet_name_A:
        model = models.resnet18(weights=None)
    elif model_name == resnet_name_B:
        model = models.resnet34(weights=None)
    elif model_name == resnet_name_C:
        model = models.resnet50(weights=None)
    elif model_name == resnet_name_D:
        model = models.resnet101(weights=None)
    elif model_name == resnet_name_E:
        model = models.resnet152(weights=None)
    else:
        raise Exception('model name error')
    return model

def get_pretrained_resnet(model_name):
    if model_name == resnet_name_A:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == resnet_name_B:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == resnet_name_C:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == resnet_name_D:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif model_name == resnet_name_E:
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise Exception('model name error')
    return model

class ResNet(nn.Module):
    def __init__(self, num_classes=40, model_name='resnet18', pretrained=True):
        super().__init__()
        if pretrained:
            print('Using pretrained model')
            resnet = get_pretrained_resnet(model_name)
        else:
            print('No pretrained model')
            resnet = get_resnet(model_name)
        self.fe = nn.Sequential(*list(resnet.children()))[:-1]
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        emb = self.fe(x).squeeze()
        out = self.fc(emb).squeeze()
        return emb, out
    