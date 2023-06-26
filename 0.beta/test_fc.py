
import setpath
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from prepare.show import get_material_dir
from run.start import get_device
from prepare.eegdataset import GeneralEEGPointDataset, MySubset
from torch.utils.tensorboard import SummaryWriter

from utils.eegutils import get_simple_log_dir
data_path = '/data0/tianjunchao/dataset/CVPR2021-02785/data/img_pkl/32x32'


# 定义超参数
hidden_size = 256
num_layers = 2
num_classes = 40
learning_rate = 0.001
num_epochs = 100
batch_size = 128
window_size = 100

if window_size > 0:
    input_size = 717 - window_size + 1
else:
    input_size = 717

dataset = GeneralEEGPointDataset(path=data_path, window_size=100)
# dataset = get_test_setting(dataset)
# train_loader = DataLoader(dataset, batch_size=128)
device = get_device()
material_dir = get_material_dir()

log_dir = get_simple_log_dir()
summary = SummaryWriter(log_dir=log_dir)



# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.base(x)
        out = self.fc(x)
        return out


k_fold = KFold(n_splits=5, shuffle=True)
# model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
model = MLPModel(input_size, hidden_size, num_classes).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):

    if fold > 0:
        break

    # use train_ids and valid_ids to get train_loader and valid_loader
    # don't use Mysubset, use sampler instead
    train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)




    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        # compute acc
        correct = 0
        total = 0

        # train model
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.squeeze()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 反向传播并优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to tensorboard
            summary.add_scalar('train loss', loss.item(), epoch * total_step + i)

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), 100 * correct / total))
        # train acc
        summary.add_scalar('train acc', 100 * correct / total, epoch)
        
        # test model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs = inputs.squeeze()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                # test loss
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Valid Acc: {:.2f}%'.format(100 * correct / total))
            summary.add_scalar('valid loss', loss.item(), epoch)
            summary.add_scalar('valid acc', 100 * correct / total, epoch)


        