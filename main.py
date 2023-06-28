from model.simple import ResNet
from prepare.eegdataset import C_GeneralEEGImageDataset, GeneralEEGImageDataset, GeneralEEGPointDataset, MySubset, FeatureEEGImageDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
from run.others import SmallNet, SimpleCNN
from utils.eegutils import get_test_setting, get_log_dir
from prepare.show import get_material_dir, plot_tsne
from run.start import get_args, get_dataset, get_device, MyEarlyStopping, run
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

args = get_args()

summary = SummaryWriter(log_dir=get_log_dir(args))
dataset = get_dataset(args)

k_fold = KFold(n_splits=args.k, shuffle=True)
device = get_device()

train_transforms = transforms.Compose([
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])
valid_transforms = transforms.Compose([
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])


# args.epochs = 3
# dataset = get_test_setting(dataset)
print('args: ', args)

for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):

    if fold > 0:
        break

    train_dataset = MySubset(dataset, train_ids, train_transforms)
    valid_dataset = MySubset(dataset, valid_ids, valid_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=3, prefetch_factor=2)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=1, prefetch_factor=1)

    print('Fold -', fold, ' num of train and test: ',
          len(train_ids), len(valid_ids))

    # 模型
    if 'resnet' in args.model_name:
        model = ResNet(num_classes=args.num_classes,
                       model_name=args.model_name, pretrained=args.pretrain)
    elif args.model_name == 'simplecnn':
        print('Using simplecnn')
        model = SimpleCNN(num_classes=args.num_classes).to(device)
    elif 'smallnet' in args.model_name:
        print('Using smallnet'+args.model_name)
        model = SmallNet(type=args.model_name[-1]).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    early_stopping = MyEarlyStopping(patience=10)

    for epoch in range(args.epochs):

        train_acc, train_loss = run(
            device, train_loader, model, summary, epoch, task='Train', optimizer=optimizer)
        test_acc, test_loss = run(
            device, valid_loader, model, summary, epoch, task='Test')
        print('Epoch: {} Train Acc/Loss: {:.2f}/{:.2f} Test Acc/Loss: {:.2f}/{:.2f} Lr: {:.4f}'.format(
            epoch, train_acc, train_loss, test_acc, test_loss, optimizer.param_groups[0]['lr']))

        early_stopping.check(test_acc)
        if early_stopping.stop:
            print(f'Early stopping after {epoch} epochs')
            break
print('Done')
