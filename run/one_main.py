from run.resnet import TesNet
from prepare.eegdataset import C_GeneralEEGImageDataset, GeneralEEGImageDataset, GeneralEEGPointDataset, MySubset, FeatureEEGImageDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
from run.others import SmallNet, SimpleCNN
from utils.eegutils import get_test_setting, get_log_dir
from prepare.show import get_material_dir, plot_tsne
from run.start import get_args, get_data_dir, get_device, MyEarlyStopping, run
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

args = get_args()
# args.description = '加上归一化跑一遍resnet18看看'
# args.target = 'test'

data_dir = get_data_dir()
model_dir = get_log_dir(args, "one")
summary = SummaryWriter(log_dir=model_dir)


# dataset = GeneralEEGImageDataset(
#     path=data_dir, n_channels=args.n_channels, grid_size=args.grid_size, window_size=20)
dataset = FeatureEEGImageDataset(
    path=data_dir, n_channels=args.n_channels, grid_size=args.grid_size)
# dataset = GeneralEEGImageDataset(
#     path=data_dir, n_channels=args.n_channels, grid_size=args.grid_size)

if args.target == 'test':
    args.epochs = 3
    dataset = get_test_setting(dataset)

print('args: ', args)

k_fold = KFold(n_splits=args.k, shuffle=True)


device = get_device()


# print('resize, rotate, crop')
train_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0),antialias=True),
    # transforms.
    # resize(256)
    # transforms.RandomCrop(224, padding=4),
    # transforms.Resize((256, 256),antialias=True),
    # transforms.RandomRotation(30),
    # transforms.RandomCrop(224),
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])
valid_transforms = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])

load_path = '4.materials/model/0417/2233/resnet18_407/model.pth'
material_path = get_material_dir()
for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):

    if fold > 0:
        break

    # 对训练集数据应用变换
    # dataset.transform = train_transforms
    train_dataset = MySubset(dataset, train_ids, train_transforms)
    # 对验证集数据应用变换
    # dataset.transform = valid_transforms
    valid_dataset = MySubset(dataset, valid_ids, valid_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=3, prefetch_factor=2)
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=1, prefetch_factor=1)

    print('Fold -', fold, ' num of train and test: ',
          len(train_ids), len(valid_ids))

    # 模型
    if 'resnet' in args.model_name:
        if args.ptr == 'yes':
            print('Using pretrained model')
            model = TesNet(num_classes=args.num_classes,
                           model_name=args.model_name)
            # model.load(load_path)
        else:
            print('No pretrained model')
            model = TesNet(num_classes=args.num_classes,
                           model_name=args.model_name, pretrained=False)
    elif args.model_name == 'simplecnn':
        print('Using simplecnn')
        model = SimpleCNN(num_classes=args.num_classes).to(device)
    elif 'smallnet' in args.model_name:
        print('Using smallnet'+args.model_name)
        model = SmallNet(type=args.model_name[-1]).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler1 = LinearLR(optimizer, start_factor=0.5,
                             total_iters=len(train_loader)*5, last_epoch=-1)
    lr_scheduler2 = CosineAnnealingLR(
        optimizer, T_max=args.epochs*len(train_loader), eta_min=1e-4)

    # # lr_scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5)
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #                             momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs)

    # 创建EarlyStopping实例
    early_stopping = MyEarlyStopping(patience=10)

    for epoch in range(args.epochs):

        train_acc, train_loss = run(
            device, train_loader, model, summary, epoch, task='Train', optimizer=optimizer, lr_schedulers=[lr_scheduler1, lr_scheduler2])
        test_acc, test_loss = run(
            device, valid_loader, model, summary, epoch, task='Test')
        print('Epoch: {} Train Acc/Loss: {:.2f}/{:.2f} Test Acc/Loss: {:.2f}/{:.2f} Lr: {:.4f}'.format(
            epoch, train_acc, train_loss, test_acc, test_loss, optimizer.param_groups[0]['lr']))

        early_stopping.check(test_acc)
        if early_stopping.stop:
            print(f'Early stopping after {epoch} epochs')
            break

    # plot t-sne and save
    # if args.is_parallel == 'yes':
    #     model = model.module
        # plt = plot_tsne(model, train_loader, device, args.model_name, target='train epoch '+str(epoch),  material_dir=material_path,plot=True)
        # add picture to tensorboard
        # summary.add_figure('tsne train', plt.gcf(), epoch)

        # plt = plot_tsne(model, valid_loader, device, args.model_name, target='test epoch '+str(epoch), material_dir=material_path,plot=True)
        # add picture to tensorboard
        # summary.add_figure('tsne test', plt.gcf(), epoch)

        # break
print('Done')
