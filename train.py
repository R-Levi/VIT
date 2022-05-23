import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from ImageNetDataset import ImageNet
#from model import ViT
from my_model import ViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_SET_MODEL = 'CIFAR100'
BATCH_SIZE = 1024
NUMBER_OF_EPOCHS = 500
SAVE_EPOCH_MODEL = 100
LOAD_MODEL_LOC = None
SAVE_MODEL_LOC = 'model_'

LR = 2e-3
LR_d = 150
In_channels=3
Img_size=32
Patch_size = 4
Depth = 8  #4
Num_class=100
Drop_out = 0

PRINT = True
PRINT_GRAPH = True
PRINT_CM = False

# measures accuracy of predictions at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader, num_classes=100,epoch=100):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    cm = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            y_ = model(x.to(DEVICE))
            y_top1 = torch.argmax(y_, dim=1)
            maxk = max((1,5))
            _,y_top5 = torch.topk(y_,maxk,1,True,True)
            # print(y_.shape)
            # print(y.shape)
            # print(y_top5.shape)
            correct_top1 += (y_top1 == y.to(DEVICE)).sum()
            correct_top5 += torch.eq(y_top5,y.to(DEVICE).view(-1,1)).sum()
            total+=x.shape[0]

            if i < 20:
                for label,pre in zip(y,y_top1.cpu().numpy()):
                    cm[label][pre] += 1

    if PRINT_CM and epoch==NUMBER_OF_EPOCHS:
        class_labels = list(range(num_classes))
        ax = sn.heatmap(
            cm,
            annot=True,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels, 
            fmt='g')
        ax.set(
            xlabel="prediction",
            ylabel="truth",
            title="Confusion Matrix for " + ("Training set" if loader.dataset.train else "Validation dataset"))
        #plt.show()
        plt.savefig(os.path.join(os.path.join("images",f"{DATA_SET_MODEL}"),"my_cm_train.png" if loader.dataset.train else "my_cm_val.png"))
    # print(correct.cpu().numpy())
    # print(total)
    return (correct_top1.cpu().numpy() / total).item() * 100 ,(correct_top5.cpu().numpy() / total).item() * 100


# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, scheduler, train_loader, validation_loader):
    epoch = 1
    if LOAD_MODEL_LOC:
        logger.write(f"load ckpt {LOAD_MODEL_LOC}....")
        print(f"load ckpt {LOAD_MODEL_LOC}....")
        model.load_state_dict(torch.load(LOAD_MODEL_LOC))
        epoch = int(LOAD_MODEL_LOC.split('_')[-1])+1
    accuracy_per_epoch_train_top1 = []
    accuracy_per_epoch_train_top5 = []
    accuracy_per_epoch_val_top1 = []
    accuracy_per_epoch_val_top5 = []

    for epoch in range(epoch,NUMBER_OF_EPOCHS+1):
        model.train()
        progress = tqdm(train_loader)
        
        for i, (x, y) in enumerate(progress):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # if i > BATCHES_PER_EPOCH:
            #     break
            y_ = model(x)
            loss = loss_function(y_, y)

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item(),lr=scheduler.state_dict()['_last_lr'])
            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        train_top1,train_top5 = accuracy(model, train_loader, num_classes=Num_class, epoch=epoch)
        val_top1,val_top5 = accuracy(model, validation_loader, num_classes=Num_class, epoch=epoch)
        accuracy_per_epoch_train_top1.append(train_top1)
        accuracy_per_epoch_train_top5.append(train_top5)
        accuracy_per_epoch_val_top1.append(val_top1)
        accuracy_per_epoch_val_top5.append(val_top5)

        if SAVE_MODEL_LOC and epoch%SAVE_EPOCH_MODEL==0:
            logger.write(f"save model {epoch}....")
            print(f"save model {epoch}....")
            torch.save(model.state_dict(), os.path.join(os.path.join("model",f"{DATA_SET_MODEL}"),SAVE_MODEL_LOC + str(epoch)+'.pth'))

        if PRINT:
            tqdm.write("Train Top 1 Accuracy  for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_train_top1[-1]))
            tqdm.write("Train Top 5 Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_train_top5[-1]))
            tqdm.write("Test Top 1 Accuracy  for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_val_top1[-1]))
            tqdm.write("Test Top 5 Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_val_top5[-1]))


            logger.write("Train Top 1 Accuracy  for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_train_top1[-1]))
            logger.write("Train Top 5 Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_train_top5[-1]))
            logger.write("Test Top 1 Accuracy  for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_val_top1[-1]))
            logger.write("Test Top 5 Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy_per_epoch_val_top5[-1]))

        if PRINT_GRAPH and epoch == NUMBER_OF_EPOCHS :
            plt.figure(figsize=(10, 10), dpi=100)
            plt.plot(range(1,epoch + 1), accuracy_per_epoch_train_top1,
                     color='b', marker='o', linestyle='dashed', label='Training')
            plt.plot(range(1, epoch + 1), accuracy_per_epoch_val_top1,
                     color='r', marker='o', linestyle='dashed', label='Validation')
            plt.plot(range(1, epoch + 1), accuracy_per_epoch_train_top5,
                     color='green', marker='_', linestyle='dashed', label='Training')
            plt.plot(range(1, epoch + 1), accuracy_per_epoch_val_top5,
                     color='greenyellow', marker='_', linestyle='dashed', label='Validation')
            plt.legend()
            plt.title("Graph of accuracy over time")
            plt.xlabel("epoch #")
            plt.ylabel("accuracy %")
            if epoch < 20:
                plt.xticks(range(1, epoch + 1))
            plt.ylim(0, 100)
            # plt.show()
            plt.savefig(os.path.join(os.path.join("images",f"{DATA_SET_MODEL}"),"my_accuracy.png"))
        

if __name__ == "__main__":
    from utils import logger
    global logger
    logger = logger(f"log",f"{DATA_SET_MODEL}_log")
    logger.write(f"load data {DATA_SET_MODEL}")
    print(f"load data {DATA_SET_MODEL}")
    if not os.path.exists(os.path.join("model",f"{DATA_SET_MODEL}")):
        os.makedirs(os.path.join("model",f"{DATA_SET_MODEL}"))
    if not os.path.exists(os.path.join("images",f"{DATA_SET_MODEL}")):
        os.makedirs(os.path.join("images",f"{DATA_SET_MODEL}"))

    """准备数据"""
    train_dataset = None
    validation_dataset = None
    '''针对ImageNet定义Transform'''
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),  # 转化为张量
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if DATA_SET_MODEL == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))  # 进行归一化
                ])
            )

        validation_dataset = torchvision.datasets.CIFAR100(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                ])
            )
    elif DATA_SET_MODEL=='MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))

        validation_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
    elif DATA_SET_MODEL == 'ImageNet':
        train_dataset = ImageNet(
            dir = "data/mini-imagenet", is_train = True,
            json_path = "/home/levi/workplace/PyTorch-ViT-Vision-Transformer-main/data/mini-imagenet/classes_name.json",
            transform= train_transforms
        )

        validation_dataset = ImageNet(
            dir="data/mini-imagenet", is_train=False,
            json_path="/home/levi/workplace/PyTorch-ViT-Vision-Transformer-main/data/mini-imagenet/classes_name.json",
            transform=  val_transforms
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=True,
        pin_memory=True)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        pin_memory=True)

    logger.write(f"train data total {len(train_dataset)}")
    logger.write(f"val data total {len(validation_dataset)}")
    print(f"train data total {len(train_dataset)}")
    print(f"val data total {len(validation_dataset)}")
    # x,y = train_dataset[0]
    # print(x.shape)
    # plt.imshow(x)
    # plt.show()

    logger.write("load model ...")
    print("load model ...")

    """模型准备"""
    model = ViT(in_channels=In_channels,patch_size=Patch_size,img_size=Img_size,num_class=Num_class,depth=Depth).to(DEVICE)
    logger.write(f"In_channels={In_channels},Img_size={Img_size},Patch_size = {Patch_size},Depth = {Depth} ,Num_class={Num_class},Drop_out={Drop_out}")
    print(f"In_channels={In_channels},Img_size={Img_size},Patch_size = {Patch_size},Depth = {Depth} ,Num_class={Num_class},Drop_out={Drop_out}")
    loss_function = nn.CrossEntropyLoss()

    def weight_init(m):
        for m in m.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_uniform_(m.weight.data)
                # torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    model.apply(weight_init)
    #optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(
        # [paras for paras in model.parameters() if paras.requires_grad is True],
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=0.001
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=LR_d,gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[150,200],gamma=0.1)

    logger.write(f"optimizer:{optimizer.state_dict()}")
    logger.write(f"scheduler:{scheduler.state_dict()}")
    logger.write(f"in_channels={In_channels},img_size={Img_size},num_class={Num_class}")
    logger.write("start training ...")
    print(f"start training ... on {DEVICE}")
    logger.write(f"start training ... on {DEVICE}")

    train(model, loss_function, optimizer, scheduler, train_loader, validation_loader)
