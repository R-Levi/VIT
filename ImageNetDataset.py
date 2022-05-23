import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import json
from PIL import Image

class ImageNet(Dataset):
    def __init__(self,dir='data/mini-imagenet',is_train=True,json_path = "/home/levi/workplace/PyTorch-ViT-Vision-Transformer-main/data/mini-imagenet/classes_name.json",transform=None):
        super().__init__()
        self.root = dir
        self.csv_dir = os.path.join(self.root,'new_train.csv' if is_train else 'new_val.csv' )
        self.data_file = []
        self.label = []
        self.json_path = json_path
        self.transform = transform

        data = pd.read_csv(self.csv_dir)
        json_data = json.load(open(self.json_path, "r"))
        json_data = dict([(k,v[0]) for k, v in json_data.items()])

        file_name = data['filename']
        label = data['label']
        for name in file_name:
            self.data_file.append(os.path.join(self.root,os.path.join("images",name)))
        for _label in label:
            self.label.append(list(json_data).index(_label))

    def __getitem__(self, index):
        img_path = self.data_file[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        return image,label
    def __len__(self):
        return len(self.data_file)


if __name__ == '__main__':

    # data = pd.read_csv("data/mini-imagenet/new_train.csv")
    # print(data['filename'])
    # print(data['label'])

    json_path = "/home/levi/workplace/PyTorch-ViT-Vision-Transformer-main/data/mini-imagenet/classes_name.json"  # 指向imagenet的索引标签文件
    # # load imagenet labels
    # label_dict = json.load(open(json_path, "r"))
    # label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])
    # print(label_dict)
    # print(list(label_dict).index('n01440764'))
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.RandomResizedCrop(224)
    ])
    train_dataset = ImageNet(is_train=False,json_path=json_path,transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=1,
        shuffle=True,
        pin_memory=True)
    for i,(img,label) in enumerate(train_loader):
        print(img[0])
        image = transforms.ToPILImage()(img[0])
        print(image)
        image.show()
        print(label[0])
        break

