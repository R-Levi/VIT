import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

class MHA(nn.Module):
    def __init__(self, num_embeddings=65, len_embedding=256, num_heads=8):
        super(MHA, self).__init__()

        self.len_embedding = len_embedding
        self.num_embeddings = num_embeddings
        self.num_heads = num_heads

        assert (len_embedding % num_heads == 0)
        self.len_head = len_embedding // num_heads

        self.WK = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )
        self.WQ = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )
        self.WV = nn.Conv1d(
            in_channels=self.len_embedding,
            out_channels=num_heads * self.len_head,
            kernel_size=1
        )

        self.WZ = nn.Conv1d(
            in_channels=num_heads * self.len_head,
            out_channels=len_embedding,
            kernel_size=1
        )
        
    def forward(self, inp):
        inp = inp.swapaxes(1, 2)
        K = self.WK(inp).T.reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        Q = self.WQ(inp).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()
        V = self.WV(inp).reshape(self.num_heads, self.num_embeddings, self.len_head).squeeze()

        score = Q.bmm(K.transpose(dim0=1, dim1=2))

        indexes = torch.softmax(score / self.len_head, dim=2)
        
        Z = indexes.bmm(V)
        Z = Z.moveaxis(1, 2)
        Z = Z.flatten(start_dim=0, end_dim=1)
        Z = Z.unsqueeze(dim=0)

        output = self.WZ(Z)
        output = output.swapaxes(1, 2)
        return output


class Encoder(nn.Module):
    def __init__(self, num_embeddings=65, len_embedding=256, num_heads=8):
        super(Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.len_embedding = len_embedding
        self.MHA = MHA(num_embeddings, len_embedding, num_heads)
        self.ff = nn.Linear(len_embedding, len_embedding)
        self.BatchNorm1d = nn.BatchNorm1d(self.num_embeddings)


    def forward(self, inp):
        output = self.BatchNorm1d(inp)
        skip = self.MHA(inp) + output
        output = self.BatchNorm1d(skip)
        output = self.ff(output) + skip

        return output


class ViT(nn.Module):
    def __init__(self, num_encoders=5, len_embedding=128, num_heads=8, patch_size=4, input_res=32, num_classes=100,input_channel=3):
        super(ViT, self).__init__()
        patches_per_dim = (input_res // patch_size) * (input_res // patch_size)
        self.inp_channels = input_channel
        self.num_encoders = num_encoders
        self.positional_embedding = nn.Embedding(patches_per_dim + 1, len_embedding)
        self.cls_token = nn.Parameter(torch.rand(1, len_embedding))
        self.convolution_embedding = nn.Conv2d(
            in_channels=self.inp_channels,
            out_channels=len_embedding,
            kernel_size=patch_size,
            stride=patch_size)
        self.classification_head = nn.Linear(
            in_features=len_embedding,
            out_features=num_classes,
            bias=False)

        self.stack_of_encoders = nn.ModuleList()
        for i in range(num_encoders):
            self.stack_of_encoders.append(Encoder(patches_per_dim + 1, len_embedding, num_heads))

        self._init_weight()

    def forward(self, x):
        z = self.convolution_embedding(x)
        z = z.flatten(start_dim=2, end_dim=3).swapaxes(1, 2)
        z = torch.cat((self.cls_token, z.squeeze()))
        for e in range(len(z)):
            z[e] = z[e] + self.positional_embedding.weight[e]

        z = z.T.unsqueeze(dim=0)
        z = z.swapaxes(1, 2)

        for encoder in self.stack_of_encoders:
            z = encoder(z)

        z = z[:, 0, :]

        y_ = self.classification_head(z)
        return y_
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.01)
                # nn.init.normal_(m.weight.data,0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT()
    print(model)
    model.to(DEVICE)
    print(next(model.parameters()).device)
    inp = torch.rand((3, 3, 32, 32))
    res = model(inp.to(DEVICE))
    print(res.shape)
