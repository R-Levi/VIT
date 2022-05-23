"""
implement VIT
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self,in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,emb_size,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.postions = nn.Parameter(torch.randn((img_size//patch_size)**2+1,emb_size))

    def forward(self,x):
        b,_,_,_ = x.shape
        x = self.projection(x)#b,(h,w),len_e
        cls_token = repeat(self.cls_token,'() n e -> b n e',b=b)
        # print(cls_token.shape)
        # print(x.shape)
        x = torch.cat([cls_token,x],dim=1)#32 65 128
        # print(self.postions.shape)
        x+=self.postions
        return x

class MHA(nn.Module):
    def __init__(self,emb_size=128,num_heads=8,dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size,emb_size)

    def forward(self,x,mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print("queries.shape",queries.shape)
        # print("keys.shape",keys.shape)
        # print("values.shape",values.shape)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # print("att2' shape: ", att.shape)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        # print("out2's shape: ", out.shape)
        out = self.projection(out)
        # print("out3's shape: ", out.shape)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self,emb_size,expand,drop_out):
        super().__init__(
            nn.Linear(emb_size,emb_size*expand),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(emb_size*expand,emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,emb_size=128,ff_expand=4,ff_expand_drop=0,drop_out=0,**kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MHA(emb_size,**kwargs),
                nn.Dropout(drop_out)

            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size,ff_expand,ff_expand_drop),
                nn.Dropout(drop_out)
            )),
        )

class TransformerEnconder(nn.Sequential):
    def __init__(self,depth=12,**kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )

class ClassficationHead(nn.Sequential):
    def __init__(self,emb_size=128,num_class=10):
        super().__init__(
            Reduce('b n e -> b e',reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size,num_class)
        )

class ViT(nn.Sequential):
    def __init__(self,in_channels=1,patch_size=4,emb_size=128,img_size=28,depth=4,num_class=10,**kwargs):
        super().__init__(
            PatchEmbedding(in_channels,patch_size,emb_size,img_size),
            TransformerEnconder(depth,emb_size=emb_size,**kwargs),
            ClassficationHead(emb_size,num_class)
        )

if __name__ == '__main__':
    x = torch.randn(32,3,32,32)
    # out = PatchEmbedding()(x)
    # out = MHA()(x=out,mask=None)
    # print(out.shape)
    print(summary(ViT(), (3, 32, 32), device='cpu'))
    model = ViT(emb_size=48)
    print(model)
    out = model(x)
    print(out.shape)