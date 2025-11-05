from torch import nn, Tensor
from einops.layers.torch import Rearrange
from layers import ConvLeakyRelu2d
import torch

class Discriminator(nn.Module):
    """
    Use to discriminate fused images and source images.
    """
    # dim 是 图片的维度， size 是图片大小
    def __init__(self, dim: int = 64, size: tuple = (256, 256),patch_height=32, patch_width=32):
        super(Discriminator, self).__init__()

        self.toPatch = Rearrange('b c (h p1) (w p2) -> b (c h w) p1 p2 ', p1=patch_height, p2=patch_width)
        self.conv = nn.Sequential(
            ConvLeakyRelu2d(dim, dim*2, kernel_size=3, stride=2, padding=1, norm='Batch', activation='LReLU'),
            ConvLeakyRelu2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1, norm='Batch', activation='LReLU'),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim*4, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

        # self.linear1 = nn.Linear((size[0] // 8) * (size[1] // 8) * 128, 128)
        # self.linear2 = nn.Linear(128, 1)

    def forward(self, vis, ir, fu):
        # 鉴别器在test的时候不用，所以无需考虑test
        vis_patch = self.toPatch(vis) # b c (h p1) (w p2) -> b (c h w) p1 p2
        ir_patch = self.toPatch(ir) # 16,64,32,32
        fu_patch = self.toPatch(fu)

        # 求vis和ir每个patch的方差
        vis_var = torch.var(vis_patch, dim=(2, 3), keepdim=True)
        ir_var = torch.var(ir_patch, dim=(2, 3), keepdim=True) # 16,64,1,1
        var = vis_patch * (vis_var>=ir_var) + ir_patch * (ir_var>vis_var)

        # 鉴别
        real = self.conv(var) # b, 1
        fake = self.conv(fu_patch)

        return real, fake
