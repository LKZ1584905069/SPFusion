import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from layers import ConvLeakyRelu2d
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# 每个Transformer有depth层
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        # 图像和图像块的宽高
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 总共有多少 patch
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # 每个patch的维度 L=C*H*W
        patch_dim = channels * patch_height * patch_width  # patch_dim=1024

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        # embedding
        self.vi_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.ir_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 修改位置编码,这个位置我们希望是给不同的块一个不同相加的权重
        self.pos_vi_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 1, ,1024
        self.pos_ir_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 这个cls_token 应该不要
        self.dropout = nn.Dropout(emb_dropout)

        # 标准的Transformer 块
        self.vi_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.ir_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 变回特征图 16,16,1024 -- 16,1,H,W
        self.to_fea = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width,
                                w=image_width // patch_width, h=image_height // patch_height)

    def forward(self, vis, ir, thread=0.5):
        b, c, h, w = vis.shape
        # 将两张图变成 patch  b,b,h,w-----b c (h p1) (w p2) -> b (h w) (p1 p2 c)
        patch_vi = self.to_patch(vis)  # 16,16,1024
        patch_ir = self.to_patch(ir)
        # 获取 IR 图像每个 pathc 的均值
        mean_patch_ir = patch_ir.mean(dim=2)  # 16,16
        # 要选择的irpatch块
        choose_ir_patch = mean_patch_ir > thread  # 16,16
        choose_ir_patch = choose_ir_patch.unsqueeze(-1)
        # 重新组装块
        new_vi_patch = patch_vi * (~choose_ir_patch) + patch_ir * choose_ir_patch  # 16,16,1024
        new_ir_patch = patch_vi * choose_ir_patch + patch_ir * (~choose_ir_patch)

        # embedding
        embeding_vi = self.vi_embedding(new_vi_patch)
        embeding_ir = self.ir_embedding(new_ir_patch)



        # pos
        # pos_vi = embeding_vi + self.pos_vi_embedding
        # pos_ir = embeding_ir + self.pos_ir_embedding
        # pos
        pos_vi = embeding_vi
        pos_ir = embeding_ir

        # dropout
        drop_vi = self.dropout(pos_vi)
        drop_ir = self.dropout(pos_ir)

        # Transformer 这里只用了一层
        trans_vi = self.vi_transformer(drop_vi)  # 16,16,1024
        trans_ir = self.ir_transformer(drop_ir)

        # print(trans_vi.shape)
        # 变换回来
        fea_vi = self.to_fea(trans_vi)
        fea_ir = self.to_fea(trans_ir)

        return fea_vi, fea_ir



class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()
        self.conv1_1 = ConvLeakyRelu2d(in_channels=in_planes, out_channels= in_planes//4, kernel_size=3, padding=1,
                                       activation='LReLU')
        self.conv1_2 = ConvLeakyRelu2d(in_channels=in_planes // 4, out_channels= 1, kernel_size=3, padding=1,
                                       activation='Sigmoid')

        self.conv2_1 = ConvLeakyRelu2d(in_channels=in_planes, out_channels=in_planes // 4, kernel_size=5, padding=2,
                                     activation='LReLU')
        self.conv2_2 = ConvLeakyRelu2d(in_channels=in_planes // 4, out_channels=1, kernel_size=5, padding=2,
                                       activation='Sigmoid')
        self.conv3_1 = ConvLeakyRelu2d(in_channels=in_planes, out_channels=in_planes // 4, kernel_size=7, padding=3,
                                       activation='LReLU')
        self.conv3_2 = ConvLeakyRelu2d(in_channels=in_planes // 4, out_channels=1, kernel_size=7, padding=3,
                                       activation='Sigmoid')

        self.conv4_1 = ConvLeakyRelu2d(in_channels=in_planes, out_channels=in_planes // 4, kernel_size=9, padding=4,
                                       activation='LReLU')
        self.conv4_2 = ConvLeakyRelu2d(in_channels=in_planes // 4, out_channels=1, kernel_size=9, padding=4,
                                       activation='Sigmoid')


    def forward(self, x):
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)

        conv2 = self.conv2_1(x)
        conv2 = self.conv2_2(conv2)

        conv3 = self.conv3_1(x)
        conv3 = self.conv3_2(conv3)

        conv4 = self.conv4_1(x)
        conv4 = self.conv4_2(conv4)

        out = (conv1 + conv2 + conv3 + conv4) / 4.
        return out


# 通道注意力机制采用（混合 - IR）作为VIS的通道注意力特征 * VIS，同理 IR 也是
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


'''
    使用两个卷积做成一对
'''


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # VIS 编码器
        self.vis1 = nn.Sequential(ConvLeakyRelu2d(1, 16, activation='PReLU'),
                                  ConvLeakyRelu2d(16, 16, norm='Batch', activation='PReLU'))
        self.vis2 = nn.Sequential(ConvLeakyRelu2d(16, 16, activation='PReLU'),
                                  ConvLeakyRelu2d(16, 16, norm='Batch', activation='PReLU'))
        self.vis3 = nn.Sequential(ConvLeakyRelu2d(32, 32, activation='PReLU'),
                                  ConvLeakyRelu2d(32, 16, norm='Batch', activation='PReLU'))
        self.vis4 = nn.Sequential(ConvLeakyRelu2d(48, 48, activation='PReLU'),
                                  ConvLeakyRelu2d(48, 16, norm='Batch', activation='PReLU'))

        # IR 编码器
        self.ir1 = nn.Sequential(ConvLeakyRelu2d(1, 16, activation='PReLU'),
                                 ConvLeakyRelu2d(16, 16, norm='Batch', activation='PReLU'))
        self.ir2 = nn.Sequential(ConvLeakyRelu2d(16, 16, activation='PReLU'),
                                 ConvLeakyRelu2d(16, 16, norm='Batch', activation='PReLU'))
        self.ir3 = nn.Sequential(ConvLeakyRelu2d(32, 32, activation='PReLU'),
                                 ConvLeakyRelu2d(32, 16, norm='Batch', activation='PReLU'))
        self.ir4 = nn.Sequential(ConvLeakyRelu2d(48, 48, activation='PReLU'),
                                 ConvLeakyRelu2d(48, 16, norm='Batch', activation='PReLU'))

    def forward(self, vis, ir):
        vis1 = self.vis1(vis)
        vis2 = self.vis2(vis1)
        vis3 = self.vis3(torch.cat((vis1, vis2), 1))
        vis4 = self.vis4(torch.cat((vis1, vis2, vis3), 1))
        ir1 = self.ir1(ir)
        ir2 = self.ir2(ir1)
        ir3 = self.ir3(torch.cat((ir1, ir2), 1))
        ir4 = self.ir4(torch.cat((ir1, ir2, ir3), 1))

        return vis4, ir4


# 局部注意力模块
class LocalAttenMocel(nn.Module):
    def __init__(self):
        super(LocalAttenMocel, self).__init__()
        self.sam_vi = SpatialAttention(in_planes=16)
        self.cam_vi = ChannelAttention(in_planes=16)
        self.sam_ir = SpatialAttention(in_planes=16)
        self.cam_ir = ChannelAttention(in_planes=16)

    def forward(self, vis5, ir5):
        
        vi_cam = self.cam_vi(vis5) * vis5
        vi_sam = self.sam_vi(vi_cam) * vi_cam

        ir_cam = self.cam_ir(ir5) * ir5
        ir_sam = self.sam_ir(ir_cam) * ir_cam
        
        return vi_sam, ir_sam


# 解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.de2 = nn.Sequential(ConvLeakyRelu2d(64, 32, norm='Batch', activation='PReLU'),
                                 ConvLeakyRelu2d(32, 16, norm='Batch', activation='PReLU'))

        self.de4 = ConvLeakyRelu2d(16, 1, activation='Tanh')

    def forward(self, x):
        de1 = self.de2(x)
        de4 = self.de4(de1)

        return de4


# Encoder 和 ViT 都是同时包含了 VIS 和 IR
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Encoder 得到两个图像的编码特征图
        self.Encoder = Encoder()
        # 得到两个图像的局部注意力模块
        self.LAM = LocalAttenMocel()
        # 得到两个图像的全局注意力模块
        self.ViT = ViT(
            image_size=256,  # 图像大小
            patch_size=32,  # patch大小（分块的大小）
            dim=1024,  # position embedding的维度  dim = patch_size * patch_size
            depth=6,  # encoder和decoder中block层数是6
            heads=16,  # multi-head中head的数量为16
            mlp_dim=2048,
            dropout=0,  #
            emb_dropout=0.1
        )
        # Decoder 解码器
        self.Decoder = Decoder()

        # 初始化四个可学习的参数，初始默认为 1
        self.gamma1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma4 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, vis, ir, is_test=False):
        b, v, h, w = vis.shape

        # max 和 mean 用于LAM
        max = torch.max(vis,ir)
        mean = vis/2. + ir/2.

        # 得到特征图
        fea_vi, fea_ir = self.Encoder(vis=vis, ir=ir)  # 16,16,256,256
        # 得到局部注意力模块
        lam_vi, lam_ir = self.LAM(vis5=fea_vi - max, ir5=fea_ir-mean)  # 16,16,256,256
        # 得到全局注意力
        # 判断是否是测试阶段
        if is_test:
            vis = F.interpolate(vis, size=(256, 256), scale_factor=None, mode='bicubic', align_corners=None)
            ir = F.interpolate(ir, size=(256, 256), scale_factor=None, mode='bicubic', align_corners=None)
            gam_att_vi, gam_att_ir = self.ViT(vis=vis, ir=ir)
            gam_att_vi = F.interpolate(gam_att_vi, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None)
            gam_att_ir = F.interpolate(gam_att_ir, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None)
            gam_vi = gam_att_vi * fea_vi + fea_vi  # 16,16,256,256
            gam_ir = gam_att_ir * fea_ir + fea_ir  # 16,16,256,256
            # 解码
            fusion = self.Decoder(torch.cat((self.gamma1 * lam_vi, self.gamma2 * lam_ir,
                                             self.gamma3 * gam_vi, self.gamma4 * gam_ir), dim=1))

            return fusion
        else:
            gam_att_vi, gam_att_ir = self.ViT(vis=vis, ir=ir)  # 16,1,256,256
            # 全局注意力图乘以特征图
            gam_vi = gam_att_vi * fea_vi + fea_vi  # 16,16,256,256
            gam_ir = gam_att_ir * fea_ir + fea_ir  # 16,16,256,256
            # 解码
            fusion = self.Decoder(torch.cat((self.gamma1 * lam_vi, self.gamma2 * lam_ir,
                                             self.gamma3 * gam_vi, self.gamma4 * gam_ir), dim=1))

        return fusion








