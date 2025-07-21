import torch
import torch.nn as nn
from geoseg.models.ResNet import ResNet101
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
class AuxHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x
class Block(nn.Module): # MFAB  (CCAM+D_FNN)
    def __init__(self, dim=512, num_heads=16, mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CCAM(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = D_FNN(in_channels=dim, out_channels=dim)
    def forward(self, x):
        x = x + self.drop_path(self.norm1(self.attn(x))) #torch.Size([1, 512, 16, 16])
        x = x + self.drop_path(self.mlp(x)) # torch.Size([1, 512, 16, 16])
        return x
class BoundaryEnhancementNet(nn.Module):    # BFR





class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=(256, 512, 1024, 2048),
                 decode_channels=64,
                 dilation = [[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 edge_pooling = [[3,5],[5,7]],
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()
        self.edge4 = BoundaryEnhancementNet(decode_channels,decode_channels,edge_pooling[0])
        self.edge3 = BoundaryEnhancementNet(decode_channels, decode_channels, edge_pooling[0])
        self.edge2 = BoundaryEnhancementNet(decode_channels, decode_channels, edge_pooling[1])
        self.edge1 = BoundaryEnhancementNet(decode_channels, decode_channels, edge_pooling[1])
        self.Conv4 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.Conv1 = ConvBN(encode_channels[-4], decode_channels, 1)
        self.b4 = Block(dim=decode_channels, num_heads=16)
        self.p3 = Fusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])
        self.p2 = Fusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])
        self.p1 = Fusion(decode_channels)
        self.b1 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[3])
        # self.seg_head = MAF(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)
        self.seg_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.Conv1_1 =Conv(2, 512, 1)
        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

            self.upE4 = nn.UpsamplingBilinear2d(scale_factor=1)
            self.upE3 = nn.UpsamplingBilinear2d(scale_factor=1)

            self.aux_head = AuxHead(decode_channels, num_classes)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            res4 = self.Conv4(res4)
            res3 = self.Conv2(res3)
            res1 = self.Conv1(res1)
            res4 = self.edge4(res4)
            res3 = self.edge4(res3)
            res2 = self.edge4(res2)
            res1 = self.edge4(res1)
            x = self.b4(res4)
            h4 = self.up4(x)
            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x =self.b1(x) # torch.Size([1, 512, 128, 128])
            x = self.seg_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)
            return x, ah
        else:
            res4 = self.Conv4(res4)  # torch.Size([1, 512, 16, 16])
            res3 = self.Conv2(res3)  # torch.Size([1, 512, 32, 32])
            res1 = self.Conv1(res1)
            res4 = self.edge4(res4)
            res3 = self.edge4(res3)
            res2 = self.edge4(res2)
            res1 = self.edge4(res1)
            x = self.b4(res4)
            x = self.p3(x, res3)
            x = self.b3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            x = self.p1(x, res1)
            x = self.b1(x)
            x = self.seg_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            return x
class MNet(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.1,
                 num_classes=6,
                 backbone=ResNet101
                 ):
        super().__init__()
        self.backbone = backbone()
        self.decoder = Decoder(encode_channels, decode_channels, dropout=dropout, num_classes=num_classes)
    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
