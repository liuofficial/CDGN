from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange
import numpy as np
import math
# from deconv import DEConv, GradConv
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time


def FFT(fea):
    fft = torch.fft.fft2(fea, dim=(-2, -1))
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    return amplitude, phase


def IFFT(A, P):
    ifft = A * torch.exp(1j * P)
    ifft = torch.fft.ifft2(ifft, dim=(-2, -1)).real
    return ifft


class CRC(nn.Module):
    def __init__(self, in_c, out_c, ks=3):
        super(CRC, self).__init__()
        self.c1 = nn.Conv2d(in_c, out_c, ks, 1, ks // 2)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(out_c, out_c, ks, 1, ks // 2)

    def forward(self, x):
        return self.c2(self.relu(self.c1(x))) + x


class CALayer(nn.Module):
    def __init__(self, in_c, ratio=4):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_c, in_c // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_c // ratio, in_c, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SALayer(nn.Module):
    def __init__(self, ks=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, ks, 1, ks // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RSAB(nn.Module):
    def __init__(self):
        super(RSAB, self).__init__()
        self.sa = SALayer()

    def forward(self, x):
        return x + self.sa(x) * x


class RCAB(nn.Module):
    def __init__(self, in_c):
        super(RCAB, self).__init__()
        self.ca = CALayer(in_c)

    def forward(self, x):
        return x + self.ca(x) * x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WSA(nn.Module):
    def __init__(self, hsi_c, ws=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.window_size = ws
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.W_q = nn.Sequential(nn.Conv2d(hsi_c, hsi_c, 1),
                                 nn.Conv2d(hsi_c, hsi_c, 3, 1, 1, groups=hsi_c))
        self.W_k = nn.Sequential(nn.Conv2d(hsi_c, hsi_c, 1),
                                 nn.Conv2d(hsi_c, hsi_c, 3, 1, 1, groups=hsi_c))
        self.W_v = nn.Sequential(nn.Conv2d(hsi_c, hsi_c, 1),
                                 nn.Conv2d(hsi_c, hsi_c, 3, 1, 1, groups=hsi_c))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hsi_c, hsi_c)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        h, w = X.shape[-2:]
        q = self.W_q(X)
        q = window_partition(q.permute(0, 2, 3, 1), self.window_size)
        q = rearrange(q, 'b h w c -> b (h w) c')
        k = self.W_k(X)
        k = window_partition(k.permute(0, 2, 3, 1), self.window_size)
        k = rearrange(k, 'b h w c -> b (h w) c')
        v = self.W_v(X)
        v = window_partition(v.permute(0, 2, 3, 1), self.window_size)
        v = rearrange(v, 'b h w c -> b (h w) c')
        attn = torch.einsum('b n c, b m c-> b n m', q, k)
        attn = torch.softmax(attn / math.sqrt(q.shape[-1]), dim=-1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias
        attn = self.attn_drop(attn)
        attn = self.softmax(attn)
        out = torch.einsum('b n m, b m c -> b n c', attn, v)
        out = self.proj_drop(self.proj(out))
        out = rearrange(out, 'b (h w) c -> b h w c', h=self.window_size, w=self.window_size)
        out = window_reverse(out, self.window_size, h, w).permute(0, 3, 1, 2)
        return out


class Mlp(nn.Module):
    def __init__(self, in_c, drop=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(in_c, in_c, 1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(in_c, in_c, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransBlock(nn.Module):
    def __init__(self, in_c, window_size=8, drop=0.1):
        super().__init__()
        self.wsa = WSA(in_c, window_size, attn_drop=drop, proj_drop=drop)
        self.mlp = Mlp(in_c, drop)

    def forward(self, X):
        B, C, H, W = X.shape
        Xln = torch.layer_norm(X, [C, H, W])
        fea = self.wsa(Xln) + X
        return fea


class TransBlock(nn.Module):
    def __init__(self, in_c, window_size=8, drop=0.1):
        super().__init__()
        self.wsa = WSA(in_c, window_size, attn_drop=drop, proj_drop=drop)
        self.mlp = Mlp(in_c, drop)

    def forward(self, X):
        B, C, H, W = X.shape
        Xln = torch.layer_norm(X, [C, H, W])
        fea = self.wsa(Xln) + X
        out = torch.layer_norm(fea, [C, H, W])
        out = self.mlp(out) + fea
        return out


class FSA(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.head = nn.Conv2d(in_c, in_c, 1)
        self.clcx = nn.Sequential(nn.Conv2d(in_c, in_c, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_c, in_c, 3, 1, 1))
        self.clcA = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_c, in_c, 1))
        self.clcP = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_c, in_c, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fea = self.head(x)
        out = self.clcx(fea)
        A, P = FFT(fea)
        A = A + self.clcA(A)
        P = P + self.clcP(P)
        scores = IFFT(A, P)
        out = out * self.sigmoid(scores)
        return out


class FreB(nn.Module):
    def __init__(self, in_c, drop=0.1):
        super().__init__()
        self.freb = FSA(in_c)
        self.mlp = Mlp(in_c, drop)

    def forward(self, X):
        B, C, H, W = X.shape
        Xln = torch.layer_norm(X, [C, H, W])
        fea = self.freb(Xln) + X
        out = torch.layer_norm(fea, [C, H, W])
        out = self.mlp(out) + fea
        return out


class HARM(nn.Module):
    def __init__(self, in_c, window_size=8, drop=0.1):
        super().__init__()
        self.head = nn.Conv2d(in_c, in_c, 1)
        self.tb1 = TransBlock(in_c, window_size, drop)
        self.fb1 = FreB(in_c, drop)
        self.cat1 = nn.Conv2d(in_c * 2, in_c, 1)
        self.cat2 = nn.Conv2d(in_c * 2, in_c, 1)
        self.tb2 = TransBlock(in_c, window_size, drop)
        self.fb2 = FreB(in_c, drop)
        self.tail = nn.Conv2d(in_c * 2, in_c, 1)

    def forward(self, X):
        fea = self.head(X)
        tsa1 = self.tb1(fea)
        fsa1 = self.fb1(fea)
        tsa2 = self.tb2(self.cat1(torch.cat([tsa1, fsa1], dim=1)))
        fsa2 = self.fb2(self.cat2(torch.cat([fsa1, tsa1], dim=1)))
        out = self.tail(torch.cat([tsa2, fsa2], dim=1))
        return out


class DAEN(nn.Module):
    def __init__(self, hsi_c=31, mid_c=48, factor=8):
        super(DAEN, self).__init__()
        self.up_spa = nn.Upsample(scale_factor=factor, mode='bilinear')
        self.up_spe = nn.Conv2d(hsi_c, mid_c, 1)
        self.down_spe = nn.Conv2d(mid_c, hsi_c, 1)

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.e1 = CRC(mid_c, mid_c, 3)
        self.e2 = CRC(mid_c, mid_c, 3)
        self.e3 = CRC(mid_c, mid_c, 3)
        self.e4 = CRC(mid_c, mid_c, 3)

        self.rsab1 = RSAB()
        self.rsab2 = RSAB()
        self.rsab3 = RSAB()
        self.rsab4 = RSAB()

        self.d4 = CRC(mid_c, mid_c, 3)
        self.up3 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d3 = CRC(mid_c, mid_c, 3)
        self.up2 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d2 = CRC(mid_c, mid_c, 3)
        self.up1 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d1 = CRC(mid_c, mid_c, 3)

        self.cat4 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat3 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat2 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat1 = nn.Conv2d(mid_c * 2, mid_c, 1)

    def forward(self, Y):
        Yup = self.up_spa(Y)
        Fy = self.up_spe(Yup)
        e1 = self.e1(Fy)
        e2 = self.e2(self.down(e1))
        e3 = self.e3(self.down(e2))
        e4 = self.e4(self.down(e3))

        d4 = self.d4(e4)
        da4 = self.cat4(torch.cat([d4, self.rsab4(e4)], dim=1))
        d3 = self.d3(self.up3(da4))
        da3 = self.cat3(torch.cat([d3, self.rsab3(e3)], dim=1))
        d2 = self.d2(self.up2(da3))
        da2 = self.cat2(torch.cat([d2, self.rsab2(e2)], dim=1))
        d1 = self.d1(self.up1(da2))
        da1 = self.cat1(torch.cat([d1, self.rsab1(e1)], dim=1))
        res = self.down_spe(da1)
        Xra = res + Yup
        Da = [da1, da2, da3, da4]
        return Xra, Da


class DEEN(nn.Module):
    def __init__(self, msi_c=3, hsi_c=31, mid_c=48):
        super(DEEN, self).__init__()
        self.msToHs = nn.Conv2d(msi_c, hsi_c, 1)
        self.up_spe = nn.Conv2d(hsi_c, mid_c, 1)
        self.down_spe = nn.Conv2d(mid_c, hsi_c, 1)

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.e1 = CRC(mid_c, mid_c, 1)
        self.e2 = CRC(mid_c, mid_c, 1)
        self.e3 = CRC(mid_c, mid_c, 1)
        self.e4 = CRC(mid_c, mid_c, 1)

        self.rcab1 = RCAB(mid_c)
        self.rcab2 = RCAB(mid_c)
        self.rcab3 = RCAB(mid_c)
        self.rcab4 = RCAB(mid_c)

        self.d4 = CRC(mid_c, mid_c, 1)
        self.up3 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d3 = CRC(mid_c, mid_c, 1)
        self.up2 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d2 = CRC(mid_c, mid_c, 1)
        self.up1 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=2, stride=2)
        self.d1 = CRC(mid_c, mid_c, 1)

        self.cat4 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat3 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat2 = nn.Conv2d(mid_c * 2, mid_c, 1)
        self.cat1 = nn.Conv2d(mid_c * 2, mid_c, 1)

    def forward(self, Z):
        Zup = self.msToHs(Z)
        Fz = self.up_spe(Zup)
        e1 = self.e1(Fz)
        e2 = self.e2(self.down(e1))
        e3 = self.e3(self.down(e2))
        e4 = self.e4(self.down(e3))

        d4 = self.d4(e4)
        de4 = self.cat4(torch.cat([d4, self.rcab4(e4)], dim=1))

        d3 = self.d3(self.up3(de4))
        de3 = self.cat3(torch.cat([d3, self.rcab3(e3)], dim=1))

        d2 = self.d2(self.up2(de3))
        de2 = self.cat2(torch.cat([d2, self.rcab2(e2)], dim=1))

        d1 = self.d1(self.up1(de2))
        de1 = self.cat1(torch.cat([d1, self.rcab1(e1)], dim=1))

        res = self.down_spe(de1)
        Xra = res + Zup
        De = [de1, de2, de3, de4]
        return Xra, De


class DGM(nn.Module):
    def __init__(self, in_c):
        super(DGM, self).__init__()
        self.sa = SALayer()
        self.cata1 = nn.Conv2d(in_c * 2, in_c, 1)
        self.cata2 = nn.Conv2d(in_c * 2, in_c, 1)

        self.ca = CALayer(in_c, ratio=8)
        self.cate1 = nn.Conv2d(in_c * 2, in_c, 1)
        self.cate2 = nn.Conv2d(in_c * 2, in_c, 1)

        self.cat = nn.Conv2d(in_c * 2, in_c, 1)

    def forward(self, X, Da, De):
        outa = torch.cat([self.cata1(torch.cat([X, Da], dim=1)), self.sa(Da) * X], dim=1)
        outa = self.cata2(outa)
        oute = torch.cat([self.cate1(torch.cat([X, De], dim=1)), self.ca(De) * X], dim=1)
        oute = self.cate2(oute)
        out = self.cat(torch.cat([outa, oute], dim=1))
        return out


class DGRN(nn.Module):
    def __init__(self, hsi_c, msi_c, factor=8, nf=48):
        super().__init__()
        self.spa_up = nn.Upsample(scale_factor=factor, mode='bilinear')
        self.catConv = nn.Sequential(nn.Conv2d(hsi_c + msi_c, nf, 1),
                                     nn.ReLU())

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.dgm1 = DGM(nf)
        self.dgm2 = DGM(nf)
        self.dgm3 = DGM(nf)
        self.dgm4 = DGM(nf)

        self.harm1 = HARM(nf)
        self.harm2 = HARM(nf)
        self.harm3 = HARM(nf)
        self.harm4 = HARM(nf)

        self.tail = nn.Conv2d(nf * 4, hsi_c, 1)

    def forward(self, Y, Z, Da, De):
        Yup = self.spa_up(Y)
        X = self.catConv(torch.cat([Yup, Z], dim=1))

        out1 = self.harm1(X)
        out1 = self.dgm1(out1, Da[0], De[0])

        out2 = self.harm2(self.down(out1))
        out2 = self.dgm2(out2, Da[1], De[1])
        fea2 = F.interpolate(out2, scale_factor=2, mode='bilinear')

        out3 = self.harm3(self.down(out2))
        out3 = self.dgm3(out3, Da[2], De[2])
        fea3 = F.interpolate(out3, scale_factor=4, mode='bilinear')

        out4 = self.harm4(self.down(out3))
        out4 = self.dgm4(out4, Da[3], De[3])
        fea4 = F.interpolate(out4, scale_factor=8, mode='bilinear')

        sr = self.tail(torch.cat([out1, fea2, fea3, fea4], dim=1)) + Yup
        sr = torch.clamp(sr, 0, 1)

        return sr


class X_Module(nn.Module):
    def __init__(self, hsi_c, msi_c, factor=8, nf=48):
        super().__init__()

        self.subNetSpa = DAEN(hsi_c, nf, factor)
        self.subNetSpe = DEEN(msi_c, hsi_c, nf)
        self.resNet = DGRN(hsi_c, msi_c, factor, nf)

        self.init_weights(self)

    def forward(self, Y, Z):
        Xra, Da = self.subNetSpa(Y)
        Xre, De = self.subNetSpe(Z)
        X = self.resNet(Y, Z, Da, De)
        return X, Xra, Xre,  Da, De

    def init_weights(model, init_type='normal'):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # print(name)
                # for name, parameters in m.named_parameters():
                #     print(name, ':', parameters.size())
                if init_type == 'normal':
                    nn.init.kaiming_normal_(m.weight.data)
                elif init_type == 'uniform':
                    nn.init.kaiming_uniform_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        pass


if __name__ == '__main__':
    X = torch.ones([1, 31, 64, 64]).cuda()
    Y = torch.ones([1, 31, 8, 8]).cuda()
    Z = torch.ones([1, 3, 64, 64]).cuda()
    net = X_Module(hsi_c=31, msi_c=3, factor=8, nf=64).cuda()
    start = time.perf_counter()
    sr = net(Y, Z)
    end = time.perf_counter()
    total_params = sum(p.numel() for p in net.parameters())
    print("total_params: {}, time: {}".format(total_params, end - start))
