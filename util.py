import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import random
import cv2
from scipy.ndimage import gaussian_filter
import torch
import scipy.interpolate as spi
from torch.autograd import Variable
import tensorly as tl
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn
from scipy.ndimage import rotate
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.util import view_as_blocks


# ---------- 基础函数 ----------
def uqi(x, y, eps=1e-12):
    """Universal Quality Index (UQI)"""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    mx, my = x.mean(), y.mean()
    C = np.cov(x, y)
    num = 4 * C[0, 1] * mx * my
    den = (C[0, 0] + C[1, 1]) * (mx**2 + my**2)
    return 0.0 if den < eps else num / den

def block_uqi(img1, img2, block_size=32):
    """按块计算 UQI 平均值"""
    assert img1.shape == img2.shape, "Images must have the same size"
    N, M = img1.shape
    N = (N // block_size) * block_size
    M = (M // block_size) * block_size
    img1, img2 = img1[:N, :M], img2[:N, :M]
    blocks1 = view_as_blocks(img1, (block_size, block_size))
    blocks2 = view_as_blocks(img2, (block_size, block_size))
    Qmap = np.zeros(blocks1.shape[:2])
    for i in range(Qmap.shape[0]):
        for j in range(Qmap.shape[1]):
            Qmap[i, j] = uqi(blocks1[i, j], blocks2[i, j])
    return np.mean(Qmap)

def blur_down(img, sigma):
    """用高斯模糊模拟 MTF 降采样"""
    return gaussian_filter(img.astype(np.float64), sigma=sigma)

def resize_LRHS(I_LRHS, target_shape):
    """将 LRHS 上采样到 HRHS 尺度"""
    ratio_h = target_shape[0] / I_LRHS.shape[0]
    ratio_w = target_shape[1] / I_LRHS.shape[1]
    I_up = np.zeros(target_shape, dtype=np.float64)
    for b in range(I_LRHS.shape[2]):
        I_up[:, :, b] = zoom(I_LRHS[:, :, b], (ratio_h, ratio_w), order=3)
    return I_up

# ---------- 光谱失真 D_lambda ----------
def D_lambda(I_F, I_LRHS, p=1):
    H, W, B = I_F.shape
    sum_val = 0.0
    for i in range(B):
        for j in range(i + 1, B):
            Q_F = block_uqi(I_F[:, :, i], I_F[:, :, j])
            Q_L = block_uqi(I_LRHS[:, :, i], I_LRHS[:, :, j])
            sum_val += abs(Q_F - Q_L) ** p
    return (2 * sum_val / (B * (B - 1))) ** (1 / p)

# ---------- 空间失真 D_s ----------
def D_s(I_F, I_LRHS, I_HRMS, blur_sigma=2.0, block_size=32, q=1):
    H, W, B = I_F.shape
    if I_HRMS.ndim == 3:
        I_HRMS_gray = np.mean(I_HRMS, axis=2)
    else:
        I_HRMS_gray = I_HRMS
    HRMS_blur = blur_down(I_HRMS_gray, sigma=blur_sigma)

    D_s_val = 0.0
    # print(I_F.shape, I_HRMS_gray.shape)
    for b in range(B):
        Q_high = block_uqi(I_F[:, :, b], I_HRMS_gray, block_size)
        Q_low = block_uqi(I_LRHS[:, :, b], HRMS_blur, block_size)
        D_s_val += abs(Q_high - Q_low) ** q
    return (D_s_val / B) ** (1 / q)

# ---------- 综合指标 QNR ----------
def QNR(I_F, I_LRHS, I_HRMS, p=1, q=1, blur_sigma=2.0, block_size=32):
    # 如果 LRHS 尺度比 HRHS 小，先上采样
    if I_LRHS.shape[:2] != I_F.shape[:2]:
        I_LRHS_up = resize_LRHS(I_LRHS, I_F.shape)
    else:
        I_LRHS_up = I_LRHS

    D_lambda_val = D_lambda(I_F, I_LRHS_up, p)
    D_s_val = D_s(I_F, I_LRHS_up, I_HRMS, blur_sigma, block_size, q)
    QNR_val = (1 - D_lambda_val) * (1 - D_s_val)
    return D_lambda_val, D_s_val, QNR_val


def FFT(fea):
    fft = torch.fft.fft2(fea, dim=(-2, -1))
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    return amplitude, phase

def IFFT(A, P):
    ifft = A * torch.exp(1j * P)
    ifft = torch.fft.ifft2(ifft, dim=(-2, -1)).real
    return ifft


def dot(m1, m2):
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def CC(reference, target):
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')
        target_temp = target[:, :, i].flatten(order='F')
        cc = np.corrcoef(ref_temp, target_temp)
        out[i] = cc[0, 1]
    return np.mean(out)


def SAM(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 1 / (2 ** 52)  # 浮点精度
    prod_scal = dot(reference, target)  # 取各通道相同位置组成的向量进行内积运算
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)  # 二范数乘积矩阵
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps  # 除法避免除数为0
    map = np.arccos(prod_scal / prod_map)  # 求得映射矩阵
    prod_scal = np.reshape(prod_scal, [pixels, 1])
    prod_norm = np.reshape(prod_norm, [pixels, 1])
    z = np.argwhere(prod_norm == 0)[:, 0]  # 求得prod_norm中为0位置的行号向量
    # 去除这些行，方便后续进行点除运算
    prod_scal = np.delete(prod_scal, z, axis=0)
    prod_norm = np.delete(prod_norm, z, axis=0)
    # 求取平均光谱角度
    angolo = np.sum(np.arccos(np.clip(prod_scal / prod_norm, -1, 1))) / prod_scal.shape[0]
    # angolo = np.sum(np.arccos(prod_scal / prod_norm)) / prod_scal.shape[0]

    # 转换为度数
    angle_sam = np.real(angolo) * 180 / np.pi

    return angle_sam, map


def SSIM(reference, target):
    rows, cols, bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += SSIM_BAND(reference[:, :, i], target[:, :, i])
    mssim /= bands
    return mssim


def SSIM_BAND(reference, target):
    return compare_ssim(reference, target, data_range=1.0)


def PSNR(reference, target):
    max_pixel = 1.0
    return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))


def RMSE(reference, target):
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def ERGAS(references, target, ratio):
    rows, cols, bands = references.shape
    d = 1 / ratio
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels
    uk = np.mean(tar_temp, axis=0)
    relative_rmse2 = rmse2 / uk ** 2
    total_relative_rmse = np.sum(relative_rmse2)
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)
    return out

class Loss_SAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label, output):
        b, c, h, w = label.shape
        x_norm = torch.sqrt(torch.sum(torch.square(label), dim=1))
        y_norm = torch.sqrt(torch.sum(torch.square(output), dim=1))
        xy_norm = torch.multiply(x_norm, y_norm)
        xy = torch.sum(torch.multiply(label, output), dim=1)
        dist = torch.mean(torch.arccos(torch.minimum(torch.divide(xy, xy_norm + 1e-8), torch.tensor(1.0 - 1.0e-9))), dim=[1, 2])
        dist = torch.multiply(torch.tensor(180.0 / np.pi), dist)
        dist = torch.mean(dist)
        return dist


def gauss_kernel(row_size, col_size, sigma):
    kernel = cv2.getGaussianKernel(row_size, sigma)
    kernel = kernel * cv2.getGaussianKernel(col_size, sigma).T
    return kernel

def anisotropic_gaussian_kernel(size, sigmaX, sigmaY, angle_degrees):
    center = size // 2
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1 / (2 * np.pi * sigmaX * sigmaY)) * np.exp(-((x - center)**2 / (2 * sigmaX**2) + (y - center)**2 / (2 * sigmaY**2)))
    kernel /= np.sum(kernel)

    # 旋转核
    rotated_kernel = rotate(kernel, angle_degrees, reshape=False, mode='constant', cval=0.0)
    return rotated_kernel

def intersect(list1, list2):
    list1 = list(list1)
    elem = list(set(list1).intersection(set(list2)))
    elem.sort()
    res = np.zeros(len(elem))
    for i in range(0, len(elem)):
        res[i] = list1.index(elem[i])
    res = res.astype("int32")
    return res

def create_spec_resp(data_num, genPath):
    if data_num == 0:   # CAVE
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(400, 700 + 1, 10)
        spec_resp = mat['spec_resp']
        R = spec_resp[spec_rng - 377, 1:4].T
    if data_num == 1:  # harvard
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(420, 720 + 1, 10)
        spec_resp = mat['spec_resp']
        R = spec_resp[spec_rng - 377, 1:4].T
    if data_num == 2:
        band = 102
        file = os.path.join(genPath, 'srf/ikonos.mat')  # 350 : 5 : 1035
        mat = sio.loadmat(file)
        spec_rng = np.arange(430, 861)
        spec_resp = mat['spec_resp']
        ms_bands = range(1, 5)
        valid_ik_bands = intersect(spec_resp[:, 0], spec_rng)
        no_wa = len(valid_ik_bands)
        # Spline interpolation
        xx = np.linspace(1, no_wa, band)
        x = range(1, no_wa + 1)
        R = np.zeros([5, band])
        for i in range(0, 5):
            ipo3 = spi.splrep(x, spec_resp[valid_ik_bands, i + 1], k=3)
            R[i, :] = spi.splev(xx, ipo3)
        R = R[ms_bands, :]
    if data_num == 3:
        # Chikusei  128 X 4
        band = 128
        file = os.path.join(genPath, 'srf/ikonos.mat')
        mat = sio.loadmat(file)
        spec_rng = np.arange(375, 1015, 5)
        spec_resp = mat['spec_resp']
        R = spec_resp[(spec_rng - 350) // 5, 2:6].T
    c = 1 / np.sum(R, axis=1)
    R = np.multiply(R, c.reshape([c.size, 1]))
    return R

def create_hrms_lrhs(hs, B, R, ratio, hs_snr, ms_snr, noise=True):
    hrms = tl.tenalg.mode_dot(hs, R, mode=2)
    # add noise for ms
    ms_sig = (np.sum(np.power(hrms.flatten(), 2)) / (10 ** (ms_snr / 10)) / hrms.size) ** 0.5
    np.random.seed(1)
    if noise is True:
        hrms = np.add(hrms, ms_sig * np.random.randn(hrms.shape[0], hrms.shape[1], hrms.shape[2]))
    # blur
    lrhs = cv2.filter2D(hs, -1, B, borderType=cv2.BORDER_REFLECT)
    # add noise for hs
    hs_sig = (np.sum(np.power(lrhs.flatten(), 2)) / (10 ** (hs_snr / 10)) / lrhs.size) ** 0.5
    np.random.seed(0)
    if noise is True:
        lrhs = np.add(lrhs, hs_sig * np.random.randn(lrhs.shape[0], lrhs.shape[1], lrhs.shape[2]))
    # down sampling
    lrhs = lrhs[0::ratio, 0::ratio, :]
    

    return hrms, lrhs

def create_hrms_lrhs_inte(hs, B, R, ratio, hs_snr, ms_snr, noise=True, down_type=0):
    # 1. 生成 HRMS（光谱降采样）
    hrms = tl.tenalg.mode_dot(hs, R, mode=2)
    
    # 2. 添加 MS 噪声
    ms_sig = (np.sum(np.power(hrms.flatten(), 2)) / (10 ** (ms_snr / 10)) / hrms.size) ** 0.5
    np.random.seed(1)
    if noise:
        hrms = hrms + ms_sig * np.random.randn(*hrms.shape)

    # 3. 模糊处理（对每个波段卷积）
    lrhs = np.stack([
        cv2.filter2D(hs[:, :, i], -1, B, borderType=cv2.BORDER_REFLECT)
        for i in range(hs.shape[2])
    ], axis=2)

    # 4. 添加 HS 噪声
    hs_sig = (np.sum(np.power(lrhs.flatten(), 2)) / (10 ** (hs_snr / 10)) / lrhs.size) ** 0.5
    np.random.seed(0)
    if noise:
        lrhs = lrhs + hs_sig * np.random.randn(*lrhs.shape)

    # 5. 双三次插值下采样
    h, w, c = lrhs.shape
    if down_type == 0:
        lrhs = lrhs[0::ratio, 0::ratio, :]
    elif down_type == 1:
        lrhs = np.stack([
            cv2.resize(lrhs[:, :, i], (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
            for i in range(c)
        ], axis=2)
    elif down_type == 2:
        lrhs = np.stack([
            cv2.resize(lrhs[:, :, i], (w // ratio, h // ratio), interpolation=cv2.INTER_NEAREST)
            for i in range(c)
        ], axis=2)
    else:
        lrhs = np.stack([
            cv2.resize(lrhs[:, :, i], (w // ratio, h // ratio), interpolation=cv2.INTER_LINEAR)
            for i in range(c)
        ], axis=2)

    return hrms, lrhs

class degDataPreprocessing(object):
    def __init__(self, dataNum, genPath, factor):
        self.dataNum = dataNum
        self.genPath = genPath
        self.factor = factor
        self.R = torch.tensor(create_spec_resp(self.dataNum, self.genPath))

    def __call__(self, hr, type='train'):
        hr = hr.numpy()
        if type == 'train':
            random_ker = random.randint(0, 3)
            if random_ker == 0:
                ks, sigmax, sigmay, angle = 15, 3.4, 2.8, 0
            elif random_ker == 1:
                ks, sigmax, sigmay, angle = 13, 2, 1.8, 45
            elif random_ker == 2:
                ks, sigmax, sigmay, angle = 11, 1.5, 1, 60
            else:
                ks, sigmax, sigmay, angle = 9, 2, 1.5, 135
        else:
            random_ker = random.randint(0, 3)
            if random_ker == 0:
                ks, sigmax, sigmay, angle = 15, 3.3, 2.5, 45
            elif random_ker == 1:
                ks, sigmax, sigmay, angle = 13, 2.5, 2, 60
            elif random_ker == 2:
                ks, sigmax, sigmay, angle = 11, 2, 1.8, 120
            else:
                ks, sigmax, sigmay, angle = 9, 1.8, 1.8, 0
            
        B = anisotropic_gaussian_kernel(ks, sigmax, sigmay, angle)
        hs_snr, ms_snr = 30, 40
        hr = hr.transpose(0, 1, 3, 4, 2)
        batch_size, N, H, W, C = hr.shape
        msi = np.zeros((batch_size, N, H, W, 3))
        hsi = np.zeros((batch_size, N, H//self.factor, W//self.factor, C))
        for i in range(batch_size):
            for j in range(N):
                img = hr[i, j, :, :, :]
                hrms, lrhs = create_hrms_lrhs(img, B, self.R, self.factor, hs_snr, ms_snr, noise=True)
                msi[i, j, :, :, :] = hrms
                hsi[i, j, :, :, :] = lrhs
        msi = msi.astype(np.float32)
        msi = torch.from_numpy(np.ascontiguousarray(msi.transpose(0, 1, 4, 2, 3)))
        hsi = hsi.astype(np.float32)
        hsi = torch.from_numpy(np.ascontiguousarray(hsi.transpose(0, 1, 4, 2, 3)))
        return msi, hsi





        

