import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import glob
from util import create_hrms_lrhs_inte
import random


class TrainDataLoader(Dataset):
    def __init__(self, root, psfPath, srfPath, factor=8, patch_size=16, seed=0):
        super(TrainDataLoader, self).__init__()
        self.factor = factor
        self.hs_snr = 30
        self.ms_snr = 40
        self.hr_size = patch_size * factor
        self.lr_size = patch_size
        self.image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
        self.hrhs_list = []
        self.lrhs_list = []
        self.hrms_list = []
        self.stride = 48
        self.psf = sio.loadmat(psfPath)['train']
        self.srf = sio.loadmat(srfPath)['train']
        random.seed(seed)
        self.rand_ker_list = [random.randint(0, len(self.psf) - 1) for _ in range(len(self.image_names))]
        self.rand_srf_list = [random.randint(0, len(self.srf) - 1) for _ in range(len(self.image_names))]
        self.down_type = [random.randint(0, 3) for _ in range(len(self.image_names))]
        
        for ind in range(len(self.image_names)):
            hrhs = sio.loadmat(self.image_names[ind])
            hrhs = hrhs["HS"].astype(np.float32)
            hrhs = hrhs[0: 1024, 0: 1024, :]
            B = self.psf[self.rand_ker_list[ind], :, :]
            R = self.srf[self.rand_srf_list[ind], :, :]
            hrms, lrhs = create_hrms_lrhs_inte(hrhs, B, R, self.factor, self.hs_snr, self.ms_snr, noise=True, down_type=self.down_type[ind])
            H, W, C = hrhs.shape  
            
            # 计算patch的数量
            n_rows = (H - self.hr_size) // self.stride + 1
            n_cols = (W - self.hr_size) // self.stride + 1
            
            # 遍历图像并提取patch
            for i in range(n_rows):
                for j in range(n_cols):
                    # 计算当前patch的左上角坐标
                    hr_row = i * self.stride
                    hr_col = j * self.stride
                    lr_row = i * self.stride // self.factor
                    lr_col = j * self.stride // self.factor
                    # 提取当前patch
                    hrhs_patch = hrhs[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    hrms_patch = hrms[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    lrhs_patch = lrhs[lr_row: lr_row + self.lr_size, lr_col: lr_col + self.lr_size, :]

                    self.hrhs_list.append(hrhs_patch)
                    self.hrms_list.append(hrms_patch)
                    self.lrhs_list.append(lrhs_patch)
            
    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs, hrms, lrhs = self.hrhs_list[index], self.hrms_list[index], self.lrhs_list[index]
        hrhs = hrhs.astype(np.float32)
        hrms = hrms.astype(np.float32)
        lrhs = lrhs.astype(np.float32)
        hrhs = torch.from_numpy(np.ascontiguousarray(hrhs.transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(hrms.transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(lrhs.transpose(2, 0, 1)))

        return hrhs, lrhs, hrms

class TestDataLoader(Dataset):
    def __init__(self, root, psfPath, srfPath, factor=8, patch_size=16):
        super(TestDataLoader, self).__init__()
        self.factor = factor
        self.hs_snr = 30
        self.ms_snr = 40
        self.hr_size = patch_size * factor
        self.lr_size = patch_size
        self.image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
        self.hrhs_list = []
        self.lrhs_list = []
        self.hrms_list = []
        self.stride = 64
        self.psf = sio.loadmat(psfPath)['test']
        self.srf = sio.loadmat(srfPath)['test']
        random.seed(5)
        self.rand_ker_list = [random.randint(0, len(self.psf) - 1) for _ in range(len(self.image_names))]
        self.rand_srf_list = [random.randint(0, len(self.srf) - 1) for _ in range(len(self.image_names))]
        self.down_type = [random.randint(0, 3) for _ in range(len(self.image_names))]
        
        for ind in range(len(self.image_names)):
            hrhs = sio.loadmat(self.image_names[ind])
            hrhs = hrhs["HS"].astype(np.float32)
            hrhs = hrhs[0: 1024, 0: 1024, :]
            B = self.psf[self.rand_ker_list[ind], :, :]
            R = self.srf[self.rand_srf_list[ind], :, :]
            hrms, lrhs = create_hrms_lrhs_inte(hrhs, B, R, self.factor, self.hs_snr, self.ms_snr, noise=True, down_type=self.down_type[ind])
            H, W, C = hrhs.shape  
            
            # 计算patch的数量
            n_rows = (H - self.hr_size) // self.stride + 1
            n_cols = (W - self.hr_size) // self.stride + 1
            
            # 遍历图像并提取patch
            for i in range(n_rows):
                for j in range(n_cols):
                    # 计算当前patch的左上角坐标
                    hr_row = i * self.stride
                    hr_col = j * self.stride
                    lr_row = i * self.stride // self.factor
                    lr_col = j * self.stride // self.factor
                    # 提取当前patch
                    hrhs_patch = hrhs[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    hrms_patch = hrms[hr_row: hr_row + self.hr_size, hr_col: hr_col + self.hr_size, :]
                    lrhs_patch = lrhs[lr_row: lr_row + self.lr_size, lr_col: lr_col + self.lr_size, :]

                    self.hrhs_list.append(hrhs_patch)
                    self.hrms_list.append(hrms_patch)
                    self.lrhs_list.append(lrhs_patch)
            
    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs, hrms, lrhs = self.hrhs_list[index], self.hrms_list[index], self.lrhs_list[index]
        hrhs = hrhs.astype(np.float32)
        hrms = hrms.astype(np.float32)
        lrhs = lrhs.astype(np.float32)
        hrhs = torch.from_numpy(np.ascontiguousarray(hrhs.transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(hrms.transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(lrhs.transpose(2, 0, 1)))

        return hrhs, lrhs, hrms
    

