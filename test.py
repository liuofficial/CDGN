import numpy as np
import torch
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import util
import time
import scipy.io as sio
from CDGN import X_Module
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    filename='test.log',  # 日志文件名
    filemode='a'  # 'w'覆盖，'a'追加
)


model_root = './model_path/harvard_x8.pt'
root = ''
output_save_path = './results/ours/harvard/'
piece_size = 64
stride = 32
ratio = 8
hs_snr = 30
ms_snr = 40

# if not os.path.exists(output_save_path):
#     os.makedirs(output_save_path, exist_ok=True)

run_time = 0
model = X_Module(31, 3, 8, nf=64).cuda()  # Ours
weight = torch.load(model_root)['model']
model.load_state_dict(weight)

model.eval()
image_names = sorted(glob.glob(os.path.join(root, '*.mat')))
psnrs = []
sams = []
ssims = []
ergas = []
ccs = []
rmses = []
logging.info("start!")
logging.info(model_root)
for ind in range(len(image_names)):
    name = image_names[ind].split("\\")[-1]
    name = name.split(".")[0]
    data = sio.loadmat(image_names[ind])
    X = data['X']
    tZ, tY = data['Z'], data['Y']
    output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
    num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
    start = time.perf_counter()
    for x in range(0, tZ.shape[0] - piece_size + 1, stride):
        for y in range(0, tZ.shape[1] - piece_size + 1, stride):
            end_x = x + piece_size
            if end_x + stride > tZ.shape[0]:
                end_x = tZ.shape[0]
            end_y = y + piece_size
            if end_y + stride > tZ.shape[1]:
                end_y = tZ.shape[1]
            itY = tY[x // ratio:end_x // ratio, y // ratio:end_y // ratio, :]
            itZ = tZ[x:end_x, y:end_y, :]
            itY = itY.astype(np.float32)
            itY = torch.from_numpy(np.ascontiguousarray(itY.transpose(2, 0, 1)))
            itZ = itZ.astype(np.float32)
            itZ = torch.from_numpy(np.ascontiguousarray(itZ.transpose(2, 0, 1)))
            itY, itZ = torch.unsqueeze(itY, dim=0).cuda(), torch.unsqueeze(itZ, dim=0).cuda()
            res = model(itY, itZ)
            sr = res[0]
            sr = sr.detach().cpu().squeeze(0).numpy()
            sr = np.transpose(sr, (1, 2, 0))
            output[x:end_x, y:end_y, :] += sr
            num_sum[x:end_x, y:end_y, :] += 1
    output = output / num_sum
    end = time.perf_counter()
    run_time += end - start
    psnr = util.PSNR(output, X)
    sam, sam_map = util.SAM(output, X)
    ssim = util.SSIM(output, X)
    erg = util.ERGAS(output, X, ratio)
    cc = util.CC(X, output)
    rmse = util.RMSE(X, output)
    psnrs.append(psnr)
    sams.append(sam)
    ssims.append(ssim)
    ergas.append(erg)
    ccs.append(cc)
    rmses.append(rmse)
    # save_path = os.path.join(output_save_path, name + '_sr.mat')
    # sio.savemat(save_path, {'HS': output})
    logging.info(
        "name: {}, time:{:.6f}, psnr: {:.6f}, sam: {:.6f}, ssim: {:.6f}, ergas: {:.6f}, cc: {:.6f}, rmse: {:.6f}".format(
            name, end - start, psnr, sam, ssim, erg, cc, rmse))
logging.info('Time: %ss' % (run_time / len(image_names)))
logging.info("PSNR: {:.6f}, SAM: {:.6f}, SSIM: {:.6f}, ERGAS: {:.6f}, CC: {:.6f}, RMSE: {:.6f}".format(
        sum(psnrs) / len(psnrs), sum(sams) / len(sams), sum(ssims) / len(ssims),
        sum(ergas) / len(ergas), sum(ccs) / len(ccs), sum(rmses) / len(rmses)))







