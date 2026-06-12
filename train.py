import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CDGN import X_Module
from loaddata import TrainDataLoader, TestDataLoader
import util
import numpy as np
import argparse
import logging

# 配置logging模块
logging.basicConfig(
    filename='train.log',    # 指定日志文件名
    level=logging.DEBUG,   # 设置日志级别
    format='%(asctime)s - %(message)s'  # 设置日志格式
)

# 定义一个新的 print 函数，将消息发送到日志文件
def print_to_log(*args, **kwargs):
    message = ' '.join(map(str, args))
    logging.info(message)

# 将 print 替换为 print_to_log
print = print_to_log

# 示例使用新的 print 函数
print("start")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='dcgn model', help="")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--patch_size", type=int, default=8, help="size of each image dimension")
parser.add_argument("--factor", type=int, default=8, help=" ")
parser.add_argument("--nf", type=int, default=64, help=" ")
parser.add_argument("--nb", type=int, default=5, help="The number of iterations of the overall model")
parser.add_argument("--checkpoint_save_path", type=str, default="./model_path/", help=" ")
parser.add_argument("--train_root", type=str, default="../data/harvard/train", help=" ")
parser.add_argument("--test_root", type=str, default="../data/harvard/test", help=" ")
parser.add_argument("--psf_root", type=str, default="psf35_15_15_split.mat", help=" ")
parser.add_argument("--srf_root", type=str, default="srf_28x3x31_split.mat", help=" ")
parser.add_argument("--dataNum", type=int, default=0, help=" ")
parser.add_argument("--hs_in_ch", type=int, default=31, help="number of lrhsi image channels")
parser.add_argument("--ms_in_ch", type=int, default=3, help="number of hrmsi image channels")

opt = parser.parse_args()
print(opt)
if not os.path.exists(opt.checkpoint_save_path):
    os.makedirs(opt.checkpoint_save_path, exist_ok=True)
testdata = TestDataLoader(opt.test_root, opt.psf_root, opt.srf_root, opt.factor, opt.patch_size)
testloader = DataLoader(testdata, batch_size=1)


# load models
model = X_Module(opt.hs_in_ch, opt.ms_in_ch, factor=opt.factor, nf=opt.nf)
model = model.to(device)


optim = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=100, gamma=0.5)

# Loss
l1_loss = torch.nn.L1Loss().to(device)
loss = 0.

best_psnr = 0
best_sam = 100
only_psnr_best = 0
only_sam_best = 100

for epoch in range(opt.n_epochs):
    traindata = TrainDataLoader(opt.train_root, opt.psf_root, opt.srf_root, opt.factor, opt.patch_size, epoch)
    trainloader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    model.train()
    
    for iteration, batch in enumerate(trainloader, 1):
        gt, Y, Z = batch[0], batch[1], batch[2]
        gt = gt.to(device)
        Y = Y.to(device)
        Z = Z.to(device)
        sr, Xy, Xz = model(Y, Z)
        loss_res = l1_loss(Xy, gt) + l1_loss(Xz, gt)
        loss = l1_loss(sr, gt) + 0.5 * loss_res
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optim.step()
    
    lr_scheduler.step()

    if (epoch + 1) % 100 == 0:
        torch.save(
            {
                'model': model.state_dict(),
            },
            os.path.join(opt.checkpoint_save_path, 'model_{}.pt'.format(epoch + 1))
        )

    if (epoch + 1) % 10 == 0:
        model.eval()
        psnrs = []
        sams = []
        print("epoch: {}, loss: {}".format(epoch, loss))

        for step, batch in enumerate(testloader):
            gt, Y, Z = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            res = model(Y, Z)
            sr = res[0]
            sr = sr.detach().cpu()
            gt = gt.cpu()
            sr= sr.squeeze(0).numpy()
            sr = np.transpose(sr, (1, 2, 0))
            gt = np.array(gt.cpu().squeeze(0))
            gt = np.transpose(gt, (1, 2, 0))

            psnr = util.PSNR(sr, gt)
            sam, sam_map = util.SAM(sr, gt)
            psnrs.append(psnr)
            sams.append(sam)

        avg_psnr = sum(psnrs) / len(psnrs)
        avg_sam = sum(sams) / len(sams)
        if(best_psnr <= avg_psnr and best_sam >= avg_sam):
            best_psnr = avg_psnr
            best_sam = avg_sam
            torch.save(
            {
                'model': model.state_dict(),
            },
            os.path.join(opt.checkpoint_save_path, 'model_best.pt')
        )
            
        if(only_psnr_best < avg_psnr):
            only_psnr_best = avg_psnr
            torch.save(
            {
                'model': model.state_dict(),
            },
            os.path.join(opt.checkpoint_save_path, 'model_best_psnr.pt')
        )
        
        if(only_sam_best > avg_sam):
            only_sam_best = avg_sam
            torch.save(
            {
                'model': model.state_dict(),
            },
            os.path.join(opt.checkpoint_save_path, 'model_best_sam.pt')
        )
       
        print("PSNR: {}, SAM: {}".format(avg_psnr, avg_sam))



