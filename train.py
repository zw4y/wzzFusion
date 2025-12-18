import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import FusionNet
from utils.loss import PixelGradLoss, cal_saliency_loss, cal_fre_loss
from utils.get_params_group import get_param_groups
import kornia
from kornia.metrics import AverageMeter
from configs import *
import logging
import yaml
import dataset
from tqdm import tqdm
import argparse
import numpy as np
import wandb
# 添加自动混合精度导入
from torch.cuda.amp import autocast, GradScaler


def to_device(mlist, device):
    for module in mlist:
        module.to(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True      # 加上这个能加速训练


def train(cfg_path, wb_key):
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)

    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)

    # wandb
    wandb.login(key=wb_key)
    runs = wandb.init(project=cfg.project_name,
                      name=cfg.dataset_name + '_' + cfg.exp_name,
                      config=cfg,
                      mode=cfg.wandb_mode)

    # ==================== Model ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fuse_net = FusionNet(channel=16)           # 减少通道数从32到16以节省内存
    fuse_net.to(device)
    fuse_net.train()

    optimizer = torch.optim.Adam(fuse_net.parameters(), lr=cfg.lr_i)
    lr_func = lambda x: (1 - x / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    if cfg.resume is not None:
        logging.info(f'Resume from {cfg.resume}')
        ckpt = torch.load(cfg.resume, map_location=device)
        fuse_net.load_state_dict(ckpt['fuse_net'])

    # ==================== Loss ====================
    loss_ssim = kornia.losses.SSIMLoss(window_size=11)
    loss_grad_pixel = PixelGradLoss()

    # ==================== Data ====================
    train_d = getattr(dataset, cfg.dataset_name)
    train_dataset = train_d(cfg, 'train')

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,                         # 关键：必须打乱！
        num_workers=cfg.num_workers,
        collate_fn=train_dataset.__collate_fn__ if hasattr(train_dataset, '__collate_fn__') else None,
        pin_memory=True,
        drop_last=True
    )

    # 梯度累积步数，用来模拟更大的批次大小
    accumulation_steps = 4  # 如果实际批次大小是2，这会模拟批次大小为8的效果

    # 创建GradScaler对象用于混合精度训练
    scaler = GradScaler()

    logging.info('Start training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        total_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        ssim_loss_meter = AverageMeter()
        saliency_loss_meter = AverageMeter()
        fre_loss_meter = AverageMeter()

        iter_bar = tqdm(trainloader, total=len(trainloader), ncols=100)
        for i, (data_ir, data_vi, mask, _) in enumerate(iter_bar):
            data_ir = data_ir.to(device, non_blocking=True)
            data_vi = data_vi.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # 使用autocast包装前向传播
            with autocast():
                fus_data, amp, pha = fuse_net(data_ir, data_vi)   # 完美兼容

                # ============ Losses ============
                content_loss = loss_grad_pixel(data_vi, data_ir, fus_data)

                ssim_loss_v = loss_ssim(data_vi, fus_data)
                ssim_loss_i = loss_ssim(data_ir, fus_data)
                ssim_loss = ssim_loss_i + ssim_loss_v

                saliency_loss = cal_saliency_loss(fus_data, data_ir, data_vi, mask)
                fre_loss = cal_fre_loss(amp, pha, data_ir, data_vi, mask)

                # 根据累积步数调整损失值
                total_loss = (cfg.coeff_content * content_loss +
                              cfg.coeff_ssim * ssim_loss +
                              cfg.coeff_saliency * saliency_loss +
                              cfg.coeff_fre * fre_loss) / accumulation_steps

            # 使用scaler缩放损失并反向传播
            scaler.scale(total_loss).backward()

            # 只有在累积了足够的梯度后才进行优化器步骤
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                # 使用scaler更新优化器参数
                scaler.step(optimizer)
                # 更新scaler
                scaler.update()
                optimizer.zero_grad()  # 在优化器步骤后重置梯度

            # ============ Logging ============
            total_loss_meter.update(total_loss.item() * accumulation_steps)
            content_loss_meter.update(content_loss.item())
            ssim_loss_meter.update(ssim_loss.item())
            saliency_loss_meter.update(saliency_loss.item())
            fre_loss_meter.update(fre_loss.item())
            
            # 清理变量以释放内存
            del fus_data, amp, pha, content_loss, ssim_loss_v, ssim_loss_i
            del ssim_loss, saliency_loss, fre_loss, total_loss
            
            iter_bar.set_description(f'Epoch {epoch+1}/{cfg.num_epochs}')
            iter_bar.set_postfix({
                'total': f'{total_loss.item() * accumulation_steps:.4f}',
                'cont': f'{content_loss.item():.3f}',
                'ssim': f'{ssim_loss.item():.3f}',
                'sal': f'{saliency_loss.item():.3f}',
                'fre': f'{fre_loss.item():.3f}'
            })
            
        scheduler.step()
        torch.cuda.empty_cache()  # 在每个epoch结束后清理缓存

        # ============ Epoch Summary ============
        log_dict = {
            'total_loss': total_loss_meter.avg,
            'content_loss': content_loss_meter.avg,
            'ssim_loss': ssim_loss_meter.avg,
            'saliency_loss': saliency_loss_meter.avg,
            'fre_loss': fre_loss_meter.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }
        runs.log(log_dict)

        print('*' * 70 + ' EPOCH FINISHED ' + '*' * 70)
        logging.info(
            f'Epoch {epoch+1}/{cfg.num_epochs} | '
            f'lr: {optimizer.param_groups[0]["lr"]:.2e} | '
            f'total: {total_loss_meter.avg:.4f} | '
            f'content: {content_loss_meter.avg:.3f} | '
            f'ssim: {ssim_loss_meter.avg:.3f} | '
            f'saliency: {saliency_loss_meter.avg:.3f} | '
            f'fre: {fre_loss_meter.avg:.4f}'
        )

        # ============ Save Checkpoint ============
        if (epoch + 1) % cfg.epoch_gap == 0 or (epoch + 1) == cfg.num_epochs:
            save_path = os.path.join("models", f'{cfg.exp_name}_epoch{epoch+1}.pth')
            os.makedirs("models", exist_ok=True)
            torch.save({'fuse_net': fuse_net.state_dict()}, save_path)
            logging.info(f'Checkpoint saved: {save_path}')

        torch.cuda.empty_cache()

    print("Training finished! ")
    runs.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/cfg.yaml', help='config file path')
    parser.add_argument('--auth', default='', help='wandb auth api key')
    args = parser.parse_args()
    train(args.cfg, args.auth)