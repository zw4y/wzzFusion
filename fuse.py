from modules import *
import os
import numpy as np
import torch
from utils.img_read import *
import logging
from tqdm import tqdm
import warnings
import time
import argparse

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def fuse(args):
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    fuse_net = FusionNet(channel=16)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.to(device)
    fuse_net.eval()

    time_list = []
    img_names = os.listdir(args.ir_path)

    with torch.no_grad():
        logging.info(f'Fusing {len(img_names)} images in {args.mode} mode...')
        iter_bar = tqdm(img_names, total=len(img_names), ncols=80)

        for img_name in iter_bar:
            # 1. 移入循环内部：按需读取，杜绝 OOM
            data_ir = img_read(os.path.join(args.ir_path, img_name), mode='L').unsqueeze(0)

            # 2. 截留色彩通道：兼容返回 (Y, CbCr) 元组的读取方式
            vi_read = img_read(os.path.join(args.vi_path, img_name), mode='YCbCr')
            if isinstance(vi_read, tuple) or isinstance(vi_read, list):
                data_vi = vi_read[0].unsqueeze(0)
                vi_cbcr = vi_read[1].unsqueeze(0)
            else:
                data_vi = vi_read.unsqueeze(0)
                vi_cbcr = None  # 如果不是 tuple，说明 utils.img_read 被魔改过，需排查

            # 对齐尺寸 (使用 ... 自动适配维度)
            _, _, h, w = data_ir.shape
            if h // 2 != 0 or w // 2 != 0:
                data_ir = data_ir[..., : h // 2 * 2, : w // 2 * 2]
                data_vi = data_vi[..., : h // 2 * 2, : w // 2 * 2]
                if args.mode == 'RGB' and vi_cbcr is not None:
                    vi_cbcr = vi_cbcr[..., : h // 2 * 2, : w // 2 * 2]

            data_vi, data_ir = data_vi.to(device), data_ir.to(device)

            # 3. 前向推理
            ts = time.time()
            fus_data, _, _, _, _, _, _, _, _ = fuse_net(data_ir, data_vi)
            te = time.time()
            time_list.append(te - ts)

            # 4. 色彩重组与保存
            if args.mode == 'gray':
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder)
            elif args.mode == 'RGB':
                vi_cbcr = vi_cbcr.to(device)
                fi = torch.cat((fus_data, vi_cbcr), dim=1)
                fi = ycbcr_to_rgb(fi)
                fi = tensor_to_image(fi) * 255
                fi = fi.astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder, mode='RGB')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckpt_path', type=str, default=f'models/model_lastest.pth')
    parse.add_argument('--ir_path', type=str, default='./LLVIP_50pairs/test/ir/')
    parse.add_argument('--vi_path', type=str, default='./LLVIP_50pairs/test/vi/')
    parse.add_argument('--out_dir', type=str, default=f'test_result/LLVIP_50pairs')
    parse.add_argument('--mode', type=str, default='RGB')
    args = parse.parse_args()

    fuse(args)