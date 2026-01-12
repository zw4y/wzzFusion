from modules import *
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
from utils.img_read import *
import argparse
import logging
from kornia.metrics import AverageMeter
from tqdm import tqdm
import warnings
import yaml
from configs import from_dict
import dataset
from torch.utils.data import DataLoader
from thop import profile, clever_format
import time
import cv2

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def test(args):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    # 根据checkpoint调整模型参数
    fuse_net = FusionNet(channel=32)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.to(device)
    fuse_net.eval()

    time_list = []
    with torch.no_grad():
        logging.info(f'fusing images ...')
        iter = tqdm(testloader, total=len(testloader), ncols=80)
        for data_ir, data_vi, _, img_name in iter:
            data_vi, data_ir = data_vi.to(device), data_ir.to(device)

            ts = time.time()
            fus_data, _, _ = fuse_net(data_ir, data_vi)
            te = time.time()
            time_list.append(te - ts)
            if args.mode == 'gray':
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name[0], fuse_out_folder)
            elif args.mode == 'RGB':
                vi_cbcr = vi_cbcr.to(device)
                fi = torch.cat((fus_data, vi_cbcr), dim=1)
                fi = ycbcr_to_rgb(fi)
                fi = tensor_to_image(fi) * 255
                fi = fi.astype(np.uint8)
                img_save(fi, img_name[0], fuse_out_folder, mode='RGB')

    logging.info(f'fusing images done!')
    logging.info(f'time: {np.round(np.mean(time_list[1:]), 6)}s')
    evaluate(fuse_out_folder)


def evaluate(fuse_out_folder):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    # 定义所有指标的名称列表
    metric_names = ['EN', 'SF', 'MI', 'SCD', 'VIFF', 'Qabf']
    metric_result = [AverageMeter() for _ in range(len(metric_names))]
    
    # 存储每张图片的指标值
    per_image_metrics = {}

    logging.info(f'evaluating images ...')
    iter = tqdm(testloader, total=len(testloader), ncols=80)

    for data_ir, data_vi, _, img_name in iter:
        ir = data_ir.numpy().squeeze() * 255
        vi = data_vi.numpy().squeeze() * 255
        fi_path = os.path.join(fuse_out_folder, img_name[0])
        fi = img_read(fi_path, 'L').numpy().squeeze() * 255
        h, w = fi.shape
        if h // 2 != 0 or w // 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]
        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))
        # print(ir.shape, vi.shape, fi.shape)
        
        # 计算各项指标
        en_val = Evaluator.EN(fi)
        sf_val = Evaluator.SF(fi)
        mi_val = Evaluator.MI(fi, ir, vi)
        scd_val = Evaluator.SCD(fi, ir, vi)
        viff_val = Evaluator.VIFF(fi, ir, vi)
        qabf_val = Evaluator.Qabf(fi, ir, vi)
        
        # 更新平均值
        metric_result[0].update(en_val)
        metric_result[1].update(sf_val)
        metric_result[2].update(mi_val)
        metric_result[3].update(scd_val)
        metric_result[4].update(viff_val)
        metric_result[5].update(qabf_val)
        
        # 存储每张图片的指标值
        per_image_metrics[img_name[0]] = [en_val, sf_val, mi_val, scd_val, viff_val, qabf_val]

#    # 输出每张图片的指标值
#    print("\n" * 2 + "-" * 80)
#    print("Per-image results:")
#    # 打印表头
#    header = "Image_Name\t\t" + "\t".join(metric_names)
#    print(header)
#    # 打印每张图片的指标值
#    for img_name, metrics in per_image_metrics.items():
#        result_line = f'{img_name[:15]:<15}\t' + "\t".join([str(np.round(val, 3)) for val in metrics])
#        print(result_line)
#    print("-" * 80)
    
    # 结果写入主结果文件（平均值）
    with open(f'{fuse_out_folder}_result.txt', 'w') as f:
        for i, name in enumerate(metric_names):
            f.write(f'{name}: ' + str(np.round(metric_result[i].avg, 3)) + '\n')
    
    # 将每张图片的指标值写入单独的文件
    per_img_result_file = f'{fuse_out_folder}_per_img_result.txt'
    with open(per_img_result_file, 'w') as f:
        # 写入表头
        f.write("Image_Name\t" + "\t".join(metric_names) + "\n")
        # 写入每张图片的指标值
        for img_name, metrics in per_image_metrics.items():
            f.write(f'{img_name}\t' + "\t".join([str(np.round(val, 6)) for val in metrics]) + "\n")

    logging.info(f'writing results done!')
    print("\n" * 2 + "=" * 80)
    print("Average test result :")
    # 打印指标名称行
    header = "\t\t" + "\t".join(metric_names)
    print(header)
    # 打印结果行
    result_line = "result:\t" + "\t".join([str(np.round(metric_result[i].avg, 3)) for i in range(len(metric_names))])
    print(result_line)
    print("=" * 80)
    
    print(f'Per-image results saved to: {per_img_result_file}')


if __name__ == "__main__":
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckpt_path', type=str, default=f'models/model-MSRS_over_epoch200.pth')
    parse.add_argument('--dataset_name', type=str, default=cfg.dataset_name)
    parse.add_argument('--out_dir', type=str, default=f'test_result/MSRS_per')
    parse.add_argument('--mode', type=str, default='gray')
    args = parse.parse_args()

    test(args)
    # evaluate("./test_result/res.txt")