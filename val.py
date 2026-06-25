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
import time
import cv2

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def test(args):
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    # ==================== Load Model ====================
    fuse_net = FusionNet(channel=16)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])

    fuse_net.to(device)
    fuse_net.eval()

    time_list = []

    # 直接读取文件夹中的图像
    img_names = os.listdir(args.ir_path)
    img_names = sorted(img_names)

    with torch.no_grad():
        logging.info(f'Fusing {len(img_names)} images in {args.mode} mode...')
        iter_bar = tqdm(img_names, total=len(img_names), ncols=80)

        for img_name in iter_bar:
            # ==================== Read IR ====================
            data_ir = img_read(
                os.path.join(args.ir_path, img_name),
                mode='L'
            ).unsqueeze(0)

            # ==================== Read VIS ====================
            vi_read = img_read(
                os.path.join(args.vi_path, img_name),
                mode='YCbCr'
            )

            if isinstance(vi_read, tuple) or isinstance(vi_read, list):
                data_vi = vi_read[0].unsqueeze(0)
                vi_cbcr = vi_read[1].unsqueeze(0)
            else:
                data_vi = vi_read.unsqueeze(0)
                vi_cbcr = None

            # ==================== Crop to even size ====================
            _, _, h, w = data_ir.shape
            if h % 2 != 0 or w % 2 != 0:
                data_ir = data_ir[..., : h // 2 * 2, : w // 2 * 2]
                data_vi = data_vi[..., : h // 2 * 2, : w // 2 * 2]
                if args.mode == 'RGB' and vi_cbcr is not None:
                    vi_cbcr = vi_cbcr[..., : h // 2 * 2, : w // 2 * 2]

            data_ir = data_ir.to(device)
            data_vi = data_vi.to(device)

            # ==================== Forward ====================
            ts = time.time()

            # 当前 FusionNet.forward() 返回 10 个值：
            # fusion, fused_amp, fused_pha, high_mask,
            # ir_amp, vi_amp, vi_orig, vi_enhanced, curve_A, enhance_mask
            # 验证阶段只需要最终融合图 fusion，因此直接取 [0]。
            fus_data = fuse_net(data_ir, data_vi)[0]

            te = time.time()
            time_list.append(te - ts)

            # ==================== Save Fusion Result ====================
            if args.mode == 'gray':
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder)

            elif args.mode == 'RGB':
                if vi_cbcr is None:
                    raise RuntimeError(
                        "RGB mode requires VIS image to provide CbCr channels, "
                        "but vi_cbcr is None. Please check img_read(..., mode='YCbCr')."
                    )

                vi_cbcr = vi_cbcr.to(device)
                fi = torch.cat((fus_data, vi_cbcr), dim=1)
                fi = ycbcr_to_rgb(fi)
                fi = tensor_to_image(fi) * 255
                fi = fi.astype(np.uint8)
                img_save(fi, img_name, fuse_out_folder, mode='RGB')

            else:
                raise ValueError(f"Unsupported mode: {args.mode}. Use 'gray' or 'RGB'.")

    if len(time_list) > 1:
        avg_time = np.round(np.mean(time_list[1:]), 6)
    else:
        avg_time = np.round(np.mean(time_list), 6)

    logging.info(f'Fusing images done! Avg time: {avg_time}s')

    # 生成融合图后直接评估
    evaluate(args)


def evaluate(args):
    fuse_out_folder = args.out_dir
    img_names = os.listdir(args.ir_path)
    img_names = sorted(img_names)

    metric_names = ['EN', 'SF', 'AG', 'MI', 'VIFF', 'Qabf']
    metric_result = [AverageMeter() for _ in range(len(metric_names))]
    per_image_metrics = {}

    logging.info('Evaluating images ...')
    iter_bar = tqdm(img_names, total=len(img_names), ncols=80)

    for img_name in iter_bar:
        # 直接读取灰度原图进行指标计算
        ir = img_read(os.path.join(args.ir_path, img_name), 'L').numpy().squeeze() * 255
        vi = img_read(os.path.join(args.vi_path, img_name), 'L').numpy().squeeze() * 255

        fi_path = os.path.join(fuse_out_folder, img_name)
        fi = img_read(fi_path, 'L').numpy().squeeze() * 255

        h, w = fi.shape
        if h % 2 != 0 or w % 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]

        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))

        en_val = Evaluator.EN(fi)
        sf_val = Evaluator.SF(fi)
        ag_val = Evaluator.AG(fi)
        mi_val = Evaluator.MI(fi, ir, vi)
        viff_val = Evaluator.VIFF(fi, ir, vi)
        qabf_val = Evaluator.Qabf(fi, ir, vi)

        metric_result[0].update(en_val)
        metric_result[1].update(sf_val)
        metric_result[2].update(ag_val)
        metric_result[3].update(mi_val)
        metric_result[4].update(viff_val)
        metric_result[5].update(qabf_val)

        per_image_metrics[img_name] = [
            en_val, sf_val, ag_val, mi_val, viff_val, qabf_val
        ]

    # ==================== Save Average Metrics ====================
    with open(f'{fuse_out_folder}_result.txt', 'w') as f:
        for i, name in enumerate(metric_names):
            f.write(f'{name}: ' + str(np.round(metric_result[i].avg, 3)) + '\n')

    # ==================== Save Per-image Metrics ====================
    per_img_result_file = f'{fuse_out_folder}_per_img_result.txt'
    with open(per_img_result_file, 'w') as f:
        f.write("Image_Name\t" + "\t".join(metric_names) + "\n")
        for img_name, metrics in per_image_metrics.items():
            f.write(
                f'{img_name}\t' +
                "\t".join([str(np.round(val, 6)) for val in metrics]) +
                "\n"
            )

    print("\n" + "=" * 80)
    print("Average test result :")
    print("\t\t" + "\t".join(metric_names))
    print("result:\t" + "\t".join([
        str(np.round(metric_result[i].avg, 3))
        for i in range(len(metric_names))
    ]))
    print("=" * 80)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckpt_path', type=str, default=f'models/model_lastest.pth')
    parse.add_argument('--ir_path', type=str, default='./LLVIP_50pairs/test/ir/')
    parse.add_argument('--vi_path', type=str, default='./LLVIP_50pairs/test/vi/')
    parse.add_argument('--out_dir', type=str, default=f'test_result/LLVIP_50pairs')
    parse.add_argument('--mode', type=str, default='RGB')
    args = parse.parse_args()

    test(args)