import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ifft2


class MultiScaleDirSobel(nn.Module):
    """
    多尺度方向感知 Sobel 算子
    - 分别返回独立的 |Gx| 和 |Gy|（不合并）
    - 支持 3×3, 5×5, 7×7 三个尺度
    """
    def __init__(self):
        super().__init__()
        # === 3×3 Sobel ===
        kx3 = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], dtype=torch.float32)
        ky3 = torch.tensor([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]], dtype=torch.float32)

        # === 5×5 Sobel ===
        kx5 = torch.tensor([[-1., -2., 0., 2., 1.],
                             [-4., -8., 0., 8., 4.],
                             [-6.,-12., 0.,12., 6.],
                             [-4., -8., 0., 8., 4.],
                             [-1., -2., 0., 2., 1.]], dtype=torch.float32)
        ky5 = torch.tensor([[-1., -4., -6., -4., -1.],
                             [-2., -8.,-12., -8., -2.],
                             [ 0.,  0.,  0.,  0.,  0.],
                             [ 2.,  8., 12.,  8.,  2.],
                             [ 1.,  4.,  6.,  4.,  1.]], dtype=torch.float32)

        # === 7×7 Sobel ===
        kx7 = torch.tensor([[-1., -4., -5.,  0.,  5.,  4.,  1.],
                             [-6.,-24.,-30.,  0., 30., 24.,  6.],
                             [-15.,-60.,-75., 0., 75., 60., 15.],
                             [-20.,-80.,-100.,0.,100., 80., 20.],
                             [-15.,-60.,-75., 0., 75., 60., 15.],
                             [-6.,-24.,-30.,  0., 30., 24.,  6.],
                             [-1., -4., -5.,  0.,  5.,  4.,  1.]], dtype=torch.float32)
        ky7 = kx7.t().contiguous()

        # 不做归一化，保持原始 Sobel 量级（与原 Sobelxy 一致）
        # 大尺度核的响应天然更大，通过 PixelGradLoss 中的 scale_weights 来平衡

        self.register_buffer('kx3', kx3.view(1, 1, 3, 3))
        self.register_buffer('ky3', ky3.view(1, 1, 3, 3))
        self.register_buffer('kx5', kx5.view(1, 1, 5, 5))
        self.register_buffer('ky5', ky5.view(1, 1, 5, 5))
        self.register_buffer('kx7', kx7.view(1, 1, 7, 7))
        self.register_buffer('ky7', ky7.view(1, 1, 7, 7))

    def forward(self, x):
        """
        返回: [(|Gx_3|, |Gy_3|), (|Gx_5|, |Gy_5|), (|Gx_7|, |Gy_7|)]
        """
        # 兼容 AMP 混合精度：将核转换到与输入相同的 dtype 和 device
        kx3 = self.kx3.to(dtype=x.dtype, device=x.device)
        ky3 = self.ky3.to(dtype=x.dtype, device=x.device)
        kx5 = self.kx5.to(dtype=x.dtype, device=x.device)
        ky5 = self.ky5.to(dtype=x.dtype, device=x.device)
        kx7 = self.kx7.to(dtype=x.dtype, device=x.device)
        ky7 = self.ky7.to(dtype=x.dtype, device=x.device)

        gx3 = F.conv2d(x, kx3, padding=1, groups=x.shape[1]).abs()
        gy3 = F.conv2d(x, ky3, padding=1, groups=x.shape[1]).abs()
        gx5 = F.conv2d(x, kx5, padding=2, groups=x.shape[1]).abs()
        gy5 = F.conv2d(x, ky5, padding=2, groups=x.shape[1]).abs()
        gx7 = F.conv2d(x, kx7, padding=3, groups=x.shape[1]).abs()
        gy7 = F.conv2d(x, ky7, padding=3, groups=x.shape[1]).abs()
        return [(gx3, gy3), (gx5, gy5), (gx7, gy7)]


class PixelGradLoss(nn.Module):
    """
    像素损失 + 方向感知多尺度梯度损失
    - 像素损失：L1(max(ir, vis), fused)  —— 不变
    - 梯度损失：对 x/y 方向独立、在 3/5/7 三个尺度上分别计算
    """
    def __init__(self):
        super().__init__()
        self.ms_sobel = MultiScaleDirSobel()
        # 多尺度权重：
        # 3×3 核元素绝对值和=8, 5×5=96(≈12×), 7×7=1440(≈180×)
        # 权重按比例缩小，使各尺度对总损失的贡献量级接近
        # 3×3 权重为 1.0（主导，与原 Sobel 一致）
        self.scale_weights = [1.0, 1.0 / 12.0, 1.0 / 180.0]

    def forward(self, image_vis, image_ir, fus_img):
        image_y = image_vis[:, :1, :, :]

        # ===== 像素损失（不变）=====
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, fus_img)

        # ===== 方向感知多尺度梯度损失 =====
        vis_grads = self.ms_sobel(image_y)
        ir_grads = self.ms_sobel(image_ir)
        fus_grads = self.ms_sobel(fus_img)

        loss_grad = 0.0
        for s, w in enumerate(self.scale_weights):
            # x 方向
            target_gx = torch.max(vis_grads[s][0], ir_grads[s][0])
            loss_gx = F.l1_loss(fus_grads[s][0], target_gx)
            # y 方向
            target_gy = torch.max(vis_grads[s][1], ir_grads[s][1])
            loss_gy = F.l1_loss(fus_grads[s][1], target_gy)
            loss_grad = loss_grad + w * (loss_gx + loss_gy)

        # 权重与原来保持同量级：原来是 5*pixel + 10*grad
        return 5 * loss_in + 10 * loss_grad


def cal_sf_loss(fus_img):
    """
    可微分 Spatial Frequency 损失
    SF = sqrt(RF² + CF²)
    RF = sqrt(mean((I[i,j] - I[i,j-1])²))  行频率
    CF = sqrt(mean((I[i,j] - I[i-1,j])²))  列频率
    损失 = 1 - SF（SF 越大损失越小）
    """
    # 水平差分：相邻列之差
    diff_h = fus_img[:, :, :, 1:] - fus_img[:, :, :, :-1]
    # 垂直差分：相邻行之差
    diff_v = fus_img[:, :, 1:, :] - fus_img[:, :, :-1, :]

    # RF 和 CF
    rf = torch.sqrt(torch.mean(diff_h ** 2) + 1e-8)
    cf = torch.sqrt(torch.mean(diff_v ** 2) + 1e-8)

    # SF
    sf = torch.sqrt(rf ** 2 + cf ** 2)

    return 1.0 - sf


def cal_freq_hf_loss(fused_amp, ir_amp, vi_amp, high_mask):
    """
    高频 FFT 损失：约束融合图像的高频幅度不低于两个源图像的最大值
    复用 DecoupledHLFuse 内部的 high_mask，保证"高频"定义一致
    """
    fused_hf = fused_amp * high_mask
    ir_hf = ir_amp * high_mask
    vi_hf = vi_amp * high_mask

    target_hf = torch.max(ir_hf, vi_hf)
    return F.l1_loss(fused_hf, target_hf)


def cal_saliency_loss(fus, ir, vi, mask):
    loss_tar = F.l1_loss(fus * mask, ir * mask)
    loss_back = F.l1_loss(fus * (1 - mask), vi * (1 - mask))
    return 5 * loss_tar + loss_back


def cc(img1, img2, eps=1e-8):
    img1 = img1 - img1.mean(dim=(-2, -1), keepdim=True)
    img2 = img2 - img2.mean(dim=(-2, -1), keepdim=True)
    num = torch.sum(img1 * img2, dim=(-2, -1))
    denom = torch.sqrt(torch.sum(img1**2, dim=(-2, -1))) * torch.sqrt(torch.sum(img2**2, dim=(-2, -1))) + eps
    return torch.clamp(num / denom, -1.0, 1.0).mean()


def cal_fre_loss(amp, pha, ir, vi, mask):
    recon = ifft2(amp, pha)
    recon = torch.clamp(recon, 0, 1)
    loss_ir = cc(recon * mask, ir * mask)
    loss_vi = cc(recon * (1 - mask), vi * (1 - mask))
    return -(loss_ir + loss_vi)   # 负号！方向正确！


class EnhanceLoss(nn.Module):
    """
    低光增强自监督损失（基于 Zero-DCE）
    包含三项：
    - L_spa: 空间一致性损失，保持增强前后的局部结构
    - L_exp: 曝光控制损失，约束暗区增强后的亮度
    - L_TV:  全变分损失，约束曲线参数的平滑性
    """
    def __init__(self, patch_size=16, mean_val=0.5):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

        # 空间一致性的四方向差分核（用 register_buffer 兼容 AMP）
        self.register_buffer('w_left',  torch.FloatTensor([[0,0,0],[-1,1,0],[0,0,0]]).view(1,1,3,3))
        self.register_buffer('w_right', torch.FloatTensor([[0,0,0],[0,1,-1],[0,0,0]]).view(1,1,3,3))
        self.register_buffer('w_up',    torch.FloatTensor([[0,-1,0],[0,1,0],[0,0,0]]).view(1,1,3,3))
        self.register_buffer('w_down',  torch.FloatTensor([[0,0,0],[0,1,0],[0,-1,0]]).view(1,1,3,3))
        self.spa_pool = nn.AvgPool2d(4)

    def forward(self, vi_orig, vi_enhanced, curve_A):
        """
        vi_orig:     增强前的 VIS (B,1,H,W)
        vi_enhanced: 增强后的 VIS (B,1,H,W)
        curve_A:     曲线参数图 (B,curve_num,H,W)
        """
        # 兼容 AMP：同时转换 dtype 和 device
        w_left = self.w_left.to(dtype=vi_orig.dtype, device=vi_orig.device)
        w_right = self.w_right.to(dtype=vi_orig.dtype, device=vi_orig.device)
        w_up = self.w_up.to(dtype=vi_orig.dtype, device=vi_orig.device)
        w_down = self.w_down.to(dtype=vi_orig.dtype, device=vi_orig.device)

        # ===== L_spa: 空间一致性损失 =====
        org_pool = self.spa_pool(vi_orig)
        enh_pool = self.spa_pool(vi_enhanced)

        D_org_left = F.conv2d(org_pool, w_left, padding=1)
        D_org_right = F.conv2d(org_pool, w_right, padding=1)
        D_org_up = F.conv2d(org_pool, w_up, padding=1)
        D_org_down = F.conv2d(org_pool, w_down, padding=1)

        D_enh_left = F.conv2d(enh_pool, w_left, padding=1)
        D_enh_right = F.conv2d(enh_pool, w_right, padding=1)
        D_enh_up = F.conv2d(enh_pool, w_up, padding=1)
        D_enh_down = F.conv2d(enh_pool, w_down, padding=1)

        L_spa = torch.mean(
            (D_org_left - D_enh_left) ** 2 +
            (D_org_right - D_enh_right) ** 2 +
            (D_org_up - D_enh_up) ** 2 +
            (D_org_down - D_enh_down) ** 2
        )

        # ===== L_exp: 曝光控制损失（只约束暗区）=====
        dark_mask = torch.sigmoid(10.0 * (0.35 - vi_orig))
        enh_mean = self.pool(vi_enhanced)
        mask_mean = self.pool(dark_mask)
        # 只在暗区计算曝光损失
        L_exp = torch.mean(mask_mean * (enh_mean - self.mean_val) ** 2)

        # ===== L_TV: 曲线平滑性损失 =====
        b, c, h, w = curve_A.shape
        h_tv = torch.mean((curve_A[:, :, 1:, :] - curve_A[:, :, :h-1, :]) ** 2)
        w_tv = torch.mean((curve_A[:, :, :, 1:] - curve_A[:, :, :, :w-1]) ** 2)
        L_tv = h_tv + w_tv

        return L_spa + L_exp + L_tv