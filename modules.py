import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ========================== 正确的完整 FFT ==========================
def fft2(x):
    x = x.squeeze(1)
    spec = torch.fft.fft2(x, norm='ortho')
    amp = torch.abs(spec).unsqueeze(1)
    pha = torch.angle(spec).unsqueeze(1)
    return amp, pha


def ifft2(amp, pha):
    real = amp.squeeze(1) * torch.cos(pha.squeeze(1))
    imag = amp.squeeze(1) * torch.sin(pha.squeeze(1))
    spec = torch.complex(real, imag)
    img = torch.fft.ifft2(spec, norm='ortho')
    return torch.abs(img).unsqueeze(1)


# ========================== VIS 低光增强（Phase 4.3：升级版） ==========================
class LowLightEnhance(nn.Module):
    """
    Dark-region-only curve enhancement (Zero-DCE style, upgraded)
    - 加深加宽：5层64通道（原3层32通道）
    - 残差连接防止梯度消失
    - 返回增强前图像和 Alpha map 供自监督损失使用
    Assumes input in [0,1]
    """
    def __init__(self, curve_num: int = 8):
        super().__init__()
        self.curve_num = curve_num
        self.curve_net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, curve_num, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        # x: (B,1,H,W) in [0,1]
        x_orig = x  # 保存增强前的图像

        # ===== 像素级暗区 soft mask =====
        dark_mask = torch.sigmoid(10.0 * (0.35 - x))

        A = self.curve_net(x) * dark_mask
        out = x
        for i in range(self.curve_num):
            out = out + A[:, i:i+1] * out * (1.0 - out)

        out = torch.clamp(out, 0.0, 1.0)
        return out, x_orig, A


# ========================== Tokenizer ==========================
class Tokenizer(nn.Module):
    def __init__(self, patch_size=3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=1, padding=patch_size // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.unfold(x)
        tokens = tokens.transpose(1, 2)
        return tokens, b, h, w


# ========================== INN Detail Refinement ==========================
class DetailNode(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        half = dim // 2
        self.theta_phi = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1), nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1)
        )
        self.theta_rho = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1), nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1)
        )
        self.theta_eta = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1), nn.ReLU(),
            nn.Conv2d(half, half, 3, padding=1)
        )
        self.shuffle = nn.Conv2d(dim, dim, 1)

    def forward(self, z1, z2):
        z = self.shuffle(torch.cat([z1, z2], dim=1))
        z1, z2 = z.chunk(2, dim=1)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2) * 0.1) + self.theta_eta(z2) * 0.1
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3, dim=64):
        super().__init__()
        self.nodes = nn.ModuleList([DetailNode(dim) for _ in range(num_layers)])

    def forward(self, x):
        z1, z2 = x.chunk(2, dim=1)
        for node in self.nodes:
            z1, z2 = node(z1, z2)
        return torch.cat([z1, z2], dim=1)


# ========================== 空间域 DMRM ==========================
class DMRM(nn.Module):
    def __init__(self, in_ch=1, out_ch=32, dim=64):
        super().__init__()

        # ===== Edge detection (固定 Sobel 核，无参数) =====
        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # ===== Modality-specific Stems (输入: 原图+边缘图 = 2通道) =====
        self.ir_stem = nn.Sequential(
            nn.Conv2d(in_ch * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )
        self.vi_stem = nn.Sequential(
            nn.Conv2d(in_ch * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )

        self.tokenizer = Tokenizer(patch_size=3)

        # ===== Projection =====
        self.vis_proj = nn.Conv2d(dim, dim, 1)
        self.ir_proj = nn.Conv2d(dim, dim, 1)
        self.joint_proj = nn.Conv2d(dim * 2, dim, 1)

        # ===== Similarity-driven weight map =====
        self.weight_map = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )

        # ===== INN refinement =====
        self.inn_refine = DetailFeatureExtraction(num_layers=3, dim=dim)

        # ===== Output heads =====
        self.out = nn.Conv2d(dim, out_ch, 1)
        self.res = nn.Conv2d(in_ch, out_ch, 1)

        # ===== Learnable scalars =====
        self.beta = nn.Parameter(torch.tensor(0.08))    # SDM weight
        self.gamma = nn.Parameter(torch.tensor(0.08))   # Frequency guidance weight
        self.tau = nn.Parameter(torch.tensor(0.4))      # Uncertainty threshold

        # ===== Frequency alignment =====
        self.freq_align = nn.Conv2d(out_ch, dim, 1)

    def forward(self, ir, vi, frefus):
        """
        ir, vi:   (B,1,H,W)
        frefus:   (B,out_ch,H,W)  frequency-domain fused features
        """

        # ===== Edge extraction (Phase 3.2) =====
        sobel_x = self.sobel_x.to(dtype=ir.dtype)
        sobel_y = self.sobel_y.to(dtype=ir.dtype)

        ir_edge = F.conv2d(ir, sobel_x, padding=1).abs() + F.conv2d(ir, sobel_y, padding=1).abs()
        vi_edge = F.conv2d(vi, sobel_x, padding=1).abs() + F.conv2d(vi, sobel_y, padding=1).abs()

        # ===== Stem features (原图 + 边缘图拼接) =====
        ir_f = self.ir_stem(torch.cat([ir, ir_edge], dim=1))
        vi_f = self.vi_stem(torch.cat([vi, vi_edge], dim=1))

        # ===== Token similarity =====
        ir_tok, b, H, W = self.tokenizer(ir_f)
        vi_tok, _, _, _ = self.tokenizer(vi_f)

        ir_norm = F.normalize(ir_tok, dim=-1)
        vi_norm = F.normalize(vi_tok, dim=-1)
        sim = torch.sum(ir_norm * vi_norm, dim=-1, keepdim=True)
        sim_map = sim.transpose(1, 2).reshape(b, 1, H, W)   # (B,1,H,W)

        # ===== Branch weighting =====
        vis_w = self.weight_map(sim_map)
        ir_w = 1.0 - vis_w

        vis_branch = self.vis_proj(vi_f) * vis_w
        ir_branch = self.ir_proj(ir_f) * ir_w
        joint_branch = self.joint_proj(torch.cat([ir_f, vi_f], dim=1))

        f_spatial_raw = vis_branch + ir_branch + 0.5 * joint_branch

        # ===== SDM (difference injection only in low-sim regions) =====
        sdm = torch.abs(ir_f - vi_f)
        f_spatial = f_spatial_raw + self.beta * sdm * (1.0 - sim_map)

        # ============================================================
        # ★ MODIFIED PART: Frequency guidance × uncertainty
        # ============================================================

        # Frequency alignment
        freq_guidance = self.freq_align(frefus)
        freq_guidance = F.normalize(freq_guidance, dim=1)

        # Uncertainty (low similarity → high confidence for refinement)
        conf = torch.sigmoid(5.0 * (self.tau - sim_map))   # (B,1,H,W)

        # Frequency-guided modulation (only on uncertain regions)
        guidance = 1.0 + self.gamma * freq_guidance * conf
        f_spatial_guided = f_spatial * guidance

        # ============================================================

        # ===== INN refinement (only on uncertain regions) =====
        refine_region = self.inn_refine(f_spatial_guided * conf)
        pass_region = f_spatial_guided * (1.0 - conf)
        f_spatial_refined = refine_region + pass_region

        # ===== Output =====
        feat = self.out(f_spatial_refined) + self.res(ir)
        return feat


# ========================== 频率域 DHLF ==========================
class DecoupledHLFuse(nn.Module):
    def __init__(self, channel=8, init_cutoff=0.12):
        super().__init__()
        self.channel = channel
        self.cutoff = nn.Parameter(torch.tensor(init_cutoff))

        self.low_enhance = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )
        # VIS 高频门控（输出1通道 vi_gate）
        self.high_denoise = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        # IR 高频门控（独立网络，替代硬编码衰减）
        self.ir_high_gate = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )
        self.pha_adjust = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(1, channel * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(channel * 2, channel, 3, padding=1), nn.GELU(),
            nn.Conv2d(channel, channel, 1), nn.ReLU(inplace=True)
        )

    def forward(self, ir_amp, ir_pha, vi_amp, vi_pha):
        b, _, h, w = ir_amp.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=ir_amp.device),
            torch.linspace(-1, 1, w, device=ir_amp.device),
            indexing='ij'
        )
        dist = torch.sqrt(xx ** 2 + yy ** 2)
        low_mask = torch.sigmoid(60 * (self.cutoff.clamp(0.05, 0.45) - dist))
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = 1.0 - low_mask

        ir_low, ir_high = ir_amp * low_mask, ir_amp * high_mask
        vi_low, vi_high = vi_amp * low_mask, vi_amp * high_mask

        low_weight = self.low_enhance(torch.cat([vi_low, ir_low], dim=1))
        fused_low = ir_low * low_weight + vi_low * (1 - low_weight)

        high_cat = torch.cat([vi_high, ir_high], dim=1)
        vi_gate = torch.sigmoid(self.high_denoise(high_cat))
        ir_gate = self.ir_high_gate(high_cat)

        ir_residual = ir_high * ir_gate

        fused_high = vi_high * vi_gate + ir_residual

        fused_amp = fused_low + fused_high
        pha_delta = torch.tanh(self.pha_adjust(torch.cat([vi_pha, ir_pha], dim=1)))
        fused_pha = vi_pha + pha_delta[:, 1:2] * low_mask * 0.4

        intensity = ifft2(fused_amp, fused_pha)
        frefus = self.refine(intensity)
        return frefus, fused_amp, fused_pha, high_mask


# ========================== 融合头 ==========================
class FuseBlock(nn.Module):
    def __init__(self, dim=24, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, spatial_f, frefus):
        x = torch.cat([spatial_f, frefus], dim=1)
        out = self.net(x)
        # per-sample 归一化：每张图独立计算 min/max，训练与推理行为一致
        b = out.shape[0]
        out_min = out.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        out_max = out.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        out = (out - out_min) / (out_max - out_min + 1e-8)
        return out


# ========================== 主网络 ==========================
class FusionNet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.channel = channel
        self.llie = LowLightEnhance(curve_num=8)
        self.dmrm = DMRM(in_ch=1, out_ch=channel)
        self.freq_fuse = DecoupledHLFuse(channel=channel)
        self.fuse_block = FuseBlock(dim=channel * 2)

    def forward(self, ir, vi):
        vi_enhanced, vi_orig, curve_A = self.llie(vi)
        ir_amp, ir_pha = fft2(ir)
        vi_amp, vi_pha = fft2(vi_enhanced)
        frefus, fused_amp, fused_pha, high_mask = self.freq_fuse(ir_amp, ir_pha, vi_amp, vi_pha)
        spatial_feat = self.dmrm(ir, vi_enhanced, frefus)
        fusion = self.fuse_block(spatial_feat, frefus)
        return fusion, fused_amp, fused_pha, high_mask, ir_amp, vi_amp, vi_orig, vi_enhanced, curve_A