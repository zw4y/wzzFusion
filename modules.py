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


# ========================== Tokenizer (3x3 unfold) ==========================
class Tokenizer(nn.Module):
    def __init__(self, patch_size=3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=1, padding=patch_size//2)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = self.unfold(x)              # (B, C*p*p, L)
        tokens = tokens.transpose(1, 2)      # (B, L, C*p*p)
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
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )

        self.tokenizer = Tokenizer(patch_size=3)

        self.vis_proj = nn.Conv2d(dim, dim, 1)
        self.ir_proj = nn.Conv2d(dim, dim, 1)
        self.joint_proj = nn.Conv2d(dim*2, dim, 1)

        self.weight_map = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )

        # INN for uncertainty refinement
        self.inn_refine = DetailFeatureExtraction(num_layers=3, dim=dim)

        self.out = nn.Conv2d(dim, out_ch, 1)
        self.res = nn.Conv2d(in_ch, out_ch, 1)

        # 可学习参数
        self.beta = nn.Parameter(torch.tensor(0.08))   # SDM 权重
        self.gamma = nn.Parameter(torch.tensor(0.08))  # Frequency Guidance 权重
        self.tau = nn.Parameter(torch.tensor(0.4))     # 可学习不确定阈值（更强！）

        # 频率域特征对齐投影
        self.freq_align = nn.Conv2d(out_ch, dim, 1)     # frefus (out_ch) → dim

    def forward(self, ir, vi, frefus):
        ir_f = self.stem(ir)
        vi_f = self.stem(vi)

        # Token 相似度结构对齐
        ir_tok, b, H, W = self.tokenizer(ir_f)
        vi_tok, _, _, _ = self.tokenizer(vi_f)

        ir_norm = F.normalize(ir_tok, dim=-1)
        vi_norm = F.normalize(vi_tok, dim=-1)
        sim = torch.sum(ir_norm * vi_norm, dim=-1, keepdim=True)
        sim_map = sim.transpose(1, 2).reshape(b, 1, H, W)  # (B,1,H,W)

        # 三分支融合
        vis_w = self.weight_map(sim_map)
        ir_w = 1 - vis_w

        vis_branch = self.vis_proj(vi_f) * vis_w
        ir_branch = self.ir_proj(ir_f) * ir_w
        joint_branch = self.joint_proj(torch.cat([ir_f, vi_f], dim=1))

        f_spatial_raw = vis_branch + ir_branch + 0.5 * joint_branch

        # SDM 只在低相似区域注入（更聪明）
        sdm = torch.abs(ir_f - vi_f)
        f_spatial = f_spatial_raw + self.beta * sdm * (1 - sim_map)

        # 频率域一致性引导（加对齐投影！）
        freq_guidance = self.freq_align(frefus)  # (B,out_ch,H,W) → (B,dim,H,W)
        guidance = 1 + self.gamma * F.normalize(freq_guidance, dim=1)
        f_spatial_guided = f_spatial * guidance

        # soft confidence mask（梯度友好）
        conf = torch.sigmoid(5 * (self.tau - sim_map))   # soft [0,1]
        refine_region = self.inn_refine(f_spatial_guided * conf)
        pass_region = f_spatial_guided * (1 - conf)
        f_spatial_refined = refine_region + pass_region

        # 输出头 + 残差
        feat = self.out(f_spatial_refined) + self.res(ir)
        return feat, feat


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
        self.high_denoise = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        self.pha_adjust = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(1, channel*2, 3, padding=1), nn.GELU(),
            nn.Conv2d(channel*2, channel, 3, padding=1), nn.GELU(),
            nn.Conv2d(channel, channel, 1), nn.ReLU(inplace=True)
        )

    def forward(self, ir_amp, ir_pha, vi_amp, vi_pha):
        b, _, h, w = ir_amp.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=ir_amp.device),
            torch.linspace(-1, 1, w, device=ir_amp.device),
            indexing='ij'
        )
        dist = torch.sqrt(xx**2 + yy**2)
        low_mask = torch.sigmoid(60 * (self.cutoff.clamp(0.05, 0.45) - dist))
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = 1.0 - low_mask

        ir_low, ir_high = ir_amp * low_mask, ir_amp * high_mask
        vi_low, vi_high = vi_amp * low_mask, vi_amp * high_mask

        low_weight = self.low_enhance(torch.cat([vi_low, ir_low], dim=1))
        fused_low = ir_low * low_weight + vi_low * (1 - low_weight)

        high_cat = torch.cat([vi_high, ir_high], dim=1)
        gates = torch.sigmoid(self.high_denoise(high_cat))
        vi_gate = gates[:, 0:1]
        ir_suppress = gates[:, 1:2]

        ir_residual = ir_high * (1.0 - ir_suppress)
        ir_residual = torch.clamp(ir_residual, max=0.06)
        ir_residual = ir_residual * 0.05
        ir_residual = F.avg_pool2d(ir_residual, 3, stride=1, padding=1) * 0.85

        fused_high = vi_high * vi_gate + ir_residual

        fused_amp = fused_low + fused_high
        pha_delta = torch.tanh(self.pha_adjust(torch.cat([vi_pha, ir_pha], dim=1)))
        fused_pha = vi_pha + pha_delta[:, 1:2] * low_mask * 0.4

        intensity = ifft2(fused_amp, fused_pha)
        frefus = self.refine(intensity)
        return frefus, fused_amp, fused_pha


# ========================== 融合头 ==========================
class FuseBlock(nn.Module):
    def __init__(self, dim=24, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch*2, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, ir_f, vi_f, frefus):
        x = torch.cat([ir_f, vi_f, frefus], dim=1)
        out = self.net(x)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out


# ========================== 主网络（关键：DMRM forward 传 frefus）==========================
class FusionNet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.channel = channel
        self.dmrm = DMRM(in_ch=1, out_ch=channel)
        self.freq_fuse = DecoupledHLFuse(channel=channel)
        self.fuse_block = FuseBlock(dim=channel*3)

    def forward(self, ir, vi):
        ir_amp, ir_pha = fft2(ir)
        vi_amp, vi_pha = fft2(vi)

        frefus, fused_amp, fused_pha = self.freq_fuse(ir_amp, ir_pha, vi_amp, vi_pha)

        # 关键：把 frefus 传进空间域，实现频率引导
        ir_feat, vi_feat = self.dmrm(ir, vi, frefus)

        fusion = self.fuse_block(ir_feat, vi_feat, frefus)
        return fusion, fused_amp, fused_pha