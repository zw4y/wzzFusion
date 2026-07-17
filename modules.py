import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== 中心化 FFT ==========================
def fft2(x):
    x = x.squeeze(1)
    spec = torch.fft.fft2(x, norm='ortho')
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    amp = torch.abs(spec).unsqueeze(1)
    pha = torch.angle(spec).unsqueeze(1)
    return amp, pha


def ifft2(amp, pha):
    real = amp.squeeze(1) * torch.cos(pha.squeeze(1))
    imag = amp.squeeze(1) * torch.sin(pha.squeeze(1))
    spec = torch.complex(real, imag)
    spec = torch.fft.ifftshift(spec, dim=(-2, -1))
    img = torch.fft.ifft2(spec, norm='ortho')
    return torch.abs(img).unsqueeze(1)


# ========================== VIS 低光增强（简化自适应 soft mask 版） ==========================
class LowLightEnhance(nn.Module):

    def __init__(
        self,
        curve_num: int = 8,
        base_tau: float = 0.35,
        k: float = 10.0,
        tau_min: float = 0.28,
        tau_max: float = 0.45,
        adapt_ratio: float = 0.25,
    ):
        super().__init__()
        self.curve_num = curve_num
        self.base_tau = base_tau
        self.k = k
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.adapt_ratio = adapt_ratio

        # 与原始版本保持一致：5层64通道，最后用 Tanh 预测曲线参数
        self.curve_net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, curve_num, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        # x: (B,1,H,W), in [0,1]
        x_orig = x

        # ============================================================
        # 1) Image-level adaptive threshold
        # ============================================================
        # 图像越暗，tau 越高，使更多区域参与增强；
        # 图像越亮，tau 越低，减少正常曝光区域的不必要增强。
        illum_mean = x.mean(dim=(2, 3), keepdim=True).detach()
        tau = self.base_tau + self.adapt_ratio * (self.base_tau - illum_mean)
        tau = tau.clamp(self.tau_min, self.tau_max)

        # ============================================================
        # 2) Adaptive soft dark mask
        # ============================================================
        # 形式仍然接近原始公式 M = sigmoid(k * (tau - Ivi))，
        # 区别是 tau 由每张图像的全局亮度自适应确定。
        enhance_mask = torch.sigmoid(self.k * (tau - x))

        # ============================================================
        # 3) Curve enhancement
        # ============================================================
        A = self.curve_net(x) * enhance_mask

        out = x
        for i in range(self.curve_num):
            out = out + A[:, i:i + 1] * out * (1.0 - out)

        out = torch.clamp(out, 0.0, 1.0)
        return out, x_orig, A, enhance_mask


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


# ========================== Swin Cross-Attention 模块 ==========================

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口还原为特征图
    Args:
        windows: (B * num_windows, window_size, window_size, C)
        window_size: int
        H, W: 特征图高宽
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowCrossAttention(nn.Module):
    """
    窗口内的 Cross-Attention
    query_feat 出 Q，context_feat 出 K/V
    包含可学习的相对位置编码
    """
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        # ===== 相对位置编码 =====
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
        coords_flat = coords.view(2, -1)  # (2, ws*ws)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, query_feat, context_feat):
        """
        Args:
            query_feat:   (B*num_windows, window_size*window_size, dim)
            context_feat: (B*num_windows, window_size*window_size, dim)
        Returns:
            out: (B*num_windows, window_size*window_size, dim)
        """
        Bw, N, C = query_feat.shape

        q = self.q_proj(query_feat).reshape(Bw, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context_feat).reshape(Bw, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 加入相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    """
    双向 Cross-Attention + FFN
    IR tokens attend to VI tokens，VI tokens attend to IR tokens
    """
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.norm_ir1 = nn.LayerNorm(dim)
        self.norm_vi1 = nn.LayerNorm(dim)
        self.cross_attn_ir = WindowCrossAttention(dim, num_heads, window_size)
        self.cross_attn_vi = WindowCrossAttention(dim, num_heads, window_size)

        self.norm_ir2 = nn.LayerNorm(dim)
        self.norm_vi2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn_ir = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )
        self.ffn_vi = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )

    def forward(self, ir_tokens, vi_tokens):
        """
        Args:
            ir_tokens: (B*nW, N, dim)
            vi_tokens: (B*nW, N, dim)
        Returns:
            ir_tokens, vi_tokens: 更新后的 tokens
        """
        # 双向 cross-attention
        ir_normed = self.norm_ir1(ir_tokens)
        vi_normed = self.norm_vi1(vi_tokens)
        ir_cross = self.cross_attn_ir(ir_normed, vi_normed)   # IR query, VI context
        vi_cross = self.cross_attn_vi(vi_normed, ir_normed)   # VI query, IR context
        ir_tokens = ir_tokens + ir_cross
        vi_tokens = vi_tokens + vi_cross

        # FFN
        ir_tokens = ir_tokens + self.ffn_ir(self.norm_ir2(ir_tokens))
        vi_tokens = vi_tokens + self.ffn_vi(self.norm_vi2(vi_tokens))

        return ir_tokens, vi_tokens


class SwinCrossFusion(nn.Module):
    """
    Swin 风格的窗口 Cross-Attention 融合模块

    特性：
    - 窗口内双向 cross-attention（IR↔VI）
    - Shifted window 实现跨窗口信息交换
    - sim_map 通过交互后特征的余弦相似度计算（数值行为与原始版本一致）
    - 相对位置编码，分辨率无关（训练320测试640无需适配）
    """
    def __init__(self, dim=64, num_heads=4, window_size=8, num_layers=2):
        super().__init__()
        self.window_size = window_size
        self.num_layers = num_layers

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, window_size) for _ in range(num_layers)
        ])

        # 融合投影：将 cross-attention 交互后的 IR/VI 特征合并
        self.fuse_proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, ir_f, vi_f):
        """
        Args:
            ir_f: (B, C, H, W) — IR stem 输出的特征
            vi_f: (B, C, H, W) — VI stem 输出的特征
        Returns:
            f_spatial_raw: (B, C, H, W) — 融合后的空间特征
            sim_map:       (B, 1, H, W) — 相似度图（交互后特征的余弦相似度）
        """
        B, C, H, W = ir_f.shape
        ws = self.window_size

        # ===== Padding（确保 H, W 能被 window_size 整除）=====
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            ir_f = F.pad(ir_f, (0, pad_w, 0, pad_h))
            vi_f = F.pad(vi_f, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = ir_f.shape

        # (B, C, H, W) → (B, H, W, C)
        ir_x = ir_f.permute(0, 2, 3, 1).contiguous()
        vi_x = vi_f.permute(0, 2, 3, 1).contiguous()

        for i, blk in enumerate(self.blocks):
            # 奇数层做 shifted window
            shift = ws // 2 if i % 2 == 1 else 0

            if shift > 0:
                ir_shifted = torch.roll(ir_x, shifts=(-shift, -shift), dims=(1, 2))
                vi_shifted = torch.roll(vi_x, shifts=(-shift, -shift), dims=(1, 2))
            else:
                ir_shifted = ir_x
                vi_shifted = vi_x

            # Window partition → (B*nW, ws*ws, C)
            ir_windows = window_partition(ir_shifted, ws).view(-1, ws * ws, C)
            vi_windows = window_partition(vi_shifted, ws).view(-1, ws * ws, C)

            # Cross-attention
            ir_windows, vi_windows = blk(ir_windows, vi_windows)

            # Window reverse
            ir_shifted = window_reverse(ir_windows.view(-1, ws, ws, C), ws, Hp, Wp)
            vi_shifted = window_reverse(vi_windows.view(-1, ws, ws, C), ws, Hp, Wp)

            # Reverse shift
            if shift > 0:
                ir_x = torch.roll(ir_shifted, shifts=(shift, shift), dims=(1, 2))
                vi_x = torch.roll(vi_shifted, shifts=(shift, shift), dims=(1, 2))
            else:
                ir_x = ir_shifted
                vi_x = vi_shifted

        # (B, H, W, C) → (B, C, H, W)
        ir_out = ir_x.permute(0, 3, 1, 2).contiguous()
        vi_out = vi_x.permute(0, 3, 1, 2).contiguous()

        # 去除 padding
        if pad_h > 0 or pad_w > 0:
            ir_out = ir_out[:, :, :H, :W]
            vi_out = vi_out[:, :, :H, :W]

        # 融合
        f_spatial_raw = self.fuse_proj(torch.cat([ir_out, vi_out], dim=1))

        # ===== sim_map: 交互后特征的逐像素余弦相似度 =====
        # cross-attention 交互后的特征包含跨模态上下文信息
        # 余弦相似度范围 [-1, 1]，与原始版本的 sim_map 数值行为一致
        # SDM 和 INN 的 (1-sim_map) 和 conf 可以正常工作
        sim_map = F.cosine_similarity(ir_out, vi_out, dim=1).unsqueeze(1)  # (B, 1, H, W)

        return f_spatial_raw, sim_map


# ========================== 空间域 SASF（Swin Cross-Attention 版）==========================
class SASF(nn.Module):
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

        # ===== Swin Cross-Attention 融合 =====
        self.cross_fusion = SwinCrossFusion(dim=dim, num_heads=4, window_size=8, num_layers=2)

        # ===== Learnable scalars =====
        self.beta = nn.Parameter(torch.tensor(0.08))    # SDM weight
        self.gamma = nn.Parameter(torch.tensor(0.08))   # Frequency guidance weight
        self.tau = nn.Parameter(torch.tensor(0.4))       # Uncertainty threshold

        # ===== INN refinement =====
        self.inn_refine = DetailFeatureExtraction(num_layers=3, dim=dim)

        # ===== Output heads =====
        self.out = nn.Conv2d(dim, out_ch, 1)
        self.res = nn.Conv2d(in_ch, out_ch, 1)

        # ===== Frequency alignment =====
        self.freq_align = nn.Conv2d(out_ch, dim, 1)

    def forward(self, ir, vi, frefus):
        """
        Args:
            ir:     (B, 1, H, W) — 红外图像
            vi:     (B, 1, H, W) — 可见光图像（已增强）
            frefus: (B, out_ch, H, W) — 频率域融合特征
        Returns:
            feat:   (B, out_ch, H, W) — 空间域融合特征
        """

        # ===== Edge extraction =====
        sobel_x = self.sobel_x.to(dtype=ir.dtype)
        sobel_y = self.sobel_y.to(dtype=ir.dtype)

        ir_edge = F.conv2d(ir, sobel_x, padding=1).abs() + F.conv2d(ir, sobel_y, padding=1).abs()
        vi_edge = F.conv2d(vi, sobel_x, padding=1).abs() + F.conv2d(vi, sobel_y, padding=1).abs()

        ir_f = self.ir_stem(torch.cat([ir, ir_edge], dim=1))
        vi_f = self.vi_stem(torch.cat([vi, vi_edge], dim=1))

        # Cross-attention 融合（核心，不变）
        f_spatial_raw, sim_map = self.cross_fusion(ir_f, vi_f)
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
        fy = 2.0 * torch.fft.fftshift(
            torch.fft.fftfreq(h, device=ir_amp.device)
        )
        fx = 2.0 * torch.fft.fftshift(
            torch.fft.fftfreq(w, device=ir_amp.device)
        )
        yy, xx = torch.meshgrid(
            fy,
            fx,
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
        self.sasf = SASF(in_ch=1, out_ch=channel)
        self.freq_fuse = DecoupledHLFuse(channel=channel)
        self.fuse_block = FuseBlock(dim=channel * 2)

    def forward(self, ir, vi):
        vi_enhanced, vi_orig, curve_A, enhance_mask = self.llie(vi)
        ir_amp, ir_pha = fft2(ir)
        vi_amp, vi_pha = fft2(vi_enhanced)
        frefus, fused_amp, fused_pha, high_mask = self.freq_fuse(ir_amp, ir_pha, vi_amp, vi_pha)
        spatial_feat = self.sasf(ir, vi_enhanced, frefus)
        fusion = self.fuse_block(spatial_feat, frefus)
        return fusion, fused_amp, fused_pha, high_mask, ir_amp, vi_amp, vi_orig, vi_enhanced, curve_A, enhance_mask
