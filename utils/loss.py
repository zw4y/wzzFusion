import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ifft2


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [0.,  0.,  0.],
                                 [1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('weight_x', kernel_x.repeat(1, 1, 1, 1))
        self.register_buffer('weight_y', kernel_y.repeat(1, 1, 1, 1))

    def forward(self, x):

        weight_x = self.weight_x.to(x.device)
        weight_y = self.weight_y.to(x.device)
        gx = F.conv2d(x, weight_x, padding=1, groups=x.shape[1])
        gy = F.conv2d(x, weight_y, padding=1, groups=x.shape[1])
        return torch.abs(gx) + torch.abs(gy)


class PixelGradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, fus_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, fus_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        fus_grad = self.sobelconv(fus_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, fus_grad)

        return 5 * loss_in + 10 * loss_grad



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