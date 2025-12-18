from torch import nn


def get_param_groups(module) -> tuple:
    group = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers
    for v in module.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            "bias"
            group[2].append(v.bias)
        if isinstance(v, bn):
            "weight (no decay)"
            group[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            "weight (with decay)"
            group[0].append(v.weight)
    return group
 # PyTorch 模型参数的精细化分组，核心目标是为不同类型的参数（普通权重、归一化层权重、偏置）分配不同的优化策略（尤其是权重衰减），是深度学习训练中优化器配置的常用技巧，有助于提升模型训练效果。