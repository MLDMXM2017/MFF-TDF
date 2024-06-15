import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional, Type, Union

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'std'], k_size=3):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.weight = nn.Parameter(torch.rand(2))

        self.pool_types = pool_types
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        weight = torch.sigmoid(self.weight)
        out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        out = out.permute(0, 3, 2, 1).contiguous() 
        out = self.conv(out)   
        out = out.permute(0, 3, 2, 1).contiguous() 
        scale = self.sigmoid(out)
        return scale

class SpatialGate(nn.Module):
    def __init__(self, pool_types, k_size=3):
        super(SpatialGate, self).__init__()
        self.kernel_size = k_size
        self.weight = nn.Parameter(torch.rand(2))
        self.conv = nn.Sequential(nn.Conv2d(1, 1, self.kernel_size, stride=1, padding= (self.kernel_size - 1) // 2, bias=False),
                                     nn.AvgPool2d(3, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_Avg = x.mean(1, keepdim=True)
        x_Std = x.std(1, keepdim=True)
        # x_Max, _ = x.max(1, keepdim=True)
        weight = torch.sigmoid(self.weight)
        out = (0.5 + weight[0]) * x_Avg + (0.5 + weight[1]) * x_Std
        out = self.conv(out) 
        scale = self.sigmoid(out) # broadcasting
        return scale

class SelfAtten(nn.Module):
    def __init__(self, gate_channels, kernel_size=3, pool_types=['avg', 'std']):
        super(SelfAtten, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, pool_types, kernel_size)
        self.SpatialGate = SpatialGate(pool_types, kernel_size)
    def forward(self, x):
        scale_c = self.ChannelGate(x)
        scale_s = self.SpatialGate(x)
        scale = 0.3333333 * scale_c + 0.666666 * scale_s
        return scale * x

def replace_relu_with_relu6(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU6(inplace=True))
        else:
            replace_relu_with_relu6(child)

def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=7,
        stride=stride,
        padding=dilation + 2,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=dilation + 1,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) if kernel_size == 3 else (
                     conv5x5(inplanes, planes, stride) if kernel_size == 5 else 
                     conv7x7(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes) if kernel_size == 3 else (
                     conv5x5(planes, planes) if kernel_size == 5 else 
                     conv7x7(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride        
        self.attention = SelfAtten(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention(out) # !!!!!!

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) if kernel_size == 3 else (
                     conv5x5(width, width, stride, groups, dilation) if kernel_size == 5 else 
                     conv7x7(width, width, stride, groups, dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention = SelfAtten(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.attention(out) # !!!!!!!

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module): 
    def __init__(self, 
                 kernel_size: int, 
                 layers: list, 
                 zero_init_residual: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, kernel_size, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, kernel_size, 128, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        kernel_size: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                kernel_size, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    kernel_size,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) * mask

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class TIPNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.base_1 = ResNet(3, [1, 2])
        self.base_2 = ResNet(5, [1, 2]) 
        self.base_3 = ResNet(7, [1, 2]) 
        self.mask_proess = nn.MaxPool2d(8, 8)

    def forward(self, X):
        mask0 = torch.where(X[:, 1, :, :] > -1.7, 1.0, 0.0).unsqueeze(1)
        mask1 = self.mask_proess(mask0)
        mask_rate = mask1.shape[3] * mask1.shape[2] / mask1.sum(3).sum(2)
        X1 = self.base_1(X, mask1)
        X2 = self.base_2(X, mask1)
        X3 = self.base_3(X, mask1)
        f = torch.concatenate((X1, X2, X3), dim=1) * mask_rate
        return f

def init_weights(net):
    for m in list(net.modules()):
        if m not in list(net.base.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels + m.out_channels) / m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape 
                init_range = math.sqrt(6.0 / (n[0] + n[1])) # 23.10.20 Changed
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
    net.embed[3].weight.data.zero_()
    net.embed[3].bias.data.zero_()
    
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std

class TDF(nn.Module): 
    def __init__(self, num_features, num_labels, drop_rate):
        super().__init__()
        self.num_features = num_features
        mid = 512
        out1 = 512
        out2 = 512
        out3 = 512

        self.base = TIPNet()
        self.embed = nn.Sequential(nn.Linear(3, mid),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(drop_rate),
                                    nn.Linear(mid, (out1 + out2 + out3)), # 0_init
                                    nn.Sigmoid())
        self.embed_concat = nn.Sequential(nn.Linear(num_features - 3, mid),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(drop_rate),
                                    nn.Linear(mid, mid),
                                    nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(drop_rate),
                                    nn.Linear((out1 + out2 + out3) + mid, num_labels))

    def forward(self, X, X_feature):
        X1 = self.base(X)
        age_and_sex = torch.concatenate([X_feature[:, :1], X_feature[:, -2:]], 1) # Age and Sex in 0 and last 2
        Xi = X1 * (2 * self.embed(age_and_sex))
        Xc = self.embed_concat(X_feature[:, 1:-2]) # Others
        X3 = torch.concatenate([Xi, Xc], 1)
        X4 = self.head(X3)
        return X4

def get_net(num_features, num_labels, drop_rate):
    net = TDF(num_features, num_labels, drop_rate)
    init_weights(net)
    return net
