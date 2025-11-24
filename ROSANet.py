import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import sys, os
from cc_attention import CrissCrossAttention
from utils.pyt_utils import load_model
# from inplace_abn import _backend
from inplace_abn._backend import *
from inplace_abn import InPlaceABN, InPlaceABNSync
####################################################################
from ARConv import ARConv
from models.exchange_modules import *



BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def predict_whole(outs, tile_size):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    if isinstance(outs, list):
        outs = outs[0]
    prediction = interp(outs)
    return prediction


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out







class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_module, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                             stride=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=6, dilation=6, bias=False),
                                   InPlaceABNSync(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=12, dilation=12, bias=False),
                                   InPlaceABNSync(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=18, dilation=18, bias=False),
                                   InPlaceABNSync(planes))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        return torch.cat((x0, x1, x2, x3), dim=1)


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output
###############
# class ARConv_Block(nn.Module):
#     def __init__(self, in_planes, flag=False):
#         super(ARConv_Block, self).__init__()
#         self.head_conv = nn.Conv2d(in_planes*2, in_planes, 3, 1, 1)
#         self.flag = flag
#         self.conv1 = ARConv(in_planes, in_planes, 3, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = ARConv(in_planes, in_planes, 3, 1, 1)
#
#     def forward(self, x, epoch=500, hw_range=[0,18]):
#         x_proj = self.head_conv(x)
#         res = self.conv1(x_proj, epoch, hw_range)
#         res = self.relu(res)
#         res = self.conv2(res, epoch, hw_range)
#         x = x_proj + res
#         return x

class ARConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(ARConv_Block, self).__init__()
        self.head_conv = nn.Conv2d(in_planes * 2, in_planes, 3, 1, 1)
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.flag = flag
        self.conv1 = ARConv(in_planes, in_planes, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ARConv(in_planes, in_planes, 3, 1, 1)

    def forward(self, x, epoch=500, hw_range=[0, 18]):
        x_proj = self.head_conv(x)
        self.bn1(x_proj)
        self.relu1(x_proj)
        res = self.conv1(x_proj, epoch, hw_range)
        res = self.relu2(res)
        res = self.conv2(res, epoch, hw_range)
        x = x_proj + res
        return x

class ARConv_Block_Lite(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(ARConv_Block_Lite, self).__init__()
        self.head_conv = nn.Conv2d(in_planes * 2, in_planes, 3, 1, 1)
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.flag = flag
        self.conv1 = ARConv(in_planes, in_planes, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ARConv(in_planes, in_planes, 3, 1, 1)

    def forward(self, x, epoch=500, hw_range=[0, 18]):
        x_proj = self.head_conv(x)
        self.bn1(x_proj)
        self.relu1(x_proj)
        res = self.conv1(x_proj, epoch, hw_range)
        res = self.relu2(res)
        res = self.conv2(res, epoch, hw_range)
        x = x_proj + res
        return x
###############
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]
class Featureselect(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=126,num_parallel=6, bn_threshold=2e-2):
            super(Featureselect, self).__init__()
            self.inp_dim = in_channels
            self.num_parallel = num_parallel
            self.bn_threshold = bn_threshold
            self.exchange = Exchange6(in_channels)
            self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
            self.conv1 = ModuleParallel(
                nn.Conv2d(mid_channels//num_parallel, mid_channels//num_parallel, kernel_size=3, stride=1, padding=1, bias=True))
            self.conv2 = ModuleParallel(
                nn.Conv2d(mid_channels//num_parallel, mid_channels//num_parallel, kernel_size=3, stride=1, padding=1, bias=True))
            self.relu = ModuleParallel(nn.ReLU(inplace=False))
            self.bn1 = BatchNorm2dParallel(mid_channels//num_parallel,num_parallel=6)
            self.bn2 = BatchNorm2dParallel(mid_channels//num_parallel,num_parallel=6)

            self.bn2_list = []
            for module in self.bn2.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn2_list.append(module)
            self.restore_block = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )
    def forward(self, x):
        out=self.conv0(x) #变成126通道方便进行切分
        out = torch.chunk(out, 6, dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = torch.cat(out, dim=1)
        out = self.restore_block(out)

        return out

class Featureselect2(nn.Module):
    def __init__(self, in_channels , num_parallel=2, bn_threshold=2e-2):
        super(Featureselect2, self).__init__()
        self.inp_dim = in_channels
        self.num_parallel = num_parallel
        self.bn_threshold = bn_threshold
        self.exchange = Exchange()

        self.conv1 = ModuleParallel(
            nn.Conv2d(in_channels , in_channels ,kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = ModuleParallel(
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, stride=1,padding=1, bias=True))
        self.relu = ModuleParallel(nn.ReLU(inplace=False))
        self.bn1 = BatchNorm2dParallel(in_channels, num_parallel=2)
        self.bn2 = BatchNorm2dParallel(in_channels, num_parallel=2)

        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x,y):
        out = [x,y]
        residual = out

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)
        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)
        x, y = out[0],out[1]


        return x,y

# class ChannelReplacer(nn.Module):
#     def __init__(self, thresh=1e-3, verbose=False, stage_name="stage", log_dir=r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录"):
#         super(ChannelReplacer, self).__init__()
#         self.thresh = thresh
#         self.verbose = verbose
#         self.stage_name = stage_name
#         self.log_dir = log_dir
#
#         # 确保目录存在
#         os.makedirs(self.log_dir, exist_ok=True)
#
#     def forward(self, x, y):
#         B, C, H, W = x.shape
#         x_max = x.view(B, C, -1).max(dim=2)[0]
#         y_max = y.view(B, C, -1).max(dim=2)[0]
#
#         x_mask = (x_max < self.thresh).float().unsqueeze(-1).unsqueeze(-1)
#         y_mask = (y_max < self.thresh).float().unsqueeze(-1).unsqueeze(-1)
#
#         if self.verbose:
#             with torch.no_grad():
#                 # 找出被替换通道的最大值（数值）
#                 x_low_values = x_max[x_max < self.thresh].cpu().numpy()
#                 y_low_values = y_max[y_max < self.thresh].cpu().numpy()
#
#                 # 构建输出日志
#                 log_str = (
#                     f"[Replace Info - {self.stage_name}] x_low_values: {x_low_values.tolist()}\n"
#                     f"[Replace Info - {self.stage_name}] y_low_values: {y_low_values.tolist()}\n"
#                 )
#
#                 log_file = os.path.join(self.log_dir, f"filter_{self.stage_name}.txt")
#                 os.makedirs(self.log_dir, exist_ok=True)
#                 with open(log_file, 'a', encoding='utf-8') as f:
#                     f.write(log_str)
#
#         x_new = x * (1 - x_mask) + y * x_mask
#         y_new = y * (1 - y_mask) + x * y_mask
#
#         return x_new, y_new
class ChannelReplacer(nn.Module):
    def __init__(self, thresh=1e-3, verbose=False, log_dir=None, stage_name="stage1"):
        super(ChannelReplacer, self).__init__()
        self.thresh = thresh
        self.verbose = verbose
        self.log_dir = log_dir
        self.stage_name = stage_name
        self.x_max_replaced_values = []
        self.y_max_replaced_values = []

    def forward(self, x, y):
        B, C, H, W = x.shape
        x_max = x.view(B, C, -1).max(dim=2)[0]  # (B, C)
        y_max = y.view(B, C, -1).max(dim=2)[0]

        x_mask = (x_max < self.thresh)  # (B, C)
        y_mask = (y_max < self.thresh)  # (B, C)

        # 替换操作
        x_mask_ = x_mask.float().unsqueeze(-1).unsqueeze(-1)
        y_mask_ = y_mask.float().unsqueeze(-1).unsqueeze(-1)
        x_new = x * (1 - x_mask_) + y * x_mask_
        y_new = y * (1 - y_mask_) + x * y_mask_
        if self.verbose:
            # ✅ 保存最大值信息
            self.save_replacement_values(
                x_max.unsqueeze(-1).unsqueeze(-1),
                y_max.unsqueeze(-1).unsqueeze(-1),
                x_mask.unsqueeze(-1).unsqueeze(-1),
                y_mask.unsqueeze(-1).unsqueeze(-1)
            )

        return x_new, y_new

    def save_replacement_values(self, x_max, y_max, x_mask, y_mask):
        x_vals = x_max[x_mask.bool()].detach().cpu().numpy().tolist()
        y_vals = y_max[y_mask.bool()].detach().cpu().numpy().tolist()
        self.x_max_replaced_values.append(x_vals)
        self.y_max_replaced_values.append(y_vals)

    def write_epoch_log(self, epoch=None):
        if not self.verbose:
            return
        if not self.x_max_replaced_values and not self.y_max_replaced_values:
            return

        # 展平所有数据
        all_x_vals = [v for step in self.x_max_replaced_values for v in step]
        all_y_vals = [v for step in self.y_max_replaced_values for v in step]

        log_str = ""
        if epoch is not None:
            log_str += f"Epoch {epoch}:\n"
        log_str += f"[Replace Info - {self.stage_name}] x_channels: {len(all_x_vals)}\n"
        log_str += f"[Replace Info - {self.stage_name}] x_maxpool_vals: {[round(v, 6) for v in all_x_vals]}\n"
        log_str += f"[Replace Info - {self.stage_name}] y_channels: {len(all_y_vals)}\n"
        log_str += f"[Replace Info - {self.stage_name}] y_maxpool_vals: {[round(v, 6) for v in all_y_vals]}\n"

        log_file = os.path.join(self.log_dir, f"filter_{self.stage_name}.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')

        self.x_max_replaced_values.clear()
        self.y_max_replaced_values.clear()

class Featureselect3(nn.Module):  #加入了激活稀疏
        def __init__(self, in_channels, num_parallel=2, bn_threshold=2e-2,maxpool_thresh=1e-3,verbose=True, stage_name='stage1',log_dir = r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录"):
            super(Featureselect3, self).__init__()
            self.inp_dim = in_channels
            self.num_parallel = num_parallel
            self.bn_threshold = bn_threshold

            self.replacer = ChannelReplacer(thresh=maxpool_thresh,verbose=verbose,log_dir=log_dir,stage_name = stage_name)
            self.exchange = Exchange()

            self.conv1 = ModuleParallel(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            self.conv2 = ModuleParallel(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            self.relu = ModuleParallel(nn.ReLU(inplace=False))
            self.bn1 = BatchNorm2dParallel(in_channels, num_parallel=2)
            self.bn2 = BatchNorm2dParallel(in_channels, num_parallel=2)

            self.bn2_list = []
            for module in self.bn2.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn2_list.append(module)


        def forward(self, x, y):

            in_residual = [x, y]
            x, y = self.replacer(x, y)
            x = x + in_residual[0]
            y = y + in_residual[1]
            out = [x, y]
            residual = [x, y]
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if len(x) > 1:
                out = self.exchange(out, self.bn2_list, self.bn_threshold)
            out = self.relu(out)
            out = [out[l] + residual[l] for l in range(self.num_parallel)]
            out = self.relu(out)
            x, y = out[0], out[1]

            return x, y

###############
class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, ex_channels=None, num_classes=5):
        super(ResNet, self).__init__()
        #img branch
        self.img_branch = nn.Sequential(
            conv3x3(img_channels, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 128),
            BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), nn.AdaptiveAvgPool2d((64, 64))

        )

        #Dist&Dem branch
        self.ex_branch = nn.Sequential(
            conv3x3(ex_channels, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 128),
            BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),nn.AdaptiveAvgPool2d((64, 64))
        )

        self.layer1 = self._make_layer_img(block, 128, 64, layers[0])
        self.layer1_d = self._make_layer_d(block, 128, 64, layers[0])
        self.attention_1 = self.attention(256)
        self.attention_1_d = self.attention(256)

        self.layer2 = self._make_layer_img(block, 256, 128, layers[1], stride=2)
        self.layer2_d = self._make_layer_d(block, 256, 128, layers[1], stride=2)
        self.attention_2 = self.attention(512)
        self.attention_2_d = self.attention(512)

        self.layer3 = self._make_layer_img(block, 512, 256, layers[2], stride=1, dilation=2)
        self.layer3_d = self._make_layer_d(block, 512, 256, layers[2], stride=1, dilation=2)
        self.attention_3 = self.attention(1024)
        self.attention_3_d = self.attention(1024)

        self.layer4 = self._make_layer_img(block, 1024, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.layer4_d = self._make_layer_d(block, 1024, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.attention_4 = self.attention(2048)
        self.attention_4_d = self.attention(2048)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.ARconvs = nn.ModuleList([ARConv_Block(32, flag=True),ARConv_Block(64, flag=True),ARConv_Block(128, flag=True),None])
        # self.ARconvs = nn.ModuleList([ARConv_Block(256, flag=True),
        #                                     ARConv_Block(512, flag=True),
        #                                     ARConv_Block(1024, flag=True),
        #                                     ARConv_Block(2048, flag=True)])
        # self.Featureselects = nn.ModuleList([
        #     Featureselect(256, 256, mid_channels=258,num_parallel=6, bn_threshold=2e-1),
        #     Featureselect(512, 512, mid_channels=510, num_parallel=6, bn_threshold=2e-1),
        #     Featureselect(1024, 1024, mid_channels=1026, num_parallel=6, bn_threshold=2e-1),
        #     Featureselect(2048, 2048, mid_channels=2046, num_parallel=6, bn_threshold=2e-1)])
        #

        self.reduce_convs = nn.ModuleList([
            nn.Sequential(  # Stage 1: 512 → 64
                nn.Conv2d(512, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # Stage 2: 1024 → 128
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # Stage 3: 2048 → 256
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
        ])
        self.expand_convs = nn.ModuleList([
            nn.Sequential(  # Stage 1: 32 → 256
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # Stage 2: 64 → 512
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # Stage 3: 128 → 1024
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1024, kernel_size=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
            ),
        ])
        # self.Featureselects = nn.ModuleList([
        #     Featureselect2(256, num_parallel = 2, bn_threshold = 2e-1),
        #     Featureselect2(512, num_parallel=2, bn_threshold=2e-1),
        #     Featureselect2(1024, num_parallel=2, bn_threshold=2e-1),
        #     Featureselect2(2048, num_parallel=2, bn_threshold=2e-1)
        #     ])

        self.Featureselects = nn.ModuleList([
            Featureselect3(256, num_parallel = 2, bn_threshold = 4e-1,maxpool_thresh=3e-1,verbose=False, log_dir= r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录", stage_name = 'stage1'),
            Featureselect3(512, num_parallel=2, bn_threshold=4e-1, maxpool_thresh=3e-1,verbose=False, log_dir= r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录", stage_name = 'stage2'),
            Featureselect3(1024, num_parallel=2, bn_threshold=4e-1, maxpool_thresh=3e-1,verbose=False, log_dir= r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录", stage_name = 'stage3'),
            Featureselect3(2048, num_parallel=2, bn_threshold=4e-1, maxpool_thresh=3e-1,verbose=False, log_dir= r"D:\zotero_article\SFA-DFNet-main\IAFFNet\实验记录", stage_name = 'stage4'),
        ])


    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    # 增强模型对特征的选择性表达，从而提高模型的性能,把张量每个通道转化成一个权重，通道数量不变
    # model = AttentionModule(num_channels=2)
    # Input shape: torch.Size([1, 2, 3, 3])
    # Attention weights
    # shape: torch.Size([1, 2, 1, 1])
    # Attention weights:
    # tensor([[[[0.5041]],
    #          [[0.5062]]]])

    def _make_layer_img(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None    # 处理输入通道数和输出通道数之间不匹配的情况（例如，步幅为2或输入输出通道数不同）。如果stride不为1或输入和输出通道数不匹配，就需要进行降采样（即使用1x1卷积将输入的尺寸调整为匹配输出的尺寸）。
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)



    # 处理输入通道数和输出通道数之间不匹配的情况（例如，步幅为2或输入输出通道数不同）。如果stride不为1或输入和输出通道数不匹配，就需要进行降采样（即使用1x1卷积将输入的尺寸调整为匹配输出的尺寸）。

    def _make_layer_d(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, x_ex):
        x = self.img_branch(x) #128
        y = self.ex_branch(x_ex)#128

        x = self.layer1(x)
        y = self.layer1_d(y)
        x_attention = self.attention_1(x)
        y_attention = self.attention_1_d(y)
        x = torch.mul(x, x_attention) #256
        y = torch.mul(y, y_attention)
        del x_attention, y_attention
        x,y = self.Featureselects[0](x,y) #x,y分别是256,64,64
        res =  x + y
        # x = x + y
        x = torch.cat([x, y], dim=1)#512
        x = self.reduce_convs[0](x)
        x = self.ARconvs[0](x)
        x = self.expand_convs[0](x)
        x = F.relu(x + res, inplace=True)
        x_low = x

        x = self.layer2(x)##x,y分别是512,32,32
        y = self.layer2_d(y)
        x_attention = self.attention_2(x)
        y_attention = self.attention_2_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        del x_attention, y_attention
        x,y = self.Featureselects[1](x,y)
        res =  x + y
        # x = x + y
        x = torch.cat([x, y], dim=1)#512
        x = self.reduce_convs[1](x)
        x = self.ARconvs[1](x)
        x = self.expand_convs[1](x)
        x = x + res
        x = F.relu(x + res, inplace=True)

        # x = torch.cat([x, y], dim=1)
        # x = self.ARconvs[1](x)


        x = self.layer3(x)#x,y分别是1024,32,32
        y = self.layer3_d(y)
        x_attention = self.attention_3(x)
        y_attention = self.attention_3_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        del x_attention, y_attention
        x,y = self.Featureselects[2](x,y)
        res = x + y
        # x = torch.cat([x, y], dim=1)
        # x = self.ARconvs[2](x)
        x = torch.cat([x, y], dim=1)  # 512
        x = self.reduce_convs[2](x)
        x = self.ARconvs[2](x)
        x = self.expand_convs[2](x)
        x = x + res
        x = F.relu(x + res, inplace=True)
        x_dsn = self.dsn(x)

        x = self.layer4(x)
        y = self.layer4_d(y)
        x_attention = self.attention_4(x)
        y_attention = self.attention_4_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        del x_attention, y_attention
        x,y = self.Featureselects[3](x,y)
        x = x + y
        # x = torch.cat([x, y], dim=1)#512
        # x = self.ARconvs[3](x)


        return x, x_dsn, x_low


class ROSANet(nn.Module):
    def __init__(self, block, size, in_channels=10, ex_channels=1,
                 n_classes=5, criterion=None, recurrence=2, use_rcca=True):
        super(ROSANet, self).__init__()

        # Main stream and branch(branch extract dist features)
        self.backbone = ResNet(block, [3, 4, 6, 3], img_channels=in_channels, ex_channels=ex_channels, num_classes=n_classes)
        # ASPP
        self.aspp = ASPP_module(2048, 256)
        # RCCA
        self.rcca = RCCAModule(2048, 512, num_classes=n_classes)

        # global pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   BatchNorm2d(256),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   BatchNorm2d(256),
                                   nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        self.last_conv = nn.Conv2d(n_classes*2, n_classes, kernel_size=1, bias=False)

        self.size = size
        self.criterion = criterion
        self.recurrence = recurrence
        self.use_rcca = use_rcca

        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_ex, labels=None):
        # x                  8x2048x29x29
        # low_level_features 8x256x57x57
        # x_dsn              8x5x29x29
        x, x_dsn, low_level_features = self.backbone(x, x_ex)
        x_r = x

        # x                  8x2048x57x57
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        # x_aspp             8x1024x57x57
        x_aspp = self.aspp(x)#
        # x_                 8x256x57x57
        x_ = self.global_avg_pool(x)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_aspp, x_), dim=1)

        # x                  8x256x57x57
        x = self.conv(x)
        x = self.bn(x)

        # low_level_features 8x48x57x57
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)

        # x                  8x5x57x57
        x = self.conv3(x)

        # x_rcca             8x5x29x29
        if self.use_rcca:
            x_rcca = self.rcca(x_r)
            x_rcca = F.interpolate(x_rcca, size=x.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, x_rcca), dim=1)
            x = self.last_conv(x)

        x_dsn = F.interpolate(x_dsn, size=x.size()[2:], mode='bilinear', align_corners=True)
        outs = [x, x_dsn]

        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels), predict_whole(outs, self.size)
        else:
            return predict_whole(outs, self.size)


def Seg_Model(in_channel, ex_channels, num_classes, size, criterion=None, recurrence=0, use_rcca=True, **kwargs):
    model = ROSANet(Bottleneck, size=size, in_channels=in_channel, ex_channels=ex_channels,
                   n_classes=num_classes, criterion=criterion, recurrence=recurrence, use_rcca=use_rcca)
    return model

