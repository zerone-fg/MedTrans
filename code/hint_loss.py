from collections import OrderedDict
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoResample(nn.Module):
    def __init__(self, enable: bool = True):
        super().__init__()
        self.enable = enable

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        """
        Args:
            feat_1: [n1, bs, emb_dim]
            feat_2: [n2, bs, emb_dim]
        """
        if not self.enable:
            return feat_1, feat_2

        shape_1 = feat_1.shape
        shape_2 = feat_2.shape
        if shape_1 == shape_2:
            return feat_1, feat_2

        assert shape_1[1:] == shape_2[1:], "bs and emb_dim must be equal"
        exchange = False
        if shape_1[0] > shape_2[0]:
            feat_long = feat_1
            feat_short = feat_2
            assert shape_1[0] % shape_2[0] == 0
            step = shape_1[0] // shape_2[0]
        else:
            exchange = True
            feat_long = feat_2
            feat_short = feat_1
            assert shape_2[0] % shape_1[0] == 0
            step = shape_2[0] // shape_1[0]

        # [n, bs, emb_dim] -> [bs, emb_dim, n]
        feat_long = feat_long.permute(1, 2, 0)
        feat_long = F.avg_pool1d(feat_long, kernel_size=step, stride=step)
        # [bs, emb_dim, n] -> [n, bs, emb_dim]
        feat_long = feat_long.permute(2, 0, 1)

        if exchange:
            feat_long, feat_short = feat_short, feat_long
        return feat_long, feat_short


class DistillHint(nn.Module):
    def __init__(
        self,
        start_id: int = 1,
        proj_student: bool = False,
        embed_dim_in: int = None,
        embed_dim_out: int = None,
        norm: bool = False,
        auto_resample: bool = False
    ):
        """
        Args:
            start_id: 1 for one [cls] token, 2 for one [cls] and one [dist] token
        """
        super().__init__()
        self.start_id = start_id
        self.proj = nn.Identity()
        self.resample = AutoResample(auto_resample)
        self.norm_s = nn.Identity()
        self.norm_t = nn.Identity()
        self.mse_loss = nn.MSELoss(reduction='mean')

        if proj_student:
            # self.proj = nn.LazyLinear(embed_dim)
            self.proj = nn.Linear(embed_dim_in, embed_dim_out)
        if norm:
            self.norm_s = nn.LayerNorm(embed_dim_out)
            self.norm_t = nn.LayerNorm(embed_dim_out)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:

        feat_cnn = feat_s
        feat_vit = feat_t

        # [n + 1(2), bs, emb_dim] -> [n, bs, emb_dim]
        feat_vit = feat_vit.flatten(2).permute(2, 0, 1)
        # [bs, emb_dim, w, h] -> [wh, bs, emb_dim]
        feat_cnn = feat_cnn.flatten(2).permute(2, 0, 1)

        # proj student
        feat_vit = self.proj(feat_vit)
        feat_cnn, feat_vit = self.resample(feat_cnn, feat_vit)
        # print(feat_vit.shape, feat_cnn.shape)
        assert feat_vit.shape == feat_cnn.shape

        feat_vit = self.norm_s(feat_vit)
        feat_cnn = self.norm_t(feat_cnn)

        # loss = self.mse_loss(feat_vit.detach(), feat_cnn)
        loss = self.mse_loss(feat_vit, feat_cnn)
        return loss


def compute_proto(pred, feature):
    b, d, h, w = feature.shape
    b, num_classes, h, w = pred.shape
    mask = pred.view(b, num_classes, -1)
    feature = feature.view(b, d, -1).transpose(-1, -2)
    centers_c = torch.zeros((b, num_classes, d))
    for classes in range(num_classes):
        for i in range(b):
            centers_c[i, classes, :] = torch.sum(
                feature[i, :, :] * nn.Softmax(dim=-1)(mask[i, classes, :]).unsqueeze(1), dim=0)  # (1, 1,c)
    return centers_c.cuda()


class ClassConsisten(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_s = nn.Identity()
        self.norm_t = nn.Identity()

    def forward(self, feature1, feature2, outputs_soft_1, outputs_soft_2):  # 计算类别中心和特征的相似度图

        b, d, h, w = feature1.shape
        feature1 = self.norm_s(feature1)
        feature2 = self.norm_t(feature2)

        outputs1_r = F.interpolate(outputs_soft_1, feature1.shape[-2:], mode="bilinear", align_corners=True)
        outputs2_r = F.interpolate(outputs_soft_2, feature2.shape[-2:], mode="bilinear", align_corners=True)

        proto_c = compute_proto(outputs1_r, feature1)  # (16, 4, 256) (b, cls, d)
        proto_t = compute_proto(outputs2_r, feature2)  # (16, 4, 768) (b, cls, d)

        feature1 = feature1.permute(0, 2, 3, 1).reshape(b, -1, d)  # (b, h*w, d)
        feature2 = feature2.permute(0, 2, 3, 1).reshape(b, -1, d)  # (b, h*w, d)

        proto_c = proto_c.unsqueeze(2).repeat(1, 1, h * w, 1)  # (b, cls, h*w, d)
        proto_t = proto_t.unsqueeze(2).repeat(1, 1, h * w, 1)

        feature1 = feature1.unsqueeze(1).repeat(1, 4, 1, 1)
        feature2 = feature2.unsqueeze(1).repeat(1, 4, 1, 1)

        sim_c = F.cosine_similarity(proto_c, feature1, dim=-1)
        sim_t = F.cosine_similarity(proto_t, feature2, dim=-1)

        loss_cls = nn.MSELoss(reduction="sum")(sim_t.view(-1), sim_c.view(-1))
        return loss_cls


