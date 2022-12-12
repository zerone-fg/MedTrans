import torch
from torch import nn
from networks.unet import ConvBlock, DownBlock, UpBlock
from networks.swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed, BasicLayer, PatchMerging, PatchExpand, \
    BasicLayer_up, FinalPatchExpand_X4
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from networks.unet import Decoder
import copy
import math


class FeatureSelector(nn.Module):
    def __init__(self, dim_cnn, dim_trans, h_c, w_c, h_t, w_t):
        super(FeatureSelector, self).__init__()
        inplane_cnn = dim_trans // 4
        inplane_trans = dim_trans // 4

        self.spatial_interaction_cnn_1 = nn.Sequential(
            nn.Conv2d(2 * dim_trans, inplane_cnn, kernel_size=3, padding=1),
            nn.BatchNorm2d(inplane_cnn),
            nn.ReLU(),
            nn.Conv2d(inplane_cnn, 1, kernel_size=1))

        self.spatial_interaction_trans_1 = nn.Sequential(
            nn.Conv2d(2 * dim_trans, inplane_trans, kernel_size=3, padding=1),
            nn.BatchNorm2d(inplane_trans),
            nn.ReLU(),
            nn.Conv2d(inplane_trans, 1, kernel_size=1), )

        self.fusion1 = nn.Linear(dim_trans, dim_cnn)
        self.fusion2 = nn.Linear(dim_trans, dim_trans)

        self.cnn_filter = nn.Parameter(torch.randn(1, 1, h_c, math.ceil(w_c / 2 + 1), dtype=torch.float32) * 0.02)
        self.trans_filter = nn.Parameter(torch.randn(1, 1, h_t, math.ceil(w_t / 2 + 1), dtype=torch.float32) * 0.02)

        self.chans1 = nn.Conv2d(dim_cnn, dim_trans, 1)

    def forward(self, x_cnn, h_c, w_c, x_trans, h_t, w_t):
        x_cnn = self.chans1(x_cnn)

        B, c_c, _, _ = x_cnn.shape
        x_cnn = x_cnn.permute(0, 2, 3, 1).view(B, h_c, w_c, c_c)
        x_cnn = x_cnn.to(torch.float32)
        x_cnn = torch.fft.rfft2(x_cnn, dim=(1, 2), norm='ortho')  # (16, 56, 29, 96)

        B, _, c_t = x_trans.shape
        x_trans = x_trans.view(B, h_t, w_t, c_t)
        x_trans = x_trans.to(torch.float32)
        x_trans = torch.fft.rfft2(x_trans, dim=(1, 2), norm='ortho')

        # # spatial attention
        x_trans = x_trans.permute(0, 3, 1, 2)
        x_cnn = x_cnn.permute(0, 3, 1, 2)

        att_real_trans = self.spatial_interaction_cnn_1(torch.cat((x_cnn.real, x_cnn.imag), dim=1))
        att_real_cnn = self.spatial_interaction_trans_1(torch.cat((x_trans.real, x_trans.imag), dim=1))

        x_trans = F.sigmoid(att_real_cnn) * x_cnn + x_trans
        x_cnn = F.sigmoid(att_real_trans) * x_trans + x_cnn

        x_cnn = self.cnn_filter * x_cnn
        x_trans = self.trans_filter * x_trans

        x_cnn = x_cnn.permute(0, 2, 3, 1)
        x_trans = x_trans.permute(0, 2, 3, 1)

        # IFFT
        x_cnn = torch.fft.irfft2(x_cnn, s=(h_c, w_c), dim=(1, 2), norm='ortho')
        x_trans = torch.fft.irfft2(x_trans, s=(h_t, w_t), dim=(1, 2), norm='ortho')

        x_cnn = self.fusion1(x_cnn)
        x_trans = self.fusion2(x_trans)

        x_cnn = x_cnn.permute(0, 3, 1, 2)
        x_trans = x_trans.reshape(B, h_t * w_t, c_t)

        return x_cnn, x_trans


class DualBranch(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super(DualBranch, self).__init__()

        self.params = {'in_chns': in_chans,
                       'feature_chns': [16, 32, 64, 128, 256],
                       'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                       'class_num': num_classes,
                       'bilinear': False,
                       'acti_func': 'relu'}

        ## unet encoder
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']

        assert (len(self.ft_chns) == 5)

        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        ## swin unet encoder
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                       i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # decoder 部分
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(
                                                 self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(
                                                 self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(
                                                 self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (
                                                 i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(
                img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(
                in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        self.fusion_module = nn.ModuleList()
        self.norm = norm_layer(self.num_features)
        self.fusion_module.append(FeatureSelector(64, 96, 56, 56, 56, 56))
        self.fusion_module.append(FeatureSelector(128, 192, 28, 28, 28, 28))
        self.fusion_module.append(FeatureSelector(256, 384, 14, 14, 14, 14))

        self.cnn_decoder = Decoder(self.params)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_encoder(self, x):
        c0 = self.in_conv(x)  # (16, 16, 224, 224)
        c1 = self.down1(c0)  # (16, 32, 112, 112)
        c2 = self.down2(c1)  # (16, 64, 56, 56)

        trans = self.patch_embed(x)
        if self.ape:
            trans = trans + self.absolute_pos_embed
        trans = self.pos_drop(trans)
        x_downsample_trans = []  # (16, 56, 56,96)

        #  t0和c2融合
        c2_r, trans_r = self.fusion_module[0](c2, 56, 56, trans, 56, 56)
        c2 = c2_r + c2
        trans = trans_r + trans
        x_downsample_trans.append(trans)

        t0 = self.layers[0](trans)
        c3 = self.down3(c2)  # (16, 128, 28, 28)
        c3_r, t0_r = self.fusion_module[1](c3, 28, 28, t0, 28, 28)
        c3 = c3_r + c3
        t0 = t0_r + t0
        x_downsample_trans.append(t0)

        c4 = self.down4(c3)
        t1 = self.layers[1](t0)
        c4_r, t1_r = self.fusion_module[2](c4, 14, 14, t1, 14, 14)
        c4 = c4 + c4_r
        t1 = t1 + t1_r
        x_downsample_trans.append(t1)

        #  单独t3计算
        t2 = self.layers[2](t1)  # (16, 7, 7, 768)
        x_downsample_trans.append(t2)
        t3 = self.layers[3](t2)
        x_downsample_trans.append(t3)

        t3 = self.norm(t3)

        return [c0, c1, c2, c3, c4], (t3, x_downsample_trans)

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward_decoder(self, feature_cnn, feature_trans):
        x_trans = self.forward_up_features(feature_trans[0], feature_trans[1])
        x_trans = self.up_x4(x_trans)
        x_cnn = self.cnn_decoder(feature_cnn)
        return x_cnn, x_trans

    def forward(self, x):
        feature_cnn, feature_trans = self.forward_encoder(x)
        x_c_f, x_t_f = feature_cnn[-1], feature_trans[0]
        out_cnn, out_trans = self.forward_decoder(feature_cnn, feature_trans)
        return out_cnn, out_trans, x_c_f, x_t_f


class MedTrans(nn.Module):
    def __init__(self, classes, config):
        super(MedTrans, self).__init__()
        self.num_classes = classes
        self.model = DualBranch(img_size=config.DATA.IMG_SIZE,
                                   patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                   in_chans=config.MODEL.SWIN.IN_CHANS,
                                   num_classes=self.num_classes,
                                   embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                   depths=config.MODEL.SWIN.DEPTHS,
                                   num_heads=config.MODEL.SWIN.NUM_HEADS,
                                   window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                   mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                   qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                   qk_scale=config.MODEL.SWIN.QK_SCALE,
                                   drop_rate=config.MODEL.DROP_RATE,
                                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                   ape=config.MODEL.SWIN.APE,
                                   patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.load_from(config)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits_cnn, logits_trans, x_c_f, x_t_f = self.model(x)
        return logits_cnn, logits_trans, x_c_f, x_t_f

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.model.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.model.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.model.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
