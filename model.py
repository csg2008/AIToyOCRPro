from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.conv_bn_act import create_conv2d
from timm.models.vision_transformer import VisionTransformer, register_model
from torchvision.ops.deform_conv import DeformConv2d
from decoder import HybridDecoder
from svtr import SVTRv2, UltraLightweightSVTRv2Neck
from viptr import VIPTRv2T_CH

@register_model
def vit_tiny_patch8_224(pretrained=False, **kwargs):
    '''ViT tiny'''
    # 去掉 VisionTransformer 不认识的参数以免报错
    for k in ('pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir'):
        kwargs.pop(k, None)

    model = VisionTransformer(
        img_size=224,
        patch_size=8,
        embed_dim=192,   # tiny 级别
        depth=12,
        num_heads=3,     # 192 // 64
        mlp_ratio=2,
        **kwargs
    )
    return model

class HGNetV2Backbone(nn.Module):
    """HGNetV2 主干网络"""
    def __init__(
        self,
        model_name: str = 'hgnetv2_b4',
        in_channels: int = 3,
    ):
        super().__init__()
        channels = {
            'hgnetv2_b0': 16,
            'hgnetv2_b1': 32,
            'hgnetv2_b2': 32,
            'hgnetv2_b3': 32,
            'hgnetv2_b4': 48,
            'hgnetv2_b5': 64,
            'hgnetv2_b6': 96,
        }

        assert model_name in channels, f"Backbone {model_name} not support"

        # 加载预训练模型
        self.model = timm.create_model(
            model_name,
            pretrained = False,
            features_only=True,
            in_chans=in_channels,
        )
        self.model.stem = create_conv2d(
            in_channels=in_channels,
            out_channels=channels[model_name],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 32, 128)
            features = self.model(dummy_input)
            self.num_features = features[-1].shape[1]
            self.feature_dims = [f.shape[1] for f in features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, 3, H, W] - 输入图像
        Returns:
            features: [B, C, H', W'] - 最后一层特征
        """
        # 提取多层特征
        multi_features = self.model(x)

        # 返回最后一层特征
        return multi_features[-1]

class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXtV2 主干网络"""
    def __init__(
        self,
        model_name: str = 'convnextv2_nano',
        in_channels: int = 3,
    ):
        super().__init__()
        channels = {
            'convnextv2_atto': 40,
            'convnextv2_femto': 48,
            'convnextv2_pico': 64,
            'convnextv2_nano': 80,
            'convnextv2_tiny': 96,
            'convnextv2_small': 96,
            'convnextv2_base': 128,
            'convnextv2_large': 192,
            'convnextv2_huge': 352,
        }

        assert model_name in channels, f"Backbone {model_name} not support"

        self.model = timm.create_model(
            model_name,
            features_only=True,
            pretrained=False,
            in_chans=in_channels
        )
        self.model.stem_0 = create_conv2d(
            in_channels=in_channels,
            out_channels=channels[model_name],
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 32, 128)
            features = self.model(dummy_input)
            self.num_features = features[-1].shape[1]
            self.feature_dims = [f.shape[1] for f in features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, 3, H, W] - 输入图像
        Returns:
            features: [B, C, H', W'] - 最后一层特征
        """
        return self.model(x)[-1]

class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 主干网络"""

    def __init__(
        self,
        model_name: str = 'mobilenetv4_conv_small',
        in_channels: int = 3,
    ):
        super().__init__()
        channels = {
            'mobilenetv4_conv_small': [32, 960],
            'mobilenetv4_conv_medium': [128, 960],
            'mobilenetv4_conv_large': [96, 960],
        }

        assert model_name in channels, f"Backbone {model_name} not support"

        backbone = timm.create_model(
            model_name,
            features_only=True,
            pretrained=False,
            in_chans=in_channels
        )
        conv2d = create_conv2d(
            in_channels=in_channels,
            out_channels=channels[model_name][0],
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )

        if hasattr(backbone.blocks[0][0], 'conv_exp'):
            backbone.blocks[0][0].conv_exp = conv2d
        else:
            backbone.blocks[0][0].conv = conv2d

        self.model = backbone.blocks
        self.num_features = channels[model_name][1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, 3, H, W] - 输入图像
        Returns:
            features: [B, C, H', W'] - 最后一层特征
        """
        return self.model(x)

class RepVitBackbone(nn.Module):
    """RepVit 主干网络"""
    def __init__(
            self,
            arch: str = "repvit_m2_5",
            in_channels: int = 3
        ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=arch,
            in_chans=in_channels,
            num_classes=0,          # 去掉分类头
            # img_size=None,
            pretrained=False,
            features_only=True,
            global_pool=None,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 128)
            feats = self.backbone(dummy)
            self.out_channels = feats[-1].shape[1]  # 通道数
            self.feature_dims = [f.shape[1] for f in feats]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, 3, H, W] - 输入图像
        Returns:
            features: [B, C, H', W'] - 最后一层特征
        """
        return self.backbone(x)

class SVTRNeck(nn.Module):
    """SVTR 风格的 Neck - 将CNN特征转换为序列"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: int,
        depth: int,
        kernel_size: List[int] = [1, 3],
        use_guide: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_guide = use_guide

        # 1. 通道降维
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # 2. SVTR Mixing 模块
        self.mixing_blocks = nn.ModuleList()
        for i in range(depth):
            block = SVTRMixingBlock(
                dim=out_channels,
                kernel_size=kernel_size,
                hidden_dims=hidden_dims,
                dropout=0.1
            )
            self.mixing_blocks.append(block)

        # 3. 特征转换层
        self.feature_transform = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, C, H, W] - CNN特征
        Returns:
            sequence: [B, T, D] - 序列特征
        """
        B, C, H, W = x.shape

        # 1. 通道降维
        x = self.input_proj(x)  # [B, out_channels, H, W]

        # 2. SVTR Mixing（在 1×W 上做）
        for mixing_block in self.mixing_blocks:
            x = mixing_block(x)

        # 3. 转换为序列
        # 将高度维度合并到宽度维度
        x = x.permute(0, 2, 3, 1)  # [B, H, W, D]
        x = x.reshape(B, H * W, -1)  # [B, H*W, D]

        # 4. 特征变换
        x = self.feature_transform(x)   # [B, W, D]

        return x

class SVTRMixingBlock(nn.Module):
    """SVTR Mixing Block - 局部和全局特征混合"""
    def __init__(
        self,
        dim: int,
        kernel_size: List[int],
        hidden_dims: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 局部混合
        self.local_mixing = nn.Sequential(
            nn.Conv2d(
                dim, hidden_dims,
                kernel_size=kernel_size[1],
                padding=kernel_size[1]//2,
                groups=dim  # 分组卷积
            ),
            nn.BatchNorm2d(hidden_dims),
            nn.GELU(),
            nn.Conv2d(hidden_dims, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

        # 全局混合
        self.global_mixing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dims, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dims, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        residual = x

        # 局部混合
        local_features = self.local_mixing(x)

        # 全局混合
        global_attention = self.global_mixing(x)
        global_features = x * global_attention

        # 融合
        mixed_features = local_features + global_features

        # 残差连接
        output = residual + mixed_features

        return output

class HybridNeck(nn.Module):
    """混合 Neck - 用于融合不同来源的特征"""
    def __init__(self, backbone_out_channels: int, d_model: int, mode = "attention"):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.backbone_out_channels = backbone_out_channels

        # 投影到 d_model 维度（用于 key/value）
        self.feature_proj = nn.Conv2d(backbone_out_channels, d_model, 1)

        # LayerNorm for stability (optional but recommended)
        self.norm = nn.LayerNorm(d_model)

        # 训练阶段才用的 attention 参数
        self.query = nn.Parameter(torch.empty(1, d_model, 1, 1))
        nn.init.xavier_uniform_(self.query)

        # 注意力的 key 投影（1x1 conv）
        self.key_proj = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, C, H, W)
        Output: (B, W, D_model)
        """
        B, C, H, W = x.shape
        # 1. 先投影到 d_model 通道
        x = self.feature_proj(x)              # (B,D,H,W)

        if self.training or self.mode == "attention":
            # ---------- attention branch ----------
            # 2. 计算 key: (B, D, H, W)
            key = self.key_proj(x)            # (B,D,H,W)
            # 3. 广播 query 到 (B, D, 1, W)
            query = self.query.expand(B, -1, 1, W)  # (B,D,1,W)
            # 4. 计算注意力分数: (B, H, W)
            # key: (B, D, H, W), query: (B, D, 1, W)
            attn = (key * query).sum(1) / (self.d_model ** 0.5)  # (B,H,W)
            attn = torch.softmax(attn, dim=1)
            # 5. 对 H 维加权求和: (B, D, H, W) * (B, 1, H, W) -> sum over H
            x_pooled = (x * attn.unsqueeze(1)).sum(dim=2)        # (B,D,W)
        else:
            # ---------- avg pool branch ----------
            x_pooled = x.mean(dim=2)          # (B,D,W)

        # 6. 转置为 (B, W, D)
        x_seq = x_pooled.permute(0, 2, 1)     # (B,W,D)

        # 7. LayerNorm (序列维度)
        return self.norm(x_seq)

class RepVitMultiScaleNeck(nn.Module):
    """多尺度RepViT Neck - 高精度场景"""

    def __init__(self, backbone_channels: List[int], out_channels: int, hidden_dims: int = 256):
        super().__init__()

        # 多尺度特征融合
        self.scale_adapters = nn.ModuleList([
            nn.Conv2d(channels, hidden_dims, 1) for channels in backbone_channels
        ])

        # 特征融合 - 使用轻量级注意力
        self.fusion_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dims * len(backbone_channels), hidden_dims, 1),
            nn.Sigmoid()
        )

        # 垂直压缩和序列转换
        self.vertical_compress = nn.AdaptiveAvgPool2d((1, None))
        self.sequence_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, out_channels)
        )

    def forward(self, multi_scale_features):
        # 适配不同尺度的特征
        adapted_features = []
        for feat, adapter in zip(multi_scale_features, self.scale_adapters):
            adapted = adapter(feat)
            # 上采样到相同空间尺寸
            target_size = multi_scale_features[-1].shape[2:]
            if adapted.shape[2:] != target_size:
                adapted = F.interpolate(adapted, size=target_size, mode='bilinear', align_corners=False)
            adapted_features.append(adapted)

        # 融合多尺度特征
        fused = torch.cat(adapted_features, dim=1)
        attention_weights = self.fusion_attention(fused)

        # 加权融合
        weighted_features = adapted_features[-1] * attention_weights

        # 垂直压缩和序列转换
        compressed = self.vertical_compress(weighted_features)
        sequence = compressed.squeeze(2).permute(0, 2, 1)
        output = self.sequence_proj(sequence)

        return output

class RecNetwork(nn.Module):
    """文本识别网络"""
    def __init__(
        self,
        train_mode: str,
        model_name: str,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_text_length: int,
        dropout: float,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        max_rope_len: int = 30000,
        use_svd: bool = False,
        k=8,
        r=64,
        dynamic_k=0
    ):
        super().__init__()
        self.train_mode = train_mode
        self.num_classes = num_classes
        self.max_text_length = max_text_length
        self.use_ctc = True if train_mode == 'ctc' or train_mode == 'hybrid' else False
        self.use_ar = True if train_mode == 'ar' or train_mode == 'hybrid' else False

        if model_name.find('vit') == 0:
            # 创建 ViT 模型（不限制输入尺寸）
            self.backbone = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=False,
                in_chans=in_channels,
                num_classes=0,          # 去掉分类头
                img_size=None,
                dynamic_img_size=True,
                global_pool='',         # ✅ 关键：禁止池化，保留 patch 序列
            )

            # 获取ViT的特征维度
            self.feature_dims = self.backbone.embed_dim

            # 将ViT输出映射到解码器维度
            self.neck = nn.Linear(self.feature_dims, hidden_dim)
        elif model_name.find('hgnetv2') == 0:
            self.backbone = HGNetV2Backbone(
                model_name=model_name,
                in_channels=in_channels
            )

            self.neck = HybridNeck(self.backbone.num_features, hidden_dim)
        elif model_name.find('convnextv2') == 0:
            self.backbone = ConvNeXtV2Backbone(
                model_name=model_name,
                in_channels=in_channels
            )

            self.neck = HybridNeck(self.backbone.num_features, hidden_dim)
        elif model_name.find('viptr2') == 0:
            self.backbone = VIPTRv2T_CH(in_chans = in_channels, output_channel = hidden_dim)

            self.neck = nn.Identity()
        elif model_name.find('mobilenetv4') == 0:
            self.backbone = MobileNetV4Backbone(
                model_name=model_name,
                in_channels=in_channels
            )

            self.neck = HybridNeck(self.backbone.num_features, hidden_dim)
        elif model_name.find('svtrv2') == 0:
            cfg = SVTRv2.svtr_cfg[model_name[7:]]
            self.backbone = SVTRv2(in_channels=in_channels, **cfg)

            # 使用SVTRv2专属Neck，平衡计算性能与精度
            self.neck = UltraLightweightSVTRv2Neck(
                in_channels=self.backbone.num_features,
                out_channels=hidden_dim,
                hidden_dims=256,
                dropout=dropout
            )
        elif model_name.find('repvit') == 0:
            arch = model_name[7:]
            scale = {
                'tiny': 'repvit_m1_0',
                'small': 'repvit_m1_1',
                'base': 'repvit_m1_5',
                'large': 'repvit_m2_3',
            }
            self.backbone = RepVitBackbone(arch=scale[arch], in_channels=in_channels)
            self.embed_dim = self.backbone.out_channels

            # 使用RepVitBackbone专属Neck，平衡计算性能与精度
            self.neck = RepVitMultiScaleNeck(self.backbone.feature_dims, hidden_dim)
        else:
            raise ValueError(f"Unsupported backbone model: {model_name}")

        # 3. 解码器
        self.decoder = HybridDecoder(
            train_mode=train_mode,
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            max_text_length=max_text_length,
            dropout=dropout,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token,
            max_rope_len=max_rope_len,
            use_svd=use_svd,
            k=k,
            r=r,
            dynamic_k=dynamic_k
        )

        self._init_weights()

    def _init_weights(self):
        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("backbone."):
                continue
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_features: bool = False,
        epoch: int | None = None,
        eval_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 1, H, W] - 单通道，动态宽度
            targets: 目标文本 [B, L] (训练时用)
            return_features: 是否返回中间特征
            eval_mode: 训练时是否走推理分支（解决模型训练时不知道自回归推理错误字符）

        Returns:
            包含预测结果的字典
        """
        # 1. Backbone 特征提取
        features = self.backbone(x)  # [B, C, H', W']

        # 2. Neck 特征处理
        neck_features = self.neck(features)  # [B, T, D] - 转换为序列形式
        # 查看特征尺寸
        # b,c,h,w = x.shape
        # px_per_frame = w / neck_features.shape[1]
        # print(f'px_per_frame={px_per_frame:.1f} w: {w}  neck_w: {neck_features.shape[1]}')
        # print(f'neck_features reshape: {neck_features.shape=}, {neck_features.min()=}, {neck_features.max()=}')

        # 3. Decoder 解码
        if self.training:
            # 训练时使用双分支
            decoder_outputs = self.decoder(
                neck_features,
                targets=targets,
                eval_mode=eval_mode,
                use_ctc=self.use_ctc,
                use_ar=self.use_ar
            )
        else:
            # 推理时仅使用CTC分支（高效）
            use_ar = True if self.train_mode == 'ar' else False
            decoder_outputs = self.decoder(
                neck_features,
                eval_mode=eval_mode,
                use_ctc=self.use_ctc,
                use_ar=use_ar
            )

        if return_features:
            decoder_outputs['backbone_features'] = features
            decoder_outputs['neck_features'] = neck_features

        return decoder_outputs

    def predict(self, x: torch.Tensor) -> List[str]:
        """推理接口"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            ctc_logits = outputs['ctc_logits']

            # CTC解码
            decoded = self.ctc_decode(ctc_logits)
            return decoded

    def ctc_decode(self, logits: torch.Tensor) -> List[str]:
        """CTC解码 - 支持动态长度"""
        # 使用贪心解码
        predictions = torch.argmax(logits, dim=-1)  # [B, T]

        # 后处理
        results = []
        for pred in predictions:
            # 去除连续重复
            pred = [pred[0]] + [pred[i] for i in range(1, len(pred)) if pred[i] != pred[i-1]]
            # 去除空白符 (0)
            pred = [p for p in pred if p != 0]
            results.append(pred)

        return results
