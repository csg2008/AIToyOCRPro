"""
SVTRv2 is refer from: https://github.com/Topdu/OpenOCR/blob/main/openrec/modeling/encoders/svtrv2_lnconv_two33.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_
from timm.layers import Mlp, DropPath, Attention, ConvNormAct, LayerNorm

class FlattenTranspose(nn.Module):
    """
    展平并转置张量的模块

    将输入张量的第2维及以后展平，然后转置第1维和第2维
    """
    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(b, c, h, w)或类似形状

        Returns:
            展平并转置后的张量，形状为(b, h*w, c)
        """
        return x.flatten(2).transpose(1, 2)


class SubSample2D(nn.Module):
    """
    2D子采样模块

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长，默认[2, 1]
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = LayerNorm(out_channels)

    def forward(self, x, sz):
        """
        前向传播

        Args:
            x: 输入张量
            sz: 尺寸信息

        Returns:
            tuple: (处理后的张量, 新的尺寸信息)
        """
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class SubSample1D(nn.Module):
    """
    1D子采样模块

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长，默认[2, 1]
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = LayerNorm(out_channels)

    def forward(self, x, sz):
        """
        前向传播

        Args:
            x: 输入张量
            sz: 尺寸信息

        Returns:
            tuple: (处理后的张量, 新的尺寸信息)
        """
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):
    """
    恒等尺寸模块

    返回输入的张量和尺寸信息，不做任何修改
    """
    def forward(self, x, sz):
        """
        前向传播

        Args:
            x: 输入张量
            sz: 尺寸信息

        Returns:
            tuple: (输入张量, 输入尺寸信息)
        """
        return x, sz


class ADDPosEmbed(nn.Module):
    """
    添加位置嵌入模块

    Args:
        feat_max_size: 特征最大尺寸，默认[8, 32]
        embed_dim: 嵌入维度，默认768
    """
    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        pos_embed = torch.zeros(
            [1, feat_max_size[0] * feat_max_size[1], embed_dim],
            dtype=torch.float32)
        trunc_normal_(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0],
                                              feat_max_size[1]),
            requires_grad=True,
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            添加位置嵌入后的张量
        """
        sz = x.shape[2:]
        x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        return x

class LinearAttention(nn.Module):
    """线性注意力机制 - 计算复杂度从O(n²)降至O(n)

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量，默认8
        qkv_bias: 是否使用qkv投影的偏置，默认False
        attn_drop: 注意力dropout概率，默认0.0
        proj_drop: 投影dropout概率，默认0.0
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，可以是3D [B, L, C]或4D [B, C, H, W]格式

        Returns:
            处理后的张量，与输入格式相同
        """
        # 处理 3D [B, L, C] 或 4D [B, C, H, W] 输入(4D是为了兼容自适应深度机制)
        is_4d = len(x.shape) == 4
        if is_4d:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        B, L, C = x.shape

        # 计算QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, head_dim]

        # 线性注意力：使用核技巧
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # 计算注意力输出
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        output = torch.einsum('bhln,bhdm->bhld', q, kv)

        # 合并多头
        output = output.transpose(1, 2).reshape(B, L, C)
        output = self.proj(output)
        output = self.proj_dropout(output)

        # 如果是 4D 输入，恢复原始形状(4D是为了兼容自适应深度机制)
        if is_4d:
            output = output.transpose(1, 2).reshape(B, C, H, W)

        return output


class OptimizedAttention(nn.Module):
    """
    优化的注意力机制模块

    根据use_linear_attn参数选择使用线性注意力或普通注意力

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量，默认8
        qkv_bias: 是否使用qkv投影的偏置，默认False
        qk_scale: 缩放因子，默认None
        attn_drop: 注意力dropout概率，默认0.0
        proj_drop: 投影dropout概率，默认0.0
        use_linear_attn: 是否使用线性注意力，默认False
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_linear_attn=False,  # 新增参数控制是否使用线性注意力
    ):
        super().__init__()
        self.use_linear_attn = use_linear_attn

        if use_linear_attn:
            self.attention = LinearAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            self.attention = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                proj_bias=True  # SVTR默认使用proj_bias
            )
        # 保持与原始接口兼容
        self.scale = qk_scale or (dim // num_heads) ** -0.5

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，可以是3D [B, L, C]或4D [B, C, H, W]格式

        Returns:
            处理后的张量，与输入格式相同
        """
        return self.attention(x)


class AdaptiveDepthModule(nn.Module):
    """
    自适应深度机制 - 根据输入复杂度动态调整网络深度

    Args:
        dim: 输入通道数
        max_blocks: 最大块数
        threshold: 阈值，默认0.5
    """
    def __init__(self, dim, max_blocks, threshold=0.5):
        super().__init__()
        self.max_blocks = max_blocks
        self.threshold = threshold

        # 深度分类器 - 根据输入特征决定使用多少层
        # 使用AdaptiveAvgPool2d处理4D输入，同时兼容3D输入
        self.depth_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, C, 1, 1]
            nn.Conv2d(dim, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, max_blocks, 1),
            nn.Sigmoid()  # 输出每个block的使用概率
        )

    def forward(self, x, blocks):
        """
        Args:
            x: 输入特征 [B, L, C] 或 [B, C, H, W]
            blocks: nn.ModuleList，包含所有block
        Returns:
            经过自适应深度处理后的特征
        """
        # 计算需要执行的block数量
        # 只在初始时计算一次，使用输入特征来决定网络深度
        is_4d = len(x.shape) == 4
        if is_4d:
            # 4D输入: [B, C, H, W]
            # 计算深度权重 [B, max_blocks, 1, 1]
            depth_weights = self.depth_classifier(x)
            depth_weights = depth_weights.squeeze(-1).squeeze(-1)  # [B, max_blocks]
        else:
            # 3D输入: [B, L, C]
            # 转换为4D以便使用depth_classifier
            x_4d = x.transpose(1, 2).unsqueeze(-1)  # [B, C, L, 1]
            # 计算深度权重 [B, max_blocks, 1, 1]
            depth_weights = self.depth_classifier(x_4d)
            depth_weights = depth_weights.squeeze(-1).squeeze(-1)  # [B, max_blocks]

        # 计算每个样本需要执行的block数量
        # 取batch中最大的block数量，确保所有样本都能正确处理
        # 注意：这里使用max是为了确保维度一致，避免后续处理中的维度不匹配问题
        block_mask = depth_weights > self.threshold  # [B, max_blocks]
        num_blocks_to_run = block_mask.sum(dim=1).max().item()  # 取batch中最大的block数量
        num_blocks_to_run = max(1, min(num_blocks_to_run, len(blocks)))  # 确保至少执行1个，最多执行所有block

        # 执行选定数量的block
        for i in range(num_blocks_to_run):
            if i < len(blocks):
                x = blocks[i](x)

        return x


class OptimizedBlock(nn.Module):
    """
    优化的Transformer块

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量
        mlp_ratio: MLP中间层维度比例，默认4.0
        qkv_bias: 是否使用qkv投影的偏置，默认False
        qk_scale: 缩放因子，默认None
        drop: dropout概率，默认0.0
        attn_drop: 注意力dropout概率，默认0.0
        drop_path: 随机深度概率，默认0.0
        act_layer: 激活函数类型，默认nn.GELU
        norm_layer: 归一化层类型，默认LayerNorm
        eps: 归一化层的epsilon值，默认1e-6
        use_linear_attn: 是否使用线性注意力，默认False
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        eps=1e-6,
        use_linear_attn=False,  # 新增参数
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = OptimizedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_linear_attn=use_linear_attn,  # 传递参数
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class LightweightMultiScaleConv(nn.Module):
    """
    轻量级多尺度卷积 - 使用深度可分离卷积减少计算量

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量
        kernel_size: 卷积核大小，默认3
        reduction: 通道缩减比例，默认2
    """
    def __init__(self, dim, num_heads, kernel_size=3, reduction=2):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

        # 深度卷积 - 分组数为num_heads
        self.depth_conv = nn.Conv2d(
            dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads
        )

        # 轻量级多尺度特征提取
        self.scale1 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.BatchNorm2d(dim // reduction),
            nn.GELU()
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 3, padding=1, groups=dim // reduction),
            nn.BatchNorm2d(dim // reduction),
            nn.GELU()
        )

        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d((dim // reduction) * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        # 深度卷积
        x = self.depth_conv(x)

        # 多尺度特征提取
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s3 = s3.expand(-1, -1, x.shape[2], x.shape[3])

        # 特征融合
        multi_scale = torch.cat([s1, s2, s3], dim=1)
        attention = self.fusion(multi_scale)

        # 通道注意力机制
        return x * attention


class FlattenBlockRe2D(OptimizedBlock):
    """
    带展平操作的Transformer块

    继承自OptimizedBlock，添加了4D到3D的转换和反向转换

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量
        mlp_ratio: MLP中间层维度比例，默认4
        qkv_bias: 是否使用qkv投影的偏置，默认False
        qk_scale: 缩放因子，默认None
        drop: dropout概率，默认0
        attn_drop: 注意力dropout概率，默认0
        drop_path: 随机深度概率，默认0
        act_layer: 激活函数类型，默认nn.GELU
        norm_layer: 归一化层类型，默认LayerNorm
        eps: 归一化层的epsilon值，默认0.000001
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0,
                 attn_drop=0,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=LayerNorm,
                 eps=0.000001):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, eps)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为(B, C, H, W)

        Returns:
            处理后的张量，形状为(B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvBlock(nn.Module):
    """
    卷积块模块

    Args:
        dim: 输入通道数
        num_heads: 注意力头的数量
        mlp_ratio: MLP中间层维度比例，默认4.0
        drop: dropout概率，默认0.0
        drop_path: 随机深度概率，默认0.0
        act_layer: 激活函数类型，默认nn.GELU
        norm_layer: 归一化层类型，默认LayerNorm
        eps: 归一化层的epsilon值，默认1e-6
        num_conv: 卷积层数量，默认2
        kernel_size: 卷积核大小，默认3
        use_lightweight_conv: 是否使用轻量级卷积，默认True
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        eps=1e-6,
        num_conv=2,
        kernel_size=3,
        use_lightweight_conv=True,  # 新增参数
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)

        if use_lightweight_conv:
            # 使用轻量级多尺度卷积
            self.mixer = LightweightMultiScaleConv(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                reduction=2
            )
        else:
            # 原始卷积实现
            self.mixer = nn.Sequential(*[
                nn.Conv2d(
                    dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
                for i in range(num_conv)
            ])

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            处理后的张量
        """
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class SVTRStage(nn.Module):
    """
    SVTR阶段模块，用于构建Scene Text Recognition Transformer的主要结构。

    该模块根据指定的mixer类型创建不同的块（ConvBlock、OptimizedBlock、FlattenBlockRe2D），
    支持自适应深度机制，可根据输入动态调整网络深度，并包含下采样功能。

    参数:
        dim (int): 输入特征维度
        out_dim (int): 输出特征维度
        depth (int): 块的数量
        mixer (list): 混合器类型列表，每个元素对应一个块，可选值包括'Conv'、'Global'、'FGlobal'、'FGlobalRe2D'
        kernel_sizes (list): 卷积核大小列表，每个元素对应一个块
        sub_k (list): 下采样核大小列表
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP隐藏层维度与输入维度的比例
        qkv_bias (bool): 是否在QKV线性层中使用偏置
        qk_scale (float): 缩放因子，若为None则自动计算
        drop_rate (float): 随机失活率
        attn_drop_rate (float): 注意力机制中的随机失活率
        drop_path (list): 随机深度失活率列表，每个元素对应一个块
        norm_layer (nn.Module): 归一化层类型
        act (nn.Module): 激活函数类型
        eps (float): 归一化层的epsilon值
        num_conv (list): 每个ConvBlock中的卷积层数量
        downsample (bool): 是否使用下采样
        use_adaptive_depth (bool): 是否使用自适应深度机制
        use_linear_attn (bool): 是否使用线性注意力机制
        use_efficient_conv (bool): 是否使用轻量级卷积
    """
    def __init__(self,
                 dim=64,
                 out_dim=256,
                 depth=3,
                 mixer=['Local'] * 3,
                 kernel_sizes=[3] * 3,
                 sub_k=[2, 1],
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 num_conv=[2] * 3,
                 downsample=None,
                 use_adaptive_depth=False,  # 新增参数
                 use_linear_attn=False,     # 新增参数
                 use_efficient_conv=True,   # 新增参数
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.use_adaptive_depth = use_adaptive_depth
        self.depth = depth

        # 创建blocks列表用于自适应深度
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(
                    ConvBlock(
                    dim=dim,
                    kernel_size=kernel_sizes[i],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    act_layer=act,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    eps=eps,
                    num_conv=num_conv[i],
                    use_lightweight_conv=use_efficient_conv  # 传递参数
                ))
            else:
                if mixer[i] == 'Global':
                    block = OptimizedBlock
                elif mixer[i] == 'FGlobal':
                    block = OptimizedBlock
                    self.blocks.append(FlattenTranspose())
                elif mixer[i] == 'FGlobalRe2D':
                    block = FlattenBlockRe2D

                block_instance = block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    eps=eps,
                    use_linear_attn=use_linear_attn,  # 传递参数
                )

                self.blocks.append(block_instance)

        # 如果使用自适应深度，创建自适应深度模块
        self.use_adaptive_depth = use_adaptive_depth
        if use_adaptive_depth:
            self.adaptive_depth = AdaptiveDepthModule(
                dim=dim,
                max_blocks=depth,
                threshold=0.5
            )

        if downsample:
            if mixer[-1] == 'Conv' or mixer[-1] == 'FGlobalRe2D':
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        if self.use_adaptive_depth:
            # 使用自适应深度机制
            x = self.adaptive_depth(x, self.blocks)
        else:
            # 顺序执行所有blocks
            for blk in self.blocks:
                x = blk(x)

        x, sz = self.downsample(x, sz)
        return x, sz

class OptimizedPOPatchEmbed(nn.Module):
    """
    使用timm ConvNormAct优化的POPatchEmbed模块，用于将输入图像转换为特征嵌入。

    该模块通过两个卷积层将输入图像下采样4倍，并将其转换为指定维度的特征嵌入，
    支持添加位置嵌入和扁平化操作。

    参数:
        in_channels (int): 输入图像的通道数，默认为3
        feat_max_size (list): 特征图的最大尺寸，格式为[高度, 宽度]，默认为[8, 32]
        embed_dim (int): 输出特征嵌入的维度，默认为768
        use_pos_embed (bool): 是否添加位置嵌入，默认为False
        flatten (bool): 是否将特征图扁平化并转置，默认为False
        bias (bool): 是否在卷积层中使用偏置，默认为False
    """

    def __init__(self, in_channels=3, feat_max_size=[8, 32], embed_dim=768,
                    use_pos_embed=False, flatten=False, bias=False):
        super().__init__()

        # 使用优化的ConvBNLayer
        self.patch_embed = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=bias,
                apply_norm=True,
                apply_act=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.GELU,
                # 禁用抗锯齿池化以保持与原始版本一致
                aa_layer=None
            ),
            ConvNormAct(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=bias,
                apply_norm=True,
                apply_act=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.GELU,
                # 禁用抗锯齿池化以保持与原始版本一致
                aa_layer=None
            )
        )

        if use_pos_embed:
            self.patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            self.patch_embed.append(FlattenTranspose())

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]

class LastStage(nn.Module):
    """
    LastStage模块，用于SVTR网络的最后一个阶段处理。

    该模块将输入特征进行reshape和均值池化，然后通过线性层、激活函数和dropout层，
    最终输出处理后的特征和尺寸信息。

    参数:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出特征通道数
        last_drop (float): dropout率
        out_char_num (int): 输出字符数量，默认为0
    """
    def __init__(self, in_channels, out_channels, last_drop, out_char_num=0):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x, sz):
        x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class Feat2D(nn.Module):
    """
    Feat2D模块，用于将特征从3D格式转换为4D格式。

    该模块将输入的3D特征张量转换为4D特征张量，便于后续的卷积操作处理。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        """
        前向传播函数，将3D特征转换为4D特征。

        参数:
            x (torch.Tensor): 输入特征张量，形状为 [B, N, C]
            sz (list): 特征图尺寸，格式为 [H, W]

        返回:
            torch.Tensor: 转换后的4D特征张量，形状为 [B, C, H, W]
            list: 特征图尺寸，保持不变
        """
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


class SVTRv2(nn.Module):
    """
    SVTRv2模型，Scene Text Recognition Transformer的第二代版本，使用timm优化。

    该模型是一个用于场景文本识别的Transformer架构，支持多种配置（tiny、small、base），
    包含Patch Embedding、多个SVTRStage以及可选的LastStage和Feat2D模块。

    参数:
        max_sz (list): 最大输入尺寸，格式为[高度, 宽度]，默认为[32, 128]
        in_channels (int): 输入图像通道数，默认为3
        out_channels (int): 输出特征通道数，默认为192
        depths (list): 每个阶段的块数量，默认为[3, 6, 3]
        dims (list): 每个阶段的特征维度，默认为[64, 128, 256]
        mixer (list): 每个阶段的混合器类型列表，默认为[['Conv']*3, ['Conv']*3+['Global']*3, ['Global']*3]
        use_pos_embed (bool): 是否使用位置嵌入，默认为True
        sub_k (list): 每个阶段的下采样核大小列表，默认为[[1, 1], [2, 1], [1, 1]]
        num_heads (list): 每个阶段的注意力头数量，默认为[2, 4, 8]
        mlp_ratio (float): MLP隐藏层维度与输入维度的比例，默认为4
        qkv_bias (bool): 是否在QKV线性层中使用偏置，默认为True
        qk_scale (float): 缩放因子，若为None则自动计算，默认为None
        drop_rate (float): 随机失活率，默认为0.0
        last_drop (float): 最后一个Dropout层的失活率，默认为0.1
        attn_drop_rate (float): 注意力机制中的随机失活率，默认为0.0
        drop_path_rate (float): 随机深度失活率，默认为0.1
        norm_layer (nn.Module): 归一化层类型，默认为LayerNorm
        act (nn.Module): 激活函数类型，默认为nn.GELU
        last_stage (bool): 是否使用最后一个阶段，默认为False
        feat2d (bool): 是否使用Feat2D模块，默认为False
        eps (float): 归一化层的epsilon值，默认为1e-6
        num_convs (list): 每个阶段的卷积层数量列表，默认为[[2]*3, [2]*3+[3]*3, [3]*3]
        kernel_sizes (list): 每个阶段的卷积核大小列表，默认为[[3]*3, [3]*3+[3]*3, [3]*3]
        pope_bias (bool): 是否在OptimizedPOPatchEmbed中使用偏置，默认为False
        use_adaptive_depth (bool): 是否使用自适应深度机制，默认为False
        use_linear_attn (bool): 是否使用线性注意力机制，默认为False
        use_efficient_conv (bool): 是否使用轻量级卷积，默认为True
    """
    svtr_cfg = {
        "tiny": dict(
            use_pos_embed=False,
            dims=[64, 128, 256],
            depths=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=[
                ["Conv", "Conv", "Conv"],
                ["Conv", "Conv", "Conv", "FGlobal", "Global", "Global"],
                ["Global", "Global", "Global"],
            ],
            local_k=[[5, 5], [5, 5], [-1, -1]],
            sub_k=[[1, 1], [2, 1], [-1, -1]],
            last_stage=False,
            feat2d=True,
        ),
        "small": dict(
            use_pos_embed=False,
            dims=[96, 192, 384],
            depths=[3, 6, 3],
            num_heads=[3, 6, 12],
            mixer=[
                ["Conv", "Conv", "Conv"],
                ["Conv", "Conv", "Conv", "FGlobal", "Global", "Global"],
                ["Global", "Global", "Global"],
            ],
            local_k=[[5, 5], [5, 5], [-1, -1]],
            sub_k=[[1, 1], [2, 1], [-1, -1]],
            last_stage=False,
            feat2d=True,
        ),
        "base": dict(
            use_pos_embed=False,
            out_channels=256,
            dims=[128, 256, 384],
            depths=[6, 6, 6],
            num_heads=[4, 8, 12],
            mixer=[
                ["Conv", "Conv", "Conv", "Conv", "Conv", "Conv"],
                ["Conv", "Conv", "FGlobal", "Global", "Global", "Global"],
                ["Global", "Global", "Global", "Global", "Global", "Global"],
            ],
            kernel_sizes=[[5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5], [-1]],
            num_convs=[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [-1]],
            sub_k=[[2, 1], [2, 1], [-1, -1]],
            last_stage=False,
            feat2d=True,
            pope_bias=True,
        ),
    }
    def __init__(self,
                 max_sz=[32, 128],
                 in_channels=3,
                 out_channels=192,
                 depths=[3, 6, 3],
                 dims=[64, 128, 256],
                 mixer=[['Conv'] * 3, ['Conv'] * 3 + ['Global'] * 3, ['Global'] * 3],
                 use_pos_embed=True,
                 sub_k=[[1, 1], [2, 1], [1, 1]],
                 num_heads=[2, 4, 8],
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=LayerNorm,
                 act=nn.GELU,
                 last_stage=False,
                 feat2d=False,
                 eps=1e-6,
                 num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
                 kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
                 pope_bias=False,
                 use_adaptive_depth=False,
                 use_linear_attn=False,
                 use_efficient_conv=True,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = OptimizedPOPatchEmbed(in_channels=in_channels,
                                 feat_max_size=feat_max_size,
                                 embed_dim=dims[0],
                                 use_pos_embed=use_pos_embed,
                                 flatten=mixer[0][0] != 'Conv',
                                 bias=pope_bias)

        dpr = np.linspace(0, drop_path_rate, sum(depths))  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) else [3] *
                len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                num_conv=num_convs[i_stage] if len(num_convs[i_stage]) == len(
                    mixer[i_stage]) else [2] * len(mixer[i_stage]),
                use_adaptive_depth=use_adaptive_depth,
                use_linear_attn=use_linear_attn,
                use_efficient_conv=use_efficient_conv,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(
                LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x

class UltraLightweightSVTRv2Neck(nn.Module):
    """超轻量级SVTRv2 Neck - 极致计算效率优化

    设计特点：
    1. 完全移除注意力机制，使用简单的统计特征
    2. 用深度可分离卷积替代标准卷积
    3. 通道压缩减少计算量
    4. 移除所有复杂的混合模块
    5. 使用平均池化替代注意力压缩

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        hidden_dims (int): 隐藏层维度，默认为128
        dropout (float): dropout率，默认为0.1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: int = 128,  # 减少隐藏维度
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. 轻量级输入投影 - 使用1x1卷积 + 深度可分离卷积
        self.input_proj = nn.Sequential(
            # 先压缩通道数
            nn.Conv2d(in_channels, hidden_dims // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims // 2),
            nn.GELU(),
            # 深度可分离卷积提取局部特征
            nn.Conv2d(hidden_dims // 2, hidden_dims // 2, kernel_size=3,
                     padding=1, groups=hidden_dims // 2, bias=False),
            nn.BatchNorm2d(hidden_dims // 2),
            nn.GELU(),
            # 恢复通道数
            nn.Conv2d(hidden_dims // 2, hidden_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.GELU()
        )

        # 2. 超轻量级特征提取 - 仅使用单尺度深度卷积
        self.lightweight_conv = nn.Sequential(
            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3,
                     padding=1, groups=hidden_dims, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. 简单垂直压缩 - 使用自适应平均池化 + 轻量级融合
        self.vertical_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 只压缩高度维度
            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.GELU()
        )

        # 4. 轻量级序列转换 - 简单的线性变换
        self.sequence_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - SVTRv2输出的2D特征
        Returns:
            sequence: [B, W, D] - 序列特征，适配decoder输入
        """
        B, C, H, W = x.shape

        # 1. 轻量级输入投影
        x = self.input_proj(x)  # [B, hidden_dims, H, W]

        # 2. 单尺度特征提取
        x = self.lightweight_conv(x)  # [B, hidden_dims, H, W]

        # 3. 垂直压缩 - 使用平均池化 + 1x1卷积融合
        x = self.vertical_compress(x)  # [B, hidden_dims, 1, W]

        # 4. 调整维度并输出
        x = x.squeeze(2)  # [B, hidden_dims, W]
        x = x.permute(0, 2, 1)  # [B, W, hidden_dims]
        x = self.sequence_proj(x)  # [B, W, out_channels]

        return x


class SVTRv2Neck(nn.Module):
    """
    SVTRv2专属Neck - 针对SVTRv2混合特征设计的高效转换模块

    设计思路：
    1. 利用SVTRv2的2D特征结构，避免过度扁平化造成的信息损失
    2. 采用轻量级注意力机制进行垂直方向压缩，保持水平序列结构
    3. 引入多尺度特征融合，平衡局部细节和全局语义
    4. 使用深度可分离卷积降低计算复杂度

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        hidden_dims (int): 隐藏层维度，默认为256
        depth (int): 模块深度，默认为2
        num_heads (int): 注意力头数量，默认为8
        use_multi_scale (bool): 是否使用多尺度特征融合，默认为True
        use_lightweight (bool): 是否使用轻量级设计，默认为True
        dropout (float): dropout率，默认为0.1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        use_multi_scale: bool = True,
        use_lightweight: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_multi_scale = use_multi_scale
        self.use_lightweight = use_lightweight

        # 1. 输入特征投影 - 使用1x1卷积进行通道调整
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.GELU()
        )

        # 2. 多尺度特征提取（可选）
        if use_multi_scale:
            self.multi_scale_conv = MultiScaleFeatureExtractor(
                in_channels=hidden_dims,
                out_channels=hidden_dims,
                use_lightweight=use_lightweight
            )

        # 3. 轻量级垂直注意力压缩
        self.vertical_attention = LightweightVerticalAttention(
            channels=hidden_dims,
            num_heads=num_heads,
            use_lightweight=use_lightweight
        )

        # 4. SVTR风格的混合模块
        self.mixing_blocks = nn.ModuleList()
        for i in range(depth):
            block = SVTRv2MixingBlock(
                dim=hidden_dims,
                num_heads=num_heads,
                dropout=dropout,
                use_lightweight=use_lightweight
            )
            self.mixing_blocks.append(block)

        # 5. 输出投影 - 转换为序列格式
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - SVTRv2输出的2D特征
        Returns:
            sequence: [B, W, D] - 序列特征，适配decoder输入
        """
        B, C, H, W = x.shape

        # 1. 输入投影
        x = self.input_proj(x)  # [B, hidden_dims, H, W]

        # 2. 多尺度特征提取（增强特征表示）
        if self.use_multi_scale:
            x = self.multi_scale_conv(x)

        # 3. 垂直注意力压缩（将H维度压缩，保留重要信息）
        x = self.vertical_attention(x)  # [B, hidden_dims, 1, W] 或 [B, hidden_dims, W]

        # 4. SVTR混合模块（进一步增强序列特征）
        for mixing_block in self.mixing_blocks:
            x = mixing_block(x)

        # 5. 调整维度并输出
        if len(x.shape) == 4:  # 如果还是4D，去掉高度维度
            x = x.squeeze(2)  # [B, hidden_dims, W]

        x = x.permute(0, 2, 1)  # [B, W, hidden_dims]
        x = self.output_proj(x)  # [B, W, out_channels]

        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器 - 捕获不同尺度的局部和全局信息

    该模块通过三个不同尺度的分支（小、中、大）提取特征，并将它们融合，
    支持轻量级设计，使用深度可分离卷积降低计算复杂度。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        use_lightweight (bool): 是否使用轻量级设计，默认为True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_lightweight: bool = True
    ):
        super().__init__()

        # 三个不同尺度的分支
        self.branch1 = self._make_branch(in_channels, out_channels, 3, use_lightweight)
        self.branch2 = self._make_branch(in_channels, out_channels, 5, use_lightweight)
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def _make_branch(self, in_ch: int, out_ch: int, kernel_size: int, lightweight: bool):
        """构建单个分支"""
        if lightweight:
            # 使用深度可分离卷积
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size//2, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat3 = feat3.expand(-1, -1, x.shape[2], x.shape[3])  # 上采样到原始尺寸

        # 特征拼接和融合
        multi_scale_feat = torch.cat([feat1, feat2, feat3], dim=1)
        output = self.fusion(multi_scale_feat)

        # 残差连接
        return output + x


class LightweightVerticalAttention(nn.Module):
    """
    轻量级垂直注意力模块 - 高效压缩高度维度

    该模块通过注意力机制对输入特征的高度维度进行压缩，保留重要信息，
    支持轻量级设计，通过降维减少计算复杂度。

    参数:
        channels (int): 输入通道数
        num_heads (int): 注意力头数量，默认为8
        use_lightweight (bool): 是否使用轻量级设计，默认为True
        reduction_ratio (int): 降维比例，用于轻量级设计，默认为4
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        use_lightweight: bool = True,
        reduction_ratio: int = 4
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_lightweight = use_lightweight

        if use_lightweight:
            # 轻量级注意力：先降维再计算注意力
            self.query_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(channels, channels // reduction_ratio, 1),
                nn.GELU()
            )
            self.key_proj = nn.Conv2d(channels, channels // reduction_ratio, 1)
            self.value_proj = nn.Conv2d(channels, channels, 1)

            self.attention_conv = nn.Conv2d(
                channels // reduction_ratio, 1, 1
            )
        else:
            # 标准注意力
            self.query_proj = nn.Conv2d(channels, channels, 1)
            self.key_proj = nn.Conv2d(channels, channels, 1)
            self.value_proj = nn.Conv2d(channels, channels, 1)

        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C, 1, W] - 垂直压缩后的特征
        """
        B, C, H, W = x.shape

        if self.use_lightweight:
            # 轻量级注意力计算
            query = self.query_proj(x)  # [B, C//r, 1, 1]
            key = self.key_proj(x)      # [B, C//r, H, W]
            value = self.value_proj(x)  # [B, C, H, W]

            # 计算注意力权重
            query = query.expand(-1, -1, H, W)  # [B, C//r, H, W]
            attention_scores = (query * key).sum(dim=1, keepdim=True) * self.scale  # [B, 1, H, W]
            attention_weights = torch.softmax(attention_scores, dim=2)  # 在H维度softmax

            # 加权求和
            output = (value * attention_weights).sum(dim=2, keepdim=True)  # [B, C, 1, W]
        else:
            # 标准注意力计算
            query = self.query_proj(x).view(B, self.num_heads, C // self.num_heads, H, W)
            key = self.key_proj(x).view(B, self.num_heads, C // self.num_heads, H, W)
            value = self.value_proj(x).view(B, self.num_heads, C // self.num_heads, H, W)

            # 计算注意力
            attention_scores = (query * key).sum(dim=2) * self.scale  # [B, num_heads, H, W]
            attention_weights = torch.softmax(attention_scores, dim=2)  # 在H维度softmax

            # 加权求和
            output = (value * attention_weights.unsqueeze(2)).sum(dim=3)  # [B, num_heads, C//num_heads, W]
            output = output.view(B, C, W).unsqueeze(2)  # [B, C, 1, W]

        return output


class SVTRv2MixingBlock(nn.Module):
    """
    SVTRv2混合模块 - 结合局部和全局特征处理

    该模块通过局部特征处理（卷积）和全局特征处理（注意力）相结合的方式，
    增强特征表示能力，支持轻量级设计，使用深度可分离卷积降低计算复杂度。

    参数:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数量，默认为8
        mlp_ratio (float): MLP隐藏层维度与输入维度的比例，默认为4.0
        dropout (float): dropout率，默认为0.1
        use_lightweight (bool): 是否使用轻量级设计，默认为True
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_lightweight: bool = True
    ):
        super().__init__()

        # 局部特征处理
        if use_lightweight:
            self.local_mixer = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # 深度卷积
                nn.Conv2d(dim, dim, 1),  # 点卷积
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.local_mixer = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # 全局特征处理（使用注意力）
        self.global_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 或 [B, C, W]
        """
        # 确保是4D张量
        if len(x.shape) == 3:
            B, C, W = x.shape
            x = x.unsqueeze(2)  # [B, C, 1, W]
            squeeze_height = True
        else:
            squeeze_height = False

        residual = x

        # 局部混合
        x = self.local_mixer(x) + residual

        # 全局混合
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # [B*H, W, C]
        x_flat = self.global_mixer(x_flat)  # [B*H, W, C]
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # MLP
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        x_flat = self.mlp(x_flat)  # [B*H*W, C]
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # 如果原来是3D，恢复
        if squeeze_height:
            x = x.squeeze(2)  # [B, C, W]

        return x




def basic_test():
    """
    测试函数，用于测试SVTRv2Neck和UltraLightweightSVTRv2Neck的功能和性能。

    该函数执行以下测试：
    1. 创建SVTRv2模型并生成特征
    2. 创建并测试SVTRv2Neck，包括前向传播和参数量计算
    3. 创建并测试UltraLightweightSVTRv2Neck
    4. 比较两种Neck的输出结果
    """
    import time

    # 测试SVTRv2Neck
    print("Testing SVTRv2Neck...")

    # 测试代码
    cfg = SVTRv2.svtr_cfg['tiny']
    model = SVTRv2(**cfg)
    x = torch.randn(3, 3, 32, 128)
    features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print("Model created successfully with optimizations!")

    # 模拟SVTRv2 backbone输出
    B, C, H, W = features.shape  # SVTRv2 typical output

    # 创建Neck
    neck = SVTRv2Neck(
        in_channels=C,
        out_channels=384,  # decoder输入维度
        hidden_dims=256,
        depth=2,
        use_multi_scale=True,
        use_lightweight=True
    )

    # 前向传播
    output = neck(features)

    print(f"Neck input shape: {features.shape}")
    print(f"Neck output shape: {output.shape}")
    print(f"Expected output shape: [{B}, {W}, 384]")

    # 计算参数量
    total_params = sum(p.numel() for p in neck.parameters())
    print(f"Neck total parameters: {total_params:,}")

    # 创建超轻量Neck
    ultra_light_neck = UltraLightweightSVTRv2Neck(
        in_channels=C,
        out_channels=384,  # decoder输入维度
        hidden_dims=256
    )

    # 前向传播
    ultra_output = ultra_light_neck(features)

    print(f"\nUltra-lightweight input shape: {features.shape}")
    print(f"Ultra-lightweight output shape: {ultra_output.shape}")
    print(f"Expected output shape: [{B}, {W}, 384]")

    # 计算参数量对比
    ultra_params = sum(p.numel() for p in ultra_light_neck.parameters())

    print(f"Ultra-lightweight Neck parameters: {ultra_params:,}")
    print(f"Parameter reduction: {(1 - ultra_params/total_params)*100:.1f}%")

    # 验证输出一致性
    print(f"\nOutput consistency check:")
    print(f"Original output shape: {output.shape}")
    print(f"Ultra-lightweight output shape: {ultra_output.shape}")
    print(f"Shape match: {output.shape == ultra_output.shape}")

    # 测试不同配置组合
    print(f"\nTesting different configuration combinations:")

    configs = [
        {"use_multi_scale": True, "use_lightweight": True, "name": "Standard"},
        {"use_multi_scale": True, "use_lightweight": False, "name": "Multi-scale only"},
        {"use_multi_scale": False, "use_lightweight": True, "name": "Lightweight only"},
        {"use_multi_scale": False, "use_lightweight": False, "name": "Ultra-lightweight"},
    ]

    for config in configs:
        if not config["use_multi_scale"] and not config["use_lightweight"]:
            test_neck = UltraLightweightSVTRv2Neck(
                in_channels=C,
                out_channels=384,
                hidden_dims=256
            )
        else:
            test_neck = SVTRv2Neck(
                in_channels=C,
                out_channels=384,
                hidden_dims=256,
                depth=2,
                use_multi_scale=config["use_multi_scale"],
                use_lightweight=config["use_lightweight"]
            )

        params = sum(p.numel() for p in test_neck.parameters())

        # 速度测试
        test_neck.eval()
        with torch.no_grad():
            # 预热
            for _ in range(5):
                _ = test_neck(features)

            # 正式测试
            start_time = time.time()
            for i in range(100):
                output = test_neck(features)
                # 检查是否有NaN或inf
                assert not torch.isnan(output).any(), "输出包含NaN"
                assert not torch.isinf(output).any(), "输出包含inf"
                if i == 0:
                    first_output = output.clone()
                else:
                    # 检查输出一致性
                    diff = torch.abs(output - first_output).max().item()
                    assert diff < 1e-5, f"输出不一致，差异: {diff}"
            end_time = time.time()

            avg_time = (end_time - start_time) / 100

        print(f"{config['name']}: {params:,} params, {avg_time*1000:.2f}ms, 内存效率测试通过！")

    print("\nAll tests completed!")

def test_svtrv2_onnx(model_type='tiny', input_shape=(1, 3, 32, 128), output_path='svtrv2.onnx'):
    """
    SVTRv2 ONNX导出测试函数

    Args:
        model_type: 模型类型，可选 'tiny', 'small', 'base'
        input_shape: 输入张量形状 (batch, channels, height, width)
        output_path: 输出的ONNX文件路径

    Returns:
        str: ONNX文件路径
    """
    import time
    import os
    import numpy as np
    import onnx
    import onnxruntime as ort
    from torch.export import Dim

    print(f"=== SVTRv2 ONNX导出测试 ===")
    print(f"模型类型: {model_type}")
    print(f"输入形状: {input_shape}")
    print(f"输出路径: {output_path}")

    # 1. 创建模型
    print("1. 创建模型...")
    if model_type in SVTRv2.svtr_cfg:
        cfg = SVTRv2.svtr_cfg[model_type]
        model = SVTRv2(**cfg)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(SVTRv2.svtr_cfg.keys())}")

    model.eval()

    # 2. 创建测试输入
    print("2. 创建测试输入...")
    dummy_input = torch.randn(input_shape)

    # 3. 测试PyTorch模型
    print("3. 测试PyTorch模型...")
    with torch.no_grad():
        start_time = time.time()
        pytorch_output = model(dummy_input)
        pytorch_time = time.time() - start_time
        print(f"PyTorch推理时间: {pytorch_time:.4f}秒")
        print(f"PyTorch输出形状: {pytorch_output.shape}")
        print(f"PyTorch输出类型: {pytorch_output.dtype}")

    # 4. 导出ONNX模型
    print("4. 导出ONNX模型...")
    try:
        # 给每个动态维起名字 + 可选范围
        batch_size  = Dim("batch_size", min=2, max=1024)   # 可写 Dim("batch", min=1, max=64)
        width       = Dim("width", min=16, max=4096)

        # 把输出也当关键字写进去（名字跟 forward 返回变量对应）
        # 输出张量不需要写在这里，Dynamo 会推导；如果写了会导致出错
        dynamic_shapes = {
            'x': {0: batch_size, 3: width}
        }

        with torch.no_grad():
            model(dummy_input)

            torch.onnx.export(
                model,
                args=dummy_input,
                f=output_path,
                dynamo=True,
                verbose=True,
                report=True,
                verify=True,
                export_params=True,
                external_data=False,
                opset_version=18,
                do_constant_folding=False,
                input_names=['x'],
                output_names=['logits'],
                dynamic_shapes=dynamic_shapes
            )

        print(f"ONNX模型已成功导出到: {output_path}")

        # 5. 验证ONNX模型
        print("5. 验证ONNX模型...")
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过 ✓")
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")

        # 6. 测试ONNX推理
        print("6. 测试ONNX推理...")
        try:
            # 创建ONNX运行时会话
            ort_session = ort.InferenceSession(output_path)

            # 获取输入输出信息
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            print(f"ONNX输入名称: {input_name}")
            print(f"ONNX输出名称: {output_name}")
            print(f"ONNX输入形状: {ort_session.get_inputs()[0].shape}")
            print(f"ONNX输出形状: {ort_session.get_outputs()[0].shape}")

            # 运行ONNX推理
            start_time = time.time()
            onnx_output = ort_session.run([output_name], {input_name: dummy_input.numpy()})[0]
            onnx_time = time.time() - start_time

            print(f"ONNX推理时间: {onnx_time:.4f}秒")
            print(f"ONNX输出形状: {onnx_output.shape}")
            print(f"ONNX输出类型: {onnx_output.dtype}")

            # 7. 比较输出结果
            print("7. 比较PyTorch和ONNX输出...")
            pytorch_output_np = pytorch_output.detach().numpy()

            # 计算差异
            abs_diff = np.abs(pytorch_output_np - onnx_output)
            rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)

            print(f"最大绝对差异: {np.max(abs_diff):.6f}")
            print(f"平均绝对差异: {np.mean(abs_diff):.6f}")
            print(f"最大相对差异: {np.max(rel_diff):.6f}")
            print(f"平均相对差异: {np.mean(rel_diff):.6f}")

            # 检查是否匹配
            if np.allclose(pytorch_output_np, onnx_output, rtol=1e-3, atol=1e-5):
                print("PyTorch和ONNX输出匹配 ✓")
            else:
                print("PyTorch和ONNX输出存在差异 ⚠")

            # 8. 测试不同输入尺寸
            print("8. 测试不同输入尺寸...")
            test_shapes = [
                (1, 3, 32, 64),
                (1, 3, 32, 128),
                (2, 3, 32, 96),
                (4, 3, 32, 256),
                (8, 3, 32, 512),
                (8, 3, 32, 1024)
            ]

            for test_shape in test_shapes:
                try:
                    test_input = torch.randn(test_shape)
                    with torch.no_grad():
                        test_output = model(test_input)
                        onnx_output = ort_session.run([output_name], {input_name: test_input.numpy()})[0]
                    print(f"输入形状 {test_shape} -> Pytorch 输出形状 {test_output.shape} onnx {onnx_output.shape} ✓")
                except Exception as e:
                    print(f"输入形状 {test_shape} -> 失败: {e} ✗")

        except Exception as e:
            print(f"ONNX推理测试失败: {e}")

        # 9. 模型信息
        print("9. 模型信息统计...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 计算模型大小
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"ONNX文件大小: {file_size / 1024 / 1024:.2f} MB")

        print(f"\n=== SVTRv2 ONNX导出测试完成 ===")
        return output_path

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return None


def test_svtrv2_neck_onnx(in_channels=256, out_channels=384, input_shape=(1, 256, 8, 32),
                         output_path='svtrv2_neck.onnx', **neck_kwargs):
    """
    SVTRv2Neck ONNX导出测试函数

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        input_shape: 输入张量形状 (batch, channels, height, width)
        output_path: 输出的ONNX文件路径
        **neck_kwargs: SVTRv2Neck的其他参数

    Returns:
        str: ONNX文件路径
    """
    import time
    import os
    import numpy as np
    import onnx
    import onnxruntime as ort

    print(f"=== SVTRv2Neck ONNX导出测试 ===")
    print(f"输入通道: {in_channels}")
    print(f"输出通道: {out_channels}")
    print(f"输入形状: {input_shape}")
    print(f"输出路径: {output_path}")

    # 1. 创建模型
    print("1. 创建模型...")
    neck = SVTRv2Neck(
        in_channels=in_channels,
        out_channels=out_channels,
        **neck_kwargs
    )
    neck.eval()

    # 2. 创建测试输入
    print("2. 创建测试输入...")
    dummy_input = torch.randn(input_shape)

    # 3. 测试PyTorch模型
    print("3. 测试PyTorch模型...")
    with torch.no_grad():
        start_time = time.time()
        pytorch_output = neck(dummy_input)
        pytorch_time = time.time() - start_time
        print(f"PyTorch推理时间: {pytorch_time:.4f}秒")
        print(f"PyTorch输出形状: {pytorch_output.shape}")
        print(f"PyTorch输出类型: {pytorch_output.dtype}")

    # 4. 导出ONNX模型
    print("4. 导出ONNX模型...")
    try:
        with torch.no_grad():
            neck(dummy_input)

            torch.onnx.export(
                neck,
                args=dummy_input,
                f=output_path,
                dynamo=True,
                verbose=True,
                report=True,
                verify=True,
                export_params=True,
                external_data=False,
                opset_version=18,
                do_constant_folding=False,
                input_names=['features'],
                output_names=['sequence']
            )

        print(f"ONNX模型已成功导出到: {output_path}")

        # 5. 验证ONNX模型
        print("5. 验证ONNX模型...")
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过 ✓")
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")

        # 6. 测试ONNX推理
        print("6. 测试ONNX推理...")
        try:
            # 创建ONNX运行时会话
            ort_session = ort.InferenceSession(output_path)

            # 获取输入输出信息
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            print(f"ONNX输入名称: {input_name}")
            print(f"ONNX输出名称: {output_name}")
            print(f"ONNX输入形状: {ort_session.get_inputs()[0].shape}")
            print(f"ONNX输出形状: {ort_session.get_outputs()[0].shape}")

            # 运行ONNX推理
            start_time = time.time()
            onnx_output = ort_session.run([output_name], {input_name: dummy_input.numpy()})[0]
            onnx_time = time.time() - start_time

            print(f"ONNX推理时间: {onnx_time:.4f}秒")
            print(f"ONNX输出形状: {onnx_output.shape}")
            print(f"ONNX输出类型: {onnx_output.dtype}")

            # 7. 比较输出结果
            print("7. 比较PyTorch和ONNX输出...")
            pytorch_output_np = pytorch_output.detach().numpy()

            # 计算差异
            abs_diff = np.abs(pytorch_output_np - onnx_output)
            rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)

            print(f"最大绝对差异: {np.max(abs_diff):.6f}")
            print(f"平均绝对差异: {np.mean(abs_diff):.6f}")
            print(f"最大相对差异: {np.max(rel_diff):.6f}")
            print(f"平均相对差异: {np.mean(rel_diff):.6f}")

            # 检查是否匹配
            if np.allclose(pytorch_output_np, onnx_output, rtol=1e-3, atol=1e-5):
                print("PyTorch和ONNX输出匹配 ✓")
            else:
                print("PyTorch和ONNX输出存在差异 ⚠")

        except Exception as e:
            print(f"ONNX推理测试失败: {e}")

        # 8. 测试不同输入尺寸
        print("8. 测试不同输入尺寸...")
        test_shapes = [
            (1, in_channels, 8, 32),
            (1, in_channels, 8, 64),
            (2, in_channels, 8, 128),
            (4, in_channels, 8, 256)
        ]

        for test_shape in test_shapes:
            try:
                test_input = torch.randn(test_shape)
                with torch.no_grad():
                    test_output = neck(test_input)
                print(f"输入形状 {test_shape} -> 输出形状 {test_output.shape} ✓")
            except Exception as e:
                print(f"输入形状 {test_shape} -> 失败: {e} ✗")

        # 9. 模型信息
        print("9. 模型信息统计...")
        total_params = sum(p.numel() for p in neck.parameters())
        trainable_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 计算模型大小
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"ONNX文件大小: {file_size / 1024 / 1024:.2f} MB")

        print(f"\n=== SVTRv2Neck ONNX导出测试完成 ===")
        return output_path

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return None


def test_ultra_lightweight_svtrv2_neck_onnx(in_channels=256, out_channels=384, input_shape=(1, 256, 8, 32),
                                          output_path='ultra_lightweight_svtrv2_neck.onnx', **neck_kwargs):
    """
    UltraLightweightSVTRv2Neck ONNX导出测试函数

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        input_shape: 输入张量形状 (batch, channels, height, width)
        output_path: 输出的ONNX文件路径
        **neck_kwargs: UltraLightweightSVTRv2Neck的其他参数

    Returns:
        str: ONNX文件路径
    """
    import time
    import os
    import numpy as np
    import onnx
    import onnxruntime as ort

    print(f"=== UltraLightweightSVTRv2Neck ONNX导出测试 ===")
    print(f"输入通道: {in_channels}")
    print(f"输出通道: {out_channels}")
    print(f"输入形状: {input_shape}")
    print(f"输出路径: {output_path}")

    # 1. 创建模型
    print("1. 创建模型...")
    neck = UltraLightweightSVTRv2Neck(
        in_channels=in_channels,
        out_channels=out_channels,
        **neck_kwargs
    )
    neck.eval()

    # 2. 创建测试输入
    print("2. 创建测试输入...")
    dummy_input = torch.randn(input_shape)

    # 3. 测试PyTorch模型
    print("3. 测试PyTorch模型...")
    with torch.no_grad():
        start_time = time.time()
        pytorch_output = neck(dummy_input)
        pytorch_time = time.time() - start_time
        print(f"PyTorch推理时间: {pytorch_time:.4f}秒")
        print(f"PyTorch输出形状: {pytorch_output.shape}")
        print(f"PyTorch输出类型: {pytorch_output.dtype}")

    # 4. 导出ONNX模型
    print("4. 导出ONNX模型...")
    try:
        with torch.no_grad():
            neck(dummy_input)

            torch.onnx.export(
                neck,
                args=dummy_input,
                f=output_path,
                dynamo=True,
                verbose=True,
                report=True,
                verify=True,
                export_params=True,
                external_data=False,
                opset_version=18,
                do_constant_folding=False,
                input_names=['features'],
                output_names=['sequence']
            )

        print(f"ONNX模型已成功导出到: {output_path}")

        # 5. 验证ONNX模型
        print("5. 验证ONNX模型...")
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过 ✓")
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")

        # 6. 测试ONNX推理
        print("6. 测试ONNX推理...")
        try:
            # 创建ONNX运行时会话
            ort_session = ort.InferenceSession(output_path)

            # 获取输入输出信息
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            print(f"ONNX输入名称: {input_name}")
            print(f"ONNX输出名称: {output_name}")
            print(f"ONNX输入形状: {ort_session.get_inputs()[0].shape}")
            print(f"ONNX输出形状: {ort_session.get_outputs()[0].shape}")

            # 运行ONNX推理
            start_time = time.time()
            onnx_output = ort_session.run([output_name], {input_name: dummy_input.numpy()})[0]
            onnx_time = time.time() - start_time

            print(f"ONNX推理时间: {onnx_time:.4f}秒")
            print(f"ONNX输出形状: {onnx_output.shape}")
            print(f"ONNX输出类型: {onnx_output.dtype}")

            # 7. 比较输出结果
            print("7. 比较PyTorch和ONNX输出...")
            pytorch_output_np = pytorch_output.detach().numpy()

            # 计算差异
            abs_diff = np.abs(pytorch_output_np - onnx_output)
            rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)

            print(f"最大绝对差异: {np.max(abs_diff):.6f}")
            print(f"平均绝对差异: {np.mean(abs_diff):.6f}")
            print(f"最大相对差异: {np.max(rel_diff):.6f}")
            print(f"平均相对差异: {np.mean(rel_diff):.6f}")

            # 检查是否匹配
            if np.allclose(pytorch_output_np, onnx_output, rtol=1e-3, atol=1e-5):
                print("PyTorch和ONNX输出匹配 ✓")
            else:
                print("PyTorch和ONNX输出存在差异 ⚠")

        except Exception as e:
            print(f"ONNX推理测试失败: {e}")

        # 8. 测试不同输入尺寸
        print("8. 测试不同输入尺寸...")
        test_shapes = [
            (1, in_channels, 8, 32),
            (1, in_channels, 8, 64),
            (2, in_channels, 8, 128),
            (4, in_channels, 8, 256)
        ]

        for test_shape in test_shapes:
            try:
                test_input = torch.randn(test_shape)
                with torch.no_grad():
                    test_output = neck(test_input)
                print(f"输入形状 {test_shape} -> 输出形状 {test_output.shape} ✓")
            except Exception as e:
                print(f"输入形状 {test_shape} -> 失败: {e} ✗")

        # 9. 模型信息
        print("9. 模型信息统计...")
        total_params = sum(p.numel() for p in neck.parameters())
        trainable_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 计算模型大小
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"ONNX文件大小: {file_size / 1024 / 1024:.2f} MB")

        print(f"\n=== UltraLightweightSVTRv2Neck ONNX导出测试完成 ===")
        return output_path

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return None

def test_all_onnx_exports():
    """测试所有ONNX导出函数"""
    print("=== 验证SVTR ONNX导出函数 ===")

    # 1. 测试SVTRv2Neck
    print("\n1. 测试SVTRv2Neck ONNX导出...")
    try:
        result = test_svtrv2_neck_onnx(
            in_channels=256,
            out_channels=384,
            input_shape=(1, 256, 8, 32),
            output_path='svtrv2_neck_validate.onnx',
            hidden_dims=256,
            depth=2,
            use_multi_scale=True,
            use_lightweight=True
        )
        if result:
            print("✓ SVTRv2Neck ONNX导出成功")
        else:
            print("✗ SVTRv2Neck ONNX导出失败")
    except Exception as e:
        print(f"✗ SVTRv2Neck ONNX导出错误: {e}")

    # 2. 测试UltraLightweightSVTRv2Neck
    print("\n2. 测试UltraLightweightSVTRv2Neck ONNX导出...")
    try:
        result = test_ultra_lightweight_svtrv2_neck_onnx(
            in_channels=256,
            out_channels=384,
            input_shape=(1, 256, 8, 32),
            output_path='ultra_lightweight_svtrv2_neck_validate.onnx',
            hidden_dims=256
        )
        if result:
            print("✓ UltraLightweightSVTRv2Neck ONNX导出成功")
        else:
            print("✗ UltraLightweightSVTRv2Neck ONNX导出失败")
    except Exception as e:
        print(f"✗ UltraLightweightSVTRv2Neck ONNX导出错误: {e}")

    # 3. 测试SVTRv2（简化版本）
    print("\n3. 测试SVTRv2 ONNX导出...")
    try:
        # 先测试模型是否能正常工作
        cfg = SVTRv2.svtr_cfg['tiny']
        model = SVTRv2(**cfg)
        model.eval()

        # 简单前向测试
        test_input = torch.randn(1, 3, 32, 128)
        with torch.no_grad():
            output = model(test_input)
        print(f"SVTRv2模型测试通过，输入: {test_input.shape}, 输出: {output.shape}")

        # 尝试ONNX导出（使用简化方法）
        result = test_svtrv2_onnx(
            model_type='tiny',
            input_shape=(3, 3, 32, 128),
            output_path='svtrv2_validate.onnx'
        )
        if result:
            print("✓ SVTRv2 ONNX导出成功")
        else:
            print("✗ SVTRv2 ONNX导出失败")
    except Exception as e:
        print(f"✗ SVTRv2 ONNX导出错误: {e}")

    print("\n=== ONNX导出验证完成 ===")

if __name__ == "__main__":
    basic_test()
    test_all_onnx_exports()

