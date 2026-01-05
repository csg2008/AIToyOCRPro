"""
VIPTRNet is refer from: https://github.com/yyedekkun/VIPTR/blob/main/modules/VIPTRv2T_ch.py
"""
import time
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_, DropPath, ConvNormAct, Mlp, Attention, LayerScale, PatchEmbed
from torch.export import draft_export

class DWConv2d(nn.Module):
    """
    深度可分离卷积模块

    Args:
        dim: 输入和输出的通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 卷积填充
    """
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (b, h, w, c)

        Returns:
            输出张量，形状为 (b, h, w, c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x

class MHSA_Block(nn.Module):
    """
    多头自注意力模块

    Args:
        dim: 输入和输出的通道数
        num_heads: 注意力头的数量
        mlp_ratio: MLP中间层通道数的比例
        qkv_bias: 是否使用qkv投影的偏置
        qk_scale: 缩放因子，如果为None则使用默认值
        drop: 投影层的 dropout 概率
        attn_drop: 注意力层的 dropout 概率
        drop_path_rate: 随机深度的 dropout 概率
        act_layer: 激活函数类型
        norm_layer: 归一化层类型
        epsilon: 归一化层的 epsilon 参数
        prenorm: 是否在注意力之前进行归一化
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 epsilon=1e-6,
                 prenorm=False):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = norm_layer(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)

        self.mixer = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # self.drop_path = DropPath(local_rank,drop_path) if drop_path > 0. else Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        if isinstance(norm_layer, str):
            self.norm2 = norm_layer(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=True
        )
        self.prenorm = prenorm

    def forward(self, x, size=None):
        """
        前向传播

        Args:
            x: 输入张量
            size: 可选的尺寸信息

        Returns:
            输出张量
        """
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OSRA_Attention(nn.Module):  ### OSRA
    """
    自注意力与局部卷积结合的注意力模块

    Args:
        dim: 输入和输出的通道数
        num_heads: 注意力头的数量
        qk_scale: 缩放因子，如果为None则使用默认值
        attn_drop: 注意力层的 dropout 概率
        sr_ratio: 空间压缩比率
    """
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvNormAct(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_layer=nn.BatchNorm2d,
                           act_layer=nn.GELU),
                ConvNormAct(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_layer=nn.BatchNorm2d,
                           apply_act=False), )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, size, relative_pos_enc=None):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (b, n, c)
            size: 输入的空间尺寸 (h, w)
            relative_pos_enc: 相对位置编码，可选

        Returns:
            输出张量，形状为 (b, n, c)
        """
        B, N, C = x.shape
        H, W = size
        x = x.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).contiguous().reshape(B, C, -1)
        x = x.permute(0, 2, 1).contiguous()
        return x

class OSRA_Block(nn.Module):
    """
    自注意力与局部卷积结合的注意力模块块

    Args:
        dim: 输入和输出的通道数，默认64
        sr_ratio: 空间压缩比率，默认1
        num_heads: 注意力头的数量，默认1
        mlp_ratio: MLP中间层通道数的比例，默认4
        norm_cfg: 归一化层类型，默认nn.LayerNorm
        act_cfg: 激活函数类型，默认nn.GELU
        drop: 投影层的 dropout 概率，默认0
        drop_path: 随机深度的 dropout 概率，默认0
        layer_scale_init_value: 层缩放的初始值，默认1e-5
        grad_checkpoint: 是否使用梯度检查点，默认False
    """

    def __init__(self,
                 dim=64,
                 sr_ratio=1,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=nn.LayerNorm, # dict(type='GN', num_groups=1),
                 act_cfg=nn.GELU, # dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_cfg(dim)
        self.token_mixer = OSRA_Attention(dim, num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = norm_cfg(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_cfg,
            drop=drop,
            bias=True
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        """
        内部前向传播实现

        Args:
            x: 输入张量
            relative_pos_enc: 相对位置编码，可选

        Returns:
            输出张量
        """
        x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        """
        前向传播

        Args:
            x: 输入张量
            relative_pos_enc: 相对位置编码，可选

        Returns:
            输出张量
        """
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x


class RelPos2d(nn.Module):
    """
    2D相对位置编码模块

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        initial_value: 初始值
        heads_range: 头的范围
    """
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        self._cached_rel_pos = None
        self._cached_size = None

    def generate_2d_decay(self, H: int, W: int):
        """
        生成2D衰减矩阵

        Args:
            H: 高度
            W: 宽度

        Returns:
            2D衰减矩阵
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_decay(self, l: int):
        """
        生成1D衰减矩阵

        Args:
            l: 长度

        Returns:
            1D衰减矩阵
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        """
        前向传播

        Args:
            slen: 尺寸元组，包含高度和宽度
            activate_recurrent: 是否激活循环模式，默认False
            chunkwise_recurrent: 是否激活分块循环模式，默认False

        Returns:
            相对位置编码
        """
        if activate_recurrent:
            retention_rel_pos = self.decay.exp()
        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = (mask_h, mask_w)
        else:
            cache_key = (slen[0], slen[1], False, False)
            if self._cached_rel_pos is not None and self._cached_size == cache_key:
                return self._cached_rel_pos
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = mask
            self._cached_rel_pos = retention_rel_pos
            self._cached_size = cache_key
        return retention_rel_pos

class MaSAd(nn.Module):
    """
    分块多头自注意力模块

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        value_factor: 值的缩放因子，默认1
    """

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置参数
        """
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (b, h, w, c)
            rel_pos: 相对位置编码，包含高度和宽度的掩码
            chunkwise_recurrent: 是否使用分块循环模式，默认False
            incremental_state: 增量状态，默认None

        Returns:
            输出张量，形状为 (b, h, w, c)
        """
        bsz, h, w, _ = x.size()
        mask_h, mask_w = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

class MaSA(nn.Module):
    """
    多头自注意力模块

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        value_factor: 值的缩放因子，默认1
    """

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置参数
        """
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (b, h, w, c)
            rel_pos: 相对位置编码，形状为 (n, l, l)
            chunkwise_recurrent: 是否使用分块循环模式，默认False
            incremental_state: 增量状态，默认None

        Returns:
            输出张量，形状为 (b, h, w, c)
        """
        bsz, h, w, _ = x.size()
        mask = rel_pos

        assert h * w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        qr = qr.flatten(2, 3)  # (b n l d1)
        kr = kr.flatten(2, 3)  # (b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # (b n h w d2)
        vr = vr.flatten(2, 3)  # (b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2)  # (b n l l)
        qk_mat = qk_mat + mask  # (b n l l)
        qk_mat = torch.softmax(qk_mat, -1)  # (b n l l)
        output = torch.matmul(qk_mat, vr)  # (b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # (b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络模块

    Args:
        embed_dim: 嵌入维度
        ffn_dim: 前馈神经网络中间层的维度
        activation_fn: 激活函数，默认F.gelu
        dropout: 输出dropout概率，默认0.0
        activation_dropout: 激活后dropout概率，默认0.0
        layernorm_eps: 层归一化的epsilon参数，默认1e-6
        subln: 是否在FFN中间层使用层归一化，默认False
        subconv: 是否在FFN中间层使用深度可分离卷积，默认False
    """
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            layernorm_eps=1e-6,
            subln=False,
            subconv=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # self.out_dim = out_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        """
        重置参数
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RetBlock(nn.Module):
    """
    保留块模块

    Args:
        retention: 保留类型，可选 'chunk' 或 'whole'
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        ffn_dim: 前馈神经网络中间层的维度
        out_dim: 输出维度
        drop_path: 随机深度的 dropout 概率，默认0.
        layerscale: 是否使用层缩放，默认False
        layer_init_values: 层缩放的初始值，默认1e-5
    """

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, out_dim, drop_path=0., layerscale=False,
                 layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        # self.out_dim = out_dim if out_dim is not None else embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = MaSAd(embed_dim, num_heads)
        else:
            self.retention = MaSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
    ):
        """
        前向传播

        Args:
            x: 输入张量
            incremental_state: 增量状态，默认None
            chunkwise_recurrent: 是否使用分块循环模式，默认False
            retention_rel_pos: 保留相对位置编码，默认None

        Returns:
            输出张量
        """
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent,
                                              incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(
                self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim: 输入通道数
        out_dim: 输出通道数
        norm_layer: 归一化层，默认nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, (2, 1), 1)
        # self.norm = nn.BatchNorm2d(out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (B, H, W, C)

        Returns:
            输出张量，形状为 (B, oh, ow, oc)
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b oh ow oc)
        x = self.norm(x)

        return x

class LePEAttention(nn.Module):
    """
    局部增强位置编码注意力模块

    Args:
        dim: 输入通道数
        resolution: 分辨率
        idx: 索引
        split_size: 分割大小，默认7
        dim_out: 输出通道数，默认与输入通道数相同
        num_heads: 注意力头的数量，默认8
        attn_drop: 注意力dropout概率，默认0.
        proj_drop: 投影dropout概率，默认0.
        qk_scale: qk缩放因子，默认None
    """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, size):
        """
        将图像转换为CSWin窗口

        Args:
            x: 输入张量
            size: 尺寸

        Returns:
            转换后的张量
        """
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func, size):
        """
        获取局部增强位置编码

        Args:
            x: 输入张量
            func: 函数
            size: 尺寸

        Returns:
            处理后的张量和局部增强位置编码
        """
        B, N, C = x.shape
        H, W = size
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, size):
        """
        前向传播

        Args:
            qkv: 包含q、k、v的元组
            size: 尺寸

        Returns:
            输出张量
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        H, W = size
        B, L, C = q.shape

        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size

        q = self.im2cswin(q, size)
        k = self.im2cswin(k, size)
        v, lepe = self.get_lepe(v, self.get_v, size)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)
        return x


class CSWinBlock(nn.Module):
    """
    CSWin块模块

    Args:
        dim: 输入通道数
        reso: 分辨率
        num_heads: 注意力头的数量
        split_size: 分割大小，默认7
        mlp_ratio: MLP中间层通道数的比例，默认4.
        qkv_bias: 是否使用qkv投影的偏置，默认False
        qk_scale: 缩放因子，默认None
        drop: 投影层的 dropout 概率，默认0.
        attn_drop: 注意力层的 dropout 概率，默认0.
        drop_path: 随机深度的 dropout 概率，默认0.
        act_layer: 激活函数类型，默认nn.GELU
        norm_layer: 归一化层类型，默认nn.LayerNorm
        last_stage: 是否为最后一个阶段，默认False
    """
    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        last_stage = False
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=True
        )

        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        """
        前向传播

        Args:
            x: 输入张量
            size: 尺寸

        Returns:
            输出张量
        """
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], size)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], size)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv, size)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    # Avoid calling int() on shape elements (these can be SymInt during export/tracing)
    # Use integer (floor) division with shape components to stay export-friendly.
    # B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    B = img_splits_hw.shape[0] // ((H // H_sp) * (W // W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class BasicLayer(nn.Module):
    """
    Vision Transformer的基本层模块

    Args:
        embed_dim: 嵌入维度
        out_dim: 输出维度
        depth: 层深度
        num_heads: 注意力头的数量
        init_value: 初始化值
        heads_range: 注意力头的范围
        mlp_ratio: MLP中间层维度比例，默认4.
        split_size: 分割大小，默认1
        sr_ratio: 空间缩减比例，默认1
        qkv_bias: 是否使用qkv偏置，默认True
        qk_scale: qk缩放因子，默认None
        drop_rate: dropout概率，默认0.
        attn_drop: 注意力dropout概率，默认0.0
        drop_path: 随机深度概率，可以是列表或单个值，默认0.
        norm_layer: 归一化层类型，默认nn.LayerNorm
        chunkwise_recurrent: 是否使用分块递归，默认False
        downsample: 下采样模块，默认None
        use_checkpoint: 是否使用检查点，默认False
        mixer_type: 混合器类型，可选'Global'、'Local1'、'Local2'、'Global1'、'Global2'，默认'Global'
        layerscale: 是否使用层缩放，默认False
        layer_init_values: 层初始化值，默认1e-5
    """
    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float, mlp_ratio=4., split_size=1, sr_ratio=1,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop=0.0,
                 drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False, mixer_type='Global',
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        self.mixer_type = mixer_type
        if mixer_type == 'Local1':
            self.blocks = nn.ModuleList([
                CSWinBlock(
                    dim=embed_dim, num_heads=num_heads, reso=25, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                    drop=drop_rate, attn_drop=attn_drop,
                    drop_path=drop_path[i], norm_layer=norm_layer)
                for i in range(depth)])

        elif mixer_type == 'Local2':
            if chunkwise_recurrent:
                flag = 'chunk'
            else:
                flag = 'whole'
            self.Relpos = RelPos2d(embed_dim, num_heads, init_value, heads_range)

            # build blocks
            self.blocks = nn.ModuleList([
                RetBlock(flag, embed_dim, num_heads, int(mlp_ratio * embed_dim),
                         drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
                for i in range(depth)])

        elif mixer_type == 'Global1':
            self.blocks = nn.ModuleList([
                OSRA_Block(
                    dim=embed_dim, sr_ratio=sr_ratio, num_heads=num_heads // 2, mlp_ratio=mlp_ratio,
                    norm_cfg=norm_layer,
                    drop=drop_rate, drop_path=drop_path[i], act_cfg=nn.GELU,
                ) for i in range(depth)]
            )
        elif mixer_type == 'Global2':
            self.blocks = nn.ModuleList([
                MHSA_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop, drop_path_rate=drop_path[i], act_layer=nn.GELU,
                    norm_layer=norm_layer,
                ) for i in range(depth)]
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, size):
        """
        前向传播

        Args:
            x: 输入张量，形状为(b, h, w, d)
            size: 尺寸，通常为(h, w)

        Returns:
            输出张量
        """
        b, h, w, d = x.size()
        if self.mixer_type == 'Local1':
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == 'Local2':
            rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
            for blk in self.blocks:
                if self.use_checkpoint:
                    tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                                      retention_rel_pos=rel_pos)
                    x = checkpoint.checkpoint(tmp_blk, x)
                else:
                    x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent,
                            retention_rel_pos=rel_pos)

        elif self.mixer_type == 'Global1':
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                # x = x.permute(0, 3, 1, 2).contiguous()
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        elif self.mixer_type == 'Global2':
            for blk in self.blocks:
                x = x.flatten(1).reshape(b, -1, d)
                x = blk(x, size)
                x = x.reshape(b, h, w, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class BasicLayerV2(nn.Module):
    """
    Vision Transformer的基本层模块V2

    Args:
        embed_dim: 嵌入维度
        out_dim: 输出维度
        depth: 层深度
        num_heads: 注意力头的数量
        init_value: 初始化值
        heads_range: 注意力头的范围
        mlp_ratio: MLP中间层维度比例，默认4.
        split_size: 分割大小，默认1
        sr_ratio: 空间缩减比例，默认1
        qkv_bias: 是否使用qkv偏置，默认True
        qk_scale: qk缩放因子，默认None
        drop_rate: dropout概率，默认0.
        attn_drop: 注意力dropout概率，默认0.0
        drop_path: 随机深度概率，可以是列表或单个值，默认0.
        norm_layer: 归一化层类型，默认nn.LayerNorm
        chunkwise_recurrent: 是否使用分块递归，默认False
        downsample: 下采样模块，默认None
        use_checkpoint: 是否使用检查点，默认False
        mixer_type: 混合器类型，可选'Local'或'Global'，默认'Global'
        layerscale: 是否使用层缩放，默认False
        layer_init_values: 层初始化值，默认1e-5
    """
    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float, mlp_ratio=4., split_size=1, sr_ratio=1,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop=0.0,
                 drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging = None, use_checkpoint=False, mixer_type='Global',
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        self.mixer_type = mixer_type
        if mixer_type == 'Local':
            self.blocks = nn.ModuleList([
                CSWinBlock(
                    dim=embed_dim, num_heads=num_heads, reso=25, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                    drop=drop_rate, attn_drop=attn_drop,
                    drop_path=drop_path[i], norm_layer=norm_layer)
                for i in range(depth)])

        elif mixer_type == 'Global':
            self.blocks = nn.ModuleList([
                OSRA_Block(
                    dim=embed_dim, sr_ratio=sr_ratio, num_heads=num_heads//2, mlp_ratio=mlp_ratio, norm_cfg=norm_layer,
                    drop=drop_rate, drop_path=drop_path[i], act_cfg=nn.GELU,
                ) for i in range(depth)]
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, size):
        """
        前向传播

        Args:
            x: 输入张量，形状为(b, h, w, d)
            size: 尺寸，通常为(h, w)

        Returns:
            输出张量
        """
        b, h, w, d = x.size()
        if self.mixer_type == 'Local':
            x = x.flatten(1).reshape(b, -1, d)
            for blk in self.blocks:
                x = blk(x, size)
            x = x.reshape(b, h, w, -1)
        elif self.mixer_type == 'Global':
            x = x.flatten(1).reshape(b, -1, d)
            for blk in self.blocks:
                x = blk(x, size)
            x = x.reshape(b, h, w, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VIPTRNetV1(nn.Module):
    """
    Vision Transformer for Scene Text Recognition Version 1

    Args:
        in_chans: 输入通道数，默认3
        out_dim: 输出维度，默认192
        embed_dims: 各层嵌入维度列表，默认[96, 192, 384, 768]
        depths: 各层深度列表，默认[2, 2, 6, 2]
        num_heads: 各层注意力头数量列表，默认[3, 6, 12, 24]
        init_values: 初始化值列表，默认[1, 1, 1, 1]
        heads_ranges: 注意力头范围列表，默认[3, 3, 3, 3]
        mlp_ratios: MLP中间层维度比例列表，默认[3, 3, 3, 3]
        split_sizes: 分割大小列表，默认[1, 2, 2, 4]
        sr_ratios: 空间缩减比例列表，默认[8, 4, 2, 1]
        drop_path_rate: 随机深度概率，默认0.1
        norm_layer: 归一化层类型，默认nn.LayerNorm
        patch_norm: 是否使用补丁归一化，默认True
        use_checkpoints: 是否使用检查点列表，默认[False, False, False, False]
        mixer_types: 混合器类型列表，默认['Local1', 'Local1', 'Global2', 'Global2']
        chunkwise_recurrents: 是否使用分块递归列表，默认[True, True, False, False]
        layerscales: 是否使用层缩放列表，默认[False, False, False, False]
        layer_init_values: 层初始化值，默认1e-6
    """
    def __init__(self, in_chans=3, out_dim=192,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], split_sizes=[1, 2, 2, 4],
                 sr_ratios=[8, 4, 2, 1], drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False],
                 mixer_types=['Local1', 'Local1', 'Global2', 'Global2'],
                 chunkwise_recurrents=[True, True, False, False],
                 layerscales=[False, False, False, False], layer_init_values=1e-6):
        super().__init__()

        self.out_dim = out_dim
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=None,  # 动态图像大小
            patch_size=4,  # 3x3 stride=2 两次，相当于 patch_size=4
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=None,  # 原实现使用 BatchNorm2d 在 ConvNormAct 内部
            flatten=False,  # 不展平，保持 (B, C, H, W) 格式
            output_fmt='NHWC'  # 输出格式为 (B, H, W, C)
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                split_size=split_sizes[i_layer],
                sr_ratio=sr_ratios[i_layer],
                # ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop=0.0,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                # norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer in [0, 2]) else None,
                use_checkpoint=use_checkpoints[i_layer],
                mixer_type=mixer_types[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        #### CHN settings ####
        self.pooling = nn.AdaptiveAvgPool2d([1, 40])
        self.last_conv = nn.Conv2d(
            in_channels=embed_dims[self.num_layers - 1],
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.1)
        ########
        self.norm = nn.LayerNorm(embed_dims[-1], eps=layer_init_values)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像张量，形状为(b, c, h, w)

        Returns:
            输出特征张量
        """
        x = self.patch_embed(x)
        _, H, W, _ = x.shape
        for layer in self.layers:
            x = layer(x, (H, W))
            H = x.shape[1]
            # print(x.shape)  # nhwc
        x = self.norm(x)

        x = x.permute(0, 3, 1, 2).contiguous()  # nchw
        x = self.pooling(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)  # bchw
        x = x.permute(0, 3, 1, 2).contiguous()  # bwch
        x = x.squeeze(3)

        return x

class VIPTRNetV2(nn.Module):
    """
    Vision Transformer for Scene Text Recognition Version 2

    Args:
        in_chans: 输入通道数，默认3
        out_dim: 输出维度，默认192
        embed_dims: 各层嵌入维度列表，默认[96, 192, 384, 768]
        depths: 各层深度列表，默认[2, 2, 6, 2]
        num_heads: 各层注意力头数量列表，默认[3, 6, 12, 24]
        init_values: 初始化值列表，默认[1, 1, 1, 1]
        heads_ranges: 注意力头范围列表，默认[3, 3, 3, 3]
        mlp_ratios: MLP中间层维度比例列表，默认[3, 3, 3, 3]
        split_sizes: 分割大小列表，默认[1, 2, 2, 4]
        sr_ratios: 空间缩减比例列表，默认[8, 4, 2, 1]
        drop_path_rate: 随机深度概率，默认0.1
        norm_layer: 归一化层类型，默认nn.LayerNorm
        patch_norm: 是否使用补丁归一化，默认True
        use_checkpoints: 是否使用检查点列表，默认[False, False, False, False]
        mixer_types: 混合器类型列表，默认['Local', 'Local', 'Global', 'Global']
        chunkwise_recurrents: 是否使用分块递归列表，默认[True, True, False, False]
        layerscales: 是否使用层缩放列表，默认[False, False, False, False]
        layer_init_values: 层初始化值，默认1e-6
    """

    def __init__(self, in_chans=3, out_dim=192,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], split_sizes=[1, 2, 2, 4],
                 sr_ratios=[8, 4, 2, 1], drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=[False, False, False, False], mixer_types=['Local']*2+['Global']*2,
                 chunkwise_recurrents=[True, True, False, False],
                 layerscales=[False, False, False, False], layer_init_values=1e-6):
        super().__init__()

        self.out_dim = out_dim
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=None,  # 动态图像大小
            patch_size=4,  # 3x3 stride=2 两次，相当于 patch_size=4
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=None,  # 原实现使用 BatchNorm2d 在 ConvNormAct 内部
            flatten=False,  # 不展平，保持 (B, C, H, W) 格式
            output_fmt='NHWC'  # 输出格式为 (B, H, W, C)
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerV2(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                split_size=split_sizes[i_layer],
                sr_ratio=sr_ratios[i_layer],
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop=0.0,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer in [0, 2]) else None,
                use_checkpoint=use_checkpoints[i_layer],
                mixer_type=mixer_types[i_layer],
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.pooling = nn.AdaptiveAvgPool2d((embed_dims[self.num_layers - 1], 1))
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dims[self.num_layers - 1], out_dim, bias=False),
            nn.Hardswish(),
            nn.Dropout(p=0.1)
        )
        self.norm = nn.LayerNorm(embed_dims[-1], eps=layer_init_values)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像张量，形状为(b, c, h, w)

        Returns:
            输出特征张量
        """
        x = self.patch_embed(x)
        _, H, W, _ = x.shape
        for layer in self.layers:
            x = layer(x, (H, W))
            H = x.shape[1]
            # print(x.shape)  # nhwc
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # nwch
        x = self.pooling(x)
        x = x.squeeze(3)  # .reshape(b, W, -1)
        x = self.mlp_head(x)

        return x


def VIPTRv1(in_chans: int = 3, output_channel: int = 256):
    """
    创建VIPTRNetV1模型的函数

    Args:
        in_chans: 输入通道数，默认3
        output_channel: 输出通道数，默认256

    Returns:
        VIPTRNetV1实例
    """
    return VIPTRNetV1(
        out_dim=output_channel,
        in_chans=in_chans,
        embed_dims=[64, 128, 128, 256],
        depths=[3, 3, 3, 3],
        num_heads=[2, 4, 4, 8],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 4, 4],  # 3 3 4 4
        split_sizes=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False, False], # [True, False, False, False]
        layerscales=[False, False, False, False]
    )

def VIPTRv1L(in_chans: int = 3, output_channel: int = 256):
    """
    创建大尺寸VIPTRNetV1模型的函数

    Args:
        in_chans: 输入通道数，默认3
        output_channel: 输出通道数，默认256

    Returns:
        VIPTRNetV1实例
    """
    return VIPTRNetV1(
        out_dim=output_channel,
        in_chans=in_chans,
        embed_dims=[192, 256, 256, 512],
        depths=[3, 7, 2, 9],
        num_heads=[6, 8, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 4, 4],  # 4 4 4 4
        split_sizes=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False], # [True, False, False, False]
        layerscales=[False, False, False, False]
    )

def VIPTRv2(in_chans: int = 3, output_channel: int = 256):
    """
    创建VIPTRNetV2模型的函数

    Args:
        in_chans: 输入通道数，默认3
        output_channel: 输出通道数，默认256

    Returns:
        VIPTRNetV2实例
    """
    return VIPTRNetV2(
        out_dim=output_channel,
        in_chans=in_chans,
        embed_dims=[64, 128, 256],
        depths=[3, 3, 3],
        num_heads=[2, 4, 8],
        init_values=[2, 2, 2],
        heads_ranges=[4, 4, 6],
        mlp_ratios=[3, 4, 4], # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2], # [8, 4, 2]
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False],
        layerscales=[False, False, False]
    )

def VIPTRv2B(in_chans: int = 3, output_channel: int = 256):
    """
    创建大尺寸VIPTRNetV2模型的函数

    Args:
        in_chans: 输入通道数，默认3
        output_channel: 输出通道数，默认256

    Returns:
        VIPTRNetV2实例
    """
    return VIPTRNetV2(
        out_dim=output_channel,
        in_chans=in_chans,
        embed_dims=[128, 256, 384],
        depths=[3, 6, 9],
        num_heads=[4, 8, 12],
        init_values=[2, 2, 2],
        heads_ranges=[6, 6, 6],
        mlp_ratios=[4, 4, 4], # 4 4 4
        split_sizes=[1, 2, 4],
        sr_ratios=[4, 2, 2], # [8, 4, 2]
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False],
        layerscales=[False, False, False]
    )

def VIPTRv2T_CH(in_chans: int = 3, output_channel: int = 256):
    """
    创建VIPTRNetV2模型的CH版本函数

    Args:
        in_chans: 输入通道数，默认3
        output_channel: 输出通道数，默认256

    Returns:
        VIPTRNetV2实例
    """
    return VIPTRNetV2(
        out_dim=output_channel,
        in_chans=in_chans,
        embed_dims=[64, 128, 128, 256],
        depths=[3, 3, 3, 3],
        num_heads=[2, 4, 4, 8],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 4, 4], # 4 4 4 4
        split_sizes=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, False, False, False],
        layerscales=[False, False, False, False]
    )

def test_viptrv2(model_type='VIPTRv2T_CH', input_shape=(1, 3, 32, 512), output_path='viptrv2.onnx'):
    """
    VIPTRNetV2 ONNX导出测试函数

    Args:
        model_type: 模型类型，可选 'VIPTRv2', 'VIPTRv2B', 'VIPTRv2T_CH'
        input_shape: 输入张量形状 (batch, channels, height, width)
        output_path: 输出的ONNX文件路径

    Returns:
        str: ONNX文件路径
    """
    import os
    import numpy as np
    import onnx
    import onnxruntime as ort
    from torch.export import Dim

    print(f"=== VIPTRNetV2 ONNX导出测试 ===")
    print(f"模型类型: {model_type}")
    print(f"输入形状: {input_shape}")
    print(f"输出路径: {output_path}")

    # 1. 创建模型
    print("1. 创建模型...")
    if model_type == 'VIPTRv2':
        model = VIPTRv2()
    elif model_type == 'VIPTRv2B':
        model = VIPTRv2B()
    elif model_type == 'VIPTRv2T_CH':
        model = VIPTRv2T_CH()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

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
            "x":   {0: batch_size, 3: width}
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

            # print(draft_export(
            #     model,
            #     args=(dummy_input,),
            #     dynamic_shapes=dynamic_shapes
            # ))
            # os._exit(0)

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

        print(f"\n=== VIPTRNetV2 ONNX导出测试完成 ===")
        return output_path

    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return None

if __name__ == '__main__':
    try:
        result = test_viptrv2(
            model_type='VIPTRv2T_CH',
            input_shape=(2, 3, 32, 512),
            output_path='viptrv2_test.onnx'
        )
        if result:
            print(f"完整版本测试成功，输出文件: {result}")
        else:
            print("完整版本测试失败")
    except Exception as e:
        print(f"完整版本测试失败: {e}")