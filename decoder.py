import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim
from typing import Tuple, Optional, List, Dict
from timm.layers.pos_embed_sincos import (
    rope_rotate_half,
    build_rotary_pos_embed,
    apply_rot_embed,
    apply_rot_embed_cat,
    RotaryEmbeddingCat
)

# -------------------- 公共 RoPE 工具 --------------------
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_pos_ids: torch.Tensor,
    k_pos_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用显式位置ID对查询(Q)和键(K)张量应用旋转位置编码(RoPE)。

    该函数实现了旋转位置编码的核心逻辑，通过预计算的cos/sin值和位置ID，
    对Q和K张量进行旋转变换，使模型能够捕捉序列中的位置信息。

    Args:
        q (torch.Tensor): 查询张量，形状为[B, nhead, q_len, head_dim]
        k (torch.Tensor): 键张量，形状为[B, nhead, kv_len, head_dim]
        cos (torch.Tensor): 预计算的余弦值，形状为[max_rope_len, head_dim]
        sin (torch.Tensor): 预计算的正弦值，形状为[max_rope_len, head_dim]
        q_pos_ids (torch.Tensor): 查询token的绝对位置ID，形状为[q_len]
        k_pos_ids (torch.Tensor): 键token的绝对位置ID，形状为[kv_len]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - q_out: 应用RoPE后的查询张量，形状与q相同
            - k_out: 应用RoPE后的键张量，形状与k相同

    Note:
        - 该实现支持变长序列和KV缓存机制
        - 使用显式位置ID可以灵活处理各种位置编码场景
        - 旋转操作通过rotate_half函数实现，符合RoPE的数学原理
    """
    # Expand cos/sin to [1, 1, L, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Gather positions
    q_cos = cos[:, :, q_pos_ids]  # [1, 1, q_len, head_dim]
    q_sin = sin[:, :, q_pos_ids]
    k_cos = cos[:, :, k_pos_ids]  # [1, 1, kv_len, head_dim]
    k_sin = sin[:, :, k_pos_ids]

    q_out = q * q_cos + rope_rotate_half(q) * q_sin
    k_out = k * k_cos + rope_rotate_half(k) * k_sin
    return q_out, k_out

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用到输入张量 x (Query 或 Key) 上。
    Args:
        x (torch.Tensor): 输入张量，形状 (B, H, T, D_head)
        cos (torch.Tensor): 预计算的 cos 值，形状 (T, D_head)
        sin (torch.Tensor): 预计算的 sin 值，形状 (T, D_head)
    Returns:
        torch.Tensor: 旋转后的张量。
    """
    # 广播 cos/sin 到 (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 使用 half=True 保持与原实现兼容
    return apply_rot_embed(x, sin, cos, half=True)


def precompute_rope_sin_cos(max_len: int, head_dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """预先计算 sin/cos 表，[max_len, head_dim]"""
    # 注意：timm 的 build_rotary_pos_embed 默认是为图像设计的，in_pixels=True
    # 对于语言模型，需要设置 in_pixels=False

    # timm 的 build_rotary_pos_embed 返回的是每个维度的 sin 和 cos 值，形状为 [max_len, head_dim//2]
    # 需要将它们复制一份并拼接，以匹配原函数的输出形状 [max_len, head_dim]
    cos_half, sin_half = build_rotary_pos_embed(
        feat_shape=[max_len],
        dim=head_dim,
        max_res=max_len,
        temperature=base,
        in_pixels=False
    )

    # 复制并拼接，使形状变为 [max_len, head_dim]
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)

    # 如果 head_dim 是奇数，需要截断最后一个元素
    if head_dim % 2 == 1:
        cos = cos[:, :head_dim]
        sin = sin[:, :head_dim]

    return cos, sin

class PositionalEncoding(nn.Module):
    """1-D 正弦位置编码，支持 batch_first=True"""
    def __init__(self, d_model, dropout: float = 0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                    # [T, C]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)          # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]   # 自动广播到 [B, T, C]
        return self.dropout(x)

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 的实现。
    它不直接作用于输入，而是在自注意力计算中用于旋转 Query 和 Key 向量。
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 直接使用 precompute_rope_sin_cos 函数预计算 cos 和 sin
        # 这样可以确保返回的形状与原实现一致
        cos_cached, sin_cached = precompute_rope_sin_cos(max_seq_len, dim, base)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """返回预计算好的 cos 和 sin，截取到当前需要的序列长度"""
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


class CandidateNet(nn.Module):
    """
    候选选择网络

    输入: [B, d]  输出: [B, K] 字符索引
    训练时用 Gumbel-Softmax / 直接 Top-K，
    推理时用 Top-K 硬索引。
    """
    def __init__(self, d: int, num_classes: int, K: int = 100, temp: float = 1.0):
        """
        初始化候选选择网络

        Args:
            d (int): 输入特征维度，即输入向量的长度
            num_classes (int): 总的类别数量，用于计算均匀先验分布
            K (int, optional): 候选数量，即输出的候选索引个数。默认 100
            temp (float, optional): Gumbel-Softmax 的温度参数，控制采样的随机性。
                                   温度越高分布越均匀，温度越低越尖锐。默认 1.0

        Note:
            - 网络结构：d -> 256 -> ReLU -> K，输出 K 个候选的 logits
            - idx_base 是均匀先验，确保候选覆盖整个类别空间
            - 训练时使用 Gumbel-Softmax 进行可导采样
            - 推理时使用确定性 Top-K 选择
        """
        super().__init__()
        self.K = K
        self.temp = temp

        # 网络结构：输入 -> 256 -> ReLU -> K 个候选 logits
        self.fc = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, K)  # 输出 K 个 logits
        )

        # 均匀先验：确保候选均匀分布在类别空间中
        # 例如：num_classes=1000, K=100 -> idx_base=[0, 10, 20, ..., 990]
        self.register_buffer('idx_base', torch.arange(K)*num_classes//K)  # 均匀先验

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        前向传播：根据输入特征选择候选索引

        Args:
            x (torch.Tensor): 输入特征张量，形状为 [B, d]
                             B 是 batch size，d 是特征维度
            training (bool, optional): 是否为训练模式。训练时使用 Gumbel-Softmax 采样，
                                      推理时使用确定性 Top-K 选择。默认 True

        Returns:
            torch.Tensor: 选择的候选索引，形状为 [B, K]，数据类型为 long

        计算过程：
        训练阶段：
        1. 通过 MLP 得到 K 个候选的得分
        2. 添加 Gumbel 噪声增加随机性
        3. 使用 Softmax 计算概率分布
        4. 多项式采样选择候选
        5. 加上均匀先验偏移量

        推理阶段：
        1. 通过 MLP 得到 K 个候选的得分
        2. 直接选择得分最高的候选
        3. 加上均匀先验偏移量
        """
        score = self.fc(x)                      # [B, K]  K 个候选的得分

        if training:
            # Gumbel-Softmax 采样（可导）：在训练时增加随机性，支持梯度回传
            # Gumbel 噪声：-log(-log(uniform + eps) + eps)，eps 防止数值不稳定
            gumbel = -torch.log(-torch.log(torch.rand_like(score)+1e-10)+1e-10)
            # 温度调节的 Softmax：温度控制分布的尖锐程度
            prob = F.softmax((score + gumbel) / self.temp, dim=1)
            # 多项式采样：根据概率分布采样 K 个候选（有放回）
            idx = torch.multinomial(prob, num_samples=self.K, replacement=True)  # [B, K]
            # 加上均匀先验偏移量
            idx = idx + self.idx_base.unsqueeze(0)  # [B, K] + [1, K] -> [B, K]
        else:
            # 推理阶段：确定性选择得分最高的候选
            _, idx = score.topk(self.K, dim=1)  # [B, K]
            # 加上均匀先验偏移量
            idx = idx + self.idx_base.unsqueeze(0)  # [B, K] + [1, K] -> [B, K]

        return idx                                # [B, K]  long 类型的索引

class BlockSVDLinear(nn.Module):
    """
    基于块奇异值分解的线性层

    把 [d, c] 权重按列分成 k 块，每块做秩-r SVD
    forward:  x -> U -> S -> V^T -> out
    参数量 ≈ k * (d*r + r + c//k*r)  vs. 原 d*c
    """
    def __init__(self, in_features: int, out_features: int, k: int = 8, r: int = 64):
        """
        初始化 BlockSVDLinear 层

        Args:
            in_features (int): 输入特征维度，对应权重矩阵的行数
            out_features (int): 输出特征维度，对应权重矩阵的列数
            k (int, optional): 块的数量，必须能整除 out_features。默认 8
            r (int, optional): SVD 的秩，控制近似精度和参数量。默认 64

        Note:
            - out_features 必须能被 k 整除，确保均匀分块
            - r 越小参数量越少，但可能降低模型表达能力
            - 每个块的参数量：in_features*r + r + (out_features/k)*r
        """
        super().__init__()
        assert out_features % k == 0, f"out_features ({out_features}) must be divisible by k ({k})"
        self.k, self.r = k, r
        self.c_per = out_features // k  # 每个块处理的列数
        self.layers = nn.ModuleList()

        # 创建 k 个 SVD 块，每个块包含 U 和 V^T 两个线性层
        for i in range(k):
            u = nn.Linear(in_features, r, bias=False)   # U 矩阵：in_features -> r
            v = nn.Linear(r, self.c_per, bias=False)    # V^T 矩阵：r -> c_per
            self.layers.append(nn.Sequential(u, v))

        # 全局偏置项，形状为 [out_features]
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, in_features]
                             B 是 batch size，in_features 是输入特征维度

        Returns:
            torch.Tensor: 输出张量，形状为 [B, out_features]

        计算过程：
        1. 将输入 x 通过 k 个 SVD 块并行处理
        2. 每个块输出形状为 [B, c_per] 的特征
        3. 在列维度上拼接所有块的输出，得到 [B, out_features]
        4. 加上全局偏置项
        """
        # 通过所有 SVD 块处理输入，得到 k 个 [B, c_per] 的输出
        outs = [m(x) for m in self.layers]            # list of [B, c_per]

        # 在列维度上拼接所有输出，并加上偏置
        return torch.cat(outs, dim=1) + self.bias

class DynamicBlockSVDLinear(nn.Module):
    """
    动态块奇异值分解线性层 (Dynamic Block-SVD Linear)

    核心思想：结合静态块 SVD 和动态候选选择，通过 CandidateNet 动态识别
    重要的输出位置，只计算被选中的块，进一步减少计算量。

    与静态 BlockSVDLinear 的区别：
    - 静态：计算所有块的输出，参数量减少但计算量不变
    - 动态：只计算重要块的输出，参数量和计算量都大幅减少

    适用场景：输出维度很大但每个样本只需要稀疏输出的情况
    """
    def __init__(self, d: int, num_classes: int, k: int = 8, r: int = 64, dynamic_k: int = 100):
        """
        初始化动态块 SVD 线性层

        Args:
            d (int): 输入特征维度
            num_classes (int): 输出类别总数，即最终的输出维度
            k (int, optional): 静态块的数量，必须能整除 num_classes。默认 8
            r (int, optional): SVD 的秩，控制每个块的参数量和表达能力。默认 64
            dynamic_k (int, optional): 动态选择的候选数量，控制稀疏度。默认 100

        Note:
            - 结合了静态块 SVD 和动态候选选择的优势
            - 先通过 CandidateNet 动态选择重要的输出位置
            - 只计算被选中的块，显著减少计算量
            - 参数量：k * (d*r + r + (num_classes/k)*r) + CandidateNet 参数
            - 计算复杂度：从 O(B*T*num_classes) 降低到 O(B*T*dynamic_k)
            - 缓存优化：在推理时缓存候选选择结果，避免重复计算
        """
        super().__init__()
        self.k = k
        self.dynamic_k = dynamic_k
        self.blocks = nn.ModuleList()
        self.c_per = num_classes // k  # 每个块处理的列数

        # 创建 k 个 SVD 块，每个块包含 U 和 V^T
        for i in range(k):
            u = nn.Linear(d, r, bias=False)           # U 矩阵：d -> r
            v = nn.Linear(r, self.c_per, bias=False)  # V^T 矩阵：r -> c_per
            self.blocks.append(nn.Sequential(u, v))

        # 全局偏置项，形状为 [num_classes]
        self.bias = nn.Parameter(torch.zeros(num_classes))

        # 候选网络：动态选择重要的输出位置
        self.cand_net = CandidateNet(d, num_classes, K=dynamic_k)

        # 缓存优化：缓存候选选择结果
        self.cache_enabled = False  # 默认关闭缓存
        self._cached_idx = None
        self._cached_block_id = None
        self._cached_offset = None
        self._cached_uniq_idx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        动态前向传播：只计算被选中的重要块

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, T, d]
                             B 是 batch size，T 是时间步数，d 是特征维度

        Returns:
            torch.Tensor: 输出张量，形状为 [B, T, num_classes]

        计算过程：
        1. 对每个时间步，用 CandidateNet 选择重要的候选索引
        2. 合并所有时间步的候选，去除重复
        3. 将候选索引映射到对应的块和块内偏移
        4. 只计算被选中的块，避免计算整个输出
        5. 将块输出映射回全局输出空间
        6. 加上全局偏置

        性能优势：
        - 计算复杂度从 O(B*T*num_classes) 降低到 O(B*T*dynamic_k)
        - 特别适合输出维度很大但每个样本只需要少量输出的场景
        - 动态稀疏性：根据输入内容自适应选择重要的输出位置

        示例：
        - 输入：x 形状 [32, 100, 512]，batch_size=32，序列长度=100，特征维度=512
        - 输出：形状 [32, 100, 10000]，输出维度=10000
        - 传统方法：需要计算 32*100*10000 = 32M 次操作
        - 动态 SVD：只计算 32*100*100 = 320K 次操作，减少 99%
        """
        B, T, d = x.size()

        # 第一步：批量对所有时间步选择候选索引（优化：一次调用处理所有时间步）
        # 缓存优化：在推理时使用缓存，避免重复计算
        if self.cache_enabled and not self.training and self._cached_idx is not None:
            idx = self._cached_idx
            uniq_idx = self._cached_uniq_idx
            block_id = self._cached_block_id
            offset = self._cached_offset
        else:
            # 重塑输入为 [B*T, d]，一次性处理所有时间步
            x_flat = x.view(B * T, d)
            idx_flat = self.cand_net(x_flat, self.training)  # [B*T, dynamic_k]

            # 重塑回 [B, T, dynamic_k]
            idx = idx_flat.view(B, T, self.dynamic_k)

            # 合并所有时间步的候选并去重，得到唯一的候选索引
            uniq_idx = torch.unique(idx)  # 1-D tensor

            # 裁剪 uniq_idx 到有效范围 [0, num_classes-1]
            # 防止 CandidateNet 生成的索引超出范围
            num_classes_val = self.bias.numel()
            uniq_idx = torch.clamp(uniq_idx, min=0, max=num_classes_val - 1)

            # 第二步：将候选索引映射到块 ID 和块内偏移
            # block_id: 候选属于哪个块，offset: 在块内的具体位置
            block_id = uniq_idx // self.c_per
            offset   = uniq_idx % self.c_per

            # 缓存候选选择结果（仅在推理时）
            if self.cache_enabled and not self.training:
                self._cached_idx = idx.detach().clone()
                self._cached_uniq_idx = uniq_idx.detach().clone()
                self._cached_block_id = block_id.detach().clone()
                self._cached_offset = offset.detach().clone()

        # 第三步：初始化输出张量，只计算被选中的位置
        out = torch.zeros(B, T, self.bias.numel(), device=x.device, dtype=x.dtype)

        # 第四步：批量计算被选中的块（使用静态循环和掩码，兼容 dynamo/ONNX）
        # 计算所有块的输出，然后使用索引掩码选择需要的列
        for block_idx in range(self.k):
            sub_u = self.blocks[block_idx][0](x)
            sub_v = self.blocks[block_idx][1](sub_u)

            mask = block_id == block_idx
            count = mask.sum().item()
            if count == 0:
                continue

            cols = offset[mask]       # 当前块内需要计算的列
            global_cols = uniq_idx[mask]  # 全局列索引

            # 使用 gather 批量提取需要的列（优化：避免循环）
            # sub_v[:, :, cols] -> 使用 gather 进行批量索引
            gathered = torch.gather(sub_v, 2, cols.unsqueeze(0).unsqueeze(0).expand(B, T, -1))

            # 将结果映射回全局输出空间
            out[:, :, global_cols] = gathered

        # 第五步：加上全局偏置
        out = out + self.bias
        return out

    def enable_cache(self, enabled: bool = True):
        """
        启用或禁用候选选择缓存

        Args:
            enabled (bool): True 启用缓存，False 禁用缓存
        """
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def clear_cache(self):
        """清除缓存的候选选择结果"""
        self._cached_idx = None
        self._cached_block_id = None
        self._cached_offset = None
        self._cached_uniq_idx = None

class HybridLinear(nn.Module):
    '''混合使用普通 Linear 和 Block-SVD Linear 的全连接层'''
    def __init__(self,
                 use_svd: bool,
                 in_features: int,
                 num_classes: int,
                 k=8,
                 r=64,
                 dynamic_k=0):                # 0=关动态，>0=开
        super().__init__()

        if dynamic_k > 0 and use_svd:
            self.fc = DynamicBlockSVDLinear(in_features, num_classes, k=k, r=r, dynamic_k=dynamic_k)
        elif use_svd:
            self.fc = BlockSVDLinear(in_features, num_classes, k=k, r=r)
        else:
            self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：根据配置选择不同的线性层实现进行计算

        该方法根据初始化时的配置，调用相应的线性层实现：
        - 当 use_svd=True 且 dynamic_k>0 时，使用 DynamicBlockSVDLinear
        - 当 use_svd=True 但 dynamic_k≤0 时，使用 BlockSVDLinear
        - 当 use_svd=False 时，使用普通 nn.Linear

        Args:
            x (torch.Tensor): 输入张量，形状根据具体实现而定

        Returns:
            torch.Tensor: 线性变换后的输出张量，形状根据具体实现而定
        """
        return self.fc(x)

# -------------------- 单层 RoPE-Decoder --------------------
class RoPETransformerDecoderLayer(nn.Module):
    """
    单层 RoPE-Attention TransformerDecoder

    支持显式传入 start_pos 以处理任意绝对位置（如缓存推理、滑动窗口等）
    """
    __constants__ = ['d_model', 'nhead', 'head_dim']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", max_rope_len: int = 30000):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        self.dropout_p = dropout

        # 1. 手写 QKV 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # 2. 交叉 Attention 仍用官方实现（编码器侧无 RoPE）
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 3. FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        # 4. RoPE 缓冲
        cos, sin = precompute_rope_sin_cos(max_rope_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: torch.Tensor = torch.tensor(0)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        带旋转位置编码(RoPE)的Transformer解码器层前向传播

        该方法实现了一个完整的Transformer解码器层，包括：
        1. 带RoPE的自注意力机制
        2. 交叉注意力机制（与编码器输出交互）
        3. 前馈神经网络
        4. 支持KV缓存的高效自回归生成

        Args:
            tgt (torch.Tensor): 目标序列张量，形状为[B, L, C]，
                               其中B是批量大小，L是目标序列长度，C是模型维度
            memory (torch.Tensor): 编码器输出的记忆张量，形状为[B, T, C]，
                                  其中T是编码器序列长度
            tgt_mask (Optional[torch.Tensor]): 目标序列的因果掩码，可选，
                                             形状为[L, L]或[B, nhead, L, kv_len]，
                                             True表示需要掩码的位置
            memory_key_padding_mask (Optional[torch.Tensor]): 记忆序列的键填充掩码，可选，
                                                          形状为[B, T]，
                                                          True表示填充位置
            cache (Optional[Tuple[torch.Tensor, torch.Tensor]]): 缓存的键值对，可选，
                                                              形状为(k_cache, v_cache)，
                                                              其中k_cache和v_cache的形状均为[B, nhead, cache_len, head_dim]
            start_pos (torch.Tensor): 目标序列在完整序列中的起始位置，
                                     用于RoPE的绝对位置编码计算，默认值为0

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - 输出张量，形状为[B, L, C]，经过解码器层处理后的目标序列表示
                - 更新后的缓存，形状为(k_new, v_new)，用于后续自回归生成

        处理流程：
        1. QKV投影：将输入张量投影到查询、键、值空间
        2. KV缓存拼接：如果提供了缓存，将当前键值与缓存拼接
        3. 位置编码生成：根据start_pos生成查询和键的绝对位置ID
        4. RoPE应用：将旋转位置编码应用到查询和键张量
        5. 自注意力计算：使用Scaled Dot-Product Attention计算自注意力
        6. 残差连接与层归一化：将自注意力输出与输入相加，然后进行层归一化
        7. 交叉注意力计算：与编码器输出进行交叉注意力计算
        8. 残差连接与层归一化：将交叉注意力输出与上一步结果相加，然后进行层归一化
        9. 前馈网络：通过两层线性变换和激活函数处理
        10. 残差连接与层归一化：将前馈网络输出与上一步结果相加，然后进行层归一化

        关键技术点：
        - RoPE：旋转位置编码，通过显式位置ID和预计算的cos/sin值实现
        - KV缓存：支持高效的自回归生成，避免重复计算
        - FlashAttention：自动使用PyTorch的高效注意力实现
        - 因果掩码：确保解码过程中只能看到当前及之前的位置

        使用场景：
        - 自回归生成任务（如机器翻译、文本生成）
        - 带KV缓存的高效推理
        - 需要精确位置信息的序列建模任务
        """
        B, L = tgt.shape[:2]
        device = tgt.device

        # 1. 投影 QKV
        q = self.q_proj(tgt).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, L, head_dim]
        k = self.k_proj(tgt).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(tgt).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # 2. Concatenate with cache if provided
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)  # [B, nhead, cache_len + L, head_dim]
            v = torch.cat([v_cache, v], dim=2)
            kv_len = k.size(2)
            # Efficient position ID creation: avoid repeated arange allocation
            k_pos_ids = torch.arange(kv_len, device=device)
            q_pos_ids = torch.arange(start_pos, start_pos + L, device=device)
        else:
            # No cache: positions start at `start_pos`
            pos_ids = torch.arange(start_pos, start_pos + L, device=device)
            k_pos_ids = pos_ids
            q_pos_ids = pos_ids

        # 3. Apply RoPE with explicit position IDs
        q_rope, k_rope = apply_rope(q, k, self.cos, self.sin, q_pos_ids, k_pos_ids)

        # 4. 自注意力 (FlashAttention when available)
        # Prepare attention mask
        attn_mask = tgt_mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # True means "mask this position" → fill with -inf
                attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
            # F.scaled_dot_product_attention 要求 mask 维度 [B, nhead, q_len, kv_len]
            if attn_mask.dim() == 2:
                # Expand [L, L] → [B, nhead, L, kv_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.nhead, -1, -1)

        # Self-attention with RoPE
        # pylint: disable=not-callable
        out = F.scaled_dot_product_attention(          # type: ignore[operator]
            q_rope, k_rope, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout1.p if self.training else 0.0
        )
        attn_out = out.transpose(1, 2).reshape(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)

        # 5. 交叉 Attention (no RoPE)
        attn2, _ = self.multihead_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout2(attn2)
        tgt = self.norm2(tgt)

        # 6. FFN
        ff = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff)
        tgt = self.norm3(tgt)

        return tgt, (k, v)          # 不断梯度，原地复用显存

class RopeTransformerArDecoder(nn.Module):
    """
    基于 Rope + Transformer 的自回归解码器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        max_text_length: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_rope_len: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_text_length = max_text_length
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.num_classes)

        # 用 RoPE 解码层
        layer = RoPETransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            max_rope_len=max_rope_len)
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, encoder_features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练阶段：teacher forcing，无需 cache
        """
        memory = self.input_proj(encoder_features)
        tgt = self.embedding(targets) * math.sqrt(self.hidden_dim)  # [B, L, hidden_dim]
        mask = self.generate_square_subsequent_mask(targets.size(1)).to(targets.device)
        mem_mask = (memory.abs().sum(dim=-1) == 0)   # 建议后续换成显式 mask

        for layer in self.layers:
            tgt, _ = layer(tgt, memory,
                           tgt_mask=mask,
                           memory_key_padding_mask=mem_mask)
        logits = self.output_proj(tgt)
        return logits, tgt

    @torch.no_grad()
    def generate_step(self,
             pos: torch.Tensor,
             tgt_t: torch.Tensor,                      # [B, 1]  当前 token id
             memory: torch.Tensor,                     # [B, T, C]
             memory_key_padding_mask: torch.Tensor,    # [B, T]
             *caches,
             already_embedded: bool = False
             ) -> Tuple[torch.Tensor, List[torch.Tensor]]:  # 返回 (logit, new_caches...)
        """
        单步前向，用于 ONNX 导出。
        caches 展开成 n_layers*2 个 torch.Tensor：
            [k_cache_0, v_cache_0, k_cache_1, v_cache_1, ...]
        输出同样展开。
        """
        B, device = tgt_t.size(0), tgt_t.device
        L = tgt_t.size(1)                       # 1
        # tgt = self.embedding(tgt_t)             # [B, 1, d]
        if not already_embedded:
            tgt = self.embedding(tgt_t) * math.sqrt(self.hidden_dim)           # 训练/旧接口
        else:
            tgt = tgt_t                           # 来自 ONNXWrapper，已是 float
        mask = torch.zeros((L, L), device=device, dtype=torch.float32)
        # 把 caches 还原成 Tuple[(k,v), ...]
        num_layers = len(self.layers)
        assert len(caches) == num_layers * 2, \
            f"caches length {len(caches)} != 2 * {num_layers}"

        cache_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i in range(len(self.layers)):
            k_cache = caches[2*i]
            v_cache = caches[2*i+1]
            if k_cache.numel() == 0 or v_cache.numel() == 0:            # 空 cache 用 0-shape 表示
                cache_list.append(None)
            else:
                cache_list.append((k_cache, v_cache))

        new_caches = []
        for i, layer in enumerate(self.layers):
            tgt, (new_k, new_v) = layer(tgt, memory,
                                        tgt_mask=mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        cache=cache_list[i],
                                        start_pos=pos
                                        )
            new_caches.extend([new_k, new_v])   # 平坦化

        logit = self.output_proj(tgt[:, -1:, :])  # [B, 1, num_classes]
        return logit, new_caches

    def generate(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理：自回归 + KV-cache
        """
        B, device = encoder_features.size(0), encoder_features.device
        memory = self.input_proj(encoder_features)
        mem_mask = (memory.abs().sum(dim=-1) == 0)

        # 初始输入 <sos>
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        caches = [None] * len(self.layers)          # 每层 (k_cache, v_cache)

        for step in range(self.max_text_length - 1):
            # 1. 只取最后一个 token 的 embedding
            tgt_step = self.embedding(generated[:, -1:]) * math.sqrt(self.hidden_dim)        # [B, 1, d]

            # 2. 构造 1×kv_len 的 causal mask
            kv_len = generated.size(1)                          # 当前总长度
            mask_1d = torch.zeros((1, kv_len), device=device)   # [1, kv_len]
            # 如果 kv_len>1，则把未来位置置 -inf
            if kv_len > 1:
                mask_1d = torch.cat([torch.zeros(1, kv_len-1, device=device),
                                    torch.full((1, 1), float('-inf'), device=device)], dim=1)

            # 3. 扩到 [B, nhead, 1, kv_len]
            mask_1d = mask_1d.unsqueeze(0).unsqueeze(0).expand(B, self.layers[0].nhead, -1, -1)

            # 4. 逐层前向，更新 cache
            for i, layer in enumerate(self.layers):
                tgt_step, caches[i] = layer(tgt_step, memory,
                                            tgt_mask=mask_1d,
                                            memory_key_padding_mask=mem_mask,
                                            cache=caches[i],
                                            start_pos=step
                                            )

            # 5. 采样下一 token
            logit = self.output_proj(tgt_step)                  # [B, 1, num_classes]
            nxt = logit.argmax(dim=-1, keepdim=False).squeeze(-1)  # [B, 1]
            nxt = nxt.unsqueeze(1)                     # [B, 1]
            generated = torch.cat([generated, nxt], dim=1)

            if (nxt == self.eos_token).all():
                break

        # 补齐到 max_text_length（用于后续 loss 计算）
        pad_len = self.max_text_length - generated.size(1)
        if pad_len > 0:
            generated = torch.cat([generated,
                                torch.full((B, pad_len), self.pad_token,
                                            dtype=torch.long, device=device)], dim=1)

        # 再跑一遍拿完整 logits（teacher forcing 模式，用真实长度 mask 即可）
        tgt = self.embedding(generated) * math.sqrt(self.hidden_dim)
        mask = self.generate_square_subsequent_mask(self.max_text_length).to(device)
        for i, layer in enumerate(self.layers):
            tgt, _ = layer(tgt, memory,
                        tgt_mask=mask,
                        memory_key_padding_mask=mem_mask,
                        cache=None)
        logits = self.output_proj(tgt)

        return logits, tgt

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成下三角掩码"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def beam_search(self, logit: torch.Tensor, k: int = 5, max_len: int = 70) -> List[int]:
        """
        使用束搜索（Beam Search）进行自回归解码，生成最优序列

        束搜索是一种启发式搜索算法，通过维护k个最有可能的候选序列（束），
        在每一步扩展每个束，然后选择得分最高的k个新束，直到达到最大长度或所有束都结束。

        Args:
            logit (torch.Tensor): 预计算的log_softmax概率张量，形状为[L, V]，
                                 其中L是序列长度，V是词汇表大小
            k (int, optional): 束大小，即每一步保留的候选序列数量。默认5
            max_len (int, optional): 生成序列的最大长度。默认70

        Returns:
            List[int]: 生成的最优序列，包含sos_token和eos_token

        算法步骤：
        1. 初始化：从sos_token开始，初始化一个束
        2. 迭代生成：
            a. 对每个束，扩展所有可能的下一个token
            b. 计算每个扩展序列的得分
            c. 选择得分最高的k个序列作为新的束
        3. 终止条件：达到最大长度或所有束都包含eos_token
        4. 返回得分最高的序列

        得分计算：
            序列得分 = 所有token的log_softmax概率之和 / 序列长度^0.7
            除以序列长度的幂是为了避免偏向较长的序列
        """
        # logit [L,V] 已 log_softmax
        beams = [(0.0, [self.sos_token])]
        for step in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == self.eos_token:   # 已结束
                    new_beams.append((score, seq))
                    continue
                # 取最后一步的 logit
                lgt = logit[step]       # [V]
                topk_score, topk_idx = torch.topk(lgt, k)
                for s, idx in zip(topk_score, topk_idx):
                    new_beams.append((score + s.item(), seq + [idx.item()]))
            # 去重 + 取 topk
            new_beams = sorted(new_beams, key=lambda x: x[0] / len(x[1])**0.7, reverse=True)[:k]
            beams = new_beams
        return beams[0][1]   # 得分最高序列

class RopeTransformerNarDecoder(nn.Module):
    """
    基于 Rope + Transformer 的高性能并行解码器
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        max_text_length: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_rope_len: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_text_length = max_text_length

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_classes)

        # === 固定长度的可学习 content query（纯语义）===
        self.register_buffer("content_query", torch.empty(max_text_length, hidden_dim))
        # content query 初始值要用 xavier/kaiming 做 small norm，否则训练初期梯度爆炸
        nn.init.xavier_uniform_(self.content_query)

        # 解码层（必须内部正确实现 RoPE）
        layer = RoPETransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            max_rope_len=max_rope_len
        )
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        非自回归前向
        Returns:
            logits: [B, L, V]
            tgt:    [B, L, C]  （供中间特征可视化或蒸馏）
        """
        B = x.size(0)
        memory = self.input_proj(x)                       # [B, T, C]
        mem_mask = (memory.abs().sum(-1) == 0)            # [B, T]

        # ③ 零拷贝 broadcast，ONNX 优化后无 expand 节点
        tgt = self.content_query.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            tgt, _ = layer(tgt, memory, memory_key_padding_mask=mem_mask)

        logits = self.output_proj(tgt)
        return logits, tgt

class RopeMultiHeadAttention(nn.Module):
    """集成了 RoPE 的多头自注意力层"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE 模块，作用于每个头的维度
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状 (B, T, D_model)
        """
        B, T, _ = x.shape

        # 1. 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分为多头
        # (B, T, D_model) -> (B, T, nhead, head_dim) -> (B, nhead, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 3. 应用 RoPE（关键：传入 x.device）
        cos, sin = self.rotary_emb(T, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # 4. 高效的 Scaled Dot-Product Attention
        # PyTorch 2.0+ 会自动选择最优实现（如 FlashAttention）
        # pylint: disable=not-callable
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # 5. 合并多头
        # (B, nhead, T, head_dim) -> (B, T, nhead, head_dim) -> (B, T, D_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 6. 输出投影
        return self.out_proj(attn_output)

class RopeMultiHeadAttentionGQA(nn.Module):
    """
    带 RoPE 和 Skip-Attention 的 Group Query Attention
    用法：把 nhead_kv 设成 nhead//4 即可得到 GQA-4
    支持局部-全局Skip-Attention优化
    支持MLA (Multi-head Latent Attention) 优化（未测试有效性）
    """
    def __init__(self,
                 d_model: int,                      # 模型维度
                 nhead: int,                        # 多头数量
                 nhead_kv: int | None = None,       # KV头数量（可选）
                 dropout: float = 0.1,              # Dropout比例
                 use_weighted: bool = True,         # 是否使用Weighted GQA
                 skip_window: int = 24,             # Skip-Attention局部窗口大小（针对中文优化）
                 global_tokens: int = 8,            # 全局token数量
                 use_skip_attention: bool = False,  # 是否使用Skip-Attention
                 use_mla: bool = False,             # 是否使用Multi-head Latent Attention
                 latent_dim: int = 64):             # 潜在维度大小
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nhead_kv = nhead_kv if nhead_kv is not None else nhead
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "RoPE need even head_dim"

        # 1 组 Q 头对应 nhead // nhead_kv 组 K/V 头
        self.n_rep = self.nhead // self.nhead_kv

        # Skip-Attention参数
        self.skip_window = skip_window
        self.global_tokens = global_tokens
        self.use_skip_attention = use_skip_attention

        # MLA参数
        self.use_mla = use_mla
        self.latent_dim = latent_dim

        # 是否使用Weighted GQA
        self.use_weighted = use_weighted

        # 线性投影 —— Q 保持原通道，K/V 通道按组缩减
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.nhead_kv * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.nhead_kv * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE 模块，作用于每个头的维度
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Weighted GQA: 为每个查询头学习一个权重
        if self.use_weighted:
            # 每个查询头对应一个权重，用于加权其对应的KV组
            self.head_weights = nn.Parameter(torch.ones(self.nhead))

            # 初始化权重
            nn.init.xavier_uniform_(self.head_weights.unsqueeze(0)).squeeze(0)

        # MLA: 多头潜在注意力
        if self.use_mla:
            # 潜在查询投影
            self.latent_q_proj = nn.Linear(d_model, self.nhead * self.latent_dim)
            # 潜在键值投影
            self.latent_k_proj = nn.Linear(d_model, self.nhead_kv * self.latent_dim)
            self.latent_v_proj = nn.Linear(d_model, self.nhead_kv * self.latent_dim)
            # 潜在注意力输出投影
            self.latent_out_proj = nn.Linear(self.nhead * self.latent_dim, d_model)
            # 潜在注意力与普通注意力的融合权重
            self.mla_fusion_weight = nn.Parameter(torch.tensor(0.5))

    def create_skip_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建Skip-Attention的稀疏注意力掩码，优化长序列注意力计算

        Skip-Attention是一种稀疏注意力机制，通过限制每个token只能关注局部窗口内的token
        和少量全局token，大幅减少计算复杂度，同时保持模型性能。

        Args:
            seq_len (int): 序列长度，即注意力掩码的维度
            device (torch.device): 设备类型，用于创建掩码张量

        Returns:
            torch.Tensor: 稀疏注意力掩码，形状为[seq_len, seq_len]，
                        值为0.0表示允许注意力，值为-inf表示禁止注意力

        掩码结构：
        1. 局部注意力窗口：每个token可以关注其前后skip_window//2个token
        2. 全局token：前global_tokens个token可以关注所有token，同时所有token也可以关注它们

        示例：
        假设seq_len=10，skip_window=4（半窗口为2），global_tokens=2
        掩码矩阵将呈现以下结构（0.0表示允许注意力，-inf表示禁止）：
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 全局token 0
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 全局token 1
            [0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf],  # token 2
            [-inf, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf],  # token 3
            [-inf, -inf, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf],  # token 4
            # ... 后续token类似 ...
        ]
        """
        half_window = self.skip_window // 2

        # ONNX兼容的掩码创建方法
        # 创建距离矩阵：|i - j|
        # 使用广播机制创建距离矩阵，避免torch.arange
        i = torch.arange(seq_len, device=device, dtype=torch.float32).view(-1, 1)
        j = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, -1)
        dist_matrix = torch.abs(i - j)

        # 局部窗口掩码：距离 <= half_window 的位置为 0.0，其他为 -inf
        # ONNX兼容：使用大的负数替代 -inf，避免数值问题
        NEG_INF = -1e9  # ONNX兼容的负无穷大替代值
        mask = torch.where(dist_matrix <= half_window,
                          torch.tensor(0.0, device=device, dtype=torch.float32),
                          torch.tensor(NEG_INF, device=device, dtype=torch.float32))

        # 全局token可以 attend to all
        # 使用数学运算替代动态切片赋值
        if self.global_tokens > 0:
            # 创建全局token掩码：前global_tokens行和列设为0.0
            global_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)

            # 为前global_tokens行和列创建掩码
            row_mask = (i < self.global_tokens).float()
            col_mask = (j < self.global_tokens).float()

            # 如果行或列是全局token，则该位置为0.0
            global_indicator = torch.maximum(row_mask, col_mask)

            # 应用全局token掩码：全局位置设为0.0，其他保持原值
            mask = mask * (1.0 - global_indicator) + global_indicator * 0.0

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状 (B, T, D_model)
        """
        B, T, _ = x.shape

        # 1. 线性投影
        q = self.q_proj(x)                       # (B,T,D)
        k = self.k_proj(x)                       # (B,T,nhead_kv*head_dim)
        v = self.v_proj(x)

        # 2. 拆分为多头
        # (B, T, D_model) -> (B, T, nhead, head_dim) -> (B, nhead, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead_kv, self.head_dim).transpose(1, 2)

        # 3. RoPE：旋转后再 repeat K/V 到 Q 的头数
        cos, sin = self.rotary_emb(T, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.use_weighted:
            # Weighted GQA: 在复制K/V之前应用可学习的权重
            # 优化：使用 expand + contiguous 替代 repeat_interleave，减少显存分配
            # 首先复制K/V到Q的头数
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.head_dim)

            # 应用头部权重: [nhead] -> [1, nhead, 1, 1] 用于广播
            head_weights = self.head_weights.view(1, self.nhead, 1, 1)
            q = q * head_weights
        else:
            # 标准GQA: 每组 K/V 复制 n_rep 次
            # 优化：使用 expand + contiguous 替代 repeat_interleave，减少显存分配
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.head_dim)

        # 4. Skip-Attention：应用稀疏注意力掩码
        attn_mask = self.create_skip_mask(T, x.device) if self.use_skip_attention else None

        # 5. 高效的 Scaled Dot-Product Attention
        # PyTorch 2.0+ 会自动选择最优实现（如 FlashAttention）
        # pylint: disable=not-callable
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # 6. 合并多头
        # (B, nhead, T, head_dim) -> (B, T, nhead, head_dim) -> (B, T, D_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 7. MLA: Multi-head Latent Attention
        if self.use_mla:
            # 潜在空间注意力计算
            # 7.1 潜在QKV投影
            latent_q = self.latent_q_proj(x)  # (B,T,nhead*latent_dim)
            latent_k = self.latent_k_proj(x)  # (B,T,nhead_kv*latent_dim)
            latent_v = self.latent_v_proj(x)

            # 7.2 拆分潜在多头
            latent_q = latent_q.view(B, T, self.nhead, self.latent_dim).transpose(1, 2)
            latent_k = latent_k.view(B, T, self.nhead_kv, self.latent_dim).transpose(1, 2)
            latent_v = latent_v.view(B, T, self.nhead_kv, self.latent_dim).transpose(1, 2)

            # 7.3 复制K/V到Q的头数
            # 优化：使用 expand + reshape 替代 repeat_interleave，减少显存分配
            latent_k = latent_k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.latent_dim)
            latent_v = latent_v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.nhead, T, self.latent_dim)

            # 7.4 潜在空间注意力
            latent_out = F.scaled_dot_product_attention(
                latent_q, latent_k, latent_v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )

            # 7.5 合并潜在多头
            latent_out = latent_out.transpose(1, 2).contiguous().view(B, T, self.nhead * self.latent_dim)
            latent_out = self.latent_out_proj(latent_out)  # (B,T,D_model)

            # 7.6 融合普通注意力和潜在注意力
            # 使用可学习的融合权重，确保权重在[0,1]之间
            fusion_weight = torch.sigmoid(self.mla_fusion_weight)
            out = fusion_weight * out + (1 - fusion_weight) * latent_out

        # 8. 输出投影
        return self.out_proj(out)

class RopeTransformerEncoderLayer(nn.Module):
    """使用 RoPE 注意力的 Transformer Encoder 层 (Pre-LN 结构)

    支持的优化：
    - 算子融合: 融合 LayerNorm + Attention + FFN 为一个算子，减少内存访问
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1,
                 use_skip_attention: bool = False, skip_window: int = 24, global_tokens: int = 8,
                 use_mla: bool = False, latent_dim: int = 64, use_fused_ops: bool = False):
        super().__init__()
        self.use_fused_ops = use_fused_ops

        self.self_attn = RopeMultiHeadAttentionGQA(
            d_model, nhead, dropout=dropout,
            skip_window=skip_window, global_tokens=global_tokens,
            use_skip_attention=use_skip_attention,
            use_mla=use_mla,
            latent_dim=latent_dim
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Pre-LN 结构: Norm -> Attention -> Add -> Norm -> FFN -> Add
        if self.use_fused_ops:
            return self._fused_forward(src)
        else:
            return self._standard_forward(src)

    def _standard_forward(self, src: torch.Tensor) -> torch.Tensor:
        """标准前向传播：不使用算子融合"""
        src2 = self.self_attn(self.norm1(src))
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src

    def _fused_forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        融合前向传播：融合 LayerNorm + Attention + FFN 为一个算子

        优化原理：
        1. 减少中间结果的内存读写
        2. 合并多个操作为单个 kernel 调用
        3. 提高缓存命中率

        注意：这是一个简化的融合实现，真正的算子融合需要 CUDA kernel 编程
        这里通过减少中间变量和优化内存访问来模拟融合效果
        """
        # Pre-LN 结构: Norm -> Attention -> Add -> Norm -> FFN -> Add
        # 融合优化：减少中间变量的创建和内存访问

        # 第一步：Attention 分支
        norm1_out = self.norm1(src)
        attn_out = self.self_attn(norm1_out)

        # 融合：直接在残差连接上应用 dropout，避免创建临时变量
        src = src + self.dropout1(attn_out)

        # 第二步：FFN 分支
        norm2_out = self.norm2(src)

        # 融合：合并 linear1 + activation + dropout + linear2
        # 避免中间结果的存储
        ff_out = self.linear1(norm2_out)
        ff_out = self.activation(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)

        # 融合：直接在残差连接上应用 dropout
        src = src + self.dropout2(ff_out)

        return src

class RopeTransformerEncoder(nn.Module):
    """
    使用 RoPE 的 Transformer Encoder Decoder

    支持的优化：
    - Gradient Checkpointing: 减少显存占用，支持更大的 batch size
    - 算子融合: 融合 LayerNorm + Attention + FFN 为一个算子
    - MLA (Multi-head Latent Attention): 通过动态低秩分解减少参数量和计算量
    - Skip-Attention: 稀疏注意力机制，减少长序列计算复杂度
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        max_text_length: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_rope_len: int,
        use_svd: bool,
        k: int,
        r: int,
        dynamic_k: int,
        use_skip_attention: bool = True,
        skip_window: int = 24,          # 针对中文文本优化
        global_tokens: int = 8,
        use_mla: bool = False,          # 是否使用Multi-head Latent Attention
        latent_dim: int = 64,            # 潜在维度大小
        use_gradient_checkpointing: bool = True,  # 是否使用 gradient checkpointing
        use_fused_ops: bool = True              # 是否使用算子融合
    ):
        super().__init__()
        self.pad_token = pad_token
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fused_ops = use_fused_ops

        layer = RopeTransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            use_skip_attention=use_skip_attention,
            skip_window=skip_window,
            global_tokens=global_tokens,
            use_mla=use_mla,
            latent_dim=latent_dim,
            use_fused_ops=use_fused_ops
        )

        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

        # 最终输出前的归一化
        self.norm = nn.LayerNorm(hidden_dim)
        # CTC Loss 的分类头 - 支持动态低秩分解优化
        self.head = HybridLinear(
            use_svd=use_svd,
            in_features=hidden_dim,
            num_classes=num_classes,
            k=k,
            r=r,
            dynamic_k=dynamic_k
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 形状为 (B, T, D_model)
        """
        output = src

        # 使用 gradient checkpointing 减少显存占用
        if self.use_gradient_checkpointing and self.training:
            for layer in self.layers:
                output = torch.utils.checkpoint.checkpoint(layer, output, use_reentrant=False)
        else:
            for layer in self.layers:
                output = layer(output)

        output = self.norm(output)

        # Head: (B, T, D_model) -> (B, T, NumClasses)
        logits = self.head(output)
        # logits = self.head(output).clip(min=-8., max=8.)
        # 先看统计值
        # with torch.no_grad():
        #     print(f'logits min={logits.min():.2f}  max={logits.max():.2f}')

        return logits

class LightTransformerDecoder(nn.Module):
    """基于标准 Transformer Encoder 的 CTC 解码头"""
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        max_text_length: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_rope_len: int
    ):
        super().__init__()
        self.pad_token = pad_token
        self.num_classes = num_classes
        self.d_model = hidden_dim
        self.num_layers = num_layers

        # 若输入维度不一致，先投影
        self.input_proj = nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True             # pre-norm
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos = PositionalEncoding(hidden_dim, dropout)
        # CTC Loss 的分类头
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：基于标准Transformer编码器的CTC解码头

        该方法实现了基于标准Transformer编码器的CTC解码流程，包括：
        1. 输入特征投影（如果输入维度与模型维度不一致）
        2. 位置编码添加
        3. Transformer编码器处理
        4. 输出分类头映射

        Args:
            x (torch.Tensor): 输入特征张量，形状为[B, T, C_in]，
                            其中B是批量大小，T是时间步数，C_in是输入特征维度

        Returns:
            torch.Tensor: 输出logits张量，形状为[B, T, NumClasses]，
                        其中NumClasses是词汇表大小，用于后续CTC损失计算

        处理流程：
        1. 输入投影：将输入特征从C_in维度投影到hidden_dim维度
        2. 位置编码：为每个时间步添加正弦余弦位置编码
        3. Transformer编码：通过多层Transformer编码器提取上下文特征
        4. 分类头：将上下文特征映射到词汇表概率分布
        """
        # x: [B, T, C_in]
        x = self.input_proj(x)                      # [B, T, C_out]

        # 位置编码
        x = self.pos(x)
        x = self.tr(x)
        # Head: (B, T, D_model) -> (B, T, NumClasses)
        logits = self.head(x)
        # logits = self.head(x).clip(min=-8., max=8.)
        # 先看统计值
        # with torch.no_grad():
        #     print(f'logits min={logits.min():.2f}  max={logits.max():.2f}')

        return logits

class HybridDecoder(nn.Module):
    """
    采用CTC分支和自回归分支的混合解码器
    """
    def __init__(
        self,
        train_mode: str,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        max_text_length: int,
        dropout: float,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        num_layers: int,
        num_heads: int,
        max_rope_len: int,
        use_svd: bool,
        k: int,
        r: int,
        dynamic_k: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_text_length = max_text_length
        self.hidden_dim = hidden_dim
        self.train_mode = train_mode

        # CTC解码分支（Transformer-based）
        if train_mode == 'ctc' or train_mode == 'hybrid':
            self.ctc_decoder = RopeTransformerEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                max_text_length=max_text_length,
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                num_layers=num_layers,
                num_heads=num_heads,
                max_rope_len=max_rope_len,
                dropout=dropout,
                use_svd=use_svd,
                k=k,
                r=r,
                dynamic_k=dynamic_k
            )
        else:
            self.ctc_decoder = nn.Identity()

        # 自回归解码分支（Transformer-based）
        if train_mode == 'ar' or train_mode == 'hybrid':
            self.ar_decoder = RopeTransformerArDecoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                max_text_length=max_text_length,
                pad_token=pad_token,
                sos_token=sos_token,
                eos_token=eos_token,
                num_layers=num_layers,
                num_heads=num_heads,
                max_rope_len=max_rope_len,
                dropout=dropout
            )
        else:
            self.ar_decoder = nn.Identity()

        # 知识蒸馏用的特征对齐层
        if train_mode == 'hybrid':
            self.feature_align = nn.Linear(in_channels, hidden_dim)
        else:
            self.feature_align = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
        use_ctc: bool = True,
        use_ar: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [B, T, C]
            targets: 目标文本 [B, L] (训练时用)
            eval_mode: 训练时是否走推理分支（解决模型训练时不知道自回归推理错误字符）
            use_ctc: 是否使用CTC分支
            use_ar: 是否使用自回归分支

        Returns:
            包含各分支输出的字典
        """
        results = {}

        # CTC分支
        if use_ctc:
            ctc_logits = self.ctc_decoder(x)  # [B, T, num_classes]
            results['ctc_logits'] = ctc_logits

        # 自回归分支
        if use_ar:
            if self.training and not eval_mode:
                # 训练时使用teacher forcing
                ar_logits, ar_features = self.ar_decoder(x, targets)
                results['ar_logits'] = ar_logits
                results['ar_features'] = ar_features
            else:
                # 推理时使用自回归生成
                ar_logits, ar_features = self.ar_decoder.generate(x)
                results['ar_logits'] = ar_logits
                results['ar_features'] = ar_features

        # 特征对齐（用于知识蒸馏）
        if self.train_mode == 'hybrid':
            results['aligned_features'] = self.feature_align(x)

        return results

def topk_log_softmax(logits: torch.Tensor, K: int, dim: int = -1, training: bool = True, topk_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对logits张量执行Top-K LogSoftmax操作，只保留概率最高的K个类别

    该函数是标准LogSoftmax的优化版本，通过只保留每个位置概率最高的K个类别，
    大幅减少后续计算的复杂度，同时保持模型性能。特别适合处理大词汇表场景。

    Args:
        logits (torch.Tensor): 输入logits张量，形状为[B, T, num_classes]，
                            其中B是批量大小，T是时间步数，num_classes是词汇表大小
        K (int): 要保留的类别数量，通常远小于num_classes（K << num_classes）
        dim (int, optional): 执行Top-K操作的维度，必须是最后一维（即num_classes维）。默认-1
        training (bool, optional): 是否处于训练模式。训练模式下会进行数值稳定性处理。默认True
        topk_indices (Optional[torch.Tensor], optional): 预计算的Top-K索引，形状为[B, T, K]，
                                                        推理时可直接传入，避免重复计算。默认None

    Returns:
        torch.Tensor: 输出log_softmax概率张量，形状与logits相同[B, T, num_classes]，
                    只保留Top-K类别的概率，其余位置为-inf

    实现原理：
    1. 数值稳定性处理：训练模式下先减去每行最大值，避免exp计算溢出
    2. Top-K索引选择：根据logits值选择概率最高的K个类别索引
    3. 掩码生成：创建与logits同形的掩码，只保留Top-K类别位置
    4. 稀疏LogSoftmax：只对Top-K类别执行LogSoftmax计算，其余位置设为-inf

    训练与推理差异：
    - 训练：计算全局最大值进行数值稳定化，然后生成Top-K索引
    - 推理：直接使用外部传入的Top-K索引，从索引中计算最大值

    使用场景：
    - 大词汇表场景下的高效Softmax计算
    - 结合Dynamic Block-SVD Linear等稀疏结构使用
    - 减少内存占用和计算复杂度
    """
    assert dim == -1 or dim == logits.dim() - 1
    B, T, C = logits.shape

    if training:
        # 训练：先算全局 max 保数值稳定，再取 Top-K
        maxes = logits.max(dim=-1, keepdim=True).values        # [B,T,1]
        logits_stable = logits - maxes                         # [B,T,C]
        # Top-K 索引
        if topk_indices is None:
            topk_indices = logits_stable.topk(K, dim=-1).indices  # [B,T,K]
    else:
        # 推理：直接用外部传进来的索引（Dynamic Linear 已算好）
        if topk_indices is None:
            topk_indices = logits.topk(K, dim=-1).indices
        maxes = logits.gather(-1, topk_indices).max(dim=-1, keepdim=True).values  # [B,T,1]

    # 只对 Top-K 列做 exp
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
    logits_topk = logits.masked_fill(~mask, float('-inf'))
    log_probs_topk = F.log_softmax(logits_topk, dim=-1)        # [B,T,C]  非 Top-K 为 -inf

    return log_probs_topk

def transplant_linear_to_blocksVD(src_linear, blocksVD):
    """
    复制一个普通 Linear 的权重到 BlockSVDLinear 结构中
    Args:
    src_linear: nn.Linear  [d, c]
    blocksVD  : BlockSVDLinear
    用 torch.svd 做低秩分解，复制到 U/V 并保留 bias
    """
    import numpy as np
    W = src_linear.weight.data.cpu().numpy()   # [c, d]
    bias = src_linear.bias.data
    k = blocksVD.k
    r = blocksVD.r
    c_per = W.shape[0] // k
    with torch.no_grad():
        for i in range(k):
            Wi = W[i*c_per:(i+1)*c_per]        # [c_per, d]
            U, S, Vh = np.linalg.svd(Wi, full_matrices=False)
            Ur = U[:, :r] * np.sqrt(S[:r])     # [c_per, r]
            Vr = Vh[:r, :] * np.sqrt(S[:r]).reshape(-1,1)  # [r, d]
            # 复制到网络
            blocksVD.layers[i][0].weight.copy_(torch.from_numpy(Vr))   # U
            blocksVD.layers[i][1].weight.copy_(torch.from_numpy(Ur.T)) # V^T
        blocksVD.bias.copy_(bias)

def get_optimal_hyper_params(num_classes):
    """
    根据类别数量，生成 DynamicBlockSVDLinear 和 Top-K LogSoftmax 的最优超参数。

    Args:
        num_classes: 总类别数

    Returns:
        Dictionary of hyperparameters.
    """
    # Block-SVD Linear 的超参数
    k = max(8, min(32, int(math.log2(num_classes) * 1.2)))  # 基于类别数动态调整分块数
    r = 64                                                   # 秩固定为 64
    dynamic_k = min(int(0.04 * num_classes), 1000)           # 动态 Top-K 列数

    # Top-K LogSoftmax 的超参数
    K = min(int(0.04 * num_classes), 1000)                   # 保留列数
    chunks = min(max(1, num_classes // 2000), 8)             # 根据类别数调整分段数
    temperature = 0.7 if num_classes > 20000 else 1.0        # 大类别数时降低温度

    return {
        "block_svd": {
            "k": k,
            "r": r,
            "dynamic_k": dynamic_k
        },
        "log_softmax": {
            "K": K,
            "chunks": chunks,
            "temperature": temperature
        }
    }

def export_onnx_v1():
    device = "cpu"
    model = RopeTransformerArDecoder(
        in_channels=512,
        hidden_dim=256,
        num_classes=37,            # 0-9 A-Z blank
        max_text_length=25,
        pad_token=0,
        sos_token=1,
        eos_token=2,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_rope_len=30000
    ).to(device).eval()

    # 1. 随机生成导出用的 dummy 输入
    B = 2
    dummy_encoder = torch.randn(B, 50, 512, device=device)
    dummy_targets = torch.randint(0, 37, (B, 25), device=device)
    logits, _ = model(dummy_encoder, dummy_targets)
    print("train logits:", logits.shape)          # [B, 25, 39]

    logits, _ = model.generate(dummy_encoder)  # 先跑一次，触发 @torch.no_grad()
    print("infer logits:", logits.shape)      # [B, 25, 39]

    # 2. 声明动态维度
    batch_dim = Dim("batch")
    dynamic_shapes = {
        "encoder_features": {0: batch_dim},
        "targets": {0: batch_dim},
    }

    # 2. 导出 ONNX
    torch.onnx.export(
        model,
        (dummy_encoder, dummy_targets),          # 注意这里是 tuple
        "nrtp_rope_kvcache.onnx",
        input_names=["encoder_features", "targets"],
        output_names=["logits", "features"],
        # dynamic_shapes=dynamic_shapes,
        dynamic_axes={
            "encoder_features": {0: "batch"},
            "targets": {0: "batch"},
            "logits": {0: "batch"},
            "features": {0: "batch"}
        },
        dynamo=False,
        opset_version=18,
        do_constant_folding=True
    )
    print("ONNX 导出成功 → nrtp_rope_kvcache.onnx")

def export_onnx_step():
    '''导出单步推理的 ONNX 模型并测试'''

    import argparse
    import onnxruntime as ort
    import numpy as np
    from torch.export import Dim

    # -------------- 外壳：把 embedding 包进来 --------------
    class ONNXWrapper(nn.Module):
        """
        只做两件事：
        1. 把 int64 [B, 1] 的 token_id  embedding 成 float32 [B, 1, d]
        2. 调用原模型 forward，把剩余输入原封不动传进去
        """
        def __init__(self, inner: RopeTransformerArDecoder):
            super().__init__()
            self.inner = inner          # 原来的解码器
            # 把 embedding 层提出来，防止导出时找不到权重
            self.embedding = inner.embedding

        def forward(self,
                    pos: torch.Tensor,
                    token_id: torch.Tensor,          # [B, 1]  int64
                    memory: torch.Tensor,            # [B, T, C]
                    mem_mask: torch.Tensor,
                    *caches):                  # 2*layers 个 cache
            # 1. 预处理：id -> emb
            token_emb = self.embedding(token_id) * math.sqrt(self.inner.hidden_dim)          # [B, 1, d]

            # 2. 原模型 forward（已 @torch.no_grad()）
            logit, new_caches = self.inner.generate_step(pos, token_emb, memory, mem_mask, *caches, already_embedded=True)
            return (logit, *new_caches)      # 保持和之前一样的扁平输出


    # 外层包预处理，ONNX 输入依旧是 [B, 1] token_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--max_txt", type=int, default=25)
    parser.add_argument("--max_enc", type=int, default=50)
    parser.add_argument("--out", type=str, default="ar_rope_step_preproc.onnx")
    args = parser.parse_args()

    device = "cpu"
    base_model = RopeTransformerArDecoder(
        in_channels=512,
        hidden_dim=args.hidden,
        num_classes=37,
        max_text_length=args.max_txt,
        pad_token=0, sos_token=1, eos_token=2,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=0.0,
        max_rope_len=30000
    ).to(device).eval()

    # 包外壳
    model = ONNXWrapper(base_model).eval()

    # ---- dummy 输入 ----
    B = 2
    pos = torch.tensor(1)
    dummy_token_id = torch.full((B, 1), 1, dtype=torch.long, device=device)      # [B, 1]  int64
    dummy_mem      = torch.randn(B, args.max_enc, args.hidden)   # [B, T, hidden_dim]
    dummy_mask     = torch.zeros(B, args.max_enc, dtype=torch.bool, device=device)
    dummy_caches = []
    for _ in range(args.layers):
        dummy_caches.append(torch.randn(B, args.heads, 1, args.hidden // args.heads) * 1e-3)
        dummy_caches.append(torch.randn(B, args.heads, 1, args.hidden // args.heads) * 1e-3)

    # ---- 先跑一遍确保图能 trace ----
    with torch.no_grad():
        outputs = model(pos, dummy_token_id, dummy_mem, dummy_mask, *dummy_caches)
        print("step ok, logit & cache shape:", outputs[0].shape)

    # ---- 导出 ONNX ----
    input_names = ["pos", "token_id", "memory", "mem_mask"] + [f"cache_{i}" for i in range(2 * args.layers)]
    output_names = ["logit"] + [f"new_cache_{i}" for i in range(2 * args.layers)]

    # 动态轴
    dynamic = {
        "token_id": {0: "B"},
        "memory": {0: "B", 1: "T"},
        "mem_mask": {0: "B", 1: "T"},
        "logit": {0: "B"},
    }
    for i in range(2 * args.layers):
        dynamic[f'cache_{i}']     = {0: 'B', 1: 'nhead', 2: 'cache_len', 3: 'head_dim'}      # [B,nhead,cache_len,head_dim]
        dynamic[f'new_cache_{i}'] = {0: 'B', 1: 'nhead', 2: 'cache_len', 3: 'head_dim'}
    # 1. 构造动态维度字典（dynamo_export 格式）
    #    注意：dynamo_export 要求用 Dict[str, Dict[int, str]]
    pos_dim = Dim("pos")
    batch_dim = Dim("batch")
    time_dim = Dim("time")
    nhead_dim = Dim("n_head")
    cache_dim = Dim("cache_len")
    head_dim = Dim("head_dim")
    dynamic_shapes = {
        "pos": pos_dim,
        "token_id": {0: batch_dim, 1: time_dim},          # [B, L]  这里 L==1
        "memory": {0: batch_dim, 1: time_dim},
        "mem_mask": {0: batch_dim, 1: time_dim},
    }
    for i in range(2 * args.layers):
        dynamic_shapes[f"cache_{i}"] = {0: batch_dim, 1: nhead_dim, 2: cache_dim, 3: head_dim}

    torch.onnx.export(
        model,
        (pos, dummy_token_id, dummy_mem, dummy_mask, *dummy_caches),
        args.out,
        input_names=input_names,
        output_names=output_names,
        # dynamic_shapes=dynamic_shapes,
        dynamic_axes=dynamic,
        opset_version=18,
        dynamo=False,
        do_constant_folding=False,
    )
    print(f"ONNX 导出成功（带预处理）→ {args.out}")

    sess = ort.InferenceSession("ar_rope_step_preproc.onnx")
    print("INPUTS:", [n.name for n in sess.get_inputs()])
    # 输入依旧是 2 维 token_id
    pos_offset = np.array(1, dtype=np.int64)
    token = np.full((B, 1), 1, dtype=np.int64)  # [B, 1]
    memory = np.random.randn(B, args.max_enc, args.hidden).astype(np.float32)
    mem_mask = np.zeros((B, args.max_enc), dtype=bool)
    print('token:', token.shape, token)

    outputs = sess.run(None, {
        "pos": pos_offset,
        "token_id": token,
        "memory": memory,
        "mem_mask": mem_mask,
        **{f"cache_{i}": np.zeros((0, 8, 0, 32), dtype=np.float32)
        for i in range(2 * args.layers)}  # 6 层 * 2
    })
    print("step ok, logit & cache shape:", outputs[0].shape)   # [2, 1, 39]

def infer_onnx_step():
    '''使用 ONNX Runtime 做单步推理'''

    import numpy as np
    import onnxruntime as ort

    class RoPEOnnxDecoder:
        def __init__(self,
                    onnx_path: str,
                    max_text_length: int = 25,
                    sos_token: int = 1,
                    eos_token: int = 2,
                    pad_token: int = 0):
            self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider'])
            self.max_len = max_text_length
            self.sos = sos_token
            self.eos = eos_token
            self.pad = pad_token

            # 缓存节点名字
            self.input_names = [n.name for n in self.sess.get_inputs()]
            self.output_names = [n.name for n in self.sess.get_outputs()]
            self.num_layers = (len(self.input_names) - 3) // 2   # 除 token/memory/mem_mask 外都是 cache

        def reset_cache(self, batch_size: int):
            """返回空 cache （0-shape）"""
            return [np.zeros((0, 0, 0, 0), dtype=np.float32) for _ in range(2 * self.num_layers)]

        def decode(self, memory: np.ndarray, mem_mask: np.ndarray):
            """
            memory: [B, T, 512]  float32
            mem_mask: [B, T]   bool (True=pad)
            返回 list[str] 长度 = B
            """
            B = memory.shape[0]
            device = memory.device if hasattr(memory, 'device') else 'cpu'

            # 初始输入
            token = np.full((B, 1), self.sos, dtype=np.int64)  # [B, 1]
            caches = self.reset_cache(B)

            generated = [[] for _ in range(B)]

            for step in range(self.max_len - 1):
                pos = np.array(step, dtype=np.int64)
                feed = {
                    "pos": pos,
                    'token_id': token,
                    'memory': memory,
                    'mem_mask': mem_mask
                }
                feed.update({f'cache_{i}': caches[i] for i in range(2 * self.num_layers)})

                outputs = self.sess.run(self.output_names, feed)
                logit = outputs[0]                      # [B, 1, 39]
                caches = outputs[1:]                    # list of new caches

                nxt = logit.argmax(axis=-1, keepdims=True)  # [B, 1, 1]
                nxt = nxt.squeeze(-1)                # [B, 1, 1] → [B, 1]
                token = nxt.astype(np.int64)

                # 收集结果
                for b in range(B):
                    generated[b].append(int(nxt[b, 0]))

                # 早停
                if np.all(nxt[:, 0] == self.eos):
                    break

            # 转字符串（这里简单映射，可按自己 vocab 来）
            def to_str(ids):
                s = []
                for i in ids:
                    if i == self.eos or i == self.pad:
                        break
                    if 0 <= i < 37:
                        s.append(chr(i + 48) if i < 10 else chr(i - 10 + 65))
                return ''.join(s)

            return [to_str(ids) for ids in generated]


    # -------------------- 测试 --------------------
    # 1. 随机特征当例子，实际用真实 encoder 输出
    B, T = 2, 50
    memory = torch.randn(B, T, 256).numpy()
    mem_mask = np.zeros((B, T), dtype=bool)   # 无 pad

    decoder = RoPEOnnxDecoder("ar_rope_step_preproc.onnx",
                                max_text_length=25,
                                sos_token=1,
                                eos_token=2,
                                pad_token=0)

    texts = decoder.decode(memory, mem_mask)
    print("ONNX 推理结果:", texts)          # ['HELLO', 'WORLD'] 等

def export_onnx_rope_encoder():
    """
    导出单个支持动态 batch_size 的 RopeTransformerEncoder ONNX 文件
    支持 dynamo=True 与 opset_version>=18，动态维度用 Dim 对象
    导出后测试不同 batch_size 输入
    """
    import onnxruntime as ort
    import numpy as np

    # 配置参数
    device = "cpu"
    in_channels = 256
    hidden_dim = 256
    num_classes = 25000  # 大类别数，验证动态SVD优化效果
    max_text_length = 70
    pad_token = 0
    sos_token = 1
    eos_token = 2
    num_layers = 3
    num_heads = 8
    dropout = 0.1
    max_rope_len = 30000
    use_svd = True
    dynamic_k = 1000
    k = 10
    r = 64

    use_skip_attention = True  # 启用 Skip-Attention
    skip_window = 24
    global_tokens = 8
    use_mla = True  # 启用 MLA (Multi-head Latent Attention)
    latent_dim = 64
    use_gradient_checkpointing = False  # 推理时关闭checkpointing
    use_fused_ops = True  # 启用算子融合

    # 计算参数量和优化效果
    original_params = in_channels * num_classes
    svd_params = k * (in_channels * r + r + (num_classes // k) * r)
    param_reduction = 100 * (1 - svd_params / original_params)

    print(f"=" * 60)
    print(f"BlockSVDLinear 优化配置（静态SVD）")
    print(f"=" * 60)
    print(f"num_classes: {num_classes}")
    print(f"k (分块数): {k} (确保能整除 {num_classes})")
    print(f"r (SVD秩): {r}")
    print(f"dynamic_k: {dynamic_k} (ONNX导出时禁用动态选择)")
    print(f"原始参数量: {original_params:,}")
    print(f"SVD参数量: {svd_params:,}")
    print(f"参数量减少: {param_reduction:.1f}%")
    print(f"说明: 动态SVD在ONNX导出时禁用，推理时可启用以获得更高性能")
    print(f"=" * 60)

    # 创建模型 - 启用所有优化
    model = RopeTransformerEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        max_text_length=max_text_length,
        pad_token=pad_token,
        sos_token=sos_token,
        eos_token=eos_token,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_rope_len=max_rope_len,
        use_svd=use_svd,
        k=k,
        r=r,
        dynamic_k=dynamic_k,
        use_skip_attention=use_skip_attention,
        skip_window=skip_window,
        global_tokens=global_tokens,
        use_mla=use_mla,
        latent_dim=latent_dim,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_fused_ops=use_fused_ops
    ).to(device).eval()

    # 定义动态维度
    batch_dim = Dim("batch", min=2, max=16)  # 支持 batch_size 1-8
    seq_len_dim = Dim("seq_len", min=10, max=100)  # 支持序列长度 10-100

    # 生成导出用的 dummy 输入（使用最小 batch_size）
    T = 50  # 固定序列长度用于导出
    dummy_input = torch.randn(3, T, in_channels, device=device)

    print(f"\n模型配置:")
    print(f"  - 模型维度: {hidden_dim}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 层数: {num_layers}")
    print(f"  - Skip-Attention: window={skip_window}, global_tokens={global_tokens}")
    print(f"  - MLA: latent_dim={latent_dim}")
    print(f"  - 算子融合: {use_fused_ops}")
    print(f"  - SVD: 静态SVD (k={k}, r={r})")

    # 先跑一遍确保模型能正常运行
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\n模型输出形状: {output.shape}")
        print(f"预期输出维度: ({3}, {T}, {num_classes})")

    # 导出单个 ONNX 文件，支持动态 batch_size
    onnx_filename = "rope_transformer_encoder_optimized.onnx"

    # 定义动态轴配置
    # 注意：DynamicBlockSVDLinear 使用动态稀疏选择，依赖数据依赖的控制流
    # torch.export/dynamo 无法处理这种动态性，因此使用 legacy TorchScript 导出
    if use_svd and dynamic_k > 0:
        dynamo = False
        dynamic_shapes = None
        dynamic_axes = {
            "src": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"}
        }
    else:
        dynamo = True
        dynamic_axes = None
        dynamic_shapes = {
            "src": {0: batch_dim, 1: seq_len_dim}
        }

    # 导出 ONNX 模型
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_filename,
        dynamo=dynamo,
        verbose=True,
        report=True,
        verify=True,
        export_params=True,
        external_data=False,
        opset_version=18,
        do_constant_folding=False,
        input_names=["src"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes
    )

    print(f"\nONNX 导出成功: {onnx_filename}")
    print(f"优化效果:")
    print(f"  - 参数量减少: SVD优化可将参数量减少约 {100 * (1 - (k * (in_channels * r + r + (num_classes // k) * r)) / (in_channels * num_classes)):.1f}%")
    print(f"  - 注意力计算: MLA和Skip-Attention可进一步减少计算量")
    print(f"  - 内存访问: 算子融合可减少内存访问次数")
    print(f"  - 动态SVD: 训练时可启用以获得更高性能，ONNX导出时使用静态SVD")

    # 使用 ONNX Runtime 加载导出的模型
    sess = ort.InferenceSession(onnx_filename)

    # 测试不同 batch_size
    test_batch_sizes = [1, 2, 4, 8]

    for B in test_batch_sizes:
        print(f"\n=== 测试 batch_size={B} ===")

        # 生成随机输入
        test_input = torch.randn(B, T, in_channels, device=device)

        # PyTorch 输出
        with torch.no_grad():
            torch_output = model(test_input).numpy()

        # ONNX Runtime 输出
        test_input_np = test_input.numpy()
        ort_outputs = sess.run(None, {"src": test_input_np})

        # 计算差异
        diff = np.abs(torch_output - ort_outputs[0]).mean()
        print(f"输入形状: {test_input.shape}")
        print(f"PyTorch 输出形状: {torch_output.shape}")
        print(f"ONNX 输出形状: {ort_outputs[0].shape}")
        print(f"PyTorch 与 ONNX Runtime 输出差异: {diff:.8f}")

        if diff < 1e-5:
            print(f"✓ batch_size={B} 验证通过")
        else:
            print(f"✗ batch_size={B} 验证失败，差异较大")

    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    print(get_optimal_hyper_params(25000))
    export_onnx_step()
    infer_onnx_step()
    export_onnx_rope_encoder()
