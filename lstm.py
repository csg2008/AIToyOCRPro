"""
xLSTM is refer from: https://github.com/muditbhargava66/PyxLSTM
xLSTM: Extended Long Short-Term Memory

This package implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM combines sLSTM (Scalar LSTM) and mLSTM (Matrix LSTM) in a novel
architecture to achieve state-of-the-art performance on various language
modeling tasks.

This __init__.py file imports and exposes the main components of the xLSTM model.

Author: Mudit Bhargava
Date: June 2024
"""

__version__ = "2.0.0"
__author__ = "Mudit Bhargava"

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple

class sLSTM(nn.Module):
    """
    sLSTM layer implementation.

    This layer applies multiple sLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of sLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the sLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_state[layer_idx]
                h, c = layer(x, (h, c))
                hidden_state[layer_idx] = (h, c)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class sLSTMCell(nn.Module):
    """
    sLSTM cell implementation.

    This cell uses exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        """
        Forward pass of the sLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)

        i, f, g, o = gates.chunk(4, 1)

        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

class mLSTM(nn.Module):
    """
    mLSTM layer implementation.

    This layer applies multiple mLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of mLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([mLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the mLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, C = hidden_state[layer_idx]
                h, C = layer(x, (h, C))
                hidden_state[layer_idx] = (h, C)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class mLSTMCell(nn.Module):
    """
    mLSTM cell implementation.

    This cell uses a matrix memory state and exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, input, hx):
        """
        Forward pass of the mLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, C = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)

        i, f, o = gates.chunk(3, 1)

        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        o = torch.sigmoid(o)

        q = self.W_q(input)
        k = self.W_k(input)
        v = self.W_v(input)

        C = f.unsqueeze(2) * C + i.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        h = o * torch.bmm(q.unsqueeze(1), C).squeeze(1)

        return h, C

class xLSTMBlock(nn.Module):
    """
    xLSTM block implementation.

    This block can use either sLSTM or mLSTM as its core, surrounded by
    normalization, activation, and projection layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state in LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, lstm_type="slstm"):
        super(xLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the xLSTM block.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        output = self.activation(lstm_output)
        output = self.norm(output)
        output = self.proj(output)
        output = self.dropout_layer(output + input_seq)  # Residual connection
        return output, hidden_state

class xLSTM(nn.Module):
    """
    xLSTM model implementation.

    This model uses a combination of sLSTM and mLSTM blocks in a residual architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Size of the token embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([
            xLSTMBlock(embedding_size, hidden_size, num_layers, dropout, lstm_type)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model.

        Args:
            input_seq (Tensor): Input sequence of token indices.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output logits and final hidden states.
        """
        embedded_seq = self.embedding(input_seq)

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])

        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states

class CTCModule(nn.Module):
    """CTC输出模块"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x: (B, seq_len, dim)
        return: (B, seq_len, num_classes) log_probs
        """
        output = self.linear(x)
        return F.log_softmax(output, dim=2)

class PatchEmbed(nn.Module):
    """图像分块嵌入"""
    def __init__(self, img_size=(32, 280), patch_size=(4, 4), in_chans=1, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x

class HybridVisionXLSTM_OCR(nn.Module):
    """
    混合模型：Vision-LSTM前端 + xLSTM序列精炼后端

    Args:
        num_classes: 字符类别数
        img_size: 输入图像尺寸 (H, W)
        patch_size: 分块大小 (H, W)
        embed_dim: 嵌入维度
        num_vision_blocks: Vision-LSTM块数量
        num_xlstm_blocks: 序列精炼xLSTM块数量
        xlstm_hidden_dim: xLSTM隐藏层维度
    """

    def __init__(self, num_classes: int = 37,
                 img_size=(32, 280), patch_size=(4, 4), in_chans=3,
                 embed_dim: int = 384,
                 num_vision_blocks: int = 3,
                 num_xlstm_blocks: int = 2,
                 xlstm_hidden_dim: int = 256):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        # ========== Stage 1: Vision-LSTM 前端 ==========
        # 分块嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_pos_embed(num_patches, embed_dim)

        # 交替mLSTM块（空间建模）
        self.vision_blocks = nn.ModuleList()
        for i in range(num_vision_blocks):
            # 使用文件中已定义的mLSTM类
            self.vision_blocks.append(mLSTM(embed_dim, embed_dim, num_layers=1, dropout=0.1))

        self.vision_norm = nn.LayerNorm(embed_dim)

        # ========== Stage 2: 序列转换层 ==========
        # 将空间特征转换为序列特征
        self.seq_proj = nn.Linear(embed_dim, xlstm_hidden_dim)

        # ========== Stage 3: xLSTM 序列精炼 ==========
        # 额外的xLSTM块（时序建模）
        self.xlstm_blocks = nn.ModuleList()
        for i in range(num_xlstm_blocks):
            # 使用文件中已定义的xLSTMBlock类
            self.xlstm_blocks.append(xLSTMBlock(xlstm_hidden_dim, xlstm_hidden_dim, num_layers=1, dropout=0.1, lstm_type="mlstm"))

        # ========== Stage 4: CTC输出 ==========
        self.ctc_module = CTCModule(xlstm_hidden_dim, num_classes)

    def count_parameters(self):
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_pos_embed(self, num_patches, embed_dim):
        """初始化2D位置编码"""
        # 创建一个简单的正弦位置编码
        # 生成位置索引
        positions = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
        # 生成维度索引
        dims = torch.arange(embed_dim, dtype=torch.float32).unsqueeze(0)
        # 计算位置编码因子
        factor = torch.exp(dims * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        # 计算位置编码
        pos_enc = positions * factor
        # 交替使用sin和cos
        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
        # 调整形状并复制到pos_embed
        self.pos_embed.data.copy_(pos_enc.unsqueeze(0))

    def _alternate_scan(self, x, block_idx):
        """交替扫描策略"""
        B, N, D = x.shape
        H, W = self.grid_size

        grid = x.reshape(B, H, W, D)
        if block_idx % 2 == 1:  # 偶数块反向扫描
            grid = torch.flip(grid, dims=[1, 2])
        return grid.reshape(B, N, D)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        return: (B, seq_len, num_classes)
        """
        B = x.size(0)

        # ========== Vision-LSTM 前端 ==========
        # 分块嵌入
        x = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # 交替mLSTM块处理空间信息
        for idx, block in enumerate(self.vision_blocks):
            x_scanned = self._alternate_scan(x, idx)
            # mLSTM的forward返回(output, hidden_state)，我们只需要output
            x_out, _ = block(x_scanned)
            x = x + x_out  # 残差连接

        x = self.vision_norm(x)  # (B, N, embed_dim)

        # ========== 序列转换 ==========
        # 按行聚合空间特征 → 序列
        H, W = self.grid_size
        x_seq = x.reshape(B, H, W, self.embed_dim)
        x_seq = x_seq.mean(dim=1)  # (B, W, embed_dim)

        # 投影到xLSTM维度
        x_seq = self.seq_proj(x_seq)  # (B, W, xlstm_hidden_dim)

        # ========== xLSTM 序列精炼 ==========
        for block in self.xlstm_blocks:
            # xLSTMBlock的forward返回(output, hidden_state)，我们只需要output
            x_seq, _ = block(x_seq)

        # ========== CTC输出 ==========
        return self.ctc_module(x_seq)  # (B, W, num_classes)

def test_hybrid_model_forward():
    """测试前向传播"""
    print("Testing HybridVisionXLSTM_OCR forward pass...")

    model = HybridVisionXLSTM_OCR(
        num_classes=20,
        img_size=(32, 280),
        patch_size=(4, 4),
        embed_dim=256,
        num_vision_blocks=2,
        num_xlstm_blocks=1,
        xlstm_hidden_dim=128
    )

    # 测试不同batch size
    for batch_size in [1, 4, 8]:
        dummy_input = torch.randn(batch_size, 3, 32, 280)

        with torch.no_grad():
            output = model(dummy_input)

        # 验证输出形状
        assert output.shape[0] == batch_size, f"Batch size mismatch for bs={batch_size}"
        assert output.shape[2] == 20, "Num classes mismatch"
        expected_seq_len = 280 // 4  # width // patch_size
        assert output.shape[1] == expected_seq_len, f"Seq len mismatch: {output.shape[1]} vs {expected_seq_len}"

        print(f"  ✓ Batch size {batch_size}: output shape {output.shape}")

    print("✓ Forward pass test passed\n")

def test_hybrid_model_gradient():
    """测试梯度流"""
    print("Testing gradient flow in hybrid model...")

    # 使用更小的模型配置，减少计算复杂度
    model = HybridVisionXLSTM_OCR(
        num_classes=15,
        num_vision_blocks=1,
        num_xlstm_blocks=1,
        embed_dim=128,
        xlstm_hidden_dim=128
    )

    dummy_input = torch.randn(2, 3, 32, 280)

    # 前向传播
    preds = model(dummy_input)  # (B, seq_len, num_classes)
    B, seq_len, num_classes = preds.shape

    # 使用简单的MSE损失来测试梯度流
    # 创建一个与preds形状相同的目标张量
    target = torch.randn_like(preds)

    # 使用MSE损失
    criterion = torch.nn.MSELoss()
    loss = criterion(preds, target)

    # 反向传播
    loss.backward()

    # 检查梯度是否存在
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            has_grad = True

    assert has_grad, "No gradients found"
    print("  ✓ Gradients propagated successfully")

    print("✓ Gradient flow test passed\n")

def test_hybrid_model_components():
    """测试各组件独立工作"""
    print("Testing hybrid model components...")

    model = HybridVisionXLSTM_OCR(
        embed_dim=192,
        num_vision_blocks=3,
        num_xlstm_blocks=2
    )

    # 统计各组件参数
    vision_params = sum(p.numel() for p in model.vision_blocks.parameters())
    xlstm_params = sum(p.numel() for p in model.xlstm_blocks.parameters())
    total_params = model.count_parameters()

    print(f"  Vision-LSTM params: {vision_params:,}")
    print(f"  xLSTM refiner params: {xlstm_params:,}")
    print(f"  Total params: {total_params:,}")

    # 验证组件可独立前向
    dummy_input = torch.randn(1, 3, 32, 280)

    # 只测试Vision-LSTM前端
    x = model.patch_embed(dummy_input) + model.pos_embed
    for idx, block in enumerate(model.vision_blocks):
        x_scanned = model._alternate_scan(x, idx)
        # mLSTM的forward返回(output, hidden_state)，我们只需要output
        x_out, _ = block(x_scanned)
        x = x + x_out
    assert x.shape[1] == model.patch_embed.num_patches, "Vision-LSTM output shape error"
    print("  ✓ Vision-LSTM frontend works")

    # 完整前向
    output = model(dummy_input)
    assert len(output.shape) == 3, "Final output shape error"
    print("  ✓ Full hybrid model works")

    print("✓ Component test passed\n")

if __name__ == "__main__":
    import numpy as np

    test_hybrid_model_forward()
    test_hybrid_model_gradient()
    test_hybrid_model_components()
    print("All hybrid model tests passed!")
