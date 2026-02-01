import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class EnhancedCTCLoss(nn.Module):
    """
    统一的增强CTC损失函数
    整合形近字混淆处理、尾部空白字符惩罚、字符级Focal Loss、自适应Margin机制、温度退火机制

    核心改进：
    1. 路径级形近字权重应用，而非简单的事后加权
    2. 自适应尾部空白惩罚，考虑序列长度和空白分布
    3. 高效的权重缓存和数值稳定计算
    4. 支持动态参数调整和梯度裁剪
    """

    def __init__(self,
                 vocab_size: int,
                 blank: int,
                 confuse_weight_dict: Optional[Dict[int, float]],
                 confuse_gamma: float = 1.0,           # 形近字权重缩放因子
                 confuse_temperature: float = 1.0,     # 形近字温度参数
                 eos_penalty: float = 0.1,             # 尾部空白惩罚系数
                 eos_penalty_decay: float = 0.9,       # 尾部惩罚衰减系数
                 eos_window_size: int = 3,             # 尾部窗口大小
                 eos_adaptive: bool = True,            # 是否使用自适应尾部惩罚
                 char_focal: int = 0,                  # Focal Loss类型：0-不启用，1-字符级，2-样本级
                 focal_gamma: float = 2.0,             # Focal Loss gamma参数
                 focal_scale: float = 1.0,             # Focal Loss缩放因子
                 adaptive_margin: bool = False,        # 是否启用自适应Margin
                 margin: float = 0.3,                  # 固定Margin值
                 margin_max: float = 0.5,              # 自适应Margin最大值
                 temperature_annealing: bool = True,  # 是否启用温度退火
                 gradient_clip: bool = True,           # 是否启用梯度裁剪
                 reduction: str = 'mean'):
        super().__init__()
        self.blank = blank
        self.confuse_gamma = confuse_gamma
        self.confuse_temperature = confuse_temperature
        self.eos_penalty = eos_penalty
        self.eos_penalty_decay = eos_penalty_decay
        self.eos_window_size = eos_window_size
        self.eos_adaptive = eos_adaptive
        self.char_focal = char_focal
        self.focal_gamma = focal_gamma
        self.focal_scale = focal_scale
        self.adaptive_margin = adaptive_margin
        self.margin = margin
        self.margin_max = margin_max
        self.temperature_annealing = temperature_annealing
        self.gradient_clip = gradient_clip
        self.reduction = reduction

        # 使用优化的权重表或自定义权重
        self.confuse_weight_dict = confuse_weight_dict

        # 初始化字符权重张量
        self.register_buffer('char_weights', torch.ones(vocab_size))
        self._init_char_weights_optimized(vocab_size)

        # 基础CTC损失
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)

        # 自适应Margin参数
        if self.adaptive_margin:
            self.margin_delta = nn.Parameter(torch.zeros(1))

        # 温度退火缩放因子 - 统一使用buffer管理
        self.register_buffer('focal_scale_buffer', torch.tensor(float(self.focal_scale)))

        # 预计算衰减权重 - 避免重复计算
        self.register_buffer('decay_weights_cache', self._precompute_decay_weights())

    def _precompute_decay_weights(self) -> torch.Tensor:
        """预计算衰减权重，避免在每次forward中重复计算"""
        weights = torch.pow(
            self.eos_penalty_decay,
            torch.arange(self.eos_window_size, dtype=torch.float32)
        ).flip(0)  # 反转，使末尾权重最大
        return weights

    def schedule(self, epoch: int, max_epoch: int):
        """
        温度退火：focal强度从0→1
        采用余弦退火策略
        """
        if self.temperature_annealing and max_epoch > 1:
            self.focal_scale_buffer.fill_(
                0.5 * (1 - math.cos(math.pi * epoch / max_epoch))
            )

    def _init_char_weights_optimized(self, vocab_size: int):
        """优化的字符权重初始化 - 使用向量化操作"""
        # 创建字符到权重的映射数组
        char_weights_array = torch.ones(vocab_size)

        for idx in range(vocab_size):
            weight = self.confuse_weight_dict.get(idx, 1.0)
            # 应用gamma缩放和温度参数
            char_weights_array[idx] = (weight ** self.confuse_gamma) ** (1.0 / self.confuse_temperature)

        self.char_weights.copy_(char_weights_array)

    def _apply_char_focal(self, log_probs: torch.Tensor,
                         input_lengths: torch.Tensor,
                         targets: torch.Tensor,
                         target_lengths: torch.Tensor,
                         raw_loss: torch.Tensor) -> torch.Tensor:
        """
        应用字符级Focal Loss
        Args:
            log_probs: [T, B, V] 已log_softmax的张量
            input_lengths: [B] 输入序列长度
            targets: 目标序列
            target_lengths: [B] 目标序列长度
            raw_loss: [B] 基础CTC损失
        Returns:
            应用focal权重后的损失
        """
        # 提前返回条件 - 避免不必要的计算
        if self.char_focal != 1 or self.focal_gamma == 0 or self.focal_scale_buffer.item() == 0:
            return raw_loss

        # 计算字符级focal权重
        T, B, V = log_probs.shape
        device = log_probs.device

        # 1. 提取非blank最大概率 - 屏蔽blank标签
        nb_logp = log_probs.clone()
        nb_logp[..., self.blank] = -float('inf')
        max_nb_logp, _ = nb_logp.max(dim=-1)  # [T, B]

        # 2. 构建有效帧mask - 使用广播机制
        mask = torch.arange(T, device=device).view(T, 1) < input_lengths.view(1, B)

        # 3. 计算字符级focal权重 - 数值稳定性优化
        pt = max_nb_logp.exp().clamp(min=1e-7, max=1-1e-7)
        focal_power = -self.focal_gamma * self.focal_scale_buffer.item()

        # 使用更稳定的数值计算 - 避免大指数运算
        weight = torch.exp(focal_power * torch.log1p(-pt))

        # 4. 样本级平均 - 只计算有效帧
        valid_frames = mask.sum(dim=0).clamp_min(1)
        focal_weights = (weight * mask).sum(dim=0) / valid_frames

        return raw_loss * focal_weights

    def _apply_adaptive_margin(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用自适应Margin
        Args:
            logits: [B, T, V] 未归一化的logits
        Returns:
            应用margin后的logits
        """
        # 提前返回条件 - 无需应用margin
        if not self.adaptive_margin and self.margin == 0:
            return logits

        # 计算margin值
        if self.adaptive_margin:
            # 自适应学习margin值 - 使用sigmoid限制范围
            delta = self.margin_delta.sigmoid() * self.margin_max
        else:
            # 固定margin值
            delta = self.margin

        # 只在margin值不为0时进行修改
        if delta == 0:
            return logits

        # 降低blank标签的置信度
        logits = logits.clone()
        logits[..., self.blank] -= delta

        return logits

    def _compute_path_weights_vectorized(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        计算CTC路径权重
        基于目标序列中的字符重要性为不同路径分配权重

        Args:
            targets: [B, L] 目标序列
            target_lengths: [B] 目标序列长度

        Returns:
            [B] 路径权重
        """
        B, L = targets.shape
        device = targets.device

        # 创建有效目标mask
        valid_mask = torch.arange(L, device=device).view(1, L) < target_lengths.view(B, 1)

        # 获取目标字符的权重 - 向量化操作
        char_weights = self.char_weights[targets]  # [B, L]

        # 使用调和平均数计算路径权重 - 向量化操作
        # 避免除零，添加小常数
        reciprocal_weights = torch.reciprocal(char_weights + 1e-8)

        # 计算每个样本的平均调和权重
        path_weights = torch.reciprocal(
            (reciprocal_weights * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
        )

        return path_weights

    def _compute_eos_penalty_vectorized(self,
                                       logits: torch.Tensor,
                                       input_lengths: torch.Tensor,
                                       targets: torch.Tensor,
                                       target_lengths: torch.Tensor) -> torch.Tensor:
        """
        计算尾部空白字符惩罚

        Args:
            logits: [B, T, V] 模型输出
            input_lengths: [B] 输入序列长度
            targets: [B, L] 目标序列
            target_lengths: [B] 目标序列长度

        Returns:
            尾部空白惩罚损失
        """
        B, T, V = logits.shape
        device = logits.device

        # 计算空白字符概率 [B, T]
        blank_probs = F.softmax(logits, dim=-1)[:, :, self.blank]

        # 自适应尾部窗口大小
        if self.eos_adaptive:
            # 根据目标序列长度调整窗口大小
            adaptive_window = torch.clamp(
                torch.round(target_lengths.float() * 0.3).long(),
                min=1,
                max=self.eos_window_size
            )
        else:
            adaptive_window = torch.full((B,), self.eos_window_size, device=device)

        # 计算每个样本的起始索引 [B]
        start_indices = torch.max(
            torch.zeros_like(input_lengths),
            input_lengths - adaptive_window
        )

        # 创建位置索引矩阵 [B, T]
        position_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # 向量化条件判断：start_indices <= position_indices < input_lengths [B, T]
        tail_mask = (position_indices >= start_indices.unsqueeze(1)) & \
                    (position_indices < input_lengths.unsqueeze(1))

        # 计算每个样本的实际窗口大小 [B]
        actual_window_sizes = tail_mask.sum(dim=1).clamp_min(1)
        max_window = adaptive_window.max().item()

        # 初始化动态衰减权重矩阵 [B, T]
        dynamic_decay_weights = torch.zeros(B, T, device=device)

        if max_window > 0 and tail_mask.any():
            # 创建衰减权重模板 [max_window]
            decay_template = torch.pow(
                self.eos_penalty_decay,
                torch.arange(max_window, dtype=torch.float32, device=device)
            ).flip(0)

            # 完全向量化创建衰减权重
            # 创建每个位置在窗口中的相对索引 [B, T]
            relative_indices = position_indices - start_indices.unsqueeze(1)
            # 创建有效索引mask：relative_indices < adaptive_window [B, T]
            valid_relative_indices_mask = relative_indices < adaptive_window.unsqueeze(1)
            # 合并所有mask：tail_mask + valid_relative_indices_mask [B, T]
            combined_mask = tail_mask & valid_relative_indices_mask
            # 使用scatter_向量化填充衰减权重
            # 将relative_indices展平为1D，然后用decay_template填充
            valid_indices = combined_mask.nonzero(as_tuple=True)
            if valid_indices[0].numel() > 0:
                dynamic_decay_weights[valid_indices] = decay_template[relative_indices[valid_indices]]

        # 向量化计算加权概率 [B]
        weighted_probs = (blank_probs * dynamic_decay_weights * tail_mask.float()).sum(dim=1)
        # 计算衰减权重和 [B]
        decay_weights_sum = (dynamic_decay_weights * tail_mask.float()).sum(dim=1).clamp_min(1e-8)
        # 计算每个样本的平均加权概率 [B]
        weighted_tail_probs = weighted_probs / decay_weights_sum
        # 计算总惩罚
        eos_penalty_loss = (weighted_tail_probs * self.eos_penalty).mean()

        return eos_penalty_loss

    def _compute_loss_components(self, logits, targets, input_lengths, target_lengths, training):
        """
        计算损失组件，供forward和get_loss_components共享

        Args:
            logits: [B, T, V] 未归一化的logits
            targets: [B, L] 目标序列
            input_lengths: [B] 输入序列长度
            target_lengths: [B] 目标序列长度
            training: 是否为训练模式

        Returns:
            包含各损失组件的字典
        """
        B, T, V = logits.shape
        device = logits.device

        # 应用自适应Margin - 降低blank标签置信度
        logits_margin = self._apply_adaptive_margin(logits) if (training and (self.adaptive_margin or self.margin != 0)) else logits

        # 计算基础CTC损失（每个样本）
        log_probs = F.log_softmax(logits_margin, dim=-1).permute(1, 0, 2)  # [T, B, V]
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)  # [B]

        # 将无穷大损失替换为0，避免梯度爆炸
        ctc_losses = torch.where(torch.isinf(ctc_losses), torch.tensor(0.0, device=device), ctc_losses)

        # 保存原始CTC损失用于组件分析
        base_ctc_losses = ctc_losses.clone()

        # 应用字符级Focal Loss - 聚焦最难识别的字符
        char_focal_losses = ctc_losses.clone()
        if training and self.char_focal == 1 and self.focal_gamma != 0 and self.focal_scale_buffer.item() != 0:
            char_focal_losses = self._apply_char_focal(log_probs, input_lengths, targets, target_lengths, ctc_losses)

        # 应用样本级Focal Loss - 聚焦困难样本
        sample_focal_losses = char_focal_losses.clone()
        if training and self.char_focal == 2 and self.focal_gamma != 0 and self.focal_scale_buffer.item() != 0:
            # 使用更稳定的数值计算
            pt = (-char_focal_losses.clamp(max=10.0)).exp()  # 限制最大值避免数值溢出
            focal_weights = (1 - pt.clamp(min=1e-7, max=1-1e-7)) ** (self.focal_gamma * self.focal_scale_buffer.item())
            sample_focal_losses = focal_weights * char_focal_losses

        # 计算路径权重（基于目标字符的重要性）
        path_weights = self._compute_path_weights_vectorized(targets, target_lengths)

        # 应用路径权重到CTC损失
        weighted_ctc_losses = sample_focal_losses * path_weights

        # 计算尾部空白字符惩罚
        eos_penalty_loss = self._compute_eos_penalty_vectorized(logits_margin, input_lengths, targets, target_lengths)

        return {
            'logits_margin': logits_margin,
            'log_probs': log_probs,
            'base_ctc_losses': base_ctc_losses,
            'char_focal_losses': char_focal_losses,
            'sample_focal_losses': sample_focal_losses,
            'path_weights': path_weights,
            'weighted_ctc_losses': weighted_ctc_losses,
            'eos_penalty_loss': eos_penalty_loss
        }

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算增强CTC损失
        新增：字符级Focal Loss、自适应Margin、温度退火

        Args:
            logits: [B, T, V] 未归一化的logits
            targets: [B, L] 目标序列
            input_lengths: [B] 输入序列长度
            target_lengths: [B] 目标序列长度

        Returns:
            增强CTC损失
        """
        # 计算所有损失组件
        components = self._compute_loss_components(
            logits, targets, input_lengths, target_lengths, self.training
        )

        # 组合总损失
        total_loss = components['weighted_ctc_losses'].mean() + components['eos_penalty_loss']

        # 梯度裁剪（防止梯度爆炸）
        if self.training and self.gradient_clip:
            total_loss = torch.clamp(total_loss, min=-1000.0, max=1000.0)

        return total_loss

    def get_loss_components(self,
                           logits: torch.Tensor,
                           targets: torch.Tensor,
                           input_lengths: torch.Tensor,
                           target_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取损失的各个组成部分，用于分析和调试
        新增：字符级Focal、自适应Margin、温度退火组件分解

        Returns:
            包含各个损失组件的字典
        """
        # 计算所有损失组件
        components = self._compute_loss_components(
            logits, targets, input_lengths, target_lengths, self.training
        )

        # 计算各组件的数值
        loss_components = {
            'base_ctc_loss': components['base_ctc_losses'].mean(),
            'char_focal_loss': components['char_focal_losses'].mean(),
            'sample_focal_loss': components['sample_focal_losses'].mean(),
            'path_weights': components['path_weights'].mean(),
            'weighted_ctc_loss': components['weighted_ctc_losses'].mean(),
            'eos_penalty_loss': components['eos_penalty_loss'],
            'total_loss': components['weighted_ctc_losses'].mean() + components['eos_penalty_loss']
        }

        # 如果启用了自适应Margin，添加margin值
        if self.adaptive_margin:
            current_margin = self.margin_delta.sigmoid().item() * self.margin_max
            loss_components['adaptive_margin'] = current_margin

        # 如果启用了温度退火或Focal Loss（字符级或样本级），添加focal缩放因子
        if self.temperature_annealing or self.char_focal != 0:
            loss_components['focal_scale'] = self.focal_scale_buffer.item()

        return loss_components

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    def __init__(
        self,
        temperature: float = 4.0,
        alpha_feat: float = 0.5,
        alpha_logit: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_feat = alpha_feat
        self.alpha_logit = alpha_logit

    def schedule(self, epoch: int, max_epoch: int):
        """
        学习率衰减调度
        """

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失

        Args:
            teacher_features: 教师模型特征 [B, L, D]
            student_features: 学生模型特征 [B, L, D]
            teacher_logits: 教师模型logits [B, L, V]
            student_logits: 学生模型logits [B, L, V]

        Returns:
            包含各种损失的字典
        """

        # 使用交叉注意力机制对齐学生和教师的序列长度，以替代线性插值
        # 教师序列作为Query，学生序列作为Key和Value
        # 这会生成一个与教师序列长度相同、但内容来自学生的对齐表示

        # 1. 对齐特征层 (Align Features)
        # teacher_features: [B, L_teacher, D]
        # student_features: [B, L_student, D]
        # aligned_student_features: [B, L_teacher, D]
        # pylint: disable=not-callable
        aligned_student_features = F.scaled_dot_product_attention(
            query=teacher_features,
            key=student_features,
            value=student_features
        )
        # 特征层面对齐损失 (MSE)
        feature_loss = F.mse_loss(aligned_student_features, teacher_features.detach())

        # 2. 对齐 Logits 层 (Align Logits)
        # teacher_logits: [B, L_teacher, V]
        # student_logits: [B, L_student, V]
        # aligned_student_logits: [B, L_teacher, V]
        # pylint: disable=not-callable
        aligned_student_logits = F.scaled_dot_product_attention(
            query=teacher_logits,
            key=student_logits,
            value=student_logits
        )
        # 更新 student_logits 以用于后续的 KL 散度计算
        student_logits = aligned_student_logits

        # 学生梯度裁剪防止 Inf
        # 1. 先关 inf 源（可选，但建议保留）
        # teacher_logits = torch.clamp(teacher_logits, -35.0, 35.0)
        # student_logits = torch.clamp(student_logits, -35.0, 35.0)

        # 输出层面对齐损失 (KL散度)
        T = self.temperature

        # 软化概率分布
        teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        kl_elem = teacher_soft.exp() * (teacher_soft - student_soft)          # [B, T, V]

        # mask（有就加，没有就跳过）
        if mask is not None:
            kl_elem = kl_elem.masked_fill(~mask.unsqueeze(-1), 0.0)

        # 手动 mean + ε
        token_count = max(kl_elem.numel() if mask is None else mask.sum().item() * kl_elem.size(-1), 1)
        kl_loss = kl_elem.sum() / token_count * (T * T)

        # KL散度损失
        # kl_loss = F.kl_div(student_soft, teacher_soft.detach(), reduction='batchmean') * (T * T)
        # kl_loss2 = self.debug_kl(teacher_logits, student_logits, T, None)

        # 总蒸馏损失
        total_loss = self.alpha_feat * feature_loss + self.alpha_logit * kl_loss

        return {
            'feature_loss': feature_loss,
            'kl_loss': kl_loss,
            'total_distill_loss': total_loss
        }

    def debug_kl(self, teacher_logits, student_logits, T=8.0, mask=None):
        print("========== KL debug ==========")
        print(f"T={T}")
        print(f"teacher {teacher_logits.shape}  min={teacher_logits.min().item():.4f}  max={teacher_logits.max().item():.4f}")
        print(f"student {student_logits.shape}  min={student_logits.min().item():.4f}  max={student_logits.max().item():.4f}")
        if mask is not None:
            print(f"mask sum = {mask.sum().item()}  shape = {mask.shape}")

        t_prob = F.log_softmax(teacher_logits / T, dim=-1)
        s_prob = F.log_softmax(student_logits / T, dim=-1)
        kl = t_prob.exp() * (t_prob - s_prob)
        if mask is not None:
            kl = kl.masked_fill(~mask.unsqueeze(-1), 0.0)
        token_count = kl.numel() if mask is None else mask.sum() * kl.size(-1)
        token_count = max(token_count, 1)
        kl_loss = kl.sum() / token_count * (T * T)
        print(f"kl_loss = {kl_loss.item()}")
        print("==============================")
        return kl_loss

class QuantizationAwareLoss(torch.nn.Module):
    """量化感知损失函数"""

    def __init__(self, quantization_manager):
        super().__init__()
        self.quantization_manager = quantization_manager

    def schedule(self, epoch: int, max_epoch: int):
        """
        学习率衰减调度
        """

    def forward(self, logits):
        # 如果有原始模型，计算量化损失
        if (self.quantization_manager and self.quantization_manager.config['enabled'] and
            hasattr(self.quantization_manager, 'original_model') and
            'input_features' in logits):

            # 获取原始模型的输出作为教师信号
            with torch.no_grad():
                # 确保传递有效的输入张量，而不是None
                input_features = logits.get('input_features', None)
                if input_features is not None:
                    original_logits = self.quantization_manager.original_model(input_features)

                    # 计算量化损失，处理logits中没有total_loss的情况
                    # 尝试获取ctc_logits作为默认张量的来源
                    default_tensor = torch.zeros(1, device=logits['ctc_logits'].device) if 'ctc_logits' in logits else torch.zeros(1)
                    quantization_loss = self.quantization_manager.get_quantization_loss(
                        logits.get('features', default_tensor),
                        original_logits.get('features', default_tensor)
                    )

                    return quantization_loss

        return torch.tensor(0.0, dtype=torch.float32)

class RecognitionLoss(nn.Module):
    """文本识别总损失"""
    def __init__(
        self,
        max_epoch: int,
        vocab_size: int,
        ignore_index: int,
        ctc_weight: float,
        ar_weight: float,
        distill_weight: float,
        distill_start_epoch: int,
        confuse_weight_dict: Optional[Dict[str, float]],
        quantization_manager = None,
    ):
        super().__init__()
        self.cur_epoch = 0
        self.max_epoch = max_epoch
        self.ignore_index = ignore_index
        self.ctc_weight = ctc_weight
        self.ar_weight = ar_weight
        self.distill_weight = distill_weight
        self.distill_start_epoch = distill_start_epoch

        # CTC损失
        # self.ctc_loss = nn.CTCLoss(blank=ignore_index, reduction='mean', zero_infinity=True)
        self.ctc_loss = EnhancedCTCLoss(vocab_size=vocab_size, blank=ignore_index, confuse_weight_dict=confuse_weight_dict)

        # 交叉熵损失（用于自回归）
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.05)

        # 蒸馏损失
        self.distill_loss = DistillationLoss()

        self.quantization_loss = QuantizationAwareLoss(quantization_manager)

    def schedule(self, epoch: int):
        """
        学习率衰减调度
        """
        self.cur_epoch = epoch
        self.ctc_loss.schedule(epoch, self.max_epoch)
        self.distill_loss.schedule(epoch, self.max_epoch)
        self.quantization_loss.schedule(epoch, self.max_epoch)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        targets_ce: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            predictions: 模型预测结果
            targets: 目标文本 [B, L]
            target_lengths: 目标长度 [B]

        Returns:
            包含各种损失的字典
        """
        total_loss = 0
        loss_dict = {}

        # CTC损失
        if 'ctc_logits' in predictions:
            ctc_logits = predictions['ctc_logits']  # [B, T, V]
            # log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]

            input_lengths = torch.full((ctc_logits.size(0),), ctc_logits.size(1),
                                     dtype=torch.long, device=ctc_logits.device)

            ctc_loss = self.ctc_loss(ctc_logits, targets, input_lengths, target_lengths)
            total_loss += self.ctc_weight * ctc_loss
            loss_dict['ctc_loss'] = ctc_loss
        else:
            loss_dict['ctc_loss'] = torch.tensor(0.0, dtype=torch.float32)

        # 自回归损失
        if 'ar_logits' in predictions:
            ar_logits = predictions['ar_logits']  # [B, L, V]
            B, L, V = ar_logits.shape
            ar_logits = ar_logits[:, :targets_ce.size(1), :]       # 截断到 targets_ce 长度
            ar_loss = self.ce_loss(
                ar_logits[:, :-1, :].reshape(-1, V),
                targets_ce[:, 1:].reshape(-1)
            )
            total_loss += self.ar_weight * ar_loss
            loss_dict['ar_loss'] = ar_loss
        else:
            loss_dict['ar_loss'] = torch.tensor(0.0, dtype=torch.float32)

        # 蒸馏损失（仅在训练时，且达到指定epoch后计算）
        if self.training and self.cur_epoch >= self.distill_start_epoch and all(k in predictions for k in ['ctc_logits', 'ar_logits', 'aligned_features']):
            # 这里简化处理，实际应该使用教师模型的输出
            distill_losses = self.distill_loss(
                teacher_features=predictions['ar_features'],
                student_features=predictions['aligned_features'],
                teacher_logits=predictions['ar_logits'],
                student_logits=predictions['ctc_logits'],
                mask=mask
            )
            distill_total = distill_losses['total_distill_loss']
            total_loss += self.distill_weight * distill_total
            loss_dict.update(distill_losses)
        else:
            loss_dict['kl_loss'] = torch.tensor(0.0, dtype=torch.float32)
            loss_dict['feature_loss'] = torch.tensor(0.0, dtype=torch.float32)
            loss_dict['total_distill_loss'] = torch.tensor(0.0, dtype=torch.float32)

        # 量化感知损失
        if self.quantization_loss is not None:
            quantization_total = self.quantization_loss(predictions)
            total_loss += quantization_total
            loss_dict['quantization_loss'] = quantization_total
        else:
            loss_dict['quantization_loss'] = torch.tensor(0.0, dtype=torch.float32)

        loss_dict['total_loss'] = total_loss

        return loss_dict
