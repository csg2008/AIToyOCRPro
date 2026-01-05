import os
import re
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use('Agg')

class VisualizationConfig:
    """可视化配置类，将相关参数分组管理

    该类用于集中管理所有可视化相关的参数，提高代码可读性和可维护性。
    不同的可视化类型可能使用不同的参数子集。

    Attributes:
        visualization_type (str): 可视化类型，可选值包括 'colorbar', 'heatmap', 'enhanced', 'error_analysis'
        top_k (int): 显示概率最高的前k个字符，用于Top-K预测可视化
        prob_threshold (float): 概率阈值，低于此值的预测结果可能不会显示
        show_attention (bool): 是否显示注意力权重（仅适用于AR模型）
        show_confidence (bool): 是否显示置信度指示器
        beam_size (int): 束搜索的束大小（仅适用于AR模型的束搜索可视化）
        max_length (int): 最大显示长度，避免可视化结果过长
        patch_size (int): ViT模型的patch大小，用于计算stride
        show_error_types (bool): 是否显示错误类型分析
        error_analysis_mode (str): 错误分析模式，可选值包括 'character', 'sequence', 'confidence'
        robustness_analysis (bool): 是否启用鲁棒性分析
        noise_types (list): 噪声类型列表
        noise_levels (list): 噪声强度级别列表
        robustness_save_dir (str): 鲁棒性分析结果保存目录
        training_analysis (bool): 是否启用训练过程分析
        training_history_file (str): 训练历史文件路径
        epochs_to_compare (list): 要对比的epoch列表
        training_save_dir (str): 训练分析结果保存目录
        augmentation_analysis (bool): 是否启用数据增强效果分析
        augmentation_types (list): 增强类型列表
        augmentation_levels (list): 增强强度级别列表
        augmentation_save_dir (str): 增强分析结果保存目录
        deep_attention_analysis (bool): 是否启用深度注意力分析
        attention_layers_to_visualize (list): 要可视化的层索引列表
        attention_heads_to_visualize (list): 要可视化的头索引列表
        attention_save_dir (str): 注意力分析结果保存目录
        multi_model_comparison (bool): 是否启多模型对比分析
        models_to_compare (list): 要对比的模型列表
        model_names (list): 模型名称列表
        comparison_save_dir (str): 对比分析结果保存目录
    """
    def __init__(self,
                 visualization_type: str = 'colorbar',
                 top_k: int = 5,
                 prob_threshold: float = 0.15,
                 show_attention: bool = True,
                 show_confidence: bool = True,
                 beam_size: int = 3,
                 max_length: int = 70,
                 patch_size: int = 16,
                 show_error_types: bool = True,
                 error_analysis_mode: str = 'character',
                 robustness_analysis: bool = False,
                 noise_types: list = None,
                 noise_levels: list = None,
                 robustness_save_dir: str = './robustness_analysis',
                 training_analysis: bool = False,
                 training_history_file: str = './training_history.pkl',
                 epochs_to_compare: list = None,
                 training_save_dir: str = './training_analysis',
                 augmentation_analysis: bool = False,
                 augmentation_types: list = None,
                 augmentation_levels: list = None,
                 augmentation_save_dir: str = './augmentation_analysis',
                 deep_attention_analysis: bool = False,
                 attention_layers_to_visualize: list = None,
                 attention_heads_to_visualize: list = None,
                 attention_save_dir: str = './attention_analysis',
                 multi_model_comparison: bool = False,
                 models_to_compare: list = None,
                 model_names: list = None,
                 comparison_save_dir: str = './multi_model_comparison'):
        self.visualization_type = visualization_type
        self.top_k = top_k
        self.prob_threshold = prob_threshold
        self.show_attention = show_attention
        self.show_confidence = show_confidence
        self.beam_size = beam_size
        self.max_length = max_length
        self.patch_size = patch_size
        self.show_error_types = show_error_types
        self.error_analysis_mode = error_analysis_mode

        self.robustness_analysis = robustness_analysis
        self.noise_types = noise_types or ['gaussian', 'blur', 'rotation', 'occlusion', 'contrast']
        self.noise_levels = noise_levels or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.robustness_save_dir = robustness_save_dir

        self.training_analysis = training_analysis
        self.training_history_file = training_history_file
        self.epochs_to_compare = epochs_to_compare or [0, 10, 20, 30, 40, 50]
        self.training_save_dir = training_save_dir

        self.augmentation_analysis = augmentation_analysis
        self.augmentation_types = augmentation_types or ['rotation', 'scale', 'shear', 'elastic', 'color_jitter']
        self.augmentation_levels = augmentation_levels or [0.1, 0.2, 0.3]
        self.augmentation_save_dir = augmentation_save_dir

        self.deep_attention_analysis = deep_attention_analysis
        self.attention_layers_to_visualize = attention_layers_to_visualize or [0, 6, 12]
        self.attention_heads_to_visualize = attention_heads_to_visualize or [0, 4, 8]
        self.attention_save_dir = attention_save_dir

        self.multi_model_comparison = multi_model_comparison
        self.models_to_compare = models_to_compare or []
        self.model_names = model_names or []
        self.comparison_save_dir = comparison_save_dir

class ModelOutput:
    """模型输出结果类，统一不同模型的输出格式

    该类用于封装不同模型结构的输出结果，提供统一的访问接口，
    简化后续可视化处理逻辑，支持标准模型结构和ViT等特殊模型结构。

    Attributes:
        logits (torch.Tensor): 模型原始输出，形状为 [B, T, V]
        prob (torch.Tensor): 经过softmax归一化后的概率，形状为 [T, V]
        pred_id (torch.Tensor): 每个时间步的最高概率字符ID，形状为 [T]
        prob_values (torch.Tensor): 每个时间步的最高概率值，形状为 [T]
        attention_weights (torch.Tensor): 注意力权重，形状为 [B, L, T]（仅适用于AR模型）
        sequence_features (torch.Tensor): 序列特征，形状为 [B, T, D]（仅适用于标准模型结构）
        pred_text (str): 预测的文本字符串
        gt_text (str): 真实文本字符串
    """
    def __init__(self,
                 logits: torch.Tensor = None,
                 prob: torch.Tensor = None,
                 pred_id: torch.Tensor = None,
                 prob_values: torch.Tensor = None,
                 attention_weights: torch.Tensor = None,
                 sequence_features: torch.Tensor = None,
                 pred_text: str = '',
                 gt_text: str = ''):
        self.logits = logits
        self.prob = prob
        self.pred_id = pred_id
        self.prob_values = prob_values
        self.attention_weights = attention_weights
        self.sequence_features = sequence_features
        self.pred_text = pred_text
        self.gt_text = gt_text

class OCRErrorAnalyzer:
    """OCR错误分析器，用于分析和分类OCR识别错误"""

    @staticmethod
    def analyze_errors(pred_text: str, gt_text: str) -> dict:
        """分析OCR识别错误类型

        Args:
            pred_text (str): 预测的文本字符串
            gt_text (str): 真实文本字符串

        Returns:
            dict: 错误分析结果，包含各种错误类型的统计信息
        """
        errors = {
            'total_chars': len(gt_text),
            'correct_chars': 0,
            'substitution_errors': 0,
            'deletion_errors': 0,
            'insertion_errors': 0,
            'swap_errors': 0,
            'space_errors': 0,
            'case_errors': 0,
            'punctuation_errors': 0,
            'error_positions': [],
            'error_details': []
        }

        # 基本字符错误分析
        pred_idx = 0
        gt_idx = 0
        n_pred = len(pred_text)
        n_gt = len(gt_text)

        while pred_idx < n_pred and gt_idx < n_gt:
            if pred_text[pred_idx] == gt_text[gt_idx]:
                # 正确识别
                errors['correct_chars'] += 1
                pred_idx += 1
                gt_idx += 1
            else:
                # 错误识别
                error_type = 'substitution'
                errors['substitution_errors'] += 1
                errors['error_positions'].append(gt_idx)
                errors['error_details'].append({
                    'position': gt_idx,
                    'gt_char': gt_text[gt_idx],
                    'pred_char': pred_text[pred_idx],
                    'error_type': error_type
                })
                pred_idx += 1
                gt_idx += 1

        # 处理剩余的字符
        while gt_idx < n_gt:
            # 删除错误
            errors['deletion_errors'] += 1
            errors['error_positions'].append(gt_idx)
            errors['error_details'].append({
                'position': gt_idx,
                'gt_char': gt_text[gt_idx],
                'pred_char': '',
                'error_type': 'deletion'
            })
            gt_idx += 1

        while pred_idx < n_pred:
            # 插入错误
            errors['insertion_errors'] += 1
            errors['error_details'].append({
                'position': gt_idx,
                'gt_char': '',
                'pred_char': pred_text[pred_idx],
                'error_type': 'insertion'
            })
            pred_idx += 1

        # 高级错误分析
        errors['case_errors'] = sum(1 for gt, pred in zip(gt_text, pred_text)
                                  if len(gt) > 0 and len(pred) > 0 and gt != pred and gt.lower() == pred.lower())

        errors['space_errors'] = sum(1 for gt, pred in zip(gt_text, pred_text)
                                   if (gt == ' ' and pred != ' ') or (gt != ' ' and pred == ' '))

        punctuation = ",;:!?()[]{}<>"
        errors['punctuation_errors'] = sum(1 for gt, pred in zip(gt_text, pred_text)
                                         if (gt in punctuation and pred not in punctuation) or
                                         (gt not in punctuation and pred in punctuation))

        # 计算交换错误
        for i in range(len(gt_text) - 1):
            if i + 1 < len(pred_text) and gt_text[i] == pred_text[i+1] and gt_text[i+1] == pred_text[i]:
                errors['swap_errors'] += 1

        return errors

    @staticmethod
    def visualize_errors(pred_text: str, gt_text: str, save_path: str, errors: dict):
        """可视化OCR识别错误

        Args:
            pred_text (str): 预测的文本字符串
            gt_text (str): 真实文本字符串
            save_path (str): 输出图像保存路径
            errors (dict): 错误分析结果
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(gt_text)*0.5), 6))

        # 1. 预测结果与真实结果对比
        escaped_gt_text = re.escape(gt_text)
        escaped_pred_text = re.escape(pred_text)
        ax1.set_title(f'Prediction vs Ground Truth\nGT: {escaped_gt_text}\nPred: {escaped_pred_text}')

        max_len = max(len(gt_text), len(pred_text))
        ax1.set_xlim(0, max_len)
        ax1.set_ylim(-1.5, 1.5)

        # 绘制GT文本
        for i, ch in enumerate(gt_text):
            ax1.text(i + 0.5, 1, ch, ha='center', va='center', fontsize=12, fontweight='bold')

        # 绘制预测文本
        for i, ch in enumerate(pred_text):
            is_correct = i < len(gt_text) and ch == gt_text[i]
            color = 'green' if is_correct else 'red'
            ax1.text(i + 0.5, -1, ch, ha='center', va='center', fontsize=12, fontweight='bold', color=color)

        # 绘制连接线
        for i in range(min(len(gt_text), len(pred_text))):
            color = 'green' if gt_text[i] == pred_text[i] else 'red'
            ax1.plot([i+0.5, i+0.5], [0.5, -0.5], color=color, linestyle='--', alpha=0.5)

        # 2. 错误类型统计
        error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']
        error_counts = [errors[etype + '_errors'] for etype in error_types]

        bars = ax2.bar(error_types, error_counts, color=['red', 'orange', 'yellow', 'purple', 'blue', 'cyan', 'magenta'])
        ax2.set_title('Error Type Distribution')
        ax2.set_xlabel('Error Type')
        ax2.set_ylabel('Count')

        # 在柱状图上显示具体数值
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Error analysis saved → {save_path}')

@torch.no_grad()
def get_model_output(model: torch.nn.Module, img_tensor: torch.Tensor, train_mode: str, gt_text: str = '') -> ModelOutput:
    """统一获取模型输出结果

    该函数根据训练模式和模型结构，统一获取模型的输出结果，并封装为ModelOutput对象。
    支持CTC、hybrid和AR三种训练模式，支持标准模型结构和ViT等特殊模型结构。

    可视化原理：
    1. 对于标准模型结构（backbone -> neck -> decoder），按顺序调用各组件获取输出
    2. 对于ViT等特殊模型，直接调用模型获取logits
    3. 统一计算概率、预测ID和最大概率值
    4. 封装为ModelOutput对象返回，便于后续可视化处理

    Args:
        model (torch.nn.Module): OCR模型实例
        img_tensor (torch.Tensor): 输入图像张量，形状为 [B, 3, H, W]
        train_mode (str): 训练模式，可选值包括 'ctc', 'hybrid', 'ar'
        gt_text (str): 真实文本字符串，用于错误分析

    Returns:
        ModelOutput: 封装后的模型输出结果
    """
    model.eval()

    if train_mode == 'ctc' or train_mode == 'hybrid':
        # CTC模式
        if hasattr(model, 'backbone') and hasattr(model, 'neck') and hasattr(model, 'decoder'):
            features = model.backbone(img_tensor)
            if features.dim() == 4:
                # 标准CNN/SVTRv2模型结构：backbone -> neck -> decoder
                sequence_features = model.neck(features)
                logits = model.decoder.ctc_decoder(sequence_features)
            else:
                # ViT/VIPTRNetV2模型：backbone输出序列特征[ B, T, D]
                sequence_features = model.neck(features)
                logits = model.decoder.ctc_decoder(sequence_features)
                sequence_features = sequence_features  # 保留用于后续判断
        else:
            # 纯ViT模型直接输出logits
            outputs = model(img_tensor)
            logits = outputs['ctc_logits']
            sequence_features = None

        prob = logits.softmax(-1)[0].cpu()
        pred_id = prob.argmax(-1)
        prob_values = prob.max(-1).values

        return ModelOutput(
            logits=logits,
            prob=prob,
            pred_id=pred_id,
            prob_values=prob_values,
            sequence_features=sequence_features,
            gt_text=gt_text
        )
    else:
        # AR模式
        features = model.backbone(img_tensor)
        neck_features = model.neck(features)
        ar_logits, _ = model.decoder.ar_decoder.generate(neck_features)

        prob = torch.softmax(ar_logits, -1)[0].cpu()
        pred_id = ar_logits.argmax(-1)[0].cpu()
        prob_values = prob.max(-1).values

        # 获取注意力权重（如果模型支持）
        attention_weights = None
        if hasattr(model.decoder.ar_decoder, 'get_attention_weights'):
            attention_weights = model.decoder.ar_decoder.get_attention_weights()

        return ModelOutput(
            logits=ar_logits,
            prob=prob,
            pred_id=pred_id,
            prob_values=prob_values,
            attention_weights=attention_weights,
            gt_text=gt_text
        )

@torch.no_grad()
def convert_pred_to_text(pred_id: torch.Tensor, idx2char: dict, blank_id: int = None, eos_id: int = None, sos_id: int = None) -> str:
    """将预测的字符ID转换为文本字符串

    Args:
        pred_id (torch.Tensor): 预测的字符ID序列
        idx2char (dict): 字符ID到字符的映射字典
        blank_id (int, optional): 空白符ID，默认为None
        eos_id (int, optional): 结束符ID，默认为None
        sos_id (int, optional): 起始符ID，默认为None

    Returns:
        str: 转换后的文本字符串
    """
    text = []
    prev_char = None

    for char_id in pred_id.tolist():
        if char_id in [eos_id, sos_id]:
            continue
        if char_id == blank_id:
            prev_char = None
            continue
        if char_id != prev_char:
            text.append(idx2char[char_id])
            prev_char = char_id

    return ''.join(text)

def create_color_map(char_ids: set, blank_id: int = None) -> dict:
    """创建字符到颜色的映射

    该函数为不同字符创建唯一的颜色映射，用于可视化中的字符区分。
    空白符使用特殊颜色，其他字符从预定义的颜色映射表中分配颜色。

    可视化原理：
    1. 使用matplotlib的tab20颜色映射表，提供20种不同颜色
    2. 空白符使用浅灰色，便于区分
    3. 其他字符按顺序分配颜色，循环使用颜色表

    Args:
        char_ids (set): 需要分配颜色的字符ID集合
        blank_id (int, optional): 空白符ID，默认为None

    Returns:
        dict: 字符ID到颜色的映射字典
    """
    cmap = plt.colormaps['tab20'].colors
    color_map = {}

    # 特殊字符颜色
    if blank_id is not None:
        color_map[blank_id] = (0.8, 0.8, 0.8)  # 空白符用浅灰色

    # 其他字符分配颜色
    for i, char_id in enumerate(char_ids):
        if char_id not in color_map:
            color_map[char_id] = cmap[i % len(cmap)]

    return color_map

def get_adjusted_color(base_color: tuple, confidence: float, prob_threshold: float) -> tuple:
    """根据置信度调整颜色亮度

    该函数根据预测的置信度，调整基础颜色的亮度，用于可视化中直观表示预测置信度。
    低置信度预测使用较暗的颜色，高置信度预测使用原始颜色。

    可视化原理：
    1. 低置信度（< prob_threshold）：颜色亮度降低50%
    2. 中置信度（prob_threshold <= 置信度 < 0.8）：颜色亮度降低20%
    3. 高置信度（>= 0.8）：使用原始颜色
    4. 通过颜色亮度直观表示预测的可靠性

    Args:
        base_color (tuple): 基础颜色，格式为 (R, G, B)
        confidence (float): 预测置信度，范围 [0, 1]
        prob_threshold (float): 概率阈值，用于区分低置信度和中置信度

    Returns:
        tuple: 调整后的颜色，格式为 (R, G, B)
    """
    if confidence < prob_threshold:
        return tuple(c * 0.5 for c in base_color)
    elif confidence < 0.8:
        return tuple(c * 0.8 for c in base_color)
    else:
        return base_color

@torch.no_grad()
def visualize_ar_alignment(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                          config: VisualizationConfig, model_config: dict):
    """自回归解码器对齐可视化

    可视化自回归(AR)解码器的对齐过程，包括每一步的预测结果、注意力权重和生成路径。
    与CTC不同，AR是逐步生成的，因此可以显示生成过程中的动态变化。

    可视化原理：
    1. 主对齐图：显示每一步的预测结果，颜色表示预测字符，亮度表示置信度
    2. 置信度曲线：显示每一步预测的置信度变化
    3. 注意力权重热力图：显示解码器对输入序列的注意力分布
    4. GT对比：显示预测结果与真实文本的对比

    Args:
        model (torch.nn.Module): AR解码器模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典，包含blank_id、eos_id、idx2char、sos_id等
    """
    max_length = config.max_length
    show_attention = config.show_attention
    blank_id = model_config.get('blank_id')
    eos_id = model_config.get('eos_id')
    idx2char = model_config.get('idx2char')
    sos_id = model_config.get('sos_id')

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ar', gt_text=gt_text)
    pred_id = model_output.pred_id
    prob_values = model_output.prob_values
    attention_weights = model_output.attention_weights

    # 转换预测ID为文本
    pred_text = convert_pred_to_text(pred_id, idx2char, blank_id, eos_id, sos_id)
    model_output.pred_text = pred_text

    # 限制显示长度
    L = min(len(pred_id), max_length)
    pred_id = pred_id[:L]
    prob_values = prob_values[:L]

    # 创建可视化
    fig_height = 8 if show_attention and attention_weights is not None else 6
    fig = plt.figure(figsize=(max(12, L*0.5), fig_height))

    if show_attention and attention_weights is not None:
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_conf = fig.add_subplot(gs[1])
        ax_attn = fig.add_subplot(gs[2])
        ax_gt = fig.add_subplot(gs[3])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_conf = fig.add_subplot(gs[1])
        ax_gt = fig.add_subplot(gs[2])

    # 创建颜色映射
    cmap = plt.colormaps['tab20'].colors
    color_map = {sos_id: (0.8, 0.8, 0.8), eos_id: (0.6, 0.6, 0.6)}
    if blank_id is not None:
        color_map[blank_id] = (0.9, 0.9, 0.9)

    # 1. 主对齐图 - 显示每一步的预测
    for i in range(L):
        char_id = pred_id[i].item()
        confidence = prob_values[i].item()

        # 根据置信度调整颜色
        if char_id in color_map:
            base_color = color_map[char_id]
        else:
            base_color = cmap[char_id % len(cmap)]

        color = get_adjusted_color(base_color, confidence, 0.5)

        # 绘制预测块
        rect = matplotlib.patches.Rectangle(
            (i, 0), 1, 1,
            facecolor=color,
            edgecolor='black' if confidence > 0.8 else 'gray',
            linewidth=1)
        ax_main.add_patch(rect)

        # 显示字符（高置信度才显示）
        if confidence > 0.6 and char_id not in [sos_id, eos_id, blank_id]:
            ax_main.text(i+0.5, 0.5, idx2char[char_id],
                        ha='center', va='center', fontsize=10,
                        color='white' if sum(color) < 1.5 else 'black',
                        fontweight='bold' if confidence > 0.9 else 'normal')

        # 置信度指示器
        ax_main.add_patch(matplotlib.patches.Rectangle(
            (i, -0.15), 1, 0.1,
            facecolor='green', alpha=confidence,
            edgecolor='none'))

    ax_main.set_xlim(0, L)
    ax_main.set_ylim(-0.2, 1.2)
    escaped_gt_text = re.escape(gt_text)
    escaped_pred_text = re.escape(pred_text)
    ax_main.set_title(f'AR Decoder Alignment - GT: {escaped_gt_text} | Pred: {escaped_pred_text} | Length: {L}', usetex=False)
    ax_main.set_ylabel('Generation Steps')
    ax_main.grid(True, alpha=0.3)

    # 2. 置信度曲线
    ax_conf.plot(range(L), prob_values.numpy(), 'b-', linewidth=2, label='Confidence')
    ax_conf.fill_between(range(L), prob_values.numpy(), alpha=0.3)
    ax_conf.set_xlim(0, L-1)
    ax_conf.set_ylim(0, 1.1)
    ax_conf.set_ylabel('Confidence')
    ax_conf.set_title('Prediction Confidence over Generation Steps')
    ax_conf.grid(True, alpha=0.3)
    ax_conf.legend()

    # 3. 注意力权重可视化（如果有）
    if show_attention and attention_weights is not None:
        attn_weights = attention_weights[0].cpu()  # [L, T]
        im = ax_attn.imshow(attn_weights.T, aspect='auto', cmap='Blues',
                           extent=[0, L-1, 0, attn_weights.shape[1]])
        ax_attn.set_xlabel('Generation Steps')
        ax_attn.set_ylabel('Input Positions')
        ax_attn.set_title('Attention Weights')
        plt.colorbar(im, ax=ax_attn, label='Attention Weight')

    # 4. GT对比
    ax_gt.barh(0, L, color='lightgray', alpha=0.3)
    for i, ch in enumerate(gt_text[:L]):
        pred_char = idx2char[pred_id[i].item()]
        is_correct = pred_char == ch
        color = 'green' if is_correct else 'red'
        ax_gt.barh(0, 1, left=i, height=0.8, color=color, alpha=0.7)
        ax_gt.text(i + 0.5, 0, ch, ha='center', va='center',
                  fontsize=10, fontweight='bold')

    ax_gt.set_xlim(0, L)
    ax_gt.set_ylim(-0.6, 0.6)
    ax_gt.set_yticks([0])
    ax_gt.set_yticklabels(['GT'])
    ax_gt.set_xlabel('Generation Steps')
    ax_gt.set_title('Ground Truth Comparison')
    ax_gt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'AR alignment saved → {save_path}')

    # 生成错误分析可视化
    if config.show_error_types and gt_text != pred_text:
        error_analysis_save_path = save_path.replace('.png', '_error_analysis.png')
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)
        OCRErrorAnalyzer.visualize_errors(pred_text, gt_text, error_analysis_save_path, errors)

@torch.no_grad()
def visualize_ar_beam_search(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                            config: VisualizationConfig, model_config: dict):
    """束搜索AR对齐可视化

    可视化束搜索(beam search)的对齐过程，显示多个候选路径的生成结果，
    便于对比不同束搜索路径的差异和质量。

    可视化原理：
    1. 为每个束大小生成一个序列
    2. 显示每个束路径的预测结果
    3. 颜色表示预测置信度，绿色表示高置信度，红色表示低置信度
    4. 底部显示真实文本作为对比

    Args:
        model (torch.nn.Module): AR解码器模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典
    """
    max_length = config.max_length
    beam_size = config.beam_size
    blank_id = model_config.get('blank_id')
    eos_id = model_config.get('eos_id')
    idx2char = model_config.get('idx2char')
    sos_id = model_config.get('sos_id')

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ar', gt_text=gt_text)
    logits = model_output.logits

    # 获取log_probs
    log_probs = torch.log_softmax(logits[0], dim=-1)  # [L, V]

    beam_paths = []
    beam_confidences = []
    beam_texts = []

    # 生成多个束路径
    for k in range(1, beam_size + 1):
        beam_seq = model.decoder.ar_decoder.beam_search(log_probs, k=k, max_len=max_length)

        # 将序列转换为tensor并获取置信度
        beam_tensor = torch.tensor(beam_seq, device=logits.device).unsqueeze(0)  # [1, L]
        pred_probs = torch.softmax(logits, -1)[0].cpu()  # [L, V]

        # 为生成的序列获取对应的置信度
        confidence = []
        for i, token_id in enumerate(beam_seq):
            if i < pred_probs.shape[0]:
                confidence.append(pred_probs[i, token_id])
            else:
                confidence.append(torch.tensor(0.0))

        confidence = torch.stack(confidence)

        L = min(len(beam_seq), max_length)
        beam_paths.append(torch.tensor(beam_seq[:L]))
        beam_confidences.append(confidence[:L])

        # 转换为文本
        beam_text = convert_pred_to_text(beam_paths[-1], idx2char, blank_id, eos_id, sos_id)
        beam_texts.append(beam_text)

    # 创建可视化
    fig, axes = plt.subplots(beam_size + 2, 1, figsize=(max(12, max_length*0.5),
                                                     2*(beam_size+2)), sharex=True)

    # 绘制每个beam路径
    for beam_idx, (path, conf, text) in enumerate(zip(beam_paths, beam_confidences, beam_texts)):
        ax = axes[beam_idx]

        L = len(path)
        for i in range(L):
            char_id = path[i].item()
            confidence = conf[i].item()

            if char_id not in [sos_id, eos_id, blank_id]:
                color = plt.colormaps['RdYlGn'](confidence)  # 颜色表示置信度
                ax.barh(0, 1, left=i, height=0.8, color=color, alpha=0.8)

                if confidence > 0.7:
                    ax.text(i + 0.5, 0, idx2char[char_id],
                           ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(0, L)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([0])
        ax.set_yticklabels([f'Beam {beam_idx+1}'])
        escaped_text = re.escape(text)
        ax.set_title(f'Beam Search Path {beam_idx+1} - Text: {escaped_text}')
        ax.grid(True, alpha=0.3)

    # 预测文本对比
    ax_compare = axes[-2]
    ax_compare.barh(0, max_length, color='lightgray', alpha=0.3)
    ax_compare.set_title('Beam Text Comparison')
    ax_compare.set_yticks(range(beam_size))
    ax_compare.set_yticklabels([f'Beam {i+1}' for i in range(beam_size)])

    for i, text in enumerate(beam_texts):
        is_correct = text == gt_text
        color = 'green' if is_correct else 'red'
        ax_compare.text(0, i, f'{text} (Score: {sum(beam_confidences[i]).item():.2f})',
                       fontsize=10, fontweight='bold', color=color)

    ax_compare.set_xlim(0, max_length)
    ax_compare.grid(True, alpha=0.3)

    # GT对比
    ax_gt = axes[-1]
    for i, ch in enumerate(gt_text[:max_length]):
        ax_gt.barh(0, 1, left=i, height=0.8, color='lightgray', alpha=0.5)
        ax_gt.text(i + 0.5, 0, ch, ha='center', va='center',
                  fontsize=10, fontweight='bold')

    ax_gt.set_xlim(0, max_length)
    ax_gt.set_ylim(-0.6, 0.6)
    ax_gt.set_yticks([0])
    ax_gt.set_yticklabels(['GT'])
    ax_gt.set_xlabel('Generation Steps')
    escaped_gt_text = re.escape(gt_text)
    ax_gt.set_title(f'Ground Truth - Text: {escaped_gt_text}')
    ax_gt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'AR beam search alignment saved → {save_path}')

@torch.no_grad()
def visualize_ctc_colorbar(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                           config: VisualizationConfig, model_config: dict):
    """CTC颜色条对齐可视化

    创建类似CNN风格的颜色条对齐可视化，适用于各种CTC模型，包括CNN、ViT和SVTRv2等。
    每个时间步用一个彩色条表示，颜色对应预测的字符类别，字符显示在条上。

    可视化原理：
    1. 颜色条表示每个时间步的预测结果，颜色对应预测字符
    2. 颜色亮度表示预测置信度
    3. 底部标记真实文本字符的位置
    4. 直观显示预测结果与真实文本的对齐关系

    Args:
        model (torch.nn.Module): CTC模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典
    """
    blank_id = model_config.get('blank_id')
    idx2char = model_config.get('idx2char')

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ctc', gt_text=gt_text)
    prob = model_output.prob
    pred_id = model_output.pred_id
    prob_values = model_output.prob_values

    # 转换预测ID为文本
    pred_text = convert_pred_to_text(pred_id, idx2char, blank_id)
    model_output.pred_text = pred_text

    T = prob.shape[0]

    # 计算stride
    backbone_out = model.backbone(img_tensor)
    if backbone_out.dim() == 4:
        # 标准模型结构（CNN/SVTRv2）：输出 [B, C, H, W]
        _, _, feat_H, feat_W = backbone_out.shape
        img_W = img_tensor.shape[-1]
        stride = img_W // feat_W
    else:
        # ViT/VIPTRNetV2模型：输出 [B, T, D]，使用patch_size作为stride
        stride = config.patch_size

    # 创建颜色映射
    common_chars = set(pred_id.tolist()[:100])
    color_map = create_color_map(common_chars, blank_id)

    # 绘制对齐条
    fig, ax = plt.subplots(figsize=(max(12, T*0.3), 4))

    for t in range(T):
        char_id = int(pred_id[t])
        confidence = prob_values[t].item()

        base_color = color_map.get(char_id, (0.5, 0.5, 0.5))
        color = get_adjusted_color(base_color, confidence, 0.3)

        rect = matplotlib.patches.Rectangle(
            (t, 0), 1, 1,
            facecolor=color,
            edgecolor='black' if confidence > 0.8 else 'gray',
            linewidth=0.5)
        ax.add_patch(rect)

        # 显示字符（高置信度且非特殊字符）
        if confidence > 0.5 and char_id != blank_id:
            ax.text(t+0.5, 0.5, idx2char[char_id],
                   ha='center', va='center', fontsize=8,
                   color='white' if sum(color) < 1.5 else 'black',
                   fontweight='bold' if confidence > 0.9 else 'normal')

    # GT坐标映射
    for idx, ch in enumerate(gt_text):
        center = (idx + 0.5) * stride
        if center < T * stride:
            t_idx = int(center // stride)
            if t_idx < T:
                ax.text(center/stride, -0.3, ch, ha='center', va='top',
                       color='red', fontsize=9, fontweight='bold')

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 1.5)
    escaped_gt_text = re.escape(gt_text)
    escaped_pred_text = re.escape(pred_text)
    ax.set_title(f'CTC Alignment - GT: {escaped_gt_text} | Pred: {escaped_pred_text} | stride={stride}', usetex=False)
    ax.set_xlabel('Sequence Time Steps')
    ax.set_ylabel('Predictions')
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'CTC colorbar alignment saved → {save_path}')

    # 生成错误分析可视化
    if config.show_error_types and gt_text != pred_text:
        error_analysis_save_path = save_path.replace('.png', '_error_analysis.png')
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)
        OCRErrorAnalyzer.visualize_errors(pred_text, gt_text, error_analysis_save_path, errors)

@torch.no_grad()
def visualize_ctc_heatmap(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                          config: VisualizationConfig, model_config: dict):
    """CTC热力图对齐可视化

    创建热力图风格的CTC对齐可视化，显示每个时间步上所有字符类别的概率分布，
    便于分析模型在每个时间步的预测偏好和不确定性。

    可视化原理：
    1. 上半部分：概率热力图，显示每个时间步对所有字符类别的预测概率
    2. 下半部分：最高概率曲线，显示每个时间步的最大概率值
    3. 在热力图上标记真实文本字符的位置
    4. 通过颜色深浅表示概率大小，直观显示模型的预测分布

    Args:
        model (torch.nn.Module): CTC模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典
    """
    idx2char = model_config.get('idx2char')

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ctc', gt_text=gt_text)
    prob = model_output.prob
    pred_id = model_output.pred_id

    # 转换预测ID为文本
    pred_text = convert_pred_to_text(pred_id, idx2char, model_config.get('blank_id'))
    model_output.pred_text = pred_text

    T = prob.shape[0]

    # 计算stride
    backbone_out = model.backbone(img_tensor)
    if backbone_out.dim() == 4:
        # 标准模型结构（CNN/SVTRv2）：输出 [B, C, H, W]
        _, _, feat_H, feat_W = backbone_out.shape
        img_W = img_tensor.shape[-1]
        stride = img_W // feat_W
    else:
        # ViT/VIPTRNetV2模型：输出 [B, T, D]，使用patch_size作为stride
        stride = config.patch_size

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, T*0.2), 6),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 1. 概率热力图
    im = ax1.imshow(prob.T, aspect='auto', cmap='hot', vmin=0, vmax=1)

    # 设置y轴标签
    max_char_id = len(idx2char)
    step = max(1, max_char_id // 20)
    ax1.set_yticks(range(0, max_char_id, step))
    ax1.set_yticklabels([idx2char[i] for i in range(0, max_char_id, step)])

    ax1.set_xlabel('Time Steps (T)')
    ax1.set_ylabel('Character Classes')
    escaped_gt_text = re.escape(gt_text)
    escaped_pred_text = re.escape(pred_text)
    ax1.set_title(f'CTC Probability Heatmap - GT: {escaped_gt_text} | Pred: {escaped_pred_text}')

    # 添加颜色条
    plt.colorbar(im, ax=ax1, label='Probability')

    # 在热力图上标记GT位置
    for idx, ch in enumerate(gt_text):
        center = (idx + 0.5) * stride
        t_idx = int(center // stride)
        if t_idx < T:
            # 找到字符对应的id
            char_id = None
            for i, c in idx2char.items():
                if c == ch:
                    char_id = i
                    break

            if char_id is not None and char_id < max_char_id:
                # 标记GT字符位置
                rect = matplotlib.patches.Rectangle(
                    (t_idx-0.5, char_id-0.5), 1, 1,
                    fill=False, edgecolor='cyan', linewidth=2)
                ax1.add_patch(rect)

    # 2. 最高概率曲线
    max_probs = prob.max(-1).values
    ax2.plot(range(T), max_probs.numpy(), 'b-', linewidth=2, label='Max Probability')
    ax2.fill_between(range(T), max_probs.numpy(), alpha=0.3)
    ax2.set_xlim(0, T-1)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Max Prob')
    ax2.set_title('Maximum Probability per Time Step')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'CTC heatmap alignment saved → {save_path}')

@torch.no_grad()
def visualize_ctc_enhanced(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                           config: VisualizationConfig, model_config: dict):
    """增强版CTC对齐可视化

    创建包含多个子图的综合可视化，展示CTC对齐的全面信息，包括主对齐图、Top-K预测、
    置信度曲线、字符分布统计和GT对齐质量评估。

    可视化原理：
    1. 主对齐图：颜色条风格，显示最高预测字符和置信度
    2. Top-K预测热图：显示每个时间步的前3个最可能的预测
    3. 置信度曲线：显示每个时间步的最高概率值
    4. 字符分布统计：显示预测字符的出现频率
    5. GT对齐质量评估：显示本地窗口内的对齐质量分数
    6. 综合展示模型的预测性能和对齐特性

    Args:
        model (torch.nn.Module): CTC模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典
    """
    blank_id = model_config.get('blank_id')
    idx2char = model_config.get('idx2char')
    top_k = config.top_k
    prob_threshold = config.prob_threshold

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ctc', gt_text=gt_text)
    prob = model_output.prob
    pred_id = model_output.pred_id
    prob_values = model_output.prob_values

    # 转换预测ID为文本
    pred_text = convert_pred_to_text(pred_id, idx2char, blank_id)
    model_output.pred_text = pred_text

    T = prob.shape[0]

    # 计算stride
    backbone_out = model.backbone(img_tensor)
    if backbone_out.dim() == 4:
        # 标准模型结构（CNN/SVTRv2）：输出 [B, C, H, W]
        _, _, feat_H, feat_W = backbone_out.shape
        img_W = img_tensor.shape[-1]
        stride = img_W // feat_W
    else:
        # ViT/VIPTRNetV2模型：输出 [B, T, D]，使用patch_size作为stride
        stride = config.patch_size

    # 获取Top-K预测
    top_k_probs, top_k_ids = torch.topk(prob, k=min(top_k, prob.shape[-1]), dim=-1)

    # 创建可视化
    fig = plt.figure(figsize=(max(14, T*0.4), 10))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.4)

    # 1. 主对齐图 - 颜色条风格
    ax_main = fig.add_subplot(gs[0])

    # 创建颜色映射
    common_chars = set(pred_id.tolist()[:200])
    color_map = create_color_map(common_chars, blank_id)

    for t in range(T):
        char_id = int(pred_id[t])
        confidence = prob_values[t].item()

        # 背景色表示最高预测
        if char_id == blank_id:
            color = (0.9, 0.9, 0.9)  # 空白符用浅灰
        else:
            color = color_map.get(char_id, plt.colormaps['tab20'].colors[char_id % 20])

        # 根据置信度调整透明度
        alpha = min(confidence + 0.3, 1.0)
        rect = matplotlib.patches.Rectangle(
            (t, 0), 1, 1, facecolor=color, alpha=alpha,
            edgecolor='black' if confidence > 0.8 else 'gray', linewidth=0.5)
        ax_main.add_patch(rect)

        # 显示高置信度字符
        if confidence > prob_threshold and char_id != blank_id:
            ax_main.text(t+0.5, 0.5, idx2char[char_id],
                        ha='center', va='center', fontsize=9,
                        color='white' if sum(color[:3]) < 1.5 else 'black',
                        fontweight='bold')

    # GT映射
    for idx, ch in enumerate(gt_text):
        center = (idx + 0.5) * stride
        t_idx = int(center // stride)
        if t_idx < T:
            pred_char = idx2char[pred_id[t_idx].item()]
            is_correct = pred_char == ch
            color = 'green' if is_correct else 'red'
            ax_main.text(center/stride, -0.25, ch, ha='center', va='top',
                        color=color, fontsize=10, fontweight='bold')

    ax_main.set_xlim(0, T)
    ax_main.set_ylim(-0.4, 1.2)
    escaped_gt_text = re.escape(gt_text)
    escaped_pred_text = re.escape(pred_text)
    ax_main.set_title(f'Enhanced CTC Alignment - GT: {escaped_gt_text} | Pred: {escaped_pred_text} | stride={stride}',
                     usetex=False)
    ax_main.set_ylabel('Main Prediction')
    ax_main.grid(True, alpha=0.3)

    # 2. Top-K预测热图
    ax_topk = fig.add_subplot(gs[1])

    # 采样显示，避免过密
    step = max(1, T // 30)
    sampled_positions = range(0, T, step)

    for i, t in enumerate(sampled_positions):
        for k in range(min(3, top_k_ids.shape[1])):  # 只显示前3个
            char_id = top_k_ids[t][k].item()
            prob_val = top_k_probs[t][k].item()

            if prob_val > 0.1:  # 只显示有意义的预测
                y_pos = k
                x_pos = i
                color = plt.colormaps['RdYlGn'](prob_val)
                ax_topk.barh(y_pos, 1, left=x_pos, height=0.8,
                            color=color, alpha=0.8)

                if prob_val > 0.3:
                    ax_topk.text(x_pos + 0.5, y_pos, idx2char[char_id],
                                ha='center', va='center', fontsize=7)

    ax_topk.set_xlim(0, len(sampled_positions))
    ax_topk.set_ylim(-0.5, 2.5)
    ax_topk.set_yticks(range(3))
    ax_topk.set_yticklabels(['1st', '2nd', '3rd'])
    ax_topk.set_xticks(range(0, len(sampled_positions), max(1, len(sampled_positions)//10)))
    ax_topk.set_xticklabels([f'{t}' for t in sampled_positions[::max(1, len(sampled_positions)//10)]])
    ax_topk.set_xlabel('Sampled Time Steps')
    ax_topk.set_title('Top-3 Predictions (Sampled)')
    ax_topk.grid(True, alpha=0.3)

    # 3. 置信度曲线
    ax_conf = fig.add_subplot(gs[2])
    ax_conf.plot(range(T), prob_values.numpy(), 'b-', linewidth=2, label='Max Confidence')
    ax_conf.fill_between(range(T), prob_values.numpy(), alpha=0.3)
    ax_conf.axhline(y=prob_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold={prob_threshold}')
    ax_conf.set_xlim(0, T-1)
    ax_conf.set_ylim(0, 1.1)
    ax_conf.set_ylabel('Confidence')
    ax_conf.set_title('Prediction Confidence')
    ax_conf.grid(True, alpha=0.3)
    ax_conf.legend()

    # 4. 字符分布统计
    ax_dist = fig.add_subplot(gs[3])

    # 统计每个字符的出现次数
    char_counts = {}
    for t in range(T):
        char_id = pred_id[t].item()
        if char_id != blank_id:
            char_counts[char_id] = char_counts.get(char_id, 0) + 1

    if char_counts:
        chars = list(char_counts.keys())[:20]  # 只显示前20个
        counts = [char_counts[c] for c in chars]
        char_labels = [idx2char[c] for c in chars]

        bars = ax_dist.bar(range(len(chars)), counts,
                          color=[plt.colormaps['RdYlGn'](i/len(chars)) for i in range(len(chars))])
        ax_dist.set_xticks(range(len(chars)))
        ax_dist.set_xticklabels(char_labels, rotation=45)
        ax_dist.set_ylabel('Count')
        ax_dist.set_title('Character Frequency Distribution')
        ax_dist.grid(True, alpha=0.3)

    # 5. GT对齐质量评估
    ax_quality = fig.add_subplot(gs[4])

    # 计算对齐质量
    alignment_scores = []
    window_size = max(3, T // 10)

    for i in range(T - window_size + 1):
        window_start = i
        window_end = i + window_size

        # 找到对应的GT字符
        gt_start = int(window_start * stride)
        gt_end = int(window_end * stride)

        window_gt = gt_text[gt_start:gt_end]
        window_pred = [idx2char[pred_id[t].item()] for t in range(window_start, window_end)
                      if pred_id[t].item() != blank_id]

        # 简单的对齐评分
        pred_str = ''.join(window_pred)
        score = len(set(window_gt) & set(pred_str)) / max(len(window_gt), 1)
        alignment_scores.append(score)

    if alignment_scores:
        ax_quality.plot(range(len(alignment_scores)), alignment_scores, 'g-', linewidth=2)
        ax_quality.fill_between(range(len(alignment_scores)), alignment_scores, alpha=0.3, color='green')
        ax_quality.set_xlabel('Window Position')
        ax_quality.set_ylabel('Alignment Score')
        ax_quality.set_title(f'Local Alignment Quality (window={window_size})')
        ax_quality.grid(True, alpha=0.3)
        ax_quality.set_ylim(0, 1.1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Enhanced CTC alignment saved → {save_path}')

    # 生成错误分析可视化
    if config.show_error_types and gt_text != pred_text:
        error_analysis_save_path = save_path.replace('.png', '_error_analysis.png')
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)
        OCRErrorAnalyzer.visualize_errors(pred_text, gt_text, error_analysis_save_path, errors)

@torch.no_grad()
def visualize_ctc_compact(model: torch.nn.Module, img_tensor: torch.Tensor, gt_text: str, save_path: str,
                          config: VisualizationConfig, model_config: dict):
    """紧凑版CTC对齐可视化

    创建超紧凑的对齐可视化，专为超大字库设计，只显示最关键的信息，
    包括高置信度的预测字符、置信度热力条和真实文本对齐标记。

    可视化原理：
    1. 第一行：只显示高置信度的预测字符，减少视觉干扰
    2. 第二行：置信度热力条，通过颜色渐变显示置信度变化
    3. 第三行：GT位置标记，显示真实文本与预测的对齐关系
    4. 紧凑设计，适合在有限空间内展示关键信息

    Args:
        model (torch.nn.Module): CTC模型
        img_tensor (torch.Tensor): 输入图像张量，形状为 [1, 3, H, W]
        gt_text (str): 真实文本字符串
        save_path (str): 输出图像保存路径
        config (VisualizationConfig): 可视化配置对象
        model_config (dict): 模型配置字典
    """
    blank_id = model_config.get('blank_id')
    idx2char = model_config.get('idx2char')

    # 获取模型输出
    model_output = get_model_output(model, img_tensor, train_mode='ctc', gt_text=gt_text)
    prob = model_output.prob
    pred_id = model_output.pred_id
    prob_values = model_output.prob_values

    # 转换预测ID为文本
    pred_text = convert_pred_to_text(pred_id, idx2char, blank_id)
    model_output.pred_text = pred_text

    T = prob.shape[0]
    stride = config.patch_size

    # 创建紧凑可视化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(10, T*0.2), 4),
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    # 1. 预测字符（只显示高置信度）
    for t in range(T):
        char_id = pred_id[t].item()
        confidence = prob_values[t].item()

        if confidence > 0.7 and char_id != blank_id:  # 高置信度才显示
            color = plt.colormaps['RdYlGn'](confidence)  # 置信度颜色映射
            ax1.barh(0, 1, left=t, height=0.8, color=color, alpha=0.8)
            ax1.text(t + 0.5, 0, idx2char[char_id], ha='center', va='center',
                    fontsize=8, fontweight='bold')

    ax1.set_xlim(0, T)
    ax1.set_ylim(-0.6, 0.6)
    escaped_gt_text = re.escape(gt_text)
    escaped_pred_text = re.escape(pred_text)
    ax1.set_title(f'CTC Alignment - GT: {escaped_gt_text} | Pred: {escaped_pred_text}')
    ax1.set_yticks([])

    # 2. 置信度热力条
    ax2.imshow(prob_values.unsqueeze(0), aspect='auto', cmap='RdYlGn', extent=[0, T, 0, 1])
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_label('Confidence')

    # 3. GT位置标记
    for idx, ch in enumerate(gt_text):
        t_pos = idx * stride
        t_idx = int(t_pos // stride)
        if t_idx < T:
            pred_char = idx2char[pred_id[t_idx].item()]
            is_correct = pred_char == ch
            color = 'green' if is_correct else 'red'
            ax3.barh(0, 1, left=t_idx, height=0.8, color=color, alpha=0.6)
            ax3.text(t_idx + 0.5, 0, ch, ha='center', va='center', fontsize=8)

    ax3.set_xlim(0, T)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Compact CTC alignment saved → {save_path}')

    # 生成错误分析可视化
    if config.show_error_types and gt_text != pred_text:
        error_analysis_save_path = save_path.replace('.png', '_error_analysis.png')
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)
        OCRErrorAnalyzer.visualize_errors(pred_text, gt_text, error_analysis_save_path, errors)

@torch.no_grad()
def debug_ar_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int, output_dir: str,
                      model_config: dict, config: VisualizationConfig):
    """调试自回归(AR)解码器对齐可视化

    从数据加载器中获取一个批次的图像数据，使用AR模型生成预测结果，
    并调用相应的可视化函数生成AR对齐图。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): AR解码器模型
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        output_dir (str): 输出目录路径
        model_config (dict): 模型配置字典
        config (VisualizationConfig): 可视化配置对象
    """
    plt.rcParams['text.usetex'] = False

    # 获取数据
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)

    # 生成GT文本
    gt_txt = ''
    for i in batch['labels'][0].tolist():
        if i not in [model_config.get('blank_id'), model_config.get('sos_id'), model_config.get('eos_id')]:
            gt_txt += model_config.get('idx2char')[i]

    # 生成对齐图
    visualize_ar_alignment(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/ar_align_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    # 束搜索对齐（可选）
    if epoch % 5 == 0:  # 每5个epoch生成一次束搜索图
        visualize_ar_beam_search(
            model=model,
            img_tensor=img_tensor,
            gt_text=gt_txt,
            save_path=f'{output_dir}/visualizations/ar_beam_ep{epoch}.png',
            config=config,
            model_config=model_config
        )

@torch.no_grad()
def debug_svtrv2_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int,
                          output_dir: str, model_config: dict, config: VisualizationConfig):
    """调试SVTRv2 CTC对齐可视化

    从数据加载器中获取一个批次的图像数据，使用SVTRv2模型生成预测结果，
    并调用相应的可视化函数生成三种风格的SVTRv2对齐图。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): SVTRv2 CTC模型
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        output_dir (str): 输出目录路径
        model_config (dict): 模型配置字典
        config (VisualizationConfig): 可视化配置对象
    """
    plt.rcParams['text.usetex'] = False

    # 获取数据
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)

    # 生成GT文本
    gt_txt = ''
    for i in batch['labels'][0].tolist():
        if i != model_config.get('blank_id'):
            gt_txt += model_config.get('idx2char')[i]

    # 生成多种类型的对齐图
    visualize_ctc_colorbar(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/svtrv2_colorbar_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_heatmap(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/svtrv2_heatmap_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_enhanced(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/svtrv2_enhanced_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

@torch.no_grad()
def debug_vit_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int,
                       output_dir: str, model_config: dict, config: VisualizationConfig):
    """调试ViT CTC对齐可视化

    从数据加载器中获取一个批次的图像数据，使用ViT模型生成预测结果，
    并调用相应的可视化函数生成三种风格的ViT对齐图。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): ViT CTC模型
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        output_dir (str): 输出目录路径
        model_config (dict): 模型配置字典
        config (VisualizationConfig): 可视化配置对象
    """
    plt.rcParams['text.usetex'] = False

    # 获取数据
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)

    # 生成GT文本
    gt_txt = ''
    for i in batch['labels'][0].tolist():
        if i != model_config.get('blank_id'):
            gt_txt += model_config.get('idx2char')[i]

    # 生成多种类型的对齐图
    visualize_ctc_colorbar(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/vit_colorbar_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_heatmap(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/vit_heatmap_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_enhanced(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/vit_enhanced_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_compact(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/vit_compact_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

@torch.no_grad()
def debug_viptrv2_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int,
                          output_dir: str, model_config: dict, config: VisualizationConfig):
    """调试VIPTRNetV2模型的对齐可视化

    VIPTRNetV2使用OSRA（One-Shot Relative Attention）机制，具有独特的Vision Transformer架构特点。
    该函数生成专门针对VIPTRNetV2的可视化，包括OSRA注意力、Patch Embedding和DWConv2d特性。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): VIPTRNetV2模型实例
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        output_dir (str): 输出目录路径
        model_config (dict): 模型配置字典
        config (VisualizationConfig): 可视化配置对象

    可视化特性：
        1. OSRA相对位置编码可视化
        2. Patch Embedding注意力权重展示
        3. DWConv2d深度可分离特性分析
        4. 与标准ViT对比展示差异
    """
    plt.rcParams['text.usetex'] = False

    # 获取数据
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)

    # 生成GT文本
    gt_txt = ''
    for i in batch['labels'][0].tolist():
        if i != model_config.get('blank_id'):
            gt_txt += model_config.get('idx2char')[i]

    # 创建输出目录
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)

    # 1. 生成基础CTC对齐图（复用ViT可视化）
    visualize_ctc_colorbar(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/viptrv2_colorbar_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_heatmap(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/viptrv2_heatmap_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_enhanced(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/viptrv2_enhanced_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    visualize_ctc_compact(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/viptrv2_compact_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

    # 2. 生成VIPTRNetV2特性分析图（新增）
    try:
        # 尝试获取VIPTRNetV2的内部特征
        if hasattr(model, 'layers'):
            # 2.1 OSRA相对位置编码可视化
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 2.1.1 相对位置编码对比
            ax1 = axes[0, 0]
            if hasattr(model, 'patch_embed'):
                patch_embed = model.patch_embed
                if hasattr(patch_embed, 'pos_embed'):
                    pos_embed = patch_embed.pos_embed
                    if pos_embed is not None:
                        # 可视化相对位置编码
                        rel_pos = pos_embed.weight.detach().cpu().numpy()
                        if len(rel_pos.shape) == 2:
                            # 展示相对位置编码的分布
                            im = ax1.imshow(rel_pos, aspect='auto', cmap='viridis')
                            ax1.set_title('VIPTRNetV2 Relative Position Encoding')
                            ax1.set_xlabel('Position Index')
                            ax1.set_ylabel('Embedding Dimension')
                            plt.colorbar(im, ax=ax1, label='Value')
                        else:
                            ax1.text(0.5, 0.5, 'Position encoding not available',
                                   ha='center', va='center', fontsize=12)

            # 2.1.2 DWConv2d深度可分离特性
            ax2 = axes[0, 1]
            if hasattr(model, 'layers'):
                # 查找第一个DWConv2d层
                for layer in model.layers:
                    if hasattr(layer, 'local_conv'):
                        dw_conv = layer.local_conv
                        if hasattr(dw_conv, 'conv'):
                            # 可视化深度可分离卷积核
                            conv_weights = dw_conv.conv.weight.detach().cpu().numpy()
                            if len(conv_weights.shape) == 4:
                                # 展示4个深度可分离通道
                                for i in range(4):
                                    ax2.plot(conv_weights[i, 0], label=f'Channel {i}', linewidth=2)
                                ax2.set_title('VIPTRNetV2 DWConv2d Depthwise Separation')
                                ax2.set_xlabel('Spatial Position')
                                ax2.set_ylabel('Weight Value')
                                ax2.legend()
                                ax2.grid(True, alpha=0.3)
                        break
                    break

            # 2.2 注意力权重对比（如果有）
            ax3 = axes[1, 0]
            ax3.text(0.5, 0.5, 'VIPTRNetV2 OSRA Attention Analysis',
                   ha='center', va='center', fontsize=14, fontweight='bold')

            if hasattr(model, 'layers') and len(model.layers) > 0:
                layer = model.layers[0]
                if hasattr(layer, 'token_mixer'):
                    # 可视化OSRA注意力权重
                    if hasattr(layer.token_mixer, 'mixer'):
                        mixer = layer.token_mixer.mixer
                        if hasattr(mixer, 'attn'):
                            attn_weights = mixer.attn.attention_weights.detach().cpu()
                            if attn_weights is not None and len(attn_weights) > 0:
                                # 展示注意力权重分布
                                im = ax3.imshow(attn_weights[0], aspect='auto', cmap='plasma')
                                ax3.set_title('OSRA Attention Weights')
                                ax3.set_xlabel('Query Position')
                                ax3.set_ylabel('Key Position')
                                plt.colorbar(im, ax=ax3, label='Attention Weight')
                            else:
                                ax3.text(0.5, 0.3, 'OSRA attention not available',
                                       ha='center', va='center', fontsize=10)

            # 2.3 模型架构信息
            ax4 = axes[1, 1]
            ax4.text(0.5, 0.5, 'VIPTRNetV2 Architecture Features:\n'
                      '- OSRA (One-Shot Relative Attention)\n'
                      '- Patch Embedding\n'
                      '- DWConv2d Depthwise Separation\n'
                      '- Multi-layer Transformer Blocks',
                   ha='left', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.savefig(f'{output_dir}/visualizations/viptrv2_features_ep{epoch}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f'VIPTRNetV2 features saved → {output_dir}/visualizations/viptrv2_features_ep{epoch}.png')
    except Exception as e:
        print(f'Warning: Could not extract VIPTRNetV2 features: {e}')

@torch.no_grad()
def debug_cnn_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int,
                       output_dir: str, model_config: dict, config: VisualizationConfig):
    """调试CNN CTC对齐可视化

    从数据加载器中获取一个批次的图像数据，使用CNN模型生成预测结果，
    并调用相应的可视化函数生成CNN风格的对齐图。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): CNN CTC模型
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        output_dir (str): 输出目录路径
        model_config (dict): 模型配置字典
        config (VisualizationConfig): 可视化配置对象
    """
    plt.rcParams['text.usetex'] = False

    # 获取数据
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)

    # 生成GT文本
    gt_txt = ''
    for i in batch['labels'][0].tolist():
        if i != model_config.get('blank_id'):
            gt_txt += model_config.get('idx2char')[i]

    # 生成CNN风格的对齐图
    visualize_ctc_colorbar(
        model=model,
        img_tensor=img_tensor,
        gt_text=gt_txt,
        save_path=f'{output_dir}/visualizations/cnn_colorbar_ep{epoch}.png',
        config=config,
        model_config=model_config
    )

def _run_robustness_analysis(device, model, loader, idx2char, blank_id, config):
    """运行鲁棒性分析

    从数据加载器获取样本图像，使用OCRRobustnessAnalyzer分析模型在不同噪声干扰下的性能，
    并调用visualize_robustness_analysis生成可视化结果。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): OCR模型实例
        loader (DataLoader): 数据加载器，用于获取测试图像
        idx2char (dict): 字符ID到字符的映射字典
        blank_id (int): 空白符ID
        config (VisualizationConfig): 可视化配置对象，包含鲁棒性分析相关参数
    """
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)
    gt_txt = ''.join([idx2char[i] for i in batch['labels'][0].tolist()
                      if i not in (blank_id, config.sos_id, config.eos_id)])

    analyzer = OCRRobustnessAnalyzer(model, device)
    results = analyzer.analyze_robustness(
        img_tensor, gt_txt, config.noise_types, config.noise_levels, idx2char, blank_id
    )

    os.makedirs(config.robustness_save_dir, exist_ok=True)
    visualize_robustness_analysis(
        results,
        f'{config.robustness_save_dir}/robustness_analysis.png',
        config.noise_types, config.noise_levels
    )

def _run_training_analysis(device, model, loader, idx2char, blank_id, config, epoch):
    """运行训练过程分析

    从文件加载训练历史，使用TrainingProcessAnalyzer分析训练过程中的性能变化，
    并调用visualize_training_progress生成可视化结果。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): OCR模型实例
        loader (DataLoader): 数据加载器，用于获取测试图像
        idx2char (dict): 字符ID到字符的映射字典
        blank_id (int): 空白符ID
        config (VisualizationConfig): 可视化配置对象，包含训练分析相关参数
        epoch (int): 当前训练轮次，用于命名输出文件
    """
    history_manager = TrainingHistoryManager(config.training_history_file)
    history = history_manager.load_history()

    if history:
        analyzer = TrainingProcessAnalyzer(history_manager)
        results = analyzer.analyze_training_progress(
            config.epochs_to_compare, idx2char, blank_id
        )

        os.makedirs(config.training_save_dir, exist_ok=True)
        visualize_training_progress(
            results,
            f'{config.training_save_dir}/training_progress_ep{epoch}.png',
            config.epochs_to_compare
        )

def _run_augmentation_analysis(device, model, loader, idx2char, blank_id, config):
    """运行数据增强效果分析

    从数据加载器获取样本图像，使用AugmentationEffectAnalyzer分析不同增强策略的效果，
    并调用visualize_augmentation_effects生成可视化结果。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): OCR模型实例
        loader (DataLoader): 数据加载器，用于获取测试图像
        idx2char (dict): 字符ID到字符的映射字典
        blank_id (int): 空白符ID
        config (VisualizationConfig): 可视化配置对象，包含增强分析相关参数
    """
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)
    gt_txt = ''.join([idx2char[i] for i in batch['labels'][0].tolist()
                      if i not in (blank_id, config.sos_id, config.eos_id)])

    analyzer = AugmentationEffectAnalyzer(model, device)
    results = analyzer.analyze_augmentation_effects(
        img_tensor, gt_txt, config.augmentation_types, config.augmentation_levels, idx2char, blank_id
    )

    os.makedirs(config.augmentation_save_dir, exist_ok=True)
    visualize_augmentation_effects(
        results,
        f'{config.augmentation_save_dir}/augmentation_effects.png',
        config.augmentation_types, config.augmentation_levels
    )

def _run_deep_attention_analysis(device, model, loader, idx2char, config):
    """运行深度注意力分析

    从数据加载器获取样本图像，使用DeepAttentionAnalyzer分析Transformer模型的多层多头注意力，
    并调用visualize_deep_attention生成可视化结果。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): Transformer模型实例
        loader (DataLoader): 数据加载器，用于获取测试图像
        idx2char (dict): 字符ID到字符的映射字典
        config (VisualizationConfig): 可视化配置对象，包含注意力分析相关参数
    """
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)
    gt_txt = ''.join([idx2char[i] for i in batch['labels'][0].tolist()
                      if i not in (config.sos_id, config.eos_id)])

    analyzer = DeepAttentionAnalyzer(model)
    results = analyzer.analyze_attention_patterns(
        img_tensor, config.attention_layers_to_visualize,
        config.attention_heads_to_visualize, gt_txt, idx2char
    )

    os.makedirs(config.attention_save_dir, exist_ok=True)
    visualize_deep_attention(
        results,
        f'{config.attention_save_dir}/deep_attention.png',
        config.attention_layers_to_visualize, config.attention_heads_to_visualize, gt_txt
    )

def _run_multi_model_comparison(device, models, model_names, loader, idx2char, blank_id, config):
    """运行多模型对比分析

    从数据加载器获取样本图像，使用MultiModelComparator对比多个模型的性能，
    并调用visualize_multi_model_comparison生成可视化结果。

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        models (list): OCR模型实例列表
        model_names (list): 模型名称列表，与models一一对应
        loader (DataLoader): 数据加载器，用于获取测试图像
        idx2char (dict): 字符ID到字符的映射字典
        blank_id (int): 空白符ID
        config (VisualizationConfig): 可视化配置对象，包含多模型对比相关参数
    """
    batch = next(iter(loader))
    img_tensor = batch['images'][0:1].to(device, non_blocking=True)
    gt_txt = ''.join([idx2char[i] for i in batch['labels'][0].tolist()
                      if i not in (blank_id, config.sos_id, config.eos_id)])

    comparator = MultiModelComparator(models, model_names, device)
    results = comparator.compare_models(img_tensor, gt_txt, idx2char, blank_id)

    os.makedirs(config.comparison_save_dir, exist_ok=True)
    visualize_multi_model_comparison(
        results,
        f'{config.comparison_save_dir}/multi_model_comparison.png'
    )

class NoiseGenerator:
    """噪声生成器，用于生成各种类型的噪声干扰"""

    @staticmethod
    def add_gaussian_noise(img_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(img_tensor) * noise_level
        return torch.clamp(img_tensor + noise, 0, 1)

    @staticmethod
    def add_blur(img_tensor: torch.Tensor, blur_level: float) -> torch.Tensor:
        """添加模糊干扰（使用高斯模糊）"""
        import torchvision.transforms.functional as F
        kernel_size = int(blur_level * 10) + 1
        return F.gaussian_blur(img_tensor, kernel_size=kernel_size)

    @staticmethod
    def add_rotation(img_tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """添加旋转干扰"""
        import torchvision.transforms.functional as F
        return F.rotate(img_tensor, angle)

    @staticmethod
    def add_occlusion(img_tensor: torch.Tensor, occlusion_level: float) -> torch.Tensor:
        """添加遮挡干扰"""
        img = img_tensor.clone()
        h, w = img_tensor.shape[-2:]
        occlusion_size = int(min(h, w) * occlusion_level)

        x = torch.randint(0, w - occlusion_size, (1,)).item()
        y = torch.randint(0, h - occlusion_size, (1,)).item()

        img[:, :, y:y+occlusion_size, x:x+occlusion_size] = 0
        return img

    @staticmethod
    def adjust_contrast(img_tensor: torch.Tensor, contrast_level: float) -> torch.Tensor:
        """调整对比度"""
        import torchvision.transforms.functional as F
        factor = 1.0 + contrast_level * 2
        return F.adjust_contrast(img_tensor, factor)

class OCRRobustnessAnalyzer:
    """OCR鲁棒性分析器，分析模型在不同噪声干扰下的性能"""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.noise_generator = NoiseGenerator()

    @torch.no_grad()
    def analyze_robustness(self, img_tensor: torch.Tensor, gt_text: str,
                         noise_types: list, noise_levels: list,
                         idx2char: dict, blank_id: int) -> dict:
        """分析模型在不同噪声干扰下的鲁棒性

        Args:
            img_tensor: 原始图像张量
            gt_text: 真实文本
            noise_types: 噪声类型列表
            noise_levels: 噪声强度级别列表
            idx2char: 字符ID到字符的映射
            blank_id: 空白符ID

        Returns:
            dict: 鲁棒性分析结果
        """
        results = {
            'original': self._evaluate(img_tensor, gt_text, idx2char, blank_id),
            'noise_results': {}
        }

        for noise_type in noise_types:
            results['noise_results'][noise_type] = {}
            for level in noise_levels:
                noisy_img = self._add_noise(img_tensor, noise_type, level)
                metrics = self._evaluate(noisy_img, gt_text, idx2char, blank_id)
                results['noise_results'][noise_type][level] = metrics

        return results

    def _add_noise(self, img_tensor: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
        """添加指定类型的噪声"""
        noise_map = {
            'gaussian': lambda x: self.noise_generator.add_gaussian_noise(x, level),
            'blur': lambda x: self.noise_generator.add_blur(x, level),
            'rotation': lambda x: self.noise_generator.add_rotation(x, level * 30),
            'occlusion': lambda x: self.noise_generator.add_occlusion(x, level),
            'contrast': lambda x: self.noise_generator.adjust_contrast(x, level)
        }

        return noise_map.get(noise_type, lambda x: x)(img_tensor)

    def _evaluate(self, img_tensor: torch.Tensor, gt_text: str,
                 idx2char: dict, blank_id: int) -> dict:
        """评估模型在给定图像上的性能"""
        model_output = get_model_output(self.model, img_tensor.to(self.device),
                                     train_mode='ctc', gt_text=gt_text)

        pred_text = convert_pred_to_text(model_output.pred_id, idx2char, blank_id)
        cer = self._calculate_cer(pred_text, gt_text)
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)

        return {
            'pred_text': pred_text,
            'cer': cer,
            'accuracy': 1 - cer,
            'errors': errors
        }

    def _calculate_cer(self, pred_text: str, gt_text: str) -> float:
        """计算字符错误率（Character Error Rate）"""
        try:
            from Levenshtein import distance
            return distance(pred_text, gt_text) / max(len(gt_text), 1)
        except ImportError:
            print("Warning: python-Levenshtein not installed, using simple CER calculation")
            if len(pred_text) != len(gt_text):
                return 1.0
            errors = sum(1 for p, g in zip(pred_text, gt_text) if p != g)
            return errors / max(len(gt_text), 1)

@torch.no_grad()
def visualize_robustness_analysis(robustness_results: dict, save_path: str,
                                 noise_types: list, noise_levels: list):
    """可视化鲁棒性分析结果

    可视化原理：
    本函数通过4个子图全面展示OCR模型在不同噪声干扰下的鲁棒性表现：
    1. CER变化曲线：展示模型在不同噪声强度下的字符错误率变化趋势，曲线上升越快说明鲁棒性越差
    2. 准确率变化曲线：展示模型在不同噪声强度下的准确率变化，曲线下降越快说明鲁棒性越差
    3. 错误类型热力图：展示不同噪声类型和强度下，各类错误（替换、删除、插入等）的分布情况
       热力图颜色越深表示错误数量越多，可识别模型在特定噪声下的主要错误类型
    4. 鲁棒性雷达图：综合展示模型对不同噪声类型的鲁棒性评分，评分越高表示越鲁棒
       雷达图面积越大表示整体鲁棒性越好

    Args:
        robustness_results (dict): 鲁棒性分析结果字典，包含以下键：
            - 'original' (dict): 原始图像（无噪声）的性能指标
                * 'cer' (float): 字符错误率
                * 'accuracy' (float): 准确率
                * 'errors' (dict): 错误统计
            - 'noise_results' (dict): 噪声干扰下的性能指标
                * 结构: {noise_type: {noise_level: {metrics}}}
                * noise_type (str): 噪声类型，如 'gaussian', 'blur', 'rotation', 'occlusion', 'contrast'
                * noise_level (float): 噪声强度级别
                * metrics (dict): 包含 'cer', 'accuracy', 'errors' 等指标
        save_path (str): 可视化结果保存路径，如 './robustness_analysis/robustness.png'
        noise_types (list): 噪声类型列表，如 ['gaussian', 'blur', 'rotation', 'occlusion', 'contrast']
        noise_levels (list): 噪声强度级别列表，如 [0.1, 0.2, 0.3, 0.4, 0.5]

    Returns:
        None: 函数将可视化结果保存到指定路径，不返回值

    可视化输出：
    - 生成一个2x2的子图布局，包含：
      1. 左上：CER vs Noise Level 折线图
      2. 右上：Accuracy vs Noise Level 折线图
      3. 左下：Error Type Distribution Heatmap 热力图
      4. 右下：Robustness Score Radar Chart 雷达图
    - 图片保存为高分辨率（300 DPI）PNG格式

    使用示例：
        >>> robustness_results = {
        ...     'original': {'cer': 0.05, 'accuracy': 0.95, 'errors': {...}},
        ...     'noise_results': {
        ...         'gaussian': {0.1: {'cer': 0.08, 'accuracy': 0.92, 'errors': {...}}, ...},
        ...         'blur': {0.1: {'cer': 0.07, 'accuracy': 0.93, 'errors': {...}}, ...}
        ...     }
        ... }
        >>> visualize_robustness_analysis(
        ...     robustness_results,
        ...     './robustness_analysis/robustness.png',
        ...     ['gaussian', 'blur', 'rotation'],
        ...     [0.1, 0.2, 0.3, 0.4, 0.5]
        ... )
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. CER随噪声强度变化曲线
    ax1 = axes[0, 0]
    for noise_type in noise_types:
        cer_values = [robustness_results['noise_results'][noise_type][level]['cer']
                     for level in noise_levels]
        ax1.plot(noise_levels, cer_values, marker='o', label=noise_type, linewidth=2)

    ax1.axhline(y=robustness_results['original']['cer'], color='black',
                linestyle='--', label='Original', linewidth=2)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Character Error Rate (CER)')
    ax1.set_title('CER vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 准确率随噪声强度变化曲线
    ax2 = axes[0, 1]
    for noise_type in noise_types:
        acc_values = [robustness_results['noise_results'][noise_type][level]['accuracy']
                     for level in noise_levels]
        ax2.plot(noise_levels, acc_values, marker='s', label=noise_type, linewidth=2)

    ax2.axhline(y=robustness_results['original']['accuracy'], color='black',
                linestyle='--', label='Original', linewidth=2)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 错误类型分布热力图
    ax3 = axes[1, 0]
    error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']

    heatmap_data = []
    for noise_type in noise_types:
        row = []
        for level in noise_levels:
            errors = robustness_results['noise_results'][noise_type][level]['errors']
            row.append([errors[etype + '_errors'] for etype in error_types])
        heatmap_data.append(row)

    im = ax3.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax3.set_xticks(range(len(noise_levels)))
    ax3.set_xticklabels([f'{level:.1f}' for level in noise_levels])
    ax3.set_yticks(range(len(noise_types)))
    ax3.set_yticklabels(noise_types)
    ax3.set_xlabel('Noise Level')
    ax3.set_ylabel('Noise Type')
    ax3.set_title('Error Type Distribution Heatmap')
    plt.colorbar(im, ax=ax3, label='Error Count')

    # 4. 鲁棒性评分雷达图
    ax4 = axes[1, 1]
    categories = noise_types
    N = len(categories)

    robustness_scores = []
    for noise_type in noise_types:
        avg_cer = np.mean([robustness_results['noise_results'][noise_type][level]['cer']
                          for level in noise_levels])
        score = 1 - avg_cer
        robustness_scores.append(score)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    robustness_scores += robustness_scores[:1]
    angles += angles[:1]

    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, robustness_scores, 'o-', linewidth=2)
    ax4.fill(angles, robustness_scores, alpha=0.25)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Robustness Score Radar Chart')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Robustness analysis saved → {save_path}')

class TrainingHistoryManager:
    """训练历史管理器，用于记录和加载训练过程中的模型状态"""

    def __init__(self, history_file: str):
        self.history_file = history_file
        self.history = {
            'epochs': [],
            'losses': [],
            'metrics': [],
            'model_outputs': {},
            'sample_images': {}
        }

    def save_epoch(self, epoch: int, loss: float, metrics: dict,
                   model_output: ModelOutput, img_tensor: torch.Tensor):
        """保存一个epoch的训练信息

        该方法将一个epoch的训练信息保存到历史记录中，并将整个历史记录持久化到文件。

        Args:
            epoch (int): 当前训练轮次
            loss (float): 该epoch的损失值
            metrics (dict): 该epoch的性能指标字典
            model_output (ModelOutput): 该epoch的模型输出结果
            img_tensor (torch.Tensor): 用于测试的图像张量
        """
        self.history['epochs'].append(epoch)
        self.history['losses'].append(loss)
        self.history['metrics'].append(metrics)
        self.history['model_outputs'][epoch] = model_output
        self.history['sample_images'][epoch] = img_tensor.cpu()

        import pickle
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.history, f)

    def load_history(self) -> dict:
        """加载训练历史

        从文件中加载训练历史记录，返回历史字典。如果文件不存在，返回None。

        Returns:
            dict or None: 训练历史字典，包含epochs、losses、metrics、model_outputs和sample_images等键
        """
        import pickle
        try:
            with open(self.history_file, 'rb') as f:
                self.history = pickle.load(f)
            return self.history
        except FileNotFoundError:
            return None

    def get_epochs_data(self, epochs: list) -> dict:
        """获取指定epoch的数据

        根据输入的epoch列表，从历史记录中提取对应的训练数据。

        Args:
            epochs (list): 需要获取数据的epoch列表

        Returns:
            dict: 指定epoch的数据字典，结构为 {epoch: {loss, metrics, model_output, img_tensor}}
        """
        data = {}
        for epoch in epochs:
            if epoch in self.history['model_outputs']:
                data[epoch] = {
                    'loss': self.history['losses'][epoch],
                    'metrics': self.history['metrics'][epoch],
                    'model_output': self.history['model_outputs'][epoch],
                    'img_tensor': self.history['sample_images'][epoch]
                }
        return data

class TrainingProcessAnalyzer:
    """训练过程分析器，分析模型在训练过程中的性能变化"""

    def __init__(self, history_manager: TrainingHistoryManager):
        self.history_manager = history_manager

    def analyze_training_progress(self, epochs: list, idx2char: dict, blank_id: int) -> dict:
        """分析训练过程中的性能变化

        该方法分析指定epoch的训练性能变化，包括损失、对齐质量、置信度和错误类型等指标。

        Args:
            epochs (list): 需要分析的epoch列表
            idx2char (dict): 字符ID到字符的映射字典
            blank_id (int): 空白符ID

        Returns:
            dict: 训练过程分析结果，包含以下键：
                - 'loss_trend': 损失值变化趋势
                - 'metric_trends': 其他性能指标变化趋势
                - 'alignment_quality_trend': 对齐质量变化趋势
                - 'confidence_trend': 平均置信度变化趋势
                - 'error_type_trends': 各错误类型变化趋势
                - 'epoch_comparisons': 各epoch的详细比较数据
        """
        epochs_data = self.history_manager.get_epochs_data(epochs)

        results = {
            'loss_trend': [],
            'metric_trends': {},
            'alignment_quality_trend': [],
            'confidence_trend': [],
            'error_type_trends': {},
            'epoch_comparisons': {}
        }

        for epoch in epochs:
            if epoch in epochs_data:
                results['loss_trend'].append(epochs_data[epoch]['loss'])

                for metric_name, metric_value in epochs_data[epoch]['metrics'].items():
                    if metric_name not in results['metric_trends']:
                        results['metric_trends'][metric_name] = []
                    results['metric_trends'][metric_name].append(metric_value)

                model_output = epochs_data[epoch]['model_output']
                alignment_quality = self._calculate_alignment_quality(
                    model_output, idx2char, blank_id
                )
                results['alignment_quality_trend'].append(alignment_quality)

                avg_confidence = model_output.prob_values.mean().item()
                results['confidence_trend'].append(avg_confidence)

                pred_text = convert_pred_to_text(model_output.pred_id, idx2char, blank_id)
                gt_text = model_output.gt_text
                errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)

                for error_type in ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']:
                    if error_type not in results['error_type_trends']:
                        results['error_type_trends'][error_type] = []
                    results['error_type_trends'][error_type].append(
                        errors[error_type + '_errors']
                    )

                results['epoch_comparisons'][epoch] = {
                    'pred_text': pred_text,
                    'gt_text': gt_text,
                    'cer': self._calculate_cer(pred_text, gt_text),
                    'avg_confidence': avg_confidence,
                    'alignment_quality': alignment_quality,
                    'errors': errors
                }

        return results

    def _calculate_alignment_quality(self, model_output: ModelOutput,
                                    idx2char: dict, blank_id: int) -> float:
        """计算对齐质量分数

        基于模型输出和真实文本，计算对齐质量分数。
        对齐质量 = 1 - 字符错误率 (CER)

        Args:
            model_output (ModelOutput): 模型输出结果
            idx2char (dict): 字符ID到字符的映射字典
            blank_id (int): 空白符ID

        Returns:
            float: 对齐质量分数，范围[0, 1]，分数越高表示对齐越准确
        """
        pred_id = model_output.pred_id
        gt_text = model_output.gt_text

        pred_text = convert_pred_to_text(pred_id, idx2char, blank_id)
        cer = self._calculate_cer(pred_text, gt_text)

        return 1 - cer

    def _calculate_cer(self, pred_text: str, gt_text: str) -> float:
        """计算字符错误率

        使用Levenshtein距离计算字符错误率（Character Error Rate）。
        如果Levenshtein库不可用，则使用简单的字符比较方法。

        Args:
            pred_text (str): 预测的文本字符串
            gt_text (str): 真实文本字符串

        Returns:
            float: 字符错误率，范围[0, 1]，分数越高表示错误越多
        """
        try:
            from Levenshtein import distance
            return distance(pred_text, gt_text) / max(len(gt_text), 1)
        except ImportError:
            if len(pred_text) != len(gt_text):
                return 1.0
            errors = sum(1 for p, g in zip(pred_text, gt_text) if p != g)
            return errors / max(len(gt_text), 1)

@torch.no_grad()
def visualize_training_progress(training_results: dict, save_path: str, epochs: list):
    """可视化训练过程中的性能变化

    可视化原理：
    本函数通过8个子图全面展示OCR模型在训练过程中的性能演变：
    1. 损失变化曲线：展示训练损失随epoch的变化趋势，理想情况下应持续下降
       损失下降速度和收敛情况反映模型学习效率
    2. 对齐质量变化曲线：展示CTC对齐质量随epoch的变化，质量越高表示对齐越准确
       对齐质量 = 1 - CER，范围[0, 1]
    3. 平均置信度变化曲线：展示模型预测置信度随epoch的变化，置信度越高表示模型越确定
       置信度范围[0, 1]，理想情况下应逐渐提高
    4. CER变化曲线：展示字符错误率随epoch的变化，CER越低表示性能越好
       CER下降趋势反映模型学习效果
    5. 错误类型趋势：展示7种错误类型（替换、删除、插入、交换、空格、大小写、标点）随epoch的变化
       可识别模型在不同训练阶段的薄弱环节
    6. 其他指标趋势：展示其他训练指标（如准确率、F1等）随epoch的变化
       可根据需要自定义添加更多指标
    7. 性能综合雷达图：展示最后一个epoch的综合性能（对齐质量、置信度、准确率）
       雷达图面积越大表示整体性能越好
    8. Epoch对比表格：详细对比不同epoch的预测结果、CER、置信度和错误统计
       表格形式便于快速查看关键指标变化

    Args:
        training_results (dict): 训练过程分析结果字典，包含以下键：
            - 'loss_trend' (list): 损失值列表，每个epoch一个值
            - 'alignment_quality_trend' (list): 对齐质量列表，范围[0, 1]
            - 'confidence_trend' (list): 平均置信度列表，范围[0, 1]
            - 'error_type_trends' (dict): 错误类型趋势，键为错误类型，值为错误数量列表
                * 'substitution' (list): 替换错误数量列表
                * 'deletion' (list): 删除错误数量列表
                * 'insertion' (list): 插入错误数量列表
                * 'swap' (list): 交换错误数量列表
                * 'space' (list): 空格错误数量列表
                * 'case' (list): 大小写错误数量列表
                * 'punctuation' (list): 标点符号错误数量列表
            - 'metric_trends' (dict): 其他指标趋势，键为指标名，值为指标值列表
            - 'epoch_comparisons' (dict): 各epoch详细对比数据
                * 结构: {epoch: {metrics}}
                * metrics (dict): 包含 'pred_text', 'gt_text', 'cer', 'avg_confidence',
                  'alignment_quality', 'errors' 等
        save_path (str): 可视化结果保存路径，如 './training_analysis/training_progress.png'
        epochs (list): 要可视化的epoch列表，如 [1, 5, 10, 20, 30]

    Returns:
        None: 函数将可视化结果保存到指定路径，不返回值

    可视化输出：
    - 生成一个3x3的子图布局（最后一个位置为空），包含：
      1. 左上：Training Loss Trend 折线图
      2. 中上：Alignment Quality Trend 折线图
      3. 右上：Prediction Confidence Trend 折线图
      4. 左中：CER Trend 折线图
      5. 中中：Error Type Trends 多条折线图
      6. 右中：Other Metrics Trend 多条折线图
      7. 左下：Performance Radar 雷达图
      8. 中下-右下：Epoch Comparison Table 表格
    - 图片保存为高分辨率（300 DPI）PNG格式

    使用示例：
        >>> training_results = {
        ...     'loss_trend': [2.5, 1.8, 1.2, 0.8, 0.5],
        ...     'alignment_quality_trend': [0.6, 0.75, 0.85, 0.9, 0.93],
        ...     'confidence_trend': [0.65, 0.78, 0.87, 0.92, 0.95],
        ...     'error_type_trends': {
        ...         'substitution': [10, 8, 5, 3, 2],
        ...         'deletion': [5, 4, 3, 2, 1],
        ...         ...
        ...     },
        ...     'metric_trends': {'accuracy': [0.85, 0.9, 0.93, 0.95, 0.96]},
        ...     'epoch_comparisons': {
        ...         1: {'pred_text': 'hello', 'gt_text': 'hello', 'cer': 0.0, ...},
        ...         5: {'pred_text': 'world', 'gt_text': 'world', 'cer': 0.0, ...},
        ...         ...
        ...     }
        ... }
        >>> visualize_training_progress(
        ...     training_results,
        ...     './training_analysis/training_progress.png',
        ...     [1, 5, 10, 20, 30]
        ... )
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 损失变化曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, training_results['loss_trend'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Trend')
    ax1.grid(True, alpha=0.3)

    # 2. 对齐质量变化曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, training_results['alignment_quality_trend'], 'g-o',
             linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Alignment Quality')
    ax2.set_title('Alignment Quality Trend')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # 3. 平均置信度变化曲线
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, training_results['confidence_trend'], 'r-o',
             linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Confidence')
    ax3.set_title('Prediction Confidence Trend')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # 4. CER变化曲线
    ax4 = fig.add_subplot(gs[1, 0])
    cer_values = [training_results['epoch_comparisons'][ep]['cer'] for ep in epochs]
    ax4.plot(epochs, cer_values, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Character Error Rate')
    ax4.set_title('CER Trend')
    ax4.grid(True, alpha=0.3)

    # 5. 错误类型趋势
    ax5 = fig.add_subplot(gs[1, 1])
    error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']
    colors = ['red', 'orange', 'yellow', 'purple', 'blue', 'cyan', 'magenta']

    for error_type, color in zip(error_types, colors):
        if error_type in training_results['error_type_trends']:
            ax5.plot(epochs, training_results['error_type_trends'][error_type],
                    marker='o', label=error_type, color=color, linewidth=2)

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Error Count')
    ax5.set_title('Error Type Trends')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. 指标对比
    ax6 = fig.add_subplot(gs[1, 2])
    for metric_name, metric_values in training_results['metric_trends'].items():
        ax6.plot(epochs, metric_values, marker='s', label=metric_name, linewidth=2)

    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Metric Value')
    ax6.set_title('Other Metrics Trend')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. 性能综合雷达图
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    categories = ['Alignment Quality', 'Confidence', 'Accuracy']
    N = len(categories)

    last_epoch = epochs[-1]
    values = [
        training_results['alignment_quality_trend'][-1],
        training_results['confidence_trend'][-1],
        1 - training_results['epoch_comparisons'][last_epoch]['cer']
    ]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax7.plot(angles, values, 'o-', linewidth=2)
    ax7.fill(angles, values, alpha=0.25)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(categories)
    ax7.set_ylim(0, 1)
    ax7.set_title(f'Performance Radar (Epoch {last_epoch})')
    ax7.grid(True)

    # 8. Epoch对比表格
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('tight')
    ax8.axis('off')

    table_data = [['Epoch', 'GT Text', 'Pred Text', 'CER', 'Confidence', 'Errors']]
    for epoch in epochs:
        comp = training_results['epoch_comparisons'][epoch]
        error_summary = f"Sub:{comp['errors']['substitution_errors']} " \
                       f"Del:{comp['errors']['deletion_errors']} " \
                       f"Ins:{comp['errors']['insertion_errors']}"
        table_data.append([
            epoch,
            comp['gt_text'][:20] + '...' if len(comp['gt_text']) > 20 else comp['gt_text'],
            comp['pred_text'][:20] + '...' if len(comp['pred_text']) > 20 else comp['pred_text'],
            f"{comp['cer']:.3f}",
            f"{comp['avg_confidence']:.3f}",
            error_summary
        ])

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax8.set_title('Epoch Comparison Table', pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training progress analysis saved → {save_path}')

class DataAugmentationVisualizer:
    """数据增强效果可视化器，用于生成和可视化各种数据增强效果"""

    @staticmethod
    def apply_rotation(img_tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """应用旋转变换"""
        import torchvision.transforms.functional as F
        return F.rotate(img_tensor, angle)

    @staticmethod
    def apply_scale(img_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """应用缩放变换"""
        import torchvision.transforms.functional as F
        h, w = img_tensor.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = F.resize(img_tensor, [new_h, new_w])
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        else:
            padded = torch.zeros_like(img_tensor)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            padded[:, :, start_h:start_h+new_h, start_w:start_w+new_w] = scaled
            return padded

    @staticmethod
    def apply_shear(img_tensor: torch.Tensor, shear: float) -> torch.Tensor:
        """应用剪切变换"""
        import torchvision.transforms.functional as F
        return F.affine(img_tensor, angle=0, translate=[0, 0], scale=1.0,
                       shear=[shear, shear])

    @staticmethod
    def apply_color_jitter(img_tensor: torch.Tensor, brightness: float,
                          contrast: float, saturation: float) -> torch.Tensor:
        """应用颜色抖动"""
        import torchvision.transforms.functional as F
        img = F.adjust_brightness(img_tensor, 1.0 + brightness)
        img = F.adjust_contrast(img, 1.0 + contrast)
        img = F.adjust_saturation(img, 1.0 + saturation)
        return img

class AugmentationEffectAnalyzer:
    """数据增强效果分析器，分析不同增强策略对模型性能的影响"""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.augmenter = DataAugmentationVisualizer()

    @torch.no_grad()
    def analyze_augmentation_effects(self, img_tensor: torch.Tensor, gt_text: str,
                                   augmentation_types: list, augmentation_levels: list,
                                   idx2char: dict, blank_id: int) -> dict:
        """分析不同数据增强策略的效果"""
        results = {
            'original': self._evaluate(img_tensor, gt_text, idx2char, blank_id),
            'augmentation_results': {}
        }

        for aug_type in augmentation_types:
            results['augmentation_results'][aug_type] = {}
            for level in augmentation_levels:
                augmented_img = self._apply_augmentation(img_tensor, aug_type, level)
                metrics = self._evaluate(augmented_img, gt_text, idx2char, blank_id)
                results['augmentation_results'][aug_type][level] = metrics

        return results

    def _apply_augmentation(self, img_tensor: torch.Tensor, aug_type: str, level: float) -> torch.Tensor:
        """应用指定类型的增强"""
        aug_map = {
            'rotation': lambda x: self.augmenter.apply_rotation(x, level * 30),
            'scale': lambda x: self.augmenter.apply_scale(x, 1.0 + level),
            'shear': lambda x: self.augmenter.apply_shear(x, level * 20),
            'color_jitter': lambda x: self.augmenter.apply_color_jitter(
                x, brightness=level, contrast=level, saturation=level)
        }

        return aug_map.get(aug_type, lambda x: x)(img_tensor)

    def _evaluate(self, img_tensor: torch.Tensor, gt_text: str,
                 idx2char: dict, blank_id: int) -> dict:
        """评估模型在给定图像上的性能"""
        model_output = get_model_output(self.model, img_tensor.to(self.device),
                                     train_mode='ctc', gt_text=gt_text)

        pred_text = convert_pred_to_text(model_output.pred_id, idx2char, blank_id)
        cer = self._calculate_cer(pred_text, gt_text)
        errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)

        return {
            'pred_text': pred_text,
            'cer': cer,
            'accuracy': 1 - cer,
            'avg_confidence': model_output.prob_values.mean().item(),
            'errors': errors
        }

    def _calculate_cer(self, pred_text: str, gt_text: str) -> float:
        """计算字符错误率"""
        try:
            from Levenshtein import distance
            return distance(pred_text, gt_text) / max(len(gt_text), 1)
        except ImportError:
            if len(pred_text) != len(gt_text):
                return 1.0
            errors = sum(1 for p, g in zip(pred_text, gt_text) if p != g)
            return errors / max(len(gt_text), 1)

@torch.no_grad()
def visualize_augmentation_effects(augmentation_results: dict, save_path: str,
                                 augmentation_types: list, augmentation_levels: list):
    """可视化数据增强效果分析结果

    可视化原理：
    本函数通过5个子图全面展示不同数据增强策略对OCR模型性能的影响：
    1. CER变化曲线：展示不同增强策略在不同强度下的字符错误率变化
       曲线上升越快说明该增强策略对模型影响越大
       与原始图像（虚线）对比可评估增强策略的鲁棒性
    2. 准确率变化曲线：展示不同增强策略在不同强度下的准确率变化
       曲线下降越快说明该增强策略对模型影响越大
       可识别哪些增强策略在什么强度范围内是可接受的
    3. 错误类型分布热力图：展示不同增强策略和强度下，各类错误的分布情况
       热力图颜色越深表示错误数量越多
       可识别特定增强策略导致的错误类型倾向
    4. 增强效果雷达图：综合展示模型对不同增强策略的鲁棒性评分
       评分 = 1 - 平均CER，范围[0, 1]
       雷达图面积越大表示模型对增强策略越鲁棒
    5. 增强效果对比表格：详细对比不同增强策略在不同强度下的性能指标
       表格包含CER、准确率、置信度和主要错误统计
       便于快速查找最优增强策略和强度参数

    Args:
        augmentation_results (dict): 数据增强效果分析结果字典，包含以下键：
            - 'original' (dict): 原始图像（无增强）的性能指标
                * 'cer' (float): 字符错误率
                * 'accuracy' (float): 准确率
                * 'avg_confidence' (float): 平均置信度
                * 'errors' (dict): 错误统计
            - 'augmentation_results' (dict): 增强后的性能指标
                * 结构: {augmentation_type: {augmentation_level: {metrics}}}
                * augmentation_type (str): 增强类型，如 'rotation', 'scale', 'shear', 'color_jitter'
                * augmentation_level (float): 增强强度级别
                * metrics (dict): 包含 'cer', 'accuracy', 'avg_confidence', 'errors' 等指标
        save_path (str): 可视化结果保存路径，如 './augmentation_analysis/augmentation_effects.png'
        augmentation_types (list): 增强类型列表，如 ['rotation', 'scale', 'shear', 'color_jitter']
        augmentation_levels (list): 增强强度级别列表，如 [0.1, 0.2, 0.3, 0.4, 0.5]

    Returns:
        None: 函数将可视化结果保存到指定路径，不返回值

    可视化输出：
    - 生成一个4x3的子图布局（部分位置为空），包含：
      1. 左上：CER vs Augmentation Level 折线图
      2. 中上：Accuracy vs Augmentation Level 折线图
      3. 右上：Error Type Distribution Heatmap 热力图
      4. 左中：Augmentation Robustness Radar Chart 雷达图
      5. 中中-右中：Augmentation Effect Comparison Table 表格
    - 图片保存为高分辨率（300 DPI）PNG格式

    使用示例：
        >>> augmentation_results = {
        ...     'original': {'cer': 0.05, 'accuracy': 0.95, 'avg_confidence': 0.9, 'errors': {...}},
        ...     'augmentation_results': {
        ...         'rotation': {0.1: {'cer': 0.06, 'accuracy': 0.94, 'avg_confidence': 0.88, 'errors': {...}}, ...},
        ...         'scale': {0.1: {'cer': 0.07, 'accuracy': 0.93, 'avg_confidence': 0.87, 'errors': {...}}, ...}
        ...     }
        ... }
        >>> visualize_augmentation_effects(
        ...     augmentation_results,
        ...     './augmentation_analysis/augmentation_effects.png',
        ...     ['rotation', 'scale', 'shear', 'color_jitter'],
        ...     [0.1, 0.2, 0.3, 0.4, 0.5]
        ... )
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. CER随增强强度变化曲线
    ax1 = fig.add_subplot(gs[0, 0])
    for aug_type in augmentation_types:
        cer_values = [augmentation_results['augmentation_results'][aug_type][level]['cer']
                     for level in augmentation_levels]
        ax1.plot(augmentation_levels, cer_values, marker='o',
                label=aug_type, linewidth=2)

    ax1.axhline(y=augmentation_results['original']['cer'], color='black',
                linestyle='--', label='Original', linewidth=2)
    ax1.set_xlabel('Augmentation Level')
    ax1.set_ylabel('Character Error Rate (CER)')
    ax1.set_title('CER vs Augmentation Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 准确率随增强强度变化曲线
    ax2 = fig.add_subplot(gs[0, 1])
    for aug_type in augmentation_types:
        acc_values = [augmentation_results['augmentation_results'][aug_type][level]['accuracy']
                     for level in augmentation_levels]
        ax2.plot(augmentation_levels, acc_values, marker='s',
                label=aug_type, linewidth=2)

    ax2.axhline(y=augmentation_results['original']['accuracy'], color='black',
                linestyle='--', label='Original', linewidth=2)
    ax2.set_xlabel('Augmentation Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Augmentation Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 错误类型分布热力图
    ax3 = fig.add_subplot(gs[0, 2])
    error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']

    heatmap_data = []
    for aug_type in augmentation_types:
        row = []
        for level in augmentation_levels:
            errors = augmentation_results['augmentation_results'][aug_type][level]['errors']
            row.append([errors[etype + '_errors'] for etype in error_types])
        heatmap_data.append(row)

    im = ax3.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax3.set_xticks(range(len(augmentation_levels)))
    ax3.set_xticklabels([f'{level:.1f}' for level in augmentation_levels])
    ax3.set_yticks(range(len(augmentation_types)))
    ax3.set_yticklabels(augmentation_types)
    ax3.set_xlabel('Augmentation Level')
    ax3.set_ylabel('Augmentation Type')
    ax3.set_title('Error Type Distribution Heatmap')
    plt.colorbar(im, ax=ax3, label='Error Count')

    # 4. 增强效果对比雷达图
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    categories = augmentation_types
    N = len(categories)

    robustness_scores = []
    for aug_type in augmentation_types:
        avg_cer = np.mean([augmentation_results['augmentation_results'][aug_type][level]['cer']
                          for level in augmentation_levels])
        score = 1 - avg_cer
        robustness_scores.append(score)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    robustness_scores += robustness_scores[:1]
    angles += angles[:1]

    ax4.plot(angles, robustness_scores, 'o-', linewidth=2)
    ax4.fill(angles, robustness_scores, alpha=0.25)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Augmentation Robustness Radar Chart')
    ax4.grid(True)

    # 5. 增强效果对比表格
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('tight')
    ax5.axis('off')

    table_data = [['Augmentation', 'Level', 'CER', 'Accuracy', 'Confidence', 'Major Errors']]
    for aug_type in augmentation_types:
        for level in augmentation_levels:
            result = augmentation_results['augmentation_results'][aug_type][level]
            errors = result['errors']
            major_errors = f"Sub:{errors['substitution_errors']} " \
                           f"Del:{errors['deletion_errors']} " \
                           f"Ins:{errors['insertion_errors']}"
            table_data.append([
                aug_type,
                f'{level:.1f}',
                f"{result['cer']:.3f}",
                f"{result['accuracy']:.3f}",
                f"{result['avg_confidence']:.3f}",
                major_errors
            ])

    table = ax5.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax5.set_title('Augmentation Effect Comparison Table', pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Augmentation effects analysis saved → {save_path}')

class AttentionExtractor:
    """注意力提取器，用于提取Transformer模型的多层多头注意力权重"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_weights = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子以捕获注意力权重

        该方法在模型的Transformer层上注册前向钩子，用于在模型前向传播时捕获注意力权重。
        目前仅支持带有decoder.ar_decoder.transformer结构的模型。
        """
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights[name] = output[1].detach().cpu()
            return hook

        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'ar_decoder'):
            ar_decoder = self.model.decoder.ar_decoder
            if hasattr(ar_decoder, 'transformer'):
                transformer = ar_decoder.transformer
                for idx, layer in enumerate(transformer.layers):
                    layer_name = f'layer_{idx}'
                    layer.register_forward_hook(get_attention_hook(layer_name))

    def extract_attention(self, img_tensor: torch.Tensor) -> dict:
        """提取注意力权重

        运行模型前向传播，通过已注册的钩子捕获注意力权重。

        Args:
            img_tensor (torch.Tensor): 输入图像张量

        Returns:
            dict: 注意力权重字典，键为层名称（如 'layer_0'），值为注意力权重张量
        """
        self.attention_weights = {}
        with torch.no_grad():
            _ = self.model(img_tensor)
        return self.attention_weights

    def get_layer_attention(self, layer_idx: int) -> torch.Tensor:
        """获取指定层的注意力权重

        Args:
            layer_idx (int): 层索引

        Returns:
            torch.Tensor or None: 指定层的注意力权重张量，若不存在则返回None
        """
        layer_name = f'layer_{layer_idx}'
        return self.attention_weights.get(layer_name, None)

    def get_head_attention(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """获取指定层和头的注意力权重

        Args:
            layer_idx (int): 层索引
            head_idx (int): 头索引

        Returns:
            torch.Tensor or None: 指定层和头的注意力权重张量，若不存在则返回None
        """
        layer_attention = self.get_layer_attention(layer_idx)
        if layer_attention is None:
            return None

        if len(layer_attention.shape) == 4:
            return layer_attention[0, head_idx]

        return None

class DeepAttentionAnalyzer:
    """深度注意力分析器，分析Transformer模型的多层多头注意力模式"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_extractor = AttentionExtractor(model)

    @torch.no_grad()
    def analyze_attention_patterns(self, img_tensor: torch.Tensor,
                                   layers: list, heads: list,
                                   gt_text: str, idx2char: dict) -> dict:
        """分析多层多头注意力模式

        该方法提取并分析Transformer模型的多层多头注意力模式，包括层分析、头分析和注意力流分析。

        Args:
            img_tensor (torch.Tensor): 输入图像张量
            layers (list): 需要分析的层索引列表
            heads (list): 需要分析的头索引列表
            gt_text (str): 真实文本字符串
            idx2char (dict): 字符ID到字符的映射字典

        Returns:
            dict: 注意力模式分析结果，包含以下键：
                - 'attention_weights': 原始注意力权重字典
                - 'layer_analysis': 各层注意力分析结果
                - 'head_analysis': 各头注意力分析结果
                - 'attention_flow': 注意力流分析结果
                - 'cross_self_comparison': 交叉自注意力比较结果
        """
        attention_weights = self.attention_extractor.extract_attention(img_tensor)

        results = {
            'attention_weights': attention_weights,
            'layer_analysis': {},
            'head_analysis': {},
            'attention_flow': {},
            'cross_self_comparison': {}
        }

        for layer_idx in layers:
            layer_name = f'layer_{layer_idx}'
            if layer_name in attention_weights:
                layer_attn = attention_weights[layer_name]
                results['layer_analysis'][layer_idx] = self._analyze_layer_attention(
                    layer_attn, gt_text, idx2char
                )

        for layer_idx in layers:
            for head_idx in heads:
                head_attn = self.attention_extractor.get_head_attention(layer_idx, head_idx)
                if head_attn is not None:
                    results['head_analysis'][(layer_idx, head_idx)] = self._analyze_head_attention(
                        head_attn, gt_text, idx2char
                    )

        results['attention_flow'] = self._analyze_attention_flow(
            attention_weights, layers, gt_text, idx2char
        )

        return results

    def _analyze_layer_attention(self, layer_attn: torch.Tensor,
                                gt_text: str, idx2char: dict) -> dict:
        """分析单层的注意力模式

        分析指定层的注意力权重，计算平均注意力、熵、稀疏度等指标。

        Args:
            layer_attn (torch.Tensor): 层注意力权重张量
            gt_text (str): 真实文本字符串
            idx2char (dict): 字符ID到字符的映射字典

        Returns:
            dict: 层注意力分析结果，包含平均注意力、熵、稀疏度等指标
        """
        avg_attention = layer_attn.mean(dim=1)

        entropy = -(avg_attention * torch.log(avg_attention + 1e-10)).sum(dim=1)

        max_attention_pos = avg_attention.argmax(dim=1)

        return {
            'avg_attention': avg_attention,
            'entropy': entropy,
            'max_attention_pos': max_attention_pos,
            'attention_sparsity': (avg_attention < 0.01).float().mean()
        }

    def _analyze_head_attention(self, head_attn: torch.Tensor,
                                gt_text: str, idx2char: dict) -> dict:
        """分析单个头的注意力模式

        分析指定头的注意力权重，计算均值、标准差、最大值、熵等指标，并分类注意力模式。

        Args:
            head_attn (torch.Tensor): 头注意力权重张量
            gt_text (str): 真实文本字符串
            idx2char (dict): 字符ID到字符的映射字典

        Returns:
            dict: 头注意力分析结果，包含注意力权重、均值、标准差、最大值、熵和注意力模式
        """
        mean_attention = head_attn.mean()
        std_attention = head_attn.std()
        max_attention = head_attn.max()

        entropy = -(head_attn * torch.log(head_attn + 1e-10)).sum()

        attention_pattern = self._classify_attention_pattern(head_attn)

        return {
            'attention_weights': head_attn,
            'mean': mean_attention,
            'std': std_attention,
            'max': max_attention,
            'entropy': entropy,
            'pattern': attention_pattern
        }

    def _classify_attention_pattern(self, attention: torch.Tensor) -> str:
        """分类注意力模式

        根据注意力权重的分布特征，将注意力模式分为以下四类：
        - 'local': 局部注意力，主要关注相邻位置
        - 'uniform': 均匀注意力，注意力分布较为均匀
        - 'focused': 聚焦注意力，有明显的焦点位置
        - 'global': 全局注意力，关注整个序列

        Args:
            attention (torch.Tensor): 注意力权重张量，形状为 [seq_len, seq_len]

        Returns:
            str: 注意力模式类型
        """
        seq_len = attention.shape[0]

        diag_weights = torch.diag(attention)
        off_diag_weights = attention - torch.diag(diag_weights)

        if diag_weights.mean() > off_diag_weights.mean() * 2:
            return 'local'
        elif attention.std() < 0.1:
            return 'uniform'
        elif attention.max() > 0.5:
            return 'focused'
        else:
            return 'global'

    def _analyze_attention_flow(self, attention_weights: dict, layers: list,
                                gt_text: str, idx2char: dict) -> dict:
        """分析注意力在不同层之间的流动

        分析相邻层之间的注意力变化和相关性，评估注意力信息在层间的传递情况。

        Args:
            attention_weights (dict): 各层注意力权重字典
            layers (list): 层索引列表
            gt_text (str): 真实文本字符串
            idx2char (dict): 字符ID到字符的映射字典

        Returns:
            dict: 注意力流分析结果，包含层间注意力变化和相关性
        """
        flow_data = []

        for i in range(len(layers) - 1):
            layer1_name = f'layer_{layers[i]}'
            layer2_name = f'layer_{layers[i+1]}'

            if layer1_name in attention_weights and layer2_name in attention_weights:
                attn1 = attention_weights[layer1_name]
                attn2 = attention_weights[layer2_name]

                attention_change = (attn2 - attn1).abs().mean()

                correlation = torch.corrcoef(
                    attn1.flatten().unsqueeze(0),
                    attn2.flatten().unsqueeze(0)
                )[0, 1]

                flow_data.append({
                    'from_layer': layers[i],
                    'to_layer': layers[i+1],
                    'attention_change': attention_change.item(),
                    'correlation': correlation.item()
                })

        return flow_data

@torch.no_grad()
def visualize_deep_attention(attention_results: dict, save_path: str,
                            layers: list, heads: list, gt_text: str):
    """可视化深度注意力分析结果

    可视化原理：
    本函数通过15个子图全面展示Transformer模型的多层多头注意力机制：
    1. 平均注意力随层数变化：展示各层的平均注意力权重，反映注意力强度分布
       通常浅层注意力较强，深层注意力更聚焦
    2. 注意力熵随层数变化：展示各层注意力分布的熵值，反映注意力集中程度
       熵值越低表示注意力越集中，熵值越高表示注意力越分散
    3. 注意力稀疏度随层数变化：展示各层注意力的稀疏性，反映注意力选择能力
       稀疏度越高表示注意力越集中于少数位置
    4. 注意力流分析：展示相邻层之间的注意力变化和相关性的双轴图
       蓝色曲线表示注意力变化幅度，橙色曲线表示层间相关性
       可识别注意力在不同层之间的演变规律
    5-8. 不同层注意力热力图（4个示例）：展示各层的平均注意力权重热力图
       横轴为目标位置，纵轴为源位置，颜色深浅表示注意力权重
       可识别各层的注意力模式差异
    9-12. 不同头注意力热力图（4个示例）：展示特定层和头的注意力权重热力图
       每个头可能学习不同的注意力模式（局部、全局、聚焦、均匀）
       标题显示注意力模式分类结果
    13. 头注意力模式分布柱状图：统计所有注意力头的模式分布
       模式类型：local（局部）、global（全局）、focused（聚焦）、uniform（均匀）
       可识别模型学习的主要注意力策略
    14. 最后一层注意力特征雷达图：综合展示最后一层的注意力特征
       包含平均注意力、熵、稀疏度三个维度
       雷达图面积越大表示该层注意力特征越丰富
    15. 头注意力模式对比表格：详细对比每个头的注意力统计信息
       包含层索引、头索引、模式、均值、标准差、最大值、熵
       便于快速查找和对比不同头的注意力特性

    Args:
        attention_results (dict): 深度注意力分析结果字典，包含以下键：
            - 'layer_analysis' (dict): 各层分析结果
                * 结构: {layer_idx: {metrics}}
                * layer_idx (int): 层索引
                * metrics (dict): 包含 'avg_attention', 'entropy', 'attention_sparsity' 等
            - 'head_analysis' (dict): 各头分析结果
                * 结构: {(layer_idx, head_idx): {metrics}}
                * layer_idx (int): 层索引
                * head_idx (int): 头索引
                * metrics (dict): 包含 'attention_weights', 'mean', 'std', 'max', 'entropy', 'pattern' 等
            - 'attention_weights' (dict): 注意力权重字典
                * 结构: {f'layer_{layer_idx}': attention_tensor}
                * attention_tensor (torch.Tensor): 注意力权重张量
            - 'attention_flow' (list): 注意力流分析数据
                * 每个元素包含 'from_layer', 'to_layer', 'attention_change', 'correlation'
        save_path (str): 可视化结果保存路径，如 './attention_analysis/deep_attention.png'
        layers (list): 要可视化的层索引列表，如 [0, 1, 2, 3, 4, 5]
        heads (list): 要可视化的头索引列表，如 [0, 1, 2, 3, 4, 5, 6, 7]
        gt_text (str): 真实文本，用于标注和参考

    Returns:
        None: 函数将可视化结果保存到指定路径，不返回值

    可视化输出：
    - 生成一个4x4的子图布局（最后一个位置为空），包含：
      1. 左上：Average Attention per Layer 折线图
      2. 中上：Attention Entropy per Layer 折线图
      3. 右上：Attention Sparsity per Layer 折线图
      4. 最右上：Attention Flow Between Layers 双轴图
      5-8. 第一行：Layer 0-3 Average Attention 热力图
      9-12. 第二行：Layer-Head Attention 热力图（带模式标注）
      13. 左下：Head Attention Pattern Distribution 柱状图
      14. 中下：Layer N Attention Features 雷达图
      15. 右下：Head Attention Pattern Comparison 表格
    - 图片保存为高分辨率（300 DPI）PNG格式

    使用示例：
        >>> attention_results = {
        ...     'layer_analysis': {
        ...         0: {'avg_attention': tensor(...), 'entropy': tensor(...), 'attention_sparsity': tensor(...)},
        ...         1: {...},
        ...         ...
        ...     },
        ...     'head_analysis': {
        ...         (0, 0): {'attention_weights': tensor(...), 'mean': 0.5, 'std': 0.2, 'pattern': 'local', ...},
        ...         (0, 1): {...},
        ...         ...
        ...     },
        ...     'attention_weights': {
        ...         'layer_0': tensor(...),
        ...         'layer_1': tensor(...),
        ...         ...
        ...     },
        ...     'attention_flow': [
        ...         {'from_layer': 0, 'to_layer': 1, 'attention_change': 0.1, 'correlation': 0.8},
        ...         ...
        ...     ]
        ... }
        >>> visualize_deep_attention(
        ...     attention_results,
        ...     './attention_analysis/deep_attention.png',
        ...     [0, 1, 2, 3, 4, 5],
        ...     [0, 1, 2, 3, 4, 5, 6, 7],
        ...     'hello world'
        ... )
    """
    num_layers = len(layers)
    num_heads = len(heads)

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 1. 多层注意力热力图（平均注意力）
    ax1 = fig.add_subplot(gs[0, 0])
    layer_avg_attentions = []
    for layer_idx in layers:
        if layer_idx in attention_results['layer_analysis']:
            layer_avg_attentions.append(
                attention_results['layer_analysis'][layer_idx]['avg_attention'].mean().item()
            )

    ax1.plot(layers, layer_avg_attentions, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Attention Weight')
    ax1.set_title('Average Attention per Layer')
    ax1.grid(True, alpha=0.3)

    # 2. 注意力熵随层数变化
    ax2 = fig.add_subplot(gs[0, 1])
    layer_entropies = []
    for layer_idx in layers:
        if layer_idx in attention_results['layer_analysis']:
            layer_entropies.append(
                attention_results['layer_analysis'][layer_idx]['entropy'].mean().item()
            )

    ax2.plot(layers, layer_entropies, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Attention Entropy')
    ax2.set_title('Attention Entropy per Layer')
    ax2.grid(True, alpha=0.3)

    # 3. 注意力稀疏度随层数变化
    ax3 = fig.add_subplot(gs[0, 2])
    layer_sparsities = []
    for layer_idx in layers:
        if layer_idx in attention_results['layer_analysis']:
            layer_sparsities.append(
                attention_results['layer_analysis'][layer_idx]['attention_sparsity'].item()
            )

    ax3.plot(layers, layer_sparsities, '^-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Attention Sparsity')
    ax3.set_title('Attention Sparsity per Layer')
    ax3.grid(True, alpha=0.3)

    # 4. 注意力流分析
    ax4 = fig.add_subplot(gs[0, 3])
    if attention_results['attention_flow']:
        flow_data = attention_results['attention_flow']
        from_layers = [f['from_layer'] for f in flow_data]
        changes = [f['attention_change'] for f in flow_data]
        correlations = [f['correlation'] for f in flow_data]

        ax4.plot(from_layers, changes, 'o-', label='Attention Change', linewidth=2)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(from_layers, correlations, 's-', label='Correlation',
                      color='orange', linewidth=2)

        ax4.set_xlabel('Layer Transition')
        ax4.set_ylabel('Attention Change', color='blue')
        ax4_twin.set_ylabel('Correlation', color='orange')
        ax4.set_title('Attention Flow Between Layers')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')

    # 5-8. 显示不同层的注意力热力图（4个示例）
    for i, layer_idx in enumerate(layers[:4]):
        ax = fig.add_subplot(gs[1, i])
        layer_name = f'layer_{layer_idx}'
        if layer_name in attention_results['attention_weights']:
            layer_attn = attention_results['attention_weights'][layer_name]
            avg_attn = layer_attn[0].mean(dim=0)

            im = ax.imshow(avg_attn.cpu().numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'Layer {layer_idx} Average Attention')
            ax.set_xlabel('Target Position')
            ax.set_ylabel('Source Position')
            plt.colorbar(im, ax=ax, label='Attention Weight')

    # 9-12. 显示不同头的注意力热力图（4个示例）
    for i, (layer_idx, head_idx) in enumerate(list(zip(layers[:2], heads[:4]))):
        ax = fig.add_subplot(gs[2, i])
        if (layer_idx, head_idx) in attention_results['head_analysis']:
            head_attn = attention_results['head_analysis'][(layer_idx, head_idx)]['attention_weights']
            pattern = attention_results['head_analysis'][(layer_idx, head_idx)]['pattern']

            im = ax.imshow(head_attn.cpu().numpy(), cmap='plasma', aspect='auto')
            ax.set_title(f'Layer {layer_idx} Head {head_idx}\nPattern: {pattern}')
            ax.set_xlabel('Target Position')
            ax.set_ylabel('Source Position')
            plt.colorbar(im, ax=ax, label='Attention Weight')

    # 13. 头注意力模式分布
    ax13 = fig.add_subplot(gs[3, 0])
    pattern_counts = {'local': 0, 'global': 0, 'focused': 0, 'uniform': 0}
    for key in attention_results['head_analysis']:
        pattern = attention_results['head_analysis'][key]['pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    colors = ['green', 'blue', 'red', 'orange']

    bars = ax13.bar(patterns, counts, color=colors, alpha=0.7)
    ax13.set_xlabel('Attention Pattern')
    ax13.set_ylabel('Count')
    ax13.set_title('Head Attention Pattern Distribution')
    ax13.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax13.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    # 14. 层级注意力特征雷达图
    ax14 = fig.add_subplot(gs[3, 1], projection='polar')
    categories = ['Avg Attention', 'Entropy', 'Sparsity']
    N = len(categories)

    last_layer = layers[-1]
    if last_layer in attention_results['layer_analysis']:
        values = [
            attention_results['layer_analysis'][last_layer]['avg_attention'].mean().item(),
            attention_results['layer_analysis'][last_layer]['entropy'].mean().item(),
            attention_results['layer_analysis'][last_layer]['attention_sparsity'].item()
        ]

        values = [v / max(values) for v in values]

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax14.plot(angles, values, 'o-', linewidth=2)
        ax14.fill(angles, values, alpha=0.25)
        ax14.set_xticks(angles[:-1])
        ax14.set_xticklabels(categories)
        ax14.set_ylim(0, 1)
        ax14.set_title(f'Layer {last_layer} Attention Features')
        ax14.grid(True)

    # 15. 注意力模式对比表格
    ax15 = fig.add_subplot(gs[3, 2:])
    ax15.axis('tight')
    ax15.axis('off')

    table_data = [['Layer', 'Head', 'Pattern', 'Mean', 'Std', 'Max', 'Entropy']]
    for (layer_idx, head_idx) in attention_results['head_analysis'].keys():
        head_data = attention_results['head_analysis'][(layer_idx, head_idx)]
        table_data.append([
            layer_idx,
            head_idx,
            head_data['pattern'],
            f"{head_data['mean']:.3f}",
            f"{head_data['std']:.3f}",
            f"{head_data['max']:.3f}",
            f"{head_data['entropy']:.3f}"
        ])

    table = ax15.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#9C27B0')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax15.set_title('Head Attention Pattern Comparison', pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Deep attention analysis saved → {save_path}')

class MultiModelComparator:
    """多模型对比分析器，用于对比多个OCR模型的性能"""

    def __init__(self, models: list, model_names: list, device: str):
        self.models = models
        self.model_names = model_names
        self.device = device
        assert len(models) == len(model_names), "模型数量和名称数量不匹配"

    @torch.no_grad()
    def compare_models(self, img_tensor: torch.Tensor, gt_text: str,
                       idx2char: dict, blank_id: int) -> dict:
        """对比多个模型在同一输入上的性能"""
        results = {
            'model_results': {},
            'comparison_metrics': {},
            'error_type_comparison': {},
            'confidence_comparison': {},
            'ranking': {}
        }

        for model, model_name in zip(self.models, self.model_names):
            model_output = get_model_output(model, img_tensor.to(self.device),
                                         train_mode='ctc', gt_text=gt_text)

            pred_text = convert_pred_to_text(model_output.pred_id, idx2char, blank_id)
            cer = self._calculate_cer(pred_text, gt_text)
            errors = OCRErrorAnalyzer.analyze_errors(pred_text, gt_text)

            results['model_results'][model_name] = {
                'pred_text': pred_text,
                'gt_text': gt_text,
                'cer': cer,
                'accuracy': 1 - cer,
                'avg_confidence': model_output.prob_values.mean().item(),
                'confidence_distribution': model_output.prob_values.cpu().numpy(),
                'errors': errors,
                'model_output': model_output
            }

        results['comparison_metrics'] = self._calculate_comparison_metrics(
            results['model_results']
        )

        results['error_type_comparison'] = self._compare_error_types(
            results['model_results']
        )

        results['confidence_comparison'] = self._compare_confidence_distributions(
            results['model_results']
        )

        results['ranking'] = self._calculate_ranking(
            results['comparison_metrics']
        )

        return results

    def _calculate_cer(self, pred_text: str, gt_text: str) -> float:
        """计算字符错误率"""
        try:
            from Levenshtein import distance
            return distance(pred_text, gt_text) / max(len(gt_text), 1)
        except ImportError:
            if len(pred_text) != len(gt_text):
                return 1.0
            errors = sum(1 for p, g in zip(pred_text, gt_text) if p != g)
            return errors / max(len(gt_text), 1)

    def _calculate_comparison_metrics(self, model_results: dict) -> dict:
        """计算对比指标"""
        metrics = {
            'cer': {},
            'accuracy': {},
            'avg_confidence': {},
            'min_confidence': {},
            'max_confidence': {},
            'std_confidence': {}
        }

        for model_name, result in model_results.items():
            metrics['cer'][model_name] = result['cer']
            metrics['accuracy'][model_name] = result['accuracy']
            metrics['avg_confidence'][model_name] = result['avg_confidence']
            metrics['min_confidence'][model_name] = result['confidence_distribution'].min()
            metrics['max_confidence'][model_name] = result['confidence_distribution'].max()
            metrics['std_confidence'][model_name] = result['confidence_distribution'].std()

        return metrics

    def _compare_error_types(self, model_results: dict) -> dict:
        """对比错误类型"""
        error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']
        comparison = {error_type: {} for error_type in error_types}

        for model_name, result in model_results.items():
            errors = result['errors']
            for error_type in error_types:
                comparison[error_type][model_name] = errors[error_type + '_errors']

        return comparison

    def _compare_confidence_distributions(self, model_results: dict) -> dict:
        """对比置信度分布"""
        distributions = {}

        for model_name, result in model_results.items():
            conf_dist = result['confidence_distribution']

            distributions[model_name] = {
                'mean': conf_dist.mean(),
                'median': np.median(conf_dist),
                'std': conf_dist.std(),
                'q25': np.percentile(conf_dist, 25),
                'q75': np.percentile(conf_dist, 75)
            }

        return distributions

    def _calculate_ranking(self, comparison_metrics: dict) -> dict:
        """计算模型排名"""
        ranking = {}

        for metric_name in ['cer', 'accuracy', 'avg_confidence']:
            if metric_name == 'cer':
                sorted_models = sorted(comparison_metrics[metric_name].items(),
                                    key=lambda x: x[1])
            else:
                sorted_models = sorted(comparison_metrics[metric_name].items(),
                                    key=lambda x: x[1], reverse=True)

            ranking[metric_name] = {
                model_name: rank + 1
                for rank, (model_name, _) in enumerate(sorted_models)
            }

        overall_scores = {}
        for model_name in self.model_names:
            total_rank = sum(ranking[metric][model_name] for metric in ranking)
            overall_scores[model_name] = total_rank

        ranking['overall'] = {
            model_name: rank + 1
            for rank, (model_name, _) in enumerate(
                sorted(overall_scores.items(), key=lambda x: x[1])
            )
        }

        return ranking

@torch.no_grad()
def visualize_multi_model_comparison(comparison_results: dict, save_path: str):
    """可视化多模型对比分析结果

    可视化原理：
    本函数通过14个子图全面展示多个OCR模型的性能对比：
    1. CER对比柱状图：展示各模型的字符错误率，绿色表示最优，红色表示其他
       CER越低表示性能越好，柱状图顶部显示具体数值
    2. 准确率对比柱状图：展示各模型的准确率，绿色表示最优，红色表示其他
       准确率越高表示性能越好，范围[0, 1]
    3. 平均置信度对比柱状图：展示各模型的平均预测置信度
       置信度越高表示模型越确定，范围[0, 1]
    4. 置信度标准差对比柱状图：展示各模型置信度的稳定性
       标准差越低表示置信度越稳定，绿色表示最优
    5. 错误类型分布堆叠柱状图：展示各模型在不同错误类型上的分布
       包含7种错误类型：替换、删除、插入、交换、空格、大小写、标点
       可识别各模型的错误类型倾向
    6. 置信度分布箱线图：展示各模型置信度的分布情况
       箱线图显示中位数、四分位数、最小值、最大值和异常值
       可对比各模型置信度的集中程度和稳定性
    7. 模型性能雷达图：综合展示各模型在准确率、置信度、稳定性三个维度的表现
       雷达图面积越大表示综合性能越好
       稳定性 = 1 - 置信度标准差
    8. 综合排名柱状图：展示各模型的整体排名
       排名基于CER、准确率、置信度三个指标的综合评分
       金色表示第1名，银色表示第2名，棕色表示其他
    9. 各指标排名热力图：展示各模型在不同指标上的排名
       包含CER、准确率、置信度、综合排名四个指标
       颜色越绿表示排名越好（数字越小），越红表示排名越差
       热力图单元格内显示具体排名数字
    10. 预测结果对比表格：详细对比各模型的预测结果
        包含模型名称、真实文本、预测文本、CER、准确率、置信度、错误摘要
        便于快速查看各模型的具体预测表现
    11-14. 模型预测文本对比（4个示例）：展示每个模型的预测文本
        绿色表示预测正确，红色表示预测错误
        同时显示CER、准确率、置信度等关键指标

    Args:
        comparison_results (dict): 多模型对比分析结果字典，包含以下键：
            - 'model_results' (dict): 各模型的详细结果
                * 结构: {model_name: {metrics}}
                * model_name (str): 模型名称
                * metrics (dict): 包含 'pred_text', 'gt_text', 'cer', 'accuracy',
                  'avg_confidence', 'confidence_distribution', 'errors', 'model_output' 等
            - 'comparison_metrics' (dict): 对比指标汇总
                * 'cer' (dict): 各模型的CER
                * 'accuracy' (dict): 各模型的准确率
                * 'avg_confidence' (dict): 各模型的平均置信度
                * 'min_confidence' (dict): 各模型的最小置信度
                * 'max_confidence' (dict): 各模型的最大置信度
                * 'std_confidence' (dict): 各模型的置信度标准差
            - 'error_type_comparison' (dict): 错误类型对比
                * 结构: {error_type: {model_name: count}}
                * error_type (str): 错误类型，如 'substitution', 'deletion' 等
            - 'confidence_comparison' (dict): 置信度分布对比
                * 结构: {model_name: {distribution_metrics}}
                * distribution_metrics (dict): 包含 'mean', 'median', 'std', 'q25', 'q75' 等
            - 'ranking' (dict): 模型排名
                * 'cer' (dict): CER排名
                * 'accuracy' (dict): 准确率排名
                * 'avg_confidence' (dict): 置信度排名
                * 'overall' (dict): 综合排名
        save_path (str): 可视化结果保存路径，如 './multi_model_comparison/comparison.png'

    Returns:
        None: 函数将可视化结果保存到指定路径，不返回值

    可视化输出：
    - 生成一个4x4的子图布局（最后两个位置为空），包含：
      1. 左上：CER Comparison 柱状图
      2. 中上：Accuracy Comparison 柱状图
      3. 右上：Average Confidence Comparison 柱状图
      4. 最右上：Confidence Stability Comparison 柱状图
      5. 左中：Error Type Distribution Comparison 堆叠柱状图
      6. 中中：Confidence Distribution Comparison 箱线图
      7. 右中：Model Performance Radar Chart 雷达图
      8. 最右中：Model Overall Ranking 柱状图
      9. 左下：Ranking Heatmap 热力图
      10. 中下-右下：Model Prediction Comparison Table 表格
      11-14. 最下一行：Model Prediction Text Comparison 文本对比（4个模型）
    - 图片保存为高分辨率（300 DPI）PNG格式

    使用示例：
        >>> comparison_results = {
        ...     'model_results': {
        ...         'SVTRv2': {'pred_text': 'hello', 'gt_text': 'hello', 'cer': 0.0, 'accuracy': 1.0, ...},
        ...         'ViT': {'pred_text': 'hello', 'gt_text': 'hello', 'cer': 0.0, 'accuracy': 1.0, ...},
        ...         'CNN': {'pred_text': 'helo', 'gt_text': 'hello', 'cer': 0.2, 'accuracy': 0.8, ...}
        ...     },
        ...     'comparison_metrics': {
        ...         'cer': {'SVTRv2': 0.0, 'ViT': 0.0, 'CNN': 0.2},
        ...         'accuracy': {'SVTRv2': 1.0, 'ViT': 1.0, 'CNN': 0.8},
        ...         'avg_confidence': {'SVTRv2': 0.95, 'ViT': 0.93, 'CNN': 0.85},
        ...         'std_confidence': {'SVTRv2': 0.05, 'ViT': 0.07, 'CNN': 0.12}
        ...     },
        ...     'error_type_comparison': {
        ...         'substitution': {'SVTRv2': 0, 'ViT': 0, 'CNN': 1},
        ...         'deletion': {'SVTRv2': 0, 'ViT': 0, 'CNN': 0},
        ...         ...
        ...     },
        ...     'confidence_comparison': {
        ...         'SVTRv2': {'mean': 0.95, 'median': 0.96, 'std': 0.05, ...},
        ...         'ViT': {'mean': 0.93, 'median': 0.94, 'std': 0.07, ...},
        ...         'CNN': {'mean': 0.85, 'median': 0.86, 'std': 0.12, ...}
        ...     },
        ...     'ranking': {
        ...         'cer': {'SVTRv2': 1, 'ViT': 1, 'CNN': 3},
        ...         'accuracy': {'SVTRv2': 1, 'ViT': 1, 'CNN': 3},
        ...         'avg_confidence': {'SVTRv2': 1, 'ViT': 2, 'CNN': 3},
        ...         'overall': {'SVTRv2': 1, 'ViT': 2, 'CNN': 3}
        ...     }
        ... }
        >>> visualize_multi_model_comparison(
        ...     comparison_results,
        ...     './multi_model_comparison/comparison.png'
        ... )
    """
    model_names = list(comparison_results['model_results'].keys())
    num_models = len(model_names)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 1. CER对比柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    cer_values = [comparison_results['comparison_metrics']['cer'][name] for name in model_names]
    colors = ['green' if v == min(cer_values) else 'red' for v in cer_values]
    bars = ax1.bar(model_names, cer_values, color=colors, alpha=0.7)
    ax1.set_ylabel('Character Error Rate (CER)')
    ax1.set_title('CER Comparison')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 准确率对比柱状图
    ax2 = fig.add_subplot(gs[0, 1])
    acc_values = [comparison_results['comparison_metrics']['accuracy'][name] for name in model_names]
    colors = ['green' if v == max(acc_values) else 'red' for v in acc_values]
    bars = ax2.bar(model_names, acc_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. 平均置信度对比柱状图
    ax3 = fig.add_subplot(gs[0, 2])
    conf_values = [comparison_results['comparison_metrics']['avg_confidence'][name]
                  for name in model_names]
    colors = ['green' if v == max(conf_values) else 'red' for v in conf_values]
    bars = ax3.bar(model_names, conf_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Confidence')
    ax3.set_title('Average Confidence Comparison')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. 置信度标准差对比柱状图
    ax4 = fig.add_subplot(gs[0, 3])
    std_values = [comparison_results['comparison_metrics']['std_confidence'][name]
                 for name in model_names]
    colors = ['green' if v == min(std_values) else 'red' for v in std_values]
    bars = ax4.bar(model_names, std_values, color=colors, alpha=0.7)
    ax4.set_ylabel('Confidence Std Dev')
    ax4.set_title('Confidence Stability Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # 5. 错误类型对比堆叠柱状图
    ax5 = fig.add_subplot(gs[1, 0])
    error_types = ['substitution', 'deletion', 'insertion', 'swap', 'space', 'case', 'punctuation']
    error_colors = ['red', 'orange', 'yellow', 'purple', 'blue', 'cyan', 'magenta']

    bottom = np.zeros(num_models)
    for error_type, color in zip(error_types, error_colors):
        error_counts = [comparison_results['error_type_comparison'][error_type][name]
                       for name in model_names]
        ax5.bar(model_names, error_counts, bottom=bottom,
                label=error_type, color=color, alpha=0.7)
        bottom += np.array(error_counts)

    ax5.set_ylabel('Error Count')
    ax5.set_title('Error Type Distribution Comparison')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. 置信度分布对比箱线图
    ax6 = fig.add_subplot(gs[1, 1])
    confidence_data = []
    for model_name in model_names:
        conf_dist = comparison_results['model_results'][model_name]['confidence_distribution']
        confidence_data.append(conf_dist)

    bp = ax6.boxplot(confidence_data, labels=model_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax6.set_ylabel('Confidence')
    ax6.set_title('Confidence Distribution Comparison')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. 模型性能雷达图
    ax7 = fig.add_subplot(gs[1, 2], projection='polar')
    categories = ['Accuracy', 'Confidence', 'Stability']
    N = len(categories)

    model_scores = []
    for model_name in model_names:
        scores = [
            comparison_results['comparison_metrics']['accuracy'][model_name],
            comparison_results['comparison_metrics']['avg_confidence'][model_name],
            1 - comparison_results['comparison_metrics']['std_confidence'][model_name]
        ]
        model_scores.append(scores)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    for i, (model_name, scores) in enumerate(zip(model_names, model_scores)):
        scores_plot = scores + scores[:1]
        angles_plot = angles + angles[:1]
        ax7.plot(angles_plot, scores_plot, 'o-', linewidth=2, label=model_name)
        ax7.fill(angles_plot, scores_plot, alpha=0.15)

    ax7.set_xticks(angles)
    ax7.set_xticklabels(categories)
    ax7.set_ylim(0, 1)
    ax7.set_title('Model Performance Radar Chart')
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax7.grid(True)

    # 8. 综合排名柱状图
    ax8 = fig.add_subplot(gs[1, 3])
    overall_ranks = [comparison_results['ranking']['overall'][name] for name in model_names]
    colors = ['gold' if rank == 1 else 'silver' if rank == 2 else 'brown'
               for rank in overall_ranks]
    bars = ax8.barh(model_names, overall_ranks, color=colors, alpha=0.7)
    ax8.set_xlabel('Overall Rank')
    ax8.set_title('Model Overall Ranking')
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        ax8.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'#{int(width)}', ha='left', va='center', fontweight='bold')

    # 9. 各指标排名热力图
    ax9 = fig.add_subplot(gs[2, 0])
    metrics = ['cer', 'accuracy', 'avg_confidence', 'overall']
    rank_matrix = []

    for metric in metrics:
        row = [comparison_results['ranking'][metric][name] for name in model_names]
        rank_matrix.append(row)

    im = ax9.imshow(rank_matrix, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=num_models)
    ax9.set_xticks(range(num_models))
    ax9.set_xticklabels(model_names)
    ax9.set_yticks(range(len(metrics)))
    ax9.set_yticklabels(['CER', 'Accuracy', 'Confidence', 'Overall'])
    ax9.set_title('Ranking Heatmap (Lower is Better)')
    plt.colorbar(im, ax=ax9, label='Rank')

    for i in range(len(metrics)):
        for j in range(num_models):
            text = ax9.text(j, i, str(rank_matrix[i][j]),
                           ha="center", va="center", color="black", fontweight='bold')

    # 10. 预测结果对比表格
    ax10 = fig.add_subplot(gs[2, 1:])
    ax10.axis('tight')
    ax10.axis('off')

    table_data = [['Model', 'GT Text', 'Pred Text', 'CER', 'Accuracy', 'Confidence', 'Errors']]
    for model_name in model_names:
        result = comparison_results['model_results'][model_name]
        errors = result['errors']
        error_summary = f"Sub:{errors['substitution_errors']} " \
                       f"Del:{errors['deletion_errors']} " \
                       f"Ins:{errors['insertion_errors']}"
        table_data.append([
            model_name,
            result['gt_text'][:20] + '...' if len(result['gt_text']) > 20 else result['gt_text'],
            result['pred_text'][:20] + '...' if len(result['pred_text']) > 20 else result['pred_text'],
            f"{result['cer']:.3f}",
            f"{result['accuracy']:.3f}",
            f"{result['avg_confidence']:.3f}",
            error_summary
        ])

    table = ax10.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#FF5722')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax10.set_title('Model Prediction Comparison Table', pad=20)

    # 11-14. 显示每个模型的预测文本对比（4个示例）
    for i, model_name in enumerate(model_names[:4]):
        ax = fig.add_subplot(gs[3, i])
        result = comparison_results['model_results'][model_name]

        gt_text = result['gt_text']
        pred_text = result['pred_text']

        ax.text(0.5, 0.7, f'GT: {gt_text}',
               ha='center', va='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)

        ax.text(0.5, 0.5, f'Pred: {pred_text}',
               ha='center', va='center', fontsize=10,
               color='green' if pred_text == gt_text else 'red',
               transform=ax.transAxes)

        ax.text(0.5, 0.3,
               f'CER: {result["cer"]:.3f} | Acc: {result["accuracy"]:.3f} | Conf: {result["avg_confidence"]:.3f}',
               ha='center', va='center', fontsize=9,
               transform=ax.transAxes)

        ax.set_title(f'{model_name}', fontweight='bold')
        ax.axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Multi-model comparison saved → {save_path}')


@torch.no_grad()
def debug_virtual_alignment(device: str, model: torch.nn.Module, loader: DataLoader, epoch: int, backbone: str, train_mode: str, output_dir: str,
                           blank_id: int = None, eos_id: int = None, idx2char: dict = None, sos_id: int = None, max_length: int = 70,
                           enable_robustness_analysis: bool = False,
                           enable_training_analysis: bool = False,
                           enable_augmentation_analysis: bool = False,
                           enable_deep_attention_analysis: bool = False,
                           enable_multi_model_comparison: bool = False):
    """调试对齐可视化的综合函数（增强版）

    该函数根据指定的主干网络类型和训练模式，调用相应的对齐可视化函数。
    支持多种主干网络和训练模式的组合：
    - SVTRv2主干 + CTC训练模式：生成四种风格的SVTRv2可视化
    - ViT主干 + CTC训练模式：生成四种风格的ViT可视化
    - 其他CNN主干 + CTC训练模式：生成CNN风格的可视化
    - AR训练模式：生成AR对齐可视化
    - 混合训练模式：生成适合的可视化
    - 鲁棒性分析：分析模型在不同噪声干扰下的性能
    - 训练过程分析：分析训练过程中的性能变化
    - 数据增强效果分析：分析不同增强策略的效果
    - 注意力机制深度分析：分析Transformer的多层多头注意力
    - 多模型对比分析：对比多个模型的性能

    Args:
        device (str): 设备名称，如 'cuda' 或 'cpu'
        model (torch.nn.Module): 完整的OCR模型，包含backbone、neck和decoder
        loader (DataLoader): 数据加载器，用于获取测试图像
        epoch (int): 当前训练轮次，用于命名输出文件
        backbone (str): 主干网络类型，如 'svtrv2', 'vit', 'resnet'
        train_mode (str): 训练模式，如 'ctc', 'ar', 'hybrid'
        output_dir (str): 输出目录，用于保存可视化图像
        blank_id (int, optional): 空白符ID，默认为None
        eos_id (int, optional): 结束符ID，默认为None
        idx2char (dict, optional): 字符ID到字符的映射字典，默认为None
        sos_id (int, optional): 起始符ID，默认为None
        max_length (int, optional): 最大文本长度，默认为70
        enable_robustness_analysis (bool, optional): 是否启用鲁棒性分析，默认为False
        enable_training_analysis (bool, optional): 是否启用训练过程分析，默认为False
        enable_augmentation_analysis (bool, optional): 是否启用数据增强效果分析，默认为False
        enable_deep_attention_analysis (bool, optional): 是否启用深度注意力分析，默认为False
        enable_multi_model_comparison (bool, optional): 是否启多模型对比分析，默认为False
    """
    # 创建模型配置字典
    model_config = {
        'blank_id': blank_id,
        'eos_id': eos_id,
        'idx2char': idx2char,
        'sos_id': sos_id
    }

    # 创建可视化配置
    config = VisualizationConfig(max_length=max_length)

    # 根据新参数启用相应功能
    if enable_robustness_analysis:
        config.robustness_analysis = True
        config.robustness_save_dir = f'{output_dir}/robustness_analysis'

    if enable_training_analysis:
        config.training_analysis = True
        config.training_save_dir = f'{output_dir}/training_analysis'

    if enable_augmentation_analysis:
        config.augmentation_analysis = True
        config.augmentation_save_dir = f'{output_dir}/augmentation_analysis'

    if enable_deep_attention_analysis:
        config.deep_attention_analysis = True
        config.attention_save_dir = f'{output_dir}/attention_analysis'

    if enable_multi_model_comparison and config.models_to_compare:
        config.multi_model_comparison = True
        config.comparison_save_dir = f'{output_dir}/multi_model_comparison'

    use_ctc = train_mode == 'ctc' or train_mode == 'hybrid'
    plt.rcParams['text.usetex'] = False

    # 根据主干网络和训练模式选择可视化方法
    if use_ctc:
        if 'svtrv2' in backbone.lower():
            debug_svtrv2_alignment(device, model, loader, epoch, output_dir, model_config, config)
        elif 'vit' in backbone.lower():
            debug_vit_alignment(device, model, loader, epoch, output_dir, model_config, config)
        elif 'viptr2' in backbone.lower():
            # VIPTRNetV2使用OSRA（One-Shot Relative Attention）机制，输出结构与ViT兼容
            debug_viptrv2_alignment(device, model, loader, epoch, output_dir, model_config, config)
        else:
            debug_cnn_alignment(device, model, loader, epoch, output_dir, model_config, config)
    else:
        debug_ar_alignment(device, model, loader, epoch, output_dir, model_config, config)

    # 运行新增的分析功能
    if config.robustness_analysis:
        _run_robustness_analysis(device, model, loader, idx2char, blank_id, config)

    if config.training_analysis:
        _run_training_analysis(device, model, loader, idx2char, blank_id, config, epoch)

    if config.augmentation_analysis:
        _run_augmentation_analysis(device, model, loader, idx2char, blank_id, config)

    if config.deep_attention_analysis and train_mode == 'ar':
        _run_deep_attention_analysis(device, model, loader, idx2char, config)

    if config.multi_model_comparison and config.models_to_compare:
        _run_multi_model_comparison(device, config.models_to_compare, config.model_names,
                                   loader, idx2char, blank_id, config)
