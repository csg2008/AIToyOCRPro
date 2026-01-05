"""
量化配置管理器 - 动态调整量化参数和策略
提供多种量化策略和优化选项
"""
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.export import export
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import optuna
from optuna import Trial
from functools import reduce
import operator
import itertools
import warnings
from torchao.quantization import (
    Int8DynActInt4WeightLinear,
    Int8DynActInt4WeightQuantizer,
    quantize_,
    int8_dynamic_activation_int4_weight,
    int8_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
)
from torchao.quantization.qat import (
    Int8DynActInt4WeightQATQuantizer,
    Int4WeightOnlyQATQuantizer,
)

class PruningConfig:
    """剪枝配置类"""
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', False)
        self.pruning_strategy = config.get('pruning_strategy', 'l1_unstructured')
        self.pruning_ratio = config.get('pruning_ratio', 0.3)
        self.pruning_layers = config.get('pruning_layers', ['backbone', 'neck', 'decoder'])
        self.pruning_epoch = config.get('pruning_epoch', 20)
        self.min_acc_drop = config.get('min_acc_drop', 0.01)
        self.finetune_epochs = config.get('finetune_epochs', 10)
        self.prune_criteria = config.get('prune_criteria', 'l1')

        # 分层剪枝比例
        self.layer_specific_ratios = config.get('layer_specific_ratios', {
            'backbone': 0.2,
            'neck': 0.3,
            'decoder': 0.1
        })

        # 剪枝方法参数
        self.prune_params = config.get('prune_params', {
            'n': 1,
            'dim': 0
        })

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'pruning_strategy': self.pruning_strategy,
            'pruning_ratio': self.pruning_ratio,
            'pruning_layers': self.pruning_layers,
            'pruning_epoch': self.pruning_epoch,
            'min_acc_drop': self.min_acc_drop,
            'finetune_epochs': self.finetune_epochs,
            'prune_criteria': self.prune_criteria,
            'layer_specific_ratios': self.layer_specific_ratios,
            'prune_params': self.prune_params
        }

class PruningManager:
    """剪枝管理器"""
    def __init__(self, config: PruningConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.pruned_layers = []
        self.original_model = None
        self.pruning_applied = False

    def get_pruning_ratio(self, layer_name: str) -> float:
        """获取特定层的剪枝比例"""
        for layer_type, ratio in self.config.layer_specific_ratios.items():
            if layer_type in layer_name:
                return ratio
        return self.config.pruning_ratio

    def apply_pruning(self, epoch: int, current_acc: float, best_acc: float) -> bool:
        """应用剪枝"""
        if not self.config.enabled:
            return False

        # 检查是否达到剪枝条件
        if epoch != self.config.pruning_epoch:
            return False

        # 检查精度是否足够高
        if current_acc < best_acc * 0.95:
            return False

        print(f"🎯 开始剪枝...")
        print(f"   - 当前精度: {current_acc:.4f}")
        print(f"   - 最佳精度: {best_acc:.4f}")
        print(f"   - 剪枝策略: {self.config.pruning_strategy}")

        # 保存原始模型
        self.original_model = self.model.state_dict()

        # 应用剪枝
        if self.config.pruning_strategy == 'l1_unstructured':
            self._apply_l1_unstructured_pruning()
        elif self.config.pruning_strategy == 'l1_structured':
            self._apply_l1_structured_pruning()
        elif self.config.pruning_strategy == 'ln_structured':
            self._apply_ln_structured_pruning()

        self.pruning_applied = True
        print(f"✅ 剪枝完成")
        return True

    def _apply_l1_unstructured_pruning(self):
        """应用L1非结构化剪枝"""
        # 遍历模型的所有层
        for name, module in self.model.named_modules():
            # 只剪枝卷积层和全连接层
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 检查是否在指定的剪枝层中
                if any(layer_type in name for layer_type in self.config.pruning_layers):
                    pruning_ratio = self.get_pruning_ratio(name)
                    print(f"   - 剪枝层: {name}, 剪枝比例: {pruning_ratio:.2f}")

                    # 应用L1非结构化剪枝
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    self.pruned_layers.append((name, module))

    def _apply_l1_structured_pruning(self):
        """应用L1结构化剪枝"""
        # 遍历模型的所有层
        for name, module in self.model.named_modules():
            # 只剪枝卷积层和全连接层
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 检查是否在指定的剪枝层中
                if any(layer_type in name for layer_type in self.config.pruning_layers):
                    pruning_ratio = self.get_pruning_ratio(name)
                    print(f"   - 剪枝层: {name}, 剪枝比例: {pruning_ratio:.2f}")

                    # 应用L1结构化剪枝
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
                    self.pruned_layers.append((name, module))

    def _apply_ln_structured_pruning(self):
        """应用LN结构化剪枝"""
        # 遍历模型的所有层
        for name, module in self.model.named_modules():
            # 只剪枝卷积层和全连接层
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 检查是否在指定的剪枝层中
                if any(layer_type in name for layer_type in self.config.pruning_layers):
                    pruning_ratio = self.get_pruning_ratio(name)
                    print(f"   - 剪枝层: {name}, 剪枝比例: {pruning_ratio:.2f}")

                    # 应用LN结构化剪枝
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, **self.config.prune_params)
                    self.pruned_layers.append((name, module))

    def remove_pruning(self):
        """移除剪枝包装，使剪枝永久化"""
        if not self.pruning_applied:
            return

        print(f"🔧 永久化剪枝...")
        for name, module in self.pruned_layers:
            prune.remove(module, 'weight')
        print(f"✅ 剪枝永久化完成")

    def restore_original_model(self):
        """恢复原始模型"""
        if self.original_model is not None:
            self.model.load_state_dict(self.original_model)
            self.pruning_applied = False
            self.pruned_layers = []
            print(f"🔄 恢复原始模型完成")

    def calculate_pruning_ratio(self) -> float:
        """计算实际剪枝比例"""
        if not self.pruning_applied:
            return 0.0

        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 检查是否有weight_mask属性（剪枝后的标志）
                if hasattr(module, 'weight_mask'):
                    weight = module.weight
                    mask = module.weight_mask
                    total_params += weight.numel()
                    pruned_params += (mask == 0).sum().item()
                else:
                    weight = module.weight
                    total_params += weight.numel()

        if total_params == 0:
            return 0.0

        return pruned_params / total_params

    def get_pruned_model_info(self) -> Dict:
        """获取剪枝后的模型信息"""
        info = {
            'pruning_applied': self.pruning_applied,
            'pruned_layers_count': len(self.pruned_layers),
            'pruning_ratio': self.calculate_pruning_ratio(),
            'pruned_layers': [name for name, _ in self.pruned_layers]
        }
        return info

    def is_pruning_time(self, epoch: int) -> bool:
        """检查是否到达剪枝时间"""
        return self.config.enabled and epoch == self.config.pruning_epoch

    def is_finetuning_time(self, epoch: int) -> bool:
        """检查是否处于微调阶段"""
        if not self.config.enabled or not self.pruning_applied:
            return False

        start_finetune_epoch = self.config.pruning_epoch + 1
        end_finetune_epoch = start_finetune_epoch + self.config.finetune_epochs

        return start_finetune_epoch <= epoch < end_finetune_epoch

    def get_finetune_lr_multiplier(self, epoch: int) -> float:
        """获取微调阶段的学习率乘数"""
        if not self.is_finetuning_time(epoch):
            return 1.0

        # 微调阶段使用较低的学习率
        return 0.1

def create_pruning_config(args, config_file: Optional[str] = None) -> PruningConfig:
    """创建剪枝配置"""
    config = {
        'enabled': args.enable_pruning,
        'pruning_strategy': args.pruning_strategy,
        'pruning_ratio': args.pruning_ratio,
        'pruning_layers': args.pruning_layers,
        'pruning_epoch': args.pruning_epoch,
        'min_acc_drop': args.min_acc_drop,
        'finetune_epochs': args.finetune_epochs,
        'prune_criteria': args.prune_criteria,
        'layer_specific_ratios': {
            'backbone': args.backbone_pruning_ratio,
            'neck': args.neck_pruning_ratio,
            'decoder': args.decoder_pruning_ratio
        }
    }

    return PruningConfig(config)

class QuantizationStrategy(Enum):
    """量化策略枚举"""
    INT8_DYN_ACT_INT4_WEIGHT = "int8_dyn_act_int4_weight"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    INT4_WEIGHT_ONLY = "int4_weight_only"
    INT8_DYNAMIC_ACT_INT8_WEIGHT = "int8_dynamic_activation_int8_weight"
    MIXED_PRECISION = "mixed_precision"

class ObserverType(Enum):
    """观察器类型枚举"""
    MOVING_AVERAGE = "moving_average"
    MIN_MAX = "min_max"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"

class QuantizationGranularity(Enum):
    """量化粒度枚举"""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"

@dataclass
class QuantizationConfig:
    """量化配置数据类"""

    # 基础配置
    enabled: bool = True
    strategy: QuantizationStrategy = QuantizationStrategy.INT8_DYN_ACT_INT4_WEIGHT
    quantization_aware_training: bool = True
    post_training_quantization: bool = False

    # 训练配置
    qat_epochs: int = 5
    ptq_epochs: int = 2
    calibration_batches: int = 100
    qat_learning_rate_multiplier: float = 0.1

    # 量化位数配置
    weight_bits: int = 4
    activation_bits: int = 8
    mixed_precision: bool = True

    # 观察器配置
    observer_type: ObserverType = ObserverType.MOVING_AVERAGE
    observer_momentum: float = 0.1
    quantization_granularity: QuantizationGranularity = QuantizationGranularity.PER_CHANNEL

    # 层配置
    quantization_layers: List[str] = field(default_factory=lambda: ['linear', 'conv2d', 'attention'])
    excluded_layers: List[str] = field(default_factory=lambda: ['embedding', 'layernorm', 'batchnorm'])

    # 损失配置
    quantization_loss_weight: float = 0.01
    temperature_distillation: float = 4.0
    distillation_weight: float = 0.3

    # 高级配置
    dynamic_quantization: bool = False
    static_quantization: bool = True
    symmetric_quantization: bool = True
    clipping_threshold: float = 1.0

    # 性能优化
    enable_cuda_graphs: bool = False
    memory_efficient: bool = True
    compile_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'strategy': self.strategy.value,
            'quantization_aware_training': self.quantization_aware_training,
            'post_training_quantization': self.post_training_quantization,
            'qat_epochs': self.qat_epochs,
            'ptq_epochs': self.ptq_epochs,
            'calibration_batches': self.calibration_batches,
            'qat_learning_rate_multiplier': self.qat_learning_rate_multiplier,
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits,
            'mixed_precision': self.mixed_precision,
            'observer_type': self.observer_type.value,
            'observer_momentum': self.observer_momentum,
            'quantization_granularity': self.quantization_granularity.value,
            'quantization_layers': self.quantization_layers,
            'excluded_layers': self.excluded_layers,
            'quantization_loss_weight': self.quantization_loss_weight,
            'temperature_distillation': self.temperature_distillation,
            'distillation_weight': self.distillation_weight,
            'dynamic_quantization': self.dynamic_quantization,
            'static_quantization': self.static_quantization,
            'symmetric_quantization': self.symmetric_quantization,
            'clipping_threshold': self.clipping_threshold,
            'enable_cuda_graphs': self.enable_cuda_graphs,
            'memory_efficient': self.memory_efficient,
            'compile_model': self.compile_model,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """从字典创建配置"""
        config = cls()

        # 基础配置
        config.enabled = config_dict.get('enabled', True)
        config.strategy = QuantizationStrategy(config_dict.get('strategy', 'int8_dyn_act_int4_weight'))
        config.quantization_aware_training = config_dict.get('quantization_aware_training', True)
        config.post_training_quantization = config_dict.get('post_training_quantization', False)

        # 训练配置
        config.qat_epochs = config_dict.get('qat_epochs', 5)
        config.ptq_epochs = config_dict.get('ptq_epochs', 2)
        config.calibration_batches = config_dict.get('calibration_batches', 100)
        config.qat_learning_rate_multiplier = config_dict.get('qat_learning_rate_multiplier', 0.1)

        # 量化位数配置
        config.weight_bits = config_dict.get('weight_bits', 4)
        config.activation_bits = config_dict.get('activation_bits', 8)
        config.mixed_precision = config_dict.get('mixed_precision', True)

        # 观察器配置
        config.observer_type = ObserverType(config_dict.get('observer_type', 'moving_average'))
        config.observer_momentum = config_dict.get('observer_momentum', 0.1)
        config.quantization_granularity = QuantizationGranularity(
            config_dict.get('quantization_granularity', 'per_channel')
        )

        # 层配置
        config.quantization_layers = config_dict.get('quantization_layers', ['linear', 'conv2d', 'attention'])
        config.excluded_layers = config_dict.get('excluded_layers', ['embedding', 'layernorm', 'batchnorm'])

        # 损失配置
        config.quantization_loss_weight = config_dict.get('quantization_loss_weight', 0.01)
        config.temperature_distillation = config_dict.get('temperature_distillation', 4.0)
        config.distillation_weight = config_dict.get('distillation_weight', 0.3)

        # 高级配置
        config.dynamic_quantization = config_dict.get('dynamic_quantization', False)
        config.static_quantization = config_dict.get('static_quantization', True)
        config.symmetric_quantization = config_dict.get('symmetric_quantization', True)
        config.clipping_threshold = config_dict.get('clipping_threshold', 1.0)

        # 性能优化
        config.enable_cuda_graphs = config_dict.get('enable_cuda_graphs', False)
        config.memory_efficient = config_dict.get('memory_efficient', True)
        config.compile_model = config_dict.get('compile_model', False)

        return config

    def save_to_file(self, filepath: str, format: str = 'json'):
        """保存配置到文件"""
        config_dict = self.to_dict()

        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() in ['yaml', 'yml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的格式: {format}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'QuantizationConfig':
        """从文件加载配置"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif filepath.endswith(('.yaml', '.yml')):
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")

        return cls.from_dict(config_dict)

class AdaptiveQuantizationConfig:
    """自适应量化配置 - 根据模型和任务动态调整参数"""

    def __init__(self, model: nn.Module, task_type: str = 'ocr'):
        self.model = model
        self.task_type = task_type
        self.base_config = QuantizationConfig()

    def analyze_model_structure(self) -> Dict[str, Any]:
        """分析模型结构，为量化配置提供建议"""
        analysis = {
            'total_params': 0,
            'linear_layers': 0,
            'conv_layers': 0,
            'attention_layers': 0,
            'embedding_layers': 0,
            'layer_norm_layers': 0,
            'largest_layer_params': 0,
            'model_size_mb': 0,
        }

        total_params = 0
        layer_counts = {
            'linear': 0,
            'conv': 0,
            'attention': 0,
            'embedding': 0,
            'layernorm': 0,
        }

        largest_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_counts['linear'] += 1
                params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                total_params += params
                largest_params = max(largest_params, params)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layer_counts['conv'] += 1
                params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                total_params += params
                largest_params = max(largest_params, params)
            elif isinstance(module, nn.MultiheadAttention):
                layer_counts['attention'] += 1
                # 估算注意力层参数量
                params = 4 * module.embed_dim * module.embed_dim  # Q, K, V, O 投影
                total_params += params
                largest_params = max(largest_params, params)
            elif isinstance(module, nn.Embedding):
                layer_counts['embedding'] += 1
                params = module.num_embeddings * module.embedding_dim
                total_params += params
            elif isinstance(module, nn.LayerNorm):
                layer_counts['layernorm'] += 1
                params = 2 * module.normalized_shape[0]  # weight + bias
                total_params += params

        analysis['total_params'] = total_params
        analysis['linear_layers'] = layer_counts['linear']
        analysis['conv_layers'] = layer_counts['conv']
        analysis['attention_layers'] = layer_counts['attention']
        analysis['embedding_layers'] = layer_counts['embedding']
        analysis['layer_norm_layers'] = layer_counts['layernorm']
        analysis['largest_layer_params'] = largest_params
        analysis['model_size_mb'] = total_params * 4 / (1024 * 1024)  # float32

        return analysis

    def recommend_config(self, target_compression_ratio: float = 0.25,
                        preserve_accuracy: bool = True) -> QuantizationConfig:
        """根据模型分析结果推荐量化配置"""
        analysis = self.analyze_model_structure()

        config = QuantizationConfig()

        # 根据模型大小调整策略
        model_size_mb = analysis['model_size_mb']
        if model_size_mb < 10:  # 小模型
            config.weight_bits = 8
            config.activation_bits = 8
            config.qat_epochs = 3
            config.quantization_loss_weight = 0.005
        elif model_size_mb < 50:  # 中等模型
            config.weight_bits = 4
            config.activation_bits = 8
            config.qat_epochs = 5
            config.quantization_loss_weight = 0.01
        else:  # 大模型
            config.weight_bits = 4
            config.activation_bits = 4
            config.qat_epochs = 8
            config.quantization_loss_weight = 0.02

        # 根据任务类型调整
        if self.task_type == 'ocr':
            # OCR任务对精度要求高，使用保守的量化策略
            config.strategy = QuantizationStrategy.INT8_DYN_ACT_INT4_WEIGHT
            config.observer_type = ObserverType.MOVING_AVERAGE
            config.quantization_granularity = QuantizationGranularity.PER_CHANNEL
            config.temperature_distillation = 6.0
            config.distillation_weight = 0.5
        elif self.task_type == 'classification':
            # 分类任务可以使用更激进的量化
            config.strategy = QuantizationStrategy.INT4_WEIGHT_ONLY
            config.observer_type = ObserverType.MIN_MAX
            config.quantization_granularity = QuantizationGranularity.PER_TENSOR
        elif self.task_type == 'detection':
            # 检测任务平衡精度和速度
            config.strategy = QuantizationStrategy.INT8_WEIGHT_ONLY
            config.observer_type = ObserverType.PERCENTILE
            config.quantization_granularity = QuantizationGranularity.PER_CHANNEL

        # 根据压缩比目标调整
        if target_compression_ratio < 0.2:  # 高压缩比
            config.weight_bits = 4
            config.activation_bits = 4
            config.qat_epochs = max(8, config.qat_epochs)
        elif target_compression_ratio > 0.5:  # 低压缩比
            config.weight_bits = 8
            config.activation_bits = 8
            config.qat_epochs = min(3, config.qat_epochs)

        # 根据精度要求调整
        if preserve_accuracy:
            config.qat_learning_rate_multiplier = 0.05  # 更低的学习率
            config.quantization_loss_weight = 0.02  # 更高的量化损失权重
            config.temperature_distillation = 8.0  # 更高的蒸馏温度
            config.distillation_weight = 0.7  # 更高的蒸馏权重

        return config

    def optimize_config_for_hardware(self, hardware_target: str = 'cpu') -> QuantizationConfig:
        """针对特定硬件优化配置"""
        config = self.recommend_config()

        if hardware_target == 'cpu':
            # CPU优化：使用对称量化，减少计算复杂度
            config.symmetric_quantization = True
            config.quantization_granularity = QuantizationGranularity.PER_TENSOR
            config.enable_cuda_graphs = False
            config.memory_efficient = True
        elif hardware_target == 'gpu':
            # GPU优化：使用通道级量化，提高精度
            config.symmetric_quantization = False
            config.quantization_granularity = QuantizationGranularity.PER_CHANNEL
            config.enable_cuda_graphs = True
            config.memory_efficient = False
        elif hardware_target == 'mobile':
            # 移动端优化：激进的量化策略
            config.weight_bits = 4
            config.activation_bits = 4
            config.symmetric_quantization = True
            config.quantization_granularity = QuantizationGranularity.PER_TENSOR
            config.memory_efficient = True
            config.compile_model = True

        return config

class QuantizationConfigValidator:
    """量化配置验证器 - 验证配置的有效性和兼容性"""

    @staticmethod
    def validate_config(config: QuantizationConfig) -> List[str]:
        """验证配置并返回错误信息列表"""
        errors = []

        # 基础验证
        if not config.enabled:
            return errors

        # 训练策略验证
        if config.quantization_aware_training and config.post_training_quantization:
            errors.append("不能同时启用QAT和PTQ")

        if not config.quantization_aware_training and not config.post_training_quantization:
            errors.append("必须选择QAT或PTQ中的一种训练策略")

        # 量化位数验证
        if config.weight_bits not in [1, 2, 4, 8]:
            errors.append("权重量化位数必须是1, 2, 4, 或 8")

        if config.activation_bits not in [1, 2, 4, 8, 16]:
            errors.append("激活量化位数必须是1, 2, 4, 8, 或 16")

        # 训练轮数验证
        if config.qat_epochs < 1:
            errors.append("QAT训练轮数必须大于0")

        if config.ptq_epochs < 1:
            errors.append("PTQ训练轮数必须大于0")

        # 校准批次验证
        if config.calibration_batches < 10:
            errors.append("校准批次数量至少为10")

        # 超参数范围验证
        if not (0.001 <= config.qat_learning_rate_multiplier <= 1.0):
            errors.append("QAT学习率倍数必须在0.001到1.0之间")

        if not (0.0 <= config.quantization_loss_weight <= 1.0):
            errors.append("量化损失权重必须在0.0到1.0之间")

        if not (1.0 <= config.temperature_distillation <= 20.0):
            errors.append("蒸馏温度必须在1.0到20.0之间")

        if not (0.0 <= config.distillation_weight <= 1.0):
            errors.append("蒸馏权重必须在0.0到1.0之间")

        # 观察器动量验证
        if not (0.0 <= config.observer_momentum <= 1.0):
            errors.append("观察器动量必须在0.0到1.0之间")

        # 裁剪阈值验证
        if config.clipping_threshold <= 0:
            errors.append("裁剪阈值必须大于0")

        return errors

    @staticmethod
    def warn_config(config: QuantizationConfig) -> List[str]:
        """检查配置并返回警告信息列表"""
        warnings = []

        if not config.enabled:
            return warnings

        # 精度警告
        if config.weight_bits < 4 and config.activation_bits < 8:
            warnings.append("低量化位数可能导致显著的精度损失")

        # 训练警告
        if config.qat_epochs < 3:
            warnings.append("QAT训练轮数较少，可能无法充分优化量化参数")

        if config.calibration_batches < 50:
            warnings.append("校准批次较少，可能影响量化精度")

        # 性能警告
        if config.enable_cuda_graphs and config.quantization_granularity == QuantizationGranularity.PER_CHANNEL:
            warnings.append("CUDA图与通道级量化同时使用可能影响性能")

        # 超参数警告
        if config.qat_learning_rate_multiplier > 0.5:
            warnings.append("QAT学习率倍数较高，可能导致训练不稳定")

        if config.quantization_loss_weight > 0.1:
            warnings.append("量化损失权重较高，可能影响主任务性能")

        if config.temperature_distillation > 10.0:
            warnings.append("蒸馏温度过高，可能导致知识蒸馏效果不佳")

        return warnings

@dataclass
class QuantizationMetrics:
    """量化评估指标"""
    # 精度指标
    original_accuracy: float = 0.0
    quantized_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    accuracy_drop_ratio: float = 0.0

    # 模型大小指标
    original_model_size_mb: float = 0.0
    quantized_model_size_mb: float = 0.0
    compression_ratio: float = 0.0
    size_reduction_ratio: float = 0.0

    # 推理速度指标
    original_inference_time_ms: float = 0.0
    quantized_inference_time_ms: float = 0.0
    speedup_ratio: float = 0.0

    # 内存使用指标
    original_memory_usage_mb: float = 0.0
    quantized_memory_usage_mb: float = 0.0
    memory_reduction_ratio: float = 0.0

    # 计算复杂度指标
    original_flops: int = 0
    quantized_flops: int = 0
    flops_reduction_ratio: float = 0.0

    # 量化误差指标
    mse_error: float = 0.0
    mae_error: float = 0.0
    snr_db: float = 0.0
    psnr_db: float = 0.0

    # 激活分布指标
    weight_clipping_ratio: float = 0.0
    activation_clipping_ratio: float = 0.0
    quantization_noise_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'accuracy': {
                'original': self.original_accuracy,
                'quantized': self.quantized_accuracy,
                'drop': self.accuracy_drop,
                'drop_ratio': self.accuracy_drop_ratio,
            },
            'model_size': {
                'original_mb': self.original_model_size_mb,
                'quantized_mb': self.quantized_model_size_mb,
                'compression_ratio': self.compression_ratio,
                'reduction_ratio': self.size_reduction_ratio,
            },
            'inference_speed': {
                'original_time_ms': self.original_inference_time_ms,
                'quantized_time_ms': self.quantized_inference_time_ms,
                'speedup_ratio': self.speedup_ratio,
            },
            'memory_usage': {
                'original_mb': self.original_memory_usage_mb,
                'quantized_mb': self.quantized_memory_usage_mb,
                'reduction_ratio': self.memory_reduction_ratio,
            },
            'computational_complexity': {
                'original_flops': self.original_flops,
                'quantized_flops': self.quantized_flops,
                'reduction_ratio': self.flops_reduction_ratio,
            },
            'quantization_error': {
                'mse': self.mse_error,
                'mae': self.mae_error,
                'snr_db': self.snr_db,
                'psnr_db': self.psnr_db,
            },
            'activation_distribution': {
                'weight_clipping_ratio': self.weight_clipping_ratio,
                'activation_clipping_ratio': self.activation_clipping_ratio,
                'quantization_noise_std': self.quantization_noise_std,
            }
        }

class ModelAnalyzer:
    """模型分析器 - 分析模型结构和复杂度"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_info = []

    def analyze_model_structure(self) -> Dict[str, Any]:
        """分析模型结构"""
        analysis = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'total_layers': 0,
            'layer_types': {},
            'parameter_distribution': {},
            'memory_usage_mb': 0,
            'model_size_mb': 0,
        }

        total_params = 0
        trainable_params = 0
        layer_types = {}
        param_distribution = {}

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                layer_type = module.__class__.__name__
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

                # 计算参数数量
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                total_params += params
                trainable_params += trainable

                if params > 0:
                    param_distribution[name] = {
                        'type': layer_type,
                        'parameters': params,
                        'trainable': trainable,
                        'size_mb': params * 4 / (1024 * 1024)  # float32
                    }

        analysis['total_parameters'] = total_params
        analysis['trainable_parameters'] = trainable_params
        analysis['total_layers'] = sum(layer_types.values())
        analysis['layer_types'] = layer_types
        analysis['parameter_distribution'] = param_distribution
        analysis['memory_usage_mb'] = self._estimate_memory_usage()
        analysis['model_size_mb'] = total_params * 4 / (1024 * 1024)

        return analysis

    def _estimate_memory_usage(self) -> float:
        """估算模型内存使用"""
        # 考虑模型参数、梯度和优化器状态
        param_size = sum(p.numel() for p in self.model.parameters()) * 4  # float32
        grad_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad) * 4
        optimizer_size = grad_size * 2  # Adam优化器有动量和方差

        total_bytes = param_size + grad_size + optimizer_size
        return total_bytes / (1024 * 1024)  # MB

    def count_flops(self, input_shape: Tuple[int, ...]) -> int:
        """估算模型FLOPs"""
        # 简化的FLOPs估算
        flops = 0

        def hook_fn(module, input, output):
            nonlocal flops

            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs估算
                batch_size = input[0].shape[0]
                output_shape = output.shape
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = batch_size * output_shape[2] * output_shape[3] * module.out_channels
                flops += kernel_flops * output_elements

            elif isinstance(module, nn.Linear):
                # Linear FLOPs估算
                flops += module.in_features * module.out_features * input[0].shape[0]

        hooks = []
        for module in self.model.modules():
            hooks.append(module.register_forward_hook(hook_fn))

        # 运行一次前向传播
        dummy_input = torch.randn(input_shape)
        self.model(dummy_input)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        return flops

class QuantizationEvaluator:
    """量化评估器 - 全面评估量化效果"""

    def __init__(self, original_model: nn.Module, quantized_model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.original_model = original_model
        self.quantized_model = quantized_model
        self.device = device
        self.analyzer = ModelAnalyzer(original_model)

    def evaluate_quantization(self, dataloader, num_batches: int = 100) -> QuantizationMetrics:
        """全面评估量化效果"""
        print("🔍 开始量化效果评估...")

        metrics = QuantizationMetrics()

        # 1. 评估模型精度
        print("📊 评估模型精度...")
        original_accuracy = self._evaluate_accuracy(self.original_model, dataloader, num_batches)
        quantized_accuracy = self._evaluate_accuracy(self.quantized_model, dataloader, num_batches)

        metrics.original_accuracy = original_accuracy
        metrics.quantized_accuracy = quantized_accuracy
        metrics.accuracy_drop = original_accuracy - quantized_accuracy
        metrics.accuracy_drop_ratio = metrics.accuracy_drop / original_accuracy if original_accuracy > 0 else 0

        # 2. 评估模型大小
        print("📏 评估模型大小...")
        original_size = self._get_model_size(self.original_model)
        quantized_size = self._get_model_size(self.quantized_model)

        metrics.original_model_size_mb = original_size
        metrics.quantized_model_size_mb = quantized_size
        metrics.compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        metrics.size_reduction_ratio = (original_size - quantized_size) / original_size if original_size > 0 else 0.0

        # 3. 评估推理速度
        print("⚡ 评估推理速度...")
        original_time = self._benchmark_inference(self.original_model, dataloader, num_batches=50)
        quantized_time = self._benchmark_inference(self.quantized_model, dataloader, num_batches=50)

        metrics.original_inference_time_ms = original_time
        metrics.quantized_inference_time_ms = quantized_time
        metrics.speedup_ratio = original_time / quantized_time if quantized_time > 0 else 1.0

        # 4. 评估内存使用
        print("💾 评估内存使用...")
        original_memory = self._measure_memory_usage(self.original_model, dataloader, num_batches=20)
        quantized_memory = self._measure_memory_usage(self.quantized_model, dataloader, num_batches=20)

        metrics.original_memory_usage_mb = original_memory
        metrics.quantized_memory_usage_mb = quantized_memory
        metrics.memory_reduction_ratio = (original_memory - quantized_memory) / original_memory if original_memory > 0 else 0.0

        # 5. 评估量化误差
        print("🔬 评估量化误差...")
        mse, mae, snr, psnr = self._calculate_quantization_error(dataloader, num_batches=30)

        metrics.mse_error = mse
        metrics.mae_error = mae
        metrics.snr_db = snr
        metrics.psnr_db = psnr

        # 6. 评估激活分布
        print("📈 评估激活分布...")
        weight_clip, act_clip, noise_std = self._analyze_activation_distribution(dataloader, num_batches=20)

        metrics.weight_clipping_ratio = weight_clip
        metrics.activation_clipping_ratio = act_clip
        metrics.quantization_noise_std = noise_std

        print("✅ 量化评估完成！")
        return metrics

    def _evaluate_accuracy(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """评估模型精度"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(images)

                # 假设是CTC模型，使用CTC解码
                if 'ctc_logits' in outputs:
                    predictions = outputs['ctc_logits'].argmax(dim=-1)
                    # 简化的CTC解码和准确率计算
                    for pred, label in zip(predictions, labels):
                        pred_text = self._ctc_decode(pred.cpu().numpy())
                        label_text = self._ctc_decode(label.cpu().numpy())
                        if pred_text == label_text:
                            correct += 1
                        total += 1

        return correct / total if total > 0 else 0.0

    def _ctc_decode(self, predictions: np.ndarray) -> str:
        """简化的CTC解码"""
        # 去除重复和空白
        decoded = []
        prev = None
        for pred in predictions:
            if pred != prev and pred != 0:  # 假设0是空白符
                decoded.append(pred)
            prev = pred
        return ''.join(map(str, decoded))

    def _get_model_size(self, model: nn.Module) -> float:
        """获取模型大小（MB）"""
        # 保存模型状态字典并计算大小
        state_dict = model.state_dict()
        size_bytes = 0

        for tensor in state_dict.values():
            size_bytes += tensor.numel() * tensor.element_size()

        return size_bytes / (1024 * 1024)  # MB

    def _benchmark_inference(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """基准测试推理时间"""
        model.eval()
        times = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(self.device)

                # 预热
                if i == 0:
                    for _ in range(5):
                        _ = model(images)

                # 正式测试
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                _ = model(images)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()

                times.append((end_time - start_time) * 1000)  # ms

        return np.mean(times) if times else 0.0

    def _measure_memory_usage(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """测量内存使用"""
        model.eval()
        memory_usage = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(self.device)

                # 清理内存
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # 记录初始内存
                initial_memory = self._get_current_memory_usage()

                # 推理
                _ = model(images)

                # 记录峰值内存
                peak_memory = self._get_current_memory_usage()
                memory_usage.append(peak_memory - initial_memory)

        return np.mean(memory_usage) if memory_usage else 0.0

    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    def _calculate_quantization_error(self, dataloader, num_batches: int) -> Tuple[float, float, float, float]:
        """计算量化误差"""
        self.original_model.eval()
        self.quantized_model.eval()

        mse_errors = []
        mae_errors = []
        snr_values = []
        psnr_values = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(self.device)

                # 获取原始和量化模型的输出
                original_output = self.original_model(images)
                quantized_output = self.quantized_model(images)

                # 提取特征进行比较
                if isinstance(original_output, dict):
                    # 处理字典输出
                    for key in original_output:
                        if torch.is_tensor(original_output[key]):
                            orig_feat = original_output[key].flatten()
                            quant_feat = quantized_output[key].flatten()

                            # 计算误差
                            mse = F.mse_loss(quant_feat, orig_feat).item()
                            mae = F.l1_loss(quant_feat, orig_feat).item()

                            # 计算SNR
                            signal_power = torch.mean(orig_feat ** 2).item()
                            noise_power = torch.mean((quant_feat - orig_feat) ** 2).item()
                            snr = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else float('inf')

                            # 计算PSNR
                            max_value = torch.max(torch.abs(orig_feat)).item()
                            psnr = 20 * np.log10(max_value / (np.sqrt(mse) + 1e-10)) if mse > 0 else float('inf')

                            mse_errors.append(mse)
                            mae_errors.append(mae)
                            snr_values.append(snr)
                            psnr_values.append(psnr)

        return (np.mean(mse_errors) if mse_errors else 0.0,
                np.mean(mae_errors) if mae_errors else 0.0,
                np.mean(snr_values) if snr_values else 0.0,
                np.mean(psnr_values) if psnr_values else 0.0)

    def _analyze_activation_distribution(self, dataloader, num_batches: int) -> Tuple[float, float, float]:
        """分析激活分布"""
        self.original_model.eval()
        self.quantized_model.eval()

        weight_clips = []
        act_clips = []
        noise_stds = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(self.device)

                # 获取激活值
                original_activations = self._get_activations(self.original_model, images)
                quantized_activations = self._get_activations(self.quantized_model, images)

                for orig_act, quant_act in zip(original_activations, quantized_activations):
                    # 计算裁剪比例
                    orig_range = torch.max(torch.abs(orig_act)).item()
                    quant_range = torch.max(torch.abs(quant_act)).item()
                    clipping_ratio = 1.0 - (quant_range / (orig_range + 1e-10))

                    # 计算量化噪声
                    noise = quant_act - orig_act
                    noise_std = torch.std(noise).item()

                    act_clips.append(clipping_ratio)
                    noise_stds.append(noise_std)

        return 0.0, np.mean(act_clips) if act_clips else 0.0, np.mean(noise_stds) if noise_stds else 0.0

    def _get_activations(self, model: nn.Module, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """获取模型激活值"""
        activations = []

        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                activations.append(output.detach().flatten())

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        _ = model(input_tensor)

        for hook in hooks:
            hook.remove()

        return activations

    def generate_report(self, metrics: QuantizationMetrics, output_path: str = 'quantization_report.json'):
        """生成量化评估报告"""
        report = {
            'summary': {
                'accuracy_drop': f"{metrics.accuracy_drop:.4f} ({metrics.accuracy_drop_ratio*100:.2f}%)",
                'compression_ratio': f"{metrics.compression_ratio:.2f}x",
                'speedup_ratio': f"{metrics.speedup_ratio:.2f}x",
                'memory_reduction': f"{metrics.memory_reduction_ratio*100:.1f}%",
            },
            'detailed_metrics': metrics.to_dict(),
            'recommendations': self._generate_recommendations(metrics),
            'grade': self._grade_quantization(metrics),
        }

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 量化评估报告已保存到: {output_path}")
        return report

    def _generate_recommendations(self, metrics: QuantizationMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 精度相关建议
        if metrics.accuracy_drop_ratio > 0.05:  # 超过5%的精度下降
            recommendations.append("精度下降较大，建议增加QAT训练轮数或降低量化强度")
            recommendations.append("考虑使用知识蒸馏来保持精度")

        if metrics.accuracy_drop_ratio < 0.01:  # 小于1%的精度下降
            recommendations.append("精度保持良好，可以尝试更激进的量化策略")

        # 压缩相关建议
        if metrics.compression_ratio < 2.0:  # 压缩比小于2倍
            recommendations.append("压缩比偏低，可以尝试更低的量化位数")

        if metrics.compression_ratio > 8.0:  # 压缩比大于8倍
            recommendations.append("压缩比很高，需要密切关注精度变化")

        # 速度相关建议
        if metrics.speedup_ratio < 1.2:  # 加速比小于1.2倍
            recommendations.append("加速效果不明显，检查量化实现或硬件兼容性")

        # 内存相关建议
        if metrics.memory_reduction_ratio < 0.3:  # 内存减少小于30%
            recommendations.append("内存减少有限，考虑优化内存访问模式")

        # 量化误差相关建议
        if metrics.snr_db < 20:  # SNR小于20dB
            recommendations.append("量化噪声较大，建议改进量化策略或校准方法")

        return recommendations

    def _grade_quantization(self, metrics: QuantizationMetrics) -> str:
        """给量化效果打分"""
        score = 0

        # 精度分数 (40分)
        if metrics.accuracy_drop_ratio <= 0.01:
            score += 40
        elif metrics.accuracy_drop_ratio <= 0.03:
            score += 30
        elif metrics.accuracy_drop_ratio <= 0.05:
            score += 20
        else:
            score += 10

        # 压缩分数 (30分)
        if metrics.compression_ratio >= 4.0:
            score += 30
        elif metrics.compression_ratio >= 2.0:
            score += 20
        elif metrics.compression_ratio >= 1.5:
            score += 10

        # 速度分数 (20分)
        if metrics.speedup_ratio >= 1.5:
            score += 20
        elif metrics.speedup_ratio >= 1.2:
            score += 15
        elif metrics.speedup_ratio >= 1.1:
            score += 10

        # 内存分数 (10分)
        if metrics.memory_reduction_ratio >= 0.5:
            score += 10
        elif metrics.memory_reduction_ratio >= 0.3:
            score += 7
        elif metrics.memory_reduction_ratio >= 0.1:
            score += 5

        # 评级
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (中等)"
        elif score >= 60:
            return "C (及格)"
        else:
            return "D (需改进)"

    def visualize_results(self, metrics: QuantizationMetrics, save_path: str = 'quantization_visualization.png'):
        """可视化量化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('量化效果可视化', fontsize=16)

        # 1. 精度对比
        ax = axes[0, 0]
        categories = ['原始模型', '量化模型']
        accuracies = [metrics.original_accuracy, metrics.quantized_accuracy]
        bars = ax.bar(categories, accuracies, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('准确率')
        ax.set_title('模型精度对比')
        ax.set_ylim(0, 1.1 * max(accuracies))

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}', ha='center', va='bottom')

        # 2. 模型大小对比
        ax = axes[0, 1]
        sizes = [metrics.original_model_size_mb, metrics.quantized_model_size_mb]
        bars = ax.bar(categories, sizes, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('模型大小 (MB)')
        ax.set_title('模型大小对比')

        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{size:.1f}MB', ha='center', va='bottom')

        # 3. 推理速度对比
        ax = axes[0, 2]
        times = [metrics.original_inference_time_ms, metrics.quantized_inference_time_ms]
        bars = ax.bar(categories, times, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('推理时间 (ms)')
        ax.set_title('推理速度对比')

        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}ms', ha='center', va='bottom')

        # 4. 压缩比和加速比
        ax = axes[1, 0]
        ratios = [metrics.compression_ratio, metrics.speedup_ratio]
        ratio_labels = ['压缩比', '加速比']
        bars = ax.bar(ratio_labels, ratios, color=['green', 'orange'], alpha=0.7)
        ax.set_ylabel('比值')
        ax.set_title('压缩和加速效果')

        for bar, ratio, label in zip(bars, ratios, ratio_labels):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}x', ha='center', va='bottom')

        # 5. 量化误差
        ax = axes[1, 1]
        error_metrics = ['MSE', 'MAE']
        errors = [metrics.mse_error, metrics.mae_error]
        bars = ax.bar(error_metrics, errors, color=['purple', 'brown'], alpha=0.7)
        ax.set_ylabel('误差值')
        ax.set_title('量化误差')
        ax.set_yscale('log')  # 对数坐标

        for bar, error, metric in zip(bars, errors, error_metrics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{error:.4f}', ha='center', va='bottom')

        # 6. 信噪比
        ax = axes[1, 2]
        snr_metrics = ['SNR (dB)', 'PSNR (dB)']
        snr_values = [metrics.snr_db, metrics.psnr_db]
        bars = ax.bar(snr_metrics, snr_values, color=['cyan', 'magenta'], alpha=0.7)
        ax.set_ylabel('dB')
        ax.set_title('信号质量')

        for bar, snr, metric in zip(bars, snr_values, snr_metrics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{snr:.1f}dB', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 量化可视化结果已保存到: {save_path}")

class QuantizationManager:
    """量化管理器 - 处理所有量化相关操作"""

    def __init__(self, config: Dict, model: torch.nn.Module):
        self.config = config
        self.model = model
        self.quantizer = None
        self.calibration_data = []
        self.original_model = None

    def prepare_model_for_quantization(self) -> torch.nn.Module:
        """准备模型进行量化"""
        if not self.config['enabled']:
            return self.model

        print("🔧 准备模型量化...")

        # 保存原始模型用于知识蒸馏
        self.original_model = self.model

        # 根据配置选择量化策略
        strategy = self.config['quantization_strategy']

        if self.config['quantization_aware_training']:
            print(f"🎯 启用量化感知训练 (QAT): {strategy}")
            self.model = self._apply_qat_quantization()
        elif self.config['post_training_quantization']:
            print(f"🎯 启用训练后量化 (PTQ): {strategy}")
            self.model = self._apply_ptq_quantization()
        else:
            print("⚠️  量化已启用但未指定训练策略")

        return self.model

    def _apply_qat_quantization(self) -> torch.nn.Module:
        """应用量化感知训练"""
        strategy = self.config['quantization_strategy']

        if strategy == 'int8_dyn_act_int4_weight':
            # 使用默认参数初始化量化器
            quantizer = Int8DynActInt4WeightQATQuantizer()
        elif strategy == 'int4_weight_only':
            quantizer = Int4WeightOnlyQATQuantizer()
        else:
            raise ValueError(f"不支持的QAT策略: {strategy}")

        # 应用量化
        quantized_model = quantizer.prepare(self.model)

        print(f"✅ QAT量化应用完成: {strategy}")
        return quantized_model

    def _apply_ptq_quantization(self) -> torch.nn.Module:
        """应用训练后量化"""
        strategy = self.config['quantization_strategy']

        if strategy == 'int8_dyn_act_int4_weight':
            quantizer = int8_dynamic_activation_int4_weight
        elif strategy == 'int8_weight_only':
            quantizer = int8_weight_only
        elif strategy == 'int4_weight_only':
            quantizer = int4_weight_only
        elif strategy == 'int8_dynamic_activation_int8_weight':
            quantizer = int8_dynamic_activation_int8_weight
        else:
            raise ValueError(f"不支持的PTQ策略: {strategy}")

        # 应用量化
        quantize_(self.model, quantizer())

        print(f"✅ PTQ量化应用完成: {strategy}")
        return self.model

    def calibrate_model(self, dataloader: DataLoader, num_batches: int = None):
        """校准量化模型"""
        if not self.config['enabled'] or not self.config['post_training_quantization']:
            return

        print("📊 开始模型校准...")

        if num_batches is None:
            num_batches = self.config['calibration_batches']

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc='校准')):
                if i >= num_batches:
                    break

                images = batch['images'].cuda()
                _ = self.model(images)

        print(f"✅ 模型校准完成 ({num_batches} 批次)")

    def get_quantization_loss(self, quantized_features: torch.Tensor,
                            original_features: torch.Tensor) -> torch.Tensor:
        """计算量化损失用于知识蒸馏"""
        if not self.config['enabled']:
            return torch.tensor(0.0, device=quantized_features.device)

        # 特征蒸馏损失
        distillation_loss = F.kl_div(
            F.log_softmax(quantized_features / self.config['temperature_distillation'], dim=-1),
            F.softmax(original_features / self.config['temperature_distillation'], dim=-1),
            reduction='batchmean'
        )

        # 量化感知损失
        quantization_loss = F.mse_loss(quantized_features, original_features.detach())

        total_loss = (self.config['distillation_weight'] * distillation_loss +
                     self.config['quantization_loss_weight'] * quantization_loss)

        return total_loss

    @torch.no_grad()
    def export_quantized_model(self, pruning: bool, path: str, epoch: int, best_cer: float, best_em: float, example_input: torch.Tensor, opt: Dict, scaler: Dict, model: nn.Module):
        """导出量化模型"""

        # 确保模型在评估模式
        model.eval()

        state = {
            'pruning': pruning,
            'model': model.state_dict(), 'opt': opt,
            'scaler': scaler, 'epoch': epoch,
            'best_cer': best_cer, 'best_em': best_em,
            'config': self.config
        }
        if self.config['enabled']:
            print("📤 导出量化模型...")

            # 使用torch.export导出量化模型
            try:
                state['quantization_model'] = export(model, (example_input,))

                # 保存导出的程序
                torch.save(state, path)

                print(f"✅ 量化模型导出完成: {path}")

            except Exception as e:
                print(f"⚠️  导出失败，使用备用方法: {e}")

                # 备用方法：只保存状态字典
                del state['quantization_model']

                torch.save(state, path)
                print(f"✅ 模型状态字典导出完成: {path}")
        else:
            print("⚠️  量化未启用，导出原始模型")
            torch.save(state, path)
            return

# 自定义 ParameterGrid 实现，替代 sklearn.model_selection.ParameterGrid 的依赖
class ParameterGrid:
    """生成参数网格的所有组合"""
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        keys, values = zip(*self.param_grid.items())
        result = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for item in result:
            yield item

    def __len__(self):

        return reduce(operator.mul, (len(v) for v in self.param_grid.values()), 1)

@dataclass
class OptimizationResult:
    """优化结果"""
    best_config: QuantizationConfig
    best_metrics: QuantizationMetrics
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    best_trial: int
    optimization_time: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'best_config': self.best_config.to_dict(),
            'best_metrics': self.best_metrics.to_dict(),
            'optimization_history': self.optimization_history,
            'total_trials': self.total_trials,
            'best_trial': self.best_trial,
            'optimization_time': self.optimization_time
        }

class QuantizationObjective:
    """量化目标函数 - 用于优化"""

    def __init__(self, original_model: nn.Module, dataloader, device: str,
                 accuracy_weight: float = 0.6, compression_weight: float = 0.2,
                 speed_weight: float = 0.2, min_accuracy_threshold: float = 0.95):
        self.original_model = original_model
        self.dataloader = dataloader
        self.device = device
        self.accuracy_weight = accuracy_weight
        self.compression_weight = compression_weight
        self.speed_weight = speed_weight
        self.min_accuracy_threshold = min_accuracy_threshold

    def __call__(self, trial: Trial, config: QuantizationConfig) -> float:
        """计算目标函数值"""
        try:
            # 根据trial建议修改配置
            modified_config = self._suggest_config_changes(trial, config)

            # 创建量化模型
            quantized_model = self._create_quantized_model(modified_config)

            # 评估量化效果
            evaluator = QuantizationEvaluator(
                self.original_model, quantized_model, self.device
            )
            metrics = evaluator.evaluate_quantization(
                self.dataloader, num_batches=20  # 减少评估批次以加速优化
            )

            # 计算综合得分
            score = self._calculate_score(metrics, modified_config)

            # 保存trial信息
            trial.set_user_attr('metrics', metrics.to_dict())
            trial.set_user_attr('config', modified_config.to_dict())

            return score

        except Exception as e:
            warnings.warn(f"Trial failed: {e}")
            return float('inf')  # 返回一个很大的值表示失败

    def _suggest_config_changes(self, trial: Trial, base_config: QuantizationConfig) -> QuantizationConfig:
        """根据trial建议修改配置"""
        config = QuantizationConfig.from_dict(base_config.to_dict())

        # 量化位数
        config.weight_bits = trial.suggest_categorical('weight_bits', [4, 8])
        config.activation_bits = trial.suggest_categorical('activation_bits', [4, 8])

        # QAT训练轮数
        config.qat_epochs = trial.suggest_int('qat_epochs', 3, 10)

        # 学习率倍数
        config.qat_learning_rate_multiplier = trial.suggest_float(
            'qat_learning_rate_multiplier', 0.01, 0.5, log=True
        )

        # 量化损失权重
        config.quantization_loss_weight = trial.suggest_float(
            'quantization_loss_weight', 0.001, 0.1, log=True
        )

        # 蒸馏温度
        config.temperature_distillation = trial.suggest_float(
            'temperature_distillation', 2.0, 10.0
        )

        # 蒸馏权重
        config.distillation_weight = trial.suggest_float(
            'distillation_weight', 0.1, 0.8
        )

        # 观察器动量
        config.observer_momentum = trial.suggest_float(
            'observer_momentum', 0.01, 0.2
        )

        # 校准批次数量
        config.calibration_batches = trial.suggest_int(
            'calibration_batches', 50, 200, step=25
        )

        return config

    def _create_quantized_model(self, config: QuantizationConfig) -> nn.Module:
        """创建量化模型"""
        # 这里应该实现具体的量化逻辑
        # 简化版本：返回原始模型的副本
        return self.original_model

    def _calculate_score(self, metrics: QuantizationMetrics, config: QuantizationConfig) -> float:
        """计算综合得分"""
        # 精度得分（越高越好）
        accuracy_score = metrics.quantized_accuracy / metrics.original_accuracy

        # 如果精度低于阈值，给予惩罚
        if accuracy_score < self.min_accuracy_threshold:
            accuracy_penalty = (self.min_accuracy_threshold - accuracy_score) * 10
            accuracy_score -= accuracy_penalty

        # 压缩得分（越高越好）
        compression_score = metrics.compression_ratio / 4.0  # 归一化到0-1范围
        compression_score = min(compression_score, 1.0)  # 限制最大值

        # 速度得分（越高越好）
        speed_score = metrics.speedup_ratio / 2.0  # 归一化到0-1范围
        speed_score = min(speed_score, 1.0)  # 限制最大值

        # 综合得分
        total_score = (
            self.accuracy_weight * accuracy_score +
            self.compression_weight * compression_score +
            self.speed_weight * speed_score
        )

        return -total_score  # Optuna最小化目标函数

class QuantizationHyperparameterOptimizer:
    """量化超参数优化器"""

    def __init__(self, original_model: nn.Module, dataloader, device: str):
        self.original_model = original_model
        self.dataloader = dataloader
        self.device = device

    def optimize_with_optuna(self, base_config: QuantizationConfig,
                           n_trials: int = 50,
                           timeout: int = 3600,
                           n_jobs: int = 1) -> OptimizationResult:
        """使用Optuna进行贝叶斯优化"""
        print(f"🔍 开始贝叶斯优化，试验次数: {n_trials}")

        start_time = time.time()

        # 创建目标函数
        objective = QuantizationObjective(
            self.original_model, self.dataloader, self.device
        )

        # 创建study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        # 运行优化
        study.optimize(
            lambda trial: objective(trial, base_config),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        optimization_time = time.time() - start_time

        # 获取最佳结果
        best_trial = study.best_trial
        best_config = QuantizationConfig.from_dict(best_trial.user_attrs['config'])
        best_metrics = QuantizationMetrics()
        # 这里需要从trial中恢复metrics，简化处理

        # 构建优化历史
        history = []
        for trial in study.trials:
            if trial.value is not None:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'metrics': trial.user_attrs.get('metrics', {}),
                    'config': trial.user_attrs.get('config', {})
                })

        result = OptimizationResult(
            best_config=best_config,
            best_metrics=best_metrics,
            optimization_history=history,
            total_trials=len(study.trials),
            best_trial=best_trial.number,
            optimization_time=optimization_time
        )

        print(f"✅ 贝叶斯优化完成，最佳试验: {best_trial.number}, 得分: {best_trial.value:.4f}")
        return result

    def optimize_with_grid_search(self, base_config: QuantizationConfig,
                                param_grid: Optional[Dict[str, List]] = None) -> OptimizationResult:
        """使用网格搜索进行优化"""
        print("🔍 开始网格搜索优化")

        start_time = time.time()

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'weight_bits': [4, 8],
                'activation_bits': [4, 8],
                'qat_epochs': [3, 5, 8],
                'qat_learning_rate_multiplier': [0.05, 0.1, 0.2],
                'quantization_loss_weight': [0.005, 0.01, 0.02],
                'temperature_distillation': [4.0, 6.0, 8.0],
                'distillation_weight': [0.3, 0.5, 0.7],
                'observer_momentum': [0.05, 0.1, 0.15],
                'calibration_batches': [50, 100, 150]
            }

        # 生成所有参数组合
        grid = list(ParameterGrid(param_grid))
        print(f"📊 总参数组合数: {len(grid)}")

        best_score = float('inf')
        best_config = None
        best_metrics = None
        best_trial = 0
        history = []

        for i, params in enumerate(tqdm(grid, desc="网格搜索")):
            try:
                # 创建配置
                config = QuantizationConfig.from_dict(base_config.to_dict())
                for key, value in params.items():
                    setattr(config, key, value)

                # 创建量化模型
                quantized_model = self._create_quantized_model_for_grid_search(config)

                # 评估
                evaluator = QuantizationEvaluator(
                    self.original_model, quantized_model, self.device
                )
                metrics = evaluator.evaluate_quantization(
                    self.dataloader, num_batches=10  # 减少评估批次
                )

                # 计算得分
                score = self._calculate_simple_score(metrics)

                # 保存历史
                history.append({
                    'trial_number': i,
                    'params': params,
                    'score': score,
                    'metrics': metrics.to_dict(),
                    'config': config.to_dict()
                })

                # 更新最佳结果
                if score < best_score:
                    best_score = score
                    best_config = config
                    best_metrics = metrics
                    best_trial = i

            except Exception as e:
                warnings.warn(f"网格搜索试验 {i} 失败: {e}")
                continue

        optimization_time = time.time() - start_time

        result = OptimizationResult(
            best_config=best_config,
            best_metrics=best_metrics,
            optimization_history=history,
            total_trials=len(grid),
            best_trial=best_trial,
            optimization_time=optimization_time
        )

        print(f"✅ 网格搜索完成，最佳得分: {best_score:.4f}")
        return result

    def optimize_with_random_search(self, base_config: QuantizationConfig,
                                  n_trials: int = 100) -> OptimizationResult:
        """使用随机搜索进行优化"""
        print(f"🔍 开始随机搜索优化，试验次数: {n_trials}")

        start_time = time.time()

        best_score = float('inf')
        best_config = None
        best_metrics = None
        best_trial = 0
        history = []

        for i in range(n_trials):
            try:
                # 随机生成参数
                config = self._generate_random_config(base_config)

                # 创建量化模型
                quantized_model = self._create_quantized_model_for_grid_search(config)

                # 评估
                evaluator = QuantizationEvaluator(
                    self.original_model, quantized_model, self.device
                )
                metrics = evaluator.evaluate_quantization(
                    self.dataloader, num_batches=10
                )

                # 计算得分
                score = self._calculate_simple_score(metrics)

                # 保存历史
                history.append({
                    'trial_number': i,
                    'config': config.to_dict(),
                    'score': score,
                    'metrics': metrics.to_dict()
                })

                # 更新最佳结果
                if score < best_score:
                    best_score = score
                    best_config = config
                    best_metrics = metrics
                    best_trial = i

            except Exception as e:
                warnings.warn(f"随机搜索试验 {i} 失败: {e}")
                continue

        optimization_time = time.time() - start_time

        result = OptimizationResult(
            best_config=best_config,
            best_metrics=best_metrics,
            optimization_history=history,
            total_trials=n_trials,
            best_trial=best_trial,
            optimization_time=optimization_time
        )

        print(f"✅ 随机搜索完成，最佳得分: {best_score:.4f}")
        return result

    def _create_quantized_model_for_grid_search(self, config: QuantizationConfig) -> nn.Module:
        """为网格搜索创建量化模型"""
        # 简化版本：返回原始模型的副本
        # 实际实现中应该应用真正的量化
        return self.original_model

    def _generate_random_config(self, base_config: QuantizationConfig) -> QuantizationConfig:
        """生成随机配置"""
        import random

        config = QuantizationConfig.from_dict(base_config.to_dict())

        # 随机参数
        config.weight_bits = random.choice([4, 8])
        config.activation_bits = random.choice([4, 8])
        config.qat_epochs = random.randint(3, 10)
        config.qat_learning_rate_multiplier = random.uniform(0.01, 0.5)
        config.quantization_loss_weight = random.uniform(0.001, 0.1)
        config.temperature_distillation = random.uniform(2.0, 10.0)
        config.distillation_weight = random.uniform(0.1, 0.8)
        config.observer_momentum = random.uniform(0.01, 0.2)
        config.calibration_batches = random.choice([50, 75, 100, 125, 150, 175, 200])

        return config

    def _calculate_simple_score(self, metrics: QuantizationMetrics) -> float:
        """计算简单得分"""
        # 精度得分
        accuracy_score = metrics.quantized_accuracy / metrics.original_accuracy

        # 压缩得分
        compression_score = min(metrics.compression_ratio / 4.0, 1.0)

        # 速度得分
        speed_score = min(metrics.speedup_ratio / 2.0, 1.0)

        # 综合得分
        total_score = 0.6 * accuracy_score + 0.2 * compression_score + 0.2 * speed_score

        return -total_score  # 负值用于最小化

class AdaptiveHyperparameterOptimizer:
    """自适应超参数优化器 - 根据模型和任务自动选择优化策略"""

    def __init__(self, model: nn.Module, dataloader, device: str):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.base_optimizer = QuantizationHyperparameterOptimizer(model, dataloader, device)

    def optimize(self, base_config: QuantizationConfig,
                optimization_budget: str = 'medium',
                target_metric: str = 'balanced',
                method: str = 'bayesian') -> OptimizationResult:
        """
        自适应优化

        Args:
            base_config: 基础配置
            optimization_budget: 优化预算 ('low', 'medium', 'high')
            target_metric: 目标指标 ('accuracy', 'compression', 'speed', 'balanced')
            method: 优化方法 ('bayesian', 'grid_search', 'random_search')
        """
        print(f"🎯 开始自适应优化，预算: {optimization_budget}, 目标: {target_metric}, 方法: {method}")

        # 根据预算选择优化参数
        if optimization_budget == 'low':
            n_trials = 20
            timeout = 600  # 10分钟
            n_jobs = 1
        elif optimization_budget == 'medium':
            n_trials = 50
            timeout = 1800  # 30分钟
            n_jobs = 1  # 避免多进程问题
        else:  # high
            n_trials = 100
            timeout = 3600  # 1小时
            n_jobs = 1  # 避免多进程问题

        # 根据目标调整权重
        if target_metric == 'accuracy':
            accuracy_weight, compression_weight, speed_weight = 0.8, 0.1, 0.1
        elif target_metric == 'compression':
            accuracy_weight, compression_weight, speed_weight = 0.3, 0.6, 0.1
        elif target_metric == 'speed':
            accuracy_weight, compression_weight, speed_weight = 0.3, 0.1, 0.6
        else:  # balanced
            accuracy_weight, compression_weight, speed_weight = 0.6, 0.2, 0.2

        # 根据指定的方法选择优化策略
        if method == 'bayesian':
            # 使用贝叶斯优化
            result = self.base_optimizer.optimize_with_optuna(
                base_config, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs
            )
        elif method == 'grid_search':
            # 使用网格搜索
            result = self.base_optimizer.optimize_with_grid_search(base_config)
        elif method == 'random_search':
            # 使用随机搜索
            result = self.base_optimizer.optimize_with_random_search(
                base_config, n_trials=n_trials
            )
        else:
            # 默认使用贝叶斯优化
            result = self.base_optimizer.optimize_with_optuna(
                base_config, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs
            )

        return result

    def progressive_optimization(self, base_config: QuantizationConfig,
                               n_stages: int = 3) -> List[OptimizationResult]:
        """渐进式优化 - 分阶段优化不同参数"""
        print(f"🚀 开始渐进式优化，阶段数: {n_stages}")

        results = []
        current_config = QuantizationConfig.from_dict(base_config.to_dict())

        # 阶段1: 优化量化位数
        print("📊 阶段1: 优化量化位数")
        stage1_config = QuantizationConfig.from_dict(current_config.to_dict())
        stage1_config.qat_epochs = 3  # 快速评估
        stage1_config.calibration_batches = 50

        param_grid = {
            'weight_bits': [4, 8],
            'activation_bits': [4, 8],
            'observer_type': ['moving_average', 'min_max'],
            'quantization_granularity': ['per_channel', 'per_tensor']
        }

        result1 = self.base_optimizer.optimize_with_grid_search(stage1_config, param_grid)
        results.append(result1)
        current_config = result1.best_config

        # 阶段2: 优化训练参数
        print("📈 阶段2: 优化训练参数")
        stage2_config = QuantizationConfig.from_dict(current_config.to_dict())

        param_grid = {
            'qat_epochs': [3, 5, 8],
            'qat_learning_rate_multiplier': [0.05, 0.1, 0.2],
            'calibration_batches': [50, 100, 150]
        }

        result2 = self.base_optimizer.optimize_with_grid_search(stage2_config, param_grid)
        results.append(result2)
        current_config = result2.best_config

        # 阶段3: 优化损失函数参数
        print("🎯 阶段3: 优化损失函数参数")
        stage3_config = QuantizationConfig.from_dict(current_config.to_dict())

        param_grid = {
            'quantization_loss_weight': [0.005, 0.01, 0.02],
            'temperature_distillation': [4.0, 6.0, 8.0],
            'distillation_weight': [0.3, 0.5, 0.7],
            'observer_momentum': [0.05, 0.1, 0.15]
        }

        result3 = self.base_optimizer.optimize_with_grid_search(stage3_config, param_grid)
        results.append(result3)

        print("✅ 渐进式优化完成")
        return results

# 预定义配置模板
QUANTIZATION_CONFIG = {
    'enabled': True,                            # 是否启用量化
    'qat_epochs': 5,                           # QAT训练轮数
    'ptq_epochs': 2,                           # PTQ训练轮数
    'quantization_strategy': 'int8_dyn_act_int4_weight',  # 量化策略
    'calibration_batches': 100,                # 校准批次数量
    'quantization_aware_training': True,       # 是否使用QAT
    'post_training_quantization': False,       # 是否使用PTQ
    'mixed_precision': True,                   # 混合精度训练
    'quantization_layers': ['linear', 'conv2d', 'attention'],  # 需要量化的层类型
    'excluded_layers': ['embedding', 'layernorm'],  # 排除的层
    'quantization_bits': {
        'weight': 4,                           # 权重量化位数
        'activation': 8,                       # 激活量化位数
    },
    'observer_type': 'moving_average',         # 观察器类型
    'observer_momentum': 0.1,                  # 观察器动量
    'quantization_granularity': 'per_channel', # 量化粒度
    'qat_learning_rate_multiplier': 0.1,       # QAT学习率倍数
    'quantization_loss_weight': 0.01,          # 量化损失权重
    'temperature_distillation': 4.0,           # 知识蒸馏温度
    'distillation_weight': 0.3,                # 知识蒸馏权重
}
PREDEFINED_CONFIGS = {
    'ocr_conservative': QuantizationConfig(
        enabled=True,
        strategy=QuantizationStrategy.INT8_DYN_ACT_INT4_WEIGHT,
        quantization_aware_training=True,
        qat_epochs=8,
        weight_bits=4,
        activation_bits=8,
        observer_type=ObserverType.MOVING_AVERAGE,
        quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        quantization_loss_weight=0.02,
        temperature_distillation=6.0,
        distillation_weight=0.5,
        qat_learning_rate_multiplier=0.05,
    ),

    'ocr_balanced': QuantizationConfig(
        enabled=True,
        strategy=QuantizationStrategy.INT8_DYN_ACT_INT4_WEIGHT,
        quantization_aware_training=True,
        qat_epochs=5,
        weight_bits=4,
        activation_bits=8,
        observer_type=ObserverType.MOVING_AVERAGE,
        quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        quantization_loss_weight=0.01,
        temperature_distillation=4.0,
        distillation_weight=0.3,
        qat_learning_rate_multiplier=0.1,
    ),

    'ocr_aggressive': QuantizationConfig(
        enabled=True,
        strategy=QuantizationStrategy.INT4_WEIGHT_ONLY,
        quantization_aware_training=True,
        qat_epochs=10,
        weight_bits=4,
        activation_bits=4,
        observer_type=ObserverType.MIN_MAX,
        quantization_granularity=QuantizationGranularity.PER_TENSOR,
        quantization_loss_weight=0.03,
        temperature_distillation=8.0,
        distillation_weight=0.7,
        qat_learning_rate_multiplier=0.05,
    ),

    'mobile_optimized': QuantizationConfig(
        enabled=True,
        strategy=QuantizationStrategy.INT4_WEIGHT_ONLY,
        quantization_aware_training=True,
        qat_epochs=6,
        weight_bits=4,
        activation_bits=4,
        observer_type=ObserverType.MIN_MAX,
        quantization_granularity=QuantizationGranularity.PER_TENSOR,
        quantization_loss_weight=0.02,
        temperature_distillation=6.0,
        distillation_weight=0.5,
        qat_learning_rate_multiplier=0.08,
        memory_efficient=True,
        compile_model=True,
    ),

    'server_optimized': QuantizationConfig(
        enabled=True,
        strategy=QuantizationStrategy.INT8_DYN_ACT_INT4_WEIGHT,
        quantization_aware_training=True,
        qat_epochs=4,
        weight_bits=4,
        activation_bits=8,
        observer_type=ObserverType.MOVING_AVERAGE,
        quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        quantization_loss_weight=0.01,
        temperature_distillation=4.0,
        distillation_weight=0.3,
        qat_learning_rate_multiplier=0.1,
        enable_cuda_graphs=True,
        memory_efficient=False,
    ),
}

def get_config_template(template_name: str) -> QuantizationConfig:
    """获取预定义配置模板"""
    if template_name not in PREDEFINED_CONFIGS:
        available_templates = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(f"未知的配置模板: {template_name}. 可用模板: {available_templates}")

    return PREDEFINED_CONFIGS[template_name]

def create_optimal_config(model: nn.Module, task_type: str = 'ocr',
                         hardware_target: str = 'cpu',
                         target_compression_ratio: float = 0.25,
                         preserve_accuracy: bool = True) -> QuantizationConfig:
    """创建最优量化配置"""
    adaptive_config = AdaptiveQuantizationConfig(model, task_type)

    # 获取推荐配置
    config = adaptive_config.recommend_config(target_compression_ratio, preserve_accuracy)

    # 针对硬件优化
    config = adaptive_config.optimize_config_for_hardware(hardware_target)

    # 验证配置
    errors = QuantizationConfigValidator.validate_config(config)
    if errors:
        raise ValueError(f"配置验证失败: {errors}")

    # 检查警告
    warnings = QuantizationConfigValidator.warn_config(config)
    if warnings:
        print("⚠️  配置警告:")
        for warning in warnings:
            print(f"  - {warning}")

    return config

# 使用示例和工具函数
def create_optimization_study(device: str, model: nn.Module, dataloader: DataLoader, output_dir: str, study_name: str, method: str = 'bayesian',
                            n_trials: int = 50, param_config: dict = None,
                            optimization_target: str = 'balanced', dry_run: bool = False) -> None:
    """
    创建并运行超参数自动优化研究

    Args:
        output_dir: 输出目录
        study_name: 研究名称
        method: 优化方法 ('bayesian', 'grid_search', 'random_search')
        n_trials: 试验次数
        param_config: 参数配置文件路径或字典
        optimization_target: 优化目标 ('balanced', 'accuracy', 'compression', 'speed')
        dry_run: 是否只验证代码流程，不执行实际优化
    """

    # 创建研究目录
    study_dir = Path(output_dir) / f"optimization_study_{study_name}"
    study_dir.mkdir(parents=True, exist_ok=True)

    # 创建基础配置
    base_config = create_optimal_config(
        model=model,
        task_type='ocr',
        hardware_target='cpu',
        target_compression_ratio=0.25,
        preserve_accuracy=True
    )

    # 创建优化器
    optimizer = AdaptiveHyperparameterOptimizer(model, dataloader, device)

    # 根据优化目标调整预算
    if optimization_target == 'balanced':
        budget = 'medium'
    elif optimization_target == 'accuracy':
        budget = 'high'  # 更高预算以获得更好精度
    else:
        budget = 'medium'

    # 运行优化
    print(f"🚀 开始优化研究: {study_name}")
    print(f"   - 优化方法: {method}")
    print(f"   - 试验次数: {n_trials}")
    print(f"   - 优化目标: {optimization_target}")
    print(f"   - 输出目录: {study_dir}")

    # 创建研究配置
    study_config = {
        'study_name': study_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'output_directory': str(study_dir),
        'optimization_method': method,
        'n_trials': n_trials,
        'optimization_target': optimization_target,
        'base_config': base_config.to_dict(),
        'evaluation_metrics': ['accuracy', 'compression', 'speed', 'memory']
    }

    # 运行不同的优化方法
    if dry_run:
        print("🔍 执行dry_run，跳过实际优化...")
        # 创建模拟结果，用于验证流程

        # 模拟优化结果
        result = OptimizationResult(
            best_config=base_config,
            best_metrics=QuantizationMetrics(),
            optimization_history=[
                {
                    'trial_number': 0,
                    'score': -0.85,
                    'params': {
                        'weight_bits': 4,
                        'activation_bits': 8,
                        'qat_epochs': 5,
                        'qat_learning_rate_multiplier': 0.1,
                        'quantization_loss_weight': 0.01,
                        'temperature_distillation': 4.0,
                        'distillation_weight': 0.3,
                        'observer_momentum': 0.1,
                        'calibration_batches': 100
                    },
                    'metrics': {
                        'accuracy': {
                            'original': 0.98,
                            'quantized': 0.975,
                            'drop': 0.005,
                            'drop_ratio': 0.0051
                        },
                        'model_size': {
                            'original_mb': 15.2,
                            'quantized_mb': 4.8,
                            'compression_ratio': 3.17,
                            'reduction_ratio': 0.684
                        }
                    },
                    'config': base_config.to_dict()
                }
            ],
            total_trials=1,
            best_trial=0,
            optimization_time=10.5
        )
    else:
        if method == 'bayesian':
            print("🎯 运行贝叶斯优化...")
            result = optimizer.optimize(
                base_config=base_config,
                optimization_budget=budget,
                target_metric=optimization_target,
                method=method
            )
        elif method == 'grid_search':
            print("📋 运行网格搜索...")
            result = optimizer.optimize(
                base_config=base_config,
                optimization_budget=budget,
                target_metric=optimization_target,
                method=method
            )
        elif method == 'random_search':
            print("🎲 运行随机搜索...")
            result = optimizer.optimize(
                base_config=base_config,
                optimization_budget=budget,
                target_metric=optimization_target,
                method=method
            )
        else:
            raise ValueError(f"不支持的优化方法: {method}")

    # 保存优化结果
    print("💾 保存优化结果...")

    # 保存研究配置
    config_path = study_dir / 'study_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(study_config, f, indent=2, ensure_ascii=False)

    # 保存最佳配置
    best_config_path = study_dir / 'best_quantization_config.json'
    result.best_config.save_to_file(str(best_config_path))

    # 保存优化历史
    history_path = study_dir / 'optimization_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(result.optimization_history, f, indent=2, ensure_ascii=False)

    # 保存最佳指标
    best_metrics_path = study_dir / 'best_metrics.json'
    with open(best_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result.best_metrics.to_dict(), f, indent=2, ensure_ascii=False)

    # 生成优化报告
    generate_optimization_report(result, study_dir, study_name)

    print(f"✅ 优化研究完成！")
    print(f"📁 结果保存在: {study_dir}")
    print(f"🏆 最佳配置已保存到: {best_config_path}")


def generate_optimization_report(result: OptimizationResult, study_dir: Path, study_name: str) -> None:
    """
    生成优化报告

    Args:
        result: 优化结果
        study_dir: 输出目录
        study_name: 研究名称
    """
    report = {
        'study_name': study_name,
        'total_trials': result.total_trials,
        'best_trial': result.best_trial,
        'optimization_time': result.optimization_time,
        'best_config': result.best_config.to_dict(),
        'best_metrics': result.best_metrics.to_dict(),
        'optimization_history': result.optimization_history,
        'summary': {
            'accuracy_improvement': 'N/A',
            'compression_ratio': f"{result.best_metrics.compression_ratio:.2f}x",
            'speedup_ratio': f"{result.best_metrics.speedup_ratio:.2f}x",
            'memory_reduction': f"{result.best_metrics.memory_reduction_ratio*100:.1f}%"
        }
    }

    # 保存报告
    report_path = study_dir / 'optimization_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 生成可视化
    print("📈 生成优化可视化...")
    visualize_optimization_results(result, study_dir)

    print(f"📄 优化报告已生成: {report_path}")


def visualize_optimization_results(result: OptimizationResult, study_dir: Path) -> None:
    """
    可视化优化结果

    Args:
        result: 优化结果
        study_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    # 提取历史数据
    trial_numbers = [item['trial_number'] for item in result.optimization_history]
    scores = [item['score'] for item in result.optimization_history]

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 1. 优化得分变化
    plt.subplot(1, 2, 1)
    plt.plot(trial_numbers, scores, 'b-', alpha=0.7, label='Optimization Score')
    plt.scatter(trial_numbers, scores, c='red', s=20, alpha=0.5)
    plt.xlabel('Trial Number')
    plt.ylabel('Score (Lower is Better)')
    plt.title('Optimization Score Over Trials')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. 最佳配置参数
    plt.subplot(1, 2, 2)
    best_config = result.best_config.to_dict()

    # 提取关键参数
    params = {
        'Weight Bits': best_config['weight_bits'],
        'Activation Bits': best_config['activation_bits'],
        'QAT Epochs': best_config['qat_epochs'],
        'Learning Rate Multiplier': best_config['qat_learning_rate_multiplier'],
        'Quantization Loss Weight': best_config['quantization_loss_weight'],
        'Distillation Temperature': best_config['temperature_distillation'],
        'Distillation Weight': best_config['distillation_weight'],
        'Calibration Batches': best_config['calibration_batches']
    }

    # 转换为适合图表的数据
    param_names = list(params.keys())
    param_values = list(params.values())

    # 使用水平条形图显示参数
    plt.barh(param_names, param_values, color='green', alpha=0.7)
    plt.xlabel('Value')
    plt.title('Best Configuration Parameters')
    plt.grid(True, alpha=0.3)

    # 保存图表
    plt.tight_layout()
    viz_path = study_dir / 'optimization_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 优化可视化已保存: {viz_path}")

def compare_optimization_methods(results: Dict[str, OptimizationResult]) -> None:
    """比较不同优化方法的结果"""
    print("\n📊 优化方法比较:")
    print("="*80)

    for method, result in results.items():
        print(f"\n🔍 {method.upper()} 优化结果:")
        print(f"  - 最佳试验: #{result.best_trial}")
        print(f"  - 优化时间: {result.optimization_time:.1f}秒")
        print(f"  - 总试验数: {result.total_trials}")

        if result.best_metrics:
            print(f"  - 精度保持: {(1 - result.best_metrics.accuracy_drop_ratio)*100:.1f}%")
            print(f"  - 压缩比: {result.best_metrics.compression_ratio:.2f}x")
            print(f"  - 加速比: {result.best_metrics.speedup_ratio:.2f}x")

# 示例用法
if __name__ == "__main__":
    # 创建示例模型

    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.linear1 = nn.Linear(64 * 30 * 30, 128)
            self.linear2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.linear1(x))
            return self.linear2(x)

    model = ExampleModel()

    # 创建最优配置
    config = create_optimal_config(
        model=model,
        task_type='ocr',
        hardware_target='mobile',
        target_compression_ratio=0.2,
        preserve_accuracy=True
    )

    print("🎯 最优量化配置:")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    # 保存配置
    config.save_to_file('optimal_quantization_config.json')
    print("✅ 配置已保存到 optimal_quantization_config.json")
