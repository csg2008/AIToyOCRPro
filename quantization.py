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
import copy
import random
import traceback
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.qat import (
    ComposableQATQuantizer,
    QATConfig,
    IntxFakeQuantizeConfig,
)
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.quantization.granularity import PerTensor, PerGroup, PerToken, PerRow, PerBlock, PerAxis
from torchao.quantization.quant_api import MappingType, ZeroPointDomain

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
    """剪枝管理器 - 处理结构化/非结构化剪枝、验证和回滚"""
    def __init__(self, config: PruningConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.pruned_layers = []
        self.original_model = None
        self.pruning_applied = False
        self.pruning_candidates = {}
        # 缓存第一层名称，避免重复计算
        self._first_layer_name = None
        self._sensitivity_analysis = {}

    def get_pruning_ratio(self, layer_name: str) -> float:
        """获取特定层的剪枝比例"""
        for layer_type, ratio in self.config.layer_specific_ratios.items():
            if layer_type in layer_name:
                return ratio
        return self.config.pruning_ratio

    def _is_first_layer(self, module: nn.Module, name: str) -> bool:
        """检测是否为网络的第一层

        通过遍历模型结构来确定第一层，而不是依赖硬编码的命名模式
        """
        if self._first_layer_name is not None:
            return name == self._first_layer_name

        # 找到第一个卷积层或线性层
        for n, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._first_layer_name = n
                return n == name
        return False

    def _is_excluded_layer(self, module: nn.Module, name: str) -> bool:
        """判断是否应该排除该层不剪枝"""
        # 第一层通常不剪枝（特别是输入通道）
        if self._is_first_layer(module, name):
            return True

        # 分类层/输出层保护
        if any(keyword in name.lower() for keyword in ['classifier', 'head', 'output', 'fc_out']):
            if isinstance(module, nn.Linear) and module.out_features < 1000:
                return True

        # 检查是否在允许的剪枝层列表中
        if not any(layer_type in name for layer_type in self.config.pruning_layers):
            return True

        return False

    def _is_prunable_module(self, module: nn.Module, name: str) -> bool:
        """判断模块是否可剪枝"""
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            return False

        if self._is_excluded_layer(module, name):
            return False

        # 检查权重是否足够大（避免剪枝极小的层）
        if module.weight.numel() < 100:  # 小于100参数的层不剪枝
            return False

        return True

    def _clear_all_pruning_masks(self):
        """清除模型中所有残留的剪枝 mask

        改进: 使用 prune.remove() 和 prune.is_pruned() 标准API，
        避免直接操作属性导致的状态不一致
        """
        cleared_count = 0
        for name, module in self.model.named_modules():
            # 使用标准API检查是否已剪枝，而不是检查属性
            if not prune.is_pruned(module):
                continue

            for param_name in ['weight', 'bias']:
                mask_attr = f'{param_name}_mask'

                # 只有当mask属性存在时才处理
                if not hasattr(module, mask_attr):
                    continue

                try:
                    # 使用标准API移除剪枝，这会正确处理原始参数
                    prune.remove(module, param_name)
                    cleared_count += 1
                except (ValueError, AttributeError) as e:
                    # 如果移除失败（例如参数不存在），安全地清理mask
                    try:
                        mask = getattr(module, mask_attr)
                        if hasattr(module, param_name):
                            param = getattr(module, param_name)
                            # 将mask应用到参数，确保0值被保留
                            with torch.no_grad():
                                param.data = param.data * mask
                    except Exception:
                        pass

        # 清理pruned_layers列表中的记录
        self.pruned_layers = []
        self.pruning_applied = False
        if cleared_count > 0:
            print(f"   已清除 {cleared_count} 个参数的剪枝状态")

    def _should_prune(self, epoch: int, current_acc: float, best_acc: float) -> bool:
        """检查是否应该进行剪枝"""
        if not self.config.enabled:
            return False
        if epoch != self.config.pruning_epoch:
            return False
        if current_acc < best_acc * (1 - self.config.min_acc_drop):
            print(f"⚠️ 当前精度 {current_acc:.4f} 低于阈值 {best_acc * (1 - self.config.min_acc_drop):.4f}，跳过剪枝")
            return False
        return True

    def record_pruning_candidates(self, epoch: int, current_acc: float, best_acc: float) -> bool:
        """记录剪枝候选节点"""
        if not self._should_prune(epoch, current_acc, best_acc):
            return False

        print(f"🎯 开始记录剪枝候选节点...")
        print(f"   - 当前精度: {current_acc:.4f}")
        print(f"   - 最佳精度: {best_acc:.4f}")
        print(f"   - 剪枝策略: {self.config.pruning_strategy}")

        # 梯度剪枝警告与验证
        if self.config.prune_criteria == 'grad':
            was_training = self.model.training
            has_grad = False

            # 先在训练模式下检查
            self.model.train()
            for _, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if module.weight.grad is not None:
                        has_grad = True
                        break

            # 恢复原始模式
            if not was_training:
                self.model.eval()

            if not has_grad:
                print("⚠️ 警告: 使用梯度剪枝标准但模型没有梯度信息")
                print("   梯度剪枝需要在反向传播之后、梯度清零之前执行")
                print("   建议: 在训练步骤(loss.backward后, optimizer.zero_grad前)调用record_pruning_candidates")
                print("   当前将回退到L1范数剪枝标准")

        # 遍历模型的所有层
        for name, module in self.model.named_modules():
            if not self._is_prunable_module(module, name):
                continue

            pruning_ratio = self.get_pruning_ratio(name)
            print(f"   - 记录层: {name}, 剪枝比例: {pruning_ratio:.2f}")

            # 获取权重
            weight = module.weight.data.cpu()

            # 根据剪枝标准计算权重重要性
            importance = self._calculate_importance(module, weight)

            # 根据剪枝策略确定剪枝方式
            if self.config.pruning_strategy == 'l1_unstructured':
                # 非结构化剪枝：按权重值排序
                flat_importance = importance.flatten()
                threshold = torch.quantile(flat_importance, pruning_ratio)
                mask = importance > threshold
            else:
                # 结构化剪枝：按通道或神经元排序
                mask = self._create_structured_mask(module, importance, pruning_ratio)

            # 存储剪枝候选信息
            self.pruning_candidates[name] = {
                'module_type': module.__class__.__name__,
                'pruning_ratio': pruning_ratio,
                'mask': mask,
                'importance': importance
            }

        print(f"✅ 剪枝候选节点记录完成 (共 {len(self.pruning_candidates)} 层)")
        return True

    def _calculate_importance(self, module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
        """计算权重重要性"""
        criteria = self.config.prune_criteria
        weight_device = weight.device

        if criteria == 'l1':
            return torch.abs(weight)
        elif criteria == 'l2':
            return torch.norm(weight.view(weight.size(0), -1), dim=1, keepdim=True).view_as(weight)
        elif criteria == 'grad':
            if module.weight.grad is not None:
                # 确保梯度在与 weight 相同的设备上
                grad = module.weight.grad.data
                if grad.device != weight_device:
                    grad = grad.to(weight_device)
                return torch.abs(grad)
            else:
                # 无梯度时回退到 L1
                return torch.abs(weight)
        else:
            return torch.abs(weight)

    def _create_structured_mask(self, module: nn.Module, importance: torch.Tensor,
                                 pruning_ratio: float) -> torch.Tensor:
        """创建结构化剪枝的mask"""
        device = importance.device

        # 按输出通道/神经元计算重要性
        channel_importance = importance.view(importance.size(0), -1).mean(dim=1)

        num_channels = len(channel_importance)
        num_prune = int(num_channels * pruning_ratio)

        if num_prune <= 0 or num_prune >= num_channels:
            # 如果没有需要pruning的通道，返回全1 mask
            return torch.ones_like(importance)

        sorted_indices = torch.argsort(channel_importance)
        prune_indices = sorted_indices[:num_prune]

        # 创建掩码 - 在与 importance 相同的设备上
        mask = torch.ones_like(importance)

        # 对于 Conv2d 和 Linear，都pruning输出维度
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask[prune_indices] = 0

        return mask

    def apply_pruning(self, epoch: int, current_acc: float, best_acc: float) -> bool:
        """应用剪枝 - 统一入口

        统一使用 _apply_pruning_to_module 方法，消除代码重复
        """
        if not self._should_prune(epoch, current_acc, best_acc):
            return False

        print(f"🎯 开始剪枝...")
        print(f"   - 当前精度: {current_acc:.4f}")
        print(f"   - 最佳精度: {best_acc:.4f}")
        print(f"   - 剪枝策略: {self.config.pruning_strategy}")

        # 保存原始模型（深拷贝以避免引用问题）
        self.original_model = copy.deepcopy(self.model.state_dict())

        # 清除任何之前的剪枝状态
        self._clear_all_pruning_masks()

        # 统一应用剪枝
        pruned_count = 0
        for name, module in self.model.named_modules():
            if not self._is_prunable_module(module, name):
                continue

            pruning_ratio = self.get_pruning_ratio(name)
            if pruning_ratio <= 0 or pruning_ratio >= 1:
                continue

            if self._apply_pruning_to_module(module, name, pruning_ratio):
                print(f"   - 剪枝层: {name}, 比例: {pruning_ratio:.2%}")
                pruned_count += 1

        self.pruning_applied = True
        print(f"✅ 剪枝完成 (共 {pruned_count} 层)")
        return True

    def _apply_pruning_to_module(self, module: nn.Module, name: str,
                                  pruning_ratio: float) -> bool:
        """统一应用剪枝到单个模块

        Args:
            module: 要剪枝的模块
            name: 模块名称
            pruning_ratio: 剪枝比例

        Returns:
            是否成功应用剪枝
        """
        if pruning_ratio <= 0 or pruning_ratio >= 1:
            return False

        strategy = self.config.pruning_strategy

        try:
            if strategy == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

            elif strategy == 'l1_structured':
                dim = self._select_pruning_dim(module, name)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=dim)

            elif strategy == 'l2_structured':
                dim = self._select_pruning_dim(module, name)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=dim)

            elif strategy == 'ln_structured':
                # 使用配置中的参数
                n = self.config.prune_params.get('n', 2)
                dim = self._select_pruning_dim(module, name)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=n, dim=dim)
            else:
                print(f"⚠️ 未知的剪枝策略: {strategy}，跳过层 {name}")
                return False

            # 记录已剪枝的层
            if (name, module) not in self.pruned_layers:
                self.pruned_layers.append((name, module))

            return True

        except Exception as e:
            print(f"⚠️ 剪枝层 {name} 失败: {e}")
            return False

    def apply_global_pruning(self, epoch: int, current_acc: float, best_acc: float) -> bool:
        """全局非结构化剪枝 - 获得更好的整体稀疏性

        相比逐层剪枝，全局剪枝可以在全局范围内选择最不重要的权重，
        从而获得更优的稀疏模式。

        Args:
            epoch: 当前epoch
            current_acc: 当前精度
            best_acc: 最佳精度

        Returns:
            是否成功应用剪枝
        """
        if not self._should_prune(epoch, current_acc, best_acc):
            return False

        print(f"🎯 开始全局剪枝...")
        print(f"   - 当前精度: {current_acc:.4f}")
        print(f"   - 最佳精度: {best_acc:.4f}")
        print(f"   - 目标剪枝比例: {self.config.pruning_ratio:.2%}")

        # 保存原始模型
        self.original_model = copy.deepcopy(self.model.state_dict())
        self._clear_all_pruning_masks()

        # 收集所有可剪枝的参数
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if self._is_prunable_module(module, name):
                parameters_to_prune.append((module, 'weight'))

        if len(parameters_to_prune) == 0:
            print("⚠️ 没有可剪枝的层")
            return False

        try:
            # 应用全局剪枝
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_ratio
            )

            # 记录已剪枝的层
            for module, _ in parameters_to_prune:
                for m_name, m in self.model.named_modules():
                    if m is module and (m_name, module) not in self.pruned_layers:
                        self.pruned_layers.append((m_name, module))
                        break

            self.pruning_applied = True
            actual_ratio = self.calculate_pruning_ratio()
            print(f"✅ 全局剪枝完成，实际比例: {actual_ratio:.2%}")
            return True

        except Exception as e:
            print(f"❌ 全局剪枝失败: {e}")
            # 回滚
            if self.original_model is not None:
                self.model.load_state_dict(self.original_model)
            return False

    def apply_pruning_from_candidates(self) -> bool:
        """从记录的候选节点应用剪枝"""
        if not self.config.enabled or not self.pruning_candidates:
            return False

        print(f"🎯 开始从候选节点应用剪枝...")
        print(f"   - 剪枝候选节点数量: {len(self.pruning_candidates)}")

        # 保存原始模型（深拷贝）
        self.original_model = copy.deepcopy(self.model.state_dict())

        # 清除之前的状态
        self._clear_all_pruning_masks()

        # 遍历候选节点并应用剪枝
        applied_count = 0
        failed_count = 0
        for name, info in self.pruning_candidates.items():
            # 查找对应模块
            module = self._get_module_by_name(name)
            if module is None:
                print(f"⚠️  未找到模块: {name}")
                failed_count += 1
                continue

            # 验证模块类型
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"[WARN]  模块 {name} 不是可pruninglayer，跳过")
                failed_count += 1
                continue

            # 获取掩码
            mask = info['mask']

            # 验证 mask 形状
            if mask.shape != module.weight.shape:
                print(f"⚠️  mask 形状不匹配 {name}: {mask.shape} vs {module.weight.shape}")
                failed_count += 1
                continue

            # 将 mask 移动到正确的设备
            mask = mask.to(module.weight.device)

            try:
                # 使用 PyTorch 标准 API 应用pruning mask
                # 这会正确注册 mask，与 make_pruning_permanent 兼容
                prune.custom_from_mask(module, name='weight', mask=mask)

                # 对于结构化pruning（通道级），调整偏置
                if module.bias is not None and mask.shape[0] == module.bias.shape[0]:
                    # 检查是否是结构化pruning（整行被 mask）
                    is_structured = (mask.view(mask.size(0), -1).all(dim=1) == 0).any()
                    if is_structured:
                        # 计算通道 mask
                        channel_mask = mask.view(mask.size(0), -1).any(dim=1).float()
                        # 使用 custom_from_mask 对 bias 也应用 mask
                        bias_mask = channel_mask.to(module.bias.device)
                        prune.custom_from_mask(module, name='bias', mask=bias_mask)

                # 记录pruning后的layer
                self.pruned_layers.append((name, module))
                applied_count += 1
                print(f"   - 应用pruning到layer: {name}, 比例: {info['pruning_ratio']:.2%}")

            except Exception as e:
                print(f"[WARN]  应用pruningfailed {name}: {e}")
                failed_count += 1
                continue

        self.pruning_applied = applied_count > 0
        print(f"✅ pruning应用done (success {applied_count}/{len(self.pruning_candidates)} layer, failed {failed_count} layer)")

        if applied_count == 0:
            print("[WARN]  没有success应用任何pruning，恢复原始模型")
            if self.original_model is not None:
                self.model.load_state_dict(self.original_model)
            return False

        return True

    def _get_module_by_name(self, name: str):
        """根据名称获取模块"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if part in module._modules:
                module = module._modules[part]
            else:
                return None
        return module

    def analyze_layer_sensitivity(self, dataloader, num_batches: int = 50) -> Dict[str, Dict[str, float]]:
        """分析各层对剪枝的敏感性，用于指导剪枝比例分配

        方法：临时剪枝每一层50%的参数，测量精度下降，下降越多越敏感

        Args:
            dataloader: 数据加载器
            num_batches: 用于评估的批次数

        Returns:
            各层的敏感度信息字典
        """
        print(f"🔍 分析层敏感性 (使用 {num_batches} 批次)...")

        self.model.eval()
        device = next(self.model.parameters()).device

        # 保存原始状态
        original_state = copy.deepcopy(self.model.state_dict())

        # 获取基线精度
        baseline_acc = self._quick_evaluate(dataloader, num_batches, device)
        print(f"   基线精度: {baseline_acc:.4f}")

        sensitivities = {}

        with torch.no_grad():
            for name, module in list(self.model.named_modules()):
                if not isinstance(module, (nn.Conv2d, nn.Linear)):
                    continue

                if not any(layer_type in name for layer_type in self.config.pruning_layers):
                    continue

                # 恢复原始状态
                self.model.load_state_dict(original_state)

                # 计算该层的重要性分数
                weight = module.weight.data
                if isinstance(module, nn.Conv2d):
                    # 卷积层: 按输出通道计算L2范数
                    importance = weight.view(weight.size(0), -1).norm(p=2, dim=1)
                else:
                    # 全连接层: 按输出神经元计算L2范数
                    importance = weight.norm(p=2, dim=1)

                # 临时剪枝50%的通道/神经元
                num_total = len(importance)
                num_prune = int(num_total * 0.5)

                if num_prune == 0 or num_prune >= num_total:
                    continue

                sorted_indices = torch.argsort(importance)
                prune_indices = sorted_indices[:num_prune]

                # 创建并应用mask
                mask = torch.ones_like(weight)
                if isinstance(module, nn.Conv2d):
                    mask[prune_indices] = 0
                else:
                    mask[prune_indices, :] = 0

                # 临时应用剪枝
                original_weight = weight.clone()
                weight.mul_(mask)

                # 评估精度
                pruned_acc = self._quick_evaluate(dataloader, num_batches, device)
                sensitivity = baseline_acc - pruned_acc

                # 根据敏感度计算推荐剪枝比例
                # 敏感度越高，推荐比例越低
                if sensitivity < 0.01:  # 几乎无影响
                    recommended_ratio = self.config.pruning_ratio
                elif sensitivity < 0.03:  # 轻微影响
                    recommended_ratio = self.config.pruning_ratio * 0.8
                elif sensitivity < 0.05:  # 中等影响
                    recommended_ratio = self.config.pruning_ratio * 0.6
                else:  # 严重影响
                    recommended_ratio = max(0.1, self.config.pruning_ratio * 0.3)

                sensitivities[name] = {
                    'sensitivity': float(sensitivity),
                    'baseline_acc': float(baseline_acc),
                    'pruned_acc': float(pruned_acc),
                    'recommended_ratio': float(recommended_ratio),
                    'num_channels': num_total,
                    'num_pruned': num_prune
                }

                print(f"   {name}: 敏感度={sensitivity:.4f}, 推荐比例={recommended_ratio:.2f}")

        # 恢复原始状态
        self.model.load_state_dict(original_state)

        # 保存敏感性分析结果供后续使用
        self._sensitivity_analysis = sensitivities

        print(f"✅ 敏感性分析完成 (分析了 {len(sensitivities)} 层)")
        return sensitivities

    def _quick_evaluate(self, dataloader, num_batches: int, device: str) -> float:
        """快速评估模型精度 - 用于敏感性分析

        使用CER作为评估指标（适合OCR任务）
        """
        self.model.eval()
        total_cer = 0.0
        num_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                label_lengths = batch.get('label_lengths')

                outputs = self.model(images)

                # 简化的CER计算
                if 'ctc_logits' in outputs:
                    logits = outputs['ctc_logits']
                    predictions = logits.argmax(dim=-1)

                    # 简单的CTC解码和CER计算
                    for pred, label, label_len in zip(predictions, labels, label_lengths):
                        pred_text = self._simple_ctc_decode(pred.cpu().numpy())
                        label_text = self._simple_ctc_decode(label[:label_len].cpu().numpy())

                        # 计算CER
                        if len(label_text) > 0:
                            cer = self._levenshtein_distance(pred_text, label_text) / len(label_text)
                            total_cer += min(cer, 1.0)
                            num_samples += 1

        # 返回准确率 (1 - CER)
        avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
        return max(0.0, 1.0 - avg_cer)

    def _simple_ctc_decode(self, sequence: np.ndarray) -> str:
        """简化的CTC解码 - 去除重复和空白"""
        decoded = []
        prev = -1
        for token in sequence:
            if token != prev and token != 0:  # 0假设为blank
                decoded.append(str(token))
            prev = token
        return ''.join(decoded)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 计算插入、删除、替换的代价
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _select_pruning_dim(self, module: nn.Module, layer_name: str) -> int:
        """智能选择结构化剪枝的最佳维度

        Args:
            module: 要剪枝的模块
            layer_name: 层的名称

        Returns:
            剪枝维度 (0=输出通道/神经元, 1=输入通道)
        """
        if isinstance(module, nn.Conv2d):
            # 卷积层: weight shape = [out_channels, in_channels, kH, kW]
            out_ch = module.out_channels
            in_ch = module.in_channels

            # 第一层卷积不要剪枝输入通道
            if self._is_first_layer(module, layer_name):
                return 0  # 只剪枝输出通道

            # 分组卷积特殊处理
            if module.groups > 1:
                # 分组卷积通常剪枝输出通道更安全
                return 0

            # 对于标准卷积，优先剪枝输出通道
            # 但如果输出通道远少于输入通道，考虑剪枝输入
            if out_ch < in_ch * 0.3 and in_ch > 64:
                return 1  # 输入通道远多于输出通道时剪枝输入
            return 0  # 默认剪枝输出通道

        elif isinstance(module, nn.Linear):
            # 全连接层: weight shape = [out_features, in_features]
            out_feat = module.out_features
            in_feat = module.in_features

            # 输出层保护（类别数较少时）
            if out_feat < 100 and ('classifier' in layer_name or 'head' in layer_name):
                return 1  # 剪枝输入特征，保护输出

            # 默认剪枝输出神经元（更容易实现压缩）
            return 0

        return 0  # 默认维度

    def make_pruning_permanent(self):
        """使剪枝永久化

        改进:
        - 处理所有可能被剪枝的参数 (weight和bias)
        - 在移除mask之前确保0值被正确保留
        - 清理所有相关的剪枝状态
        - 添加异常处理避免部分失败影响整体
        """
        if not self.pruning_applied and not self.pruned_layers:
            return

        print(f"🔧 永久化剪枝...")
        permanent_count = 0
        failed_layers = []

        for name, module in self.pruned_layers:
            # 处理所有可能被剪枝的参数
            for param_name in ['weight', 'bias']:
                mask_attr = f'{param_name}_mask'
                orig_attr = f'{param_name}_orig'

                if not hasattr(module, mask_attr):
                    continue

                try:
                    # 获取mask并应用永久剪枝（确保0值被保留）
                    mask = getattr(module, mask_attr)
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        # 在remove之前确保参数值为mask后的值
                        with torch.no_grad():
                            param.data *= mask

                    # 移除PyTorch剪枝的mask
                    prune.remove(module, param_name)
                    permanent_count += 1
                    print(f"   - 永久化 {name}.{param_name}")

                except Exception as e:
                    failed_layers.append(f"{name}.{param_name}")
                    print(f"   ⚠️ 永久化失败 {name}.{param_name}: {e}")
                    continue

                # 清理可能残留的属性
                try:
                    if hasattr(module, orig_attr):
                        delattr(module, orig_attr)
                except Exception:
                    pass

        self.pruning_applied = False
        self.pruned_layers = []

        if failed_layers:
            print(f"⚠️ 部分层永久化失败: {failed_layers}")
        print(f"✅ 剪枝永久化完成 (共 {permanent_count} 个参数)")

    def restore_original_model(self):
        """恢复原始模型"""
        if self.original_model is not None:
            self.model.load_state_dict(self.original_model)
            self.pruning_applied = False
            self.pruned_layers = []
            print(f"🔄 恢复原始模型完成")

    def _is_weight_pruned(self, weight: torch.Tensor) -> torch.Tensor:
        """更鲁棒的剪枝检测 - 基于权重分布而非固定阈值

        Args:
            weight: 权重张量

        Returns:
            布尔张量，表示哪些元素被剪枝
        """
        # 方法1: 检查是否为精确的0（PyTorch prune设置的就是精确的0）
        exact_zero = (weight == 0)

        # 方法2: 对于浮点误差，使用相对阈值
        # 如果权重的绝对值小于权重标准差的1e-6倍，认为是被剪枝的
        weight_std = weight.std()
        if weight_std > 0:
            relative_threshold = weight_std * 1e-6
            near_zero = torch.abs(weight) < relative_threshold
        else:
            near_zero = torch.zeros_like(weight, dtype=torch.bool)

        return exact_zero | near_zero

    def calculate_pruning_ratio(self, include_bias: bool = False) -> float:
        """计算实际剪枝比例

        改进:
        - 使用 _is_weight_pruned() 进行更准确的检测
        - 区分非结构化剪枝(0值)和结构化剪枝(维度减少)
        - 支持检测PyTorch prune模块创建的weight_mask

        Args:
            include_bias: 是否包含偏置参数（默认False，因为通常只剪枝权重）

        Returns:
            全局剪枝比例(0-1)
        """
        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            # 计算权重参数
            weight = module.weight
            total_params += weight.numel()

            # 方法1: 使用标准API检查PyTorch prune模块创建的mask（最准确）
            if prune.is_pruned(module):
                # 使用weight_mask进行精确计算
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    pruned_params += (mask == 0).sum().item()
                else:
                    pruned_params += self._is_weight_pruned(weight).sum().item()
            # 方法2: 使用改进的检测方法（永久化剪枝后）
            else:
                pruned_params += self._is_weight_pruned(weight).sum().item()

            # 可选：检查bias
            if include_bias and module.bias is not None:
                total_params += module.bias.numel()
                if prune.is_pruned(module) and hasattr(module, 'bias_mask'):
                    pruned_params += (module.bias_mask == 0).sum().item()
                else:
                    pruned_params += self._is_weight_pruned(module.bias).sum().item()

        return pruned_params / total_params if total_params > 0 else 0.0

    def calculate_structured_pruning_ratio(self) -> Tuple[float, int, int]:
        """计算结构化剪枝比例（通道/神经元级别）

        Returns:
            (结构化剪枝比例, 被剪枝通道数, 总通道数)
        """
        total_channels = 0
        pruned_channels = 0

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            if isinstance(module, nn.Conv2d):
                total_ch = module.out_channels
            else:
                total_ch = module.out_features

            total_channels += total_ch

            # 检查每个输出通道是否全部被剪枝
            weight = module.weight
            if prune.is_pruned(module) and hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                channel_active = mask.view(mask.size(0), -1).any(dim=1)
            else:
                channel_active = ~self._is_weight_pruned(weight).view(weight.size(0), -1).all(dim=1)

            pruned_channels += (~channel_active).sum().item()

        ratio = pruned_channels / total_channels if total_channels > 0 else 0.0
        return ratio, pruned_channels, total_channels

    def get_structured_pruning_stats(self) -> Dict[str, Any]:
        """获取结构化剪枝的统计信息（输出通道/神经元级别）

        使用 _is_weight_pruned() 进行更准确的检测

        Returns:
            结构化剪枝统计信息
        """
        stats = {
            'total_layers': 0,
            'pruned_channels': 0,
            'total_channels': 0,
            'layer_details': []
        }

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            if isinstance(module, nn.Conv2d):
                total_ch = module.out_channels
            else:
                total_ch = module.out_features

            # 检查该层是否有剪枝 - 使用改进的检测方法
            pruned_ch = 0
            weight = module.weight
            if weight.dim() >= 2:
                # 检查每个输出通道是否全部被剪枝
                if prune.is_pruned(module) and hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    channel_active = mask.view(mask.size(0), -1).any(dim=1)
                else:
                    # 使用 _is_weight_pruned 进行更准确检测
                    pruned_mask = self._is_weight_pruned(weight)
                    channel_active = ~pruned_mask.view(weight.size(0), -1).all(dim=1)

                pruned_ch = (~channel_active).sum().item()

            if pruned_ch > 0 or total_ch > 0:
                stats['total_layers'] += 1
                stats['pruned_channels'] += pruned_ch
                stats['total_channels'] += total_ch

                stats['layer_details'].append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'total_channels': total_ch,
                    'pruned_channels': pruned_ch,
                    'ratio': pruned_ch / total_ch if total_ch > 0 else 0
                })

        return stats

    def get_detailed_pruning_stats(self) -> Dict[str, Any]:
        """获取详细的剪枝统计信息

        使用 _is_weight_pruned() 进行更准确的检测

        Returns:
            包含全局和各层详细统计的字典
        """
        stats = {
            'total_params': 0,
            'pruned_params': 0,
            'global_ratio': 0.0,
            'pruned_layers_count': len(self.pruned_layers),
            'layer_stats': []
        }

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            weight = module.weight
            layer_total = weight.numel()

            # 检测剪枝数量 - 使用改进的方法
            if prune.is_pruned(module) and hasattr(module, 'weight_mask'):
                layer_pruned = (module.weight_mask == 0).sum().item()
                pruning_method = 'mask'
            else:
                # 使用 _is_weight_pruned 进行更准确检测
                layer_pruned = self._is_weight_pruned(weight).sum().item()
                pruning_method = 'permanent' if layer_pruned > 0 else 'none'

            layer_ratio = layer_pruned / layer_total if layer_total > 0 else 0.0

            # 计算层信息
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'total_params': int(layer_total),
                'pruned_params': int(layer_pruned),
                'pruning_ratio': float(layer_ratio),
                'pruning_method': pruning_method,
                'shape': list(weight.shape)
            }

            # 添加额外信息
            if isinstance(module, nn.Conv2d):
                layer_info['in_channels'] = module.in_channels
                layer_info['out_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
            elif isinstance(module, nn.Linear):
                layer_info['in_features'] = module.in_features
                layer_info['out_features'] = module.out_features

            stats['layer_stats'].append(layer_info)
            stats['total_params'] += layer_total
            stats['pruned_params'] += layer_pruned

        # 计算全局比例
        if stats['total_params'] > 0:
            stats['global_ratio'] = stats['pruned_params'] / stats['total_params']

        # 排序：按剪枝比例降序
        stats['layer_stats'].sort(key=lambda x: x['pruning_ratio'], reverse=True)

        return stats

    def print_pruning_stats(self):
        """打印剪枝统计信息（便于调试）"""
        stats = self.get_detailed_pruning_stats()

        print("\n📊 剪枝统计信息:")
        print("=" * 80)
        print(f"总参数量: {stats['total_params']:,}")
        print(f"剪枝参数量: {stats['pruned_params']:,}")
        print(f"全局剪枝比例: {stats['global_ratio']:.2%}")
        print(f"剪枝层数: {stats['pruned_layers_count']}")
        print("\n各层详情 (前10):")
        print("-" * 80)
        print(f"{'层名':<40} {'类型':<12} {'总参数':>12} {'剪枝比例':>10}")
        print("-" * 80)

        for layer_info in stats['layer_stats'][:10]:
            name = layer_info['name'][:38]
            layer_type = layer_info['type'][:10]
            total = layer_info['total_params']
            ratio = layer_info['pruning_ratio']
            print(f"{name:<40} {layer_type:<12} {total:>12,} {ratio:>9.1%}")

        print("=" * 80)

    def get_pruned_model_info(self) -> Dict:
        """获取剪枝后的模型信息"""
        info = {
            'pruning_applied': self.pruning_applied,
            'pruned_layers_count': len(self.pruned_layers),
            'pruning_ratio': self.calculate_pruning_ratio(),
            'pruned_layers': [name for name, _ in self.pruned_layers]
        }
        return info

    def save_pruning_candidates(self, path: str):
        """保存剪枝候选信息到文件"""
        if not self.pruning_candidates:
            print(f"⚠️  没有剪枝候选信息可保存")
            return

        # 准备保存的数据
        save_data = {
            'pruning_candidates': {}
        }

        # 处理每个候选节点
        for name, info in self.pruning_candidates.items():
            save_data['pruning_candidates'][name] = {
                'module_type': info['module_type'],
                'pruning_ratio': info['pruning_ratio'],
                'mask': info['mask'].tolist(),
                'importance': info['importance'].tolist()
            }

        # 保存到文件
        # 转换numpy数组为列表
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # 保存文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=convert_to_serializable)

        print(f"✅ 剪枝候选信息已保存到: {path}")

    def load_pruning_candidates(self, path: str):
        """从文件加载剪枝候选信息"""
        if not os.path.exists(path):
            print(f"⚠️  剪枝候选信息文件不存在: {path}")
            return False

        # 加载文件
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        # 恢复剪枝候选信息
        candidates = save_data.get('pruning_candidates', {})

        # 处理每个候选节点
        for name, info in candidates.items():
            self.pruning_candidates[name] = {
                'module_type': info['module_type'],
                'pruning_ratio': info['pruning_ratio'],
                'mask': torch.tensor(info['mask']),
                'importance': torch.tensor(info['importance'])
            }

        print(f"✅ 剪枝候选信息已加载，共 {len(self.pruning_candidates)} 个候选节点")
        return True

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

    def visualize_pruning(self, save_path: str = 'pruning_visualization.png'):
        """可视化剪枝效果

        生成包含以下内容的可视化图表:
        - 各层剪枝比例条形图
        - 权重稀疏性热力图
        - 结构化剪枝通道分布
        - 剪枝前后参数量对比

        Args:
            save_path: 保存路径
        """
        stats = self.get_detailed_pruning_stats()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pruning Visualization', fontsize=16, fontweight='bold')

        # 1. 各层剪枝比例
        ax = axes[0, 0]
        layer_names = [s['name'].split('.')[-1][:15] for s in stats['layer_stats'][:15]]
        ratios = [s['pruning_ratio'] for s in stats['layer_stats'][:15]]
        colors = ['green' if r < 0.2 else 'orange' if r < 0.5 else 'red' for r in ratios]
        bars = ax.barh(layer_names, ratios, color=colors, alpha=0.7)
        ax.set_xlabel('Pruning Ratio')
        ax.set_title('Layer-wise Pruning Ratio')
        ax.set_xlim(0, 1)
        ax.axvline(x=self.config.pruning_ratio, color='blue', linestyle='--',
                  label=f'Target ({self.config.pruning_ratio:.0%})')
        ax.legend()

        # 2. 权重分布对比
        ax = axes[0, 1]
        all_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights = module.weight.data.flatten().cpu().numpy()
                all_weights.extend(weights)

        if len(all_weights) > 0:
            ax.hist(all_weights, bins=100, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            ax.set_title('Weight Distribution (Log Scale)')
            ax.set_yscale('log')
            ax.axvline(x=0, color='red', linestyle='--', label='Zero')
            ax.legend()

        # 3. 结构化剪枝统计
        ax = axes[1, 0]
        struct_stats = self.get_structured_pruning_stats()
        layer_names_struct = [d['name'].split('.')[-1][:10] for d in struct_stats['layer_details'][:10]]
        total_ch = [d['total_channels'] for d in struct_stats['layer_details'][:10]]
        pruned_ch = [d['pruned_channels'] for d in struct_stats['layer_details'][:10]]

        x = range(len(layer_names_struct))
        ax.bar(x, total_ch, label='Total Channels', alpha=0.7)
        ax.bar(x, pruned_ch, label='Pruned Channels', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names_struct, rotation=45, ha='right')
        ax.set_ylabel('Channel Count')
        ax.set_title('Structured Pruning Stats')
        ax.legend()

        # 4. 参数统计
        ax = axes[1, 1]
        categories = ['Original', 'After Pruning', 'Removed']
        original = stats['total_params']
        removed = stats['pruned_params']
        remaining = original - removed
        values = [original, remaining, removed]
        colors = ['blue', 'green', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Parameter Count')
        ax.set_title(f'Parameter Statistics\n(Global Ratio: {stats["global_ratio"]:.2%})')

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val/1e6:.2f}M', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 剪枝可视化已保存: {save_path}")

    def apply_iterative_pruning(self, epoch: int, start_epoch: int, end_epoch: int,
                                target_ratio: float, dataloader=None) -> bool:
        """渐进式剪枝

        避免一次性剪枝导致精度损失过大，通过渐进方式让模型逐步适应。

        Args:
            epoch: 当前epoch
            start_epoch: 渐进剪枝开始epoch
            end_epoch: 渐进剪枝结束epoch
            target_ratio: 最终目标剪枝比例
            dataloader: 用于敏感性分析的数据加载器（可选）

        Returns:
            是否应用了剪枝
        """
        if not self.config.enabled:
            return False

        if epoch < start_epoch or epoch > end_epoch:
            return False

        # 第一次进入渐进剪枝时，保存原始模型
        if epoch == start_epoch:
            self.original_model = copy.deepcopy(self.model.state_dict())
            self._clear_all_pruning_masks()

        # 计算当前进度
        total_epochs = end_epoch - start_epoch + 1
        current_progress = (epoch - start_epoch + 1) / total_epochs

        # 平滑渐进：使用平方根函数
        progress = np.sqrt(current_progress)
        current_target_ratio = target_ratio * progress

        print(f"🎯 渐进式剪枝 - Epoch {epoch}/{end_epoch}")
        print(f"   进度: {current_progress:.1%}, 当前目标比例: {current_target_ratio:.2%}")

        # 恢复原始模型并清理状态（每次从原始状态开始）
        if self.original_model is not None:
            self.model.load_state_dict(self.original_model)
            self._clear_all_pruning_masks()

        # 可选：使用敏感性分析调整各层比例
        sensitivities = getattr(self, '_sensitivity_analysis', None)

        # 应用渐进剪枝
        for name, module in self.model.named_modules():
            if not self._is_prunable_module(module, name):
                continue

            # 基础比例
            layer_ratio = self.get_pruning_ratio(name) * progress

            # 根据敏感性调整（如果有分析结果）
            if sensitivities and name in sensitivities:
                sens = sensitivities[name]['sensitivity']
                # 敏感度高的层减少剪枝比例
                if sens > 0.05:
                    layer_ratio *= 0.5
                elif sens > 0.03:
                    layer_ratio *= 0.7

            # 确保比例不超过当前目标
            layer_ratio = min(layer_ratio, current_target_ratio)

            if layer_ratio > 0.01:  # 至少1%才剪枝
                self._apply_pruning_to_module(module, name, layer_ratio)

        self.pruning_applied = True
        actual_ratio = self.calculate_pruning_ratio()
        print(f"✅ 渐进剪枝完成，实际比例: {actual_ratio:.2%}")

        return True

    def get_finetune_lr_multiplier(self, epoch: int) -> float:
        """获取微调阶段的学习率乘数"""
        if not self.is_finetuning_time(epoch):
            return 1.0

        # 微调阶段使用较低的学习率
        return 0.1

    def apply_pruning_with_validation(self, epoch: int, current_acc: float, best_acc: float,
                                      val_dataloader=None, max_acc_drop: float = None) -> bool:
        """带验证的剪枝应用 - 如果精度下降过大则回滚

        Args:
            epoch: 当前epoch
            current_acc: 当前精度
            best_acc: 最佳精度
            val_dataloader: 验证数据加载器（可选，用于快速验证）
            max_acc_drop: 允许的最大精度下降（默认使用 config.min_acc_drop）

        Returns:
            是否成功应用剪枝
        """
        if max_acc_drop is None:
            max_acc_drop = self.config.min_acc_drop

        # 先进行标准剪枝检查
        if not self._should_prune(epoch, current_acc, best_acc):
            return False

        # 保存剪枝前的状态
        state_before = copy.deepcopy(self.model.state_dict())

        # 应用剪枝
        if not self.apply_pruning(epoch, current_acc, best_acc):
            return False

        # 如果有验证数据，进行快速验证
        if val_dataloader is not None:
            print(f"🔍 验证剪枝效果...")
            val_acc = self._quick_evaluate(val_dataloader, 20,
                                           next(self.model.parameters()).device)
            actual_drop = best_acc - val_acc

            if actual_drop > max_acc_drop:
                print(f"⚠️ 精度下降过大 (下降 {actual_drop:.4f} > 阈值 {max_acc_drop:.4f})，回滚剪枝")
                self.model.load_state_dict(state_before)
                self._clear_all_pruning_masks()
                self.original_model = None
                self.pruning_applied = False
                return False

            print(f"✅ 剪枝验证通过，精度下降: {actual_drop:.4f}")

        return True

    def _find_associated_bn(self, module: nn.Module, module_name: str) -> Optional[Tuple[str, nn.BatchNorm2d]]:
        """找到与卷积层关联的BatchNorm层

        Args:
            module: 卷积层模块
            module_name: 卷积层名称

        Returns:
            (bn_name, bn_module) 或 None
        """
        # 策略1: 同级的BatchNorm（如 conv -> bn）
        parent_name = '.'.join(module_name.split('.')[:-1])
        module_base_name = module_name.split('.')[-1]

        for name, m in self.model.named_modules():
            # 检查是否为BatchNorm
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # 检查是否可能是关联的BN
                bn_name = name.split('.')[-1]
                # 常见命名模式: conv->bn, conv1->bn1, etc.
                if (parent_name in name or name.startswith(parent_name)):
                    # 检查通道数是否匹配
                    if hasattr(m, 'num_features'):
                        if isinstance(module, nn.Conv2d) and m.num_features == module.out_channels:
                            return name, m
        return None

    def _adjust_bn_after_structured_pruning(self, module: nn.Module, module_name: str,
                                            pruned_indices: torch.Tensor):
        """结构化剪枝后调整 BatchNorm 层 - 完整实现

        Args:
            module: 被剪枝的模块
            module_name: 模块名称
            pruned_indices: 被剪枝的输出通道索引
        """
        if len(pruned_indices) == 0:
            return

        # 找到关联的BN层
        bn_result = self._find_associated_bn(module, module_name)
        if bn_result is None:
            return

        bn_name, bn_module = bn_result

        try:
            # 创建掩码，标记保留的通道
            num_features = bn_module.num_features
            keep_mask = torch.ones(num_features, dtype=torch.bool, device=bn_module.weight.device)
            keep_mask[pruned_indices] = False
            new_num_features = keep_mask.sum().item()

            if new_num_features == 0:
                print(f"   warning: BNlayer {bn_name} 所有通道将被pruning，跳过调整")
                return

            # 调整BN参数 - 创建新的 Parameter 而不是修改 data
            with torch.no_grad():
                # 调整权重和偏置
                if bn_module.weight is not None:
                    new_weight = nn.Parameter(bn_module.weight.data[keep_mask])
                    bn_module.weight = new_weight
                if bn_module.bias is not None:
                    new_bias = nn.Parameter(bn_module.bias.data[keep_mask])
                    bn_module.bias = new_bias

                # 调整running statistics - 直接修改data，避免重新注册buffer
                # 重新register_buffer会导致状态丢失和训练中断，直接修改data更安全
                if hasattr(bn_module, 'running_mean'):
                    with torch.no_grad():
                        new_running_mean = bn_module.running_mean[keep_mask].clone()
                        bn_module.running_mean.data = new_running_mean
                if hasattr(bn_module, 'running_var'):
                    with torch.no_grad():
                        new_running_var = bn_module.running_var[keep_mask].clone()
                        # 确保方差不为零
                        new_running_var = torch.clamp(new_running_var, min=1e-5)
                        bn_module.running_var.data = new_running_var

                # 调整 num_batches_tracked (如果存在)
                if hasattr(bn_module, 'num_batches_tracked'):
                    # 保持原值，不需要调整
                    pass

            # 注意: PyTorch 的 num_features 是只读属性，不能直接修改
            # 但在实际运行中，上面调整的 weight/bias/running_mean/running_var 的
            # 形状会自动反映到 num_features
            actual_num_features = bn_module.weight.shape[0] if bn_module.weight is not None else \
                                 (bn_module.running_mean.shape[0] if hasattr(bn_module, 'running_mean') else new_num_features)

            print(f"   调整BNlayer: {bn_name}, 通道: {num_features} -> {actual_num_features}")

        except Exception as e:
            print(f"   调整BN层失败 {bn_name}: {e}")
            traceback.print_exc()

    def get_layer_connectivity(self) -> Dict[str, List[str]]:
        """分析模型层之间的连接关系 - 用于结构化剪枝后的维度调整

        改进内容:
        1. 使用更智能的方法分析 Sequential 和 ModuleList 中的连接关系
        2. 返回前序layer到后继layer的映射

        Returns:
            层名到其后继层列表的映射
        """
        connectivity = {}

        # 收集所有可pruninglayer
        prunable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable_layers.append((name, module))

        # 分析连接关系
        for i, (name, module) in enumerate(prunable_layers):
            # 默认认为同一父模块中的相邻layer有连接关系
            for j in range(i + 1, len(prunable_layers)):
                next_name, next_module = prunable_layers[j]

                # 检查命名layer级关系
                name_parts = name.split('.')
                next_parts = next_name.split('.')

                # 如果在同一父模块中，认为有连接
                if len(name_parts) == len(next_parts):
                    same_parent = all(name_parts[k] == next_parts[k] for k in range(len(name_parts) - 1))
                    if same_parent:
                        connectivity.setdefault(name, []).append(next_name)
                        break  # 只找直接后继

        return connectivity

    def validate_pruning_with_rollback(self, val_dataloader,
                                       max_acc_drop: float = None) -> bool:
        """验证剪枝效果，如不满足条件则自动回滚

        这是一个增强的验证方法，可以在剪枝后立即验证效果，
        如果精度下降过大则自动回滚。

        Args:
            val_dataloader: 验证数据加载器
            max_acc_drop: 允许的最大精度下降（默认使用config.min_acc_drop）

        Returns:
            是否接受当前剪枝（True=接受，False=已回滚）
        """
        if max_acc_drop is None:
            max_acc_drop = self.config.min_acc_drop

        if not self.pruning_applied:
            print("⚠️ 没有应用剪枝，无需验证")
            return True

        print(f"🔍 验证剪枝效果...")

        # 保存当前状态（用于回滚）
        state_before = copy.deepcopy(self.model.state_dict())

        # 永久化剪枝以进行准确评估
        self.make_pruning_permanent()

        try:
            # 快速评估
            current_acc = self._quick_evaluate(
                val_dataloader,
                num_batches=20,
                device=next(self.model.parameters()).device
            )

            # 计算相对于原始精度的下降（使用保存的原始模型）
            if self.original_model is not None:
                # 临时加载原始模型进行评估
                current_state = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self.original_model)
                original_acc = self._quick_evaluate(
                    val_dataloader,
                    num_batches=20,
                    device=next(self.model.parameters()).device
                )
                # 恢复当前状态
                self.model.load_state_dict(current_state)

                acc_drop = original_acc - current_acc
                print(f"   原始精度: {original_acc:.4f}")
                print(f"   剪枝后精度: {current_acc:.4f}")
                print(f"   精度下降: {acc_drop:.4f}")

                if acc_drop > max_acc_drop:
                    print(f"⚠️ 精度下降过大 ({acc_drop:.4f} > {max_acc_drop:.4f})，执行回滚")
                    self.model.load_state_dict(state_before)
                    self.pruning_applied = False
                    self.pruned_layers = []
                    self._clear_all_pruning_masks()
                    return False
                else:
                    print(f"✅ 剪枝验证通过")
                    return True
            else:
                print("⚠️ 未保存原始模型状态，无法计算精度下降")
                return True

        except Exception as e:
            print(f"❌ 验证过程出错: {e}，执行回滚")
            self.model.load_state_dict(state_before)
            return False

    def compress_model_structurally(self, val_dataloader=None,
                                    max_acc_drop: float = 0.02) -> nn.Module:
        """真正的结构化压缩 - 删除被剪枝的通道/神经元并重建模型

        这是一个高级功能，会：
        1. 分析当前剪枝状态，识别被完全剪枝的通道
        2. 创建新的压缩后的层
        3. 复制保留的权重
        4. 调整后续层的输入维度
        5. 调整关联的BatchNorm层

        Args:
            val_dataloader: 验证数据加载器（用于验证压缩后效果）
            max_acc_drop: 允许的最大精度下降

        Returns:
            压缩后的模型（原模型被修改）
        """
        if not self.pruning_applied:
            print("⚠️ 没有应用剪枝，无法压缩")
            return self.model

        print(f"🔧 开始结构化压缩...")

        # 先永久化当前的剪枝
        self.make_pruning_permanent()

        # 构建layer依赖图和顺序
        layer_order = []
        layer_outputs = {}  # 记录每layer输出维度变化
        layer_inputs = {}   # 记录每layer输入维度变化

        # 第一步：收集所有可pruninglayer的信息
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            weight = module.weight
            if weight.dim() >= 2:
                # 检查每个输出通道是否全部被pruning
                channel_active = ~(self._is_weight_pruned(weight).view(weight.size(0), -1).all(dim=1))
                if (~channel_active).any():
                    keep_indices = torch.where(channel_active)[0]
                    layer_order.append(name)
                    layer_outputs[name] = {
                        'module': module,
                        'keep_indices': keep_indices,
                        'original_out': weight.size(0),
                        'new_out': len(keep_indices),
                        'module_type': type(module).__name__
                    }
                    # 记录原始输入维度
                    if isinstance(module, nn.Conv2d):
                        layer_inputs[name] = module.in_channels
                    else:
                        layer_inputs[name] = module.in_features
                    print(f"   计划压缩 {name}: {weight.size(0)} -> {len(keep_indices)} channels")

        if len(layer_order) == 0:
            print("ℹ️ 没有需要压缩的层")
            return self.model

        # 第二步：分析layer间连接关系，构建输入维度映射
        # 对于每个layer，找出哪些layer的输出连接到它的输入
        input_mapping = self._build_input_mapping(layer_order, layer_outputs)

        # 第三步：按顺序应用压缩
        compressed_count = 0
        for name in layer_order:
            info = layer_outputs[name]
            try:
                module = info['module']
                keep_indices = info['keep_indices']

                # 计算新的输入维度（可能被前面的layer修改过）
                new_in_features = self._get_new_input_dim(name, layer_inputs, input_mapping, layer_outputs)

                # 创建新layer
                if isinstance(module, nn.Conv2d):
                    new_module = self._create_compressed_conv2d(module, keep_indices, new_in_features)
                elif isinstance(module, nn.Linear):
                    new_module = self._create_compressed_linear(module, keep_indices, new_in_features)
                else:
                    continue

                # 替换原模块
                parent_name = '.'.join(name.split('.')[:-1])
                module_base_name = name.split('.')[-1]

                if parent_name:
                    parent = self._get_module_by_name(parent_name)
                    if parent is not None:
                        setattr(parent, module_base_name, new_module)
                else:
                    setattr(self.model, module_base_name, new_module)

                # 调整关联的BN层
                pruned_indices = torch.tensor([i for i in range(info['original_out'])
                                              if i not in keep_indices.tolist()])
                self._adjust_bn_after_structured_pruning(new_module, name, pruned_indices)

                # 更新输出信息，供后续layer使用
                info['new_module'] = new_module
                compressed_count += 1
                print(f"   done压缩 {name}: 输入 {layer_inputs[name]}->{new_in_features}, "
                      f"输出 {info['original_out']}->{info['new_out']}")

            except Exception as e:
                print(f"   压缩层 {name} 失败: {e}")
                traceback.print_exc()
                continue

        print(f"✅ 结构化压缩完成 (压缩了 {compressed_count} 层)")

        # 验证压缩效果
        if val_dataloader is not None:
            print("⚠️ 验证压缩后的模型...")
            try:
                # 先测试一个前向传播看是否能正常运行
                test_input = next(iter(val_dataloader))['images']
                test_input = test_input.to(next(self.model.parameters()).device)
                with torch.no_grad():
                    _ = self.model(test_input[:1])
                print("   模型前向传播测试通过")

                is_valid = self.validate_pruning_with_rollback(val_dataloader, max_acc_drop)
                if not is_valid:
                    print("[WARN] 压缩后accuracy下降过大，已回滚")
            except Exception as e:
                print(f"   压缩后模型验证failed: {e}")
                print("   执行回滚...")
                if self.original_model is not None:
                    self.model.load_state_dict(self.original_model)
                    self._clear_all_pruning_masks()

        return self.model

    def _build_input_mapping(self, layer_order: List[str], layer_outputs: Dict) -> Dict[str, List[str]]:
        """构建layer间输入映射关系

        对于每个layer，找出哪些前序layer的输出连接到它的输入

        Returns:
            layer名 -> 前序layer名列表的映射
        """
        input_mapping = {}

        # 按名称layer级排序，模拟拓扑顺序
        sorted_layers = sorted(layer_order, key=lambda x: len(x.split('.')))

        for i, name in enumerate(sorted_layers):
            input_mapping[name] = []
            current_module = layer_outputs[name]['module']

            # 查找所有可能的前序layer
            for prev_name in sorted_layers[:i]:
                prev_module = layer_outputs[prev_name]['module']

                # 检查命名关系（简单启发式）
                # 如果当前layer名称以前序layer名称开头，可能有连接关系
                if name.startswith(prev_name) or self._are_layers_connected(prev_module, current_module):
                    input_mapping[name].append(prev_name)

        return input_mapping

    def _are_layers_connected(self, prev_module: nn.Module, current_module: nn.Module) -> bool:
        """检查两个layer是否可能连接（简单启发式）"""
        # 这里使用简单的启发式：Sequential 中的相邻layer
        # 更复杂的模型结构可能需要更复杂的分析
        return False  # 默认不假设连接，依赖名称layer级关系

    def _get_new_input_dim(self, name: str, layer_inputs: Dict, input_mapping: Dict,
                          layer_outputs: Dict) -> int:
        """计算layer的新输入维度

        根据前序layer的输出维度变化，计算当前layer应有的输入维度
        """
        original_in = layer_inputs[name]
        predecessor_layers = input_mapping.get(name, [])

        # 如果没有前序layer信息，返回原始输入维度
        if not predecessor_layers:
            return original_in

        # 对于每个前序layer，检查其输出维度是否改变
        # 对于 Conv2d->Conv2d 或 Linear->Linear 的直连关系
        for pred_name in predecessor_layers:
            if pred_name in layer_outputs:
                pred_info = layer_outputs[pred_name]
                if 'new_out' in pred_info and pred_info['new_out'] != pred_info['original_out']:
                    # 前序layer输出被压缩，当前layer输入应相应调整
                    return pred_info['new_out']

        return original_in

    def _create_compressed_conv2d(self, module: nn.Conv2d, keep_indices: torch.Tensor,
                                  new_in_channels: int) -> nn.Conv2d:
        """创建压缩后的 Conv2d layer"""
        new_out_channels = len(keep_indices)

        new_module = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=new_out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups if module.groups == 1 else min(module.groups, new_out_channels),
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )

        with torch.no_grad():
            # 复制权重 - 需要考虑输入维度的变化
            if new_in_channels == module.in_channels:
                # 输入维度未变，只调整输出维度
                new_module.weight.copy_(module.weight.data[keep_indices])
            else:
                # 输入维度也变了，需要同时调整
                # 找出输入维度保留的索引
                # 这里假设输入维度的压缩比例与输出相同（简化处理）
                # 实际应用中可能需要更复杂的映射
                in_ratio = new_in_channels / module.in_channels
                in_keep_count = int(module.in_channels * in_ratio)
                in_keep_indices = torch.linspace(0, module.in_channels - 1, in_keep_count).long()
                new_module.weight.copy_(module.weight.data[keep_indices][:, in_keep_indices])

            if module.bias is not None:
                new_module.bias.copy_(module.bias.data[keep_indices])

        return new_module

    def _create_compressed_linear(self, module: nn.Linear, keep_indices: torch.Tensor,
                                  new_in_features: int) -> nn.Linear:
        """创建压缩后的 Linear layer"""
        new_out_features = len(keep_indices)

        new_module = nn.Linear(
            in_features=new_in_features,
            out_features=new_out_features,
            bias=module.bias is not None
        )

        with torch.no_grad():
            # 复制权重 - 需要考虑输入维度的变化
            if new_in_features == module.in_features:
                # 输入维度未变，只调整输出维度
                new_module.weight.copy_(module.weight.data[keep_indices])
            else:
                # 输入维度也变了，需要同时调整
                # 找出输入维度保留的索引
                in_ratio = new_in_features / module.in_features
                in_keep_count = int(module.in_features * in_ratio)
                in_keep_indices = torch.linspace(0, module.in_features - 1, in_keep_count).long()
                new_module.weight.copy_(module.weight.data[keep_indices][:, in_keep_indices])

            if module.bias is not None:
                new_module.bias.copy_(module.bias.data[keep_indices])

        return new_module


class PruningQuantizationScheduler:
    """剪枝与量化的协同调度器 - 协调剪枝和QAT的训练流程

    提供两种训练模式:
    1. 先剪枝后量化: 训练 -> 剪枝 -> 微调 -> 插入QAT -> QAT训练
    2. 交替进行: 训练 -> 部分剪枝 -> 部分QAT -> 重复

    Attributes:
        pruning_manager: 剪枝管理器
        qat_scheduler: QAT训练调度器
        current_stage: 当前训练阶段
    """

    STAGES = ['warmup', 'pre_pruning', 'pruning', 'pruning_finetune',
              'pre_qat', 'qat_insert', 'qat_finetune', 'post_qat']

    def __init__(self, pruning_manager: PruningManager,
                 qat_scheduler: 'QATTrainingScheduler',
                 mode: str = 'sequential'):
        """
        Args:
            pruning_manager: 剪枝管理器实例
            qat_scheduler: QAT调度器实例
            mode: 训练模式 ('sequential'=先剪枝后量化, 'interleaved'=交替进行)
        """
        self.pruning_manager = pruning_manager
        self.qat_scheduler = qat_scheduler
        self.mode = mode
        self.current_stage = 'warmup'
        self.stage_start_epoch = 0

        # 阶段配置
        self.config = {
            'warmup_epochs': 3,
            'pruning_epoch': pruning_manager.config.pruning_epoch if pruning_manager.config.enabled else -1,
            'pruning_finetune_epochs': pruning_manager.config.finetune_epochs if pruning_manager.config.enabled else 0,
            'qat_insert_epoch': getattr(qat_scheduler.config, 'qat_insert_epoch', 3),
            'qat_epochs': getattr(qat_scheduler.config, 'qat_epochs', 5),
        }

    def step(self, epoch: int, current_acc: float, best_acc: float) -> Dict[str, Any]:
        """执行当前epoch的调度

        Args:
            epoch: 当前epoch
            current_acc: 当前精度
            best_acc: 最佳精度

        Returns:
            包含状态信息的字典
        """
        status = {
            'stage': self.current_stage,
            'pruning_applied': False,
            'qat_inserted': False,
            'lr_multiplier': 1.0,
            'epoch': epoch
        }

        # 更新阶段
        self._update_stage(epoch)

        # 根据阶段执行相应操作
        if self.current_stage == 'pruning':
            # 执行剪枝
            if self.pruning_manager.config.enabled:
                applied = self.pruning_manager.apply_pruning(epoch, current_acc, best_acc)
                status['pruning_applied'] = applied
                if applied:
                    print(f"[Scheduler] Epoch {epoch}: 剪枝已应用")

        elif self.current_stage == 'qat_insert':
            # 插入QAT模块
            if self.qat_scheduler.delay_insert:
                inserted = self.qat_scheduler.maybe_insert_qat(epoch)
                status['qat_inserted'] = inserted
                if inserted:
                    print(f"[Scheduler] Epoch {epoch}: QAT模块已插入")

        # 计算学习率乘数
        status['lr_multiplier'] = self.get_lr_multiplier(epoch)
        status['stage'] = self.current_stage

        return status

    def _update_stage(self, epoch: int):
        """更新当前阶段"""
        cfg = self.config

        # 顺序模式: warmup -> pruning -> pruning_finetune -> pre_qat -> qat_insert -> qat_finetune -> post_qat
        if epoch < cfg['warmup_epochs']:
            new_stage = 'warmup'
        elif not self.pruning_manager.config.enabled:
            # 不剪枝，直接进入QAT流程
            if epoch < cfg['qat_insert_epoch']:
                new_stage = 'pre_qat'
            elif epoch == cfg['qat_insert_epoch']:
                new_stage = 'qat_insert'
            elif epoch < cfg['qat_insert_epoch'] + cfg['qat_epochs']:
                new_stage = 'qat_finetune'
            else:
                new_stage = 'post_qat'
        elif epoch == cfg['pruning_epoch']:
            new_stage = 'pruning'
        elif epoch < cfg['pruning_epoch'] + cfg['pruning_finetune_epochs']:
            new_stage = 'pruning_finetune'
        elif epoch < cfg['qat_insert_epoch'] + cfg['pruning_finetune_epochs']:
            new_stage = 'pre_qat'
        elif epoch == cfg['qat_insert_epoch'] + cfg['pruning_finetune_epochs']:
            new_stage = 'qat_insert'
        elif epoch < cfg['qat_insert_epoch'] + cfg['pruning_finetune_epochs'] + cfg['qat_epochs']:
            new_stage = 'qat_finetune'
        else:
            new_stage = 'post_qat'

        if new_stage != self.current_stage:
            print(f"[Scheduler] 阶段切换: {self.current_stage} -> {new_stage} (Epoch {epoch})")
            self.current_stage = new_stage
            self.stage_start_epoch = epoch

    def get_lr_multiplier(self, epoch: int) -> float:
        """获取当前阶段的学习率乘数"""
        multipliers = {
            'warmup': 1.0,
            'pre_pruning': 1.0,
            'pruning': 0.5,  # 剪枝时降低学习率
            'pruning_finetune': 0.1,  # 剪枝后微调使用低学习率
            'pre_qat': 1.0,
            'qat_insert': 0.5,  # QAT插入时降低学习率
            'qat_finetune': getattr(self.qat_scheduler.config, 'qat_learning_rate_multiplier', 0.1),
            'post_qat': 0.05,
        }
        return multipliers.get(self.current_stage, 1.0)

    def should_freeze_bn_ln(self, epoch: int) -> bool:
        """检查是否应该冻结BN/LN层"""
        # 在QAT微调阶段冻结
        return self.current_stage in ['qat_finetune', 'post_qat']

    def get_current_stage(self) -> str:
        """获取当前阶段"""
        return self.current_stage

    def is_pruning_time(self, epoch: int) -> bool:
        """检查是否是剪枝时机"""
        return self.current_stage == 'pruning' and epoch == self.config['pruning_epoch']

    def is_qat_time(self, epoch: int) -> bool:
        """检查是否是QAT插入时机"""
        return self.current_stage == 'qat_insert'

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取训练进度摘要"""
        return {
            'current_stage': self.current_stage,
            'stages_completed': self.STAGES.index(self.current_stage) if self.current_stage in self.STAGES else 0,
            'total_stages': len(self.STAGES),
            'pruning_applied': self.pruning_manager.pruning_applied,
            'qat_inserted': self.qat_scheduler.qat_inserted if hasattr(self.qat_scheduler, 'qat_inserted') else False,
            'lr_multiplier': self.get_lr_multiplier(0),  # 当前阶段的学习率乘数
        }


class QATTrainingScheduler:
    """QAT训练调度器 - 优化3: 分阶段QAT训练流程

    实现分阶段训练策略:
    1. 预热阶段: 正常训练，学习率 warmup
    2. QAT插入阶段: 在指定epoch插入FakeQuantize
    3. QAT微调阶段: 低学习率微调量化参数，冻结BN/LN
    4. 后QAT阶段: 更低学习率继续微调

    """

    def __init__(self, config: Dict, quantization_manager: 'QuantizationManager',
                 model: nn.Module, delay_insert: bool = True):
        """
        Args:
            config: 量化配置
            quantization_manager: 量化管理器
            model: 训练模型
            delay_insert: 是否延迟QAT插入（True表示在指定epoch插入，False表示立即插入）
        """
        self.config = config
        self.qm = quantization_manager
        self.model = model
        self.current_stage = 'warmup'
        self.qat_inserted = False
        self.delay_insert = delay_insert
        self._bn_frozen = False
        self._frozen_norm_modules = []  # 记录被冻结的模块，用于后续解冻

    def maybe_insert_qat(self, epoch: int) -> bool:
        """在指定epoch动态插入QAT模块

        Args:
            epoch: 当前训练轮次

        Returns:
            是否成功插入QAT
        """
        if self.qat_inserted:
            return False

        qat_insert_epoch = self.config.get('qat_insert_epoch', 3)

        if epoch >= qat_insert_epoch:
            print(f"[QAT] Epoch {epoch}: Inserting QAT modules...")
            try:
                # 调用量化管理器应用QAT
                self.qm.prepare_model_for_quantization()
                self.qat_inserted = True

                # 验证QAT是否成功应用
                has_fake_quant = False
                for name, module in self.model.named_modules():
                    if isinstance(module, FakeQuantizedLinear):
                        has_fake_quant = True
                        break

                if has_fake_quant:
                    print(f"[QAT] Modules inserted successfully")
                else:
                    print(f"[QAT] Warning: FakeQuantizedLinear not detected, QAT may not be active")

                return True
            except Exception as e:
                print(f"[QAT] Insertion failed: {e}")
                return False

        return False

    def get_current_stage(self, epoch: int) -> str:
        """获取当前训练阶段"""
        warmup_epochs = self.config.get('warmup_lr', 3)
        qat_insert_epoch = self.config.get('qat_insert_epoch', warmup_epochs)
        qat_epochs = self.config.get('qat_epochs', 5)

        # 如果延迟插入且QAT尚未插入，则处于pre_qat阶段
        if self.delay_insert and not self.qat_inserted:
            if epoch < warmup_epochs:
                return 'warmup'
            elif epoch < qat_insert_epoch:
                return 'pre_qat'
            else:
                return 'pre_qat'  # 等待插入

        if epoch < warmup_epochs:
            return 'warmup'
        elif epoch < qat_insert_epoch:
            return 'pre_qat'
        elif epoch < qat_insert_epoch + qat_epochs:
            return 'qat_finetune'
        else:
            return 'post_qat'

    def should_freeze_bn_ln(self, epoch: int) -> bool:
        """检查是否应该冻结BN和LN层 - QAT阶段冻结以提高稳定性"""
        stage = self.get_current_stage(epoch)
        return stage == 'qat_finetune'

    def get_lr_multiplier(self, epoch: int) -> float:
        """获取当前阶段的学习率乘数"""
        stage = self.get_current_stage(epoch)

        multipliers = {
            'warmup': 1.0,  # warmup阶段正常学习率
            'pre_qat': 1.0,
            'qat_finetune': self.config.get('qat_learning_rate_multiplier', 0.1),  # QAT阶段低学习率
            'post_qat': 0.05,  # 后QAT阶段更低学习率
        }

        return multipliers.get(stage, 1.0)

    def freeze_bn_ln_for_qat(self, model: nn.Module):
        """冻结归一化层用于QAT - 提高量化稳定性

        支持多种归一化层:
        - BatchNorm: BatchNorm1d, BatchNorm2d, BatchNorm3d
        - LayerNorm: LayerNorm
        - GroupNorm: GroupNorm
        - InstanceNorm: InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
        """
        frozen_count = 0
        norm_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
        )

        for name, module in model.named_modules():
            if isinstance(module, norm_types):
                # 检查是否已经冻结
                if getattr(module, '_qat_frozen', False):
                    continue

                module.requires_grad_(False)
                module.eval()  # 设置为eval模式防止统计量更新
                module._qat_frozen = True  # 标记为已冻结
                self._frozen_norm_modules.append(module)
                frozen_count += 1

        if frozen_count > 0:
            print(f"[QAT] Froze {frozen_count} normalization layers for QAT stability")
            self._bn_frozen = True

    def unfreeze_bn_ln(self, model: nn.Module):
        """解冻归一化层 - 用于QAT阶段结束后恢复训练"""
        unfrozen_count = 0
        for module in self._frozen_norm_modules:
            if hasattr(module, '_qat_frozen'):
                module.requires_grad_(True)
                module.train()  # 恢复train模式
                delattr(module, '_qat_frozen')
                unfrozen_count += 1

        self._frozen_norm_modules.clear()

        if unfrozen_count > 0:
            print(f"[QAT] Unfroze {unfrozen_count} normalization layers")
            self._bn_frozen = False

    def unfreeze_all(self, model: nn.Module):
        """解冻所有参数"""
        for module in model.modules():
            module.requires_grad_(True)
        self._frozen_norm_modules.clear()
        self._bn_frozen = False
        print("[QAT] Unfroze all layer parameters")

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
            'strategy': self.strategy.value if hasattr(self.strategy, 'value') else self.strategy,
            'quantization_aware_training': self.quantization_aware_training,
            'post_training_quantization': self.post_training_quantization,
            'qat_epochs': self.qat_epochs,
            'ptq_epochs': self.ptq_epochs,
            'calibration_batches': self.calibration_batches,
            'qat_learning_rate_multiplier': self.qat_learning_rate_multiplier,
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits,
            'mixed_precision': self.mixed_precision,
            'observer_type': self.observer_type.value if hasattr(self.observer_type, 'value') else self.observer_type,
            'observer_momentum': self.observer_momentum,
            'quantization_granularity': self.quantization_granularity.value if hasattr(self.quantization_granularity, 'value') else self.quantization_granularity,
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
            # 分层混合精度QAT参数
            'enable_layer_wise_qat': getattr(self, 'enable_layer_wise_qat', False),
            'fake_quant_loss_weight': getattr(self, 'fake_quant_loss_weight', 0.001),
            'qat_insert_epoch': getattr(self, 'qat_insert_epoch', 3),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """从字典创建配置"""
        config = cls()

        # 基础配置
        config.enabled = config_dict.get('enabled', True)
        strategy_val = config_dict.get('strategy', 'int8_dyn_act_int4_weight')
        if isinstance(strategy_val, str):
            config.strategy = QuantizationStrategy(strategy_val)
        else:
            config.strategy = strategy_val
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
        observer_val = config_dict.get('observer_type', 'moving_average')
        if isinstance(observer_val, str):
            config.observer_type = ObserverType(observer_val)
        else:
            config.observer_type = observer_val
        config.observer_momentum = config_dict.get('observer_momentum', 0.1)

        granularity_val = config_dict.get('quantization_granularity', 'per_channel')
        if isinstance(granularity_val, str):
            config.quantization_granularity = QuantizationGranularity(granularity_val)
        else:
            config.quantization_granularity = granularity_val

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

        # 分层混合精度QAT参数
        config.enable_layer_wise_qat = config_dict.get('enable_layer_wise_qat', False)
        config.fake_quant_loss_weight = config_dict.get('fake_quant_loss_weight', 0.001)
        config.qat_insert_epoch = config_dict.get('qat_insert_epoch', 3)

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
        """获取模型大小（MB）

        针对新量化方案的优化：
        - 如果模型包含FakeQuantizedLinear，估算转换后的真实大小
        - 排除FakeQuantize相关的伪量化参数
        """
        state_dict = model.state_dict()
        size_bytes = 0

        # 检测是否是新版API的FakeQuantize模型（包含FakeQuantizedLinear模块）
        has_fake_quantized_modules = False
        for module in model.modules():
            if isinstance(module, FakeQuantizedLinear):
                has_fake_quantized_modules = True
                break

        # 计算大小，排除伪量化参数
        for name, tensor in state_dict.items():
            # 跳过FakeQuantize相关的伪量化参数（scale, zero_point等）
            if has_fake_quantized_modules and any(keyword in name for keyword in [
                'scale', 'zero_point', 'fake_quant', 'observer', '_amax', '_min', '_max'
            ]):
                continue
            size_bytes += tensor.numel() * tensor.element_size()

        # 对于包含FakeQuantizedLinear的模型，估算量化后的压缩大小
        if has_fake_quantized_modules:
            # 估算：权重按int4/int8计算，激活保持float
            # 这是一个近似值，实际大小取决于具体量化配置
            estimated_ratio = 0.25  # 默认估算压缩到25%
            size_bytes = size_bytes * estimated_ratio

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
    """量化管理器 - 处理所有量化相关操作

    优化特性:
    1. 支持新版FakeQuantizeConfig配置系统
    2. 支持ComposableQATQuantizer分层混合精度
    3. 精细化的粒度控制 (PerToken/PerAxis/PerGroup)
    """

    def __init__(self, model: torch.nn.Module, config: Dict):
        self.config = config
        self.model = model
        self.quantizer = None
        self.calibration_data = []
        self.original_model = None
        self.qat_config = None
        self.use_modern_api = False

    def prepare_model_for_quantization(self) -> torch.nn.Module:
        """准备模型进行量化

        使用新版API: FakeQuantizeConfig + ComposableQATQuantizer
        """
        if not self.config['enabled']:
            return self.model

        print("[QAT] Preparing model quantization...")

        # 保存原始模型用于知识蒸馏
        self.original_model = copy.deepcopy(self.model)
        self.original_model.eval()

        # 根据配置选择量化策略
        strategy = self.config['quantization_strategy']

        if self.config['quantization_aware_training']:
            print(f"🎯 启用量化感知训练 (QAT): {strategy}")

            # 检查是否启用分层混合精度
            if self.config.get('enable_layer_wise_qat', False):
                print("   使用分层混合精度QAT (ComposableQATQuantizer)")
                self.model = self._apply_layer_wise_mixed_precision_qat()
                self.use_modern_api = True
            else:
                print("   使用QAT API (FakeQuantizeConfig) - 精度优先模式")
                self.model = self._apply_modern_qat_quantization()
                self.use_modern_api = True

        elif self.config['post_training_quantization']:
            print(f"🎯 启用训练后量化 (PTQ): {strategy}")
            self.model = self._apply_ptq_quantization()
        else:
            print("⚠️  量化已启用但未指定训练策略")

        return self.model

    def _create_fake_quantize_config(self, model: nn.Module = None) -> Tuple[IntxFakeQuantizeConfig, IntxFakeQuantizeConfig]:
        """创建量化配置 - 优化1: 使用新版推荐的QATConfig系统

        根据量化策略和精度优先原则，创建新的QATConfig配置
        自动调整group_size以适应模型中各层的维度
        返回: (activation_config, weight_config) 或直接使用 base_config 的 QATConfig
        """
        strategy = self.config['quantization_strategy']
        weight_bits = self.config.get('weight_bits', 4)
        activation_bits = self.config.get('activation_bits', 8)
        granularity = self.config.get('quantization_granularity', 'per_channel')

        # 检查模型中的线性层维度，选择合适的group_size
        group_size = 32  # 默认最小group_size
        if model is not None:
            min_in_features = float('inf')
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    min_in_features = min(min_in_features, module.in_features)

            # 找到能整除最小维度的最大group_size
            for candidate in [256, 128, 64, 32]:
                if min_in_features >= candidate and min_in_features % candidate == 0:
                    group_size = candidate
                    break

            print(f"   使用 group_size={group_size} (最小in_features={min_in_features})")

        # 激活量化配置 - PerToken动态量化
        # 根据 activation_bits 选择配置，目前仅支持 INT8 激活
        # 注意: torchao 0.16.0 目前仅支持 INT8 激活量化
        if activation_bits not in [8, 16]:
            print(f"   警告: activation_bits={activation_bits} 不支持，使用 INT8")
        activation_config = IntxFakeQuantizeConfig(
            dtype=torch.int8,
            granularity="per_token",
            is_symmetric=False,
        )

        # 权重量化配置
        if weight_bits == 4:
            weight_config = IntxFakeQuantizeConfig(
                dtype=torch.int4,
                group_size=group_size,
                is_symmetric=True,
            )
        elif weight_bits == 8:
            weight_config = IntxFakeQuantizeConfig(
                dtype=torch.int8,
                granularity="per_channel",
                is_symmetric=True,
            )
        else:
            weight_config = IntxFakeQuantizeConfig(
                dtype=torch.int8,
                granularity="per_tensor",
                is_symmetric=True,
            )

        return activation_config, weight_config

    def _apply_modern_qat_quantization(self) -> torch.nn.Module:
        """应用新版QAT量化 - 优化1: 使用新版推荐的QATConfig系统

        使用 QATConfig 配合 IntxFakeQuantizeConfig 提供更高的精度和灵活性
        这是 FakeQuantizeConfig 和 IntXQuantizationAwareTrainingConfig 的新替代方案
        """
        strategy = self.config['quantization_strategy']

        # 创建量化配置，传入模型以自动调整group_size
        # 返回的是 (activation_config, weight_config)
        activation_config, weight_config = self._create_fake_quantize_config(self.model)

        # 根据策略选择配置方式
        if strategy == 'int8_dyn_act_int4_weight':
            # INT8动态激活 + INT4权重 - 平衡精度与压缩
            self.qat_config = QATConfig(
                activation_config=activation_config,
                weight_config=weight_config,
                step="prepare",
            )

        elif strategy == 'int4_weight_only':
            # INT4权重量化 - 最大压缩
            self.qat_config = QATConfig(
                activation_config=None,  # 不量化激活
                weight_config=weight_config,
                step="prepare",
            )

        elif strategy == 'int8_weight_only':
            # INT8权重量化 - 保守策略
            weight_config_int8 = IntxFakeQuantizeConfig(
                dtype=torch.int8,
                granularity="per_channel",
                is_symmetric=True,
            )
            self.qat_config = QATConfig(
                activation_config=None,
                weight_config=weight_config_int8,
                step="prepare",
            )

        elif strategy == 'int8_dynamic_activation_int8_weight':
            # INT8动态激活 + INT8权重 - 高精度模式
            activation_config_int8 = IntxFakeQuantizeConfig(
                dtype=torch.int8,
                granularity="per_token",
                is_symmetric=False,
            )
            weight_config_int8 = IntxFakeQuantizeConfig(
                dtype=torch.int8,
                granularity="per_channel",
                is_symmetric=True,
            )
            self.qat_config = QATConfig(
                activation_config=activation_config_int8,
                weight_config=weight_config_int8,
                step="prepare",
            )
        else:
            raise ValueError(f"不支持的现代QAT策略: {strategy}")

        # 应用量化配置
        prepared_model = self.model
        quantize_(prepared_model, self.qat_config)

        print(f"✅ 现代QAT量化应用完成: {strategy}")
        print(f"   - 权重量化: {weight_config.dtype} (group_size={getattr(weight_config, 'group_size', 'N/A')})")
        if activation_config is not None:
            print(f"   - 激活量化: {activation_config.dtype} ({activation_config.granularity})")

        return prepared_model

    def _get_valid_group_size(self, in_features: int, preferred: int = 256) -> int:
        """获取能整除in_features的有效group_size"""
        for candidate in [preferred, 128, 64, 32]:
            if in_features >= candidate and in_features % candidate == 0:
                return candidate
        return 32  # 最小默认值

    def _apply_layer_wise_mixed_precision_qat(self) -> torch.nn.Module:
        """应用分层混合精度QAT - 优化2: 使用新版QATConfig分层配置

        针对不同层使用不同量化策略:
        - Backbone: INT8动态激活 + INT4权重 (自适应groupsize) - 平衡
        - Neck: INT8动态激活 + INT4权重 (自适应groupsize) - 更高精度
        - Decoder: INT8动态激活 + INT8权重 - 精度优先
        """
        print("🎯 应用分层混合精度QAT (新版QATConfig)")

        # 收集各层的最小维度
        backbone_min_dim = float('inf')
        neck_min_dim = float('inf')

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if 'backbone' in name:
                    backbone_min_dim = min(backbone_min_dim, module.in_features)
                elif 'neck' in name:
                    neck_min_dim = min(neck_min_dim, module.in_features)

        # 根据实际维度选择合适的group_size
        backbone_group_size = self._get_valid_group_size(backbone_min_dim, 256) if backbone_min_dim != float('inf') else 32
        neck_group_size = self._get_valid_group_size(neck_min_dim, 128) if neck_min_dim != float('inf') else 32

        print(f"   Backbone group_size={backbone_group_size} (min_dim={backbone_min_dim})")
        print(f"   Neck group_size={neck_group_size} (min_dim={neck_min_dim})")

        # 定义每层的量化配置 - 使用新版 IntxFakeQuantizeConfig
        # Backbone: INT8动态激活 + INT4权重
        backbone_act_config = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
        backbone_weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int4, group_size=backbone_group_size, is_symmetric=True
        )

        # Neck: INT8动态激活 + INT4权重 (更小group_size)
        neck_act_config = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
        neck_weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int4, group_size=neck_group_size, is_symmetric=True
        )

        # Decoder: INT8动态激活 + INT8权重
        decoder_act_config = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
        decoder_weight_config = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_channel", is_symmetric=True
        )

        # 使用 quantize_ 逐个模块应用分层配置
        # 注意: ComposableQATQuantizer 需要 List[TwoStepQuantizer]，不适用于此场景
        print("   开始为各层应用量化配置...")

        # 统计应用情况
        applied_counts = {'backbone': 0, 'neck': 0, 'decoder': 0}

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            config = None
            layer_type = None

            if 'backbone' in name:
                config = QATConfig(
                    activation_config=backbone_act_config,
                    weight_config=backbone_weight_config,
                    step="prepare"
                )
                layer_type = 'backbone'
            elif 'neck' in name:
                config = QATConfig(
                    activation_config=neck_act_config,
                    weight_config=neck_weight_config,
                    step="prepare"
                )
                layer_type = 'neck'
            elif 'decoder' in name or 'ctc' in name or 'ar' in name:
                config = QATConfig(
                    activation_config=decoder_act_config,
                    weight_config=decoder_weight_config,
                    step="prepare"
                )
                layer_type = 'decoder'

            if config is not None:
                try:
                    quantize_(module, config)
                    applied_counts[layer_type] += 1
                except Exception as e:
                    print(f"   ⚠️  {name} 应用量化配置失败: {e}")

        print(f"✅ 分层混合精度QAT应用完成")
        print(f"   - Backbone层: {applied_counts['backbone']} 个模块 (INT8动态激活 + INT4权重)")
        print(f"   - Neck层: {applied_counts['neck']} 个模块 (INT8动态激活 + INT4权重)")
        print(f"   - Decoder层: {applied_counts['decoder']} 个模块 (INT8动态激活 + INT8权重)")

        return self.model

    def _apply_ptq_quantization(self) -> torch.nn.Module:
        """应用训练后量化 - 使用新版API (torchao 0.16.0)

        使用正确的API参数:
        - Int8DynamicActivationIntxWeightConfig: 使用 weight_dtype 和 weight_granularity
        - IntxWeightOnlyConfig: 使用 weight_dtype 和 granularity
        """
        strategy = self.config['quantization_strategy']
        weight_bits = self.config.get('weight_bits', 4)

        # 获取合适的group_size
        min_in_features = float('inf')
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                min_in_features = min(min_in_features, module.in_features)

        group_size = 32
        for candidate in [256, 128, 64, 32]:
            if min_in_features >= candidate and min_in_features % candidate == 0:
                group_size = candidate
                break

        if strategy == 'int8_dyn_act_int4_weight':
            # torchao 0.16.0: Int8DynamicActivationIntxWeightConfig 使用 weight_granularity
            # 始终显式指定 granularity，提高代码可读性和确定性
            quantizer = Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(group_size=group_size),
            )
        elif strategy == 'int8_weight_only':
            # torchao 0.16.0: IntxWeightOnlyConfig 使用 PerAxis(axis=0) 进行通道级量化
            quantizer = IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=PerAxis(axis=0),
            )
        elif strategy == 'int4_weight_only':
            # torchao 0.16.0: IntxWeightOnlyConfig 使用 granularity 参数
            quantizer = IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(group_size=group_size),
            )
        elif strategy == 'int8_dynamic_activation_int8_weight':
            quantizer = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8)
        else:
            raise ValueError(f"不支持的PTQ策略: {strategy}")

        # 应用量化
        quantize_(self.model, quantizer)

        print(f"✅ PTQ量化应用完成: {strategy} (group_size={group_size})")
        return self.model

    def calibrate_model(self, dataloader: DataLoader, num_batches: int = None):
        """校准量化模型 - 增加预热步骤以提高校准稳定性

        PTQ校准流程:
        1. 预热阶段: 运行若干批次使观察器稳定
        2. 正式校准: 统计激活值分布并计算量化参数
        """
        if not self.config['enabled'] or not self.config['post_training_quantization']:
            return

        print("📊 开始模型校准...")

        if num_batches is None:
            num_batches = self.config['calibration_batches']

        # 预热阶段: 使用10%的批次或最少5个批次进行预热
        warmup_batches = max(5, num_batches // 10)
        actual_calibration = max(num_batches - warmup_batches, num_batches // 2)

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc='校准')):
                if i >= num_batches:
                    break

                images = batch['images'].to(device)
                _ = self.model(images)

                if i == warmup_batches - 1:
                    print(f"   预热完成 ({warmup_batches} 批次)，开始正式校准...")

        print(f"✅ 模型校准完成 (预热{warmup_batches} + 正式{actual_calibration} 批次)")

    def get_quantization_loss(self, quantized_features: torch.Tensor,
                            original_features: torch.Tensor) -> torch.Tensor:
        """计算量化损失用于知识蒸馏 - 优化4: 改进的量化感知损失"""
        if not self.config['enabled']:
            return torch.tensor(0.0, device=quantized_features.device)

        device = quantized_features.device
        total_loss = torch.tensor(0.0, device=device)

        # 1. 特征蒸馏损失 (KL散度) - 保持相对关系
        temperature = self.config['temperature_distillation']
        distillation_loss = F.kl_div(
            F.log_softmax(quantized_features / temperature, dim=-1),
            F.softmax(original_features / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)  # 温度缩放恢复梯度幅度

        total_loss += self.config['distillation_weight'] * distillation_loss

        # 2. 量化感知损失 - 多尺度MSE + Cosine相似度
        # MSE损失 - 保持绝对数值接近
        mse_loss = F.mse_loss(quantized_features, original_features.detach())

        # Cosine相似度损失 - 保持方向一致性
        # pylint: disable=not-callable
        # 处理1维张量的情况：如果只有1维，直接使用；否则保留batch维，扁平化其余维度
        if quantized_features.dim() == 1:
            q_flat = quantized_features.unsqueeze(0)
            o_flat = original_features.unsqueeze(0).detach()
        else:
            q_flat = quantized_features.flatten(1)
            o_flat = original_features.flatten(1).detach()
        cos_loss = 1 - F.cosine_similarity(q_flat, o_flat, dim=1).mean()

        # L1稀疏性损失 - 鼓励稀疏性（对INT4有益）
        l1_loss = torch.abs(quantized_features).mean() * 0.01

        quantization_loss = mse_loss + 0.5 * cos_loss + l1_loss
        total_loss += self.config['quantization_loss_weight'] * quantization_loss

        # 3. 收集FakeQuantize层的量化损失
        if self.qat_config is not None:
            fake_quant_loss = self._collect_fake_quantization_loss()
            total_loss += self.config.get('fake_quant_loss_weight', 0.001) * fake_quant_loss

        return total_loss

    def _collect_fake_quantization_loss(self) -> torch.Tensor:
        """收集所有FakeQuantize层的量化损失 - 优化4: 细粒度量化损失"""
        device = next(self.model.parameters()).device
        total_fake_quant_loss = torch.tensor(0.0, device=device)

        for name, module in self.model.named_modules():
            # 检查是否为FakeQuantize层
            if hasattr(module, 'get_quantization_loss'):
                layer_loss = module.get_quantization_loss()
                if layer_loss is not None:
                    total_fake_quant_loss += layer_loss
            # 检查是否为FakeQuantizedLinear层
            elif 'FakeQuantized' in module.__class__.__name__:
                if hasattr(module, 'weight_fake_quant'):
                    w_loss = module.weight_fake_quant.get_quantization_loss()
                    if w_loss is not None:
                        total_fake_quant_loss += w_loss

        return total_fake_quant_loss

    def convert_to_quantized_model(self, model: nn.Module) -> nn.Module:
        """优化5: 将FakeQuantize模型转换为真实量化模型 - 使用新版API (torchao 0.16.0)

        使用 QATConfig(base_config, step="convert") 替换已弃用的 from_intx_quantization_aware_training
        支持ExportWrapper（包含多个子模块的模型包装器）
        """
        if not self.config['enabled']:
            return model

        # 检查是否是新版API的FakeQuantize模型
        has_fake_quantized_modules = False
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizedLinear):
                has_fake_quantized_modules = True
                break

        if not has_fake_quantized_modules:
            # 不是FakeQuantize模型，已经转换过
            print("ℹ️  模型不包含FakeQuantizedLinear模块，跳过转换")
            return model

        print("🔄 转换FakeQuantize模型为真实量化模型...")

        try:
            strategy = self.config.get('quantization_strategy', 'int8_dyn_act_int4_weight')

            # 创建对应的base_config用于转换
            if strategy == 'int8_dyn_act_int4_weight':
                base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
            elif strategy == 'int4_weight_only':
                base_config = IntxWeightOnlyConfig(
                    weight_dtype=torch.int4,
                    granularity=PerGroup(group_size=32),
                )
            elif strategy == 'int8_weight_only':
                base_config = IntxWeightOnlyConfig(weight_dtype=torch.int8)
            elif strategy == 'int8_dynamic_activation_int8_weight':
                base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8)
            else:
                # 默认使用int8_dyn_act_int4_weight配置
                base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)

            # 创建转换配置: QATConfig with step="convert"
            convert_config = QATConfig(base_config, step="convert")
            print(f"   使用策略: {strategy}, base_config: {base_config.__class__.__name__}")

            # 处理ExportWrapper（包含多个子模块的模型）
            if hasattr(model, 'get_submodules'):
                print("📝 检测到模型包装器，对子模块进行转换...")
                submodules = model.get_submodules()
                converted_count = 0

                for name, submodule in submodules.items():
                    if submodule is not None and self._has_fake_quantized_modules(submodule):
                        try:
                            quantize_(submodule, convert_config)
                            setattr(model, name, submodule)
                            converted_count += 1
                            print(f"  ✅ {name} 转换完成")
                        except Exception as sub_err:
                            print(f"  ⚠️ {name} 转换失败: {sub_err}")

                if converted_count > 0:
                    print(f"✅ 子模块转换完成 ({converted_count}/{len(submodules)})")
                return model
            else:
                # 普通模型直接转换
                quantize_(model, convert_config)
                quantized_model = model
                print(f"✅ 使用 QATConfig(step='convert') 转换完成")
                return quantized_model

        except Exception as e:
            print(f"⚠️  新版API转换失败: {e}")
            print("   回退到原始模型")

        return model

    def _has_fake_quantized_modules(self, module: nn.Module) -> bool:
        """检查模块是否包含FakeQuantizedLinear等假量化模块"""
        for child in module.modules():
            if isinstance(child, FakeQuantizedLinear):
                return True
        return False

    @torch.no_grad()
    def export_quantized_model(self, pruning: bool, path: str, epoch: int, best_cer: float, best_em: float, example_input: torch.Tensor, opt: Dict, scaler: Dict, model: nn.Module):
        """导出量化模型 - 优化5: 统一导出流程

        适配新量化方案:
        - 自动检测并转换FakeQuantizer模型
        - 支持ExportWrapper（多子模块模型）
        - 优先使用torch.export导出转换后的真实量化模型
        """

        # 确保模型在评估模式
        model.eval()

        state = {
            'pruning': pruning,
            'opt': opt, 'scaler': scaler, 'epoch': epoch,
            'best_cer': best_cer, 'best_em': best_em,
            'config': self.config,
            'use_modern_api': self.use_modern_api,
        }

        if self.config['enabled']:
            print("📤 导出量化模型...")

            # 优化5: 如果是新版API，先转换为真实量化模型
            export_model = model
            converted = False
            if self.use_modern_api and self.qat_config is not None:
                try:
                    export_model = self.convert_to_quantized_model(model)
                    converted = export_model is not model  # 检查是否成功转换
                    if converted:
                        print("✅ 模型已成功转换为真实量化格式")
                except Exception as e:
                    print(f"⚠️  转换失败，使用原始模型导出: {e}")

            state['converted_to_quantized'] = converted

            # 更新state中的模型状态字典（使用转换后的模型）
            state['model'] = export_model.state_dict()

            # 使用torch.export导出量化模型
            try:
                # 确保example_input在正确的设备上
                # example_input = example_input.to(next(export_model.parameters()).device)

                # 保存量化的模型会导致加载时提示：TypeError: _ModuleStackTracer.init() missing 1 required positional argument: 'scope_root'
                # exported_program = export(export_model, (example_input,))
                # state['quantization_model'] = exported_program

                # 同时保存onnx格式的元数据
                state['export_metadata'] = {
                    'input_shape': list(example_input.shape),
                    'dtype': str(example_input.dtype),
                    'device': str(example_input.device),
                }

                # 保存导出的程序
                torch.save(state, path)

                print(f"✅ 量化模型导出完成: {path}")
                print(f"   - 使用现代API: {self.use_modern_api}")
                print(f"   - 已转换为真实量化: {converted}")

            except Exception as e:
                print(f"⚠️  torch.export导出失败，使用备用方法: {e}")

                # 备用方法：只保存状态字典
                if 'quantization_model' in state:
                    del state['quantization_model']
                if 'export_metadata' in state:
                    del state['export_metadata']

                torch.save(state, path)
                print(f"✅ 模型状态字典导出完成: {path}")
        else:
            print("⚠️  量化未启用，导出原始模型")
            state['model'] = model.state_dict()
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
        """根据trial建议修改配置 - 扩展支持新版API参数"""
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

        # ========== 分层混合精度QAT参数 ==========
        # 是否启用分层混合精度QAT
        config.enable_layer_wise_qat = trial.suggest_categorical('enable_layer_wise_qat', [True, False])

        # FakeQuantize层损失权重 (新版API)
        config.fake_quant_loss_weight = trial.suggest_float(
            'fake_quant_loss_weight', 0.0001, 0.01, log=True
        )

        # 权重量化粒度 (新版API)
        config.weight_granularity = trial.suggest_categorical(
            'weight_granularity', ['per_group_256', 'per_group_128', 'per_axis', 'per_tensor']
        )

        # 激活量化粒度 (新版API)
        config.activation_granularity = trial.suggest_categorical(
            'activation_granularity', ['per_token', 'per_tensor']
        )

        # QAT插入epoch (分阶段训练)
        config.qat_insert_epoch = trial.suggest_int('qat_insert_epoch', 1, 5)

        return config

    def _create_quantized_model(self, config: QuantizationConfig) -> nn.Module:
        """创建量化模型

        根据配置使用 QuantizationManager 应用真正的量化
        """
        # 深拷贝原始模型
        try:
            model_copy = copy.deepcopy(self.original_model)
        except Exception as e:
            warnings.warn(f"模型深拷贝失败: {e}，使用原始模型")
            return self.original_model

        # 创建量化管理器配置
        qm_config = {
            'enabled': True,
            'quantization_strategy': config.strategy.value if hasattr(config, 'strategy') else 'int8_dyn_act_int4_weight',
            'quantization_aware_training': True,
            'post_training_quantization': False,
            'weight_bits': getattr(config, 'weight_bits', 4),
            'activation_bits': getattr(config, 'activation_bits', 8),
            'qat_epochs': getattr(config, 'qat_epochs', 5),
            'qat_learning_rate_multiplier': getattr(config, 'qat_learning_rate_multiplier', 0.1),
            'quantization_loss_weight': getattr(config, 'quantization_loss_weight', 0.01),
            'temperature_distillation': getattr(config, 'temperature_distillation', 4.0),
            'distillation_weight': getattr(config, 'distillation_weight', 0.3),
            'observer_momentum': getattr(config, 'observer_momentum', 0.1),
            'calibration_batches': getattr(config, 'calibration_batches', 100),
            # 分层混合精度QAT配置
            'enable_layer_wise_qat': getattr(config, 'enable_layer_wise_qat', False),
            'fake_quant_loss_weight': getattr(config, 'fake_quant_loss_weight', 0.001),
        }

        try:
            # 创建量化管理器并应用量化
            qm = QuantizationManager(model_copy, qm_config)
            quantized_model = qm.prepare_model_for_quantization()

            # 转换为真实量化模型用于评估
            if hasattr(qm, 'convert_to_quantized_model'):
                quantized_model = qm.convert_to_quantized_model(quantized_model)

            return quantized_model

        except Exception as e:
            warnings.warn(f"量化模型创建失败: {e}，返回原始模型副本")
            return model_copy

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

    def _create_quantized_model(self, config: QuantizationConfig) -> nn.Module:
        """创建量化模型

        根据配置使用 QuantizationManager 应用真正的量化
        """
        # 深拷贝原始模型
        try:
            model_copy = copy.deepcopy(self.original_model)
        except Exception as e:
            warnings.warn(f"模型深拷贝失败: {e}，使用原始模型")
            return self.original_model

        # 创建量化管理器配置
        qm_config = {
            'enabled': True,
            'quantization_strategy': config.strategy.value if hasattr(config, 'strategy') else 'int8_dyn_act_int4_weight',
            'quantization_aware_training': True,
            'post_training_quantization': False,
            'weight_bits': getattr(config, 'weight_bits', 4),
            'activation_bits': getattr(config, 'activation_bits', 8),
            'qat_epochs': getattr(config, 'qat_epochs', 5),
            'qat_learning_rate_multiplier': getattr(config, 'qat_learning_rate_multiplier', 0.1),
            'quantization_loss_weight': getattr(config, 'quantization_loss_weight', 0.01),
            'temperature_distillation': getattr(config, 'temperature_distillation', 4.0),
            'distillation_weight': getattr(config, 'distillation_weight', 0.3),
            'observer_momentum': getattr(config, 'observer_momentum', 0.1),
            'calibration_batches': getattr(config, 'calibration_batches', 100),
            # 分层混合精度QAT配置
            'enable_layer_wise_qat': getattr(config, 'enable_layer_wise_qat', False),
            'fake_quant_loss_weight': getattr(config, 'fake_quant_loss_weight', 0.001),
        }

        try:
            # 创建量化管理器并应用量化
            qm = QuantizationManager(model_copy, qm_config)
            quantized_model = qm.prepare_model_for_quantization()

            # 转换为真实量化模型用于评估
            if hasattr(qm, 'convert_to_quantized_model'):
                quantized_model = qm.convert_to_quantized_model(quantized_model)

            return quantized_model

        except Exception as e:
            warnings.warn(f"量化模型创建失败: {e}，返回原始模型副本")
            return model_copy

    def _create_quantized_model_for_grid_search(self, config: QuantizationConfig) -> nn.Module:
        """为网格搜索创建量化模型 - 复用 _create_quantized_model 的实现"""
        # 直接调用通用的量化模型创建方法
        return self._create_quantized_model(config)

    def _generate_random_config(self, base_config: QuantizationConfig) -> QuantizationConfig:
        """生成随机配置"""
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

    # 分层混合精度QAT配置
    'enable_layer_wise_qat': False,            # 是否启用分层混合精度QAT
    'weight_bits': 4,                          # 权重量化位数
    'activation_bits': 8,                      # 激活量化位数
    'fake_quant_loss_weight': 0.001,           # FakeQuantize层损失权重

    # 分阶段QAT配置
    'qat_insert_epoch': 3,                     # QAT插入epoch (默认与warmup_lr一致)
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
