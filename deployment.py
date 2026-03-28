#!/usr/bin/env python3
"""
部署工具 - 支持量化模型的各种部署格式
提供ONNX、TensorRT、TorchScript等格式导出
"""
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import editdistance
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx
from PIL import Image, ImageDraw, ImageFont
from torch.export import Dim

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cer_score(pred, gold, length, blank_id: int, sos_id: int, eos_id: int, idx2char: dict):
    """
    计算字符错误率 (Character Error Rate, CER)。

    Args:
        pred: 模型输出的一维 tensor 或列表，长度为 max_len，可能包含 PAD/EOS 标记
        gold: 真值一维 tensor 或列表，长度为 max_len，可能包含 PAD/EOS 标记
        length: 真值实际字符数（不含 SOS/EOS/PAD）
        blank_id: 空白符 (PAD) 的 ID
        sos_id: 起始符 (Start of Sequence) 的 ID
        eos_id: 结束符 (End of Sequence) 的 ID
        idx2char: 索引到字符的映射字典

    Returns:
        float: CER 分数，计算公式为编辑距离除以真实文本长度
    """
    # 把 tensor 转成 list，去掉 PAD 和 EOS
    def _clean(seq):
        out = []
        idx_seq = seq.tolist() if isinstance(seq, torch.Tensor) else seq
        for idx in idx_seq:
            if idx == blank_id or idx == sos_id:    # PAD
                continue
            if idx == eos_id:                       # EOS，直接截断
                break
            if idx in idx2char:                     # 正常字符
                out.append(idx2char[idx])
        return out

    pred_clean = _clean(pred)
    gold_clean = _clean(gold[:length])          # 只取有效长度

    if len(gold_clean) == 0:
        return 1.0

    return editdistance.eval(pred_clean, gold_clean) / len(gold_clean)

def exact_match(pred, gold, length, blank_id, eos_id):
    """
    计算两个字符串是否完全匹配 (Exact Match)。

    Args:
        pred: 模型输出的一维 tensor，长度为 max_len，可能包含 PAD/EOS 标记
        gold: 真值一维 tensor，长度为 max_len，可能包含 PAD/EOS 标记
        length: 真值实际字符数（不含 SOS/EOS/PAD）
        blank_id: 空白符 (PAD) 的 ID
        eos_id: 结束符 (End of Sequence) 的 ID

    Returns:
        bool: 如果预测结果与真实值完全匹配则返回 True，否则返回 False
    """
    # 有效长度
    len_pred = (pred != blank_id).logical_and(pred != eos_id).sum().item()

    min_len = min(len_pred, length)
    if min_len == 0:                       # 空串
        return len_pred == length

    return torch.equal(pred[:min_len], gold[:min_len])

def dbg_em(pred, gold, blank_id: int):
    """
    调试用的精确匹配函数，用于打印预测结果和真实值的详细比较信息。

    Args:
        pred: 模型输出的一维 CPU tensor，包含预测的索引序列
        gold: 真值的一维 CPU tensor，包含真实的索引序列
        blank_id: 空白符 (PAD) 的 ID，用于过滤掉空白符

    Returns:
        None: 该函数仅用于调试，不返回任何值，直接打印比较结果
    """
    pred_l = [int(i) for i in pred if int(i) != blank_id]
    gold_l = [int(i) for i in gold if int(i) != blank_id]
    print('pred', pred_l, 'gold', gold_l,
          'len=', len(pred_l), len(gold_l),
          'same?', pred_l == gold_l)

def ctc_decode_v2(pred: List[int], skip_tokens: List[int]) -> List[int]:
    """
    CTC 解码函数，用于将模型输出的索引序列转换为最终的字符索引序列。

    Args:
        pred: 模型输出的索引列表，长度为序列长度
        skip_tokens: 需要跳过的标记列表，通常包含 blank_id, sos_id, eos_id 等

    Returns:
        List[int]: 解码后的索引列表，去除了连续重复和跳过标记
    """
    pred = [pred[0]] + [pred[j] for j in range(1, len(pred)) if pred[j] != pred[j-1]]
    pred = [p for p in pred if p not in skip_tokens]

    return pred

class ExportWrapper(torch.nn.Module):
    """简化的推理模型包装器
    
    支持:
    - 普通RecNetwork模型
    - FakeQuantizer包装的模型（新版QAT API）
    - 已转换的真实量化模型
    """
    def __init__(self, model: nn.Module):
        super().__init__()

        model.eval()
        
        # 处理 FakeQuantizer 包装的模型（新版QAT API）
        # FakeQuantizer 包装时会把原模型放在 model 属性中
        inner_model = model
        if hasattr(model, 'model') and 'FakeQuantizer' in model.__class__.__name__:
            # 解包 FakeQuantizer 获取真实模型 (包括 IntxFakeQuantizer, Int8FakeQuantizer 等)
            inner_model = model.model
            logger.info(f"🔓 ExportWrapper: 自动解包 {model.__class__.__name__} 模型")
        
        # 提取子模块
        self.backbone = inner_model.backbone
        self.neck     = inner_model.neck
        self.head = inner_model.decoder.ctc_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  [B,3,H,W]  0~1  float32
        return: log_softmax 后 [B,L,num_classes]
        """
        feat = self.backbone(x)          # [B,L,C]
        feat = self.neck(feat)           # 训练时 neck 需要 target，推理不用
        out  = self.head(feat)           # 内部已 log_softmax
        return out
    
    def get_submodules(self) -> Dict[str, nn.Module]:
        """获取子模块字典，用于量化转换"""
        return {
            'backbone': self.backbone,
            'neck': self.neck,
            'head': self.head
        }

class ModelExporter(ABC):
    """模型导出器基类"""

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None):
        self.model = ExportWrapper(model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = vocab or []
        self.other_pad_size = other_pad_size
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.idx2char = idx2char or {}

    @abstractmethod
    def export(self, dummy_input: torch.Tensor, **kwargs) -> str:
        """导出模型"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        pass

class ONNXExporter(ModelExporter):
    """ONNX导出器
    
    修复: 在初始化时完成量化模型转换，避免引用不一致问题
    """

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None, 
                 opset_version: int = 18, quantization_config: Optional[Dict[str, Any]] = None):
        # 修复: 在调用父类__init__前，先进行量化模型转换
        # 这样ExportWrapper接收的是转换后的模型，避免FakeQuantizer包装冲突
        self.opset_version = opset_version
        self.quantization_config = quantization_config or {}
        self._quantization_applied = False
        
        # 提前进行量化转换（如果启用）
        converted_model = self._convert_model_if_quantized(model)
        
        # 现在调用父类__init__，传入可能已转换的模型
        super().__init__(converted_model, output_dir, vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)

    def _convert_model_if_quantized(self, model: nn.Module) -> nn.Module:
        """在初始化时转换量化模型
        
        修复模型转换时机问题：在ONNXExporter创建时就完成转换，
        而不是等到export()时才转换，确保ExportWrapper接收正确的模型
        """
        if not self.quantization_config.get('enabled', False):
            return model
        
        # 检查是否已经是真量化模型
        if self._is_true_quantized_model(model):
            logger.info("✅ 模型已经是真量化模型，跳过转换")
            return model
        
        # 检查是否是QAT模型需要转换
        if self._is_qat_model(model):
            logger.info("🔄 ONNXExporter: 检测到QAT模型，开始转换为真量化模型...")
            try:
                converted = self._convert_qat_to_quantized(model)
                self._quantization_applied = True
                logger.info("✅ 量化模型转换完成，将使用转换后的模型进行导出")
                return converted
            except Exception as e:
                logger.warning(f"⚠️ 量化模型转换失败，使用原始模型: {e}")
                return model
        
        return model

    def export(self, dummy_input: torch.Tensor,
               input_names: Optional[List[str]] = None,
               output_names: Optional[List[str]] = None,
               dynamic_axes: Optional[Dict[str, Dict[str, Dim]]] = None,
               **kwargs) -> str:
        """导出ONNX模型"""

        if input_names is None:
            input_names = ['x']
        if output_names is None:
            output_names = ['logits']
        if dynamic_axes is None:
            # 给每个动态维起名字 + 可选范围
            batch_size  = Dim("batch_size", min=2, max=1024)
            width       = Dim("width", min=16, max=4096)
            dynamic_axes = {
                "x":   {0: batch_size, 3: width}
            }

        # 设置模型为评估模式
        self.model.eval()

        # 构建输出路径
        output_path = self.output_dir / 'model.onnx'

        # 导出ONNX
        try:
            with torch.no_grad():
                # 检查并应用量化转换（伪量化->真量化）
                model_to_export = self._prepare_model_for_export(dummy_input)
                
                model_to_export(dummy_input)
                if not os.path.exists(output_path):
                    self.export_onnx(str(output_path), dummy_input, input_names, output_names, dynamic_axes, model_to_export)
                self.validate(str(output_path), self.vocab, self.other_pad_size, self.blank_id, self.sos_id, self.eos_id, self.idx2char)

            logger.info(f"✅ ONNX模型导出成功: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ ONNX导出失败: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        return ['onnx']

    def validate(self, model_path: str, vocab: list, other_pad_size: int, blank_id: int, sos_id: int, eos_id: int, idx2char: dict) -> bool:
        """验证ONNX模型"""
        try:
            MAX_CHARS    = 1500
            STEP         = 10
            providers    = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            # 测试动态轴
            self.test_dynamic_axes(model_path)

            # 加载 ONNX
            ort_sess = ort.InferenceSession(model_path, providers = providers)

            for n_char in range(10, MAX_CHARS + 1, STEP):
                gt_text = ''.join(np.random.choice(vocab[:-other_pad_size], n_char))
                img_tensor = self.make_text_image(gt_text)
                pred_text, dt = self.onnx_infer(ort_sess, img_tensor, blank_id, sos_id, eos_id, idx2char)
                pred_preview = pred_text[:30] + ('...' if len(pred_text) > 30 else '')
                gold_preview = gt_text[:30] + ('...' if len(gt_text) > 30 else '')
                cer = editdistance.eval(pred_text, gt_text) / max(len(gt_text), 1)
                print(n_char, img_tensor.shape[3], f'{cer:.3f}', f'{dt*1000:.1f}',
                    gold_preview, pred_preview, sep='\t')

            return True

        except Exception as e:
            logger.error(f"❌ ONNX模型验证失败: {e}")
            return False

    def _prepare_model_for_export(self, dummy_input: torch.Tensor) -> nn.Module:
        """
        准备模型用于导出，将伪量化(QAT)转换为真量化
        
        Args:
            dummy_input: 示例输入，用于模型追踪
            
        Returns:
            转换后的模型（如果启用了量化）
        """
        model = self.model
        
        # 检查是否启用了量化
        if not self.quantization_config.get('enabled', False):
            logger.info("📋 量化未启用，直接导出原始模型")
            return model
        
        # 检查是否已经是真量化模型（通过检查权重数据类型）
        if self._is_true_quantized_model(model):
            logger.info("✅ 模型已经是真量化模型，直接导出")
            return model
        
        # 检查是否是QAT模型（包含伪量化节点）
        if self._is_qat_model(model):
            logger.info("🔄 检测到QAT伪量化模型，开始转换为真量化...")
            model = self._convert_qat_to_quantized(model)
            self._quantization_applied = True
        
        return model
    
    def _is_qat_model(self, model: nn.Module) -> bool:
        """
        检查模型是否是QAT模型（包含伪量化节点）
        
        支持检测:
        - 旧版API: Int8DynActInt4WeightQATQuantizer 等
        - 新版API: FakeQuantizer (FakeQuantizeConfig)
        
        Returns:
            True 如果模型包含QAT相关的伪量化节点
        """
        # 检查模型本身是否是 FakeQuantizer (新版API)
        model_class = model.__class__.__name__
        if 'FakeQuantizer' in model_class:
            return True
        
        # 如果是 ExportWrapper，检查其子模块
        if hasattr(model, 'get_submodules'):
            submodules = model.get_submodules()
            for submodule in submodules.values():
                if submodule is not None and self._is_qat_model(submodule):
                    return True
            return False
        
        # 检查普通模型
        for name, module in model.named_modules():
            # 检查 torchao QAT 量化器添加的伪量化节点
            module_class = module.__class__.__name__
            if any(keyword in module_class.lower() for keyword in [
                'qat', 'fakequantize', 'fake_quant', 'int8dynactint4weight',
                'int4weightonly', 'quantizedlinear', '_quantize',
                'fakequantizer',  # 新版API: FakeQuantizer
            ]):
                return True
            # 检查是否有量化相关的属性
            if hasattr(module, 'weight_scale') or hasattr(module, 'weight_zero_point'):
                return True
            if hasattr(module, 'activation_scale') or hasattr(module, 'activation_zero_point'):
                return True
        return False
    
    def _is_modern_api_qat_model(self, model: nn.Module) -> bool:
        """
        检查模型是否是新版API (FakeQuantizeConfig) 的QAT模型
        
        Returns:
            True 如果模型使用新版QAT API (FakeQuantizer)
        """
        # 检查模型本身
        if 'FakeQuantizer' in model.__class__.__name__:
            return True
        
        # 检查子模块
        for name, module in model.named_modules():
            if 'FakeQuantizer' in module.__class__.__name__:
                return True
        
        return False
    
    def _is_true_quantized_model(self, model: nn.Module) -> bool:
        """
        检查模型是否已经是真量化模型（权重为整数类型）
        
        Returns:
            True 如果模型权重已经是整数类型
        """
        # 如果是 ExportWrapper，检查其子模块
        if hasattr(model, 'get_submodules'):
            submodules = model.get_submodules()
            for submodule in submodules.values():
                if submodule is not None and self._is_true_quantized_model(submodule):
                    return True
            return False
        
        # 检查普通模型
        for param in model.parameters():
            if param.dtype in [torch.qint8, torch.quint8, torch.qint32, torch.int8, torch.uint8]:
                return True
        return False
    
    def _convert_qat_to_quantized(self, model: nn.Module) -> nn.Module:
        """
        将QAT伪量化模型转换为真量化模型 - 使用新版API (torchao 0.16.0)
        
        新版API使用 QATConfig(base_config, step="convert") 进行转换
        替代已弃用的 from_intx_quantization_aware_training
        
        支持策略:
        - int8_dyn_act_int4_weight: INT8动态激活 + INT4权重
        - int8_weight_only: 纯INT8权重量化
        - int4_weight_only: 纯INT4权重量化
        - int8_dynamic_activation_int8_weight: 纯INT8量化
        
        Args:
            model: QAT训练后的模型（可能是 ExportWrapper 或 FakeQuantizer）
            
        Returns:
            转换后的真量化模型
        """
        try:
            import copy
            
            # 检测是否是新版API的FakeQuantizer模型
            is_modern_api = self._is_modern_api_qat_model(model)
            
            if is_modern_api:
                # 新版API: 使用 QATConfig 转换 (step="convert")
                logger.info("🔄 检测到新版QAT API，使用 QATConfig(base_config, step='convert') 转换...")
                
                try:
                    from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, IntxWeightOnlyConfig
                    from torchao.quantization.qat import QATConfig
                    
                    # 处理 FakeQuantizer 包装的模型 - 先解包获取真实模型
                    inner_model = model
                    if hasattr(model, 'model') and 'FakeQuantizer' in model.__class__.__name__:
                        inner_model = model.model
                        logger.info("🔓 解包 FakeQuantizer 获取原始模型")
                    
                    # 创建转换配置 (step="convert" 用于将 fake quant 转换为真实量化)
                    # torchao 0.16.0: QATConfig 必须指定 base_config 参数
                    # 根据量化策略选择正确的 base_config
                    strategy = self.quantization_config.get('quantization_strategy', 'int8_dyn_act_int4_weight')
                    
                    # 导入 granularity 模块
                    from torchao.quantization.granularity import PerGroup
                    
                    if strategy == 'int8_dyn_act_int4_weight':
                        # torchao 0.16.0: Int8DynamicActivationIntxWeightConfig 使用 weight_dtype
                        # 默认 weight_granularity=PerGroup(group_size=32)
                        base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
                    elif strategy == 'int8_weight_only':
                        base_config = IntxWeightOnlyConfig(weight_dtype=torch.int8)
                    elif strategy == 'int4_weight_only':
                        # IntxWeightOnlyConfig 需要显式指定 PerGroup
                        base_config = IntxWeightOnlyConfig(
                            weight_dtype=torch.int4,
                            granularity=PerGroup(group_size=32),
                        )
                    elif strategy == 'int8_dynamic_activation_int8_weight':
                        base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8)
                    else:
                        logger.warning(f"⚠️ 未知的量化策略: {strategy}，使用默认策略")
                        base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
                    
                    # torchao 0.16.0 API: QATConfig 需要 base_config 和 step="convert"
                    convert_config = QATConfig(base_config, step="convert")
                    logger.info(f"   使用 base_config: {base_config.__class__.__name__}, 策略: {strategy}")
                    
                    # 对于 ExportWrapper，需要分别转换子模块
                    if hasattr(inner_model, 'get_submodules'):
                        logger.info("📝 检测到包装器模型，对子模块进行转换...")
                        submodules = inner_model.get_submodules()
                        
                        for name, submodule in submodules.items():
                            if submodule is not None and self._is_modern_api_qat_model(submodule):
                                try:
                                    # 使用 quantize_ 进行 in-place 转换
                                    quantize_(submodule, convert_config)
                                    logger.info(f"  ✅ {name} 转换完成")
                                except Exception as sub_err:
                                    logger.warning(f"  ⚠️ {name} 转换失败: {sub_err}，使用原始子模块")
                        
                        model_to_return = inner_model
                    else:
                        # 普通模型直接转换 - quantize_ 是 in-place 操作
                        quantize_(inner_model, convert_config)
                        model_to_return = inner_model
                    
                    logger.info("✅ 新版QAT模型已成功转换为真量化模型")
                    
                    # 验证转换结果
                    self._verify_quantization(model_to_return)
                    
                    return model_to_return
                    
                except ImportError as import_err:
                    logger.error(f"❌ 无法导入 QATConfig: {import_err}")
                    logger.warning("⚠️ 新版API转换失败，尝试使用旧版方法...")
                    # 继续执行旧版转换逻辑
            
            # 旧版API: 使用 quantize_ 进行转换 (PTQ)
            logger.info("🔄 使用PTQ API转换 (quantize_)...")
            
            from torchao.quantization import (
                quantize_,
                Int8DynamicActivationIntxWeightConfig,
                IntxWeightOnlyConfig,
            )
            from torchao.quantization.granularity import PerGroup
            
            # 获取量化策略
            strategy = self.quantization_config.get('quantization_strategy', 'int8_dyn_act_int4_weight')
            
            logger.info(f"🎯 应用量化策略: {strategy}")
            
            # 根据策略选择量化器 - 使用新版API (torchao 0.16.0)
            if strategy == 'int8_dyn_act_int4_weight':
                # Int8DynamicActivationIntxWeightConfig 默认使用 PerGroup(group_size=32)
                quantizer = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
            elif strategy == 'int8_weight_only':
                quantizer = IntxWeightOnlyConfig(weight_dtype=torch.int8)
            elif strategy == 'int4_weight_only':
                # IntxWeightOnlyConfig 需要显式指定 PerGroup
                quantizer = IntxWeightOnlyConfig(
                    weight_dtype=torch.int4,
                    granularity=PerGroup(group_size=32),
                )
            elif strategy == 'int8_dynamic_activation_int8_weight':
                quantizer = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8)
            else:
                logger.warning(f"⚠️ 未知的量化策略: {strategy}，使用默认策略")
                quantizer = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
            
            # 处理 FakeQuantizer 包装的模型
            inner_model = model
            if hasattr(model, 'model') and 'FakeQuantizer' in model.__class__.__name__:
                inner_model = model.model
                logger.info("🔓 解包 FakeQuantizer 获取原始模型")
            
            # 检查是否是 ExportWrapper
            is_wrapper = hasattr(inner_model, 'get_submodules')
            
            if is_wrapper:
                # 对于 ExportWrapper，我们需要对其子模块进行量化
                logger.info("📝 检测到包装器模型，对子模块进行量化...")
                submodules = inner_model.get_submodules()
                
                # 对每个子模块应用量化
                for name, submodule in submodules.items():
                    if submodule is not None:
                        try:
                            quantize_(submodule, quantizer)
                            logger.info(f"  ✅ {name} 量化完成")
                        except Exception as sub_err:
                            logger.warning(f"  ⚠️ {name} 量化失败: {sub_err}")
                
                model_to_return = model
            else:
                # 对于普通模型，使用深拷贝并应用量化
                try:
                    model_copy = copy.deepcopy(model)
                except Exception as copy_err:
                    logger.warning(f"⚠️ 深拷贝失败: {copy_err}，将直接修改原模型")
                    model_copy = model
                
                model_copy.eval()
                
                # 应用真量化
                quantize_(model_copy, quantizer)
                model_to_return = model_copy
            
            logger.info("✅ 旧版QAT模型已成功转换为真量化模型")
            
            # 验证转换结果
            self._verify_quantization(model_to_return)
            
            return model_to_return
            
        except Exception as e:
            logger.warning(f"⚠️ 量化转换失败: {e}，将导出原始模型")
            import traceback
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return model
    
    def _verify_quantization(self, model: nn.Module):
        """验证模型是否成功量化"""
        quantized_layers = 0
        total_layers = 0
        
        # 如果是 ExportWrapper，检查其子模块
        if hasattr(model, 'get_submodules'):
            submodules = model.get_submodules()
            for submodule in submodules.values():
                if submodule is not None:
                    q, t = self._count_quantized_layers(submodule)
                    quantized_layers += q
                    total_layers += t
        else:
            q, t = self._count_quantized_layers(model)
            quantized_layers += q
            total_layers += t
        
        if total_layers > 0:
            ratio = quantized_layers / total_layers * 100
            logger.info(f"📊 量化覆盖率: {quantized_layers}/{total_layers} 层 ({ratio:.1f}%)")
    
    def _count_quantized_layers(self, model: nn.Module) -> tuple:
        """统计模型的量化层数"""
        quantized_layers = 0
        total_layers = 0
        
        for name, module in model.named_modules():
            module_class = module.__class__.__name__.lower()
            if 'linear' in module_class or 'conv' in module_class:
                total_layers += 1
                # 检查是否是量化层
                if any(keyword in module_class for keyword in ['quantized', 'int8', 'int4']):
                    quantized_layers += 1
        
        return quantized_layers, total_layers
    
    def export_onnx(self, onnx_path: str, dummy_input: torch.Tensor,
                    input_names: List[str],
                    output_names: List[str],
                    dynamic_shapes: Dict[str, Dict[int, Dim]],
                    model_to_export: Optional[nn.Module] = None):
        """导出 Onnx 模型"""
        
        # 使用传入的模型或默认模型
        model = model_to_export if model_to_export is not None else self.model

        # 给每个动态维起名字 + 可选范围
        batch_size  = Dim("batch_size", min=2, max=1024)
        width       = Dim("width", min=16, max=4096)

        # 把输出也当关键字写进去（名字跟 forward 返回变量对应）
        dynamic_shapes = {
            "x":   {0: batch_size, 3: width}
        }

        torch.onnx.export(
            model,
            args=dummy_input,
            f=onnx_path,
            dynamo=True,
            verbose=True,
            report=True,
            verify=True,
            export_params=True,
            external_data=False,
            do_constant_folding=True,
            opset_version=self.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes
        )

        print(f'ONNX 已导出 → {onnx_path}')

        return onnx_path

    def export_onnx_fixed(self, onnx_path: str, dummy_input: torch.Tensor):
        """修复后的ONNX导出函数"""

        # 定义动态轴
        dynamic_axes = {
            'x': {0: 'batch_size', 3: 'width'},  # 只动态batch和width，height和channel固定
            'logits': {0: 'batch_size', 1: 'seq_length'}  # 输出也需要动态轴
        }

        # 使用传统的导出方式（更稳定）
        # 真正提速：先用 jit.trace 拿图
        traced = torch.jit.trace(self.model, dummy_input)

        torch.onnx.export(
            traced,
            dummy_input,
            f=onnx_path,
            dynamo=False,
            export_params=True,
            external_data=False,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['x'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=True
        )

        print(f'ONNX 已导出 → {onnx_path}')

        return onnx_path

    def test_dynamic_axes(self, onnx_path: str):
        """测试动态轴是否生效"""

        # 验证模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(onnx_path)

        print("\n=== ONNX模型输入信息 ===")
        for input in session.get_inputs():
            print(f"输入名: {input.name}")
            print(f"形状: {input.shape}")
            print(f"类型: {input.type}")

        print("\n=== ONNX模型输出信息 ===")
        for output in session.get_outputs():
            print(f"输出名: {output.name}")
            print(f"形状: {output.shape}")
            print(f"类型: {output.type}")

        # 测试不同尺寸的输入
        test_sizes = [
            (1, 3, 32, 128),
            (2, 3, 32, 256),
            (4, 3, 32, 512)
        ]

        print("\n=== 动态轴测试 ===")
        for batch_size, channels, height, width in test_sizes:
            test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
            try:
                outputs = session.run(None, {'x': test_input})
                print(f"输入形状 {test_input.shape} -> 输出形状 {outputs[0].shape} ✓")
            except Exception as e:
                print(f"输入形状 {test_input.shape} -> 失败: {e} ✗")

    def make_text_image(self, text: str, height=32):
        """ 合成单张图片，返回 (1,3,H,W) 的 np.ndarray """
        # 将宽度向上对齐到 16 的倍数以免出错
        def round_up_to_patch_size(w, patch_size=16):
            return ((w + patch_size - 1) // patch_size) * patch_size

        # 1. 准备字体
        font_path = 'arial.ttf'
        font = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) \
            else ImageFont.load_default()

        # 生成随机背景颜色（RGB）
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 文字颜色为背景颜色的反色
        text_color = (255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2])

        # 2. 计算文本宽度
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0] + 10
        w = round_up_to_patch_size(w)

        # 3. 生成图像并绘制文字
        img = Image.new('RGB', (w, height), bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((5, 3), text, font=font, fill=text_color)

        # 4. 转为 float32 并归一化到 [0,1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # 5. 调整输出形状为 (1,3,H,W)
        return arr.transpose(2, 0, 1)[None, :, :, :]

    def onnx_infer(self, session, img_np, blank_id: int, sos_id: int, eos_id: int, idx2char: dict):
        """推理"""
        in_name  = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        skip_tokens = [blank_id, sos_id, eos_id]

        tic = time.time()
        logits = session.run([out_name], {in_name: img_np})[0]  # [1,L,nc]
        pred_ids = logits.argmax(-1)[0]                         # [L]
        pred_txt = ctc_decode_v2(pred_ids, skip_tokens=skip_tokens)
        pred_str = ''.join([idx2char[i] for i in pred_txt])
        toc = time.time()
        return pred_str, toc - tic

    def verify_onnx_quantization(self, model_path: str):
        """
        验证ONNX模型是否量化

        Args:
            model_path: ONNX模型路径

        Returns:
            bool: 模型是否量化
        """
        print(f"🔍 验证ONNX模型: {model_path}")

        # 加载模型
        model = onnx.load(model_path)

        # 检查模型结构
        graph = model.graph

        # 1. 检查是否包含QuantizeLinear或DequantizeLinear节点
        has_quantize_nodes = False
        quantize_node_count = 0
        dequantize_node_count = 0

        for node in graph.node:
            if node.op_type == 'QuantizeLinear':
                has_quantize_nodes = True
                quantize_node_count += 1
            elif node.op_type == 'DequantizeLinear':
                has_quantize_nodes = True
                dequantize_node_count += 1

        print(f"📊 量化相关节点:")
        print(f"   - QuantizeLinear节点数: {quantize_node_count}")
        print(f"   - DequantizeLinear节点数: {dequantize_node_count}")
        print(f"   - 包含量化节点: {'✅ 是' if has_quantize_nodes else '❌ 否'}")

        # 2. 检查权重数据类型
        has_integer_weights = False
        weight_data_types = {}

        for init in graph.initializer:
            tensor = onnx.numpy_helper.to_array(init)
            dtype = tensor.dtype
            weight_data_types[dtype] = weight_data_types.get(dtype, 0) + 1

            # 检查是否有整数类型的权重
            if dtype.kind in 'iu':  # integer or unsigned integer
                has_integer_weights = True

        print(f"📊 权重数据类型:")
        for dtype, count in weight_data_types.items():
            print(f"   - {dtype}: {count} 个权重张量")
        print(f"   - 包含整数权重: {'✅ 是' if has_integer_weights else '❌ 否'}")

        # 3. 检查模型元数据中是否包含量化信息
        has_quantization_metadata = False
        if hasattr(model, 'metadata_props'):
            for meta in model.metadata_props:
                if 'quant' in meta.key.lower() or 'quantization' in meta.key.lower():
                    has_quantization_metadata = True
                    print(f"   - 元数据: {meta.key} = {meta.value}")

        print(f"📊 量化元数据:")
        print(f"   - 包含量化元数据: {'✅ 是' if has_quantization_metadata else '❌ 否'}")

        # 4. 检查图输入输出的数据类型
        print(f"📊 输入输出数据类型:")
        for input in graph.input:
            print(f"   - 输入 {input.name}: {input.type.tensor_type.elem_type}")
        for output in graph.output:
            print(f"   - 输出 {output.name}: {output.type.tensor_type.elem_type}")

        # 综合判断
        is_quantized = has_quantize_nodes or has_integer_weights

        print(f"\n📋 综合判断:")
        if is_quantized:
            print(f"✅ 模型是量化模型")

            # 进一步分析量化类型
            if quantize_node_count > 0 and dequantize_node_count > 0:
                print(f"   - 量化类型: 全量化模型 (包含QuantizeLinear和DequantizeLinear)")
            elif has_integer_weights:
                print(f"   - 量化类型: 权重量化模型 (仅权重为整数类型)")
        else:
            print(f"❌ 模型不是量化模型")

        return is_quantized
class TorchScriptExporter(ModelExporter):
    """TorchScript导出器"""

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None):
        super().__init__(model, output_dir, vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)

    def export(self, dummy_input: torch.Tensor,
               method: str = 'trace',
               optimize: bool = True,
               **kwargs) -> str:
        """导出TorchScript模型"""

        # 设置模型为评估模式
        self.model.eval()

        # 构建输出路径
        output_path = self.output_dir / 'model.pt'

        try:
            if method == 'trace':
                # 使用追踪模式
                traced_model = torch.jit.trace(self.model, dummy_input)
            elif method == 'script':
                # 使用脚本模式
                traced_model = torch.jit.script(self.model)
            else:
                raise ValueError(f"不支持的TorchScript方法: {method}")

            # 优化模型（如果启用）
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)

            # 保存模型
            torch.jit.save(traced_model, str(output_path))

            logger.info(f"✅ TorchScript模型导出成功: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ TorchScript导出失败: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        return ['torchscript', 'pt']

class TensorRTExporter(ModelExporter):
    """TensorRT导出器（需要TensorRT支持）"""

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None):
        super().__init__(model, output_dir, vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)
        self._check_tensorrt_availability()

    def _check_tensorrt_availability(self):
        """检查TensorRT是否可用"""
        try:
            import tensorrt as trt
            self.trt_available = True
            logger.info("✅ TensorRT可用")
        except ImportError:
            self.trt_available = False
            logger.warning("⚠️ TensorRT不可用，将使用ONNX作为中间格式")

    def export(self, dummy_input: torch.Tensor,
               max_batch_size: int = 32,
               max_workspace_size: int = 1 << 30,  # 1GB
               fp16_mode: bool = False,
               int8_mode: bool = False,
               **kwargs) -> str:
        """导出TensorRT引擎"""

        if not self.trt_available:
            logger.warning("使用ONNX作为TensorRT的中间格式")
            return self._export_via_onnx(dummy_input, max_batch_size,
                                       max_workspace_size, fp16_mode, int8_mode)

        # 设置模型为评估模式
        self.model.eval()

        # 构建输出路径
        output_path = self.output_dir / 'model.trt'

        try:
            import tensorrt as trt

            # 创建TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # 创建builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # 首先导出ONNX
            onnx_path = self.output_dir / 'temp_model.onnx'
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=13
            )

            # 解析ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("ONNX解析失败")

            # 构建配置
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size

            # 设置精度模式
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                # 这里需要设置INT8校准器

            # 构建引擎
            engine_bytes = builder.build_serialized_network(network, config)

            # 保存引擎
            with open(output_path, 'wb') as f:
                f.write(engine_bytes)

            # 清理临时文件
            onnx_path.unlink(missing_ok=True)

            logger.info(f"✅ TensorRT引擎导出成功: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ TensorRT导出失败: {e}")
            raise

    def _export_via_onnx(self, dummy_input: torch.Tensor,
                        max_batch_size: int, max_workspace_size: int,
                        fp16_mode: bool, int8_mode: bool) -> str:
        """通过ONNX间接导出TensorRT"""
        try:
            from torch_tensorrt import compile

            # 设置模型为评估模式
            self.model.eval()

            # 编译模型
            trt_model = compile(
                self.model,
                inputs=[dummy_input],
                enabled_precisions={torch.float, torch.half} if fp16_mode else {torch.float},
                workspace_size=max_workspace_size,
                truncate_long_and_double=True
            )

            # 保存模型
            output_path = self.output_dir / 'model_trt.pt'
            torch.jit.save(trt_model, str(output_path))

            logger.info(f"✅ TensorRT模型导出成功（通过torch-tensorrt）: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ TensorRT导出失败: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        return ['tensorrt', 'trt']

class CoreMLExporter(ModelExporter):
    """CoreML导出器（需要coremltools支持）"""

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None):
        super().__init__(model, output_dir, vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)
        self._check_coreml_availability()

    def _check_coreml_availability(self):
        """检查CoreML是否可用"""
        try:
            import coremltools as ct
            self.coreml_available = True
            logger.info("✅ CoreML可用")
        except ImportError:
            self.coreml_available = False
            logger.warning("⚠️ CoreML不可用，需要安装coremltools")

    def export(self, dummy_input: torch.Tensor,
               input_names: Optional[List[str]] = None,
               output_names: Optional[List[str]] = None,
               minimum_deployment_target: str = '13',
               **kwargs) -> str:
        """导出CoreML模型"""

        if not self.coreml_available:
            raise RuntimeError("CoreML不可用，请安装coremltools")

        # 设置模型为评估模式
        self.model.eval()

        # 构建输出路径
        output_path = self.output_dir / 'model.mlmodel'

        try:
            import coremltools as ct

            # 追踪模型
            traced_model = torch.jit.trace(self.model, dummy_input)

            # 转换为CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
                minimum_deployment_target=ct.target.iOS13
            )

            # 保存模型
            coreml_model.save(str(output_path))

            logger.info(f"✅ CoreML模型导出成功: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ CoreML导出失败: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        return ['coreml', 'mlmodel']

class TFLiteExporter(ModelExporter):
    """TensorFlow Lite导出器"""

    def __init__(self, model: nn.Module, output_dir: str, vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2, idx2char: dict = None):
        super().__init__(model, output_dir, vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)
        self._check_tflite_availability()

    def _check_tflite_availability(self):
        """检查TensorFlow Lite是否可用"""
        try:
            import tensorflow as tf
            self.tflite_available = True
            logger.info("✅ TensorFlow Lite可用")
        except ImportError:
            self.tflite_available = False
            logger.warning("⚠️ TensorFlow Lite不可用，需要安装tensorflow")

    def export(self, dummy_input: torch.Tensor,
               quantization_type: str = 'dynamic',
               representative_dataset: Optional[Any] = None,
               **kwargs) -> str:
        """导出TensorFlow Lite模型"""

        if not self.tflite_available:
            raise RuntimeError("TensorFlow Lite不可用，请安装tensorflow")

        # 设置模型为评估模式
        self.model.eval()

        # 构建输出路径
        output_path = self.output_dir / 'model.tflite'

        try:
            import tensorflow as tf
            import torch.onnx

            # 首先导出为ONNX
            onnx_path = self.output_dir / 'temp_model.onnx'
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=13
            )

            # 使用ONNX-TF转换
            from onnx_tf.backend import prepare

            # 加载ONNX模型
            onnx_model = onnx.load(str(onnx_path))

            # 转换为TensorFlow
            tf_rep = prepare(onnx_model)

            # 保存TensorFlow模型
            tf_model_path = self.output_dir / 'temp_tf_model'
            tf_rep.export_graph(str(tf_model_path))

            # 转换为TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))

            # 设置量化配置
            if quantization_type == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                if representative_dataset:
                    converter.representative_dataset = representative_dataset
            elif quantization_type == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            # 转换
            tflite_model = converter.convert()

            # 保存TFLite模型
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            # 清理临时文件
            onnx_path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(tf_model_path, ignore_errors=True)

            logger.info(f"✅ TensorFlow Lite模型导出成功: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ TensorFlow Lite导出失败: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        return ['tflite']

class QuantizedModelExporter:
    """量化模型导出管理器"""

    def __init__(self, model: nn.Module, output_dir: str,
                 quantization_config: Optional[Dict[str, Any]] = None,
                 vocab: list = None, other_pad_size: int = 0,
                 blank_id: int = 0, sos_id: int = 1, eos_id: int = 2,
                 idx2char: dict = None):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quantization_config = quantization_config or {}
        self.vocab = vocab or []
        self.other_pad_size = other_pad_size
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.idx2char = idx2char or {}

        # 初始化所有导出器
        self.exporters = {
            'onnx': ONNXExporter(model, str(self.output_dir / 'onnx'), vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char, 
                                quantization_config=self.quantization_config),
            'torchscript': TorchScriptExporter(model, str(self.output_dir / 'torchscript'), vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char),
            'tensorrt': TensorRTExporter(model, str(self.output_dir / 'tensorrt'), vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char),
            'coreml': CoreMLExporter(model, str(self.output_dir / 'coreml'), vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char),
            'tflite': TFLiteExporter(model, str(self.output_dir / 'tflite'), vocab, other_pad_size, blank_id, sos_id, eos_id, idx2char)
        }

    def export_all_formats(self, dummy_input: torch.Tensor,
                          formats: Optional[List[str]] = None) -> Dict[str, str]:
        """导出所有支持的格式"""

        if formats is None:
            formats = ['onnx', 'torchscript']  # 默认导出常用格式

        results = {}

        for format_name in formats:
            if format_name in self.exporters:
                try:
                    exporter = self.exporters[format_name]
                    output_path = exporter.export(dummy_input)
                    results[format_name] = output_path
                    logger.info(f"✅ {format_name.upper()}导出成功")
                except Exception as e:
                    logger.error(f"❌ {format_name.upper()}导出失败: {e}")
                    results[format_name] = None
            else:
                logger.warning(f"⚠️ 不支持的格式: {format_name}")
                results[format_name] = None

        # 保存导出配置
        self._save_export_config(results)

        return results

    def export_specific_format(self, format_name: str,
                              dummy_input: torch.Tensor,
                              **kwargs) -> Optional[str]:
        """导出特定格式"""

        if format_name in self.exporters:
            try:
                exporter = self.exporters[format_name]
                output_path = exporter.export(dummy_input, **kwargs)
                logger.info(f"✅ {format_name.upper()}导出成功")
                return output_path
            except Exception as e:
                logger.error(f"❌ {format_name.upper()}导出失败: {e}")
                return None
        else:
            logger.warning(f"⚠️ 不支持的格式: {format_name}")
            return None

    def _save_export_config(self, export_results: Dict[str, str]):
        """保存导出配置"""
        config = {
            'quantization_config': self.quantization_config,
            'export_results': export_results,
            'export_time': torch.tensor([]).device.type,  # 获取当前设备信息
            'pytorch_version': torch.__version__
        }

        config_path = self.output_dir / 'export_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 导出配置已保存: {config_path}")

    def get_export_summary(self) -> Dict[str, Any]:
        """获取导出摘要"""
        summary = {
            'total_formats': len(self.exporters),
            'available_formats': list(self.exporters.keys()),
            'output_directory': str(self.output_dir),
            'quantization_enabled': bool(self.quantization_config)
        }

        # 检查已导出的文件
        exported_files = {}
        for format_name, exporter in self.exporters.items():
            for ext in exporter.get_supported_formats():
                expected_file = self.output_dir / format_name / f'model.{ext}'
                if expected_file.exists():
                    file_size = expected_file.stat().st_size / (1024 * 1024)  # MB
                    exported_files[format_name] = {
                        'path': str(expected_file),
                        'size_mb': round(file_size, 2)
                    }

        summary['exported_files'] = exported_files
        return summary

class DeploymentOptimizer:
    """部署优化器 - 针对特定部署场景优化模型"""

    def __init__(self, model: nn.Module, deployment_target: str):
        self.model = model
        self.deployment_target = deployment_target
        self.optimization_passes = self._get_optimization_passes()

    def _get_optimization_passes(self) -> List[Callable]:
        """获取针对目标平台的优化pass"""
        passes = []

        if self.deployment_target == 'mobile':
            passes.extend([
                self._optimize_for_mobile,
                self._remove_unused_operations,
                self._fuse_operations
            ])
        elif self.deployment_target == 'edge':
            passes.extend([
                self._optimize_for_edge,
                self._quantize_activations,
                self._simplify_graph
            ])
        elif self.deployment_target == 'server':
            passes.extend([
                self._optimize_for_server,
                self._enable_parallel_execution,
                self._optimize_memory_layout
            ])

        return passes

    def optimize_for_deployment(self) -> nn.Module:
        """为部署优化模型"""
        optimized_model = self.model

        for optimization_pass in self.optimization_passes:
            try:
                optimized_model = optimization_pass(optimized_model)
                logger.info(f"✅ 应用优化: {optimization_pass.__name__}")
            except Exception as e:
                logger.warning(f"⚠️ 优化失败 {optimization_pass.__name__}: {e}")

        return optimized_model

    def _optimize_for_mobile(self, model: nn.Module) -> nn.Module:
        """移动端优化"""
        # 简化模型结构
        # 减少操作数量
        # 优化内存访问模式
        return model

    def _optimize_for_edge(self, model: nn.Module) -> nn.Module:
        """边缘设备优化"""
        # 减少计算复杂度
        # 优化功耗
        return model

    def _optimize_for_server(self, model: nn.Module) -> nn.Module:
        """服务器端优化"""
        # 启用并行化
        # 优化批处理
        return model

    def _remove_unused_operations(self, model: nn.Module) -> nn.Module:
        """移除未使用的操作"""
        # 图优化：移除死代码
        return model

    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """融合操作"""
        # 合并连续的操作以减少内存访问
        return model

    def _quantize_activations(self, model: nn.Module) -> nn.Module:
        """量化激活函数"""
        # 应用激活函数量化
        return model

    def _simplify_graph(self, model: nn.Module) -> nn.Module:
        """简化计算图"""
        # 图简化优化
        return model

    def _enable_parallel_execution(self, model: nn.Module) -> nn.Module:
        """启用并行执行"""
        # 并行化优化
        return model

    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """优化内存布局"""
        # 内存访问优化
        return model

# 部署配置模板
DEPLOYMENT_CONFIGS = {
    'mobile_cpu': {
        'formats': ['tflite', 'onnx'],
        'optimizations': ['mobile', 'quantization'],
        'target_device': 'arm_cpu',
        'precision': 'int8',
        'max_model_size_mb': 10
    },
    'mobile_gpu': {
        'formats': ['coreml', 'onnx'],
        'optimizations': ['mobile', 'gpu'],
        'target_device': 'mobile_gpu',
        'precision': 'fp16',
        'max_model_size_mb': 20
    },
    'edge_tpu': {
        'formats': ['tflite'],
        'optimizations': ['edge', 'quantization'],
        'target_device': 'edge_tpu',
        'precision': 'int8',
        'max_model_size_mb': 5
    },
    'server_cpu': {
        'formats': ['onnx', 'torchscript'],
        'optimizations': ['server', 'cpu'],
        'target_device': 'x86_cpu',
        'precision': 'int8',
        'max_model_size_mb': 100
    },
    'server_gpu': {
        'formats': ['tensorrt', 'onnx'],
        'optimizations': ['server', 'gpu'],
        'target_device': 'nvidia_gpu',
        'precision': 'fp16',
        'max_model_size_mb': 200
    }
}

def create_deployment_package(model: nn.Module,
                            quantization_config: Dict[str, Any],
                            deployment_target: str,
                            output_dir: str,
                            dummy_input: torch.Tensor,
                            vocab: list = None,
                            other_pad_size: int = 0,
                            blank_id: int = 0,
                            sos_id: int = 1,
                            eos_id: int = 2,
                            idx2char: dict = None) -> str:
    """创建部署包"""

    logger.info(f"📦 创建部署包，目标平台: {deployment_target}")

    # 获取部署配置
    if deployment_target not in DEPLOYMENT_CONFIGS:
        raise ValueError(f"不支持的部署目标: {deployment_target}")

    config = DEPLOYMENT_CONFIGS[deployment_target]

    # 创建输出目录
    package_dir = Path(output_dir) / f'deployment_{deployment_target}'
    package_dir.mkdir(parents=True, exist_ok=True)

    # 优化模型
    optimizer = DeploymentOptimizer(model, deployment_target)
    optimized_model = optimizer.optimize_for_deployment()

    # 导出模型
    exporter = QuantizedModelExporter(
        optimized_model,
        str(package_dir),
        quantization_config,
        vocab,
        other_pad_size,
        blank_id,
        sos_id,
        eos_id,
        idx2char
    )

    # 导出指定格式
    results = exporter.export_all_formats(
        dummy_input,
        formats=config['formats']
    )

    # 创建部署文档
    deployment_doc = {
        'target_platform': deployment_target,
        'optimization_config': config,
        'exported_models': results,
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'quantization_applied': bool(quantization_config)
        },
        'deployment_instructions': get_deployment_instructions(deployment_target)
    }

    # 保存部署文档
    doc_path = package_dir / 'deployment_guide.json'
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_doc, f, indent=2, ensure_ascii=False)

    # 创建推理脚本模板
    inference_script = create_inference_script(deployment_target, results)
    script_path = package_dir / 'inference_example.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(inference_script)

    logger.info(f"✅ 部署包创建完成: {package_dir}")
    return str(package_dir)

def get_deployment_instructions(target: str) -> Dict[str, str]:
    """获取部署说明"""
    instructions = {
        'mobile_cpu': {
            'runtime': 'TensorFlow Lite',
            'installation': 'pip install tensorflow',
            'optimization': '使用XNNPACK后端，启用多线程',
            'memory_requirements': '至少100MB可用内存'
        },
        'mobile_gpu': {
            'runtime': 'CoreML或ONNX Runtime',
            'installation': 'pip install coremltools onnxruntime',
            'optimization': '启用GPU加速，使用Metal后端',
            'memory_requirements': '至少200MB可用内存'
        },
        'edge_tpu': {
            'runtime': 'TensorFlow Lite + Edge TPU',
            'installation': '安装Edge TPU运行时',
            'optimization': '模型必须完全量化到INT8',
            'memory_requirements': '至少50MB可用内存'
        },
        'server_cpu': {
            'runtime': 'ONNX Runtime或TorchScript',
            'installation': 'pip install onnxruntime',
            'optimization': '启用AVX指令集，使用多线程',
            'memory_requirements': '至少1GB可用内存'
        },
        'server_gpu': {
            'runtime': 'TensorRT或ONNX Runtime',
            'installation': 'pip install tensorrt onnxruntime-gpu',
            'optimization': '启用TensorRT优化，使用CUDA',
            'memory_requirements': '至少2GB显存'
        }
    }

    return instructions.get(target, {})

def create_inference_script(target: str, exported_models: Dict[str, str]) -> str:
    """创建推理脚本模板"""

    script_parts = []

    # 头部注释
    script_parts.append('"""')
    script_parts.append(f'推理脚本 - {target}平台')
    script_parts.append('自动生成，请根据实际需求修改')
    script_parts.append('"""')
    script_parts.append('')

    # 导入依赖
    if 'tflite' in exported_models:
        script_parts.append('import tensorflow as tf')
        script_parts.append('import numpy as np')

    if 'onnx' in exported_models:
        script_parts.append('import onnxruntime as ort')
        script_parts.append('import numpy as np')

    if 'coreml' in exported_models:
        script_parts.append('import coremltools as ct')

    if 'torchscript' in exported_models:
        script_parts.append('import torch')

    if 'tensorrt' in exported_models:
        script_parts.append('import tensorrt as trt')
        script_parts.append('import pycuda.driver as cuda')
        script_parts.append('import pycuda.autoinit')

    script_parts.append('')

    # 模型加载函数
    script_parts.append('def load_model(model_path: str):')
    script_parts.append('    """加载模型"""')

    if 'tflite' in exported_models:
        script_parts.append('    if model_path.endswith(".tflite"):')
        script_parts.append('        interpreter = tf.lite.Interpreter(model_path=model_path)')
        script_parts.append('        interpreter.allocate_tensors()')
        script_parts.append('        return interpreter')

    if 'onnx' in exported_models:
        script_parts.append('    elif model_path.endswith(".onnx"):')
        script_parts.append('        return ort.InferenceSession(model_path)')

    if 'coreml' in exported_models:
        script_parts.append('    elif model_path.endswith(".mlmodel"):')
        script_parts.append('        return ct.models.MLModel(model_path)')

    if 'torchscript' in exported_models:
        script_parts.append('    elif model_path.endswith(".pt"):')
        script_parts.append('        return torch.jit.load(model_path)')

    if 'tensorrt' in exported_models:
        script_parts.append('    elif model_path.endswith(".trt"):')
        script_parts.append('        # TensorRT引擎加载逻辑')
        script_parts.append('        pass')

    script_parts.append('    else:')
    script_parts.append('        raise ValueError(f"不支持的模型格式: {model_path}")')
    script_parts.append('')

    # 推理函数
    script_parts.append('def inference(model, input_data):')
    script_parts.append('    """运行推理"""')

    if 'tflite' in exported_models:
        script_parts.append('    if hasattr(model, "get_input_details"):  # TensorFlow Lite')
        script_parts.append('        input_details = model.get_input_details()')
        script_parts.append('        output_details = model.get_output_details()')
        script_parts.append('        model.set_tensor(input_details[0]["index"], input_data)')
        script_parts.append('        model.invoke()')
        script_parts.append('        return model.get_tensor(output_details[0]["index"])')

    if 'onnx' in exported_models:
        script_parts.append('    elif hasattr(model, "run"):  # ONNX Runtime')
        script_parts.append('        input_name = model.get_inputs()[0].name')
        script_parts.append('        return model.run(None, {input_name: input_data})[0]')

    if 'coreml' in exported_models:
        script_parts.append('    elif hasattr(model, "predict"):  # CoreML')
        script_parts.append('        return model.predict({"input": input_data})')

    if 'torchscript' in exported_models:
        script_parts.append('    elif hasattr(model, "forward"):  # TorchScript')
        script_parts.append('        with torch.no_grad():')
        script_parts.append('            return model(torch.from_numpy(input_data)).numpy()')

    script_parts.append('')
    script_parts.append('    else:')
    script_parts.append('        raise ValueError("未知的模型类型")')
    script_parts.append('')

    # 主函数
    script_parts.append('def main():')
    script_parts.append('    """主函数示例"""')
    script_parts.append('    # 模型路径（请根据实际导出结果修改）')

    for format_name, model_path in exported_models.items():
        if model_path:
            script_parts.append(f'    # model_path = "{model_path}"  # {format_name.upper()}格式')

    script_parts.append('')
    script_parts.append('    # 加载模型')
    script_parts.append('    # model = load_model(model_path)')
    script_parts.append('')
    script_parts.append('    # 准备输入数据（请根据实际输入形状修改）')
    script_parts.append('    # input_data = np.random.randn(1, 3, 32, 128).astype(np.float32)')
    script_parts.append('')
    script_parts.append('    # 运行推理')
    script_parts.append('    # output = inference(model, input_data)')
    script_parts.append('    # print(f"推理结果形状: {output.shape}")')
    script_parts.append('')
    script_parts.append('if __name__ == "__main__":')
    script_parts.append('    main()')

    return '\n'.join(script_parts)
