# AIToyOCRPro - AI Agent Guide

## Project Overview

AIToyOCRPro 是一个基于 PyTorch 的 OCR 文本识别网络量化感知训练（QAT, Quantization-Aware Training）项目。该项目提供完整的模型量化、剪枝、知识蒸馏和超参数自动优化方案，旨在保持高精度的同时显著减少模型大小和推理延迟。

### Key Information

| 属性 | 值 |
|------|-----|
| 主要语言 | Python 3.12+ |
| 深度学习框架 | PyTorch 2.9+ |
| 量化库 | torchao 0.16.0 (PyTorch官方量化库) |
| 主要功能 | OCR文本识别、模型量化、剪枝、部署 |
| 操作系统支持 | Windows/Linux/macOS |

## Project Structure

```
AIToyOCRPro/
├── main.py                 # 主入口 - 训练、评估、部署、优化研究
├── model.py                # RecNetwork 模型架构定义
├── decoder.py              # 混合解码器 (CTC + AR双分支)
├── quantization.py         # 量化与剪枝核心实现
├── loss.py                 # 损失函数 (CTC + 蒸馏 + 量化感知)
├── data.py                 # 数据加载与合成分布式数据集
├── deployment.py           # 部署工具 (ONNX/TensorRT/CoreML/TFLite)
├── debug.py                # 可视化与调试工具
├── transforms.py           # 数据增强变换
├── tia.py                  # TIA文本图像增强 (移动最小二乘法)
├── viptr.py                # VIPTRv2 主干网络实现
├── svtr.py                 # SVTRv2 主干网络实现
├── lstm.py                 # xLSTM 实现 (sLSTM + mLSTM)
├── test_*.py               # 测试文件 (4个测试模块)
├── requirements.txt        # 依赖配置
├── README.md               # 详细文档
├── arial.ttf               # 字体文件 (数据合成用)
└── output/                 # 输出目录
    ├── models/             # 保存的模型文件 (.pth)
    ├── reports/            # 评估报告 (.json)
    ├── visualizations/     # 可视化图表 (.png)
    ├── logs/               # 训练日志
    └── deployment_*/       # 部署包 (ONNX/TensorRT等)
```

## Technology Stack

### Core Dependencies

| 包名 | 用途 | 版本要求 |
|------|------|----------|
| torch | 深度学习框架 | 2.9+ |
| torchao | PyTorch官方量化库 | 0.16.0 |
| timm | 图像模型库 | latest |
| numpy | 数值计算 | latest |
| Pillow | 图像处理 | latest |
| opencv-python | 计算机视觉 | latest |
| albumentations | 数据增强 | latest |
| lmdb | LMDB数据集支持 | latest |
| editdistance | 编辑距离计算 | latest |
| optuna | 超参数优化 | latest |
| onnxscript/onnxruntime | ONNX导出与推理 | latest |
| matplotlib | 可视化 | latest |
| tqdm | 进度条 | latest |

### Model Architecture

```
输入图像 [B, 3, H, W]
    ↓
Backbone (主干网络) - 可选:
├── HGNetV2 (hgnetv2_b0 ~ b6)
├── ConvNeXtV2 (convnextv2_atto ~ huge)
├── MobileNetV4 (mobilenetv4_conv_small/medium/large)
├── RepViT (repvit_tiny/small/base/large)
├── ViT (vit_tiny_patch16_224)
├── VIPTRv2 (viptr2) [默认]
└── SVTRv2 (svtrv2_tiny/small/base/large)
    ↓
Neck (特征融合)
├── HybridNeck
├── SVTRNeck
├── RepVitMultiScaleNeck
├── UltraLightweightSVTRv2Neck
└── nn.Identity (viptr2)
    ↓
Decoder (混合解码器)
├── CTC Decoder: RopeTransformerEncoder
│   ├── RoPE + GQA (Group Query Attention)
│   ├── Skip-Attention (可选)
│   └── MLA (Multi-head Latent Attention, 可选)
└── AR Decoder: RopeTransformerArDecoder
    ├── RoPE Transformer Decoder Layer
    ├── KV-Cache 推理优化
    └── Beam Search 支持
```

## Build and Run Commands

### Environment Setup

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; import torchao; print('环境就绪')"
```

### Training Commands

```bash
# 基础训练 (CTC模式)
python main.py --mode both --model_name viptr2 --epochs 60

# 混合模式训练 (CTC + AR)
python main.py --mode both --train_mode hybrid --epochs 60

# 使用预定义配置模板
python main.py --mode both --template ocr_balanced        # 平衡配置
python main.py --mode both --template ocr_conservative    # 高精度
python main.py --mode both --template ocr_aggressive      # 高压缩
python main.py --mode both --template mobile_optimized    # 移动端
```

### Quantization Commands

```bash
# QAT (量化感知训练) - 推荐
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --quantization_strategy int8_dyn_act_int4_weight \
    --qat_epochs 8

# 分层混合精度QAT
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --enable_layer_wise_qat \
    --qat_epochs 10

# 分阶段QAT训练
python main.py \
    --enable_quantization \
    --quantization_mode qat \
    --warmup_lr 3 \
    --qat_insert_epoch 3 \
    --qat_epochs 8 \
    --epochs 20

# PTQ (训练后量化)
python main.py \
    --enable_quantization \
    --quantization_mode ptq \
    --quantization_strategy int8_dyn_act_int4_weight
```

### Pruning Commands

```bash
# 基础剪枝
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20

# 训练期间应用剪枝 (推荐)
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20 \
    --apply_pruning_during_training \
    --validate_pruning

# 结构化压缩 (移动端部署)
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_structured \
    --pruning_ratio 0.4 \
    --apply_pruning_during_training \
    --structural_compression
```

### Optimization Study Commands

```bash
# 贝叶斯优化
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 50 \
    --optimization_target balanced \
    --study_name my_study

# 快速验证 (dry run)
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 5 \
    --dry_run \
    --batch_size 4 \
    --num_train 4 \
    --num_val 4
```

### Evaluation Commands

```bash
# 全面评估
python main.py --mode evaluate --generate_report --visualize

# 仅评估不训练
python main.py --mode evaluate --num_val 500
```

### Testing Commands

```bash
# 运行所有测试
python test_quantization.py
python test_decoder.py
python test_loss.py
python test_svtr.py

# 运行特定测试类别
python test_quantization.py --test basic       # 基础测试
python test_quantization.py --test core        # 核心功能
python test_quantization.py --test optimized   # 优化功能
python test_quantization.py --test integration # 集成测试
```

## Code Style Guidelines

### Python Code Style

1. **命名规范**:
   - 类名: `PascalCase` (如 `QuantizationManager`, `RecNetwork`)
   - 函数名: `snake_case` (如 `prepare_model_for_quantization`)
   - 常量: `UPPER_CASE` (如 `VOCAB_SIZE`, `BLANK`)
   - 私有方法: `_leading_underscore` (如 `_is_weight_pruned`)

2. **文档字符串**:
   - 使用 `"""triple double quotes"""`
   - 类文档包含功能描述和属性说明
   - 函数文档包含 Args, Returns, Note 等章节
   - 使用中文注释和文档

3. **类型注解**:
   - 函数参数和返回值使用类型注解
   - 示例: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
   - 复杂类型使用 `typing` 模块: `Optional`, `Dict`, `List`, `Tuple`

4. **代码组织**:
   - 每个模块文件开始包含模块文档字符串
   - 相关功能分组，使用注释分隔
   - 导入顺序: 标准库 → 第三方库 → 本地模块

### Example Code Pattern

```python
class QuantizationManager:
    """
    量化管理器 - 处理模型量化的核心逻辑
    
    提供量化感知训练(QAT)和训练后量化(PTQ)功能
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        初始化量化管理器
        
        Args:
            model: 待量化的模型
            config: 量化配置字典
        """
        self.model = model
        self.config = config
        
    def prepare_model_for_quantization(self) -> nn.Module:
        """
        准备模型进行量化
        
        Returns:
            准备好的量化模型
        """
        # 实现逻辑
        pass
```

## Testing Instructions

### Test Structure

- `test_quantization.py`: 剪枝和量化功能测试
- `test_decoder.py`: 解码器功能测试
- `test_loss.py`: 损失函数测试
- `test_svtr.py`: SVTR模型测试

### Running Tests

```bash
# 运行单个测试文件
python test_quantization.py

# 运行特定测试类别
python test_quantization.py --test basic

# 注意: Windows上部分测试可能因编码问题失败，建议在Linux环境运行完整测试
```

### Adding New Tests

1. 在对应的 `test_*.py` 文件中添加测试函数
2. 使用 `print_header()`, `print_subheader()`, `print_result()` 工具函数
3. 测试函数应返回 `bool` 表示通过/失败
4. 在 `TEST_SUITES` 字典中注册新测试

## Configuration System

### Configuration Files

项目支持 JSON 配置文件，默认路径为 `quantization_config.json`:

```json
{
  "enabled": true,
  "strategy": "int8_dyn_act_int4_weight",
  "quantization_aware_training": true,
  "qat_epochs": 8,
  "weight_bits": 4,
  "activation_bits": 8,
  "qat_learning_rate_multiplier": 0.1,
  "quantization_loss_weight": 0.01,
  "temperature_distillation": 4.0,
  "distillation_weight": 0.3
}
```

### Predefined Templates

| 模板名称 | 策略 | QAT轮数 | 适用场景 |
|----------|------|---------|----------|
| `ocr_conservative` | int8_dyn_act_int4_weight | 8 | 高精度优先 |
| `ocr_balanced` | int8_dyn_act_int4_weight | 5 | 平衡精度与压缩 |
| `ocr_aggressive` | int4_weight_only | 10 | 最大压缩比 |
| `mobile_optimized` | int4_weight_only | 6 | 移动端部署 |
| `server_optimized` | int8_dyn_act_int4_weight | 4 | 服务器GPU |

## Output Directory Structure

```
output/
├── models/
│   ├── ocr_latest.pth              # 最新模型
│   ├── ocr_best_cer.pth            # 最佳CER模型
│   ├── ocr_best_em.pth             # 最佳EM模型
│   ├── original_model.pth          # 原始模型
│   ├── original_model.onnx         # 原始ONNX
│   └── quantized_model.onnx        # 量化ONNX
├── reports/
│   ├── quantization_report.json    # 量化评估报告
│   └── optimization_report.json    # 优化研究报告
├── visualizations/
│   └── *.png                       # 可视化图表
├── logs/                           # 训练日志
└── deployment_{target}/            # 部署包
    ├── onnx/
    ├── tensorrt/
    ├── coreml/
    └── tflite/
```

## Key Classes and APIs

### QuantizationManager

核心量化管理类，主要方法:

| 方法 | 功能 |
|------|------|
| `prepare_model_for_quantization()` | 准备模型进行量化 |
| `_apply_modern_qat_quantization()` | 应用QAT量化 |
| `_apply_layer_wise_mixed_precision_qat()` | 分层混合精度QAT |
| `get_quantization_loss()` | 计算量化感知损失 |
| `export_quantized_model()` | 导出量化模型 |

### PruningManager

剪枝管理类，主要方法:

| 方法 | 功能 |
|------|------|
| `apply_pruning()` | 标准逐层剪枝 |
| `apply_global_pruning()` | 全局非结构化剪枝 |
| `validate_pruning_with_rollback()` | 验证并回滚 |
| `compress_model_structurally()` | 结构化压缩 |
| `visualize_pruning()` | 生成可视化图表 |

### HybridDecoder

混合解码器，支持:

- `RopeTransformerEncoder`: CTC解码分支
- `RopeTransformerArDecoder`: AR解码分支
- `FeatureAlign`: 特征对齐用于知识蒸馏

## Security Considerations

1. **模型文件安全**: 加载外部 `.pth` 文件时使用 `weights_only=True` (PyTorch 2.0+)
2. **路径安全**: 使用 `pathlib.Path` 处理文件路径，避免路径遍历攻击
3. **输入验证**: 命令行参数使用 `argparse` 进行类型和范围验证
4. **资源限制**: 训练时设置合理的 `num_workers` 避免系统资源耗尽

## Common Development Tasks

### Adding a New Backbone

1. 在 `model.py` 中创建新的 Backbone 类 (继承 `nn.Module`)
2. 在 `RecNetwork.__init__` 中添加模型名称映射
3. 确保输出特征维度与 Neck 兼容

### Adding a New Quantization Strategy

1. 在 `quantization.py` 的 `QuantizationStrategy` 枚举中添加新策略
2. 在 `QuantizationManager.prepare_model_for_quantization()` 中实现策略逻辑
3. 更新配置验证和文档

### Adding a New Test

1. 在对应的 `test_*.py` 中创建测试类或函数
2. 使用提供的工具函数打印测试结果
3. 在 `TEST_SUITES` 中注册测试
4. 确保测试可在隔离环境中运行

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 减小 `batch_size` 或启用梯度检查点
2. **量化精度下降**: 增加 `qat_epochs` 或启用分层混合精度
3. **Windows 编码错误**: 设置 `PYTHONIOENCODING=utf-8` 环境变量
4. **ONNX 导出失败**: 确保使用正确的 `example_input` 维度

### Debug Mode

使用 `debug.py` 中的可视化工具进行调试:

```python
from debug import visualize_predictions, debug_virtual_alignment

# 可视化预测结果
visualize_predictions(model, dataloader, config)
```

## Version History

- **v3.0** (2026-03-29): 当前版本，集成 torchao QAT API，分层混合精度，分阶段训练

## References

- VIPTRNet: https://github.com/yyedekkun/VIPTR
- SVTRv2: https://github.com/Topdu/OpenOCR
- xLSTM: https://github.com/muditbhargava66/PyxLSTM
- torchao: https://github.com/pytorch/ao
