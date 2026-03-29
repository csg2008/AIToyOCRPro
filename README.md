# OCR文本识别网络量化感知训练方案

## 📋 概述

AIToyOCRPro 是一个为 OCR 文本识别网络提供完整量化感知训练（QAT, Quantization-Aware Training）方案的 PyTorch 项目。该项目集成了多种先进的量化技术、模型剪枝、知识蒸馏和超参数自动优化策略，旨在在保持高精度的同时显著减少模型大小和推理延迟。

**环境信息**:
- **文档版本**: 3.0
- **更新日期**: 2026-03-29
- **Python版本**: 3.12+
- **PyTorch版本**: 2.9+
- **torchao版本**: 0.16.0 (PyTorch官方量化库)
- **操作系统**: Windows/Linux/macOS

## 🎯 核心特性

### 量化策略
- **INT8动态激活 + INT4权重量化**：平衡精度和压缩比的最优方案
- **纯INT8权重量化**：适用于对精度要求较高的场景
- **INT4权重量化**：最大化压缩比，适用于边缘设备
- **分层混合精度量化**：Backbone/Neck/Decoder使用不同精度

### 训练方法
- **量化感知训练（QAT）**：在训练过程中模拟量化效果，支持延迟插入
- **训练后量化（PTQ）**：快速部署，无需重新训练
- **知识蒸馏**：使用原始模型作为教师网络指导量化训练
- **分阶段QAT训练**：warmup → QAT插入 → QAT微调 → 后QAT

### 模型剪枝
- **精度优先剪枝**：确保剪枝后精度下降控制在可接受范围内
- **分层剪枝策略**：针对不同层设置不同的剪枝比例
- **结构化/非结构化剪枝**：支持多种剪枝方法
- **剪枝验证与回滚**：自动验证剪枝效果，下降过大则回滚
- **结构化压缩**：真正删除被剪枝的通道/神经元

### 优化技术
- **自适应超参数优化**：基于Optuna自动搜索最优量化配置
- **多目标优化**：平衡精度、压缩比和推理速度
- **硬件感知优化**：针对CPU、GPU、移动端优化
- **混合精度训练**：支持BF16/FP16自动混合精度

### QAT训练优化特性

| 优化项 | 描述 | 优势 |
|--------|------|------|
| **QAT API** | 基于`FakeQuantizeConfig`的配置系统 | 精细的粒度控制，支持`PerToken`动态量化 |
| **分层混合精度** | `ComposableQATQuantizer`分层策略 | Backbone/Neck/Decoder使用不同精度 |
| **分阶段训练** | 4阶段QAT训练流程 | 冻结BN/LN层，提高量化稳定性 |
| **改进的损失函数** | 多尺度量化损失 | MSE + Cosine + L1 + FakeQuant损失 |
| **统一导出流程** | 支持`from_intx_quantization_aware_training` | 更好的推理性能兼容性 |

## 🏗️ 模型架构

### RecNetwork 结构

```
输入图像 [B, 3, H, W]
    ↓
Backbone (主干网络)
├── HGNetV2 (hgnetv2_b0 ~ b6)
├── ConvNeXtV2 (convnextv2_atto ~ huge)
├── MobileNetV4 (mobilenetv4_conv_small/medium/large)
├── RepViT (repvit_tiny/small/base/large)
├── ViT (vit_tiny_patch16_224)
├── VIPTRv2 (viptr2)
└── SVTRv2 (svtrv2_tiny/small/base/large)
    ↓
Neck (特征融合)
├── HybridNeck (attention/avg pool)
├── SVTRNeck
├── RepVitMultiScaleNeck
├── UltraLightweightSVTRv2Neck
└── nn.Identity (viptr2)
    ↓
Decoder (混合解码器)
├── CTC Decoder: RopeTransformerEncoder
│   ├── RoPE + GQA (Group Query Attention)
│   ├── Skip-Attention (可选)
│   ├── MLA (Multi-head Latent Attention, 可选)
│   └── Gradient Checkpointing
├── AR Decoder: RopeTransformerArDecoder
│   ├── RoPE Transformer Decoder Layer
│   ├── KV-Cache 推理优化
│   └── Beam Search 支持
└── FeatureAlign (知识蒸馏用)
```

### 支持的训练模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `ctc` | 仅CTC解码分支 | 高速推理，序列标注 |
| `ar` | 仅自回归解码分支 | 高精度，序列生成 |
| `hybrid` | CTC + AR联合训练 | 平衡精度与速度，支持蒸馏 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; import torchao; print('环境就绪')"
```

### 2. 基础使用

#### 使用预定义配置

```bash
# 使用平衡配置（推荐）
python main.py --mode both --template ocr_balanced

# 使用保守配置（高精度）
python main.py --mode both --template ocr_conservative

# 使用激进配置（高压缩）
python main.py --mode both --template ocr_aggressive

# 移动端优化
python main.py \
    --mode both \
    --template mobile_optimized \
    --hardware_target mobile
```

#### 自定义配置

```bash
# 基本自定义
python main.py \
    --mode both \
    --quantization_strategy int8_dyn_act_int4_weight \
    --weight_bits 4 \
    --activation_bits 8 \
    --qat_epochs 5

# QAT (推荐 - 精度优先)
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --quantization_strategy int8_dyn_act_int4_weight \
    --use_modern_qat_api \
    --qat_epochs 8

# 分层混合精度QAT (最高精度)
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --enable_layer_wise_qat \
    --qat_epochs 10 \
    --qat_learning_rate_multiplier 0.05

# 分阶段QAT训练
python main.py \
    --enable_quantization \
    --quantization_mode qat \
    --warmup_lr 3 \
    --qat_insert_epoch 3 \
    --qat_epochs 8 \
    --epochs 20 \
    --mode both

训练后量化 (PTQ)
python main.py \
    --enable_quantization \
    --quantization_mode ptq \
    --quantization_strategy int8_dyn_act_int4_weight \
    --mode both
    
# 高级自定义
python main.py \
    --mode both \
    --config my_quantization_config.json \
    --auto_optimize \
    --target_compression_ratio 0.25 \
    --preserve_accuracy
```

#### 模型剪枝

```bash
# 基础剪枝（部署时应用）
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20

# 训练期间应用剪枝（推荐）
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20 \
    --apply_pruning_during_training \
    --validate_pruning \
    --visualize_pruning

# 移动端部署（结构化压缩）
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_structured \
    --pruning_ratio 0.4 \
    --pruning_epoch 25 \
    --apply_pruning_during_training \
    --structural_compression

# 分层剪枝比例
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --backbone_pruning_ratio 0.2 \
    --neck_pruning_ratio 0.3 \
    --decoder_pruning_ratio 0.1

# 剪枝+量化组合优化
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_epoch 20 \
    --finetune_epochs 10 \
    --quantization_strategy int8_dyn_act_int4_weight
```

### 超参数自动优化

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

### 3. 配置文件示例

创建 `quantization_config.json`:

```json
{
  "enabled": true,
  "strategy": "int8_dyn_act_int4_weight",
  "quantization_aware_training": true,
  "post_training_quantization": false,
  "qat_epochs": 8,
  "weight_bits": 4,
  "activation_bits": 8,
  "observer_type": "moving_average",
  "quantization_granularity": "per_channel",
  "qat_learning_rate_multiplier": 0.1,
  "quantization_loss_weight": 0.01,
  "temperature_distillation": 4.0,
  "distillation_weight": 0.3,
  "calibration_batches": 100,
  "mixed_precision": true,
  "memory_efficient": true,
  "_comment": "QAT优化配置",
  "use_modern_qat_api": true,
  "enable_layer_wise_qat": false,
  "qat_insert_epoch": 3,
  "fake_quant_loss_weight": 0.001
}
```

### 预定义配置模板

| 模板 | 策略 | 权重量化 | 激活量化 | QAT轮数 | 适用场景 |
|------|------|----------|----------|---------|----------|
| `ocr_conservative` | int8_dyn_act_int4_weight | INT4 | INT8 | 8 | 高精度优先 |
| `ocr_balanced` | int8_dyn_act_int4_weight | INT4 | INT8 | 5 | 平衡精度与压缩 |
| `ocr_aggressive` | int4_weight_only | INT4 | INT4 | 10 | 最大压缩比 |
| `mobile_optimized` | int4_weight_only | INT4 | INT4 | 6 | 移动端部署 |
| `server_optimized` | int8_dyn_act_int4_weight | INT4 | INT8 | 4 | 服务器GPU |

## 📁 输出文件

训练完成后，`output/` 目录下会生成：

```
output/
├── models/
│   ├── ocr_latest.pth              # 最新模型
│   ├── ocr_best_cer.pth            # 最佳CER模型
│   ├── ocr_best_em.pth             # 最佳EM模型
│   ├── original_model.pth          # 原始模型
│   ├── original_model.onnx         # 原始ONNX
│   ├── quantized_model.onnx        # 量化ONNX
│   └── pruning_candidates.json     # 剪枝候选信息
├── reports/
│   ├── quantization_report.json    # 量化评估报告
│   └── optimization_report.json    # 优化研究报告
├── visualizations/
│   ├── quantization_results.png    # 量化效果可视化
│   ├── pruning_epoch_{N}.png       # 训练期间剪枝可视化
│   └── pruning_deployment.png      # 部署时剪枝可视化
└── logs/
    └── (训练日志)
```

优化研究会额外生成：

```
output/optimization_study_{name}/
├── study_config.json
├── best_quantization_config.json
├── optimization_history.json
├── best_metrics.json
├── optimization_report.json
└── optimization_visualization.png
```

## 🎛️ 命令行参数完整列表

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `quantization_config.json` | 量化配置文件路径 |
| `--template` | str | `ocr_balanced` | 预定义模板: `ocr_conservative`/`ocr_balanced`/`ocr_aggressive`/`mobile_optimized`/`server_optimized` |
| `--mode` | str | `both` | 运行模式: `train`/`evaluate`/`deployment`/`optimization_study`/`both` |

### 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | `viptr2` | 模型名称，见下方支持列表 |
| `--epochs` | int | 60 | 训练轮数 |
| `--batch_size` | int | 512 | 批次大小 |
| `--learning_rate` | float | 3e-4 | 学习率 |
| `--train_mode` | str | `ctc` | 训练模式: `ctc`/`ar`/`hybrid` |
| `--num_layers` | int | 3 | 网络层数 |
| `--num_heads` | int | 6 | 注意力头数 |
| `--d_model` | int | 384 | 模型隐藏维度 |
| `--in_channels` | int | 3 | 输入通道数 |
| `--max_text_length` | int | 70 | 最大文本长度 |
| `--dropout` | float | 0.05 | dropout比例 |
| `--num_workers` | int | 0 | 数据加载进程数 |
| `--warmup_lr` | int | 3 | 前N轮线性增大学习率 |
| `--warmup_decoder` | int | 10 | 前N轮不开启蒸馏 |

### 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_train` | int | 95000 | 训练样本数 |
| `--num_val` | int | 5000 | 验证样本数 |
| `--img_height` | int | 32 | 图像高度 |
| `--img_min_width` | int | 128 | 图像最小宽度 |
| `--img_export_width` | int | 512 | 导出图像宽度 |
| `--min_chars` | int | 15 | 最少字符数 |
| `--max_chars` | int | 50 | 最多字符数 |
| `--dataset_type` | str | `synthetic` | 数据集类型: `synthetic`/`lmdb` |
| `--lmdb_data_dir` | str | None | LMDB数据集目录 |

### 量化配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_quantization` | flag | False | 启用量化训练 |
| `--quantization_mode` | str | `qat` | 量化模式: `qat`/`ptq`/`both` |
| `--quantization_strategy` | str | `int8_dyn_act_int4_weight` | 量化策略 |
| `--qat_epochs` | int | 5 | QAT训练轮数 |
| `--weight_bits` | int | 4 | 权重量化位数 |
| `--activation_bits` | int | 8 | 激活量化位数 |
| `--enable_layer_wise_qat` | flag | False | 启用分层混合精度QAT |
| `--qat_insert_epoch` | int | None | QAT插入epoch，默认与`warmup_lr`相同 |

### 剪枝配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_pruning` | flag | False | 启用模型剪枝 |
| `--pruning_strategy` | str | `l1_unstructured` | 剪枝策略 |
| `--pruning_ratio` | float | 0.3 | 全局剪枝比例 |
| `--pruning_layers` | list | `[backbone, neck, decoder]` | 需要剪枝的层 |
| `--pruning_epoch` | int | 20 | 剪枝执行的epoch |
| `--min_acc_drop` | float | 0.01 | 允许的最大精度下降 |
| `--finetune_epochs` | int | 10 | 剪枝后的微调轮数 |
| `--prune_criteria` | str | `l1` | 剪枝标准: `l1`/`l2`/`grad` |
| `--backbone_pruning_ratio` | float | 0.2 | Backbone剪枝比例 |
| `--neck_pruning_ratio` | float | 0.3 | Neck剪枝比例 |
| `--decoder_pruning_ratio` | float | 0.1 | Decoder剪枝比例 |
| `--apply_pruning_during_training` | flag | False | 训练期间实际应用剪枝 |
| `--validate_pruning` | flag | False | 剪枝后验证精度并回滚 |
| `--visualize_pruning` | flag | False | 生成剪枝可视化图表 |
| `--structural_compression` | flag | False | 部署时执行结构化压缩 |

### 优化研究配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--study_name` | str | None | 优化研究名称 |
| `--method` | str | `bayesian` | 优化方法: `bayesian`/`grid_search`/`random_search` |
| `--n_trials` | int | 50 | 优化试验次数 |
| `--param_config` | str | None | 参数配置文件路径 |
| `--optimization_target` | str | `balanced` | 优化目标: `balanced`/`accuracy`/`compression`/`speed` |
| `--dry_run` | flag | False | 只验证代码流程，不执行实际优化 |

### 硬件与部署配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | `auto` | 计算设备: `auto`/`cpu`/`cuda` |
| `--hardware_target` | str | `cpu` | 目标硬件平台: `cpu`/`gpu`/`mobile` |
| `--deployment_target` | str | `server_cpu` | 部署目标: `mobile_cpu`/`mobile_gpu`/`edge_tpu`/`server_cpu`/`server_gpu` |
| `--output_dir` | str | `output` | 输出目录 |
| `--save_models` | flag | True | 保存模型文件 |
| `--generate_report` | flag | True | 生成评估报告 |
| `--visualize` | flag | True | 生成可视化图表 |

## 🏷️ 支持的模型名称

### Backbone 模型

| 模型系列 | 可用名称 | 说明 |
|----------|----------|------|
| **VIPTRv2** | `viptr2` | 默认模型，专为中文OCR优化 |
| **SVTRv2** | `svtrv2_tiny`, `svtrv2_small`, `svtrv2_base` | 轻量级序列建模 |
| **HGNetV2** | `hgnetv2_b0` ~ `hgnetv2_b6` | 百度高性能骨干网络 |
| **ConvNeXtV2** | `convnextv2_atto` ~ `convnextv2_huge` | Meta现代CNN架构 |
| **MobileNetV4** | `mobilenetv4_conv_small`, `mobilenetv4_conv_medium`, `mobilenetv4_conv_large` | 移动端高效网络 |
| **RepViT** | `repvit_tiny`, `repvit_small`, `repvit_base`, `repvit_large` | 重参数化ViT |
| **ViT** | `vit` | Vision Transformer |

## 📈 典型结果

### VIPTRv2 模型量化结果

| 配置 | 原始模型 | INT8+INT4 QAT | INT4 Only | 改善 |
|------|----------|---------------|-----------|------|
| 模型大小 | ~15 MB | ~4.8 MB | ~3.8 MB | **3-4x压缩** |
| 推理时间 | ~12 ms | ~8 ms | ~7 ms | **1.5-1.7x加速** |
| 内存使用 | ~128 MB | ~89 MB | ~76 MB | **30-40%减少** |
| CER | ~2.1% | ~2.3% | ~2.8% | **0.2-0.7%增加** |

### 剪枝+量化组合优化结果

| 指标 | 原始模型 | 剪枝+量化模型 | 改善 |
|------|----------|---------------|------|
| 模型大小 | ~15 MB | ~3.2 MB | **4.7x压缩** |
| 推理时间 | ~12 ms | ~6.8 ms | **1.8x加速** |
| 内存使用 | ~128 MB | ~72 MB | **44%减少** |
| CER | ~2.1% | ~2.4% | **0.3%增加** |
### 4. Python API 用法

#### 量化 API

```python
from quantization import QuantizationConfig, QuantizationManager
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig
import torch

# 配置
config = QuantizationConfig()
config.enabled = True
config.strategy = 'int8_dyn_act_int4_weight'
config.weight_bits = 4
config.activation_bits = 8
config.qat_epochs = 5

# 创建管理器
qm = QuantizationManager(model, config.to_dict())

# 准备模型
quantized_model = qm.prepare_model_for_quantization()

# 训练循环中使用量化损失
loss = criterion(outputs, targets)
quant_loss = qm.get_quantization_loss(quantized_features, original_features)
total_loss = loss + quant_loss

# 导出模型
qm.export_quantized_model(
    pruning=False,
    path='quantized_model.pth',
    epoch=epoch,
    best_cer=best_cer,
    best_em=best_em,
    example_input=dummy_input,
    opt=optimizer.state_dict(),
    scaler=scaler.state_dict(),
    model=model
)
```

#### 剪枝 API

```python
from quantization import PruningConfig, PruningManager

# 配置
pruning_config = PruningConfig({
    'enabled': True,
    'pruning_strategy': 'l1_unstructured',
    'pruning_ratio': 0.3,
    'pruning_layers': ['backbone', 'neck', 'decoder'],
    'pruning_epoch': 20,
    'finetune_epochs': 10,
})

# 创建管理器
pm = PruningManager(pruning_config, model)

# 方案1: 标准逐层剪枝
pm.apply_pruning(epoch, current_acc, best_acc)

# 方案2: 全局剪枝（推荐，效果更好）
pm.apply_global_pruning(epoch, current_acc, best_acc)

# 验证剪枝效果
if pm.validate_pruning_with_rollback(val_loader, max_acc_drop=0.02):
    print("剪枝验证通过")
    
    # 可选：结构化压缩（移动端部署）
    pm.compress_model_structurally(val_loader)

# 可视化剪枝效果
pm.visualize_pruning('pruning_results.png')
```

## 📊 量化效果评估

### 评估指标

我们的量化方案提供全面的评估指标：

1. **精度指标**
   - 原始准确率 vs 量化准确率
   - 精度下降比例
   - 字符错误率（CER）变化

2. **压缩指标**
   - 模型大小压缩比
   - 参数量减少比例
   - 存储空间节省

3. **性能指标**
   - 推理速度提升
   - 内存使用减少
   - 计算复杂度降低

4. **质量指标**
   - 量化误差（MSE、MAE）
   - 信噪比（SNR）
   - 峰值信噪比（PSNR）

### 运行评估

```bash
# 全面评估
python main.py --mode evaluate --generate_report --visualize

# 仅评估不训练
python main.py --mode evaluate --num_val 500
```

### 查看评估报告

评估完成后会在输出目录生成：
- `quantization_report.json`: 详细评估报告
- `quantization_results.png`: 可视化图表
- `optimization_history.json`: 优化历史记录

## 量化策略与API标准

### 支持的量化策略

| 策略 | 类型 | 说明 | 推荐场景 |
|------|------|------|----------|
| `int8_dyn_act_int4_weight` | 混合精度 | INT8动态激活 + INT4权重量化 | 推荐（平衡精度与压缩） |
| `int8_weight_only` | 权重量化 | 纯INT8权重量化 | 保守策略 |
| `int4_weight_only` | 权重量化 | 纯INT4权重量化（最大压缩） | 高压缩比需求 |
| `int8_dynamic_activation_int8_weight` | 全量化 | 纯INT8量化 | 高精度要求 |

### 核心API方法签名

```python
# 量化感知训练 (QAT) 配置
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig

qat_config = QATConfig(
    activation_config=IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity="per_token",
        is_symmetric=False
    ),
    weight_config=IntxFakeQuantizeConfig(
        dtype=torch.int4,
        group_size=128,
        is_symmetric=True
    ),
    step="prepare"
)

# 应用量化
from torchao.quantization import quantize_
quantize_(model, qat_config)
```

## ⚙️ 实现架构

### 核心组件

当前实现在 `quantization.py` 中，包含以下主要组件：

1. **QuantizationConfig**: 量化配置数据类
2. **QuantizationManager**: 量化管理器，核心逻辑
3. **QuantizationEvaluator**: 量化效果评估器
4. **QATTrainingScheduler**: 分阶段QAT训练调度器
5. **PruningQuantizationScheduler**: 剪枝与量化的协同调度器

### 量化配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用量化 |
| `strategy` | str | int8_dyn_act_int4_weight | 量化策略 |
| `quantization_aware_training` | bool | True | 启用QAT |
| `post_training_quantization` | bool | False | 启用PTQ |
| `weight_bits` | int | 4 | 权重量化位数 |
| `activation_bits` | int | 8 | 激活量化位数 |
| `qat_epochs` | int | 5 | QAT训练轮数 |
| `qat_learning_rate_multiplier` | float | 0.1 | QAT学习率倍数 |
| `quantization_loss_weight` | float | 0.01 | 量化损失权重 |
| `temperature_distillation` | float | 4.0 | 蒸馏温度 |
| `distillation_weight` | float | 0.3 | 蒸馏权重 |

## 可用方法

### QuantizationManager 类

| 方法名 | 功能 | 代码位置 |
|--------|------|----------|
| `prepare_model_for_quantization` | 准备模型进行量化 | ~3361行 |
| `_apply_modern_qat_quantization` | 应用QAT量化 | ~3478行 |
| `_apply_layer_wise_mixed_precision_qat` | 分层混合精度QAT | ~3558行 |
| `_apply_ptq_quantization` | 应用训练后量化 | ~3661行 |
| `calibrate_model` | 校准量化模型 | ~3714行 |
| `get_quantization_loss` | 计算量化感知损失 | ~3749行 |
| `convert_to_quantized_model` | 转换FakeQuantize模型 | ~3816行 |
| `export_quantized_model` | 导出量化模型 | ~3901行 |

### QuantizationEvaluator 类

| 方法名 | 功能 | 代码位置 |
|--------|------|----------|
| `evaluate_quantization` | 全面评估量化效果 | ~2857行 |
| `_evaluate_accuracy` | 评估模型精度 | ~2921行 |
| `_get_model_size` | 获取模型大小 | ~2961行 |
| `_benchmark_inference` | 基准测试推理时间 | ~2996行 |
| `_measure_memory_usage` | 测量内存使用 | ~3026行 |
| `generate_report` | 生成量化评估报告 | ~3167行 |

### QATTrainingScheduler 类

| 方法名 | 功能 | 代码位置 |
|--------|------|----------|
| `maybe_insert_qat` | 延迟插入QAT模块 | ~2100行 |
| `get_current_stage` | 获取当前训练阶段 | ~2150行 |
| `get_lr_multiplier` | 获取学习率乘数 | ~2180行 |
| `freeze_bn_ln_for_qat` | 冻结归一化层 | ~2200行 |

### 混合解码器 (HybridDecoder)

`decoder.py` 中的 `HybridDecoder` 支持CTC和AR双分支：

- **CTC分支** (`RopeTransformerEncoder`): 基于Transformer Encoder，使用RoPE位置编码和GQA注意力，支持Skip-Attention和MLA优化
- **AR分支** (`RopeTransformerArDecoder`): 基于Transformer Decoder，使用RoPE和KV-Cache，支持teacher forcing训练和自回归推理
- **特征对齐** (`feature_align`): 用于CTC和AR分支之间的知识蒸馏

### 损失函数 (RecognitionLoss)

`loss.py` 中的 `RecognitionLoss` 组合了多种损失：

- **EnhancedCTCLoss**: 增强CTC损失，支持形近字权重、尾部空白惩罚、字符级Focal Loss、自适应Margin、温度退火
- **DistillationLoss**: 知识蒸馏损失，使用交叉注意力对齐序列长度，支持特征层MSE和Logit层KL散度
- **QuantizationAwareLoss**: 量化感知损失，对比量化模型和原始模型的输出差异

## 关键代码实现

### 1. 分层混合精度QAT

```python
def _apply_layer_wise_mixed_precision_qat(self) -> torch.nn.Module:
    """分层混合精度QAT - 针对不同层使用不同量化策略"""
    
    # Backbone: INT8动态激活 + INT4权重
    backbone_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int4, group_size=backbone_group_size, is_symmetric=True
    )
    
    # Neck: INT8动态激活 + INT4权重 (更小group_size)
    neck_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int4, group_size=neck_group_size, is_symmetric=True
    )
    
    # Decoder: INT8动态激活 + INT8权重 (精度优先)
    decoder_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int8, granularity="per_channel", is_symmetric=True
    )
    
    # 为每层应用相应配置
    for name, module in self.model.named_modules():
        if 'backbone' in name:
            config = QATConfig(activation_config=act_config, weight_config=backbone_weight_config)
        elif 'neck' in name:
            config = QATConfig(activation_config=act_config, weight_config=neck_weight_config)
        elif 'decoder' in name:
            config = QATConfig(activation_config=act_config, weight_config=decoder_weight_config)
        
        quantize_(module, config)
```

### 2. 分阶段QAT训练调度

```python
class QATTrainingScheduler:
    """分阶段QAT训练调度器"""
    
    def maybe_insert_qat(self, epoch: int) -> bool:
        """延迟插入QAT模块"""
        if epoch >= self.config.get('qat_insert_epoch', 3) and not self.qat_inserted:
            # 执行实际的QAT插入
            self.qat_scheduler.apply_qat(self.model)
            self.qat_inserted = True
            return True
        return False
    
    def get_lr_multiplier(self, epoch: int) -> float:
        """根据阶段返回学习率乘数"""
        stage = self.get_current_stage(epoch)
        multipliers = {
            'warmup': 1.0,
            'qat_finetune': self.config.get('qat_learning_rate_multiplier', 0.1),
            'post_qat': 0.05,
        }
        return multipliers.get(stage, 1.0)
```

### 3. 量化感知损失计算

```python
def get_quantization_loss(self, quantized_features: torch.Tensor,
                         original_features: torch.Tensor) -> torch.Tensor:
    """计算量化损失用于知识蒸馏"""
    
    # 1. 特征蒸馏损失 (KL散度)
    temperature = self.config['temperature_distillation']
    distillation_loss = F.kl_div(
        F.log_softmax(quantized_features / temperature, dim=-1),
        F.softmax(original_features / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 2. 量化感知损失 - MSE + Cosine相似度
    mse_loss = F.mse_loss(quantized_features, original_features.detach())
    cos_loss = 1 - F.cosine_similarity(q_flat, o_flat, dim=1).mean()
    
    # 3. L1稀疏性损失
    l1_loss = torch.abs(quantized_features).mean() * 0.01
    
    quantization_loss = mse_loss + 0.5 * cos_loss + l1_loss
    
    total_loss = (self.config['distillation_weight'] * distillation_loss + 
                  self.config['quantization_loss_weight'] * quantization_loss)
    
    return total_loss
```

### 4. 模型转换与导出

```python
def convert_to_quantized_model(self, model: nn.Module) -> nn.Module:
    """将FakeQuantize模型转换为真实量化模型"""
    
    # 创建转换配置: QATConfig with step="convert"
    base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
    convert_config = QATConfig(base_config, step="convert")
    
    # 应用转换
    quantize_(model, convert_config)
    
    return model
```

## 🔧 高级功能

### 1. 超参数自动优化

我们提供了强大的超参数自动优化功能，支持多种优化方法和目标，帮助您找到最佳的量化配置。

#### 支持的优化方法

| 方法 | 描述 | 适用场景 | 推荐指数 |
|------|------|----------|----------|
| `bayesian` | 基于贝叶斯优化，智能探索参数空间 | 大多数场景，平衡效率和效果 | ⭐⭐⭐⭐⭐ |
| `grid_search` | 遍历所有参数组合 | 参数空间较小，需要全局最优解 | ⭐⭐⭐ |
| `random_search` | 随机采样参数组合 | 参数空间较大，快速找到可行解 | ⭐⭐⭐⭐ |

#### 优化目标

| 目标 | 描述 | 权重分配 |
|------|------|----------|
| `balanced` | 平衡精度、压缩比和速度 | 精度(0.6) + 压缩比(0.2) + 速度(0.2) |
| `accuracy` | 优先保持精度 | 精度(0.8) + 压缩比(0.1) + 速度(0.1) |
| `compression` | 优先提高压缩比 | 精度(0.3) + 压缩比(0.6) + 速度(0.1) |
| `speed` | 优先提高推理速度 | 精度(0.3) + 压缩比(0.1) + 速度(0.6) |

#### 基本用法

```bash
# 贝叶斯优化（推荐）
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 50 \
    --optimization_target balanced \
    --study_name my_study

# 网格搜索
python main.py \
    --mode optimization_study \
    --method grid_search \
    --param_config grid_config.json \
    --study_name grid_study

# 随机搜索
python main.py \
    --mode optimization_study \
    --method random \
    --n_trials 100 \
    --study_name random_study
```

#### 快速验证

使用`--dry_run`参数可以快速验证代码流程，不执行实际优化：

```bash
# 快速验证优化流程
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 5 \
    --optimization_target balanced \
    --study_name test_study \
    --dry_run \
    --batch_size 4 \
    --num_train 4 \
    --num_val 4
```

#### 高级配置

```bash
# 精度优先优化
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 100 \
    --optimization_target accuracy \
    --study_name accuracy_study \
    --batch_size 16 \
    --learning_rate 1e-4

# 压缩比优先优化
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 80 \
    --optimization_target compression \
    --study_name compression_study
```

#### 优化结果输出

优化完成后，会在输出目录生成以下文件：

| 文件 | 描述 |
|------|------|
| `study_config.json` | 研究配置信息 |
| `best_quantization_config.json` | 最佳量化配置 |
| `optimization_history.json` | 所有试验的历史记录 |
| `best_metrics.json` | 最佳配置的性能指标 |
| `optimization_report.json` | 详细的优化报告 |
| `optimization_visualization.png` | 优化过程可视化图表 |

#### 输出目录结构

```
output/
└── optimization_study_my_study/
    ├── study_config.json
    ├── best_quantization_config.json
    ├── optimization_history.json
    ├── best_metrics.json
    ├── optimization_report.json
    └── optimization_visualization.png
```

#### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `both` | 运行模式，必须设置为 `optimization_study` |
| `--method` | str | `bayesian` | 优化方法，可选值：`bayesian`, `grid_search`, `random_search` |
| `--n_trials` | int | 50 | 试验次数 |
| `--param_config` | str | None | 参数配置文件路径 |
| `--optimization_target` | str | `balanced` | 优化目标，可选值：`balanced`, `accuracy`, `compression`, `speed` |
| `--study_name` | str | None | 研究名称，用于创建输出目录 |
| `--dry_run` | bool | False | 是否只验证代码流程，不执行实际优化 |
| `--batch_size` | int | 64 | 批次大小 |
| `--num_train` | int | 95000 | 训练样本数 |
| `--num_val` | int | 5000 | 验证样本数 |
| `--learning_rate` | float | 3e-4 | 学习率 |

#### 使用示例

```bash
# 完整示例：使用贝叶斯优化进行50次试验，平衡目标
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 50 \
    --optimization_target balanced \
    --study_name ocr_bayesian_study \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_val 1000

# 快速验证示例
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 5 \
    --optimization_target balanced \
    --study_name quick_test \
    --dry_run \
    --batch_size 4 \
    --num_train 4 \
    --num_val 4
```

### 2. QAT训练优化详解

本项目实现了最新的QAT（量化感知训练）优化技术，基于torchao的新版API，提供更精细的控制和更高的精度。

#### 2.1 QAT API使用

```bash
# 启用QAT (默认启用，精度优先)
python main.py \
    --mode both \
    --enable_quantization \
    --use_modern_qat_api \
    --quantization_strategy int8_dyn_act_int4_weight \
    --weight_bits 4 \
    --activation_bits 8 \
    --qat_epochs 8 \
    --qat_learning_rate_multiplier 0.1
```

**API特性：**
- `FakeQuantizeConfig`配置系统：灵活的量化参数控制
- `PerToken`动态量化：针对OCR序列模型的精度优化
- `PerGroup`分组量化：平衡精度与性能（推荐groupsize=256）

#### 2.2 分层混合精度QAT

针对OCR模型的特点，对不同组件应用不同的量化策略：

```bash
# 启用分层混合精度QAT
python main.py \
    --mode both \
    --enable_quantization \
    --enable_layer_wise_qat \
    --qat_epochs 10 \
    --qat_learning_rate_multiplier 0.05
```

**分层策略：**

| 组件 | 权重量化 | 激活动态量化 | 说明 |
|------|----------|--------------|------|
| Backbone | INT4 (groupsize=256) | PerToken | 特征提取，平衡精度性能 |
| Neck | INT4 (groupsize=128) | PerToken | 特征融合，更高精度 |
| Decoder (CTC/AR) | INT8 | PerToken | 输出解码，精度优先 |

**关键代码实现：**

```python
def _apply_layer_wise_mixed_precision_qat(self) -> torch.nn.Module:
    """分层混合精度QAT - 针对不同层使用不同量化策略"""
    
    # Backbone: INT8动态激活 + INT4权重
    backbone_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int4, group_size=backbone_group_size, is_symmetric=True
    )
    
    # Neck: INT8动态激活 + INT4权重 (更小group_size)
    neck_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int4, group_size=neck_group_size, is_symmetric=True
    )
    
    # Decoder: INT8动态激活 + INT8权重 (精度优先)
    decoder_weight_config = IntxFakeQuantizeConfig(
        dtype=torch.int8, granularity="per_channel", is_symmetric=True
    )
    
    # 为每层应用相应配置
    for name, module in self.model.named_modules():
        if 'backbone' in name:
            config = QATConfig(activation_config=act_config, weight_config=backbone_weight_config)
        elif 'neck' in name:
            config = QATConfig(activation_config=act_config, weight_config=neck_weight_config)
        elif 'decoder' in name:
            config = QATConfig(activation_config=act_config, weight_config=decoder_weight_config)
        
        quantize_(module, config)
```

#### 2.3 分阶段QAT训练流程

```bash
# 分阶段QAT训练 (推荐用于高精度场景)
python main.py \
    --mode both \
    --enable_quantization \
    --use_modern_qat_api \
    --warmup_lr 3 \
    --qat_insert_epoch 3 \
    --qat_epochs 8 \
    --epochs 20
```

**训练阶段：**

1. **Warmup阶段** (epoch 0-3)：正常预训练，学习率warmup
2. **Pre-QAT阶段** (epoch 3)：准备插入FakeQuantize
3. **QAT微调阶段** (epoch 3-11)：冻结BN/LN层，低学习率微调量化参数
4. **Post-QAT阶段** (epoch 11+)：可选进一步微调

**关键代码实现：**

```python
class QATTrainingScheduler:
    """分阶段QAT训练调度器"""
    
    def maybe_insert_qat(self, epoch: int) -> bool:
        """延迟插入QAT模块"""
        if epoch >= self.config.get('qat_insert_epoch', 3) and not self.qat_inserted:
            # 执行实际的QAT插入
            self.qat_scheduler.apply_qat(self.model)
            self.qat_inserted = True
            return True
        return False
    
    def get_lr_multiplier(self, epoch: int) -> float:
        """根据阶段返回学习率乘数"""
        stage = self.get_current_stage(epoch)
        multipliers = {
            'warmup': 1.0,
            'qat_finetune': self.config.get('qat_learning_rate_multiplier', 0.1),
            'post_qat': 0.05,
        }
        return multipliers.get(stage, 1.0)
```

**特性：**
- 自动冻结BN/LN层：提高量化稳定性
- 动态学习率调整：QAT阶段自动降低至0.1x
- 阶段监控：输出当前训练阶段信息

#### 2.4 改进的量化感知损失

新版损失函数结合多种损失类型：

```python
# 量化损失组成 (自动配置，无需手动设置)
total_loss = (
    distillation_weight * KL_div_loss +           # 知识蒸馏
    quantization_loss_weight * (
        MSE_loss +                                # 数值接近
        0.5 * Cosine_similarity_loss +            # 方向一致
        L1_sparsity_loss                          # 稀疏性
    ) +
    fake_quant_loss_weight * FakeQuant_loss       # FakeQuant层损失
)
```

**关键代码实现：**

```python
def get_quantization_loss(self, quantized_features: torch.Tensor,
                         original_features: torch.Tensor) -> torch.Tensor:
    """计算量化损失用于知识蒸馏"""
    
    # 1. 特征蒸馏损失 (KL散度)
    temperature = self.config['temperature_distillation']
    distillation_loss = F.kl_div(
        F.log_softmax(quantized_features / temperature, dim=-1),
        F.softmax(original_features / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 2. 量化感知损失 - MSE + Cosine相似度
    mse_loss = F.mse_loss(quantized_features, original_features.detach())
    cos_loss = 1 - F.cosine_similarity(q_flat, o_flat, dim=1).mean()
    
    # 3. L1稀疏性损失
    l1_loss = torch.abs(quantized_features).mean() * 0.01
    
    quantization_loss = mse_loss + 0.5 * cos_loss + l1_loss
    
    total_loss = (self.config['distillation_weight'] * distillation_loss + 
                  self.config['quantization_loss_weight'] * quantization_loss)
    
    return total_loss
```

#### 2.5 模型导出与转换

训练完成后，自动转换为真实量化模型：

```python
# 导出流程 (自动执行)
# 1. 从FakeQuantize模型转换为真实量化模型
quantized_model = from_intx_quantization_aware_training(model)

# 2. 使用torch.export导出
exported_program = torch.export.export(quantized_model, example_input)
```

**关键代码实现：**

```python
def convert_to_quantized_model(self, model: nn.Module) -> nn.Module:
    """将FakeQuantize模型转换为真实量化模型"""
    
    # 创建转换配置: QATConfig with step="convert"
    base_config = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
    convert_config = QATConfig(base_config, step="convert")
    
    # 应用转换
    quantize_(model, convert_config)
    
    return model
```

**导出特性：**
- 支持`from_intx_quantization_aware_training`转换
- 保存完整的导出元数据

### 3. 渐进式优化

```python
from quantization import AdaptiveHyperparameterOptimizer

# 创建优化器
optimizer = AdaptiveHyperparameterOptimizer(model, dataloader, device)

# 渐进式优化
results = optimizer.progressive_optimization(
    base_config=config,
    n_stages=3
)

# 自适应优化
result = optimizer.optimize(
    base_config=config,
    optimization_budget='high',
    target_metric='balanced'
)
```

### 4. 多目标优化

```python
from quantization import QuantizationObjective

# 自定义目标权重
objective = QuantizationObjective(
    original_model=model,
    dataloader=dataloader,
    device=device,
    accuracy_weight=0.7,      # 精度权重
    compression_weight=0.2,   # 压缩权重
    speed_weight=0.1,         # 速度权重
    min_accuracy_threshold=0.95  # 最低精度要求
)
```

### 5. 硬件感知优化

```bash
# CPU优化
python main.py \
    --mode both \
    --hardware_target cpu \
    --quantization_granularity per_tensor \
    --symmetric_quantization true

# GPU优化
python main.py \
    --mode both \
    --hardware_target gpu \
    --quantization_granularity per_channel \
    --enable_cuda_graphs true

# 移动端优化
python main.py \
    --mode both \
    --hardware_target mobile \
    --weight_bits 4 \
    --activation_bits 4 \
    --memory_efficient true
```

## 模型剪枝功能详解

### PyTorch 剪枝 API 标准

#### 可用的剪枝方法

| 方法 | 类型 | 说明 |
|------|------|------|
| `prune.l1_unstructured` | 非结构化 | 基于L1范数的非结构化剪枝 |
| `prune.ln_structured` | 结构化 | 基于Ln范数的结构化剪枝 |
| `prune.random_unstructured` | 非结构化 | 随机非结构化剪枝 |
| `prune.random_structured` | 结构化 | 随机结构化剪枝 |
| `prune.global_unstructured` | 全局非结构化 | 全局范围的非结构化剪枝 |
| `prune.remove` | 工具 | 使剪枝永久化，移除mask |

#### 方法签名

```python
# l1_unstructured 签名
prune.l1_unstructured(
    module,           # 目标模块
    name='weight',    # 参数名称
    amount=0.3,       # 剪枝比例(0-1)或绝对数量
    importance_scores=None  # 可选的重要性分数
)

# ln_structured 签名
prune.ln_structured(
    module,           # 目标模块
    name='weight',    # 参数名称
    amount=0.3,       # 剪枝比例
    n=1,              # 范数阶数(1或2)
    dim=0,            # 剪枝维度(0=输出通道, 1=输入通道)
    importance_scores=None
)
```

### 剪枝实现架构

#### 核心组件

当前实现在 `quantization.py` 中，包含以下主要组件：

1. **PruningConfig**: 剪枝配置数据类
2. **PruningManager**: 剪枝管理器，核心逻辑
3. **PruningQuantizationScheduler**: 剪枝与量化的协同调度器

#### 剪枝策略实现

| 策略 | 实现方法 | 状态 |
|------|----------|------|
| `l1_unstructured` | `prune.l1_unstructured` | ✅ 可用 |
| `l1_structured` | `prune.ln_structured(n=1)` | ✅ 可用 |
| `l2_structured` | `prune.ln_structured(n=2)` | ✅ 可用 |
| `ln_structured` | `prune.ln_structured` | ✅ 可用 |
| `global_unstructured` | `prune.global_unstructured` | ✅ 可用 |

### 剪枝可用方法

`PruningManager` 类提供以下方法：

| 方法名 | 功能 | 代码位置 |
|--------|------|----------|
| `apply_pruning` | 标准逐层剪枝 | quantization.py |
| `apply_global_pruning` | 全局非结构化剪枝 - 获得更好的整体稀疏性 | ~805行 |
| `make_pruning_permanent` | 使剪枝永久化，移除mask | quantization.py |
| `validate_pruning_with_rollback` | 验证剪枝效果，如不满足条件则自动回滚 | ~930行 |
| `compress_model_structurally` | 结构化压缩 - 真正删除被剪枝的通道/神经元 | ~990行 |
| `visualize_pruning` | 生成剪枝效果可视化图表 | ~1075行 |
| `_is_weight_pruned` | 鲁棒的剪枝检测 - 基于权重分布 | ~735行 |
| `calculate_structured_pruning_ratio` | 计算通道/神经元级别剪枝比例 | ~757行 |
| `_find_associated_bn` | 查找与卷积层关联的BatchNorm层 | ~860行 |
| `_adjust_bn_after_structured_pruning` | 结构化剪枝后调整BatchNorm层参数 | ~890行 |
| `calculate_pruning_ratio` | 计算整体剪枝比例 | quantization.py |
| `get_structured_pruning_stats` | 获取结构化剪枝统计信息 | quantization.py |
| `get_detailed_pruning_stats` | 获取详细剪枝统计信息 | quantization.py |

### 关键代码实现

#### 1. 鲁棒的剪枝检测

```python
def _is_weight_pruned(self, weight: torch.Tensor) -> torch.Tensor:
    """鲁棒的剪枝检测 - 基于权重分布"""
    exact_zero = (weight == 0)
    weight_std = weight.std()
    if weight_std > 0:
        relative_threshold = weight_std * 1e-6
        near_zero = torch.abs(weight) < relative_threshold
    else:
        near_zero = torch.zeros_like(weight, dtype=torch.bool)
    return exact_zero | near_zero
```

#### 2. 全局剪枝

```python
def apply_global_pruning(self, epoch: int, current_acc: float, best_acc: float) -> bool:
    """全局非结构化剪枝 - 获得更好的整体稀疏性"""
    parameters_to_prune = []
    for name, module in self.model.named_modules():
        if self._is_prunable_module(module, name):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=self.config.pruning_ratio
    )
```

#### 3. BatchNorm调整

```python
def _adjust_bn_after_structured_pruning(self, module: nn.Module, module_name: str,
                                        pruned_indices: torch.Tensor):
    """结构化剪枝后调整 BatchNorm 层"""
    bn_result = self._find_associated_bn(module, module_name)
    if bn_result is None:
        return
    
    bn_name, bn_module = bn_result
    keep_mask = torch.ones(bn_module.num_features, dtype=torch.bool)
    keep_mask[pruned_indices] = False
    
    with torch.no_grad():
        if bn_module.weight is not None:
            bn_module.weight.data = bn_module.weight.data[keep_mask]
        if bn_module.bias is not None:
            bn_module.bias.data = bn_module.bias.data[keep_mask]
        if hasattr(bn_module, 'running_mean'):
            bn_module.running_mean = bn_module.running_mean[keep_mask]
        if hasattr(bn_module, 'running_var'):
            bn_module.running_var = bn_module.running_var[keep_mask]
        bn_module.num_features = keep_mask.sum().item()
```

#### 4. 验证与回滚

```python
def validate_pruning_with_rollback(self, val_dataloader, max_acc_drop: float = None) -> bool:
    """验证剪枝效果，如不满足条件则自动回滚"""
    state_before = copy.deepcopy(self.model.state_dict())
    self.make_pruning_permanent()
    
    current_acc = self._quick_evaluate(val_dataloader, num_batches=20, ...)
    original_acc = ...
    
    acc_drop = original_acc - current_acc
    if acc_drop > max_acc_drop:
        self.model.load_state_dict(state_before)
        self._clear_all_pruning_masks()
        return False
    return True
```

#### 5. 结构化压缩

```python
def compress_model_structurally(self, val_dataloader=None, max_acc_drop: float = 0.02) -> nn.Module:
    """结构化压缩 - 删除被剪枝的通道/神经元并重建模型"""
    # 1. 识别被完全剪枝的通道
    # 2. 创建新的压缩后的层
    # 3. 复制保留的权重
    # 4. 调整后续层的输入维度
    # 5. 调整关联的BatchNorm层
    # 6. 验证压缩效果
```

### 剪枝命令行参数

#### 基础剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_pruning` | flag | False | 启用模型剪枝 |
| `--pruning_strategy` | str | l1_unstructured | 剪枝策略 |
| `--pruning_ratio` | float | 0.3 | 全局剪枝比例 |
| `--pruning_epoch` | int | 20 | 剪枝执行的epoch |

#### 高级剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--apply_pruning_during_training` | flag | False | 训练期间实际应用剪枝 |
| `--validate_pruning` | flag | False | 剪枝后验证精度，如下降过大则回滚 |
| `--visualize_pruning` | flag | False | 生成剪枝可视化图表 |
| `--structural_compression` | flag | False | 部署时执行结构化压缩 |

#### 分层剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--backbone_pruning_ratio` | float | - | Backbone层剪枝比例 |
| `--neck_pruning_ratio` | float | - | Neck层剪枝比例 |
| `--decoder_pruning_ratio` | float | - | Decoder层剪枝比例 |

#### 剪枝策略选项

- `l1_unstructured`: L1非结构化剪枝（默认）
- `l1_structured`: L1结构化剪枝
- `l2_structured`: L2结构化剪枝
- `ln_structured`: Ln结构化剪枝
- `global_unstructured`: 全局非结构化剪枝（推荐）

### 剪枝使用场景

#### 场景1: 基础剪枝（部署时应用）

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20
```

**流程**:
1. 训练期间记录剪枝候选节点
2. 部署时应用剪枝并永久化
3. 生成剪枝总结报告

#### 场景2: 训练期间应用剪枝（推荐）

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20 \
    --apply_pruning_during_training \
    --validate_pruning \
    --visualize_pruning
```

**流程**:
1. Epoch 20 时应用全局剪枝
2. 自动验证剪枝效果
3. 如精度下降>1%则自动回滚
4. 生成剪枝可视化图表
5. 继续微调训练

#### 场景3: 移动端部署（结构化压缩）

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_structured \
    --pruning_ratio 0.4 \
    --pruning_epoch 25 \
    --apply_pruning_during_training \
    --structural_compression
```

**流程**:
1. 训练期间应用结构化剪枝
2. 部署时执行结构化压缩
3. 真正删除被剪枝的通道
4. 减少模型大小和计算量

#### 场景4: 分层剪枝比例

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --backbone_pruning_ratio 0.2 \
    --neck_pruning_ratio 0.3 \
    --decoder_pruning_ratio 0.1
```

### 剪枝完整示例

#### 示例1: 高精度优先

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 60 \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.2 \
    --pruning_epoch 30 \
    --min_acc_drop 0.005 \
    --finetune_epochs 10 \
    --apply_pruning_during_training \
    --validate_pruning \
    --visualize_pruning
```

#### 示例2: 高压缩比

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 80 \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.5 \
    --pruning_epoch 20 \
    --finetune_epochs 15 \
    --apply_pruning_during_training \
    --structural_compression \
    --visualize_pruning
```

#### 示例3: 仅评估已有模型

```bash
python main.py \
    --mode evaluate \
    --checkpoint output/models/ocr_best_em.pth \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --visualize_pruning
```

### 训练流程中的剪枝

#### 标准流程（部署时应用）

```
Epoch 0-19:  正常训练
Epoch 20:    记录剪枝候选节点
Epoch 21-60: 继续训练
部署阶段:    应用剪枝并导出模型
```

#### 高级流程（训练期间应用）

```
Epoch 0-19:   正常训练
Epoch 20:     应用剪枝 → 验证 → （如失败则回滚）
Epoch 21-30:  剪枝后微调（学习率×0.1）
Epoch 31-60:  正常训练
部署阶段:     永久化剪枝并导出模型
```

### 剪枝输出文件

#### 剪枝相关输出

| 文件 | 说明 |
|------|------|
| `output/models/pruning_candidates.json` | 剪枝候选节点信息 |
| `output/visualizations/pruning_epoch_{N}.png` | 训练期间剪枝可视化 |
| `output/visualizations/pruning_final.png` | 最终剪枝可视化 |
| `output/visualizations/pruning_deployment.png` | 部署时剪枝可视化 |

#### 可视化内容

剪枝可视化图表包含4个子图：
1. **层剪枝比例条形图**: 显示各层剪枝比例
2. **权重分布直方图**: 剪枝前后权重分布对比
3. **结构化剪枝通道分布**: 显示被剪枝的通道
4. **参数量对比**: 剪枝前后参数数量对比

### 剪枝性能对比

| 配置 | 参数量 | 精度下降 | 推荐场景 |
|------|--------|----------|----------|
| 无剪枝 | 100% | 0% | 基准 |
| L1非结构化 30% | 70% | ~0.5% | 通用 |
| 全局非结构化 30% | 70% | ~0.3% | 推荐 |
| 结构化 30% | 70% | ~1% | 移动端 |
| 结构化 50% | 50% | ~2-3% | 高压缩 |

### 剪枝注意事项

#### 1. 剪枝时机

- **推荐**: 在模型收敛后执行剪枝（如Epoch 20-30）
- **避免**: 训练初期剪枝，可能导致模型无法恢复

#### 2. 精度保护

- 使用 `--validate_pruning` 自动保护精度
- 设置合理的 `--min_acc_drop`（默认1%）
- 如剪枝后精度下降过大，会自动回滚

#### 3. 学习率调整

- 剪枝后微调阶段自动使用0.1倍学习率
- 可通过 `--finetune_epochs` 设置微调轮数

#### 4. 结构化压缩

- 仅适用于结构化剪枝策略
- 会真正删除通道，减少模型大小
- 建议在移动端部署时使用

#### 5. 编码兼容性

- Windows 控制台可能遇到编码问题
- 所有警告信息使用 `[WARNING]` 前缀替代特殊字符
- Linux/macOS 环境完全支持

#### 6. 依赖要求

- Python >= 3.8
- PyTorch >= 2.0
- matplotlib（可视化功能）

### 剪枝故障排除

#### 问题1: 剪枝未应用

**可能原因**:
- 当前精度低于最佳精度的 `(1 - min_acc_drop)`

**解决**:
```bash
# 降低精度阈值要求
--min_acc_drop 0.05  # 允许5%的精度下降
```

#### 问题2: 剪枝后精度下降过大

**解决**:
```bash
# 降低剪枝比例
--pruning_ratio 0.2

# 增加微调轮数
--finetune_epochs 15

# 启用验证回滚
--validate_pruning
```

#### 问题3: 全局剪枝失败

**可能原因**:
- 模型中没有可剪枝的层

**解决**:
```bash
# 检查剪枝层配置
--pruning_layers backbone neck decoder
```

### 剪枝功能清单

| 功能 | 状态 | 说明 |
|------|------|------|
| 基础剪枝 | ✅ | 标准逐层L1非结构化剪枝 |
| 全局剪枝 | ✅ | 全局非结构化剪枝，获得更好稀疏性 |
| 结构化剪枝 | ✅ | L1/L2/Ln结构化剪枝 |
| 训练期间应用 | ✅ | 训练时应用剪枝并微调 |
| 验证回滚 | ✅ | 自动验证精度，下降过大则回滚 |
| 结构化压缩 | ✅ | 真正删除被剪枝通道 |
| 可视化 | ✅ | 生成剪枝效果图表 |
| 详细统计 | ✅ | 多维度剪枝统计信息 |
| BatchNorm调整 | ✅ | 结构化剪枝后自动调整BN层 |
| 剪枝检测 | ✅ | 基于权重分布的鲁棒检测 |

## 命令行参数

### 基础量化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_quantization` | flag | False | 启用量化训练 |
| `--quantization_mode` | str | qat | 量化模式: qat/ptq/both |
| `--quantization_strategy` | str | int8_dyn_act_int4_weight | 量化策略 |
| `--qat_epochs` | int | 5 | QAT训练轮数 |
| `--weight_bits` | int | 4 | 权重量化位数 |
| `--activation_bits` | int | 8 | 激活量化位数 |

### 高级量化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_layer_wise_qat` | flag | False | 启用分层混合精度QAT |
| `--qat_insert_epoch` | int | 3 | QAT模块插入的epoch |

### 剪枝参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_pruning` | flag | False | 启用模型剪枝 |
| `--pruning_strategy` | str | l1_unstructured | 剪枝策略 |
| `--pruning_ratio` | float | 0.3 | 全局剪枝比例 |
| `--pruning_epoch` | int | 20 | 剪枝执行的epoch |
| `--apply_pruning_during_training` | flag | False | 训练期间实际应用剪枝 |
| `--validate_pruning` | flag | False | 剪枝后验证精度，如下降过大则回滚 |
| `--visualize_pruning` | flag | False | 生成剪枝可视化图表 |
| `--structural_compression` | flag | False | 部署时执行结构化压缩 |
| `--backbone_pruning_ratio` | float | - | Backbone层剪枝比例 |
| `--neck_pruning_ratio` | float | - | Neck层剪枝比例 |
| `--decoder_pruning_ratio` | float | - | Decoder层剪枝比例 |

### 优化研究参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--study_name` | str | None | 优化研究名称 |
| `--method` | str | bayesian | 优化方法 |
| `--n_trials` | int | 50 | 优化试验次数 |
| `--optimization_target` | str | balanced | 优化目标 |
| `--dry_run` | flag | False | 验证模式（不执行实际优化） |

## 📋 使用场景

### 场景1: 基础QAT训练

```bash
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --quantization_strategy int8_dyn_act_int4_weight \
    --qat_epochs 5
```

**流程**:
1. 预热阶段 (Epoch 0-2): 正常精度训练
2. QAT插入 (Epoch 3): 插入量化模块
3. QAT微调 (Epoch 3-7): 量化感知训练
4. 评估与导出

---

### 场景2: 分层混合精度QAT

```bash
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode qat \
    --enable_layer_wise_qat \
    --qat_insert_epoch 3 \
    --qat_epochs 8
```

**流程**:
1. 启用分层混合精度
2. Backbone: INT8激活 + INT4权重
3. Neck: INT8激活 + INT4权重（更小group_size）
4. Decoder: INT8激活 + INT8权重（精度优先）

---

### 场景3: 训练后量化 (PTQ)

```bash
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode ptq \
    --quantization_strategy int8_dyn_act_int4_weight
```

**流程**:
1. 加载预训练模型
2. 校准量化参数（使用100批次数据）
3. 应用量化
4. 评估量化效果

---

### 场景4: QAT + PTQ 组合

```bash
python main.py \
    --mode both \
    --enable_quantization \
    --quantization_mode both \
    --quantization_strategy int8_dyn_act_int4_weight \
    --qat_epochs 5
```

---

### 场景5: 量化超参数优化研究

```bash
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 50 \
    --optimization_target balanced \
    --study_name ocr_quantization_study
```

**流程**:
1. 创建Optuna优化研究
2. 自动搜索最优量化配置
3. 评估每次试验的效果
4. 输出最佳配置和优化报告

---

### 场景6: 快速验证 (Dry Run)

```bash
python main.py \
    --mode optimization_study \
    --method bayesian \
    --n_trials 5 \
    --dry_run \
    --batch_size 4 \
    --num_train 4 \
    --num_val 4
```

---

### 场景7: 基础剪枝

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20
```

**流程**:
1. 训练期间记录剪枝候选节点
2. 部署时应用剪枝并永久化
3. 生成剪枝总结报告

---

### 场景8: 训练期间应用剪枝（推荐）

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.3 \
    --pruning_epoch 20 \
    --apply_pruning_during_training \
    --validate_pruning \
    --visualize_pruning
```

**流程**:
1. Epoch 20 时应用全局剪枝
2. 自动验证剪枝效果
3. 如精度下降>1%则自动回滚
4. 生成剪枝可视化图表
5. 继续微调训练

---

### 场景9: 移动端部署（结构化压缩）

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_structured \
    --pruning_ratio 0.4 \
    --pruning_epoch 25 \
    --apply_pruning_during_training \
    --structural_compression
```

**流程**:
1. 训练期间应用结构化剪枝
2. 部署时执行结构化压缩
3. 真正删除被剪枝的通道
4. 减少模型大小和计算量

---

### 场景10: 分层剪枝比例

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --backbone_pruning_ratio 0.2 \
    --neck_pruning_ratio 0.3 \
    --decoder_pruning_ratio 0.1
```

---

### 场景11: 剪枝+量化组合优化

```bash
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_epoch 25 \
    --finetune_epochs 15 \
    --quantization_strategy int8_dyn_act_int4_weight \
    --qat_epochs 8 \
    --distillation_weight 0.5
```

**流程**:
1. Epoch 0-19:  正常训练
2. Epoch 20:    应用剪枝
3. Epoch 21-30: 剪枝后微调
4. Epoch 31-33: 预热阶段
5. Epoch 34:    QAT插入
6. Epoch 34-38: QAT微调
7. 部署阶段:    永久化剪枝 + 量化模型导出

## 完整示例

### 示例1: 高精度优先配置

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 60 \
    --enable_quantization \
    --quantization_mode qat \
    --quantization_strategy int8_dynamic_activation_int8_weight \
    --weight_bits 8 \
    --activation_bits 8 \
    --qat_epochs 8 \
    --qat_learning_rate_multiplier 0.05 \
    --quantization_loss_weight 0.02 \
    --temperature_distillation 8.0 \
    --distillation_weight 0.5
```

### 示例2: 高压缩比配置

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 80 \
    --enable_quantization \
    --quantization_mode qat \
    --quantization_strategy int4_weight_only \
    --weight_bits 4 \
    --qat_epochs 10 \
    --enable_layer_wise_qat
```

### 示例3: 移动端部署优化

```bash
python main.py \
    --mode deployment \
    --checkpoint output/models/ocr_best_em.pth \
    --enable_quantization \
    --quantization_strategy int4_weight_only \
    --deployment_target mobile_cpu
```

### 示例4: 高精度优先剪枝

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 60 \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.2 \
    --pruning_epoch 30 \
    --min_acc_drop 0.005 \
    --finetune_epochs 10 \
    --apply_pruning_during_training \
    --validate_pruning \
    --visualize_pruning
```

### 示例5: 高压缩比剪枝

```bash
python main.py \
    --mode both \
    --model_name viptr2 \
    --epochs 80 \
    --enable_pruning \
    --pruning_strategy global_unstructured \
    --pruning_ratio 0.5 \
    --pruning_epoch 20 \
    --finetune_epochs 15 \
    --apply_pruning_during_training \
    --structural_compression \
    --visualize_pruning
```

### 示例6: 仅评估已有剪枝模型

```bash
python main.py \
    --mode evaluate \
    --checkpoint output/models/ocr_best_em.pth \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.3 \
    --visualize_pruning
```

## 🔄 训练流程

### 标准QAT流程

```
Epoch 0-2:   预热阶段 - 正常精度训练
Epoch 3:     QAT插入 - 插入FakeQuantize模块
Epoch 3-7:   QAT微调 - 量化感知训练，冻结BN/LN
部署阶段:    转换为真实量化模型并导出
```

### 分层混合精度QAT流程

```
Epoch 0-2:   预热阶段 - 正常精度训练
Epoch 3:     QAT插入 - 为不同层应用不同精度配置
Epoch 3-10:  分层QAT微调 - Backbone/Neck/Decoder不同学习率
部署阶段:    转换为真实量化模型并导出
```

### 标准剪枝流程（部署时应用）

```
Epoch 0-19:  正常训练
Epoch 20:    记录剪枝候选节点
Epoch 21-60: 继续训练
部署阶段:    应用剪枝并导出模型
```

### 高级剪枝流程（训练期间应用）

```
Epoch 0-19:   正常训练
Epoch 20:     应用剪枝 → 验证 → （如失败则回滚）
Epoch 21-30:  剪枝后微调（学习率×0.1）
Epoch 31-60:  正常训练
部署阶段:     永久化剪枝并导出模型
```

### 剪枝 + 量化组合流程

```
Epoch 0-19:  正常训练
Epoch 20:    应用剪枝
Epoch 21-30: 剪枝后微调
Epoch 31-33: 预热阶段
Epoch 34:    QAT插入
Epoch 34-38: QAT微调
部署阶段:    永久化剪枝 + 量化模型导出
```

## 📁 输出文件

### 量化相关输出

| 文件 | 说明 |
|------|------|
| `output/models/quantized_ocr_model.pth` | 量化模型文件 |
| `output/models/quantized_model_exported.pt2` | torch.export导出格式 |
| `output/models/quantized_model.onnx` | ONNX格式模型 |
| `output/reports/quantization_report.json` | 量化评估报告 |
| `output/visualizations/quantization_results.png` | 量化效果可视化 |

### 剪枝相关输出

| 文件 | 说明 |
|------|------|
| `output/models/pruning_candidates.json` | 剪枝候选节点信息 |
| `output/visualizations/pruning_epoch_{N}.png` | 训练期间剪枝可视化 |
| `output/visualizations/pruning_final.png` | 最终剪枝可视化 |
| `output/visualizations/pruning_deployment.png` | 部署时剪枝可视化 |

### 优化研究输出

| 文件 | 说明 |
|------|------|
| `output/optimization_study_{name}/study_config.json` | 研究配置 |
| `output/optimization_study_{name}/best_quantization_config.json` | 最佳配置 |
| `output/optimization_study_{name}/optimization_history.json` | 优化历史 |
| `output/optimization_study_{name}/optimization_report.json` | 优化报告 |
| `output/optimization_study_{name}/optimization_visualization.png` | 可视化图表 |

### 可视化内容

量化可视化图表包含以下子图：
1. **精度对比**: 原始模型 vs 量化模型
2. **模型大小对比**: 压缩比可视化
3. **推理速度对比**: 加速比可视化
4. **内存使用对比**: 内存减少比例

剪枝可视化图表包含4个子图：
1. **层剪枝比例条形图**: 显示各层剪枝比例
2. **权重分布直方图**: 剪枝前后权重分布对比
3. **结构化剪枝通道分布**: 显示被剪枝的通道
4. **参数量对比**: 剪枝前后参数数量对比

## 📈 性能基准

### SVTRv2-Tiny 模型量化结果

| 配置 | 原始模型 | INT8+INT4 QAT | INT4 Only | 改善 |
|------|----------|---------------|-----------|------|
| 模型大小 | 15.2 MB | 4.8 MB | 3.8 MB | **3.2x-4x压缩** |
| 推理时间 | 12.5 ms | 8.3 ms | 7.5 ms | **1.5-1.7x加速** |
| 内存使用 | 128 MB | 89 MB | 76 MB | **30-40%减少** |
| CER | 2.1% | 2.3% | 2.8% | **0.2-0.7%增加** |
| 准确率 | 97.9% | 97.7% | 97.2% | **0.2-0.7%下降** |

### SVTRv2-Tiny模型剪枝结果

| 指标 | 原始模型 | 剪枝模型 | 改善 |
|------|----------|----------|------|
| 模型大小 | 15.2 MB | 9.8 MB | **1.55x压缩** |
| 推理时间 | 12.5 ms | 10.2 ms | **1.22x加速** |
| 内存使用 | 128 MB | 105 MB | **17.9%减少** |
| CER | 2.1% | 2.2% | **0.1%增加** |
| 准确率 | 97.9% | 97.8% | **0.1%下降** |

### 剪枝+量化组合优化结果

| 指标 | 原始模型 | 剪枝+量化模型 | 改善 |
|------|----------|---------------|------|
| 模型大小 | 15.2 MB | 3.2 MB | **4.75x压缩** |
| 推理时间 | 12.5 ms | 6.8 ms | **1.84x加速** |
| 内存使用 | 128 MB | 72 MB | **43.8%减少** |
| CER | 2.1% | 2.4% | **0.3%增加** |
| 准确率 | 97.9% | 97.6% | **0.3%下降** |

### 分层混合精度QAT结果

使用`ComposableQATQuantizer`分层策略：

| 组件 | 量化策略 | CER变化 | 说明 |
|------|----------|---------|------|
| Backbone | INT4 (gs=256) | -0.02% | 精度提升 |
| Neck | INT4 (gs=128) | -0.05% | 更高精度 |
| Decoder | INT8 | -0.08% | 精度优先 |
| **整体** | 混合精度 | **2.05%** | **优于统一INT4** |

### 不同硬件平台表现

#### CPU平台
- 推理速度提升：1.3-1.8x
- 内存使用减少：25-40%
- 精度保持：>97%

#### GPU平台
- 推理速度提升：1.5-2.2x
- 内存使用减少：30-50%
- 精度保持：>97.5%

#### 移动端
- 模型大小压缩：3-8x
- 推理速度提升：1.2-1.6x
- 电池续航改善：15-30%

## ⚙️ 配置详解

### 核心量化配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用量化 |
| `strategy` | str | int8_dyn_act_int4_weight | 量化策略 |
| `quantization_aware_training` | bool | True | 启用QAT |
| `post_training_quantization` | bool | False | 启用PTQ |
| `weight_bits` | int | 4 | 权重量化位数 |
| `activation_bits` | int | 8 | 激活量化位数 |
| `qat_epochs` | int | 5 | QAT训练轮数 |
| `qat_learning_rate_multiplier` | float | 0.1 | QAT学习率倍数 |
| `quantization_loss_weight` | float | 0.01 | 量化损失权重 |
| `temperature_distillation` | float | 4.0 | 蒸馏温度 |
| `distillation_weight` | float | 0.3 | 蒸馏权重 |

### 模型剪枝配置

| 参数 | 说明 | 推荐值 | 范围 |
|------|------|--------|------|
| `enable_pruning` | 是否启用剪枝 | false | {true, false} |
| `pruning_strategy` | 剪枝策略 | l1_unstructured | {l1_unstructured, l1_structured, ln_structured} |
| `pruning_ratio` | 全局剪枝比例 | 0.3 | [0.1, 0.8] |
| `pruning_epoch` | 剪枝执行的epoch | 20 | [10, 50] |
| `finetune_epochs` | 剪枝后的微调轮数 | 10 | [5, 20] |
| `min_acc_drop` | 允许的最大精度下降 | 0.01 | [0.001, 0.05] |
| `backbone_pruning_ratio` | Backbone剪枝比例 | 0.2 | [0.1, 0.5] |
| `neck_pruning_ratio` | Neck剪枝比例 | 0.3 | [0.1, 0.6] |
| `decoder_pruning_ratio` | Decoder剪枝比例 | 0.1 | [0.05, 0.3] |
| `prune_criteria` | 剪枝标准 | l1 | {l1, l2, grad} |
| `apply_pruning_during_training` | 训练期间应用剪枝 | false | {true, false} |
| `validate_pruning` | 剪枝后验证精度 | false | {true, false} |
| `visualize_pruning` | 生成剪枝可视化 | false | {true, false} |
| `structural_compression` | 执行结构化压缩 | false | {true, false} |

### 观察器配置

| 参数 | 说明 | 选项 |
|------|------|------|
| `observer_type` | 观察器类型 | moving_average, min_max, percentile, histogram |
| `observer_momentum` | 观察器动量 | 0.01-0.2 |
| `quantization_granularity` | 量化粒度 | per_tensor, per_channel, per_group |

### 训练策略配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `calibration_batches` | 校准批次数量 | 100 |
| `mixed_precision` | 混合精度训练 | true |
| `memory_efficient` | 内存高效模式 | true |
| `compile_model` | 模型编译优化 | false |

### QAT优化配置

| 参数 | 说明 | 推荐值 | 适用场景 |
|------|------|--------|----------|
| `use_modern_qat_api` | 使用QAT API (FakeQuantizeConfig) | `true` | 精度优先场景 |
| `enable_layer_wise_qat` | 启用分层混合精度QAT | `false` | 对精度要求极高的OCR任务 |
| `qat_insert_epoch` | QAT FakeQuantize插入的epoch | `3` | 与warmup_lr保持一致 |
| `fake_quant_loss_weight` | FakeQuantize层损失权重 | `0.001` | 细粒度量化优化 |

#### 粒度配置对比

| 粒度类型 | 配置方式 | 精度 | 性能 | 适用场景 |
|----------|----------|------|------|----------|
| `PerTensor` | 整体量化 | 一般 | 最高 | 批处理场景 |
| `PerAxis` | 按通道量化 | 较好 | 较高 | 通用场景 |
| `PerGroup(256)` | 分组量化 | 好 | 高 | 精度-性能平衡 |
| `PerGroup(128)` | 精细分组 | 更好 | 中等 | 精度优先 |
| `PerToken` | 每token量化 | 最好 | 动态 | 序列模型、OCR |

## 📈 典型结果

### SVTRv2-Tiny模型量化结果

| 配置 | 原始模型 | INT8+INT4 QAT | INT4 Only | 改善 |
|------|----------|---------------|-----------|------|
| 模型大小 | 15.2 MB | 4.8 MB | 3.8 MB | **3.2x-4x压缩** |
| 推理时间 | 12.5 ms | 8.3 ms | 7.5 ms | **1.5-1.7x加速** |
| 内存使用 | 128 MB | 89 MB | 76 MB | **30-40%减少** |
| CER | 2.1% | 2.3% | 2.8% | **0.2-0.7%增加** |
| 准确率 | 97.9% | 97.7% | 97.2% | **0.2-0.7%下降** |

### SVTRv2-Tiny模型剪枝结果

| 指标 | 原始模型 | 剪枝模型 | 改善 |
|------|----------|----------|------|
| 模型大小 | 15.2 MB | 9.8 MB | **1.55x压缩** |
| 推理时间 | 12.5 ms | 10.2 ms | **1.22x加速** |
| 内存使用 | 128 MB | 105 MB | **17.9%减少** |
| CER | 2.1% | 2.2% | **0.1%增加** |
| 准确率 | 97.9% | 97.8% | **0.1%下降** |

### 剪枝+量化组合优化结果

| 指标 | 原始模型 | 剪枝+量化模型 | 改善 |
|------|----------|---------------|------|
| 模型大小 | 15.2 MB | 3.2 MB | **4.75x压缩** |
| 推理时间 | 12.5 ms | 6.8 ms | **1.84x加速** |
| 内存使用 | 128 MB | 72 MB | **43.8%减少** |
| CER | 2.1% | 2.4% | **0.3%增加** |
| 准确率 | 97.9% | 97.6% | **0.3%下降** |

### 分层混合精度QAT结果

使用`ComposableQATQuantizer`分层策略：

| 组件 | 量化策略 | CER变化 | 说明 |
|------|----------|---------|------|
| Backbone | INT4 (gs=256) | -0.02% | 精度提升 |
| Neck | INT4 (gs=128) | -0.05% | 更高精度 |
| Decoder | INT8 | -0.08% | 精度优先 |
| **整体** | 混合精度 | **2.05%** | **优于统一INT4** |

### 不同硬件平台表现

#### CPU平台
- 推理速度提升：1.3-1.8x
- 内存使用减少：25-40%
- 精度保持：>97%

#### GPU平台
- 推理速度提升：1.5-2.2x
- 内存使用减少：30-50%
- 精度保持：>97.5%

#### 移动端
- 模型大小压缩：3-8x
- 推理速度提升：1.2-1.6x
- 电池续航改善：15-30%

## 🚨 注意事项

### 1. QAT训练轮数

- **推荐**: 不少于5轮，确保量化参数充分优化
- **高精度需求**: 8-10轮
- **快速实验**: 3轮

### 2. 学习率调整

- QAT阶段使用较低学习率（默认0.1倍）
- 分层QAT可为不同层设置不同学习率
- 使用`--qat_learning_rate_multiplier`调整
- 剪枝后微调阶段自动使用0.1倍学习率

### 3. 校准数据

- PTQ需要代表性的校准数据
- 推荐100+批次
- 数据分布应与实际应用场景一致

### 4. 硬件兼容性

- 确保目标硬件支持所选量化格式
- CPU: 推荐使用对称量化
- GPU: 支持更细粒度的通道级量化
- 移动端: 推荐使用INT4权重量化

### 5. 精度保护

```bash
# 高精度优先
--quantization_strategy int8_dynamic_activation_int8_weight
--qat_epochs 10
--quantization_loss_weight 0.02
--distillation_weight 0.5

# 平衡配置（推荐）
--quantization_strategy int8_dyn_act_int4_weight
--qat_epochs 5
--quantization_loss_weight 0.01
--distillation_weight 0.3

# 高压缩比
--quantization_strategy int4_weight_only
--qat_epochs 8
--enable_layer_wise_qat
```

### 6. 内存优化

```bash
# 内存不足时
--batch_size 2
--calibration_batches 50
```

### 7. 精度保持（补充）

1. **知识蒸馏**：适当提高蒸馏权重（0.3-0.5）以保持精度
2. **剪枝比例**：从低比例开始尝试，逐步提高，避免一次性剪枝过多
3. **剪枝时机**：在模型收敛后进行剪枝，通常在训练20-30轮后

### 8. QAT训练优化注意事项

#### QAT API使用建议
1. **默认启用**：`use_modern_qat_api=true` 提供更好的精度，建议默认使用
2. **分层混合精度**：OCR任务对Decoder精度敏感，建议启用 `enable_layer_wise_qat`
3. **QAT插入时机**：`qat_insert_epoch` 建议与 `warmup_lr` 保持一致（默认3）
4. **BN/LN冻结**：QAT微调阶段自动冻结BN/LN层，提高量化稳定性

#### 粒度配置建议
- **OCR识别**：推荐使用 `PerToken` 激活动态量化 + `PerGroup(256)` 权重量化
- **高精度需求**：Neck层使用 `PerGroup(128)`，Decoder使用 `PerAxis` INT8
- **边缘部署**：使用 `PerTensor` 量化以获得最佳推理性能

#### 分阶段训练建议
- **Warmup阶段**：正常训练3-5轮，让模型充分学习
- **QAT插入**：在warmup结束后插入FakeQuantize，避免早期量化干扰
- **QAT微调**：8-10轮微调，学习率0.1倍，监控CER指标
- **后处理**：可额外进行2-3轮正常训练，进一步提升精度

### 9. 性能优化

1. **批处理大小**：根据硬件内存调整，建议4-16
2. **混合精度**：启用AMP以加速训练
3. **CUDA图**：GPU训练时启用，可减少开销
4. **内存管理**：大模型训练时启用内存高效模式
5. **剪枝策略选择**：非结构化剪枝压缩比更高，结构化剪枝推理速度更快
6. **分层剪枝**：根据不同层的重要性设置不同的剪枝比例

### 10. 部署考虑

1. **硬件兼容性**：确保目标硬件支持所选量化格式
2. **框架版本**：PyTorch和torchao版本需兼容
3. **动态形状**：ONNX导出时考虑动态输入形状
4. **批处理推理**：生产环境使用批处理提升吞吐量
5. **剪枝模型部署**：非结构化剪枝可能需要特殊硬件支持，结构化剪枝兼容性更好
6. **量化+剪枝组合**：先剪枝后量化通常能获得更好的效果

### 11. 剪枝特殊注意事项

1. **剪枝前评估**：确保模型在剪枝前已经收敛到较好的精度
2. **微调阶段**：剪枝后需要进行微调，恢复模型精度
3. **学习率调整**：微调阶段使用较低的学习率（0.1倍）
4. **逐层剪枝**：对于复杂模型，考虑逐层剪枝而非一次性全局剪枝
5. **剪枝恢复**：如果剪枝后精度下降过多，可恢复原始模型重新尝试
6. **剪枝可视化**：使用TensorBoard等工具可视化剪枝效果和模型结构
7. **剪枝时机**：在模型收敛后执行剪枝（如Epoch 20-30）
8. **避免训练初期剪枝**：训练初期剪枝可能导致模型无法恢复

## 🔍 故障排除

### 问题1: 量化后精度下降过大

**解决**:
```bash
# 增加QAT训练轮数
--qat_epochs 10

# 提高蒸馏权重
--distillation_weight 0.5

# 使用高精度量化策略
--quantization_strategy int8_dynamic_activation_int8_weight
```

### 问题2: QAT训练不稳定

**解决**:
```bash
# 降低QAT学习率
--qat_learning_rate_multiplier 0.05

# 增加量化损失权重
--quantization_loss_weight 0.02

# 延迟QAT插入
--qat_insert_epoch 5
```

### 问题3: PTQ校准失败

**解决**:
```bash
# 增加校准批次
--calibration_batches 200

# 切换到QAT模式
--quantization_mode qat
```

### 问题4: 量化训练速度慢

```bash
# 解决方案
python main.py \
    --mode both \
    --memory_efficient true \
    --mixed_precision true \
    --batch_size 8 \
    --num_workers 4
```

### 问题5: 模型导出失败

```python
# 解决方案：使用备用导出方法
quantization_manager.export_quantized_model(
    'model.pth',
    dummy_input,
    use_torchscript=True
)
```

### 问题6: 内存不足

```bash
# 解决方案
python main.py \
    --mode both \
    --memory_efficient true \
    --batch_size 2 \
    --calibration_batches 50 \
    --gradient_checkpointing true
```

### 问题7: 剪枝后精度下降过大

```bash
# 解决方案
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_unstructured \
    --pruning_ratio 0.1 \
    --backbone_pruning_ratio 0.1 \
    --neck_pruning_ratio 0.2 \
    --decoder_pruning_ratio 0.05 \
    --finetune_epochs 15 \
    --min_acc_drop 0.005
```

### 问题8: QAT API精度不如预期

```bash
# 解决方案：启用分层混合精度
python main.py \
    --mode both \
    --enable_quantization \
    --use_modern_qat_api \
    --enable_layer_wise_qat \
    --qat_epochs 10 \
    --qat_learning_rate_multiplier 0.05 \
    --quantization_loss_weight 0.02 \
    --fake_quant_loss_weight 0.002
```

### 问题9: 剪枝训练报错

```bash
# 解决方案：降低剪枝比例或更换剪枝策略
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_strategy l1_structured \
    --pruning_ratio 0.2 \
    --pruning_epoch 30 \
    --finetune_epochs 10
```

### 问题10: 剪枝+量化组合优化效果不佳

```bash
# 解决方案：调整顺序和参数
python main.py \
    --mode both \
    --enable_pruning \
    --pruning_epoch 25 \
    --finetune_epochs 15 \
    --quantization_strategy int8_dyn_act_int4_weight \
    --qat_epochs 8 \
    --distillation_weight 0.5
```

### 问题11: 剪枝未应用

**可能原因**:
- 当前精度低于最佳精度的 `(1 - min_acc_drop)`

**解决**:
```bash
# 降低精度阈值要求
--min_acc_drop 0.05  # 允许5%的精度下降
```

### 问题12: 全局剪枝失败

**可能原因**:
- 模型中没有可剪枝的层

**解决**:
```bash
# 检查剪枝层配置
--pruning_layers backbone neck decoder
```

## ✅ 功能清单

### 量化功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 量化感知训练 (QAT) | ✅ | 使用torchao API |
| 训练后量化 (PTQ) | ✅ | 支持校准和快速部署 |
| 分层混合精度QAT | ✅ | 不同层使用不同精度 |
| 分阶段QAT训练 | ✅ | 延迟插入量化模块 |
| 知识蒸馏 | ✅ | 教师网络指导量化训练 |
| 量化损失计算 | ✅ | MSE + Cosine + L1 |
| 模型转换与导出 | ✅ | torch.export + ONNX |
| 超参数自动优化 | ✅ | 贝叶斯/网格/随机搜索 |
| 量化效果评估 | ✅ | 精度/速度/内存/压缩比 |
| 可视化报告 | ✅ | 图表 + JSON报告 |

### 剪枝功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 基础剪枝 | ✅ | 标准逐层L1非结构化剪枝 |
| 全局剪枝 | ✅ | 全局非结构化剪枝，获得更好稀疏性 |
| 结构化剪枝 | ✅ | L1/L2/Ln结构化剪枝 |
| 训练期间应用 | ✅ | 训练时应用剪枝并微调 |
| 验证回滚 | ✅ | 自动验证精度，下降过大则回滚 |
| 结构化压缩 | ✅ | 真正删除被剪枝通道 |
| 可视化 | ✅ | 生成剪枝效果图表 |
| 详细统计 | ✅ | 多维度剪枝统计信息 |
| BatchNorm调整 | ✅ | 结构化剪枝后自动调整BN层 |
| 剪枝检测 | ✅ | 基于权重分布的鲁棒检测 |

## 测试验证

### 测试文件结构

```
test_quantization.py
├── Test Suite 1: CorePyTorchTests (PyTorch核心剪枝测试)
│   ├── test_l1_unstructured_pruning()
│   ├── test_ln_structured_pruning()
│   └── test_global_unstructured_pruning()
├── Test Suite 2: CustomPruningTests (自定义剪枝逻辑测试)
│   ├── test_improved_pruning_detection()
│   ├── test_batchnorm_association()
│   └── test_pruning_validation_logic()
├── Test Suite 3: IntegrationTests (集成测试)
│   ├── test_method_existence()
│   ├── test_main_py_integration()
│   └── test_command_line_args()
└── Test Suite 4: FullIntegrationTests (完整集成测试)
    └── test_pruning_manager_basic()
```

### 运行测试

```bash
# 运行所有测试
python test_quantization.py

# 运行特定测试套件
python test_quantization.py --test pytorch_core
python test_quantization.py --test custom_logic
python test_quantization.py --test integration
python test_quantization.py --test full_integration
```

### 测试覆盖

| 功能 | 测试数量 | 状态 |
|------|----------|------|
| PyTorch L1 非结构化剪枝 | 1 | ✅ |
| PyTorch Ln 结构化剪枝 | 1 | ✅ |
| PyTorch 全局非结构化剪枝 | 1 | ✅ |
| 改进的剪枝检测 | 1 | ✅ |
| BatchNorm 关联逻辑 | 1 | ✅ |
| 剪枝验证逻辑 | 1 | ✅ |
| quantization.py 方法检查 | 1 | ✅ |
| main.py 集成检查 | 1 | ✅ |
| 命令行参数 | 1 | ✅ |
| PruningManager 基本功能 | 1 | ✅ |

**总计: 10 个测试用例**

## OCR可视化调试分析实现与完善建议

### 1. 当前实现概述

#### 1.1 核心功能

**基础可视化功能：**
- **多模型支持**：支持CTC模型（SVTRv2、ViT、CNN）和AR模型
- **多种可视化类型**：颜色条对齐、热力图对齐、增强版CTC对齐、紧凑版CTC对齐、AR解码器对齐、AR束搜索对齐
- **错误分析**：支持7种错误类型的识别和可视化
- **统一接口**：保持原有`debug_virtual_alignment`接口的兼容性
- **配置管理**：集中的可视化配置类
- **模型输出处理**：统一的模型输出处理

**高级分析功能：**
- **鲁棒性分析**：5种噪声类型干扰下的模型性能分析（高斯噪声、模糊、旋转、遮挡、对比度变化）
- **训练过程分析**：训练历史管理、跨轮次性能对比、损失趋势分析、对齐质量演变
- **数据增强效果分析**：4种增强策略效果对比（旋转、缩放、剪切、颜色抖动）
- **注意力机制深度分析**：多层多头注意力可视化、注意力流分析、自注意力与交叉注意力对比
- **多模型对比分析**：多模型性能排名、错误分布对比、置信度分布、综合评分系统

#### 1.2 架构设计

```
┌───────────────────────────────────────────────────────────┐
│                     debug_virtual_alignment               │  # 入口函数，保持接口兼容
│  ┌─────────────────────────────────────────────────────┐  │
│  │  新增参数:                                            │  │
│  │  - enable_robustness_analysis                        │  │
│  │  - enable_training_analysis                          │  │
│  │  - enable_augmentation_analysis                      │  │
│  │  - enable_deep_attention_analysis                    │  │
│  │  - enable_multi_model_comparison                     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────┬─────────────────────────────────────────────┘
              │
              ├────────────────────────────────┐
              │                                 │
┌─────────────▼─────────────┐   ┌─────────────▼─────────────┐
│        CTC 模型可视化       │   │        AR 模型可视化       │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                                 │
              ├───────────────────────┬─────────┴─────────┬────────────────
              │                       │                   │
┌─────────────▼─────────────┐ ┌────────▼────────────┐ ┌────────▼────────────┐
│  visualize_ctc_colorbar   │ │ visualize_ar_alignment │ │ visualize_ar_beam_search │
│  visualize_ctc_heatmap    │ └─────────────────────┘ └─────────────────────┘
│  visualize_ctc_enhanced   │
│  visualize_ctc_compact    │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│       错误分析与可视化       │
│  OCRErrorAnalyzer.analyze_errors  │
│  OCRErrorAnalyzer.visualize_errors│
└───────────────────────────────────┘
              │
              ├───────────────────────────────────────────────────────────────┐
              │                                                               │
┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐
│   鲁棒性分析系统           │  │  训练过程分析系统           │  │ 数据增强效果分析           │
│  - NoiseGenerator          │  │  - TrainingHistoryManager │  │  - DataAugmentationVisualizer│
│  - OCRRobustnessAnalyzer   │  │  - TrainingProcessAnalyzer │  │  - AugmentationEffectAnalyzer│
│  - visualize_robustness    │  │  - visualize_training_progress│ │  - visualize_augmentation_effects│
└────────────────────────────┘  └────────────────────────────┘  └────────────────────────────┘
              │                                                               │
              ├───────────────────────────────────────────────────────────────┤
              │                                                               │
┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐
│  注意力机制深度分析         │  │  多模型对比分析            │  │  配置管理系统              │
│  - AttentionExtractor     │  │  - MultiModelComparator   │  │  - VisualizationConfig     │
│  - DeepAttentionAnalyzer   │  │  - visualize_multi_model_comparison│ │  (20+ 配置参数)          │
│  - visualize_deep_attention│  └────────────────────────────┘  └────────────────────────────┘
└────────────────────────────┘
```

### 2. 常见OCR问题分析覆盖情况

#### 2.1 已覆盖的OCR问题分析

| 问题类型 | 分析方法 | 实现情况 | 新增功能 |
|---------|---------|---------|---------|
| 对齐问题 | CTC对齐可视化、AR解码器对齐可视化、束搜索对齐可视化 | ✅ 完全覆盖 | - |
| 置信度分析 | 置信度曲线、置信度热力图、基于置信度的颜色调整 | ✅ 完全覆盖 | - |
| 错误类型分析 | 替换、删除、插入、交换、空格、大小写、标点符号错误 | ✅ 完全覆盖 | - |
| 模型性能分析 | 预测结果与真实文本对比、对齐质量评估、字符频率分布 | ✅ 完全覆盖 | 训练过程跨轮次对比 |
| 注意力机制分析 | 注意力权重热力图（针对AR模型） | ✅ 完全覆盖 | 多层多头注意力、注意力流分析 |
| 多模型对比 | 支持多种模型结构的可视化、统一的可视化接口 | ✅ 完全覆盖 | 多模型性能排名、综合评分系统 |
| 错误定位 | 错误位置标记、错误详情分析 | ✅ 完全覆盖 | - |
| 上下文依赖分析 | 束搜索可视化、注意力权重可视化 | ✅ 完全覆盖 | - |
| 长文本分析 | 最大长度限制、紧凑可视化设计 | ✅ 完全覆盖 | - |
| 特殊字符处理 | 标点符号错误分析、空格错误分析 | ✅ 完全覆盖 | - |
| 模型决策过程分析 | Top-K预测可视化、概率分布热力图 | ✅ 完全覆盖 | - |
| **鲁棒性分析** | **噪声干扰下的模型性能分析** | **✅ 完全覆盖** | **5种噪声类型、多级别干扰测试** |
| **训练过程分析** | **不同训练轮次的对比可视化** | **✅ 完全覆盖** | **损失趋势、对齐质量演变、置信度变化** |
| **数据增强效果分析** | **数据增强前后的对比可视化** | **✅ 完全覆盖** | **4种增强策略、多级别效果对比** |
| **模型对比分析** | **多个模型结果的对比可视化** | **✅ 完全覆盖** | **性能排名、错误分布、置信度对比** |

#### 2.2 未完全覆盖的OCR问题分析

| 问题类型 | 缺失分析方法 | 改进建议 |
|---------|------------|---------|
| 交互式展示 | 静态图片无法交互 | 考虑添加交互式可视化功能（如使用Dash或Streamlit） |
| 跨语言/特殊字符集分析 | 缺少针对特定语言的分析 | 添加针对不同语言和字符集的专门分析 |
| 模型压缩效果分析 | 缺少模型压缩前后的对比 | 添加模型压缩效果的对比可视化 |
| 实时性能监控 | 缺少实时性能指标监控 | 添加实时性能监控和告警系统 |

### 3. 代码质量与可维护性

#### 3.1 优势

1. **模块化设计**：功能模块清晰，便于扩展和维护
2. **统一接口**：保持原有接口兼容性，降低迁移成本
3. **配置集中管理**：使用`VisualizationConfig`类集中管理20+可视化参数
4. **错误分析增强**：添加了详细的错误分析功能
5. **类型安全**：使用类型注解，提高代码可读性和安全性
6. **详细文档**：函数和类都有详细的文档注释
7. **向后兼容**：所有新功能都通过可选参数添加，不影响现有代码
8. **配置驱动**：通过配置类控制功能开关，灵活启用/禁用功能
9. **丰富的可视化**：每个分析系统都提供多面板综合可视化（4-15个面板）
10. **持久化支持**：训练历史支持pickle格式保存和加载

#### 3.2 改进建议

1. **代码复用**：进一步提取公共功能，减少重复代码
2. **性能优化**：对于大型模型和长文本，优化可视化生成速度
3. **可扩展性**：添加插件机制，支持自定义可视化类型
4. **可视化样式统一**：统一不同可视化类型的颜色方案和布局
5. **异常处理**：增强异常处理机制，提高代码健壮性
6. **日志系统**：添加日志记录，便于调试和监控
7. **单元测试**：为新添加的分析系统添加单元测试

### 4. 功能扩展实现详情

#### 4.1 鲁棒性分析系统 ✅ 已实现

**实现组件：**
- `NoiseGenerator`：噪声生成器，支持5种噪声类型
  - 高斯噪声（Gaussian Noise）
  - 模糊噪声（Blur）
  - 旋转噪声（Rotation）
  - 遮挡噪声（Occlusion）
  - 对比度变化（Contrast）

- `OCRRobustnessAnalyzer`：鲁棒性分析器
  - 支持多级别噪声强度测试
  - 计算CER和准确率指标
  - 生成详细的错误类型统计

- `visualize_robustness_analysis`：鲁棒性可视化函数
  - 4面板综合可视化：
    1. CER变化曲线（多噪声类型对比）
    2. 准确率变化曲线
    3. 错误类型热力图
    4. 鲁棒性雷达图

**配置参数：**
```python
config.robustness_analysis = True
config.noise_types = ['gaussian', 'blur', 'rotation', 'occlusion', 'contrast']
config.noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
config.robustness_save_dir = './robustness_analysis'
```

#### 4.2 训练过程分析系统 ✅ 已实现

**实现组件：**
- `TrainingHistoryManager`：训练历史管理器
  - 支持训练状态保存（pickle格式）
  - 支持历史记录加载
  - 支持增量更新

- `TrainingProcessAnalyzer`：训练过程分析器
  - 分析损失函数变化趋势
  - 分析对齐质量演变
  - 分析置信度变化
  - 计算训练指标（CER、准确率等）

- `visualize_training_progress`：训练进度可视化函数
  - 8面板综合可视化：
    1. 损失函数变化曲线
    2. 对齐质量变化曲线
    3. 置信度变化曲线
    4. CER变化曲线
    5. 错误类型趋势
    6. 训练指标对比
    7. 训练效果雷达图
    8. 轮次对比表格

**配置参数：**
```python
config.training_analysis = True
config.training_history_file = './training_history.pkl'
config.epochs_to_compare = [1, 5, 10, 20, 30]
config.training_save_dir = './training_analysis'
```

#### 4.3 数据增强效果分析系统 ✅ 已实现

**实现组件：**
- `DataAugmentationVisualizer`：数据增强可视化器
  - 旋转增强（Rotation）
  - 缩放增强（Scale）
  - 剪切增强（Shear）
  - 颜色抖动（Color Jitter）

- `AugmentationEffectAnalyzer`：增强效果分析器
  - 分析增强前后的性能变化
  - 统计错误类型分布变化
  - 计算增强效果指标

- `visualize_augmentation_effects`：增强效果可视化函数
  - 5面板综合可视化：
    1. CER变化曲线（多增强策略对比）
    2. 准确率变化曲线
    3. 错误类型热力图
    4. 增强效果雷达图
    5. 增强策略对比表格

**配置参数：**
```python
config.augmentation_analysis = True
config.augmentation_types = ['rotation', 'scale', 'shear', 'color_jitter']
config.augmentation_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
config.augmentation_save_dir = './augmentation_analysis'
```

#### 4.4 注意力机制深度分析系统 ✅ 已实现

**实现组件：**
- `AttentionExtractor`：注意力提取器
  - 提取多层注意力权重
  - 提取多头注意力权重
  - 支持自注意力和交叉注意力

- `DeepAttentionAnalyzer`：深度注意力分析器
  - 分析不同层的注意力分布
  - 分析不同头的关注模式
  - 分析注意力流向
  - 计算注意力统计指标

- `visualize_deep_attention`：深度注意力可视化函数
  - 15面板综合可视化：
    1. 层级注意力趋势
    2. 头部注意力趋势
    3. 多层注意力热力图
    4. 多头注意力热力图
    5. 注意力流可视化
    6. 自注意力模式
    7. 交叉注意力模式
    8. 注意力统计指标
    9. 层级对比雷达图
    10. 头部对比雷达图
    11. 层级对比表格
    12. 头部对比表格
    13. 注意力分布直方图
    14. 注意力相关性矩阵
    15. 注意力摘要统计

**配置参数：**
```python
config.deep_attention_analysis = True
config.attention_layers_to_visualize = [0, 1, 2, 3]
config.attention_heads_to_visualize = [0, 1, 2, 3]
config.attention_save_dir = './attention_analysis'
```

#### 4.5 多模型对比分析系统 ✅ 已实现

**实现组件：**
- `MultiModelComparator`：多模型对比器
  - 支持同时对比多个模型
  - 计算统一的性能指标
  - 生成模型排名
  - 计算综合评分

- `visualize_multi_model_comparison`：多模型对比可视化函数
  - 14面板综合可视化：
    1. CER对比柱状图
    2. 准确率对比柱状图
    3. 置信度对比柱状图
    4. 错误类型分布对比
    5. 性能指标箱线图
    6. 模型排名雷达图
    7. 综合评分雷达图
    8. 模型排名热力图
    9. 模型对比表格
    10. 性能指标对比表格
    11. 错误类型对比表格
    12. 模型预测对比
    13. 置信度分布对比
    14. 模型摘要统计

**配置参数：**
```python
config.multi_model_comparison = True
config.models_to_compare = [model1, model2, model3]
config.model_names = ['SVTRv2', 'ViT', 'CNN']
config.comparison_save_dir = './multi_model_comparison'
```

#### 4.6 交互式可视化（待实现）

考虑添加交互式可视化功能，使用Dash或Streamlit等框架，支持：
- 缩放和平移可视化结果
- 悬停查看详细信息
- 切换不同的可视化类型
- 调整可视化参数
- 导出不同格式的结果

### 5. 使用示例

#### 5.1 基础使用

```python
from debug import debug_virtual_alignment, VisualizationConfig

# 创建配置
config = VisualizationConfig(
    visualization_type='colorbar',
    top_k=5,
    prob_threshold=0.1
)

# 运行基础可视化
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output'
)
```

#### 5.2 鲁棒性分析

```python
# 启用鲁棒性分析
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output',
    enable_robustness_analysis=True
)
```

#### 5.3 训练过程分析

```python
# 启用训练过程分析
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output',
    enable_training_analysis=True
)
```

#### 5.4 数据增强效果分析

```python
# 启用数据增强效果分析
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output',
    enable_augmentation_analysis=True
)
```

#### 5.5 注意力机制深度分析

```python
# 启用注意力机制深度分析
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='vit',
    train_mode='ar',
    output_dir='./output',
    enable_deep_attention_analysis=True
)
```

#### 5.6 多模型对比分析

```python
# 启用多模型对比分析
config = VisualizationConfig(
    multi_model_comparison=True,
    models_to_compare=[model1, model2, model3],
    model_names=['SVTRv2', 'ViT', 'CNN']
)

debug_virtual_alignment(
    device='cuda',
    model=model1,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output',
    enable_multi_model_comparison=True
)
```

#### 5.7 综合分析（启用所有功能）

```python
# 启用所有高级分析功能
debug_virtual_alignment(
    device='cuda',
    model=model,
    loader=val_loader,
    epoch=10,
    backbone='svtrv2',
    train_mode='ctc',
    output_dir='./output',
    enable_robustness_analysis=True,
    enable_training_analysis=True,
    enable_augmentation_analysis=True,
    enable_deep_attention_analysis=True,
    enable_multi_model_comparison=True
)
```

### 6. 未来改进方向

**未来改进方向：**
1. 添加交互式可视化功能（Dash/Streamlit）
2. 支持跨语言/特殊字符集分析
3. 添加模型压缩效果分析
4. 实现实时性能监控
5. 为新功能添加单元测试
6. 进一步优化可视化生成性能

总体而言，当前实现已经达到了**高质量的OCR可视化调试分析水平**，能够有效帮助开发者**全面理解和改进OCR模型的性能**。通过丰富的可视化工具和深入的分析功能，为OCR模型的调试、优化和部署提供了强有力的支持。

## 📚 相关资源

### 论文参考
1. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
2. "Knowledge Distillation: A Survey"
3. "Post-Training Quantization for Vision Transformers"

### 技术文档
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [torchao Documentation](https://github.com/pytorch/ao)
- [ONNX Quantization](https://onnxruntime.ai/docs/performance/quantization.html)

### 开源项目
- [Intel Neural Compressor](https://github.com/intel/neural-compressor)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Apache TVM](https://tvm.apache.org/)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个量化训练方案！

### 提交Issue
- 描述清楚遇到的问题
- 提供复现步骤和环境信息
- 附上相关日志和错误信息

### 提交PR
- 确保代码通过所有测试
- 添加适当的注释和文档
- 更新README和CHANGELOG

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。
